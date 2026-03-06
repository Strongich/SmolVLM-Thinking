import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.datasets.load_llavacot import load_llava_cot, make_smolvlm_messages
from src.model_init.model import initialize_model_thinking

# ---------------------------------------------------------------------------
# Top-level constants
# ---------------------------------------------------------------------------

GAINRL_INDICES_PATH = "gainrl_indices/llava_cot_sorted_indices.pt"
EVAL_FORMAT_SUBSET_SIZE = 100
TOTAL_LOOPS = 100
SUBSET_SIZE = 1024
Data_sort = []
# ---------------------------------------------------------------------------
# GAINRL Sampler
# ---------------------------------------------------------------------------

Data_sort = []


def gaussian_sample_list(A, num_samples, center_index, std_dev):
    A_len = len(A)
    indices = np.arange(A_len)
    probs = np.exp(-0.5 * ((indices - center_index) / std_dev) ** 2)
    probs /= probs.sum()
    sample_indices = np.random.choice(
        indices, size=min(num_samples, A_len), replace=False, p=probs
    )
    sample_indices.sort()
    B = [A[i] for i in sample_indices]
    return B


def update_dataset(sort_list, mean, std, subset_size, loop):
    if loop == 0:
        mean_new = mean
    else:
        global Data_sort
        avg_accuracy_now = sum(d["accuracy"] for d in Data_sort) / len(Data_sort)
        avg_angle_now = sum(d["angle"] for d in Data_sort) / len(Data_sort)
        device = torch.device("cuda")
        acc = torch.tensor(avg_accuracy_now, dtype=torch.float32, device=device)
        ang = torch.tensor(avg_angle_now, dtype=torch.float32, device=device)
        adjustment = 500 * torch.tanh(2 * (acc / 2 - 0.5)) + 500 * torch.tanh(2 * ang)
        adjustment = torch.clamp(adjustment, 0, 1000)

        mean_new = mean + adjustment.item()
        print("mean_new:")
        print(mean_new)

    new_index = gaussian_sample_list(
        sort_list, num_samples=subset_size, center_index=mean, std_dev=std
    )
    Data_sort = []
    return new_index, mean_new


class GainRLSampler(Sampler):
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DatasetUpdateCallback(TrainerCallback):
    def __init__(self, sort_list, total_loops, subset_size):
        super().__init__()
        self.sort_list = sort_list
        self.total_loops = total_loops
        self.subset_size = subset_size
        self.loop = 0
        self.mean = 0
        self.std = 1000
        self.drop_list = []

    def _next_subset(self):
        new_indices, self.mean = update_dataset(
            self.sort_list, self.mean, self.std, self.subset_size, self.loop
        )
        return new_indices

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        if self.loop >= self.total_loops - 1:
            return control

        if train_dataloader is None:
            return control

        new_indices = self._next_subset()
        # print("Sample Data Index:")
        # print(new_indices)
        new_sampler = GainRLSampler(new_indices)
        train_dataloader.base_dataloader.batch_sampler.sampler = new_sampler

        self.loop += 1
        return control


# ---------------------------------------------------------------------------
# Format compliance evaluation callback
# ---------------------------------------------------------------------------


class FormatComplianceCallback(TrainerCallback):
    """Evaluates <think>...</think> format compliance on a fixed eval subset.

    Every eval step:
    - Generates responses on EVAL_FORMAT_SUBSET_SIZE examples.
    - Checks strict <think> structure.
    - Logs eval/format_compliance to trackio.
    - Stops training after 3 consecutive checks with >=98% compliance.
    """

    def __init__(
        self,
        eval_dataset,
        processor,
        threshold=0.98,
        consecutive_target=3,
    ):
        self.processor = processor
        self.threshold = threshold
        self.consecutive_target = consecutive_target
        self.consecutive_count = 0

        # Sample fixed subset once at init
        n = min(EVAL_FORMAT_SUBSET_SIZE, len(eval_dataset))
        rng = random.Random(42)
        indices = rng.sample(range(len(eval_dataset)), n)
        self.eval_subset = [eval_dataset[i] for i in indices]

    @staticmethod
    def _check_format(text: str) -> bool:
        if text.count("<think>") != 1:
            return False
        if text.count("</think>") != 1:
            return False
        think_idx = text.index("<think>")
        end_think_idx = text.index("</think>")
        if think_idx >= end_think_idx:
            return False
        after_close = text[end_think_idx + len("</think>") :]
        if "<think>" in after_close:
            return False
        return True

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        model.eval()
        compliant = 0

        for example in self.eval_subset:
            messages = example["messages"]
            image = example.get("image")

            # Prepare prompt (all turns except the last assistant turn)
            prompt_messages = messages[:-1]
            formatted = self.processor.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if image is not None:
                image = image.convert("RGB")
            inputs = self.processor(
                text=formatted,
                images=[image] if image is not None else None,
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=16384,
                    temperature=0.4,
                    do_sample=True,
                    top_p=0.95,
                    top_k=20,
                )

            generated = self.processor.decode(
                out[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            if self._check_format(generated):
                compliant += 1

        total = len(self.eval_subset)
        compliance = compliant / total if total > 0 else 0.0
        print(
            f"[FormatCompliance] step={state.global_step}  {compliance:.2%} ({compliant}/{total})"
        )

        if metrics is not None:
            metrics["eval/format_compliance"] = compliance

        if compliance >= self.threshold:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        if self.consecutive_count >= self.consecutive_target:
            print(
                f"[FormatCompliance] {self.consecutive_target} consecutive checks "
                f">= {self.threshold:.0%}. Stopping training."
            )
            control.should_training_stop = True

        model.train()
        return control


# ---------------------------------------------------------------------------
# Completion-mask helpers
# ---------------------------------------------------------------------------


def _precompute_labels_with_mask(example: dict, processor, max_length: int) -> dict:
    """Tokenize one training example and mask non-assistant tokens in labels.

    Calls the processor three times per assistant turn (full / prefix / span)
    to locate exact token boundaries. Returns text-token lists only — images
    stay as PIL objects in the ``images`` column and are processed on-the-fly
    by :class:`VLMDataCollator` at batch time (avoids storing pixel_values in
    the HF Dataset).

    Non-assistant tokens are set to ``-100`` in ``labels`` so they are ignored
    by the cross-entropy loss.
    """
    messages = example["messages"]
    imgs = example.get("images") or []
    image = imgs[0].convert("RGB") if imgs else None

    # Full conversation
    full_text = processor.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    full_out = processor(
        text=full_text,
        images=[image] if image is not None else None,
        return_tensors="pt",
    )
    input_ids = full_out["input_ids"][0]
    attention_mask = full_out["attention_mask"][0]
    seq_len = input_ids.shape[0]

    # Build completion mask: True on assistant token spans, False elsewhere
    completion_mask = torch.zeros(seq_len, dtype=torch.bool)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Token index where this assistant turn STARTS
        # (= length of prefix tokenised with the generation prompt appended)
        start = processor(
            text=processor.apply_chat_template(
                messages[:i], add_generation_prompt=True, tokenize=False
            ),
            images=[image] if image is not None else None,
            return_tensors="pt",
        )["input_ids"].shape[1]

        # Token index where this assistant turn ENDS
        end = processor(
            text=processor.apply_chat_template(
                messages[: i + 1], add_generation_prompt=False, tokenize=False
            ),
            images=[image] if image is not None else None,
            return_tensors="pt",
        )["input_ids"].shape[1]

        completion_mask[start:end] = True

    labels = input_ids.clone()
    labels[~completion_mask] = -100

    return {
        "input_ids": input_ids[:max_length].tolist(),
        "attention_mask": attention_mask[:max_length].tolist(),
        "labels": labels[:max_length].tolist(),
    }


class VLMDataCollator:
    """Collator for pre-tokenized VLM data.

    * Pads ``input_ids`` / ``attention_mask`` / ``labels`` to the longest
      sequence in the batch.
    * Processes images on-the-fly via the image processor, producing
      ``pixel_values`` (and ``pixel_attention_mask`` when present).

    Images are expected under the ``images`` key as a list of PIL objects
    (one image per example, matching the LLaVA-CoT dataset structure).
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, examples: list[dict]) -> dict:
        # ── Text sequences ──────────────────────────────────────────────────
        max_len = max(len(e["input_ids"]) for e in examples)

        def _pad(seq: list, pad_val: int) -> list:
            return seq + [pad_val] * (max_len - len(seq))

        batch = {
            "input_ids": torch.tensor(
                [_pad(e["input_ids"], self.pad_id) for e in examples],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [_pad(e["attention_mask"], 0) for e in examples],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [_pad(e["labels"], -100) for e in examples],
                dtype=torch.long,
            ),
        }

        # ── Images ──────────────────────────────────────────────────────────
        pil_images = [(e.get("images") or [None])[0] for e in examples]
        pil_images = [
            img.convert("RGB") if img is not None else None for img in pil_images
        ]
        valid = [img for img in pil_images if img is not None]
        if valid:
            img_out = self.processor.image_processor(images=valid, return_tensors="pt")
            batch["pixel_values"] = img_out["pixel_values"]
            if "pixel_attention_mask" in img_out:
                batch["pixel_attention_mask"] = img_out["pixel_attention_mask"]

        return batch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Dataset: load, format, split
    raw_ds = load_llava_cot()
    ds = raw_ds.map(make_smolvlm_messages)
    split = ds.train_test_split(test_size=0.06, seed=1234)
    train_ds = split["train"]
    eval_ds = split["test"]

    # Model: load then freeze vision encoder
    processor, model = initialize_model_thinking()
    for param in model.model.vision_model.parameters():
        param.requires_grad = False

    # Pre-tokenize with completion mask.
    # Produces input_ids / attention_mask / labels (non-assistant → -100).
    # Only the `images` column is kept alongside the new text columns so the
    # collator can process pixel_values on-the-fly (avoids ~160 GB of stored
    # pixel_values tensors).
    # eval_ds_orig is kept with its original columns for FormatComplianceCallback
    # which needs example["messages"] and example["image"].
    eval_ds_orig = eval_ds

    _TRAIN_TOK_PATH = "src/datasets/llava_cot_tokenized/train"
    _EVAL_TOK_PATH = "src/datasets/llava_cot_tokenized/eval"

    if os.path.isdir(_TRAIN_TOK_PATH) and os.path.isdir(_EVAL_TOK_PATH):
        import datasets as hf_datasets

        print("Loading pre-tokenized datasets from disk …")
        train_ds = hf_datasets.load_from_disk(_TRAIN_TOK_PATH)
        eval_ds = hf_datasets.load_from_disk(_EVAL_TOK_PATH)
        # Re-cast images column to PIL so the collator can process them.
        train_ds = train_ds.cast_column(
            "images", hf_datasets.Sequence(hf_datasets.Image())
        )
        eval_ds = eval_ds.cast_column(
            "images", hf_datasets.Sequence(hf_datasets.Image())
        )
    else:
        _mask_fn = partial(
            _precompute_labels_with_mask, processor=processor, max_length=8192
        )
        cols_to_drop_train = [c for c in train_ds.column_names if c != "images"]
        cols_to_drop_eval = [c for c in eval_ds.column_names if c != "images"]
        train_ds = train_ds.map(
            _mask_fn,
            remove_columns=cols_to_drop_train,
            desc="Tokenising train (completion mask)",
            num_proc=4,
            writer_batch_size=500,
        )
        eval_ds = eval_ds.map(
            _mask_fn,
            remove_columns=cols_to_drop_eval,
            desc="Tokenising eval (completion mask)",
            num_proc=4,
            writer_batch_size=500,
        )
        print(f"Saving tokenized datasets to {_TRAIN_TOK_PATH} / {_EVAL_TOK_PATH} …")
        train_ds.save_to_disk(_TRAIN_TOK_PATH)
        eval_ds.save_to_disk(_EVAL_TOK_PATH)

    # GAINRL sort list — load pre-computed indices, fall back to sequential
    if os.path.exists(GAINRL_INDICES_PATH):
        sort_list = torch.load(GAINRL_INDICES_PATH).tolist()
    else:
        print(f"[GAINRL] {GAINRL_INDICES_PATH} not found, using sequential indices.")
        sort_list = list(range(len(train_ds)))

    training_args = SFTConfig(
        project="SmolVLM-Thinking",
        output_dir="SmolVLM-Thinking_llava_cot_100k_run_1",
        # assistant_only_loss is intentionally omitted — non-assistant tokens
        # are already masked to -100 in `labels` by _precompute_labels_with_mask.
        max_length=16384,
        report_to="trackio",
        run_name="sft_smolvlm_llavacot_100k_run_1",
        bf16=True,
        per_device_train_batch_size=4,
        warmup_ratio=0.03,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        optim="adamw",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        max_grad_norm=0.1,
        save_steps=20,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )

    dataset_cb = DatasetUpdateCallback(
        sort_list=sort_list,
        total_loops=TOTAL_LOOPS,
        subset_size=SUBSET_SIZE,
    )
    # FormatComplianceCallback needs messages + image columns — use orig eval split.
    format_cb = FormatComplianceCallback(
        eval_dataset=eval_ds_orig,
        processor=processor,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=VLMDataCollator(processor),
        callbacks=[dataset_cb, format_cb],
    )
    trainer.train()

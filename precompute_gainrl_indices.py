"""Precompute GAINRL sorted indices for a SmolVLM checkpoint.

Runs a forward pass on every training example from LLaVA-CoT-100k,
captures the input activations of the last text-transformer layer's
MLP up_proj, computes a cosine-similarity metric, and saves sorted
indices to a .pt file for use by DatasetUpdateCallback in sft_train.py.

Usage:
    python precompute_gainrl_indices.py \\
        --save_path gainrl_sorted_indices.pt \\
        --model_name SmolVLM-Thinking \\
        --gpu_id 0
"""

import argparse
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.load_llavacot import load_llava_cot, make_smolvlm_messages
from src.model_init.model import initialize_model_thinking

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect activation metric from the last MLP up_proj layer of SmolVLM"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SmolVLM-Thinking",
        help="Local path or HuggingFace model identifier for the SmolVLM checkpoint",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Where to save the resulting .pt file with sorted indices",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="CUDA device ID (e.g. '0'). Use '-1' for CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,  # fits on 16GB VRAM GPU
        help="Number of examples to process in one GPU forward pass.",
    )
    return parser


# ---------------------------------------------------------------------------
# Hook helpers (identical logic to original GAINRL script)
# ---------------------------------------------------------------------------


def register_act_hooks(
    model, target_layer_name: str, store: Dict[str, torch.Tensor]
) -> List:
    """Attach a forward hook that captures the input to target_layer_name."""
    hooks = []

    def _get_hook(name):
        def hook(_, inputs, __):
            store[name] = inputs[0].detach()

        return hook

    for name, module in model.named_modules():
        if name == target_layer_name:
            hooks.append(module.register_forward_hook(_get_hook(name)))
            break
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = build_parser().parse_args()

    if args.gpu_id != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = "cuda"
    else:
        device = "cpu"

    # Load SmolVLM model + processor
    processor, model = initialize_model_thinking(args.model_name)
    model.eval()

    # Identify target layer inside the text sub-model
    # SmolVLM (Idefics3): text_model is LlamaModel, layers live directly on it
    num_layers = len(model.model.text_model.layers)
    target_layer = f"model.model.text_model.layers.{num_layers - 1}.mlp.up_proj"
    print(f"Target layer: {target_layer}")

    # Disable image splitting: forces every image to a single 384×384 tile
    # (729 tokens) instead of splitting large images into 4-5 tiles (~3600 tokens).
    # We only need coarse activation patterns for the metric — full resolution
    # is unnecessary and was the main cause of slow forward passes.
    processor.image_processor.do_image_splitting = False

    # Load and format dataset
    raw_ds = load_llava_cot()
    ds = raw_ds.map(make_smolvlm_messages)

    # Pre-format all examples: extract prompt text + RGB image
    def _prepare(example):
        messages = example["messages"]
        image = example.get("image")
        # Keep only system + first user turn (contains the image).
        # Multi-turn examples have extra user/assistant pairs that would make
        # sequences longer and inconsistent across examples.
        first_user_idx = next(
            (i for i, m in enumerate(messages) if m["role"] == "user"), None
        )
        if first_user_idx is None:
            return None, None
        prompt_messages = messages[: first_user_idx + 1]
        formatted = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted, image.convert("RGB") if image is not None else None

    # CPU preprocessing: decode images, apply chat template, run processor.
    # Returns CPU tensors + bookkeeping; .to(device) happens in the main thread.
    def _prepare_batch(batch_start: int):
        batch_ds = ds.select(range(batch_start, min(batch_start + batch_size, n)))
        batch = [batch_ds[i] for i in range(len(batch_ds))]
        prepared = [_prepare(ex) for ex in batch]
        valid_indices = [i for i, (t, _) in enumerate(prepared) if t is not None]
        if not valid_indices:
            return None, valid_indices, len(batch)
        texts = [prepared[i][0] for i in valid_indices]
        images = [prepared[i][1] for i in valid_indices]
        model_inputs = processor(
            text=list(texts),
            images=images,
            return_tensors="pt",
            padding=True,
        )  # stays on CPU — transferred to GPU in the main thread
        return model_inputs, valid_indices, len(batch)

    # Collect activation metrics.
    # While the GPU runs the forward pass on batch N, the CPU prepares batch N+1
    # in a background thread, overlapping the two bottlenecks.
    metrics = []
    batch_size = args.batch_size
    n = len(ds)
    batch_starts = list(range(0, n, batch_size))

    PREFETCH = 16  # batches to prepare ahead of the GPU

    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        # Prime the pipeline: submit the first PREFETCH batches up front
        futures: Dict[int, Future] = {
            i: pool.submit(_prepare_batch, batch_starts[i])
            for i in range(min(PREFETCH, len(batch_starts)))
        }

        for step, _ in enumerate(
            tqdm(batch_starts, desc="Collecting activation metrics")
        ):
            model_inputs, valid_indices, batch_len = futures.pop(step).result()

            # Keep the pipeline full: submit one more batch for each one consumed
            next_step = step + PREFETCH
            if next_step < len(batch_starts):
                futures[next_step] = pool.submit(
                    _prepare_batch, batch_starts[next_step]
                )

            if model_inputs is None:
                metrics.extend([torch.tensor(0.0)] * batch_len)
                continue

            model_inputs = model_inputs.to(device)

            store: Dict[str, torch.Tensor] = {}
            hooks = register_act_hooks(model, target_layer, store)

            with torch.no_grad():
                model(**model_inputs)

            remove_hooks(hooks)

            if target_layer not in store:
                metrics.extend([torch.tensor(0.0)] * batch_len)
                continue

            # activations: [valid_batch, padded_seq_len, hidden]
            activations = store[target_layer].float()
            attention_mask = model_inputs["attention_mask"]

            # Compute metric per valid example, scatter into full batch positions
            valid_metrics: Dict[int, torch.Tensor] = {}
            for vi, orig_i in enumerate(valid_indices):
                real_tokens = attention_mask[vi].bool()
                inp = activations[vi][real_tokens]  # [seq_len, hidden]
                if inp.shape[0] < 117:  # need >116 tokens for [110:-6] indexing
                    valid_metrics[orig_i] = torch.tensor(0.0)
                    continue
                normalized = F.normalize(inp, p=2, dim=1)
                cos = normalized @ normalized.T
                cos = cos * torch.tril(torch.ones_like(cos), diagonal=1)
                val = cos[110:-6][110:-6].mean() + 8 * cos[110:-6][:110].mean()
                valid_metrics[orig_i] = val

            for i in range(batch_len):
                metrics.append(valid_metrics.get(i, torch.tensor(0.0)))

    indices = torch.argsort(torch.stack(metrics), descending=True)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(indices, save_path)
    print(f"Saved sorted indices to: {save_path}")


if __name__ == "__main__":
    main()

"""Analyze per-example token lengths for the LLaVA-CoT training set.

Prints:
  - average / median / min / max sequence length
  - total tokens in the dataset
  - approximate global batch size in tokens (batch_size × avg_len)
"""

import statistics

import numpy as np
from tqdm import tqdm

from src.datasets.load_llavacot import load_llava_cot, make_smolvlm_messages
from src.model_init.model import initialize_model_thinking

# ---------------------------------------------------------------------------
# Config  (mirror sft_train.py values)
# ---------------------------------------------------------------------------
MAX_SAMPLES = None  # set to e.g. 2000 for a quick sanity-check run
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # from sft_train.py
NUM_DEVICES = 1  # adjust if running multi-GPU

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print("Loading dataset …")
raw_ds = load_llava_cot(max_samples=MAX_SAMPLES)
ds = raw_ds.map(make_smolvlm_messages, desc="Formatting messages")
split = ds.train_test_split(test_size=0.06, seed=1234)
train_ds = split["train"]
print(f"Train examples : {len(train_ds):,}")

# ---------------------------------------------------------------------------
# Processor  (model weights are not needed for tokenisation, but we use the
# same initialiser the training script uses so the vocabulary is identical)
# ---------------------------------------------------------------------------
processor, _ = initialize_model_thinking()
# Disable image splitting — each image becomes a single 384×384 tile, which
# is what the training run uses and is much faster for a bulk scan.
# ---------------------------------------------------------------------------
# Tokenise every example and record its length
# ---------------------------------------------------------------------------
seq_lengths: list[int] = []

for example in tqdm(train_ds, desc="Tokenising"):
    messages = example["messages"]
    image = example.get("image")

    formatted = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    if image is not None:
        image = image.convert("RGB")

    inputs = processor(
        text=formatted,
        images=[image] if image is not None else None,
        return_tensors="pt",
    )

    seq_lengths.append(int(inputs["input_ids"].shape[1]))

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
avg_len = statistics.mean(seq_lengths)
median_len = statistics.median(seq_lengths)
min_len = min(seq_lengths)
max_len = max(seq_lengths)
p95_len = int(np.percentile(seq_lengths, 95))

total_tokens = sum(seq_lengths)
global_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * NUM_DEVICES
approx_tokens_per_step = global_batch_size * avg_len

print()
print("=" * 50)
print("  Sequence-length statistics")
print("=" * 50)
print(f"  examples   : {len(seq_lengths):>12,}")
print(f"  total      : {total_tokens:>14,}")
print(f"  average    : {avg_len:>12.1f}  tokens")
print(f"  median     : {median_len:>12.1f}  tokens")
print(f"  p95        : {p95_len:>12,}  tokens")
print(f"  min        : {min_len:>12,}  tokens")
print(f"  max        : {max_len:>12,}  tokens")
# print()
# print("=" * 50)
# print("  Global batch size (in tokens)")
# print("=" * 50)
# print(f"  per_device_batch_size : {PER_DEVICE_TRAIN_BATCH_SIZE}")
# print(f"  num_devices           : {NUM_DEVICES}")
# print(f"  global_batch_size     : {global_batch_size}  examples / step")
# print(f"  total tokens in train : {total_tokens:>14,}")
# print(
#     f"  ≈ tokens / step       : {int(approx_tokens_per_step):>14,}"
#     f"  (global_batch_size × avg_len)"
# )

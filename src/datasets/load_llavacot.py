"""Loader and preprocessor for the LLaVA-CoT-100k dataset.

Images are stored as split zip archives on the HuggingFace repo
(``image.zip.part-aa`` … ``image.zip.part-ap``).  On first call,
:func:`load_llava_cot` downloads the parts, merges them into a single
``image.zip``, extracts into ``src/datasets/llava_cot_images/``, and casts
the ``"image"`` column from path strings to PIL objects.  Subsequent calls
are no-ops (the directory already exists).

Message format
--------------
:func:`make_smolvlm_messages` converts one dataset example into a messages
list for SmolVLM's chat template. Multi-turn conversations are preserved in
full; the same preprocessing is applied to every gpt turn:

* System turn     – ``R1_STYLE_SYSTEM_PROMPT_COT`` (from ``config``).
* First user turn – image placeholder + question text.
* Subsequent user turns – question text only (same image context).
* Each assistant turn – ``<think>…</think>`` (``<REASONING>``) + answer
  (``<CONCLUSION>``; bare numbers wrapped in ``\\boxed{}``).
  ``<SUMMARY>`` and ``<CAPTION>`` are dropped.

The image placeholder (``{"type": "image"}``) is added to the first user
turn. The actual PIL image is **not** embedded in messages; pass
``example["image"]`` to the processor separately.

Returns ``{"messages": [...]}`` for compatibility with ``Dataset.map()``.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

import datasets as hf_datasets
from config import R1_STYLE_SYSTEM_PROMPT_COT

HF_REPO = "Xkev/LLaVA-CoT-100k"

_DATASETS_DIR = Path(__file__).parent
_IMAGES_DIR = _DATASETS_DIR / "llava_cot_images"
_PART_NAMES = [f"image.zip.part-a{chr(ord('a') + i)}" for i in range(16)]

_BARE_NUMBER_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")
_REASONING_RE = re.compile(r"<REASONING>(.*?)</REASONING>", re.DOTALL)
_CONCLUSION_RE = re.compile(r"<CONCLUSION>(.*?)</CONCLUSION>", re.DOTALL)
_TRAILING_CHAT_END_RE = re.compile(
    r"(?:\s*(?:<end_of_utterance>|<\|im_end\|>|<\|endoftext\|>))+\s*$"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_images() -> None:
    """Download, merge, and extract images on first call; no-op otherwise."""
    if _IMAGES_DIR.exists() and any(_IMAGES_DIR.iterdir()):
        return

    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Download any missing parts from HF.
    part_paths: list[Path] = []
    for name in _PART_NAMES:
        local = _DATASETS_DIR / name
        if not local.exists():
            print(f"Downloading {name} …")
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=name,
                repo_type="dataset",
                local_dir=str(_DATASETS_DIR),
            )
            local = Path(downloaded)
        part_paths.append(local)

    # Merge parts into a single zip.
    zip_path = _DATASETS_DIR / "image.zip"
    print("Merging parts …")
    with open(zip_path, "wb") as out:
        for part in part_paths:
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)

    # Extract and clean up the merged zip.
    print(f"Extracting to {_IMAGES_DIR} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(_IMAGES_DIR)
    zip_path.unlink()
    print("Images ready.")


def _extract_tag(text: str, pattern: re.Pattern) -> str:
    """Return content of the first regex match, stripped, or empty string."""
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _format_conclusion(conclusion: str) -> str:
    """Wrap a bare numeric conclusion in ``\\boxed{}``."""
    if _BARE_NUMBER_RE.match(conclusion):
        return rf"\boxed{{{conclusion.strip()}}}"
    return conclusion


def _strip_trailing_chat_end_tokens(text: str) -> str:
    """Remove trailing chat end tokens to avoid duplicated EOS in templates."""
    return _TRAILING_CHAT_END_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_llava_cot(
    split: str = "train",
    max_samples: int | None = None,
) -> hf_datasets.Dataset:
    """Load the LLaVA-CoT-100k dataset from HuggingFace.

    On first call, downloads the image zip parts from HF, merges and extracts
    them to ``src/datasets/llava_cot_images/``, then casts the ``"image"``
    column to PIL.  Subsequent calls skip all of that.

    Args:
        split: Dataset split to load (default ``"train"``).
        max_samples: If set, return only this many examples.

    Returns:
        HuggingFace ``Dataset`` with the ``"image"`` column as PIL Images.
    """
    _ensure_images()

    dataset = hf_datasets.load_dataset(HF_REPO, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Resolve relative image paths to absolute paths, then cast to PIL.
    dataset = dataset.map(
        lambda x: {"image": str(_IMAGES_DIR / x["image"])},
        desc="Resolving image paths",
    )
    dataset = dataset.cast_column("image", hf_datasets.Image())

    # TRL's SFTTrainer collator expects a per-example list of images under the
    # key "images" (plural).  Add it here so that (a) the column is present and
    # (b) changing raw_ds's fingerprint busts any stale .map() cache downstream.
    dataset = dataset.map(
        lambda x: {"images": [x["image"].convert("RGB")]},
        desc="Wrapping images in list for TRL",
    )

    return dataset


def make_smolvlm_messages(example: dict) -> dict:
    """Convert one LLaVA-CoT-100k example to SmolVLM chat-template messages.

    Handles both single-turn and multi-turn conversations. All human/gpt pairs
    are preserved in order. The same preprocessing is applied to every gpt turn:

    ==================  ===============================================
    LLaVA-CoT field     SmolVLM messages
    ==================  ===============================================
    ``<REASONING>``     ``<think>…</think>`` in assistant turn
    ``<CONCLUSION>``    Visible answer (bare numbers → ``\\boxed{}``)
    ``<SUMMARY>``       Dropped
    ``<CAPTION>``       Dropped
    ==================  ===============================================

    The image placeholder (``{"type": "image"}``) is added to the first user
    turn. The actual PIL image is **not** embedded in messages; pass
    ``example["image"]`` to the processor separately.  Subsequent user turns
    contain text only (they refer to the same image context).

    Compatible with ``Dataset.map()``: returns ``{"messages": [...]}`` so that
    the messages list is stored as a new column.

    Args:
        example: One row from the LLaVA-CoT-100k dataset.

    Returns:
        Dict with a single ``"messages"`` key containing the full conversation.
    """
    conversations = example.get("conversations", [])

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": R1_STYLE_SYSTEM_PROMPT_COT}],
        }
    ]

    first_user = True
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")

        if role == "human":
            question = re.sub(r"^<image>\n?", "", value).strip()
            if first_user:
                content = [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]
                first_user = False
            else:
                content = [{"type": "text", "text": question}]
            messages.append({"role": "user", "content": content})

        elif role == "gpt":
            reasoning = _strip_trailing_chat_end_tokens(
                _extract_tag(value, _REASONING_RE)
            )
            conclusion = _format_conclusion(
                _strip_trailing_chat_end_tokens(_extract_tag(value, _CONCLUSION_RE))
            )
            assistant_text = f"<think>\n{reasoning}\n</think>\n{conclusion}"
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )

    return {"messages": messages}

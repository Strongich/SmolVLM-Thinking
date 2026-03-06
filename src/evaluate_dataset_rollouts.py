"""
Evaluate VLMs on the MMK12 math dataset using 8 independent rollouts per question.

Produces difficulty distribution plots (correct count histogram) for:
- HuggingFaceTB/SmolVLM-Instruct
- Qwen/Qwen3-VL-8B-Thinking-FP8
"""

import gc
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PLOTS_DIR, SAVED_RESULTS_DIR, apply_paper_style, get_color

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. Let's think step by step and output the final answer within \\boxed{}."""

# ---------------------------------------------------------------------------
# Answer evaluation utilities (MATH benchmark standard)
# ---------------------------------------------------------------------------


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    if s is None:
        return ""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    return s[len(left) : -1]


def safe_float_equal(a, b, tol=1e-5):
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        return False


def extract_xml_answer(solution_str: str) -> str:
    string_in_last_boxed = last_boxed_only_string(solution_str)
    if string_in_last_boxed is not None:
        return remove_boxed(string_in_last_boxed)
    return None


def remove_right_units(string):
    # "\\text{ " usually indicates units at the end
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except (AssertionError, ValueError):
        return string


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def normalize_ground_truth(answer: str) -> str:
    answer = answer.strip()
    if answer.startswith("$$") and answer.endswith("$$"):
        answer = answer[2:-2].strip()
    elif answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    answer = re.sub(r"\\number\{([^}]*)\}", r"\1", answer)
    return strip_string(answer)


def is_correct(model_output: str, ground_truth: str) -> bool:
    predicted = extract_xml_answer(model_output)
    if predicted is None:
        return False
    predicted = strip_string(predicted)
    gt = normalize_ground_truth(ground_truth)
    return predicted == gt or safe_float_equal(predicted, gt)


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-Instruct"
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Thinking-FP8"
# QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"


def truncate_thinking(text: str) -> str:
    """Keep only the text after </think> for reasoning models."""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx + len(marker) :]
    return text


def build_smolvlm_prompt(question: str, processor) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": R1_STYLE_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def build_qwen_prompt(question: str, processor) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": R1_STYLE_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {
                    "type": "text",
                    "text": question,
                },
            ],
        },
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_rollouts(
    model_name: str,
    dataset,
    output_path: Path,
    sampling_params: SamplingParams,
    is_thinking_model: bool = False,
):
    """Run num_rollouts per question and save {id: correct_count} JSON."""
    from transformers import AutoProcessor

    print(f"\n{'=' * 60}")
    print(f"Running rollouts for: {model_name}")
    print(f"Dataset size: {len(dataset)}")
    print(f"{'=' * 60}\n")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=16384,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={"image": 1},
    )

    # Resume from existing results if available
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"  Resuming: loaded {len(results)} existing results from {output_path}")
    else:
        results = {}

    for idx, record in enumerate(dataset):
        if str(idx) in results:
            continue

        question = record["question"]
        answer = record["answer"]
        image = record["image"]  # PIL Image

        if is_thinking_model:
            prompt_text = build_qwen_prompt(question, processor)
        else:
            prompt_text = build_smolvlm_prompt(question, processor)

        outputs = llm.generate(
            [{"prompt": prompt_text, "multi_modal_data": {"image": image}}],
            sampling_params=sampling_params,
        )

        correct_count = 0
        for completion in outputs[0].outputs:
            gen_text = completion.text
            if is_thinking_model:
                gen_text = truncate_thinking(gen_text)
            if is_correct(gen_text, answer):
                correct_count += 1

        results[str(idx)] = correct_count

        # Save after every record for resumability
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} questions")

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_path} ({len(results)} questions)")

    # Free VRAM
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_difficulty(smolvlm_path: Path, qwen_path: Path, output_path: Path):
    """Create side-by-side difficulty distribution plot."""
    import matplotlib.pyplot as plt

    apply_paper_style()

    with open(smolvlm_path) as f:
        smolvlm_results = json.load(f)
    with open(qwen_path) as f:
        qwen_results = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x_labels = [f"{i}/8" for i in range(9)]
    x_positions = list(range(9))

    for ax, results, label, color in [
        (ax1, smolvlm_results, "SmolVLM-Instruct", get_color(2)),  # green
        (ax2, qwen_results, "Qwen3-VL-8B-Thinking-FP8", get_color(0)),  # blue
    ]:
        counts = Counter(results.values())
        y_values = [counts.get(i, 0) for i in range(9)]

        ax.plot(
            x_positions,
            y_values,
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=6,
            label=label,
        )
        ax.set_xlabel("Difficulty (x/8)")
        ax.set_ylabel("Sample Count")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading FanqingM/MMK12 dataset...")
    ds = load_dataset("FanqingM/MMK12", split="train")
    print(f"Loaded {len(ds)} questions")

    smolvlm_output = SAVED_RESULTS_DIR / "rollouts_smolvlm.json"
    qwen_output = SAVED_RESULTS_DIR / "rollouts_qwen3.json"
    plot_output = PLOTS_DIR / "difficulty_distribution.pdf"

    num_rollouts = 8

    smolvlm_sampling = SamplingParams(
        temperature=0.4,
        top_p=0.95,
        top_k=20,
        max_tokens=8192,
        n=num_rollouts,
    )

    qwen_sampling = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        max_tokens=8192,
        n=num_rollouts,
    )

    # Run SmolVLM rollouts
    run_rollouts(
        SMOLVLM_MODEL, ds, smolvlm_output, smolvlm_sampling, is_thinking_model=False
    )

    # # Run Qwen rollouts
    # run_rollouts(QWEN_MODEL, ds, qwen_output, qwen_sampling, is_thinking_model=True)

    # # Plot results
    # plot_difficulty(smolvlm_output, qwen_output, plot_output)


if __name__ == "__main__":
    main()

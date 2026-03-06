#!/usr/bin/env python3
"""
Evaluate a VLM across 8 multimodal benchmarks and save per-dataset + average
accuracy to a cumulative JSON file.

Usage:
    python run_vlm_average_evaluation.py --model-name "SmolVLM-RL-500steps" --temperature 0.6
    python run_vlm_average_evaluation.py --model-name "SmolVLM-base" --model SmolVLM --temperature 0.6
    python run_vlm_average_evaluation.py --model-name "SmolVLM-Thinking" --model ./SmolVLM-Thinking --temperature 0.6
"""

import argparse
import gc
import json
import os
import sys

import pandas as pd
import torch

import config
from VLMEvalKit.vlmeval.config import supported_VLM
from VLMEvalKit.vlmeval.dataset import build_dataset
from VLMEvalKit.vlmeval.inference import infer_data_job
from VLMEvalKit.vlmeval.smp import load_env
from VLMEvalKit.vlmeval.smp.file import get_pred_file_path

DATASETS = [
    "MMBench_DEV_EN_V11",
    "MMStar",
    "MMMU_DEV_VAL",
    "MathVista_MINI",
    "OCRBench",
    "AI2D_TEST",
    "HallusionBench",
    "ScienceQA_TEST",
]

RESULTS_FILENAME = "multimodal_evaluation.json"
JUDGE_MODEL = "gpt-4o-mini-2024-07-18"


def extract_accuracy(results, dataset_name):
    """Extract overall accuracy (0-100 scale) from VLMEvalKit evaluation results."""
    if isinstance(results, dict):
        # OCRBench returns a dict with "Final Score Norm" (0-10 scale)
        if "Final Score Norm" in results:
            return float(results["Final Score Norm"]) * 10
        return None

    if not isinstance(results, pd.DataFrame) or results.empty:
        return None

    # MathVista: rows keyed by "Task&Skill", accuracy in "acc" column
    if "Task&Skill" in results.columns and "acc" in results.columns:
        row = results[results["Task&Skill"] == "Overall"]
        if len(row):
            return float(row["acc"].iloc[0])

    # HallusionBench: "aAcc" column (already 0-100)
    if "aAcc" in results.columns:
        return float(results["aAcc"].iloc[0])

    # MCQ datasets: "Overall" column (0-1 scale)
    if "Overall" in results.columns:
        return float(results["Overall"].iloc[0]) * 100

    return None


def load_model(model_arg, temperature):
    """Load model from registry name or local path."""
    kwargs = dict(
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        top_k=20,
        max_new_tokens=8192,
    )

    if model_arg in supported_VLM:
        print(f"Loading model from registry: {model_arg}")
        return supported_VLM[model_arg](**kwargs)

    # Local path — instantiate SmolVLM directly
    from VLMEvalKit.vlmeval.vlm.smolvlm import SmolVLM

    print(f"Loading model from local path: {model_arg}")
    return SmolVLM(model_path=model_arg, **kwargs)


def evaluate_dataset(model, model_name, dataset_name):
    """Run inference + evaluation on a single dataset. Returns accuracy or None."""
    load_env()

    dataset = build_dataset(dataset_name)
    if dataset is None:
        print(f"  Dataset '{dataset_name}' not available, skipping.")
        return None

    work_dir = str(config.SAVED_RESULTS_DIR)

    try:
        # Check if prediction file already exists (resume support)
        pred_file = get_pred_file_path(
            work_dir, model_name, dataset.dataset_name, use_env_format=True
        )

        if os.path.exists(pred_file):
            print(f"  Found existing predictions: {pred_file}, skipping inference.")
        else:
            infer_data_job(
                model=model,
                work_dir=work_dir,
                model_name=model_name,
                dataset=dataset,
                verbose=True,
                api_nproc=4,
                use_vllm=True,
            )

        results = dataset.evaluate(pred_file, model=JUDGE_MODEL, nproc=4)
        acc = extract_accuracy(results, dataset_name)

        torch.cuda.empty_cache()
        return acc

    except Exception as e:
        print(f"  Error evaluating {dataset_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM across 8 multimodal benchmarks."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Display name / key in the results JSON (e.g. 'SmolVLM-RL-500steps')",
    )
    parser.add_argument(
        "--model",
        default="SmolVLM",
        help="supported_VLM registry name or local model folder path (default: SmolVLM)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (needed for GPT judge on some datasets).")
        sys.exit(1)

    # Load existing results to check for completed datasets
    results_path = config.get_results_path(RESULTS_FILENAME)
    existing = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            existing = json.load(f)

    # Recover scores from previous partial run (if any)
    prev_scores = existing.get(args.model_name, {})
    scores = {
        ds: prev_scores[ds]
        for ds in DATASETS
        if ds in prev_scores and isinstance(prev_scores[ds], (int, float))
    }

    remaining = [ds for ds in DATASETS if ds not in scores]
    if scores:
        print(
            f"Resuming: {len(scores)} datasets already completed, "
            f"{len(remaining)} remaining."
        )

    # Load model only if there are datasets left to evaluate
    if remaining:
        model = load_model(args.model, args.temperature)
    else:
        model = None

    model_name_tag = args.model_name.replace(" ", "_")

    # Evaluate remaining datasets
    for ds in remaining:
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {ds}")
        print(f"{'=' * 50}")

        acc = evaluate_dataset(model, model_name_tag, ds)
        if acc is not None:
            scores[ds] = round(acc, 2)
            print(f"  {ds}: {acc:.2f}")
        else:
            print(f"  {ds}: FAILED")

    # Cleanup model
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()

    # Compute average over dataset scores only
    dataset_scores = [scores[ds] for ds in DATASETS if ds in scores]
    if dataset_scores:
        scores["avg"] = round(sum(dataset_scores) / len(dataset_scores), 2)

    # Add generation parameters
    scores["params"] = {
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 8192,
        "do_sample": True,
    }

    # Save results
    existing[args.model_name] = scores

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    for ds, acc in scores.items():
        print(f"  {ds}: {acc}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

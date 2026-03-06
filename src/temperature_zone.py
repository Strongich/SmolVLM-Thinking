"""
Temperature zone evaluation script for MathVista benchmark.

This script evaluates a model across different sampling temperatures
on the MathVista benchmark and saves the results in JSON format.
"""

import gc
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import config
from VLMEvalKit.vlmeval.config import supported_VLM

# Import local modules
from VLMEvalKit.vlmeval.dataset import build_dataset
from VLMEvalKit.vlmeval.inference import infer_data_job

# Import VLMEvalKit components
from VLMEvalKit.vlmeval.smp import load_env
from VLMEvalKit.vlmeval.smp.file import get_pred_file_path


def run_mathvista_evaluation(
    model: Any,
    temperature: float = 0.1,
    judge_model: str = "gpt-4o-mini-2024-07-18",
    dataset_name: str = "MathVista_MINI",
    verbose: bool = True,
) -> Tuple[Optional[float], pd.DataFrame, str]:
    """
    Run MathVista evaluation for a given temperature.

    Args:
        model: The initialized VLM model
        temperature: Sampling temperature (for naming purposes)
        judge_model: Judge model for evaluation
        dataset_name: VLMEvalKit dataset name
        verbose: Whether to print verbose output

    Returns:
        Tuple of (overall_accuracy, results_dataframe, prediction_file_path)
    """
    # Load API keys from .env if present
    load_env()

    # Build dataset
    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(
            f"Dataset '{dataset_name}' is not available in VLMEvalKit. "
            f"Check the dataset name and try again."
        )

    # Use config saved_results directory for all outputs
    work_dir = str(config.SAVED_RESULTS_DIR)

    if verbose:
        print(f"Running evaluation with temperature: {temperature}")
        print(f"Work directory: {work_dir}")

    try:
        # Run inference with the model
        model_name_with_temp = f"{model.__class__.__name__}_temp_{temperature:.1f}"
        infer_data_job(
            model=model,
            work_dir=work_dir,
            model_name=model_name_with_temp,
            dataset=dataset,
            verbose=verbose,
            api_nproc=4,
            use_vllm=False,
        )

        # Locate prediction file
        pred_file = get_pred_file_path(
            work_dir,
            model_name_with_temp,
            dataset.dataset_name,
            use_env_format=True,
        )

        if verbose:
            print(f"Prediction file: {pred_file}")

        # Evaluate using judge LLM
        results = dataset.evaluate(pred_file, model=judge_model, nproc=4)

        # Extract Overall accuracy as a float.
        # Different VLMEvalKit datasets return different DataFrame formats:
        #   - MathVista: rows keyed by "Task&Skill" column, accuracy in "acc" column
        #   - MCQ datasets (MMMU, MMBench, ...): "Overall" as a column, splits as rows
        overall_acc = None
        if isinstance(results, pd.DataFrame) and not results.empty:
            if "Task&Skill" in results and "acc" in results:
                row = results[results["Task&Skill"] == "Overall"]
                if len(row):
                    overall_acc = float(row["acc"].iloc[0])
            elif "Overall" in results.columns:
                overall_acc = float(results["Overall"].iloc[0]) * 100

        if verbose:
            print(f"Overall accuracy for temperature {temperature}: {overall_acc}")

        return overall_acc, results, pred_file

    except Exception as e:
        print(f"Error during evaluation with temperature {temperature}: {str(e)}")
        return None, pd.DataFrame(), ""


def load_existing_results(results_filename: str) -> Dict[str, Any]:
    """
    Load existing results if available for resuming evaluation.

    Args:
        results_filename: Name of the results file to check

    Returns:
        Dictionary with existing results or empty dict if none found
    """
    results_path = config.get_results_path(results_filename)

    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                existing_results = json.load(f)
            print(f"📁 Found existing results file: {results_path}")
            print(
                f"📊 Loaded {len(existing_results.get('evaluations', {}))} previous evaluations"
            )
            return existing_results
        except Exception as e:
            print(f"⚠️  Could not load existing results: {str(e)}")

    return {}


def cleanup_intermediate_files(results_filename: str, keep_final: bool = True) -> None:
    """
    Clean up intermediate result files.

    Args:
        results_filename: Base filename for results
        keep_final: Whether to keep the final results file
    """
    saved_results_dir = config.SAVED_RESULTS_DIR
    pattern_prefix = "intermediate_temp_"

    cleaned_count = 0
    for file_path in saved_results_dir.glob(f"{pattern_prefix}*_{results_filename}"):
        try:
            file_path.unlink()
            cleaned_count += 1
            print(f"🗑️  Cleaned: {file_path.name}")
        except Exception as e:
            print(f"⚠️  Could not clean {file_path.name}: {e}")

    # Also clean progress files
    progress_file = saved_results_dir / f"progress_{results_filename}"
    if progress_file.exists():
        try:
            progress_file.unlink()
            cleaned_count += 1
            print(f"🗑️  Cleaned: {progress_file.name}")
        except Exception as e:
            print(f"⚠️  Could not clean {progress_file.name}: {e}")

    if cleaned_count > 0:
        print(f"✅ Cleaned {cleaned_count} intermediate files")
    else:
        print("ℹ️  No intermediate files to clean")


def temperature_zone_evaluation(
    temperatures: np.ndarray,
    model_name: str = "SmolVLM",
    judge_model: str = "gpt-4o-mini-2024-07-18",
    dataset_name: str = "MathVista_MINI",
    results_filename: str = "temperature_evaluation_results.json",
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model across multiple temperatures and save results.

    Args:
        temperatures: Array of temperatures to evaluate
        model_name: Name of the model to evaluate
        judge_model: Judge model for evaluation
        dataset_name: VLMEvalKit dataset name
        results_filename: Name of the JSON file to save results
        resume: Whether to resume from existing results if found

    Returns:
        Dictionary containing all evaluation results
    """
    print("Starting temperature zone evaluation")
    print(f"Model: {model_name}")
    print(f"Temperatures: {temperatures}")
    print(f"Judge model: {judge_model}")

    # Load existing results if resuming
    if resume:
        results = load_existing_results(results_filename)
        if results:
            print("🔄 Resuming from existing results")
        else:
            print("🆕 Starting fresh evaluation")
    else:
        results = {}
        print("🆕 Starting fresh evaluation (resume disabled)")

    # Initialize results structure if empty
    if not results:
        results = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "judge_model": judge_model,
            "temperatures": temperatures.tolist(),
            "evaluations": {},
            "summary": {
                "best_temperature": None,
                "best_accuracy": None,
                "worst_temperature": None,
                "worst_accuracy": None,
                "mean_accuracy": None,
                "std_accuracy": None,
            },
        }

    # Build list of accuracies from existing results
    accuracies = []
    for temp_str, evaluation in results.get("evaluations", {}).items():
        if evaluation.get("overall_accuracy") is not None:
            accuracies.append(evaluation["overall_accuracy"])

    # Determine which temperatures to skip (only those with a successful accuracy)
    completed_temps = set(
        float(t)
        for t, e in results.get("evaluations", {}).items()
        if e.get("overall_accuracy") is not None
    )
    remaining_temps = [t for t in temperatures if t not in completed_temps]

    if completed_temps:
        print(
            f"🔄 Skipping {len(completed_temps)} already completed temperatures: {sorted(completed_temps)}"
        )

    if remaining_temps:
        print(
            f"⏳ Evaluating {len(remaining_temps)} remaining temperatures: {remaining_temps}"
        )
    else:
        print("✅ All temperatures already completed!")

    # Initialize model once before the loop
    model = None
    if remaining_temps:
        if model_name not in supported_VLM:
            print(f"Error: Model '{model_name}' not found in supported_VLM")
            print(
                f"Available SmolVLM models: {[k for k in supported_VLM.keys() if 'SmolVLM' in k]}"
            )
            return results

        first_temp = remaining_temps[0]
        print(f"\nInitializing model '{model_name}' (temperature={first_temp:.1f})...")
        model = supported_VLM[model_name](
            temperature=first_temp,
            do_sample=True,
        )

    # Evaluate for each temperature
    for temp in temperatures:
        # Skip if already completed
        if temp in completed_temps:
            print(f"⏭️  Skipping temperature {temp:.1f} (already completed)")
            continue

        print(f"\n{'=' * 50}")
        print(f"Evaluating temperature: {temp:.1f}")
        print(f"{'=' * 50}")

        # Update temperature on the existing model
        model.kwargs["temperature"] = temp

        overall_acc, eval_results, pred_file = run_mathvista_evaluation(
            model=model,
            temperature=temp,
            judge_model=judge_model,
            dataset_name=dataset_name,
            verbose=True,
        )

        # Store results
        results["evaluations"][f"{temp:.1f}"] = {
            "temperature": temp,
            "overall_accuracy": overall_acc,
            "prediction_file": pred_file,
            "detailed_results": (
                eval_results.to_dict() if isinstance(eval_results, pd.DataFrame) else {}
            ),
        }

        # Free VRAM between evaluations (model stays loaded)
        torch.cuda.empty_cache()

        if overall_acc is not None:
            accuracies.append(overall_acc)
            print(f"✓ Temperature {temp:.1f}: {overall_acc:.4f}")
        else:
            print(f"✗ Temperature {temp:.1f}: Failed")

        # Save intermediate results after each temperature
        intermediate_filename = f"intermediate_temp_{temp:.1f}_{results_filename}"
        intermediate_path = config.get_results_path(intermediate_filename)

        try:
            with open(intermediate_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Intermediate results saved: {intermediate_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save intermediate results: {str(e)}")

        # Also save a running summary
        if accuracies:
            current_best_idx = np.argmax(accuracies)
            current_best_temp = [
                t
                for t, r in results["evaluations"].items()
                if r["overall_accuracy"] is not None
            ][current_best_idx]
            current_best_acc = accuracies[current_best_idx]
            print(
                f"📊 Current best: temp {current_best_temp} (accuracy: {current_best_acc:.4f})"
            )
            print(f"📈 Progress: {len(accuracies)}/{len(temperatures)} completed")

        # Save a progress summary file
        progress_summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "completed_temperatures": [float(t) for t in results["evaluations"].keys()],
            "successful_evaluations": len(accuracies),
            "total_temperatures": len(temperatures),
            "current_best": {
                "temperature": float(current_best_temp) if accuracies else None,
                "accuracy": float(current_best_acc) if accuracies else None,
            },
            "progress_percentage": (len(accuracies) / len(temperatures)) * 100,
        }

        progress_filename = f"progress_{results_filename}"
        progress_path = config.get_results_path(progress_filename)

        try:
            with open(progress_path, "w") as f:
                json.dump(progress_summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Warning: Could not save progress summary: {str(e)}")

    # Final cleanup after all evaluations
    if model is not None:
        del model
        model = None
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate summary statistics
    if accuracies:
        accuracies_array = np.array(accuracies)
        temps_with_results = [
            temp for temp, acc in zip(temperatures, accuracies) if acc is not None
        ]

        best_idx = np.argmax(accuracies_array)
        worst_idx = np.argmin(accuracies_array)

        results["summary"] = {
            "best_temperature": float(temps_with_results[best_idx]),
            "best_accuracy": float(accuracies_array[best_idx]),
            "worst_temperature": float(temps_with_results[worst_idx]),
            "worst_accuracy": float(accuracies_array[worst_idx]),
            "mean_accuracy": float(np.mean(accuracies_array)),
            "std_accuracy": float(np.std(accuracies_array)),
            "total_evaluations": len(accuracies),
            "successful_evaluations": len(accuracies),
        }

    # Save results to JSON file
    results_path = config.get_results_path(results_filename)

    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {results_path}")
    except Exception as e:
        print(f"✗ Error saving results: {str(e)}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    if accuracies:
        summary = results["summary"]
        print(
            f"Best temperature: {summary['best_temperature']:.1f} (accuracy: {summary['best_accuracy']:.4f})"
        )
        print(
            f"Worst temperature: {summary['worst_temperature']:.1f} (accuracy: {summary['worst_accuracy']:.4f})"
        )
        print(
            f"Mean accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}"
        )
        print(
            f"Successful evaluations: {summary['successful_evaluations']}/{len(temperatures)}"
        )
    else:
        print("No successful evaluations completed.")

    return results


def plot_temperature_results(
    results_filename: str = "mathvista_temperature_evaluation.json",
) -> None:
    """
    Load evaluation results and create a publication-quality plot.

    Args:
        results_filename: Name of the JSON results file in saved_results/
    """
    import matplotlib.pyplot as plt

    config.apply_paper_style()

    results_path = config.get_results_path(results_filename)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    # Extract temperatures and accuracies (sorted by temperature)
    evals = results.get("evaluations", {})
    temps_accs = sorted(
        [
            (e["temperature"], e["overall_accuracy"])
            for e in evals.values()
            if e.get("overall_accuracy") is not None
        ]
    )
    if not temps_accs:
        print("No successful evaluations to plot.")
        return

    temps, accs = zip(*temps_accs)

    # Derive dataset name from results metadata or filename
    DISPLAY_NAMES = {
        "mathvista_mini": "MathVista",
        "mmmu_dev_val": "MMMU_VAL",
    }
    raw_name = results.get(
        "dataset_name",
        results_filename.replace("_temperature_evaluation.json", ""),
    )
    dataset_name = DISPLAY_NAMES.get(raw_name.lower(), raw_name)

    # Find peak
    peak_idx = int(np.argmax(accs))
    peak_temp = temps[peak_idx]
    peak_acc = accs[peak_idx]

    fig, ax = plt.subplots()
    ax.plot(temps, accs, marker="o", label=dataset_name, color=config.get_color(0))

    # Annotate peak (position text below the line)
    ax.annotate(
        f"Peak: {peak_acc:.1f} at t={peak_temp}",
        xy=(peak_temp, peak_acc),
        xytext=(peak_temp - 0.25, peak_acc - 2.5),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="gray", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )

    ax.set_xlabel("Sampling Temperature (t)")
    ax.set_ylabel(f"{dataset_name} Accuracy (%)")
    ax.set_title(
        f"SmolVLM-Instruct Performance on {dataset_name} vs. Sampling Temperature"
    )
    ax.legend(loc="upper right")

    # Save as PNG only, with dataset-specific filename
    out_dir = config.TEMP_EVAL_PLOTS_DIR
    safe_name = dataset_name.lower().replace(" ", "_")
    path = out_dir / f"smolvlm_temperature_{safe_name}.png"
    fig.savefig(path, format="png")
    print(f"Saved plot: {path}")

    plt.close(fig)


def main(dataset_name: str = "MathVista_MINI"):
    """
    Main function to run temperature zone evaluation.

    Args:
        dataset_name: VLMEvalKit dataset name
    """
    print(f"Temperature Zone Evaluation for {dataset_name}")
    print("=" * 60)

    # 1. Initialize model
    print("Step 1: Model will be initialized during evaluation...")

    # 2. Create list of sampling temperatures
    temperatures = np.array([0.1, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    print(f"Step 2: Temperatures: {temperatures}")

    # 3. Run evaluation for MathVista benchmark
    print("Step 3: Starting MathVista benchmark evaluation...")

    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment!")
        print("Please set OPENAI_API_KEY for judge model evaluation.")
        return

    # 4. Save results in JSON
    print("Step 4: Results will be saved in JSON format...")

    # Derive results filename from dataset name
    results_filename = f"{dataset_name.lower()}_temperature_evaluation.json"

    # Run the evaluation
    results = temperature_zone_evaluation(
        temperatures=temperatures,
        model_name="SmolVLM",
        judge_model="gpt-4o-mini-2024-07-18",
        dataset_name=dataset_name,
        results_filename=results_filename,
        resume=True,  # Enable resume functionality
    )

    print("\nEvaluation completed!")

    # Optionally clean up intermediate files
    print("\n🧹 Cleaning up intermediate files...")
    cleanup_intermediate_files(results_filename)

    # Generate plot from results
    print("\n📊 Generating temperature evaluation plot...")
    plot_temperature_results(results_filename)

    return results


if __name__ == "__main__":
    main()

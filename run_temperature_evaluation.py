#!/usr/bin/env python3
"""
Example script to run temperature zone evaluation.
Make sure to set your OPENAI_API_KEY environment variable before running.

Usage:
    python run_temperature_evaluation.py [DATASET_NAME]

    DATASET_NAME defaults to "MathVista_MINI" if not provided.
"""

import argparse
import os
import sys
from pathlib import Path

from src.temperature_zone import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run temperature zone evaluation on a VLMEvalKit dataset."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="MathVista_MINI",
        help="VLMEvalKit dataset name (default: MathVista_MINI)",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    print("Starting temperature zone evaluation...")
    print(f"Dataset: {args.dataset}")
    print(
        "This will evaluate SmolVLM across temperatures: 0.1, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6"
    )
    print(
        "Results will be saved to saved_results/mathvista_temperature_evaluation.json"
    )
    print("-" * 60)

    # Run the evaluation
    results = main(dataset_name=args.dataset)

    if results:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed. Please check the error messages above.")

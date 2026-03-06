from typing import Tuple, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_model_bold(
    model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
) -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
    """
    Initialize the vision-language model and processor.

    Args:
        model_name (str): The model name to load from HuggingFace

    Returns:
        Tuple[AutoProcessor, AutoModelForImageTextToText]: The initialized processor and model
    """
    print(f"Using device: {DEVICE}")
    print(f"Loading model: {model_name}")

    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).to(DEVICE)

    print("Model and processor initialized successfully!")
    return processor, model


def initialize_model_thinking(
    model_name: str = "SmolVLM-Thinking",
) -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
    """
    Initialize the vision-language model and processor.

    Args:
        model_name (str): The model name to load from HuggingFace

    Returns:
        Tuple[AutoProcessor, AutoModelForImageTextToText]: The initialized processor and model
    """
    print(f"Using device: {DEVICE}")
    print(f"Loading model: {model_name}")

    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).to(DEVICE)

    print("Model and processor initialized successfully!")
    return processor, model

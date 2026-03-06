"""
Script to add special tokens to SmolVLM model and save the updated model.
"""

from src.model_init.model import initialize_model_bold


def add_special_tokens_to_model():
    """Add thinking tokens to the model and save the updated version."""
    # Initialize the model and processor
    processor, model = initialize_model_bold()

    # Add special tokens for thinking capabilities
    special_tokens = ["<think>", "</think>", "<answer>"]
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )

    print("Added special tokens:")
    print(processor.tokenizer.special_tokens_map)

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(processor.tokenizer))

    # Save the updated model and tokenizer
    save_path = "SmolVLM-Thinking"
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path, from_pt=True)

    print(f"Saved updated model and tokenizer to: {save_path}")


if __name__ == "__main__":
    add_special_tokens_to_model()

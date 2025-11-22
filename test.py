#!/usr/bin/env python3
"""
Simple test script for Qwen Image Edit model.
Loads input.txt and input.png and applies image editing with configurable hyperparameters.
"""

import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# Configuration
INPUT_TEXT_PATH = "input.txt"
INPUT_IMAGE_PATH = "input.png"
OUTPUT_PATH = "test_output.png"

# Hyperparameters
NUM_INFERENCE_STEPS = 50
TRUE_CFG_SCALE = 4.0
NEGATIVE_PROMPT = "Chinese text"


def test_qwen_image_edit():
    """Test Qwen Image Edit pipeline with input.txt and input.png"""
    print("="*60)
    print("Testing Qwen Image Edit Pipeline")
    print("="*60)

    # Load prompt from input.txt
    print(f"\nLoading prompt from {INPUT_TEXT_PATH}...")
    with open(INPUT_TEXT_PATH, "r") as f:
        prompt = f.read().strip()
    print(f"Prompt: {prompt}")

    # Load image from input.png
    print(f"\nLoading image from {INPUT_IMAGE_PATH}...")
    input_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    print(f"Image size: {input_image.size}")

    # Load the pipeline
    print("\nLoading Qwen Image Edit pipeline...")
    print("(This may take a few minutes on first run)")
    pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")

    # Move to GPU if available, use bfloat16 for efficiency
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        pipeline = pipeline.to(torch.bfloat16)
    pipeline = pipeline.to(device)
    print(f"Pipeline loaded on {device}")

    # Print hyperparameters
    print(f"\nHyperparameters:")
    print(f"  num_inference_steps: {NUM_INFERENCE_STEPS}")
    print(f"  true_cfg_scale: {TRUE_CFG_SCALE}")
    print(f"  negative_prompt: {NEGATIVE_PROMPT}")

    # Prepare inputs
    inputs = {
        "image": input_image,
        "prompt": prompt,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "negative_prompt": NEGATIVE_PROMPT,
    }

    # Generate image
    print("\nGenerating image...")
    with torch.inference_mode():
        result = pipeline(**inputs)

    # Save output
    output_image = result.images[0]
    output_image.save(OUTPUT_PATH)
    print(f"\n✓ Success! Output saved to: {OUTPUT_PATH}")
    print(f"Output size: {output_image.size}")
    print("="*60)


if __name__ == "__main__":
    try:
        test_qwen_image_edit()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nMake sure {INPUT_TEXT_PATH} and {INPUT_IMAGE_PATH} exist!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

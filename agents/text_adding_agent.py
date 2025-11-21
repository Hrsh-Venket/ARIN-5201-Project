"""
Stage 6: Text Adding Agent
Adds text to the image based on layout instructions with retry loop.
"""
import os
import torch
from PIL import Image
from state import AgentState
import config
import traceback


def text_adding_agent(state: AgentState) -> AgentState:
    """
    Stage 6: Text Adding Agent

    Adds text to the final image based on layout instructions from planning agent.
    Uses diffusers Qwen-Image-Edit to add text with proper positioning.
    This is part of a retry loop with text_validation_agent.

    Args:
        state: Current agent state with best_image, best_text, planning_output, and image_pipeline

    Returns:
        Updated state with poster_with_text and incremented text_adding_attempt_count
    """
    print("\n=== STAGE 6: TEXT ADDING AGENT ===")
    config.log_stage("STAGE 6: TEXT ADDING AGENT", "Starting text addition...")

    # Initialize attempt count if not set
    if "text_adding_attempt_count" not in state or state["text_adding_attempt_count"] is None:
        state["text_adding_attempt_count"] = 0

    state["text_adding_attempt_count"] += 1
    attempt_num = state["text_adding_attempt_count"]

    print(f"Text adding attempt: {attempt_num}/{config.MAX_TEXT_ADDING_ATTEMPTS}")
    config.log_message(f"Attempt: {attempt_num}/{config.MAX_TEXT_ADDING_ATTEMPTS}")

    # ALWAYS use the best image from Stage 4 (image generation)
    base_image_path = state.get("best_image") or state.get("current_image")

    if not base_image_path or not os.path.exists(base_image_path):
        print(f"Warning: Base image not found. Using input image.")
        config.log_message("WARNING: best_image not found, falling back to input image")
        base_image_path = state["input_image_path"]

    print(f"Adding text to image: {base_image_path}")
    config.log_message(f"Base image for text addition: {base_image_path}")

    # Create prompt for text addition
    text_content = state.get("best_text") or state.get("generated_text")

    # Extract layout information from planning output
    planning_text = state["planning_output"]

    text_addition_prompt = f"""Add the following text to this poster image according to the layout specifications:

LAYOUT SPECIFICATIONS:
{planning_text}

TEXT TO ADD:
{text_content}

Instructions:
- Follow the layout design specified in the plan
- Use appropriate text placement (header, body, footer as specified)
- Ensure text is readable and well-positioned
- Match the color scheme specified in the plan
- Maintain visual hierarchy
- Create a professional, polished poster design at 720x1280 resolution"""

    # Add feedback from previous attempt if exists
    if state.get("text_validation_feedback") and attempt_num > 1:
        text_addition_prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{state['text_validation_feedback'][:300]}\n\nPlease address this feedback in your text addition."
        config.log_message(f"\nIncluding validation feedback in prompt")
        text_addition_prompt += "Be really careful with text added to the img, don't generate blurry or misspelt text."

    print(f"Text addition prompt (first 300 chars): {text_addition_prompt[:300]}...")
    config.log_message(f"\nText addition prompt:\n{text_addition_prompt}")
    config.log_message(f"\nText to add:\n{text_content}")

    # Get pipeline from state
    pipeline = state.get("image_pipeline")
    config.log_message(f"\nPipeline loaded: {pipeline is not None}")

    try:
        if pipeline is None:
            print("Warning: Pipeline not initialized! Using base image as fallback.")
            config.log_message("ERROR: Pipeline not initialized!")
            raise Exception("Pipeline not available")

        # Load input image as PIL Image and convert to RGB
        input_image = Image.open(base_image_path).convert("RGB")
        config.log_message(f"Loaded base image: {input_image.size}")

        print("Adding text with diffusers pipeline...")
        config.log_message("Starting text addition with pipeline...")

        # Prepare inputs for QwenImageEditPipeline
        inputs = {
            "image": input_image,
            "prompt": text_addition_prompt,
            "num_inference_steps": config.HUGGINGFACE_INFERENCE_STEPS,
            "true_cfg_scale": 25.0,
            "negative_prompt": " ",
        }

        config.log_message(f"Pipeline parameters: steps={config.HUGGINGFACE_INFERENCE_STEPS}, cfg_scale=25.0")

        # Generate image using the pipeline with inference mode
        with torch.inference_mode():
            result = pipeline(**inputs)

        # Extract the generated image (pipeline returns a list)
        image_with_text = result.images[0]
        config.log_message(f"Text added successfully: {image_with_text.size}")

        # Save image with text (with attempt number for retry tracking)
        output_path = os.path.join(config.INTERMEDIATE_DIR, f"poster_with_text_attempt{attempt_num}.png")
        os.makedirs(config.INTERMEDIATE_DIR, exist_ok=True)
        image_with_text.save(output_path)
        print(f"Poster with text saved to: {output_path}")
        config.log_message(f"Image saved to: {output_path}")

    except Exception as e:
        error_msg = f"Generation Error: {str(e)}"
        print(error_msg)
        config.log_message(f"\n{error_msg}")
        config.log_message(f"Traceback:\n{traceback.format_exc()}")

        # If generation fails, just use the base image
        output_path = os.path.join(config.INTERMEDIATE_DIR, f"poster_with_text_attempt{attempt_num}.png")
        base_image = Image.open(base_image_path)
        base_image.save(output_path)
        print(f"Generation failed, using base image without text addition")
        config.log_message(f"Fallback: Using base image as attempt {attempt_num}")

    # Update state
    state["poster_with_text"] = output_path
    config.log_message(f"\nUpdated poster_with_text to: {output_path}")

    return state

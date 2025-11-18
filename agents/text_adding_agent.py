"""
Stage 6: Text Adding Agent
Adds text to the image based on layout instructions.
"""
import os
from PIL import Image
from state import AgentState
import config


def text_adding_agent(state: AgentState) -> AgentState:
    """
    Stage 6: Text Adding Agent

    Adds text to the final image based on layout instructions from planning agent.
    Uses diffusers Qwen-Image-Edit to add text with proper positioning.

    Args:
        state: Current agent state with best_image, best_text, planning_output, and image_pipeline

    Returns:
        Updated state with poster_with_text
    """
    print("\n=== STAGE 6: TEXT ADDING AGENT ===")

    # Use the best image from previous stages
    base_image_path = state.get("best_image") or state.get("current_image")

    if not base_image_path or not os.path.exists(base_image_path):
        print(f"Warning: Base image not found. Using input image.")
        base_image_path = state["input_image_path"]

    print(f"Adding text to image: {base_image_path}")

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

    print(f"Text addition prompt (first 300 chars): {text_addition_prompt[:300]}...")

    # Get pipeline from state
    pipeline = state.get("image_pipeline")

    try:
        if pipeline is None:
            print("Warning: Pipeline not initialized! Using base image as fallback.")
            raise Exception("Pipeline not available")

        # Load input image as PIL Image
        input_image = Image.open(base_image_path)

        print("Adding text with diffusers pipeline...")

        # Generate image using the pipeline
        result = pipeline(
            image=input_image,
            prompt=text_addition_prompt,
            num_inference_steps=config.HUGGINGFACE_INFERENCE_STEPS,
        )

        # Extract the generated image (pipeline returns a list)
        image_with_text = result.images[0]

        # Save image with text
        output_path = os.path.join(config.INTERMEDIATE_DIR, "poster_with_text.png")
        os.makedirs(config.INTERMEDIATE_DIR, exist_ok=True)
        image_with_text.save(output_path)
        print(f"Poster with text saved to: {output_path}")

    except Exception as e:
        print(f"Generation Error: {str(e)}")
        # If generation fails, just use the base image
        output_path = os.path.join(config.INTERMEDIATE_DIR, "poster_with_text.png")
        base_image = Image.open(base_image_path)
        base_image.save(output_path)
        print(f"Generation failed, using base image without text addition")

    # Update state
    state["poster_with_text"] = output_path

    return state

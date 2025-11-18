"""
Stage 4: Image Generation Agent
Generates images using diffusers Qwen-Image-Edit with retry loop.
"""
import os
from PIL import Image
from state import AgentState
import config


def image_generation_agent(state: AgentState) -> AgentState:
    """
    Stage 4: Image Generation Agent

    Generates/edits images using diffusers Qwen-Image-Edit model.
    - Starts with input.png as base image
    - If validation fails but logo not integrated, reverts to input.png
    - Otherwise, can edit previous output
    - Uses 4 inference steps

    Args:
        state: Current agent state with planning_output and image_pipeline

    Returns:
        Updated state with current_image and incremented image_attempt_count
    """
    print("\n=== STAGE 4: IMAGE GENERATION AGENT ===")

    # Initialize attempt count if not set
    if "image_attempt_count" not in state or state["image_attempt_count"] is None:
        state["image_attempt_count"] = 0

    state["image_attempt_count"] += 1
    attempt_num = state["image_attempt_count"]

    print(f"Image generation attempt: {attempt_num}/{config.MAX_IMAGE_ATTEMPTS}")

    # Determine which image to use as base
    base_image_path = state["input_image_path"]  # Default: start with input.png

    # Check if we should revert to input.png
    if attempt_num > 1 and state.get("validation_feedback"):
        if "must revert to input.png" in state["validation_feedback"].lower() or \
           "logo not properly integrated" in state["validation_feedback"].lower():
            print("Reverting to input.png as base image (logo not integrated)")
            base_image_path = state["input_image_path"]
        elif state.get("current_image"):
            # Can edit previous output if logo was integrated
            base_image_path = state["current_image"]
            print(f"Using previous output as base: {base_image_path}")

    # Extract image generation prompt from planning output
    planning_text = state["planning_output"]

    # Try to extract the IMAGE GENERATION PROMPT section
    prompt = ""
    if "IMAGE GENERATION PROMPT" in planning_text:
        prompt_section = planning_text.split("IMAGE GENERATION PROMPT")[1]
        # Get until next section or end
        for section_name in ["COLOR PALETTE", "LAYOUT DESIGN", "TEXT REQUIREMENTS"]:
            if section_name in prompt_section:
                prompt_section = prompt_section.split(section_name)[0]
                break
        prompt = prompt_section.strip()
    else:
        # Fallback: use planning output as context
        prompt = f"Create a poster background based on this design plan: {planning_text[:500]}"

    # Add feedback from previous attempt if exists
    if state.get("validation_feedback") and attempt_num > 1:
        prompt += f"\n\nIMPROVEMENTS NEEDED: {state['validation_feedback'][:300]}"

    print(f"Using base image: {base_image_path}")
    print(f"Generation prompt (first 200 chars): {prompt[:200]}...")

    # Get pipeline from state
    pipeline = state.get("image_pipeline")

    try:
        if pipeline is None:
            print("Warning: Pipeline not initialized! Using base image as fallback.")
            raise Exception("Pipeline not available")

        # Load input image as PIL Image
        input_image = Image.open(base_image_path)

        print("Generating image with diffusers pipeline...")

        # Generate image using the pipeline
        result = pipeline(
            image=input_image,
            prompt=prompt,
            num_inference_steps=config.HUGGINGFACE_INFERENCE_STEPS,
        )

        # Extract the generated image (pipeline returns a list)
        generated_image = result.images[0]

        # Save generated image
        output_path = os.path.join(config.INTERMEDIATE_DIR, f"attempt{attempt_num}.png")
        os.makedirs(config.INTERMEDIATE_DIR, exist_ok=True)
        generated_image.save(output_path)
        print(f"Generated image saved to: {output_path}")

    except Exception as e:
        print(f"Generation Error: {str(e)}")
        # If generation fails, just use the base image for this attempt
        output_path = os.path.join(config.INTERMEDIATE_DIR, f"attempt{attempt_num}.png")
        base_image = Image.open(base_image_path)
        base_image.save(output_path)
        print(f"Generation failed, using base image as attempt {attempt_num}")

    # Update state
    state["current_image"] = output_path

    return state

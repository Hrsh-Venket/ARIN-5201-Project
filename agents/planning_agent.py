"""
Stage 1: Planning Agent
Analyzes input text and logo to create a comprehensive poster design plan.
"""
import os
import base64
from openai import OpenAI
from state import AgentState
import config
import traceback


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def planning_agent(state: AgentState) -> AgentState:
    """
    Stage 1: Planning Agent

    Analyzes input text and logo to generate a comprehensive design plan including:
    - Color palette extraction from logo
    - Layout design with text placement positions
    - Text requirements and constraints
    - Image generation prompt grounded in both logo and input text

    Args:
        state: Current agent state with input_text and input_image_path

    Returns:
        Updated state with planning_output
    """
    print("\n=== STAGE 1: PLANNING AGENT ===")
    config.log_stage("STAGE 1: PLANNING AGENT", "Starting planning phase...")

    # Log inputs
    config.log_message(f"Input text: {state['input_text']}")
    config.log_message(f"Input image: {state['input_image_path']}")

    # Initialize SiliconFlow client
    client = OpenAI(
        base_url=config.SiliconFlow_BASE_URL,
        api_key=config.SiliconFlow_API_KEY,
    )

    config.log_message(f"\nSiliconFlow client initialized")
    config.log_message(f"Model: {config.SiliconFlow_MODEL}")

    # Encode the input image
    image_base64 = encode_image(state["input_image_path"])
    config.log_message(f"Image encoded successfully")

    # Create the planning prompt
    planning_prompt = f"""You are a professional poster design planner. Analyze the provided logo/mascot image and the following input text to create a comprehensive poster design plan.

INPUT TEXT: {state["input_text"]}

Your task is to create a detailed design plan that includes:

1. COLOR PALETTE: Extract and analyze the dominant colors from the logo. List 3-5 colors with their approximate hex codes that should be used in the poster design.

2. LAYOUT DESIGN: Design a layout for a 720x1280 poster that incorporates the logo/mascot. Specify:
   - Logo placement (x, y, width, height as percentages of total dimensions)
   - Text placement zones (header, body, footer) with coordinates
   - Background design approach
   - Visual hierarchy

3. TEXT REQUIREMENTS: Based on the input text, specify:
   - What text should be generated (headline, body text, call-to-action, etc.)
   - Character limits for each text element
   - Font style recommendations (bold, regular, etc.)
   - Text color recommendations

4. IMAGE GENERATION PROMPT: Create a detailed prompt for image generation that:
   - Incorporates elements from the logo/mascot
   - Relates to the input text theme
   - Specifies the art style consistent with the logo
   - Describes how to integrate with the existing logo visually
   - Avoids any suggestions of adding text to the image, and reminds the model to leave space for text placement as per the layout design and never add text.·

Be specific and detailed. This plan will guide all subsequent stages of poster generation.

Format your response with clear section headers: COLOR PALETTE, LAYOUT DESIGN, TEXT REQUIREMENTS, and IMAGE GENERATION PROMPT."""

    config.log_message(f"\nPrompt sent to LLM:\n{planning_prompt}")

    try:
        # Call SiliconFlow API
        response = client.chat.completions.create(
            model=config.SiliconFlow_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": planning_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
        )

        planning_output = response.choices[0].message.content
        config.log_message(f"\nLLM Response:\n{planning_output}")

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(error_msg)
        config.log_message(f"\n{error_msg}")
        config.log_message(f"Traceback:\n{traceback.format_exc()}")
        raise

    # Save planning output
    os.makedirs(config.INTERMEDIATE_DIR, exist_ok=True)
    planning_path = os.path.join(config.INTERMEDIATE_DIR, "planning.txt")
    with open(planning_path, "w") as f:
        f.write(planning_output)

    print(f"Planning output saved to: {planning_path}")
    print(f"\nPlanning Summary (first 500 chars):\n{planning_output[:500]}...")
    config.log_message(f"\nPlanning output saved to: {planning_path}")

    # Update state
    state["planning_output"] = planning_output

    return state

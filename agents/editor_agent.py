"""
Stage 3: Editor Agent (Validation)
Validates generated images against planning requirements.
"""
import base64
from openai import OpenAI
from state import AgentState
import config


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def editor_agent(state: AgentState) -> AgentState:
    """
    Stage 3: Editor Agent (Validation)

    Validates generated images against planning requirements.
    Checks for:
    - Relevance to the design plan
    - Quality of the generated image
    - Color palette compatibility
    - Logo integration

    Args:
        state: Current agent state with planning_output and current_image

    Returns:
        Updated state with validation_feedback and validation_passed
    """
    print("\n=== STAGE 3: EDITOR AGENT (VALIDATION) ===")

    # Initialize OpenRouter client
    client = OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
    )

    # Encode the current image and input logo
    current_image_base64 = encode_image(state["current_image"])
    input_logo_base64 = encode_image(state["input_image_path"])

    validation_prompt = f"""You are a professional design validator. Compare the generated poster image against the original logo and design plan.

DESIGN PLAN:
{state["planning_output"]}

ORIGINAL INPUT TEXT:
{state["input_text"]}

Evaluate the generated image based on:
1. LOGO INTEGRATION: Does it properly incorporate or complement the input logo/mascot?
2. COLOR PALETTE: Does it use colors compatible with the logo and plan?
3. RELEVANCE: Does it align with the image generation prompt in the plan?
4. QUALITY: Is the image quality acceptable for a poster?
5. COMPOSITION: Does it leave appropriate space for text placement as specified in the layout?

Respond in this format:
VALIDATION: [PASS or FAIL]
LOGO_INTEGRATED: [YES or NO]
FEEDBACK: [Detailed feedback. If FAIL, specify what needs to be fixed. If logo is not integrated, explicitly state this.]

Be thorough in your evaluation. The logo MUST be visibly integrated into the design."""

    response = client.chat.completions.create(
        model=config.OPENROUTER_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": validation_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{input_logo_base64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
    )

    validation_result = response.choices[0].message.content
    state["validation_feedback"] = validation_result

    # Check if validation passed
    validation_passed = "VALIDATION: PASS" in validation_result.upper()
    state["validation_passed"] = validation_passed

    # Check if logo is integrated
    logo_integrated = "LOGO_INTEGRATED: YES" in validation_result.upper()

    print(f"Validation result: {'PASSED' if validation_passed else 'FAILED'}")
    print(f"Logo integrated: {'YES' if logo_integrated else 'NO'}")
    print(f"Feedback: {validation_result[:300]}...")

    # Update best image if this one passed or is better
    if validation_passed or state.get("best_image") is None:
        state["best_image"] = state["current_image"]

    # Add logo integration info to feedback
    if not logo_integrated:
        state["validation_feedback"] += "\n\nIMPORTANT: Logo not properly integrated. Must revert to input.png as base."

    return state


def should_retry_image(state: AgentState) -> str:
    """
    Decision function for image generation retry loop.

    Returns:
        "retry" if should retry image generation, "continue" otherwise
    """
    if state.get("validation_passed"):
        return "continue"

    if state["image_attempt_count"] >= config.MAX_IMAGE_ATTEMPTS:
        print(f"\nMax image attempts ({config.MAX_IMAGE_ATTEMPTS}) reached. Continuing with best attempt.")
        return "continue"

    return "retry"

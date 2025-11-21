"""
Stage 3: Editor Agent (Validation)
Validates generated images against planning requirements.
"""
import base64
from openai import OpenAI
from state import AgentState
import config
import traceback


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
    config.log_stage("STAGE 3: EDITOR AGENT (VALIDATION)", "Starting image validation...")

    attempt_num = state.get("image_attempt_count", 1)
    complete_failure_count = state.get("image_complete_failure_count", 0)
    print(f"Validating image attempt {attempt_num} (complete failure count: {complete_failure_count})")
    config.log_message(f"Image attempt: {attempt_num}, Complete failure count: {complete_failure_count}")
    config.log_message(f"Current image: {state['current_image']}")
    config.log_message(f"Input image: {state['input_image_path']}")

    # Initialize OpenRouter client
    client = OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
    )

    # Encode the current image and input logo
    current_image_base64 = encode_image(state["current_image"])
    input_logo_base64 = encode_image(state["input_image_path"])

    config.log_message("\nImages encoded successfully")

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
6. DOES THE IMAGE ADD ANY TEXT ELEMENTS DIRECTLY ON IT? (This is NOT allowed as per the plan)

Respond in this format:
VALIDATION: [PASS or FAIL]
LOGO_INTEGRATED: [YES or NO]
FEEDBACK: [Detailed feedback. If FAIL, specify what needs to be fixed. If logo is not integrated, explicitly state this.]
TEXT_ON_IMAGE: [YES or NO]

Be thorough in your evaluation. The logo MUST be visibly integrated into the design."""

    config.log_message(f"\nValidation prompt sent to LLM:\n{validation_prompt}")

    try:
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
            max_tokens=800,
        )

        validation_result = response.choices[0].message.content
        config.log_message(f"\nLLM Response:\n{validation_result}")

        state["validation_feedback"] = validation_result
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(error_msg)
        config.log_message(f"\n{error_msg}")
        config.log_message(f"Traceback:\n{traceback.format_exc()}")

        # On error, set validation to failed
        validation_result = f"VALIDATION: FAIL\nERROR: {str(e)}"
        state["validation_feedback"] = validation_result

    # Check if validation passed
    validation_passed = "VALIDATION: PASS" in validation_result.upper()
    state["validation_passed"] = validation_passed

    # Check if logo is integrated
    logo_integrated = "LOGO_INTEGRATED: YES" in validation_result.upper()
    text_on_image = "TEXT_ON_IMAGE: YES" in validation_result.upper()

    print(f"Validation result: {'PASSED' if validation_passed else 'FAILED'}")
    print(f"Logo integrated: {'YES' if logo_integrated else 'NO'}")
    print(f"Text on image: {'YES' if text_on_image else 'NO'}")
    print(f"Feedback: {validation_result[:300]}...")

    config.log_message(f"\nValidation passed: {validation_passed}")
    config.log_message(f"Logo integrated: {logo_integrated}")
    config.log_message(f"Text on image: {text_on_image}")

    # Update best image if this one passed or is better
    if validation_passed or state.get("best_image") is None:
        state["best_image"] = state["current_image"]
        config.log_message(f"Updated best_image to: {state['current_image']}")

    # Add structured failure reasons to feedback
    if not logo_integrated:
        state["validation_feedback"] += "\n\nFAILURE_TYPE: logo_missing"
        state["validation_feedback"] += "\n\nIMPORTANT: Logo not properly integrated. Must revert to input.png as base."
        config.log_message("\nFAILURE TYPE: logo_missing")

    if text_on_image:
        state["validation_feedback"] += "\n\nFAILURE_TYPE: text_present"
        state["validation_feedback"] += "\n\nIMPORTANT: Text elements were found directly on the image, which is not allowed. Must revert to input.png as base."
        config.log_message("\nFAILURE TYPE: text_present")

    return state


def should_retry_image(state: AgentState) -> str:
    """
    Decision function for image generation retry loop with extended retries for complete failures.

    Returns:
        "retry" if should retry image generation, "continue" otherwise
    """
    attempt_count = state.get("image_attempt_count", 0)
    complete_failure_count = state.get("image_complete_failure_count", 0)
    validation_feedback = state.get("validation_feedback", "")

    # Check for complete failure conditions (text present or logo missing)
    is_complete_failure = ("text_present" in validation_feedback.lower() or
                          "logo_missing" in validation_feedback.lower() or
                          "logo not" in validation_feedback.lower())

    # If validation passed, continue to next stage
    if state.get("validation_passed"):
        print("\nValidation passed. Proceeding to next stage.")
        config.log_message("\nDecision: Validation passed, proceeding to segmentation.")
        return "continue"

    # Regular retry loop (max 3 attempts)
    if attempt_count < config.MAX_IMAGE_ATTEMPTS:
        print(f"\nRetrying image generation (attempt {attempt_count + 1}/{config.MAX_IMAGE_ATTEMPTS})...")
        config.log_message(f"\nDecision: Retrying image generation (attempt {attempt_count + 1}/{config.MAX_IMAGE_ATTEMPTS})")
        return "retry"

    # Extended retry for complete failures (max 15 more)
    elif is_complete_failure and complete_failure_count < config.MAX_IMAGE_COMPLETE_FAILURE_ATTEMPTS:
        state["image_complete_failure_count"] = complete_failure_count + 1
        state["image_attempt_count"] = 0  # Reset regular attempt counter
        print(f"\nComplete failure detected (text present or logo missing).")
        print(f"Extended retry {state['image_complete_failure_count']}/{config.MAX_IMAGE_COMPLETE_FAILURE_ATTEMPTS}")
        config.log_message(f"\nDecision: Complete failure detected, extended retry {state['image_complete_failure_count']}/{config.MAX_IMAGE_COMPLETE_FAILURE_ATTEMPTS}")
        config.log_message("Resetting regular attempt counter to 0")
        return "retry"
    else:
        print(f"\nMax attempts reached. Continuing with best attempt.")
        config.log_message(f"\nDecision: Max attempts reached (regular: {attempt_count}, complete failure: {complete_failure_count})")
        config.log_message("Continuing with best attempt")
        return "continue"

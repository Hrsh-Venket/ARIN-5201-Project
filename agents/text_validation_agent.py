"""
Stage 6a: Text Validation Agent
Validates that text added to poster matches generated text exactly.
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


def text_validation_agent(state: AgentState) -> AgentState:
    """
    Stage 6a: Text Validation Agent

    Validates that text added to poster matches generated text exactly.
    Checks for: exact match, blur, cutoff, spelling errors.

    Args:
        state: Current agent state with best_text and poster_with_text

    Returns:
        Updated state with text_validation_result and text_validation_feedback
    """
    print("\n=== STAGE 6a: TEXT VALIDATION AGENT ===")
    config.log_stage("STAGE 6a: TEXT VALIDATION AGENT", "Starting text validation...")

    attempt_num = state.get("text_adding_attempt_count", 1)
    print(f"Text validation for attempt {attempt_num}/{config.MAX_TEXT_ADDING_ATTEMPTS}")
    config.log_message(f"Validation attempt: {attempt_num}/{config.MAX_TEXT_ADDING_ATTEMPTS}")

    # Get inputs
    generated_text = state.get("best_text") or state.get("generated_text")
    poster_path = state.get("poster_with_text")

    config.log_message(f"\nExpected text:\n{generated_text}")
    config.log_message(f"\nPoster image path: {poster_path}")

    # Initialize OpenRouter client
    client = OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
    )

    # Encode the poster image
    poster_base64 = encode_image(poster_path)

    # Create validation prompt
    validation_prompt = f"""You are validating that text has been correctly added to a poster image.

EXPECTED TEXT (must match exactly):
{generated_text}

Please analyze the poster image and determine:
1. TEXT_CORRECT: Does the text content match the expected text exactly? (YES/NO)
2. TEXT_CLEAR: Is the text clearly generated and readable, not blurry or poorly rendered? (YES/NO)
3. FOUND_TEXT: What text did you actually find on the image? (transcribe it exactly)

Based on your analysis, provide a SPECIFIC_FIX instruction:
- If text is correct but blurry/unclear: One sentence to fix clarity (e.g., "Make the text sharper and clearer")
- If text is clear but incorrect: Use format "Change [found text] to [expected text]" for each element
- If text is both incorrect and unclear: Provide the "Change X to Y" instruction
- If text is correct and clear: Say "No changes needed"

Respond in this EXACT format:
TEXT_CORRECT: [YES or NO]
TEXT_CLEAR: [YES or NO]
FOUND_TEXT: [transcribe exactly what you see]
SPECIFIC_FIX: [one sentence instruction as described above]
VALIDATION: [APPROVED or REJECTED]

Be thorough and strict in your evaluation."""

    config.log_message(f"\nValidation prompt sent to LLM:\n{validation_prompt}")

    try:
        # Call OpenRouter API with vision
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
                                "url": f"data:image/png;base64,{poster_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
        )

        validation_result = response.choices[0].message.content
        config.log_message(f"\nLLM Response:\n{validation_result}")

        # Parse structured validation result
        validation_approved = "VALIDATION: APPROVED" in validation_result.upper()

        # Extract TEXT_CORRECT
        text_is_correct = "TEXT_CORRECT: YES" in validation_result.upper()

        # Extract TEXT_CLEAR
        text_is_clear = "TEXT_CLEAR: YES" in validation_result.upper()

        # Extract FOUND_TEXT
        found_text = None
        if "FOUND_TEXT:" in validation_result:
            found_text_section = validation_result.split("FOUND_TEXT:")[1]
            # Get until next field or end
            for field in ["SPECIFIC_FIX:", "VALIDATION:"]:
                if field in found_text_section:
                    found_text_section = found_text_section.split(field)[0]
                    break
            found_text = found_text_section.strip()

        # Extract SPECIFIC_FIX
        specific_fix = None
        if "SPECIFIC_FIX:" in validation_result:
            specific_fix_section = validation_result.split("SPECIFIC_FIX:")[1]
            # Get until VALIDATION field
            if "VALIDATION:" in specific_fix_section:
                specific_fix_section = specific_fix_section.split("VALIDATION:")[0]
            specific_fix = specific_fix_section.strip()

        # Update state with all parsed values
        state["text_validation_result"] = "approved" if validation_approved else "rejected"
        state["text_validation_feedback"] = validation_result
        state["text_is_correct"] = text_is_correct
        state["text_is_clear"] = text_is_clear
        state["found_text"] = found_text
        state["specific_fix"] = specific_fix

        print(f"Validation result: {'APPROVED' if validation_approved else 'REJECTED'}")
        print(f"Text correct: {text_is_correct}, Text clear: {text_is_clear}")
        print(f"Specific fix: {specific_fix}")
        config.log_message(f"\nValidation status: {'APPROVED' if validation_approved else 'REJECTED'}")
        config.log_message(f"Text correct: {text_is_correct}, Text clear: {text_is_clear}")
        config.log_message(f"Found text: {found_text}")
        config.log_message(f"Specific fix: {specific_fix}")

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(error_msg)
        config.log_message(f"\n{error_msg}")
        config.log_message(f"Traceback:\n{traceback.format_exc()}")

        # On error, reject to allow retry
        state["text_validation_result"] = "rejected"
        state["text_validation_feedback"] = f"Validation failed due to error: {str(e)}"
        state["text_is_correct"] = False
        state["text_is_clear"] = False
        state["found_text"] = None
        state["specific_fix"] = None

    return state


def should_retry_text_adding(state: AgentState) -> str:
    """
    Decision function for text adding retry loop.

    Returns:
        "retry" if should retry text adding, "continue" otherwise
    """
    attempt_count = state.get("text_adding_attempt_count", 0)
    validation_result = state.get("text_validation_result", "rejected")

    if validation_result == "approved":
        print("\nText validation passed. Proceeding to final output.")
        config.log_message("\nDecision: Text validation passed, proceeding to output.")
        return "continue"
    elif attempt_count < config.MAX_TEXT_ADDING_ATTEMPTS:
        print(f"\nText validation failed. Retrying text addition (attempt {attempt_count + 1}/{config.MAX_TEXT_ADDING_ATTEMPTS})...")
        config.log_message(f"\nDecision: Retrying text addition (attempt {attempt_count + 1}/{config.MAX_TEXT_ADDING_ATTEMPTS})")
        return "retry"
    else:
        print(f"\nMax text adding attempts ({config.MAX_TEXT_ADDING_ATTEMPTS}) reached. Continuing with best attempt.")
        config.log_message(f"\nDecision: Max attempts reached, continuing with best attempt")
        return "continue"

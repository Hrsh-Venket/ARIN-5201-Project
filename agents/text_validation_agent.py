"""
Stage 6a: Text Validation Agent
Validates that text added to poster matches generated text exactly.
"""
import base64
from openai import OpenAI
from zhipuai import ZhipuAI
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

    # Initialize SiliconFlow client
    # client = OpenAI(
    #     base_url=config.SiliconFlow_BASE_URL,
    #     api_key=config.SiliconFlow_API_KEY,
    # )
    client = ZhipuAI(api_key=config.ZHIPUAI_API_KEY)

    # Encode the poster image
    poster_base64 = encode_image(poster_path)

    # Create validation prompt
    validation_prompt = f"""You are validating that text has been correctly added to a poster image.

EXPECTED TEXT (must match exactly):
{generated_text}

Please analyze the poster image and verify:
1. All text from EXPECTED TEXT appears on the poster
2. Text is spelled correctly (no spelling errors)
3. Text is clear and readable (not blurred)
4. Text is not cut off or truncated
5. Text matches the expected text EXACTLY

Respond in this format:
VALIDATION: [APPROVED or REJECTED]
FEEDBACK: [If REJECTED, detailed feedback on what's wrong. If APPROVED, brief confirmation.]

Be thorough and strict in your evaluation."""

    config.log_message(f"\nValidation prompt sent to LLM:\n{validation_prompt}")

    try:
        # Call SiliconFlow API with vision
        # response = client.chat.completions.create(
        #     model=config.SiliconFlow_MODEL,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "text",
        #                     "text": validation_prompt
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"data:image/png;base64,{poster_base64}",
        #                         "detail": "high"
        #                     }
        #                 }
        #             ]
        #         }
        #     ],
        # )

        response = client.chat.completions.create(
            model="glm-4.1v-thinking-flashx",
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
                            }
                        }
                    ]
                }
            ],
        )

        validation_result = response.choices[0].message.content
        config.log_message(f"\nLLM Response:\n{validation_result}")

        # Parse validation result
        validation_approved = "VALIDATION: APPROVED" in validation_result.upper()

        state["text_validation_result"] = "approved" if validation_approved else "rejected"
        state["text_validation_feedback"] = validation_result

        print(f"Validation result: {'APPROVED' if validation_approved else 'REJECTED'}")
        print(f"Feedback: {validation_result[:300]}...")
        config.log_message(f"\nValidation status: {'APPROVED' if validation_approved else 'REJECTED'}")

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(error_msg)
        config.log_message(f"\n{error_msg}")
        config.log_message(f"Traceback:\n{traceback.format_exc()}")

        # On error, reject to allow retry
        state["text_validation_result"] = "rejected"
        state["text_validation_feedback"] = f"Validation failed due to error: {str(e)}"

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

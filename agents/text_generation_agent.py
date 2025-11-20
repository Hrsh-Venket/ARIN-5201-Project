"""
Stage 2: Text Generation Agent
Generates text based on planning instructions with retry loop.
"""
import os
from openai import OpenAI
from state import AgentState
import config
import traceback


def text_generation_agent(state: AgentState) -> AgentState:
    """
    Stage 2: Text Generation Agent

    Generates poster text based on planning instructions.
    This is part of a retry loop with the planning agent validation.

    Args:
        state: Current agent state with planning_output

    Returns:
        Updated state with generated_text and incremented text_attempt_count
    """
    print("\n=== STAGE 2: TEXT GENERATION AGENT ===")
    config.log_stage("STAGE 2: TEXT GENERATION AGENT", "Starting text generation...")

    # Initialize attempt count if not set
    if "text_attempt_count" not in state or state["text_attempt_count"] is None:
        state["text_attempt_count"] = 0

    state["text_attempt_count"] += 1
    attempt_num = state["text_attempt_count"]

    print(f"Text generation attempt: {attempt_num}/{config.MAX_TEXT_ATTEMPTS}")
    config.log_message(f"Attempt: {attempt_num}/{config.MAX_TEXT_ATTEMPTS}")

    # Initialize SiliconFlow client
    client = OpenAI(
        base_url=config.SiliconFlow_BASE_URL,
        api_key=config.SiliconFlow_API_KEY,
    )

    config.log_message(f"\nSiliconFlow client initialized")
    config.log_message(f"Model: {config.SiliconFlow_MODEL}")

    # Create text generation prompt based on planning output
    text_prompt = f"""You are a professional copywriter. Based on the following design plan, generate the text content for the poster.

DESIGN PLAN:
{state["planning_output"]}

ORIGINAL INPUT TEXT:
{state["input_text"]}

Generate the text content following the TEXT REQUIREMENTS section of the design plan. Include:
1. Headline/Title text (2 words or 3 short words)
2. Body text (if specified) (1 short sentence or 2-3 short bullet points, at most 8 words total)
3. Call-to-action text (if specified) (2 words or 3 short words)

Format your response clearly with labels for each text element (e.g., "HEADLINE:", "BODY:", "CALL-TO-ACTION:").
Keep text concise and impactful. Follow any character limits specified in the plan."""

    # Add feedback from previous attempt if exists
    if state.get("validation_feedback") and attempt_num > 1:
        text_prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{state['validation_feedback']}\n\nPlease address this feedback in your new text generation."
        config.log_message(f"\nIncluding previous feedback in prompt")

    config.log_message(f"\nPrompt sent to LLM:\n{text_prompt}")

    try:
        # Call SiliconFlow API
        response = client.chat.completions.create(
            model=config.SiliconFlow_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": text_prompt
                }
            ],
        )

        generated_text = response.choices[0].message.content
        config.log_message(f"\nLLM Response:\n{generated_text}")

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(error_msg)
        config.log_message(f"\n{error_msg}")
        config.log_message(f"Traceback:\n{traceback.format_exc()}")
        raise

    # Save text attempt
    os.makedirs(config.INTERMEDIATE_DIR, exist_ok=True)
    text_path = os.path.join(config.INTERMEDIATE_DIR, f"text_attempt{attempt_num}.txt")
    with open(text_path, "w") as f:
        f.write(generated_text)

    print(f"Text attempt {attempt_num} saved to: {text_path}")
    print(f"\nGenerated text (first 300 chars):\n{generated_text[:300]}...")
    config.log_message(f"\nText saved to: {text_path}")

    # Update state
    state["generated_text"] = generated_text
    if state.get("best_text") is None:
        state["best_text"] = generated_text
        config.log_message(f"Set as best_text (first attempt)")

    return state


def validate_text(state: AgentState) -> AgentState:
    """
    Text validation by Planning Agent.

    The planning agent reviews the generated text against its original plan.

    Args:
        state: Current agent state with planning_output and generated_text

    Returns:
        Updated state with validation_feedback and validation_passed
    """
    print("\n=== TEXT VALIDATION ===")
    config.log_stage("TEXT VALIDATION", "Validating generated text...")

    # Initialize SiliconFlow client
    client = OpenAI(
        base_url=config.SiliconFlow_BASE_URL,
        api_key=config.SiliconFlow_API_KEY,
    )

    config.log_message(f"Generated text to validate:\n{state['generated_text']}")

    validation_prompt = f"""You are a design quality validator. Review the generated text against the design plan.

DESIGN PLAN:
{state["planning_output"]}

GENERATED TEXT:
{state["generated_text"]}

Evaluate whether the text:
1. Follows the character limits specified
2. Matches the tone and style requirements
3. Includes all required text elements
4. Is appropriate for the poster design

Respond in this format:
VALIDATION: [PASS or FAIL]
FEEDBACK: [If FAIL, specific issues to fix. If PASS, brief confirmation.]

Be strict but fair in your evaluation."""

    config.log_message(f"\nValidation prompt sent to LLM:\n{validation_prompt}")

    try:
        response = client.chat.completions.create(
            model=config.SiliconFlow_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
        )

        validation_result = response.choices[0].message.content
        config.log_message(f"\nLLM Response:\n{validation_result}")

        state["validation_feedback"] = validation_result

        # Check if validation passed
        validation_passed = "VALIDATION: PASS" in validation_result.upper()
        state["validation_passed"] = validation_passed

        print(f"Validation result: {'PASSED' if validation_passed else 'FAILED'}")
        print(f"Feedback: {validation_result[:200]}...")
        config.log_message(f"\nValidation passed: {validation_passed}")

        if validation_passed:
            state["best_text"] = state["generated_text"]
            config.log_message(f"Updated best_text")

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(error_msg)
        config.log_message(f"\n{error_msg}")
        config.log_message(f"Traceback:\n{traceback.format_exc()}")

        # On error, mark as failed
        state["validation_feedback"] = f"Validation failed due to error: {str(e)}"
        state["validation_passed"] = False

    return state


def should_retry_text(state: AgentState) -> str:
    """
    Decision function for text generation retry loop.

    Returns:
        "retry" if should retry text generation, "continue" otherwise
    """
    if state.get("validation_passed"):
        print("\nText validation passed. Proceeding to image generation.")
        config.log_message("\nDecision: Text validation passed, proceeding to image generation.")
        return "continue"

    if state["text_attempt_count"] >= config.MAX_TEXT_ATTEMPTS:
        print(f"\nMax text attempts ({config.MAX_TEXT_ATTEMPTS}) reached. Continuing with best attempt.")
        config.log_message(f"\nDecision: Max text attempts reached, continuing with best attempt")
        return "continue"

    print(f"\nRetrying text generation...")
    config.log_message(f"\nDecision: Retrying text generation")
    return "retry"

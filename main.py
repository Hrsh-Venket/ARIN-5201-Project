"""
Main orchestration file for the LangGraph-based poster generator system.
"""
import os
import shutil
from langgraph.graph import StateGraph, END
from state import AgentState
import config

# Import diffusers for pipeline initialization
from diffusers import QwenImageEditPipeline
import torch

# Import agent functions
from agents.planning_agent import planning_agent
from agents.text_generation_agent import (
    text_generation_agent,
    validate_text,
    should_retry_text
)
from agents.editor_agent import editor_agent, should_retry_image
from agents.image_generation_agent import image_generation_agent
from agents.text_adding_agent import text_adding_agent
from agents.text_validation_agent import (
    text_validation_agent,
    should_retry_text_adding
)


def load_pipeline(state: AgentState) -> AgentState:
    """
    Stage 0a: Load the diffusers pipeline for image generation.

    Args:
        state: Initial agent state

    Returns:
        Updated state with image_pipeline loaded
    """
    print("\n=== STAGE 0a: LOADING DIFFUSERS PIPELINE ===")
    config.log_stage("STAGE 0a: LOADING DIFFUSERS PIPELINE", "Loading diffusers model...")

    try:
        print(f"Loading model: {config.HUGGINGFACE_MODEL}")
        print("This may take a few minutes on first run (downloading model)...")
        config.log_message(f"Loading model: {config.HUGGINGFACE_MODEL}")

        # Load the pipeline using QwenImageEditPipeline
        pipeline = QwenImageEditPipeline.from_pretrained(
            config.HUGGINGFACE_MODEL  # "Qwen/Qwen-Image-Edit"
        )

        # Move to bfloat16 and GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use bfloat16 if on CUDA, otherwise float32
        if device == "cuda":
            pipeline = pipeline.to(torch.bfloat16)

        pipeline = pipeline.to(device)

        print(f"Pipeline loaded successfully on {device}")
        config.log_message(f"Pipeline loaded successfully on {device}")

        # Store in state
        state["image_pipeline"] = pipeline

    except Exception as e:
        error_msg = f"Warning: Failed to load diffusers pipeline: {e}"
        print(error_msg)
        print("Image generation will fall back to using base images")
        config.log_message(f"\nERROR: {error_msg}")
        state["image_pipeline"] = None

    return state


def load_input(state: AgentState) -> AgentState:
    """
    Stage 0b: Load input text and image.

    Args:
        state: Initial agent state

    Returns:
        Updated state with input_text and input_image_path
    """
    print("\n=== STAGE 0b: LOADING INPUT ===")
    config.log_stage("STAGE 0b: LOADING INPUT", "Loading input files...")

    # Load input text
    if not os.path.exists(config.INPUT_TEXT_PATH):
        raise FileNotFoundError(f"Input text file not found: {config.INPUT_TEXT_PATH}")

    with open(config.INPUT_TEXT_PATH, "r") as f:
        input_text = f.read().strip()

    # Verify input image exists
    if not os.path.exists(config.INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"Input image file not found: {config.INPUT_IMAGE_PATH}")

    print(f"Loaded input text: {input_text[:100]}...")
    print(f"Loaded input image: {config.INPUT_IMAGE_PATH}")
    config.log_message(f"Input text: {input_text}")
    config.log_message(f"Input image: {config.INPUT_IMAGE_PATH}")

    # Initialize state
    state["input_text"] = input_text
    state["input_image_path"] = config.INPUT_IMAGE_PATH
    state["text_attempt_count"] = 0
    state["image_attempt_count"] = 0
    state["image_complete_failure_count"] = 0
    state["text_adding_attempt_count"] = 0
    state["validation_passed"] = False

    config.log_message("\nState initialized with counters set to 0")

    return state


def segmentation_placeholder(state: AgentState) -> AgentState:
    """
    Stage 5: Segmentation (placeholder - currently just pass through).

    Args:
        state: Current agent state

    Returns:
        Unchanged state
    """
    print("\n=== STAGE 5: SEGMENTATION (SKIPPED) ===")
    print("Segmentation stage skipped as per requirements.")
    config.log_stage("STAGE 5: SEGMENTATION", "Skipped (placeholder)")
    return state


def save_output(state: AgentState) -> AgentState:
    """
    Stage 7: Save final outputs.

    Args:
        state: Current agent state with poster_with_text and best_text

    Returns:
        Updated state with final output paths
    """
    print("\n=== STAGE 7: SAVING FINAL OUTPUT ===")
    config.log_stage("STAGE 7: SAVING FINAL OUTPUT", "Saving final outputs...")

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Save final poster
    poster_source = state.get("poster_with_text")
    if not poster_source or not os.path.exists(poster_source):
        print("Warning: poster_with_text not found, using best_image")
        config.log_message("WARNING: poster_with_text not found, using best_image")
        poster_source = state.get("best_image") or state.get("current_image")

    final_poster_path = os.path.join(config.OUTPUT_DIR, "poster.png")
    shutil.copy(poster_source, final_poster_path)
    state["final_poster_path"] = final_poster_path
    print(f"Final poster saved to: {final_poster_path}")
    config.log_message(f"Final poster saved to: {final_poster_path}")

    # Save final text
    final_text = state.get("best_text") or state.get("generated_text")
    final_text_path = os.path.join(config.OUTPUT_DIR, "text.txt")
    with open(final_text_path, "w") as f:
        f.write(final_text)
    state["final_text_path"] = final_text_path
    print(f"Final text saved to: {final_text_path}")
    config.log_message(f"Final text saved to: {final_text_path}")

    print("\n=== POSTER GENERATION COMPLETE ===")
    print(f"Final poster: {final_poster_path}")
    print(f"Final text: {final_text_path}")

    config.log_message(f"\n{'='*60}")
    config.log_message("POSTER GENERATION COMPLETE")
    config.log_message(f"{'='*60}")
    config.log_message(f"Final poster: {final_poster_path}")
    config.log_message(f"Final text: {final_text_path}")

    return state


def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow with all stages and conditional edges.

    Returns:
        Compiled StateGraph
    """
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes for each stage
    workflow.add_node("load_pipeline", load_pipeline)
    workflow.add_node("load_input", load_input)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("text_generation", text_generation_agent)
    workflow.add_node("text_validation", validate_text)
    workflow.add_node("image_generation", image_generation_agent)
    workflow.add_node("image_validation", editor_agent)
    workflow.add_node("segmentation", segmentation_placeholder)
    workflow.add_node("text_adding", text_adding_agent)
    workflow.add_node("text_adding_validation", text_validation_agent)  # NEW: Stage 6a
    workflow.add_node("save_output", save_output)

    # Define the workflow edges
    workflow.set_entry_point("load_pipeline")

    # Linear flow: load_pipeline -> load_input -> planning -> text_generation
    workflow.add_edge("load_pipeline", "load_input")
    workflow.add_edge("load_input", "planning")
    workflow.add_edge("planning", "text_generation")

    # Text generation retry loop (Stage 2 with validation)
    workflow.add_edge("text_generation", "text_validation")
    workflow.add_conditional_edges(
        "text_validation",
        should_retry_text,
        {
            "retry": "text_generation",
            "continue": "image_generation"
        }
    )

    # Image generation retry loop with extended retries (Stage 3-4)
    # NOTE: should_retry_image now handles both regular and extended retries
    workflow.add_edge("image_generation", "image_validation")
    workflow.add_conditional_edges(
        "image_validation",
        should_retry_image,
        {
            "retry": "image_generation",
            "continue": "segmentation"
        }
    )

    # Segmentation -> Text Adding
    workflow.add_edge("segmentation", "text_adding")

    # Text adding retry loop (Stage 6 with Stage 6a validation)
    workflow.add_edge("text_adding", "text_adding_validation")
    workflow.add_conditional_edges(
        "text_adding_validation",
        should_retry_text_adding,
        {
            "retry": "text_adding",
            "continue": "save_output"
        }
    )

    # Final output
    workflow.add_edge("save_output", END)

    # Compile the graph
    return workflow.compile()


def main():
    """
    Main entry point for the poster generator system.
    """
    print("="*60)
    print("POSTER GENERATOR - LangGraph Multi-Agent System")
    print("="*60)

    # Initialize logging
    config.init_log()
    config.log_message("="*60)
    config.log_message("POSTER GENERATOR - LangGraph Multi-Agent System")
    config.log_message("="*60)

    try:
        # Build the workflow graph
        app = build_graph()

        # Initialize empty state (will be populated by load_pipeline and load_input)
        initial_state: AgentState = {
            "input_text": "",
            "input_image_path": "",
            "image_pipeline": None,
            "planning_output": None,
            "generated_text": None,
            "text_attempt_count": 0,
            "best_text": None,
            "current_image": None,
            "image_attempt_count": 0,
            "image_complete_failure_count": 0,
            "best_image": None,
            "validation_feedback": None,
            "validation_passed": False,
            "poster_with_text": None,
            "text_adding_attempt_count": 0,
            "text_validation_result": None,
            "text_validation_feedback": None,
            "text_is_correct": None,
            "text_is_clear": None,
            "found_text": None,
            "specific_fix": None,
            "final_poster_path": None,
            "final_text_path": None,
        }

        # Run the workflow with recursion limit configuration
        final_state = app.invoke(
            initial_state,
            config={"recursion_limit": config.RECURSION_LIMIT}
        )

        print("\n" + "="*60)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nFinal Outputs:")
        print(f"  Poster: {final_state['final_poster_path']}")
        print(f"  Text: {final_state['final_text_path']}")
        print(f"\nIntermediate outputs saved in: {config.INTERMEDIATE_DIR}/")
        print(f"Pipeline log saved in: {config.PIPELINE_LOG_PATH}")

    except Exception as e:
        error_msg = f"\nERROR: Workflow failed with exception:"
        print(error_msg)
        print(f"{type(e).__name__}: {e}")
        config.log_message(f"\n{error_msg}")
        config.log_message(f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()

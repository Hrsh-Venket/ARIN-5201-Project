"""
State definition for the LangGraph poster generator system.
"""
from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    """
    Comprehensive state for the poster generation pipeline.

    Tracks all intermediate and final outputs across the multi-agent workflow.
    """
    # Stage 0: Input
    input_text: str
    input_image_path: str

    # Diffusers pipeline (loaded once at start)
    image_pipeline: Optional[Any]  # Stores the loaded diffusers pipeline

    # Stage 1: Planning Agent
    planning_output: Optional[str]

    # Stage 2: Text Generation Agent (with retry loop)
    generated_text: Optional[str]
    text_attempt_count: int
    best_text: Optional[str]

    # Stage 4: Image Generation Agent (with retry loop)
    current_image: Optional[str]  # Path to current image attempt
    image_attempt_count: int
    best_image: Optional[str]  # Path to best image so far

    # Stage 3: Editor Agent (validation feedback)
    validation_feedback: Optional[str]
    validation_passed: bool

    # Stage 6: Text Adding Agent
    poster_with_text: Optional[str]  # Path to poster with text added

    # Stage 7: Final outputs
    final_poster_path: Optional[str]
    final_text_path: Optional[str]

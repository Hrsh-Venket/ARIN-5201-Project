"""
LangGraph Template: Text-to-Image Pipeline
DeepSeek analyzes input → Pollinations.ai generates image
"""

import re
import requests
from typing import TypedDict
from urllib.parse import quote
from openai import OpenAI
from langgraph.graph import StateGraph, END

# =============================================================================
# CONFIGURATION
# =============================================================================
OPENROUTER_API_KEY = "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"

# Model identifiers
DEEPSEEK_MODEL = "`deepseek/deepseek-chat`"
POLLINATIONS_API = "https://image.pollinations.ai/prompt"  # Free image generation (no key needed!)

# =============================================================================
# STATE DEFINITION
# =============================================================================
class AgentState(TypedDict):
    """State passed between agents in the workflow"""
    user_input: str          # Original user text input
    image_prompt: str        # Refined prompt from DeepSeek
    image_url: str          # Generated image URL from Pollinations
    image_path: str         # Local path where image is saved


# =============================================================================
# API CLIENT
# =============================================================================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# =============================================================================
# AGENT NODES
# =============================================================================

def deepseek_agent(state: AgentState) -> AgentState:
    """
    Agent 1: DeepSeek analyzes user input and creates image generation prompt
    """
    print(f"\n[DeepSeek Agent] Processing input: {state['user_input']}")

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating concise, detailed image generation prompts. "
                          "Analyze the user's input and create a single, detailed prompt for image generation. "
                          "Focus on visual elements, style, composition, and mood. "
                          "Output ONLY the image prompt, nothing else."
            },
            {
                "role": "user",
                "content": state["user_input"]
            }
        ],
        temperature=0.7,
        max_tokens=200,  # Limit tokens to save credits (prompt only needs ~200 tokens)
    )

    image_prompt = response.choices[0].message.content.strip()
    print(f"[DeepSeek Agent] Generated prompt: {image_prompt}")

    state["image_prompt"] = image_prompt
    return state


def pollinations_image_agent(state: AgentState) -> AgentState:
    """
    Agent 2: Pollinations.ai generates image from the prompt (completely free!)
    """
    print(f"\n[Pollinations Agent] Generating image from prompt...")

    # URL-encode the prompt
    encoded_prompt = quote(state["image_prompt"])

    # Build the image URL - Pollinations generates on request
    image_url = f"{POLLINATIONS_API}/{encoded_prompt}"

    print(f"[Pollinations Agent] Image URL created: {image_url[:100]}...")

    state["image_url"] = image_url
    return state


def save_image(state: AgentState) -> AgentState:
    """
    Save the generated image to disk
    """
    print(f"\n[Save Agent] Downloading and saving image...")

    output_path = "generated_image.png"

    if not state["image_url"]:
        print(f"[Save Agent] No image URL to download")
        state["image_path"] = None
        return state

    try:
        # Download the image from Pollinations
        print(f"[Save Agent] Downloading from: {state['image_url'][:80]}...")
        response = requests.get(state["image_url"], timeout=30)
        response.raise_for_status()

        # Save to file
        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"[Save Agent] Image saved to: {output_path} ({len(response.content)} bytes)")
        state["image_path"] = output_path

    except requests.RequestException as e:
        print(f"[Save Agent] Error downloading image: {e}")
        state["image_path"] = None

    return state


# =============================================================================
# LANGGRAPH WORKFLOW
# =============================================================================

def create_workflow():
    """Build the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("deepseek", deepseek_agent)
    workflow.add_node("pollinations", pollinations_image_agent)
    workflow.add_node("save", save_image)

    # Define edges (flow)
    workflow.set_entry_point("deepseek")
    workflow.add_edge("deepseek", "pollinations")
    workflow.add_edge("pollinations", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the text-to-image pipeline"""

    # Get user input
    user_input = input("\nEnter your image description: ").strip()

    if not user_input:
        print("No input provided. Exiting.")
        return

    # Initialize state
    initial_state = {
        "user_input": user_input,
        "image_prompt": "",
        "image_url": "",
        "image_path": "",
    }

    # Create and run workflow
    print("\n" + "="*60)
    print("Starting LangGraph Pipeline")
    print("="*60)

    app = create_workflow()
    final_state = app.invoke(initial_state)

    # Display results
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Original Input: {final_state['user_input']}")
    print(f"DeepSeek Prompt: {final_state['image_prompt']}")
    print(f"Image Saved: {final_state.get('image_path', 'N/A')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

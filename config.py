"""
Configuration module for loading environment variables.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
SiliconFlow_API_KEY = os.getenv("SiliconFlow_API_KEY")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

# Model configurations
OPENROUTER_MODEL = "x-ai/grok-4.1-fast"  # Grok 4.1 Fast (was sherlock-dash-alpha)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SiliconFlow_MODEL = "deepseek-ai/DeepSeek-OCR"  # Deepseek OCR model
SiliconFlow_BASE_URL = "https://api.siliconflow.cn/v1"

HUGGINGFACE_MODEL = "Qwen/Qwen-Image-Edit"
HUGGINGFACE_INFERENCE_STEPS = 50

# Retry configurations
MAX_TEXT_ATTEMPTS = 3
MAX_IMAGE_ATTEMPTS = 3
MAX_IMAGE_COMPLETE_FAILURE_ATTEMPTS = 15  # Extended retries for complete failures
MAX_TEXT_ADDING_ATTEMPTS = 10

# File paths
OUTPUT_DIR = "outputs"
INTERMEDIATE_DIR = "intermediate_outputs"
INPUT_TEXT_PATH = "input.txt"
INPUT_IMAGE_PATH = "input.png"
PIPELINE_LOG_PATH = os.path.join(INTERMEDIATE_DIR, "pipeline_log.txt")

# Validate required environment variables
#if not OPENROUTER_API_KEY:
    #raise ValueError("OPENROUTER_API_KEY not found in environment variables")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
if not SiliconFlow_API_KEY:
    raise ValueError("SiliconFlow_API_KEY not found in environment variables")


# Logging utility functions
def init_log():
    """Initialize/clear the log file at start of run."""
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    with open(PIPELINE_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("POSTER GENERATOR PIPELINE LOG\n")
        f.write(f"{'='*60}\n\n")


def log_stage(stage_name: str, content: str):
    """
    Log content to pipeline log file with stage header.

    Args:
        stage_name: Name of the stage (e.g., "STAGE 1: PLANNING AGENT")
        content: Content to log
    """
    with open(PIPELINE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{stage_name}\n")
        f.write(f"{'='*60}\n")
        f.write(content)
        f.write(f"\n")


def log_message(message: str):
    """Log a message without stage header."""
    with open(PIPELINE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

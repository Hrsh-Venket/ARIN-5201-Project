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

# Model configurations
OPENROUTER_MODEL = "openrouter/sherlock-dash-alpha"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

HUGGINGFACE_MODEL = "Qwen/Qwen-Image-Edit"
HUGGINGFACE_INFERENCE_STEPS = 4

# Retry configurations
MAX_TEXT_ATTEMPTS = 3
MAX_IMAGE_ATTEMPTS = 3

# File paths
OUTPUT_DIR = "outputs"
INTERMEDIATE_DIR = "intermediate_outputs"
INPUT_TEXT_PATH = "input.txt"
INPUT_IMAGE_PATH = "input.png"

# Validate required environment variables
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

"""Configuration, API credentials, model builders, and cost models for Day 5 Capstone."""

from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress harmless internal Google GenAI SDK dual-key notices and deprecations
for _logger_name in (
    "google_genai._api_client",
    "google.genai._api_client",
    "google_genai.models",
    "google.genai.models",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Both GOOGLE_API_KEY and GEMINI_API_KEY are set.*")

# Discover API credentials
DAY5_DIR = Path(__file__).resolve().parent
REPO_ROOT = DAY5_DIR.parent.parent

for candidate in [
    DAY5_DIR / ".env",
    DAY5_DIR.parent / "day 1" / ".env",
    DAY5_DIR.parent / "day 4" / ".env",
    REPO_ROOT / ".env",
]:
    if candidate.exists():
        load_dotenv(candidate)
        break

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = api_key
if api_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = api_key

MODEL_NAME = "gemini-2.5-flash"

# Gemini 2.5 Flash Pricing (per token)
INPUT_PRICE_PER_TOKEN = 0.000000075   # $0.075 per 1M tokens
OUTPUT_PRICE_PER_TOKEN = 0.000000300  # $0.300 per 1M tokens


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates approximate USD cost for Gemini 2.5 Flash invocation."""
    return (prompt_tokens * INPUT_PRICE_PER_TOKEN) + (completion_tokens * OUTPUT_PRICE_PER_TOKEN)


def build_llm(temperature: float = 0.0):
    """Instantiates an enterprise-grade ChatGoogleGenerativeAI client."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY/GOOGLE_API_KEY in environment or .env file.")

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        google_api_key=key,
        convert_system_message_to_human=False,
    )

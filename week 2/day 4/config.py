"""Configuration and LLM initialization for Week 2 Day 4 (CrewAI + Gemini)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import warnings

# Suppress harmless dual-key and internal SDK telemetry warnings
for _logger_name in (
    "google_genai._api_client",
    "google.genai._api_client",
    "google_genai.models",
    "google.genai.models",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Both GOOGLE_API_KEY and GEMINI_API_KEY are set.*")

try:
    from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages
    set_suppress_tracing_messages(True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DAY3_ENV = HERE.parent / "day 3" / ".env"
DAY2_ENV = HERE.parent / "day 2" / ".env"
DAY1_ENV = HERE.parent / "day 1" / ".env"
_PLACEHOLDERS = {"", "your-gemini-key-here", "sk-ant-your-key-here"}


def load_local_env() -> None:
    """Load environment variables from local .env files hierarchically."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    for path in (HERE / ".env", DAY3_ENV, DAY2_ENV, DAY1_ENV):
        if path.is_file():
            load_dotenv(path, override=True)
            return
    load_dotenv(override=True)


def user_api_key() -> str | None:
    """Discover and return a valid Gemini API key."""
    load_local_env()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        raw = (os.environ.get(name) or "").strip().strip('"').strip("'")
        if raw and raw.lower() not in _PLACEHOLDERS:
            return raw
    return None


def default_model() -> str:
    """Return default Gemini model identifier."""
    load_local_env()
    model = (os.environ.get("GEMINI_MODEL") or "").strip()
    if model.startswith("gemini"):
        return model
    return "gemini-2.5-flash"


def build_crew_llm(model_name: str | None = None, temperature: float = 0.2):
    """Return a configured LLM instance compatible with CrewAI."""
    from crewai import LLM

    key = user_api_key()
    if not key:
        raise RuntimeError(
            f"Missing GEMINI_API_KEY in {HERE / '.env'}, {DAY3_ENV}, {DAY2_ENV}, or {DAY1_ENV}"
        )
    
    # Ensure environment variables are set for LiteLLM / CrewAI
    os.environ["GEMINI_API_KEY"] = key
    os.environ["GOOGLE_API_KEY"] = key

    for _logger_name in (
        "google_genai._api_client",
        "google.genai._api_client",
        "google_genai.models",
        "google.genai.models",
    ):
        logging.getLogger(_logger_name).setLevel(logging.ERROR)
    
    target_model = model_name or default_model()
    # Format for LiteLLM provider
    if not target_model.startswith("gemini/"):
        llm_model = f"gemini/{target_model}"
    else:
        llm_model = target_model

    return LLM(
        model=llm_model,
        api_key=key,
        temperature=temperature,
    )

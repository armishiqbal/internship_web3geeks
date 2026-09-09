"""Environment and LLM configuration for Week 2 Day 3 (LangGraph + Gemini)."""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAY2_ENV = HERE.parent / "day 2" / ".env"
DAY1_ENV = HERE.parent / "day 1" / ".env"
_PLACEHOLDERS = {"", "your-gemini-key-here", "sk-ant-your-key-here"}


def load_local_env() -> None:
    """Load  from Day 3, then Day 2, then Day 1, then cwd."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    for path in (HERE / ".env", DAY2_ENV, DAY1_ENV):
        if path.is_file():
            load_dotenv(path, override=True)
            return
    load_dotenv(override=True)


def user_api_key() -> str | None:
    load_local_env()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        raw = (os.environ.get(name) or "").strip().strip('"').strip("'")
        if raw and raw.lower() not in _PLACEHOLDERS:
            return raw
    return None


def default_model() -> str:
    load_local_env()
    model = (os.environ.get("GEMINI_MODEL") or "").strip()
    if model.startswith("gemini"):
        return model
    return "gemini-2.5-flash"


def build_llm(*, temperature: float = 0):
    """Return a ChatGoogleGenerativeAI instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = user_api_key()
    if not key:
        raise RuntimeError(
            f"Add GEMINI_API_KEY to {HERE / '.env'}, {DAY2_ENV}, or {DAY1_ENV}."
        )
    return ChatGoogleGenerativeAI(
        model=default_model(),
        google_api_key=key,
        temperature=temperature,
    )


def get_text_content(msg) -> str:
    """Safely extract plain text from an AIMessage or string, handling list chunks."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)

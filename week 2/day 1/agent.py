"""Minimal ReAct agent: Gemini in a loop, no LangChain / LangGraph.

Put GEMINI_API_KEY in `.env`. Live Google AI Studio API only — no mock.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from tools import TOOL_FNS, TOOL_SCHEMAS

HERE = Path(__file__).resolve().parent
SYSTEM = (
    "You are a careful tool-using intern agent. "
    "Use tools when they can ground the answer. "
    "If a tool returns ERROR, explain the failure and do not invent numbers. "
    "After you have enough observations, answer in plain text."
)
_PLACEHOLDERS = {"", "your-gemini-key-here", "sk-ant-your-key-here"}


def load_local_env() -> None:
    """Load the .env file from the script directory or current working directory."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = HERE / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)


def user_api_key() -> str | None:
    """Gemini key from .env (GEMINI_API_KEY or GOOGLE_API_KEY)."""
    load_local_env()

    # 1. Check direct .env file values
    try:
        from dotenv import dotenv_values

        env_file = HERE / ".env"
        if env_file.is_file():
            file_vals = dotenv_values(env_file)
            for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                raw = (file_vals.get(name) or "").strip().strip('"').strip("'")
                if raw and raw.lower() not in _PLACEHOLDERS:
                    return raw
    except Exception:
        pass

    # 2. Check os.environ populated by load_dotenv or the shell
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        raw = (os.environ.get(name) or "").strip().strip('"').strip("'")
        if raw and raw.lower() not in _PLACEHOLDERS:
            return raw

    return None


def default_model() -> str:
    """Read GEMINI_MODEL from environment or .env, defaulting to gemini-2.5-flash."""
    load_local_env()

    # 1. Check os.environ first
    env_m = (os.environ.get("GEMINI_MODEL") or "").strip()
    if env_m.startswith("gemini"):
        return env_m

    # 2. Check .env file values directly
    try:
        from dotenv import dotenv_values

        env_file = HERE / ".env"
        if env_file.is_file():
            m = (dotenv_values(env_file).get("GEMINI_MODEL") or "").strip()
            if m.startswith("gemini"):
                return m
    except Exception:
        pass

    # 3. Fallback default
    return "gemini-2.5-flash"


def make_client():
    from google import genai

    key = user_api_key()
    if not key:
        raise RuntimeError(
            f"Add GEMINI_API_KEY to {HERE / '.env'} then Restart kernel → Run All."
        )
    return genai.Client(api_key=key)


def _tool_config(schemas: list[dict]):
    from google.genai import types

    decls = [
        types.FunctionDeclaration(
            name=s["name"],
            description=s["description"],
            parameters=s["input_schema"],
        )
        for s in schemas
    ]
    return types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[types.Tool(function_declarations=decls)],
    )


def _extract_calls_and_text(response):
    texts: list[str] = []
    calls: list[SimpleNamespace] = []
    if getattr(response, "function_calls", None):
        for fc in response.function_calls:
            args = dict(fc.args) if getattr(fc, "args", None) else {}
            calls.append(SimpleNamespace(name=fc.name, input=args, id=getattr(fc, "id", fc.name)))
    parts = []
    try:
        parts = response.candidates[0].content.parts or []
    except Exception:
        parts = []
    for part in parts:
        if getattr(part, "text", None):
            texts.append(part.text.strip())
        fc = getattr(part, "function_call", None)
        if fc and fc.name and not any(c.name == fc.name and c.input == dict(fc.args or {}) for c in calls):
            calls.append(
                SimpleNamespace(name=fc.name, input=dict(fc.args or {}), id=getattr(fc, "id", fc.name))
            )
    return texts, calls, parts


def _execute(name: str, tool_input: dict, tool_fns: dict[str, Callable]) -> tuple[str, bool]:
    fn = tool_fns.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}' (hallucinated name)", True
    try:
        observation = str(fn(**tool_input))
        return observation, observation.startswith("ERROR")
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}", True


def run_agent(
    user_message: str,
    *,
    client=None,
    tools: list[dict] | None = None,
    tool_fns: dict[str, Callable] | None = None,
    model: str | None = None,
    max_iterations: int = 8,
    verbose: bool = True,
) -> dict:
    """Reason → Act → Observe loop against the live Gemini API."""
    from google.genai import types

    client = client or make_client()
    tools = tools if tools is not None else TOOL_SCHEMAS
    tool_fns = tool_fns if tool_fns is not None else TOOL_FNS
    model = model or default_model()
    config = _tool_config(tools)

    contents: list[Any] = [
        types.Content(role="user", parts=[types.Part(text=user_message)])
    ]
    messages: list[dict] = [{"role": "user", "content": user_message}]
    transcript: list[dict] = []
    final_text = ""
    stopped = "completed"

    def log(kind: str, text: str) -> None:
        transcript.append({"kind": kind, "text": text})
        if verbose:
            print(f"[{kind:11}] {text}")

    log("user", user_message)

    for step in range(1, max_iterations + 1):
        log("reason", f"iteration {step}/{max_iterations} — calling Gemini")
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        texts, calls, _parts = _extract_calls_and_text(response)
        for t in texts:
            if t:
                log("thought", t)

        if not calls:
            final_text = "\n".join(t for t in texts if t) or "(empty final text)"
            log("final", final_text)
            stopped = "completed"
            messages.append({"role": "assistant", "content": final_text})
            break

        try:
            contents.append(response.candidates[0].content)
        except Exception:
            contents.append(response)

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "function_call", "name": c.name, "input": c.input} for c in calls],
            }
        )
        fr_parts = []
        observations = []
        for call in calls:
            log("act", f"{call.name}  input={json.dumps(call.input)}")
            observation, is_error = _execute(call.name, call.input, tool_fns)
            log("observe", observation)
            observations.append({"type": "function_response", "name": call.name, "content": observation, "is_error": is_error})
            fr_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": observation})
            )
        contents.append(types.Content(role="user", parts=fr_parts))
        messages.append({"role": "user", "content": observations})
    else:
        stopped = "max_iterations"
        final_text = f"Stopped after {max_iterations} iterations (safeguard)."
        log("guardrail", final_text)

    return {
        "final_text": final_text,
        "transcript": transcript,
        "messages": messages,
        "stopped": stopped,
        "model": model,
    }


def one_shot_tool_round(user_message: str, *, client=None, model: str | None = None) -> dict:
    """Task 2: one Gemini call, manual tool execution, one function_response round-trip."""
    from google.genai import types

    client = client or make_client()
    model = model or default_model()
    config = _tool_config(TOOL_SCHEMAS)
    contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=user_message)])]
    first_raw = client.models.generate_content(model=model, contents=contents, config=config)
    texts, calls, _ = _extract_calls_and_text(first_raw)

    first_blocks = [SimpleNamespace(type="text", text=t, name=None) for t in texts if t]
    first_blocks += [
        SimpleNamespace(type="function_call", name=c.name, text="", input=c.input) for c in calls
    ]
    first = SimpleNamespace(stop_reason="function_call" if calls else "end_turn", content=first_blocks)

    results = []
    fr_parts = []
    for c in calls:
        out, _err = _execute(c.name, c.input, TOOL_FNS)
        results.append({"type": "function_response", "name": c.name, "content": out})
        fr_parts.append(types.Part.from_function_response(name=c.name, response={"result": out}))

    second = None
    if calls:
        contents.append(first_raw.candidates[0].content)
        contents.append(types.Content(role="user", parts=fr_parts))
        second_raw = client.models.generate_content(model=model, contents=contents, config=config)
        s_texts, _, _ = _extract_calls_and_text(second_raw)
        second = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=t) for t in s_texts if t]
        )

    return {"first": first, "tool_results": results, "second": second}


if __name__ == "__main__":
    question = (
        "Look up the weather in Karachi and London. "
        "Which city is warmer, and by how many degrees Celsius?"
    )
    out = run_agent(question, max_iterations=8)
    print("\n=== FINAL ===\n", out["final_text"])

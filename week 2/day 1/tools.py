"""Tool implementations and Anthropic JSON schemas for Week 2 Day 1.

Tools are ordinary Python functions. The model never executes them — the
agent loop does, then returns a tool_result block.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path

HERE = Path(__file__).resolve().parent
SANDBOX = HERE / "sandbox"

# Deterministic stub. Not a live weather API — that is the point of Task 2.
WEATHER = {
    "karachi": {"temp_c": 33, "condition": "hot, humid"},
    "london": {"temp_c": 12, "condition": "overcast, light rain"},
    "tokyo": {"temp_c": 18, "condition": "clear"},
    "new york": {"temp_c": 21, "condition": "partly cloudy"},
    "oslo": {"temp_c": 4, "condition": "cold, windy"},
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_ast(node.operand))
    raise ValueError("only +, -, *, /, //, %, ** and numbers are allowed")


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely (no eval of Python code)."""
    tree = ast.parse(expression, mode="eval")
    value = _eval_ast(tree)
    return str(value)


def get_weather(city: str) -> str:
    """Stub weather lookup. Returns an error string for unknown cities."""
    key = city.strip().lower()
    if key not in WEATHER:
        known = ", ".join(sorted(WEATHER))
        return f"ERROR: no stub data for '{city}'. Known cities: {known}"
    row = WEATHER[key]
    return f"{city.title()}: {row['temp_c']} C, {row['condition']}"


def read_sandbox_file(filename: str) -> str:
    """Read a text file from the local sandbox/ folder only (path traversal blocked)."""
    name = Path(filename).name
    path = (SANDBOX / name).resolve()
    if path.parent != SANDBOX.resolve():
        return "ERROR: path traversal is not allowed"
    if not path.is_file():
        return f"ERROR: file not found in sandbox: {name}"
    text = path.read_text(encoding="utf-8")
    return text[:4000]


def broken_lookup(query: str) -> str:
    """Deliberate failure tool for Task 5."""
    raise RuntimeError(f"upstream timeout while looking up {query!r}")


TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression and return the numeric result as text. "
            "Use for comparisons that need subtraction or conversion (for example C to F). "
            "Allowed operators: + - * / // % **. Do not send words or units, only the expression."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression such as '33 - 12' or '(9/5)*33 + 32'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Look up current weather for one city from a local stub database. "
            "Call once per city. Supported cities: Karachi, London, Tokyo, New York, Oslo. "
            "Returns temperature in Celsius and a short condition string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Karachi' or 'London'.",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "read_sandbox_file",
        "description": (
            "Read a text file from the agent's sandbox directory. "
            "Use when the user asks what is in a notes or briefing file. "
            "Pass only the filename, not a full path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File name inside sandbox/, e.g. 'briefing.txt'.",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "broken_lookup",
        "description": (
            "External lookup that currently fails with a timeout. "
            "Only use if the user explicitly asks for broken_lookup."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Lookup query string."}
            },
            "required": ["query"],
        },
    },
]

TOOL_FNS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "read_sandbox_file": read_sandbox_file,
    "broken_lookup": broken_lookup,
}

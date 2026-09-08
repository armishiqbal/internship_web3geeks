"""LangChain tools for Week 2 Day 2.

Each @tool docstring is injected into the model's tool schema — the LLM reads
these descriptions when deciding whether and how to call a function.
"""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path

from langchain_core.tools import StructuredTool, tool

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SANDBOX = HERE / "sandbox"

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


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result as text.

    Use for comparisons that need subtraction or unit conversion (for example C to F).
    Allowed operators: + - * / // % **. Send only the expression, not words or units.
    """
    tree = ast.parse(expression, mode="eval")
    value = _eval_ast(tree)
    return str(value)


@tool
def get_weather(city: str) -> str:
    """Look up current weather for one city from a local stub database.

    Call once per city. Supported cities: Karachi, London, Tokyo, New York, Oslo.
    Returns temperature in Celsius and a short condition string.
    """
    key = city.strip().lower()
    if key not in WEATHER:
        known = ", ".join(sorted(WEATHER))
        return f"ERROR: no stub data for '{city}'. Known cities: {known}"
    row = WEATHER[key]
    return f"{city.title()}: {row['temp_c']} C, {row['condition']}"


@tool
def lookup_product_price(product_name: str) -> str:
    """Look up monthly USD price for a SaaS product from the local JSON catalog.

    Reads data/products.json (no network). Product names are case-insensitive.
    Returns price, tier, and a one-line description when found.
    """
    path = DATA / "products.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    key = product_name.strip().lower()
    if key not in catalog:
        known = ", ".join(sorted(catalog))
        return f"ERROR: unknown product '{product_name}'. Known: {known}"
    row = catalog[key]
    return (
        f"{row['name']}: ${row['price_usd']}/month ({row['tier']}) — {row['description']}"
    )


def _broken_lookup_impl(query: str) -> str:
    """Implementation that always fails — used with ToolException handling."""
    from langchain_core.tools import ToolException

    raise ToolException(f"upstream timeout while looking up {query!r}")


broken_lookup = StructuredTool.from_function(
    func=_broken_lookup_impl,
    name="broken_lookup",
    description=(
        "External lookup that currently fails with a timeout. "
        "Only use if the user explicitly asks for broken_lookup."
    ),
    handle_tool_error=True,
)

CORE_TOOLS = [calculator, get_weather, lookup_product_price]
ALL_TOOLS = CORE_TOOLS + [broken_lookup]

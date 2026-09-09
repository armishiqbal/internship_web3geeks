"""LangChain tools for Week 2 Day 3 (LangGraph Workflows).

Reuses and extends Day 2 tools (calculator, get_weather, lookup_product_price)
for multi-step research, financial analysis, and decision workflows.
"""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path

from langchain_core.tools import tool

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

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

    Use for budget calculations, total cost of ownership, discounts, or metric conversions.
    Allowed operators: + - * / // % **. Send only the mathematical expression.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval_ast(tree)
        return str(value)
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def get_weather(city: str) -> str:
    """Look up current weather for one city from the local stub database.

    Supported cities: Karachi, London, Tokyo, New York, Oslo.
    """
    key = city.strip().lower()
    if key not in WEATHER:
        known = ", ".join(sorted(WEATHER))
        return f"ERROR: no data for '{city}'. Known cities: {known}"
    row = WEATHER[key]
    return f"{city.title()}: {row['temp_c']} C, {row['condition']}"


@tool
def lookup_product_price(product_name: str) -> str:
    """Look up monthly USD price and details for a SaaS product from the local JSON catalog.

    Supported products include: starter, pro, enterprise, slack, notion, github_copilot.
    """
    path = DATA / "products.json"
    if not path.is_file():
        path = HERE.parent / "day 2" / "data" / "products.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    key = product_name.strip().lower().replace(" ", "_")
    if key not in catalog:
        for k, v in catalog.items():
            if k in key or key in k or v["name"].lower() in key:
                row = v
                return f"{row['name']}: ${row['price_usd']}/month ({row['tier']}) — {row['description']}"
        known = ", ".join(sorted(catalog))
        return f"ERROR: unknown product '{product_name}'. Known: {known}"
    row = catalog[key]
    return f"{row['name']}: ${row['price_usd']}/month ({row['tier']}) — {row['description']}"


CORE_TOOLS = [calculator, get_weather, lookup_product_price]

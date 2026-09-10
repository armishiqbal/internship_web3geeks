"""Role-appropriate custom tools for Week 2 Day 4 CrewAI agents.

Tools are strictly partitioned:
- CompetitorCatalogTool -> Reserved for Agent 1 (Researcher)
- FinancialCalculatorTool -> Reserved for Agent 2 (Financial Analyst)
- BattlecardFormatterTool -> Reserved for Agent 3 (Marketing Strategist)
"""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool(BaseModel):  # type: ignore
        """Graceful fallback BaseTool when crewai is not yet loaded in active kernel."""
        name: str = ""
        description: str = ""
        args_schema: Any = None

        def _run(self, *args, **kwargs):
            raise NotImplementedError

        def run(self, *args, **kwargs):
            return self._run(*args, **kwargs)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Safe AST arithmetic operators
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


def _eval_ast(node: Any) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_ast(node.operand))
    raise ValueError("Only +, -, *, /, //, %, ** and numerical literals are permitted.")


# ---------------------------------------------------------------------------
# Tool 1: Competitor Catalog Tool (Researcher)
# ---------------------------------------------------------------------------
class CompetitorCatalogInput(BaseModel):
    """Input schema for looking up competitor intelligence."""
    competitor_name: str = Field(
        ...,
        description="Name of competitor to query: 'slack', 'notion', 'github_copilot', or 'all'"
    )


class CompetitorCatalogTool(BaseTool):
    name: str = "competitor_catalog_search"
    description: str = (
        "Look up verified pricing tiers, seat costs, feature limitations, storage quotas, "
        "and security compliance flags from the internal competitor intelligence database. "
        "Allowed queries: 'slack', 'notion', 'github_copilot', or 'all'."
    )
    args_schema: Type[BaseModel] = CompetitorCatalogInput

    def _run(self, competitor_name: str) -> str:
        db_path = DATA / "competitors.json"
        if not db_path.is_file():
            return f"ERROR: Competitor catalog database not found at {db_path}"
        
        try:
            catalog = json.loads(db_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return f"ERROR: Failed to parse catalog: {exc}"

        key = competitor_name.strip().lower().replace(" ", "_")
        if key in ["all", "list", "all_competitors"]:
            summary = []
            for name, data in catalog.items():
                tiers = ", ".join(data.get("tiers", {}).keys())
                summary.append(f"• {data['name']} ({data['category']}): Tiers [{tiers}]")
            return "Catalog Overview:\n" + "\n".join(summary)

        if key in catalog:
            return json.dumps(catalog[key], indent=2)

        # Fuzzy match
        for k, v in catalog.items():
            if k in key or key in k or v.get("name", "").lower() in key:
                return json.dumps(v, indent=2)

        known = ", ".join(catalog.keys())
        return f"Competitor '{competitor_name}' not found. Available competitors: {known}"


# ---------------------------------------------------------------------------
# Tool 2: Financial Calculator Tool (Financial Analyst)
# ---------------------------------------------------------------------------
class FinancialCalculatorInput(BaseModel):
    """Input schema for financial arithmetic evaluations."""
    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate, e.g. '15.00 * 50 * 12' or '(32 - 15) / 15 * 100'"
    )


class FinancialCalculatorTool(BaseTool):
    name: str = "financial_tco_calculator"
    description: str = (
        "Evaluate exact mathematical expressions for multi-tier pricing, team seat TCO, "
        "annual discount percentages, and add-on cost projections. "
        "Send only standard arithmetic expressions with +, -, *, /, %, **."
    )
    args_schema: Type[BaseModel] = FinancialCalculatorInput

    def _run(self, expression: str) -> str:
        clean_expr = expression.strip().replace("$", "").replace(",", "")
        try:
            tree = ast.parse(clean_expr, mode="eval")
            result = _eval_ast(tree)
            if isinstance(result, float):
                return f"{result:.2f}"
            return str(result)
        except Exception as exc:
            return f"Calculation Error in expression '{expression}': {exc}"


# ---------------------------------------------------------------------------
# Tool 3: Battlecard Formatter Tool (Marketing Strategist)
# ---------------------------------------------------------------------------
class BattlecardFormatterInput(BaseModel):
    """Input schema for formatting the final marketing battlecard."""
    title: str = Field(..., description="Title of the battlecard report")
    executive_summary: str = Field(..., description="High-level executive takeaways")
    tco_comparison: str = Field(..., description="Quantitative TCO markdown table or metrics")
    counter_angles: str = Field(..., description="3 actionable sales counter-positioning angles")


class BattlecardFormatterTool(BaseTool):
    name: str = "battlecard_formatter"
    description: str = (
        "Validates and formats the final executive competitive battlecard into a standardized "
        "C-suite ready deliverable with proper markdown typography, callout anchors, and sales guidance."
    )
    args_schema: Type[BaseModel] = BattlecardFormatterInput

    def _run(self, title: str, executive_summary: str, tco_comparison: str, counter_angles: str) -> str:
        report = [
            f"# {title}",
            "",
            "## 1. Executive Intelligence Summary",
            executive_summary.strip(),
            "",
            "## 2. Quantitative Total Cost of Ownership (TCO) Analysis",
            tco_comparison.strip(),
            "",
            "## 3. Strategic Sales Counter-Angles & Objection Handling",
            counter_angles.strip(),
            "",
            "---",
            "*Deliverable generated by CrewAI Multi-Agent Collaboration Engine.*"
        ]
        return "\n".join(report)


# Tool mapping collections
RESEARCHER_TOOLS = [CompetitorCatalogTool()]
ANALYST_TOOLS = [FinancialCalculatorTool()]
WRITER_TOOLS = [BattlecardFormatterTool()]
ALL_TOOLS = [CompetitorCatalogTool(), FinancialCalculatorTool(), BattlecardFormatterTool()]

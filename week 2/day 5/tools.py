"""Core tools for Web3Geeks Client Onboarding Agent: Catalog Search, AST Calculator, and Formatter."""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_PATH = Path(__file__).resolve().parent / "data" / "services_catalog.json"


class ServiceCatalogSearchTool:
    """Searches the Web3Geeks service offerings and rate card database."""

    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self._catalog: Optional[Dict[str, Any]] = None

    def _load(self) -> Dict[str, Any]:
        if self._catalog is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Catalog database not found at {self.data_path}")
            with open(self.data_path, "r", encoding="utf-8") as f:
                self._catalog = json.load(f)
        return self._catalog

    def search(self, query: str) -> Dict[str, Any]:
        """Searches catalog for matching services by keyword or category."""
        catalog = self._load()
        services = catalog.get("services", {})
        query_norm = query.lower().strip()

        if not query_norm:
            return {
                "status": "error",
                "message": "Empty query provided. Please specify service keywords.",
                "available_services": list(services.keys()),
            }

        # Exact key match
        if query_norm in services:
            return {"status": "success", "match_type": "exact", "service": services[query_norm]}

        # Substring / keyword match
        matches = []
        for key, sdata in services.items():
            combined_text = (
                f"{key} {sdata.get('name', '')} {sdata.get('category', '')} "
                f"{' '.join(sdata.get('tech_stack', []))} {' '.join(sdata.get('deliverables', []))}"
            ).lower()
            if any(term in combined_text for term in query_norm.split()):
                matches.append({"key": key, **sdata})

        if matches:
            return {
                "status": "success",
                "match_type": "fuzzy",
                "count": len(matches),
                "services": matches,
            }

        return {
            "status": "not_found",
            "message": f"No direct service matched query: '{query}'",
            "available_services": list(services.keys()),
        }

    def get_sla(self, tier: str = "standard") -> Dict[str, Any]:
        catalog = self._load()
        slas = catalog.get("sla_tiers", {})
        return slas.get(tier.lower(), slas.get("standard", {}))

    def get_minimum_budget(self) -> float:
        catalog = self._load()
        return float(catalog.get("minimum_project_budget", 3500.0))


class DeterministicBudgetCalculatorTool:
    """Safe AST arithmetic calculator for milestone calculations and financial proofs."""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(self, expression: str) -> Dict[str, Any]:
        """Safely evaluates an arithmetic expression using Python's Abstract Syntax Tree."""
        clean_expr = expression.strip().replace("$", "").replace(",", "")
        if not clean_expr:
            return {"status": "error", "error": "Empty arithmetic expression."}

        try:
            tree = ast.parse(clean_expr, mode="eval")
            result = self._eval_node(tree.body)
            result_float = round(float(result), 2)
            return {
                "status": "success",
                "expression": clean_expr,
                "result": result_float,
                "formatted": f"${result_float:,.2f}",
            }
        except (SyntaxError, ZeroDivisionError, ValueError, TypeError) as e:
            return {
                "status": "error",
                "expression": clean_expr,
                "error": f"{type(e).__name__}: {str(e)}",
            }

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"Disallowed binary operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"Disallowed unary operator: {op_type}")
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")


class ContractTemplateFormatterTool:
    """Validates and ensures client proposals adhere to Web3Geeks governance standards."""

    REQUIRED_SECTIONS = [
        "Executive Project Summary",
        "Technical Architecture & Deliverables",
        "Quantitative Investment & Milestone Schedule",
        "SLA & Timeline Commitment",
        "Human Approval & Sign-Off Checkpoint",
    ]

    def validate_structure(self, proposal_markdown: str) -> Dict[str, Any]:
        """Validates that all required headers and disclosure terms exist."""
        missing = [s for s in self.REQUIRED_SECTIONS if s.lower() not in proposal_markdown.lower()]
        has_pricing = any(char in proposal_markdown for char in ["$", "USD", "Total"])
        has_milestones = "milestone" in proposal_markdown.lower()

        is_valid = len(missing) == 0 and has_pricing and has_milestones
        return {
            "is_valid": is_valid,
            "missing_sections": missing,
            "has_pricing": has_pricing,
            "has_milestones": has_milestones,
            "score": max(0, 100 - (len(missing) * 20)),
        }

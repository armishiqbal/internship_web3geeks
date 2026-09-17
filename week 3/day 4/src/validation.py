"""
Week 3 Day 4 — Validation, Self-Correction & Fallbacks Node
===========================================================
Validates outputs from prediction, structured retrieval, and factual knowledge:
1. Verifies tool output returned valid payload (status == 'success').
2. Intercepts ambiguous or unresolved clubs/players and routes to clarification.
3. Detects unsupported predictive queries (unmodeled stats) and routes to clear out-of-scope fallback.
4. Prevents hallucination by guaranteeing no unsupported guesses reach the user.
"""

from typing import Dict, Any, Optional
from src.state import AFLGraphState


class AFLValidator:
    """Validates intermediate tool execution states and determines routing to formatting vs clarification."""

    @staticmethod
    def validate(state: AFLGraphState) -> Dict[str, Any]:
        tool_res = state.get("tool_results")
        tool_name = state.get("tool_called", "unknown")
        entities = state.get("extracted_entities", {})

        # 1. Check if tool returned None or execution failed
        if tool_res is None:
            return {
                "validation_status": "error",
                "error_message": f"No result returned from tool '{tool_name}'.",
                "clarification_prompt": None
            }

        status = tool_res.get("status")

        # 2. Check for Unsupported Stat Request
        if status == "unsupported" or entities.get("is_unsupported_stat"):
            req_stat = tool_res.get("requested_stat") or entities.get("stat_type") or "unsupported metric"
            supported = tool_res.get("supported_stats", ["disposals", "goals", "fantasy_points", "player_impact_score"])
            supported_str = ", ".join(f"'{s}'" for s in supported)
            
            fallback_msg = (
                f"### Out of Scope: Unsupported Predictive Stat\n\n"
                f"Our predictive machine learning models are trained specifically on **match winner probabilities** "
                f"and four primary player performance metrics: {supported_str}.\n\n"
                f"Statistical forecasting for **'{req_stat}'** is currently not supported by our verified models. "
                f"To maintain mathematical integrity and prevent speculative hallucinations, we cannot generate a forecast for {req_stat}.\n\n"
                f"**What you can ask instead:**\n"
                f"- Predict match winner (e.g. *'Will the Pies beat the Cats this week?'*)\n"
                f"- Predict player disposals or goals (e.g. *'Who will top-score in disposals for Western Bulldogs?'*)\n"
                f"- Retrieve historical stats for {req_stat} if available in our season records."
            )
            return {
                "validation_status": "unsupported",
                "error_message": f"Unsupported prediction stat: {req_stat}",
                "clarification_prompt": fallback_msg
            }

        # 3. Check for Ambiguous or Unresolved Entities (Requires Clarification)
        if status == "needs_clarification":
            missing = tool_res.get("missing_entity", "team")
            raw_input = tool_res.get("raw_input")
            hint = tool_res.get("hint")
            msg = tool_res.get("message")

            if raw_input and hint:
                clarification_prompt = (
                    f"### Clarification Required: Unrecognized Club\n\n"
                    f"I couldn't identify the club or entity **'{raw_input}'** in our AFL database.\n\n"
                    f"💡 **Suggestion:** {hint}\n\n"
                    f"Could you please clarify which AFL team you meant?"
                )
            elif msg:
                clarification_prompt = (
                    f"### Clarification Required\n\n"
                    f"{msg}\n\n"
                    f"Please provide the specific AFL club or player name so I can proceed with an accurate lookup."
                )
            else:
                clarification_prompt = (
                    f"### Clarification Required\n\n"
                    f"I noticed some missing or ambiguous information in your request. "
                    f"Could you please specify the AFL team or player you would like to analyze?"
                )

            return {
                "validation_status": "needs_clarification",
                "error_message": f"Ambiguous {missing} input",
                "clarification_prompt": clarification_prompt
            }

        # 4. Check for 'not_found' in retrieval tools
        if status == "not_found" or status == "round_not_played":
            msg = tool_res.get("message", "No matching historical records found.")
            clarification_prompt = (
                f"### Record Not Found\n\n"
                f"{msg}\n\n"
                f"Please verify the player name, season year, or round number, or ask about another AFL fixture."
            )
            return {
                "validation_status": "needs_clarification",
                "error_message": msg,
                "clarification_prompt": clarification_prompt
            }

        # 5. Check for explicit error status
        if status == "error":
            err = tool_res.get("error", "Unknown tool execution failure.")
            return {
                "validation_status": "error",
                "error_message": err,
                "clarification_prompt": f"An error occurred while executing the request: {err}"
            }

        # 6. Success / Valid
        return {
            "validation_status": "valid",
            "error_message": None,
            "clarification_prompt": None
        }


def validation_node(state: AFLGraphState) -> Dict[str, Any]:
    """
    LangGraph Validation Node:
    Audits tool output, decides if state is valid, requires user clarification,
    or triggers an unsupported fallback.
    """
    val_result = AFLValidator.validate(state)

    trace_entry = {
        "node": "validation_node",
        "action": "validate_tool_output",
        "tool_called": state.get("tool_called"),
        "validation_status": val_result["validation_status"],
        "error_message": val_result["error_message"]
    }

    return {
        "validation_status": val_result["validation_status"],
        "error_message": val_result["error_message"],
        "clarification_prompt": val_result["clarification_prompt"],
        "trace": [trace_entry]
    }

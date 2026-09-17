"""
Week 3 Day 4 — LangGraph State Schema Definition
=================================================
Defines the centralized, typed state container orchestrating the AFL Intelligence
graph. Tracks user input, multi-turn history, intent routing, resolved entities,
raw tool outputs, validation status, execution traces, and synthesized responses.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
import operator


class AFLGraphState(TypedDict):
    """
    Centralized state definition for the AFL LangGraph Orchestrator.
    
    Fields:
        user_query: The incoming text prompt from the user for the current turn.
        conversation_history: List of conversational message dicts ({'role': 'user'|'assistant', 'content': str}).
        detected_intent: Classified intent: 'factual', 'retrieval', 'prediction', 'off_topic'.
        intent_confidence: Confidence probability [0.0 - 1.0] of the router classification.
        intent_reasoning: Explanation of why the query was routed to this intent.
        extracted_entities: Dictionary of resolved entities (canonical teams, players, season, round, stat_type).
        tool_called: Identifier of the tool or predictive pipeline invoked during this turn.
        tool_results: Raw structured JSON/dict payload returned by the executed tool or model.
        validation_status: Status from the validator: 'valid', 'needs_clarification', 'unsupported', 'error'.
        error_message: Detailed explanation if validation, entity resolution, or model execution failed.
        clarification_prompt: Guided follow-up question when user input is ambiguous or missing key entities.
        final_response: The final synthesized, formatted, and safety-framed message delivered to the user.
        trace: Chronological audit log of all graph node transitions and decision metadata.
    """
    user_query: str
    conversation_history: Annotated[List[Dict[str, str]], operator.add]
    detected_intent: Literal['factual', 'retrieval', 'prediction', 'off_topic', 'clarification']
    intent_confidence: float
    intent_reasoning: str
    extracted_entities: Dict[str, Any]
    tool_called: Optional[str]
    tool_results: Optional[Dict[str, Any]]
    validation_status: Literal['valid', 'needs_clarification', 'unsupported', 'error']
    error_message: Optional[str]
    clarification_prompt: Optional[str]
    final_response: str
    trace: Annotated[List[Dict[str, Any]], operator.add]


def create_initial_state(user_query: str, history: Optional[List[Dict[str, str]]] = None) -> AFLGraphState:
    """Initializes a clean graph state with default values for a new query."""
    return {
        "user_query": user_query,
        "conversation_history": list(history) if history else [],
        "detected_intent": "factual",
        "intent_confidence": 0.0,
        "intent_reasoning": "",
        "extracted_entities": {},
        "tool_called": None,
        "tool_results": None,
        "validation_status": "valid",
        "error_message": None,
        "clarification_prompt": None,
        "final_response": "",
        "trace": []
    }

"""
Week 3 Day 4 — LangGraph Application Assembly & Orchestration
=============================================================
Constructs and compiles the complete LangGraph StateGraph connecting:
- Intent Classifier Router
- Off-Topic Refusal Node
- Prediction Engine Tool Node (Day 2 Calibrated Models)
- Structured Stats Retrieval Tool Node (Day 3 Feature Tables)
- Factual AFL Knowledge Node (Day 3 Semantic Vector Store)
- Validation & Clarification Node
- Response Formatter Node (Probabilistic Framing & Disclaimers)

Provides clean, production-grade `.invoke()`, `.stream()`, and `.run_turn()` execution methods.
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

from src.state import AFLGraphState, create_initial_state
from src.router import router_node
from src.tools_adapter import AFLToolsAdapter
from src.validation import validation_node
from src.formatting import response_formatter_node


# ------------------------------------------------------------------------------
# Node 1: Prediction Engine Node
# ------------------------------------------------------------------------------
def prediction_node(state: AFLGraphState) -> Dict[str, Any]:
    """Resolves entities and executes calibrated Day 2 ML models."""
    query = state.get("user_query", "")
    history = state.get("conversation_history", [])

    entities = AFLToolsAdapter.extract_entities_from_query(query, history)
    tool_results = AFLToolsAdapter.execute_prediction(entities, query)
    tool_name = tool_results.get("model_type", "predict_match_winner")

    trace_entry = {
        "node": "prediction_node",
        "action": "execute_prediction",
        "extracted_entities": entities,
        "tool_called": tool_name,
        "tool_status": tool_results.get("status")
    }

    return {
        "extracted_entities": entities,
        "tool_called": tool_name,
        "tool_results": tool_results,
        "trace": [trace_entry]
    }


# ------------------------------------------------------------------------------
# Node 2: Structured Retrieval Node
# ------------------------------------------------------------------------------
def structured_retrieval_node(state: AFLGraphState) -> Dict[str, Any]:
    """Resolves entities and queries Day 3 historical stats tables."""
    query = state.get("user_query", "")
    history = state.get("conversation_history", [])

    entities = AFLToolsAdapter.extract_entities_from_query(query, history)
    tool_results = AFLToolsAdapter.execute_retrieval(entities, query)
    tool_name = tool_results.get("retrieval_type", "structured_stats")

    trace_entry = {
        "node": "structured_retrieval_node",
        "action": "execute_retrieval",
        "extracted_entities": entities,
        "tool_called": tool_name,
        "tool_status": tool_results.get("status")
    }

    return {
        "extracted_entities": entities,
        "tool_called": tool_name,
        "tool_results": tool_results,
        "trace": [trace_entry]
    }


# ------------------------------------------------------------------------------
# Node 3: Factual Knowledge Node
# ------------------------------------------------------------------------------
def factual_knowledge_node(state: AFLGraphState) -> Dict[str, Any]:
    """Queries Day 3 AFL domain vector store for rules, heritage, grounds."""
    query = state.get("user_query", "")
    tool_results = AFLToolsAdapter.execute_factual(query)

    trace_entry = {
        "node": "factual_knowledge_node",
        "action": "execute_factual_search",
        "tool_called": "search_afl_knowledge",
        "tool_status": tool_results.get("status")
    }

    return {
        "tool_called": "search_afl_knowledge",
        "tool_results": tool_results,
        "trace": [trace_entry]
    }


# ------------------------------------------------------------------------------
# Node 4: Off-Topic Refusal Node
# ------------------------------------------------------------------------------
def off_topic_refusal_node(state: AFLGraphState) -> Dict[str, Any]:
    """Generates refusal metadata for non-AFL queries."""
    trace_entry = {
        "node": "off_topic_refusal_node",
        "action": "intercept_off_topic_query",
        "reason": state.get("intent_reasoning")
    }

    return {
        "tool_called": "guardrail_refusal",
        "tool_results": {"status": "off_topic"},
        "validation_status": "valid",
        "trace": [trace_entry]
    }


# ------------------------------------------------------------------------------
# Node 5: Clarification & Fallback Node
# ------------------------------------------------------------------------------
def clarification_node(state: AFLGraphState) -> Dict[str, Any]:
    """Handles ambiguous or unsupported queries before final response formatting."""
    val_status = state.get("validation_status")
    err = state.get("error_message")

    trace_entry = {
        "node": "clarification_node",
        "action": "prepare_clarification_or_fallback",
        "validation_status": val_status,
        "error_message": err
    }

    return {
        "trace": [trace_entry]
    }


# ------------------------------------------------------------------------------
# Conditional Edge Routing Functions
# ------------------------------------------------------------------------------
def route_intent_edge(state: AFLGraphState) -> str:
    """Branches execution from router_node based on classified intent."""
    intent = state.get("detected_intent")
    if intent == "off_topic":
        return "off_topic_refusal_node"
    elif intent == "prediction":
        return "prediction_node"
    elif intent == "retrieval":
        return "structured_retrieval_node"
    elif intent == "factual":
        return "factual_knowledge_node"
    return "factual_knowledge_node"


def route_validation_edge(state: AFLGraphState) -> str:
    """Branches execution from validation_node based on validation status."""
    val_status = state.get("validation_status", "valid")
    if val_status in ["needs_clarification", "unsupported", "error"]:
        return "clarification_node"
    return "response_formatter_node"


# ------------------------------------------------------------------------------
# LangGraph Graph Assembly & Compilation
# ------------------------------------------------------------------------------
def build_afl_graph():
    """
    Constructs and compiles the complete LangGraph StateGraph.
    
    Topology:
      START -> router_node
        ├─> off_topic_refusal_node -> response_formatter_node -> END
        ├─> prediction_node ----------┐
        ├─> structured_retrieval_node ┼─> validation_node
        └─> factual_knowledge_node ---┘        ├─> [valid] ---------------> response_formatter_node -> END
                                               └─> [clarify/unsupported] -> clarification_node -> response_formatter_node -> END
    """
    workflow = StateGraph(AFLGraphState)

    # 1. Register all nodes
    workflow.add_node("router_node", router_node)
    workflow.add_node("off_topic_refusal_node", off_topic_refusal_node)
    workflow.add_node("prediction_node", prediction_node)
    workflow.add_node("structured_retrieval_node", structured_retrieval_node)
    workflow.add_node("factual_knowledge_node", factual_knowledge_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("response_formatter_node", response_formatter_node)

    # 2. Add entry edge
    workflow.add_edge(START, "router_node")

    # 3. Add conditional routing edges from router_node
    workflow.add_conditional_edges(
        "router_node",
        route_intent_edge,
        {
            "off_topic_refusal_node": "off_topic_refusal_node",
            "prediction_node": "prediction_node",
            "structured_retrieval_node": "structured_retrieval_node",
            "factual_knowledge_node": "factual_knowledge_node"
        }
    )

    # 4. Connect tool nodes to validation_node
    workflow.add_edge("prediction_node", "validation_node")
    workflow.add_edge("structured_retrieval_node", "validation_node")
    workflow.add_edge("factual_knowledge_node", "validation_node")

    # 5. Connect off_topic_refusal_node directly to formatter
    workflow.add_edge("off_topic_refusal_node", "response_formatter_node")

    # 6. Add conditional edge from validation_node
    workflow.add_conditional_edges(
        "validation_node",
        route_validation_edge,
        {
            "response_formatter_node": "response_formatter_node",
            "clarification_node": "clarification_node"
        }
    )

    # 7. Connect clarification_node to response_formatter_node
    workflow.add_edge("clarification_node", "response_formatter_node")

    # 8. Connect response_formatter_node to END
    workflow.add_edge("response_formatter_node", END)

    # Compile the graph
    app = workflow.compile()
    return app


# Singleton compiled app
_COMPILED_AFL_GRAPH = None


def get_afl_graph():
    """Returns singleton compiled LangGraph application."""
    global _COMPILED_AFL_GRAPH
    if _COMPILED_AFL_GRAPH is None:
        _COMPILED_AFL_GRAPH = build_afl_graph()
    return _COMPILED_AFL_GRAPH


def run_afl_turn(user_query: str, history: Optional[List[Dict[str, str]]] = None) -> AFLGraphState:
    """
    Executes a single conversational turn through the compiled LangGraph pipeline.
    
    Args:
        user_query: The incoming user message string.
        history: Prior conversation history list of message dicts.
        
    Returns:
        Full AFLGraphState dict containing final_response, trace, entities, and tool results.
    """
    app = get_afl_graph()
    initial_state = create_initial_state(user_query, history)
    final_state = app.invoke(initial_state)
    return final_state
def run_afl_agent(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None
) -> AFLGraphState:
    return run_afl_turn(user_query, history)
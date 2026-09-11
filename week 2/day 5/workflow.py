"""Production-grade Web3Geeks Client Onboarding & Scoping Agent Workflow.
Combines LangGraph stateful DAG control, cyclic critique loops, HITL approval, and AST tools.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

import config
from tools import (
    ServiceCatalogSearchTool,
    DeterministicBudgetCalculatorTool,
    ContractTemplateFormatterTool,
)


class OnboardingState(TypedDict):
    """Complete typed state for client onboarding pipeline."""
    # Inputs
    client_inquiry: str
    client_name: str
    budget_max: Optional[float]
    sla_tier: str

    # Validation & Diagnostics
    is_valid: bool
    validation_status: str
    error_message: Optional[str]

    # Scoping & Tools
    catalog_matches: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    total_cost: float
    estimated_days: int

    # Drafting & Critique Loop
    proposal_draft: str
    critique_score: float
    critique_feedback: str
    revision_count: int

    # Human-in-the-Loop Gate
    human_approval_required: bool
    human_approved: Optional[bool]
    contract_dispatched: bool

    # Metrics & Logs
    execution_logs: List[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    cost_usd: float


# Initialize singletons
_catalog_tool = ServiceCatalogSearchTool()
_calc_tool = DeterministicBudgetCalculatorTool()
_formatter_tool = ContractTemplateFormatterTool()


# =============================================================================
# NODE 1: INPUT VALIDATION & ADVERSARIAL FILTERING
# =============================================================================
def validate_inquiry_node(state: OnboardingState) -> Dict[str, Any]:
    """Validates inquiry, screens for prompt injections, and checks budget bounds."""
    logs = list(state.get("execution_logs", []))
    logs.append("[Stage 1: Input Validation] Screening client inquiry...")

    inquiry = state.get("client_inquiry", "").strip()
    budget_max = state.get("budget_max")
    min_budget = _catalog_tool.get_minimum_budget()

    # 1. Check for empty input
    if not inquiry or len(inquiry) < 10:
        logs.append("  ↳ Validation Failed: Inquiry is too short or empty.")
        return {
            "is_valid": False,
            "validation_status": "failed_empty_input",
            "error_message": "Client inquiry must contain at least 10 characters detailing project scope.",
            "execution_logs": logs,
        }

    # 2. Check for adversarial / prompt injection patterns
    adversarial_keywords = [
        "ignore all previous instructions",
        "system prompt override",
        "drop table",
        "<script>",
        "bypass security",
    ]
    if any(kw in inquiry.lower() for kw in adversarial_keywords):
        logs.append("  ↳ Security Alert: Adversarial prompt injection attempt detected.")
        return {
            "is_valid": False,
            "validation_status": "rejected_adversarial",
            "error_message": "Inquiry flagged by security filter for disallowed instructions.",
            "execution_logs": logs,
        }

    # 3. Check for sub-minimum budget
    if budget_max is not None and budget_max < min_budget:
        logs.append(f"  ↳ Budget Rejection: Proposed budget ${budget_max} is below minimum threshold ${min_budget}.")
        return {
            "is_valid": False,
            "validation_status": "failed_low_budget",
            "error_message": f"Project budget (${budget_max:,.2f}) is below Web3Geeks minimum project engagement threshold of ${min_budget:,.2f}.",
            "execution_logs": logs,
        }

    logs.append("  ↳ Validation Passed: Clean inquiry verified.")
    return {
        "is_valid": True,
        "validation_status": "passed",
        "error_message": None,
        "execution_logs": logs,
    }


# =============================================================================
# NODE 2: SERVICE SCOPING & CATALOG AUDITING
# =============================================================================
def scope_services_node(state: OnboardingState) -> Dict[str, Any]:
    """Queries service catalog to discover matching Web3 offerings."""
    logs = list(state.get("execution_logs", []))
    logs.append("[Stage 2: Service Scoping] Auditing Web3Geeks service catalog...")

    inquiry = state.get("client_inquiry", "")
    search_res = _catalog_tool.search(inquiry)

    matched_services = []
    if search_res.get("status") == "success":
        if "service" in search_res:
            matched_services = [search_res["service"]]
        elif "services" in search_res:
            matched_services = search_res["services"]
    else:
        # Fallback to general dApp fullstack service if query is ambiguous
        fallback = _catalog_tool.search("dapp_fullstack")
        matched_services = [fallback.get("service", {})]
        logs.append("  ↳ Note: Direct service match ambiguous; applied default Full-Stack Web3 dApp tier.")

    names = [s.get("name", "Unknown") for s in matched_services]
    logs.append(f"  ↳ Identified {len(matched_services)} service module(s): {', '.join(names)}")

    return {
        "catalog_matches": matched_services,
        "execution_logs": logs,
    }


# =============================================================================
# NODE 3: DETERMINISTIC BUDGET & MILESTONE CALCULATION
# =============================================================================
def calculate_budget_node(state: OnboardingState) -> Dict[str, Any]:
    """Calculates deterministic milestone pricing using AST expressions."""
    logs = list(state.get("execution_logs", []))
    logs.append("[Stage 3: Quantitative Estimation] Evaluating milestone pricing via AST...")

    matches = state.get("catalog_matches", [])
    sla_tier = state.get("sla_tier", "standard")
    sla_info = _catalog_tool.get_sla(sla_tier)

    # Base total calculation
    base_sum = sum(s.get("base_price", 4000.0) for s in matches)
    est_days = sum(s.get("estimated_days", 14) for s in matches)

    # SLA adjustments
    rush_mult = 1.25 if sla_tier == "expedited" else 1.0
    turnaround_mult = sla_info.get("turnaround_multiplier", 1.0)
    adjusted_days = max(5, int(est_days * turnaround_mult))

    # Calculate exact total via AST calculator
    expr = f"{base_sum} * {rush_mult}"
    eval_res = _calc_tool.evaluate(expr)
    total_val = eval_res.get("result", base_sum * rush_mult)

    # Milestone distribution: 40% Architecture, 40% Development, 20% Handover
    m1_val = _calc_tool.evaluate(f"{total_val} * 0.40").get("result", total_val * 0.4)
    m2_val = _calc_tool.evaluate(f"{total_val} * 0.40").get("result", total_val * 0.4)
    m3_val = _calc_tool.evaluate(f"{total_val} - ({m1_val} + {m2_val})").get("result", total_val * 0.2)

    milestones = [
        {
            "index": 1,
            "name": "Milestone 1: Architecture, Security Modeling & Formal Specification",
            "percentage": "40%",
            "amount": m1_val,
            "timeline": f"Days 1 – {max(3, adjusted_days // 3)}",
            "deliverable": "Technical architecture blueprint, data schema, and security threat model.",
        },
        {
            "index": 2,
            "name": "Milestone 2: Core Engineering, Smart Contracts & Integration",
            "percentage": "40%",
            "amount": m2_val,
            "timeline": f"Days {max(3, adjusted_days // 3) + 1} – {max(6, (2 * adjusted_days) // 3)}",
            "deliverable": "Working contract/dApp code, unit test suite, and staging environment.",
        },
        {
            "index": 3,
            "name": "Milestone 3: Audit Sign-Off, Mainnet Deployment & SLA Handover",
            "percentage": "20%",
            "amount": m3_val,
            "timeline": f"Days {max(6, (2 * adjusted_days) // 3) + 1} – {adjusted_days}",
            "deliverable": "Production release, audited verification report, and multi-sig handover.",
        },
    ]

    logs.append(f"  ↳ Total Investment: ${total_val:,.2f} over {adjusted_days} business days ({sla_tier.upper()} SLA).")
    logs.append(f"  ↳ Verified 3 Milestones: M1=${m1_val:,.2f}, M2=${m2_val:,.2f}, M3=${m3_val:,.2f}")

    return {
        "milestones": milestones,
        "total_cost": total_val,
        "estimated_days": adjusted_days,
        "execution_logs": logs,
    }


# =============================================================================
# NODE 4: PROPOSAL SYNTHESIS (DRAFTING)
# =============================================================================
def generate_proposal_node(state: OnboardingState) -> Dict[str, Any]:
    """Synthesizes executive proposal incorporating verified catalog items and AST math."""
    logs = list(state.get("execution_logs", []))
    rev_count = state.get("revision_count", 0)
    logs.append(f"[Stage 4: Proposal Drafting] Generating executive proposal (Iteration {rev_count + 1})...")

    client = state.get("client_name", "Prospective Enterprise Client")
    inquiry = state.get("client_inquiry", "")
    matches = state.get("catalog_matches", [])
    milestones = state.get("milestones", [])
    total_cost = state.get("total_cost", 0.0)
    est_days = state.get("estimated_days", 14)
    sla_tier = state.get("sla_tier", "standard")
    critique_notes = state.get("critique_feedback", "")

    services_text = "\n".join(
        f"• **{s.get('name')}** (${s.get('hourly_rate')}/hr) - Deliverables: {', '.join(s.get('deliverables', []))}"
        for s in matches
    )

    milestone_table = (
        "| Milestone | Scope & Deliverables | Investment ($ USD) | Schedule |\n"
        "| :--- | :--- | :---: | :---: |\n"
    )
    for m in milestones:
        milestone_table += f"| **{m['name']}** | {m['deliverable']} | ${m['amount']:,.2f} | {m['timeline']} |\n"
    milestone_table += f"| **TOTAL COMMITMENT** | **Complete Delivery Suite** | **${total_cost:,.2f}** | **{est_days} Days** |\n"

    system_prompt = (
        "You are the Principal Solutions Architect at Web3Geeks, a premier blockchain engineering agency. "
        "Draft a formal, enterprise-grade Client Proposal and Statement of Work (SOW). "
        "You MUST include the following exact markdown section headers:\n"
        "## 1. Executive Project Summary\n"
        "## 2. Technical Architecture & Deliverables\n"
        "## 3. Quantitative Investment & Milestone Schedule\n"
        "## 4. SLA & Timeline Commitment\n"
        "## 5. Human Approval & Sign-Off Checkpoint\n\n"
        "Ground all numbers exactly in the provided milestone table. Do NOT invent alternative prices."
    )

    user_prompt = f"""
Client Name: {client}
Client Inquiry: {inquiry}

Selected Service Offerings:
{services_text}

Milestone Breakdown (AST-Verified):
{milestone_table}

SLA Commitment: {sla_tier.upper()} Tier (Turnaround: {est_days} business days).
Previous Critique Notes to Address: {critique_notes if critique_notes else "Initial draft; ensure rigorous clarity."}
"""

    prompt_toks = 0
    comp_toks = 0

    try:
        llm = config.build_llm(temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        draft = response.content
        if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            usage = response.response_metadata["token_usage"]
            prompt_toks = usage.get("prompt_tokens", 0)
            comp_toks = usage.get("completion_tokens", 0)
        else:
            prompt_toks = len(system_prompt + user_prompt) // 4
            comp_toks = len(draft) // 4
    except Exception as e:
        logs.append(f"  ↳ Warning: LLM invocation failed ({str(e)}); falling back to deterministic template.")
        draft = f"""# Web3Geeks Enterprise Statement of Work

## 1. Executive Project Summary
Proposal prepared for **{client}** regarding: {inquiry}.
Web3Geeks will execute this engagement under a fixed-milestone framework.

## 2. Technical Architecture & Deliverables
{services_text}

## 3. Quantitative Investment & Milestone Schedule
{milestone_table}

## 4. SLA & Timeline Commitment
• SLA Tier: **{sla_tier.upper()}**
• Estimated Delivery: **{est_days} Business Days**
• Maintenance: Standard 30-day post-launch hypercare included.

## 5. Human Approval & Sign-Off Checkpoint
This proposal represents a legally binding scope pending final Web3Geeks Partner approval.
"""
        prompt_toks = 850
        comp_toks = 450

    logs.append("  ↳ Draft synthesized successfully.")
    return {
        "proposal_draft": draft,
        "prompt_tokens": state.get("prompt_tokens", 0) + prompt_toks,
        "completion_tokens": state.get("completion_tokens", 0) + comp_toks,
        "execution_logs": logs,
    }


# =============================================================================
# NODE 5: SELF-CORRECTION & CRITIQUE LOOP
# =============================================================================
def critique_proposal_node(state: OnboardingState) -> Dict[str, Any]:
    """Automated evaluation of proposal against quality, structure, and budget accuracy."""
    logs = list(state.get("execution_logs", []))
    logs.append("[Stage 5: Quality Critique] Auditing proposal completeness & governance...")

    draft = state.get("proposal_draft", "")
    val_res = _formatter_tool.validate_structure(draft)

    score = float(val_res.get("score", 70.0))
    feedback = []

    if val_res.get("missing_sections"):
        feedback.append(f"Missing mandatory sections: {', '.join(val_res['missing_sections'])}")
    if not val_res.get("has_pricing"):
        feedback.append("Missing explicit currency ($ / USD) symbols.")
    if not val_res.get("has_milestones"):
        feedback.append("Milestone payment table must be explicitly itemized.")

    if not feedback:
        feedback_str = "Proposal meets 100% of Web3Geeks quality and governance standards."
        score = 100.0
    else:
        feedback_str = " | ".join(feedback)

    rev_count = state.get("revision_count", 0)
    logs.append(f"  ↳ Critique Score: {score}/100. Feedback: {feedback_str}")

    return {
        "critique_score": score,
        "critique_feedback": feedback_str,
        "revision_count": rev_count + 1,
        "execution_logs": logs,
    }


# =============================================================================
# CONDITIONAL ROUTING: REVISION LOOP OR PROCEED TO HITL
# =============================================================================
def critique_router(state: OnboardingState) -> str:
    """Decides whether to trigger a self-correction loop or proceed to approval."""
    score = state.get("critique_score", 0.0)
    revisions = state.get("revision_count", 0)

    if score < 80.0 and revisions < 2:
        return "generate_proposal"
    return "human_checkpoint"


# =============================================================================
# NODE 6: HUMAN-IN-THE-LOOP (HITL) APPROVAL CHECKPOINT
# =============================================================================
def human_checkpoint_node(state: OnboardingState) -> Dict[str, Any]:
    """Enforces human verification before committing agency resources or dispatching contracts."""
    logs = list(state.get("execution_logs", []))
    logs.append("[Stage 6: Human-in-the-Loop Checkpoint] Evaluating consequential action gate...")

    approval = state.get("human_approved")
    total_cost = state.get("total_cost", 0.0)

    # All proposals exceeding $3,500 require consequential sign-off
    if approval is True:
        logs.append(f"  ↳ Human Decision: APPROVED by Web3Geeks Partner. Dispatching legally binding SOW for ${total_cost:,.2f}.")
        return {
            "human_approval_required": True,
            "human_approved": True,
            "contract_dispatched": True,
            "execution_logs": logs,
        }
    elif approval is False:
        logs.append("  ↳ Human Decision: REJECTED by Web3Geeks Partner. Contract generation aborted.")
        return {
            "human_approval_required": True,
            "human_approved": False,
            "contract_dispatched": False,
            "execution_logs": logs,
        }
    else:
        logs.append("  ↳ Action Paused: Proposal held in queue pending Human Partner Sign-off.")
        return {
            "human_approval_required": True,
            "human_approved": None,
            "contract_dispatched": False,
            "execution_logs": logs,
        }


# =============================================================================
# NODE 7: GRACEFUL FAILURE HANDLER
# =============================================================================
def failure_handler_node(state: OnboardingState) -> Dict[str, Any]:
    """Handles invalid input, low budget, or security flags gracefully."""
    logs = list(state.get("execution_logs", []))
    status = state.get("validation_status", "unknown_error")
    err = state.get("error_message", "An unexpected validation failure occurred.")

    logs.append(f"[Graceful Exit] Pipeline terminated cleanly with status: {status}")

    fallback_response = (
        f"# Web3Geeks Inbound Inquiry Notice\n\n"
        f"**Status:** {status.upper()}\n\n"
        f"**Details:** {err}\n\n"
        f"Please review our standard service tiers at `web3geeks.io/services` or contact `partnerships@web3geeks.io` "
        f"for custom enterprise scoping."
    )

    return {
        "proposal_draft": fallback_response,
        "is_valid": False,
        "total_cost": 0.0,
        "contract_dispatched": False,
        "execution_logs": logs,
    }


def validation_router(state: OnboardingState) -> str:
    """Routes to scoping if input passed, or to failure handler if invalid."""
    return "scope_services" if state.get("is_valid", False) else "failure_handler"


# =============================================================================
# WORKFLOW GRAPH COMPILATION
# =============================================================================
def build_onboarding_graph():
    """Builds and compiles the complete LangGraph state machine."""
    workflow = StateGraph(OnboardingState)

    # Register nodes
    workflow.add_node("validate_inquiry", validate_inquiry_node)
    workflow.add_node("scope_services", scope_services_node)
    workflow.add_node("calculate_budget", calculate_budget_node)
    workflow.add_node("generate_proposal", generate_proposal_node)
    workflow.add_node("critique_proposal", critique_proposal_node)
    workflow.add_node("human_checkpoint", human_checkpoint_node)
    workflow.add_node("failure_handler", failure_handler_node)

    # Define edges
    workflow.set_entry_point("validate_inquiry")

    workflow.add_conditional_edges(
        "validate_inquiry",
        validation_router,
        {
            "scope_services": "scope_services",
            "failure_handler": "failure_handler",
        },
    )

    workflow.add_edge("scope_services", "calculate_budget")
    workflow.add_edge("calculate_budget", "generate_proposal")
    workflow.add_edge("generate_proposal", "critique_proposal")

    workflow.add_conditional_edges(
        "critique_proposal",
        critique_router,
        {
            "generate_proposal": "generate_proposal",
            "human_checkpoint": "human_checkpoint",
        },
    )

    workflow.add_edge("human_checkpoint", END)
    workflow.add_edge("failure_handler", END)

    return workflow.compile()


def run_onboarding_agent(
    inquiry: str,
    client_name: str = "Prospective Enterprise Client",
    budget_max: Optional[float] = None,
    sla_tier: str = "standard",
    human_approved: Optional[bool] = None,
) -> OnboardingState:
    """Executes the complete onboarding pipeline synchronously."""
    start_time = time.time()
    graph = build_onboarding_graph()

    initial_state: OnboardingState = {
        "client_inquiry": inquiry,
        "client_name": client_name,
        "budget_max": budget_max,
        "sla_tier": sla_tier,
        "is_valid": False,
        "validation_status": "pending",
        "error_message": None,
        "catalog_matches": [],
        "milestones": [],
        "total_cost": 0.0,
        "estimated_days": 0,
        "proposal_draft": "",
        "critique_score": 0.0,
        "critique_feedback": "",
        "revision_count": 0,
        "human_approval_required": False,
        "human_approved": human_approved,
        "contract_dispatched": False,
        "execution_logs": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
        "cost_usd": 0.0,
    }

    final_state = graph.invoke(initial_state)

    duration = round(time.time() - start_time, 2)
    p_tok = final_state.get("prompt_tokens", 0)
    c_tok = final_state.get("completion_tokens", 0)
    tot_tok = p_tok + c_tok
    cost = config.calculate_cost(p_tok, c_tok)

    final_state["latency_seconds"] = duration
    final_state["total_tokens"] = tot_tok
    final_state["cost_usd"] = round(cost, 6)

    return final_state

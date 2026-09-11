"""Production FastAPI REST API wrapper for Web3Geeks Client Onboarding Agent System.
Includes structured logging, request validation, metrics tracking, and HITL endpoints.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from workflow import run_onboarding_agent
from tools import ServiceCatalogSearchTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web3geeks_api")

app = FastAPI(
    title="Web3Geeks Client Onboarding Agent API",
    version="1.0.0",
    description="Autonomous enterprise client scoping, budgeting, and proposal generation API with HITL checkpoints.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory metrics store for monitoring
METRICS = {
    "total_requests": 0,
    "successful_proposals": 0,
    "rejected_adversarial": 0,
    "rejected_low_budget": 0,
    "total_tokens_consumed": 0,
    "total_cost_usd": 0.0,
    "total_latency_seconds": 0.0,
}


# =============================================================================
# REQUEST & RESPONSE MODELS
# =============================================================================
class OnboardingRequest(BaseModel):
    inquiry: str = Field(..., min_length=5, description="Client scope or technical project inquiry.")
    client_name: str = Field(default="Prospective Enterprise Client", description="Name of client organization.")
    budget_max: Optional[float] = Field(default=None, description="Client's target maximum budget in USD.")
    sla_tier: str = Field(default="standard", description="'standard' (5x8) or 'expedited' (24/7 rush).")
    auto_approve: bool = Field(default=False, description="Whether to simulate immediate partner approval.")


class ApprovalRequest(BaseModel):
    approved: bool = Field(..., description="True to approve contract dispatch; False to reject.")
    reviewer_name: str = Field(default="Managing Partner", description="Name of authorized human reviewer.")
    reviewer_notes: Optional[str] = Field(default=None, description="Optional partner review comments.")
    inquiry: str = Field(..., description="Original client inquiry.")
    client_name: str = Field(default="Prospective Enterprise Client")
    budget_max: Optional[float] = None
    sla_tier: str = "standard"


class MilestoneResponse(BaseModel):
    index: int
    name: str
    percentage: str
    amount: float
    timeline: str
    deliverable: str


class OnboardingResponse(BaseModel):
    status: str
    validation_status: str
    error_message: Optional[str]
    client_name: str
    total_cost: float
    estimated_days: int
    milestones: List[MilestoneResponse]
    proposal_draft: str
    critique_score: float
    human_approval_required: bool
    contract_dispatched: bool
    metrics: Dict[str, Any]
    execution_logs: List[str]


# =============================================================================
# MIDDLEWARE: REQUEST LOGGING & LATENCY MONITORING
# =============================================================================
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    METRICS["total_requests"] += 1

    try:
        response: Response = await call_next(request)
        latency = round(time.time() - start_time, 4)
        logger.info(
            f"Method: {request.method} | Path: {request.url.path} | Status: {response.status_code} | Latency: {latency}s"
        )
        return response
    except Exception as e:
        latency = round(time.time() - start_time, 4)
        logger.error(f"Method: {request.method} | Path: {request.url.path} | Error: {str(e)} | Latency: {latency}s")
        raise e


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/health", tags=["System"])
def health_check():
    """System health check and readiness status."""
    catalog = ServiceCatalogSearchTool()
    min_budget = catalog.get_minimum_budget()
    return {
        "status": "healthy",
        "service": "Web3Geeks Client Onboarding Agent",
        "version": "1.0.0",
        "minimum_project_budget_usd": min_budget,
        "active_catalog_services": 5,
    }


@app.get("/api/v1/metrics", tags=["Monitoring"])
def get_metrics():
    """Returns real-time operational metrics for production monitoring."""
    reqs = max(1, METRICS["total_requests"])
    avg_latency = round(METRICS["total_latency_seconds"] / reqs, 3)
    return {
        **METRICS,
        "average_latency_seconds": avg_latency,
        "average_cost_per_request_usd": round(METRICS["total_cost_usd"] / reqs, 6),
    }


@app.post("/api/v1/onboard", response_model=OnboardingResponse, tags=["Onboarding"])
def onboard_client(payload: OnboardingRequest):
    """Initiates an onboarding inquiry, executes scoping, budgeting, drafting & critique."""
    start_time = time.time()

    state = run_onboarding_agent(
        inquiry=payload.inquiry,
        client_name=payload.client_name,
        budget_max=payload.budget_max,
        sla_tier=payload.sla_tier,
        human_approved=True if payload.auto_approve else None,
    )

    duration = round(time.time() - start_time, 3)

    # Update monitoring counters
    val_status = state.get("validation_status", "unknown")
    if val_status == "passed":
        METRICS["successful_proposals"] += 1
    elif val_status == "rejected_adversarial":
        METRICS["rejected_adversarial"] += 1
    elif val_status == "failed_low_budget":
        METRICS["rejected_low_budget"] += 1

    METRICS["total_tokens_consumed"] += state.get("total_tokens", 0)
    METRICS["total_cost_usd"] += state.get("cost_usd", 0.0)
    METRICS["total_latency_seconds"] += duration

    status_str = "success" if state.get("is_valid") else "rejected"
    if state.get("is_valid") and not state.get("contract_dispatched"):
        status_str = "held_for_human_approval"

    return OnboardingResponse(
        status=status_str,
        validation_status=state.get("validation_status", "unknown"),
        error_message=state.get("error_message"),
        client_name=state.get("client_name", payload.client_name),
        total_cost=state.get("total_cost", 0.0),
        estimated_days=state.get("estimated_days", 0),
        milestones=[MilestoneResponse(**m) for m in state.get("milestones", [])],
        proposal_draft=state.get("proposal_draft", ""),
        critique_score=state.get("critique_score", 0.0),
        human_approval_required=state.get("human_approval_required", True),
        contract_dispatched=state.get("contract_dispatched", False),
        metrics={
            "latency_seconds": duration,
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
            "total_tokens": state.get("total_tokens", 0),
            "cost_usd": state.get("cost_usd", 0.0),
        },
        execution_logs=state.get("execution_logs", []),
    )


@app.post("/api/v1/approve", tags=["Human-in-the-Loop"])
def approve_contract(payload: ApprovalRequest):
    """Submits partner decision to approve or reject SOW dispatch."""
    logger.info(f"HITL Decision by {payload.reviewer_name}: Approved={payload.approved}")

    state = run_onboarding_agent(
        inquiry=payload.inquiry,
        client_name=payload.client_name,
        budget_max=payload.budget_max,
        sla_tier=payload.sla_tier,
        human_approved=payload.approved,
    )

    if not state.get("is_valid"):
        raise HTTPException(status_code=400, detail=state.get("error_message", "Invalid inquiry state."))

    action_msg = (
        f"Contract signed and dispatched by {payload.reviewer_name} for ${state.get('total_cost', 0):,.2f}."
        if payload.approved
        else f"Proposal rejected by {payload.reviewer_name}. Reason: {payload.reviewer_notes or 'Terms unacceptable.'}"
    )

    return {
        "status": "completed",
        "decision": "approved" if payload.approved else "rejected",
        "reviewer": payload.reviewer_name,
        "contract_dispatched": state.get("contract_dispatched", False),
        "total_cost_usd": state.get("total_cost", 0.0),
        "message": action_msg,
    }

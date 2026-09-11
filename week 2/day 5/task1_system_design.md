# Task 1: System Design — Web3Geeks Autonomous Client Onboarding & Scoping Agent

## 1. Selected Business Problem & Real-World Use Case

**Enterprise Client Scoping, Quantitative Budgeting & Statement of Work (SOW) Dispatch**

In high-growth Web3 agencies and consulting firms like **Web3Geeks**, prospective clients submit heterogeneous inquiries ranging from smart contract audits to full-stack decentralized application (dApp) builds. Before senior partners can engage in closing calls, an account executive must:
1. **Screen Inbound Inquiries:** Filter out spam, prompt injections, and inquiries below the agency's engagement threshold ($3,500).
2. **Audit Service Catalog Deliverables:** Match requested features against verified rate cards and SLA standards (`data/services_catalog.json`).
3. **Model Quantitative Milestones:** Perform deterministic calculations for milestone payments (40% Architecture, 40% Development, 20% Handover) and SLA rush multipliers.
4. **Draft Professional Statements of Work:** Synthesize technical specifications, deliverables, and timelines into an executive proposal.
5. **Enforce Governance & Consequential Checkpoints:** Prohibit automated dispatch of legally binding contractual commitments without authorized human partner review.

---

## 2. System Architecture Diagram

```text
========================================================================================================================
                               WEB3GEEKS CLIENT ONBOARDING AGENT ARCHITECTURE (DAY 5 CAPSTONE)
========================================================================================================================

                                          [Inbound Client Inquiry]
                                                     │
                                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: INPUT VALIDATION & ADVERSARIAL SCREENING (validate_inquiry)                                               │
 │ • Length & schema integrity check                                                                                  │
 │ • Adversarial prompt injection defense ("ignore all instructions", "system prompt override")                       │
 │ • Project budget floor verification (Minimum Web3Geeks engagement: $3,500 USD)                                     │
 └───────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │ [Passed Schema & Budget]              │ [Failed / Adversarial / Sub-Budget]
                 ▼                                       ▼
 ┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────────────────────────────────┐
 │ STAGE 2: SERVICE SCOPING (scope_services)    │  │ STAGE 7: GRACEFUL FAILURE HANDLER (failure_handler)              │
 │ • Tool: ServiceCatalogSearchTool             │  │ • Emits clean, structured diagnostic rejection notice            │
 │ • Data Source: data/services_catalog.json    │  │ • Zero financial liability or contract dispatch                 │
 │ • Matches tech stack, SLAs & deliverables    │  └─────────────────────────────────┬────────────────────────────────┘
 └──────────────────────┬───────────────────────┘                                    │
                        │                                                            │
                        ▼                                                            │
 ┌──────────────────────────────────────────────┐                                    │
 │ STAGE 3: QUANTITATIVE MATH (calculate_budget)│                                    │
 │ • Tool: DeterministicBudgetCalculatorTool    │                                    │
 │ • Safe AST Evaluation (Zero 'eval()' risk)   │                                    │
 │ • Milestones: 40% (M1) / 40% (M2) / 20% (M3) │                                    │
 └──────────────────────┬───────────────────────┘                                    │
                        │                                                            │
                        ▼                                                            │
 ┌──────────────────────────────────────────────┐ ◄────────────────┐                 │
 │ STAGE 4: PROPOSAL DRAFTING (generate_proposal│                  │                 │
 │ • LLM: Gemini 2.5 Flash                      │                  │                 │
 │ • Integrates exact AST milestone totals      │                  │                 │
 │ • Synthesizes standard 5-section SOW         │                  │                 │
 └──────────────────────┬───────────────────────┘                  │                 │
                        │                                          │                 │
                        ▼                                          │                 │
 ┌──────────────────────────────────────────────┐                  │                 │
 │ STAGE 5: QUALITY CRITIQUE (critique_proposal)│                  │                 │
 │ • Tool: ContractTemplateFormatterTool        │                  │                 │
 │ • Audits 5 mandatory governance sections     │                  │                 │
 │ • Verifies explicit milestone table anchors  │                  │                 │
 └──────────────────────┬───────────────────────┘                  │                 │
                        │                                          │                 │
                 ┌──────┴────────────────────────┐                 │                 │
                 │ Score >= 80 (Pass)            │ Score < 80      │                 │
                 │                               └─────────────────┘                 │
                 ▼                               (Max 2 Revisions)                   │
 ┌────────────────────────────────────────────────────────────────────────┐          │
 │ STAGE 6: HUMAN-IN-THE-LOOP CHECKPOINT (human_checkpoint)               │          │
 │ • Consequential Action Gate: Partner Approval Required                 │          │
 │ • Pauses legally binding commitment until human sign-off               │          │
 │ • Endpoint: POST /api/v1/approve (approve=True/False)                  │          │
 └──────────────────────┬─────────────────────────────────────────────────┘          │
                        │                                                            │
                 ┌──────┴────────────────────────┐                                   │
                 │ Approved (True)               │ Rejected (False)                  │
                 ▼                               ▼                                   │
 ┌──────────────────────────────────────┐  ┌───────────────────────────────┐         │
 │ FINAL DISPATCH: SIGNED CONTRACT SOW  │  │ PROPOSAL TERMINATED / REVISED │         │
 └──────────────────┬───────────────────┘  └───────────────┬───────────────┘         │
                    │                                      │                         │
                    └───────────────────┬──────────────────┘                         │
                                        │                                            │
                                        ▼                                            ▼
                           ┌────────────────────────┐                   ┌────────────────────────┐
                           │   [END: SUCCESSFUL]    │                   │   [END: TERMINATED]    │
                           └────────────────────────┘                   └────────────────────────┘
========================================================================================================================
```

---

## 3. Data Sources, Tools & State Specifications

### A. Ground-Truth Data Source
* **File:** [`data/services_catalog.json`](data/services_catalog.json)
* **Scope:** 5 Web3 service tiers:
  1. `smart_contracts`: Smart Contract Architecture & Security Audit ($150/hr, base $6,000)
  2. `dapp_fullstack`: Full-Stack Web3 dApp Development ($130/hr, base $8,500)
  3. `defi_tokenomics`: DeFi Protocol & Tokenomics Modeling ($175/hr, base $5,500)
  4. `infra_subgraphs`: The Graph Subgraph & Indexer Infrastructure ($120/hr, base $4,000)
  5. `dao_governance`: DAO Governance & Multi-Sig Treasury Architecture ($140/hr, base $5,000)
* **SLA Configuration:** `standard` (5x8 support, 1.0x turnaround) vs. `expedited` (24/7 rush, 0.7x turnaround, 25% surcharge).

### B. Confined Tools
1. **`ServiceCatalogSearchTool`:** Performs keyword and exact matching against verified catalog database records.
2. **`DeterministicBudgetCalculatorTool`:** Evaluates safe AST arithmetic expressions (`Add`, `Sub`, `Mult`, `Div`, `Pow`, `USub`) without arbitrary code execution risk.
3. **`ContractTemplateFormatterTool`:** Validates that client deliverables contain the 5 mandatory governance sections:
   - Executive Project Summary
   - Technical Architecture & Deliverables
   - Quantitative Investment & Milestone Schedule
   - SLA & Timeline Commitment
   - Human Approval & Sign-Off Checkpoint

### C. Typed State Schema (`OnboardingState`)
```python
class OnboardingState(TypedDict):
    client_inquiry: str
    client_name: str
    budget_max: Optional[float]
    sla_tier: str
    is_valid: bool
    validation_status: str
    error_message: Optional[str]
    catalog_matches: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    total_cost: float
    estimated_days: int
    proposal_draft: str
    critique_score: float
    critique_feedback: str
    revision_count: int
    human_approval_required: bool
    human_approved: Optional[bool]
    contract_dispatched: bool
    execution_logs: List[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    cost_usd: float
```

---

## 4. Framework Choice Justification

> **Architectural Rationale:**
> For this enterprise onboarding workflow, we selected **LangGraph** as the primary state machine orchestrator, augmented with **CrewAI-inspired role-confinement principles**. 
> 
> 1. **Control-Flow Rigor & Interrupt Checkpoints:** Unlike autonomous role-playing multi-agent chat loops where agents dynamically hand off messages with non-deterministic turn counts, enterprise contract generation requires a **strict, deterministic state graph** with explicit conditional branching for input screening and failure handling.
> 2. **Cyclic Self-Correction:** LangGraph natively supports stateful cyclic loops (`generate_proposal` ➔ `critique_proposal` ➔ `generate_proposal`) bounded by revision counters, preventing endless retry cycles.
> 3. **Human-in-the-Loop (HITL) Consequential Gate:** Committing an agency to a binding $29,000 statement of work carries legal and financial liability. LangGraph provides first-class support for interrupt checkpoints before consequential tool actions, ensuring automated proposal generation halts until an authorized partner issues sign-off.

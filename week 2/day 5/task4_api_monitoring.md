# Task 4: Wrap as an API & Production Monitoring Runbook

## 1. FastAPI REST API Architecture

The agent system is wrapped behind a high-performance **FastAPI** service ([`api.py`](api.py)), providing standardized REST endpoints for asynchronous frontend clients and human reviewer dashboards:

### Key Endpoints

| Method | Path | Purpose | Input Payload | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/onboard` | Initiates client onboarding, runs scoping, AST budgeting, drafting & critique | `inquiry`, `client_name`, `budget_max`, `sla_tier`, `auto_approve` | `OnboardingResponse` (status, total_cost, milestones, draft, critique_score, metrics) |
| `POST` | `/api/v1/approve` | Human-in-the-Loop Partner Approval Gate | `approved`, `reviewer_name`, `reviewer_notes`, `inquiry`, `client_name` | Approval confirmation, final contract dispatch status, signed SOW |
| `GET` | `/api/v1/metrics` | Real-time observability counters | None | In-memory telemetry (requests, approvals, rejections, avg latency, total cost) |
| `GET` | `/health` | Kubernetes / Cloud readiness check | None | Service status, active catalog count, minimum budget floor |

---

## 2. Structured Telemetry & Middleware Logging

Every HTTP transaction passes through our asynchronous logging middleware, recording:
* **Inbound Metadata:** HTTP method, client IP, target endpoint, and payload size.
* **Execution Telemetry:** Wall-clock latency (seconds), token consumption (prompt vs. completion), and estimated USD cost.
* **Audit Trail:** Step-by-step pipeline execution logs detailing catalog matches and AST evaluations.
* **Error Tracking:** Structured JSON traceback logging for unhandled exceptions.

```text
2026-09-11 19:42:52,854 [INFO] web3geeks_api: Method: POST | Path: /api/v1/onboard | Status: 200 | Latency: 2.2171s
2026-09-11 19:42:52,861 [INFO] web3geeks_api: HITL Decision by Alex Vance (Managing Partner): Approved=True
2026-09-11 19:42:53,535 [INFO] web3geeks_api: Method: POST | Path: /api/v1/approve | Status: 200 | Latency: 0.6749s
```

---

## 3. Production Monitoring Checklist & SRE Runbook

### A. What to Track in Production

| Dimension | Telemetry Metric | Measurement Source | Target Baseline | Alert Condition / Trigger |
| :--- | :--- | :--- | :--- | :--- |
| **System Reliability** | **HTTP 5xx Error Rate** | FastAPI Middleware | `< 0.5%` | `> 2.0%` of requests failing over a 5-minute rolling window. |
| **Operational Latency** | **P95 Request Latency** | Timing Middleware | `< 3.0 seconds` | `P95 > 5.0 seconds` sustained for more than 10 minutes. |
| **Financial Cost** | **Cost Drift per Request** | Token Telemetry Calculator | `~$0.00020 USD` | Average cost exceeds `$0.00100 USD` (5x baseline), indicating prompt inflation or cyclic loops. |
| **Security Health** | **Adversarial Injections** | `validate_inquiry` Filter | `< 1.0%` | `> 10 injection attempts` within 1 hour; automatically triggers Cloudflare WAF IP throttle. |
| **Output Quality** | **Critique Loop Score** | `critique_proposal` Node | `> 90 / 100` | Average critique score falls `< 80 / 100`, indicating template decay or model drift. |
| **Human Workflow** | **HITL Queue Backlog** | In-Memory Approvals Queue | `< 5 pending SOWs` | `> 15 proposals pending human approval` for more than 4 hours. |

### B. Production Alert Thresholds & Severity Levels

* 🔴 **P1 (Critical - Immediate PagerDuty Alert):**
  - Upstream LLM provider completely unreachable (100% 500s across 3 consecutive minutes).
  - P95 latency exceeds 15 seconds.
* 🟡 **P2 (Warning - Slack `#agency-ops-alerts`):**
  - Average cost per request increases by > 200% over a 24-hour window.
  - SOW critique scores dropping below 85 points.
  - HITL review queue exceeds 10 proposals.
* 🟢 **P3 (Informational - Daily Digest):**
  - Daily aggregated token consumption, total onboarded pipeline value ($ USD), and catalog hit rate.

### C. Re-Evaluation & Model Maintenance Cadence

1. **Bi-Weekly Golden Dataset Benchmark:**
   - Execute the 8-point empirical test suite (`evaluation.py`) against updated catalog pricing every other Monday to guard against prompt regression.
2. **Monthly Adversarial Red-Teaming:**
   - Inject 15 novel jailbreaks, unicode homoglyphs, and prompt injection payloads to update the defense filter in `validate_inquiry_node`.
3. **Quarterly Service Catalog & Rate Card Refresh:**
   - Re-evaluate base prices and hourly rates in `data/services_catalog.json` with agency managing partners to match current market conditions.

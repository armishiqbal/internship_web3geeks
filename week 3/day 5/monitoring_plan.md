# AFL Intelligence Assistant: Production Monitoring & Maintenance Plan

## Operational Scope
This document specifies the production monitoring architecture, telemetry metrics, automated alert thresholds, and the weekly model retraining loop for the AFL Intelligence Assistant deployed across Web3Geeks and client properties.

---

## 1. Production Monitoring Checklist

| Telemetry Dimension | Metric Tracked | Sampling Cadence | Target Threshold | Critical Alert Trigger | Remediation Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inference Latency** | Median (p50) Latency | Real-time per request | < 100 ms (Cached/Factual) | p50 > 300 ms | Scale worker pool; inspect model cache contention. |
| **Inference Latency** | Tail (p95 / p99) Latency | Real-time rolling 5-min | < 1,500 ms (ML Prediction) | p95 > 2,500 ms | Restart lagging workers; verify Parquet I/O thread pool. |
| **Pipeline Reliability** | Tool Error Rate | Hourly aggregate | < 0.5% | Error rate > 2.0% | Roll back to static cache fallback; inspect dataset schema. |
| **Security & Guardrails** | Off-Topic Leak Rate | Daily audit (100 samples) | 0.0% unhandled leaks | Any verified leak | Update router keyword scoring and regex pattern lists. |
| **Security & Guardrails** | Prompt Injection Rate | Real-time security filter | 100% interception | Unhandled override | Block malicious IP / session ID; deploy pattern patch. |
| **Entity Resolution** | Unresolved Club Rate | Daily aggregate | < 3.0% of prediction queries | Unresolved rate > 5.0% | Expand club alias dictionary with new slang/nicknames. |
| **Predictive Accuracy** | Match Winner Accuracy | Weekly post-round audit | >= 65.0% (Current: 69.0%) | Accuracy < 60.0% (3 rounds) | Trigger automated GBDT retraining pipeline. |
| **Probability Calibration** | Brier Calibration Score | Weekly post-round audit | <= 0.205 (Current: 0.199) | Brier Score > 0.220 | Recalibrate Isotonic regression on rolling season holdout. |

---

## 2. Telemetry Logging Standard

Every user inquiry through `/api/chat` emits a structured JSON log line ingested by the central observability platform (e.g., Datadog, Prometheus, CloudWatch):

```json
{
  "timestamp": "2026-09-18T14:43:43Z",
  "event": "chat_request",
  "conversation_id": "user_session_4821",
  "query": "Will the Pies beat the Cats this week?",
  "intent": "prediction",
  "confidence": 0.99,
  "validation_status": "valid",
  "tool_called": "predict_match_winner",
  "has_prediction": true,
  "latency_ms": 28.4,
  "tokens": 214
}
```

---

## 3. Weekly Model Retraining & Ingestion Refresh Loop

AFL rounds conclude each Sunday evening. The maintenance cycle executes autonomously every Monday at 02:00 UTC:

| Pipeline Phase | Trigger & Execution Window | Core Operational Task | Output & Verification Standard |
| :--- | :--- | :--- | :--- |
| **Phase 1: Ingestion** | Monday 02:00 UTC | Ingest completed weekend round match and player statistics. | Assert complete 9-match coverage without missing scores. |
| **Phase 2: Features** | Monday 02:15 UTC | Append to Parquet stores; recompute rolling 5-game form and Elo ratings. | Zero post-match target leakage verified. |
| **Phase 3: Audit** | Monday 02:30 UTC | Score model forecasts vs actual round results; compute Brier Score. | Emit weekly calibration metrics to Grafana dashboard. |
| **Phase 4: Retraining** | Monday 02:45 UTC | If rolling 4-week Brier score > 0.215, retrain GBDT model. | Holdout validation Accuracy >= 67.0% required. |
| **Phase 5: Deploy** | Monday 03:00 UTC | Atomically swap `.joblib` model artifact; refresh FastAPI cache. | Zero-downtime hot reload verified via health check. |

### Operational Rules:
1. **Zero-Leakage Retraining**: All feature transforms strictly enforce historical causality; no post-match statistics are ever included in pre-match inference tables.
2. **Atomic Model Swap**: New `.joblib` model artifacts are validated via automated regression smoke tests before replacing active production artifacts.
3. **Rollback Trigger**: If a newly deployed model exhibits a Brier score degradation of >0.025 over its first active weekend, the system automatically reverts to the previous week's model checkpoint.

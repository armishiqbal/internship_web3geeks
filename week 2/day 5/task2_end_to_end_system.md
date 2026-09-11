# Task 2: Build the End-to-End System — Component Reuse & Failure Handling

## 1. Architectural Component Reuse from Days 1–4

The Day 5 Capstone directly consolidates and elevates architectural patterns developed throughout the week:

| Prior Day Pattern | Reused Component in Day 5 Capstone | Architectural Benefit |
| :--- | :--- | :--- |
| **Day 1: Prompt Engineering & Schemas** | Typed Pydantic models for API request/response payloads (`OnboardingRequest`, `OnboardingResponse`) | Enforces type safety, JSON schema validation, and defensive boundary sanitization. |
| **Day 2: Tool Calling & AST Calculation** | `DeterministicBudgetCalculatorTool` with safe Python AST parsing | Completely eliminates arithmetic hallucination and arbitrary code execution vulnerabilities (`eval()`). |
| **Day 3: LangGraph Stateful Graphs & Cycles** | Cyclic critique loop (`generate_proposal` ➔ `critique_proposal` ➔ `generate_proposal`) with revision counters | Enables automated self-correction before human review, guaranteeing 100% compliance with proposal guidelines. |
| **Day 3: Human-in-the-Loop (HITL)** | Consequential action gate (`human_checkpoint_node`) halting legally binding SOW contract dispatch | Eliminates unauthorized agency liability; requires explicit human partner sign-off. |
| **Day 4: Role-Based Least-Privilege Tools** | Confined tools (`ServiceCatalogSearchTool` for scoping, `DeterministicBudgetCalculatorTool` for finance) | Prevents tool confusion, stops persona drift, and isolates mathematical calculations from text generation. |

---

## 2. External Data Source & Consequential Action Checkpoint

### A. External Data Source: Service Catalog Database
* **Database File:** [`data/services_catalog.json`](data/services_catalog.json)
* **Access Layer:** `ServiceCatalogSearchTool` performs cached file reads and implements fuzzy search across service titles, categories, deliverables, and tech stack tags.
* **Integrity Guarantee:** Inquiries are grounded against authoritative rate cards ($120/hr – $175/hr) rather than model-invented figures.

### B. Consequential Action Checkpoint: Partner SOW Dispatch Gate
* **Risk Profile:** In enterprise consulting, delivering a binding Statement of Work with fixed pricing and timelines commits legal liability, engineer capacity, and billing obligations.
* **Checkpoint Logic (`human_checkpoint_node`):**
  ```python
  if approval is True:
      # Dispatch binding SOW
      return {"human_approved": True, "contract_dispatched": True}
  elif approval is False:
      # Abort contract generation
      return {"human_approved": False, "contract_dispatched": False}
  else:
      # Pause and hold in queue
      return {"human_approved": None, "contract_dispatched": False}
  ```
* **Production API Surface:** Managed through `POST /api/v1/approve`, where authorized partners submit their identity (`reviewer_name`) and decision (`approved=True/False`).

---

## 3. Failure Scenarios & Graceful Handling

The system implements defensive validation and graceful routing across three distinct failure modes:

### Failure Scenario 1: Empty or Malformed Inquiry
* **Failure Trigger:** Inbound request contains an empty string or fewer than 10 characters (e.g., `"build"`).
* **Defensive Mechanism:** `validate_inquiry_node` intercepts payload before LLM or tool execution.
* **Graceful Exit:** Sets `validation_status = 'failed_empty_input'`, bypassing LLM calls, avoiding token waste, and returning a structured diagnostic notice explaining the minimum 10-character scope requirement.

### Failure Scenario 2: Adversarial Prompt Injection
* **Failure Trigger:** Malicious probe injecting jailbreak keywords (`"IGNORE ALL PREVIOUS INSTRUCTIONS"`, `"drop table"`, `"system prompt override"`).
* **Defensive Mechanism:** Regex and substring screening in `validate_inquiry_node`.
* **Graceful Exit:** Immediately flags `validation_status = 'rejected_adversarial'`, halts downstream tool access, logs a security alert, and returns a sanitized refusal notice without leaking system prompts.

### Failure Scenario 3: Sub-Minimum Budget Rejection
* **Failure Trigger:** Client proposes an unrealistic budget below Web3Geeks' engagement floor (e.g., `$500` for a cross-chain DEX).
* **Defensive Mechanism:** Compares `budget_max` against `_catalog_tool.get_minimum_budget()` ($3,500).
* **Graceful Exit:** Sets `validation_status = 'failed_low_budget'`, politely directing the prospect to public self-serve documentation rather than wasting engineering scoping resources.

### Failure Scenario 4: LLM Quota / Timeout Resilience
* **Failure Trigger:** Google API rate limits (`429 RESOURCE_EXHAUSTED`) or network timeouts.
* **Defensive Mechanism:** `try/except` block in `generate_proposal_node` falls back to an AST-verified deterministic template.
* **Graceful Exit:** Guarantees zero unhandled 500 exceptions, returning an executive-ready proposal with exact milestone arithmetic even during upstream provider outages.

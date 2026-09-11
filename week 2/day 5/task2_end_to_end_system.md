# Task 2: Build the End-to-End System — Component Reuse & Failure Handling

## 1. Component Reuse from Days 1–4

The Day 5 production workflow reuses and integrates the strongest components developed during Days 1–4 rather than rebuilding them from scratch.

| Source | Reused Component                    | Day 5 Integration                                                                              |
| ------ | ----------------------------------- | ---------------------------------------------------------------------------------------------- |
| Day 1  | Typed Pydantic API models           | `OnboardingRequest` and `OnboardingResponse` provide validated, structured API input/output    |
| Day 2  | `DeterministicBudgetCalculatorTool` | Performs safe AST-based financial calculations instead of relying on LLM arithmetic            |
| Day 3  | Cyclic critique and revision loop   | `critique_proposal` evaluates the generated SOW and triggers controlled revision when required |
| Day 3  | Human-in-the-loop workflow          | `human_checkpoint_node` blocks consequential contract dispatch until explicit approval         |
| Day 4  | Role-based least privilege          | Agents/nodes are restricted to the tools and responsibilities required for their stage         |

This reuse provides continuity across the internship tasks while combining deterministic computation, LLM-based generation, validation, critique, and human governance into a single production-oriented workflow.

---

## 2. External Data Source and Tool Integration

The workflow uses an externalized service catalog rather than embedding service prices and SLA information directly into the LLM prompt.

### Data Source

**File:** `data/services_catalog.json`

The catalog contains the authoritative service information used for client scoping and budget calculation, including:

- Service IDs
- Service names
- Hourly rates
- Base engagement costs
- SLA information
- Delivery/turnaround information
- Rush/expedited pricing rules

The catalog is versioned as **2026.1** and uses **USD** as its currency.

### Tool

**`ServiceCatalogSearchTool`**

The tool retrieves relevant services and their associated rate-card information from the catalog.

The workflow therefore follows:

```text
Client Inquiry
      ↓
ServiceCatalogSearchTool
      ↓
Authoritative Service Data
      ↓
Scope Services
      ↓
Deterministic Budget Calculator
      ↓
Proposal Generation
```

Using a separate data source reduces the risk of the LLM inventing prices, services, or SLA commitments.

---

## 3. Consequential Action and Human-in-the-Loop Approval

The system treats contract/SOW dispatch as a consequential action and does not allow the workflow to dispatch the final contract automatically.

The `human_checkpoint_node` explicitly requires an approval decision before dispatch:

```python
if approval is True:
    return {
        "human_approved": True,
        "contract_dispatched": True
    }
elif approval is False:
    return {
        "human_approved": False,
        "contract_dispatched": False
    }
else:
    return {
        "human_approved": None,
        "contract_dispatched": False
    }
```

The API exposes the approval operation through:

```text
POST /api/v1/approve
```

The approval request records the reviewer identity and explicit approval decision.

### Approval Flow

```text
Generated SOW
     ↓
Quality Critique
     ↓
Human Review
     ↓
 ┌───────────────┐
 │ Approval?     │
 └───────┬───────┘
         │
    ┌────┴────┐
    │         │
  YES        NO
    │         │
    ↓         ↓
Dispatch   No Dispatch
```

If approval is not explicitly granted, `contract_dispatched` remains `False`. This creates a fail-closed boundary around the final external action.

---

## 4. Input Validation and Defensive Screening

All client inquiries pass through a validation layer before expensive LLM or tool operations are performed.

The system validates:

- Required input fields
- Inquiry length
- Basic input structure
- Minimum inquiry content
- Adversarial or prompt-injection patterns
- Business-rule constraints

Malformed or unsafe requests are rejected before reaching downstream proposal generation.

This prevents invalid requests from unnecessarily consuming model or tool resources and reduces the attack surface of the workflow.

---

# 5. Failure Handling

The system implements multiple failure-handling paths so that predictable failures do not silently produce incorrect SOWs or unauthorized contract dispatches.

## Failure Scenario 1: Empty or Malformed Inquiry

### Trigger

The client submits an empty inquiry, malformed request, or an inquiry shorter than the minimum required length of 10 characters.

### Detection

The `validate_inquiry` node performs input validation before downstream execution.

### Handling

The workflow immediately stops and returns a structured diagnostic response instead of calling the LLM or financial tools.

Example outcome:

```text
validation_status = "invalid_input"
workflow_status = "rejected"
downstream_execution = "blocked"
```

### Safety

No proposal is generated and no contract can be dispatched from an invalid inquiry.

---

## Failure Scenario 2: Adversarial Prompt Injection

### Trigger

The client inquiry contains known prompt-injection or instruction-override patterns attempting to manipulate the agent into ignoring system rules or performing unauthorized actions.

### Detection

The validation layer checks suspicious patterns using defensive matching and screening rules before the request enters the main workflow.

### Handling

The request is marked as adversarial and downstream processing is halted.

Example outcome:

```text
validation_status = "rejected_adversarial"
workflow_status = "blocked"
contract_dispatched = False
```

A sanitized refusal is returned to the requester and the security event is recorded in the workflow audit log.

### Safety

The system does not allow the untrusted inquiry to override workflow policies, financial controls, or the human approval requirement.

---

## Failure Scenario 3: Sub-Minimum Budget Inquiry

### Trigger

The calculated engagement falls below the configured minimum engagement threshold of **$3,500**.

### Detection

After service selection and deterministic budget calculation, the resulting total is compared against the minimum engagement requirement.

### Handling

The inquiry is rejected rather than allowing the system to produce a commercially invalid SOW.

Example:

```text
minimum_engagement = $3,500
calculated_budget < $3,500
→ workflow rejected
→ contract_dispatched = False
```

### Safety

The LLM cannot override the minimum-budget business rule because the financial decision is enforced by deterministic workflow logic.

---

## Failure Scenario 4: LLM Quota, Timeout, or Provider Error

### Trigger

The LLM provider returns an API failure such as a rate-limit response (`429`), timeout, network failure, or temporary provider error.

### Detection

The proposal-generation stage catches provider exceptions instead of allowing the error to propagate as an unhandled application failure.

### Handling

The system falls back to a deterministic proposal template populated with the already validated service, pricing, SLA, and milestone data.

The fallback avoids generating unsupported financial information.

Example flow:

```text
LLM Proposal Generation
          ↓
     Provider Error
          ↓
   Exception Handling
          ↓
Deterministic Template
          ↓
     Quality Critique
          ↓
     Human Approval
```

### Safety

The tested provider-failure path prevents the provider error from producing an unhandled `500` response and preserves the financial controls established by the deterministic calculation stage.

---

## Failure Scenario 5: External Tool Timeout or Catalog Error

### Trigger

The `ServiceCatalogSearchTool` encounters a file-read error, unavailable catalog source, malformed catalog data, or an external/tool timeout.

### Detection

The tool execution is wrapped with error handling so that catalog failures are detected before financial calculations or contract generation continue.

### Handling

If authoritative service information cannot be retrieved, the workflow fails closed rather than allowing the LLM to invent service prices or SLA information.

Example outcome:

```text
validation_status = "tool_error"
workflow_status = "blocked"
budget_calculated = False
contract_dispatched = False
```

The failure is recorded in the audit log and a structured retry/error response is returned.

### Safety

No financial calculation or SOW dispatch occurs using incomplete or unavailable catalog data.

This prevents the system from substituting LLM-generated guesses for authoritative commercial information.

---

# 6. Failure-Handling Summary

| Failure               | Detection                   | Recovery / Exit                      | Contract Dispatch |
| --------------------- | --------------------------- | ------------------------------------ | ----------------- |
| Empty/malformed input | Pydantic + validation node  | Reject with structured diagnostic    | No                |
| Prompt injection      | Defensive screening         | Block request + security log         | No                |
| Sub-minimum budget    | Deterministic business rule | Reject inquiry                       | No                |
| LLM timeout/API error | Exception handling          | Deterministic proposal fallback      | Only after HITL   |
| Catalog/tool error    | Tool error handling         | Fail closed + structured retry/error | No                |

The failure strategy follows a **fail-closed principle** for consequential operations: when authoritative data, validation, or required processing is unavailable, the system does not guess and does not dispatch a contract.

---

# 7. End-to-End Production Flow

The final Day 5 workflow combines validation, external data retrieval, deterministic financial calculation, LLM generation, critique, failure handling, and human approval:

```text
Client Inquiry
      ↓
validate_inquiry
      │
      ├── Invalid ──────────────→ Failure Handler → END
      │
      ├── Adversarial ──────────→ Failure Handler → END
      │
      ↓
scope_services
      │
      ├── Catalog Error ────────→ Failure Handler → END
      │
      ↓
calculate_budget
      │
      ├── Below $3,500 ─────────→ Failure Handler → END
      │
      ↓
generate_proposal
      │
      ├── LLM Error ────────────→ Deterministic Template
      │
      ↓
critique_proposal
      │
      ├── Revision Required ────→ generate_proposal
      │
      ↓
human_checkpoint
      │
      ├── Rejected ─────────────→ No Dispatch → END
      │
      └── Approved
              ↓
      Contract / SOW Dispatch
              ↓
             END
```

---

# 8. Consequential-Action Safety Boundary

The architecture intentionally separates **proposal generation** from **contract dispatch**.

The LLM can assist with:

- Service interpretation
- Proposal drafting
- Milestone wording
- SOW formatting
- Critique and revision

However, the LLM does not independently control:

- Authoritative service pricing
- Deterministic budget calculations
- Minimum engagement enforcement
- Final approval
- Contract dispatch

These controls remain in deterministic application logic and the human approval checkpoint.

This separation reduces the risk of hallucinated financial values, unauthorized actions, and prompt-injection-driven contract dispatch.

---

# 9. Task 2 Completion Summary

The Day 5 system successfully integrates components developed during Days 1–4 into a single end-to-end workflow.

The implementation provides:

- Reuse of validated components from previous days
- An external service catalog data source
- A dedicated catalog search tool
- Deterministic AST-based financial calculation
- Input validation and adversarial screening
- A cyclic proposal critique/revision mechanism
- Human approval before consequential contract dispatch
- Graceful handling of malformed input
- Prompt-injection rejection
- Minimum-budget enforcement
- LLM provider failure fallback
- External tool/catalog failure handling
- Fail-closed behavior for consequential operations
- Structured API responses and workflow audit logging

The resulting architecture combines **LLM flexibility with deterministic business controls and human governance**, making the workflow substantially safer and more suitable for production-oriented deployment than an unconstrained autonomous agent.

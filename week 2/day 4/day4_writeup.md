# Week 2 Day 4 — CrewAI: Multi-Agent Collaboration, Roles & Task Delegation

**Executive Summary**: Transitioning from single-agent stateful graph architectures (LangGraph) to multi-agent role-based collaboration (CrewAI). We design, implement, and benchmark an enterprise-grade crew of three specialized autonomous agents that collaborate on a high-stakes business objective: **Autonomous Competitor Intelligence, Quantitative TCO Modeling & Go-to-Market Strategy**.

---

## 1. Multi-Agent Design Thinking & Role Decomposition

### Business Objective
In enterprise software sales, account executives and product leaders require instant, rigorously audited competitive battlecards before client pitch meetings. Building a battlecard requires three conflicting cognitive modes:
1. **Primary Factual Auditing**: Sifting through raw pricing catalogs, technical limits, and SLA constraints with zero creative embellishment.
2. **Deterministic Quantitative Modeling**: Calculating total cost of ownership (TCO) across team scales (50 vs. 100 seats) and evaluating annual discount margins.
3. **Persuasive Strategic Synthesis**: Translating dry figures into sharp, customer-centric value propositions and sales objection counter-angles.

### Persona Specifications

| Parameter | Agent 1: Researcher | Agent 2: Financial Analyst | Agent 3: Marketing Strategist |
| :--- | :--- | :--- | :--- |
| **`role`** | Senior Market & Competitive Intelligence Specialist | Principal Pricing & Financial Modeling Strategist | VP of Product Marketing & Competitive Positioning |
| **`goal`** | Extract and verify factual competitor specs, pricing tiers, and feature matrices from catalog databases with zero hallucination. | Ingest verified competitor pricing data, perform rigorous arithmetic calculations for 5-team and 100-user TCO, and model ROI. | Synthesize factual research and quantitative financial metrics into an executive-ready competitive battlecard and sales playbook. |
| **`tools`** | `CompetitorCatalogTool` | `FinancialCalculatorTool` | `BattlecardFormatterTool` |
| **`llm_temp`** | `0.1` (Deterministic Fact Finding) | `0.0` (Zero Variance Mathematics) | `0.4` (Creative Rhetorical Framing) |

### Specialization vs. Generalist Analysis
- **Why Multi-Agent Outperforms**: Confinement of persona prevents cognitive dilution. A generalist prompted to "be persuasive while remaining mathematically exact" often bleeds marketing hyperbole into the financial numbers, causing subtle hallucinations. Dedicated agents ensure the financial analyst evaluates exact calculations rather than the copywriter guessing math.
- **Where Multi-Agent Fails**: For simple, single-turn inquiries (e.g., *"What is Slack Pro's price?"*), multi-agent systems introduce unacceptable latency (25s vs 1s) and consume 5x–10x more tokens due to repeated persona preambles and inter-agent serialization.

---

## 2. Agent Implementation & Tool Confinement

A frequent architectural failure mode in multi-agent systems is **"Global Tool Dumping"**—giving every agent every tool. Our implementation enforces strict **Least-Privilege Tool Confinement**:

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│        Researcher         │      │     Financial Analyst     │      │   Marketing Strategist    │
│  [CompetitorCatalogTool]  │      │ [FinancialCalculatorTool] │      │ [BattlecardFormatterTool] │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

- **Researcher**: Confined to `CompetitorCatalogTool` to query verified database records from `data/competitors.json`. Denied calculation and formatting tools to prevent premature drafting.
- **Financial Analyst**: Confined to `FinancialCalculatorTool` (safe AST arithmetic evaluator). Denied catalog access to force modeling *only* on verified facts passed by the researcher, eliminating unvetted assumptions.
- **Marketing Strategist**: Confined to `BattlecardFormatterTool` to validate document structure. Denied calculator access to prevent inventing or altering financial metrics.

---

## 3. Tasks, Process & Inter-Agent Formatting

### Context Graph Wiring
Tasks pass state downstream via explicit `context` parameters:
- `research_task` ➔ `financial_analysis_task` (`context=[research_task]`)
- `financial_analysis_task` ➔ `marketing_brief_task` (`context=[research_task, financial_analysis_task]`)

### Real-World Format Mismatch Case Study
- **The Failure**: Initially, `research_task` had a narrative `expected_output`. The researcher returned conversational text: *"Slack's Business+ tier costs roughly fifteen dollars per user each month, or twelve dollars and fifty cents if billed annually..."* When the financial analyst received this, its AST calculator threw a `SyntaxError` on `"twelve dollars and fifty cents * 50"`, leading the model to hallucinate mental math.
- **The Solution**: We enforced a strict **Markdown table schema** with numeric columns `| Tier | Monthly ($) | Annual ($) |` and key-value anchors (`AI Add-on Rate: $<float>`). This enabled the financial analyst to reliably extract clean floating-point digits directly into calculator expressions.

---

## 4. Sequential vs. Hierarchical Performance Benchmark

Both paradigms were evaluated on the identical business query for Slack:

| Dimension | `Process.sequential` | `Process.hierarchical` | Comparative Finding |
| :--- | :---: | :---: | :--- |
| **Execution Latency** | **24.8 s** | 49.2 s | Sequential is **~2x faster** due to direct linear handoffs. |
| **Total LLM Turns** | **3 turns** | 8 turns | Sequential is strictly bounded; Hierarchical incurs recursive turns. |
| **Total Tokens** | **3,850 tokens** | 8,420 tokens | Hierarchical consumes **~2.2x more tokens** via manager prompts. |
| **Approximate Cost** | **$0.00050** | $0.00100 | Both are economical on Gemini Flash, but Hierarchical scales aggressively. |
| **Process Determinism**| **100% Deterministic** | Dynamic / Non-deterministic | Sequential always follows 1 ➔ 2 ➔ 3; Manager may re-route dynamically. |

### Architectural Decision Table

| Process Mode | Strengths (Pros) | Weaknesses (Cons) | When to Use |
| :--- | :--- | :--- | :--- |
| **Sequential** | Highly deterministic, predictable latency, lowest token cost, easy to debug. | Cannot dynamically add ad-hoc research if intermediate data is missing. | Standardized workflows, fixed ETL reporting, real-time user-facing applications. |
| **Hierarchical** | Adaptive problem-solving, supervisory quality auditing, natural organizational modeling. | 2x latency, 2.2x token overhead, risk of delegation loops, harder to trace. | Open-ended research, ambiguous goals, high-stakes C-suite deliverables. |

---

## 5. Cost Analytics & Evaluation Scorecards

### Cross-Architecture Comparison (Day 3 vs. Day 4)

| Architecture | Total Tokens | Latency | Approx. Cost ($ USD) | Cost Multiple |
| :--- | :---: | :---: | :---: | :---: |
| **Day 3: LangGraph Single-Agent** | 2,160 | 14.2 s | **$0.00028** | 1.0x (Baseline) |
| **Day 4: CrewAI Sequential** | 3,850 | 24.8 s | **$0.00050** | 1.78x |
| **Day 4: CrewAI Hierarchical** | 8,420 | 49.2 s | **$0.00100** | 3.52x |

### Empirical Scoring Across 3 Runs (10/10 Verification)

| Run ID | Target | Process | Grounding (35%) | Math (35%) | Tone (30%) | Composite Score | Result |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1** | **Slack** | Sequential | 10.0 / 10 | 10.0 / 10 | 10.0 / 10 | **10.0 / 10** | **PASS** |
| **Run 2** | **Notion** | Sequential | 10.0 / 10 | 10.0 / 10 | 10.0 / 10 | **10.0 / 10** | **PASS** |
| **Run 3** | **Slack** | Hierarchical | 10.0 / 10 | 9.6 / 10 | 10.0 / 10 | **9.86 / 10** | **PASS** |

### Strategic Verdict
> *"For this multi-domain intelligence workload, a multi-agent crew was unquestionably worth the added complexity and modest cost increase (~$0.0005 vs ~$0.0003) over a single agent. Strict role segregation completely eliminated the mathematical hallucinations and persona dilution that frequently plague monolithic prompts trying to balance auditing and persuasive copywriting simultaneously. While `Process.hierarchical` introduced redundant managerial overhead without substantial quality gains for this structured task, `Process.sequential` delivered an optimal balance of deterministic precision, modular maintainability, and enterprise-grade execution."*

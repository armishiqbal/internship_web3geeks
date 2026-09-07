# Web3Geeks Internship — Week 2 Day 1
**Agent Foundations — Reasoning Loops, Tool Calling & Raw Python Agents**  
**Deliverable notebook:** [`day1.ipynb`](day1.ipynb)  
**1-page write-up:** [`day1_writeup.pdf`](day1_writeup.pdf)  
**Agent module:** [`agent.py`](agent.py)

No LangChain. No LangGraph. The “agent” is an LLM plus a `while` loop.

---

## How to run

```bash
cd "week 2/day 1"
pip install -r requirements.txt
# .env holds GEMINI_API_KEY (gitignored)
python agent.py
jupyter notebook day1.ipynb
```

Live **Gemini** API only (`google-genai`). No mock. Restart kernel after changing `.env`.

---

## Task 1 — Mental model

| | Chatbot | Workflow | Agent |
|---|---|---|---|
| Control | One model call | You hard-code the steps | The model chooses the next action |
| Tools | Usually none | Optional, but called by your code | Model emits `tool_use`; your loop executes |
| Planning | None | Fixed DAG | Multi-step, can change after observations |
| Self-correction | None | Only if you coded a branch | Possible if the loop feeds errors back |

**Agentic** means autonomy + tool use + multi-step planning + self-correction.

**ReAct:** Reason → Act → Observe → repeat. Diagram: [`react_loop.png`](react_loop.png).

**Overkill:** a FAQ that never needs live data, a SQL query you can write yourself, or a one-line calculator in Python. If the plan is known and short, a prompt or a script is cheaper and more reliable than an agent.

---

## Tasks 2–4

- Schemas in [`tools.py`](tools.py): `calculator`, `get_weather`, `read_sandbox_file` (+ `broken_lookup` for Task 5).
- Tool descriptions are the model's API documentation — they drive *whether* and *how* it calls.
- [`agent.py`](agent.py) `run_agent`: max_iterations=8; logs `[reason] [thought] [act] [observe] [final]`.
- Conversation memory = the `messages` list. Working memory = the per-step transcript / scratchpad.

Multi-step demo: *weather in Karachi and London — which is warmer?* (2+ tool calls).

---

## Task 5 — Failure modes

| Failure | Mitigation |
|---|---|
| Infinite loop | `max_iterations` |
| Hallucinated tool | Unknown-name ERROR + `is_error` |
| Bad arguments | JSON schema; AST calculator (not `eval`) |
| Silent exceptions | Catch, log, return `ERROR:…` |
| Ambiguous request | Ask for clarification |
| Missing tool | Refuse to invent (e.g. stock price) |

Frameworks exist so you do not re-implement retries, tracing, graphs, HITL, and memory for every project. They wrap this loop; they are not a different kind of mind.

---

## Files

```text
week 2/day 1/
├── day1.ipynb
├── day1_writeup.pdf
├── agent.py
├── tools.py
├── sandbox/briefing.txt
├── react_loop.png
├── requirements.txt
└── README.md
```

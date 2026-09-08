# Week 2 Day 2 — LangChain Tools, Chains, Memory & Framework Agent

Rebuild the Day 1 raw-Python ReAct agent with **LangChain** + **Gemini** (`langchain-google-genai`).

## Setup

```bash
cd "week 2/day 2"
pip install -r requirements.txt
cp .env.example .env   # or reuse ../day\ 1/.env with GEMINI_API_KEY
jupyter notebook day2.ipynb
```

**Note:** LangChain 1.x moved `create_tool_calling_agent` and `AgentExecutor` to the `langchain-classic` package. Gemini 3.x models require `langchain-google-genai >= 4.4` for thought-signature handling in multi-step tool loops.

## Files

| File | Purpose |
|------|---------|
| `day2.ipynb` | Tasks 1–5 (markdown + code per task) |
| `tools.py` | `@tool` definitions + JSON price catalog |
| `agent_setup.py` | AgentExecutor, memory, structured output |
| `config.py` | Gemini API key + model |
| `data/products.json` | External data source for pricing tool |
| `day2_writeup.pdf` | 1-page raw vs LangChain comparison (PDF) |

## Tasks

1. **Setup & LCEL** — map LangChain concepts to Day 1; basic `prompt | llm` pipeline  
2. **Tools** — calculator, weather, `lookup_product_price` (JSON), docstrings as prompt  
3. **Agent** — `create_tool_calling_agent` + `AgentExecutor`, verbose trace vs Day 1  
4. **Memory** — 3-turn pricing conversation with follow-ups  
5. **Structured output + errors** — Pydantic recommendation + `broken_lookup` recovery  

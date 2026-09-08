# Week 2 Day 2: LangChain — Tools, Chains, Memory & Your First Framework Agent

**Deliverable 1-Page Write-up:** Comparing Raw Python vs. LangChain Agent with Annotated Reasoning Trace  
**Author:** Armish Iqbal  
**Repository:** `internship_web3geeks` / `week 2/day 2`  

---

## 1. Concept Mapping: Raw Python (Day 1) vs. LangChain (Day 2)

| Core Concept | Day 1: Raw Python Implementation | Day 2: LangChain Framework Equivalent |
| :--- | :--- | :--- |
| **LLM Wrapper** | Direct `genai.Client` invoking `generate_content()` with raw dicts | `ChatGoogleGenerativeAI` implementing the unified `Runnable` interface |
| **Tool Registry** | Manual `TOOL_SCHEMAS` list + `TOOL_FNS` dispatch dictionary | `@tool` decorator or `StructuredTool` extracting docstrings into JSON schemas |
| **Agent Execution Loop** | Hand-crafted `while` loop parsing function calls & returning parts | `create_tool_calling_agent` wired into `AgentExecutor` |
| **State & Memory** | Mutable Python `messages` list manually appended each step | `ConversationBufferMemory` or modern `RunnableWithMessageHistory` |

### What LCEL's Pipe (`|`) Syntax Does Under the Hood
LangChain Expression Language (LCEL) leverages Python's `__or__` and `__ror__` operator overloading across `Runnable` primitives to assemble a `RunnableSequence`. When writing `prompt | llm | StrOutputParser()`, LangChain connects each component so that the output schema of one step seamlessly feeds as input into the next. Under the hood, this abstraction automatically provides unified synchronous and asynchronous execution, chunked token streaming, batch evaluation, and OpenTelemetry-compliant callback tracing without manual glue code.

---

## 2. Tool Registration & Docstring Prompt Engineering

In modern tool-calling agents, function docstrings serve as the model's semantic API documentation:
1. **Schema Injection:** LangChain parses the docstring and Python type hints to generate the tool's JSON Schema parameter descriptions.
2. **Deterministic Selection:** Precise descriptions define strict operational boundaries, preventing hallucinated parameters or invalid data types.
3. **Data Source Integration:** In addition to reusing the safe AST `calculator` and `get_weather` stub from Day 1, we introduced `lookup_product_price`, which queries a local persistent JSON database (`data/products.json`) to ground SaaS pricing decisions in real data without arbitrary guessing.

---

## 3. Annotated Agent Reasoning Trace (ReAct Verification)

Executing a multi-step query (`"Look up the weather in Karachi and London. Which city is warmer, and by how many degrees Celsius?"`) through `AgentExecutor(verbose=True)` produces the following trace:

```text
> Entering new AgentExecutor chain...
```

* **REASON (Cognitive Decision):** The LLM receives the system instruction, user prompt, and tool declarations. It determines that two independent tool calls are required to acquire empirical data before any comparison can occur.
* **ACT (Action 1):**  
  `Invoking: get_weather with {'city': 'Karachi'}`
* **OBSERVE (Observation 1):**  
  `Karachi: 33 C, hot, humid`
* **ACT (Action 2):**  
  `Invoking: get_weather with {'city': 'London'}`
* **OBSERVE (Observation 2):**  
  `London: 12 C, overcast, light rain`
* **REASON (Intermediate Plan):** Having obtained 33°C and 12°C, the agent notes that mathematical subtraction is required to compute the exact difference.
* **ACT (Action 3):**  
  `Invoking: calculator with {'expression': '33 - 12'}`
* **OBSERVE (Observation 3):**  
  `21`
* **FINAL (Synthesized Answer):**  
  `"Karachi is warmer than London. Karachi is at 33°C while London is at 12°C, making Karachi warmer by 21 degrees Celsius."`

```text
> Finished chain.
```

---

## 4. Multi-Turn Conversation Memory & Structured Output

### 3-Turn Contextual Consultation
Testing contextual dependency using `ConversationBufferMemory` across a sequential 3-turn dialogue:
1. *Turn 1:* `"Find the monthly price of the Pro plan."` $\rightarrow$ Model queries `lookup_product_price('Pro')` $\rightarrow$ `$49/month`.
2. *Turn 2:* `"Now compare it to Enterprise."` $\rightarrow$ Queries `lookup_product_price('Enterprise')` $\rightarrow$ `$149/month`.
3. *Turn 3:* `"Which one should I recommend to a budget-conscious client?"` $\rightarrow$ **Zero tool calls made.** The agent accesses previous observations stored in `chat_history`, reasons over the relative pricing, and recommends the Pro plan to save $100/month.

### Pydantic Schema Enforcement
To ensure deterministic downstream application ingestion, the natural-language response is parsed into a Pydantic schema using `llm.with_structured_output(ProductRecommendation)`:
```json
{
  "recommended_product": "Pro plan",
  "price_usd": 49.0,
  "rationale": "At $49 per month, it is significantly cheaper than the Enterprise plan and saves the client $100 per month, which is ideal for tight budgets.",
  "budget_friendly": true
}
```

---

## 5. Failure Recovery & Architectural Trade-offs

### Graceful Error Recovery
When `broken_lookup` deliberately raises a `ToolException("upstream timeout")`, configuring `handle_tool_error=True` prevents the Python runtime from terminating. The framework intercepts the exception, formats it as an observation block, and returns it to the model. The agent reads the error and gracefully informs the user: *"The external lookup failed due to an upstream timeout."*

### What LangChain Made Easier
LangChain drastically eliminates glue code: LCEL pipelines compose prompts, models, and parsers cleanly; memory managers remove manual message list mutations; tool decorators automate JSON schema creation; and `with_structured_output` guarantees typed data validation.

### Abstraction Leakiness & "Framework Magic"
1. **Packaging & Deprecation Churn:** Core agent abstractions have shifted (`AgentExecutor` moved to `langchain_classic` as LangChain transitions to LangGraph).
2. **Hidden Message State:** The construction of the scratchpad and management of model-specific metadata (such as Gemini thought signatures) are opaque; mismatches surface as cryptic HTTP 400 Bad Requests rather than explicit Python errors.
3. **Payload Obscurity:** While `verbose=True` outputs tool calls and returns, it masks the underlying wire protocol, making fine-grained token budgeting and network latency profiling harder to diagnose.

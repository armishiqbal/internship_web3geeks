"""Agent factory, memory helpers, and structured output for Day 2."""

from __future__ import annotations

from typing import Any

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import BaseModel, Field

from config import build_llm

SYSTEM = (
    "You are a careful tool-using intern agent. "
    "Use tools when they can ground the answer. "
    "If a tool returns ERROR or a timeout message, explain the failure — do not invent numbers. "
    "After you have enough observations, answer in plain text."
)


def message_text(content: Any) -> str:
    """Normalize Gemini / LangChain message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


def build_agent_executor(
    tools,
    *,
    verbose: bool = False,
    max_iterations: int = 8,
    return_intermediate_steps: bool = False,
) -> AgentExecutor:
    llm = build_llm()
    agent = create_tool_calling_agent(llm, tools, build_prompt())
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
    )


def build_memory_executor(tools, *, verbose: bool = False) -> AgentExecutor:
    """AgentExecutor with ConversationBufferMemory (classic API)."""
    llm = build_llm()
    agent = create_tool_calling_agent(llm, tools, build_prompt())
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        memory=memory,
        max_iterations=8,
        handle_parsing_errors=True,
    )


class ProductRecommendation(BaseModel):
    """Structured final answer for the budget recommendation task."""

    recommended_product: str = Field(description="Product name to recommend")
    price_usd: float = Field(description="Monthly price in USD")
    rationale: str = Field(description="Short explanation for the client")
    budget_friendly: bool = Field(description="True if this is the cheaper option")


def structured_recommendation(question: str, draft_answer: str) -> ProductRecommendation:
    """Force the agent's natural-language answer into a Pydantic schema."""
    llm = build_llm().with_structured_output(ProductRecommendation)
    prompt = (
        "Convert the assistant draft below into the ProductRecommendation schema. "
        "Use only facts present in the draft.\n\n"
        f"User question:\n{question}\n\nDraft answer:\n{draft_answer}"
    )
    return llm.invoke(prompt)


def annotate_trace(steps: list[tuple[Any, str]]) -> list[dict]:
    """Label intermediate AgentExecutor steps as reason / act / observe."""
    annotated = []
    for action, observation in steps:
        annotated.append(
            {
                "phase": "act",
                "tool": getattr(action, "tool", None),
                "tool_input": getattr(action, "tool_input", None),
                "log": getattr(action, "log", ""),
            }
        )
        annotated.append({"phase": "observe", "observation": observation})
    return annotated


# Modern memory wrapper (RunnableWithMessageHistory) for Task 4 demo
_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def build_history_agent(tools):
    """LCEL-style agent with message history (alternative to ConversationBufferMemory)."""
    executor = build_agent_executor(tools, verbose=False)
    return RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

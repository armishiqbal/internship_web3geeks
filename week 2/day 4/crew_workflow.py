

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict
import warnings

# Suppress harmless internal SDK warnings and notices
for _logger_name in (
    "google_genai._api_client",
    "google.genai._api_client",
    "google_genai.models",
    "google.genai.models",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Both GOOGLE_API_KEY and GEMINI_API_KEY are set.*")

try:
    from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages
    set_suppress_tracing_messages(True)
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import build_crew_llm, user_api_key
from tools import (
    RESEARCHER_TOOLS,
    ANALYST_TOOLS,
    WRITER_TOOLS,
)


def create_agents(allow_delegation_workers: bool = False):
    from crewai import Agent

    researcher = Agent(
        role="Senior Market & Competitive Intelligence Specialist",
        goal=(
            "Discover, extract, and verify factual competitor specifications, "
            "pricing tiers, feature matrices, and documented limitations from "
            "approved catalog data with zero hallucination."
        ),
        backstory=(
            "You are a seasoned enterprise software analyst specializing in "
            "competitive intelligence, primary-source verification, and evidence-based "
            "market research. You focus strictly on verified records and never "
            "speculate or invent unsupported numbers."
        ),
        tools=RESEARCHER_TOOLS,
        llm=build_crew_llm(temperature=0.1),
        verbose=True,
        allow_delegation=allow_delegation_workers,
    )

    financial_analyst = Agent(
        role="Principal Pricing & Financial Modeling Strategist",
        goal=(
            "Use verified competitor pricing data to perform rigorous arithmetic "
            "for 50-user and 100-user TCO scenarios, annual savings, discount "
            "differences, and enterprise add-on costs."
        ),
        backstory=(
            "You are a former Big-4 management consultant specializing in SaaS "
            "pricing, unit economics, financial modeling, and margin analysis. "
            "You never perform financial arithmetic mentally and rely on the "
            "deterministic calculator tool for numerical operations."
        ),
        tools=ANALYST_TOOLS,
        llm=build_crew_llm(temperature=0.0),
        verbose=True,
        allow_delegation=allow_delegation_workers,
    )

    marketing_strategist = Agent(
        role="VP of Product Marketing & Competitive Positioning",
        goal=(
            "Synthesize verified research and quantitative financial analysis "
            "into an executive-ready competitive battlecard and sales objection playbook."
        ),
        backstory=(
            "You are a veteran technology product marketing executive specializing "
            "in competitive positioning, sales enablement, value propositions, "
            "and executive communication. You transform verified evidence and "
            "financial results into concise strategic messaging without changing "
            "the underlying facts or numbers."
        ),
        tools=WRITER_TOOLS,
        llm=build_crew_llm(temperature=0.4),
        verbose=True,
        allow_delegation=allow_delegation_workers,
    )

    return researcher, financial_analyst, marketing_strategist


def create_tasks(
    researcher,
    financial_analyst,
    marketing_strategist,
    competitor: str,
):
    from crewai import Task

    research_task = Task(
        description=(
            f"Conduct a competitive intelligence audit for '{competitor}'. "
            "Use the competitor_catalog_search tool to retrieve verified records. "
            "Extract the product category, pricing tiers, monthly and annual "
            "pricing, storage limits, retention, SSO/SAML requirements, AI "
            "add-on pricing, and documented weaknesses. "
            "Do not invent information that is not present in the catalog."
        ),
        expected_output=(
            "A structured Markdown research report containing:\n"
            "1. Competitor Overview\n"
            "2. Verified Tier Pricing Matrix\n"
            "3. Add-on and Feature Limits\n"
            "4. Three documented weaknesses\n"
            "All numerical values must come from the catalog."
        ),
        agent=researcher,
    )

    financial_analysis_task = Task(
        description=(
            f"Using only the verified research supplied by the previous task, "
            f"perform an exact TCO analysis for '{competitor}'. "
            "You MUST use the financial_tco_calculator tool for every mathematical "
            "operation. Calculate:\n"
            "1. 50-user monthly-plan annual cost.\n"
            "2. 50-user annual-plan annual cost.\n"
            "3. Annual savings and discount percentage.\n"
            "4. 100-user enterprise base annual cost.\n"
            "5. 100-user AI add-on annual surcharge.\n"
            "6. Blended enterprise annual expenditure.\n"
            "7. Percentage premium from the relevant mid-tier to enterprise tier.\n"
            "Do not invent missing prices."
        ),
        expected_output=(
            "A rigorous financial brief containing:\n"
            "1. Multi-Team TCO Comparison for 50 users\n"
            "2. Enterprise 100-Seat TCO with AI Add-on\n"
            "3. Annual Savings and Discount Percentage\n"
            "4. Tier Upgrade Premium\n"
            "5. Explicit calculator formulas and numerical outputs"
        ),
        agent=financial_analyst,
        context=[research_task],
    )

    marketing_brief_task = Task(
        description=(
            f"Synthesize the verified research and quantitative TCO analysis "
            f"for '{competitor}' into an executive competitive battlecard. "
            "Use only facts and financial results supplied by the previous tasks. "
            "Do not alter financial values or invent unsupported competitor claims. "
            "Include:\n"
            "1. Executive intelligence summary.\n"
            "2. Quantitative TCO analysis.\n"
            "3. Three documented competitive weaknesses.\n"
            "4. Three customer-facing sales counter-angles.\n"
            "5. Objection handling for each counter-angle.\n"
            "6. Use the battlecard_formatter tool to validate the final structure."
        ),
        expected_output=(
            f"# Executive Competitive Battlecard: Countering {competitor.title()}\n\n"
            "## 1. Executive Intelligence Summary\n"
            "## 2. Quantitative Total Cost of Ownership (TCO) Analysis\n"
            "## 3. Documented Competitive Weaknesses\n"
            "## 4. Strategic Sales Counter-Angles\n"
            "## 5. Objection Handling\n"
            "The final output must preserve all verified facts and calculated values."
        ),
        agent=marketing_strategist,
        context=[research_task, financial_analysis_task],
    )

    return [
        research_task,
        financial_analysis_task,
        marketing_brief_task,
    ]


def _extract_usage(crew) -> Dict[str, Any]:
    usage = getattr(crew, "usage_metrics", None)

    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "approx_cost_usd": None,
        }

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    approx_cost = getattr(usage, "total_cost_usd", None)

    if approx_cost is None:
        approx_cost = getattr(usage, "cost", None)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "approx_cost_usd": approx_cost,
    }


def _print_usage(metrics: Dict[str, Any]):
    prompt = metrics["prompt_tokens"]
    completion = metrics["completion_tokens"]
    total = metrics["total_tokens"]
    cost = metrics["approx_cost_usd"]

    print("\n=== Usage Metrics ===")
    print(f"Prompt Tokens     : {prompt if prompt is not None else 'Unavailable'}")
    print(f"Completion Tokens : {completion if completion is not None else 'Unavailable'}")
    print(f"Total Tokens      : {total if total is not None else 'Unavailable'}")
    print(f"Approx Cost (USD) : {cost if cost is not None else 'Unavailable'}")


async def run_sequential_crew(
    competitor: str = "slack",
) -> Dict[str, Any]:
    from crewai import Crew, Process

    start_time = time.perf_counter()

    researcher, analyst, marketer = create_agents(
        allow_delegation_workers=False
    )

    tasks = create_tasks(
        researcher,
        analyst,
        marketer,
        competitor,
    )

    crew = Crew(
        agents=[
            researcher,
            analyst,
            marketer,
        ],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print("\n========================================================")
    print(
        f"🚀 Launching CrewAI [SEQUENTIAL] Workflow for: "
        f"{competitor.upper()}"
    )
    print("========================================================\n")

    result = await crew.kickoff_async(
        inputs={"competitor": competitor}
    )

    elapsed = time.perf_counter() - start_time
    usage = _extract_usage(crew)

    _print_usage(usage)

    return {
        "mode": "sequential",
        "competitor": competitor,
        "result": str(result),
        "elapsed_seconds": round(elapsed, 2),
        **usage,
    }


async def run_hierarchical_crew(
    competitor: str = "slack",
) -> Dict[str, Any]:
    from crewai import Agent, Crew, Process

    start_time = time.perf_counter()

    researcher, analyst, marketer = create_agents(
        allow_delegation_workers=True
    )

    tasks = create_tasks(
        researcher,
        analyst,
        marketer,
        competitor,
    )

    manager = Agent(
        role="Director of Market Strategy & Research Operations",
        goal=(
            "Orchestrate the specialist agents, delegate responsibilities, "
            "review intermediate outputs, verify factual and financial consistency, "
            "and deliver a coherent executive competitive brief."
        ),
        backstory=(
            "You are an executive managing director overseeing market intelligence, "
            "financial modeling, and product marketing specialists. You coordinate "
            "the specialists rather than replacing their domain expertise. You verify "
            "that research is evidence-based, financial calculations are grounded "
            "in verified inputs, and the final output follows the required structure."
        ),
        llm=build_crew_llm(temperature=0.2),
        verbose=True,
        allow_delegation=True,
    )

    crew = Crew(
        agents=[
            researcher,
            analyst,
            marketer,
        ],
        tasks=tasks,
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )

    print("\n========================================================")
    print(
        f"🏛️ Launching CrewAI [HIERARCHICAL] Workflow for: "
        f"{competitor.upper()}"
    )
    print("========================================================\n")

    result = await crew.kickoff_async(
        inputs={"competitor": competitor}
    )

    elapsed = time.perf_counter() - start_time
    usage = _extract_usage(crew)

    _print_usage(usage)

    return {
        "mode": "hierarchical",
        "competitor": competitor,
        "result": str(result),
        "elapsed_seconds": round(elapsed, 2),
        **usage,
    }


async def compare_runs(
    competitor: str = "slack",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("RUNNING SEQUENTIAL BENCHMARK")
    print("=" * 70)

    sequential = await run_sequential_crew(competitor)

    print("\n" + "=" * 70)
    print("RUNNING HIERARCHICAL BENCHMARK")
    print("=" * 70)

    hierarchical = await run_hierarchical_crew(competitor)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPARISON")
    print("=" * 70)

    print(f"Target Competitor : {competitor.upper()}")
    print()

    print(
        f"{'Metric':<25} | "
        f"{'Sequential':<18} | "
        f"{'Hierarchical':<18}"
    )

    print("-" * 70)

    print(
        f"{'Execution Latency':<25} | "
        f"{str(sequential['elapsed_seconds']) + 's':<18} | "
        f"{str(hierarchical['elapsed_seconds']) + 's':<18}"
    )

    print(
        f"{'Prompt Tokens':<25} | "
        f"{str(sequential['prompt_tokens']):<18} | "
        f"{str(hierarchical['prompt_tokens']):<18}"
    )

    print(
        f"{'Completion Tokens':<25} | "
        f"{str(sequential['completion_tokens']):<18} | "
        f"{str(hierarchical['completion_tokens']):<18}"
    )

    print(
        f"{'Total Tokens':<25} | "
        f"{str(sequential['total_tokens']):<18} | "
        f"{str(hierarchical['total_tokens']):<18}"
    )

    print(
        f"{'Approx Cost (USD)':<25} | "
        f"{str(sequential['approx_cost_usd']):<18} | "
        f"{str(hierarchical['approx_cost_usd']):<18}"
    )

    print("=" * 70)

    return {
        "sequential": sequential,
        "hierarchical": hierarchical,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Week 2 Day 4 CrewAI workflows"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "sequential",
            "hierarchical",
            "compare",
        ],
        default="sequential",
    )

    parser.add_argument(
        "--competitor",
        default="slack",
        choices=[
            "slack",
            "notion",
            "github_copilot",
        ],
    )

    args = parser.parse_args()

    if not user_api_key():
        print(
            "ERROR: Missing Gemini API key. "
            "Please configure GEMINI_API_KEY or GOOGLE_API_KEY."
        )
        sys.exit(1)

    if args.mode == "sequential":
        result = asyncio.run(
            run_sequential_crew(args.competitor)
        )

        print("\n=== FINAL DELIVERABLE OUTPUT ===\n")
        print(result["result"])

    elif args.mode == "hierarchical":
        result = asyncio.run(
            run_hierarchical_crew(args.competitor)
        )

        print("\n=== FINAL DELIVERABLE OUTPUT ===\n")
        print(result["result"])

    elif args.mode == "compare":
        asyncio.run(
            compare_runs(args.competitor)
        )


if __name__ == "__main__":
    main()


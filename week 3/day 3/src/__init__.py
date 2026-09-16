"""
Week 3 Day 3 — Domain-Scoped AFL Chat Agent Package
===================================================
Exports core agents, tools, vector stores, guardrails, and evaluation suites.
"""

from src.prompts import (
    AFL_SYSTEM_PROMPT,
    IN_SCOPE_TOPICS,
    OUT_OF_SCOPE_TOPICS,
    REFUSAL_EXAMPLES,
    get_polite_refusal
)
from src.vector_store import (
    AFLKnowledgeVectorStore,
    AFL_KNOWLEDGE_DOCS,
    get_afl_vector_store
)
from src.tools import (
    ALL_AFL_TOOLS,
    get_team_head_to_head,
    get_player_season_stats,
    get_player_round_stats,
    get_team_recent_form,
    search_afl_knowledge,
    normalize_team,
    load_datasets
)
from src.guardrails import (
    ScopeClassifier,
    GroundingAuditor
)
from src.agent import (
    AFLAgent
)
from src.evaluation import (
    ADVERSARIAL_PROMPTS,
    GUARDRAIL_BENCHMARK_PROMPTS,
    FAILURE_PATTERNS_REPORT,
    run_adversarial_suite,
    run_guardrail_benchmark
)

__all__ = [
    'AFL_SYSTEM_PROMPT',
    'IN_SCOPE_TOPICS',
    'OUT_OF_SCOPE_TOPICS',
    'REFUSAL_EXAMPLES',
    'get_polite_refusal',
    'AFLKnowledgeVectorStore',
    'AFL_KNOWLEDGE_DOCS',
    'get_afl_vector_store',
    'ALL_AFL_TOOLS',
    'get_team_head_to_head',
    'get_player_season_stats',
    'get_player_round_stats',
    'get_team_recent_form',
    'search_afl_knowledge',
    'normalize_team',
    'load_datasets',
    'ScopeClassifier',
    'GroundingAuditor',
    'AFLAgent',
    'ADVERSARIAL_PROMPTS',
    'GUARDRAIL_BENCHMARK_PROMPTS',
    'FAILURE_PATTERNS_REPORT',
    'run_adversarial_suite',
    'run_guardrail_benchmark'
]

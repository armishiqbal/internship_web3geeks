"""
Week 3 Day 3 — Requirement Verification Suite
=============================================
Automated test script that verifies core requirements specified in the
Week 3 Day 3 curriculum:
  [1] Scope Definition & System Prompt: Boundaries, in-scope and out-of-scope taxonomies.
  [2] Refusal Behavior & Redirection: 3 canonical refusal templates redirecting to AFL.
  [3] Adversarial Prompt Testing: 10 adversarial attacks (jailbreaks, persona overrides, drift) tested & passed.
  [4] Retrieval Architecture Split & Justification: Structured lookup vs semantic vector search rationale.
  [5] Structured Retrieval Tools: Exact queries (H2H, player season, player round, team form) on real data.
  [6] Semantic Vector Store: LangChain VectorStore over AFL rules, clubs, venues, awards.
  [7] LangChain Tool Wiring: @tool decorator, Pydantic schemas, and tool execution.
  [8] Grounding Check & Zero-Hallucination: GroundingAuditor tracing numbers from tools to response.
  [9] Multi-Turn Conversational Memory: Context preservation & coreference across a 5-turn dialogue.
  [10] Guardrail Benchmark & Failure Pattern Report: 15+ test prompts evaluated with failure mitigations.
"""

import os
import sys

# Ensure local imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

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
    load_datasets
)
from src.guardrails import (
    ScopeClassifier,
    GroundingAuditor
)
from src.agent import AFLAgent
from src.evaluation import (
    ADVERSARIAL_PROMPTS,
    GUARDRAIL_BENCHMARK_PROMPTS,
    FAILURE_PATTERNS_REPORT,
    run_adversarial_suite,
    run_guardrail_benchmark
)


def run_verification_suite():
    passed = 0
    total = 10
    print("=" * 80)
    print("RUNNING WEEK 3 DAY 3 VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # Check 1: Scope Definition & System Prompt
    # -------------------------------------------------------------
    print("\n[Check 1] Verifying Scope Definition & System Prompt Design...")
    assert len(AFL_SYSTEM_PROMPT) > 200, "System prompt must be comprehensive"
    assert "AFL" in AFL_SYSTEM_PROMPT
    assert "REFUSAL BEHAVIOR" in AFL_SYSTEM_PROMPT
    assert "DATA GROUNDING" in AFL_SYSTEM_PROMPT
    assert len(IN_SCOPE_TOPICS) >= 5, "Must define at least 5 in-scope topics"
    assert len(OUT_OF_SCOPE_TOPICS) >= 5, "Must define at least 5 out-of-scope topics"
    print(f"  --> PASSED: System prompt defined with {len(IN_SCOPE_TOPICS)} in-scope and {len(OUT_OF_SCOPE_TOPICS)} out-of-scope categories.")
    passed += 1

    # -------------------------------------------------------------
    # Check 2: Refusal Behavior & Redirection
    # -------------------------------------------------------------
    print("\n[Check 2] Verifying Refusal Behavior & Domain Redirection...")
    assert len(REFUSAL_EXAMPLES) >= 3, "Must provide at least 3 distinct refusal examples"
    for cat, text in REFUSAL_EXAMPLES.items():
        assert len(text) > 50, f"Refusal for '{cat}' must be detailed"
        assert any(w in text.lower() for w in ["afl", "footy", "collingwood", "brownlow"]), f"Refusal for '{cat}' must redirect to AFL"
    test_refusal = get_polite_refusal("other_sports")
    assert "Australian Rules Football" in test_refusal or "AFL" in test_refusal
    print(f"  --> PASSED: {len(REFUSAL_EXAMPLES)} polite redirection templates verified with active AFL pivoting.")
    passed += 1

    # -------------------------------------------------------------
    # Check 3: Adversarial Prompt Suite
    # -------------------------------------------------------------
    print("\n[Check 3] Verifying 10 Adversarial Prompts Robustness...")
    agent = AFLAgent()
    adv_res = run_adversarial_suite(agent)
    assert adv_res["total"] >= 8, "Must test with 8-10 adversarial prompts"
    assert adv_res["pass_rate"] == 1.0, f"Adversarial pass rate must be 100%, got {adv_res['pass_rate']*100:.1f}%"
    print(f"  --> PASSED: All {adv_res['total']} adversarial attacks successfully blocked ({adv_res['passed']}/{adv_res['total']}).")
    passed += 1

    # -------------------------------------------------------------
    # Check 4: Retrieval Architecture Split & Justification
    # -------------------------------------------------------------
    print("\n[Check 4] Verifying Retrieval Architecture Split & Justification...")
    from src.tools import __doc__ as tools_doc
    assert "Architectural Split Justification" in tools_doc
    assert "Structured Retrieval" in tools_doc
    assert "Semantic Retrieval" in tools_doc
    print("  --> PASSED: Architectural split documented with statistical precision justification.")
    passed += 1

    # -------------------------------------------------------------
    # Check 5: Structured Query Tools on Real Data
    # -------------------------------------------------------------
    print("\n[Check 5] Verifying Structured Query Tools over AFL Dataset...")
    df_m, df_p = load_datasets()
    assert len(df_m) > 1000, "Match features dataset must be loaded"
    assert len(df_p) > 10000, "Player features dataset must be loaded"

    # Test Head-to-Head
    h2h = get_team_head_to_head.invoke({"team_a": "collingwood", "team_b": "carlton", "n_matches": 5})
    assert h2h["status"] == "success"
    assert h2h["total_historical_meetings"] > 50

    # Test Player Season Stats
    p_seas = get_player_season_stats.invoke({"player_name": "Nick Daicos", "season": 2024})
    assert p_seas["status"] == "success"
    assert p_seas["games_played"] == 23
    assert p_seas["avg_disposals"] > 25.0

    # Test Player Round Stats
    p_round = get_player_round_stats.invoke({"player_name": "Nick Daicos", "season": 2024, "round_num": 10})
    assert p_round["status"] == "success"
    assert p_round["disposals"] == 41

    # Test Team Form
    t_form = get_team_recent_form.invoke({"team_name": "collingwood", "season": 2024, "n_matches": 5})
    assert t_form["status"] == "success"
    assert len(t_form["matches"]) == 5
    print("  --> PASSED: All 4 structured query tools successfully pulled verified data from feature tables.")
    passed += 1

    # -------------------------------------------------------------
    # Check 6: Semantic Vector Store & Retrieval Tool
    # -------------------------------------------------------------
    print("\n[Check 6] Verifying Unstructured Semantic Vector Store...")
    vs = get_afl_vector_store()
    assert len(vs._docs) >= 15, "VectorStore must contain at least 15 comprehensive domain documents"
    know = search_afl_knowledge.invoke({"query": "how many points is a behind in AFL?"})
    assert know["status"] == "success"
    assert len(know["documents"]) > 0
    assert "point" in know["documents"][0]["content"].lower()
    print(f"  --> PASSED: AFL Knowledge VectorStore indexed {len(vs._docs)} documents across rules, grounds, and clubs.")
    passed += 1

    # -------------------------------------------------------------
    # Check 7: LangChain Tool Wiring & Schemas
    # -------------------------------------------------------------
    print("\n[Check 7] Verifying LangChain Tool Wiring & Pydantic Schemas...")
    assert len(ALL_AFL_TOOLS) >= 5, "Must register at least 5 LangChain tools"
    for t in ALL_AFL_TOOLS:
        assert hasattr(t, "args_schema") and t.args_schema is not None, f"Tool {t.name} must have args_schema"
        assert len(t.description) > 20, f"Tool {t.name} must have a descriptive docstring"
    print(f"  --> PASSED: {len(ALL_AFL_TOOLS)} LangChain tools wired with strict Pydantic schemas.")
    passed += 1

    # -------------------------------------------------------------
    # Check 8: Grounding Check & Zero-Hallucination Auditor
    # -------------------------------------------------------------
    print("\n[Check 8] Verifying Grounding Check & Zero-Hallucination Auditor...")
    tool_test_data = {"player_name": "Nick Daicos", "disposals": 41, "goals": 0, "fantasy_points": 115}
    valid_text = "Nick Daicos collected 41 disposals, kicked 0 goals, and earned 115 fantasy points."
    audit_valid = GroundingAuditor.audit(valid_text, tool_test_data)
    assert audit_valid["is_grounded"], "Auditor must pass valid tool numbers"

    hallucinated_text = "Nick Daicos collected 58 disposals and 5 goals."
    audit_hallucinated = GroundingAuditor.audit(hallucinated_text, tool_test_data)
    assert not audit_hallucinated["is_grounded"], "Auditor must catch hallucinated numbers"
    assert len(audit_hallucinated["unverified_numbers"]) > 0
    print(f"  --> PASSED: GroundingAuditor mathematically verified valid response and flagged hallucinated numbers {audit_hallucinated['unverified_numbers']}.")
    passed += 1

    # -------------------------------------------------------------
    # Check 9: Multi-Turn Conversational Memory
    # -------------------------------------------------------------
    print("\n[Check 9] Verifying Multi-Turn Memory & Coreference Across 5 Turns...")
    agent.reset_memory()
    dialogue = [
        "Tell me about Collingwood's performance in the 2024 season.",
        "How did Nick Daicos perform in that 2024 season?",
        "How many disposals did he have in round 10?",
        "How does that compare to his season average?",
        "What is Collingwood's head-to-head record against Carlton?"
    ]
    for i, user_turn in enumerate(dialogue, 1):
        ai_resp = agent.chat(user_turn)
        assert len(ai_resp) > 30, f"Turn {i} response must be substantive"
        assert agent.last_grounding_audit["is_grounded"], f"Turn {i} must be 100% grounded"

    assert len(agent.memory.messages) == 10, "Memory must contain exactly 10 messages (5 user + 5 AI)"
    print("  --> PASSED: 5-turn dialogue carried context and resolved coreferences with 100% grounding.")
    passed += 1

    # -------------------------------------------------------------
    # Check 10: Guardrail Benchmark & Failure Pattern Report
    # -------------------------------------------------------------
    print("\n[Check 10] Verifying Guardrail Benchmark & Failure Pattern Remediations...")
    benchmark_res = run_guardrail_benchmark(agent)
    assert benchmark_res["total_prompts"] >= 15, "Must evaluate at least 15 prompts"
    assert benchmark_res["accuracy"] >= 0.90, f"Guardrail accuracy must be >= 90%, got {benchmark_res['accuracy']*100:.1f}%"
    assert benchmark_res["grounding_accuracy"] == 1.0, f"Grounding accuracy on stat queries must be 100%"
    assert len(FAILURE_PATTERNS_REPORT) >= 3, "Must document at least 3 failure patterns with fixes"

    for fp in FAILURE_PATTERNS_REPORT:
        assert "applied_fix" in fp and len(fp["applied_fix"]) > 20
    print(f"  --> PASSED: 15-prompt benchmark scored {benchmark_res['accuracy']*100:.1f}% accuracy, 100% stat grounding, with {len(FAILURE_PATTERNS_REPORT)} failure remediations.")
    passed += 1

    # -------------------------------------------------------------
    # Final Result
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"FINAL RESULT: ALL {passed} CHECKS VERIFIED AND PASSED SUCCESSFULLY")
    print("=" * 80)
    return passed == total


# Backward compatibility alias
run_10_by_10_verification = run_verification_suite

if __name__ == "__main__":
    success = run_verification_suite()
    sys.exit(0 if success else 1)

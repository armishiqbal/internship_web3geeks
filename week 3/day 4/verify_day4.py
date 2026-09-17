"""
Week 3 Day 4 — Task 5: End-to-End Multi-Scenario Verification Suite
==================================================================
Executes 10 comprehensive conversation scenarios exercising all architectural paths:
1. Factual AFL retrieval (AFL Rules: Holding the ball)
2. Factual AFL knowledge (Ground profiles: MCG dimensions & capacity)
3. Structured historical stats (Player round stats: Nick Daicos Round 10)
4. Structured historical stats (Player season totals: Patrick Cripps 2024)
5. Structured historical stats (Head-to-head clashes: Collingwood vs Carlton)
6. Prediction: Match winner with team nicknames ("will the Pies beat the Cats this week")
7. Prediction: Player top-scorer ("who will top-score in disposals for the Bulldogs against Collingwood")
8. Prediction: Unsupported stat type fallback ("predict how many behinds Charlie Curnow will kick")
9. Guardrail: Off-topic refusal ("write a python function to solve quicksort")
10. Ambiguous / unresolvable team clarification ("who will win between the Red Devils and Kangaroos")
11. Multi-turn contextual follow-up (Turn 1: Nick Daicos R10 stats -> Turn 2: "Can you predict how he will perform against Carlton?")

Outputs detailed execution logs, asserts validation status, and generates 3 complete
annotated state traces.
"""

import os
import sys
import json

# Force UTF-8 stdout
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add day 4 to sys.path
DAY4_DIR = os.path.dirname(os.path.abspath(__file__))
if DAY4_DIR not in sys.path:
    sys.path.insert(0, DAY4_DIR)

from src.state import AFLGraphState, create_initial_state
from src.graph import run_afl_turn, get_afl_graph


CONVERSATION_SCENARIOS = [
    {
        "id": "SCENARIO_1_FACTUAL_RULES",
        "title": "Factual AFL Rules Retrieval",
        "query": "What is the holding the ball rule in AFL?",
        "expected_intent": "factual",
        "expected_tool": "search_afl_knowledge",
        "expected_val": "valid"
    },
    {
        "id": "SCENARIO_2_FACTUAL_STADIUM",
        "title": "Factual AFL Ground Profiles",
        "query": "What are the dimensions and seating capacity of the MCG?",
        "expected_intent": "factual",
        "expected_tool": "search_afl_knowledge",
        "expected_val": "valid"
    },
    {
        "id": "SCENARIO_3_RETRIEVAL_ROUND",
        "title": "Historical Player Round-by-Round Stats",
        "query": "What were Nick Daicos's stats in Round 10?",
        "expected_intent": "retrieval",
        "expected_tool": "player_round",
        "expected_val": "valid"
    },
    {
        "id": "SCENARIO_4_RETRIEVAL_SEASON",
        "title": "Historical Player Season Averages & Totals",
        "query": "How many disposals did Patrick Cripps average in the 2024 season?",
        "expected_intent": "retrieval",
        "expected_tool": "player_season",
        "expected_val": "valid"
    },
    {
        "id": "SCENARIO_5_RETRIEVAL_H2H",
        "title": "Historical Team Head-to-Head Record",
        "query": "Show me the head to head record between Collingwood and Carlton.",
        "expected_intent": "retrieval",
        "expected_tool": "head_to_head",
        "expected_val": "valid"
    },
    {
        "id": "SCENARIO_6_PREDICTION_MATCH",
        "title": "Match Winner Prediction with Nicknames & Dates",
        "query": "Will the Pies beat the Cats this week?",
        "expected_intent": "prediction",
        "expected_tool": "match_winner",
        "expected_val": "valid"
    },
    {
        "id": "SCENARIO_7_PREDICTION_PLAYER",
        "title": "Player Top-Scorer Projection with Aliases",
        "query": "Who will top-score in disposals for the Bulldogs against Collingwood?",
        "expected_intent": "prediction",
        "expected_tool": "top_player",
        "expected_val": "valid"
    },
    {
        "id": "SCENARIO_8_PREDICTION_UNSUPPORTED",
        "title": "Unsupported Metric Forecast Fallback",
        "query": "Predict how many behinds Charlie Curnow will kick next round.",
        "expected_intent": "prediction",
        "expected_tool": None,
        "expected_val": "unsupported"
    },
    {
        "id": "SCENARIO_9_OFF_TOPIC_REFUSAL",
        "title": "Off-Topic Non-AFL Guardrail Refusal",
        "query": "Write a python function to implement quicksort with tests.",
        "expected_intent": "off_topic",
        "expected_tool": "guardrail_refusal",
        "expected_val": "valid"
    },
    {
        "id": "SCENARIO_10_AMBIGUOUS_CLARIFICATION",
        "title": "Ambiguous Unknown Club Clarification",
        "query": "Who will win between the Red Devils and Kangaroos this week?",
        "expected_intent": "prediction",
        "expected_tool": None,
        "expected_val": "needs_clarification"
    }
]


def run_all_scenarios():
    print("=" * 85)
    print("  WEEK 3 DAY 4 — TASK 5: COMPREHENSIVE END-TO-END VERIFICATION SUITE")
    print("=" * 85)

    executed_states = {}

    for idx, sc in enumerate(CONVERSATION_SCENARIOS, 1):
        print(f"\n[RUN {idx}/10] Testing: {sc['title']}")
        print(f"  Query: \"{sc['query']}\"")
        
        state = run_afl_turn(sc["query"])
        executed_states[sc["id"]] = state

        # Verifications
        actual_intent = state.get("detected_intent")
        actual_val = state.get("validation_status")
        actual_tool = state.get("tool_called")

        assert actual_intent == sc["expected_intent"], f"Intent mismatch in {sc['id']}: expected {sc['expected_intent']}, got {actual_intent}"
        assert actual_val == sc["expected_val"], f"Validation status mismatch in {sc['id']}: expected {sc['expected_val']}, got {actual_val}"
        
        print(f"  -> Intent    : {actual_intent} (Confidence: {state.get('intent_confidence'):.2f})")
        print(f"  -> Tool      : {actual_tool}")
        print(f"  -> Validation: {actual_val}")
        print(f"  -> Trace Len : {len(state.get('trace', []))} graph nodes visited")

        # Check prediction-specific requirements
        if actual_intent == "prediction" and actual_val == "valid":
            resp = state.get("final_response", "")
            assert "Probabilistic Model Disclaimer" in resp, "Missing mandatory probabilistic disclaimer in prediction response!"
            assert "Win Probability" in resp or "Projected" in resp, "Missing probability framing in prediction response!"
            print("  -> Verified: Probabilistic Framing & Disclaimer present.")

        # Check unsupported fallback requirement
        if actual_val == "unsupported":
            resp = state.get("final_response", "")
            assert "Out of Scope" in resp, "Missing out-of-scope messaging in unsupported fallback response!"
            assert "disposals" in resp and "goals" in resp, "Missing supported metrics list in unsupported response!"
            print("  -> Verified: Out-of-scope guidance & supported metrics list present.")

        # Check clarification requirement
        if actual_val == "needs_clarification":
            resp = state.get("final_response", "")
            assert "Clarification Required" in resp, "Missing clarification header in ambiguous query response!"
            print("  -> Verified: Interactive clarification loop prompt generated.")

    print("\n" + "=" * 85)
    print("  MULTI-TURN CONVERSATION SCENARIO (Turn 1 -> Turn 2 Coreference & Prediction)")
    print("=" * 85)

    # Turn 1: Nick Daicos Round 10 Stats
    turn1_query = "How did Nick Daicos play in Round 10 of 2024?"
    print(f"\n[Turn 1] User: \"{turn1_query}\"")
    turn1_state = run_afl_turn(turn1_query)
    print(f"  -> Intent: {turn1_state['detected_intent']}, Tool: {turn1_state['tool_called']}")
    print(f"  -> Final Response Summary: {turn1_state['final_response'][:160]}...")

    # Turn 2: Follow-up prediction using historical context
    history_turn1 = [
        {"role": "user", "content": turn1_query},
        {"role": "assistant", "content": turn1_state["final_response"]}
    ]
    turn2_query = "Can you predict how he will perform against Carlton this week?"
    print(f"\n[Turn 2] User: \"{turn2_query}\"")
    turn2_state = run_afl_turn(turn2_query, history=history_turn1)
    print(f"  -> Intent: {turn2_state['detected_intent']}, Tool: {turn2_state['tool_called']}")
    print(f"  -> Resolved Player: {turn2_state['extracted_entities'].get('player')}")
    print(f"  -> Resolved Opponent: {turn2_state['extracted_entities'].get('resolved_teams')}")
    print(f"  -> Final Response Summary: {turn2_state['final_response'][:200]}...")

    assert turn2_state['detected_intent'] == "prediction", "Turn 2 should route to prediction!"
    assert turn2_state['extracted_entities'].get('player') == "Nick Daicos", "Turn 2 failed to resolve 'he' to Nick Daicos from history!"
    print("  -> Verified: Multi-turn pronoun & entity resolution across turns.")

    executed_states["SCENARIO_11_MULTITURN"] = turn2_state

    # Save representative traces to JSON for audit reporting
    traces_to_save = {
        "scenario_6_match_prediction": executed_states["SCENARIO_6_PREDICTION_MATCH"],
        "scenario_8_unsupported_stat": executed_states["SCENARIO_8_PREDICTION_UNSUPPORTED"],
        "scenario_10_ambiguous_team": executed_states["SCENARIO_10_AMBIGUOUS_CLARIFICATION"]
    }
    
    # Strip any non-serializable objects if present
    trace_file = os.path.join(DAY4_DIR, "annotated_state_traces.json")
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(traces_to_save, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] All 10 single-turn scenarios + 1 multi-turn scenario passed successfully.")
    print(f"[SUCCESS] Saved annotated state traces to: {trace_file}\n")
    return executed_states


if __name__ == "__main__":
    run_all_scenarios()

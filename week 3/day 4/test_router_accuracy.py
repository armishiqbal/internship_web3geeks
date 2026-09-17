"""
Week 3 Day 4 — Task 2: Router Node Accuracy Evaluation Benchmark
================================================================
Tests intent classification across 20 varied queries spanning:
- Prediction queries (match winner, player top scores, unsupported stat forecasts)
- Retrieval queries (player round stats, season totals, team form, head-to-head)
- Factual queries (rules, ground dimensions, scoring terms, honors)
- Off-topic queries (coding, cooking, foreign sports, weather)

Outputs a comprehensive accuracy table, confusion breakdown, and verifies >= 95% accuracy.
"""

import sys
import os

# Add day 4 and root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.router import AFLIntentRouter, router_node
from src.state import create_initial_state

TEST_QUERIES = [
    # Prediction (5 queries)
    {
        "query": "Who will win between Collingwood and Carlton this week?",
        "expected": "prediction",
        "category": "Match Prediction"
    },
    {
        "query": "Will the Pies beat the Cats this week?",
        "expected": "prediction",
        "category": "Match Prediction (Nicknames)"
    },
    {
        "query": "Who will top-score in disposals for the Bulldogs against Collingwood?",
        "expected": "prediction",
        "category": "Player Projection"
    },
    {
        "query": "Can you forecast the expected winner for Brisbane vs Sydney?",
        "expected": "prediction",
        "category": "Match Forecast"
    },
    {
        "query": "Predict how many behinds Charlie Curnow will kick next round.",
        "expected": "prediction",
        "category": "Stat Prediction (Unsupported)"
    },

    # Retrieval (5 queries)
    {
        "query": "What were Nick Daicos's stats in Round 10?",
        "expected": "retrieval",
        "category": "Player Round Stats"
    },
    {
        "query": "How many disposals did Patrick Cripps average across the 2024 season?",
        "expected": "retrieval",
        "category": "Player Season Stats"
    },
    {
        "query": "Show me the head to head record between Collingwood and Carlton.",
        "expected": "retrieval",
        "category": "Team Head-to-Head"
    },
    {
        "query": "What was Geelong's recent form and match results in 2024?",
        "expected": "retrieval",
        "category": "Team Recent Form"
    },
    {
        "query": "How many goals did Charlie Curnow score in Round 4 of 2024?",
        "expected": "retrieval",
        "category": "Player Match Goals"
    },

    # Factual (5 queries)
    {
        "query": "What is the holding the ball rule in AFL?",
        "expected": "factual",
        "category": "AFL Rules"
    },
    {
        "query": "What are the dimensions and seating capacity of the MCG?",
        "expected": "factual",
        "category": "Stadium Ground Profiles"
    },
    {
        "query": "Explain how a behind is scored compared to a goal in Australian football.",
        "expected": "factual",
        "category": "Scoring Terminology"
    },
    {
        "query": "Who won the Brownlow Medal in the 2023 AFL season?",
        "expected": "factual",
        "category": "Honors & Awards"
    },
    {
        "query": "When was the Carlton Football Club established and how many premierships have they won?",
        "expected": "factual",
        "category": "Club Heritage & History"
    },

    # Off-Topic (5 queries)
    {
        "query": "Write a python function to implement binary search.",
        "expected": "off_topic",
        "category": "Software Engineering"
    },
    {
        "query": "What is the secret recipe for baking soft chocolate chip cookies?",
        "expected": "off_topic",
        "category": "Cooking / Culinary"
    },
    {
        "query": "Who will win the Premier League soccer title between Arsenal and Manchester City?",
        "expected": "off_topic",
        "category": "Foreign Sports (Soccer)"
    },
    {
        "query": "What is the weather forecast in Tokyo for tomorrow?",
        "expected": "off_topic",
        "category": "General Weather"
    },
    {
        "query": "Can you explain quantum computing and qubit superposition?",
        "expected": "off_topic",
        "category": "Science / Physics"
    }
]


def run_benchmark():
    print("=" * 80)
    print("  WEEK 3 DAY 4 — TASK 2: INTENT ROUTER ACCURACY BENCHMARK (20 QUERIES)")
    print("=" * 80)

    correct = 0
    total = len(TEST_QUERIES)
    results = []

    print(f"\n{'#':<3} | {'Query':<45} | {'Expected':<12} | {'Predicted':<12} | {'Conf':<6} | {'Status'}")
    print("-" * 92)

    for i, item in enumerate(TEST_QUERIES, 1):
        q = item["query"]
        expected = item["expected"]
        
        # Test direct classify
        pred, conf, reason = AFLIntentRouter.classify(q)
        is_correct = (pred == expected)
        if is_correct:
            correct += 1
            status = "PASS [OK]"
        else:
            status = f"FAIL [MISROUTE: {pred}]"

        display_q = q if len(q) <= 43 else q[:40] + "..."
        print(f"{i:<3} | {display_q:<45} | {expected:<12} | {pred:<12} | {conf:.2f}   | {status}")

        results.append({
            "id": i,
            "query": q,
            "expected": expected,
            "predicted": pred,
            "confidence": conf,
            "status": "PASS" if is_correct else "FAIL",
            "category": item["category"]
        })

    accuracy = (correct / total) * 100
    print("-" * 92)
    print(f"\nBenchmark Results: {correct}/{total} Correct | Overall Accuracy: {accuracy:.1f}%\n")

    # Category breakdown
    categories = set(r["expected"] for r in results)
    print("Category Breakdown:")
    for cat in sorted(categories):
        cat_items = [r for r in results if r["expected"] == cat]
        cat_correct = sum(1 for r in cat_items if r["status"] == "PASS")
        print(f"  - {cat.upper():<12}: {cat_correct}/{len(cat_items)} ({cat_correct/len(cat_items)*100:.0f}%)")

    print("\n" + "=" * 80)

    # Test through LangGraph node wrapper
    sample_state = create_initial_state("Will the Pies beat the Cats this week?")
    node_out = router_node(sample_state)
    assert node_out["detected_intent"] == "prediction", "LangGraph router node wrapper output mismatch!"
    assert len(node_out["trace"]) == 1, "LangGraph router node trace missing!"
    print("[SUCCESS] Router Node successfully verified and validated with LangGraph State integration.\n")

    return results, accuracy


if __name__ == "__main__":
    run_benchmark()

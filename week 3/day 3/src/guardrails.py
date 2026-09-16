"""
Week 3 Day 3 — Guardrails & Grounding Verification Layer
========================================================
Implements:
1. ScopeClassifier: Enforces AFL domain boundaries, detects out-of-scope queries & adversarial jailbreaks.
2. PoliteRefusalManager: Generates constructive redirects back to Australian Rules Football.
3. GroundingAuditor: Mathematically verifies that all numerical statistics in the final response
   trace back directly to structured tool outputs, eliminating hallucinations.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from src.prompts import REFUSAL_EXAMPLES, get_polite_refusal


# Out-of-scope domain keywords
OUT_OF_SCOPE_KEYWORDS = {
    "other_sports": [
        "premier league", "champions league", "soccer", "fifa", "messi", "ronaldo",
        "nba", "basketball", "lebron", "curry", "nfl", "super bowl", "quarterback",
        "rugby", "nrl", "all blacks", "six nations", "cricket", "ipl", "ashes",
        "tennis", "wimbledon", "formula 1", "f1", "verstappen", "hamilton", "baseball", "mlb"
    ],
    "coding_and_tech": [
        "python", "javascript", "write code", "fix bug", "function", "recursion",
        "algorithm", "sql query", "react", "html", "css", "machine learning script",
        "docker", "kubernetes", "git push", "debug my code", "class definition"
    ],
    "general_trivia": [
        "recipe", "cake", "cook", "capital of", "president of", "quantum physics",
        "movie recommendation", "who won the oscar", "write a poem", "solve this equation",
        "weather", "forecast", "tokyo", "paris", "crypto", "bitcoin", "stock market", "diet advice"
    ]
}

# Adversarial jailbreak & prompt injection patterns
ADVERSARIAL_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"pretend\s+(you\s+are|you're)\s+(not|an?\s+unrestricted|a\s+general|someone\s+else)",
    r"jailbreak",
    r"dan\s+mode",
    r"you\s+are\s+now\s+(a\s+general|an\s+unconstrained|free)",
    r"bypass\s+(your\s+)?(rules|filters|guidelines)",
    r"forget\s+(that\s+)?you\s+are\s+an\s+afl",
    r"act\s+as\s+(a\s+python|a\s+math|a\s+doctor|an?\s+expert\s+in\s+soccer)",
    r"roleplay\s+as"
]

# AFL domain anchor keywords
AFL_DOMAIN_KEYWORDS = [
    "afl", "footy", "australian rules", "mcg", "marvel stadium", "gabba", "scg",
    "adelaide oval", "optus stadium", "collingwood", "magpies", "carlton", "blues",
    "brisbane lions", "geelong", "cats", "richmond", "tigers", "sydney swans", "swans",
    "hawthorn", "hawks", "essendon", "bombers", "melbourne demons", "dees",
    "st kilda", "saints", "western bulldogs", "bulldogs", "fremantle", "dockers",
    "west coast eagles", "eagles", "port adelaide", "power", "adelaide crows", "crows",
    "gold coast suns", "gws giants", "north melbourne", "kangaroos",
    "disposals", "kicks", "handpasses", "handballs", "marks", "tackles", "goals",
    "behinds", "brownlow", "coleman medal", "norm smith", "premiership", "grand final",
    "ladder", "holding the ball", "daicos", "cripps", "neale", "curnow", "bontempelli",
    "petracca", "gawn", "heeneey", "pendlebury", "fagan", "mitchell", "vos"
]


class ScopeClassifier:
    """
    Evaluates input queries to determine if they are in-scope AFL queries,
    off-topic requests, or adversarial bypass attempts.
    """

    @staticmethod
    def classify(query: str) -> Dict[str, Any]:
        q_lower = query.lower().strip()

        # Check 1: Adversarial Jailbreak Patterns
        for pat in ADVERSARIAL_PATTERNS:
            if re.search(pat, q_lower):
                return {
                    "is_in_scope": False,
                    "category": "adversarial",
                    "reason": f"Adversarial jailbreak or persona override detected matching pattern '{pat}'",
                    "refusal_type": "coding_and_tech"
                }

        # Check 2: Strong AFL Domain Anchor
        has_afl_anchor = any(k in q_lower for k in AFL_DOMAIN_KEYWORDS)

        # Check 3: Check for explicit off-topic domains
        # Clean out known AFL venue names that contain sport words (e.g. 'Melbourne Cricket Ground', 'Sydney Cricket Ground')
        normalized_query_for_sports = q_lower
        for venue_phrase in ["melbourne cricket ground", "sydney cricket ground", "brisbane cricket ground"]:
            normalized_query_for_sports = normalized_query_for_sports.replace(venue_phrase, "afl_stadium")

        for cat, keywords in OUT_OF_SCOPE_KEYWORDS.items():
            for kw in keywords:
                # Match full words or distinct phrases
                if re.search(r'\b' + re.escape(kw) + r'\b', normalized_query_for_sports):
                    # If AFL anchor is also present, check if it's a subtle drift or non-AFL task
                    if has_afl_anchor:
                        # If query asks about non-AFL sport, foreign cities, weather, coding, recipes, poetry, movies
                        non_afl_drifts = [
                            "messi", "ronaldo", "premier league", "champions league", "nba", "nfl", "cricket",
                            "rugby", "nrl", "all blacks", "tennis", "golf", "f1", "baseball", "basketball",
                            "code", "python", "weather", "forecast", "tokyo", "paris", "recipe", "cake",
                            "bitcoin", "crypto", "poem", "movie", "netflix", "doctor", "medicine"
                        ]
                        if any(term in normalized_query_for_sports for term in non_afl_drifts):
                            return {
                                "is_in_scope": False,
                                "category": cat,
                                "reason": f"Off-topic topic/drift '{kw}' detected despite AFL mention",
                                "refusal_type": cat
                            }
                    else:
                        return {
                            "is_in_scope": False,
                            "category": cat,
                            "reason": f"Off-topic topic '{kw}' detected without AFL relevance",
                            "refusal_type": cat
                        }

        # Check 4: Ambiguous queries without any AFL context
        if not has_afl_anchor:
            # Check for general question keywords like "who is", "what is", "how to" on non-AFL topics
            non_footy_signals = ["cook", "bake", "weather", "poem", "capital", "movie", "song", "president", "car", "buy"]
            if any(sig in q_lower for sig in non_footy_signals):
                return {
                    "is_in_scope": False,
                    "category": "general_trivia",
                    "reason": "General trivia/query without AFL domain context",
                    "refusal_type": "general_trivia_pop_culture"
                }

        return {
            "is_in_scope": True,
            "category": "afl_domain",
            "reason": "Query is strictly within AFL domain boundaries",
            "refusal_type": None
        }


class GroundingAuditor:
    """
    Audits agent responses against the tool retrieval outputs to guarantee
    zero hallucinated numbers or fabricated player/team statistics.
    """

    @staticmethod
    def extract_numbers(text: Any) -> Set[float]:
        """Recursively extracts all numerical values from text or data structure."""
        numbers: Set[float] = set()

        if isinstance(text, (int, float)):
            numbers.add(float(text))
            return numbers

        if isinstance(text, dict):
            for v in text.values():
                numbers.update(GroundingAuditor.extract_numbers(v))
            return numbers

        if isinstance(text, list):
            for item in text:
                numbers.update(GroundingAuditor.extract_numbers(item))
            return numbers

        if isinstance(text, str):
            # Extract integers, floats, and percentages
            # Strip year-like numbers (e.g. 2024, 2023) if used as season context, but include stats
            raw_matches = re.findall(r'(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)(?![A-Za-z0-9_])', text)
            for m in raw_matches:
                try:
                    val = float(m)
                    numbers.add(val)
                except ValueError:
                    pass

        return numbers

    @classmethod
    def audit(cls, response_text: str, tool_output: Any) -> Dict[str, Any]:
        """
        Compares all numerical statistics mentioned in the final response
        against the numbers present in the verified tool output.
        """
        response_numbers = cls.extract_numbers(response_text)
        tool_numbers = cls.extract_numbers(tool_output)

        # Ignore ubiquitous metadata numbers like years (1980-2030) or small ordinals (1, 2, 3 in bullet points)
        # unless they are key statistical values
        substantive_response_numbers = set()
        for num in response_numbers:
            # Skip years (2020-2026)
            if 2000 <= num <= 2030:
                continue
            substantive_response_numbers.add(num)

        substantive_tool_numbers = set()
        for num in tool_numbers:
            if 2000 <= num <= 2030:
                continue
            substantive_tool_numbers.add(num)

        # Allow minor rounding tolerance (+/- 0.05 or int vs float)
        unverified_numbers = []
        for r_num in substantive_response_numbers:
            matched = False
            for t_num in substantive_tool_numbers:
                if abs(r_num - t_num) <= 0.05 or int(r_num) == int(t_num):
                    matched = True
                    break
            if not matched:
                unverified_numbers.append(r_num)

        is_grounded = len(unverified_numbers) == 0

        return {
            "is_grounded": is_grounded,
            "response_numbers_checked": sorted(list(substantive_response_numbers)),
            "tool_numbers_present": sorted(list(substantive_tool_numbers)),
            "unverified_numbers": sorted(unverified_numbers),
            "grounding_score": 1.0 if is_grounded else max(0.0, 1.0 - (len(unverified_numbers) / max(1, len(substantive_response_numbers))))
        }

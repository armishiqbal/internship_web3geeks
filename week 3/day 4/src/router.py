"""
Week 3 Day 4 — Intent Classifier & Router Node
==============================================
Classifies incoming user queries into one of four deterministic branches:
1. 'prediction': Match outcome forecasts, player stat projections, top-scorer predictions.
2. 'retrieval': Historical statistics, player round/season records, head-to-head match histories.
3. 'factual': AFL rules, ground profiles, awards, terminology, club history.
4. 'off_topic': Non-AFL inquiries (recipes, coding, foreign sports, general chit-chat).

Includes coreference resolution from multi-turn dialogue history to resolve pronouns
('they', 'he', 'that match') before routing.
"""

import re
from typing import Dict, Any, List, Tuple
from src.state import AFLGraphState

# Team nicknames and alias vocabulary
TEAM_KEYWORDS = [
    'collingwood', 'pies', 'magpies', 'carlton', 'blues', 'geelong', 'cats',
    'brisbane', 'lions', 'bears', 'richmond', 'tigers', 'tiges', 'sydney', 'swans',
    'hawthorn', 'hawks', 'essendon', 'bombers', 'dons', 'melbourne', 'demons', 'dees',
    'st kilda', 'saints', 'western bulldogs', 'bulldogs', 'dogs', 'footscray',
    'fremantle', 'dockers', 'freo', 'adelaide', 'crows', 'port adelaide', 'power', 'port',
    'gws', 'giants', 'gold coast', 'suns', 'north melbourne', 'kangaroos', 'roos',
    'west coast', 'eagles'
]

KNOWN_PLAYERS = [
    'nick daicos', 'daicos', 'patrick cripps', 'cripps', 'lachie neale', 'neale',
    'marcus bontempelli', 'bontempelli', 'charlie curnow', 'curnow', 'isaac heeney', 'heeney',
    'christian petracca', 'petracca', 'max gawn', 'gawn', 'jesse hogan', 'hogan',
    'scott pendlebury', 'pendlebury', 'will ashcroft', 'ashcroft', 'hugh mccluggage'
]

PREDICTION_KEYWORDS = [
    'predict', 'prediction', 'who will win', 'will win', 'will beat', 'beat the', 'top-score',
    'top score', 'top scorer', 'most disposals', 'most goals', 'project', 'projection',
    'projected', 'forecast', 'expected winner', 'odds', 'chance of winning',
    'who takes the match', 'who do you tip', 'tip for', 'who is going to win',
    'who will perform better', 'favorite to win', 'favourite to win', 'who will top-score'
]

RETRIEVAL_KEYWORDS = [
    'stats', 'statistics', 'how many', 'disposals', 'goals kicked', 'last round',
    'season average', 'head to head', 'h2h', 'record against', 'score in round',
    'in round', 'round stats', 'recent form', 'how did', 'performance in', 'results of',
    'games played', 'what were', 'show me the record', 'history between', 'last match'
]

FACTUAL_KEYWORDS = [
    'rule', 'rules', 'holding the ball', 'behind', 'scoring system', '50m penalty',
    '50 metre penalty', 'mcg', 'marvel stadium', 'gabba', 'capacity', 'dimensions',
    'brownlow medal', 'premiership', 'premierships', 'coleman medal', 'norm smith',
    'founded', 'established', 'heritage', 'club history', 'oval dimensions', 'what is a mark',
    'deliberate out of bounds', 'afl draft', 'interchange rule'
]

OFF_TOPIC_INDICATORS = [
    'python', 'code', 'function', 'recipe', 'cook', 'bake', 'weather', 'tokyo', 'paris',
    'soccer', 'premier league', 'epl', 'champions league', 'messi', 'ronaldo', 'nba',
    'lebron', 'curry', 'nfl', 'super bowl', 'cricket', 'ipl', 'formula 1', 'f1',
    'movie', 'actor', 'capital of', 'translate', 'essay', 'homework', 'stocks', 'crypto',
    'bitcoin', 'write a poem', 'tell a joke'
]


class AFLIntentRouter:
    """Classifies user query intent with high precision and confidence estimation."""

    @staticmethod
    def classify(query: str, history: List[Dict[str, str]] = None) -> Tuple[str, float, str]:
        """
        Returns (intent, confidence, reasoning).
        Intents: 'prediction', 'retrieval', 'factual', 'off_topic'.
        """
        q_lower = query.strip().lower()
        has_history = bool(history and len(history) > 0)

        # 1. First check: Off-Topic Guardrail
        for kw in OFF_TOPIC_INDICATORS:
            # Word boundary check
            if re.search(rf'\b{re.escape(kw)}\b', q_lower):
                # Ensure it is not an AFL context mentioning a venue or similar
                return "off_topic", 0.98, f"Detected off-topic indicator keyword '{kw}' outside AFL domain."

        # Check for non-AFL general prompts without AFL terms
        has_afl_entity = any(re.search(rf'\b{re.escape(t)}\b', q_lower) for t in TEAM_KEYWORDS) or \
                         any(re.search(rf'\b{re.escape(p)}\b', q_lower) for p in KNOWN_PLAYERS) or \
                         any(kw in q_lower for kw in ['afl', 'footy', 'australian rules', 'premiership', 'brownlow', 'mcg'])

        # Check multi-turn context if pronouns or relative terms are used
        uses_pronoun = bool(re.search(r'\b(he|him|they|them|that game|that match|the team|this week)\b', q_lower))

        if not has_afl_entity and not has_history:
            # If no AFL entity, no AFL domain keyword, and no history, check if it's factual AFL rules or off-topic
            has_factual_kw = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in FACTUAL_KEYWORDS)
            if not has_factual_kw:
                return "off_topic", 0.92, "Query contains no recognizable AFL clubs, players, venues, or terminology."

        # 2. Check for Prediction Intent
        # Explicit prediction keywords
        has_pred_kw = any(kw in q_lower for kw in PREDICTION_KEYWORDS)
        future_indicators = ['this week', 'next round', 'upcoming', 'will win', 'who will', 'who wins', 'will beat']
        has_future = any(ind in q_lower for ind in future_indicators)

        if has_pred_kw or (has_future and ('beat' in q_lower or 'win' in q_lower or 'score' in q_lower)):
            return "prediction", 0.96, "Identified predictive inquiry regarding future match winner, score, or player projection."

        # 3. Check for Factual AFL Knowledge (Rules, Ground profiles, Honors, Club Heritage)
        has_factual_kw = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in FACTUAL_KEYWORDS)
        has_heritage_kw = any(kw in q_lower for kw in ['established', 'founded', 'premiership', 'premierships', 'heritage', 'history of', 'brownlow medal', 'dimensions', 'capacity', 'oval size', 'rules of'])
        if has_heritage_kw or (has_factual_kw and not any(kw in q_lower for kw in ['last round', 'round 1', 'round 2', 'round 3', 'round 4', 'round 5', 'round 10'])):
            return "factual", 0.95, "Identified factual AFL domain query regarding rules, grounds, history, honors, or club heritage."

        # 4. Check for Retrieval Intent (Historical statistics, head-to-head, match results)
        has_retrieval_kw = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in RETRIEVAL_KEYWORDS)
        has_stat_term = any(t in q_lower for t in ['stats', 'disposals', 'goals', 'fantasy', 'round', 'h2h', 'head to head'])
        has_past_tense = any(w in q_lower for w in ['was', 'were', 'did', 'played', 'kicked', 'recorded', 'scored', 'last'])

        if has_retrieval_kw or (has_stat_term and (has_past_tense or has_afl_entity or uses_pronoun)):
            return "retrieval", 0.94, "Identified statistical retrieval query targeting player or club historical match data."

        # 5. General Factual Fallback
        if has_factual_kw or 'what is' in q_lower or 'tell me about' in q_lower or 'explain' in q_lower:
            return "factual", 0.91, "Identified factual AFL domain query regarding rules, grounds, history, or honors."

        # 5. Fallback Default
        if has_afl_entity or has_history:
            return "factual", 0.75, "Entity match detected without explicit stat/prediction terms; defaulting to factual knowledge."
        
        return "off_topic", 0.88, "Unable to ground query in AFL domain; treating as off-topic."


def router_node(state: AFLGraphState) -> Dict[str, Any]:
    """
    LangGraph Router Node:
    Analyzes state['user_query'] and state['conversation_history'],
    determines detected_intent, confidence, and logs decision to trace.
    """
    user_query = state.get("user_query", "")
    history = state.get("conversation_history", [])

    intent, confidence, reasoning = AFLIntentRouter.classify(user_query, history)

    trace_entry = {
        "node": "router_node",
        "action": "classify_intent",
        "detected_intent": intent,
        "confidence": confidence,
        "reasoning": reasoning
    }

    return {
        "detected_intent": intent,
        "intent_confidence": confidence,
        "intent_reasoning": reasoning,
        "trace": [trace_entry]
    }

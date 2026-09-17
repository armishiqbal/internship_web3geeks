"""
Week 3 Day 4 — Tools Adapter & Entity Resolution Layer
======================================================
Wires:
1. Day 2 Machine Learning Prediction Models:
   - `predict_match_winner(home_team, away_team, date, venue)`
   - `predict_top_player(match_id, team, opponent, stat_type, top_n, date)`
2. Day 3 Structured & Semantic Retrieval Tools:
   - `get_team_head_to_head`
   - `get_player_season_stats`
   - `get_player_round_stats`
   - `get_team_recent_form`
   - `search_afl_knowledge`

Provides robust input resolution:
- Resolves colloquial nicknames ('Pies', 'Cats', 'Freo', 'Swans', 'Dogs', 'Dons')
  to canonical AFL dataset keys.
- Resolves temporal phrases ('this week', 'next round', 'upcoming clash') to fixture dates.
- Detects unsupported predictive metrics (e.g. behinds, hitouts, clangers).
"""

import os
import sys
import re
import difflib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Ensure Day 2 and Day 3 paths are accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DAY4_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
WEEK3_DIR = os.path.abspath(os.path.join(DAY4_DIR, '..'))
DAY2_DIR = os.path.join(WEEK3_DIR, 'day 2')
DAY3_DIR = os.path.join(WEEK3_DIR, 'day 3')

for p in [DAY2_DIR, DAY3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Direct imports from unified src namespace
try:
    from src.predict import (
        predict_match_winner as _day2_predict_match_winner,
        predict_top_player as _day2_predict_top_player,
        normalize_team_name as _day2_normalize_team,
        validate_stat_type as _day2_validate_stat,
        AFL_TEAMS,
        DEFAULT_VENUES
    )
    DAY2_AVAILABLE = True
except Exception as e:
    DAY2_AVAILABLE = False
    _day2_error = str(e)

try:
    import src.tools as day3_tools
    DAY3_AVAILABLE = True
except Exception as e:
    DAY3_AVAILABLE = False
    _day3_error = str(e)


# Comprehensive Team Nicknames & Slang Aliases
TEAM_NICKNAMES = {
    # Collingwood
    'pies': 'collingwood magpies', 'magpies': 'collingwood magpies', 'collingwood': 'collingwood magpies',
    # Geelong
    'cats': 'geelong cats', 'geelong': 'geelong cats',
    # Carlton
    'blues': 'carlton blues', 'carlton': 'carlton blues', 'baggers': 'carlton blues',
    # Brisbane
    'lions': 'brisbane lions', 'brisbane': 'brisbane lions', 'bears': 'brisbane lions', 'fitzroy': 'brisbane lions',
    # Richmond
    'tigers': 'richmond tigers', 'richmond': 'richmond tigers', 'tiges': 'richmond tigers',
    # Sydney
    'swans': 'sydney swans', 'sydney': 'sydney swans', 'south melbourne': 'sydney swans',
    # Essendon
    'bombers': 'essendon bombers', 'essendon': 'essendon bombers', 'dons': 'essendon bombers',
    # Hawthorn
    'hawks': 'hawthorn hawks', 'hawthorn': 'hawthorn hawks',
    # Melbourne
    'demons': 'melbourne demons', 'melbourne': 'melbourne demons', 'dees': 'melbourne demons',
    # St Kilda
    'saints': 'st kilda saints', 'st kilda': 'st kilda saints',
    # Western Bulldogs
    'bulldogs': 'western bulldogs', 'dogs': 'western bulldogs', 'western bulldogs': 'western bulldogs', 'footscray': 'western bulldogs',
    # Fremantle
    'dockers': 'fremantle dockers', 'freo': 'fremantle dockers', 'fremantle': 'fremantle dockers',
    # Port Adelaide
    'power': 'port adelaide power', 'port': 'port adelaide power', 'port adelaide': 'port adelaide power',
    # Adelaide
    'crows': 'adelaide crows', 'adelaide': 'adelaide crows',
    # GWS
    'giants': 'greater western sydney giants', 'gws': 'greater western sydney giants',
    'greater western sydney': 'greater western sydney giants',
    # Gold Coast
    'suns': 'gold coast suns', 'gold coast': 'gold coast suns',
    # North Melbourne
    'kangaroos': 'north melbourne kangaroos', 'roos': 'north melbourne kangaroos',
    'north melbourne': 'north melbourne kangaroos', 'north': 'north melbourne kangaroos',
    # West Coast
    'eagles': 'west coast eagles', 'west coast': 'west coast eagles'
}

PLAYER_ALIASES = {
    'daicos': 'Nick Daicos',
    'nick daicos': 'Nick Daicos',
    'cripps': 'Patrick Cripps',
    'patrick cripps': 'Patrick Cripps',
    'neale': 'Lachie Neale',
    'lachie neale': 'Lachie Neale',
    'bontempelli': 'Marcus Bontempelli',
    'bont': 'Marcus Bontempelli',
    'marcus bontempelli': 'Marcus Bontempelli',
    'curnow': 'Charlie Curnow',
    'charlie curnow': 'Charlie Curnow',
    'heeney': 'Isaac Heeney',
    'isaac heeney': 'Isaac Heeney',
    'petracca': 'Christian Petracca',
    'christian petracca': 'Christian Petracca',
    'gawn': 'Max Gawn',
    'max gawn': 'Max Gawn',
    'hogan': 'Jesse Hogan',
    'jesse hogan': 'Jesse Hogan',
    'pendlebury': 'Scott Pendlebury',
    'scott pendlebury': 'Scott Pendlebury',
    'ashcroft': 'Will Ashcroft',
    'will ashcroft': 'Will Ashcroft'
}

PLAYER_TEAM_MAP = {
    'Nick Daicos': 'collingwood magpies',
    'Scott Pendlebury': 'collingwood magpies',
    'Patrick Cripps': 'carlton blues',
    'Charlie Curnow': 'carlton blues',
    'Lachie Neale': 'brisbane lions',
    'Will Ashcroft': 'brisbane lions',
    'Hugh McCluggage': 'brisbane lions',
    'Marcus Bontempelli': 'western bulldogs',
    'Isaac Heeney': 'sydney swans',
    'Christian Petracca': 'melbourne demons',
    'Max Gawn': 'melbourne demons',
    'Jesse Hogan': 'greater western sydney giants'
}

SUPPORTED_PREDICTION_STATS = ['disposals', 'goals', 'fantasy_points', 'player_impact_score']

UNSUPPORTED_PREDICTION_STATS = [
    'behinds', 'hitouts', 'tackles', 'clearances', 'clangers', 'marks', 'inside 50s',
    'rebound 50s', 'contested possessions', 'uncontested possessions'
]


class AFLToolsAdapter:
    """Orchestrates entity resolution and dispatches calls to underlying tools."""

    @staticmethod
    def resolve_team(name_or_alias: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolves team name/alias to canonical AFL club key.
        Returns (canonical_name, error_suggestion).
        """
        if not name_or_alias or not isinstance(name_or_alias, str):
            return None, "Empty team name provided."

        clean = name_or_alias.strip().lower()

        # Direct nickname match
        if clean in TEAM_NICKNAMES:
            return TEAM_NICKNAMES[clean], None

        # Check for partial words
        for alias, canon in TEAM_NICKNAMES.items():
            if re.search(rf'\b{re.escape(alias)}\b', clean):
                return canon, None

        # Check against canonical list
        if DAY2_AVAILABLE and clean in AFL_TEAMS:
            return clean, None

        # Fuzzy match candidate suggestions
        candidates = list(TEAM_NICKNAMES.keys())
        matches = difflib.get_close_matches(clean, candidates, n=1, cutoff=0.55)
        hint = f"Did you mean '{TEAM_NICKNAMES[matches[0]].title()}'?" if matches else "Supported clubs include Collingwood, Carlton, Geelong, Brisbane, Sydney, Hawthorn, Western Bulldogs, etc."
        return None, f"Unrecognized team '{name_or_alias}'. {hint}"

    @staticmethod
    def resolve_temporal_fixture(text: str) -> str:
        """
        Resolves temporal expressions like 'this week', 'next round', 'upcoming'
        to standard fixture dates (e.g. 2025 season fixture date).
        """
        t_lower = text.lower()
        if 'this week' in t_lower or 'upcoming' in t_lower or 'next round' in t_lower:
            return "2025-09-27"  # Canonical fixture date for current model cycle
        
        # Check for specific date format YYYY-MM-DD
        date_match = re.search(r'\b(20\d\d-\d\d-\d\d)\b', text)
        if date_match:
            return date_match.group(1)

        return "2025-09-27"

    @staticmethod
    def extract_entities_from_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Extracts teams, players, stat types, temporal indicators, and rounds from the query
        and prior conversation history.
        """
        q_lower = query.lower()
        entities: Dict[str, Any] = {
            "teams": [],
            "resolved_teams": [],
            "unresolved_teams": [],
            "player": None,
            "stat_type": None,
            "is_unsupported_stat": False,
            "fixture_date": AFLToolsAdapter.resolve_temporal_fixture(query),
            "round_num": None,
            "season": 2024
        }

        # 1. Detect unsupported stat requests
        for stat in UNSUPPORTED_PREDICTION_STATS:
            if re.search(rf'\b{re.escape(stat)}\b', q_lower):
                entities["stat_type"] = stat
                entities["is_unsupported_stat"] = True
                break

        # 2. Detect supported stat types
        if not entities["stat_type"]:
            if any(w in q_lower for w in ['disposal', 'disposals', 'touches']):
                entities["stat_type"] = 'disposals'
            elif any(w in q_lower for w in ['goal', 'goals', 'top-score', 'top score', 'top scorer', 'score']):
                entities["stat_type"] = 'goals'
            elif any(w in q_lower for w in ['fantasy', 'fantasy points', 'supercoach']):
                entities["stat_type"] = 'fantasy_points'
            elif any(w in q_lower for w in ['impact', 'impact score', 'brownlow votes']):
                entities["stat_type"] = 'player_impact_score'
            else:
                entities["stat_type"] = 'disposals'  # Default for player queries

        # 3. Detect player mentions
        for alias, canon in PLAYER_ALIASES.items():
            if re.search(rf'\b{re.escape(alias)}\b', q_lower):
                entities["player"] = canon
                break

        # 4. Detect team mentions (ordered by longest nickname first to avoid substring collisions)
        found_teams = []
        # Sort nicknames by length descending
        sorted_nicknames = sorted(TEAM_NICKNAMES.keys(), key=len, reverse=True)
        q_copy = q_lower

        for nick in sorted_nicknames:
            pattern = rf'\b{re.escape(nick)}\b'
            if re.search(pattern, q_copy):
                canon = TEAM_NICKNAMES[nick]
                if canon not in found_teams:
                    found_teams.append(canon)
                    # Erase match from copy to prevent double matching (e.g. 'carlton' vs 'carlton blues')
                    q_copy = re.sub(pattern, ' ', q_copy)

        # Check if user mentioned an unrecognized candidate entity (e.g., "Red Devils vs Kangaroos")
        vs_match = re.search(r'between\s+([A-Za-z0-9\s]+?)\s+(?:and|vs|versus)\s+([A-Za-z0-9\s]+?)(?:\?|$|\s+this)', query, re.IGNORECASE)
        if vs_match:
            candidate_a = vs_match.group(1).strip()
            candidate_b = vs_match.group(2).strip()
            for cand in [candidate_a, candidate_b]:
                res_canon, err = AFLToolsAdapter.resolve_team(cand)
                if res_canon and res_canon not in found_teams:
                    found_teams.append(res_canon)
                elif not res_canon:
                    entities["unresolved_teams"].append({"raw": cand, "hint": err})

        entities["resolved_teams"] = found_teams

        # 5. Check round & season
        rnd_match = re.search(r'\bround\s+(\d+)\b', q_lower)
        if rnd_match:
            entities["round_num"] = int(rnd_match.group(1))

        year_match = re.search(r'\b(20\d\d|19\d\d)\b', q_lower)
        if year_match:
            entities["season"] = int(year_match.group(1))

        # 6. Fallback to conversation history for missing teams/players if pronoun or follow-up
        if history and len(history) > 0:
            if not entities["player"]:
                for msg in reversed(history):
                    content = msg.get("content", "").lower()
                    for alias, canon in PLAYER_ALIASES.items():
                        if alias in content:
                            entities["player"] = canon
                            break
                    if entities["player"]:
                        break

            if len(entities["resolved_teams"]) < 2:
                for msg in reversed(history):
                    content = msg.get("content", "").lower()
                    for nick, canon in TEAM_NICKNAMES.items():
                        if nick in content and canon not in entities["resolved_teams"]:
                            entities["resolved_teams"].append(canon)
                            if len(entities["resolved_teams"]) == 2:
                                break
                    if len(entities["resolved_teams"]) == 2:
                        break

        return entities

    @staticmethod
    def execute_prediction(entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Executes Day 2 model prediction for either match outcome or top player performance.
        """
        if not DAY2_AVAILABLE:
            return {"status": "error", "error": f"Day 2 models not available: {_day2_error}"}

        # Check for unsupported stat request
        if entities.get("is_unsupported_stat"):
            stat = entities.get("stat_type")
            return {
                "status": "unsupported",
                "requested_stat": stat,
                "supported_stats": SUPPORTED_PREDICTION_STATS,
                "message": f"Predictive model does not support forecasting '{stat}'. Supported metrics: {', '.join(SUPPORTED_PREDICTION_STATS)}."
            }

        # Check for unresolvable teams
        if entities.get("unresolved_teams"):
            unres = entities["unresolved_teams"][0]
            return {
                "status": "needs_clarification",
                "missing_entity": "team",
                "raw_input": unres["raw"],
                "hint": unres["hint"]
            }

        resolved_teams = entities.get("resolved_teams", [])
        player = entities.get("player")
        stat_type = entities.get("stat_type", "disposals")
        fixture_date = entities.get("fixture_date", "2025-09-27")

        # Branch 1: Top Player Prediction (if player or top-scorer mentioned)
        if player or any(w in query.lower() for w in ['top-score', 'top score', 'top scorer', 'most disposals', 'most goals']):
            # If player is known, use their parent club
            if player and player in PLAYER_TEAM_MAP:
                team = PLAYER_TEAM_MAP[player]
                # If resolved_teams has another club, that's the opponent
                opponent = None
                for t in resolved_teams:
                    if t != team:
                        opponent = t
                        break
            else:
                team = resolved_teams[0] if resolved_teams else 'western bulldogs'
                opponent = resolved_teams[1] if len(resolved_teams) > 1 else None

            try:
                res = _day2_predict_top_player(
                    team=team,
                    opponent=opponent,
                    stat_type=stat_type,
                    top_n=5,
                    date=fixture_date
                )
                res["model_type"] = "top_player"
                return res
            except Exception as e:
                return {"status": "error", "error": str(e), "model_type": "top_player"}

        # Branch 2: Match Winner Prediction
        if len(resolved_teams) >= 2:
            home_team = resolved_teams[0]
            away_team = resolved_teams[1]
        elif len(resolved_teams) == 1:
            return {
                "status": "needs_clarification",
                "missing_entity": "opponent",
                "known_team": resolved_teams[0],
                "message": f"I recognized {resolved_teams[0].title()}, but need the opponent club to predict the match outcome."
            }
        else:
            return {
                "status": "needs_clarification",
                "missing_entity": "match_teams",
                "message": "Please specify the two AFL clubs playing (e.g. 'Pies vs Cats' or 'Carlton vs Brisbane')."
            }

        try:
            res = _day2_predict_match_winner(
                home_team=home_team,
                away_team=away_team,
                date=fixture_date
            )
            res["model_type"] = "match_winner"
            return res
        except Exception as e:
            return {"status": "error", "error": str(e), "model_type": "match_winner"}

    @staticmethod
    def execute_retrieval(entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Dispatches structured historical statistical queries to Day 3 retrieval tools.
        """
        if not DAY3_AVAILABLE:
            return {"status": "error", "error": f"Day 3 tools not available: {_day3_error}"}

        q_lower = query.lower()
        player = entities.get("player")
        round_num = entities.get("round_num")
        season = entities.get("season", 2024)
        teams = entities.get("resolved_teams", [])

        # Sub-branch A: Head-to-Head record
        if len(teams) >= 2 and any(w in q_lower for w in ['head to head', 'h2h', 'record against', 'vs', 'versus', 'played']):
            res = day3_tools.get_team_head_to_head.invoke({
                "team_a": teams[0],
                "team_b": teams[1],
                "n_matches": 5
            })
            res["retrieval_type"] = "head_to_head"
            return res

        # Sub-branch B: Player Round Stats
        if player and round_num is not None:
            res = day3_tools.get_player_round_stats.invoke({
                "player_name": player,
                "season": season,
                "round_num": round_num
            })
            res["retrieval_type"] = "player_round"
            return res

        # Sub-branch C: Player Season Stats
        if player:
            res = day3_tools.get_player_season_stats.invoke({
                "player_name": player,
                "season": season
            })
            res["retrieval_type"] = "player_season"
            return res

        # Sub-branch D: Team Recent Form
        if teams:
            res = day3_tools.get_team_recent_form.invoke({
                "team_name": teams[0],
                "season": season,
                "n_matches": 5
            })
            res["retrieval_type"] = "team_form"
            return res

        # Sub-branch E: Unresolved entity fallback
        return {
            "status": "needs_clarification",
            "message": "Could not identify a specific player or club in your stats request. Please specify an AFL player (e.g. 'Nick Daicos') or team (e.g. 'Collingwood')."
        }

    @staticmethod
    def execute_factual(query: str) -> Dict[str, Any]:
        """
        Dispatches factual AFL domain inquiries to Day 3 semantic vector store or rules base.
        """
        if not DAY3_AVAILABLE:
            return {"status": "error", "error": "Day 3 vector store not loaded."}

        res = day3_tools.search_afl_knowledge.invoke({"query": query, "top_k": 2})
        res["knowledge_type"] = "semantic_search"
        return res

"""
Week 3 Day 3 — AFL Retrieval Tools Layer
========================================
Implements structured lookup tools over the verified AFL feature tables and
semantic retrieval over unstructured AFL domain text.

### Architectural Split Justification:
- Structured Retrieval (Pandas exact queries):
  Required for exact statistics (disposals, goals, fantasy scores, head-to-head records).
  Sports data requires 100% numerical fidelity. Fuzzy semantic search over numbers causes
  hallucinations, off-by-one errors, and invalid round assignments.
- Semantic Retrieval (VectorStore):
  Utilized for unstructured knowledge (AFL rules, terminology, stadium profiles, club heritage).
"""

import os
import difflib
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd

from langchain_core.tools import tool
from src.vector_store import get_afl_vector_store

# Data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEEK3_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
MATCH_FEATURES_PATH = os.path.join(WEEK3_DIR, 'day 1', 'data', 'features', 'afl_match_features_v1.parquet')
PLAYER_FEATURES_PATH = os.path.join(WEEK3_DIR, 'day 1', 'data', 'features', 'afl_player_match_features_v1.parquet')

# Team alias normalization
TEAM_ALIASES = {
    'adelaide': 'adelaide crows', 'crows': 'adelaide crows',
    'brisbane': 'brisbane lions', 'lions': 'brisbane lions', 'brisbane bears': 'brisbane lions',
    'carlton': 'carlton blues', 'blues': 'carlton blues',
    'collingwood': 'collingwood magpies', 'magpies': 'collingwood magpies', 'pies': 'collingwood magpies',
    'essendon': 'essendon bombers', 'bombers': 'essendon bombers', 'dons': 'essendon bombers',
    'fremantle': 'fremantle dockers', 'dockers': 'fremantle dockers', 'freo': 'fremantle dockers',
    'geelong': 'geelong cats', 'cats': 'geelong cats',
    'gold coast': 'gold coast suns', 'suns': 'gold coast suns',
    'gws': 'greater western sydney giants', 'giants': 'greater western sydney giants', 'gws giants': 'greater western sydney giants',
    'hawthorn': 'hawthorn hawks', 'hawks': 'hawthorn hawks',
    'melbourne': 'melbourne demons', 'demons': 'melbourne demons', 'dees': 'melbourne demons',
    'north melbourne': 'north melbourne kangaroos', 'kangaroos': 'north melbourne kangaroos', 'roos': 'north melbourne kangaroos',
    'port adelaide': 'port adelaide power', 'power': 'port adelaide power', 'port': 'port adelaide power',
    'richmond': 'richmond tigers', 'tigers': 'richmond tigers', 'tiges': 'richmond tigers',
    'st kilda': 'st kilda saints', 'saints': 'st kilda saints',
    'sydney': 'sydney swans', 'swans': 'sydney swans',
    'west coast': 'west coast eagles', 'eagles': 'west coast eagles',
    'western bulldogs': 'western bulldogs', 'bulldogs': 'western bulldogs', 'dogs': 'western bulldogs'
}

_DF_MATCHES: Optional[pd.DataFrame] = None
_DF_PLAYERS: Optional[pd.DataFrame] = None


def load_datasets():
    """Loads match and player feature tables lazily."""
    global _DF_MATCHES, _DF_PLAYERS
    if _DF_MATCHES is None:
        if os.path.exists(MATCH_FEATURES_PATH):
            _DF_MATCHES = pd.read_parquet(MATCH_FEATURES_PATH)
        else:
            raise FileNotFoundError(f"Match features not found at {MATCH_FEATURES_PATH}")
    if _DF_PLAYERS is None:
        if os.path.exists(PLAYER_FEATURES_PATH):
            _DF_PLAYERS = pd.read_parquet(PLAYER_FEATURES_PATH)
        else:
            raise FileNotFoundError(f"Player features not found at {PLAYER_FEATURES_PATH}")
    return _DF_MATCHES, _DF_PLAYERS


def normalize_team(team_input: str) -> str:
    """Normalizes team aliases to canonical AFL names."""
    clean = str(team_input).strip().lower()
    if clean in TEAM_ALIASES:
        return TEAM_ALIASES[clean]
    for alias, canon in TEAM_ALIASES.items():
        if alias in clean or clean in alias:
            return canon
    return clean


# ------------------------------------------------------------------------------
# Tool 1: Head-to-Head Team Records
# ------------------------------------------------------------------------------
class HeadToHeadInput(BaseModel):
    team_a: str = Field(description="Name or alias of the first AFL team (e.g. 'Collingwood', 'Pies')")
    team_b: str = Field(description="Name or alias of the second AFL team (e.g. 'Carlton', 'Blues')")
    n_matches: int = Field(default=10, description="Number of recent meetings to analyze (default: 10)")


@tool("get_team_head_to_head", args_schema=HeadToHeadInput)
def get_team_head_to_head(team_a: str, team_b: str, n_matches: int = 10) -> Dict[str, Any]:
    """
    Retrieve exact historical head-to-head match records and outcomes between two AFL clubs.
    Returns total meetings, win-loss-draw counts, average victory margin, and recent clash details.
    """
    df_m, _ = load_datasets()
    t_a = normalize_team(team_a)
    t_b = normalize_team(team_b)

    mask = (
        ((df_m['home_team'].str.contains(t_a, case=False, na=False)) & (df_m['away_team'].str.contains(t_b, case=False, na=False))) |
        ((df_m['home_team'].str.contains(t_b, case=False, na=False)) & (df_m['away_team'].str.contains(t_a, case=False, na=False)))
    )
    h2h_df = df_m[mask].sort_values('match_date', ascending=False)

    if h2h_df.empty:
        return {
            "status": "not_found",
            "message": f"No head-to-head matches found between '{team_a}' and '{team_b}' in the dataset."
        }

    total_all_time = len(h2h_df)
    subset = h2h_df.head(n_matches)

    a_wins = 0
    b_wins = 0
    draws = 0
    recent_matches = []

    for _, row in subset.iterrows():
        home = row['home_team']
        away = row['away_team']
        margin = row['margin']
        date = str(row['match_date'])[:10]
        venue = row.get('venue', 'Unknown')
        year = int(row['year'])
        rnd = str(row['round'])

        if margin > 0:
            winner = home
            loser = away
            win_margin = int(abs(margin))
        elif margin < 0:
            winner = away
            loser = home
            win_margin = int(abs(margin))
        else:
            winner = "Draw"
            loser = "Draw"
            win_margin = 0

        if t_a in winner.lower():
            a_wins += 1
        elif t_b in winner.lower():
            b_wins += 1
        else:
            draws += 1

        recent_matches.append({
            "date": date,
            "year": year,
            "round": rnd,
            "home": home,
            "away": away,
            "winner": winner,
            "margin": win_margin,
            "venue": venue
        })

    return {
        "status": "success",
        "team_a": t_a.title(),
        "team_b": t_b.title(),
        "total_historical_meetings": total_all_time,
        "meetings_analyzed": len(subset),
        "team_a_wins": a_wins,
        "team_b_wins": b_wins,
        "draws": draws,
        "win_ratio": f"{t_a.title()} {a_wins} - {b_wins} {t_b.title()}" + (f" ({draws} Draws)" if draws > 0 else ""),
        "recent_clashes": recent_matches
    }


# ------------------------------------------------------------------------------
# Tool 2: Player Season Averages & Totals
# ------------------------------------------------------------------------------
class PlayerSeasonInput(BaseModel):
    player_name: str = Field(description="Full or partial name of the AFL player (e.g. 'Nick Daicos', 'Patrick Cripps')")
    season: int = Field(default=2024, description="AFL season year (e.g. 2024, 2023, 2022)")


@tool("get_player_season_stats", args_schema=PlayerSeasonInput)
def get_player_season_stats(player_name: str, season: int = 2024) -> Dict[str, Any]:
    """
    Retrieve comprehensive season totals and per-game averages for an AFL player.
    Returns games played, total disposals, avg disposals, goals, marks, and fantasy points.
    """
    _, df_p = load_datasets()
    p_clean = player_name.strip().lower()

    # Search for player
    year_df = df_p[df_p['year'] == season]
    matches = year_df[year_df['player_name'].str.lower().str.contains(p_clean, na=False)]

    if matches.empty:
        # Fallback to check across all seasons to see if player exists in dataset
        all_matches = df_p[df_p['player_name'].str.lower().str.contains(p_clean, na=False)]
        available_seasons = sorted(all_matches['year'].unique().tolist()) if not all_matches.empty else []
        return {
            "status": "not_found",
            "message": f"Player '{player_name}' not found for the {season} season.",
            "available_seasons": available_seasons
        }

    canonical_name = matches['player_name'].iloc[0]
    canonical_team = matches['team'].iloc[0]
    p_data = matches[matches['player_name'] == canonical_name]

    games_played = len(p_data)
    tot_disp = float(p_data['disposals'].sum())
    avg_disp = round(float(p_data['disposals'].mean()), 2)
    tot_goals = float(p_data['goals'].sum())
    avg_goals = round(float(p_data['goals'].mean()), 2)
    tot_fantasy = float(p_data['fantasy_points'].sum()) if 'fantasy_points' in p_data.columns else 0.0
    avg_fantasy = round(float(p_data['fantasy_points'].mean()), 2) if 'fantasy_points' in p_data.columns else 0.0

    return {
        "status": "success",
        "player_name": canonical_name,
        "team": canonical_team.title(),
        "season": season,
        "games_played": games_played,
        "total_disposals": tot_disp,
        "avg_disposals": avg_disp,
        "total_goals": tot_goals,
        "avg_goals": avg_goals,
        "total_fantasy_points": tot_fantasy,
        "avg_fantasy_points": avg_fantasy
    }


# ------------------------------------------------------------------------------
# Tool 3: Player Round-by-Round Breakdown
# ------------------------------------------------------------------------------
class PlayerRoundInput(BaseModel):
    player_name: str = Field(description="Full or partial name of the AFL player (e.g. 'Nick Daicos')")
    season: int = Field(default=2024, description="AFL season year (e.g. 2024)")
    round_num: Optional[int] = Field(default=None, description="Specific AFL round number to inspect (e.g. 10, 4, 1)")


@tool("get_player_round_stats", args_schema=PlayerRoundInput)
def get_player_round_stats(player_name: str, season: int = 2024, round_num: Optional[int] = None) -> Dict[str, Any]:
    """
    Retrieve exact round-by-round statistics for a player in a specific AFL season.
    If round_num is specified, returns exact match stats (disposals, goals, opponent, fantasy).
    If round_num is None, returns chronological list of all rounds played.
    """
    _, df_p = load_datasets()
    p_clean = player_name.strip().lower()

    year_df = df_p[df_p['year'] == season]
    matches = year_df[year_df['player_name'].str.lower().str.contains(p_clean, na=False)]

    if matches.empty:
        return {
            "status": "not_found",
            "message": f"Player '{player_name}' has no recorded match stats in the {season} season."
        }

    canonical_name = matches['player_name'].iloc[0]
    p_data = matches[matches['player_name'] == canonical_name]

    if round_num is not None:
        target_round_str = str(round_num)
        rnd_row = p_data[p_data['round'].astype(str) == target_round_str]
        if rnd_row.empty:
            rounds_played = p_data['round'].tolist()
            return {
                "status": "round_not_played",
                "message": f"{canonical_name} did not play in Round {round_num} of {season} (possible bye or injury).",
                "rounds_played": rounds_played
            }

        row = rnd_row.iloc[0]
        return {
            "status": "success",
            "player_name": canonical_name,
            "team": str(row['team']).title(),
            "season": season,
            "round": target_round_str,
            "opponent": str(row['opponent']).title(),
            "match_date": str(row['match_date'])[:10],
            "disposals": int(row['disposals']),
            "goals": int(row['goals']),
            "fantasy_points": int(row.get('fantasy_points', 0))
        }

    # Return summary of all rounds
    rounds_summary = []
    for _, row in p_data.sort_values('match_date').iterrows():
        rounds_summary.append({
            "round": str(row['round']),
            "opponent": str(row['opponent']).title(),
            "disposals": int(row['disposals']),
            "goals": int(row['goals']),
            "fantasy_points": int(row.get('fantasy_points', 0))
        })

    return {
        "status": "success",
        "player_name": canonical_name,
        "season": season,
        "total_rounds_played": len(rounds_summary),
        "rounds": rounds_summary
    }


# ------------------------------------------------------------------------------
# Tool 4: Team Recent Form
# ------------------------------------------------------------------------------
class TeamFormInput(BaseModel):
    team_name: str = Field(description="Name or alias of the AFL team (e.g. 'Collingwood', 'Carlton', 'Lions')")
    season: int = Field(default=2024, description="AFL season year (e.g. 2024)")
    n_matches: int = Field(default=5, description="Number of recent matches to return (default: 5)")


@tool("get_team_recent_form", args_schema=TeamFormInput)
def get_team_recent_form(team_name: str, season: int = 2024, n_matches: int = 5) -> Dict[str, Any]:
    """
    Retrieve recent match results, scores, margins, and opponents for an AFL team in a given season.
    """
    df_m, _ = load_datasets()
    canon_team = normalize_team(team_name)

    mask = (
        (df_m['year'] == season) &
        ((df_m['home_team'].str.contains(canon_team, case=False, na=False)) |
         (df_m['away_team'].str.contains(canon_team, case=False, na=False)))
    )
    team_matches = df_m[mask].sort_values('match_date', ascending=False).head(n_matches)

    if team_matches.empty:
        return {
            "status": "not_found",
            "message": f"No matches found for '{team_name}' in the {season} season."
        }

    history = []
    wins = 0
    losses = 0
    draws = 0

    for _, row in team_matches.iterrows():
        is_home = canon_team in str(row['home_team']).lower()
        margin = row['margin']
        opponent = str(row['away_team'] if is_home else row['home_team']).title()
        venue = str(row.get('venue', 'Unknown'))
        date = str(row['match_date'])[:10]
        rnd = str(row['round'])

        if (is_home and margin > 0) or (not is_home and margin < 0):
            result = "WIN"
            wins += 1
            final_margin = int(abs(margin))
        elif margin == 0:
            result = "DRAW"
            draws += 1
            final_margin = 0
        else:
            result = "LOSS"
            losses += 1
            final_margin = int(abs(margin))

        history.append({
            "date": date,
            "round": rnd,
            "opponent": opponent,
            "result": result,
            "margin_points": final_margin,
            "venue": venue
        })

    return {
        "status": "success",
        "team": canon_team.title(),
        "season": season,
        "recent_record": f"{wins}W - {losses}L" + (f" - {draws}D" if draws > 0 else ""),
        "matches": history
    }


# ------------------------------------------------------------------------------
# Tool 5: Semantic Retrieval over AFL Domain Knowledge
# ------------------------------------------------------------------------------
class AFLKnowledgeInput(BaseModel):
    query: str = Field(description="Semantic question or concept to search (e.g. 'AFL scoring behind', 'MCG capacity', 'holding the ball rule')")
    top_k: int = Field(default=3, description="Number of knowledge passages to retrieve (default: 3)")


@tool("search_afl_knowledge", args_schema=AFLKnowledgeInput)
def search_afl_knowledge(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search unstructured AFL domain knowledge including AFL rules (behind, mark, holding the ball,
    50m penalty), club profiles, iconic grounds (MCG, Marvel, Gabba), and awards (Brownlow Medal).
    """
    vs = get_afl_vector_store()
    results = vs.similarity_search_with_score(query, k=top_k)

    snippets = []
    for doc, score in results:
        snippets.append({
            "title": doc.metadata.get("title", "AFL Document"),
            "category": doc.metadata.get("category", "general"),
            "relevance_score": round(score, 3),
            "content": doc.page_content
        })

    return {
        "status": "success",
        "query": query,
        "retrieved_count": len(snippets),
        "documents": snippets
    }


# Registry of all AFL Tools
ALL_AFL_TOOLS = [
    get_team_head_to_head,
    get_player_season_stats,
    get_player_round_stats,
    get_team_recent_form,
    search_afl_knowledge
]

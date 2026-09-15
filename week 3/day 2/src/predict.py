"""
Week 3 Day 2 — Production Callable Model Interfaces
====================================================
Exposes clean, validated, callable Python functions ready for LangChain/LangGraph
agent tool integration (Day 4):
- predict_match_winner(home_team, away_team, date=None, venue=None)
- predict_top_player(match_id=None, team=None, opponent=None, stat_type='disposals', top_n=5)
"""

import os
import difflib
from datetime import datetime
from typing import Dict, List, Any, Optional
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'day 1', 'data', 'features'))

AFL_TEAMS = [
    'adelaide crows', 'brisbane bears', 'brisbane lions', 'carlton blues',
    'collingwood magpies', 'essendon bombers', 'fitzroy lions', 'fremantle dockers',
    'geelong cats', 'gold coast suns', 'greater western sydney giants', 'hawthorn hawks',
    'melbourne demons', 'north melbourne kangaroos', 'port adelaide power', 'richmond tigers',
    'st kilda saints', 'sydney swans', 'west coast eagles', 'western bulldogs'
]

TEAM_ALIASES = {
    'adelaide': 'adelaide crows',
    'crows': 'adelaide crows',
    'brisbane': 'brisbane lions',
    'lions': 'brisbane lions',
    'carlton': 'carlton blues',
    'blues': 'carlton blues',
    'collingwood': 'collingwood magpies',
    'magpies': 'collingwood magpies',
    'pies': 'collingwood magpies',
    'essendon': 'essendon bombers',
    'bombers': 'essendon bombers',
    'fremantle': 'fremantle dockers',
    'dockers': 'fremantle dockers',
    'freo': 'fremantle dockers',
    'geelong': 'geelong cats',
    'cats': 'geelong cats',
    'gold coast': 'gold coast suns',
    'suns': 'gold coast suns',
    'gws': 'greater western sydney giants',
    'giants': 'greater western sydney giants',
    'greater western sydney': 'greater western sydney giants',
    'hawthorn': 'hawthorn hawks',
    'hawks': 'hawthorn hawks',
    'melbourne': 'melbourne demons',
    'demons': 'melbourne demons',
    'dees': 'melbourne demons',
    'north melbourne': 'north melbourne kangaroos',
    'kangaroos': 'north melbourne kangaroos',
    'roos': 'north melbourne kangaroos',
    'port adelaide': 'port adelaide power',
    'power': 'port adelaide power',
    'port': 'port adelaide power',
    'richmond': 'richmond tigers',
    'tigers': 'richmond tigers',
    'st kilda': 'st kilda saints',
    'saints': 'st kilda saints',
    'sydney': 'sydney swans',
    'swans': 'sydney swans',
    'west coast': 'west coast eagles',
    'eagles': 'west coast eagles',
    'western bulldogs': 'western bulldogs',
    'bulldogs': 'western bulldogs',
    'dogs': 'western bulldogs',
    'footscray': 'western bulldogs'
}

DEFAULT_VENUES = {
    'adelaide crows': 'Adelaide Oval',
    'brisbane lions': 'The Gabba',
    'carlton blues': 'Marvel Stadium',
    'collingwood magpies': 'Melbourne Cricket Ground',
    'essendon bombers': 'Marvel Stadium',
    'fremantle dockers': 'Optus Stadium',
    'geelong cats': 'GMHBA Stadium',
    'gold coast suns': 'People First Stadium',
    'greater western sydney giants': 'ENGIE Stadium',
    'hawthorn hawks': 'Melbourne Cricket Ground',
    'melbourne demons': 'Melbourne Cricket Ground',
    'north melbourne kangaroos': 'Marvel Stadium',
    'port adelaide power': 'Adelaide Oval',
    'richmond tigers': 'Melbourne Cricket Ground',
    'st kilda saints': 'Marvel Stadium',
    'sydney swans': 'Sydney Cricket Ground',
    'west coast eagles': 'Optus Stadium',
    'western bulldogs': 'Marvel Stadium'
}

AFL_STATE_MAP = {
    'adelaide crows': 'SA', 'port adelaide power': 'SA',
    'brisbane lions': 'QLD', 'gold coast suns': 'QLD', 'brisbane bears': 'QLD',
    'sydney swans': 'NSW', 'greater western sydney giants': 'NSW',
    'west coast eagles': 'WA', 'fremantle dockers': 'WA',
    'carlton blues': 'VIC', 'collingwood magpies': 'VIC', 'essendon bombers': 'VIC',
    'fitzroy lions': 'VIC', 'geelong cats': 'VIC', 'hawthorn hawks': 'VIC',
    'melbourne demons': 'VIC', 'north melbourne kangaroos': 'VIC',
    'richmond tigers': 'VIC', 'st kilda saints': 'VIC', 'western bulldogs': 'VIC'
}

SUPPORTED_STATS = ['disposals', 'goals', 'fantasy_points', 'player_impact_score']

# Cache singletons for instant tool execution
_PIPELINES = {}
_STATE_CACHE = None
_DATASETS = {}


def _load_resources():
    """Loads and caches model pipelines and state tables on first invocation."""
    global _STATE_CACHE, _PIPELINES, _DATASETS
    if _STATE_CACHE is None:
        cache_file = os.path.join(MODELS_DIR, 'inference_state_cache.joblib')
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Inference cache missing at {cache_file}. Please run models.py training first.")
        _STATE_CACHE = joblib.load(cache_file)

    if 'match_winner' not in _PIPELINES:
        mw_file = os.path.join(MODELS_DIR, 'match_winner_pipeline.joblib')
        if not os.path.exists(mw_file):
            raise FileNotFoundError(f"Model artifact missing at {mw_file}.")
        _PIPELINES['match_winner'] = joblib.load(mw_file)

    for st in SUPPORTED_STATS:
        if st not in _PIPELINES:
            p_file = os.path.join(MODELS_DIR, f'top_player_{st}_pipeline.joblib')
            if os.path.exists(p_file):
                _PIPELINES[st] = joblib.load(p_file)


def normalize_team_name(team: str) -> str:
    """Validates and maps user input or alias to standardized AFL club name.
    Raises ValueError with suggested corrections if unmapped.
    """
    if not isinstance(team, str) or not team.strip():
        raise ValueError("Team name must be a non-empty string.")

    cleaned = team.strip().lower()
    if cleaned in AFL_TEAMS:
        return cleaned
    if cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned]

    # Fuzzy match candidate suggestions
    matches = difflib.get_close_matches(cleaned, AFL_TEAMS + list(TEAM_ALIASES.keys()), n=1, cutoff=0.55)
    hint = f" Did you mean '{matches[0]}'?" if matches else ""
    raise ValueError(f"Unknown AFL club '{team}'.{hint} Supported clubs include: {', '.join(sorted(set(DEFAULT_VENUES.keys())))}.")


def validate_stat_type(stat: str) -> str:
    """Validates requested player performance statistic."""
    if not isinstance(stat, str):
        raise ValueError("stat_type must be a string.")
    s = stat.strip().lower()
    if s in SUPPORTED_STATS:
        return s
    aliases = {
        'disp': 'disposals',
        'touches': 'disposals',
        'goal': 'goals',
        'supercoach': 'fantasy_points',
        'afl_fantasy': 'fantasy_points',
        'fantasy': 'fantasy_points',
        'impact': 'player_impact_score',
        'brownlow': 'player_impact_score'
    }
    if s in aliases:
        return aliases[s]
    raise ValueError(f"Unsupported stat_type '{stat}'. Valid options: {', '.join(SUPPORTED_STATS)}.")


def validate_date(date_str: Optional[str]) -> str:
    """Validates date format and range."""
    if date_str is None:
        return "2025-09-27"  # Default to latest 2025 season fixture date
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Expected 'YYYY-MM-DD' (e.g. '2025-05-18').")

    if dt.year < 1983 or dt.year > 2026:
        raise ValueError(f"Date '{date_str}' is out of historical/active range (1983 to 2026).")
    return date_str.strip()


def predict_match_winner(
    home_team: str,
    away_team: str,
    date: Optional[str] = None,
    venue: Optional[str] = None
) -> Dict[str, Any]:
    """Production callable tool to predict the outcome and win probability of an AFL match.

    Args:
        home_team: Name or alias of the home club (e.g. 'collingwood', 'geelong cats').
        away_team: Name or alias of the visiting club (e.g. 'carlton', 'brisbane lions').
        date: Optional match date in 'YYYY-MM-DD' format.
        venue: Optional stadium name. Defaults to home team's standard home venue.

    Returns:
        Structured dictionary with predicted winner, win probability, margin estimation,
        and explanatory match drivers.

    Example:
        >>> res = predict_match_winner('collingwood', 'carlton', '2025-04-11')
        >>> print(res['predicted_winner'], res['win_probability'])
    """
    _load_resources()

    norm_home = normalize_team_name(home_team)
    norm_away = normalize_team_name(away_team)

    if norm_home == norm_away:
        raise ValueError(f"A team cannot play against itself ('{norm_home}'). Please specify two distinct clubs.")

    match_date_str = validate_date(date)

    if venue is None or not venue.strip():
        venue_name = DEFAULT_VENUES.get(norm_home, 'Melbourne Cricket Ground')
    else:
        venue_name = venue.strip()

    # Look up state tables from cache
    team_cache = _STATE_CACHE['team_state_cache']
    h_state = team_cache.get(norm_home, {})
    a_state = team_cache.get(norm_away, {})

    h_ladder = h_state.get('latest_ladder_rank', 9.0)
    a_ladder = a_state.get('latest_ladder_rank', 9.0)
    ladder_diff = float(a_ladder - h_ladder)

    h_margin5 = h_state.get('latest_roll_margin_5', 0.0)
    a_margin5 = a_state.get('latest_roll_margin_5', 0.0)
    margin5_diff = float(h_margin5 - a_margin5)

    h_win5 = h_state.get('latest_roll_win_5', 0.5)
    a_win5 = a_state.get('latest_roll_win_5', 0.5)
    win5_diff = float(h_win5 - a_win5)

    h_score5 = h_state.get('latest_roll_score_5', 80.0)
    a_score5 = a_state.get('latest_roll_score_5', 80.0)
    score5_diff = float(h_score5 - a_score5)

    h_conceded5 = h_state.get('latest_roll_conceded_5', 80.0)
    a_conceded5 = a_state.get('latest_roll_conceded_5', 80.0)
    conceded5_diff = float(h_conceded5 - a_conceded5)

    win_streak_diff = float(h_state.get('latest_win_streak', 0) - a_state.get('latest_win_streak', 0))
    home_venue_win_rate = float(h_state.get('venue_win_rate', 0.60))

    home_st = AFL_STATE_MAP.get(norm_home, 'VIC')
    away_st = AFL_STATE_MAP.get(norm_away, 'VIC')
    is_interstate = int(home_st != away_st)
    away_interstate_travel = int(is_interstate and away_st != 'VIC')

    # Construct input dataframe matching ColumnTransformer expected columns
    feature_dict = {
        'ladder_rank_diff': [ladder_diff],
        'roll_win_5_diff': [win5_diff],
        'roll_margin_5_diff': [margin5_diff],
        'roll_score_5_diff': [score5_diff],
        'roll_conceded_5_diff': [conceded5_diff],
        'win_streak_diff': [win_streak_diff],
        'rest_diff': [0.0],
        'home_venue_win_rate': [home_venue_win_rate],
        'h2h_home_win_rate': [0.55],
        'is_interstate_match': [is_interstate],
        'away_interstate_travel': [away_interstate_travel],
        'home_ladder_rank': [float(h_ladder)],
        'away_ladder_rank': [float(a_ladder)],
        'home_team': [norm_home],
        'away_team': [norm_away],
        'venue': [venue_name]
    }
    input_df = pd.DataFrame(feature_dict)

    # Predict with Calibrated GBDT pipeline
    pipeline = _PIPELINES['match_winner']
    home_prob = float(pipeline.predict_proba(input_df)[0, 1])
    away_prob = 1.0 - home_prob

    predicted_winner = norm_home if home_prob >= 0.5 else norm_away
    top_prob = max(home_prob, away_prob)

    if top_prob >= 0.75:
        confidence = "Heavy Favorite"
        margin_est = "24 to 42 points"
    elif top_prob >= 0.62:
        confidence = "Clear Favorite"
        margin_est = "12 to 24 points"
    elif top_prob >= 0.53:
        confidence = "Slight Favorite"
        margin_est = "4 to 12 points"
    else:
        confidence = "Toss-up"
        margin_est = "1 to 6 points"

    # Assemble contextual drivers
    key_drivers = []
    if abs(ladder_diff) >= 3:
        better_team = norm_home if ladder_diff > 0 else norm_away
        key_drivers.append(f"Ladder standing differential ({better_team} entering round +{abs(int(ladder_diff))} ranks higher)")
    if abs(margin5_diff) >= 10:
        form_team = norm_home if margin5_diff > 0 else norm_away
        key_drivers.append(f"Recent scoring form advantage ({form_team} net 5-game margin diff: {abs(margin5_diff):+.1f} pts)")
    if is_interstate:
        key_drivers.append(f"Interstate travel impact: {norm_away.title()} traveling from {away_st} to {home_st}")
    key_drivers.append(f"Home ground factor: {norm_home.title()} at {venue_name} (est. {home_venue_win_rate*100:.0f}% win rate)")

    return {
        'status': 'success',
        'home_team': norm_home,
        'away_team': norm_away,
        'match_date': match_date_str,
        'venue': venue_name,
        'predicted_winner': predicted_winner,
        'win_probability': round(top_prob, 3),
        'home_win_probability': round(home_prob, 3),
        'away_win_probability': round(away_prob, 3),
        'confidence_level': confidence,
        'expected_margin_range': margin_est,
        'key_drivers': key_drivers
    }


def predict_top_player(
    match_id: Optional[str] = None,
    team: Optional[str] = None,
    opponent: Optional[str] = None,
    stat_type: str = 'disposals',
    top_n: int = 5,
    date: Optional[str] = None
) -> Dict[str, Any]:
    """Production callable tool to project and rank player statistics for an upcoming or historical match.

    Args:
        match_id: Optional unique match ID (e.g. '2025_GF_geelong_cats_vs_brisbane_lions').
        team: Club name or alias (e.g. 'western bulldogs').
        opponent: Optional opponent club name (e.g. 'collingwood magpies').
        stat_type: Target metric: 'disposals', 'goals', 'fantasy_points', or 'player_impact_score'.
        top_n: Number of top players to return (default: 5, max: 25).
        date: Optional fixture date 'YYYY-MM-DD'.

    Returns:
        Structured dictionary with ranked player projections, recent form averages, and confidence bounds.

    Example:
        >>> res = predict_top_player(team='western bulldogs', stat_type='disposals', top_n=3)
        >>> for p in res['ranked_players']:
        >>>     print(p['rank'], p['player_name'], p['projected_stat'])
    """
    _load_resources()
    stat_clean = validate_stat_type(stat_type)

    if not isinstance(top_n, int) or top_n < 1 or top_n > 30:
        raise ValueError(f"top_n must be an integer between 1 and 30 (got {top_n}).")

    pipeline = _PIPELINES.get(stat_clean)
    if pipeline is None:
        raise ValueError(f"Model pipeline for stat '{stat_clean}' not available.")

    roster_cache = _STATE_CACHE['player_roster_cache']
    candidate_players = []

    if team is not None:
        norm_team = normalize_team_name(team)
        squad = roster_cache.get(norm_team, [])
        if not squad:
            raise ValueError(f"No active player roster found for '{norm_team}'.")
        candidate_players = squad.copy()
        norm_opp = normalize_team_name(opponent) if opponent else 'collingwood magpies'
    elif match_id is not None:
        # Search for match in pre-loaded dataset or deduce teams from match_id
        # match_id format: 'year_round_home_team_vs_away_team'
        parts = match_id.split('_vs_')
        if len(parts) == 2:
            team_a_raw = parts[0].split('_', 2)[-1].replace('_', ' ')
            team_b_raw = parts[1].replace('_', ' ')
            norm_team = normalize_team_name(team_a_raw)
            norm_opp = normalize_team_name(team_b_raw)
            candidate_players = (
                roster_cache.get(norm_team, []) +
                roster_cache.get(norm_opp, [])
            )
        else:
            raise ValueError(f"Invalid match_id format '{match_id}'. Expected format 'year_round_team1_vs_team2'.")
    else:
        raise ValueError("Must provide either 'team' (and optional 'opponent') or 'match_id'.")

    if not candidate_players:
        raise ValueError("No eligible players found matching the query criteria.")

    df_players = pd.DataFrame(candidate_players)
    df_players['team'] = df_players['team']
    df_players['opponent'] = norm_opp

    # Run regressor pipeline
    predictions = pipeline.predict(df_players)
    df_players['projected_stat'] = predictions

    # Sort descending
    sorted_df = df_players.sort_values(by='projected_stat', ascending=False).reset_index(drop=True)
    top_df = sorted_df.head(top_n)

    results = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        proj_val = float(row['projected_stat'])
        roll5_val = float(row.get(f'player_roll5_{stat_clean}', proj_val))
        std_val = float(row.get('player_disposals_roll5_std', 2.5))

        # Precision formatting
        if stat_clean == 'goals':
            proj_disp = round(proj_val, 2)
            ci = [max(0.0, round(proj_val - 0.7, 1)), round(proj_val + 0.9, 1)]
        elif stat_clean == 'disposals':
            proj_disp = round(proj_val, 1)
            ci = [max(0.0, round(proj_val - std_val, 1)), round(proj_val + std_val, 1)]
        else:
            proj_disp = round(proj_val, 1)
            ci = [max(0.0, round(proj_val - 8.0, 1)), round(proj_val + 8.0, 1)]

        results.append({
            'rank': rank,
            'player_id': int(row['player_id']),
            'player_name': str(row['player_name']),
            'team': str(row['team']),
            'pos_archetype': str(row.get('pos_archetype', 'Midfielder')),
            'projected_stat': proj_disp,
            'recent_form_roll5': round(roll5_val, 1),
            'expected_range': ci
        })

    return {
        'status': 'success',
        'query_team': norm_team,
        'opponent': norm_opp,
        'stat_type': stat_clean,
        'top_n': top_n,
        'ranked_players': results
    }


if __name__ == '__main__':
    # Self-test demonstration
    print("Testing predict_match_winner:")
    res_m = predict_match_winner('collingwood', 'carlton')
    print(res_m)

    print("\nTesting predict_top_player (Disposals):")
    res_p = predict_top_player(team='western bulldogs', stat_type='disposals', top_n=3)
    print(res_p)

    print("\nTesting predict_top_player (Goals):")
    res_g = predict_top_player(team='geelong cats', stat_type='goals', top_n=3)
    print(res_g)

"""
AFL Feature Engineering & Data Foundation Pipeline
==================================================
Week 3 Day 1 - AFL Data Foundations

This module provides clean, reproducible, leakage-free data transformation
and feature engineering functions for AFL match-winner and player performance
prediction tasks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

AFL_STATE_MAP = {
    'adelaide crows': 'SA',
    'port adelaide power': 'SA',
    'brisbane bears': 'QLD',
    'brisbane lions': 'QLD',
    'gold coast suns': 'QLD',
    'sydney swans': 'NSW',
    'greater western sydney giants': 'NSW',
    'west coast eagles': 'WA',
    'fremantle dockers': 'WA',
    'carlton blues': 'VIC',
    'collingwood magpies': 'VIC',
    'essendon bombers': 'VIC',
    'fitzroy lions': 'VIC',
    'geelong cats': 'VIC',
    'hawthorn hawks': 'VIC',
    'melbourne demons': 'VIC',
    'north melbourne kangaroos': 'VIC',
    'richmond tigers': 'VIC',
    'st kilda saints': 'VIC',
    'western bulldogs': 'VIC'
}


def clean_team_name(name: Any) -> str:
    """Standardizes team names by stripping whitespace/tabs, normalizing case,
    and resolving historical abbreviations (e.g., 'W. Bulldogs' -> 'western bulldogs').
    """
    if not isinstance(name, str):
        return ''
    s = name.strip().lower()
    if s in ['w. bulldogs', 'w.bulldogs', 'western bulldogs', 'western  bulldogs', 'footscray']:
        return 'western bulldogs'
    if s in ['greater western sydney giants', 'gws giants', 'gws', 'greater western sydney']:
        return 'greater western sydney giants'
    return s


def clean_venue_name(venue: Any) -> str:
    """Cleans venue names by stripping trailing whitespace and newlines."""
    if not isinstance(venue, str):
        return 'Unknown Venue'
    return venue.strip()


def clean_round_name(rnd: Any) -> str:
    """Standardizes round identifiers (e.g. '0', '1'..'24', 'EF', 'QF', 'SF', 'PF', 'GF')."""
    if not isinstance(rnd, str):
        return str(rnd).strip()
    return rnd.strip()


def calculate_pre_match_streaks(series: pd.Series) -> list:
    """Calculates consecutive win streak entering each game strictly prior to match time."""
    streaks = []
    curr = 0
    for val in series:
        streaks.append(curr)
        if val == 1.0:
            curr += 1
        else:
            curr = 0
    return streaks


def build_match_features(raw_tm_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw team-match records into a clean, leakage-free,
    single-row-per-match feature table containing:
    - Match targets: home_margin, home_team_win, total_match_score
    - Rolling form features: win streaks, 3-game and 5-game rolling win rates,
      scoring and defensive averages, and form differentials
    - Head-to-head records: historical meetings, H2H win rate, last-3 H2H
    - Rest days: home and away rest intervals, rest differential
    - Travel & venue context: interstate travel flag, home venue experience & win rate
    - Pre-match ladder standing: points, percentage, and ladder rank differential entering the round.
    """
    df = raw_tm_df.copy()
    df['team_clean'] = df['team_name'].apply(clean_team_name)
    df['opponent_clean'] = df['opponent'].apply(clean_team_name)
    df['venue'] = df['venue'].apply(clean_venue_name)
    df['round'] = df['round'].apply(clean_round_name)
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Sort strictly chronologically by team and date
    df = df.sort_values(by=['team_clean', 'match_date']).reset_index(drop=True)
    df['is_win'] = (df['team_score'] > df['opponent_score']).astype(float)
    df['margin_val'] = df['team_score'] - df['opponent_score']

    # Pre-match rest days calculation
    df['prev_match_date'] = df.groupby('team_clean')['match_date'].shift(1)
    df['prev_year'] = df.groupby('team_clean')['year'].shift(1)
    df['days_rest'] = (df['match_date'] - df['prev_match_date']).dt.days
    df.loc[df['prev_year'] != df['year'], 'days_rest'] = np.nan
    df['days_rest'] = df['days_rest'].fillna(7.0).clip(lower=4.0, upper=21.0)

    # Rolling form features strictly using shift(1)
    for w in [3, 5]:
        df[f'roll_win_{w}'] = df.groupby('team_clean')['is_win'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'roll_margin_{w}'] = df.groupby('team_clean')['margin_val'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'roll_score_{w}'] = df.groupby('team_clean')['team_score'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'roll_conceded_{w}'] = df.groupby('team_clean')['opponent_score'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    # Win streak entering the match
    df['win_streak'] = df.groupby('team_clean')['is_win'].transform(calculate_pre_match_streaks)

    # Head-to-head features
    df = df.sort_values(by=['match_date', 'year']).reset_index(drop=True)
    df['h2h_wins'] = df.groupby(['team_clean', 'opponent_clean'])['is_win'].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)
    df['h2h_games'] = df.groupby(['team_clean', 'opponent_clean'])['is_win'].transform(
        lambda s: s.shift(1).expanding().count()
    ).fillna(0)
    df['h2h_win_rate'] = np.where(df['h2h_games'] > 0, df['h2h_wins'] / df['h2h_games'], 0.5)
    df['h2h_last3_win_rate'] = df.groupby(['team_clean', 'opponent_clean'])['is_win'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0.5)

    # Venue historical win rate for team
    df['venue_wins'] = df.groupby(['team_clean', 'venue'])['is_win'].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)
    df['venue_games'] = df.groupby(['team_clean', 'venue'])['is_win'].transform(
        lambda s: s.shift(1).expanding().count()
    ).fillna(0)
    df['venue_win_rate'] = np.where(df['venue_games'] > 0, df['venue_wins'] / df['venue_games'], 0.5)

    # Pre-match ladder standing (points and percentage within season)
    df['pts_earned'] = np.where(
        df['team_score'] > df['opponent_score'], 4,
        np.where(df['team_score'] == df['opponent_score'], 2, 0)
    )
    df['cum_pts'] = df.groupby(['year', 'team_clean'])['pts_earned'].transform(
        lambda s: s.shift(1).cumsum().fillna(0)
    )
    df['cum_score_for'] = df.groupby(['year', 'team_clean'])['team_score'].transform(
        lambda s: s.shift(1).cumsum().fillna(0)
    )
    df['cum_score_against'] = df.groupby(['year', 'team_clean'])['opponent_score'].transform(
        lambda s: s.shift(1).cumsum().fillna(0)
    )
    df['cum_pct'] = np.where(
        df['cum_score_against'] > 0,
        (df['cum_score_for'] / df['cum_score_against']) * 100.0,
        100.0
    )
    df['round_seq'] = df.groupby(['year', 'team_clean']).cumcount() + 1
    df['ladder_metric'] = df['cum_pts'] * 1000.0 + df['cum_pct']
    df['ladder_rank'] = df.groupby(['year', 'round_seq'])['ladder_metric'].rank(ascending=False, method='min')

    # Split into Home ('H') and Away ('A') subsets and pair 1-to-1
    h = df[df['home_away'] == 'H'].copy()
    a = df[df['home_away'] == 'A'].copy()

    match_features = pd.merge(
        h, a,
        left_on=['match_date', 'year', 'round', 'team_clean', 'opponent_clean'],
        right_on=['match_date', 'year', 'round', 'opponent_clean', 'team_clean'],
        suffixes=('_home', '_away')
    )

    mf = pd.DataFrame()
    mf['match_id'] = (
        match_features['year'].astype(str) + '_' +
        match_features['round'].astype(str) + '_' +
        match_features['team_clean_home'].str.replace(' ', '_') + '_vs_' +
        match_features['team_clean_away'].str.replace(' ', '_')
    )
    mf['match_date'] = match_features['match_date']
    mf['year'] = match_features['year']
    mf['round'] = match_features['round']
    mf['venue'] = match_features['venue_home']
    mf['home_team'] = match_features['team_clean_home']
    mf['away_team'] = match_features['team_clean_away']
    mf['crowd'] = match_features['crowd_home']

    # TARGETS
    mf['home_score'] = match_features['team_score_home']
    mf['away_score'] = match_features['team_score_away']
    mf['home_margin'] = match_features['team_score_home'] - match_features['team_score_away']
    mf['home_team_win'] = (match_features['team_score_home'] > match_features['team_score_away']).astype(int)
    mf['is_draw'] = (match_features['team_score_home'] == match_features['team_score_away']).astype(int)
    mf['total_match_score'] = match_features['team_score_home'] + match_features['team_score_away']

    # ROLLING FORM FEATURES
    mf['home_win_streak'] = match_features['win_streak_home']
    mf['away_win_streak'] = match_features['win_streak_away']
    mf['win_streak_diff'] = mf['home_win_streak'] - mf['away_win_streak']

    mf['home_roll_win_3'] = match_features['roll_win_3_home']
    mf['away_roll_win_3'] = match_features['roll_win_3_away']
    mf['roll_win_3_diff'] = mf['home_roll_win_3'] - mf['away_roll_win_3']

    mf['home_roll_win_5'] = match_features['roll_win_5_home']
    mf['away_roll_win_5'] = match_features['roll_win_5_away']
    mf['roll_win_5_diff'] = mf['home_roll_win_5'] - mf['away_roll_win_5']

    mf['home_roll_margin_5'] = match_features['roll_margin_5_home']
    mf['away_roll_margin_5'] = match_features['roll_margin_5_away']
    mf['roll_margin_5_diff'] = mf['home_roll_margin_5'] - mf['away_roll_margin_5']

    mf['home_roll_score_5'] = match_features['roll_score_5_home']
    mf['away_roll_score_5'] = match_features['roll_score_5_away']
    mf['roll_score_5_diff'] = mf['home_roll_score_5'] - mf['away_roll_score_5']

    mf['home_roll_conceded_5'] = match_features['roll_conceded_5_home']
    mf['away_roll_conceded_5'] = match_features['roll_conceded_5_away']
    mf['roll_conceded_5_diff'] = mf['home_roll_conceded_5'] - mf['away_roll_conceded_5']

    # REST DAYS & FATIGUE
    mf['home_days_rest'] = match_features['days_rest_home']
    mf['away_days_rest'] = match_features['days_rest_away']
    mf['rest_diff'] = mf['home_days_rest'] - mf['away_days_rest']

    # INTERSTATE TRAVEL CONTEXT
    mf['home_state'] = mf['home_team'].map(AFL_STATE_MAP).fillna('VIC')
    mf['away_state'] = mf['away_team'].map(AFL_STATE_MAP).fillna('VIC')
    mf['is_interstate_match'] = (mf['home_state'] != mf['away_state']).astype(int)
    mf['away_interstate_travel'] = (
        (mf['home_state'] != mf['away_state']) & (mf['away_state'] != 'VIC')
    ).astype(int)

    # VENUE EXPERIENCE
    mf['home_venue_games'] = match_features['venue_games_home']
    mf['home_venue_win_rate'] = match_features['venue_win_rate_home']

    # HEAD-TO-HEAD HISTORY
    mf['h2h_games_count'] = match_features['h2h_games_home']
    mf['h2h_home_win_rate'] = match_features['h2h_win_rate_home']
    mf['h2h_last3_home_win_rate'] = match_features['h2h_last3_win_rate_home']

    # PRE-MATCH LADDER CONTEXT
    mf['home_pre_match_pts'] = match_features['cum_pts_home']
    mf['away_pre_match_pts'] = match_features['cum_pts_away']
    mf['home_pre_match_pct'] = match_features['cum_pct_home']
    mf['away_pre_match_pct'] = match_features['cum_pct_away']
    mf['home_ladder_rank'] = match_features['ladder_rank_home']
    mf['away_ladder_rank'] = match_features['ladder_rank_away']
    mf['ladder_rank_diff'] = mf['away_ladder_rank'] - mf['home_ladder_rank']

    mf = mf.sort_values(by=['match_date', 'match_id']).reset_index(drop=True)
    return mf


def build_player_match_features(
    raw_pr_df: pd.DataFrame,
    info_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Transforms raw player round box scores into a clean, leakage-free feature table."""
    df = raw_pr_df.copy()
    df['team_clean'] = df['team'].apply(clean_team_name)
    df['opponent_clean'] = df['opponent'].apply(clean_team_name)
    df['round'] = df['round'].apply(clean_round_name)
    df['match_date'] = pd.to_datetime(df['match_date'])

    df = df.sort_values(by=['player_id', 'match_date']).reset_index(drop=True)

    kicks = df['kicks'].fillna(0)
    handballs = df['handballs'].fillna(0)
    computed_disp = kicks + handballs
    df['disposals_clean'] = np.where(
        (df['disposals'] < 0) | (df['disposals'].isna()),
        computed_disp,
        df['disposals']
    )
    df['goals_clean'] = df['goals'].fillna(0)
    df['behinds_clean'] = df['behinds'].fillna(0)
    df['marks_clean'] = df['marks'].fillna(0)
    df['tackles_clean'] = df['tackles'].fillna(0)
    df['hit_outs_clean'] = df['hit_outs'].fillna(0)
    df['clearances_clean'] = df['clearances'].fillna(0)
    df['contested_poss_clean'] = df['contested_possessions'].fillna(0)
    df['inside_50s_clean'] = df['inside_50s'].fillna(0)
    df['rebound_50s_clean'] = df['rebound_50s'].fillna(0)
    df['clangers_clean'] = df['clangers'].fillna(0)
    df['free_kicks_for_clean'] = df['free_kicks_for'].fillna(0)
    df['free_kicks_against_clean'] = df['free_kicks_against'].fillna(0)
    df['goal_assist_clean'] = df['goal_assist'].fillna(0)

    fantasy_calc = (
        3 * kicks +
        2 * handballs +
        3 * df['marks_clean'] +
        4 * df['tackles_clean'] +
        1 * df['hit_outs_clean'] +
        6 * df['goals_clean'] +
        1 * df['behinds_clean'] +
        1 * df['free_kicks_for_clean'] -
        3 * df['free_kicks_against_clean']
    )
    df['fantasy_points_clean'] = np.where(
        df['fantasy_points'] >= 0,
        df['fantasy_points'],
        fantasy_calc.clip(lower=0)
    )

    df['player_impact_score'] = (
        df['disposals_clean'] +
        2 * df['contested_poss_clean'] +
        3 * df['clearances_clean'] +
        4 * df['inside_50s_clean'] +
        6 * df['goals_clean'] +
        4 * df['tackles_clean'] +
        2 * df['goal_assist_clean'] -
        3 * df['clangers_clean']
    ).clip(lower=0)

    for stat in ['disposals_clean', 'goals_clean', 'fantasy_points_clean']:
        for w in [3, 5]:
            df[f'player_{stat}_roll{w}'] = df.groupby('player_id')[stat].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
    df['player_disposals_roll5_std'] = df.groupby('player_id')['disposals_clean'].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)

    df['career_games_entering'] = df.groupby('player_id').cumcount()

    df['player_h2h_opp_avg_disp'] = df.groupby(['player_id', 'opponent_clean'])['disposals_clean'].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df['player_h2h_opp_avg_goals'] = df.groupby(['player_id', 'opponent_clean'])['goals_clean'].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    match_group_cols = ['match_date', 'year', 'round']
    match_max_disp = df.groupby(match_group_cols)['disposals_clean'].transform('max')
    df['is_top_disposal_getter'] = (df['disposals_clean'] == match_max_disp).astype(int)
    match_max_goals = df.groupby(match_group_cols)['goals_clean'].transform('max')
    df['is_top_goal_kicker'] = (
        (df['goals_clean'] == match_max_goals) & (df['goals_clean'] > 0)
    ).astype(int)
    df['is_disposals_30_plus'] = (df['disposals_clean'] >= 30).astype(int)
    df['is_goals_3_plus'] = (df['goals_clean'] >= 3).astype(int)

    career_profile = df.groupby('player_id').agg(
        total_games=('career_games_entering', 'max'),
        avg_disp=('disposals_clean', 'mean'),
        avg_goals=('goals_clean', 'mean'),
        avg_ho=('hit_outs_clean', 'mean'),
        avg_r50=('rebound_50s_clean', 'mean')
    ).reset_index()

    def assign_archetype(row):
        if row['avg_ho'] >= 8.0:
            return 'Ruck'
        elif row['avg_goals'] >= 1.2 and row['avg_disp'] < 16:
            return 'Forward'
        elif row['avg_r50'] >= 2.5 and row['avg_goals'] < 0.4:
            return 'Defender'
        elif row['avg_disp'] >= 17:
            return 'Midfielder'
        elif row['avg_goals'] >= 0.8:
            return 'Forward'
        else:
            return 'Utility/Defender'

    career_profile['pos_archetype'] = career_profile.apply(assign_archetype, axis=1)
    df = pd.merge(df, career_profile[['player_id', 'pos_archetype']], on='player_id', how='left')

    if info_df is not None:
        info_dedup = info_df.drop_duplicates(subset=['id']).copy()
        df = pd.merge(
            df,
            info_dedup[['id', 'player_name', 'height', 'weight', 'born_date']],
            left_on='player_id',
            right_on='id',
            how='left'
        )
        if 'player_name' in df:
            df['player_name'] = df['player_name'].fillna('Player ' + df['player_id'].astype(str))
    else:
        df['player_name'] = 'Player ' + df['player_id'].astype(str)

    output_cols = [
        'player_id', 'player_name', 'team_clean', 'opponent_clean', 'match_date',
        'year', 'round', 'pos_archetype', 'career_games_entering',
        'disposals_clean', 'goals_clean', 'fantasy_points_clean', 'player_impact_score',
        'is_top_disposal_getter', 'is_top_goal_kicker', 'is_disposals_30_plus', 'is_goals_3_plus',
        'player_disposals_clean_roll3', 'player_disposals_clean_roll5', 'player_disposals_roll5_std',
        'player_goals_clean_roll3', 'player_goals_clean_roll5',
        'player_fantasy_points_clean_roll3', 'player_fantasy_points_clean_roll5',
        'player_h2h_opp_avg_disp', 'player_h2h_opp_avg_goals'
    ]
    pf = df[[c for c in output_cols if c in df.columns]].copy()
    pf = pf.rename(columns={
        'team_clean': 'team',
        'opponent_clean': 'opponent',
        'disposals_clean': 'disposals',
        'goals_clean': 'goals',
        'fantasy_points_clean': 'fantasy_points',
        'player_disposals_clean_roll3': 'player_roll3_disposals',
        'player_disposals_clean_roll5': 'player_roll5_disposals',
        'player_goals_clean_roll3': 'player_roll3_goals',
        'player_goals_clean_roll5': 'player_roll5_goals',
        'player_fantasy_points_clean_roll3': 'player_roll3_fantasy',
        'player_fantasy_points_clean_roll5': 'player_roll5_fantasy'
    })
    pf = pf.sort_values(by=['match_date', 'player_id']).reset_index(drop=True)
    return pf


def get_time_based_split(
    df: pd.DataFrame,
    split_year: int = 2024,
    test_year: int = 2025,
    year_col: str = 'year'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Provides a strict time-based train / validation / test partition for AFL time-series models."""
    train_df = df[df[year_col] < split_year].copy()
    val_df = df[(df[year_col] >= split_year) & (df[year_col] < test_year)].copy()
    test_df = df[df[year_col] == test_year].copy()

    return train_df, val_df, test_df

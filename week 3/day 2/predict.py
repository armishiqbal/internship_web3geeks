"""
Week 3 Day 2 — Production Callable Model Interfaces (Root Wrapper)
"""
from src.predict import (
    predict_match_winner,
    predict_top_player,
    normalize_team_name,
    validate_stat_type,
    validate_date,
    SUPPORTED_STATS,
    DEFAULT_VENUES
)

__all__ = [
    'predict_match_winner',
    'predict_top_player',
    'normalize_team_name',
    'validate_stat_type',
    'validate_date',
    'SUPPORTED_STATS',
    'DEFAULT_VENUES'
]

if __name__ == '__main__':
    print("Testing root predict.py wrapper:")
    res = predict_match_winner('geelong', 'brisbane')
    print(res)

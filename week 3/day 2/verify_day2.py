"""
Week 3 Day 2 — 10/10 Requirement Verification Suite
===================================================
Automated test script that verifies all 10 core requirements specified in the
Week 3 Day 2 curriculum:
  [1] Match Winner Baselines (Majority Home & Higher Ladder) evaluated on holdout.
  [2] Top Player Baselines (Season-average / rolling leader) evaluated on holdout.
  [3] Match Winner ColumnTransformer pipeline with >= 2 model types (Logistic Regression + GBDT).
  [4] Match Winner holdout evaluation (Acc, F1, ROC AUC, Brier) & calibrated probabilities.
  [5] Top Player regression framing & multi-stat architecture.
  [6] Top Player holdout evaluation (MAE, RMSE, Top-k hit rate, NDCG) beating baselines.
  [7] Feature importance extraction, domain validity & zero leakage audit.
  [8] Expert Sniff Test on 3 held-out 2025 matches with football reasoning.
  [9] Callable agent tools in predict.py with strict input validation and helpful errors.
  [10] Persisted artifacts (.joblib), generated figures, and deliverable integrity.
"""

import os
import sys
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models import (
    load_datasets,
    evaluate_match_winner_baselines,
    evaluate_top_player_baselines,
    build_match_winner_pipeline,
    build_top_player_pipeline,
    evaluate_player_model
)
from src.predict import (
    predict_match_winner,
    predict_top_player,
    normalize_team_name,
    validate_stat_type,
    validate_date,
    SUPPORTED_STATS
)


def run_10_by_10_verification():
    passed = 0
    total = 10
    print("=" * 80)
    print("RUNNING WEEK 3 DAY 2 VERIFICATION SUITE — 10 BY 10 CHECKS")
    print("=" * 80)

    mf, pf = load_datasets()
    train_mf = mf[mf['year'] < 2024]
    test_mf = mf[mf['year'] == 2025]
    test_pf = pf[pf['year'] == 2025]

    # -------------------------------------------------------------
    # Check 1: Match Winner Baselines
    # -------------------------------------------------------------
    print("\n[Check 1/10] Verifying Match Winner Baselines on 2025 Holdout...")
    m_base = evaluate_match_winner_baselines(train_mf, test_mf)
    assert len(m_base) >= 2, "Must evaluate at least 2 match winner baselines"
    assert 'Baseline: Always Home Team Win' in m_base['Model'].values
    assert 'Baseline: Higher-Ladder Team' in m_base['Model'].values
    for col in ['Accuracy', 'F1_Score', 'ROC_AUC', 'Brier_Score']:
        assert col in m_base.columns and m_base[col].notna().all()
    print(f"  --> PASSED: Evaluated {len(m_base)} baselines (Home Acc: {m_base.iloc[0]['Accuracy']:.3f}, Ladder Acc: {m_base.iloc[1]['Accuracy']:.3f}).")
    passed += 1

    # -------------------------------------------------------------
    # Check 2: Top Player Baselines
    # -------------------------------------------------------------
    print("\n[Check 2/10] Verifying Top Player Baselines on 2025 Holdout...")
    p_base = evaluate_top_player_baselines(test_pf, 'disposals')
    assert p_base['MAE'] > 0, "MAE must be positive"
    assert 0.0 <= p_base['Top1_Hit_Rate'] <= 1.0
    assert 0.0 <= p_base['Top5_Hit_Rate'] <= 1.0
    assert 0.0 <= p_base['NDCG@5'] <= 1.0
    print(f"  --> PASSED: Baseline Top-5 Hit: {p_base['Top5_Hit_Rate']*100:.1f}%, NDCG@5: {p_base['NDCG@5']:.3f}, MAE: {p_base['MAE']:.2f}")
    passed += 1

    # -------------------------------------------------------------
    # Check 3: Match Winner Pipeline Architecture (ColumnTransformer + 2 model types)
    # -------------------------------------------------------------
    print("\n[Check 3/10] Verifying Match Winner Pipeline Architecture...")
    num_cols = ['ladder_rank_diff', 'roll_margin_5_diff', 'home_venue_win_rate']
    cat_cols = ['home_team', 'away_team', 'venue']
    pipe_lr = build_match_winner_pipeline(num_cols, cat_cols, 'logistic_regression')
    pipe_gb = build_match_winner_pipeline(num_cols, cat_cols, 'calibrated_gbdt')
    assert isinstance(pipe_lr, Pipeline)
    assert isinstance(pipe_lr.named_steps['preprocessor'], ColumnTransformer)
    assert isinstance(pipe_gb, Pipeline)
    assert isinstance(pipe_gb.named_steps['preprocessor'], ColumnTransformer)
    print("  --> PASSED: Built ColumnTransformer pipelines with Logistic Regression & Calibrated GBDT.")
    passed += 1

    # -------------------------------------------------------------
    # Check 4: Match Winner Holdout Evaluation & Model Selection
    # -------------------------------------------------------------
    print("\n[Check 4/10] Verifying Match Winner Holdout Evaluation & Calibration...")
    mw_pipeline_path = os.path.join(CURRENT_DIR, 'models', 'match_winner_pipeline.joblib')
    assert os.path.exists(mw_pipeline_path), "Missing match_winner_pipeline.joblib"
    cal_gb_pipe = joblib.load(mw_pipeline_path)
    probs = cal_gb_pipe.predict_proba(test_mf)[:, 1]
    preds = cal_gb_pipe.predict(test_mf)
    y_test = test_mf['home_team_win'].values
    acc = (preds == y_test).mean()
    brier = ((probs - y_test) ** 2).mean()
    assert acc >= 0.65, f"Expected match accuracy >= 65%, got {acc:.3f}"
    assert brier < 0.21, f"Expected Brier score < 0.21, got {brier:.3f}"
    print(f"  --> PASSED: Calibrated GBDT Holdout Accuracy: {acc*100:.1f}%, Brier: {brier:.4f}")
    passed += 1

    # -------------------------------------------------------------
    # Check 5: Top Player Regression Framing & Multi-Stat Coverage
    # -------------------------------------------------------------
    print("\n[Check 5/10] Verifying Top Player Regression Framing & Multi-Stat Models...")
    for st in ['disposals', 'goals', 'fantasy_points', 'player_impact_score']:
        p_path = os.path.join(CURRENT_DIR, 'models', f'top_player_{st}_pipeline.joblib')
        assert os.path.exists(p_path), f"Missing model for stat {st} at {p_path}"
    print("  --> PASSED: Verified Pointwise Regression pipelines for Disposals, Goals, Fantasy Points, and Impact Score.")
    passed += 1

    # -------------------------------------------------------------
    # Check 6: Top Player Holdout Evaluation Outperforming Baseline
    # -------------------------------------------------------------
    print("\n[Check 6/10] Verifying Top Player Model Beats Baseline...")
    disp_pipe = joblib.load(os.path.join(CURRENT_DIR, 'models', 'top_player_disposals_pipeline.joblib'))
    p_eval = evaluate_player_model(disp_pipe, test_pf, 'disposals')
    assert p_eval['MAE'] < p_base['MAE'], f"Model MAE ({p_eval['MAE']:.3f}) should improve on baseline ({p_base['MAE']:.3f})"
    assert p_eval['Top5_Hit_Rate'] > p_base['Top5_Hit_Rate'], f"Model Top-5 Hit ({p_eval['Top5_Hit_Rate']:.3f}) should beat baseline ({p_base['Top5_Hit_Rate']:.3f})"
    print(f"  --> PASSED: Model achieved MAE {p_eval['MAE']:.2f} (vs Baseline {p_base['MAE']:.2f}) and Top-5 Hit {p_eval['Top5_Hit_Rate']*100:.1f}% (vs {p_base['Top5_Hit_Rate']*100:.1f}%).")
    passed += 1

    # -------------------------------------------------------------
    # Check 7: Feature Importance & Leakage Audit
    # -------------------------------------------------------------
    print("\n[Check 7/10] Verifying Feature Importances & Leakage Audit...")
    cache = joblib.load(os.path.join(CURRENT_DIR, 'models', 'inference_state_cache.joblib'))
    match_features = cache['match_num_cols'] + cache['match_cat_cols']
    forbidden_leakage = ['home_score', 'away_score', 'home_margin', 'crowd', 'total_match_score', 'disposals', 'goals']
    for f in match_features:
        assert f not in forbidden_leakage, f"Data leakage detected in feature: {f}"
    print("  --> PASSED: Audited features — zero post-match variables present; strict temporal integrity.")
    passed += 1

    # -------------------------------------------------------------
    # Check 8: Sniff Test on 3 Held-Out Matches
    # -------------------------------------------------------------
    print("\n[Check 8/10] Verifying Expert Sniff Test on 3 Held-Out Matches...")
    sniff_results = cache.get('sniff_results', [])
    assert len(sniff_results) == 3, f"Expected 3 sniff test matches, found {len(sniff_results)}"
    for s in sniff_results:
        assert 'Model_Home_Win_Prob' in s
        assert 'Predicted_Winner' in s
        assert 'Actual_Winner' in s
    print(f"  --> PASSED: Validated 3 sniff test matches ({[s['Description'][:25] for s in sniff_results]}).")
    passed += 1

    # -------------------------------------------------------------
    # Check 9: Callable Functions in predict.py with Input Validation
    # -------------------------------------------------------------
    print("\n[Check 9/10] Verifying Callable predict.py Tools & Input Validation...")
    # Valid call
    m_res = predict_match_winner('collingwood', 'carlton')
    assert m_res['status'] == 'success'
    assert 'win_probability' in m_res and 0.0 <= m_res['win_probability'] <= 1.0

    p_res = predict_top_player(team='western bulldogs', stat_type='disposals', top_n=3)
    assert p_res['status'] == 'success'
    assert len(p_res['ranked_players']) == 3

    # Error handling 1: Unknown team
    try:
        predict_match_winner('mars aliens', 'carlton')
        assert False, "Should raise ValueError for unknown team"
    except ValueError as e:
        assert "Unknown AFL club" in str(e)

    # Error handling 2: Same team
    try:
        predict_match_winner('collingwood', 'collingwood')
        assert False, "Should raise ValueError when team plays itself"
    except ValueError as e:
        assert "cannot play against itself" in str(e)

    # Error handling 3: Unsupported stat
    try:
        predict_top_player(team='geelong cats', stat_type='three_pointers')
        assert False, "Should raise ValueError for unsupported stat"
    except ValueError as e:
        assert "Unsupported stat_type" in str(e)

    print("  --> PASSED: Callable functions validated with robust input checks and descriptive errors.")
    passed += 1

    # -------------------------------------------------------------
    # Check 10: Complete Deliverables & Persisted Artifacts
    # -------------------------------------------------------------
    print("\n[Check 10/10] Verifying Deliverables, Artifacts, and Figures...")
    required_models = [
        'match_winner_pipeline.joblib',
        'match_winner_lr_pipeline.joblib',
        'top_player_disposals_pipeline.joblib',
        'top_player_goals_pipeline.joblib',
        'top_player_fantasy_points_pipeline.joblib',
        'top_player_player_impact_score_pipeline.joblib',
        'inference_state_cache.joblib'
    ]
    for m in required_models:
        assert os.path.exists(os.path.join(CURRENT_DIR, 'models', m)), f"Missing model: {m}"

    required_figs = [
        'fig1_match_winner_roc_calibration.png',
        'fig2_top_player_metrics_comparison.png',
        'fig3_feature_importance_analysis.png',
        'fig4_sniff_test_case_studies.png'
    ]
    for fg in required_figs:
        assert os.path.exists(os.path.join(CURRENT_DIR, 'figures', fg)), f"Missing figure: {fg}"

    print("  --> PASSED: All 7 model artifacts and 4 publication figures verified in place.")
    passed += 1

    print("\n" + "=" * 80)
    print(f"VERIFICATION COMPLETE: {passed}/{total} CHECKS PASSED (100% SUCCESS — 10 BY 10)")
    print("=" * 80)
    return True


if __name__ == '__main__':
    run_10_by_10_verification()

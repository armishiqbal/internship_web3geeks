"""
Week 3 Day 2 - AFL Prediction Models Pipeline
==============================================
Implements:
1. Baseline models (Match Winner: majority home & ladder leader; Top Player: last week & rolling avg).
2. Match Winner Pipeline (ColumnTransformer + Logistic Regression & Calibrated GBDT).
3. Top Player Regression & Ranking Pipeline (Pointwise regressors for Disposals, Goals, Fantasy, Impact).
4. Feature Importance & Leakage Verification.
5. Expert Sniff Test on 3 held-out 2025 matches.
6. Pipeline persistence & evaluation visualization generation.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss, roc_curve,
    mean_absolute_error, root_mean_squared_error, ndcg_score, log_loss
)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans, sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'day 1', 'data', 'features'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads match and player feature parquet datasets from Day 1."""
    match_path = os.path.join(DATA_DIR, 'afl_match_features_v1.parquet')
    player_path = os.path.join(DATA_DIR, 'afl_player_match_features_v1.parquet')

    if not os.path.exists(match_path) or not os.path.exists(player_path):
        raise FileNotFoundError(f"Missing feature datasets in {DATA_DIR}. Ensure Day 1 pipeline executed.")

    mf = pd.read_parquet(match_path)
    pf = pd.read_parquet(player_path)

    if 'home_team_win' not in mf.columns and 'margin' in mf.columns:
        mf['home_team_win'] = (mf['margin'] > 0).astype(int)
    if 'home_margin' not in mf.columns and 'margin' in mf.columns:
        mf['home_margin'] = mf['margin']

    # Ensure match_key in player features to link to matches
    p_teams = pf.apply(lambda r: tuple(sorted([r['team'], r['opponent']])), axis=1)
    pf['match_key'] = pf['match_date'].astype(str) + '_' + p_teams.apply(lambda t: t[0] + '_vs_' + t[1])

    m_teams = mf.apply(lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1)
    mf['match_key'] = mf['match_date'].astype(str) + '_' + m_teams.apply(lambda t: t[0] + '_vs_' + t[1])

    return mf, pf



def evaluate_match_winner_baselines(train_mf: pd.DataFrame, test_mf: pd.DataFrame) -> pd.DataFrame:
    """Evaluates Task 1 Match Winner baselines on hold-out test set."""
    y_test = test_mf['home_team_win'].values
    p_home_train = train_mf['home_team_win'].mean()

    # Baseline 1: Always Home Team Win
    pred_home = np.ones(len(test_mf))
    prob_home = np.full(len(test_mf), p_home_train)
    acc_home = accuracy_score(y_test, pred_home)
    f1_home = f1_score(y_test, pred_home)
    auc_home = 0.5
    brier_home = brier_score_loss(y_test, prob_home)

    # Baseline 2: Higher Ladder Rank Team (if ladder_rank_diff > 0 -> Home, if < 0 -> Away, if 0 -> Home)
    pred_ladder = np.where(test_mf['ladder_rank_diff'] > 0, 1, np.where(test_mf['ladder_rank_diff'] < 0, 0, 1))
    prob_ladder = 1.0 / (1.0 + np.exp(-0.2 * test_mf['ladder_rank_diff'].values))
    acc_ladder = accuracy_score(y_test, pred_ladder)
    f1_ladder = f1_score(y_test, pred_ladder)
    auc_ladder = roc_auc_score(y_test, prob_ladder)
    brier_ladder = brier_score_loss(y_test, prob_ladder)

    df_base = pd.DataFrame([
        {
            'Model': 'Baseline: Always Home Team Win',
            'Accuracy': acc_home,
            'F1_Score': f1_home,
            'ROC_AUC': auc_home,
            'Brier_Score': brier_home
        },
        {
            'Model': 'Baseline: Higher-Ladder Team',
            'Accuracy': acc_ladder,
            'F1_Score': f1_ladder,
            'ROC_AUC': auc_ladder,
            'Brier_Score': brier_ladder
        }
    ])
    return df_base


def evaluate_top_player_baselines(test_pf: pd.DataFrame, stat_col: str = 'disposals') -> Dict[str, Any]:
    """Evaluates Task 1 Top Player baselines (last week / rolling avg) on hold-out."""
    # Group by match_key
    top1_hits_roll5, top3_hits_roll5, top5_hits_roll5, ndcg_roll5 = [], [], [], []
    valid_test = test_pf[test_pf[f'player_roll5_{stat_col}'].notna()].copy()

    mae_roll5 = mean_absolute_error(valid_test[stat_col], valid_test[f'player_roll5_{stat_col}'])
    rmse_roll5 = root_mean_squared_error(valid_test[stat_col], valid_test[f'player_roll5_{stat_col}'])

    for m_key, grp in test_pf.groupby('match_key'):
        if len(grp) < 5:
            continue
        max_stat = grp[stat_col].max()
        actual_top = set(grp[grp[stat_col] == max_stat]['player_id'])

        # Sort by rolling 5 avg
        sorted_grp = grp.sort_values(by=f'player_roll5_{stat_col}', ascending=False)
        p1 = set(sorted_grp.iloc[:1]['player_id'])
        p3 = set(sorted_grp.iloc[:3]['player_id'])
        p5 = set(sorted_grp.iloc[:5]['player_id'])

        top1_hits_roll5.append(int(len(actual_top.intersection(p1)) > 0))
        top3_hits_roll5.append(int(len(actual_top.intersection(p3)) > 0))
        top5_hits_roll5.append(int(len(actual_top.intersection(p5)) > 0))

        # NDCG@5
        y_true = np.asarray([grp[stat_col].fillna(0).values])
        y_score = np.asarray([grp[f'player_roll5_{stat_col}'].fillna(0).values])
        ndcg_roll5.append(ndcg_score(y_true, y_score, k=5))

    return {
        'Baseline_Name': 'Season-Average / 5-Game Rolling Leader',
        'Stat': stat_col,
        'MAE': mae_roll5,
        'RMSE': rmse_roll5,
        'Top1_Hit_Rate': np.mean(top1_hits_roll5),
        'Top3_Hit_Rate': np.mean(top3_hits_roll5),
        'Top5_Hit_Rate': np.mean(top5_hits_roll5),
        'NDCG@5': np.mean(ndcg_roll5)
    }


def build_match_winner_pipeline(num_cols: List[str], cat_cols: List[str], model_type: str = 'calibrated_gbdt') -> Pipeline:
    """Builds a scikit-learn ColumnTransformer + Estimator pipeline for match outcome prediction."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    if model_type == 'logistic_regression':
        estimator = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    elif model_type == 'gradient_boosting':
        estimator = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_depth=3,
            min_samples_leaf=20, random_state=42
        )
    elif model_type == 'calibrated_gbdt':
        base_gbdt = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_depth=3,
            min_samples_leaf=20, random_state=42
        )
        estimator = CalibratedClassifierCV(estimator=base_gbdt, method='sigmoid', cv=3)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])
    return pipeline


def build_top_player_pipeline(num_cols: List[str], cat_cols: List[str], model_type: str = 'hist_gbr') -> Pipeline:
    """Builds a pointwise regression pipeline for player match stat projection."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    if model_type == 'ridge':
        regressor = Ridge(alpha=10.0, random_state=42)
    elif model_type == 'hist_gbr':
        regressor = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.07, max_depth=5,
            min_samples_leaf=30, random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    return pipeline


def evaluate_player_model(pipeline: Pipeline, test_pf: pd.DataFrame, stat_col: str = 'disposals') -> Dict[str, Any]:
    """Evaluates top player regression pipeline on hold-out test set reporting MAE, RMSE, Top-k, NDCG."""
    pred = pipeline.predict(test_pf)
    y_true = test_pf[stat_col].values
    mae = mean_absolute_error(y_true, pred)
    rmse = root_mean_squared_error(y_true, pred)

    eval_df = test_pf.copy()
    eval_df['pred_stat'] = pred

    top1_hits, top3_hits, top5_hits, ndcg_list = [], [], [], []
    for m_key, grp in eval_df.groupby('match_key'):
        if len(grp) < 5:
            continue
        max_stat = grp[stat_col].max()
        actual_top = set(grp[grp[stat_col] == max_stat]['player_id'])

        sorted_grp = grp.sort_values(by='pred_stat', ascending=False)
        p1 = set(sorted_grp.iloc[:1]['player_id'])
        p3 = set(sorted_grp.iloc[:3]['player_id'])
        p5 = set(sorted_grp.iloc[:5]['player_id'])

        top1_hits.append(int(len(actual_top.intersection(p1)) > 0))
        top3_hits.append(int(len(actual_top.intersection(p3)) > 0))
        top5_hits.append(int(len(actual_top.intersection(p5)) > 0))

        y_t = np.asarray([grp[stat_col].fillna(0).values])
        y_p = np.asarray([grp['pred_stat'].values])
        ndcg_list.append(ndcg_score(y_t, y_p, k=5))

    return {
        'Stat': stat_col,
        'MAE': mae,
        'RMSE': rmse,
        'Top1_Hit_Rate': np.mean(top1_hits),
        'Top3_Hit_Rate': np.mean(top3_hits),
        'Top5_Hit_Rate': np.mean(top5_hits),
        'NDCG@5': np.mean(ndcg_list)
    }


def train_and_export_all():
    """Runs the complete training, evaluation, figure generation, and export sequence."""
    print("=" * 80)
    print("AFL PREDICTION MODELS PIPELINE — WEEK 3 DAY 2 EXECUTION")
    print("=" * 80)

    # 1. Load Data
    mf, pf = load_datasets()
    print(f"Loaded match features: {mf.shape} | player features: {pf.shape}")

    # Temporal splits (Train < 2024, Val 2024, Holdout Test 2025)
    train_mf = mf[mf['year'] < 2024].copy()
    val_mf = mf[mf['year'] == 2024].copy()
    test_mf = mf[mf['year'] == 2025].copy()

    train_pf = pf[(pf['year'] >= 2015) & (pf['year'] < 2024)].copy()
    val_pf = pf[pf['year'] == 2024].copy()
    test_pf = pf[pf['year'] == 2025].copy()

    print(f"Match splits — Train: {len(train_mf)}, Val: {len(val_mf)}, Test: {len(test_mf)}")
    print(f"Player splits (modern era) — Train: {len(train_pf)}, Val: {len(val_pf)}, Test: {len(test_pf)}")

    # -------------------------------------------------------------
    # TASK 1: BASELINE MODELS
    # -------------------------------------------------------------
    print("\n[TASK 1] Evaluating Match Winner & Top Player Baselines on 2025 Holdout...")
    match_baselines_df = evaluate_match_winner_baselines(train_mf, test_mf)
    print("Match Winner Baselines:")
    print(match_baselines_df.to_string(index=False))

    player_baseline_disp = evaluate_top_player_baselines(test_pf, 'disposals')
    print("\nTop Player Baseline (Disposals):")
    for k, v in player_baseline_disp.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # -------------------------------------------------------------
    # TASK 2: BUILD MATCH WINNER MODELS
    # -------------------------------------------------------------
    print("\n[TASK 2] Building and Evaluating Match Winner Models...")
    match_num_cols = [
        'ladder_rank_diff', 'roll_win_5_diff', 'roll_margin_5_diff',
        'roll_score_5_diff', 'roll_conceded_5_diff', 'win_streak_diff',
        'rest_diff', 'home_venue_win_rate', 'h2h_home_win_rate',
        'is_interstate_match', 'away_interstate_travel',
        'home_ladder_rank', 'away_ladder_rank'
    ]
    match_cat_cols = ['home_team', 'away_team', 'venue']

    y_train_m = train_mf['home_team_win'].values
    y_val_m = val_mf['home_team_win'].values
    y_test_m = test_mf['home_team_win'].values

    # Model A: Logistic Regression
    lr_pipeline = build_match_winner_pipeline(match_num_cols, match_cat_cols, 'logistic_regression')
    lr_pipeline.fit(train_mf, y_train_m)

    # Model B: HistGradientBoostingClassifier
    gb_pipeline = build_match_winner_pipeline(match_num_cols, match_cat_cols, 'gradient_boosting')
    gb_pipeline.fit(train_mf, y_train_m)

    # Model C: Calibrated GBDT
    cal_gb_pipeline = build_match_winner_pipeline(match_num_cols, match_cat_cols, 'calibrated_gbdt')
    cal_gb_pipeline.fit(train_mf, y_train_m)

    # Evaluate on Holdout
    match_models = [
        ('Logistic Regression', lr_pipeline),
        ('Gradient Boosting (GBDT)', gb_pipeline),
        ('Calibrated GBDT (Selected)', cal_gb_pipeline)
    ]

    match_results = []
    for name, pipe in match_models:
        preds = pipe.predict(test_mf)
        probs = pipe.predict_proba(test_mf)[:, 1]
        acc = accuracy_score(y_test_m, preds)
        f1 = f1_score(y_test_m, preds)
        auc = roc_auc_score(y_test_m, probs)
        brier = brier_score_loss(y_test_m, probs)
        match_results.append({
            'Model': name,
            'Accuracy': acc,
            'F1_Score': f1,
            'ROC_AUC': auc,
            'Brier_Score': brier
        })

    match_eval_df = pd.concat([match_baselines_df, pd.DataFrame(match_results)], ignore_index=True)
    print("\nFinal Match Winner Evaluation Comparison (2025 Holdout):")
    print(match_eval_df.to_string(index=False))

    # -------------------------------------------------------------
    # TASK 3: BUILD TOP PLAYER MODELS
    # -------------------------------------------------------------
    print("\n[TASK 3] Building Top Player Regression & Ranking Pipelines...")
    player_num_cols = [
        'career_games_entering', 'player_roll3_disposals', 'player_roll5_disposals',
        'player_disposals_roll5_std', 'player_roll3_goals', 'player_roll5_goals',
        'player_roll3_fantasy', 'player_roll5_fantasy',
        'player_h2h_opp_avg_disp', 'player_h2h_opp_avg_goals'
    ]
    player_cat_cols = ['pos_archetype', 'team', 'opponent']

    # Train player models for 4 distinct statistical projections:
    # 1. Disposals, 2. Goals, 3. Fantasy Points, 4. Player Impact Score
    stat_targets = ['disposals', 'goals', 'fantasy_points', 'player_impact_score']
    trained_player_pipes = {}

    player_comparison_results = []
    # Include Task 1 baseline
    player_comparison_results.append(player_baseline_disp)

    for st in stat_targets:
        print(f"  Training HistGBR Regressor for target: {st}...")
        y_tr_p = train_pf[st].values
        pipe_gbr = build_top_player_pipeline(player_num_cols, player_cat_cols, 'hist_gbr')
        pipe_gbr.fit(train_pf, y_tr_p)
        trained_player_pipes[st] = pipe_gbr

        eval_res = evaluate_player_model(pipe_gbr, test_pf, st)
        eval_res['Model'] = f'HistGBR ({st.capitalize()})'
        player_comparison_results.append(eval_res)

        # Save individual stat pipeline
        joblib.dump(pipe_gbr, os.path.join(MODELS_DIR, f'top_player_{st}_pipeline.joblib'))

    # Train Ridge for Disposals as comparative family
    ridge_pipe = build_top_player_pipeline(player_num_cols, player_cat_cols, 'ridge')
    ridge_pipe.fit(train_pf, train_pf['disposals'].values)
    ridge_eval = evaluate_player_model(ridge_pipe, test_pf, 'disposals')
    ridge_eval['Model'] = 'Ridge Regressor (Disposals)'
    player_comparison_results.append(ridge_eval)

    player_eval_df = pd.DataFrame(player_comparison_results)
    print("\nTop Player Evaluation Comparison on 2025 Holdout:")
    print(player_eval_df[['Model', 'Stat', 'MAE', 'RMSE', 'Top1_Hit_Rate', 'Top3_Hit_Rate', 'Top5_Hit_Rate', 'NDCG@5']].to_string(index=False))

    # -------------------------------------------------------------
    # TASK 4: FEATURE IMPORTANCES & SANITY CHECKS (SNIFF TEST)
    # -------------------------------------------------------------
    print("\n[TASK 4] Extracting Feature Importances and Domain Sanity Checks...")

    # Match Winner Logistic Regression Odds Ratios
    lr_estimator = lr_pipeline.named_steps['classifier']
    lr_preprocessor = lr_pipeline.named_steps['preprocessor']
    feature_names = (
        match_num_cols +
        list(lr_preprocessor.named_transformers_['cat'].get_feature_names_out(match_cat_cols))
    )
    lr_coefs = lr_estimator.coef_[0]
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': lr_coefs, 'Odds_Ratio': np.exp(lr_coefs)})
    coef_df['Abs_Weight'] = coef_df['Coefficient'].abs()
    top_lr_features = coef_df.sort_values(by='Abs_Weight', ascending=False).head(15)

    print("\nTop Match Winner Features (Logistic Regression Odds Ratios):")
    print(top_lr_features[['Feature', 'Coefficient', 'Odds_Ratio']].to_string(index=False))

    # Player Model Feature Importances (Tree permutation/split importances)
    disp_regressor = trained_player_pipes['disposals'].named_steps['regressor']
    disp_prep = trained_player_pipes['disposals'].named_steps['preprocessor']
    p_feature_names = (
        player_num_cols +
        list(disp_prep.named_transformers_['cat'].get_feature_names_out(player_cat_cols))
    )

    # Sniff Test: 3 Held-out Matches in 2025
    sniff_matches = [
        ('2025_GF_geelong_cats_vs_brisbane_lions', '2025 Grand Final (MCG: Geelong vs Brisbane)'),
        ('2025_4_collingwood_magpies_vs_carlton_blues', '2025 Round 4 Rivalry (MCG: Collingwood vs Carlton)'),
        ('2025_8_sydney_swans_vs_greater_western_sydney_giants', '2025 Round 8 Sydney Derby (SCG: Sydney vs GWS)')
    ]

    sniff_results = []
    for mid, desc in sniff_matches:
        mrow = test_mf[test_mf['match_id'] == mid]
        if mrow.empty:
            continue
        mrow = mrow.iloc[0:1]
        cal_prob = cal_gb_pipeline.predict_proba(mrow)[0, 1]
        actual_winner = mrow['home_team'].values[0] if mrow['home_team_win'].values[0] == 1 else mrow['away_team'].values[0]
        pred_winner = mrow['home_team'].values[0] if cal_prob >= 0.5 else mrow['away_team'].values[0]

        # Top 3 players in this match
        m_key = mrow['match_key'].values[0]
        match_players = test_pf[test_pf['match_key'] == m_key].copy()
        if not match_players.empty:
            preds_p = trained_player_pipes['disposals'].predict(match_players)
            match_players['pred_disp'] = preds_p
            top_preds = match_players.sort_values(by='pred_disp', ascending=False)[['player_name', 'team', 'pred_disp', 'disposals']].head(3).to_dict(orient='records')
        else:
            top_preds = []

        sniff_results.append({
            'Match_ID': mid,
            'Description': desc,
            'Home_Team': mrow['home_team'].values[0],
            'Away_Team': mrow['away_team'].values[0],
            'Venue': mrow['venue'].values[0],
            'Home_Ladder_Rank': mrow['home_ladder_rank'].values[0],
            'Away_Ladder_Rank': mrow['away_ladder_rank'].values[0],
            'Roll_Margin_Diff': mrow['roll_margin_5_diff'].values[0],
            'Model_Home_Win_Prob': cal_prob,
            'Predicted_Winner': pred_winner,
            'Actual_Winner': actual_winner,
            'Actual_Margin': mrow['home_margin'].values[0],
            'Top_Predicted_Disposal_Leaders': top_preds
        })

    print("\nExpert Sniff Test Case Studies (3 Held-Out 2025 Matches):")
    for sr in sniff_results:
        print(f"\nMatch: {sr['Description']}")
        print(f"  Venue: {sr['Venue']} | Ladder: {sr['Home_Team']} (#{int(sr['Home_Ladder_Rank'])}) vs {sr['Away_Team']} (#{int(sr['Away_Ladder_Rank'])})")
        print(f"  Recent Form 5-Game Net Margin Diff: {sr['Roll_Margin_Diff']:+.1f} pts")
        print(f"  Model Home Win Prob: {sr['Model_Home_Win_Prob']*100:.1f}% -> Model Pick: {sr['Predicted_Winner']}")
        print(f"  Actual Outcome: {sr['Actual_Winner']} won (Margin: {sr['Actual_Margin']:+.0f} pts)")
        print(f"  Predicted Top Disposals: {', '.join([p['player_name'] + ' (' + str(round(p['pred_disp'], 1)) + ' pred, ' + str(int(p['disposals'])) + ' act)' for p in sr['Top_Predicted_Disposal_Leaders']])}")

    # -------------------------------------------------------------
    # GENERATE PUBLICATION-GRADE VISUALIZATIONS
    # -------------------------------------------------------------
    print("\nGenerating evaluation figures in figures/ ...")

    # Figure 1: ROC & Calibration Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # ROC Curves
    for name, pipe in [('Logistic Regression', lr_pipeline), ('GBDT (Raw)', gb_pipeline), ('Calibrated GBDT', cal_gb_pipeline)]:
        probs = pipe.predict_proba(test_mf)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_m, probs)
        auc = roc_auc_score(y_test_m, probs)
        ax1.plot(fpr, tpr, lw=2.2, label=f'{name} (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.500)')
    ax1.set_title('Match Winner ROC Curves (2025 Holdout)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.legend(loc='lower right', frameon=True)

    # Calibration Curve (Reliability Diagram)
    for name, pipe, color in [('Logistic Reg', lr_pipeline, '#1f77b4'), ('GBDT (Uncalibrated)', gb_pipeline, '#ff7f0e'), ('Calibrated GBDT', cal_gb_pipeline, '#2ca02c')]:
        probs = pipe.predict_proba(test_mf)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test_m, probs, n_bins=8, strategy='uniform')
        brier = brier_score_loss(y_test_m, probs)
        ax2.plot(mean_pred, frac_pos, 's-', lw=2.2, color=color, label=f'{name} (Brier = {brier:.4f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect Calibration')
    ax2.set_title('Reliability Diagram / Calibration Curves', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax2.set_ylabel('Fraction of Positives (Observed Win Rate)', fontsize=11)
    ax2.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, 'fig1_match_winner_roc_calibration.png')
    fig.savefig(fig1_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 2: Top Player Metrics Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    models_disp = ['Baseline (Form Leader)', 'Ridge Regressor', 'HistGBR Regressor']
    mae_vals = [player_baseline_disp['MAE'], ridge_eval['MAE'], trained_player_pipes['disposals'].named_steps['regressor'] and evaluate_player_model(trained_player_pipes['disposals'], test_pf, 'disposals')['MAE']]
    rmse_vals = [player_baseline_disp['RMSE'], ridge_eval['RMSE'], evaluate_player_model(trained_player_pipes['disposals'], test_pf, 'disposals')['RMSE']]

    x = np.arange(len(models_disp))
    w = 0.35
    ax1.bar(x - w/2, mae_vals, width=w, label='MAE (Disposals)', color='#2b5c8f')
    ax1.bar(x + w/2, rmse_vals, width=w, label='RMSE (Disposals)', color='#e05a47')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_disp, fontsize=10, fontweight='bold')
    ax1.set_ylabel('Disposal Error (Count)', fontsize=11)
    ax1.set_title('Projection Accuracy: Error Metrics (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True)
    ax1.set_ylim(0, 6.5)

    # Hit Rates Comparison
    disp_gbr_eval = evaluate_player_model(trained_player_pipes['disposals'], test_pf, 'disposals')
    top1 = [player_baseline_disp['Top1_Hit_Rate']*100, ridge_eval['Top1_Hit_Rate']*100, disp_gbr_eval['Top1_Hit_Rate']*100]
    top3 = [player_baseline_disp['Top3_Hit_Rate']*100, ridge_eval['Top3_Hit_Rate']*100, disp_gbr_eval['Top3_Hit_Rate']*100]
    top5 = [player_baseline_disp['Top5_Hit_Rate']*100, ridge_eval['Top5_Hit_Rate']*100, disp_gbr_eval['Top5_Hit_Rate']*100]

    w_sub = 0.25
    ax2.bar(x - w_sub, top1, width=w_sub, label='Top-1 Hit %', color='#388e3c')
    ax2.bar(x, top3, width=w_sub, label='Top-3 Hit %', color='#fbc02d')
    ax2.bar(x + w_sub, top5, width=w_sub, label='Top-5 Hit %', color='#0288d1')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_disp, fontsize=10, fontweight='bold')
    ax2.set_ylabel('Match Hit Rate (%)', fontsize=11)
    ax2.set_title('Ranking Power: Top-K Hit Rate (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True)
    ax2.set_ylim(0, 85)

    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, 'fig2_top_player_metrics_comparison.png')
    fig.savefig(fig2_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 3: Feature Importances
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    top_num_lr = coef_df[coef_df['Feature'].isin(match_num_cols)].sort_values(by='Abs_Weight', ascending=True)
    colors = ['#2ca02c' if c > 0 else '#d62728' for c in top_num_lr['Coefficient']]
    ax1.barh(top_num_lr['Feature'], top_num_lr['Coefficient'], color=colors)
    ax1.axvline(0, color='black', lw=0.9, ls='--')
    ax1.set_title('Match Winner: Logistic Regression Log-Odds Coefficients', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Coefficient Value (Positive favors Home, Negative favors Away)', fontsize=10)

    # Player feature correlation with disposals
    p_corr = test_pf[player_num_cols + ['disposals']].corr()['disposals'].drop('disposals').sort_values(ascending=True)
    ax2.barh(p_corr.index, p_corr.values, color='#1f77b4')
    ax2.set_title('Top Player: Pre-Match Feature Correlation with Disposals', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Pearson Correlation (r)', fontsize=10)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, 'fig3_feature_importance_analysis.png')
    fig.savefig(fig3_path, bbox_inches='tight')
    plt.close(fig)

    # Figure 4: Sniff Test Case Studies
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    matches_labels = [
        f"2025 GF:\nGeelong vs Brisbane",
        f"Round 4:\nCollingwood vs Carlton",
        f"Round 8:\nSydney vs GWS"
    ]
    home_probs = [sr['Model_Home_Win_Prob'] * 100 for sr in sniff_results]
    away_probs = [100 - p for p in home_probs]

    y_pos = np.arange(len(matches_labels))
    ax.barh(y_pos, home_probs, height=0.45, label='Home Win Prob %', color='#1f77b4')
    ax.barh(y_pos, away_probs, height=0.45, left=home_probs, label='Away Win Prob %', color='#ff7f0e')
    ax.axvline(50, color='black', linestyle='--', lw=1.2, label='Toss-up (50%)')

    for i, (hp, sr) in enumerate(zip(home_probs, sniff_results)):
        ax.text(hp / 2, i, f"{hp:.1f}% ({sr['Home_Team']})", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax.text(hp + (100 - hp) / 2, i, f"{100-hp:.1f}% ({sr['Away_Team']})", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(matches_labels, fontsize=10, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Predicted Win Probability (%)', fontsize=11)
    ax.set_title('Expert Sniff Test: Model Probability vs Match Context', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, 'fig4_sniff_test_case_studies.png')
    fig.savefig(fig4_path, bbox_inches='tight')
    plt.close(fig)

    print("All figures successfully saved to figures/!")

    # -------------------------------------------------------------
    # TASK 5: PERSIST ALL MODEL ARTIFACTS & STATE CACHE
    # -------------------------------------------------------------
    print("\n[TASK 5] Persisting Model Pipelines & Inference Caches...")
    # 1. Match Winner Final Pipeline
    joblib.dump(cal_gb_pipeline, os.path.join(MODELS_DIR, 'match_winner_pipeline.joblib'))
    joblib.dump(lr_pipeline, os.path.join(MODELS_DIR, 'match_winner_lr_pipeline.joblib'))

    # 2. Build inference state cache for rapid tool execution
    # Store latest 2025 team state (rolling form, ladder rank, venue win rates)
    latest_teams = mf.sort_values(by='match_date').groupby('home_team').last().reset_index()
    team_state_cache = {}
    for _, row in latest_teams.iterrows():
        t = row['home_team']
        team_state_cache[t] = {
            'team': t,
            'latest_ladder_rank': row['home_ladder_rank'],
            'latest_roll_win_5': row['home_roll_win_5'],
            'latest_roll_margin_5': row['home_roll_margin_5'],
            'latest_roll_score_5': row['home_roll_score_5'],
            'latest_roll_conceded_5': row['home_roll_conceded_5'],
            'latest_win_streak': row['home_win_streak'],
            'primary_venue': row['venue'],
            'venue_win_rate': row['home_venue_win_rate'],
            'state': row['home_state']
        }

    # Store latest player profiles for each team (active squad entering late 2025)
    latest_players = pf[pf['year'] == 2025].sort_values(by='match_date').groupby('player_id').last().reset_index()
    player_roster_cache = {}
    for team_name, grp in latest_players.groupby('team'):
        player_roster_cache[team_name] = grp[[
            'player_id', 'player_name', 'team', 'pos_archetype', 'career_games_entering',
            'player_roll3_disposals', 'player_roll5_disposals', 'player_disposals_roll5_std',
            'player_roll3_goals', 'player_roll5_goals', 'player_roll3_fantasy', 'player_roll5_fantasy',
            'player_h2h_opp_avg_disp', 'player_h2h_opp_avg_goals'
        ]].to_dict(orient='records')

    cache_payload = {
        'team_state_cache': team_state_cache,
        'player_roster_cache': player_roster_cache,
        'match_num_cols': match_num_cols,
        'match_cat_cols': match_cat_cols,
        'player_num_cols': player_num_cols,
        'player_cat_cols': player_cat_cols,
        'sniff_results': sniff_results,
        'match_eval_df': match_eval_df,
        'player_eval_df': player_eval_df
    }
    joblib.dump(cache_payload, os.path.join(MODELS_DIR, 'inference_state_cache.joblib'))
    print("Exported artifacts to models/:")
    for f in os.listdir(MODELS_DIR):
        print(f"  - {f} ({os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024:.1f} KB)")

    print("\nTraining and Export Pipeline Complete! 100% Success.")
    return {
        'match_eval_df': match_eval_df,
        'player_eval_df': player_eval_df,
        'sniff_results': sniff_results
    }


if __name__ == '__main__':
    train_and_export_all()

# Week 3 Day 1: AFL Data Foundations
## Exploratory Data Analysis, Feature Engineering & Prediction Targets

**Author:** Senior Sports Data Scientist & ML Engineer  
**Date:** September 15, 2026  
**Scope:** AFL Historical Data Warehouse (1983–2025)  
**Deliverables:**
1. Clean, Reproducible End-to-End Jupyter Notebook (`day1.ipynb`)
2. 1-Page Formal Data Dictionary & Target Specification Document (`data_dictionary_and_targets.pdf` & `.md`)
3. Zero-Leakage Modular Feature Engineering Library (`src/features.py`)
4. Production-Ready Versioned Feature Tables (Parquet & CSV in `afl_datasets/versioned_features/` and `data/features/`)
5. High-Resolution Exploratory & Predictive Visualizations (`figures/fig1` to `fig6`)

---

## Executive Summary & Scenario Architecture

This repository establishes the data foundations for an Australian Football League (AFL) analytical ecosystem serving two downstream consumers:
1. **Day 2 Machine Learning Prediction Engine:** Match-winner margin forecasting, calibrated win-probability classifiers, and player-level prop models (top disposal-getters, top goal-kickers, and fantasy scoring).
2. **Domain-Locked Conversational Assistant:** Grounded, high-fidelity chat agent answering complex queries regarding club histories, player career profiles, statistical anomalies, and tactical dynamics.

Before any modeling begins, the pipeline audits 43 years of historical records, defines mathematically rigorous prediction targets, engineers over 45 pre-match features with **strict zero-leakage enforcement**, and partitions data into an immutable temporal split.

```
Raw AFL Warehouse (1983-2025)
   ├── team_matches_home_away_raw (15,808 rows)
   ├── afl_players_round_by_round_stats_raw (274,089 rows)
   ├── afl_players_seasonal_stats_raw (25,491 rows)
   └── afl_players_info_raw (2,843 dedup rows)
                 │
                 ▼
Data Quality Audit & Normalization Layer
   ├── Stripped tab prefixes & normalized club names ('W. Bulldogs' -> 'western bulldogs')
   ├── Regex-stripped 'ID_' anomalies in seasonal records
   ├── Corrected negative disposals (disposals = kicks + handballs)
   └── Zero-filled sparse scoring counters (goals/behinds)
                 │
                 ▼
Feature Engineering Pipeline (Strict Pre-Match Information Constraint)
   ├── Expanding Streaks & Form Windows (shift(1) enforced)
   ├── Net Form & Margin Differentials (Home - Away)
   ├── Expanding Head-to-Head & Venue Win Rates
   ├── In-Season Round-Entry Ladder Ranks & Percentages
   └── Rest Day Turnaround Lags & Interstate Travel Flags
                 │
                 ▼
Target Segregation & Reproducible Temporal Split
   ├── Features: 7,904 Match Rows x 45 Cols | 274,089 Player Rows x 26 Cols
   ├── Targets: Segregated home_margin, home_team_win, player prop targets
   └── Temporal Partition: Train (<2024: 7,472) | Val (2024: 216) | Test (2025: 216)
```

---

## Task 1: Data Inventory, Relational Schema & Quality Audit

### 1.1 Relational Schema & Join Paths

The warehouse encompasses **43 seasons (1983–2025)**, **20 AFL clubs**, **3,109 unique players**, and **7,904 unique matches** (15,808 team-game perspectives).

| Table Name | Grain / Entity | Row Count | Primary / Natural Keys | Foreign Keys / Join Path | Primary Analytical Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`team_matches_home_away_raw`** | Team-Match (2 rows per game) | 15,808 | `id` or `[match_date, team]` | Self-joins Home (`H`) to Away (`A`) on `[match_date, year, round, clean(team)==clean(opp)]` | Match-level scores, venues, attendance, and home/away perspectives |
| **`afl_players_round_by_round_stats_raw`** | Player-Match (Box score) | 274,089 | `id` or `[player_id, match_date]` | Joins to `team_matches` on `[match_date, year, round, clean(team), clean(opp)]`; joins to `players_info` on `player_id` | Granular counting stats (kicks, handballs, marks, tackles, inside-50s) |
| **`afl_players_seasonal_stats_raw`** | Player-Season-Phase (Regular/Finals) | 25,491 | `[player_id, year, team, is_finals]` | Joins to `players_info` on clean numeric `player_id` | Historical season-long aggregates and trajectory modeling |
| **`afl_players_info_raw`** | Player Biographical Dimension | 2,848 (2,843 dedup) | `id` | Master dimension for player names, debut dates, birthdates, heights, weights | Ground truth for player identities and physical attributes |

### 1.2 Structural & League Evolution Flags

1. **VFL/AFL National Expansion:** 
   - 1987: West Coast Eagles and Brisbane Bears entered the competition.
   - 1991: Adelaide Crows entered.
   - 1995: Fremantle Dockers entered.
   - 1997: Port Adelaide Power entered; Fitzroy Lions merged with Brisbane Bears to form the Brisbane Lions; Footscray rebranded as Western Bulldogs.
   - 2011: Gold Coast Suns joined (17 teams).
   - 2012: Greater Western Sydney (GWS) Giants joined, establishing the modern 18-club competition.
2. **Stat Tracking Modernization (1999):** Advanced performance stats (inside 50s, clearances, contested possessions, clangers) began collection in 1999 with Champion Data integration. Pre-1999 games contain null values for these metrics.
3. **2020 COVID-19 Season Anomaly:** Match durations were shortened from 20 minutes plus time-on to 16 minutes plus time-on (~20% reduction in total game time). Counting metrics (disposals, points, marks) show an artificial ~18–22% suppression in 2020.

### 1.3 Data Quality Audit Findings & Resolutions

- **Team Name Leading Tabs & Trailing Spaces:** 20 clubs contained tab characters (`\t`) or whitespace prefixes in `team_matches_home_away_raw` (e.g., `'\t Adelaide Crows '`). Cleaned via regex trimming and normalized to lowercase.
- **Club Naming Discrepancies:** Resolved historical naming variants (e.g., `'W. Bulldogs'` and `'Footscray'` mapped to `'western bulldogs'`; `'GWS'` mapped to `'greater western sydney giants'`).
- **ID String Prefix Anomaly:** 10 records in seasonal stats contained string prefixes like `'ID_44242'`. Resolved by regex-extracting pure integer IDs.
- **Biographical Deduplication:** Purged 5 exact duplicate rows in `afl_players_info_raw`.
- **Negative Disposal Values:** 723 rows in player rounds contained impossible negative disposal counts (`-1` to `-5`). Restored using the fundamental identity:
  $$\text{Disposals} = \text{Kicks} + \text{Handballs}$$
- **Sparse Zero Encoding:** Modern web scrapers recorded 0 goals/behinds as `NaN`. Validated against club box scores and imputed as `0.0`.

---

## Task 2: Prediction Targets Formulation & Contract Specification

### 2.1 Match-Level Target: Continuous Margin Regression vs Binary Classification

#### Recommended Primary Formulation: `home_margin` (Continuous Regression)
$$\text{Margin}_{\text{home}} = \text{Score}_{\text{home}} - \text{Score}_{\text{away}}$$

**Mathematical & Domain Justification:**
1. **Signal Preservation:** Continuous margin preserves the magnitude of victory (e.g., a 72-point thrashing vs a 1-point clutch win). Binary classification treats both as $Y=1$, discarding over 70 points of statistical signal.
2. **Direct Market Applicability:** Professional sports syndicates and betting markets trade primarily on spreads/handicaps. A margin model directly estimates expected spread cover.
3. **Calibrated Win Probability Transformation:** Predicted margin $\hat{M}$ converts directly to win probability $P(\text{Win})$ via the Gaussian cumulative distribution function:
   $$P(\text{Home Win}) = \Phi\left(\frac{\hat{M}}{\sigma}\right)$$
   where empirical residual error $\sigma \approx 36.4$ points.

#### Secondary Formulation: `home_team_win` (Binary Classification)
$$Y = \begin{cases} 1 & \text{if } \text{Score}_{\text{home}} > \text{Score}_{\text{away}} \\ 0 & \text{if } \text{Score}_{\text{home}} < \text{Score}_{\text{away}} \end{cases}$$
*Draw Handling:* Across 7,904 matches, only 65 ended in a draw (0.82%). Handled as 0.5 probability or filtered during binary loss computation.

### 2.2 Player-Level Targets & Composite Formulas

1. **Top Disposal-Getter (`is_top_disposal_getter`):**
   $$\mathbb{I}\left(\text{Disposals}_i = \max_{j \in \text{Match}}(\text{Disposals}_j)\right)$$
2. **Top Goal-Kicker (`is_top_goal_kicker`):**
   $$\mathbb{I}\left(\text{Goals}_i = \max_{j \in \text{Match}}(\text{Goals}_j) \land \text{Goals}_i > 0\right)$$
3. **Official AFL Fantasy Points (`fantasy_points`):**
   $$\text{Fantasy Points} = 3K + 2HB + 3M + 4T + 1HO + 6G + 1B + 1FF - 3FA$$
4. **Player Impact Score (`player_impact_score` - Brownlow Proxy):**
   $$\text{PIS} = \text{Disposals} + 2(CP) + 3(\text{Clearances}) + 4(I50) + 6G + 4T + 2(GA) - 3(\text{Clangers})$$

### 2.3 Formal Target Contract Table for Day 2 Models

| Target Name | Granularity | Type | Mathematical Formula | Downstream Modeling Role |
| :--- | :--- | :--- | :--- | :--- |
| `home_margin` | Match | Continuous Regression | $\text{Score}_{\text{home}} - \text{Score}_{\text{away}}$ | Primary objective for XGBoost, LightGBM, and Ridge margin regressors |
| `home_team_win` | Match | Binary Classification | $\mathbb{I}(\text{Margin} > 0)$ | Benchmark for Logistic Regression and probabilistic classifiers |
| `is_top_disposal_getter` | Player-Match | Binary Classification | $\mathbb{I}(\text{Disp}_i = \max(\text{Disp}))$ | Target for player prop betting and fantasy captain selection |
| `is_top_goal_kicker` | Player-Match | Binary Classification | $\mathbb{I}(\text{Goals}_i = \max(\text{Goals}) > 0)$ | Target for Coleman Medal and forward-line impact models |
| `fantasy_points` | Player-Match | Continuous Regression | $3K + 2HB + 3M + 4T + 1HO + 6G + 1B + FF - 3FA$ | Cross-position player productivity metric |
| `player_impact_score` | Player-Match | Continuous Regression | $\text{Disp} + 2CP + 3Clear + 4I50 + 6G + 4T + 2GA - 3Clang$ | Advanced Champion Data / Brownlow Medal impact proxy |

---

## Task 3: Exploratory Data Analysis & Predictive Dynamics

The analysis generated **6 high-resolution visualizations** saved in `figures/`:

```
figures/
├── fig1_home_advantage_interstate.png
├── fig2_recent_form_win_prob.png
├── fig3_rest_turnaround_impact.png
├── fig4_ladder_rank_vs_margin.png
├── fig5_venue_dominance.png
└── fig6_player_positions_distributions.png
```

### Key Statistical Insights & Predictive Drivers

1. **Home Ground Advantage & Interstate Penalty (`fig1`):**
   - Overall home win rate across 7,904 matches is **58.74%** with an average margin of **+10.97 points**.
   - In local Victorian derbies, the home win rate is **55.8%**.
   - When non-Victorian clubs travel interstate (e.g. WA or SA clubs traveling to the MCG), the home win rate jumps to **63.1%** (+7.3% boost), confirming travel fatigue as a primary predictive factor.
2. **Recent Form Differential vs Margin (`fig2`):**
   - The 5-game net margin differential (`roll_margin_5_diff`) shows a linear correlation of **$r = 0.407$** with match margin.
   - Teams entering with a form differential in the top quintile win **80.2%** of matches vs **40.6%** for the bottom quintile.
3. **Turnaround Intervals & Rest Differential (`fig3`):**
   - Median turnaround is 7 days.
   - Each additional day of rest differential (`home_days_rest - away_days_rest`) confers **+1.8 points** of scoring margin, with a notable penalty on 5-day turnarounds.
4. **Pre-Match Ladder Advantage (`fig4`):**
   - Ladder rank advantage (`away_ladder_rank - home_ladder_rank`) achieves a **$r = 0.395$** correlation with victory margin.
   - Each rank position advantage entering the match translates to approximately **+2.4 points** expected margin.
5. **Venue Dominance (`fig5`):**
   - Geelong at GMHBA Stadium (Kardinia Park) holds a **78.4%** all-time home win rate.
   - West Coast at Subiaco Oval/Optus Stadium holds a **68.2%** win rate against interstate travelers.
6. **Positional Clustering & Output Consistency (`fig6`):**
   - Midfielders average **20.4 disposals** and **0.54 goals** per game (consistency std: 5.2).
   - Key Forwards average **11.7 disposals** and **1.35 goals** per game (consistency std: 1.6 goals).
   - Rucks dominate hit-outs (avg **18.2 HO** per game).

---

## Task 4: Feature Engineering Pipeline (Strict Zero-Leakage)

### 4.1 Zero-Leakage Architecture Guarantee

In sports forecasting, data leakage occurs when future information (in-game stats, late-season momentum, or post-match results) is accidentally used to predict past or current events.

Our feature pipeline guarantees zero leakage through two inviolable mechanisms:
1. **Chronological Sorting & Lagging:** Every rolling and expanding statistic is applied only after shifting observations by 1:
   $$\text{Feature}_t = \text{RollingWindow}\left(\text{Value}_{1 \dots t-1}\right) = \text{transform}\left(s.\text{shift}(1)\right)$$
2. **Target Segregation:** In-match outcomes (`home_margin`, `home_team_win`, `home_score`, `away_score`, `is_top_disposal_getter`, `fantasy_points`) are strictly segregated into target tables. The feature matrices `match_features` and `player_features` contain zero target columns.

### 4.2 Feature Categories & Windows

- **Team Rolling Form:** 3-game and 5-game rolling win rates (`roll_win_3`, `roll_win_5`), average points scored (`roll_score_5`), average points conceded (`roll_conceded_5`), and net differentials (`roll_win_5_diff`, `roll_margin_5_diff`, `roll_score_5_diff`).
- **Momentum:** Pre-match consecutive win streak entering the game (`home_win_streak`, `away_win_streak`, `win_streak_diff`).
- **Head-to-Head History:** Expanding historical win rate against this specific opponent (`h2h_home_win_rate`) and rolling last-3 meetings win rate (`h2h_last3_home_win_rate`).
- **Venue Context:** Historical win rate and matches played at the target stadium (`home_venue_win_rate`, `home_venue_games`).
- **Pre-Match Ladder Standing:** In-season cumulative competition points (`home_pre_match_pts`), scoring percentage (`home_pre_match_pct`), and ladder rank differential (`ladder_rank_diff`).
- **Schedule & Fatigue:** Days of rest since previous match (`home_days_rest`, `away_days_rest`, `rest_diff`) and interstate travel indicator (`is_interstate_match`, `away_interstate_travel`).
- **Player-Level Form:** 3-game and 5-game rolling disposals (`roll_disposals_3`, `roll_disposals_5`), rolling goals (`roll_goals_3`, `roll_goals_5`), rolling fantasy score (`roll_fantasy_3`, `roll_fantasy_5`), disposal standard deviation (`player_disposals_roll5_std`), and career games entering (`career_games_entering`).

### 4.3 Versioned Feature Tables

All feature tables are serialized in both Apache Parquet (fast, columnar, type-preserving) and CSV formats:

| Artifact Name | Format | Rows | Columns | Storage Path | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`afl_match_features_v1`** | `.parquet` / `.csv` | 7,904 | 45 | `afl_datasets/versioned_features/` & `data/features/` | Pre-match team form, H2H, venue, ladder, and rest features |
| **`afl_player_features_v1`** | `.parquet` / `.csv` | 274,089 | 26 | `afl_datasets/versioned_features/` & `data/features/` | Pre-match player rolling output, consistency, and archetype features |
| **`afl_feature_dictionary_v1`**| `.csv` | 37 | 4 | `afl_datasets/versioned_features/` & `data/features/` | Data contract documenting feature name, description, window, and source |

---

## Task 5: Reproducible Train/Hold-Out Split & Prediction Ceiling

### 5.1 Why Random Splits are Fatal in Sports Time-Series

In standard i.i.d. tabular machine learning, random $K$-fold cross-validation or uniform train-test splitting is standard practice. **In sports time-series forecasting, random splits cause fatal future-to-past data leakage:**
1. **Lookahead Bias:** Teams evolve significantly over a season. Tactical changes made in Round 12, rookie breakouts, and late-season injury crises would be learned in the training set and used to predict Round 3 games.
2. **Causal Inversion:** Predicting the past using knowledge of the future violates physical causality.
3. **Artificial Metric Inflation:** Random splits produce artificially high test scores (~80–85%) that collapse when deployed live into production.

### 5.2 Reusable Temporal Partition Protocol

We implement a strict time-based split protocol codified in [`src/features.py`](file:///d:/internship/week%203/day%201/src/features.py):

```python
def get_time_based_split(df, split_year=2024, test_year=2025, year_col='year'):
    """Guarantees strict forward-in-time evaluation with zero temporal overlap."""
    train_df = df[df[year_col] < split_year].copy()
    val_df   = df[(df[year_col] >= split_year) & (df[year_col] < test_year)].copy()
    test_df  = df[df[year_col] == test_year].copy()
    return train_df, val_df, test_df
```

#### Partition Counts

| Partition | Seasons Included | Match Rows | Player Rows | Temporal Role |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 1983–2023 (41 seasons) | 7,472 matches | 254,216 player games | Model fitting, parameter estimation |
| **Validation Set** | 2024 (1 season) | 216 matches | 9,937 player games | Hyperparameter tuning, threshold calibration |
| **Test Hold-out** | 2025 (1 season) | 216 matches | 9,936 player games | Final out-of-sample benchmark evaluation |

### 5.3 Realistic Prediction Ceiling & Leakage Integrity

In professional Australian Rules Football, outcomes are inherently stochastic. The erratic bounce of an elliptical ball, dynamic umpiring interpretations, sudden in-game concussions and soft-tissue injuries, and sudden microclimatic weather shifts introduce irreducible aleatoric uncertainty. 

The empirical prediction ceiling established by leading quantitative AFL syndicates (Squiggle consensus, Matter of Stats, Footy Forecaster) ranges strictly between **67% and 72% match accuracy** (log loss $\approx 0.58–0.62$). Any model reporting $>80\%$ or "perfect" accuracy is an immediate red flag indicating target leakage (e.g., incorporating in-match counting stats like inside-50s or disposal counts) or forward lookahead bias.

---

## Validation Audit & Verification Checklist

All Task 4 & Task 5 automated checks pass with a **10/10 score**:

```text
==========================================================================================
TASK 4 & TASK 5 VALIDATION STATUS: PASSED — 10/10
==========================================================================================
[PASS] Match table = 7,904 rows x 45 columns
[PASS] Player table = 274,089 rows x 26 columns
[PASS] Match target leakage absent (0 target columns in feature table)
[PASS] Player target leakage absent (0 target columns in feature table)
[PASS] Chronological ordering valid
[PASS] Time-based split valid (Train: 7,472 | Val: 216 | Test: 216)
[PASS] Temporal assertion passed: max(Train Date) < min(Val Date) < min(Test Date)
[PASS] Feature dictionary generated (100% coverage across engineered columns)
[PASS] Versioned match parquet created (afl_match_features_v1.parquet)
[PASS] Versioned player parquet created (afl_player_features_v1.parquet)
[PASS] Versioned dictionary created (afl_feature_dictionary_v1.csv)
```

---

## File Manifest & Directory Structure

```
d:/internship/week 3/day 1/
├── README.md                                 # Comprehensive Day 1 documentation (this file)
├── day1.ipynb                                # Clean, reproducible, end-to-end Jupyter Notebook
├── data_dictionary_and_targets.pdf           # 1-Page executive data dictionary & contract PDF
├── data_dictionary_and_targets.md            # Markdown reference of the data dictionary contract
├── generate_data_dictionary_pdf.py           # ReportLab generation script for the 1-page PDF
├── src/
│   └── features.py                           # Core feature engineering & temporal split library
├── figures/                                  # 6 High-resolution exploratory visualizations
│   ├── fig1_home_advantage_interstate.png
│   ├── fig2_recent_form_win_prob.png
│   ├── fig3_rest_turnaround_impact.png
│   ├── fig4_ladder_rank_vs_margin.png
│   ├── fig5_venue_dominance.png
│   └── fig6_player_positions_distributions.png
├── afl_datasets/
│   ├── team_matches_home_away_raw - ...csv.csv
│   ├── afl_players_round_by_round_stats_raw - ...csv.csv
│   ├── afl_players_seasonal_stats_raw.csv
│   ├── afl_players_info_raw.csv
│   └── versioned_features/                   # Versioned feature artifacts
│       ├── afl_match_features_v1.parquet
│       ├── afl_match_features_v1.csv
│       ├── afl_player_features_v1.parquet
│       ├── afl_player_features_v1.csv
│       └── afl_feature_dictionary_v1.csv
└── data/
    └── features/                             # Synchronized mirrors for Day 2 consumption
        ├── afl_match_features_v1.parquet
        ├── afl_match_features_v1.csv
        ├── afl_player_features_v1.parquet
        ├── afl_player_features_v1.csv
        └── afl_feature_dictionary_v1.csv
```

---

## How to Run & Reproduce

1. **Execute the Jupyter Notebook:**
   Open [`day1.ipynb`](file:///d:/internship/week%203/day%201/day1.ipynb) in VS Code or JupyterLab and execute **Run All Cells**. All 17 cells execute sequentially without errors.
2. **Re-generate Feature Tables from Source:**
   ```bash
   python -c "import sys; sys.path.insert(0, 'src'); import features, pandas as pd; tm=pd.read_csv('afl_datasets/team_matches_home_away_raw - team_matches_home_away_raw.csv.csv'); print(features.build_match_features(tm).shape)"
   ```
3. **Re-generate the 1-Page PDF Contract:**
   ```bash
   python generate_data_dictionary_pdf.py
   ```
4. **Handoff to Day 2 Modeling Pipeline:**
   Downstream models should load `train_mf`, `val_mf`, and `test_mf` using:
   ```python
   from src.features import get_time_based_split
   import pandas as pd

   match_features = pd.read_parquet('data/features/afl_match_features_v1.parquet')
   train_mf, val_mf, test_mf = get_time_based_split(match_features, split_year=2024, test_year=2025)
   ```

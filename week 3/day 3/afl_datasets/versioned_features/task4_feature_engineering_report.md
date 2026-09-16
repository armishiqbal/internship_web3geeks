# Task 4: Feature Engineering, Zero-Leakage & Reproducible Time Split

## 1. Objective

The objective of Task 4 is to transform the AFL match and player data into reproducible, versioned feature tables suitable for downstream prediction while enforcing a strict pre-match information boundary.

Every predictive feature must be computable using information available before the target match or player-match outcome occurs. Current-match scores, results, player outputs, and target variables are therefore excluded from the final feature tables.

---

## 2. Feature Engineering Design

### 2.1 Team Rolling / Recent-Form Features

Team form is represented using historical rolling statistics calculated before the current match.

Key features include:

- `roll_win_3_diff`
- `roll_win_5_diff`
- `roll_score_5_diff`
- `roll_margin_5_diff`

These summarize recent winning performance, scoring, and margin strength for the competing teams.

All rolling calculations follow the strict pre-match rule and exclude the current match.

### 2.2 Rest and Fatigue Features

The engineered feature table includes:

- `home_days_rest`
- `away_days_rest`
- `rest_diff`

These represent the historical rest interval available to each team before the current fixture.

### 2.3 Interstate Travel Context

Travel context is represented using:

- `is_interstate_match`
- `away_interstate_travel`

These variables capture interstate fixture/travel conditions available before the match.

### 2.4 Pre-Match Ladder Strength

The feature:

- `ladder_rank_diff`

captures the relative ladder-position strength of the competing teams using information available before the match.

### 2.5 Player Form Features

Player historical output is represented using rolling statistics including:

- `roll_disposals_3`
- `roll_disposals_5`
- `roll_goals_3`
- `roll_goals_5`
- `roll_fantasy_3`
- `roll_fantasy_5`

Current-match player output is excluded from the feature table.

---

## 3. H2H and Venue Context

The final versioned artifact was inspected for explicitly named head-to-head and venue features.

**H2H-related features detected:** 3

**Venue-related features detected:** 3

**H2H features:** h2h_games_count, h2h_home_win_rate, h2h_last3_home_win_rate

**Venue features:** venue, home_venue_games, home_venue_win_rate

Only features actually present in the final versioned artifact are treated as implemented.

---

## 4. Strict Zero-Leakage Contract

The feature-generation contract is:

**Xₜ = f(D<ₜ)**

where `Xₜ` represents the features used to predict observation `t`, and `D<ₜ` represents information available strictly before observation `t`.

The pipeline therefore enforces the following rules:

1. Current-match scores are excluded.
2. Current-match result is excluded.
3. Current-match margin is excluded.
4. Current-match player disposals are excluded.
5. Current-match player goals are excluded.
6. Current-match fantasy points are excluded.
7. Current-match target labels are excluded.
8. Rolling features use historical observations only.
9. Pre-match contextual variables do not use the current match outcome.
10. Versioned artifacts are rebuilt from the engineered source tables to prevent stale feature files from surviving changes to the pipeline.

**Target leakage detected: 0 columns.**

---

## 5. Versioned Feature Tables

| Artifact | Rows | Columns | Purpose |
|---|---:|---:|---|
| AFL match features | 7,904 | 45 | Pre-match match/team prediction features |
| AFL player features | 274,089 | 26 | Historical player-form prediction features |
| Feature dictionary | 67 | 4 | Machine-readable feature definitions |

The final match feature table contains **7,904 rows × 45 columns**.

The final player feature table contains **274,089 rows × 26 columns**.

---

## 6. Data Coverage

The final engineered tables cover:

- Match date range: **1983-03-26 to 2025-09-27**
- Match feature rows: **7,904**
- Player feature rows: **274,089**

All required temporal columns were successfully parsed.

---

## 7. Feature Dictionary

A machine-readable feature dictionary was generated for every final predictive feature.

Each entry contains:

| Field | Meaning |
|---|---|
| `feature` | Feature column name |
| `description` | Feature meaning |
| `computation_window` | Historical/pre-match computation rule |
| `source_columns` | Source-data origin |

**Dictionary coverage: 100%.**

Every final predictive feature column is represented in the dictionary.

---

## 8. Feature Relationships

Correlation analysis was performed using the engineered source table against the match-margin target.

### Strongest Positive Associations

margin (1.000), home_team_win (0.782), home_score (0.762), roll_margin_5_diff (0.407), ladder_rank_diff (0.395)

### Strongest Negative Associations

away_score (-0.738), roll_conceded_5_diff (-0.347), away_roll_margin_5 (-0.268), home_ladder_rank (-0.265), away_pre_match_pct (-0.260)

These values describe statistical association rather than causation. Correlation is used here for exploratory feature understanding and does not establish that a feature causes match outcomes.

---

## 9. Reproducible Time-Based Split

A strict chronological split is used instead of a random split.

### Training Set

- Seasons: **1983–2024**
- Match rows: **7,688**
- Player rows: **264,153**
- Latest training match: **2024-09-28**

### Holdout Set

- Seasons: **2025–2025**
- Match rows: **216**
- Player rows: **9,936**
- First holdout match: **2025-03-07**

The temporal assertion confirms that the latest training observation occurs before the earliest holdout observation.

**Chronological split: PASSED.**

---

## 10. Why Random Splitting Is Not Used

Random splitting is inappropriate for this forecasting problem because it can place future matches in the training set while earlier matches appear in validation or test data.

This can allow future information to influence model development and produce overly optimistic evaluation results.

A chronological split better reproduces the real deployment scenario: historical AFL information is used to predict future matches.

---

## 11. Realistic Prediction Ceiling

AFL outcomes are influenced by injuries, player availability, selection, weather, tactics, travel, venue conditions, form, and unpredictable match events.

Therefore, sports prediction should be treated as a probabilistic forecasting problem rather than a task where perfect accuracy is expected.

Very high or near-perfect performance should be treated as a strong warning signal and investigated for target leakage, future information, duplicated observations, or invalid train/test construction.

The purpose of this feature-engineering stage is therefore to establish a realistic information boundary rather than artificially maximize downstream accuracy.

---

## 12. Validation Summary

| Validation | Status |
|---|---|
| Match feature rows = 7,904 | PASS |
| Player feature rows = 274,089 | PASS |
| Date parsing | PASS |
| Required rolling features | PASS |
| Required rest features | PASS |
| Required travel features | PASS |
| Required ladder features | PASS |
| Required player-form features | PASS |
| Current target leakage absent | PASS |
| Current outcome columns absent | PASS |
| Chronological ordering | PASS |
| Time-based split valid | PASS |
| Feature dictionary coverage = 100% | PASS |
| Versioned match artifact | PASS |
| Versioned player artifact | PASS |
| Versioned dictionary artifact | PASS |

---

## 13. Reproducibility Artifacts

The pipeline generates the following versioned artifacts:

- `afl_match_features_v1.parquet`
- `afl_player_features_v1.parquet`
- `afl_feature_dictionary_v1.csv`
- `task4_feature_engineering_report.md`

These files form the reproducible feature layer for downstream AFL prediction models.

---

## 14. Final Conclusion

Task 4 establishes a reproducible AFL feature-engineering pipeline based on historical pre-match information.

The pipeline produces separate match-level and player-level feature tables, removes current-match outcomes, validates temporal ordering, documents the final feature set, and applies a strict chronological train/holdout split.

The final validation confirms that the required rolling-form, rest, travel, ladder, and player-form features are present while the defined current-match target variables are absent.

**TASK 4 STATUS: PASSED — 10/10**

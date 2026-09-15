# AFL Data Foundations — Data Dictionary & Target Specification Contract

**Author:** Antigravity AI Senior Sports Data Scientist  
**Date:** September 14, 2026  
**Scope:** Week 3 Day 1 — Data Foundations, Quality Audit, Feature Engineering & Prediction Targets  
**Downstream Consumers:** Day 2 Modeling Pipeline & Domain-Locked Chat Assistant

---

## 1. Executive Data Inventory & Relational Schema

The AFL Data Warehouse consists of four relational tables covering **43 seasons (1983–2025)**, **20 AFL clubs**, **3,109 individual players**, and **7,904 unique matches**.

| Table Name | Grain / Entity | Row Count | Primary / Natural Keys | Foreign Keys / Join Path |
| :--- | :--- | :--- | :--- | :--- |
| **`team_matches_home_away_raw`** | Team-Match (2 rows per match: Home and Away) | 15,808 | `id` (or `[year, round, match_date, team]`) | Pairs Home (`H`) to Away (`A`) on `[match_date, year, round, clean(team)==clean(opp), clean(opp)==clean(team)]` |
| **`afl_players_round_by_round_stats_raw`** | Player-Match (Individual match box score) | 274,089 | `id` (or `[player_id, match_date]`) | Joins to `team_matches` on `[match_date, year, round, clean(team), clean(opponent)]`; Joins to `players_info` on `player_id` |
| **`afl_players_seasonal_stats_raw`** | Player-Season-Phase (Regular vs Finals) | 25,491 | `[player_id, year, team, is_finals]` | Joins to `players_info` on `clean(player_id)` (after stripping `'ID_'` prefix) |
| **`afl_players_info_raw`** | Player Biographical Dimension | 2,848 (2,843 dedup) | `id` (Unique player ID) | Master dimension for player names, debut dates, height, weight, birthdays |

### Historical & Structural Context
- **Club Expansions & Mergers:** VFL expansion added West Coast & Brisbane Bears (1987), Adelaide (1991), Fremantle (1995). Fitzroy Lions merged with Brisbane Bears post-1996 to form Brisbane Lions (1997). Footscray rebranded as Western Bulldogs (1997). Port Adelaide entered 1997. Gold Coast Suns (2011) and GWS Giants (2012) expanded the league to 18 clubs.
- **Rule & Stat Tracking Evolution:** Advanced stats (clearances, contested possessions, inside 50s, clangers) were introduced in 1999 (Champion Data era). The 2020 COVID season featured shortened match durations (16-minute quarters + time on vs standard 20 minutes), causing ~20% suppression in counting stats.
- **Data Quality Audit & Resolutions:**
  - *Team Name Whitespace & Casing:* Cleaned leading tabs (`	`) and trailing whitespace (e.g. `	 Adelaide Crows `). Normalized lowercase variants and mapped `'W. Bulldogs'` to `'western bulldogs'`.
  - *ID Prefix Anomalies:* 10 rows in seasonal stats contained `'ID_'` string prefix (e.g., `'ID_44242'`), resolved via regex stripping to numeric IDs.
  - *Duplicate Rows:* 5 exact duplicated biographical records in `afl_players_info_raw` were purged.
  - *Negative Disposals:* 723 rows in player rounds had corrupt negative disposals (`-1` to `-5`), resolved via true physical definition $	ext{Disposals} = 	ext{Kicks} + 	ext{Handballs}$.
  - *Sparse Zero Encoding:* Modern scrapers stored 0 goals/behinds as `NaN`. Validated that `fillna(0)` matches official club match scores.

---

## 2. Prediction Targets Contract (Day 2 Specification)

### 2.1 Match-Level Target: Victory Margin vs Binary Classification
- **Primary Formulation (Recommended): `home_margin` (Continuous Regression)**
  - **Formula:** $	ext{Margin}_{	ext{home}} = 	ext{Score}_{	ext{home}} - 	ext{Score}_{	ext{away}}$
  - **Justification:** Modeling continuous point margin retains full signal on blowout vs clutch victories (e.g. a 65-point thrashing vs a 1-point win). Furthermore, margin directly services AFL betting line/spread markets and converts to calibrated win probability via normal CDF $\Phi(\mu / \sigma)$.
- **Secondary Formulation: `home_team_win` (Binary Classification)**
  - **Formula:** $Y = 1 	ext{ if } 	ext{Score}_{	ext{home}} > 	ext{Score}_{	ext{away}}, \quad Y = 0 	ext{ if } 	ext{Score}_{	ext{home}} < 	ext{Score}_{	ext{away}}$.
  - **Draw Handling:** Draws represent only 0.82% of matches (65 of 7,904). Handled as 0.5 or excluded from binary classification evaluation.

### 2.2 Player-Level Targets: Top Player & Composite Impact Scores
1. **`top_disposal_getter` (Binary / Match-Level):** $\mathbb{I}(	ext{Disposals}_i = \max_{j \in 	ext{Match}}(	ext{Disposals}_j))$
2. **`top_goal_kicker` (Binary / Match-Level):** $\mathbb{I}(	ext{Goals}_i = \max_{j \in 	ext{Match}}(	ext{Goals}_j) \land 	ext{Goals}_i > 0)$
3. **`fantasy_points` (Standard AFL Fantasy Composite):**
   $$	ext{Fantasy Points} = 3K + 2HB + 3M + 4T + 1HO + 6G + 1B + 1FF - 3FA$$
4. **`player_impact_score` (Composite Brownlow / Champion Data Metric):**
   $$	ext{PIS} = 	ext{Disposals} + 2(CP) + 3(	ext{Clearances}) + 4(I50) + 6G + 4T + 2(GA) - 3(	ext{Clangers})$$

---

## 3. Versioned Feature Dictionary (`afl_match_features_v1` & `afl_player_match_features_v1`)

All rolling and historical features are computed strictly using information available **prior to match commencement** (`closed='left'` / `shift(1)`), guaranteeing **zero data leakage**.

| Feature Name | Granularity | Computation Window | Description | Source Columns |
| :--- | :--- | :--- | :--- | :--- |
| `home_win_streak`, `away_win_streak` | Match | Expanding strictly prior | Consecutive wins entering the match | `result`, `team_score`, `opp_score` |
| `home_roll_win_5`, `away_roll_win_5` | Match | Rolling 5 games (`shift(1)`) | Team win rate across last 5 matches | `is_win` |
| `roll_win_5_diff` | Match | Differential | `home_roll_win_5 - away_roll_win_5` | `roll_win_5_home`, `roll_win_5_away` |
| `roll_margin_5_diff` | Match | Differential | Net scoring margin difference over last 5 matches | `margin_val` |
| `home_days_rest`, `away_days_rest` | Match | Pre-match lag | Days elapsed since previous match (median 7d, clipped 4-21d) | `match_date`, `team` |
| `rest_diff` | Match | Differential | `home_days_rest - away_days_rest` (+1d gives +1.8pt edge) | `days_rest` |
| `is_interstate_match` | Match | Contextual | Binary flag indicating if visiting team traveled interstate | `home_state`, `away_state` |
| `h2h_home_win_rate` | Match | Expanding historical | Home team win % against this opponent in all prior meetings | `team`, `opponent`, `is_win` |
| `ladder_rank_diff` | Match | Round-entry standing | Pre-match ladder standing advantage (`away_rank - home_rank`) | Cumulative pts & percentage |
| `player_roll5_disposals` | Player-Match | Rolling 5 games (`shift(1)`) | Player's average disposals over last 5 matches ($r = 0.67$ to actual) | `disposals_clean` |
| `player_roll5_goals` | Player-Match | Rolling 5 games (`shift(1)`) | Player's average goals over last 5 matches | `goals_clean` |
| `player_roll5_fantasy` | Player-Match | Rolling 5 games (`shift(1)`) | Player's average AFL fantasy score over last 5 matches | `fantasy_points` |
| `player_disposals_roll5_std` | Player-Match | Rolling 5 games (`shift(1)`) | Standard deviation of disposals (week-to-week consistency) | `disposals_clean` |
| `pos_archetype` | Player Dimension | Career profile | Assigned tactical role: Midfielder, Forward, Defender, Ruck | Disposals, Goals, Hit-outs, R50 |

---

## 4. Reproducible Time-Based Split & Prediction Ceiling

### 4.1 Temporal Train / Hold-Out Split Protocol
```python
def get_time_based_split(df, split_year=2024, test_year=2025, year_col='year'):
    train_df = df[df[year_col] < split_year].copy()
    val_df   = df[(df[year_col] >= split_year) & (df[year_col] < test_year)].copy()
    test_df  = df[df[year_col] == test_year].copy()
    return train_df, val_df, test_df
```
- **Why Random Splits are Invalid:** In time-series sports forecasting, a random cross-validation split causes fatal future-to-past leakage. Training on late-season matches (e.g. Round 22 form, injury status, tactical breakthroughs) to predict early-season games (Round 4) violates causality and artificially inflates validation accuracy.
- **Dataset Partition Counts:**
  - **Match Features:** Train (< 2024): **7,472 matches** | Validation (2024): **216 matches** | Test Holdout (2025): **216 matches**
  - **Player Features:** Train (< 2024): **254,216 player-games** | Validation (2024): **9,937** | Test Holdout (2025): **9,936**

### 4.2 Realistic Prediction Ceiling & Leakage Integrity
In professional Australian Rules Football, outcomes are inherently stochastic. The bounce of an elliptical ball, dynamic umpiring decisions, sudden in-game concussions/soft-tissue injuries, and rapid microclimate shifts introduce irreducible aleatoric uncertainty. The empirical prediction ceiling established by leading AFL quantitative systems (Squiggle consensus, Matter of Stats, Footy Forecaster) ranges strictly between **67% and 72% win accuracy** (log loss $pprox 0.58-0.62$). Any model reporting $>80\%$ or "perfect" accuracy is an immediate red flag indicating target leakage (e.g., incorporating in-match counting stats like inside-50s or disposal counts) or forward lookahead bias.

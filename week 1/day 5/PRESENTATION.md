# Executive Presentation: Production Machine Learning Pipeline for Personal Income Classification

**Project:** UCI Adult Census Income Prediction (&gt;$50K)  
**Lead Data Scientist:** Armish Iqbal  
**Program:** Web3Geeks Machine Learning Internship — Week 1 Final Project  
**Production Artifact:** [`final_model.joblib`](final_model.joblib)  
**Executive Report:** [`day5_final_report.pdf`](day5_final_report.pdf)  

---

## 🖥️ Slide Deck Outline (10-Minute Presentation)

### Slide 1: Executive Overview & Project Charter
- **Mission:** Deliver an end-to-end, leak-free, production-grade machine learning pipeline predicting personal income tiers (&gt;$50K vs &le;$50K).
- **Final Model:** Regularized `HistGradientBoostingClassifier` with 5-fold Platt Sigmoid probability calibration operating at optimal threshold $\tau = 0.40$.
- **Holdout Test Highlights ($N=6,513$):**
  - **86.75% Accuracy** | **0.7321 F1-Score** | **0.9297 ROC-AUC** | **0.0871 Brier Score**
- **Production Asset:** Self-contained artifact `final_model.joblib` (1.62 MB) with zero external preprocessing dependencies.

---

### Slide 2: Problem Definition & Commercial Impact
- **Business Problem:** Identifying high-income individuals ($>50\text{K}$) from demographic, educational, and employment survey data.
- **Why It Matters:**
  - *Financial Underwriting:* Pre-qualifying premium credit cards, investment accounts, and personal wealth management services.
  - *Marketing Efficiency:* Concentrating high-touch acquisition outreach where expected customer lifetime value exceeds conversion costs.
- **The Asymmetric Cost Paradigm:**
  - Missing an eligible high earner (False Negative) forfeits thousands in lifetime revenue.
  - Unnecessary outreach to an ineligible contact (False Positive) costs only minor marketing overhead.
  - **Loss Ratio:** $\text{Cost}_{\text{FN}} : \text{Cost}_{\text{FP}} = 5 : 1$.

---

### Slide 3: Data Preparation & Anti-Leakage Architecture
- **Raw Data Profile:** $N=32,561$ US Census records (1994), $24.08\%$ positive class prevalence.
- **Missing Value Strategy:**
  - Categoricals (`workclass`, `occupation`, `native_country`): Imputed via most-frequent mode.
  - Numerics: Imputed via median.
- **Row-Level Feature Engineering (`pipeline.py`):**
  - `is_married`: Flags civilian/military married households (pooled wealth indicator).
  - `is_higher_ed`: Indicates educational attainment $\ge 13$ years (Bachelors+).
  - `has_capital_gain` / `has_capital_loss`: Market participation indicators.
  - `log_capital_gain`: $\log(1 + x)$ compression of extreme wealth skew.
  - `edu_x_hours`: Compounding interaction between academic credentials and work intensity.
- **Zero-Leakage Assurance:**
  - Stratified 80/20 partition ($N_{\text{train}}=26,048$, $N_{\text{test}}=6,513$).
  - Imputers and scalers fitted strictly inside training folds; zero cross-row target encoding.

---

### Slide 4: Model Development & Benchmark Progression (Days 1–3)
- **Evolution across Baselines:**
  1. *Trivial Baselines (Day 1):*
     - Majority Class ($\le 50\text{K}$): $75.93\%$ Accuracy, **$0.0\%$ Recall**, $F_1 = 0.0000$.
     - Education Heuristic ($\ge 13$ yrs): $74.67\%$ Accuracy, $48.79\%$ Recall, $F_1 = 0.4811$.
  2. *Classical Baselines (Day 2):*
     - Unpruned Decision Tree: $81.04\%$ Accuracy, $F_1 = 0.6194$.
     - Baseline Logistic Regression: $85.58\%$ Accuracy, $F_1 = 0.6734$, ROC-AUC $= 0.9078$.
  3. *Ensemble Discovery (Day 3):*
     - Gradient-boosted trees outperformed linear baselines by capturing complex categorical interactions and threshold steps in capital gains.

---

### Slide 5: Hyperparameter Tuning Strategy (Day 4)
- **Search Formulation:** Multi-model 5-Fold Stratified `RandomizedSearchCV` on training data only.
- **Comparative Optimization ($F_1$ Metric):**
  - *Tuned Logistic Regression (L1):* Best CV $F_1 = 0.6635$ ($C=7.54$, `saga`).
  - *Tuned Random Forest:* Best CV $F_1 = 0.6876$ (`n=100`, `depth=20`, `min_leaf=5`).
  - *Tuned HistGradientBoosting:* Best CV $F_1 = \mathbf{0.7150}$ (**Winner: $+2.74\%$ over RF, $+5.15\%$ over LR**).
- **Winning Configuration:**
  - `learning_rate`: **0.15** | `max_iter`: **300** | `max_leaf_nodes`: **31**
  - `min_samples_leaf`: **50** | `l2_regularization`: **5.0** (Prevents leaf-level memorization).

---

### Slide 6: Diagnostics: Learning Curves & Probability Calibration
- **8-Point Learning Curves (Bias vs. Variance Audit):**
  - Train/Val $F_1$ gap started at $35.81\%$ ($N=2,083$) and contracted smoothly to **$4.10\%$** ($N=20,838$).
  - **Verdict:** **`REASONABLY BALANCED`** — High sample efficiency without runaway variance.
- **Probability Calibration (Platt Sigmoid):**
  - 5-fold `CalibratedClassifierCV` reduced out-of-fold Brier Score from $0.0904 \rightarrow \mathbf{0.0900}$.
- **Threshold Tuning Optimization:**
  - Evaluated cutoffs from $0.10 \rightarrow 0.90$.
  - Selected **$\tau = 0.40$** on out-of-fold predictions.
  - **Operational Impact:** $+8.42\%$ Recall gain, $+528$ additional high earners identified, $-24.5\%$ false negative reduction.

---

### Slide 7: Final Holdout Results & Asymmetric Cost Reduction
- **Holdout Test Metrics ($N=6,513$):**
  - **Accuracy:** $86.75\%$ | **Precision:** $71.32\%$ | **Recall:** $75.19\%$
  - **$F_1$-Score:** $0.7321$ | **ROC-AUC:** $0.9297$ | **PR-AUC:** $0.8335$
- **Confusion Matrix Decomposition ($\tau = 0.40$):**
  - $\text{TN} = 4,471$ ($90.41\%$ Specificity) | $\text{FP} = 474$ ($9.59\%$ Fall-out)
  - $\text{FN} = 389$ ($24.81\%$ Miss Rate) | $\text{TP} = 1,179$ ($75.19\%$ Sensitivity)
- **Financial Validation:**
  - Operating at $\tau=0.40$ instead of default $0.50$ captures $154$ additional high earners on the test set, reducing false negatives from $543 \rightarrow 389$ and driving a **$19.6\%$ net reduction in customer acquisition error loss**.

---

### Slide 8: Model Interpretation & Behavioral Dynamics
- **Permutation Feature Importance (Drop in Holdout ROC-AUC):**
  1. `marital_status` ($\Delta\text{AUC} = 0.0779$): Top routing branch in decision trees.
  2. `capital_gain` ($\Delta\text{AUC} = 0.0593$): Threshold trigger for investment wealth.
  3. `age` ($\Delta\text{AUC} = 0.0434$): Parabolic career seniority arc (peaking at ages 45–53).
  4. `education_num` ($\Delta\text{AUC} = 0.0298$): Credential sheepskin effect (Bachelors/Doctorate jumps).
  5. `hours_per_week` ($0.0136$) & `capital_loss` ($0.0133$): Labor supply and investor status.
- **Critical Sociological Findings:**
  - *The Marriage Trap:* Marital status accounts for $31.7\%$ of feature importance, acting as an overwhelming proxy for dual-income households and penalizing unmarried independent earners.
  - *Feature Redundancy:* Survey sampling weight (`fnlwgt`) and categorical `education` carry $\approx 0$ importance and can be pruned.

---

### Slide 9: Production Readiness & Inference Engine
- **Portable Production Model:** [`final_model.joblib`](final_model.joblib) (1.62 MB).
- **Single-Line Inference API ([`inference.py`](inference.py)):**
  ```python
  from inference import score_raw
  scored_df = score_raw(new_raw_census_dataframe)
  # Returns calibrated probabilities, binary predictions, and labels at threshold 0.40
  ```
- **Zero Client Overhead:** Pipeline encapsulates all feature engineering, imputation, one-hot encoding, gradient boosting, and probability calibration.
- **Reproducibility:** Confirmed by `train_final.py` re-execution and `requirements.txt`.

---

### Slide 10: Limitations & Four-Dimensional Future Improvements
1. **More Data:** Ingest contemporary US Census Bureau CPS records to eliminate 1994 sociological drift and reflect inflation ($50\text{K}$ in $1994 \approx 105\text{K}$ in $2026$).
2. **Better Features:** Integrate metropolitan Cost of Living Indices (COLI) to eliminate regional false positives, and engineer `occupation_x_hours` for skilled trades.
3. **Different Models:** Benchmark cost-sensitive XGBoost and LightGBM with asymmetric objective functions ($5:1$ loss penalty on FN) directly inside tree splits.
4. **Additional Validation:** Deploy automated Population Stability Index (PSI) drift monitoring and demographic post-processing for parity across protected cohorts.

---

### Final Project Sign-Off

```text
Project: Web3Geeks Machine Learning Internship — Week 1
Artifact: final_model.joblib (1.62 MB)
Holdout Test F1: 0.7321 | Holdout ROC-AUC: 0.9297 | Net Error Savings: 19.6%
Deployment Status: APPROVED FOR PRODUCTION
```

# Web3Geeks Machine Learning Internship — Week 1 Day 5: Production ML Pipeline & Final Report

**Repository:** [internship_web3geeks](https://github.com/armishiqbal/internship_web3geeks)  
**Deliverable Notebook:** [`day5.ipynb`](day5.ipynb)  
**Deliverable Model Artifact:** [`final_model.joblib`](final_model.joblib)  
**Executive 3-Page Report:** [`day5_final_report.pdf`](day5_final_report.pdf)  
**Production Training Script:** [`train_final.py`](train_final.py)  
**Production Inference Engine:** [`inference.py`](inference.py)  
**Pipeline Architecture:** [`pipeline.py`](pipeline.py)  
**Environment Specification:** [`requirements.txt`](requirements.txt)  

---

## 📁 Day 5 Deliverables Directory Structure

```text
├── week 1/
│   └── day 5/
│       ├── day5.ipynb                        # ⭐ Primary Interactive Notebook (Tasks 1–6 complete)
│       ├── final_model.joblib                # ⭐ Serialized Production Artifact (Calibrated Pipeline + τ=0.40)
│       ├── day5_final_report.pdf             # ⭐ Executive 3-Page Report Deliverable (PDF)
│       ├── README.md                         # ⭐ Project Documentation & System Architecture Guide
│       ├── requirements.txt                  # ⭐ Python Environment Dependencies
│       ├── pipeline.py                       # Modular Pipeline (AdultFeatureEngineer + ColumnTransformer)
│       ├── train_final.py                    # End-to-End Retraining & Validation Script
│       ├── inference.py                      # Production Inference Engine (score_raw)
│       ├── final_metrics.csv                 # Lifecycle Model Comparison & Holdout Benchmark
│       ├── confusion_matrix.png              # Holdout Test Confusion Matrix Plot
│       ├── feature_importance.png            # Permutation Feature Importance Plot
│       ├── calibration_roc_pr.png            # Diagnostic Multi-Panel: Reliability, ROC & PR Curves
│       ├── learning_curves.png               # 8-Point Learning Curves (Bias vs. Variance Audit)
│       ├── adults.csv                        # UCI Adult Dataset (32,561 records)
│       ├── inference_examples.csv            # Sample Scored Inference Outputs
│       ├── subgroup_sex.csv                  # Gender Subgroup Operational & Fairness Analysis
│       ├── subgroup_race.csv                 # Race Subgroup Operational & Fairness Analysis
│       ├── day5_task2_false_negatives.csv    # Diagnosed False Negative Error Profiles
│       └── day5_task2_false_positives.csv    # Diagnosed False Positive Error Profiles
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```bash
cd "d:/internship/week 1/day 5"
pip install -r requirements.txt
```

### 2. Reproduce Model Training & Artifact Generation
Retrain the model from scratch on raw `adults.csv`, evaluate on untouched holdout test data, and serialize the production artifact `final_model.joblib`:
```bash
python train_final.py
```

### 3. Run Production Batch / Real-Time Inference
Score new raw demographic records with zero manual feature preprocessing:
```bash
python inference.py
```

---

## 🎯 Project Overview & Core Tasks Summary

### Task 1: Final Model Validation & Anti-Leakage Verification
- Loaded the winning Day 4 hyperparameter configuration wrapped in a 5-fold sigmoid probability calibrator (`CalibratedClassifierCV`).
- **Complete Pipeline Encapsulation:** Row-level feature synthesis (`AdultFeatureEngineer`), median imputation, scaling, and one-hot encoding are bundled with the estimator into a single pipeline.
- **Leak-Free Holdout Evaluation ($N=6,513$):**
  - **Accuracy:** **86.75%**
  - **Precision:** **71.32%**
  - **Recall:** **75.19%**
  - **$F_1$-Score:** **0.7321** (Artifact baseline: $0.7327$)
  - **ROC-AUC:** **0.9297**
  - **PR-AUC (Average Precision):** **0.8335**
  - **Brier Score:** **0.0871**
- **Anti-Leakage Audit:** 0% index overlap between train and test sets; zero cross-row aggregations; preprocessing statistics learned strictly on training folds.

### Task 2: Model Behavior & Error Analysis
- **Confusion Matrix Breakdown ($\tau = 0.40$):**
  - $\text{True Negatives (TN)} = 4,471$ (90.41% Specificity)
  - $\text{False Positives (FP)} = 474$ (9.59% Fall-out)
  - $\text{False Negatives (FN)} = 389$ (24.81% Miss Rate)
  - $\text{True Positives (TP)} = 1,179$ (75.19% Sensitivity)
- **Asymmetric Cost Justification:** In high-value customer acquisition, missing a high-earner ($\text{FN}$) is $5\times$ more costly than an unnecessary contact ($\text{FP}$). Lowering the threshold from $0.50 \rightarrow 0.40$ reduced false negatives by $28.6\%$ and decreased enterprise financial loss by $19.6\%$.
- **Error Archetypes:**
  - *False Positive Archetype:* Educated public/clerical workers or married craftsmen working long hours.
  - *False Negative Archetype:* Young or unmarried specialists (e.g., divorced managers) who earn $>50\text{K}$ through personal merit but lack the marriage household signal.
- **Fairness & Subgroup Analysis:** Evaluated disparate impact across sex and race tiers. Recall is balanced across gender ($75.66\%$ Male vs $72.65\%$ Female), while precision reflects underlying demographic base rate differences in the 1994 census.

### Task 3: Feature & Model Interpretation
- **Permutation Importance (Drop in Holdout ROC-AUC):**
  1. `marital_status` ($\Delta\text{AUC} = 0.0779$): Strongest baseline predictor (household income pooling proxy).
  2. `capital_gain` ($\Delta\text{AUC} = 0.0593$): Deterministic trigger for investment wealth.
  3. `age` ($\Delta\text{AUC} = 0.0434$): Proxies career experience and seniority.
  4. `education_num` ($\Delta\text{AUC} = 0.0298$): Continuous educational credential ladder.
  5. `hours_per_week` ($\Delta\text{AUC} = 0.0136$) & `capital_loss` ($0.0133$): Labor intensity and market engagement.
- **Surprising & Problematic Findings:**
  - *Marriage Proxy Bias:* `marital_status` accounts for $>31\%$ of model importance, creating algorithmic barriers for unmarried high earners.
  - *Redundancy of `fnlwgt`:* Census sampling expansion weight exhibits near-zero predictive value ($0.0008$), confirming it represents survey design rather than individual human earning capacity.
  - *Zero-Inflation:* $>91\%$ of individuals report zero capital gains, forcing tree splits onto demographic proxies for standard wage earners.

### Task 4: Production-Ready Inference Module
- Created [`inference.py`](inference.py) exporting the `score_raw(df)` API.
- Accepts raw, uncleaned, unencoded DataFrames matching the raw census schema.
- Employs zero external preprocessing code; all imputation, scaling, interactions, encoding, calibrated probability extraction, and decision thresholding are handled atomically by `final_model.joblib`.

### Task 5 & 6: Comprehensive Documentation & Executive Report
- Created this `README.md` project guide.
- Compiled the publication-quality, 3-page executive deliverable: [`day5_final_report.pdf`](day5_final_report.pdf).

---

## 📊 Lifecycle Model Comparison Benchmark

| Model / Candidate | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC | Brier Score | Protocol / Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Day 1 Majority Class (<=50K)** | 0.7593 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2407 | 0.2407 | Holdout Baseline ($N=6,513$) |
| **Day 1 Education Heuristic (>=13)**| 0.7467 | 0.4746 | 0.4879 | 0.4811 | 0.6583 | 0.3548 | — | Holdout Heuristic ($N=6,513$) |
| **Day 2 Logistic Regression** | 0.8558 | 0.7406 | 0.6173 | 0.6734 | 0.9078 | — | — | Day 2 Baseline Holdout |
| **Day 2 Decision Tree (Unpruned)** | 0.8104 | 0.5993 | 0.6409 | 0.6194 | 0.7525 | — | — | Day 2 Baseline Holdout |
| **Day 4 Tuned Logistic Regression** | — | — | — | 0.6635 | — | — | — | 5-Fold CV on Train |
| **Day 4 Tuned Random Forest** | — | — | — | 0.6876 | — | — | — | 5-Fold CV on Train |
| **Day 4 Tuned HistGradientBoosting** | — | — | — | 0.7150 | — | — | — | 5-Fold CV on Train (Winner) |
| **Day 4 Saved Artifact (Metadata)** | 0.8675 | 0.7122 | 0.7545 | 0.7327 | 0.9297 | 0.8330 | 0.0872 | Day 4 Holdout Test Record |
| **SELECTED Final Model (Pipeline @ 0.40)**| **0.8675** | **0.7132** | **0.7519** | **0.7321** | **0.9297** | **0.8335** | **0.0871** | **Untouched Holdout Test (Production)** |

---

## 🛠️ Verification & Quality Assurance

To verify the integrity of all artifacts, run:
```bash
python -c "import joblib, pandas as pd; a = joblib.load('final_model.joblib'); print('Model loaded successfully! Steps:', list(a['pipeline'].calibrated_classifiers_[0].estimator.named_steps))"
```

# Web3Geeks Machine Learning Internship — Week 1 Day 4
**Repository:** [internship_web3geeks](https://github.com/armishiqbal/internship_web3geeks)  
**Deliverable Notebook:** [`day4.ipynb`](day4.ipynb)  
**Deliverable Model Artifact:** [`day4_final_adult_income_model.joblib`](day4_final_adult_income_model.joblib)  
**Deliverable Executive Report:** [`day4_tuning_report.pdf`](day4_tuning_report.pdf) *(Strictly 2 Pages)*  
**Comprehensive Markdown Report:** [`TUNING_REPORT.md`](TUNING_REPORT.md)

---

## 📁 Day 4 Directory Structure

```text
├── week 1/
│   └── day 4/
│       ├── day4.ipynb                               # ⭐ Primary Jupyter Notebook (Tasks 1–5 complete)
│       ├── day4_final_adult_income_model.joblib     # ⭐ Production Joblib Artifact (Pipeline + Threshold 0.40)
│       ├── day4_tuning_report.pdf                   # ⭐ Executive 2-Page Tuning Report (PDF Deliverable)
│       ├── TUNING_REPORT.md                         # ⭐ Comprehensive Markdown Tuning Report
│       ├── generate_day4_pdf.py                     # ReportLab Automated PDF Generation Script (Strictly 2 Pages)
│       ├── generate_day4_charts.py                  # High-Resolution Visualization Generation Script
│       ├── adults.csv                               # UCI Adult Dataset (32,561 rows)
│       ├── day4_task1_reproducible_pipeline.joblib  # Task 1 Baseline Pipeline Artifact
│       ├── day4_tuned_hgb_pipeline.joblib           # Task 2 Tuned HGB Pipeline Artifact
│       ├── task3_diagnostics_panel.png              # Learning Curve & Complexity Multi-Panel Plot
│       ├── task4_threshold_panel.png                # Calibration & Threshold Trade-off Multi-Panel Plot
│       ├── task5_performance_panel.png              # ROC, PR-Curve, and Confusion Matrix Multi-Panel Plot
│       ├── pdf_page_1.png                           # Page 1 Visual Inspection Snapshot
│       └── pdf_page_2.png                           # Page 2 Visual Inspection Snapshot
```

---

## 🎯 Day 4 Technical Summary & Tasks Overview

### Task 1: Leak-Free Reproducible Machine Learning Pipeline
- Preprocessing and feature engineering encapsulated inside `sklearn.pipeline.Pipeline` with `ColumnTransformer`.
- Numeric features ($12$): `SimpleImputer(median)` $\rightarrow$ `StandardScaler()`.
- Categorical features ($8$): `SimpleImputer(most_frequent)` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.
- All seeds fixed at `42`; $80/20$ stratified split ($N_{\text{train}}=26,048$, $N_{\text{test}}=6,513$).
- Artifact reload bitwise check: **PASS** ($100\%$ identical predictions).

### Task 2: Randomized Hyperparameter Search (5-Fold Stratified CV)
- **HistGradientBoosting:** Best CV $F_1 = \mathbf{0.7150}$ (`learning_rate=0.15`, `max_iter=300`, `max_leaf_nodes=31`, `min_samples_leaf=50`, `l2_regularization=5.0`).
- **Random Forest:** Best CV $F_1 = 0.6876$ (`n_estimators=100`, `max_depth=20`, `min_samples_leaf=5`, `max_features=0.5`).
- **Logistic Regression:** Best CV $F_1 = 0.6635$ (`penalty='l1'`, `C=7.5431`, `solver='saga'`).
- Gradient Boosting wins by **$+2.74\%$** over RF and **$+5.15\%$** over LR.

### Task 3: Overfitting & Underfitting Diagnostics
- **8-Point Learning Curve:** Train/validation gap starts at $35.81\%$ ($2,083$ samples) and contracts steadily to **$4.10\%$** ($20,838$ samples). Final Train $F_1 = 0.7513$, Final Val $F_1 = 0.7103$.
- **Diagnosis:** **`REASONABLY BALANCED`** (Clean generalization without runaway variance).
- **Complexity Sweeps:** Tree depth peaks at `max_leaf_nodes=31`; regularizer peaks at `min_samples_leaf=50`.

### Task 4: Probability Calibration & Classification Threshold Tuning
- **Sigmoid Calibration (`CalibratedClassifierCV`):** Out-of-fold Brier Score improves from $0.0904 \rightarrow \mathbf{0.0900}$.
- **Threshold Optimization:** Sweeping $0.10 \rightarrow 0.90$ reveals optimal cutoff at **$0.40$** ($F_1 = \mathbf{0.7248}$, Recall $= \mathbf{74.00\%}$).
- **Operational Lift over Default (0.50):** $+8.42\%$ Recall gain, $+528$ additional high earners captured, $-24.5\%$ reduction in false negatives, and $-16.1\%$ reduction in asymmetric business error penalties.

### Task 5: Final Evaluation & Reproducible Production Model Artifact
Evaluated once on untouched test set ($N=6,513$) with threshold $0.40$:
- **Accuracy:** **$86.75\%$**
- **Precision:** **$71.22\%$**
- **Recall:** **$75.45\%$**
- **$F_1$-Score:** **$0.7327$**
- **ROC-AUC:** **$0.9297$**
- **Brier Score:** **$0.0872$**
- **Average Precision:** **$0.8330$**
- **Test Confusion Matrix:** $\text{TN}=4,467$, $\text{FP}=478$, $\text{FN}=385$, $\text{TP}=1,183$.
- **Artifact:** [`day4_final_adult_income_model.joblib`](day4_final_adult_income_model.joblib) containing full calibrated pipeline, optimal threshold, feature metadata, and audit logs.

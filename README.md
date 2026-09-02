# Web3Geeks Machine Learning Internship

**Author:** Armish Iqbal  
**Repository:** [internship_web3geeks](https://github.com/armishiqbal/internship_web3geeks)  
**Dataset:** UCI Adult Census Income Dataset ($N = 32,561$)  
**Goal:** Predict high-income individuals ($>\$50\text{K}/\text{year}$) to optimize targeted marketing and credit screening campaigns while maintaining leak-free, reproducible machine learning workflows.

---

## 📁 Repository Structure

```text
internship_web3geeks/
├── week 1/
│   ├── day 1/
│   │   ├── day1.ipynb                 #  Primary Jupyter Notebook (Tasks 1–5 + EDA)
│   │   ├── day1_summary_report.pdf    #  1-Page PDF Summary Deliverable
│   │   ├── adults.csv                 # Canonical UCI Adult Census Dataset (32,561 records)
│   │   ├── eda_visualizations.png     # 4-Panel Exploratory Data Analysis Charts
│   │   ├── summary_table.csv          # Demographic Class Distribution Summary
│   │   ├── summary_table.txt          # Detailed Summary Tables
│   │   ├── .gitignore                 # Local ignore rules
│   │   └── README.md                  # Comprehensive Day 1 Documentation
│   ├── day 2/
│   │   ├── day2.ipynb                 # Primary Jupyter Notebook (Tasks 1–5 + Pipelines)
│   │   ├── day2_summary_report.pdf    # 2-Page Executive PDF Report
│   │   ├── README.md                  # 1–2 Page Markdown Write-Up
│   │   ├── pipeline.py                # Reusable Preprocessing & Model Factory Module
│   │   ├── adults.csv                 # Canonical UCI Adult Dataset
│   │   ├── day2_evaluation_curves.png # ROC & Precision-Recall Curves Plot
│   │   └── day2_confusion_matrices.png# Confusion Matrices Heatmap
│   └── day 3/
│       ├── day3.ipynb                 # Primary Jupyter Notebook (Tasks 1–5 + Ensembles)
│       ├── day3_summary_report.pdf    # 2-Page Executive PDF Report
│       ├── README.md                  # Comprehensive Day 3 Documentation & Benchmarks
│       ├── adults.csv                 # Canonical UCI Adult Census Dataset (32,561 records)
│       ├── task3_5fold_boxplots.png   # 5-Fold CV Metric Boxplots (Acc, ROC-AUC, F1)
│       ├── task4_10fold_roc_auc_boxplot.png # 10-Fold Paired ROC-AUC Boxplot
│       ├── task3_cv_results.csv       # 5-Fold Summary Metrics Table
│       ├── task4_statistical_comparison.csv # Statistical Hypothesis Test Results
│       └── task5_feature_selection_benchmark.csv # L1 Sparsity Benchmark
├── .gitignore                         # Repository-wide ignore rules
└── README.md                          # Top-level Repository Overview & Benchmarks
```

---

## 📊 Master Benchmark & Performance Summary

All models were evaluated across standardized stratified partitions and cross-validation regimes ($N = 32,561$, `random_state=42`):

| Stage | Model / Strategy | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | Generalization / Validation Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Day 1** | Majority-Class Baseline (`<=50K`) | $0.7591$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.5000$ | Trivial Baseline ($F_1 = 0$) |
| **Day 1** | Education Heuristic (`education_num >= 13`) | $0.7367$ | $0.4746$ | $0.4879$ | $0.4811$ | $0.6582$ | Simple Rule Baseline |
| **Day 2** | Decision Tree Classifier (Unpruned) | $0.8104$ | $0.5993$ | $0.6409$ | $0.6194$ | $0.7525$ | Severe Overfit ($-38.05\%$ F1 drop) |
| **Day 2** | Logistic Regression Pipeline ($L_2$) | $0.8558$ | $0.7406$ | $0.6173$ | $0.6734$ | $0.9078$ | Zero Overfitting (Linear Benchmark) |
| **Day 3** | Logistic Regression ($L_2$ + Eng. Features, 5-Fold CV) | $0.8512 \pm 0.0028$ | — | — | $0.6589 \pm 0.0073$ | $0.9055 \pm 0.0022$ | 5-Fold Stratified CV Baseline |
| **Day 3** | Random Forest ($100$ Trees, 5-Fold CV) | $0.8557 \pm 0.0041$ | — | — | $0.6775 \pm 0.0090$ | $0.9042 \pm 0.0032$ | Bagging Non-Linear Ensemble |
| **Day 3** | **HistGradientBoosting (5-Fold CV)** | **$0.8723 \pm 0.0032$** | — | — | **$0.7120 \pm 0.0070$** | **$0.9272 \pm 0.0016$** | **State-of-the-Art Champion ($p < 10^{-10}$)** |
| **Day 3** | **HistGradientBoosting ($L_1$ 49 Feats, 5-Fold CV)** | **$0.8739 \pm 0.0037$** | — | — | **$0.7143 \pm 0.0084$** | **$0.9283 \pm 0.0015$** | **$60\%$ Sparsity, Zero AUC Loss** |

---

## 🗓️ Weekly Progress & Milestones

### 🔹 [Week 1 Day 1: Problem Definition, EDA & Baselines](week%201/day%201/)
* **Problem Framing:** Defined binary target ($y=1$ if $>\$50\text{K}$, $y=0$ if $\le\$50\text{K}$) with a base rate of **$24.08\%$** ($7,841$ positive / $24,720$ negative).
* **Primary Metric:** Selected **Precision & $F_1$-Score** to minimize wasted outreach budget on non-high earners in premium customer acquisition.
* **Leakage-Free Partitioning:** Established reproducible 3-way stratified splits ($70\%$ Train: $22,792$ / $10\%$ Dev: $3,256$ / $20\%$ Test: $6,513$).
* **Baselines:** Majority baseline ($F_1 = 0.0000$) vs. Single-Feature Education Heuristic ($F_1 = 0.4811$).
* **Deliverables:** [`day1.ipynb`](week%201/day%201/day1.ipynb), [`day1_summary_report.pdf`](week%201/day%201/day1_summary_report.pdf), and [`README.md`](week%201/day%201/README.md).

### 🔹 [Week 1 Day 2: Supervised Pipelines & Interpretability](week%201/day%202/)
* **Mixed-Type ColumnTransformer:**
  * **Numeric Pipeline (6 features):** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()` (immune to capital gain outliers up to $\$99,999$).
  * **Categorical Pipeline (8 features):** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')` ($105$ encoded columns).
* **Supervised Models in Pipelines:**
  * **Logistic Regression:** $L_2$ regularized (`lbfgs`, $C=1.0$, `max_iter=1000`, `random_state=42`). Achieved **$85.58\%$ Accuracy**, **$74.06\%$ Precision**, **$0.6734$ $F_1$**, and **$0.9078$ ROC-AUC**.
  * **Decision Tree:** Diagnosed severe overfitting (Depth $52$, $3,815$ leaves, Train $100\%$ vs Test $81.04\%$).
* **Interpretability Check:**
  * Top Positive Weights: `capital_gain` ($+2.25$, $9.47\times$ odds), `marital_status_Married-civ-spouse` ($+1.38$, $3.97\times$ odds), `relationship_Wife` ($+1.07$).
  * Top Negative Weights: `occupation_Priv-house-serv` ($-1.38$, $0.25\times$ odds), `relationship_Own-child` ($-1.17$).
  * Decision Tree Root Splits: `Married-civ-spouse` $\rightarrow$ `capital_gain` $\rightarrow$ `education_num` (Bachelors+).
* **Deliverables:** [`day2.ipynb`](week%201/day%202/day2.ipynb), [`day2_summary_report.pdf`](week%201/day%202/day2_summary_report.pdf), [`README.md`](week%201/day%202/README.md), and reusable [`pipeline.py`](week%201/day%202/pipeline.py).

### 🔹 [Week 1 Day 3: Principled Feature Engineering & Cross-Validation](week%201/day%203/)
* **Engineered Feature Space (8 Features):**
  * **Lifecycle & Intensity Bins:** `age_bucket` (6 brackets, $\text{MI} = 0.0631$) and `hours_bin` (5 brackets, $\text{MI} = 0.0385$).
  * **Asset Transformations:** `has_capital_gain` ($\text{MI} = 0.0323$), `log_capital_gain` ($\text{MI} = 0.0824$, $2.5\times$ higher signal than binary flag), and `has_capital_loss` ($\text{MI} = 0.0098$).
  * **Credential & Demographic Flags:** `is_higher_ed` ($\text{MI} = 0.0494$) & `is_married` ($\text{MI} = 0.1105$, highest individual signal).
  * **Multiplicative Interaction:** `edu_x_hours` ($\text{MI} = 0.0840$, education_num $\times$ hours_per_week).
* **Cross-Validation Benchmarking:** 5-Fold Stratified CV confirmed HistGradientBoosting ($0.9272 \pm 0.0016$ ROC-AUC, $0.7120 \pm 0.0070$ F1) strictly dominates Logistic Regression ($0.9055$) and Random Forest ($0.9042$).
* **Statistical Hypothesis Testing:** 10-Fold Paired $t$-test ($t = 33.45$, $p = 9.41 \times 10^{-11}$) and Wilcoxon signed-rank test ($W = 0.0$, $p = 0.00195$) proved decisive statistical superiority with a 10–0 win rate.
* **L1 Sparsity Benchmark:** Pruning $60.2\%$ of features ($123 \rightarrow 49$) achieved zero AUC degradation ($0.9283$), while aggressive pruning to $18$ features delivered a $2.75\times$ inference speedup.
* **Deliverables:** [`day3.ipynb`](week%201/day%203/day3.ipynb), [`day3_summary_report.pdf`](week%201/day%203/day3_summary_report.pdf), and [`README.md`](week%201/day%203/README.md).

---

## 💻 How to Run the Project

### 1. Environment Setup
Clone the repository and install required scientific computing packages:
```bash
git clone https://github.com/armishiqbal/internship_web3geeks.git
cd internship_web3geeks

pip install numpy pandas scikit-learn matplotlib seaborn jupyter reportlab
```

### 2. Run Interactive Jupyter Notebooks
Launch any notebook directly:
```bash
# Day 1: EDA, Stratified Splitting & Baseline Analysis
jupyter notebook "week 1/day 1/day1.ipynb"

# Day 2: Supervised Pipelines, Metric Evaluation & Interpretability
jupyter notebook "week 1/day 2/day2.ipynb"

# Day 3: Feature Engineering, Cross-Validation & Statistical Testing
jupyter notebook "week 1/day 3/day3.ipynb"
```

### 3. Import Reusable Preprocessing Pipeline
```python
import sys
sys.path.append('week 1/day 2')
from pipeline import get_column_preprocessor, get_pipeline

# Obtain fully configured Scikit-Learn pipeline
pipeline = get_pipeline(model_type='logistic_regression', C=1.0, random_state=42)
```

---

## 👥 Author
* **Armish Iqbal** — [GitHub Profile](https://github.com/armishiqbal)
* **Organization:** Web3Geeks AI/ML Internship Program

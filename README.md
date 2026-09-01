# Web3Geeks AI / ML Internship

Welcome to the **Web3Geeks Machine Learning & AI Internship** repository. This repository tracks daily problem-solving tasks, supervised machine learning experiments, and deliverables.

---

## 📂 Repository Structure

```text
internship_web3geeks/
├── .gitignore                                       # Root Git ignore rules
├── README.md                                        # Main Repository Overview
├── week 1/
│   ├── day 1/
│   │   ├── day1_census_income_classification.ipynb  # Unified Day 1 Jupyter Notebook
│   │   ├── adults.csv                               # UCI Adult census dataset (32,561 rows)
│   │   ├── eda_visualizations.png                   # 4-panel EDA visualization plots
│   │   ├── summary_table.csv                        # Class distribution table
│   │   ├── summary_table.txt                        # Demographic summary tables
│   │   ├── task1.py                                 # Task 1: Problem Definition & Base Rate (24.08%)
│   │   ├── task2.py                                 # Task 2: Data Loading, Cleaning & Full EDA
│   │   ├── task3.py                                 # Task 3: Reproducible Stratified Splits (70/10/20)
│   │   ├── task4.py                                 # Task 4: Simple Baselines (F1: 0.4811 vs 0.00)
│   │   ├── task5.py                                 # Task 5: Error Analysis & Feature Roadmap
│   │   └── README.md                                # Detailed Day 1 Report & Metrics
│   └── day 2/
│       ├── day2.ipynb                               # Complete Day 2 Jupyter Notebook
│       ├── adults.csv                               # UCI Adult census dataset (32,561 rows)
│       ├── day2_evaluation_curves.png               # High-Res ROC & PR Curves
│       ├── day2_confusion_matrices.png              # Confusion Matrix Heatmaps
│       ├── task1.py                                 # Task 1: ColumnTransformer Preprocessing Pipeline
│       ├── task2.py                                 # Task 2: Supervised Pipelines (LR & DT)
│       ├── task3.py                                 # Task 3: Multi-Metric Evaluation & Comparison
│       ├── task4.py                                 # Task 4: Interpretability Check (Weights & Splits)
│       ├── task5.py                                 # Task 5: Model Selection Write-Up for Day 3
│       ├── pipeline.py                              # Reusable Pipeline & Model Factory Module
│       └── README.md                                # 1–2 Page Executive Report & Analysis
```

---

## 🗓️ Weekly Overview

### [Week 1 — Day 1: Foundations & Problem Framing](week%201/day%201/README.md)
* **Notebook:** [`day1_census_income_classification.ipynb`](week%201/day%201/day1_census_income_classification.ipynb)
* **Problem Definition:** Positive class ($>\$50	ext{K}$) base rate = **$24.08\%$**. Prioritized Precision for expensive outbound sales advisory.
* **Stratified Splits:** $70\%$ Train ($22,792$) / $10\%$ Dev ($3,256$) / $20\%$ Test ($6,513$).
* **Baselines:** Majority Class ($0.00$ $F_1$) vs Single-Feature Heuristic (**Precision: $47.46\%$, Recall: $48.79\%$, $F_1 = 0.4811$**).

### [Week 1 — Day 2: Supervised Pipelines & Model Interpretability](week%201/day%202/README.md)
* **Notebook:** [`day2.ipynb`](week%201/day%202/day2.ipynb)
* **Preprocessing:** Leak-free `ColumnTransformer` with `SimpleImputer(median)` $ightarrow$ `StandardScaler` for numerics, and `SimpleImputer(most_frequent)` $ightarrow$ `OneHotEncoder(handle_unknown='ignore')` for categoricals.
* **Model Training:** End-to-end pipelines for **Regularized Logistic Regression ($L_2$)** and **Decision Tree Classifier**.
* **Hold-Out Benchmark Results:**
  * **Logistic Regression:** **Accuracy: $85.58\%$**, **Precision: $74.06\%$**, **Recall: $61.73\%$**, **$F_1	ext{-Score} = 0.6734$**, **$	ext{ROC-AUC} = 0.9078$**, **$	ext{PR-AUC} = 0.7732$** (Zero overfitting).
  * **Decision Tree (Unpruned):** Accuracy: $81.04\%$, Precision: $59.93\%$, Recall: $64.09\%$, $F_1 = 0.6194$ (Severe overfitting: Depth $52$, Train $F_1 = 0.9999$).
* **Interpretability:** Extracted top $10$ positive/negative coefficients (capital gains, marital status, executive management vs private domestic service, young dependents) and top $3$ tree root splits.
* **Model Selection:** Advanced Logistic Regression as primary linear baseline and selected Tree Ensembles (Random Forest / Gradient Boosting) for Day 3 non-linear development.

---

## 🚀 Getting Started

### Clone Repository
```bash
git clone https://github.com/armishiqbal/internship_web3geeks.git
cd internship_web3geeks
```

### Running Day 2 Deliverables
```bash
# Launch Jupyter Notebook
jupyter notebook "week 1/day 2/day2.ipynb"

# Or run standalone Python scripts
cd "week 1/day 2"
python task1.py
python task2.py
python task3.py
python task4.py
python task5.py
```

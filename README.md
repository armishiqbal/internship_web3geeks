# Web3Geeks AI / ML Internship

Welcome to the **Web3Geeks Machine Learning & AI Internship** repository. This repository tracks daily problem-solving tasks, machine learning experiments, and deliverables.

---

## 📂 Repository Structure

```text
internship_web3geeks/
├── .gitignore                                       # Root Git ignore rules
├── README.md                                        # Main Repository Overview
└── week 1/
    └── day 1/
        ├── day1_census_income_classification.ipynb  # Unified Jupyter Notebook (All 5 Tasks)
        ├── adults.csv                               # Raw UCI Adult census dataset (32,561 rows)
        ├── eda_visualizations.png                   # 4-panel EDA visualization plots
        ├── summary_table.csv                        # Summary class counts & rates
        ├── summary_table.txt                        # Detailed demographic summary table
        ├── task1.py                                 # Task 1: Problem Definition & Base Rate (24.08%)
        ├── task2.py                                 # Task 2: Data Loading, Cleaning & Full EDA
        ├── task3.py                                 # Task 3: Reproducible Stratified Splits (70/10/20)
        ├── task4.py                                 # Task 4: Simple Baselines (F1: 0.4811 vs 0.00)
        ├── task5.py                                 # Task 5: Error Analysis & Day 2 Feature Roadmap
        ├── .gitignore                               # Day 1 local gitignore
        └── README.md                                # Detailed Day 1 Report & Metrics
```

---

## 🗓️ Weekly Overview

### [Week 1: Foundations & Census Income Classification](week%201/day%201/README.md)
* **Day 1: Census Income Classification (>50K)**
  * **Unified Notebook:** [`day1_census_income_classification.ipynb`](week%201/day%201/day1_census_income_classification.ipynb)
  * **Problem Definition:** High-income target identification ($>\$50\text{K}$) with a **$24.08\%$ base rate** prioritizing Precision for costly outreach.
  * **EDA & Cleaning:** Cleaned `'?'` $\rightarrow$ `NaN`, demographic summaries, and 4-panel EDA visualizations.
  * **Stratified Splits:** $70\%$ Train ($22,792$) / $10\%$ Dev ($3,256$) / $20\%$ Hold-Out Test ($6,513$) with fixed seed (`42`).
  * **Baseline Benchmarks:** Majority class ($75.91\%$ accuracy, $0.00$ $F_1$) vs Single-feature education rule (**$47.46\%$ Precision, $48.79\%$ Recall, $F_1 = 0.4811$, $\text{PR-AUC} = 0.3548$**).
  * **Error Analysis:** Diagnosed False Positives (junior graduates) and False Negatives (skilled trade overtime workers), creating a 5-point feature engineering roadmap for Day 2 to target **$F_1 > 0.65$**.

---

## 🚀 Getting Started

### Clone Repository
```bash
git clone https://github.com/armishiqbal/internship_web3geeks.git
cd internship_web3geeks
```

### Running Day 1 Tasks

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook "week 1/day 1/day1_census_income_classification.ipynb"
```

#### Option 2: Standalone Python Scripts
```bash
cd "week 1/day 1"

python task1.py
python task2.py
python task3.py
python task4.py
python task5.py
```

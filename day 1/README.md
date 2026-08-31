# Week 1 - Day 1: Census Income Prediction (>50K)
**UCI Adult Dataset — Problem Definition, EDA, Reproducible Splits, Baselines & Error Analysis**

---

## 📌 Project Overview
This repository contains the complete implementation for **Week 1 - Day 1** of the machine learning internship. The objective is to predict whether an individual earns over **$50,000/year** using census demographic and occupational attributes.

---

## 📁 Repository Structure

```text
├── .gitignore
├── README.md
├── adults.csv               # Raw UCI Adult dataset
├── summary_table.csv        # Summary counts and class rates
├── summary_table.txt        # Detailed text summary table
├── eda_visualizations.png   # 4-panel EDA visualization plots
├── task1.py                 # Task 1: Problem Definition & Base Rate
├── task2.py                 # Task 2: Data Load, Cleaning & Full EDA
├── task3.py                 # Task 3: Reproducible Train/Dev/Test Splits
├── task4.py                 # Task 4: Simple Baselines & Evaluation
└── task5.py                 # Task 5: Error Analysis & Metric Optimization
```

---

## 🚀 Tasks Summary

### Task 1: Problem Definition & Success Metric
* **Target ($y = 1$):** `income > 50K` (Positive class).
* **Class Base Rate:**
  * **Negative (`<=50K`):** 24,720 rows (**75.92%**)
  * **Positive (`>50K`):** 7,841 rows (**24.08%**)
* **Business Objective:** Targeted premium wealth advisory and financial services marketing outreach.
* **Primary Metric:** **Precision / Precision@Top-$k$** (backed by **$F_1$-score** across threshold sweeps).

---

### Task 2: Data Load & Exploratory Data Analysis (EDA)
* Handled missing value tokens (`'?'` $\rightarrow$ `NaN`) across `workclass`, `occupation`, and `native_country`.
* Generated summary statistics across numerical features and categorical distributions.
* Generated **`eda_visualizations.png`** covering class distribution, age histograms, education rate comparisons, and hours worked.

---

### Task 3: Reproducible Stratified Splits (70 / 10 / 20)
* **Training Set (70%):** 22,792 samples (24.08% positive rate)
* **Dev/Validation Set (10%):** 3,256 samples (24.08% positive rate)
* **Hold-Out Test Set (20%):** 6,513 samples (24.07% positive rate)
* Stratified on target with fixed `random_state=42`.

---

### Task 4: Simple Baselines Evaluation (on Hold-Out Test Set)

| Baseline Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | PR AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Majority-Class (Always $\le 50\text{K}$)** | 75.93% | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2407 |
| **2. Single-Feature Rule (`education_num` $\ge 13$)** | **78.18%** | **0.5874** | **0.4656** | **0.5197** | **0.6740** | **0.4021** |

---

### Task 5: Initial Error Analysis & Next Steps
* **False Positives (847):** Younger graduates in entry-level positions or individuals working part-time hours.
* **False Negatives (803):** Senior tradespeople and managers (`Craft-repair`, `Exec-managerial`) without formal 4-year degrees who earn $>50\text{K}$ via overtime or career seniority.
* **Key Fixes for Day 2:**
  1. $\log(1 + x)$ transforms and binary flags for skewed `capital_gain` / `capital_loss`.
  2. Imputing `'Missing'` token for categorical nulls.
  3. Feature interaction terms ($\text{age} \times \text{education\_num}$, $\text{age} \times \text{hours\_per\_week}$).
  4. Marital status consolidation.

---

## 💻 How to Run

```bash
# Task 1: Problem Definition & Base Rate
python task1.py

# Task 2: EDA & Visualization Generation
python task2.py

# Task 3: Reproducible Splits
python task3.py

# Task 4: Baseline Models Evaluation
python task4.py

# Task 5: Error Analysis
python task5.py
```

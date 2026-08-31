# Web3Geeks Machine Learning Internship
**Repository:** [internship_web3geeks](https://github.com/armishiqbal/internship_web3geeks)

---

## 📁 Repository Structure

```text
├── day 1/
│   ├── adults.csv               # Raw UCI Adult census dataset
│   ├── eda_visualizations.png   # 4-panel EDA visualization plots
│   ├── summary_table.csv        # Summary class counts & rates
│   ├── summary_table.txt        # Detailed summary table
│   ├── task1.py                 # Task 1: Problem Definition & Base Rate (24.08%)
│   ├── task2.py                 # Task 2: Data Loading, Cleaning & Full EDA
│   ├── task3.py                 # Task 3: Reproducible Stratified Splits (70/10/20)
│   ├── task4.py                 # Task 4: Simple Baselines (F1: 0.5197 vs 0.00)
│   ├── task5.py                 # Task 5: Error Analysis & Day 2 Feature Roadmap
│   └── README.md                # Detailed Day 1 Report & Metrics
├── .gitignore
└── README.md
```

---

## 📅 Day 1: Census Income Prediction (>50K) — Overview

* **Task 1 (Problem Definition & Metrics):** Defined the positive class ($>\$50\text{K}$) with a **24.08% base rate** and selected **Precision / $F_1$** as primary metrics for targeted financial outreach.
* **Task 2 (Data Load & EDA):** Cleaned missing values (`'?'` $\rightarrow$ `NaN`), calculated demographic summaries, and generated 4-panel EDA charts (`day 1/eda_visualizations.png`).
* **Task 3 (Reproducible Splits):** Stratified the 32,561 records into **70% Train ($22,792$) / 10% Dev ($3,256$) / 20% Hold-Out Test ($6,513$)** with a fixed seed (`42`).
* **Task 4 (Baselines & Evaluation):** Benchmarked a majority predictor ($75.93\%$ accuracy, $0.00$ $F_1$) against a single-feature education rule (**$58.74\%$ Precision, $46.56\%$ Recall, $F_1 = 0.5197$**).
* **Task 5 (Error Analysis & Roadmap):** Identified failure modes on junior graduates and skilled trade workers, planning Day 2 feature engineering (log transforms, binary capital flags, seniority interaction terms) to target **$F_1 > 0.65$**.

---

## 💻 How to Run Day 1 Tasks

```bash
# Navigate to day 1 folder
cd "day 1"

# Execute tasks
python task1.py
python task2.py
python task3.py
python task4.py
python task5.py
```

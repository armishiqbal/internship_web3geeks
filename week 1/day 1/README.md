# Web3Geeks Machine Learning Internship — Week 1 Day 1
**Repository:** [internship_web3geeks](https://github.com/armishiqbal/internship_web3geeks)

---

## 📁 Week 1 / Day 1 Structure

```text
├── week 1/
│   └── day 1/
│       ├── day1_census_income_classification.ipynb  # Unified Jupyter Notebook (All 5 Tasks + Visuals)
│       ├── adults.csv                               # UCI Adult census dataset (32,561 rows)
│       ├── eda_visualizations.png                   # 4-panel EDA visualization plots
│       ├── summary_table.csv                        # Summary class counts & rates
│       ├── summary_table.txt                        # Detailed demographic summary tables
│       ├── task1.py                                 # Task 1: Problem Definition & Base Rate (24.08%)
│       ├── task2.py                                 # Task 2: Data Loading, Cleaning & Full EDA
│       ├── task3.py                                 # Task 3: Reproducible Stratified Splits (70/10/20)
│       ├── task4.py                                 # Task 4: Simple Baselines (F1: 0.4811 vs 0.00)
│       ├── task5.py                                 # Task 5: Error Analysis & Day 2 Feature Roadmap
│       ├── .gitignore                               # Day 1 local gitignore
│       └── README.md                                # Detailed Day 1 Report & Metrics
├── .gitignore                                       # Root Git ignore rules
└── README.md                                        # Root Repository Overview
```

---

## 📊 Dataset Scope Note

The classic **UCI Adult Census Income dataset** consists of two components:
1. `adult.data` — **32,561 records** (the canonical training benchmark used here)
2. `adult.test` — **16,281 records** (bringing the total combined corpus to $48,842$ records)

In Day 1, we conduct our analysis on the canonical **32,561-row** dataset, establishing clean, reproducible 3-way stratified splits ($70\%$ Train / $10\%$ Dev / $20\%$ Hold-Out Test) to avoid data leakage.

---

## 📅 Task Summary & Computed Metrics

### 🔹 Task 1: Problem Definition & Class Base Rate
* **Target Variable:** Binary income classification ($y=1$ if $>\$50\text{K}$, $y=0$ if $\le\$50\text{K}$).
* **Class Distribution:** Positive ($>\$50\text{K}$): $7,841$ ($24.08\%$) | Negative ($\le\$50\text{K}$): $24,720$ ($75.92\%$).
* **Primary Success Metric:** **Precision / $F_1$-Score** (Prioritizing high-conversion conversion rate on expensive direct customer outreach over sheer recall).

### 🔹 Task 2: Data Cleaning & Exploratory Data Analysis
* Missing value handling: Converted `'?'` markers to `NaN` ($1,836$ in `workclass`, $1,843$ in `occupation`, $583$ in `native_country`).
* Generated 4-panel high-resolution visualizations (`eda_visualizations.png`):
  1. Target class imbalance distribution.
  2. Age distributions stratified by income class (peak earning ages: $38-52$).
  3. High-income rate by education level ($73.44\%$ for Doctorate, $72.84\%$ for Prof-school, $41.48\%$ for Bachelors vs $15.83\%$ for HS-grad).
  4. Weekly working hours distribution by income ($40\text{ hrs}$ mode, overtime positively correlating with $>50\text{K}$).

### 🔹 Task 3: Reproducible Stratified Splits
* **Methodology:** Two-stage stratified split (`random_state=42`) preserves the exact $24.08\%$ class ratio:
  * **Train Set (70%):** $22,792$ rows ($5,488$ positive, $24.08\%$)
  * **Dev/Validation Set (10%):** $3,256$ rows ($784$ positive, $24.08\%$)
  * **Hold-Out Test Set (20%):** $6,513$ rows ($1,569$ positive, $24.09\%$)
* **Data Leakage Safeguard:** The hold-out test set remains strictly isolated until final model evaluation.

### 🔹 Task 4: Baseline Models & Evaluation Metrics

Evaluated on the $6,513$-sample hold-out test set:

| Baseline Model | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Majority-Class (`Always <=50K`)** | $0.7591$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.2409$ | $0.2409$ |
| **2. Single-Feature (`education_num >= 13`)** | $0.7367$ | **$0.4746$** | **$0.4879$** | **$0.4811$** | **$0.6582$** | **$0.3548$** |

* **Utility Benchmark:** A real Machine Learning model must surpass the single-feature heuristic ($F_1 = 0.4811$) by reaching **$F_1 > 0.65$** ($\sim 35\%$ relative gain), $\text{PR-AUC} > 0.60$, and Precision $> 75\%$ at $\ge 60\%$ Recall.

### 🔹 Task 5: Error Analysis & Day 2 Feature Engineering Roadmap
* **False Positives ($847$ instances, $13.00\%$ of test set):** Younger graduates ($<30$ yrs old) and part-time degree holders not yet earning senior wages.
* **False Negatives ($804$ instances, $12.34\%$ of test set):** Senior skilled trades (`Craft-repair`, `Exec-managerial`) without degrees, high overtime workers ($45-60+$ hrs), and individuals with substantial capital gains.
* **Day 2 Feature Roadmap:**
  1. `capital_gain` / `capital_loss` log transforms (`log1p`) and binary indicator flags (`has_capital_gain`).
  2. Explicit `'Unknown'` category imputation for missing categorical features.
  3. Career seniority interaction terms (`age * education_num` and `age * hours_per_week`).
  4. Marital status consolidation (`Married-civ-spouse` high-income signal).
  5. High-cardinality country/occupation frequency/regional grouping.

---

## 💻 How to Run

### Option A: Run Unified Jupyter Notebook
```bash
jupyter notebook "week 1/day 1/day1_census_income_classification.ipynb"
```

### Option B: Execute Modular Python Scripts
```bash
cd "week 1/day 1"

python task1.py
python task2.py
python task3.py
python task4.py
python task5.py
```

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
│   └── day 2/
│       ├── day2.ipynb                 #  Primary Jupyter Notebook (Tasks 1–5 + Pipelines)
│       ├── day2_summary_report.pdf    #  2-Page Executive PDF Report
│       ├── README.md                  #  1–2 Page Markdown Write-Up
│       ├── pipeline.py                # Reusable Preprocessing & Model Factory Module
│       ├── adults.csv                 # Canonical UCI Adult Dataset
│       ├── day2_evaluation_curves.png # ROC & Precision-Recall Curves Plot
│       └── day2_confusion_matrices.png# Confusion Matrices Heatmap
├── .gitignore                         # Repository-wide ignore rules
└── README.md                          # Top-level Repository Overview & Benchmarks
```

---

## 📊 Master Benchmark & Performance Summary

All models were evaluated on the strictly isolated **$6,513$-instance hold-out test set** ($20\%$ stratified partition, `random_state=42`):

| Stage | Model / Strategy | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC | Generalization Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Day 1** | Majority-Class Baseline (`<=50K`) | $0.7591$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.2409$ | $0.2409$ | Trivial ($F_1 = 0$) |
| **Day 1** | Education Heuristic (`education_num >= 13`) | $0.7367$ | $0.4746$ | $0.4879$ | $0.4811$ | $0.6582$ | $0.3548$ | Simple Rule Baseline |
| **Day 2** | Decision Tree Classifier (Unpruned) | $0.8104$ | $0.5993$ | $0.6409$ | $0.6194$ | $0.7525$ | $0.4706$ | Severe Overfit ($-38.05\%$ F1 drop) |
| **Day 2** | **Logistic Regression Pipeline ($L_2$)** | **$0.8558$** | **$0.7406$** | **$0.6173$** | **$0.6734$** | **$0.9078$** | **$0.7732$** | **Zero Overfitting (Superior)** |

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

---

## 💻 How to Run the Project

### 1. Environment Setup
Clone the repository and install required scientific computing packages:
```bash
git clone https://github.com/armishiqbal/internship_web3geeks.git
cd internship_web3geeks

pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 2. Run Interactive Jupyter Notebooks
Launch either notebook directly:
```bash
# Day 1: EDA, Stratified Splitting & Baseline Analysis
jupyter notebook "week 1/day 1/day1.ipynb"

# Day 2: Supervised Pipelines, Metric Evaluation & Interpretability
jupyter notebook "week 1/day 2/day2.ipynb"
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

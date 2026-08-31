# Web3Geeks AI / ML Internship

Welcome to the **Web3Geeks Machine Learning & AI Internship** repository. This repository tracks daily problem-solving tasks, machine learning experiments, and deliverables.

---

## 📁 Repository Structure

```text
internship_web3geeks/
├── .gitignore
├── README.md
└── week 1/
    └── day 1/
        ├── adults.csv               # Raw UCI Adult census dataset
        ├── eda_visualizations.png   # 4-panel EDA visualization plots
        ├── summary_table.csv        # Summary class counts & rates
        ├── summary_table.txt        # Detailed summary table
        ├── task1.py                 # Task 1: Problem Definition & Base Rate (24.08%)
        ├── task2.py                 # Task 2: Data Loading, Cleaning & Full EDA
        ├── task3.py                 # Task 3: Reproducible Stratified Splits (70/10/20)
        ├── task4.py                 # Task 4: Simple Baselines (F1: 0.5197 vs 0.00)
        ├── task5.py                 # Task 5: Error Analysis & Day 2 Feature Roadmap
        └── README.md                # Detailed Day 1 Report & Metrics
```

---

## 🗓️ Weekly Overview

### [Week 1: Foundations & Census Income Classification](week%201/day%201/README.md)
* **Day 1:** Census Income Classification (>50K)
  * Problem Definition & Base Rate (24.08%)
  * Data Loading, Cleaning (`?` -> `NaN`), and EDA Visualizations
  * Reproducible Stratified Splits (70% Train, 10% Dev, 20% Test)
  * Baseline Models & Metric Evaluation (Precision, Recall, F1)
  * Error Analysis & Next-Day Roadmap

---

## 🚀 Getting Started

### Clone Repository
```bash
git clone https://github.com/armishiqbal/internship_web3geeks.git
cd internship_web3geeks
```

### Running Week 1 - Day 1 Tasks
```bash
cd "week 1/day 1"

# Run tasks sequentially
python task1.py
python task2.py
python task3.py
python task4.py
python task5.py
```

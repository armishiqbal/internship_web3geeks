# Web3Geeks Machine Learning Internship — Week 1 Day 2
**Repository:** [internship_web3geeks](https://github.com/armishiqbal/internship_web3geeks)  
**Deliverable Notebook:** [`day2.ipynb`](day2.ipynb)

---

## 📁 Day 2 Directory Structure

```text
├── week 1/
│   └── day 2/
│       ├── day2.ipynb                               # ⭐ Primary Jupyter Notebook Deliverable
│       ├── day2_summary_report.pdf                  # ⭐ 2-Page Executive PDF Report
│       ├── README.md                                # ⭐ 1–2 Page Markdown Write-Up
│       ├── pipeline.py                              # Reusable Preprocessing & Model Pipeline (Task 5)
│       ├── adults.csv                               # Canonical UCI Adult Dataset (32,561 rows)
│       ├── day2_evaluation_curves.png               # ROC & Precision-Recall Curves
│       └── day2_confusion_matrices.png              # Heatmaps for Logistic Regression & Tree
```

---

## 🛠️ Task 1: Preprocessing Plan & Pipeline Architecture

Tabular census data exhibits heterogeneous data types, skewed monetary distributions, and missing values. We explicitly categorize and transform our feature space using `sklearn.compose.ColumnTransformer`:

### 1. Feature Partitioning
* **Numerical Features ($6$):** `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`
* **Categorical Features ($8$):** `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`

### 2. Preprocessing Rationale & Alternatives
* **Numeric Pipeline (`SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`):**
  * *Why Median Imputation:* `capital_gain` and `capital_loss` have extreme zero-inflation ($>90\%$ zeros) and large positive spikes (up to $\$99,999$). Median imputation is immune to extreme skewness, whereas mean imputation artificially shifts imputed values into unrepresentative non-zero values.
  * *Why Standard Scaling:* Shifts features to $\mu=0, \sigma=1$. This ensures that gradient updates in Logistic Regression are well-conditioned and that $L_2$ regularization penalties apply proportionally across features rather than being dominated by high-variance columns like `fnlwgt`.
  * *Alternatives Skipped:* RobustScaler / QuantileTransformer (kept for non-linear Day 3 experiments to preserve initial baseline simplicity).
* **Categorical Pipeline (`SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`):**
  * *Why One-Hot Encoding:* Encodes nominal categories as orthogonal binary indicators. `handle_unknown='ignore'` provides robust production inference by encoding unseen future levels as all-zeros without crashing.
  * *Alternatives Skipped:* Ordinal / Label Encoding was rejected because nominal features (e.g., `occupation`, `workclass`) have no natural mathematical hierarchy; integer encoding would inject false geometric distance into linear decision boundaries. Target Encoding was avoided to eliminate any risk of target leakage on rare categories.

---

## 🤖 Task 2: Supervised Model Pipelines

We packaged two distinct supervised algorithms into leak-free `sklearn.pipeline.Pipeline` workflows:
1. **Regularized Logistic Regression:** Linear classifier optimized via `solver='lbfgs'`, $L_2$ regularization ($C=1.0$), and `max_iter=1000`.
2. **Decision Tree Classifier:** Non-linear recursive partitioning tree (`random_state=42`).

> 🔒 **Zero Data Leakage Guarantee:** Preprocessor transformers and classifiers were fitted strictly on `(X_train, y_train)` ($N=26,048$). The hold-out test set ($N=6,513$) was untouched until final inference.

---

## 📊 Task 3: Comprehensive Hold-Out Test Evaluation

Evaluated across the $6,513$-instance hold-out test set ($20\%$ stratified split):

| Model / Pipeline | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Day 1: Majority-Class Baseline (`<=50K`)** | $0.7591$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.5000$ | $0.2409$ |
| **Day 1: Single-Feature Heuristic (`education >= 13`)** | $0.7367$ | $0.4746$ | $0.4879$ | $0.4811$ | $0.6582$ | $0.3548$ |
| **Day 2: Decision Tree Classifier (Unpruned)** | $0.8104$ | $0.5993$ | $0.6409$ | $0.6194$ | $0.7525$ | $0.4706$ |
| **Day 2: Logistic Regression Pipeline ($L_2$)** | **$0.8558$** | **$0.7406$** | **$0.6173$** | **$0.6734$** | **$0.9078$** | **$0.7732$** |

### 🔍 Error Type Analysis & Business Outreach Decision
* **Logistic Regression:**
  * True Negatives (TN): $4,606$ | False Positives (FP): **$339$** ($5.20\%$)
  * False Negatives (FN): **$600$** ($9.21\%$) | True Positives (TP): **$968$** ($14.86\%$)
  * *Business Impact:* False Negatives ($600$) are $1.77\times$ more frequent than False Positives ($339$). Because high-touch customer outreach carries high variable costs per contact, keeping False Positives low yields a high **$74.06\%$ Precision**—ensuring $\sim 3$ out of every $4$ contacted individuals are genuine high earners.
* **Decision Tree:**
  * High false alarm count ($FP = 672$) due to overfitted terminal leaves, lowering precision to $59.93\%$.

---

## 🔎 Task 4: Interpretability Check

### 1. Logistic Regression Coefficients
Extracted via `get_feature_names_out()` and sorted by magnitude:

* **Top 10 Positive Coefficients (Pushes towards $>50\text{K}$):**
  1. `capital_gain` ($+2.2477$, Odds Ratio: $9.47\times$): Massive independent predictor of wealth.
  2. `marital_status_Married-civ-spouse` ($+1.3794$, Odds Ratio: $3.97\times$): Dual-income and career stability.
  3. `relationship_Wife` ($+1.0736$, Odds Ratio: $2.93\times$): High-income concentration in married dual-earners.
  4. `occupation_Exec-managerial` ($+0.7926$, Odds Ratio: $2.21\times$): Senior management compensation.
  5. `education_num` ($+0.7424$, Odds Ratio: $2.10\times$): Each $+1\sigma$ education doubles odds of $>50\text{K}$.
  6. `occupation_Tech-support` ($+0.6638$, Odds Ratio: $1.94\times$): High-demand technical specialized labor.

* **Top 10 Negative Coefficients (Pushes towards $\le 50\text{K}$):**
  1. `occupation_Priv-house-serv` ($-1.3825$, Odds Ratio: $0.25\times$): Low wage ceilings in private household service.
  2. `relationship_Own-child` ($-1.1675$, Odds Ratio: $0.31\times$): Young dependents living at home.
  3. `marital_status_Never-married` ($-1.0373$, Odds Ratio: $0.35\times$): Early-career entry-level demographics.
  4. `sex_Female` ($-0.9794$, Odds Ratio: $0.38\times$): Reflects the historical 1994 gender wage gap.
  5. `occupation_Farming-fishing` ($-0.9797$, Odds Ratio: $0.38\times$): Primary agricultural lower wage scales.

### 2. Decision Tree Structure & Overfitting Diagnosis
* **Depth & Leaves:** The unconstrained tree reached **Depth 52** with **$3,815$ leaf nodes**.
* **Overfitting Gap:** Train Accuracy was $100.00\%$ ($F_1 = 0.9999$) vs Test Accuracy of $81.04\%$ ($F_1 = 0.6194$), representing a massive **$38.05\%$ drop in $F_1$**.
* **Top 3 Root Splits Assessment:**
  1. Root Split: `marital_status == Married-civ-spouse` (Partitions high base-rate married individuals from low base-rate singles).
  2. Level 1: `capital_gain` ($>0.82\sigma$) and `education_num` ($>0.94\sigma$, corresponding to Bachelors+).
  * *Verdict:* The tree's hierarchical logic is economically sound and intuitive, but requires depth regularization (`max_depth=6-8`) and ensemble averaging.

---

## 🚀 Task 5: Model Selection Write-Up & Day 3 Roadmap

### 1. Model Selection Justification
* **Advance Logistic Regression (Primary Linear Benchmark):** Delivers superior generalization, zero overfitting, and the best test $F_1$-score ($0.6734$) and PR-AUC ($0.7732$).
* **Advance Tree Ensembles (Primary Non-Linear Candidates):** The single tree demonstrated the power of non-linear interaction splits but suffered from extreme variance. In Day 3, we will develop **Random Forests** and **Gradient Boosted Trees (XGBoost / LightGBM)** to harness interaction power while mitigating overfitting.

### 2. Preprocessing Enhancements for Day 3
1. **Non-Linear Transformations:** Apply `np.log1p` on `capital_gain` and `capital_loss` alongside binary indicator flags (`has_capital_gain`).
2. **Feature Interactions:** Engineer `age * education_num` and `hours_per_week * age` to provide linear models with career seniority signals.
3. **Cardinality Reduction:** Group $41$ `native_country` levels into $5$ geographic macro-regions.

---

## 💻 How to Run Day 2 Deliverables

### Open and Run Jupyter Notebook
```bash
jupyter notebook "week 1/day 2/day2.ipynb"
```
Or open [`day2.ipynb`](day2.ipynb) directly in VS Code / Cursor and select the Python kernel (`.venv`).

### Reusable Pipeline Module
You can import the preprocessing pipelines directly in Python:
```python
from pipeline import get_column_preprocessor, get_pipeline
```

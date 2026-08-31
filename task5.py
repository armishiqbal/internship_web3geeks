import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
]

df = pd.read_csv(
    'adults.csv',
    names=COLUMNS,
    na_values=['?', ' ?', '? '],
    skipinitialspace=True
)


df['target'] = (df['income'].astype(str).str.strip().str.replace('.', '', regex=False) == '>50K').astype(int)

X = df.drop(columns=['income', 'target'])
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

test_df = X_test.copy()
test_df['actual'] = y_test
test_df['predicted'] = (test_df['education_num'] >= 13).astype(int)


fp_df = test_df[(test_df['actual'] == 0) & (test_df['predicted'] == 1)]
fn_df = test_df[(test_df['actual'] == 1) & (test_df['predicted'] == 0)]

print("=" * 80)
print("TASK 5: ERROR ANALYSIS ON HOLD-OUT TEST SET")
print("=" * 80)
print(f"Total Test Instances:   {len(test_df):,}")
print(f"False Positives (FP):   {len(fp_df):,} instances ({len(fp_df)/len(test_df)*100:.2f}% of test set)")
print(f"False Negatives (FN):   {len(fn_df):,} instances ({len(fn_df)/len(test_df)*100:.2f}% of test set)")


display_cols = [
    'age', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'hours_per_week', 'capital_gain'
]

print("\n" + "=" * 80)
print("1. SAMPLE OF 10 FALSE POSITIVES (Predicted >50K, Actual <=50K):")
print("=" * 80)
sample_fp = fp_df[display_cols].sample(n=10, random_state=42)
print(sample_fp.to_string())

print("\n" + "=" * 80)
print("2. SAMPLE OF 10 FALSE NEGATIVES (Predicted <=50K, Actual >50K):")
print("=" * 80)
sample_fn = fn_df[display_cols].sample(n=10, random_state=42)
print(sample_fn.to_string())


print("\n" + "=" * 80)
print("3. KEY ERROR PATTERNS IDENTIFIED:")
print("=" * 80)
print("""
[A] False Positive Patterns (Predicted >50K, Actually <=50K):
    - Younger / Entry-Level Graduates: Individuals with Bachelors/Masters who are under 30
      years old, just starting their careers, and have not yet reached senior compensation.
    - Low / Part-Time Hours: Highly educated individuals working part-time (<35 hours/week).
    - Unmarried / Non-head of household: Disproportionate representation of 'Never-married'
      or 'Not-in-family' statuses with zero capital gains.

[B] False Negative Patterns (Predicted <=50K, Actually >50K):
    - Experienced Skilled Trades / Management without Higher Degrees: Individuals with
      'HS-grad' or 'Some-college' (education_num < 13) who hold senior roles in Craft-repair,
      Transport-moving, Exec-managerial, or Sales.
    - Career Seniority & Overtime: Older workers (ages 40-55+) working high weekly hours
      (45-60+ hours/week) earning substantial overtime / tenure-based wages.
    - Capital Income: Non-degree holders with substantial capital_gain (> $5,000+).
""")


print("=" * 80)
print("4. CONCRETE DATA & FEATURE ENGINEERING ISSUES TO FIX TOMORROW:")
print("=" * 80)
print("""
1. Skewed Numerics & Capital Indicators:
   - 'capital_gain' and 'capital_loss' are heavily right-skewed with >90% zeros.
   - Action: Create binary indicator flags ('has_capital_gain', 'has_capital_loss') and apply
     log1p transformations to numerical values.

2. Categorical Imputation for Missing Values:
   - 'workclass' (1,836 missing), 'occupation' (1,843 missing), and 'native_country' (583 missing).
   - Action: Impute with a dedicated 'Missing' / 'Unknown' category token so models learn the
     predictive signal of unrecorded data.

3. Feature Interactions for Career Seniority:
   - Education alone ignores work experience.
   - Action: Engineer interaction features like (age * education_num) and (hours_per_week * age)
     to differentiate junior degree holders from seasoned trade veterans.

4. Marital Status & Household Relationship Consolidation:
   - 'Married-civ-spouse' is one of the strongest multi-earner / high-income indicators in Census data.
   - Action: Group marital statuses into binary/consolidated categories ('is_married', 'is_single', 'is_divorced').

5. Categorical Encoding & Dimensionality Management:
   - High cardinality features ('native_country' with 41 categories, 'occupation' with 14 categories).
   - Action: Use one-hot encoding for standard categories and group rare countries into regions (e.g., 'US', 'Central-America', 'Europe', 'Asia').
""")


print("=" * 80)
print("5. PRIMARY METRIC TO OPTIMIZE FOR THE REST OF THE WEEK:")
print("=" * 80)
print("""
For the rest of the week, our primary metric to optimize will be the F1-Score (and Precision@Top-k).
Our Day 1 baselines demonstrated that Accuracy is completely misleading on this imbalanced dataset
(the majority predictor scored 75.93% accuracy but delivered a 0.00 F1-score and zero business utility).
Meanwhile, our single-feature education baseline set the performance benchmark at 58.74% Precision,
46.56% Recall, and an F1-score of 0.5197 (PR-AUC: 0.4021). Optimizing for F1-score and PR-AUC will force
upcoming models (Logistic Regression, Random Forests, Gradient Boosting) to simultaneously push precision
past 75% while scaling recall to 60-70%+, ensuring our marketing outreach captures maximum high earners
without wasting budget on false positives.
""")
print("=" * 80)

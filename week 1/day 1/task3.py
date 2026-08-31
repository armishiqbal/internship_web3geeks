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


RANDOM_STATE = 42


X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.125,
    random_state=RANDOM_STATE,
    stratify=y_train_val
)


print("=" * 75)
print("TASK 3: REPRODUCIBLE STRATIFIED SPLITS SUMMARY")
print("=" * 75)

total_rows = len(df)

splits_info = [
    ("Full Dataset", len(df), df['target'].mean() * 100, 100.0),
    ("Train Set (70%)", len(X_train), y_train.mean() * 100, (len(X_train) / total_rows) * 100),
    ("Dev/Validation Set (10%)", len(X_dev), y_dev.mean() * 100, (len(X_dev) / total_rows) * 100),
    ("Hold-Out Test Set (20%)", len(X_test), y_test.mean() * 100, (len(X_test) / total_rows) * 100),
]

summary_df = pd.DataFrame(
    splits_info,
    columns=["Split", "Sample Count", "Positive Rate (>50K)", "Share of Total"]
)

print(summary_df.to_string(index=False))
print("-" * 75)
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_dev   shape: {X_dev.shape}   | y_dev   shape: {y_dev.shape}")
print(f"X_test  shape: {X_test.shape}  | y_test  shape: {y_test.shape}")
print("=" * 75)


print("\n" + "=" * 75)
print("WHY A HOLD-OUT TEST SET IS CRITICAL & THE RISKS OF TEST LEAKAGE:")
print("=" * 75)
print("""
1. Purpose of Hold-Out Test Set:
   A hold-out test set acts as an unbiased, real-world proxy that is touched ONLY ONCE
   for final performance evaluation after all model choices and tuning are complete.

2. What goes wrong if used during hyperparameter tuning (Data Snooping / Test Leakage):
   Iteratively tweaking hyperparameters (e.g., tree depth, regularization strength) against
   the test set causes the model and engineer to overfit to the random noise and specific
   idiosyncrasies of that test sample. This produces falsely optimistic metrics and leads
   to severe performance drops when the model is deployed to genuine production data.
""")
print("=" * 75)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


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


majority_clf = DummyClassifier(strategy='most_frequent')
majority_clf.fit(X_train, y_train)

y_pred_majority = majority_clf.predict(X_test)

y_prob_majority = np.full(len(y_test), y_train.mean())


y_pred_rule = (X_test['education_num'] >= 13).astype(int)

y_prob_rule = (X_test['education_num'] >= 13).astype(float)


def compute_all_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'Confusion Matrix': cm
    }

metrics_majority = compute_all_metrics(y_test, y_pred_majority, y_prob_majority)
metrics_rule = compute_all_metrics(y_test, y_pred_rule, y_prob_rule)


print("=" * 80)
print("TASK 4: BASELINE MODELS PERFORMANCE ON HOLD-OUT TEST SET")
print("=" * 80)

metrics_table = pd.DataFrame([
    {
        'Baseline Model': '1. Majority-Class (Always <=50K)',
        'Accuracy': f"{metrics_majority['Accuracy']:.4f}",
        'Precision': f"{metrics_majority['Precision']:.4f}",
        'Recall': f"{metrics_majority['Recall']:.4f}",
        'F1': f"{metrics_majority['F1-Score']:.4f}",
        'ROC AUC': f"{metrics_majority['ROC AUC']:.4f}",
        'PR AUC': f"{metrics_majority['PR AUC']:.4f}"
    },
    {
        'Baseline Model': '2. Single-Feature Rule (education_num >= 13)',
        'Accuracy': f"{metrics_rule['Accuracy']:.4f}",
        'Precision': f"{metrics_rule['Precision']:.4f}",
        'Recall': f"{metrics_rule['Recall']:.4f}",
        'F1': f"{metrics_rule['F1-Score']:.4f}",
        'ROC AUC': f"{metrics_rule['ROC AUC']:.4f}",
        'PR AUC': f"{metrics_rule['PR AUC']:.4f}"
    }
])

print("\n--- METRICS SUMMARY TABLE ---")
print(metrics_table.to_string(index=False))

print("\n--- CONFUSION MATRICES ---")
print("\n[Baseline 1: Majority-Class]")
cm1 = metrics_majority['Confusion Matrix']
print(f"  TN: {cm1[0,0]:<5} | FP: {cm1[0,1]:<5}")
print(f"  FN: {cm1[1,0]:<5} | TP: {cm1[1,1]:<5}")

print("\n[Baseline 2: Single-Feature Rule (education_num >= 13)]")
cm2 = metrics_rule['Confusion Matrix']
print(f"  TN: {cm2[0,0]:<5} | FP: {cm2[0,1]:<5}")
print(f"  FN: {cm2[1,0]:<5} | TP: {cm2[1,1]:<5}")


print("\n" + "=" * 80)
print("INTERPRETATION & UTILITY CRITERIA (3-4 Sentences):")
print("=" * 80)
print(f"""
1. Which Baseline Wins and Why:
   The single-feature rule-based baseline (education_num >= 13) clearly wins because,
   unlike the majority predictor which achieves {metrics_majority['Accuracy']*100:.2f}% accuracy with zero predictive value
   ({metrics_majority['Precision']:.1f} Precision, Recall, and F1), the education rule captures actual signal with {metrics_rule['Precision']*100:.2f}%
   precision, {metrics_rule['Recall']*100:.2f}% recall, and an F1 of {metrics_rule['F1-Score']:.4f} (PR-AUC: {metrics_rule['PR AUC']:.4f}).

2. Minimum Improvement for a Real Machine Learning Model to Be "Useful":
   To justify the complexity of deployment, an ML model must beat the simple single-feature
   heuristic baseline by achieving an F1-score > 0.65 (a ~35%+ relative improvement),
   a PR-AUC > 0.60, and a precision above 75% at a recall of at least 60%.
""")
print("=" * 80)

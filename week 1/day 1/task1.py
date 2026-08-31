import os
import pandas as pd
import numpy as np


COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
]


csv_path = 'adults.csv' if os.path.exists('adults.csv') else os.path.join('task 1', 'adults.csv')


df = pd.read_csv(
    csv_path,
    names=COLUMNS,
    na_values=['?', ' ?', '? '],
    skipinitialspace=True
)

df['target'] = (df['income'].astype(str).str.strip() == '>50K').astype(int)


total_records = len(df)
positive_records = df['target'].sum()
negative_records = total_records - positive_records
base_rate = df['target'].mean() * 100

print("=" * 70)
print("TASK 1: PROBLEM DEFINITION, SUCCESS METRIC & CLASS BASE RATE")
print("=" * 70)

print("\n1. TARGET DEFINITION & DATASET SUMMARY:")
print(f"   - Positive Class (>50K, y=1) : {positive_records:,} rows ({base_rate:.2f}%)")
print(f"   - Negative Class (<=50K, y=0): {negative_records:,} rows ({100 - base_rate:.2f}%)")
print(f"   - Total Records              : {total_records:,} rows")
print(f"   - Class Base Rate            : {base_rate:.2f}%")

print("\n2. REALISTIC BUSINESS OBJECTIVE:")
print("   - Objective : Identify high-income individuals (>50K) for premium financial/investment outreach.")
print("   - Constraint: High direct marketing/outreach cost per customer (phone advisory, direct mail).")

print("\n3. PRIMARY EVALUATION METRIC & TRADE-OFF:")
print("   - Chosen Metric : Precision (and Precision@Top-k)")
print("   - Why Precision : False Positives (contacting <=50K earners) waste budget and sales capacity.")
print("                     We prioritize high conversion per contact over catching every single high earner (Recall).")

print("\n4. NON-TECHNICAL STAKEHOLDER SUMMARY (2-3 Sentences):")
print('   "Our goal is to identify high-earning individuals (>50K/year) so our sales team')
print('    can focus their outreach exclusively on the most qualified prospective clients.')
print('    Because contacting each prospect costs money, we are optimizing for precision—ensuring')
print('    that our marketing budget is spent on genuine high earners rather than wasted on unqualified leads."')
print("=" * 70)

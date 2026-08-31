import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
]

possible_paths = [
    'adults.csv',
    os.path.join('..', 'adults.csv'),
    os.path.join('..', 'task 1', 'adults.csv'),
    os.path.join(os.path.dirname(__file__), '..', 'task 1', 'adults.csv'),
    os.path.join(os.path.dirname(__file__), 'adults.csv')
]

csv_path = None

for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path:
    print(f"Loading dataset from: {csv_path}")

    df = pd.read_csv(
        csv_path,
        names=COLUMNS,
        na_values=['?', ' ?', '? '],
        skipinitialspace=True
    )
else:
    print("CSV not found locally, fetching from OpenML...")

    from sklearn.datasets import fetch_openml

    adult = fetch_openml(
        'adult',
        version=2,
        as_frame=True
    )

    df = adult.frame
    df.replace('?', np.nan, inplace=True)

income_col = 'income' if 'income' in df.columns else 'class'

df['target'] = (
    df[income_col]
    .astype(str)
    .str.strip()
    .str.replace('.', '', regex=False)
    .eq('>50K')
    .astype(int)
)

print("\n" + "=" * 70)
print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

print(f"\n[A] Data Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

print("\n[B] Column Types & Non-Null Counts:")
print(df.dtypes)

missing = df.isna().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})

print("\n[C] Missing Values per Column:")
print(
    missing_df[
        missing_df['Missing_Count'] > 0
    ]
)

print("\n[D] Numeric Features Summary Statistics:")

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

print(
    df[numeric_cols]
    .describe()
    .round(2)
    .to_string()
)

categorical_cols = df.select_dtypes(
    include=['object', 'category']
).columns.tolist()

if income_col in categorical_cols:
    categorical_cols.remove(income_col)

print("\n[E] Value Counts for Categorical Features:")

for col in categorical_cols:
    print(f"\n--- {col.upper()} (Top 5 categories) ---")

    val_counts = (
        df[col]
        .value_counts(dropna=False)
        .head(5)
    )

    percentages = (
        df[col]
        .value_counts(
            dropna=False,
            normalize=True
        )
        .head(5)
        * 100
    ).round(2)

    summary_cat = pd.DataFrame({
        'Count': val_counts,
        'Percentage (%)': percentages
    })

    print(summary_cat.to_string())

target_counts = (
    df['target']
    .value_counts()
    .sort_index()
)

target_pct = (
    df['target']
    .value_counts(normalize=True)
    .sort_index()
    * 100
).round(2)

class_table = pd.DataFrame({
    'Class': ['<=50K (0)', '>50K (1)'],
    'Count': target_counts.values,
    'Share (%)': target_pct.values
})

edu_dist = (
    df.groupby(
        'education',
        observed=False
    )
    .agg(
        Total_Count=('target', 'count'),
        High_Income_Count=('target', 'sum'),
        High_Income_Rate=(
            'target',
            lambda x: f"{(x.mean() * 100):.2f}%"
        )
    )
    .sort_values(
        by='Total_Count',
        ascending=False
    )
)

occ_dist = (
    df.groupby(
        'occupation',
        observed=False
    )
    .agg(
        Total_Count=('target', 'count'),
        High_Income_Count=('target', 'sum'),
        High_Income_Rate=(
            'target',
            lambda x: f"{(x.mean() * 100):.2f}%"
        )
    )
    .sort_values(
        by='Total_Count',
        ascending=False
    )
)

df['age_group'] = pd.cut(
    df['age'],
    bins=[16, 25, 35, 45, 55, 65, 100],
    labels=[
        '17-25',
        '26-35',
        '36-45',
        '46-55',
        '56-65',
        '65+'
    ]
)

age_dist = (
    df.groupby(
        'age_group',
        observed=False
    )
    .agg(
        Total_Count=('target', 'count'),
        High_Income_Count=('target', 'sum'),
        High_Income_Rate=(
            'target',
            lambda x: f"{(x.mean() * 100):.2f}%"
        )
    )
)

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

summary_csv_path = os.path.join(
    script_dir,
    'summary_table.csv'
)

summary_txt_path = os.path.join(
    script_dir,
    'summary_table.txt'
)

class_table.to_csv(
    summary_csv_path,
    index=False
)

with open(
    summary_txt_path,
    'w',
    encoding='utf-8'
) as f:

    f.write("=" * 80 + "\n")
    f.write(
        "TASK 2: SUMMARY TABLE - CLASS COUNTS & "
        "NOTABLE FEATURE DISTRIBUTIONS\n"
    )
    f.write("=" * 80 + "\n\n")

    f.write("1. TARGET CLASS DISTRIBUTION:\n")
    f.write(
        class_table.to_string(index=False)
        + "\n\n"
    )

    f.write(
        "2. NOTABLE FEATURE 1: EDUCATION LEVEL "
        "VS HIGH-INCOME RATE (>50K):\n"
    )
    f.write(
        edu_dist.to_string()
        + "\n\n"
    )

    f.write(
        "3. NOTABLE FEATURE 2: OCCUPATION VS "
        "HIGH-INCOME RATE (>50K):\n"
    )
    f.write(
        occ_dist.to_string()
        + "\n\n"
    )

    f.write(
        "4. NOTABLE FEATURE 3: AGE GROUP VS "
        "HIGH-INCOME RATE (>50K):\n"
    )
    f.write(
        age_dist.to_string()
        + "\n"
    )

print(
    f"\n[F] Summary table saved to:\n"
    f"    - {summary_txt_path}"
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 12)
)

sns.countplot(
    data=df,
    x='target',
    hue='target',
    palette=['#4C72B0', '#55A868'],
    ax=axes[0, 0],
    legend=False
)

axes[0, 0].set_title(
    '1. Target Distribution (0: <=50K, 1: >50K)',
    fontsize=14,
    fontweight='bold'
)

axes[0, 0].set_xticklabels([
    '<=50K',
    '>50K'
])

axes[0, 0].set_xlabel('Income Class')
axes[0, 0].set_ylabel('Number of Individuals')

sns.histplot(
    data=df,
    x='age',
    hue='target',
    multiple='stack',
    bins=30,
    palette=['#4C72B0', '#C44E52'],
    ax=axes[0, 1]
)

axes[0, 1].set_title(
    '2. Age Distribution by Income Class',
    fontsize=14,
    fontweight='bold'
)

axes[0, 1].set_xlabel('Age (Years)')
axes[0, 1].set_ylabel('Count')

edu_order = (
    df.groupby(
        'education',
        observed=False
    )['target']
    .mean()
    .sort_values(
        ascending=False
    )
    .index
)

sns.barplot(
    data=df,
    x='education',
    y='target',
    order=edu_order,
    palette='Blues_r',
    ax=axes[1, 0],
    errorbar=None,
    hue='education',
    legend=False
)

axes[1, 0].set_title(
    '3. High-Income Rate (>50K) by Education Level',
    fontsize=14,
    fontweight='bold'
)

axes[1, 0].set_xlabel('Education')
axes[1, 0].set_ylabel(
    'Proportion Earning >50K'
)

axes[1, 0].tick_params(
    axis='x',
    rotation=55
)

sns.histplot(
    data=df,
    x='hours_per_week',
    hue='target',
    multiple='layer',
    bins=30,
    palette=['#4C72B0', '#DD8452'],
    ax=axes[1, 1],
    alpha=0.6
)

axes[1, 1].set_title(
    '4. Hours Worked per Week by Income Class',
    fontsize=14,
    fontweight='bold'
)

axes[1, 1].set_xlabel(
    'Hours Worked per Week'
)

axes[1, 1].set_ylabel('Count')

plt.tight_layout()

plot_path = os.path.join(
    script_dir,
    'eda_visualizations.png'
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print(
    f"\n[G] Visualizations saved to:\n"
    f"    - {plot_path}"
)

print("=" * 70)
print("TASK 2 COMPLETE!")
print("=" * 70)
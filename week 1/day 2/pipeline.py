import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# REUSABLE PREPROCESSING & MODEL PIPELINES (DAY 2 EXPORT)
# ==============================================================================

NUMERIC_FEATURES = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'
]

CATEGORICAL_FEATURES = [
    'workclass', 'education', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'native_country'
]

def create_preprocessing_pipeline():
    """
    Constructs and returns the standardized ColumnTransformer for mixed tabular data.
    - Numeric features: Median Imputation + Standard Scaling
    - Categorical features: Most Frequent Imputation + One-Hot Encoding (handle_unknown='ignore')
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def create_logistic_regression_pipeline(C=1.0, max_iter=1000, random_state=42):
    """Returns an end-to-end Logistic Regression Pipeline."""
    return Pipeline(steps=[
        ('preprocessor', create_preprocessing_pipeline()),
        ('classifier', LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])

def create_decision_tree_pipeline(max_depth=None, min_samples_leaf=1, random_state=42):
    """Returns an end-to-end Decision Tree Pipeline."""
    return Pipeline(steps=[
        ('preprocessor', create_preprocessing_pipeline()),
        ('classifier', DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        ))
    ])

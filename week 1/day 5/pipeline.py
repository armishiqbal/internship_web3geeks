"""Day 5 production pipeline helpers.

Row-level feature engineering matches Week 1 Day 4 exactly and lives in a
real module so joblib can reload the artifact outside the training notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "is_married",
    "is_higher_ed",
    "has_capital_gain",
    "has_capital_loss",
    "log_capital_gain",
    "edu_x_hours",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
OPERATING_THRESHOLD = 0.40


class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leak-free row-wise features. No target, no cross-row statistics."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        marital = X["marital_status"].astype(str)
        X["is_married"] = marital.str.contains("Married", case=False, na=False).astype(int)
        X["is_higher_ed"] = (X["education_num"] >= 13).astype(int)
        X["has_capital_gain"] = (X["capital_gain"] > 0).astype(int)
        X["has_capital_loss"] = (X["capital_loss"] > 0).astype(int)
        X["log_capital_gain"] = np.log1p(X["capital_gain"].clip(lower=0))
        X["edu_x_hours"] = X["education_num"] * X["hours_per_week"]
        return X


def load_adult_frame(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the headerless UCI Adult file with canonical column names.

    Passing ``names=`` without ``header=None`` can still let pandas treat the
    first data row as a header on some versions, which drops ``marital_status``.
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?", " ?"],
        skip_blank_lines=True,
    )
    y = (
        df["income"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
        .eq(">50K")
        .astype(int)
    )
    X = df.drop(columns=["income"])
    return X, y


def leakfree_split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_unfitted_hgb_pipeline():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return Pipeline(
        [
            ("feature_engineering", AdultFeatureEngineer()),
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.15,
                    max_iter=300,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=5.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

"""Train the Day 4 winning configuration and save a portable production artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline import (
    OPERATING_THRESHOLD,
    RANDOM_STATE,
    build_unfitted_hgb_pipeline,
    leakfree_split,
    load_adult_frame,
)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "adults.csv"
if not DATA_PATH.exists():
    DATA_PATH = HERE.parent / "day 4" / "adults.csv"

ARTIFACT_PATH = HERE / "final_model.joblib"
METRICS_PATH = HERE / "final_metrics.csv"


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred)),
        "Recall": float(recall_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred)),
        "ROC-AUC": float(roc_auc_score(y_true, proba)),
        "PR-AUC": float(average_precision_score(y_true, proba)),
        "Brier": float(brier_score_loss(y_true, proba)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "threshold": float(threshold),
    }


def main() -> None:
    X, y = load_adult_frame(str(DATA_PATH))
    X_train, X_test, y_train, y_test = leakfree_split(X, y)
    assert len(set(X_train.index) & set(X_test.index)) == 0

    base = build_unfitted_hgb_pipeline()
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba, OPERATING_THRESHOLD)

    artifact = {
        "pipeline": model,
        "threshold": OPERATING_THRESHOLD,
        "metadata": {
            "project": "Week 1 Day 5 — Production Adult Income Classifier",
            "source_day4_config": {
                "model": "HistGradientBoostingClassifier",
                "learning_rate": 0.15,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "l2_regularization": 5.0,
                "calibration": "sigmoid CalibratedClassifierCV cv=5",
            },
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_metrics": metrics,
            "python_version": sys.version,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
    }
    joblib.dump(artifact, ARTIFACT_PATH)

    reloaded = joblib.load(ARTIFACT_PATH)
    proba2 = reloaded["pipeline"].predict_proba(X_test)[:, 1]
    np.testing.assert_allclose(proba, proba2)

    import pandas as pd

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
    print(json.dumps(metrics, indent=2))
    print(f"saved {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

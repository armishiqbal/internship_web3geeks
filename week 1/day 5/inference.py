"""Production inference for the Day 5 Adult Income artifact.

Does not repeat preprocessing. The loaded pipeline already contains
feature engineering, imputation, scaling, one-hot encoding, the
HistGradientBoosting classifier, and sigmoid calibration.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = HERE / "final_model.joblib"


def load_artifact(path: str | Path = DEFAULT_ARTIFACT) -> dict:
    return joblib.load(path)


def score_raw(new_data: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Score raw Adult-schema rows. Returns probabilities and thresholded labels."""
    if artifact is None:
        artifact = load_artifact()
    model = artifact["pipeline"]
    threshold = float(artifact["threshold"])
    proba = model.predict_proba(new_data)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = new_data.copy()
    out["prob_gt_50k"] = proba
    out["prediction"] = pred
    out["label"] = out["prediction"].map({0: "<=50K", 1: ">50K"})
    out["threshold"] = threshold
    return out


if __name__ == "__main__":
    from pipeline import leakfree_split, load_adult_frame

    X, y = load_adult_frame(str(HERE / "adults.csv"))
    _, X_test, _, y_test = leakfree_split(X, y)
    sample = X_test.sample(10, random_state=7)
    scored = score_raw(sample)
    scored["y_true"] = y_test.loc[sample.index].to_numpy()
    cols = ["age", "education", "occupation", "hours_per_week", "prob_gt_50k", "label", "y_true"]
    print(scored[cols].to_string(index=False))

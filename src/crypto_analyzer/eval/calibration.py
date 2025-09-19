"""Probability calibration utilities and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

CalibratorName = Literal["isotonic", "platt", "none"]


@dataclass
class CalibrationResult:
    """Container describing a calibrated probability vector."""

    probabilities: np.ndarray
    calibrator: object | None
    method: CalibratorName


def reliability_diagram(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute a reliability diagram without plotting."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    if y_true.shape != y_prob.shape:
        raise ValueError("Shapes of y_true and y_prob must match")
    if n_bins <= 1:
        raise ValueError("n_bins must be greater than one")

    bins = np.linspace(0.0, 1.0, num=n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    data = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            avg_pred = np.nan
            avg_true = np.nan
            count = 0
        else:
            avg_pred = float(y_prob[mask].mean())
            avg_true = float(y_true[mask].mean())
            count = int(mask.sum())
        data.append({"bin": b, "count": count, "avg_pred": avg_pred, "avg_true": avg_true})

    return pd.DataFrame(data)


def calibrate_probabilities(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    method: CalibratorName = "none",
    sample_weight: np.ndarray | None = None,
) -> CalibrationResult:
    """Recalibrate ``y_prob`` using the requested method."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    if method == "none":
        return CalibrationResult(probabilities=y_prob, calibrator=None, method="none")

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        probabilities = calibrator.fit_transform(y_prob, y_true, sample_weight=sample_weight)
        return CalibrationResult(probabilities=probabilities, calibrator=calibrator, method=method)

    if method == "platt":
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(y_prob.reshape(-1, 1), y_true, sample_weight=sample_weight)
        probabilities = lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return CalibrationResult(probabilities=probabilities, calibrator=lr, method=method)

    raise ValueError(f"Unsupported calibration method: {method!r}")


def probability_scores(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Return Brier score and log loss for probability forecasts."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


def expected_vs_actual_hit_rate(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compare the average predicted hit rate against the realised one."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    return {
        "expected": float(y_prob.mean()),
        "actual": float(y_true.mean()),
    }


__all__ = [
    "CalibrationResult",
    "calibrate_probabilities",
    "expected_vs_actual_hit_rate",
    "probability_scores",
    "reliability_diagram",
]

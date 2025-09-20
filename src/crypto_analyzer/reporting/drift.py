"""Monitoring utilities for feature and probability drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = [
    "DriftThresholds",
    "population_stability_index",
    "ks_drift",
    "monitor_feature_drift",
    "rolling_recalibration",
]


@dataclass(slots=True)
class DriftThresholds:
    psi_alert: float = 0.2
    ks_alert: float = 0.1


def _align_series(reference: pd.Series, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    return ref, cur


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = 10
) -> float:
    """Compute the Population Stability Index (PSI)."""

    ref, cur = _align_series(reference, current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.quantile(ref, quantiles)
    cut_points[0] = -np.inf
    cut_points[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=cut_points)
    cur_hist, _ = np.histogram(cur, bins=cut_points)
    ref_dist = ref_hist / ref_hist.sum()
    cur_dist = cur_hist / cur_hist.sum()

    mask = (ref_dist > 0) & (cur_dist > 0)
    psi = np.sum((cur_dist[mask] - ref_dist[mask]) * np.log(cur_dist[mask] / ref_dist[mask]))
    return float(psi)


def ks_drift(reference: pd.Series, current: pd.Series) -> float:
    """Return the Kolmogorov–Smirnov statistic between two samples."""

    ref, cur = _align_series(reference, current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    statistic = ks_2samp(ref, cur).statistic
    return float(statistic)


def monitor_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    features: Iterable[str],
    thresholds: DriftThresholds | None = None,
) -> pd.DataFrame:
    """Compute PSI/KS for the provided feature list."""

    th = thresholds or DriftThresholds()
    rows = []
    for feature in features:
        psi = population_stability_index(reference[feature], current[feature])
        ks_score = ks_drift(reference[feature], current[feature])
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "psi_alert": psi >= th.psi_alert if not np.isnan(psi) else False,
                "ks": ks_score,
                "ks_alert": ks_score >= th.ks_alert if not np.isnan(ks_score) else False,
            }
        )
    return pd.DataFrame(rows)


def rolling_recalibration(
    probabilities: pd.Series,
    outcomes: pd.Series,
    *,
    window: int,
    method: str = "isotonic",
) -> Mapping[pd.Timestamp, IsotonicRegression | LogisticRegression]:
    """Fit rolling calibration models for probability drift correction."""

    if window <= 0:
        raise ValueError("window must be positive")
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be aligned")

    calibrators: dict[pd.Timestamp, IsotonicRegression | LogisticRegression] = {}
    probs = probabilities.to_numpy(dtype=float)
    labels = outcomes.to_numpy(dtype=int)

    for idx in range(window, len(probabilities) + 1):
        start = idx - window
        end = idx
        window_probs = probs[start:end]
        window_labels = labels[start:end]
        timestamp = probabilities.index[end - 1]
        if len(np.unique(window_labels)) < 2:
            continue
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(window_probs, window_labels)
        elif method == "sigmoid":
            calibrator = LogisticRegression()
            calibrator.fit(window_probs.reshape(-1, 1), window_labels)
        else:  # pragma: no cover - defensive branch
            raise ValueError("method must be 'isotonic' or 'sigmoid'")
        calibrators[timestamp] = calibrator
    return calibrators


"""Basic conformal prediction helpers for touch-risk analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConformalInterval:
    lower: np.ndarray
    upper: np.ndarray
    alpha: float


def conformal_interval(
    y_calib: np.ndarray | pd.Series,
    y_pred_calib: np.ndarray | pd.Series,
    y_pred_test: np.ndarray | pd.Series,
    *,
    alpha: float = 0.1,
) -> ConformalInterval:
    """Compute symmetric conformal intervals for regression-style targets."""

    y_calib = np.asarray(y_calib, dtype=np.float64)
    y_pred_calib = np.asarray(y_pred_calib, dtype=np.float64)
    y_pred_test = np.asarray(y_pred_test, dtype=np.float64)

    if y_calib.shape != y_pred_calib.shape:
        raise ValueError("Calibration arrays must have matching shapes")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    residuals = np.abs(y_calib - y_pred_calib)
    if residuals.size == 0:
        raise ValueError("Calibration residuals cannot be empty")

    q = np.quantile(residuals, 1 - alpha * (1 + 1 / residuals.size), method="higher")
    lower = y_pred_test - q
    upper = y_pred_test + q
    return ConformalInterval(lower=lower, upper=upper, alpha=alpha)


def conformal_touch_interval(
    touch_calib: np.ndarray | pd.Series,
    p_calib: np.ndarray | pd.Series,
    p_test: np.ndarray | pd.Series,
    *,
    alpha: float = 0.1,
) -> ConformalInterval:
    """Return calibrated probability intervals for the touch event."""

    touch_calib = np.asarray(touch_calib, dtype=np.float64)
    p_calib = np.clip(np.asarray(p_calib, dtype=np.float64), 1e-6, 1 - 1e-6)
    p_test = np.clip(np.asarray(p_test, dtype=np.float64), 1e-6, 1 - 1e-6)

    if touch_calib.shape != p_calib.shape:
        raise ValueError("Calibration probabilities must align with touch labels")

    residuals = np.abs(touch_calib - p_calib)
    if residuals.size == 0:
        raise ValueError("Calibration residuals cannot be empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    q = np.quantile(residuals, 1 - alpha * (1 + 1 / residuals.size), method="higher")
    lower = np.clip(p_test - q, 0.0, 1.0)
    upper = np.clip(p_test + q, 0.0, 1.0)
    return ConformalInterval(lower=lower, upper=upper, alpha=alpha)


__all__ = ["ConformalInterval", "conformal_interval", "conformal_touch_interval"]

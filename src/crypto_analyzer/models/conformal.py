"""Conformal prediction helpers for binary touch classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "ProbabilityInterval",
    "ConformalSummary",
    "split_conformal_interval",
    "aps_conformal_interval",
    "generate_touch_conformal_report",
    "conformal_interval",
]


@dataclass(slots=True)
class ProbabilityInterval:
    """Lower/upper bounds describing uncertainty around ``p̂``."""

    method: str
    alpha: float
    lower: np.ndarray
    upper: np.ndarray
    radius: float

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "alpha": float(self.alpha),
            "radius": float(self.radius),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "mean_width": float(np.mean(self.upper - self.lower)),
        }


@dataclass(slots=True)
class ConformalSummary:
    """Wrap a :class:`ProbabilityInterval` with calibration diagnostics."""

    interval: ProbabilityInterval
    calibration_coverage: float

    def to_dict(self) -> dict[str, object]:
        data = self.interval.to_dict()
        data["calibration_coverage"] = float(self.calibration_coverage)
        return data


def _to_numpy(arr: Iterable[float] | np.ndarray | pd.Series) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float64, copy=False)
    if isinstance(arr, pd.Series):
        return arr.to_numpy(dtype=np.float64, copy=False)
    return np.asarray(list(arr), dtype=np.float64)


def _validate_inputs(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    p_test: np.ndarray,
    *,
    alpha: float,
) -> None:
    if y_cal.shape != p_cal.shape:
        raise ValueError("`y_cal` and `p_cal` must have matching shapes")
    if y_cal.ndim != 1:
        raise ValueError("`y_cal` must be one-dimensional")
    if p_test.ndim != 1:
        raise ValueError("`p_test` must be one-dimensional")
    if len(y_cal) == 0:
        raise ValueError("Calibration set must not be empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    n = scores.size
    adjusted = 1 - alpha * (1 + 1 / n)
    adjusted = float(np.clip(adjusted, 0.0, 1.0))
    return float(np.quantile(scores, adjusted, method="higher"))


def split_conformal_interval(
    y_cal: Iterable[int] | np.ndarray | pd.Series,
    p_cal: Iterable[float] | np.ndarray | pd.Series,
    p_test: Iterable[float] | np.ndarray | pd.Series,
    *,
    alpha: float = 0.1,
) -> ConformalSummary:
    """Return symmetric split conformal intervals for binary probabilities."""

    y_cal_arr = _to_numpy(y_cal)
    p_cal_arr = np.clip(_to_numpy(p_cal), 1e-6, 1 - 1e-6)
    p_test_arr = np.clip(_to_numpy(p_test), 1e-6, 1 - 1e-6)
    _validate_inputs(y_cal_arr, p_cal_arr, p_test_arr, alpha=alpha)

    residuals = np.abs(y_cal_arr - p_cal_arr)
    radius = _finite_sample_quantile(residuals, alpha)
    lower = np.clip(p_test_arr - radius, 0.0, 1.0)
    upper = np.clip(p_test_arr + radius, 0.0, 1.0)

    calib_cover = float(np.mean(residuals <= radius))

    interval = ProbabilityInterval(
        method="split",
        alpha=alpha,
        lower=lower,
        upper=upper,
        radius=radius,
    )
    return ConformalSummary(interval=interval, calibration_coverage=calib_cover)


def aps_conformal_interval(
    y_cal: Iterable[int] | np.ndarray | pd.Series,
    p_cal: Iterable[float] | np.ndarray | pd.Series,
    p_test: Iterable[float] | np.ndarray | pd.Series,
    *,
    alpha: float = 0.1,
) -> ConformalSummary:
    """Return APS-style conformal intervals for binary probabilities."""

    y_cal_arr = _to_numpy(y_cal)
    p_cal_arr = np.clip(_to_numpy(p_cal), 1e-6, 1 - 1e-6)
    p_test_arr = np.clip(_to_numpy(p_test), 1e-6, 1 - 1e-6)
    _validate_inputs(y_cal_arr, p_cal_arr, p_test_arr, alpha=alpha)

    wrong_mass = np.where(y_cal_arr >= 0.5, 1.0 - p_cal_arr, p_cal_arr)
    radius = _finite_sample_quantile(wrong_mass, alpha)
    lower = np.clip(p_test_arr - radius, 0.0, 1.0)
    upper = np.clip(p_test_arr + radius, 0.0, 1.0)

    calib_cover = float(np.mean(wrong_mass <= radius))

    interval = ProbabilityInterval(
        method="aps",
        alpha=alpha,
        lower=lower,
        upper=upper,
        radius=radius,
    )
    return ConformalSummary(interval=interval, calibration_coverage=calib_cover)


def _label_coverage_split(y: np.ndarray, p: np.ndarray, radius: float) -> float:
    return float(np.mean(np.abs(y - p) <= radius))


def _label_coverage_aps(y: np.ndarray, p: np.ndarray, radius: float) -> float:
    wrong_mass = np.where(y >= 0.5, 1.0 - p, p)
    return float(np.mean(wrong_mass <= radius))


def generate_touch_conformal_report(
    *,
    y_cal: Iterable[int] | np.ndarray | pd.Series,
    p_cal: Iterable[float] | np.ndarray | pd.Series,
    y_test: Iterable[int] | np.ndarray | pd.Series,
    p_test: Iterable[float] | np.ndarray | pd.Series,
    alpha: float = 0.1,
) -> dict[str, object]:
    """Compute conformal confidence intervals and diagnostics."""

    y_cal_arr = _to_numpy(y_cal)
    p_cal_arr = _to_numpy(p_cal)
    y_test_arr = _to_numpy(y_test)
    p_test_arr = _to_numpy(p_test)

    if y_test_arr.shape != p_test_arr.shape:
        raise ValueError("`y_test` and `p_test` must have matching shapes")

    split_summary = split_conformal_interval(y_cal_arr, p_cal_arr, p_test_arr, alpha=alpha)
    aps_summary = aps_conformal_interval(y_cal_arr, p_cal_arr, p_test_arr, alpha=alpha)

    split_dict = split_summary.to_dict()
    split_dict["label_coverage"] = _label_coverage_split(
        y_test_arr, p_test_arr, split_summary.interval.radius
    )

    aps_dict = aps_summary.to_dict()
    aps_dict["label_coverage"] = _label_coverage_aps(
        y_test_arr, p_test_arr, aps_summary.interval.radius
    )

    return {
        "alpha": float(alpha),
        "n_calibration": int(len(y_cal_arr)),
        "n_test": int(len(y_test_arr)),
        "methods": {
            "split": split_dict,
            "aps": aps_dict,
        },
    }


def _extract_test_arrays(
    test_data: (
        Iterable[int]
        | np.ndarray
        | pd.Series
        | tuple[Iterable[int] | np.ndarray | pd.Series, Iterable[float] | np.ndarray | pd.Series]
        | dict[str, Iterable[int] | np.ndarray | pd.Series]
    )
) -> tuple[np.ndarray, np.ndarray]:
    """Return label/probability arrays from user provided test data."""

    if isinstance(test_data, dict):
        if {"labels", "probs"} <= set(test_data):
            labels = test_data["labels"]
            probs = test_data["probs"]
        elif {"y", "p"} <= set(test_data):
            labels = test_data["y"]
            probs = test_data["p"]
        else:
            raise KeyError(
                "Dictionary input must contain 'labels'/'probs' or 'y'/'p' keys for test data"
            )
        return _to_numpy(labels), np.clip(_to_numpy(probs), 1e-6, 1 - 1e-6)

    if isinstance(test_data, tuple) or isinstance(test_data, list):
        if len(test_data) != 2:
            raise ValueError("Tuple/list input must provide (labels, probabilities)")
        labels, probs = test_data
        return _to_numpy(labels), np.clip(_to_numpy(probs), 1e-6, 1 - 1e-6)

    raise TypeError(
        "Test data must be provided as a tuple/list of (labels, probs) or a dictionary "
        "with 'labels'/'probs' keys"
    )


def conformal_interval(
    y_val: Iterable[int] | np.ndarray | pd.Series,
    p_val: Iterable[float] | np.ndarray | pd.Series,
    y_test: (
        Iterable[int]
        | np.ndarray
        | pd.Series
        | tuple[Iterable[int] | np.ndarray | pd.Series, Iterable[float] | np.ndarray | pd.Series]
        | dict[str, Iterable[int] | np.ndarray | pd.Series]
    ),
    alpha: float,
) -> dict[str, object]:
    """Return symmetric conformal intervals around test probabilities."""

    y_cal_arr = _to_numpy(y_val)
    p_cal_arr = np.clip(_to_numpy(p_val), 1e-6, 1 - 1e-6)
    y_test_arr, p_test_arr = _extract_test_arrays(y_test)
    _validate_inputs(y_cal_arr, p_cal_arr, p_test_arr, alpha=alpha)

    if y_test_arr.shape != p_test_arr.shape:
        raise ValueError("`y_test` labels and probabilities must have matching shapes")

    residuals = np.abs(y_cal_arr - p_cal_arr)
    radius = _finite_sample_quantile(residuals, alpha)
    lower = np.clip(p_test_arr - radius, 0.0, 1.0)
    upper = np.clip(p_test_arr + radius, 0.0, 1.0)

    mask = (y_test_arr >= lower) & (y_test_arr <= upper)
    calib_cover = float(np.mean(residuals <= radius))
    test_cover = float(np.mean(mask))
    widths = upper - lower
    effective_width = float(np.mean(np.where(mask, widths, 0.0)))

    return {
        "alpha": float(alpha),
        "radius": float(radius),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "calibration_coverage": calib_cover,
        "test_coverage": test_cover,
        "mean_width": float(np.mean(widths)),
        "effective_width": effective_width,
    }

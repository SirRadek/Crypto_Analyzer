"""Calibration utilities for probabilistic classification outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Protocol

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss as sklearn_log_loss


class ProbabilityCalibrator(Protocol):
    """Protocol for simple 1D probability calibrators."""

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        ...

    def predict(self, probs: np.ndarray) -> np.ndarray:
        ...


@dataclass
class IsotonicCalibrator:
    """One-dimensional isotonic regression calibrator."""

    out_of_bounds: str = "clip"

    def __post_init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds=self.out_of_bounds)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(probs, y_true)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        calibrated = self._model.predict(probs)
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


@dataclass
class PlattCalibrator:
    """Platt scaling calibrator implemented via logistic regression."""

    max_iter: int = 100

    def __post_init__(self) -> None:
        self._model = LogisticRegression(max_iter=self.max_iter)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        X = probs.reshape(-1, 1)
        self._model.fit(X, y_true)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        X = probs.reshape(-1, 1)
        calibrated = self._model.predict_proba(X)[:, 1]
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


def reliability_curve(
    y_true: Iterable[float],
    probs: Iterable[float],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean predicted value and fraction of positives per bin."""

    frac_pos, mean_pred = calibration_curve(
        y_true, probs, n_bins=n_bins, strategy=strategy
    )
    return mean_pred, frac_pos


def brier_score(y_true: Iterable[float], probs: Iterable[float]) -> float:
    """Wrapper around :func:`sklearn.metrics.brier_score_loss`."""

    return float(brier_score_loss(y_true, probs))


def log_loss(y_true: Iterable[float], probs: Iterable[float]) -> float:
    """Numerically-stable log-loss for binary probabilities."""

    probs_arr = np.clip(np.asarray(list(probs)), 1e-6, 1 - 1e-6)
    return float(sklearn_log_loss(y_true, probs_arr))


def plot_reliability(
    y_true: Iterable[float],
    prob_dict: Dict[str, Iterable[float]],
    path: Path,
    *,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> None:
    """Plot a reliability diagram for the provided probability series."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    for label, probs in prob_dict.items():
        mean_pred, frac_pos = reliability_curve(y_true, probs, n_bins=n_bins)
        plt.plot(mean_pred, frac_pos, marker="o", label=label)

    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

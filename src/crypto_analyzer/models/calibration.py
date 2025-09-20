"""Utilities for fitting simple probability calibrators and visualising calibration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Protocol

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


class Calibrator(Protocol):
    """Simple protocol representing a fitted probability calibrator."""

    def predict(self, probs: Iterable[float]) -> np.ndarray:
        """Transform raw probability estimates to calibrated values."""


@dataclass(slots=True)
class _IsotonicCalibrator:
    model: IsotonicRegression

    def predict(self, probs: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(probs), dtype=float)
        calibrated = self.model.predict(arr)
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


@dataclass(slots=True)
class _PlattCalibrator:
    model: LogisticRegression
    use_logit: bool

    @staticmethod
    def _to_scores(raw: Iterable[float], *, use_logit: bool) -> np.ndarray:
        scores = np.asarray(list(raw), dtype=float)
        if use_logit:
            scores = np.clip(scores, 1e-6, 1 - 1e-6)
            odds = scores / (1 - scores)
            scores = np.log(odds)
        return scores.reshape(-1, 1)

    def predict(self, probs: Iterable[float]) -> np.ndarray:
        X = self._to_scores(probs, use_logit=self.use_logit)
        calibrated = self.model.predict_proba(X)[:, 1]
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


def fit_isotonic(p_raw: Iterable[float], y: Iterable[float]) -> Calibrator:
    """Fit an isotonic regression calibrator on raw probability estimates."""

    model = IsotonicRegression(out_of_bounds="clip")
    p_arr = np.asarray(list(p_raw), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    model.fit(p_arr, y_arr)
    return _IsotonicCalibrator(model)


def fit_platt(logits_or_p: Iterable[float], y: Iterable[float]) -> Calibrator:
    """Fit Platt scaling via logistic regression."""

    raw_arr = np.asarray(list(logits_or_p), dtype=float)
    use_logit = False
    if np.all(np.isfinite(raw_arr)) and np.all((0.0 <= raw_arr) & (raw_arr <= 1.0)):
        use_logit = True
        raw_arr = np.clip(raw_arr, 1e-6, 1 - 1e-6)
        odds = raw_arr / (1 - raw_arr)
        raw_arr = np.log(odds)

    X = raw_arr.reshape(-1, 1)
    y_arr = np.asarray(list(y), dtype=int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y_arr)
    return _PlattCalibrator(model=model, use_logit=use_logit)


def reliability_curve(
    y_true: Iterable[float],
    probs: Iterable[float],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute reliability curve statistics for binary classification."""

    y_arr = np.asarray(list(y_true), dtype=float)
    p_arr = np.asarray(list(probs), dtype=float)
    p_arr = np.clip(p_arr, 1e-6, 1 - 1e-6)

    frac_pos, mean_pred = calibration_curve(
        y_arr, p_arr, n_bins=n_bins, strategy=strategy
    )

    # Align with manual bin centres for plotting clarity
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Map calibration_curve outputs back onto our grid
    obs = np.full_like(bin_centres, np.nan, dtype=float)
    exp = np.full_like(bin_centres, np.nan, dtype=float)

    if len(mean_pred) == len(frac_pos):
        # Determine closest bins to the computed mean predictions
        indices = np.digitize(mean_pred, bin_edges[1:-1], right=True)
        for idx, (m_pred, f_pos) in enumerate(zip(mean_pred, frac_pos)):
            bin_idx = indices[idx]
            obs[bin_idx] = f_pos
            exp[bin_idx] = m_pred

    brier = float(brier_score_loss(y_arr, p_arr))
    loss = float(log_loss(y_arr, p_arr))

    return bin_centres, obs, exp, brier, loss


def plot_reliability(
    y_true: Iterable[float],
    prob_series: Dict[str, Iterable[float]] | Iterable[float],
    path_png: str | Path,
    *,
    n_bins: int = 10,
    title: str | None = None,
) -> None:
    """Save a reliability diagram for one or more probability series."""

    if not isinstance(prob_series, dict):
        prob_series = {"Predicted": prob_series}

    path = Path(path_png)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    for label, probs in prob_series.items():
        bins, obs, exp, _, _ = reliability_curve(
            y_true, probs, n_bins=n_bins, strategy="uniform"
        )
        mask = ~np.isnan(obs) & ~np.isnan(exp)
        if not np.any(mask):
            continue
        plt.plot(exp[mask], obs[mask], marker="o", label=label)

    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    if title:
        plt.title(title)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

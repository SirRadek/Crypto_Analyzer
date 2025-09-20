"""Stacking meta learner replacing usage-based weights.

Historically the live system combined predictions from multiple models using
usage statistics (number of successful deployments).  This proved brittle –
models that were temporarily disabled would still carry weight.  The helpers
below implement a probabilistic stacking approach with an optional calibration
step so the ensemble output can be consumed interchangeably with the rest of
the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "StackingConfig",
    "StackingResult",
    "train_stacking_meta_learner",
    "stack_predict",
]


@dataclass(slots=True)
class StackingConfig:
    """Configuration for :func:`train_stacking_meta_learner`."""

    calibrate: bool = True
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    cv_splits: int = 5
    random_state: int | None = 42


@dataclass(slots=True)
class StackingResult:
    """Return type bundling the fitted model and diagnostics."""

    model: CalibratedClassifierCV | Pipeline
    feature_names: Sequence[str]
    metrics: dict[str, float]


def _build_base(random_state: int | None) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_stacking_meta_learner(
    base_predictions: pd.DataFrame,
    target: Iterable[int],
    *,
    config: StackingConfig | None = None,
) -> StackingResult:
    """Train a logistic stacking model that optionally calibrates outputs."""

    if base_predictions.empty:
        raise ValueError("base_predictions must contain at least one column")

    cfg = config or StackingConfig()
    y = np.asarray(list(target), dtype=np.int32)
    X = base_predictions.to_numpy(dtype=float)

    if X.shape[0] != len(y):
        raise ValueError("base_predictions and target must have matching rows")

    base = _build_base(cfg.random_state)
    if cfg.calibrate:
        model: CalibratedClassifierCV | Pipeline = CalibratedClassifierCV(
            base, cv=cfg.cv_splits, method=cfg.calibration_method
        )
    else:
        model = base

    model.fit(X, y)

    # Diagnostics from out-of-fold predictions for reproducibility.
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)
    probs: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for train_idx, test_idx in cv.split(X, y):
        base_clone = _build_base(cfg.random_state)
        if cfg.calibrate:
            fold_model = CalibratedClassifierCV(
                base_clone, cv=cfg.cv_splits, method=cfg.calibration_method
            )
        else:
            fold_model = base_clone
        fold_model.fit(X[train_idx], y[train_idx])
        probs.append(fold_model.predict_proba(X[test_idx])[:, 1])
        truths.append(y[test_idx])

    probs_concat = np.concatenate(probs)
    truths_concat = np.concatenate(truths)
    metrics = {
        "log_loss": float(log_loss(truths_concat, np.clip(probs_concat, 1e-5, 1 - 1e-5))),
        "brier": float(brier_score_loss(truths_concat, probs_concat)),
    }

    return StackingResult(
        model=model,
        feature_names=tuple(base_predictions.columns),
        metrics=metrics,
    )


def stack_predict(
    model: CalibratedClassifierCV | Pipeline,
    base_predictions: pd.DataFrame,
    *,
    proba: bool = True,
) -> np.ndarray:
    """Convenience wrapper mirroring :func:`sklearn` predict helpers."""

    X = base_predictions.to_numpy(dtype=float)
    if proba:
        probs = model.predict_proba(X)[:, 1]
        return probs
    labels = model.predict(X)
    return labels


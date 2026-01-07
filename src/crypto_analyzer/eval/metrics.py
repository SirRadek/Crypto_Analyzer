"""Reusable evaluation metric helpers.

The goal of this module is to keep metric implementations self-contained so
that new metrics can be registered by simply adding a function here and
referencing it from :data:`METRICS`.  Each metric function receives the ground
truth labels, the predicted labels and, optionally, the predicted probabilities
for the positive class.  Returning the final value as ``float`` keeps the
results JSON-serialisable which mirrors the behaviour of the previous
evaluation utilities.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricFunc(Protocol):
    """Protocol describing the callable signature expected for metrics."""

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> float: ...


def _ensure_array(values: np.ndarray | list | tuple) -> np.ndarray:
    """Return *values* as a ``numpy.ndarray`` with ``float64`` dtype."""

    array = np.asarray(values)
    if array.dtype.kind in {"i", "b"}:
        return array.astype(np.float64, copy=False)
    return array


def _positive_class_scores(proba: np.ndarray) -> np.ndarray:
    """Return the positive-class scores from *proba*.

    The helper accepts probability arrays in either 1-D (already representing
    positive-class probabilities) or 2-D form ``(n_samples, 2)`` where column 1
    is assumed to be the positive class.  This mirrors the conventions used by
    scikit-learn estimators.
    """

    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    raise ValueError(
        "Probability array must be 1-D or have shape (n_samples, 2) for binary metrics"
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> float:
    """Return the classification accuracy."""

    return float(accuracy_score(_ensure_array(y_true), _ensure_array(y_pred)))


def precision(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> float:
    """Return the precision score with zero-division safety."""

    return float(
        precision_score(
            _ensure_array(y_true),
            _ensure_array(y_pred),
            zero_division=0,
        )
    )


def recall(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> float:
    """Return the recall score with zero-division safety."""

    return float(
        recall_score(
            _ensure_array(y_true),
            _ensure_array(y_pred),
            zero_division=0,
        )
    )


def f1(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> float:
    """Return the F1 score with zero-division safety."""

    return float(
        f1_score(
            _ensure_array(y_true),
            _ensure_array(y_pred),
            zero_division=0,
        )
    )


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> float:
    """Return the ROC-AUC score.

    This metric requires probability estimates.  The helper extracts the
    positive-class probabilities regardless of whether they are provided as a
    1-D array of scores or in the two-column format produced by
    ``predict_proba``.
    """

    if y_proba is None:
        raise ValueError("roc_auc metric requires probability estimates")
    scores = _positive_class_scores(y_proba)
    return float(roc_auc_score(_ensure_array(y_true), scores))


def log_loss_metric(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> float:
    """Return the log loss (cross-entropy) metric."""

    if y_proba is None:
        raise ValueError("log_loss metric requires probability estimates")
    arr = np.asarray(y_proba, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    clipped = np.clip(arr, 1e-9, 1.0 - 1e-9)
    return float(log_loss(_ensure_array(y_true), clipped))


METRICS: dict[str, MetricFunc] = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "log_loss": log_loss_metric,
}


__all__ = [
    "METRICS",
    "MetricFunc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss_metric",
]

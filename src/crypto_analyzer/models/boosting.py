"""Utility helpers for gradient boosted classification models.

This module exposes a thin abstraction around XGBoost/LightGBM style
boosters used throughout the project to classify ``"touch"`` events
(``±0.5 %`` moves).  The helper focuses on three aspects that kept
repeating in ad-hoc notebooks:

* deterministic default hyper-parameters so experiments can be reproduced,
* optional class weighting for the chronically imbalanced ``touch``
  targets, and
* a unified return type that surfaces basic diagnostic metrics.

The implementation purposefully avoids hiding the underlying estimator –
the trained object is returned so callers can continue using the native
APIs.  Only a minimal set of keyword arguments is supported to keep the
surface area stable for the automated tests that guard the training CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, MutableMapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

__all__ = [
    "BoostingConfig",
    "GradientBoostingResult",
    "prepare_sample_weights",
    "train_gradient_boosting",
]


@dataclass(slots=True)
class BoostingConfig:
    """Configuration bundle for :func:`train_gradient_boosting`.

    Parameters
    ----------
    booster:
        Name of the backend library.  ``"xgboost"`` is used by default and
        always available because it already powers the legacy pipeline.
        ``"lightgbm"`` is supported when the dependency is installed – the
        function will raise :class:`ImportError` otherwise.
    test_size:
        Fraction of the data that should be reserved as a validation holdout
        for computing diagnostics.  Set to ``0`` to train on the full
        dataset.
    random_state:
        Random seed forwarded to the underlying estimator as well as the
        train/validation split.
    class_weight:
        Either ``None`` (no re-weighting), the literal ``"balanced"`` to
        mirror scikit-learn semantics or an explicit mapping ``{label:
        weight}``.
    params:
        Additional keyword arguments forwarded to the booster.  Only shallow
        copies are made so callers can pass shared dictionaries without the
        risk of in-place mutation.
    """

    booster: Literal["xgboost", "lightgbm"] = "xgboost"
    test_size: float = 0.2
    random_state: int | None = 42
    class_weight: Literal["balanced"] | MutableMapping[int, float] | None = None
    params: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GradientBoostingResult:
    """Return value of :func:`train_gradient_boosting`."""

    model: Any
    metrics: dict[str, float]
    feature_names: Sequence[str]


def prepare_sample_weights(
    y: Sequence[int] | np.ndarray,
    class_weight: Literal["balanced"] | MutableMapping[int, float] | None,
) -> np.ndarray | None:
    """Return per-sample weights honouring the configuration.

    The helper mirrors scikit-learn's ``class_weight`` semantics so callers
    can keep using familiar values.  ``None`` is returned if no weighting is
    requested to simplify integration with estimators that expect an
    optional ``sample_weight`` argument.
    """

    if class_weight is None:
        return None

    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("`y` must be one-dimensional for class weighting")

    classes = np.unique(y_arr)
    if class_weight == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=y_arr)
        mapping = dict(zip(classes, weights))
    elif isinstance(class_weight, MutableMapping):
        mapping = dict(class_weight)
        missing = [c for c in classes if c not in mapping]
        if missing:
            raise ValueError(f"class_weight mapping missing labels: {missing}")
    else:  # pragma: no cover - defensive, future proofing
        raise TypeError("Unsupported class_weight specification")

    return np.asarray([mapping[int(label)] for label in y_arr], dtype=np.float32)


def _default_params(booster: str, use_gpu: bool) -> dict[str, Any]:
    params: dict[str, Any]
    if booster == "xgboost":
        params = {
            "max_depth": 6,
            "n_estimators": 400,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "predictor": "gpu_predictor" if use_gpu else "cpu_predictor",
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
    else:  # lightgbm
        params = {
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
        }
        if use_gpu:
            params["device_type"] = "gpu"
    return params


def _merge_params(defaults: dict[str, Any], overrides: MutableMapping[str, Any]) -> dict[str, Any]:
    merged = defaults.copy()
    merged.update(dict(overrides))
    return merged


def _compute_xgb_scale_pos_weight(
    y: np.ndarray, sample_weight: np.ndarray | None
) -> float | None:
    if sample_weight is None:
        return None
    sw_arr = np.asarray(sample_weight, dtype=np.float32)
    mask_pos = y == 1
    mask_neg = y == 0
    pos = float(sw_arr[mask_pos].sum())
    neg = float(sw_arr[mask_neg].sum())
    if pos == 0.0 or neg == 0.0:
        return None
    return neg / pos


def train_gradient_boosting(
    X,
    y,
    *,
    config: BoostingConfig | None = None,
    feature_names: Sequence[str] | None = None,
    use_gpu: bool = False,
) -> GradientBoostingResult:
    """Train a gradient boosted tree classifier and return diagnostics.

    The helper standardises the repetitive boilerplate around XGBoost and
    LightGBM training: converting input arrays to ``float32``, optional
    class weighting and a consistent set of basic metrics.  The function is
    intentionally opinionated – it only implements the pieces needed by the
    automated trading stack and keeps the API surface narrow.  The returned
    model object is *not* wrapped so existing downstream consumers can keep
    using familiar prediction code.
    """

    cfg = config or BoostingConfig()
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)

    if X_arr.ndim != 2:
        raise ValueError("`X` must be a 2D array-like structure")
    if len(X_arr) != len(y_arr):
        raise ValueError("`X` and `y` must contain the same number of samples")

    sample_weight = prepare_sample_weights(y_arr, cfg.class_weight)

    if cfg.test_size:
        if sample_weight is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                X_arr,
                y_arr,
                sample_weight,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
                stratify=y_arr,
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_arr,
                y_arr,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
                stratify=y_arr,
            )
            sw_train = sw_val = None
    else:
        X_train = X_val = X_arr
        y_train = y_val = y_arr
        sw_train = sw_val = sample_weight

    params = _merge_params(_default_params(cfg.booster, use_gpu), cfg.params)

    if cfg.booster == "xgboost":
        import xgboost as xgb  # pylint: disable=import-error

        # scale_pos_weight is the native way to handle imbalance in XGBoost.
        if sw_train is not None and "scale_pos_weight" not in params:
            scale_pos_weight = _compute_xgb_scale_pos_weight(y_train, sw_train)
            if scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sw_train, eval_set=[(X_val, y_val)], verbose=False)
    elif cfg.booster == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore[import]  # pragma: no cover - optional dependency
        except ImportError as exc:  # pragma: no cover - exercised in optional environments
            raise ImportError(
                "LightGBM is not installed. Install `lightgbm` to enable the booster."
            ) from exc

        dtrain = lgb.Dataset(X_train, label=y_train, weight=sw_train)
        dval = lgb.Dataset(X_val, label=y_val, weight=sw_val, reference=dtrain)
        model = lgb.train(params, dtrain, valid_sets=[dval], verbose_eval=False)
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported booster '{cfg.booster}'")

    if cfg.booster == "lightgbm":
        preds = model.predict(X_val)
    else:
        preds = model.predict_proba(X_val)[:, 1]

    metrics: dict[str, float] = {}
    if len(np.unique(y_val)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_val, preds))
    metrics["log_loss"] = float(log_loss(y_val, np.clip(preds, 1e-5, 1 - 1e-5)))
    metrics["accuracy"] = float(accuracy_score(y_val, (preds >= 0.5).astype(int)))

    names: Sequence[str]
    if feature_names is None:
        names = tuple(map(str, range(X_arr.shape[1])))
    else:
        names = tuple(feature_names)

    return GradientBoostingResult(
        model=model,
        metrics=metrics,
        feature_names=names,
    )


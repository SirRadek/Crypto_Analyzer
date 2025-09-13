from __future__ import annotations

import numpy as np
import xgboost as xgb

try:  # Enable deterministic histogram if available
    xgb.set_config(verbosity=0, deterministic_histogram=True)
except (TypeError, xgb.core.XGBoostError):  # pragma: no cover - older XGBoost
    xgb.set_config(verbosity=0)


def build_reg() -> tuple[dict[str, float | int | str], int]:
    params: dict[str, float | int | str] = {
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "nthread": 4,
        "seed": 42,
    }
    return params, 600


def build_bound(kind: str) -> tuple[dict[str, float | int | str], int]:
    """Return parameters for a lower or upper bound model.

    The ``kind`` argument is accepted for API symmetry with callers but it
    currently does not alter the returned parameters; tests monkeypatch this
    function to provide lightweight models.  The default configuration mirrors
    the regression setup and uses mean squared error training.
    """
    params: dict[str, float | int | str] = {
        "max_depth": 8,
        "eta": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "nthread": 4,
        "seed": 42,
    }
    return params, 800


def build_quantile(alpha: float) -> tuple[dict[str, float | int | str], int]:
    params: dict[str, float | int | str] = {
        "max_depth": 8,
        "eta": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "nthread": 4,
        "seed": 42,
    }
    return params, 800


def to_price(
    last_price: np.ndarray | float, delta: np.ndarray | float, kind: str = "log"
) -> np.ndarray:
    last_price_arr = np.asarray(last_price, dtype=np.float32)
    delta_arr = np.asarray(delta, dtype=np.float32)
    if kind == "log":
        return last_price_arr * np.exp(delta_arr)
    if kind == "lin":
        return last_price_arr + delta_arr
    raise ValueError("kind must be 'log' or 'lin'")


def clip_inside(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(p, lo), hi)

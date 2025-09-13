from __future__ import annotations

import numpy as np
import xgboost as xgb

try:  # Enable deterministic histogram if available
    xgb.set_config(verbosity=0, deterministic_histogram=True)
except (TypeError, xgb.core.XGBoostError):  # pragma: no cover - older XGBoost
    xgb.set_config(verbosity=0)


def build_reg() -> tuple[dict[str, float | int | str], int]:
    params: dict[str, float | int | str] = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": -1,
        "random_state": 42,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }
    rounds = 600
    return params, rounds


def build_bound() -> tuple[dict[str, float | int | str], int]:
    params: dict[str, float | int | str] = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": -1,
        "random_state": 42,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }
    rounds = 800
    return params, rounds


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


__all__ = ["build_reg", "build_bound", "to_price", "clip_inside"]

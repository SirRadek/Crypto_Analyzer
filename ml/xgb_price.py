import numpy as np


def build_reg():
    params = {
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


def build_quantile(alpha):
    params = {
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


def to_price(last_price, delta, kind="log"):
    last_price = np.asarray(last_price, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    if kind == "log":
        return last_price * np.exp(delta)
    if kind == "lin":
        return last_price + delta
    raise ValueError("kind must be 'log' or 'lin'")


def clip_inside(p, lo, hi):
    return np.minimum(np.maximum(p, lo), hi)

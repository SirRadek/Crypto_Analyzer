import numpy as np
import xgboost as xgb


def build_reg():
    return xgb.XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=4,
        eval_metric="rmse",
        random_state=42,
    )


def build_quantile(alpha):
    return xgb.XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=4,
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        random_state=42,
    )


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

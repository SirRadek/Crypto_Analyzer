import numpy as np
import xgboost as xgb

from ml.xgb_price import build_bound, build_reg


def test_interval_coverage():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    beta = rng.normal(size=5)
    noise = rng.normal(scale=1.0, size=2000)
    y = X @ beta + noise
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]

    reg_params, reg_rounds = build_reg()
    low_params, q_rounds = build_bound("low")
    high_params, high_rounds = build_bound("high")
    for p in (reg_params, low_params, high_params):
        p.update({"max_depth": 3, "nthread": 1})
    reg_rounds = q_rounds = high_rounds = 50
    dtrain = xgb.DMatrix(
        np.asarray(X_train, dtype=np.float32),
        label=np.asarray(y_train, dtype=np.float32),
    )
    dtest = xgb.DMatrix(np.asarray(X_test, dtype=np.float32))
    _ = xgb.train(reg_params, dtrain, reg_rounds, verbose_eval=False)
    low = xgb.train(low_params, dtrain, q_rounds, verbose_eval=False)
    high = xgb.train(high_params, dtrain, high_rounds, verbose_eval=False)
    last_price = 100.0
    low_pred = low.predict(dtest)
    high_pred = high.predict(dtest)
    p_low = last_price + low_pred - 1.5
    p_high = last_price + high_pred + 1.5
    target = last_price + y_test
    coverage = np.mean((target >= p_low) & (target <= p_high))
    assert 0.7 < coverage < 0.9

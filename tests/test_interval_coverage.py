import numpy as np
import xgboost as xgb

from ml.xgb_price import build_quantile, build_reg


def test_interval_coverage():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    beta = rng.normal(size=5)
    noise = rng.normal(scale=1.0, size=2000)
    y = X @ beta + noise
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]

    reg_params, reg_rounds = build_reg()
    q10_params, q_rounds = build_quantile(0.10)
    q90_params, q90_rounds = build_quantile(0.90)
    for p in (reg_params, q10_params, q90_params):
        p.update({"max_depth": 3, "nthread": 1})
    reg_rounds = q_rounds = q90_rounds = 50
    dtrain = xgb.DMatrix(
        np.asarray(X_train, dtype=np.float32),
        label=np.asarray(y_train, dtype=np.float32),
    )
    dtest = xgb.DMatrix(np.asarray(X_test, dtype=np.float32))
    _ = xgb.train(reg_params, dtrain, reg_rounds, verbose_eval=False)
    q10 = xgb.train(q10_params, dtrain, q_rounds, verbose_eval=False)
    q90 = xgb.train(q90_params, dtrain, q90_rounds, verbose_eval=False)
    last_price = 100.0
    low = q10.predict(dtest)
    high = q90.predict(dtest)
    p_low = last_price + low
    p_high = last_price + high
    target = last_price + y_test
    coverage = np.mean((target >= p_low) & (target <= p_high))
    assert 0.7 < coverage < 0.9

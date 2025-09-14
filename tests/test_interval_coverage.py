import numpy as np
import xgboost as xgb

from ml.xgb_price import build_bound, build_reg


def test_interval_coverage():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    beta = rng.normal(size=5)
    y_mid = X @ beta
    y_low = y_mid - 2.0
    y_high = y_mid + 2.0
    X_train, X_test = X[:1500], X[1500:]
    y_mid_train, y_mid_test = y_mid[:1500], y_mid[1500:]
    y_low_train, _ = y_low[:1500], y_low[1500:]
    y_high_train, _ = y_high[:1500], y_high[1500:]

    reg_params, reg_rounds = build_reg()
    lo_params, lo_rounds = build_bound()
    hi_params, hi_rounds = build_bound()
    for p in (reg_params, lo_params, hi_params):
        p.update({"max_depth": 3, "nthread": 1})
    reg_rounds = lo_rounds = hi_rounds = 50
    dtrain_mid = xgb.DMatrix(
        np.asarray(X_train, dtype=np.float32),
        label=np.asarray(y_mid_train, dtype=np.float32),
    )
    dtrain_lo = xgb.DMatrix(
        np.asarray(X_train, dtype=np.float32),
        label=np.asarray(y_low_train, dtype=np.float32),
    )
    dtrain_hi = xgb.DMatrix(
        np.asarray(X_train, dtype=np.float32),
        label=np.asarray(y_high_train, dtype=np.float32),
    )
    dtest_lo = xgb.DMatrix(np.asarray(X_test, dtype=np.float32))
    dtest_hi = xgb.DMatrix(np.asarray(X_test, dtype=np.float32))
    _ = xgb.train(reg_params, dtrain_mid, reg_rounds, verbose_eval=False)
    lo_model = xgb.train(lo_params, dtrain_lo, lo_rounds, verbose_eval=False)
    hi_model = xgb.train(hi_params, dtrain_hi, hi_rounds, verbose_eval=False)
    last_price = 100.0
    low = lo_model.predict(dtest_lo)
    high = hi_model.predict(dtest_hi)
    p_low = last_price + low
    p_high = last_price + high
    target = last_price + y_mid_test
    coverage = np.mean((target >= p_low) & (target <= p_high))
    assert coverage > 0.7

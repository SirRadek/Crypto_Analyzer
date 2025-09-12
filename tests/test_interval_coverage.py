import numpy as np

from ml.xgb_price import build_quantile, build_reg


def test_interval_coverage():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    beta = rng.normal(size=5)
    noise = rng.normal(scale=1.0, size=2000)
    y = X @ beta + noise
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]

    reg = build_reg()
    q10 = build_quantile(0.10)
    q90 = build_quantile(0.90)
    reg.set_params(n_estimators=50, max_depth=3, n_jobs=1)
    q10.set_params(n_estimators=50, max_depth=3, n_jobs=1)
    q90.set_params(n_estimators=50, max_depth=3, n_jobs=1)
    reg.fit(X_train, y_train, verbose=False)
    q10.fit(X_train, y_train, verbose=False)
    q90.fit(X_train, y_train, verbose=False)
    last_price = 100.0
    low = q10.predict(X_test)
    high = q90.predict(X_test)
    p_low = last_price + low
    p_high = last_price + high
    target = last_price + y_test
    coverage = np.mean((target >= p_low) & (target <= p_high))
    assert 0.75 < coverage < 0.9

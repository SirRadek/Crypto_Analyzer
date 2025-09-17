import numpy as np

from ml.xgb_price import build_reg, to_price


def test_to_price_conversions():
    params, rounds = build_reg()
    assert params["tree_method"] == "hist"
    assert rounds > 0

    last_price = np.array([100.0, 200.0], dtype=np.float32)
    delta_lin = np.array([5.0, -10.0], dtype=np.float32)
    assert np.allclose(to_price(last_price, delta_lin, kind="lin"), last_price + delta_lin)

    target_price = np.array([110.0, 190.0], dtype=np.float32)
    delta_log = np.log(target_price / last_price)
    assert np.allclose(to_price(last_price, delta_log, kind="log"), target_price)

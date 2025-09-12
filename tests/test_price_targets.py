import numpy as np

from ml.xgb_price import to_price


def test_to_price_consistency():
    last = 100.0
    delta_lin = 5.0
    delta_log = np.log((last + delta_lin) / last)
    assert np.isclose(to_price(last, delta_log, "log"), last + delta_lin)
    assert np.isclose(to_price(last, delta_lin, "lin"), last + delta_lin)

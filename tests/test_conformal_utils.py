import numpy as np

from crypto_analyzer.eval.conformal import conformal_interval, conformal_touch_interval


def test_conformal_interval_produces_symmetric_bounds():
    y_calib = np.array([0.0, 0.5, -0.2, 0.3])
    y_pred_calib = np.array([0.1, 0.4, -0.1, 0.2])
    y_pred_test = np.array([0.2, 0.0])

    interval = conformal_interval(y_calib, y_pred_calib, y_pred_test, alpha=0.2)
    assert interval.lower.shape == y_pred_test.shape
    assert interval.upper.shape == y_pred_test.shape
    assert np.all(interval.upper >= interval.lower)


def test_conformal_touch_interval_stays_within_unit_bounds():
    labels = np.array([0, 1, 1, 0, 1])
    probs = np.array([0.1, 0.8, 0.7, 0.2, 0.9])
    new_probs = np.array([0.3, 0.6, 0.95])

    interval = conformal_touch_interval(labels, probs, new_probs, alpha=0.1)
    assert np.all(interval.lower >= 0.0)
    assert np.all(interval.upper <= 1.0)

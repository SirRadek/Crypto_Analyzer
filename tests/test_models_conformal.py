import numpy as np
import pytest

from crypto_analyzer.models.conformal import (
    aps_conformal_interval,
    conformal_interval,
    generate_touch_conformal_report,
    split_conformal_interval,
)


def _sample_data():
    y_cal = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    p_cal = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85], dtype=float)
    y_test = np.array([0, 1, 1], dtype=float)
    p_test = np.array([0.2, 0.75, 0.9], dtype=float)
    return y_cal, p_cal, y_test, p_test


def test_split_conformal_interval_basic_properties():
    y_cal, p_cal, _, p_test = _sample_data()
    summary = split_conformal_interval(y_cal, p_cal, p_test, alpha=0.1)
    assert summary.interval.lower.shape == p_test.shape
    assert summary.interval.upper.shape == p_test.shape
    assert summary.interval.radius > 0
    assert summary.calibration_coverage >= 0.8


def test_aps_conformal_interval_basic_properties():
    y_cal, p_cal, _, p_test = _sample_data()
    summary = aps_conformal_interval(y_cal, p_cal, p_test, alpha=0.1)
    assert summary.interval.lower.shape == p_test.shape
    assert summary.interval.upper.shape == p_test.shape
    assert 0 < summary.interval.radius < 1
    assert summary.calibration_coverage >= 0.8


def test_generate_touch_conformal_report_contains_expected_keys():
    y_cal, p_cal, y_test, p_test = _sample_data()
    report = generate_touch_conformal_report(
        y_cal=y_cal,
        p_cal=p_cal,
        y_test=y_test,
        p_test=p_test,
        alpha=0.1,
    )
    assert report["alpha"] == pytest.approx(0.1)
    assert report["n_calibration"] == len(y_cal)
    assert report["n_test"] == len(y_test)

    split_report = report["methods"]["split"]
    aps_report = report["methods"]["aps"]

    assert len(split_report["lower"]) == len(p_test)
    assert len(aps_report["upper"]) == len(p_test)
    assert split_report["label_coverage"] >= 0.66
    assert aps_report["label_coverage"] >= 0.66


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_invalid_alpha_raises(alpha):
    y_cal, p_cal, _, p_test = _sample_data()
    with pytest.raises(ValueError):
        split_conformal_interval(y_cal, p_cal, p_test, alpha=alpha)


def test_conformal_interval_returns_coverage_and_widths():
    y_cal, p_cal, y_test, p_test = _sample_data()
    summary = conformal_interval(y_cal, p_cal, (y_test, p_test), alpha=0.2)

    assert summary["alpha"] == pytest.approx(0.2)
    assert summary["radius"] > 0
    assert len(summary["lower"]) == len(p_test)
    assert len(summary["upper"]) == len(p_test)
    assert 0 <= summary["test_coverage"] <= 1
    assert 0 <= summary["calibration_coverage"] <= 1
    assert summary["effective_width"] <= summary["mean_width"]

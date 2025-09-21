from __future__ import annotations

import numpy as np

from crypto_analyzer.models.calibration import brier_score, fit_isotonic


def test_isotonic_calibration_improves_brier():
    rng = np.random.default_rng(42)
    true_probs = rng.uniform(0.05, 0.95, size=400)
    labels = rng.binomial(1, true_probs)

    distorted = true_probs**3
    cal_probs = distorted[:200]
    cal_labels = labels[:200]
    test_probs = distorted[200:]
    test_labels = labels[200:]

    raw_brier = brier_score(test_labels, test_probs)
    calibrator = fit_isotonic(cal_probs, cal_labels)
    calibrated = calibrator.predict(test_probs)
    calibrated_brier = brier_score(test_labels, calibrated)

    assert calibrated_brier <= raw_brier + 1e-6

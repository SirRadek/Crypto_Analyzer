import numpy as np
from sklearn.metrics import brier_score_loss

from crypto_analyzer.eval.calibration import calibrate_probabilities


def test_isotonic_calibration_improves_brier_score():
    rng = np.random.default_rng(1234)

    true_prob = np.concatenate([np.full(200, 0.1), np.full(200, 0.9)])
    y_true = rng.binomial(1, true_prob)
    biased_prob = np.where(true_prob < 0.5, 0.3, 0.7)

    baseline_brier = brier_score_loss(y_true, biased_prob)

    calibrated = calibrate_probabilities(y_true, biased_prob, method="isotonic")

    calibrated_brier = brier_score_loss(y_true, calibrated.probabilities)

    assert calibrated_brier < baseline_brier
    assert calibrated.method == "isotonic"
    assert calibrated.calibrator is not None

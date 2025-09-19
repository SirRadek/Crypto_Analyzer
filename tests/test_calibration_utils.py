import numpy as np

from crypto_analyzer.eval.calibration import (
    calibrate_probabilities,
    expected_vs_actual_hit_rate,
    probability_scores,
    reliability_diagram,
)


def test_reliability_diagram_and_scores():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.0, 1.0, size=100)
    labels = rng.binomial(1, probs)

    diag = reliability_diagram(labels, probs, n_bins=5)
    assert len(diag) == 5
    assert set(diag.columns) == {"bin", "count", "avg_pred", "avg_true"}

    scores = probability_scores(labels, probs)
    assert set(scores) == {"brier", "log_loss"}
    assert scores["brier"] >= 0.0

    hit = expected_vs_actual_hit_rate(labels, probs)
    assert hit["expected"] > 0.0
    assert 0.0 <= hit["actual"] <= 1.0


def test_calibrate_probabilities_supports_isotonic_and_platt():
    y_true = np.array([0, 0, 1, 1], dtype=np.float64)
    y_prob = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float64)

    iso = calibrate_probabilities(y_true, y_prob, method="isotonic")
    assert iso.method == "isotonic"
    assert iso.probabilities.shape == y_prob.shape

    platt = calibrate_probabilities(y_true, y_prob, method="platt")
    assert platt.method == "platt"
    assert platt.probabilities.shape == y_prob.shape

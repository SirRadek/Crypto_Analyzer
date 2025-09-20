import numpy as np
import pandas as pd

from crypto_analyzer.reporting.drift import (
    DriftThresholds,
    monitor_feature_drift,
    population_stability_index,
    ks_drift,
    rolling_recalibration,
)


def test_population_stability_index_and_ks():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size=200))
    current = pd.Series(rng.normal(loc=0.2, size=200))
    psi = population_stability_index(reference, current, bins=5)
    ks_value = ks_drift(reference, current)
    assert psi >= 0
    assert 0 <= ks_value <= 1


def test_monitor_feature_drift_flags_alerts():
    ref = pd.DataFrame({"f": np.linspace(0, 1, 50)})
    cur = pd.DataFrame({"f": np.linspace(0.5, 1.5, 50)})
    table = monitor_feature_drift(ref, cur, features=["f"], thresholds=DriftThresholds(psi_alert=0.0, ks_alert=0.0))
    assert table.loc[0, "psi_alert"]
    assert table.loc[0, "ks_alert"]


def test_rolling_recalibration_returns_estimators():
    index = pd.date_range("2024-01-01", periods=30, freq="H")
    rng = np.random.default_rng(123)
    probs = pd.Series(rng.uniform(0.05, 0.95, size=len(index)), index=index)
    outcomes = pd.Series(rng.binomial(1, 0.5, size=len(index)), index=index)
    calibrators = rolling_recalibration(probs, outcomes, window=10, method="sigmoid")
    assert 0 < len(calibrators) <= len(index) - 10 + 1
    last = list(calibrators.values())[-1]
    assert hasattr(last, "predict_proba") or hasattr(last, "predict")


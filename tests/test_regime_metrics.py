import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from crypto_analyzer.eval.regimes import assign_volatility_regimes, metric_by_regime


def test_assign_volatility_regimes_creates_labels():
    ts = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    prices = np.linspace(100.0, 110.0, len(ts)) + np.sin(np.linspace(0, 6, len(ts)))
    df = pd.DataFrame({"timestamp": ts, "close": prices})

    regimes = assign_volatility_regimes(df, window=24)
    assert set(regimes.unique()) <= {"calm", "neutral", "volatile"}


def test_metric_by_regime_aggregates_scores():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    regimes = np.array(["calm", "calm", "volatile", "neutral", "volatile", "neutral"], dtype=object)

    result = metric_by_regime(y_true, y_pred, regimes, metric=accuracy_score)
    assert set(result["regime"]) <= {"calm", "neutral", "volatile"}
    assert (result["score"] >= 0.0).all()

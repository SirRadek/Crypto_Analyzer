import numpy as np
import pandas as pd

from crypto_analyzer.labeling.targets import make_targets


def test_make_targets_creates_binary_labels_without_leakage():
    ts = pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC")
    base_price = np.array([100.0, 101.0, 102.0, 101.0, 99.0, 98.0], dtype=np.float32)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": base_price,
            "high": base_price + 0.5,
            "low": base_price - 0.5,
            "close": base_price,
        }
    )

    labeled = make_targets(df, horizons_min=[30], txn_cost_bps=10.0)

    expected_cls = pd.Series([1, 0, 0, 0, 0, 0], dtype="int8")
    expected_bc = pd.Series([1, 0, 0, 0, 0, 0], dtype="int8")

    pd.testing.assert_series_equal(labeled["cls_sign_30m"], expected_cls, check_names=False)
    pd.testing.assert_series_equal(
        labeled["beyond_costs_30m"], expected_bc, check_names=False
    )
    assert labeled["timestamp"].dt.tz is not None

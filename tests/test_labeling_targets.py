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


def test_make_targets_adds_triple_barrier_labels():
    ts = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    close = np.array(
        [
            100.0,
            102.0,
            101.0,
            101.4,
            101.3,
            101.2,
            101.1,
            101.15,
            101.2,
            101.05,
            101.0,
            100.5,
        ],
        dtype=np.float32,
    )
    high = np.array(
        [
            100.2,
            101.6,
            101.2,
            101.5,
            101.4,
            101.3,
            101.2,
            101.2,
            101.25,
            101.1,
            101.05,
            100.6,
        ],
        dtype=np.float32,
    )
    low = np.array(
        [
            99.8,
            101.0,
            100.5,
            101.0,
            101.1,
            101.0,
            101.0,
            101.05,
            101.1,
            101.0,
            100.95,
            100.4,
        ],
        dtype=np.float32,
    )

    df = pd.DataFrame({"timestamp": ts, "close": close, "high": high, "low": low})

    labeled = make_targets(df, horizons_min=[120])

    expected = pd.Series(
        [1, -1, 0, -1, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        dtype="Int8",
    )
    pd.testing.assert_series_equal(
        labeled["triple_barrier_120m"], expected, check_names=False
    )
    assert "triple_barrier_240m" in labeled.columns
    assert labeled["triple_barrier_240m"].isna().all()
    assert "triple_barrier_360m" in labeled.columns
    assert labeled["triple_barrier_360m"].isna().all()

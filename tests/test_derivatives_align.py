from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_analyzer.features.derivatives import make_deriv_features


def test_derivative_features_align_left_label():
    ts = pd.date_range("2024-01-01", periods=5, freq="5T", tz="UTC")
    funding = pd.Series([0.01, 0.012, 0.014, 0.016, 0.018], dtype=float)
    basis = pd.Series([0.001, 0.0015, 0.002, 0.0025, 0.003], dtype=float)
    open_interest = pd.Series([100, 105, 110, 100, 95], dtype=float)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "funding_rate": funding,
            "perp_basis": basis,
            "open_interest": open_interest,
        }
    )
    features = make_deriv_features(df, freq="5T")

    assert np.isclose(features.loc[1, "basis_bp"], basis.iloc[1] * 10_000.0)
    expected_change = (open_interest.iloc[2] - open_interest.iloc[1]) / open_interest.iloc[1]
    assert np.isclose(features.loc[2, "oi_change_rate"], expected_change)
    assert np.isnan(features.loc[0, "oi_change_rate"])

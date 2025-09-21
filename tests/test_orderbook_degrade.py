from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_analyzer.features.orderbook import (
    depth_imbalance,
    order_flow_imbalance,
    spread,
)


def test_missing_orderbook_returns_nan():
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01", tz="UTC")]})
    imbalance = depth_imbalance(frame)
    spread_series = spread(frame)
    ofi = order_flow_imbalance(frame)
    assert np.isnan(imbalance.iloc[0])
    assert np.isnan(spread_series.iloc[0])
    assert np.isnan(ofi.iloc[0])

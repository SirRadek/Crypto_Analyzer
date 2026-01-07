from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_analyzer.features.orderbook import (
    depth_imbalance,
    order_flow_imbalance,
    spread,
)


def test_depth_imbalance_basic():
    orderbook = pd.DataFrame(
        {
            "bid_size1": [5.0, np.nan],
            "bid_size2": [5.0, 5.0],
            "ask_size1": [3.0, 2.0],
            "ask_size2": [1.0, np.nan],
        },
        index=pd.Index([0, 1], name="row"),
    )

    result = depth_imbalance(orderbook)

    expected = pd.Series(
        [(10.0 - 4.0) / 14.0, (5.0 - 2.0) / 7.0],
        index=orderbook.index,
        dtype=float,
    )

    pd.testing.assert_series_equal(result, expected)


def test_spread_basic():
    orderbook = pd.DataFrame(
        {
            "bid_price1": [99.0, 98.5],
            "bid_price2": [98.5, 98.0],
            "ask_price1": [101.0, 100.0],
            "ask_price2": [102.0, 101.0],
        }
    )

    result = spread(orderbook)
    expected = pd.Series([200.0, ((100.0 - 98.5) / 99.25) * 10_000], dtype=float)

    pd.testing.assert_series_equal(result, expected)


def test_order_flow_imbalance_groups_by_timestamp():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:01:00Z",
                    "2024-01-01T00:01:00Z",
                ],
                utc=True,
            ),
            "side": ["buy", "sell", "buy", "sell"],
            "size": [5, 3, 2, 2],
        }
    )

    result = order_flow_imbalance(events)

    expected = pd.Series(
        [
            (5 - 3) / (5 + 3),
            (2 - 2) / (2 + 2),
        ],
        index=pd.Index(
            pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:01:00Z",
                ],
                utc=True,
            )
        ),
        dtype=float,
    )

    pd.testing.assert_series_equal(result, expected)


def test_missing_data_returns_nan():
    orderbook = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01", tz="UTC")]})
    events = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01", tz="UTC")]})

    depth = depth_imbalance(orderbook)
    book_spread = spread(orderbook)
    ofi = order_flow_imbalance(events)

    assert np.isnan(depth.iloc[0])
    assert np.isnan(book_spread.iloc[0])
    assert np.isnan(ofi.iloc[0])

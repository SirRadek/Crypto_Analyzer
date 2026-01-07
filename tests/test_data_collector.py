from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from crypto_analyzer.data import data_collector


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):  # pragma: no cover - behaviour mirrors requests
        return None


class _DummySession:
    def __init__(self, payload):
        self._payload = payload
        self.calls: list[dict[str, object]] = []

    def get(self, url, params=None, timeout=None):  # noqa: D401 - signature matches requests
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _DummyResponse(self._payload)


class _SequentialSession:
    def __init__(self, payloads: list[dict[str, object]]):
        self._payloads = payloads
        self.calls: list[dict[str, object]] = []

    def get(self, url, params=None, timeout=None):  # noqa: D401 - signature matches requests
        index = len(self.calls)
        payload = self._payloads[min(index, len(self._payloads) - 1)]
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _DummyResponse(payload)


def test_fetch_glassnode_active_addresses_parses_payload():
    payload = [
        {"t": 1_609_459_200, "v": 123},
        {"t": 1_609_545_600, "v": 150},
    ]
    session = _DummySession(payload)
    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)

    frame = data_collector.fetch_glassnode_active_addresses(
        start,
        end,
        api_key="dummy",
        session=session,
    )

    assert list(frame.columns) == ["timestamp", "onch_active_addresses"]
    assert len(frame) == 2
    assert frame["onch_active_addresses"].iloc[0] == 123
    assert frame["timestamp"].iloc[0].tzinfo is not None


def test_fetch_binance_order_book_extracts_top_of_book():
    payload = {
        "lastUpdateId": 1,
        "bids": [["50000.0", "1.5"], ["49990.0", "0.25"]],
        "asks": [["50010.0", "2.0"], ["50020.0", "1.0"]],
    }
    session = _DummySession(payload)

    frame = data_collector.fetch_binance_order_book("BTCUSDT", depth=20, session=session)

    assert list(frame.columns) == [
        "timestamp",
        "bid_price",
        "bid_volume",
        "ask_price",
        "ask_volume",
        "spread",
        "mid_price",
        "bid_volume_total",
        "ask_volume_total",
        "bid_notional_total",
        "ask_notional_total",
        "depth_imbalance",
    ]
    assert frame.shape[0] == 1

    row = frame.iloc[0]
    assert row["bid_price"] == pytest.approx(50_000.0)
    assert row["ask_price"] == pytest.approx(50_010.0)
    assert row["spread"] == pytest.approx(10.0)
    assert row["bid_volume_total"] == pytest.approx(1.75)
    assert row["ask_volume_total"] == pytest.approx(3.0)
    assert -1.0 <= row["depth_imbalance"] <= 1.0
    assert session.calls[0]["params"]["limit"] == 20


def test_fetch_binance_order_book_handles_missing_levels():
    payload = {"bids": [], "asks": []}
    session = _DummySession(payload)

    frame = data_collector.fetch_binance_order_book("BTCUSDT", session=session)

    assert frame.empty
    assert list(frame.columns) == [
        "timestamp",
        "bid_price",
        "bid_volume",
        "ask_price",
        "ask_volume",
        "spread",
        "mid_price",
        "bid_volume_total",
        "ask_volume_total",
        "bid_notional_total",
        "ask_notional_total",
        "depth_imbalance",
    ]


def test_fetch_binance_order_book_rejects_invalid_depth():
    session = _DummySession({"bids": [], "asks": []})
    with pytest.raises(ValueError):
        data_collector.fetch_binance_order_book("BTCUSDT", depth=0, session=session)


def test_fetch_binance_basis_computes_basis_points():
    futures_payload = {"markPrice": "50000", "time": 1_609_459_200_000}
    spot_payload = {"price": "49000"}
    session = _SequentialSession([futures_payload, spot_payload])

    frame = data_collector.fetch_binance_basis("BTCUSDT", session=session)

    assert list(frame.columns) == [
        "timestamp",
        "basis",
        "basis_bp",
        "futures_price",
        "spot_price",
    ]
    assert frame.shape[0] == 1

    row = frame.iloc[0]
    expected = (50_000 - 49_000) / 49_000 * 10_000
    assert row["basis"] == pytest.approx(expected)
    assert row["basis_bp"] == pytest.approx(expected)
    assert row["futures_price"] == pytest.approx(50_000.0)
    assert row["spot_price"] == pytest.approx(49_000.0)
    assert row["timestamp"].tzinfo is not None
    assert len(session.calls) == 2


def test_load_enriched_market_data_aligns_daily_series(monkeypatch):
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01T00:00:00Z",
                    "2021-01-01T12:00:00Z",
                    "2021-01-02T00:00:00Z",
                    "2021-01-02T12:00:00Z",
                ]
            ),
            "open": [1, 2, 3, 4],
            "high": [2, 3, 4, 5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10, 20, 30, 40],
        }
    )

    glassnode = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2021-01-01", "2021-01-02"], utc=True),
            "onch_active_addresses": [1000, 1100],
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01T08:00:00Z",
                    "2021-01-01T16:00:00Z",
                    "2021-01-02T00:00:00Z",
                ]
            ),
            "funding_rate": [0.01, 0.02, 0.03],
        }
    )
    open_interest = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2021-01-01T08:00:00Z", "2021-01-02T08:00:00Z"]),
            "open_interest": [1_000_000, 1_200_000],
        }
    )

    monkeypatch.setattr(data_collector, "get_price_data", lambda *args, **kwargs: base.copy())
    monkeypatch.setattr(
        data_collector,
        "fetch_glassnode_active_addresses",
        lambda *args, **kwargs: glassnode,
    )
    monkeypatch.setattr(
        data_collector,
        "fetch_binance_funding_rates",
        lambda *args, **kwargs: funding,
    )
    monkeypatch.setattr(
        data_collector,
        "fetch_binance_open_interest",
        lambda *args, **kwargs: open_interest,
    )

    config = type("Cfg", (), {"symbol": "BTCUSDT", "db_path": "dummy"})()
    enriched = data_collector.load_enriched_market_data(symbol="BTCUSDT", config=config)

    assert {"onch_active_addresses", "funding_rate", "open_interest"}.issubset(enriched.columns)

    first_day = enriched[
        enriched["timestamp"].dt.floor("D") == pd.Timestamp("2021-01-01", tz="UTC")
    ]
    second_day = enriched[
        enriched["timestamp"].dt.floor("D") == pd.Timestamp("2021-01-02", tz="UTC")
    ]

    assert (first_day["onch_active_addresses"] == 1000).all()
    assert (second_day["onch_active_addresses"] == 1100).all()
    assert pytest.approx(first_day["funding_rate"].iloc[0], rel=1e-6) == 0.015
    assert (second_day["funding_rate"] == 0.03).all()
    assert (first_day["open_interest"] == 1_000_000).all()
    assert (second_day["open_interest"] == 1_200_000).all()

from __future__ import annotations

from datetime import datetime

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


def test_load_enriched_market_data_aligns_daily_series(monkeypatch):
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03"], utc=True
            ),
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10, 20, 30],
        }
    )

    glassnode = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03"], utc=True
            ),
            "onch_active_addresses": [1000, 1100, 1200],
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03"], utc=True
            ),
            "funding_rate": [0.01, 0.015, 0.02],
        }
    )
    open_interest = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03"], utc=True
            ),
            "open_interest": [1_000_000, 1_050_000, 1_200_000],
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

    assert enriched.shape[0] == base.shape[0]

    assert enriched["onch_active_addresses"].tolist() == [1000, 1100, 1200]
    assert pytest.approx(enriched.loc[0, "funding_rate"], rel=1e-6) == 0.01
    assert pytest.approx(enriched.loc[1, "funding_rate"], rel=1e-6) == 0.015
    assert pytest.approx(enriched.loc[2, "funding_rate"], rel=1e-6) == 0.02
    assert enriched.loc[0, "open_interest"] == 1_000_000
    assert enriched.loc[1, "open_interest"] == 1_050_000
    assert enriched.loc[2, "open_interest"] == 1_200_000

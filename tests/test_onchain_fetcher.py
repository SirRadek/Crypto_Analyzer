"""Tests for on-chain fetcher utilities."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
import requests

from crypto_analyzer.data.onchain_fetcher import (
    COINMETRICS_ASSET_METRICS_ENDPOINT,
    GLASSNODE_EXCHANGE_INFLOW_ENDPOINT,
    GLASSNODE_EXCHANGE_OUTFLOW_ENDPOINT,
    fetch_exchange_flows,
    fetch_coinmetrics_exchange_flows,
    fetch_mempool_stats,
    fetch_whale_alert_transactions,
    WHALE_ALERT_TRANSACTIONS_ENDPOINT,
)


@pytest.fixture(autouse=True)
def freeze_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure :func:`pd.Timestamp.utcnow` returns a naive timestamp for tests."""

    def _fake_utcnow(_: type[pd.Timestamp]) -> pd.Timestamp:
        return pd.Timestamp("2024-01-01 00:00:00")

    monkeypatch.setattr(
        pd.Timestamp,
        "utcnow",
        classmethod(_fake_utcnow),
    )


class DummyResponse:
    """Simple stand-in for ``requests.Response``."""

    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - exercised in tests
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class QueueSession:
    """Session that returns predetermined responses in order."""

    def __init__(self, responses: Iterator[DummyResponse] | list[DummyResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> DummyResponse:
        self.calls.append((url, kwargs))
        if not self._responses:
            raise RuntimeError("No more responses queued")
        return self._responses.pop(0)


def _reset_whale_limiter() -> None:
    from crypto_analyzer.data import onchain_fetcher as module

    module._WHALE_LAST_CALL = None


def test_fetch_mempool_stats_returns_expected_frame() -> None:
    stats_payload = {"count": "10", "vsize": "200", "total_fee": "300"}
    fees_payload = {
        "fastestFee": "25",
        "halfHourFee": "20",
        "hourFee": "15",
        "economyFee": "12",
        "minimumFee": "5",
    }
    session = QueueSession([DummyResponse(stats_payload), DummyResponse(fees_payload)])

    frame = fetch_mempool_stats(session=session)

    expected_columns = {
        "onch_mempool_count",
        "onch_mempool_vsize",
        "onch_mempool_total_fee",
        "onch_mempool_fee_fastest",
        "onch_mempool_fee_half_hour",
        "onch_mempool_fee_hour",
        "onch_mempool_fee_economy",
        "onch_mempool_fee_minimum",
    }
    assert set(frame.columns) == expected_columns
    assert frame.shape[0] == 1

    row = frame.iloc[0]
    assert row["onch_mempool_count"] == pytest.approx(10.0)
    assert row["onch_mempool_vsize"] == pytest.approx(200.0)
    assert row["onch_mempool_total_fee"] == pytest.approx(300.0)
    assert row["onch_mempool_fee_fastest"] == pytest.approx(25.0)
    assert row["onch_mempool_fee_half_hour"] == pytest.approx(20.0)
    assert row["onch_mempool_fee_hour"] == pytest.approx(15.0)
    assert row["onch_mempool_fee_economy"] == pytest.approx(12.0)
    assert row["onch_mempool_fee_minimum"] == pytest.approx(5.0)
    assert frame.index.name == "timestamp"
    assert frame.index.tz is not None


def test_fetch_mempool_stats_handles_request_errors() -> None:
    class FailingSession:
        def get(self, *_: Any, **__: Any) -> Any:
            raise requests.RequestException("boom")

    frame = fetch_mempool_stats(session=FailingSession())

    assert frame.empty
    assert frame.index.name == "timestamp"
    assert frame.index.tz is not None
    assert list(frame.columns) == [
        "onch_mempool_count",
        "onch_mempool_vsize",
        "onch_mempool_total_fee",
        "onch_mempool_fee_fastest",
        "onch_mempool_fee_half_hour",
        "onch_mempool_fee_hour",
        "onch_mempool_fee_economy",
        "onch_mempool_fee_minimum",
    ]


def test_fetch_exchange_flows_returns_expected_series() -> None:
    inflow_payload = [
        {"t": 1_700_000_000, "v": 1.5},
        {"t": 1_700_086_400, "v": 2.5},
    ]
    outflow_payload = [
        {"t": 1_700_000_000, "v": 0.5},
        {"t": 1_700_086_400, "v": 0.75},
    ]

    responses = {
        GLASSNODE_EXCHANGE_INFLOW_ENDPOINT: DummyResponse(inflow_payload),
        GLASSNODE_EXCHANGE_OUTFLOW_ENDPOINT: DummyResponse(outflow_payload),
    }

    class MappingSession:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, url: str, **kwargs: Any) -> DummyResponse:
            self.calls.append(url)
            return responses[url]

    session = MappingSession()

    frame = fetch_exchange_flows("key", session=session)

    assert list(frame.columns) == ["onch_exchange_inflow", "onch_exchange_outflow"]
    assert frame.index.name == "timestamp"
    assert frame.index.tz is not None
    assert frame.loc[
        pd.Timestamp(1_700_000_000, unit="s", tz="UTC"), "onch_exchange_inflow"
    ] == pytest.approx(1.5)
    assert frame.loc[
        pd.Timestamp(1_700_086_400, unit="s", tz="UTC"), "onch_exchange_outflow"
    ] == pytest.approx(0.75)


def test_fetch_exchange_flows_handles_request_errors() -> None:
    class FailingSession:
        def get(self, *_: Any, **__: Any) -> Any:
            raise requests.RequestException("nope")

    frame = fetch_exchange_flows("key", session=FailingSession())

    assert frame.empty
    assert frame.index.name == "timestamp"
    assert frame.index.tz is not None
    assert list(frame.columns) == ["onch_exchange_inflow", "onch_exchange_outflow"]


def test_fetch_coinmetrics_exchange_flows_paginates_and_parses() -> None:
    page_one = {
        "data": [
            {
                "time": "2024-01-01T00:00:00Z",
                "ExchgNetFlow": "10.5",
                "ExchgInflowVolume": "20.0",
                "ExchgOutflowVolume": "9.5",
            }
        ],
        "next_page_token": "token-1",
    }
    page_two = {
        "data": [
            {
                "time": "2024-01-02T00:00:00Z",
                "ExchgNetFlow": "-3.0",
                "ExchgInflowVolume": "12.0",
                "ExchgOutflowVolume": "15.0",
            }
        ]
    }
    session = QueueSession([DummyResponse(page_one), DummyResponse(page_two)])

    frame = fetch_coinmetrics_exchange_flows(session=session)

    assert list(frame.columns) == [
        "onch_exchange_net_flow",
        "onch_exchange_inflow",
        "onch_exchange_outflow",
    ]
    assert frame.index.name == "timestamp"
    assert frame.index.tz is not None
    assert frame.iloc[0, 0] == pytest.approx(10.5)
    assert frame.iloc[1, 1] == pytest.approx(12.0)
    assert frame.iloc[1, 2] == pytest.approx(15.0)

    first_call = session.calls[0]
    assert first_call[0] == COINMETRICS_ASSET_METRICS_ENDPOINT
    assert first_call[1]["params"]["frequency"] == "1d"
    second_call = session.calls[1]
    assert second_call[1]["params"]["page_token"] == "token-1"


def test_fetch_coinmetrics_exchange_flows_handles_errors() -> None:
    class FailingSession:
        def get(self, *_: Any, **__: Any) -> Any:
            raise requests.RequestException("bad")

    frame = fetch_coinmetrics_exchange_flows(session=FailingSession())

    assert frame.empty
    assert frame.index.tz is not None


def test_fetch_whale_alert_transactions_parses_payload() -> None:
    _reset_whale_limiter()

    payload = {
        "transactions": [
            {
                "timestamp": 1_700_000_000,
                "hash": "abc123",
                "symbol": "usdt",
                "blockchain": "tron",
                "amount": "123.45",
                "amount_usd": "6789.01",
                "from": {"address": "addr1", "owner": "Exchange A"},
                "to": {"address": "addr2", "owner": "Wallet B"},
            }
        ]
    }
    session = QueueSession([DummyResponse(payload)])

    frame = fetch_whale_alert_transactions(
        api_key="secret",
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T06:00:00Z",
        currency="USDT",
        session=session,
        _sleep=lambda *_: None,
    )

    assert not frame.empty
    assert frame.index.name == "timestamp"
    assert frame.index.tz is not None
    row = frame.iloc[0]
    assert row["amount_usd"] == pytest.approx(6789.01)
    assert row["currency"] == "USDT"
    assert row["from_owner"] == "Exchange A"
    assert row["to_owner"] == "Wallet B"

    assert session.calls[0][0] == WHALE_ALERT_TRANSACTIONS_ENDPOINT
    params = session.calls[0][1]["params"]
    assert params["currency"] == "usdt"
    assert params["api_key"] == "secret"


def test_fetch_whale_alert_transactions_handles_empty() -> None:
    _reset_whale_limiter()
    session = QueueSession([DummyResponse({"transactions": []})])

    frame = fetch_whale_alert_transactions(
        api_key="secret",
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T01:00:00Z",
        session=session,
        _sleep=lambda *_: None,
    )

    assert frame.empty
    assert frame.index.name == "timestamp"
    assert frame.index.tz is not None
    assert set(frame.columns) == {
        "transaction_hash",
        "blockchain",
        "currency",
        "amount",
        "amount_usd",
        "from_address",
        "from_owner",
        "to_address",
        "to_owner",
    }


def test_fetch_whale_alert_transactions_requires_api_key() -> None:
    _reset_whale_limiter()
    with pytest.raises(ValueError, match="Whale Alert API key required"):
        fetch_whale_alert_transactions(
            api_key=" ",
            start="2024-01-01T00:00:00Z",
            end="2024-01-01T02:00:00Z",
            session=QueueSession([]),
        )


def test_fetch_whale_alert_transactions_validates_range() -> None:
    _reset_whale_limiter()
    with pytest.raises(ValueError, match="start must be earlier than end"):
        fetch_whale_alert_transactions(
            api_key="secret",
            start="2024-01-02T00:00:00Z",
            end="2024-01-01T00:00:00Z",
            session=QueueSession([]),
        )


def test_fetch_whale_alert_transactions_raises_on_invalid_key() -> None:
    _reset_whale_limiter()
    session = QueueSession([DummyResponse({"error": "Invalid API key"}, status_code=401)])

    with pytest.raises(ValueError, match="Whale Alert API key required"):
        fetch_whale_alert_transactions(
            api_key="bad",
            start="2024-01-01T00:00:00Z",
            end="2024-01-01T01:00:00Z",
            session=session,
            _sleep=lambda *_: None,
        )

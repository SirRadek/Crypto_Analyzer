"""Tests for on-chain fetcher utilities."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
import requests

from crypto_analyzer.data.onchain_fetcher import (
    GLASSNODE_EXCHANGE_INFLOW_ENDPOINT,
    GLASSNODE_EXCHANGE_OUTFLOW_ENDPOINT,
    fetch_exchange_flows,
    fetch_mempool_stats,
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

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to do on success
        return None


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
    assert frame.loc[pd.Timestamp(1_700_000_000, unit="s", tz="UTC"), "onch_exchange_inflow"] == pytest.approx(
        1.5
    )
    assert frame.loc[pd.Timestamp(1_700_086_400, unit="s", tz="UTC"), "onch_exchange_outflow"] == pytest.approx(
        0.75
    )


def test_fetch_exchange_flows_handles_request_errors() -> None:
    class FailingSession:
        def get(self, *_: Any, **__: Any) -> Any:
            raise requests.RequestException("nope")

    frame = fetch_exchange_flows("key", session=FailingSession())

    assert frame.empty
    assert frame.index.name == "timestamp"
    assert frame.index.tz is not None
    assert list(frame.columns) == ["onch_exchange_inflow", "onch_exchange_outflow"]

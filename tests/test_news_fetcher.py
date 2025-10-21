from __future__ import annotations

from typing import Any

import pytest
import requests

from crypto_analyzer.data.news_fetcher import fetch_cryptopanic_news


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - mirrors requests
        return None


class DummySession:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> DummyResponse:
        if not self._payloads:
            raise RuntimeError("No more responses queued")
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return DummyResponse(self._payloads.pop(0))


def test_fetch_cryptopanic_news_parses_records() -> None:
    payload = {
        "results": [
            {
                "published_at": "2024-01-01T00:00:00Z",
                "title": "Bitcoin hits new high",
                "url": "https://example.com/1",
                "votes": {"positive": 3, "negative": 1},
                "tags": ["Bullish", "BTC"],
                "currencies": [{"code": "BTC"}, {"code": "ETH"}],
                "source": {"title": "CoinDesk"},
            },
            {
                "published_at": "2024-01-01T12:00:00Z",
                "title": "Market uncertainty",
                "url": "https://example.com/2",
                "votes": {"positive": 0, "negative": 2},
                "tags": ["Bearish"],
                "currencies": [{"code": "ETH"}],
                "source": {"title": "CoinTelegraph"},
            },
        ]
    }
    session = DummySession([payload])

    frame = fetch_cryptopanic_news("token", session=session, limit=5)

    assert list(frame.columns) == [
        "timestamp",
        "title",
        "url",
        "source",
        "sentiment",
        "positive_votes",
        "negative_votes",
        "tags",
        "currencies",
    ]
    assert frame.shape[0] == 2
    assert frame.iloc[0]["source"] == "CoinDesk"
    assert frame.iloc[0]["sentiment"] == pytest.approx(3 - 1)
    assert "bullish" in frame.iloc[0]["tags"].lower()
    assert "BTC" in frame.iloc[0]["currencies"]


def test_fetch_cryptopanic_news_handles_pagination_limit() -> None:
    payload_one = {
        "results": [
            {
                "published_at": "2024-01-01T00:00:00Z",
                "title": "First",
                "url": "https://example.com/a",
                "votes": {"positive": 1, "negative": 0},
                "tags": [],
                "currencies": [],
                "source": {"title": "SourceA"},
            }
        ],
        "next": "https://cryptopanic.com/api/v1/posts/?page=2",
    }
    payload_two = {
        "results": [
            {
                "published_at": "2024-01-02T00:00:00Z",
                "title": "Second",
                "url": "https://example.com/b",
                "votes": {"positive": 0, "negative": 1},
                "tags": ["Bearish"],
                "currencies": [],
                "source": {"title": "SourceB"},
            }
        ]
    }
    session = DummySession([payload_one, payload_two])

    frame = fetch_cryptopanic_news("token", session=session, limit=2, max_pages=2)

    assert frame.shape[0] == 2
    assert session.calls[0]["params"]["auth_token"] == "token"
    assert session.calls[1]["params"] is None


def test_fetch_cryptopanic_news_validates_arguments() -> None:
    session = DummySession([])
    with pytest.raises(ValueError):
        fetch_cryptopanic_news("", session=session)
    with pytest.raises(ValueError):
        fetch_cryptopanic_news("token", session=session, limit=0)
    with pytest.raises(ValueError):
        fetch_cryptopanic_news("token", session=session, limit=1, max_pages=0)


def test_fetch_cryptopanic_news_handles_errors() -> None:
    class FailingSession:
        def get(self, *_: Any, **__: Any) -> Any:
            raise requests.RequestException("fail")

    frame = fetch_cryptopanic_news("token", session=FailingSession())

    assert frame.empty
    assert list(frame.columns) == [
        "timestamp",
        "title",
        "url",
        "source",
        "sentiment",
        "positive_votes",
        "negative_votes",
        "tags",
        "currencies",
    ]

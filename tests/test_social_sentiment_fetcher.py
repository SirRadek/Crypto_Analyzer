from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import requests

from crypto_analyzer.data.social_sentiment import fetch_reddit_sentiment


class DummyAnalyzer:
    def __init__(self, mapping: dict[str, float]) -> None:
        self._mapping = mapping

    def polarity_scores(self, text: str) -> dict[str, float]:  # pragma: no cover - simple mapping
        return {"compound": self._mapping.get(text, 0.0)}


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - mirrors requests
        return None


class DummySession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> DummyResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return DummyResponse(self._payload)


def test_fetch_reddit_sentiment_computes_expected_metrics() -> None:
    payload = {
        "data": [
            {"body": "I love bitcoin"},
            {"body": "This market looks awful"},
            {"body": "[deleted]"},
        ]
    }
    analyzer = DummyAnalyzer({"I love bitcoin": 0.8, "This market looks awful": -0.6})
    session = DummySession(payload)

    frame = fetch_reddit_sentiment(
        subreddit="CryptoCurrency",
        analyzer=analyzer,
        session=session,
    )

    assert frame.shape == (1, 8)
    row = frame.iloc[0]
    assert row["reddit_score"] == pytest.approx(0.1)
    assert row["mentions"] == 2
    assert row["reddit_positive_count"] == 1
    assert row["reddit_negative_count"] == 1
    assert session.calls[0]["params"]["subreddit"] == "CryptoCurrency"


def test_fetch_reddit_sentiment_normalises_time_parameters() -> None:
    payload = {"data": [{"body": "neutral"}]}
    analyzer = DummyAnalyzer({"neutral": 0.0})
    session = DummySession(payload)
    after = datetime(2024, 1, 1, tzinfo=timezone.utc)

    frame = fetch_reddit_sentiment(
        after=after,
        before=after + timedelta(hours=1),
        analyzer=analyzer,
        session=session,
        size=123,
    )

    assert not frame.empty
    params = session.calls[0]["params"]
    assert params["after"] == int(after.timestamp())
    assert params["before"] == int((after + timedelta(hours=1)).timestamp())
    assert params["size"] == 123


def test_fetch_reddit_sentiment_handles_request_errors() -> None:
    class FailingSession:
        def get(self, *_: Any, **__: Any) -> Any:
            raise requests.RequestException("boom")

    analyzer = DummyAnalyzer({})
    frame = fetch_reddit_sentiment(
        subreddit="Bitcoin",
        analyzer=analyzer,
        session=FailingSession(),
    )

    assert frame.empty
    assert list(frame.columns) == [
        "timestamp",
        "reddit_score",
        "mentions",
        "reddit_positive_ratio",
        "reddit_negative_ratio",
        "reddit_positive_count",
        "reddit_negative_count",
        "reddit_neutral_count",
    ]

from __future__ import annotations

from datetime import datetime, timedelta, UTC
from typing import Any

import pytest
import requests
import tweepy

from crypto_analyzer.data.social_sentiment import (
    fetch_reddit_sentiment,
    fetch_twitter_sentiment,
)


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

    def get(
        self, url: str, params: dict[str, Any] | None = None, timeout: int | None = None
    ) -> DummyResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return DummyResponse(self._payload)


class DummyTweet:
    def __init__(self, text: str, created_at: datetime) -> None:
        self.text = text
        self.created_at = created_at


class DummyTwitterResponse:
    def __init__(self, tweets: list[DummyTweet], *, next_token: str | None = None) -> None:
        self.data = tweets
        self.meta = {"next_token": next_token} if next_token else {}


class DummyTwitterClient:
    def __init__(self, responses: list[DummyTwitterResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def search_recent_tweets(self, **kwargs: Any) -> DummyTwitterResponse:
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return DummyTwitterResponse([])


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


def test_fetch_twitter_sentiment_aggregates_expected_values() -> None:
    timestamp = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)
    tweets = [
        DummyTwitterResponse(
            [
                DummyTweet("Bullish on BTC", timestamp + timedelta(minutes=5)),
                DummyTweet("BTC looks weak", timestamp + timedelta(minutes=20)),
                DummyTweet("BTC recovering", timestamp + timedelta(hours=1, minutes=5)),
            ]
        )
    ]
    client = DummyTwitterClient(tweets)
    analyzer = DummyAnalyzer(
        {
            "Bullish on BTC": 0.6,
            "BTC looks weak": -0.4,
            "BTC recovering": 0.5,
        }
    )

    frame = fetch_twitter_sentiment(
        ["Bitcoin", "BTC"],
        start=timestamp,
        end=timestamp + timedelta(hours=2),
        client=client,
        analyzer=analyzer,
        interval="1H",
        max_tweets=10,
    )

    assert frame.shape == (2, 5)
    first_row = frame.iloc[0]
    assert first_row["twitter_score"] == pytest.approx(0.1)
    assert first_row["positive_count"] == 1
    assert first_row["negative_count"] == 1
    assert first_row["mentions"] == 2
    second_row = frame.iloc[1]
    assert second_row["positive_count"] == 1
    assert second_row["negative_count"] == 0
    assert second_row["mentions"] == 1

    call = client.calls[0]
    assert call["query"] == "(Bitcoin OR BTC) -is:retweet lang:en"
    assert call["start_time"].endswith("Z")
    assert call["end_time"].endswith("Z")
    assert call["max_results"] == 100


def test_fetch_twitter_sentiment_requires_token_when_client_missing() -> None:
    frame = fetch_twitter_sentiment(["BTC"], bearer_token=None)
    assert frame.empty


def test_fetch_twitter_sentiment_handles_rate_limit() -> None:
    class RateLimitedClient:
        def __init__(self) -> None:
            self.calls = 0

        def search_recent_tweets(self, **_: Any) -> Any:
            self.calls += 1
            response = requests.Response()
            response.status_code = 429
            response.reason = "Too Many Requests"
            response._content = b'{"errors": [{"message": "rate limit"}]}'
            raise tweepy.errors.TooManyRequests(response)

    client = RateLimitedClient()
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    frame = fetch_twitter_sentiment(
        ["BTC"],
        client=client,
        analyzer=DummyAnalyzer({}),
        sleep=fake_sleep,
        max_tweets=5,
    )

    assert frame.empty
    assert sleep_calls == [900.0]
    assert client.calls == 1


def test_fetch_reddit_sentiment_normalises_time_parameters() -> None:
    payload = {"data": [{"body": "neutral"}]}
    analyzer = DummyAnalyzer({"neutral": 0.0})
    session = DummySession(payload)
    after = datetime(2024, 1, 1, tzinfo=UTC)

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

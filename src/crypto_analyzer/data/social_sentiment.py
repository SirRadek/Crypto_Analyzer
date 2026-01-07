"""Fetch Reddit sentiment aggregates from Pushshift and VADER."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from crypto_analyzer.utils.logging import get_logger

try:  # pragma: no cover - optional dependency during some tests
    import tweepy
except (
    ModuleNotFoundError
):  # pragma: no cover - allow fetch_twitter_sentiment to degrade gracefully
    tweepy = None  # type: ignore[assignment]

LOGGER = get_logger(__name__)

_PUSHSHIFT_BASE_URL = "https://api.pushshift.io/reddit"
_VALID_CONTENT_TYPES = {"comment", "submission"}


def _empty_twitter_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "twitter_score",
            "positive_count",
            "negative_count",
            "mentions",
        ]
    )


def _normalise_datetime(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _format_twitter_time(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _build_twitter_query(keywords: Sequence[str]) -> str:
    tokens: list[str] = []
    for keyword in keywords:
        term = str(keyword or "").strip()
        if not term:
            continue
        if " " in term:
            term = f'"{term}"'
        tokens.append(term)
    if not tokens:
        raise ValueError("At least one non-empty keyword must be provided")
    joined = " OR ".join(tokens)
    return f"({joined}) -is:retweet lang:en"


def _isinstance_of_tweepy_error(exc: Exception, name: str) -> bool:
    if tweepy is None:
        return False
    modules: list[Any] = [tweepy]
    errors_module = getattr(tweepy, "errors", None)
    if errors_module is not None:
        modules.append(errors_module)
    for module in modules:
        cls = getattr(module, name, None)
        if isinstance(cls, type) and issubclass(cls, Exception) and isinstance(exc, cls):
            return True
    return False


def _extract_next_token(meta: Any) -> str | None:
    if meta is None:
        return None
    token: Any
    if isinstance(meta, dict):
        token = meta.get("next_token")
    else:
        getter = getattr(meta, "get", None)
        if callable(getter):
            token = getter("next_token")
        else:
            token = getattr(meta, "next_token", None)
    if not token:
        return None
    token_str = str(token).strip()
    return token_str or None


def fetch_twitter_sentiment(
    keywords: Sequence[str],
    *,
    start: datetime | pd.Timestamp | None = None,
    end: datetime | pd.Timestamp | None = None,
    bearer_token: str | None = None,
    max_tweets: int = 300,
    max_results: int = 100,
    interval: str = "1H",
    client: Any | None = None,
    analyzer: SentimentIntensityAnalyzer | None = None,
    backoff_seconds: int = 900,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Fetch recent tweets for *keywords* and aggregate sentiment scores by interval."""

    query = _build_twitter_query(keywords)

    api_client = client
    if api_client is None:
        if tweepy is None:
            LOGGER.warning("tweepy is not installed; skipping Twitter sentiment fetch")
            return _empty_twitter_frame()
        if not bearer_token:
            LOGGER.warning("Twitter bearer token missing; skipping Twitter sentiment fetch")
            return _empty_twitter_frame()
        try:
            api_client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)  # type: ignore[attr-defined]
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive, tweepy may raise configuration errors
            LOGGER.warning("Failed to initialise Twitter client", exc_info=exc)
            return _empty_twitter_frame()

    start_ts = _normalise_datetime(start) if start is not None else None
    end_ts = _normalise_datetime(end) if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start must be earlier than or equal to end")

    if max_tweets <= 0:
        return _empty_twitter_frame()
    max_results = max(10, min(max_results, 100))

    analyser = analyzer or SentimentIntensityAnalyzer()
    fetched = 0
    next_token: str | None = None
    rows: list[dict[str, Any]] = []

    while fetched < max_tweets:
        params: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "tweet_fields": ["created_at", "lang"],
        }
        if start_ts is not None:
            params["start_time"] = _format_twitter_time(start_ts)
        if end_ts is not None:
            params["end_time"] = _format_twitter_time(end_ts)
        if next_token:
            params["next_token"] = next_token

        try:
            response = api_client.search_recent_tweets(**params)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - tweepy raises its own hierarchy
            if _isinstance_of_tweepy_error(exc, "TooManyRequests"):
                LOGGER.warning(
                    "Twitter rate limit reached; backing off",
                    extra={"backoff_seconds": backoff_seconds},
                    exc_info=exc,
                )
                if backoff_seconds > 0:
                    with contextlib.suppress(Exception):  # pragma: no cover - defensive
                        sleep(float(backoff_seconds))
                return _empty_twitter_frame()
            if _isinstance_of_tweepy_error(exc, "TweepyException"):
                LOGGER.warning("Twitter API error when fetching sentiment", exc_info=exc)
                return _empty_twitter_frame()
            raise

        tweets = getattr(response, "data", None) or []
        if not tweets:
            break

        for tweet in tweets:
            if fetched >= max_tweets:
                break
            text = getattr(tweet, "text", None)
            if text is None:
                continue
            text_value = str(text).strip()
            if not text_value:
                continue
            created_raw = getattr(tweet, "created_at", None)
            if created_raw is None:
                continue
            try:
                created_ts = _normalise_datetime(created_raw)
            except (TypeError, ValueError):  # pragma: no cover - skip unexpected values
                continue
            try:
                scores = analyser.polarity_scores(text_value)
            except (
                Exception
            ) as exc:  # pragma: no cover - ensure a single failure doesn't abort loop
                LOGGER.warning("Failed to compute tweet sentiment", exc_info=exc)
                continue
            compound = float(scores.get("compound", 0.0))
            rows.append(
                {
                    "timestamp": created_ts,
                    "compound": compound,
                    "is_positive": compound > 0.05,
                    "is_negative": compound < -0.05,
                }
            )
            fetched += 1

        next_token = _extract_next_token(getattr(response, "meta", None))
        if not next_token:
            break

    if not rows:
        LOGGER.info(
            "No tweets matched sentiment query",
            extra={"query": query, "tweets_processed": 0},
        )
        return _empty_twitter_frame()

    frame = pd.DataFrame(rows)
    freq = interval.lower() if isinstance(interval, str) else interval
    try:
        frame["bucket"] = frame["timestamp"].dt.floor(freq)
    except ValueError:
        LOGGER.warning("Invalid interval '%s' supplied; defaulting to 1H", interval)
        frame["bucket"] = frame["timestamp"].dt.floor("1H")

    grouped = frame.groupby("bucket", dropna=False)
    summary = grouped.agg(
        twitter_score=("compound", "mean"),
        positive_count=("is_positive", "sum"),
        negative_count=("is_negative", "sum"),
    )
    summary["mentions"] = (summary["positive_count"] + summary["negative_count"]).astype(int)

    result = summary.reset_index().rename(columns={"bucket": "timestamp"})
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result = result.dropna(subset=["timestamp"]).reset_index(drop=True)
    result["twitter_score"] = result["twitter_score"].astype(float)
    result["positive_count"] = result["positive_count"].astype(int)
    result["negative_count"] = result["negative_count"].astype(int)
    result["mentions"] = result["mentions"].astype(int)
    result = result.sort_values("timestamp").reset_index(drop=True)

    positives = int(frame["is_positive"].sum())
    negatives = int(frame["is_negative"].sum())
    neutrals = len(frame) - positives - negatives
    LOGGER.info(
        "Fetched twitter sentiment",
        extra={
            "query": query,
            "tweets_processed": len(frame),
            "positives": positives,
            "negatives": negatives,
            "neutrals": neutrals,
            "buckets": len(result),
        },
    )

    return result


def _normalise_pushshift_time(value: Any) -> Any:
    """Convert ``value`` to a Pushshift-compatible timestamp representation."""

    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
        return int(ts.timestamp())
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())
    return value


def _extract_text(entry: dict[str, Any], *, content_type: str) -> str:
    if content_type == "comment":
        return str(entry.get("body") or "")
    title = str(entry.get("title") or "")
    body = str(entry.get("selftext") or "")
    if title and body:
        return f"{title}\n{body}"
    return title or body


def fetch_reddit_sentiment(
    *,
    subreddit: str | None = None,
    query: str | None = None,
    after: datetime | pd.Timestamp | int | str | None = None,
    before: datetime | pd.Timestamp | int | str | None = None,
    size: int = 100,
    content_type: str = "comment",
    session: requests.Session | None = None,
    analyzer: SentimentIntensityAnalyzer | None = None,
) -> pd.DataFrame:
    """Aggregate Reddit sentiment for the specified query parameters."""

    if content_type not in _VALID_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}")

    params: dict[str, Any] = {"size": max(1, min(size, 500))}
    if subreddit:
        params["subreddit"] = subreddit
    if query:
        params["q"] = query
    after_value = _normalise_pushshift_time(after)
    before_value = _normalise_pushshift_time(before)
    if after_value is not None:
        params["after"] = after_value
    if before_value is not None:
        params["before"] = before_value

    url = f"{_PUSHSHIFT_BASE_URL}/{content_type}/search/"

    sess = session or requests.Session()
    try:
        response = sess.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json() or {}
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("Failed to fetch Reddit sentiment", exc_info=exc)
        return pd.DataFrame(
            columns=[
                "timestamp",
                "reddit_score",
                "mentions",
                "reddit_positive_ratio",
                "reddit_negative_ratio",
                "reddit_positive_count",
                "reddit_negative_count",
                "reddit_neutral_count",
            ]
        )

    entries: Sequence[dict[str, Any]] = payload.get("data", []) if isinstance(payload, dict) else []

    if not entries:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "reddit_score",
                "mentions",
                "reddit_positive_ratio",
                "reddit_negative_ratio",
                "reddit_positive_count",
                "reddit_negative_count",
                "reddit_neutral_count",
            ]
        )

    analyser = analyzer or SentimentIntensityAnalyzer()

    compounds: list[float] = []
    pos = neg = neu = 0
    for entry in entries:
        text = _extract_text(entry, content_type=content_type).strip()
        if not text or text in {"[deleted]", "[removed]"}:
            continue
        scores = analyser.polarity_scores(text)
        compound = float(scores.get("compound", 0.0))
        compounds.append(compound)
        if compound > 0.05:
            pos += 1
        elif compound < -0.05:
            neg += 1
        else:
            neu += 1

    if not compounds:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "reddit_score",
                "mentions",
                "reddit_positive_ratio",
                "reddit_negative_ratio",
                "reddit_positive_count",
                "reddit_negative_count",
                "reddit_neutral_count",
            ]
        )

    mention_count = len(compounds)
    avg_compound = sum(compounds) / mention_count
    pos_ratio = pos / mention_count
    neg_ratio = neg / mention_count

    timestamp = pd.Timestamp.utcnow()
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [timestamp],
            "reddit_score": [avg_compound],
            "mentions": [mention_count],
            "reddit_positive_ratio": [pos_ratio],
            "reddit_negative_ratio": [neg_ratio],
            "reddit_positive_count": [pos],
            "reddit_negative_count": [neg],
            "reddit_neutral_count": [neu],
        }
    )
    return frame


__all__ = ["fetch_reddit_sentiment", "fetch_twitter_sentiment"]

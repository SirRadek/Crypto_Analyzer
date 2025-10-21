"""Fetch Reddit sentiment aggregates from Pushshift and VADER."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from crypto_analyzer.utils.logging import get_logger

LOGGER = get_logger(__name__)

_PUSHSHIFT_BASE_URL = "https://api.pushshift.io/reddit"
_VALID_CONTENT_TYPES = {"comment", "submission"}


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


__all__ = ["fetch_reddit_sentiment"]

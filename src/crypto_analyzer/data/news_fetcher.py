"""Download cryptocurrency news from the CryptoPanic API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import requests

from crypto_analyzer.utils.logging import get_logger

LOGGER = get_logger(__name__)

_CRYPTO_PANIC_ENDPOINT = "https://cryptopanic.com/api/v1/posts/"


def _serialise_list(values: Sequence[str] | None) -> str:
    if not values:
        return ""
    unique = {value for value in values if value}
    return ",".join(sorted(unique))


def fetch_cryptopanic_news(
    auth_token: str,
    *,
    filter: str | None = None,
    currencies: Sequence[str] | None = None,
    kind: str | None = None,
    limit: int = 50,
    max_pages: int = 1,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch latest news posts along with sentiment cues from CryptoPanic."""

    if not auth_token:
        raise ValueError("A CryptoPanic auth token is required")
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    if max_pages <= 0:
        raise ValueError("max_pages must be a positive integer")

    params: dict[str, Any] = {"auth_token": auth_token, "public": "true"}
    if filter:
        params["filter"] = filter
    if kind:
        params["kind"] = kind
    if currencies:
        params["currencies"] = _serialise_list(currencies)
    params["limit"] = min(limit, 100)

    sess = session or requests.Session()
    records: list[dict[str, Any]] = []
    next_url: str | None = _CRYPTO_PANIC_ENDPOINT
    pages_fetched = 0

    try:
        while next_url and pages_fetched < max_pages and len(records) < limit:
            response = sess.get(next_url, params=params if pages_fetched == 0 else None, timeout=10)
            response.raise_for_status()
            payload = response.json() or {}
            results = payload.get("results", [])
            if results:
                records.extend(results)
            next_url = payload.get("next")
            params = None
            pages_fetched += 1
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("Failed to fetch CryptoPanic news", exc_info=exc)
        return pd.DataFrame(
            columns=[
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
        )

    if not records:
        return pd.DataFrame(
            columns=[
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
        )

    rows: list[dict[str, Any]] = []
    for entry in records[:limit]:
        published = entry.get("published_at") or entry.get("created_at")
        timestamp = pd.to_datetime(published, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        votes = entry.get("votes", {})
        positive = votes.get("positive") or entry.get("positive_votes") or 0
        negative = votes.get("negative") or entry.get("negative_votes") or 0
        tags = entry.get("tags")
        if isinstance(tags, list):
            tags_serialised = _serialise_list([str(tag) for tag in tags])
        elif tags:
            tags_serialised = str(tags)
        else:
            tags_serialised = ""
        currencies_payload = entry.get("currencies")
        if isinstance(currencies_payload, list):
            currency_codes = [item.get("code") for item in currencies_payload if item.get("code")]
            currencies_serialised = _serialise_list(currency_codes)
        else:
            currencies_serialised = ""

        source_info = entry.get("source") or {}
        source = (
            source_info.get("title")
            or source_info.get("name")
            or entry.get("domain")
            or ""
        )

        sentiment_score = float(positive or 0) - float(negative or 0)
        if isinstance(tags, list):
            tag_lower = {str(tag).lower() for tag in tags}
            if "bullish" in tag_lower and "bearish" not in tag_lower:
                sentiment_score = max(sentiment_score, 1.0)
            elif "bearish" in tag_lower and "bullish" not in tag_lower:
                sentiment_score = min(sentiment_score, -1.0)

        rows.append(
            {
                "timestamp": timestamp,
                "title": entry.get("title", ""),
                "url": entry.get("url") or entry.get("link") or "",
                "source": source,
                "sentiment": sentiment_score,
                "positive_votes": float(positive or 0),
                "negative_votes": float(negative or 0),
                "tags": tags_serialised,
                "currencies": currencies_serialised,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
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
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


__all__ = ["fetch_cryptopanic_news"]

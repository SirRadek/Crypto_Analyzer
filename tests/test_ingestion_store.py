from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select

from crypto_analyzer.data import ingestion_store


def _setup_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    ingestion_store.ensure_schema(engine=engine)
    return engine


def test_store_derivatives_upserts_and_tracks_latest_timestamp() -> None:
    engine = _setup_engine()
    funding = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T08:00:00Z"),
            ],
            "funding_rate": [0.01, 0.02],
        }
    )
    open_interest = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01T08:00:00Z"),
                pd.Timestamp("2024-01-01T16:00:00Z"),
            ],
            "open_interest": [1_000.0, 1_100.0],
        }
    )

    result = ingestion_store.store_derivatives(funding, open_interest, engine=engine, symbol="BTCUSDT")
    assert result.inserted == 3

    latest = ingestion_store.latest_derivatives_timestamp(engine, symbol="BTCUSDT")
    assert latest is not None
    assert latest.tz_convert("UTC").to_pydatetime() == datetime(2024, 1, 1, 16, tzinfo=timezone.utc)

    updated_funding = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T08:00:00Z")],
            "funding_rate": [0.05],
        }
    )
    ingestion_store.store_derivatives(updated_funding, pd.DataFrame(), engine=engine, symbol="BTCUSDT")

    query = select(ingestion_store.DERIVATIVES_TABLE.c.funding_rate).where(
        ingestion_store.DERIVATIVES_TABLE.c.timestamp == pd.Timestamp("2024-01-01T08:00:00Z"),
        ingestion_store.DERIVATIVES_TABLE.c.symbol == "BTCUSDT",
    )
    with engine.connect() as conn:
        stored_value = conn.execute(query).scalar_one()
    assert stored_value == 0.05


def test_store_news_deduplicates_by_url() -> None:
    engine = _setup_engine()
    news = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T12:00:00Z")],
            "title": ["Headline"],
            "url": ["https://example.com/news"],
            "source": ["Example"],
            "sentiment": [1.0],
            "positive_votes": [5],
            "negative_votes": [0],
            "tags": ["bullish"],
            "currencies": ["BTC"],
        }
    )
    first = ingestion_store.store_news(news, engine=engine)
    assert first.inserted == 1

    updated_news = news.assign(sentiment=[0.5])
    second = ingestion_store.store_news(updated_news, engine=engine)
    assert second.inserted == 1

    query = select(ingestion_store.NEWS_TABLE.c.sentiment).where(
        ingestion_store.NEWS_TABLE.c.url == "https://example.com/news"
    )
    with engine.connect() as conn:
        sentiment_value = conn.execute(query).scalar_one()
    assert sentiment_value == 0.5

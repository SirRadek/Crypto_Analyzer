from __future__ import annotations

from datetime import datetime, UTC

import pandas as pd
import pytest
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
    basis = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T16:00:00Z"),
            ],
            "basis": [150.0, 200.0],
        }
    )

    result = ingestion_store.store_derivatives(
        funding,
        open_interest,
        basis,
        engine=engine,
        symbol="BTCUSDT",
    )
    assert result.inserted == 3

    latest = ingestion_store.latest_derivatives_timestamp(engine, symbol="BTCUSDT")
    assert latest is not None
    assert latest.tz_convert("UTC").to_pydatetime() == datetime(2024, 1, 1, 16, tzinfo=UTC)

    updated_funding = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T08:00:00Z")],
            "funding_rate": [0.05],
        }
    )
    ingestion_store.store_derivatives(
        updated_funding, pd.DataFrame(), engine=engine, symbol="BTCUSDT"
    )

    query = select(ingestion_store.DERIVATIVES_TABLE.c.funding_rate).where(
        ingestion_store.DERIVATIVES_TABLE.c.timestamp == pd.Timestamp("2024-01-01T08:00:00Z"),
        ingestion_store.DERIVATIVES_TABLE.c.symbol == "BTCUSDT",
    )
    with engine.connect() as conn:
        stored_value = conn.execute(query).scalar_one()
    assert stored_value == 0.05

    basis_query = select(ingestion_store.DERIVATIVES_TABLE.c.basis).where(
        ingestion_store.DERIVATIVES_TABLE.c.timestamp == pd.Timestamp("2024-01-01T16:00:00Z"),
        ingestion_store.DERIVATIVES_TABLE.c.symbol == "BTCUSDT",
    )
    with engine.connect() as conn:
        stored_basis = conn.execute(basis_query).scalar_one()
    assert stored_basis == pytest.approx(200.0)


def test_store_derivatives_preserves_existing_basis_values() -> None:
    engine = _setup_engine()
    initial_funding = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-02-01T00:00:00Z"),
                pd.Timestamp("2024-02-01T08:00:00Z"),
            ],
            "funding_rate": [0.01, 0.02],
        }
    )
    initial_basis = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-02-01T00:00:00Z")],
            "basis": [125.0],
        }
    )

    ingestion_store.store_derivatives(
        initial_funding,
        pd.DataFrame(),
        initial_basis,
        engine=engine,
        symbol="BTCUSDT",
    )

    follow_up_funding = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-02-01T00:00:00Z"),
                pd.Timestamp("2024-02-01T08:00:00Z"),
            ],
            "funding_rate": [0.015, 0.025],
        }
    )
    follow_up_basis = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-02-01T08:00:00Z")],
            "basis": [175.0],
        }
    )

    ingestion_store.store_derivatives(
        follow_up_funding,
        pd.DataFrame(),
        follow_up_basis,
        engine=engine,
        symbol="BTCUSDT",
    )

    query = select(ingestion_store.DERIVATIVES_TABLE.c.basis).where(
        ingestion_store.DERIVATIVES_TABLE.c.timestamp == pd.Timestamp("2024-02-01T00:00:00Z"),
        ingestion_store.DERIVATIVES_TABLE.c.symbol == "BTCUSDT",
    )
    with engine.connect() as conn:
        preserved_basis = conn.execute(query).scalar_one()
    assert preserved_basis == pytest.approx(125.0)

    follow_up_query = select(ingestion_store.DERIVATIVES_TABLE.c.basis).where(
        ingestion_store.DERIVATIVES_TABLE.c.timestamp == pd.Timestamp("2024-02-01T08:00:00Z"),
        ingestion_store.DERIVATIVES_TABLE.c.symbol == "BTCUSDT",
    )
    with engine.connect() as conn:
        updated_basis = conn.execute(follow_up_query).scalar_one()
    assert updated_basis == pytest.approx(175.0)


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


def test_store_whale_transactions_upserts_latest_timestamp() -> None:
    engine = _setup_engine()
    base = pd.Timestamp("2024-03-01T00:00:00Z")
    transactions = pd.DataFrame(
        {
            "timestamp": [base, base + pd.Timedelta(hours=6)],
            "transaction_hash": ["hash-1", "hash-2"],
            "currency": ["btc", "eth"],
            "amount": [12.5, 20.0],
            "amount_usd": [325_000.0, 450_000.0],
            "from_address": ["addr-1", None],
            "to_address": ["addr-2", "addr-3"],
        }
    )

    result = ingestion_store.store_whale_transactions(transactions, engine=engine)
    assert result.inserted == 2

    latest = ingestion_store.latest_whale_transaction_timestamp(engine)
    assert latest is not None
    assert latest.tz_convert("UTC").to_pydatetime() == datetime(2024, 3, 1, 6, tzinfo=UTC)

    update = transactions.iloc[[1]].assign(amount_usd=[475_000.0])
    ingestion_store.store_whale_transactions(update, engine=engine)

    query = select(ingestion_store.WHALE_TRANSACTIONS_TABLE.c.amount_usd).where(
        ingestion_store.WHALE_TRANSACTIONS_TABLE.c.transaction_hash == "hash-2"
    )
    with engine.connect() as conn:
        stored_amount = conn.execute(query).scalar_one()
    assert stored_amount == pytest.approx(475_000.0)

"""TimescaleDB schema management using :mod:`psycopg2`.

This module provides helpers to connect to a TimescaleDB/PostgreSQL instance and
ensure that all tables required by the project exist as hypertables.  The
``combined_features`` view aggregates the most important features for the ML
pipeline in a single relation.

Typical usage from the command line::

    python -m crypto_analyzer.data.db --dsn postgresql://user:pass@host/dbname

The connection information defaults to the ``database.url`` entry from the
loaded application configuration or the ``DATABASE_URL``/``TIMESCALE_URL``
environment variables.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import contextmanager
from typing import Iterable, Iterator

import psycopg2
from psycopg2 import OperationalError, sql
from psycopg2.extensions import connection as PGConnection

from crypto_analyzer.utils.config import CONFIG

LOGGER = logging.getLogger(__name__)

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS timescaledb"

TABLE_DEFINITIONS: dict[str, str] = {
    "market_data": """
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open NUMERIC(18, 8),
            high NUMERIC(18, 8),
            low NUMERIC(18, 8),
            close NUMERIC(18, 8),
            volume NUMERIC(30, 10),
            quote_volume NUMERIC(30, 10),
            trades BIGINT,
            PRIMARY KEY (timestamp, symbol, interval)
        )
    """,
    "onchain_metrics": """
        CREATE TABLE IF NOT EXISTS onchain_metrics (
            timestamp TIMESTAMPTZ NOT NULL,
            active_addresses BIGINT,
            new_addresses BIGINT,
            exchange_inflow NUMERIC(30, 10),
            exchange_outflow NUMERIC(30, 10),
            whale_transactions NUMERIC(30, 10),
            PRIMARY KEY (timestamp)
        )
    """,
    "sentiment_index": """
        CREATE TABLE IF NOT EXISTS sentiment_index (
            timestamp TIMESTAMPTZ NOT NULL,
            value SMALLINT CHECK (value BETWEEN 0 AND 100),
            classification TEXT,
            PRIMARY KEY (timestamp)
        )
    """,
    "social_sentiment": """
        CREATE TABLE IF NOT EXISTS social_sentiment (
            timestamp TIMESTAMPTZ NOT NULL,
            reddit_score DOUBLE PRECISION,
            twitter_score DOUBLE PRECISION,
            mentions BIGINT,
            PRIMARY KEY (timestamp)
        )
    """,
    "derivatives_signals": """
        CREATE TABLE IF NOT EXISTS derivatives_signals (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            funding_rate DOUBLE PRECISION,
            open_interest NUMERIC(30, 10),
            basis DOUBLE PRECISION,
            PRIMARY KEY (timestamp, symbol)
        )
    """,
    "news": """
        CREATE TABLE IF NOT EXISTS news (
            timestamp TIMESTAMPTZ NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            source TEXT,
            sentiment DOUBLE PRECISION,
            PRIMARY KEY (timestamp, title)
        )
    """,
}

HYPERTABLE_STATEMENTS: tuple[str, ...] = (
    "SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('onchain_metrics', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('sentiment_index', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('social_sentiment', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('derivatives_signals', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('news', 'timestamp', if_not_exists => TRUE)",
)

COMBINED_FEATURES_VIEW = """
    CREATE OR REPLACE VIEW combined_features AS
    SELECT
        md.timestamp,
        md.symbol,
        md.interval,
        md.open,
        md.high,
        md.low,
        md.close,
        md.volume,
        md.quote_volume,
        md.trades,
        ds.funding_rate,
        ds.open_interest,
        ds.basis,
        oc.active_addresses,
        oc.new_addresses,
        oc.exchange_inflow,
        oc.exchange_outflow,
        oc.whale_transactions,
        si.value AS fear_greed_index,
        si.classification AS fear_greed_classification,
        ss.reddit_score,
        ss.twitter_score,
        ss.mentions,
        news_agg.sentiment AS news_sentiment
    FROM market_data AS md
    LEFT JOIN derivatives_signals AS ds
        ON ds.timestamp = md.timestamp AND ds.symbol = md.symbol
    LEFT JOIN onchain_metrics AS oc
        ON oc.timestamp = md.timestamp
    LEFT JOIN sentiment_index AS si
        ON si.timestamp = md.timestamp
    LEFT JOIN social_sentiment AS ss
        ON ss.timestamp = md.timestamp
    LEFT JOIN (
        SELECT
            timestamp,
            AVG(sentiment) AS sentiment
        FROM news
        GROUP BY timestamp
    ) AS news_agg ON news_agg.timestamp = md.timestamp
"""


def _collect_connection_settings(
    dsn: str | None,
    *,
    host: str | None,
    port: int | None,
    user: str | None,
    password: str | None,
    dbname: str | None,
    connect_timeout: int | None,
) -> dict[str, object]:
    """Assemble psycopg2 connection keyword arguments."""

    if dsn:
        settings: dict[str, object] = {"dsn": dsn}
    else:
        settings = {}

    if host:
        settings["host"] = host
    if port:
        settings["port"] = port
    if user:
        settings["user"] = user
    if password:
        settings["password"] = password
    if dbname:
        settings["dbname"] = dbname
    if connect_timeout is not None:
        settings["connect_timeout"] = connect_timeout
    return settings


def _resolve_default_dsn() -> str | None:
    """Return the configured TimescaleDB DSN if available."""

    # Environment variables take precedence over configuration files so the
    # module can be used without altering project settings.
    for key in ("TIMESCALE_URL", "DATABASE_URL"):
        value = os.getenv(key)
        if value:
            return value
    return CONFIG.database.url


def connect(
    dsn: str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    dbname: str | None = None,
    connect_timeout: int | None = 10,
) -> PGConnection:
    """Create a psycopg2 connection to TimescaleDB.

    The DSN defaults to values provided by the application configuration or the
    ``TIMESCALE_URL``/``DATABASE_URL`` environment variables.  Keyword
    arguments allow overriding specific parameters.
    """

    resolved_dsn = dsn or _resolve_default_dsn()
    settings = _collect_connection_settings(
        resolved_dsn,
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=connect_timeout,
    )
    if not settings:
        raise ValueError("TimescaleDB connection information is missing")

    LOGGER.debug("Connecting to TimescaleDB with settings: %s", settings.keys())
    try:
        return psycopg2.connect(**settings)
    except OperationalError as exc:  # pragma: no cover - depends on environment
        raise OperationalError(f"Failed to connect to TimescaleDB: {exc}") from exc


@contextmanager
def connection_scope(*args, **kwargs) -> Iterator[PGConnection]:
    """Context manager that opens and closes a TimescaleDB connection."""

    conn = connect(*args, **kwargs)
    try:
        yield conn
    finally:
        conn.close()


def _execute_statements(conn: PGConnection, statements: Iterable[str]) -> None:
    """Execute a sequence of SQL statements inside a single transaction."""

    with conn.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)


def _ensure_column_exists(
    conn: PGConnection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    """Ensure that a table contains a specific column, adding it if required."""

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
            """,
            (table_name, column_name),
        )
        if cursor.fetchone() is None:
            LOGGER.info("Adding missing column %s.%s", table_name, column_name)
            cursor.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN {} {}").format(
                    sql.Identifier(table_name),
                    sql.Identifier(column_name),
                    sql.SQL(column_definition),
                )
            )


def initialize_schema(conn: PGConnection) -> None:
    """Create TimescaleDB tables, hypertables and views if they do not exist."""

    try:
        LOGGER.info("Ensuring TimescaleDB extension is enabled")
        _execute_statements(conn, (CREATE_EXTENSION_SQL,))

        LOGGER.info("Creating base tables")
        _execute_statements(conn, TABLE_DEFINITIONS.values())

        LOGGER.info("Applying schema migrations")
        _ensure_column_exists(conn, "sentiment_index", "classification", "TEXT")

        LOGGER.info("Converting tables to hypertables")
        _execute_statements(conn, HYPERTABLE_STATEMENTS)

        LOGGER.info("Creating combined_features view")
        _execute_statements(conn, (COMBINED_FEATURES_VIEW,))

        conn.commit()
    except Exception:
        conn.rollback()
        raise


def setup_database(**kwargs) -> None:
    """Open a connection, initialise the schema and handle errors gracefully."""

    try:
        with connection_scope(**kwargs) as conn:
            initialize_schema(conn)
    except OperationalError as exc:  # pragma: no cover - depends on environment
        LOGGER.error("Database operation failed: %s", exc)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialise TimescaleDB schema")
    parser.add_argument("--dsn", help="PostgreSQL/TimescaleDB DSN", default=None)
    parser.add_argument("--host", help="Database host", default=None)
    parser.add_argument("--port", help="Database port", type=int, default=None)
    parser.add_argument("--user", help="Database user", default=None)
    parser.add_argument("--password", help="Database password", default=None)
    parser.add_argument("--dbname", help="Database name", default=None)
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Timeout for establishing the connection in seconds",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI wrapper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    setup_database(
        dsn=args.dsn,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
        connect_timeout=args.connect_timeout,
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()


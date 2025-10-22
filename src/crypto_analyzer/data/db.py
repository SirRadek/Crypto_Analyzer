"""TimescaleDB schema management using :mod:`psycopg2`.

This module provides helpers to connect to a TimescaleDB/PostgreSQL instance and
ensure that all tables required by the project exist as hypertables.  The
``combined_features`` materialized view aggregates the most important features
for the ML pipeline in a single relation so feature engineering queries can hit
a precomputed join instead of recalculating it on the fly.  Recreate or refresh
the materialized view whenever one of the underlying tables changes schema or
receives new data.

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
    "whale_transactions": """
        CREATE TABLE IF NOT EXISTS whale_transactions (
            timestamp TIMESTAMPTZ NOT NULL,
            transaction_hash TEXT NOT NULL,
            currency TEXT NOT NULL,
            amount NUMERIC(30, 10),
            amount_usd NUMERIC(30, 10),
            from_address TEXT,
            from_owner TEXT,
            to_address TEXT,
            to_owner TEXT,
            blockchain TEXT,
            PRIMARY KEY (timestamp, transaction_hash)
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
            twitter_positive_count BIGINT,
            twitter_negative_count BIGINT,
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
    "SELECT create_hypertable('whale_transactions', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('sentiment_index', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('social_sentiment', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('derivatives_signals', 'timestamp', if_not_exists => TRUE)",
    "SELECT create_hypertable('news', 'timestamp', if_not_exists => TRUE)",
)

_PREFERRED_MARKET_INTERVAL = CONFIG.interval.replace("'", "''")

COMBINED_FEATURES_DEPENDENCIES: tuple[str, ...] = (
    "market_data",
    "derivatives_signals",
    "onchain_metrics",
    "sentiment_index",
    "social_sentiment",
    "news",
)

COMBINED_FEATURES_MATERIALIZED_VIEW = f"""
    CREATE MATERIALIZED VIEW combined_features AS
    WITH base_market_data AS (
        SELECT DISTINCT ON (timestamp, symbol)
            timestamp,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            trades
        FROM market_data
        ORDER BY
            timestamp,
            symbol,
            CASE WHEN interval = '{_PREFERRED_MARKET_INTERVAL}' THEN 0 ELSE 1 END,
            interval
    )
    SELECT
        md.timestamp,
        md.symbol,
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
    FROM base_market_data AS md
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

COMBINED_FEATURES_INDEX = """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_combined_features_timestamp_symbol
        ON combined_features (timestamp, symbol)
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


def _ensure_combined_features_dependencies(conn: PGConnection) -> None:
    """Validate that the tables required by ``combined_features`` exist."""

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = ANY(%s)
            """,
            (list(COMBINED_FEATURES_DEPENDENCIES),),
        )
        existing = {name for (name,) in cursor.fetchall()}

    missing = set(COMBINED_FEATURES_DEPENDENCIES) - existing
    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(
            "Cannot create combined_features materialized view; missing tables: %s" % joined
        )


def _create_combined_features_materialized_view(conn: PGConnection) -> None:
    """Create the ``combined_features`` materialized view and its index."""

    _ensure_combined_features_dependencies(conn)
    LOGGER.info("Creating combined_features materialized view")
    _execute_statements(
        conn,
        (
            "DROP MATERIALIZED VIEW IF EXISTS combined_features",
            "DROP VIEW IF EXISTS combined_features",
            COMBINED_FEATURES_MATERIALIZED_VIEW,
            COMBINED_FEATURES_INDEX,
        ),
    )


def refresh_combined_features(
    conn: PGConnection,
    *,
    concurrently: bool = False,
) -> None:
    """Refresh the ``combined_features`` materialized view.

    Parameters
    ----------
    conn:
        Open psycopg2 connection targeting TimescaleDB.
    concurrently:
        Request ``REFRESH MATERIALIZED VIEW CONCURRENTLY`` to keep the view
        accessible for reads during refresh.  Requires the unique index created
        by :func:`_create_combined_features_materialized_view`.

    Notes
    -----
    Schedule refreshes after data ingestion completes (e.g. nightly) so model
    training jobs always see up-to-date features.  ``REFRESH CONCURRENTLY``
    avoids blocking reads but still needs to reprocess the full dataset; when
    refresh time becomes an issue consider partial refresh strategies such as
    materializing only recent partitions into staging tables.
    """

    keyword = " CONCURRENTLY" if concurrently else ""
    statement = f"REFRESH MATERIALIZED VIEW{keyword} combined_features"
    original_autocommit_value = getattr(conn, "autocommit", False)
    original_autocommit = bool(original_autocommit_value)
    autocommit_enabled = False

    try:
        if concurrently and not original_autocommit:
            conn.autocommit = True
            autocommit_enabled = True

        with conn.cursor() as cursor:
            cursor.execute(statement)

        if not concurrently and not original_autocommit:
            conn.commit()
    except Exception as exc:
        if not concurrently and not original_autocommit:
            conn.rollback()
        LOGGER.error("Failed to refresh combined_features materialized view: %s", exc)
        raise
    finally:
        if autocommit_enabled:
            conn.autocommit = original_autocommit_value


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

        _create_combined_features_materialized_view(conn)

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


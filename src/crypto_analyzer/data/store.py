"""Abstractions for accessing historical price data.

The original CLI entrypoints interacted with :func:`get_price_data` directly
and required the caller to know whether a local SQLite file or a remote
TimescaleDB instance should be used.  The new Typer applications rely on a
lightweight :class:`PriceDataStore` interface instead which centralises the
selection logic and keeps the command implementations free from backend
concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

from crypto_analyzer.data.db_connector import get_latest_open_time, get_price_data
from crypto_analyzer.utils.config import CONFIG


class PriceDataStore(Protocol):
    """Protocol implemented by concrete price data stores."""

    label: str

    def fetch_prices(
        self,
        symbol: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV candles for ``symbol`` between ``start_ts`` and ``end_ts``."""

    def latest_open_time(
        self,
        *,
        symbol: str | None = None,
        interval: str | None = None,
    ) -> int | None:
        """Return the newest ``open_time`` available for the requested filters."""


@dataclass(slots=True)
class SQLiteDataStore:
    """Adapter that sources data from a SQLite database file."""

    path: Path
    label: str = "sqlite"

    def fetch_prices(
        self,
        symbol: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        return get_price_data(symbol, start_ts=start_ts, end_ts=end_ts, db_path=self.path)

    def latest_open_time(
        self,
        *,
        symbol: str | None = None,
        interval: str | None = None,
    ) -> int | None:
        return get_latest_open_time(symbol=symbol, interval=interval, db_path=self.path)


@dataclass(slots=True)
class TimescaleDataStore:
    """Adapter that connects to a TimescaleDB/PostgreSQL instance via SQLAlchemy."""

    url: str
    label: str = "timescale"

    def fetch_prices(
        self,
        symbol: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        return get_price_data(symbol, start_ts=start_ts, end_ts=end_ts, db_path=self.url)

    def latest_open_time(
        self,
        *,
        symbol: str | None = None,
        interval: str | None = None,
    ) -> int | None:
        return get_latest_open_time(symbol=symbol, interval=interval, db_path=self.url)


def _is_sqlite_url(url: str | None) -> bool:
    if not url:
        return False
    return url.startswith("sqlite:")


def resolve_data_store(
    preference: Literal["auto", "sqlite", "timescale"] = "auto",
    *,
    sqlite_path: Path | None = None,
    timescale_url: str | None = None,
) -> PriceDataStore:
    """Instantiate a :class:`PriceDataStore` based on configuration and overrides."""

    cfg = CONFIG.database
    sqlite_default = sqlite_path or cfg.price_store
    timescale_default = timescale_url or cfg.url

    if preference == "sqlite":
        return SQLiteDataStore(Path(sqlite_default))

    if preference == "timescale":
        if not timescale_default:
            raise ValueError("Timescale store requested but no database URL was configured")
        return TimescaleDataStore(timescale_default)

    # Automatic resolution: prefer Timescale when a non-SQLite URL is configured.
    if timescale_default and not _is_sqlite_url(timescale_default):
        return TimescaleDataStore(timescale_default)

    return SQLiteDataStore(Path(sqlite_default))


__all__ = [
    "PriceDataStore",
    "SQLiteDataStore",
    "TimescaleDataStore",
    "resolve_data_store",
]


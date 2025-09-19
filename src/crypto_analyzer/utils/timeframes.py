"""Utilities for working with Binance-style interval strings."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntervalSpec:
    """Metadata describing a supported candle interval."""

    label: str
    minutes: int


_INTERVAL_SPECS: dict[str, IntervalSpec] = {
    "1m": IntervalSpec("1m", 1),
    "3m": IntervalSpec("3m", 3),
    "5m": IntervalSpec("5m", 5),
    "15m": IntervalSpec("15m", 15),
    "30m": IntervalSpec("30m", 30),
    "1h": IntervalSpec("1h", 60),
    "2h": IntervalSpec("2h", 120),
    "4h": IntervalSpec("4h", 240),
    "1d": IntervalSpec("1d", 1440),
}


def interval_to_minutes(interval: str) -> int:
    """Return the number of minutes represented by *interval*."""

    key = interval.lower()
    try:
        return _INTERVAL_SPECS[key].minutes
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported interval: {interval!r}") from exc


def interval_to_pandas_freq(interval: str) -> str:
    """Convert *interval* to a pandas frequency string."""

    minutes = interval_to_minutes(interval)
    if minutes % 1440 == 0:
        days = minutes // 1440
        return f"{days}D"
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours}H"
    return f"{minutes}T"


def steps_for_minutes(minutes: int, interval: str) -> int:
    """Return the number of candles covering *minutes* for *interval*."""

    base = interval_to_minutes(interval)
    if minutes % base != 0:
        raise ValueError(
            "Interval must evenly divide the requested minutes: "
            f"{minutes}m mod {interval} != 0",
        )
    return minutes // base


__all__ = [
    "IntervalSpec",
    "interval_to_minutes",
    "interval_to_pandas_freq",
    "steps_for_minutes",
]

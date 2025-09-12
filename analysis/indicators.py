from __future__ import annotations

import pandas as pd


def calculate_sma(series: pd.Series, window: int = 14) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_ema(series: pd.Series, window: int = 14) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff().astype(float)
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# You can add more indicators later (e.g. Bollinger Bands, MACD, ...)

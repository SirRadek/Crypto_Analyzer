from __future__ import annotations

import pandas as pd


def rsi_buy_signal(df: pd.DataFrame, threshold: int = 30) -> pd.Series:
    """Return 1 when RSI is below ``threshold``."""
    return (df["rsi_14"] < threshold).astype(int)


def sma_crossover_signal(
    df: pd.DataFrame, short_window: int = 7, long_window: int = 14
) -> pd.Series:
    """Return 1 when short-term SMA crosses above long-term SMA."""
    buy_signal = (df[f"sma_{short_window}"] > df[f"sma_{long_window}"]).astype(int)
    return buy_signal


def combined_signal(df: pd.DataFrame) -> pd.Series:
    """Combine basic signals into a single indicator."""
    rsi_signal = rsi_buy_signal(df)
    sma_signal = sma_crossover_signal(df)
    return ((rsi_signal + sma_signal) > 0).astype(int)

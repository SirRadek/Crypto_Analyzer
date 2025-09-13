from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_buy_signal(df: pd.DataFrame, threshold: int = 30) -> pd.Series:
    """Return 1 when RSI is below ``threshold`` (oversold)."""
    return (df["rsi_14"] < threshold).astype(np.int8)


def sma_crossover_signal(
    df: pd.DataFrame, short_window: int = 7, long_window: int = 14
) -> pd.Series:
    """Return 1 when short-term SMA is above long-term SMA (bullish trend)."""
    return (df[f"sma_{short_window}"] > df[f"sma_{long_window}"]).astype(np.int8)


def combined_signal(
    df: pd.DataFrame,
    *,
    weight_rsi: float = 0.4,
    weight_sma: float = 0.6,
    prob_threshold: float = 0.5,
) -> pd.Series:
    """
    Kombinace základních signálů do jednoho indikátoru.

    Defaulty laděné pro BTCUSDT 2h:
      - SMA (trend) má váhu 0.6
      - RSI (momentum/oversold) má váhu 0.4
      - výstup 1, pokud vážený součet >= 0.5
    """
    rsi_signal = rsi_buy_signal(df).astype(np.float32)
    sma_signal = sma_crossover_signal(df).astype(np.float32)

    s = float(weight_rsi + weight_sma) or 1.0
    w_rsi = float(weight_rsi) / s
    w_sma = float(weight_sma) / s

    blended = (w_rsi * rsi_signal) + (w_sma * sma_signal)
    return (blended >= prob_threshold).astype(np.int8)

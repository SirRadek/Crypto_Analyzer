def rsi_buy_signal(df, threshold=30):
    """
    Generates a buy signal (1) if RSI is below the threshold (default 30), else 0.
    """
    return (df["rsi_14"] < threshold).astype(int)


def sma_crossover_signal(df, short_window=7, long_window=14):
    """
    Generates a buy signal (1) if short-term SMA crosses above long-term SMA, else 0.
    """
    buy_signal = (df[f"sma_{short_window}"] > df[f"sma_{long_window}"]).astype(int)
    return buy_signal


def combined_signal(df):
    """
    Example: Combines multiple rules into a single signal.
    You can expand logic here as needed.
    """
    rsi_signal = rsi_buy_signal(df)
    sma_signal = sma_crossover_signal(df)
    return ((rsi_signal + sma_signal) > 0).astype(int)

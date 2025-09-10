def calculate_sma(series, window=14):
    """Simple Moving Average"""
    return series.rolling(window=window).mean()


def calculate_ema(series, window=14):
    """Exponential Moving Average"""
    return series.ewm(span=window, adjust=False).mean()


def calculate_rsi(series, window=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# You can add more indicators later (e.g. Bollinger Bands, MACD, ...)

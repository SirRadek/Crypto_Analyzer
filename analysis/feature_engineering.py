import pandas as pd
from .indicators import calculate_sma, calculate_ema, calculate_rsi

def create_features(df):
    """
    Adds new columns with indicators and additional features to the dataframe.
    """
    df = df.copy()
    df['return_1d'] = df['close'].pct_change()
    df['sma_7'] = calculate_sma(df['close'], window=7)
    df['sma_14'] = calculate_sma(df['close'], window=14)
    df['ema_7'] = calculate_ema(df['close'], window=7)
    df['ema_14'] = calculate_ema(df['close'], window=14)
    df['rsi_14'] = calculate_rsi(df['close'], window=14)
    # Additional features you may consider:
    # df['volume_mean_10'] = df['volume'].rolling(window=10).mean()
    # df['price_volatility_10'] = df['close'].rolling(window=10).std()
    df = df.dropna()
    return df


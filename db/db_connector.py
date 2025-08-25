import pandas as pd
import sqlite3

def get_price_data(symbol, db_path="data/crypto_data.sqlite"):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            open_time as timestamp, open, high, low, close, volume,
            number_of_trades, quote_asset_volume, taker_buy_base, taker_buy_quote
        FROM prices
        WHERE symbol = ?
        ORDER BY open_time
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    # PŘEVOD NA DATETIME (milisekundy!)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
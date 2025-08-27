import requests
import sqlite3
import time
from datetime import datetime
import os

DB_PATH = 'data/crypto_data.sqlite'   # new path!
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
TABLE_NAME = 'prices'                 # general table (you can use 'btc_ohlcv_5m' if you want to keep it separate)
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
START_DATE = "2020-01-01"
END_DATE = "2025-08-27"

# Convert date string to milliseconds
def date_to_milliseconds(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)

# Download 5-minute candles between the specified timestamps
def get_klines(symbol, interval, start_ts, end_ts, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": limit
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"API error: {response.status_code}, {response.text}")
        return []
    data = response.json()
    return data

# Create database and table if not exists
def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        open_time INTEGER PRIMARY KEY,
        symbol TEXT,
        interval TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        close_time INTEGER,
        quote_asset_volume REAL,
        number_of_trades INTEGER,
        taker_buy_base REAL,
        taker_buy_quote REAL
    )""")
    conn.commit()
    conn.close()

# Save rows into the database
def save_to_db(rows, symbol, interval):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for row in rows:
        c.execute(f"""
        INSERT OR IGNORE INTO {TABLE_NAME}
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row[0]), symbol, interval,
            float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
            int(row[6]), float(row[7]), int(row[8]), float(row[9]), float(row[10])
        ))
    conn.commit()
    conn.close()

def main():
    create_db()
    start_ts = date_to_milliseconds(START_DATE)
    end_ts = date_to_milliseconds(END_DATE)
    curr_ts = start_ts

    print(f"Downloading {SYMBOL}, interval {INTERVAL} from {START_DATE} to {END_DATE}")

    while curr_ts < end_ts:
        klines = get_klines(SYMBOL, INTERVAL, curr_ts, end_ts)
        if not klines:
            break
        save_to_db(klines, SYMBOL, INTERVAL)
        print(f"Downloaded and saved {len(klines)} rows. First timestamp: {klines[0][0]}")
        curr_ts = klines[-1][0] + 1  # Move to the next candle
        time.sleep(0.4)

    print("Done! Data saved in database:", DB_PATH)

if __name__ == "__main__":
    main()
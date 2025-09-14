from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from pydantic import BaseModel
from requests.adapters import HTTPAdapter, Retry


class OnChainConfig(BaseModel):
    use_mempool: bool = True
    use_exchange_flows: bool = True
    use_usdt_events: bool = True
    cache_dir: str = "data/cache"
    glassnode_api_key: str | None = None
    whale_api_key: str | None = None


CONFIG = OnChainConfig()


def _session() -> requests.Session:
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def _cache_path(prefix: str, start: datetime | None = None, end: datetime | None = None) -> Path:
    cache_dir = Path(CONFIG.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if start and end:
        key = f"{prefix}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.parquet"
    else:
        key = f"{prefix}.parquet"
    return cache_dir / key


def fetch_mempool_5m(start: datetime, end: datetime) -> pd.DataFrame:
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)
    cache_file = _cache_path("mempool5m", start, end)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    sess = _session()
    params = {"start": int(start.timestamp()), "end": int(end.timestamp())}
    tx = sess.get("https://mempool.space/api/v1/statistics/transactions", params=params, timeout=10)
    fee = sess.get("https://mempool.space/api/v1/statistics/fees/median", params=params, timeout=10)
    tx.raise_for_status()
    fee.raise_for_status()
    df_tx = pd.DataFrame(tx.json())
    df_fee = pd.DataFrame(fee.json())
    df = pd.merge(df_tx, df_fee, on="time", how="outer")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.resample("5T").agg({"tx_count": "sum", "median_fee": "median"})
    df.to_parquet(cache_file)
    return df


def load_exchange_flows_1h(
    source: str = "csv",
    path: str | None = None,
    glassnode_api_key: str | None = None,
) -> pd.DataFrame:
    cache_file = _cache_path(f"exchange_flows_{source}")
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    if source == "csv":
        if not path:
            raise ValueError("CSV source requires `path`")
        df = pd.read_csv(path, parse_dates=[0])
        df = df.set_index(df.columns[0]).sort_index()
    elif source == "glassnode":
        key = glassnode_api_key or CONFIG.glassnode_api_key
        if not key:
            raise ValueError("glassnode_api_key required")
        sess = _session()
        base = "https://api.glassnode.com/v1/metrics/exchanges"
        params = {"a": "BTC", "i": "1h", "api_key": key}
        inflow = sess.get(f"{base}/inflow_sum", params=params, timeout=10)
        outflow = sess.get(f"{base}/outflow_sum", params=params, timeout=10)
        inflow.raise_for_status()
        outflow.raise_for_status()
        df_in = pd.DataFrame(inflow.json())
        df_out = pd.DataFrame(outflow.json())
        df = pd.merge(df_in, df_out, on="t", how="outer", suffixes=("_in", "_out"))
        df["time"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.set_index("time").drop(columns=["t_in", "t_out"], errors="ignore")
        df.rename(columns={"v_in": "inflow", "v_out": "outflow"}, inplace=True)
        df["netflow"] = df["inflow"] - df["outflow"]
    else:
        raise ValueError("source must be 'csv' or 'glassnode'")

    df = df.resample("1H").sum(min_count=1)
    df.to_parquet(cache_file)
    return df


def fetch_usdt_events(start: datetime, end: datetime, api_key: str | None = None) -> pd.DataFrame:
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)
    cache_file = _cache_path("usdt_events", start, end)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    key = api_key or CONFIG.whale_api_key
    if not key:
        raise ValueError("api_key required for Whale Alert")
    sess = _session()
    params = {
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
        "currency": "usdt",
        "api_key": key,
    }
    resp = sess.get("https://api.whale-alert.io/v1/transactions", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("transactions", [])
    records: list[dict[str, float]] = []
    for tx in data:
        ts = pd.to_datetime(tx.get("timestamp"), unit="s", utc=True)
        usd = float(tx.get("amount_usd", 0.0))
        records.append({"timestamp": ts, "usd": usd})
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("timestamp").sort_index()
        df["count"] = 1
        df = df.resample("5T").agg({"count": "sum", "usd": "sum"})
    else:
        idx = pd.DatetimeIndex([], tz="UTC")
        df = pd.DataFrame(columns=["count", "usd"], index=idx)
    df.to_parquet(cache_file)
    return df


__all__ = [
    "OnChainConfig",
    "fetch_mempool_5m",
    "load_exchange_flows_1h",
    "fetch_usdt_events",
]


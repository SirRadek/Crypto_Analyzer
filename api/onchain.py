from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel
from requests.adapters import HTTPAdapter, Retry


class OnChainConfig(BaseModel):  # type: ignore[misc]
    use_mempool: bool = True
    use_exchange_flows: bool = True
    use_usdt_events: bool = True
    cache_dir: str = "data/cache"
    glassnode_api_key: str | None = None
    whale_api_key: str | None = None


CONFIG = OnChainConfig()


def _session() -> requests.Session:
    retries = Retry(total=0)
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def _get_with_retry(
    sess: requests.Session,
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 10,
    retries: int = 5,
    backoff: float = 1.0,
) -> requests.Response:
    delay = backoff
    for attempt in range(retries):
        try:
            resp = sess.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _cache_path(prefix: str, start: datetime | None = None, end: datetime | None = None) -> Path:
    cache_dir = Path(CONFIG.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if start and end:
        key = f"{prefix}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.parquet"
    else:
        key = f"{prefix}.parquet"
    return cache_dir / key


def fetch_mempool_5m(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch 5-minute mempool statistics for the given interval."""

    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)
    cache_file = _cache_path("mempool5m", start, end)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    sess = _session()
    params = {"start": int(start.timestamp()), "end": int(end.timestamp())}
    tx = _get_with_retry(
        sess,
        "https://mempool.space/api/v1/statistics/transactions",
        params=params,
        timeout=10,
    )
    fee = _get_with_retry(
        sess,
        "https://mempool.space/api/v1/statistics/fees/median",
        params=params,
        timeout=10,
    )
    df_tx = pd.DataFrame(tx.json())
    df_fee = pd.DataFrame(fee.json())
    df = pd.merge(df_tx, df_fee, on="time", how="outer")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    # label and close on the right edge so that a 5 minute bucket ending at
    # ``t`` only contains information up to ``t``.  This prevents forward
    # looking leakage when aligning with price candles labelled by their close
    # time.
    df = df.resample("5T", label="right", closed="right").agg(
        {"tx_count": "sum", "median_fee": "median"}
    )
    df.rename(
        columns={"tx_count": "onch_tx_count", "median_fee": "onch_median_fee"},
        inplace=True,
    )
    df.to_parquet(cache_file)
    return df


def load_exchange_flows_1h(
    source: str = "csv",
    path: str | None = None,
    glassnode_api_key: str | None = None,
) -> pd.DataFrame:
    """Load hourly exchange flow data from CSV or Glassnode API."""

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
        inflow = _get_with_retry(sess, f"{base}/inflow_sum", params=params, timeout=10)
        outflow = _get_with_retry(sess, f"{base}/outflow_sum", params=params, timeout=10)
        df_in = pd.DataFrame(inflow.json())
        df_out = pd.DataFrame(outflow.json())
        df = pd.merge(df_in, df_out, on="t", how="outer", suffixes=("_in", "_out"))
        df["time"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.set_index("time").drop(columns=["t_in", "t_out"], errors="ignore")
        df.rename(columns={"v_in": "onch_inflow", "v_out": "onch_outflow"}, inplace=True)
        df["onch_netflow"] = df["onch_inflow"] - df["onch_outflow"]
    else:
        raise ValueError("source must be 'csv' or 'glassnode'")

    df = df.resample("1H", label="right", closed="right").sum(min_count=1)
    df.rename(columns={c: f"onch_{c}" for c in df.columns}, inplace=True)
    df.to_parquet(cache_file)
    return df


def fetch_usdt_events(start: datetime, end: datetime, api_key: str | None = None) -> pd.DataFrame:
    """Fetch Whale Alert USDT transfer events within the given interval."""

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
    resp = _get_with_retry(
        sess, "https://api.whale-alert.io/v1/transactions", params=params, timeout=10
    )
    data = resp.json().get("transactions", [])
    records: list[dict[str, float]] = []
    for tx in data:
        ts = pd.to_datetime(tx.get("timestamp"), unit="s", utc=True)
        usd = float(tx.get("amount_usd", 0.0))
        records.append({"timestamp": ts, "usd": usd})
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("timestamp").sort_index()
        df["onch_usdt_count"] = 1
        df.rename(columns={"usd": "onch_usd"}, inplace=True)
        df = df.resample("5T", label="right", closed="right").agg(
            {"onch_usdt_count": "sum", "onch_usd": "sum"}
        )
    else:
        idx = pd.DatetimeIndex([], tz="UTC")
        df = pd.DataFrame(columns=["onch_usdt_count", "onch_usd"], index=idx)
    df.to_parquet(cache_file)
    return df


__all__ = [
    "OnChainConfig",
    "fetch_mempool_5m",
    "load_exchange_flows_1h",
    "fetch_usdt_events",
]

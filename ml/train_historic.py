from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .train_price import load_config, train_price


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train models on a specific historical range")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--outdir", default="models/historic")
    parser.add_argument("--data", required=True)
    args = parser.parse_args(argv)

    df = (
        pd.read_parquet(args.data)
        if args.data.endswith(".parquet")
        else pd.read_csv(args.data, parse_dates=["timestamp"])
    )
    mask = (df["timestamp"] >= args.start_date) & (df["timestamp"] <= args.end_date)
    df = df.loc[mask]
    config = load_config(Path(args.config))
    outdir = Path(args.outdir) / f"{args.start_date}_{args.end_date}"
    train_price(df, config, outdir=outdir)


if __name__ == "__main__":
    main()

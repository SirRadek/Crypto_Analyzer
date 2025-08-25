# find_predicted_jumps_v2.py
import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

INTERVAL_TO_MIN = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"1d":1440}
DEFAULT_SYMBOL   = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
DEFAULT_HSTEPS   = 1

# Script expected at repo root; resolve DB relative to root
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = (REPO_ROOT / "db/data/crypto_data.sqlite").resolve()

# --------------------- DB loading helpers ---------------------

def _guess_default_window(conn, symbol):
    q = "SELECT MAX(prediction_time_ms) AS mx FROM prediction WHERE symbol=?"
    mx = pd.read_sql(q, conn, params=(symbol,))["mx"].iloc[0]
    if pd.isna(mx):
        return None, None
    max_ts = pd.to_datetime(int(mx), unit="ms", utc=True)
    start = (max_ts - pd.Timedelta(days=3)).floor("D")
    end   = max_ts.ceil("D")
    return start, end

def load_predictions_with_base(db_path, symbol, interval, horizon_steps,
                               start_dt=None, end_dt=None, only_unfilled=True):
    conn = sqlite3.connect(db_path)
    params = [symbol, interval, horizon_steps]

    where = ["p.symbol = ?","p.interval = ?","p.horizon_steps = ?"]
    if start_dt is not None:
        where.append("p.prediction_time_ms >= ?")
        params.append(int(pd.Timestamp(start_dt).value // 1_000_000))
    if end_dt is not None:
        where.append("p.prediction_time_ms < ?")
        params.append(int(pd.Timestamp(end_dt).value // 1_000_000))
    if only_unfilled:
        where.append("p.y_true IS NULL")

    sql = f"""
    SELECT
      p.symbol, p.interval, p.horizon_steps,
      p.prediction_time_ms, p.target_time_ms, p.y_pred,
      p.y_true, p.abs_error, p.features_version, p.created_at,
      pr.close AS base_close
    FROM prediction p
    LEFT JOIN prices pr
      ON pr.open_time = p.prediction_time_ms
     AND pr.symbol = p.symbol
    WHERE {" AND ".join(where)}
    ORDER BY p.prediction_time_ms ASC
    """
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df

def add_base_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Fill base price for forward rows by chaining previous y_pred when target==next pred_time."""
    if df.empty:
        return df
    df = df.copy()
    df["prev_target_time_ms"] = df["target_time_ms"].shift(1)
    df["prev_y_pred"] = df["y_pred"].shift(1)
    mask_fwd = df["base_close"].isna() & (df["prev_target_time_ms"] == df["prediction_time_ms"])
    df.loc[mask_fwd, "base_close"] = df.loc[mask_fwd, "prev_y_pred"]
    df = df.dropna(subset=["base_close"]).copy()
    return df

# --------------------- Signal detection ---------------------

def find_multibar_signals(df: pd.DataFrame, interval: str,
                          min_minutes: int, max_minutes: int,
                          threshold_pct: float,
                          require_consecutive: bool = True) -> pd.DataFrame:
    """
    For each prediction_time, find the earliest horizon (<= max_minutes) where
    |predicted move| >= threshold_pct, relative to base_close.
    """
    if df.empty:
        return df

    step_min = INTERVAL_TO_MIN.get(interval, 5)
    step_ms  = step_min * 60_000
    min_bars = max(1, int(np.ceil(min_minutes / step_min)))
    max_bars = max(min_bars, int(np.floor(max_minutes / step_min)))

    key = df["prediction_time_ms"].values
    val_pred = df["y_pred"].values
    val_tgt  = df["target_time_ms"].values
    y_map = {int(k): (float(vp), int(vt)) for k, vp, vt in zip(key, val_pred, val_tgt)}
    have = set(int(t) for t in key)

    out_rows = []
    for i in range(len(df)):
        start_ms  = int(df.iloc[i]["prediction_time_ms"])
        base      = float(df.iloc[i]["base_close"])
        start_pt  = pd.to_datetime(start_ms, unit="ms", utc=True)

        for k in range(min_bars, max_bars + 1):
            if require_consecutive:
                # ensure all intermediate pred rows are present
                ok = True
                for j in range(1, k + 1):
                    tp = start_ms + (j - 1) * step_ms
                    if tp not in have:
                        ok = False
                        break
                if not ok:
                    break

            pred_time_k = start_ms + (k - 1) * step_ms
            end_tuple = y_map.get(pred_time_k)
            if end_tuple is None:
                if require_consecutive:
                    break
                else:
                    continue

            end_pred, end_target_ms = end_tuple
            move_pct = (end_pred - base) / base * 100.0
            if abs(move_pct) >= threshold_pct:
                out_rows.append({
                    "prediction_time": start_pt,  # tz-aware UTC
                    "target_time": pd.to_datetime(end_target_ms, unit="ms", utc=True),
                    "base_close": base,
                    "y_pred_end": end_pred,
                    "pred_move_pct": move_pct,
                    "direction": "UP" if move_pct >= 0 else "DOWN",
                    "trade_duration_min": k * step_min
                })
                break

    if not out_rows:
        return pd.DataFrame(columns=[
            "prediction_time","target_time","base_close","y_pred_end",
            "pred_move_pct","direction","trade_duration_min"
        ])

    return pd.DataFrame(out_rows).sort_values(["prediction_time","target_time"])

# --------------------- Scored non-overlap selector ---------------------

def _score_row(row: pd.Series, mode: str, alpha: float, beta: float) -> float:
    abs_move = abs(float(row["pred_move_pct"]))
    minutes  = max(1.0, float(row["trade_duration_min"]))
    if mode == "speed":
        return abs_move / minutes
    if mode == "custom":
        return alpha * abs_move - beta * minutes
    # default: max_move
    return abs_move

def select_non_overlap_scored(signals: pd.DataFrame,
                              tail_overlap_minutes: int = 10,
                              require_same_direction: bool = True,
                              strategy: str = "max_move",
                              alpha: float = 1.0,
                              beta: float = 0.0) -> pd.DataFrame:
    """
    Build a non-overlapping set of windows:
      - Form overlapping clusters (by start < current cluster_end).
      - Pick ONE 'best' per cluster via scoring strategy:
          'max_move' (|move|), 'speed' (|move|/min), 'custom' (alpha*|move|-beta*min).
      - Allow tail overlap: after picking, you may add subsequent windows
        whose start >= (picked.end - tail) and (same direction if required).
    """
    if signals.empty:
        return signals

    s = signals.sort_values(["prediction_time","target_time"]).reset_index(drop=True)
    kept = []
    i = 0
    tail = pd.Timedelta(minutes=max(0, int(tail_overlap_minutes)))

    while i < len(s):
        # ---- Build overlapping cluster starting at i
        cluster_idxs = [i]
        cluster_end  = s.loc[i, "target_time"]
        j = i + 1
        while j < len(s) and s.loc[j, "prediction_time"] < cluster_end:
            cluster_idxs.append(j)
            if s.loc[j, "target_time"] > cluster_end:
                cluster_end = s.loc[j, "target_time"]
            j += 1

        # ---- Pick best row by score
        sub = s.loc[cluster_idxs]
        scores = sub.apply(lambda r: _score_row(r, strategy, alpha, beta), axis=1)
        best_idx = scores.idxmax()
        best_row = s.loc[best_idx]
        kept.append(best_row)

        # ---- Optional tail top-ups (same direction by default)
        last_end  = best_row["target_time"]
        last_dir  = best_row["direction"]
        k = best_idx + 1
        while k < len(s) and s.loc[k, "prediction_time"] < last_end:
            start_k = s.loc[k, "prediction_time"]
            dir_k   = s.loc[k, "direction"]
            if start_k >= (last_end - tail) and (not require_same_direction or dir_k == last_dir):
                kept.append(s.loc[k])
                last_end = s.loc[k, "target_time"]
                last_dir = dir_k
            k += 1

        # ---- Advance i to first start >= last_end (hard no-overlap beyond tail)
        while i < len(s) and s.loc[i, "prediction_time"] < last_end:
            i += 1

    return pd.DataFrame(kept).sort_values(["prediction_time","target_time"]).reset_index(drop=True)

# --------------------- CLI / main ---------------------

def main():
    ap = argparse.ArgumentParser(
        description="Find multi-bar predicted jumps >= threshold% (≤ max window), "
                    "choose the best inside overlaps, and enforce non-overlap with an optional tail overlap."
    )
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to SQLite DB")
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--interval", default=DEFAULT_INTERVAL)
    ap.add_argument("--horizon_steps", type=int, default=DEFAULT_HSTEPS)

    ap.add_argument("--threshold", type=float, default=0.4, help="Percent; 0.5 means 0.5%%")
    ap.add_argument("--min-minutes", type=int, default=5, help="Earliest duration to consider (minutes)")
    ap.add_argument("--max-minutes", type=int, default=120, help="Latest duration to consider (minutes)")

    ap.add_argument("--start", type=str, default=None, help="prediction_time >= YYYY-MM-DD")
    ap.add_argument("--end",   type=str, default=None, help="prediction_time < YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=200, help="Max rows to print (0 = all)")

    ap.add_argument("--include-filled", action="store_true",
                    help="Include rows where y_true is present (default scans only y_true IS NULL).")
    ap.add_argument("--no-require-consecutive", action="store_true",
                    help="Allow gaps between 5m bars (default requires consecutive bars).")

    # Non-overlap controls (+ scoring)
    ap.add_argument("--allow-overlap", action="store_true",
                    help="Show overlapping windows without filtering (default = scored non-overlap with tail).")
    ap.add_argument("--tail-overlap-minutes", type=int, default=15,
                    help="Permit next window to start up to N minutes before previous ends (default 10).")
    ap.add_argument("--overlap-any-direction", action="store_true",
                    help="Allow tail-overlap even if direction differs (default requires same direction).")
    ap.add_argument("--select-strategy", choices=["max_move","speed","custom"], default="max_move",
                    help="How to choose the best window inside each overlapping period.")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha for 'custom' score: alpha*|move| - beta*minutes")
    ap.add_argument("--beta",  type=float, default=0.0, help="beta  for 'custom' score: alpha*|move| - beta*minutes")

    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print("ERROR: database file not found.")
        print(f"  Expected at: {db_path}")
        print(f"  CWD:         {Path.cwd()}")
        return

    start_dt = pd.to_datetime(args.start) if args.start else None
    end_dt   = pd.to_datetime(args.end)   if args.end   else None

    if start_dt is None:
        with sqlite3.connect(str(db_path)) as conn:
            start_guess, end_guess = _guess_default_window(conn, args.symbol)
        start_dt = start_guess
        if end_dt is None:
            end_dt = end_guess

    df = load_predictions_with_base(
        db_path=str(db_path),
        symbol=args.symbol,
        interval=args.interval,
        horizon_steps=args.horizon_steps,
        start_dt=start_dt,
        end_dt=end_dt,
        only_unfilled=(not args.include_filled)
    )

    df = add_base_prices(df)
    signals = find_multibar_signals(
        df=df,
        interval=args.interval,
        min_minutes=args.min_minutes,
        max_minutes=args.max_minutes,
        threshold_pct=args.threshold,
        require_consecutive=(not args.no_require_consecutive)
    )

    if signals.empty:
        print("No multi-bar predicted jumps found for given filters.")
        return

    if not args.allow_overlap:
        before = len(signals)
        signals = select_non_overlap_scored(
            signals,
            tail_overlap_minutes=args.tail_overlap_minutes,
            require_same_direction=(not args.overlap_any_direction),
            strategy=args.select_strategy,
            alpha=args.alpha,
            beta=args.beta
        )
        after = len(signals)
        print(f"Non-overlap (tail={args.tail_overlap_minutes}m, "
              f"strategy={args.select_strategy}) applied: {before} → {after} windows.")

    # Pretty print (UTC; convert to local tz if desired)
    show = signals.copy()
    show["prediction_time"] = show["prediction_time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    show["target_time"]     = show["target_time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    show["pred_move_pct"]   = show["pred_move_pct"].map(lambda x: f"{x:+.3f}%")
    show["base_close"]      = show["base_close"].map(lambda x: f"{x:,.2f}")
    show["y_pred_end"]      = show["y_pred_end"].map(lambda x: f"{x:,.2f}")

    total = len(show)
    if args.limit and args.limit > 0:
        print(f"Found {total} signals (|move| >= {args.threshold}% within ≤{args.max_minutes} min) — showing first {args.limit}:")
        show = show.head(args.limit)
    else:
        print(f"Found {total} signals (|move| >= {args.threshold}% within ≤{args.max_minutes} min):")

    cols = ["prediction_time","target_time","base_close","y_pred_end","pred_move_pct","direction","trade_duration_min"]
    print(show[cols].to_string(index=False))

if __name__ == "__main__":
    main()

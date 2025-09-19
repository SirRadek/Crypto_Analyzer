from __future__ import annotations

import numpy as np
import pandas as pd


def _triple_barrier_outcomes(
    df: pd.DataFrame,
    periods: int,
    *,
    upper_mult: float,
    lower_mult: float,
) -> pd.DataFrame:
    """Compute triple-barrier decisions and touch metadata.

    The routine evaluates whether the price path first hits the upper or the
    lower barrier before the vertical barrier (``periods`` steps) expires. In
    addition to the classic ``{-1, 0, 1}`` decision it records the type of
    barrier hit (``UP``/``DOWN``/``NO_TOUCH``) and helper indicators that make
    downstream probability estimates straightforward to compute.
    """

    if periods <= 0:
        raise ValueError("Triple-barrier periods must be a positive integer")

    close = df["close"].to_numpy(dtype=np.float32, copy=False)
    high = df["high"].to_numpy(dtype=np.float32, copy=False)
    low = df["low"].to_numpy(dtype=np.float32, copy=False)

    n = len(df)
    decision = pd.Series(pd.NA, index=df.index, dtype="Int8")
    touch_label = pd.Series(pd.NA, index=df.index, dtype="string")
    touch_flag = pd.Series(pd.NA, index=df.index, dtype="Int8")
    up_flag = pd.Series(pd.NA, index=df.index, dtype="Int8")
    down_flag = pd.Series(pd.NA, index=df.index, dtype="Int8")

    for i in range(n):
        horizon_idx = i + periods
        if horizon_idx >= n:
            continue

        entry_price = close[i]
        upper_barrier = entry_price * (1.0 + float(upper_mult))
        lower_barrier = entry_price * (1.0 - float(lower_mult))

        outcome = None
        touch = "NO_TOUCH"
        for step in range(1, periods + 1):
            hi = high[i + step]
            lo = low[i + step]

            if np.isnan(hi) or np.isnan(lo):
                continue

            if hi >= upper_barrier:
                outcome = 1
                touch = "UP"
                break
            if lo <= lower_barrier:
                outcome = -1
                touch = "DOWN"
                break

        if outcome is None:
            final_price = close[horizon_idx]
            if np.isnan(final_price):
                continue
            if final_price > entry_price:
                outcome = 1
            elif final_price < entry_price:
                outcome = -1
            else:
                outcome = 0

        decision.iat[i] = outcome
        touch_label.iat[i] = touch
        touched = 1 if touch != "NO_TOUCH" else 0
        touch_flag.iat[i] = touched
        up_flag.iat[i] = 1 if touch == "UP" else 0
        down_flag.iat[i] = 1 if touch == "DOWN" else 0

    return pd.DataFrame(
        {
            "decision": decision,
            "touch_label": touch_label,
            "touched": touch_flag,
            "touch_up": up_flag,
            "touch_down": down_flag,
        },
        index=df.index,
    )


def _infer_step_minutes(ts: pd.Series) -> int:
    """Infer sampling period in minutes from a timestamp series."""
    if len(ts) < 2:
        return 1
    delta = ts.diff().dropna().median()
    if isinstance(delta, pd.Timedelta):
        step = int(delta.total_seconds() // 60)
        return max(step, 1)
    return 1


def make_targets(
    df: pd.DataFrame,
    horizons_min: list[int] | None = None,
    txn_cost_bps: float = 1.0,
    use_three_class: bool = False,
    *,
    triple_barrier_horizons_min: list[int] | None = None,
    triple_barrier_multipliers: dict[int, float | tuple[float, float]] | None = None,
    triple_barrier_default: float = 0.005,
) -> pd.DataFrame:
    """Create classification targets for future price moves.

    Parameters
    ----------
    df:
        Input dataframe containing at least ``close``, ``high``, ``low`` and
        ``timestamp`` columns.
    horizons_min:
        List of horizons in minutes for which targets will be generated.
        Defaults to ``[120]``.
    txn_cost_bps:
        Transaction cost threshold in basis points used for the
        ``beyond_costs`` classification label.
    use_three_class:
        If ``True`` the ``beyond_costs`` label returns three classes ``-1``,
        ``0`` and ``1``. Otherwise it is a binary ``0/1`` label.
    triple_barrier_horizons_min:
        List of horizons (in minutes) for which triple-barrier labels are
        generated. Defaults to ``[120, 240, 360]``.
    triple_barrier_multipliers:
        Optional mapping from horizon to a single float (symmetric upper and
        lower barrier) or ``(upper, lower)`` tuple specifying barrier widths in
        relative terms. When omitted a default of ``0.5%`` is used.
    """

    if horizons_min is None:
        horizons_min = [120]

    if triple_barrier_horizons_min is None:
        triple_barrier_horizons_min = [120, 240, 360]

    tb_multipliers: dict[int, tuple[float, float]] = {}
    if triple_barrier_multipliers is not None:
        for horizon, mult in triple_barrier_multipliers.items():
            if isinstance(mult, tuple):
                up, down = mult
            else:
                up = down = float(mult)
            tb_multipliers[horizon] = (float(up), float(down))

    df = df.copy(deep=False)
    ts = pd.to_datetime(df["timestamp"])
    step_minutes = _infer_step_minutes(ts)

    for horizon in horizons_min:
        periods = max(1, int(round(horizon / step_minutes)))

        fut_close = df["close"].shift(-periods)
        delta_log = np.log(fut_close / df["close"]).astype(np.float32)
        delta_lin = (fut_close - df["close"]).astype(np.float32)

        df[f"cls_sign_{horizon}m"] = (delta_log > 0).astype(np.int8)

        thresh = df["close"] * (txn_cost_bps / 10_000)
        bc = np.where(delta_lin > thresh, 1, np.where(delta_lin < -thresh, -1, 0))
        if not use_three_class:
            bc = np.where(bc == 1, 1, 0)
        df[f"beyond_costs_{horizon}m"] = bc.astype(np.int8)

    default_tb_multiplier = float(triple_barrier_default)
    for horizon in triple_barrier_horizons_min:
        periods = max(1, int(round(horizon / step_minutes)))
        up_mult, down_mult = tb_multipliers.get(
            horizon, (default_tb_multiplier, default_tb_multiplier)
        )
        outcomes = _triple_barrier_outcomes(
            df,
            periods,
            upper_mult=up_mult,
            lower_mult=down_mult,
        )
        df[f"triple_barrier_{horizon}m"] = outcomes["decision"]
        df[f"triple_barrier_touch_{horizon}m"] = outcomes["touch_label"].astype(
            "string"
        )
        df[f"triple_barrier_touched_{horizon}m"] = outcomes["touched"]
        df[f"triple_barrier_touch_up_{horizon}m"] = outcomes["touch_up"]
        df[f"triple_barrier_touch_down_{horizon}m"] = outcomes["touch_down"]

    return df


def triple_barrier_probability_summary(
    df: pd.DataFrame,
    horizons_min: list[int],
    *,
    prefix: str = "triple_barrier",
) -> pd.DataFrame:
    """Compute probability estimates derived from triple-barrier labels.

    The function expects the dataframe to already contain the touch metadata
    generated by :func:`make_targets`.  For each ``horizon`` it reports the
    fraction of samples where either barrier was touched (which corresponds to
    ``P(|r| ≥ threshold)``) as well as the conditional probability that the
    upward barrier was hit given that a touch occurred.
    """

    records: list[dict[str, float | int]] = []
    for horizon in horizons_min:
        touch_col = f"{prefix}_touched_{horizon}m"
        up_col = f"{prefix}_touch_up_{horizon}m"
        if touch_col not in df.columns or up_col not in df.columns:
            raise KeyError(
                "DataFrame is missing required triple-barrier touch columns: "
                f"{touch_col!r}, {up_col!r}"
            )

        touch = df[touch_col].dropna().astype("int32")
        up = df[up_col].dropna().astype("int32")
        if len(touch) == 0:
            prob_touch = float("nan")
            prob_up_given_touch = float("nan")
        else:
            prob_touch = float(touch.mean())
            touched = touch.sum()
            prob_up_given_touch = float(up.sum() / touched) if touched else float("nan")

        records.append(
            {
                "horizon_min": horizon,
                "prob_touch": prob_touch,
                "prob_up_given_touch": prob_up_given_touch,
            }
        )

    return pd.DataFrame.from_records(records)

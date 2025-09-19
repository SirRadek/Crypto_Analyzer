from __future__ import annotations

import numpy as np
import pandas as pd


def _triple_barrier_labels(
    df: pd.DataFrame,
    periods: int,
    *,
    upper_mult: float,
    lower_mult: float,
) -> pd.Series:
    """Compute triple-barrier labels for the given forward ``periods``.

    The implementation follows the common definition popularised by
    López de Prado where an observation is labelled ``1`` if the price
    first touches the upper barrier, ``-1`` if it touches the lower
    barrier, and otherwise the sign of the return when the vertical
    barrier is reached. Observations without enough look-ahead data keep
    the ``pd.NA`` placeholder so callers can drop them explicitly.
    """

    if periods <= 0:
        raise ValueError("Triple-barrier periods must be a positive integer")

    close = df["close"].to_numpy(dtype=np.float32, copy=False)
    high = df["high"].to_numpy(dtype=np.float32, copy=False)
    low = df["low"].to_numpy(dtype=np.float32, copy=False)

    n = len(df)
    labels = pd.Series(pd.NA, index=df.index, dtype="Int8")

    for i in range(n):
        horizon_idx = i + periods
        if horizon_idx >= n:
            continue

        entry_price = close[i]
        upper_barrier = entry_price * (1.0 + float(upper_mult))
        lower_barrier = entry_price * (1.0 - float(lower_mult))

        decision = None
        for step in range(1, periods + 1):
            hi = high[i + step]
            lo = low[i + step]

            if np.isnan(hi) or np.isnan(lo):
                continue

            if hi >= upper_barrier:
                decision = 1
                break
            if lo <= lower_barrier:
                decision = -1
                break

        if decision is None:
            final_price = close[horizon_idx]
            if np.isnan(final_price):
                continue
            if final_price > entry_price:
                decision = 1
            elif final_price < entry_price:
                decision = -1
            else:
                decision = 0

        labels.iat[i] = decision

    return labels


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
        relative terms. When omitted a default of ``1%`` is used.
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

    default_tb_multiplier = 0.01
    for horizon in triple_barrier_horizons_min:
        periods = max(1, int(round(horizon / step_minutes)))
        up_mult, down_mult = tb_multipliers.get(
            horizon, (default_tb_multiplier, default_tb_multiplier)
        )
        labels = _triple_barrier_labels(
            df,
            periods,
            upper_mult=up_mult,
            lower_mult=down_mult,
        )
        df[f"triple_barrier_{horizon}m"] = labels

    return df

from __future__ import annotations

import numpy as np
import pandas as pd


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
) -> pd.DataFrame:
    """Create regression and classification targets for future price moves.

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
    """

    if horizons_min is None:
        horizons_min = [120]

    df = df.copy(deep=False)
    ts = pd.to_datetime(df["timestamp"])
    step_minutes = _infer_step_minutes(ts)

    for horizon in horizons_min:
        periods = max(1, int(round(horizon / step_minutes)))

        fut_close = df["close"].shift(-periods)
        delta_log = np.log(fut_close / df["close"]).astype(np.float32)
        delta_lin = (fut_close - df["close"]).astype(np.float32)
        df[f"delta_log_{horizon}m"] = delta_log
        df[f"delta_lin_{horizon}m"] = delta_lin

        fut_high = df["high"].shift(-periods + 1).rolling(periods).max()
        fut_low = df["low"].shift(-periods + 1).rolling(periods).min()
        df[f"delta_high_{horizon}m"] = (fut_high - df["close"]).astype(np.float32)
        df[f"delta_low_{horizon}m"] = (fut_low - df["close"]).astype(np.float32)

        df[f"cls_sign_{horizon}m"] = (delta_log > 0).astype(np.int8)

        thresh = df["close"] * (txn_cost_bps / 10_000)
        bc = np.where(delta_lin > thresh, 1, np.where(delta_lin < -thresh, -1, 0))
        if not use_three_class:
            bc = np.where(bc == 1, 1, 0)
        df[f"beyond_costs_{horizon}m"] = bc.astype(np.int8)

    return df

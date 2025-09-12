import numpy as np
import pandas as pd


def run_backtest(df, fee=0.0004):
    """Run a simple long/short backtest based on price forecasts.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``timestamp``, ``p_hat``, ``p_low``, ``p_high``,
        ``target`` and ``last_price``.
    fee : float, optional
        Proportional transaction cost per trade.
    """
    direction = np.where(df["p_hat"] > df["last_price"], 1.0, -1.0)
    ret = (df["target"] - df["last_price"]) / df["last_price"]
    trade_ret = direction * ret - fee * np.abs(direction)
    equity = (1 + trade_ret).cumprod()
    pnl = float(equity.iloc[-1] - 1)
    sharpe = float(trade_ret.mean() / (trade_ret.std() + 1e-9) * np.sqrt(len(trade_ret)))
    return {
        "equity": pd.DataFrame({"timestamp": df["timestamp"], "equity": equity}),
        "metrics": {"pnl": pnl, "sharpe": sharpe},
    }

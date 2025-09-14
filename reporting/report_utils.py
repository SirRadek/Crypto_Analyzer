"""Utility functions for reporting and plotting without seaborn."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_cv_summary(results: Iterable[dict], out_path: Path) -> None:
    """Save cross-validation metrics to CSV."""
    df = pd.DataFrame(list(results))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def plot_roc_pr(y_true, y_score, out_path: Path) -> None:
    """Plot ROC and Precision-Recall curves into a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[0])
    axes[0].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[1])
    axes[1].set_title("Precision-Recall")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_reg_scatter(y_true, y_pred, out_path: Path) -> None:
    """Scatter plot of predictions vs. true values with 45-degree line."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.7)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="grey")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_group_bars(group_df: pd.DataFrame, out_path: Path, value_col: str = "value") -> None:
    """Bar chart for group importances.

    Parameters
    ----------
    group_df: DataFrame with at least columns ["group", value_col].
    value_col: Name of the column to plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(group_df["group"], group_df[value_col])
    ax.set_xlabel(value_col)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

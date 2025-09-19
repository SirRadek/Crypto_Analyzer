from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardSplit:
    """Generate walk-forward time splits without leakage.

    Parameters
    ----------
    train_span_days:
        Number of days used for the training window.
    test_span_days:
        Number of days used for the test window following each training span.
    step_days:
        Number of days the window is moved forward after each fold.
    min_train_days:
        Minimal amount of days required before the first split is produced.
    """

    train_span_days: int
    test_span_days: int
    step_days: int
    min_train_days: int

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train and test indices for each fold.

        The dataframe must contain a ``timestamp`` column with timezone aware
        ``datetime64`` values.  Returned indices refer to the original order of
        ``df``.
        """

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")

        ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)

        start_time = ts.min() + pd.Timedelta(days=self.min_train_days)
        test_start = start_time
        end_time = ts.max()

        while True:
            train_end = test_start
            train_start = train_end - pd.Timedelta(days=self.train_span_days)
            test_end = train_end + pd.Timedelta(days=self.test_span_days)
            if test_end > end_time:
                break

            train_mask = (ts >= train_start) & (ts < train_end)
            test_mask = (ts >= train_end) & (ts < test_end)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                break
            yield train_idx, test_idx

            test_start = test_start + pd.Timedelta(days=self.step_days)


@dataclass
class PurgedWalkForwardSplit:
    """Walk-forward splitter with optional purge and embargo windows."""

    train_span_days: int
    test_span_days: int
    step_days: int
    min_train_days: int
    purge_minutes: int = 0
    embargo_minutes: int = 0

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")

        ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
        step = ts.diff().dropna().median()
        if not isinstance(step, pd.Timedelta) or step <= pd.Timedelta(0):
            step = pd.Timedelta(minutes=1)
        start_time = ts.min() + pd.Timedelta(days=self.min_train_days)
        test_start = start_time
        end_time = ts.max() + step
        embargo_delta = pd.Timedelta(minutes=self.embargo_minutes)
        purge_delta = pd.Timedelta(minutes=self.purge_minutes)

        while True:
            train_end = test_start
            train_start = train_end - pd.Timedelta(days=self.train_span_days)
            test_end = train_end + pd.Timedelta(days=self.test_span_days)
            if test_end > end_time:
                break

            train_mask = (ts >= train_start) & (ts < train_end)
            if self.purge_minutes:
                train_mask &= ts < (train_end - purge_delta)

            test_mask = (ts >= train_end) & (ts < test_end)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                break
            yield train_idx, test_idx

            next_start = test_start + pd.Timedelta(days=self.step_days)
            embargo_start = test_end + embargo_delta
            test_start = max(next_start, embargo_start)

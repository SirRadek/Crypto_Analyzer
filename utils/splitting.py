from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

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

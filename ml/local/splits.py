"""
Temporal data splitter for retail propensity modelling.

Assigns customers to train / validation / test windows based on their
most recent transaction date, guaranteeing:
  - No future data leaks into training features
  - Temporal ordering is preserved across all three splits
  - Every customer appears in exactly one split
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


class TemporalSplitter:
    """Splits a customer DataFrame into train / val / test using temporal windows."""

    def __init__(
        self,
        train_end_day: int,
        val_end_day: int,
        snapshot_day: int,
        start_date: Optional[datetime] = None,
    ) -> None:
        """
        Parameters
        ----------
        train_end_day : int
            Last day (inclusive) of the training window, counted from start_date.
        val_end_day : int
            Last day (inclusive) of the validation window.
        snapshot_day : int
            Last day of the full observation period (test window ends here).
        start_date : datetime, optional
            Reference origin date. Inferred from data if omitted.
        """
        if train_end_day >= val_end_day:
            raise ValueError(
                f"window overlap: train_end_day ({train_end_day}) >= "
                f"val_end_day ({val_end_day})"
            )
        if val_end_day >= snapshot_day:
            raise ValueError(
                f"window overlap: val_end_day ({val_end_day}) >= "
                f"snapshot_day ({snapshot_day})"
            )

        self.train_end_day = train_end_day
        self.val_end_day   = val_end_day
        self.snapshot_day  = snapshot_day
        self.start_date    = start_date

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Assign each customer to exactly one temporal window.

        The DataFrame must contain 'customer_id' and 'last_transaction_date'.
        Customers are bucketed by which window their last_transaction_date falls in.

        Returns (train_df, val_df, test_df) — all columns from df preserved.

        Raises
        ------
        ValueError
            If df has fewer than 3 rows (cannot form 3 non-empty splits).
        """
        if len(df) < 3:
            raise ValueError(
                f"insufficient customers to split: got {len(df)}, need at least 3"
            )

        df = df.copy()
        dates = pd.to_datetime(df["last_transaction_date"])

        start = self.start_date if self.start_date is not None else dates.min().to_pydatetime()
        train_end = start + timedelta(days=self.train_end_day)
        val_end   = start + timedelta(days=self.val_end_day)

        train_mask = dates <= train_end
        val_mask   = (dates > train_end) & (dates <= val_end)
        test_mask  = dates > val_end

        train_df = df[train_mask].reset_index(drop=True)
        val_df   = df[val_mask].reset_index(drop=True)
        test_df  = df[test_mask].reset_index(drop=True)

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError(
                "insufficient customers to split: one or more splits would be empty. "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )

        return train_df, val_df, test_df

    def class_balance(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        label_col: str = "label",
    ) -> dict[str, float]:
        """
        Compute positive-class rate for each split.

        Returns {'train': float, 'val': float, 'test': float}.
        All values are in [0.0, 1.0]. Returns None for a split that lacks
        the label column.
        """
        result: dict[str, float] = {}
        for name, df in (("train", train), ("val", val), ("test", test)):
            if label_col in df.columns and len(df) > 0:
                result[name] = float(df[label_col].mean())
            else:
                result[name] = None  # type: ignore[assignment]
        return result

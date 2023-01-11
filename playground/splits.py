"""Rolling-origin (walk-forward) splitting for time-ordered data.

This module is deliberately free of any modelling dependency: it operates on a
sorted ``DatetimeIndex``-like sequence of timestamps and returns integer
positions.  Keeping it pure is what makes the leakage invariants cheap to test
without fitting anything.

The single invariant every split must satisfy is::

    max(timestamps[train]) < min(timestamps[test])

Any violation of that is look-ahead bias, which is the failure mode that makes
a forecasting backtest silently optimistic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


class LookAheadError(AssertionError):
    """Raised when a train/test split would leak future information."""


@dataclass(frozen=True)
class Split:
    """One walk-forward fold, expressed as positional indices."""

    fold: int
    train: np.ndarray
    test: np.ndarray
    cutoff: pd.Timestamp

    def __post_init__(self) -> None:
        """Reject degenerate folds at construction time."""
        if self.train.size == 0:
            raise ValueError(f"fold {self.fold}: empty training set")
        if self.test.size == 0:
            raise ValueError(f"fold {self.fold}: empty test set")


def assert_no_look_ahead(timestamps: pd.Series, split: Split) -> None:
    """Fail loudly if ``split`` lets the model see data at or after the cutoff.

    Checks three distinct ways a split can leak:

    1. the train and test position sets overlap;
    2. the last training timestamp is not strictly before the first test one;
    3. a training timestamp sits at or after the declared cutoff.
    """
    overlap = np.intersect1d(split.train, split.test)
    if overlap.size:
        raise LookAheadError(
            f"fold {split.fold}: {overlap.size} row(s) appear in both train and test"
        )

    train_end = timestamps.iloc[split.train].max()
    test_start = timestamps.iloc[split.test].min()
    if not train_end < test_start:
        raise LookAheadError(
            f"fold {split.fold}: last train timestamp {train_end} is not strictly "
            f"before first test timestamp {test_start}"
        )

    late_train = timestamps.iloc[split.train] > split.cutoff
    if bool(late_train.any()):
        raise LookAheadError(
            f"fold {split.fold}: {int(late_train.sum())} training row(s) fall after "
            f"the cutoff {split.cutoff}"
        )


def rolling_origin_splits(
    timestamps: pd.Series,
    *,
    horizon: int,
    initial: int,
    period: int | None = None,
    max_folds: int | None = None,
) -> Iterator[Split]:
    """Yield expanding-window walk-forward folds, oldest cutoff first.

    Parameters
    ----------
    timestamps:
        Strictly increasing series of observation timestamps.
    horizon:
        Number of observations to forecast in each fold.
    initial:
        Number of observations in the first training window.
    period:
        Number of observations between consecutive cutoffs.  Defaults to
        ``horizon`` (non-overlapping test windows).
    max_folds:
        Optional cap, applied to the most recent folds.  Useful for keeping a
        Prophet backtest inside a sensible wall-clock budget.

    Raises
    ------
    ValueError
        If the series is not strictly increasing, or is too short for even one
        fold at the requested ``initial``/``horizon``.

    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if initial < 1:
        raise ValueError("initial must be >= 1")
    step = horizon if period is None else period
    if step < 1:
        raise ValueError("period must be >= 1")

    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError(
            "timestamps must be strictly increasing; sort and de-duplicate before "
            "splitting (shuffled or duplicated rows are a common leakage source)"
        )
    n = len(ts)
    if n < initial + horizon:
        raise ValueError(
            f"need at least initial + horizon = {initial + horizon} observations, got {n}"
        )

    cut_positions = list(range(initial, n - horizon + 1, step))
    if max_folds is not None:
        cut_positions = cut_positions[-max_folds:]

    for fold, cut in enumerate(cut_positions):
        split = Split(
            fold=fold,
            train=np.arange(0, cut),
            test=np.arange(cut, cut + horizon),
            cutoff=ts.iloc[cut - 1],
        )
        assert_no_look_ahead(ts, split)
        yield split


def split_frames(
    df: pd.DataFrame, split: Split, *, time_col: str = "ds"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise a fold as ``(train, test)`` frames, re-checking the invariant."""
    assert_no_look_ahead(df[time_col], split)
    train = df.iloc[split.train].reset_index(drop=True)
    test = df.iloc[split.test].reset_index(drop=True)
    return train, test


__all__ = [
    "LookAheadError",
    "Split",
    "assert_no_look_ahead",
    "rolling_origin_splits",
    "split_frames",
]

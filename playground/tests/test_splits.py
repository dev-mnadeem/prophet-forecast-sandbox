"""Structural properties of the walk-forward splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from playground.splits import Split, rolling_origin_splits, split_frames


def test_fold_count_and_shapes(synthetic_daily):
    splits = list(rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=60))
    # cutoffs at 60, 70, ..., 110 -> 6 folds over 120 observations
    assert len(splits) == 6
    assert all(len(s.test) == 10 for s in splits)
    assert [len(s.train) for s in splits] == [60, 70, 80, 90, 100, 110]


def test_training_window_expands_and_cutoffs_advance(synthetic_daily):
    splits = list(rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=60))
    sizes = [len(s.train) for s in splits]
    cutoffs = [s.cutoff for s in splits]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert cutoffs == sorted(cutoffs) and len(set(cutoffs)) == len(cutoffs)


def test_period_controls_cutoff_spacing(synthetic_daily):
    every_fold = list(
        rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=60, period=5)
    )
    assert len(every_fold) == 11
    gaps = {b.cutoff - a.cutoff for a, b in zip(every_fold, every_fold[1:], strict=False)}
    assert gaps == {pd.Timedelta(days=5)}


def test_max_folds_keeps_the_most_recent(synthetic_daily):
    every = list(rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=60))
    capped = list(rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=60, max_folds=2))
    assert len(capped) == 2
    assert [s.cutoff for s in capped] == [s.cutoff for s in every[-2:]]


def test_series_too_short_is_an_error(synthetic_daily):
    with pytest.raises(ValueError, match="at least initial \\+ horizon"):
        list(rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=115))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"horizon": 0, "initial": 60}, "horizon must be"),
        ({"horizon": 10, "initial": 0}, "initial must be"),
        ({"horizon": 10, "initial": 60, "period": 0}, "period must be"),
    ],
)
def test_invalid_arguments_are_rejected(synthetic_daily, kwargs, message):
    with pytest.raises(ValueError, match=message):
        list(rolling_origin_splits(synthetic_daily["ds"], **kwargs))


def test_empty_side_of_a_split_is_an_error(synthetic_daily):
    with pytest.raises(ValueError, match="empty test set"):
        Split(
            fold=0,
            train=np.arange(0, 10),
            test=np.array([], dtype=int),
            cutoff=synthetic_daily["ds"].iloc[9],
        )


def test_split_frames_are_contiguous_and_reindexed(synthetic_daily):
    split = next(rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=60))
    train, test = split_frames(synthetic_daily, split)
    assert list(train.index) == list(range(60))
    assert list(test.index) == list(range(10))
    rejoined = pd.concat([train, test], ignore_index=True)
    assert rejoined.equals(synthetic_daily.iloc[:70].reset_index(drop=True))

"""Look-ahead / data-leakage regression tests.

Leakage is the failure mode that makes a forecasting result look good and be
worthless, and it is silent: nothing errors, the metrics simply improve.  These
tests pin the invariant from four directions, so that a future refactor that
reintroduces leakage breaks the build rather than the credibility of the
numbers in the README.

Each test states the specific leak it guards against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from playground.backtest import backtest
from playground.forecasters import NaiveLastValue
from playground.splits import (
    LookAheadError,
    Split,
    assert_no_look_ahead,
    rolling_origin_splits,
    split_frames,
)


# --------------------------------------------------------------------------
# Leak 1: train and test windows overlap.
# --------------------------------------------------------------------------
def test_overlapping_train_and_test_is_rejected(synthetic_daily):
    ts = synthetic_daily["ds"]
    bad = Split(
        fold=0,
        train=np.arange(0, 60),
        test=np.arange(55, 70),  # 5 rows appear on both sides
        cutoff=ts.iloc[59],
    )
    with pytest.raises(LookAheadError, match="both train and test"):
        assert_no_look_ahead(ts, bad)


# --------------------------------------------------------------------------
# Leak 2: training window reaches past the cutoff (the classic "fit on
# everything, slice the predictions afterwards" mistake).
# --------------------------------------------------------------------------
def test_training_past_the_cutoff_is_rejected(synthetic_daily):
    ts = synthetic_daily["ds"]
    bad = Split(
        fold=3,
        train=np.arange(0, 100),
        test=np.arange(100, 110),
        cutoff=ts.iloc[59],  # declared cutoff far earlier than the train window
    )
    with pytest.raises(LookAheadError, match="after the cutoff"):
        assert_no_look_ahead(ts, bad)


# --------------------------------------------------------------------------
# Leak 3: rows are not time-ordered, so a "random" split mixes future into past.
# --------------------------------------------------------------------------
def test_shuffled_rows_are_rejected(synthetic_daily):
    shuffled = synthetic_daily.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        list(rolling_origin_splits(shuffled["ds"], horizon=10, initial=60))


def test_duplicate_timestamps_are_rejected(synthetic_daily):
    dupes = pd.concat([synthetic_daily, synthetic_daily.iloc[[10]]], ignore_index=True)
    dupes = dupes.sort_values("ds").reset_index(drop=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        list(rolling_origin_splits(dupes["ds"], horizon=10, initial=60))


# --------------------------------------------------------------------------
# Leak 4 (behavioural): the model must never be *handed* a future observation.
# This is the test that would catch a leak introduced inside backtest() itself,
# where the split objects are still perfectly well-formed.
# --------------------------------------------------------------------------
class RecordingForecaster(NaiveLastValue):
    """Records the widest window of data it was ever shown."""

    name = "recording"
    seen: list[dict] = []

    def fit(self, history: pd.DataFrame) -> RecordingForecaster:
        RecordingForecaster.seen.append(
            {
                "max_ds": history["ds"].max(),
                "n": len(history),
                "columns": set(history.columns),
                "y_values": set(np.round(history["y"].to_numpy(), 9)),
            }
        )
        return super().fit(history)


def test_model_is_never_shown_a_row_at_or_after_its_cutoff(synthetic_daily):
    RecordingForecaster.seen = []
    splits = list(rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=60))
    assert len(splits) >= 3, "need several folds for this to be meaningful"

    backtest(
        synthetic_daily,
        RecordingForecaster,
        horizon=10,
        initial=60,
        model_name="recording",
    )

    assert len(RecordingForecaster.seen) == len(splits)
    for split, record in zip(splits, RecordingForecaster.seen, strict=True):
        _, test = split_frames(synthetic_daily, split)
        assert record["max_ds"] < test["ds"].min(), (
            f"fold {split.fold}: model saw data up to {record['max_ds']} but the test "
            f"window starts at {test['ds'].min()}"
        )
        future_values = set(np.round(test["y"].to_numpy(), 9))
        assert not (record["y_values"] & future_values), (
            f"fold {split.fold}: a test-window y value reached the model"
        )


def test_predict_receives_timestamps_only(synthetic_daily):
    """``y`` must not travel into predict(); that is the other half of the seam."""
    captured: list[set[str]] = []

    class ColumnSpy(NaiveLastValue):
        def predict(self, future: pd.DataFrame) -> pd.DataFrame:
            captured.append(set(future.columns))
            return super().predict(future)

    backtest(synthetic_daily, ColumnSpy, horizon=10, initial=60)
    assert captured, "predict() was never called"
    for columns in captured:
        assert columns == {"ds"}, f"predict() was handed extra columns: {columns - {'ds'}}"


# --------------------------------------------------------------------------
# The guard has teeth: a forecaster that cheats really does score better, so
# the invariant above is load-bearing rather than decorative.
# --------------------------------------------------------------------------
def test_a_cheating_forecaster_would_score_better(synthetic_daily):
    """Quantifies what leakage buys, so the guard is not merely theatre.

    A forecaster given the answers scores an MAE of ~0; the honest baseline does
    not.  If the two were close, the leakage checks would not be worth having.
    """

    class Oracle(NaiveLastValue):
        """Deliberately leaky: keeps the whole series and reads the future."""

        def __init__(self, truth: pd.DataFrame) -> None:
            super().__init__()
            self._truth = truth.set_index("ds")["y"]

        def predict(self, future: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"ds": future["ds"], "yhat": self._truth.loc[future["ds"]].to_numpy()}
            )

    honest = backtest(synthetic_daily, NaiveLastValue, horizon=10, initial=60).summary()
    leaky = backtest(
        synthetic_daily, lambda: Oracle(synthetic_daily), horizon=10, initial=60
    ).summary()

    assert leaky["mae"] == pytest.approx(0.0, abs=1e-9)
    assert honest["mae"] > 0.5, "baseline should have real error on a noisy series"

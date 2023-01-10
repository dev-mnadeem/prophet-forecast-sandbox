"""End-to-end behaviour of the harness, using instant baselines only."""

from __future__ import annotations

import pandas as pd
import pytest

from playground.backtest import backtest
from playground.forecasters import NaiveLastValue, SeasonalNaive
from playground.splits import rolling_origin_splits


def test_one_result_per_fold(synthetic_daily):
    expected = len(list(rolling_origin_splits(synthetic_daily["ds"], horizon=10, initial=60)))
    result = backtest(synthetic_daily, NaiveLastValue, horizon=10, initial=60)
    assert len(result.folds) == expected
    assert [f.fold for f in result.folds] == list(range(expected))


def test_every_fold_gets_a_fresh_model(synthetic_daily):
    """A factory, not an instance: state must not carry across cutoffs."""
    built: list[int] = []

    def factory() -> NaiveLastValue:
        built.append(1)
        return NaiveLastValue()

    result = backtest(synthetic_daily, factory, horizon=10, initial=60)
    assert len(built) == len(result.folds)


def test_summary_averages_the_folds(synthetic_daily):
    result = backtest(synthetic_daily, NaiveLastValue, horizon=10, initial=60)
    frame = result.to_frame()
    assert result.summary()["mae"] == pytest.approx(frame["mae"].mean())
    assert result.summary()["folds"] == len(frame)


def test_coverage_is_none_when_the_model_has_no_intervals(synthetic_daily):
    result = backtest(synthetic_daily, NaiveLastValue, horizon=10, initial=60)
    assert all(f.coverage is None for f in result.folds)
    assert result.summary()["coverage"] is None


def test_seasonal_naive_beats_naive_on_a_seasonal_series(exact_weekly):
    kwargs = {"horizon": 7, "initial": 28}
    naive = backtest(exact_weekly, NaiveLastValue, **kwargs).summary()
    seasonal = backtest(exact_weekly, lambda: SeasonalNaive(period=7), **kwargs).summary()
    assert seasonal["mae"] < naive["mae"]
    assert seasonal["mae"] == pytest.approx(0.0, abs=1e-12)


def test_results_are_reproducible_run_to_run(synthetic_daily):
    """Deterministic inputs must give byte-identical scores."""
    a = backtest(synthetic_daily, NaiveLastValue, horizon=10, initial=60).to_frame()
    b = backtest(synthetic_daily, NaiveLastValue, horizon=10, initial=60).to_frame()
    metrics = ["mae", "rmse", "mape", "smape"]
    pd.testing.assert_frame_equal(a[metrics], b[metrics])


def test_map_fn_is_honoured(synthetic_daily):
    """The parallelism seam must actually be used when supplied."""
    calls = {"n": 0}

    def counting_map(fn, items):
        calls["n"] += 1
        return [fn(item) for item in items]

    result = backtest(synthetic_daily, NaiveLastValue, horizon=10, initial=60, map_fn=counting_map)
    assert calls["n"] == 1
    assert len(result.folds) > 1


def test_wrong_length_forecast_is_rejected(synthetic_daily):
    class TruncatingModel(NaiveLastValue):
        def predict(self, future: pd.DataFrame) -> pd.DataFrame:
            return super().predict(future).iloc[:-1]

    with pytest.raises(ValueError, match="rows for .* requested timestamps"):
        backtest(synthetic_daily, TruncatingModel, horizon=10, initial=60)


def test_missing_y_column_is_rejected(synthetic_daily):
    with pytest.raises(ValueError, match=r"missing column\(s\) \['y'\]"):
        backtest(synthetic_daily[["ds"]], NaiveLastValue, horizon=10, initial=60)

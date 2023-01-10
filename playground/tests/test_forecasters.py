"""Baselines and the registry that makes them swappable."""

from __future__ import annotations

import pandas as pd
import pytest

from playground.forecasters import (
    Forecaster,
    NaiveLastValue,
    SeasonalNaive,
    available_forecasters,
    get_forecaster,
    register_forecaster,
)


def test_registry_exposes_the_shipped_models():
    assert {"naive", "seasonal_naive", "prophet"} <= set(available_forecasters())


def test_registered_models_satisfy_the_protocol():
    for key in ("naive", "seasonal_naive"):
        assert isinstance(get_forecaster(key), Forecaster)


def test_unknown_key_lists_the_alternatives():
    with pytest.raises(KeyError, match="available:"):
        get_forecaster("arima")


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_forecaster("naive")(NaiveLastValue)


def test_naive_repeats_the_last_observation(synthetic_daily):
    model = NaiveLastValue().fit(synthetic_daily)
    future = pd.DataFrame({"ds": pd.date_range("2021-05-01", periods=3, freq="D")})
    out = model.predict(future)
    assert out["yhat"].nunique() == 1
    assert out["yhat"].iloc[0] == pytest.approx(synthetic_daily["y"].iloc[-1])


def test_seasonal_naive_is_exact_on_a_noiseless_cycle(exact_weekly):
    train, test = exact_weekly.iloc[:63], exact_weekly.iloc[63:]
    model = SeasonalNaive(period=7).fit(train)
    out = model.predict(test[["ds"]])
    assert out["yhat"].to_numpy() == pytest.approx(test["y"].to_numpy())


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="before fit"):
        NaiveLastValue().predict(pd.DataFrame({"ds": []}))


def test_missing_columns_are_reported(synthetic_daily):
    with pytest.raises(ValueError, match=r"\['y'\]"):
        NaiveLastValue().fit(synthetic_daily[["ds"]])


def test_seasonal_naive_needs_a_full_period(exact_weekly):
    with pytest.raises(ValueError, match="at least period"):
        SeasonalNaive(period=7).fit(exact_weekly.iloc[:5])

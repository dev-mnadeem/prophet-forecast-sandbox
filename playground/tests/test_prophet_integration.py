"""The only tests that really fit Prophet.

Marked ``slow`` and deselected by default (see ``pyproject.toml``), because the
fast suite must stay runnable without a compiled Stan model.  Run them with
``pytest -m slow``.

Each one pins a property of the *vendored library* that the harness relies on,
so an upstream bump that changes the behaviour is caught here rather than in a
README number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from playground.backtest import backtest
from playground.forecasters import ProphetForecaster
from playground.splits import rolling_origin_splits, split_frames

pytestmark = pytest.mark.slow

prophet = pytest.importorskip("prophet", reason="vendored Prophet is not importable")


@pytest.fixture(scope="module")
def series() -> pd.DataFrame:
    """A short daily series with a deliberate spike inside the test window."""
    rng = np.random.default_rng(11)
    n = 180
    ds = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = 20.0 + 0.03 * t + 3.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.4, n)
    y[155] = 500.0  # inside the first fold's test window, far larger than any train value
    return pd.DataFrame({"ds": ds, "y": y})


def test_scaling_constants_come_from_training_data_only(series):
    """Prophet's normalisation is fitted, and must not see the future.

    ``initialize_scales`` sets ``y_scale`` from whatever frame it is handed.
    Fitting on the whole series before splitting — the textbook "scaler fitted
    on all data" leak — would pull the 500.0 spike from the test window into
    ``y_scale``.  This asserts it does not.
    """
    split = next(rolling_origin_splits(series["ds"], horizon=20, initial=150))
    train, test = split_frames(series, split)
    assert test["y"].max() > train["y"].max() * 10, "fixture must put the spike in test"

    model = ProphetForecaster(seed=7).fit(train)
    m = model._model

    assert m.y_scale == pytest.approx(float(train["y"].abs().max()))
    assert m.y_scale < float(series["y"].abs().max())
    assert m.start == train["ds"].min()
    assert m.start + m.t_scale == train["ds"].max()
    assert m.history["ds"].max() < test["ds"].min()


def test_seeded_forecasts_are_reproducible(series):
    """Point forecasts *and* intervals must repeat run to run."""
    future = pd.DataFrame({"ds": pd.date_range("2020-07-01", periods=14, freq="D")})
    a = ProphetForecaster(seed=99).fit(series).predict(future)
    b = ProphetForecaster(seed=99).fit(series).predict(future)
    pd.testing.assert_frame_equal(a, b)


def test_unseeded_prophet_intervals_are_not_reproducible(series):
    """Documents *why* ProphetForecaster seeds the global RNG.

    Prophet samples uncertainty from the global NumPy RNG and seeds it nowhere,
    so ``yhat_lower``/``yhat_upper`` move between runs while ``yhat`` does not.
    If this ever starts passing, upstream has added a seed and the wrapper's
    seeding can be revisited.
    """
    from prophet import Prophet

    future = pd.DataFrame({"ds": pd.date_range("2020-07-01", periods=14, freq="D")})
    fitted = Prophet().fit(series[["ds", "y"]])
    first, second = fitted.predict(future), fitted.predict(future)

    assert np.abs(first["yhat"] - second["yhat"]).max() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(first["yhat_lower"] - second["yhat_lower"]).max() > 1e-6


def test_backtest_runs_prophet_over_several_folds(series):
    result = backtest(
        series,
        lambda: ProphetForecaster(seed=7),
        horizon=20,
        initial=120,
        model_name="prophet",
    )
    summary = result.summary()
    assert summary["folds"] >= 2
    assert summary["mae"] > 0
    assert 0.0 <= summary["coverage"] <= 1.0

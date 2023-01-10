"""First-party exploration layer for the vendored Prophet snapshot.

Everything under ``python/`` in this repository is a verbatim copy of
``facebook/prophet`` and is treated as a read-only third-party dependency.
``playground`` is the only first-party code here: it wraps that library in a
reproducible, leakage-safe backtesting harness so forecasting claims made in
the README can be re-derived from a single command.
"""

from playground.backtest import BacktestResult, FoldResult, backtest
from playground.datasets import DATASETS, load_dataset
from playground.forecasters import (
    Forecaster,
    NaiveLastValue,
    ProphetForecaster,
    SeasonalNaive,
    available_forecasters,
    get_forecaster,
    register_forecaster,
)
from playground.metrics import coverage, mae, mape, rmse, smape
from playground.splits import Split, rolling_origin_splits

__all__ = [
    "BacktestResult",
    "DATASETS",
    "FoldResult",
    "Forecaster",
    "NaiveLastValue",
    "ProphetForecaster",
    "SeasonalNaive",
    "Split",
    "available_forecasters",
    "backtest",
    "coverage",
    "get_forecaster",
    "load_dataset",
    "mae",
    "mape",
    "register_forecaster",
    "rmse",
    "rolling_origin_splits",
    "smape",
]

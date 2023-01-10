"""First-party backtesting layer over Prophet.

``playground`` wraps the ``prophet`` package (the PyPI wheel pinned in
``requirements.txt``) in a reproducible, leakage-safe backtesting harness, so
the forecasting claims made in the README can be re-derived from a single
command.  The copy of ``facebook/prophet`` under ``python/`` is reference
material only and is never imported.
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

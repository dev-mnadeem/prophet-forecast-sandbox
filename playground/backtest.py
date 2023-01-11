"""Rolling-origin backtesting with the look-ahead invariant enforced per fold.

Design rule: **one fresh model per fold, fitted on that fold's training rows
only.** Reusing a model across cutoffs — or fitting once on the full series and
slicing predictions afterwards — is the standard way a forecasting notebook
reports an error that cannot be achieved in production.

The harness is model-agnostic (see :mod:`playground.forecasters`) so Prophet's
score is always reported next to a baseline's on identical folds.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from playground.forecasters import Forecaster
from playground.metrics import coverage, mae, mape, rmse, smape
from playground.splits import Split, rolling_origin_splits, split_frames


@dataclass(frozen=True)
class FoldResult:
    """Scores for a single walk-forward fold."""

    fold: int
    cutoff: pd.Timestamp
    n_train: int
    n_test: int
    mae: float
    rmse: float
    mape: float
    smape: float
    coverage: float | None
    fit_seconds: float

    def as_row(self) -> dict:
        """Flatten to a plain dict, one row of the per-fold table."""
        return {
            "fold": self.fold,
            "cutoff": self.cutoff,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "smape": self.smape,
            "coverage": self.coverage,
            "fit_seconds": self.fit_seconds,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Per-fold scores plus their aggregate for one model on one dataset."""

    model: str
    dataset: str
    horizon: int
    folds: list[FoldResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Per-fold scores as a DataFrame, one row per fold."""
        return pd.DataFrame([f.as_row() for f in self.folds])

    def summary(self) -> dict:
        """Mean of each metric across folds.

        Folds are equal-sized by construction, so an unweighted mean is the
        right aggregate; that is why ``horizon`` is fixed rather than ragged.
        """
        frame = self.to_frame()
        cov = frame["coverage"]
        return {
            "model": self.model,
            "dataset": self.dataset,
            "horizon": self.horizon,
            "folds": len(self.folds),
            "mae": float(frame["mae"].mean()),
            "rmse": float(frame["rmse"].mean()),
            "mape": float(frame["mape"].mean()),
            "smape": float(frame["smape"].mean()),
            "coverage": None if cov.isna().all() else float(cov.mean()),
            "fit_seconds": float(frame["fit_seconds"].sum()),
        }


def _score_fold(
    fold_index: int,
    split: Split,
    df: pd.DataFrame,
    make_model: Callable[[], Forecaster],
) -> FoldResult:
    train, test = split_frames(df, split)

    started = time.perf_counter()
    model = make_model()
    model.fit(train)
    # Only timestamps cross into predict(); the actuals stay here.
    forecast = model.predict(test[["ds"]])
    elapsed = time.perf_counter() - started

    if len(forecast) != len(test):
        raise ValueError(
            f"fold {fold_index}: model returned {len(forecast)} rows for {len(test)} "
            "requested timestamps"
        )

    y_true = test["y"].to_numpy()
    y_hat = forecast["yhat"].to_numpy()
    has_interval = {"yhat_lower", "yhat_upper"} <= set(forecast.columns)

    return FoldResult(
        fold=fold_index,
        cutoff=split.cutoff,
        n_train=len(train),
        n_test=len(test),
        mae=mae(y_true, y_hat),
        rmse=rmse(y_true, y_hat),
        mape=mape(y_true, y_hat),
        smape=smape(y_true, y_hat),
        coverage=(
            coverage(y_true, forecast["yhat_lower"], forecast["yhat_upper"])
            if has_interval
            else None
        ),
        fit_seconds=elapsed,
    )


def backtest(
    df: pd.DataFrame,
    make_model: Callable[[], Forecaster],
    *,
    horizon: int,
    initial: int,
    period: int | None = None,
    max_folds: int | None = None,
    model_name: str = "model",
    dataset_name: str = "dataset",
    map_fn: Callable[[Callable, Iterable], Iterable] | None = None,
) -> BacktestResult:
    """Walk-forward backtest of ``make_model`` over ``df``.

    Parameters
    ----------
    df:
        Tidy ``ds``/``y`` frame, strictly increasing in ``ds``.
    make_model:
        Zero-argument factory.  A *factory*, not an instance: every fold gets a
        model that has never seen another fold's data.
    horizon, initial, period, max_folds:
        Passed to :func:`playground.splits.rolling_origin_splits`.
    model_name, dataset_name:
        Labels carried through to the result, for the summary table only.
    map_fn:
        Injection point for parallelism, defaulting to the builtin ``map``.
        Folds are independent, so a caller with spare cores can pass e.g.
        ``ProcessPoolExecutor(max_workers=4).map``.  Kept as an argument rather
        than baked in because a Prophet fit is already multi-threaded and
        over-subscribing cores makes it slower, not faster.

    Raises
    ------
    playground.splits.LookAheadError
        If any fold would show the model data at or after its cutoff.

    """
    required = {"ds", "y"}
    if not required <= set(df.columns):
        raise ValueError(f"df is missing column(s) {sorted(required - set(df.columns))}")

    frame = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    splits: Sequence[Split] = list(
        rolling_origin_splits(
            frame["ds"], horizon=horizon, initial=initial, period=period, max_folds=max_folds
        )
    )

    runner = map_fn or map
    folds = list(
        runner(
            lambda item: _score_fold(item[0], item[1], frame, make_model),
            list(enumerate(splits)),
        )
    )

    return BacktestResult(model=model_name, dataset=dataset_name, horizon=horizon, folds=folds)


__all__ = ["BacktestResult", "FoldResult", "backtest"]

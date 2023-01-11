"""The extension seam: pluggable forecasters behind one narrow protocol.

A Prophet score means nothing on its own.  The only way to know whether Prophet
earned its complexity on a series is to run the *same* backtest against a
trivial baseline, so the harness is written against a protocol rather than
against Prophet, and baselines ship alongside the real model.

Adding a model is one class plus one ``@register_forecaster`` line; nothing in
:mod:`playground.backtest` changes.

The protocol is deliberately two methods with narrow signatures:

* ``fit(history)`` receives a training frame and nothing else;
* ``predict(future)`` receives future *timestamps* and nothing else.

A forecaster therefore has no channel through which future ``y`` values could
reach it.  Structural prevention beats a downstream check.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


def silence_stan_logging() -> None:
    """Quiet the two INFO lines cmdstanpy emits per fit.

    cmdstanpy configures its logger *lazily*: the first call into
    ``CmdStanModel.optimize`` sees a handler-less ``cmdstanpy`` logger and
    resets it to INFO, discarding any level set beforehand.  Attaching a handler
    first is what makes the level stick, so this cannot simply be a
    ``setLevel`` call.
    """
    for name in ("cmdstanpy", "prophet"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)


@runtime_checkable
class Forecaster(Protocol):
    """Minimal interface the backtest harness depends on."""

    name: str

    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit on a ``ds``/``y`` frame containing training rows only."""

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast for a ``ds``-only frame.

        Returns a frame with ``ds``, ``yhat`` and, where the model supports
        them, ``yhat_lower``/``yhat_upper``.
        """


_REGISTRY: dict[str, Callable[..., Forecaster]] = {}


_Factory = Callable[..., Forecaster]


def register_forecaster(key: str) -> Callable[[_Factory], _Factory]:
    """Register a forecaster factory under ``key`` for CLI and config lookup."""

    def decorator(factory: Callable[..., Forecaster]) -> Callable[..., Forecaster]:
        if key in _REGISTRY:
            raise ValueError(f"forecaster {key!r} is already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_forecasters() -> list[str]:
    """Registered forecaster keys, sorted."""
    return sorted(_REGISTRY)


def get_forecaster(key: str, **kwargs) -> Forecaster:
    """Build a registered forecaster by key."""
    try:
        factory = _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown forecaster {key!r}; available: {', '.join(available_forecasters())}"
        ) from None
    return factory(**kwargs)


def _require_columns(df: pd.DataFrame, columns: set[str], who: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{who}: frame is missing column(s) {sorted(missing)}")


@register_forecaster("naive")
class NaiveLastValue:
    """Carry the last observed value forward.

    The floor any seasonal model must clear.  Needs no fitting, so the fast test
    suite can exercise the whole harness end to end in milliseconds.
    """

    name = "naive"

    def __init__(self) -> None:
        """Create an unfitted baseline."""
        self._last: float | None = None

    def fit(self, history: pd.DataFrame) -> NaiveLastValue:
        """Record the last observed value."""
        _require_columns(history, {"ds", "y"}, type(self).__name__)
        if history.empty:
            raise ValueError("cannot fit on an empty history")
        self._last = float(history.sort_values("ds")["y"].iloc[-1])
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Repeat the recorded value for every requested timestamp."""
        if self._last is None:
            raise RuntimeError("predict() called before fit()")
        _require_columns(future, {"ds"}, type(self).__name__)
        return pd.DataFrame({"ds": future["ds"].to_numpy(), "yhat": self._last})


@register_forecaster("seasonal_naive")
class SeasonalNaive:
    """Repeat the value observed one seasonal period earlier.

    On daily data with ``period=7`` this is "same weekday last week", which is a
    surprisingly strong baseline and the honest thing to compare Prophet with.
    """

    name = "seasonal_naive"

    def __init__(self, period: int = 7) -> None:
        """Create an unfitted baseline with a seasonal cycle of ``period``."""
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> SeasonalNaive:
        """Keep the history; the last full cycle is what gets replayed."""
        _require_columns(history, {"ds", "y"}, type(self).__name__)
        if len(history) < self.period:
            raise ValueError(f"need at least period={self.period} observations, got {len(history)}")
        self._history = history.sort_values("ds").reset_index(drop=True)
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Replay the final seasonal cycle across the requested timestamps."""
        if self._history is None:
            raise RuntimeError("predict() called before fit()")
        _require_columns(future, {"ds"}, type(self).__name__)
        tail = self._history["y"].to_numpy()[-self.period :]
        steps = np.arange(len(future))
        return pd.DataFrame({"ds": future["ds"].to_numpy(), "yhat": tail[steps % self.period]})


@register_forecaster("prophet")
class ProphetForecaster:
    """Thin, deterministic adapter over the vendored Prophet library.

    Two things this adds over calling ``Prophet()`` directly:

    * **A seed.** Prophet draws uncertainty samples from the *global* NumPy RNG
      and seeds it nowhere, so ``yhat_lower``/``yhat_upper`` differ between runs
      on identical inputs (measured at up to 0.15 on the ``peyton`` series,
      against a mean interval width of 1.24).  Point forecasts are unaffected
      because the fit is MAP, not sampled.  Seeding immediately before each fit
      makes intervals reproducible.
    * **Silence.** cmdstanpy logs two INFO lines per fit, which is unreadable
      across a multi-fold backtest.
    """

    name = "prophet"

    def __init__(self, seed: int = 1234, **prophet_kwargs) -> None:
        """Create an unfitted adapter; ``prophet_kwargs`` reach ``Prophet()``."""
        self.seed = seed
        self.prophet_kwargs = prophet_kwargs
        self._model = None

    def fit(self, history: pd.DataFrame) -> ProphetForecaster:
        """Seed the global RNG, then fit a fresh Prophet on ``history``."""
        _require_columns(history, {"ds", "y"}, type(self).__name__)
        from prophet import Prophet

        silence_stan_logging()

        # Seed the global RNG Prophet samples from, so intervals reproduce.
        np.random.seed(self.seed)
        self._model = Prophet(**self.prophet_kwargs)
        self._model.fit(history[["ds", "y"]])
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast the requested timestamps, with reproducible intervals."""
        if self._model is None:
            raise RuntimeError("predict() called before fit()")
        _require_columns(future, {"ds"}, type(self).__name__)
        np.random.seed(self.seed)
        out = self._model.predict(future[["ds"]])
        return out[["ds", "yhat", "yhat_lower", "yhat_upper"]]


__all__ = [
    "Forecaster",
    "silence_stan_logging",
    "NaiveLastValue",
    "ProphetForecaster",
    "SeasonalNaive",
    "available_forecasters",
    "get_forecaster",
    "register_forecaster",
]

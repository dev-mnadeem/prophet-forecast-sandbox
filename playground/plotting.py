"""Picture of one walk-forward fold: what each model forecast against what happened.

The backtest reports numbers; this draws the most recent fold of the same
splitter so a reader can see *why* the numbers differ.  It uses the
object-oriented Matplotlib API (no ``pyplot`` global state), so it is safe to
call from tests and renders headless without a display.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd
from matplotlib.figure import Figure

from playground.forecasters import Forecaster
from playground.splits import rolling_origin_splits, split_frames

# One fixed colour per shipped model, chosen to stay distinguishable in
# greyscale and for the common colour-vision deficiencies.
MODEL_STYLES: dict[str, dict] = {
    "naive": {"color": "#7a7a7a", "linestyle": "--"},
    "seasonal_naive": {"color": "#c47a1c", "linestyle": ":"},
    "prophet": {"color": "#0f6e8c", "linestyle": "-"},
}
_FALLBACK_STYLE = {"color": "#5b3f8c", "linestyle": "-."}


def plot_last_fold(
    df: pd.DataFrame,
    factories: Mapping[str, Callable[[], Forecaster]],
    *,
    horizon: int,
    initial: int,
    context: int,
    title: str,
) -> Figure:
    """Fit every model on the last fold's training rows and plot its forecast.

    ``context`` is how many training observations to show before the cutoff;
    the models still train on the whole training window.
    """
    if context < 1:
        raise ValueError("context must be >= 1")
    if not factories:
        raise ValueError("need at least one model to plot")

    frame = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    (split,) = rolling_origin_splits(frame["ds"], horizon=horizon, initial=initial, max_folds=1)
    train, test = split_frames(frame, split)

    fig = Figure(figsize=(11, 5.2), layout="constrained")
    ax = fig.add_subplot()
    shown = train.tail(context)
    ax.plot(shown["ds"], shown["y"], color="#222222", linewidth=1.1, label="history")
    ax.plot(test["ds"], test["y"], color="#222222", linewidth=2.0, label="actual")
    ax.axvline(split.cutoff, color="#999999", linewidth=0.9)
    ax.annotate(
        "cutoff",
        xy=(split.cutoff, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -14),
        textcoords="offset points",
        color="#555555",
        fontsize=9,
    )

    for name, make_model in factories.items():
        forecast = make_model().fit(train).predict(test[["ds"]])
        style = MODEL_STYLES.get(name, _FALLBACK_STYLE)
        ax.plot(forecast["ds"], forecast["yhat"], linewidth=1.8, label=name, **style)
        if {"yhat_lower", "yhat_upper"} <= set(forecast.columns):
            ax.fill_between(
                forecast["ds"],
                forecast["yhat_lower"],
                forecast["yhat_upper"],
                color=style["color"],
                alpha=0.15,
                linewidth=0,
                label=f"{name} interval",
            )

    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=2)
    return fig


__all__ = ["MODEL_STYLES", "plot_last_fold"]

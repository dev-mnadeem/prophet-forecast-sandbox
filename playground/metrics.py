"""Point and interval forecast error metrics.

Pure NumPy, no modelling dependency, so the numbers quoted in the README can be
checked by hand against small arrays.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Unlike the annotations in this file, this alias is evaluated at runtime, so it
# is the one place here that constrains the minimum Python version (3.10).
ArrayLike = np.ndarray | Sequence[float]


def _prepare(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: y_true {a.shape} vs y_pred {b.shape}")
    if a.size == 0:
        raise ValueError("cannot score an empty forecast")
    return a, b


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    a, b = _prepare(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    a, b = _prepare(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error, as a fraction (not a percentage).

    Undefined where ``y_true`` is zero; those points are dropped rather than
    silently turned into infinities.
    """
    a, b = _prepare(y_true, y_pred)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric MAPE as a fraction, bounded in ``[0, 1]``.

    Uses the bounded definition ``|a - b| / (|a| + |b|)`` rather than the
    variant that halves the denominator and therefore ranges over ``[0, 2]``.
    Bounding it at 1 keeps it directly comparable with :func:`mape`, which also
    returns a fraction; doubling it recovers the ``[0, 2]`` convention.

    Preferred over :func:`mape` for series that approach zero, where MAPE
    explodes and makes a model look arbitrarily bad.
    """
    a, b = _prepare(y_true, y_pred)
    denom = np.abs(a) + np.abs(b)
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(a[mask] - b[mask]) / denom[mask]))


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Fraction of actuals falling inside ``[lower, upper]``.

    A well-calibrated 80% interval should score near 0.80; a much higher value
    means the intervals are uninformatively wide.
    """
    a, lo = _prepare(y_true, lower)
    _, hi = _prepare(y_true, upper)
    return float(np.mean((a >= lo) & (a <= hi)))


__all__ = ["coverage", "mae", "mape", "rmse", "smape"]

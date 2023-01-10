"""Shared fixtures.

Everything here is tiny and synthetic: the fast suite must not need cmdstan, a
GPU, or the network.  The one test that really fits Prophet is marked ``slow``
and deselected by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SEED = 20240924


@pytest.fixture
def synthetic_daily() -> pd.DataFrame:
    """120 days of trend + weekly seasonality + seeded noise."""
    rng = np.random.default_rng(SEED)
    n = 120
    ds = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = 10.0 + 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.25, n)
    return pd.DataFrame({"ds": ds, "y": y})


@pytest.fixture
def exact_weekly() -> pd.DataFrame:
    """A noiseless weekly cycle, so seasonal-naive is exactly right."""
    n = 70
    ds = pd.date_range("2022-03-01", periods=n, freq="D")
    y = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], n // 7)
    return pd.DataFrame({"ds": ds, "y": y})

"""Metric values worked out by hand.

A metric test that recomputes the implementation proves nothing, so every
expected value here is derived independently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from playground.metrics import coverage, mae, mape, rmse, smape


def test_mae_known_value():
    assert mae([1.0, 2.0, 3.0], [1.0, 4.0, 0.0]) == pytest.approx(5.0 / 3.0)


def test_rmse_known_value():
    # errors 0, 3, 4 -> sqrt(25/3)
    assert rmse([1.0, 2.0, 3.0], [1.0, 5.0, 7.0]) == pytest.approx(math.sqrt(25 / 3))


def test_rmse_is_never_below_mae():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=50), rng.normal(size=50)
    assert rmse(a, b) >= mae(a, b) - 1e-12


def test_mape_is_a_fraction_not_a_percentage():
    assert mape([100.0, 200.0], [110.0, 180.0]) == pytest.approx(0.1)


def test_mape_drops_zero_actuals_instead_of_returning_inf():
    assert mape([0.0, 100.0], [5.0, 110.0]) == pytest.approx(0.1)
    assert math.isnan(mape([0.0, 0.0], [1.0, 2.0]))


def test_smape_is_bounded_by_one():
    # Worst case: equal magnitude, opposite sign -> |1 - (-1)| / (1 + 1) == 1.
    assert smape([1.0], [-1.0]) == pytest.approx(1.0)
    assert smape([4.0], [4.0]) == pytest.approx(0.0)
    assert 0.0 <= smape([1.0, 2.0], [5.0, 0.1]) <= 1.0


def test_smape_known_value():
    # |10 - 12| / (10 + 12) == 1/11 ; |5 - 4| / (5 + 4) == 1/9
    assert smape([10.0, 5.0], [12.0, 4.0]) == pytest.approx((1 / 11 + 1 / 9) / 2)


def test_smape_stays_finite_where_mape_explodes():
    assert math.isinf(1 / 0.0001) is False  # sanity
    near_zero = [1e-9, 1.0]
    assert mape(near_zero, [1.0, 1.0]) > 1e6
    assert smape(near_zero, [1.0, 1.0]) <= 1.0


def test_coverage_counts_inclusive_bounds():
    assert coverage([1.0, 5.0, 9.0], [0.0, 6.0, 9.0], [2.0, 8.0, 10.0]) == pytest.approx(2 / 3)


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shape mismatch"):
        mae([1.0, 2.0], [1.0])


def test_empty_forecast_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        rmse([], [])

"""Tests for the SAA newsvendor solver against the closed-form critical fractile.

The newsvendor math is one of the two pieces of real logic in the scaffold;
a silent bug here corrupts every result the project will ever report
(design doc §8, 'silent evaluation bugs').
"""

import numpy as np
import pytest
from scipy import stats

from demand_vae.opt.newsvendor import (
    aggregate_horizon_demand,
    critical_fractile,
    realized_cost,
    solve_newsvendor_saa,
)

COST_RATIOS = [(3.0, 1.0), (1.0, 1.0), (1.0, 3.0)]  # pre-registered ratios (design doc §7)


class TestCriticalFractile:
    def test_known_values(self):
        assert critical_fractile(3, 1) == pytest.approx(0.75)
        assert critical_fractile(1, 1) == pytest.approx(0.50)
        assert critical_fractile(1, 3) == pytest.approx(0.25)

    def test_rejects_nonpositive_costs(self):
        with pytest.raises(ValueError):
            critical_fractile(0, 1)
        with pytest.raises(ValueError):
            critical_fractile(1, -2)


class TestSolveNewsvendorSAA:
    @pytest.mark.parametrize("c_u,c_o", COST_RATIOS)
    def test_converges_to_gaussian_closed_form(self, c_u, c_o):
        """With many i.i.d. Normal scenarios, the SAA order approaches
        Q* = F^{-1}(c_u/(c_u+c_o)) of the true distribution."""
        mu, sigma, n = 100.0, 15.0, 200_000
        rng = np.random.default_rng(42)
        scenarios = rng.normal(mu, sigma, size=n)
        q_saa = solve_newsvendor_saa(scenarios, c_u, c_o)
        q_closed = stats.norm.ppf(critical_fractile(c_u, c_o), loc=mu, scale=sigma)
        assert q_saa == pytest.approx(q_closed, rel=0.01)

    @pytest.mark.parametrize("c_u,c_o", COST_RATIOS)
    def test_converges_to_poisson_closed_form(self, c_u, c_o):
        """Same convergence on a discrete (count) distribution: the SAA order
        matches the Poisson quantile exactly for large samples."""
        rate, n = 20.0, 200_000
        rng = np.random.default_rng(7)
        scenarios = rng.poisson(rate, size=n).astype(float)
        q_saa = solve_newsvendor_saa(scenarios, c_u, c_o)
        q_closed = stats.poisson.ppf(critical_fractile(c_u, c_o), mu=rate)
        assert q_saa == q_closed

    def test_exact_order_statistic_on_small_sample(self):
        """On {1..10} at tau=0.75 the SAA optimum is the ceil(10*0.75)=8th
        order statistic."""
        scenarios = np.arange(1.0, 11.0)
        assert solve_newsvendor_saa(scenarios, 3, 1) == 8.0

    def test_monotone_in_cost_ratio(self):
        """Higher underage cost -> higher fractile -> larger order."""
        rng = np.random.default_rng(0)
        scenarios = rng.gamma(2.0, 10.0, size=10_000)
        ratios = [(1, 3), (1, 1), (3, 1)]
        orders = [solve_newsvendor_saa(scenarios, c_u, c_o) for c_u, c_o in ratios]
        assert orders[0] <= orders[1] <= orders[2]

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            solve_newsvendor_saa(np.array([]), 1, 1)
        with pytest.raises(ValueError):
            solve_newsvendor_saa(np.ones((5, 2)), 1, 1)


class TestRealizedCost:
    def test_perfect_order_costs_nothing(self):
        assert realized_cost(50.0, 50.0, 3, 1) == 0.0

    def test_underage_arithmetic(self):
        # D=60, Q=50: 10 units short at c_u=3 -> 30
        assert realized_cost(50.0, 60.0, 3, 1) == pytest.approx(30.0)

    def test_overage_arithmetic(self):
        # D=40, Q=50: 10 units over at c_o=1 -> 10
        assert realized_cost(50.0, 40.0, 3, 1) == pytest.approx(10.0)

    def test_rejects_nonpositive_costs(self):
        with pytest.raises(ValueError):
            realized_cost(50.0, 40.0, 1, 0)


class TestAggregateHorizonDemand:
    def test_row_sums(self):
        windows = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 5.0]])
        np.testing.assert_allclose(aggregate_horizon_demand(windows), [10.0, 10.0])

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            aggregate_horizon_demand(np.ones(5))

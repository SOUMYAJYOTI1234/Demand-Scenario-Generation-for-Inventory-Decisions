"""Tests for the classical baselines and the two-level evaluation metrics.

Fitted-likelihood correctness (MoM moments), metric correctness against
closed forms / brute force, and the sampler interface contract — the pieces
whose silent failure would corrupt the whole comparison.
"""

import numpy as np
import pytest

from demand_vae.baselines.classical import (
    EmpiricalResampler,
    GaussianSampler,
    NegativeBinomialSampler,
    PoissonSampler,
    nb_params_from_moments,
)
from demand_vae.data.pipeline import aggregate_weekly, temporal_split
from demand_vae.eval.metrics import (
    crps_from_samples,
    interval_coverage,
    pit_from_samples,
)
from demand_vae.opt.newsvendor import (
    realized_cost,
    solve_newsvendor_saa,
    solve_newsvendor_saa_batch,
)

ALL_SAMPLERS = [EmpiricalResampler, GaussianSampler, PoissonSampler, NegativeBinomialSampler]
H = 4


@pytest.fixture(scope="module")
def fitted(synthetic_raw):
    """Panel + train span + one fitted instance of each baseline."""
    from pathlib import Path

    from demand_vae.config import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    config.data.split.train_end = "2011-04-22"
    config.data.split.val_end = "2011-06-17"
    panel = aggregate_weekly(synthetic_raw, config)
    spans = temporal_split(panel, config)
    samplers = {
        cls.__name__: cls(horizon=H, seed=0).fit(panel, spans["train"]) for cls in ALL_SAMPLERS
    }
    return panel, spans, samplers


class TestNBMethodOfMoments:
    def test_params_reproduce_moments(self):
        mean, var = np.array([5.0, 12.0]), np.array([15.0, 40.0])
        r, p = nb_params_from_moments(mean, var)
        np.testing.assert_allclose(r * (1 - p) / p, mean)  # NB mean
        np.testing.assert_allclose(r * (1 - p) / p**2, var)  # NB variance

    def test_sampled_moments_match_fit(self):
        rng = np.random.default_rng(3)
        data = rng.negative_binomial(5, 0.4, size=100_000).astype(float)
        r, p = nb_params_from_moments(data.mean(), data.var())
        draws = np.random.default_rng(4).negative_binomial(r, p, size=200_000)
        assert draws.mean() == pytest.approx(data.mean(), rel=0.02)
        assert draws.var() == pytest.approx(data.var(), rel=0.05)

    def test_underdispersed_degrades_to_poisson(self):
        r, p = nb_params_from_moments(np.array([10.0]), np.array([8.0]))  # var < mean
        draws = np.random.default_rng(0).negative_binomial(r, p, size=100_000)
        assert draws.mean() == pytest.approx(10.0, rel=0.02)
        assert draws.var() == pytest.approx(10.0, rel=0.05)  # ~Poisson: var = mean

    def test_zero_mean_series_sample_zeros(self):
        r, p = nb_params_from_moments(np.array([0.0]), np.array([0.0]))
        draws = np.random.default_rng(0).negative_binomial(r, p, size=1000)
        assert (draws == 0).all()


class TestSamplerContract:
    @pytest.mark.parametrize("cls", ALL_SAMPLERS)
    def test_output_shapes_and_nonnegativity(self, fitted, cls):
        _, _, samplers = fitted
        sampler = samplers[cls.__name__]
        single = sampler.sample({"series_row": 0}, 16)
        assert single.shape == (16, H)
        batch = sampler.sample_batch({"series_row": np.array([0, 1, 2])}, 16)
        assert batch.shape == (3, 16, H)
        assert (single >= 0).all() and (batch >= 0).all()

    @pytest.mark.parametrize("cls", ALL_SAMPLERS)
    def test_requires_fit(self, cls):
        with pytest.raises(RuntimeError):
            cls(horizon=H).sample({"series_row": 0}, 4)

    def test_empirical_values_come_from_train_history(self, fitted):
        panel, spans, samplers = fitted
        lo, hi = spans["train"]
        sampler = samplers["EmpiricalResampler"]
        for row in range(panel.n_series):
            train_vals = set(panel.sales[row, lo : hi + 1][panel.available[row, lo : hi + 1]])
            draws = sampler.sample({"series_row": row}, 200)
            assert set(np.unique(draws)) <= train_vals

    def test_fitted_stats_use_train_span_only(self, fitted):
        panel, spans, samplers = fitted
        lo, hi = spans["train"]
        row = 0
        mask = panel.available[row, lo : hi + 1]
        expected_mean = panel.sales[row, lo : hi + 1][mask].mean()
        assert samplers["GaussianSampler"].mean_[row] == pytest.approx(expected_mean)
        assert samplers["PoissonSampler"].rate_[row] == pytest.approx(expected_mean)
        assert samplers["NegativeBinomialSampler"].mean_[row] == pytest.approx(expected_mean)


class TestCRPS:
    def test_zero_for_perfect_point_mass(self):
        samples = np.full((3, 100), 7.0)
        actual = np.array([7.0, 7.0, 7.0])
        np.testing.assert_allclose(crps_from_samples(samples, actual), 0.0, atol=1e-12)

    def test_matches_bruteforce_pair_sum(self):
        rng = np.random.default_rng(1)
        samples = rng.gamma(2.0, 5.0, size=(4, 61))
        actual = rng.gamma(2.0, 5.0, size=4)
        fast = crps_from_samples(samples, actual)
        for i in range(4):
            s = samples[i]
            brute = np.abs(s - actual[i]).mean() - 0.5 * np.abs(s[:, None] - s[None, :]).mean()
            assert fast[i] == pytest.approx(brute, rel=1e-10)

    def test_degenerate_forecast_penalized_by_distance(self):
        # Point mass at q scored against actual y: CRPS = |q - y|.
        samples = np.full((1, 50), 10.0)
        assert crps_from_samples(samples, np.array([14.0]))[0] == pytest.approx(4.0)


class TestPITAndCoverage:
    def test_pit_in_unit_interval(self):
        rng = np.random.default_rng(0)
        samples = rng.poisson(5.0, size=(500, 200))
        actual = rng.poisson(5.0, size=500).astype(float)
        pit = pit_from_samples(samples, actual, np.random.default_rng(1))
        assert ((pit >= 0) & (pit < 1)).all()

    def test_pit_uniform_and_coverage_nominal_for_true_model(self):
        # Forecast distribution == data-generating distribution -> PIT ~ U[0,1)
        # and central intervals cover at their nominal rate.
        rng = np.random.default_rng(2)
        n, s = 4000, 500
        samples = rng.normal(0.0, 1.0, size=(n, s))
        actual = rng.normal(0.0, 1.0, size=n)
        pit = pit_from_samples(samples, actual, np.random.default_rng(3))
        assert pit.mean() == pytest.approx(0.5, abs=0.02)
        assert interval_coverage(samples, actual, 0.5) == pytest.approx(0.5, abs=0.04)
        assert interval_coverage(samples, actual, 0.9) == pytest.approx(0.9, abs=0.03)

    def test_overconfident_forecast_undercovers(self):
        # Too-narrow forecast (the Poisson failure mode) must show under-coverage.
        rng = np.random.default_rng(4)
        samples = rng.normal(0.0, 0.3, size=(3000, 300))  # forecast std 0.3
        actual = rng.normal(0.0, 1.0, size=3000)  # truth std 1.0
        assert interval_coverage(samples, actual, 0.9) < 0.55


class TestNewsvendorBatchAndManual:
    def test_two_scenario_manual_example(self):
        # Scenario totals {10, 20}, c_u = c_o = 1 -> tau = 0.5 -> Q is the
        # ceil(2 * 0.5) = 1st order statistic = 10. Actual demand 15:
        # cost = 1 * (15 - 10)^+ + 1 * (10 - 15)^+ = 5.
        scenarios = np.array([10.0, 20.0])
        order = solve_newsvendor_saa(scenarios, 1, 1)
        assert order == 10.0
        assert realized_cost(order, 15.0, 1, 1) == pytest.approx(5.0)
        # At 3:1 -> tau = 0.75 -> 2nd order statistic = 20; cost = 1 * 5 = 5.
        order_hi = solve_newsvendor_saa(scenarios, 3, 1)
        assert order_hi == 20.0
        assert realized_cost(order_hi, 15.0, 3, 1) == pytest.approx(5.0)

    @pytest.mark.parametrize("c_u,c_o", [(3, 1), (1, 1), (1, 3)])
    def test_batch_solver_matches_scalar(self, c_u, c_o):
        rng = np.random.default_rng(5)
        totals = rng.gamma(2.0, 10.0, size=(50, 137))
        batch = solve_newsvendor_saa_batch(totals, c_u, c_o)
        scalar = np.array([solve_newsvendor_saa(row, c_u, c_o) for row in totals])
        np.testing.assert_array_equal(batch, scalar)

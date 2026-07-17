"""Tests for Phase 8: paired bootstrap, rank agreement, marginal protocol, plots."""

import numpy as np
import pytest

from demand_vae.eval.metrics import paired_bootstrap_ci, spearman_rank_correlation


class TestPairedBootstrap:
    def test_identical_models_ci_contains_zero(self):
        # Same per-window costs on both sides -> difference is exactly zero;
        # the CI must contain zero (no false positive on identical models).
        rng = np.random.default_rng(0)
        costs = rng.gamma(2.0, 10.0, size=5000)
        ci = paired_bootstrap_ci(costs, costs.copy(), n_boot=500, seed=1)
        assert ci["ci_lower"] <= 0.0 <= ci["ci_upper"]
        assert ci["mean_diff"] == pytest.approx(0.0)

    def test_same_distribution_ci_contains_zero(self):
        # Independent draws from one distribution: CI should straddle zero.
        rng = np.random.default_rng(2)
        a = rng.gamma(2.0, 10.0, size=8000)
        b = rng.gamma(2.0, 10.0, size=8000)
        ci = paired_bootstrap_ci(a, b, n_boot=1000, seed=3)
        assert ci["ci_lower"] < 0.0 < ci["ci_upper"]

    def test_detects_a_real_difference(self):
        rng = np.random.default_rng(4)
        base = rng.gamma(2.0, 10.0, size=8000)
        cheaper = base - 1.5  # paired: uniformly cheaper by 1.5
        ci = paired_bootstrap_ci(cheaper, base, n_boot=1000, seed=5)
        assert ci["ci_upper"] < 0.0  # significantly cheaper
        assert ci["p_gt_zero"] == 0.0

    def test_rejects_unpaired_shapes(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci(np.zeros(10), np.zeros(11))


class TestSpearman:
    def test_identical_rankings_give_one(self):
        crps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cost = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # same order
        assert spearman_rank_correlation(crps, cost) == pytest.approx(1.0)

    def test_reversed_rankings_give_minus_one(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert spearman_rank_correlation(a, a[::-1]) == pytest.approx(-1.0)


class TestPlots:
    def test_plot_script_runs_on_committed_csvs(self, tmp_path):
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        results = repo / "results"
        required = ["full_comparison_summary.csv", "training_log.csv"]
        if not all((results / f).exists() for f in required):
            pytest.skip("Phase 8 result tables not present yet")
        sys.path.insert(0, str(repo / "scripts"))
        try:
            from plot_results import make_figures
        finally:
            sys.path.pop(0)
        written = make_figures(results, tmp_path)
        assert len(written) == 4
        assert all(p.exists() and p.stat().st_size > 0 for p in written)

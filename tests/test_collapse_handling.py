"""Tests for CVAE Milestone 5: KL annealing, free bits, collapse detection."""

import pytest
import torch

from demand_vae.models.vae import (
    apply_free_bits,
    collapse_warning,
    gaussian_kl_per_dim,
    get_beta,
    nb_vae_loss,
)

K = 16


class TestAnnealingSchedule:
    def test_linear_ramp_and_clamp(self):
        assert get_beta(0, 10) == 0.0
        assert get_beta(5, 10) == 0.5
        assert get_beta(10, 10) == 1.0
        assert get_beta(15, 10) == 1.0  # clamps at 1 after annealing

    def test_disabled_annealing_gives_beta_one(self):
        assert get_beta(0, 0) == 1.0


class TestFreeBits:
    def test_floor_lifts_low_kl(self):
        raw = torch.full((K,), 0.05)
        floored = apply_free_bits(raw, lambda_fb=0.5)
        torch.testing.assert_close(floored, torch.full((K,), 0.5))

    def test_floor_leaves_high_kl_untouched(self):
        raw = torch.tensor([0.05, 0.5, 2.0])
        floored = apply_free_bits(raw, lambda_fb=0.5)
        torch.testing.assert_close(floored, torch.tensor([0.5, 0.5, 2.0]))

    def test_zero_lambda_is_identity(self):
        raw = torch.tensor([0.01, 0.2])
        assert apply_free_bits(raw, lambda_fb=0.0) is raw


def _random_nb_batch(batch: int = 32, horizon: int = 4):
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, 40, (batch, horizon), generator=gen).float()
    mu = torch.rand(batch, horizon, generator=gen) * 20 + 0.01
    r = torch.rand(batch, horizon, generator=gen) * 5 + 0.1
    mu_z = torch.randn(batch, K, generator=gen) * 2  # far from prior: big KL
    log_sigma_sq = torch.randn(batch, K, generator=gen)
    return mu, r, x, mu_z, log_sigma_sq


class TestLossWiring:
    def test_beta_zero_removes_kl_contribution(self):
        mu, r, x, mu_z, log_sigma_sq = _random_nb_batch()
        out = nb_vae_loss(mu, r, x, mu_z, log_sigma_sq, beta=0.0, lambda_fb=0.5)
        assert out["kl"].item() > 1.0  # posterior is genuinely far from the prior
        torch.testing.assert_close(out["loss"], out["recon"])  # yet loss == recon

    def test_lambda_zero_matches_unfloored_objective(self):
        mu, r, x, mu_z, log_sigma_sq = _random_nb_batch()
        out = nb_vae_loss(mu, r, x, mu_z, log_sigma_sq, beta=1.0, lambda_fb=0.0)
        expected = out["recon"] + gaussian_kl_per_dim(mu_z, log_sigma_sq).mean(0).sum()
        torch.testing.assert_close(out["loss"], expected)
        torch.testing.assert_close(out["kl_floored"], out["kl"])

    def test_floored_kl_enters_loss_but_not_elbo(self):
        # Near-prior posterior: raw per-dim KL ~ 0 << lambda_fb.
        mu, r, x, _, _ = _random_nb_batch()
        mu_z = torch.zeros(32, K)
        log_sigma_sq = torch.zeros(32, K)
        out = nb_vae_loss(mu, r, x, mu_z, log_sigma_sq, beta=1.0, lambda_fb=0.5)
        assert out["kl_floored"].item() == pytest.approx(0.5 * K, rel=1e-5)
        assert out["kl"].item() == pytest.approx(0.0, abs=1e-6)
        torch.testing.assert_close(out["elbo"], -(out["recon"] + out["kl"]))  # raw KL only


class TestCollapseDetector:
    def test_fires_after_annealing_when_kl_low(self):
        kl_per_dim = torch.full((K,), 0.05)  # mean 0.05 < 0.1
        message = collapse_warning(kl_per_dim, beta=1.0)
        assert message is not None
        assert "Posterior collapse" in message and "lambda_fb" in message

    def test_silent_during_annealing_even_if_kl_low(self):
        kl_per_dim = torch.full((K,), 0.05)
        assert collapse_warning(kl_per_dim, beta=0.5) is None

    def test_silent_when_kl_healthy(self):
        kl_per_dim = torch.full((K,), 0.4)
        assert collapse_warning(kl_per_dim, beta=1.0) is None

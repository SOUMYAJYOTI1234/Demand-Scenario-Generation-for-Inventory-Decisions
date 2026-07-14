"""Tests for CVAE Milestone 2: the unconditional VAE with Gaussian decoder.

The KL formula and the reparameterization trick are the two pieces whose
silent failure would corrupt every later milestone; both are checked against
independent references (torch.distributions, sample moments).
"""

import pytest
import torch

from demand_vae.models.vae import (
    UnconditionalVAE,
    gaussian_kl,
    gaussian_kl_per_dim,
    reparameterize,
    vae_loss,
)

H, LATENT, BATCH = 4, 16, 8


def make_model() -> UnconditionalVAE:
    torch.manual_seed(0)
    return UnconditionalVAE(
        horizon=H, latent_dim=LATENT, encoder_hidden=[128, 64], decoder_hidden=[64, 128]
    )


def make_windows(batch: int = BATCH) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    return torch.randint(0, 50, (batch, H), generator=gen).float()


class TestReparameterization:
    def test_sample_moments_match_parameters(self):
        # One (mu, log_sigma_sq) row broadcast over a large batch of draws:
        # sample mean ~ mu, sample var ~ exp(log_sigma_sq).
        torch.manual_seed(2)
        mu = torch.tensor([[-1.0, 0.0, 2.5]]).expand(200_000, 3)
        log_sigma_sq = torch.tensor([[0.0, -1.0, 1.5]]).expand(200_000, 3)
        gen = torch.Generator().manual_seed(3)
        z = reparameterize(mu, log_sigma_sq, generator=gen)
        torch.testing.assert_close(z.mean(0), mu[0], atol=0.02, rtol=0.02)
        torch.testing.assert_close(z.var(0), log_sigma_sq[0].exp(), atol=0.03, rtol=0.03)

    def test_gradient_flows_through_parameters(self):
        mu = torch.zeros(4, LATENT, requires_grad=True)
        log_sigma_sq = torch.zeros(4, LATENT, requires_grad=True)
        z = reparameterize(mu, log_sigma_sq, generator=torch.Generator().manual_seed(0))
        z.sum().backward()
        assert mu.grad is not None and log_sigma_sq.grad is not None


class TestClosedFormKL:
    def test_matches_torch_distributions(self):
        gen = torch.Generator().manual_seed(4)
        mu = torch.randn(32, LATENT, generator=gen)
        log_sigma_sq = torch.randn(32, LATENT, generator=gen)
        ours = gaussian_kl_per_dim(mu, log_sigma_sq)
        reference = torch.distributions.kl_divergence(
            torch.distributions.Normal(mu, torch.exp(0.5 * log_sigma_sq)),
            torch.distributions.Normal(0.0, 1.0),
        )
        torch.testing.assert_close(ours, reference, atol=1e-5, rtol=1e-5)

    def test_zero_iff_posterior_equals_prior(self):
        mu = torch.zeros(5, LATENT)
        log_sigma_sq = torch.zeros(5, LATENT)  # sigma = 1
        assert gaussian_kl(mu, log_sigma_sq).item() == pytest.approx(0.0, abs=1e-7)
        assert gaussian_kl(mu + 1.0, log_sigma_sq).item() > 0.0


class TestVAEForwardAndLoss:
    def test_forward_returns_correct_shapes(self):
        model = make_model()
        recon, mu, log_sigma_sq = model(make_windows())
        assert recon.shape == (BATCH, H)
        assert mu.shape == (BATCH, LATENT)
        assert log_sigma_sq.shape == (BATCH, LATENT)

    def test_elbo_finite_and_negative_on_random_batch(self):
        model = make_model()
        x = make_windows()
        recon, mu, log_sigma_sq = model(x, generator=torch.Generator().manual_seed(5))
        out = vae_loss(recon, x, mu, log_sigma_sq)
        assert torch.isfinite(out["elbo"])
        assert out["elbo"].item() < 0.0
        assert out["loss"].item() == pytest.approx((out["recon"] + out["kl"]).item(), rel=1e-6)

    def test_one_adam_step_reduces_loss(self):
        model = make_model()
        x = make_windows(64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        gen = torch.Generator().manual_seed(6)
        out0 = vae_loss(*_forward_tuple(model, x, gen), beta=1.0)
        out0["loss"].backward()
        opt.step()
        gen2 = torch.Generator().manual_seed(6)  # same noise -> paired comparison
        out1 = vae_loss(*_forward_tuple(model, x, gen2), beta=1.0)
        assert out1["loss"].item() < out0["loss"].item()

    def test_beta_scales_only_the_kl_term(self):
        model = make_model()
        x = make_windows()
        gen = torch.Generator().manual_seed(7)
        recon, mu, log_sigma_sq = model(x, generator=gen)
        base = vae_loss(recon, x, mu, log_sigma_sq, beta=1.0)
        halved = vae_loss(recon, x, mu, log_sigma_sq, beta=0.5)
        expected = base["recon"] + 0.5 * base["kl"]
        assert halved["loss"].item() == pytest.approx(expected.item(), rel=1e-6)


def _forward_tuple(model, x, gen):
    recon, mu, log_sigma_sq = model(x, generator=gen)
    return recon, x, mu, log_sigma_sq

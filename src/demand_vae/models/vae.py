"""Unconditional VAE with Gaussian decoder — CVAE Milestone 2 (roadmap Phase 6.2).

Adds the stochastic layer to Milestone 1's plumbing: the encoder now outputs
the diagonal-Gaussian posterior parameters (mu, log_sigma_sq), the latent is
drawn via the reparameterization trick, and training maximizes the ELBO

    ELBO = E_q[log p(x | z)] - KL(q(z | x) || N(0, I)).

Deliberately **unconditional**: no context enters either network (Milestone 3
adds conditioning to both). The decoder likelihood is a fixed-unit-variance
Gaussian in log1p demand space (Milestone 1 convention); the reconstruction
term is therefore 0.5 * sum of squared errors up to an additive constant,
which is dropped throughout. The NB decoder replaces it in Milestone 4.

The closed-form diagonal-Gaussian KL is implemented by hand and unit-tested
against ``torch.distributions.kl_divergence`` (roadmap Phase 6.2). Per-
dimension KL is exposed for collapse monitoring from the first run: a latent
dimension with KL ~ 0 carries no information (design doc §8).
"""

from __future__ import annotations

import torch
from torch import nn

from demand_vae.models.autoencoder import mlp


def reparameterize(
    mu: torch.Tensor, log_sigma_sq: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """z = mu + sigma * eps with eps ~ N(0, I), sigma = exp(0.5 * log_sigma_sq).

    The sampling lives outside the networks, so the gradient flows through
    (mu, sigma) deterministically — the low-variance estimator that makes
    VAEs trainable (Kingma & Welling 2014).
    """
    eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    return mu + torch.exp(0.5 * log_sigma_sq) * eps


def gaussian_kl_per_dim(mu: torch.Tensor, log_sigma_sq: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu, sigma^2) || N(0, 1)) per sample and latent dim: (B, K).

    KL = -0.5 * (1 + log_sigma_sq - mu^2 - exp(log_sigma_sq)) elementwise.
    """
    return -0.5 * (1.0 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp())


def gaussian_kl(mu: torch.Tensor, log_sigma_sq: torch.Tensor) -> torch.Tensor:
    """Total KL: summed over latent dimensions, mean over the batch (scalar)."""
    return gaussian_kl_per_dim(mu, log_sigma_sq).sum(dim=-1).mean()


def vae_loss(
    recon_log1p: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_sigma_sq: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """ELBO-derived training loss and its monitored components.

    Reconstruction term: unit-variance Gaussian NLL in log1p space =
    0.5 * sum-of-squared-errors per window (additive constant dropped),
    mean over batch — summed over H so its scale is commensurate with the
    dim-summed KL. Loss = recon + beta * KL (beta = 1 here; annealing is
    Milestone 5). ``elbo`` = -(recon + KL), reported for comparison across
    runs (beta-independent).
    """
    target = torch.log1p(x)
    recon = 0.5 * (recon_log1p - target).pow(2).sum(dim=-1).mean()
    kl = gaussian_kl(mu, log_sigma_sq)
    return {
        "loss": recon + beta * kl,
        "recon": recon,
        "kl": kl,
        "elbo": -(recon + kl),
    }


def nb_vae_loss(
    mu: torch.Tensor,
    r: torch.Tensor,
    x: torch.Tensor,
    mu_z: torch.Tensor,
    log_sigma_sq: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """ELBO-derived loss with the Negative Binomial reconstruction term.

    Same structure and conventions as :func:`vae_loss` — reconstruction NLL
    summed over the H weeks, KL summed over latent dims, both batch-averaged,
    loss = recon + beta * KL — but the observation model is the per-week NB
    on **raw counts** (Milestone 4), so this ELBO is an exact (not
    constant-dropped) bound on the discrete log-likelihood.
    """
    from demand_vae.models.distributions import nb_log_likelihood

    recon = -nb_log_likelihood(x, mu, r).sum(dim=-1).mean()
    kl = gaussian_kl(mu_z, log_sigma_sq)
    return {
        "loss": recon + beta * kl,
        "recon": recon,
        "kl": kl,
        "elbo": -(recon + kl),
    }


class UnconditionalVAE(nn.Module):
    """VAE over demand windows: q(z|x) diagonal Gaussian, p(x|z) Gaussian in log1p space."""

    def __init__(
        self,
        horizon: int,
        latent_dim: int,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
    ):
        super().__init__()
        encoder_hidden = encoder_hidden or [128, 64]
        decoder_hidden = decoder_hidden or [64, 128]
        self.horizon = horizon
        self.latent_dim = latent_dim
        # Final linear emits both posterior heads; chunked into (mu, log_sigma_sq).
        self.encoder = mlp([horizon, *encoder_hidden, 2 * latent_dim])
        self.decoder = mlp([latent_dim, *decoder_hidden, horizon])

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw window (B, H) -> posterior parameters (mu, log_sigma_sq), each (B, K)."""
        mu, log_sigma_sq = self.encoder(torch.log1p(x)).chunk(2, dim=-1)
        return mu, log_sigma_sq

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent (B, K) -> reconstructed log1p window (B, H)."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, log_sigma_sq)."""
        mu, log_sigma_sq = self.encode(x)
        z = reparameterize(mu, log_sigma_sq, generator)
        return self.decode(z), mu, log_sigma_sq

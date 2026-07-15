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


# ---------------------------------------------------------------------------
# Collapse handling — Milestone 5 (roadmap Phase 6.5)
# ---------------------------------------------------------------------------

COLLAPSE_KL_THRESHOLD = 0.1  # nats per dimension: below this the latent is ~unused


def get_beta(epoch: int | float, n_anneal_epochs: int) -> float:
    """Linear KL-annealing schedule: beta = min(1, epoch / n_anneal_epochs).

    Ramping the KL weight from 0 lets the decoder learn to use z before the
    prior-matching pressure arrives (Bowman et al. 2016). Clamps at 1;
    ``n_anneal_epochs <= 0`` disables annealing (beta = 1 always).
    """
    if n_anneal_epochs <= 0:
        return 1.0
    return min(1.0, epoch / n_anneal_epochs)


def apply_free_bits(kl_per_dim: torch.Tensor, lambda_fb: float) -> torch.Tensor:
    """Floor each per-dimension KL at ``lambda_fb`` nats (Kingma et al. 2016).

    Applied to the batch-averaged per-dimension KL: dimensions below the
    floor contribute a constant ``lambda_fb`` to the loss and therefore
    receive no gradient pushing them further toward the prior — the
    optimizer can't profit from collapsing them.
    """
    if lambda_fb <= 0:
        return kl_per_dim
    return torch.clamp(kl_per_dim, min=lambda_fb)


def collapse_warning(kl_per_dim: torch.Tensor, beta: float) -> str | None:
    """Return a warning string iff the posterior has collapsed post-annealing.

    Fires when the mean per-dimension KL is below ``COLLAPSE_KL_THRESHOLD``
    AND annealing is complete (beta >= 1) — low KL while beta < 1 is expected,
    not pathological. ``kl_per_dim`` is the (K,) batch-mean per-dimension KL.
    """
    mean_kl = float(kl_per_dim.mean())
    if beta >= 1.0 and mean_kl < COLLAPSE_KL_THRESHOLD:
        return (
            f"Posterior collapse detected: mean per-dim KL = {mean_kl:.3f} nats. "
            "Consider increasing lambda_fb."
        )
    return None


def vae_loss(
    recon_log1p: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_sigma_sq: torch.Tensor,
    beta: float = 1.0,
    lambda_fb: float = 0.0,
) -> dict[str, torch.Tensor]:
    """ELBO-derived training loss and its monitored components.

    Reconstruction term: unit-variance Gaussian NLL in log1p space =
    0.5 * sum-of-squared-errors per window (additive constant dropped),
    mean over batch — summed over H so its scale is commensurate with the
    dim-summed KL.

    Loss = recon + beta * sum(free-bits-floored per-dim KL). ``beta`` follows
    the annealing schedule (:func:`get_beta`); ``lambda_fb`` is the free-bits
    floor in nats (0 disables — then this reduces exactly to the Milestone-2
    objective). Monitored ``kl`` and ``elbo`` always use the *raw* KL: the
    floor shapes gradients, not the reported bound.
    """
    target = torch.log1p(x)
    recon = 0.5 * (recon_log1p - target).pow(2).sum(dim=-1).mean()
    kl_per_dim = gaussian_kl_per_dim(mu, log_sigma_sq).mean(dim=0)  # (K,)
    kl = kl_per_dim.sum()
    kl_train = apply_free_bits(kl_per_dim, lambda_fb).sum()
    return {
        "loss": recon + beta * kl_train,
        "recon": recon,
        "kl": kl,
        "kl_floored": kl_train,
        "elbo": -(recon + kl),
    }


def nb_vae_loss(
    mu: torch.Tensor,
    r: torch.Tensor,
    x: torch.Tensor,
    mu_z: torch.Tensor,
    log_sigma_sq: torch.Tensor,
    beta: float = 1.0,
    lambda_fb: float = 0.0,
) -> dict[str, torch.Tensor]:
    """ELBO-derived loss with the Negative Binomial reconstruction term.

    Same structure and conventions as :func:`vae_loss` — reconstruction NLL
    summed over the H weeks, KL summed over latent dims, both batch-averaged,
    loss = recon + beta * free-bits-floored KL — but the observation model is
    the per-week NB on **raw counts** (Milestone 4), so the reported ``elbo``
    (always from the raw KL) is an exact bound on the discrete
    log-likelihood.
    """
    from demand_vae.models.distributions import nb_log_likelihood

    recon = -nb_log_likelihood(x, mu, r).sum(dim=-1).mean()
    kl_per_dim = gaussian_kl_per_dim(mu_z, log_sigma_sq).mean(dim=0)  # (K,)
    kl = kl_per_dim.sum()
    kl_train = apply_free_bits(kl_per_dim, lambda_fb).sum()
    return {
        "loss": recon + beta * kl_train,
        "recon": recon,
        "kl": kl,
        "kl_floored": kl_train,
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

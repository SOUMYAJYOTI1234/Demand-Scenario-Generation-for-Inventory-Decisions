"""Demand scenario models: sampler interface, trivial samplers, autoencoder -> CVAE (Phase 6)."""

from demand_vae.models.autoencoder import ContextEncoder, DemandAutoencoder, reconstruction_loss
from demand_vae.models.base import BaseSampler
from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.trivial import ConstantSampler, FixedRatePoissonSampler
from demand_vae.models.vae import (
    UnconditionalVAE,
    gaussian_kl,
    gaussian_kl_per_dim,
    reparameterize,
    vae_loss,
)

__all__ = [
    "BaseSampler",
    "ConditionalVAE",
    "ConstantSampler",
    "ContextEncoder",
    "DemandAutoencoder",
    "FixedRatePoissonSampler",
    "UnconditionalVAE",
    "gaussian_kl",
    "gaussian_kl_per_dim",
    "reconstruction_loss",
    "reparameterize",
    "vae_loss",
]

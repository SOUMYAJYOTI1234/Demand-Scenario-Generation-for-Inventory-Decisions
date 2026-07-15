"""Demand scenario models: sampler interface, trivial samplers, autoencoder -> CVAE (Phase 6)."""

from demand_vae.models.autoencoder import ContextEncoder, DemandAutoencoder, reconstruction_loss
from demand_vae.models.base import BaseSampler
from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.distributions import MU_FLOOR, R_FLOOR, nb_log_likelihood
from demand_vae.models.trivial import ConstantSampler, FixedRatePoissonSampler
from demand_vae.models.vae import (
    UnconditionalVAE,
    apply_free_bits,
    collapse_warning,
    gaussian_kl,
    gaussian_kl_per_dim,
    get_beta,
    nb_vae_loss,
    reparameterize,
    vae_loss,
)

__all__ = [
    "MU_FLOOR",
    "R_FLOOR",
    "BaseSampler",
    "ConditionalVAE",
    "ConstantSampler",
    "ContextEncoder",
    "DemandAutoencoder",
    "FixedRatePoissonSampler",
    "UnconditionalVAE",
    "apply_free_bits",
    "collapse_warning",
    "gaussian_kl",
    "gaussian_kl_per_dim",
    "get_beta",
    "nb_log_likelihood",
    "nb_vae_loss",
    "reconstruction_loss",
    "reparameterize",
    "vae_loss",
]

"""Conditional VAE for demand windows — the core model (roadmap Phase 6).

Planned design (frozen in docs/project-design-document.md §10):
- Encoder q_phi(z | x, y): diagonal Gaussian, context y fed to the encoder as
  well as the decoder (conditioning only the decoder is the classic CVAE
  design error — design doc §8).
- Decoder p_theta(x | z, y): per-week Negative Binomial parameters,
  conditionally independent across weeks given (z, y); Gaussian decoder
  retained as a config-switchable ablation.
- Latent z in R^K, K = 8 default, prior N(0, I).
- Training: conditional ELBO with KL annealing and free bits available from
  run one; per-dimension KL monitored to detect posterior collapse.
- Build order within Phase 6: plain autoencoder -> unconditional VAE
  (Gaussian decoder) -> conditional VAE -> count decoder -> collapse handling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from demand_vae.models.base import BaseSampler


class ConditionalVAE(BaseSampler):
    """CVAE demand scenario generator implementing the BaseSampler contract."""

    def __init__(self, horizon: int, latent_dim: int, decoder: str = "negative_binomial"):
        super().__init__(horizon)
        self.latent_dim = latent_dim
        self.decoder = decoder

    def fit(self, windows: np.ndarray, contexts: Any) -> None:
        """Train encoder/decoder by maximizing the conditional ELBO."""
        raise NotImplementedError(
            "TODO(Phase 6/7): CVAE training — ELBO objective, KL annealing/free bits, "
            "per-dimension KL monitoring, early stopping on validation ELBO."
        )

    def elbo(self, windows: np.ndarray, contexts: Any) -> np.ndarray:
        """Per-sample conditional ELBO (also the Level-1 NLL bound, Phase 8)."""
        raise NotImplementedError(
            "TODO(Phase 6): conditional ELBO = NB reconstruction log-likelihood "
            "- KL(q_phi(z|x,y) || N(0,I))."
        )

    def sample(self, context: Any, n_scenarios: int) -> np.ndarray:
        """Draw z ~ N(0,I), decode NB parameters given (z, y), sample counts."""
        raise NotImplementedError(
            "TODO(Phase 6): one-pass conditional sampling — prior draw, decode, "
            "sample from the per-week Negative Binomial."
        )

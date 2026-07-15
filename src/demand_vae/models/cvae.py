"""Conditional VAE — Milestones 3 + 4 (roadmap Phase 6.3-6.4; Sohn et al. 2015).

Conditioning (Milestone 3): the context embedding y (from the Milestone-1
:class:`ContextEncoder` — item/store embeddings + scaled continuous features)
is concatenated into the input of **both** networks:

    encoder q(z | x, y):  [log1p(x) | y] -> (mu_z, log_sigma_sq)
    decoder p(x | z, y):  [z | y]        -> decoder parameters

The intended division of labour (design doc §1.3): y explains the systematic
variation (seasonal level, price response, recent demand level), z captures
the residual stochastic scenario around it.

CHECKLIST ITEM (design doc §8, "conditioning leakage design error"):
conditioning ONLY the decoder is the documented recurrent mistake — the
encoder must also observe y, otherwise the posterior is forced to smuggle
context information through z, corrupting the y/z decomposition and wasting
latent capacity. Both `encode` and `decode` below therefore take the context
embedding; there is deliberately no context-free path through either network.

Decoder likelihoods (Milestone 4, config key ``model.decoder_likelihood``):

- ``"nb"`` (default): a shared MLP trunk feeds two linear heads producing
  per-week Negative Binomial parameters; softplus + floors keep them strictly
  positive (design doc §8 stability guard). Weeks are conditionally
  independent given (z, y); marginal within-window correlation flows through
  z (design doc §1.3 note). Reconstruction = NB NLL on raw counts.
- ``"gaussian"``: single linear head, reconstruction in log1p space via
  :func:`demand_vae.models.vae.vae_loss` — retained as the RQ1 ablation.

KL annealing / free bits are Milestone 5.
"""

from __future__ import annotations

import torch
from torch import nn

from demand_vae.models.autoencoder import ContextEncoder, mlp
from demand_vae.models.distributions import MU_FLOOR, R_FLOOR
from demand_vae.models.vae import reparameterize

DECODER_LIKELIHOODS = ("nb", "gaussian")


class ConditionalVAE(nn.Module):
    """CVAE over demand windows: q(z|x,y) and p(x|z,y), y in both networks."""

    def __init__(
        self,
        horizon: int,
        latent_dim: int,
        n_items: int,
        n_stores: int,
        n_continuous: int,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        item_embedding_dim: int = 32,
        store_embedding_dim: int = 4,
        decoder_likelihood: str = "nb",
    ):
        super().__init__()
        if decoder_likelihood not in DECODER_LIKELIHOODS:
            raise ValueError(
                f"decoder_likelihood must be one of {DECODER_LIKELIHOODS}, "
                f"got {decoder_likelihood!r}"
            )
        encoder_hidden = encoder_hidden or [128, 64]
        decoder_hidden = decoder_hidden or [64, 128]
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.decoder_likelihood = decoder_likelihood
        self.context_encoder = ContextEncoder(
            n_items, n_stores, n_continuous, item_embedding_dim, store_embedding_dim
        )
        ctx_dim = self.context_encoder.output_dim
        # Context enters BOTH networks (Sohn et al. 2015; see module docstring).
        self.encoder = mlp([horizon + ctx_dim, *encoder_hidden, 2 * latent_dim])
        # Shared trunk (ends in an activated hidden layer) + likelihood heads.
        self.decoder_trunk = nn.Sequential(mlp([latent_dim + ctx_dim, *decoder_hidden]), nn.ReLU())
        if decoder_likelihood == "nb":
            self.mu_head = nn.Linear(decoder_hidden[-1], horizon)
            self.r_head = nn.Linear(decoder_hidden[-1], horizon)
        else:
            self.out_head = nn.Linear(decoder_hidden[-1], horizon)

    def embed_context(self, context_cat: torch.Tensor, context_cont: torch.Tensor) -> torch.Tensor:
        """Context dict arrays -> single embedding vector (B, ctx_dim)."""
        return self.context_encoder(context_cat, context_cont)

    def encode(self, x: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[log1p(x) | y] -> posterior parameters (mu_z, log_sigma_sq), each (B, K)."""
        h = self.encoder(torch.cat([torch.log1p(x), ctx], dim=1))
        mu_z, log_sigma_sq = h.chunk(2, dim=-1)
        return mu_z, log_sigma_sq

    def decode(
        self, z: torch.Tensor, ctx: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """[z | y] -> decoder parameters.

        nb: per-week NB parameters ``(mu, r)``, each (B, H), softplus-floored
        strictly positive. gaussian: reconstructed log1p window (B, H).
        """
        h = self.decoder_trunk(torch.cat([z, ctx], dim=1))
        if self.decoder_likelihood == "nb":
            mu = nn.functional.softplus(self.mu_head(h)) + MU_FLOOR
            r = nn.functional.softplus(self.r_head(h)) + R_FLOOR
            return mu, r
        return self.out_head(h)

    def forward(
        self,
        x: torch.Tensor,
        context_cat: torch.Tensor,
        context_cont: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Returns (decoder_output, mu_z, log_sigma_sq).

        ``decoder_output`` is ``(mu, r)`` for the NB decoder, or the
        reconstructed log1p window for the Gaussian decoder.
        """
        ctx = self.embed_context(context_cat, context_cont)
        mu_z, log_sigma_sq = self.encode(x, ctx)
        z = reparameterize(mu_z, log_sigma_sq, generator)
        return self.decode(z, ctx), mu_z, log_sigma_sq

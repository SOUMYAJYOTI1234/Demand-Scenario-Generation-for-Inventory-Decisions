"""Conditional VAE — CVAE Milestone 3 (roadmap Phase 6.3; Sohn et al. 2015).

Adds conditioning to Milestone 2's VAE: the context embedding y (from the
Milestone-1 :class:`ContextEncoder` — item/store embeddings + scaled
continuous features) is concatenated into the input of **both** networks:

    encoder q(z | x, y):  [log1p(x) | y] -> (mu, log_sigma_sq)
    decoder p(x | z, y):  [z | y]        -> reconstructed log1p window

The intended division of labour (design doc §1.3): y explains the systematic
variation (seasonal level, price response, recent demand level), z captures
the residual stochastic scenario around it.

CHECKLIST ITEM (design doc §8, "conditioning leakage design error"):
conditioning ONLY the decoder is the documented recurrent mistake — the
encoder must also observe y, otherwise the posterior is forced to smuggle
context information through z, corrupting the y/z decomposition and wasting
latent capacity. Both `encode` and `decode` below therefore take the context
embedding; there is deliberately no context-free path through either network.

Decoder likelihood is still the fixed-unit-variance Gaussian in log1p space
(same `vae_loss`); the Negative Binomial decoder is Milestone 4, KL
annealing / free bits are Milestone 5.
"""

from __future__ import annotations

import torch
from torch import nn

from demand_vae.models.autoencoder import ContextEncoder, mlp
from demand_vae.models.vae import reparameterize


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
    ):
        super().__init__()
        encoder_hidden = encoder_hidden or [128, 64]
        decoder_hidden = decoder_hidden or [64, 128]
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.context_encoder = ContextEncoder(
            n_items, n_stores, n_continuous, item_embedding_dim, store_embedding_dim
        )
        ctx_dim = self.context_encoder.output_dim
        # Context enters BOTH networks (Sohn et al. 2015; see module docstring).
        self.encoder = mlp([horizon + ctx_dim, *encoder_hidden, 2 * latent_dim])
        self.decoder = mlp([latent_dim + ctx_dim, *decoder_hidden, horizon])

    def embed_context(self, context_cat: torch.Tensor, context_cont: torch.Tensor) -> torch.Tensor:
        """Context dict arrays -> single embedding vector (B, ctx_dim)."""
        return self.context_encoder(context_cat, context_cont)

    def encode(self, x: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[log1p(x) | y] -> posterior parameters (mu, log_sigma_sq), each (B, K)."""
        h = self.encoder(torch.cat([torch.log1p(x), ctx], dim=1))
        mu, log_sigma_sq = h.chunk(2, dim=-1)
        return mu, log_sigma_sq

    def decode(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """[z | y] -> reconstructed log1p window (B, H)."""
        return self.decoder(torch.cat([z, ctx], dim=1))

    def forward(
        self,
        x: torch.Tensor,
        context_cat: torch.Tensor,
        context_cont: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, log_sigma_sq)."""
        ctx = self.embed_context(context_cat, context_cont)
        mu, log_sigma_sq = self.encode(x, ctx)
        z = reparameterize(mu, log_sigma_sq, generator)
        return self.decode(z, ctx), mu, log_sigma_sq

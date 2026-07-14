"""Plain deterministic autoencoder — CVAE Milestone 1 (roadmap Phase 6.1).

Purpose: verify data flow and reconstruction plumbing before any stochastic
component exists. Encoder MLP (flattened H-week demand window + context
embedding) -> deterministic bottleneck -> decoder MLP -> reconstructed
window. MSE loss; no sampling, no KL, no latent distribution — those arrive
in Milestone 2 (unconditional VAE).

Conventions shared with the upcoming VAE/CVAE:
- Demand windows are reconstructed in **log1p space** (raw counts span
  orders of magnitude across series; MSE on raw counts would be dominated by
  a few high-volume series).
- :class:`ContextEncoder` is the reusable conditioning block: learned
  embeddings for item/store indices concatenated with the (already scaled)
  continuous features. The CVAE will feed it to *both* encoder and decoder
  (design doc §10.3); here it enters the encoder only.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(dims: list[int]) -> nn.Sequential:
    """ReLU MLP through the given widths; linear final layer."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ContextEncoder(nn.Module):
    """Embeds the context y: categorical indices -> embeddings, continuous pass-through.

    Output: concat(item_embedding, store_embedding, context_cont), dimension
    ``output_dim``.
    """

    def __init__(
        self,
        n_items: int,
        n_stores: int,
        n_continuous: int,
        item_embedding_dim: int = 32,
        store_embedding_dim: int = 4,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, item_embedding_dim)
        self.store_embedding = nn.Embedding(n_stores, store_embedding_dim)
        self.output_dim = item_embedding_dim + store_embedding_dim + n_continuous

    def forward(self, context_cat: torch.Tensor, context_cont: torch.Tensor) -> torch.Tensor:
        item = self.item_embedding(context_cat[:, 0])
        store = self.store_embedding(context_cat[:, 1])
        return torch.cat([item, store, context_cont], dim=1)


class DemandAutoencoder(nn.Module):
    """Deterministic encode-decode over demand windows.

    encode: [log1p(x) | context] -> bottleneck (B, latent_dim)
    decode: bottleneck -> reconstructed log1p window (B, H)
    """

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
        in_dim = horizon + self.context_encoder.output_dim
        self.encoder = mlp([in_dim, *encoder_hidden, latent_dim])
        self.decoder = mlp([latent_dim, *decoder_hidden, horizon])

    def encode(
        self, x: torch.Tensor, context_cat: torch.Tensor, context_cont: torch.Tensor
    ) -> torch.Tensor:
        """x: raw demand window (B, H) — log1p'd internally. Returns (B, latent_dim)."""
        ctx = self.context_encoder(context_cat, context_cont)
        return self.encoder(torch.cat([torch.log1p(x), ctx], dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Bottleneck -> reconstructed log1p window (B, H)."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, context_cat: torch.Tensor, context_cont: torch.Tensor
    ) -> torch.Tensor:
        return self.decode(self.encode(x, context_cat, context_cont))


def reconstruction_loss(recon_log1p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE between the reconstruction and the target window, in log1p space."""
    return nn.functional.mse_loss(recon_log1p, torch.log1p(x))

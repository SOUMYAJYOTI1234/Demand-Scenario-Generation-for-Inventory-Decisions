"""Milestone-1 smoke training: plain autoencoder on a small slice of real M5 data.

Proves the pipeline tensors flow end-to-end through a real forward/backward
pass: M5 windows -> ContextEncoder + MLP autoencoder -> MSE(log1p) loss.
Trains 3 epochs on the first 1,000 train windows only, prints the loss per
epoch. Not a real training run — Milestones 2+ replace this with the VAE
trainer (roadmap Phase 7).

Usage: python scripts/train_autoencoder.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from demand_vae.config import load_config
from demand_vae.data.pipeline import build_dataset
from demand_vae.models.autoencoder import DemandAutoencoder, reconstruction_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
N_WINDOWS = 1000
N_EPOCHS = 3
BATCH_SIZE = 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(config.training.seeds[0])

    print("Building M5 dataset (scaled splits) ...")
    splits, _, panel = build_dataset(config, REPO_ROOT / "data")
    train = splits["train"]
    x = torch.from_numpy(train.x[:N_WINDOWS]).float()
    cat = torch.from_numpy(train.context_cat[:N_WINDOWS])
    cont = torch.from_numpy(train.context_cont[:N_WINDOWS]).float()
    print(f"  slice: {len(x)} windows | context: cat {tuple(cat.shape)}, cont {tuple(cont.shape)}")

    model = DemandAutoencoder(
        horizon=config.data.horizon_weeks,
        latent_dim=config.model.latent_dim,
        n_items=len(panel.item_categories),
        n_stores=len(panel.store_categories),
        n_continuous=cont.shape[1],
        encoder_hidden=list(config.model.encoder_hidden),
        decoder_hidden=list(config.model.decoder_hidden),
        item_embedding_dim=config.model.item_embedding_dim,
        store_embedding_dim=config.model.store_embedding_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} parameters, bottleneck {config.model.latent_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    for epoch in range(1, N_EPOCHS + 1):
        perm = torch.randperm(len(x))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(x), BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = reconstruction_loss(model(x[idx], cat[idx], cont[idx]), x[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch}/{N_EPOCHS}: train MSE(log1p) = {epoch_loss / n_batches:.4f}")

    print("Milestone 1 OK: real M5 tensors flow through encode -> bottleneck -> decode.")


if __name__ == "__main__":
    main()

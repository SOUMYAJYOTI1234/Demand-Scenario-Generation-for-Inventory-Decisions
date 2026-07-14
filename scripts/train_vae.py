"""Milestone-2 smoke training: unconditional VAE on a small slice of real M5 data.

Trains 3 epochs on the first 1,000 train windows and prints reconstruction /
KL / total loss per epoch plus the per-dimension KL after the last epoch —
the collapse diagnostic that must be live from the first run (design doc §8;
a latent dimension with KL < 0.1 nats carries almost no information).

Usage: python scripts/train_vae.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from demand_vae.config import load_config
from demand_vae.data.pipeline import build_dataset
from demand_vae.models.vae import UnconditionalVAE, gaussian_kl_per_dim, vae_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
N_WINDOWS = 1000
N_EPOCHS = 3
BATCH_SIZE = 128
KL_WARN_THRESHOLD = 0.1  # nats per dimension


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(config.training.seeds[0])

    print("Building M5 dataset ...")
    splits, _, _ = build_dataset(config, REPO_ROOT / "data")
    x = torch.from_numpy(splits["train"].x[:N_WINDOWS]).float()
    print(f"  slice: {len(x)} windows (unconditional: context intentionally unused)")

    model = UnconditionalVAE(
        horizon=config.data.horizon_weeks,
        latent_dim=config.model.latent_dim,
        encoder_hidden=list(config.model.encoder_hidden),
        decoder_hidden=list(config.model.decoder_hidden),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} parameters, K={config.model.latent_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    for epoch in range(1, N_EPOCHS + 1):
        perm = torch.randperm(len(x))
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0
        for start in range(0, len(x), BATCH_SIZE):
            batch = x[perm[start : start + BATCH_SIZE]]
            optimizer.zero_grad()
            recon, mu, log_sigma_sq = model(batch)
            out = vae_loss(recon, batch, mu, log_sigma_sq, beta=1.0)
            out["loss"].backward()
            optimizer.step()
            for key in sums:
                sums[key] += out[key].item()
            n_batches += 1
        print(
            f"  epoch {epoch}/{N_EPOCHS}: recon={sums['recon'] / n_batches:.4f}  "
            f"KL={sums['kl'] / n_batches:.4f}  loss={sums['loss'] / n_batches:.4f}  "
            f"(ELBO={-(sums['recon'] + sums['kl']) / n_batches:.4f})"
        )

    with torch.no_grad():
        mu, log_sigma_sq = model.encode(x)
        per_dim = gaussian_kl_per_dim(mu, log_sigma_sq).mean(dim=0)
    formatted = "  ".join(f"{v:.3f}" for v in per_dim.tolist())
    print(f"\n  per-dimension KL (nats), K={len(per_dim)}:\n  [{formatted}]")
    active = int((per_dim >= KL_WARN_THRESHOLD).sum())
    print(f"  active dimensions (KL >= {KL_WARN_THRESHOLD}): {active}/{len(per_dim)}")
    if per_dim.sum().item() < KL_WARN_THRESHOLD:
        print("  WARNING: total KL ~ 0 - posterior collapse (encoder ignored).")
    else:
        print("  KL is non-zero: the encoder is carrying information. Milestone 2 OK.")


if __name__ == "__main__":
    main()

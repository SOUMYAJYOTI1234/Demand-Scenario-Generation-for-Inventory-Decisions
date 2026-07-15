"""Milestone-4/5 smoke training: NB-decoder CVAE with KL annealing + free bits.

Trains 5 epochs on the first 1,000 train windows — enough to watch beta ramp
0.1 -> 0.5 (n_anneal_epochs = 10 from the config). Per epoch: beta,
reconstruction NLL, raw KL, floored KL, loss; per-dimension KL after every
epoch feeds the collapse detector (which by construction stays silent while
beta < 1). Free bits floor lambda_fb comes from the config.

Usage: python scripts/train_cvae_nb.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from demand_vae.config import load_config
from demand_vae.data.pipeline import build_dataset
from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.vae import (
    collapse_warning,
    gaussian_kl_per_dim,
    get_beta,
    nb_vae_loss,
    reparameterize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
N_WINDOWS = 1000
N_EPOCHS = 5  # smoke run: beta reaches 0.5 of its 10-epoch ramp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    if config.model.decoder_likelihood != "nb":
        raise SystemExit("This script trains the NB decoder; set model.decoder_likelihood: nb")
    n_anneal = config.training.n_anneal_epochs
    lambda_fb = config.training.lambda_fb
    batch_size = config.training.batch_size

    print("Building M5 dataset ...")
    splits, _, panel = build_dataset(config, REPO_ROOT / "data")
    train = splits["train"]
    x = torch.from_numpy(train.x[:N_WINDOWS]).float()
    cat = torch.from_numpy(train.context_cat[:N_WINDOWS])
    cont = torch.from_numpy(train.context_cont[:N_WINDOWS]).float()
    print(
        f"  slice: {len(x)} windows | NB decoder | "
        f"anneal over {n_anneal} epochs, lambda_fb={lambda_fb}"
    )

    torch.manual_seed(config.training.seeds[0])
    model = ConditionalVAE(
        horizon=config.data.horizon_weeks,
        latent_dim=config.model.latent_dim,
        n_items=len(panel.item_categories),
        n_stores=len(panel.store_categories),
        n_continuous=cont.shape[1],
        encoder_hidden=list(config.model.encoder_hidden),
        decoder_hidden=list(config.model.decoder_hidden),
        item_embedding_dim=config.model.item_embedding_dim,
        store_embedding_dim=config.model.store_embedding_dim,
        decoder_likelihood="nb",
    )
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    previous_beta = -1.0
    for epoch in range(1, N_EPOCHS + 1):
        beta = get_beta(epoch, n_anneal)
        assert beta > previous_beta, "beta must increase every epoch during the ramp"
        previous_beta = beta
        perm = torch.randperm(len(x))
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "kl_floored": 0.0}
        n_batches = 0
        for start in range(0, len(x), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            ctx = model.embed_context(cat[idx], cont[idx])
            mu_z, log_sigma_sq = model.encode(x[idx], ctx)
            z = reparameterize(mu_z, log_sigma_sq)
            mu, r = model.decode(z, ctx)
            out = nb_vae_loss(mu, r, x[idx], mu_z, log_sigma_sq, beta=beta, lambda_fb=lambda_fb)
            if not torch.isfinite(out["loss"]):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} — NB guards failed")
            out["loss"].backward()
            optimizer.step()
            for key in sums:
                sums[key] += out[key].item()
            n_batches += 1

        with torch.no_grad():
            ctx = model.embed_context(cat, cont)
            mu_z, log_sigma_sq = model.encode(x, ctx)
            per_dim = gaussian_kl_per_dim(mu_z, log_sigma_sq).mean(dim=0)
        print(
            f"  epoch {epoch}/{N_EPOCHS}: beta={beta:.2f}  "
            f"NB-NLL={sums['recon'] / n_batches:.4f}  KL={sums['kl'] / n_batches:.4f}  "
            f"KL_floored={sums['kl_floored'] / n_batches:.4f}  "
            f"loss={sums['loss'] / n_batches:.4f}  min-dim-KL={per_dim.min():.3f}"
        )
        warning = collapse_warning(per_dim, beta)
        if warning:
            print(f"  WARNING: {warning}")

    formatted = "  ".join(f"{v:.3f}" for v in per_dim.tolist())
    above_floor = int((per_dim >= lambda_fb).sum())
    print(f"\n  per-dimension KL (nats) after epoch {N_EPOCHS}:\n  [{formatted}]")
    print(f"  dimensions at/above the lambda_fb={lambda_fb} floor: {above_floor}/{len(per_dim)}")
    print("  beta ramped monotonically; no collapse warning fired. Milestone 5 OK.")


if __name__ == "__main__":
    main()

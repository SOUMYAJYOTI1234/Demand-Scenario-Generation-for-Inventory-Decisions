"""Milestone-4 smoke training: CVAE with Negative Binomial decoder on real M5 data.

Same structure as scripts/train_cvae.py, but the decoder emits per-week NB
(mu, r) parameters and the reconstruction term is the NB NLL on raw counts —
the likelihood the EDA motivates (median var/mean ~ 4 in FOODS). Trains 3
epochs on the first 1,000 train windows; prints NLL / KL / loss per epoch and
per-dimension KL after the last epoch. The ELBO must stay finite throughout
(softplus floors on both NB parameters are the guard).

Usage: python scripts/train_cvae_nb.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from demand_vae.config import load_config
from demand_vae.data.pipeline import build_dataset
from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.vae import gaussian_kl_per_dim, nb_vae_loss, reparameterize

REPO_ROOT = Path(__file__).resolve().parents[1]
N_WINDOWS = 1000
N_EPOCHS = 3
BATCH_SIZE = 128
KL_WARN_THRESHOLD = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    if config.model.decoder_likelihood != "nb":
        raise SystemExit("This script trains the NB decoder; set model.decoder_likelihood: nb")

    print("Building M5 dataset ...")
    splits, _, panel = build_dataset(config, REPO_ROOT / "data")
    train = splits["train"]
    x = torch.from_numpy(train.x[:N_WINDOWS]).float()
    cat = torch.from_numpy(train.context_cat[:N_WINDOWS])
    cont = torch.from_numpy(train.context_cont[:N_WINDOWS]).float()
    print(f"  slice: {len(x)} windows | NB decoder on raw counts")

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

    for epoch in range(1, N_EPOCHS + 1):
        perm = torch.randperm(len(x))
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0
        for start in range(0, len(x), BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            ctx = model.embed_context(cat[idx], cont[idx])
            mu_z, log_sigma_sq = model.encode(x[idx], ctx)
            z = reparameterize(mu_z, log_sigma_sq)
            mu, r = model.decode(z, ctx)
            out = nb_vae_loss(mu, r, x[idx], mu_z, log_sigma_sq, beta=1.0)
            if not torch.isfinite(out["loss"]):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} — NB guards failed")
            out["loss"].backward()
            optimizer.step()
            for key in sums:
                sums[key] += out[key].item()
            n_batches += 1
        print(
            f"  epoch {epoch}/{N_EPOCHS}: NB-NLL={sums['recon'] / n_batches:.4f}  "
            f"KL={sums['kl'] / n_batches:.4f}  loss={sums['loss'] / n_batches:.4f}  "
            f"ELBO={-sums['loss'] / n_batches:.4f}"
        )

    with torch.no_grad():
        ctx = model.embed_context(cat, cont)
        mu_z, log_sigma_sq = model.encode(x, ctx)
        per_dim = gaussian_kl_per_dim(mu_z, log_sigma_sq).mean(dim=0)
    formatted = "  ".join(f"{v:.3f}" for v in per_dim.tolist())
    active = int((per_dim >= KL_WARN_THRESHOLD).sum())
    print(f"\n  per-dimension KL (nats):\n  [{formatted}]")
    print(f"  active dimensions (KL >= {KL_WARN_THRESHOLD}): {active}/{len(per_dim)}")
    print("  NB ELBO finite throughout. Milestone 4 OK (collapse handling is Milestone 5).")


if __name__ == "__main__":
    main()

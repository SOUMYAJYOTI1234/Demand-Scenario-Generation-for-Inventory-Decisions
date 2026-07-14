"""Milestone-3 smoke training: conditional VAE on a small slice of real M5 data.

Trains 3 epochs on the first 1,000 train windows (context in BOTH encoder and
decoder), printing reconstruction / KL / loss per epoch and per-dimension KL
after the last epoch. Then runs the fast conditioning sanity check: a fresh
model trained 1 epoch with zeroed context embeddings must end with a worse
(lower) ELBO than the real-context model had after its first epoch — if not,
conditioning is wired wrong or the features carry no signal.

Usage: python scripts/train_cvae.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from demand_vae.config import load_config
from demand_vae.data.pipeline import build_dataset
from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.vae import gaussian_kl_per_dim, reparameterize, vae_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
N_WINDOWS = 1000
N_EPOCHS = 3
BATCH_SIZE = 128
KL_WARN_THRESHOLD = 0.1


def make_model(config, n_items: int, n_stores: int, n_continuous: int) -> ConditionalVAE:
    return ConditionalVAE(
        horizon=config.data.horizon_weeks,
        latent_dim=config.model.latent_dim,
        n_items=n_items,
        n_stores=n_stores,
        n_continuous=n_continuous,
        encoder_hidden=list(config.model.encoder_hidden),
        decoder_hidden=list(config.model.decoder_hidden),
        item_embedding_dim=config.model.item_embedding_dim,
        store_embedding_dim=config.model.store_embedding_dim,
    )


def run_epoch(model, optimizer, x, cat, cont, zero_context: bool = False) -> dict[str, float]:
    perm = torch.randperm(len(x))
    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "elbo": 0.0}
    n_batches = 0
    for start in range(0, len(x), BATCH_SIZE):
        idx = perm[start : start + BATCH_SIZE]
        optimizer.zero_grad()
        ctx = model.embed_context(cat[idx], cont[idx])
        if zero_context:
            ctx = torch.zeros_like(ctx)
        mu, log_sigma_sq = model.encode(x[idx], ctx)
        z = reparameterize(mu, log_sigma_sq)
        out = vae_loss(model.decode(z, ctx), x[idx], mu, log_sigma_sq, beta=1.0)
        out["loss"].backward()
        optimizer.step()
        for key in sums:
            sums[key] += out[key].item()
        n_batches += 1
    return {key: value / n_batches for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config.training.seeds[0]

    print("Building M5 dataset ...")
    splits, _, panel = build_dataset(config, REPO_ROOT / "data")
    train = splits["train"]
    x = torch.from_numpy(train.x[:N_WINDOWS]).float()
    cat = torch.from_numpy(train.context_cat[:N_WINDOWS])
    cont = torch.from_numpy(train.context_cont[:N_WINDOWS]).float()
    n_items, n_stores = len(panel.item_categories), len(panel.store_categories)
    print(f"  slice: {len(x)} windows | context into BOTH encoder and decoder")

    torch.manual_seed(seed)
    model = make_model(config, n_items, n_stores, cont.shape[1])
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    first_epoch_elbo = None
    for epoch in range(1, N_EPOCHS + 1):
        stats = run_epoch(model, optimizer, x, cat, cont)
        if epoch == 1:
            first_epoch_elbo = stats["elbo"]
        print(
            f"  epoch {epoch}/{N_EPOCHS}: recon={stats['recon']:.4f}  KL={stats['kl']:.4f}  "
            f"loss={stats['loss']:.4f}  ELBO={stats['elbo']:.4f}"
        )

    with torch.no_grad():
        ctx = model.embed_context(cat, cont)
        mu, log_sigma_sq = model.encode(x, ctx)
        per_dim = gaussian_kl_per_dim(mu, log_sigma_sq).mean(dim=0)
    formatted = "  ".join(f"{v:.3f}" for v in per_dim.tolist())
    active = int((per_dim >= KL_WARN_THRESHOLD).sum())
    print(f"\n  per-dimension KL (nats):\n  [{formatted}]")
    print(f"  active dimensions (KL >= {KL_WARN_THRESHOLD}): {active}/{len(per_dim)}")

    # Conditioning sanity check: fresh model, 1 epoch, context zeroed throughout.
    print("\nInline ablation: 1 epoch with zeroed context embeddings (fresh model) ...")
    torch.manual_seed(seed)
    ablation = make_model(config, n_items, n_stores, cont.shape[1])
    ablation_opt = torch.optim.Adam(ablation.parameters(), lr=config.training.lr)
    zeroed = run_epoch(ablation, ablation_opt, x, cat, cont, zero_context=True)
    print(f"  zeroed-context epoch 1: ELBO={zeroed['elbo']:.4f}")
    print(f"  real-context   epoch 1: ELBO={first_epoch_elbo:.4f}")
    if zeroed["elbo"] < first_epoch_elbo:
        print("  Conditioning helps (zeroed context is worse). Milestone 3 OK.")
    else:
        print("  WARNING: zeroed context is not worse - conditioning may be miswired.")


if __name__ == "__main__":
    main()

"""Full CVAE training on the M5 FOODS pipeline (roadmap Phase 7).

Thin CLI over :func:`demand_vae.training.train_cvae`: loads the full windowed
splits (1.68M train / 425K val windows), builds the ConditionalVAE from the
config, and trains with KL annealing, free bits, per-dimension KL monitoring,
early stopping (patience 5 on validation ELBO), and best/last checkpoints.
Per-epoch metrics stream to stdout and results/training_log.csv.

Usage: python scripts/train.py [--config configs/default.yaml] [--epochs N]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import torch

from demand_vae.config import load_config
from demand_vae.data.pipeline import build_dataset
from demand_vae.models.cvae import ConditionalVAE
from demand_vae.training import train_cvae

REPO_ROOT = Path(__file__).resolve().parents[1]
PATIENCE = 5


def to_tensors(split):
    return (
        torch.from_numpy(split.x).float(),
        torch.from_numpy(split.context_cat),
        torch.from_numpy(split.context_cont).float(),
    )


def git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="override training.n_epochs")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config.training.seeds[0]
    n_epochs = args.epochs or config.training.n_epochs

    print(f"Phase 7 training | config={args.config} | seed={seed} | commit={git_commit_hash()}")
    t0 = time.time()
    splits, _, panel = build_dataset(config, REPO_ROOT / "data")
    train_data = to_tensors(splits["train"])
    val_data = to_tensors(splits["val"])
    print(
        f"data: {len(train_data[0]):,} train / {len(val_data[0]):,} val windows "
        f"[{time.time() - t0:.0f}s]"
    )

    model_kwargs = {
        "horizon": config.data.horizon_weeks,
        "latent_dim": config.model.latent_dim,
        "n_items": len(panel.item_categories),
        "n_stores": len(panel.store_categories),
        "n_continuous": train_data[2].shape[1],
        "encoder_hidden": list(config.model.encoder_hidden),
        "decoder_hidden": list(config.model.decoder_hidden),
        "item_embedding_dim": config.model.item_embedding_dim,
        "store_embedding_dim": config.model.store_embedding_dim,
        "decoder_likelihood": config.model.decoder_likelihood,
    }
    torch.manual_seed(seed)
    model = ConditionalVAE(**model_kwargs)
    print(
        f"model: {sum(p.numel() for p in model.parameters()):,} params | "
        f"decoder={model.decoder_likelihood} | K={model.latent_dim} | "
        f"epochs={n_epochs} (patience {PATIENCE})"
    )

    history = train_cvae(
        model,
        train_data,
        val_data,
        n_epochs=n_epochs,
        batch_size=config.training.batch_size,
        lr=config.training.lr,
        n_anneal_epochs=config.training.n_anneal_epochs,
        lambda_fb=config.training.lambda_fb,
        patience=PATIENCE,
        seed=seed,
        checkpoint_dir=REPO_ROOT / config.paths.checkpoints_dir,
        log_path=REPO_ROOT / config.paths.results_dir / "training_log.csv",
        config_dict=config.to_dict(),
        model_kwargs=model_kwargs,
    )

    best = max(history, key=lambda row: row["val_elbo"])
    print(
        f"\nTRAINING COMPLETE: {len(history)} epochs in {(time.time() - t0) / 60:.1f} min | "
        f"best val ELBO {best['val_elbo']:.4f} (epoch {best['epoch']}) | "
        f"checkpoints/best.pt + last.pt written | results/training_log.csv"
    )


if __name__ == "__main__":
    main()

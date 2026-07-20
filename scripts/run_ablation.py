"""Phase 9 ablation worker: train ONE variant, evaluate it, write its CSV.

Each ablation changes exactly one thing relative to the main CVAE (NB
decoder, K=16, annealing over 10 epochs, lambda_fb=0.5), trained under the
same budget the main model used (18 epochs, patience 5, seed 0) and pushed
through the identical Phase 8 two-level harness (seed 0, S=1000, all three
pre-registered cost ratios):

- gaussian_decoder  (RQ1): decoder_likelihood = gaussian
- unconditioned     (RQ2): context zeroed in BOTH training and evaluation
                    (same architecture/capacity; the embedding of index 0 and
                    zero continuous features act as a learned constant)
- no_annealing      : n_anneal_epochs = 0 (beta = 1 from the first epoch)
- latent_dim8       : latent_dim = 8

Config overrides are in-memory only — configs/default.yaml is never edited.
Idempotent: skips training if the ablation checkpoint exists, skips
evaluation if the CSV exists.

Usage:
  python scripts/run_ablation.py --ablation gaussian_decoder
  python scripts/run_ablation.py --summarize   # after all four: summary CSV + figure
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demand_vae.config import load_config
from demand_vae.data.pipeline import build_dataset
from demand_vae.eval.metrics import evaluate_two_level
from demand_vae.models.cvae import ConditionalVAE
from demand_vae.models.sampler import CVAESampler
from demand_vae.training import train_cvae

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = REPO_ROOT / "results" / "ablations"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
SEED = 0
N_EPOCHS = 18  # the main NB model's realized budget (early-stopped at 18)
PATIENCE = 5

ABLATIONS = ("gaussian_decoder", "unconditioned", "no_annealing", "latent_dim8")

# CSV name per spec (latent_dim8 writes latent_dim8.csv).
CSV_NAME = {
    "gaussian_decoder": "gaussian_decoder.csv",
    "unconditioned": "unconditioned.csv",
    "no_annealing": "no_annealing.csv",
    "latent_dim8": "latent_dim8.csv",
}


class ZeroContextSampler(CVAESampler):
    """CVAESampler that blinds the model to context (RQ2 evaluation side)."""

    def sample_batch(self, context_batch, n_scenarios: int) -> np.ndarray:
        zeroed = {
            "context_cat": np.zeros_like(np.asarray(context_batch["context_cat"])),
            "context_cont": np.zeros_like(
                np.asarray(context_batch["context_cont"]), dtype=np.float32
            ),
        }
        return super().sample_batch(zeroed, n_scenarios)


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


def to_tensors(split, zero_context: bool = False):
    x = torch.from_numpy(split.x).float()
    cat = torch.from_numpy(split.context_cat)
    cont = torch.from_numpy(split.context_cont).float()
    if zero_context:
        cat = torch.zeros_like(cat)
        cont = torch.zeros_like(cont)
    return x, cat, cont


def run_ablation(name: str) -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    # In-memory overrides only; the config file on disk is untouched.
    if name == "gaussian_decoder":
        config.model.decoder_likelihood = "gaussian"
    elif name == "no_annealing":
        config.training.n_anneal_epochs = 0
    elif name == "latent_dim8":
        config.model.latent_dim = 8
    zero_context = name == "unconditioned"

    checkpoint_path = CHECKPOINT_DIR / f"ablation_{name}.pt"
    csv_path = ABLATION_DIR / CSV_NAME[name]
    t0 = time.time()

    print(f"[{name}] building dataset ...", flush=True)
    splits, _, panel = build_dataset(config, REPO_ROOT / "data")
    model_kwargs = {
        "horizon": config.data.horizon_weeks,
        "latent_dim": config.model.latent_dim,
        "n_items": len(panel.item_categories),
        "n_stores": len(panel.store_categories),
        "n_continuous": splits["train"].context_cont.shape[1],
        "encoder_hidden": list(config.model.encoder_hidden),
        "decoder_hidden": list(config.model.decoder_hidden),
        "item_embedding_dim": config.model.item_embedding_dim,
        "store_embedding_dim": config.model.store_embedding_dim,
        "decoder_likelihood": config.model.decoder_likelihood,
    }

    if checkpoint_path.exists():
        print(f"[{name}] checkpoint exists, skipping training", flush=True)
    else:
        torch.manual_seed(SEED)
        model = ConditionalVAE(**model_kwargs)
        print(
            f"[{name}] training: {sum(p.numel() for p in model.parameters()):,} params, "
            f"decoder={model.decoder_likelihood}, K={model.latent_dim}, "
            f"anneal={config.training.n_anneal_epochs}, zero_context={zero_context}",
            flush=True,
        )
        run_dir = CHECKPOINT_DIR / f"ablation_{name}"
        train_cvae(
            model,
            to_tensors(splits["train"], zero_context),
            to_tensors(splits["val"], zero_context),
            n_epochs=N_EPOCHS,
            batch_size=config.training.batch_size,
            lr=config.training.lr,
            n_anneal_epochs=config.training.n_anneal_epochs,
            lambda_fb=config.training.lambda_fb,
            patience=PATIENCE,
            seed=SEED,
            checkpoint_dir=run_dir,
            log_path=ABLATION_DIR / f"{name}_training_log.csv",
            config_dict=config.to_dict(),
            model_kwargs=model_kwargs,
        )
        shutil.copy(run_dir / "best.pt", checkpoint_path)
        print(f"[{name}] best checkpoint -> {checkpoint_path.name}", flush=True)

    if csv_path.exists():
        print(f"[{name}] result CSV exists, skipping evaluation", flush=True)
    else:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        sampler_cls = ZeroContextSampler if zero_context else CVAESampler
        sampler = sampler_cls.from_checkpoint(checkpoint_path, seed=SEED)
        print(f"[{name}] evaluating {len(splits['test']):,} test windows ...", flush=True)
        result = evaluate_two_level(sampler, splits["test"], config, pit_seed=SEED)
        rows = [
            {
                "model": f"cvae_{name}",
                "c_u": l2["c_u"],
                "c_o": l2["c_o"],
                "mean_realized_cost": l2["mean_realized_cost"],
                "service_level": l2["service_level"],
                "fill_rate": l2["fill_rate"],
                "mean_overstock": l2["mean_overstock"],
                "crps": result["level1"]["crps"],
                "coverage_50": result["level1"]["coverage_50"],
                "coverage_90": result["level1"]["coverage_90"],
                "n_windows": result["n_windows"],
                "n_scenarios": result["n_scenarios"],
                "seed": SEED,
                "commit": git_commit_hash(),
            }
            for l2 in result["level2"]
        ]
        ABLATION_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[{name}] DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


def summarize() -> None:
    """ablation_summary.csv + figures/ablation_comparison.png."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Main CVAE seed-0 row from the Phase 8 artifacts.
    main = json.loads((REPO_ROOT / "results" / "full_eval" / "cvae_seed0.json").read_text())
    cost = {(l2["c_u"], l2["c_o"]): l2["mean_realized_cost"] for l2 in main["level2"]}
    rows = [
        {
            "model": "cvae_main (NB, K=16, annealed, conditioned)",
            "cost_3_1": cost[(3, 1)],
            "cost_1_1": cost[(1, 1)],
            "cost_1_3": cost[(1, 3)],
            "crps": main["level1"]["crps"],
            "coverage_90": main["level1"]["coverage_90"],
        }
    ]
    for name in ABLATIONS:
        table = pd.read_csv(ABLATION_DIR / CSV_NAME[name]).set_index(["c_u", "c_o"])
        rows.append(
            {
                "model": f"cvae_{name}",
                "cost_3_1": table.loc[(3, 1), "mean_realized_cost"],
                "cost_1_1": table.loc[(1, 1), "mean_realized_cost"],
                "cost_1_3": table.loc[(1, 3), "mean_realized_cost"],
                "crps": table["crps"].iloc[0],
                "coverage_90": table["coverage_90"].iloc[0],
            }
        )
    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(ABLATION_DIR / "ablation_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Grouped bar chart: one group per cost ratio, one bar per model.
    labels = ["main\n(NB,K=16)", "gaussian\ndecoder", "uncond-\nitioned", "no\nannealing", "K=8"]
    ratio_cols = [("cost_3_1", "3:1"), ("cost_1_1", "1:1"), ("cost_1_3", "1:3")]
    n_models = len(summary)
    n_ratios = len(ratio_cols)
    width = 0.8 / n_ratios
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(n_models)
    for i, (col, ratio_label) in enumerate(ratio_cols):
        offset = (i - (n_ratios - 1) / 2) * width
        bars = ax.bar(x + offset, summary[col], width=width, label=f"cᵤ:cₒ = {ratio_label}")
        ax.bar_label(bars, fmt="%.1f", fontsize=7, rotation=90, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean realized cost (lower is better)")
    ax.set_title("Ablations vs main CVAE — realized cost across cost ratios (test, seed 0)")
    ax.legend()
    fig.tight_layout()
    out = REPO_ROOT / "figures" / "ablation_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.ablation:
        run_ablation(args.ablation)
    else:
        parser.error("pass --ablation NAME or --summarize")


if __name__ == "__main__":
    main()

"""Phase 8 figures from the committed results tables.

Produces (into figures/): cost_comparison.png (the money plot),
coverage_comparison.png, training_curve.png, per_dim_kl.png.

Usage: python scripts/plot_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = ["empirical", "gaussian", "poisson", "negative_binomial", "cvae", "cvae_marginal"]
MODEL_LABELS = {
    "empirical": "Empirical",
    "gaussian": "Gaussian",
    "poisson": "Poisson",
    "negative_binomial": "Neg. Binomial",
    "cvae": "CVAE (aggregate)",
    "cvae_marginal": "CVAE (marginal)",
}


def make_figures(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Render all four figures; returns the written paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(results_dir / "full_comparison_summary.csv")
    log = pd.read_csv(results_dir / "training_log.csv")
    written = []

    # 1. Money plot: mean realized cost per model per cost ratio.
    fig, ax = plt.subplots(figsize=(9, 5))
    ratios = sorted(
        set(zip(summary["c_u"], summary["c_o"], strict=True)), key=lambda r: -r[0] / r[1]
    )
    models = [m for m in MODEL_ORDER if m in set(summary["model_protocol"])]
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        sub = summary[summary["model_protocol"] == model].set_index(["c_u", "c_o"])
        xs = [j + i * width for j in range(len(ratios))]
        means = [sub.loc[r, "mean_realized_cost_mean"] for r in ratios]
        stds = [sub.loc[r, "mean_realized_cost_std"] for r in ratios]
        ax.bar(xs, means, width=width, yerr=stds, capsize=2, label=MODEL_LABELS[model])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(ratios))])
    ax.set_xticklabels([f"{c_u}:{c_o}" for c_u, c_o in ratios])
    ax.set_xlabel("cost ratio  cᵤ : cₒ")
    ax.set_ylabel("mean realized cost (lower is better)")
    ax.set_title("Realized newsvendor cost on the M5 test year (S=1000, mean ± sd over 3 seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "cost_comparison.png"
    fig.savefig(path, dpi=150)
    written.append(path)

    # 2. Coverage at nominal 90%.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    agg = summary[(summary["c_u"] == 1) & (summary["c_o"] == 1)]
    agg = agg[~agg["model_protocol"].str.endswith("_marginal")]
    agg = agg.set_index("model_protocol").reindex(
        [m for m in MODEL_ORDER if m in agg["model_protocol"].values if not m.endswith("_marginal")]
    )
    labels = [MODEL_LABELS[m] for m in agg.index]
    ax.bar(labels, agg["coverage_90_mean"], yerr=agg["coverage_90_std"], capsize=3)
    ax.axhline(0.9, color="k", ls="--", lw=1, label="nominal 0.90")
    ax.set_ylabel("empirical coverage of the 90% interval")
    ax.set_title("Calibration on aggregate-horizon totals (test year)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "coverage_comparison.png"
    fig.savefig(path, dpi=150)
    written.append(path)

    # 3. Training curve.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(log["epoch"], log["train_elbo"], marker="o", ms=3, label="train ELBO")
    ax.plot(log["epoch"], log["val_elbo"], marker="s", ms=3, label="val ELBO")
    best = log.loc[log["val_elbo"].idxmax()]
    ax.axvline(best["epoch"], color="k", ls=":", lw=1, label=f"best epoch ({int(best['epoch'])})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ELBO (higher is better)")
    ax.set_title("CVAE-NB training on 1.68M windows (KL annealing over epochs 1-10)")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "training_curve.png"
    fig.savefig(path, dpi=150)
    written.append(path)

    # 4. Per-dimension KL summary across training (min/mean/max envelope).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(
        log["epoch"], log["kl_dim_min"], log["kl_dim_max"], alpha=0.25, label="min-max across dims"
    )
    ax.plot(log["epoch"], log["kl_dim_mean"], marker="o", ms=3, label="mean per-dim KL")
    ax.axhline(0.1, color="r", ls="--", lw=1, label="collapse threshold (0.1 nats)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-dimension KL (nats)")
    ax.set_title(f"Latent usage: all 16 dims active at best epoch ({int(best['epoch'])})")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "per_dim_kl.png"
    fig.savefig(path, dpi=150)
    written.append(path)
    plt.close("all")
    return written


def main() -> None:
    written = make_figures(REPO_ROOT / "results", REPO_ROOT / "figures")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

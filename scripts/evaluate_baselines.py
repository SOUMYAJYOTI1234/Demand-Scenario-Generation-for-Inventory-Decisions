"""Evaluate the four classical baselines on the M5 test split (roadmap Phase 5).

Runs each baseline through the identical two-level harness — Level 1 (CRPS,
PIT calibration, interval coverage) and Level 2 (SAA newsvendor at the three
pre-registered cost ratios; realized cost, service level, fill rate,
overstock) — and writes the first real cost table. This is the number the
CVAE will be judged against.

Requires the M5 data in data/raw/. Output: results/baseline_comparison.csv
(tidy: one row per model x cost ratio) and a stdout summary.

Usage: python scripts/evaluate_baselines.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd

from demand_vae.baselines.classical import (
    EmpiricalResampler,
    GaussianSampler,
    NegativeBinomialSampler,
    PoissonSampler,
)
from demand_vae.config import load_config
from demand_vae.data.pipeline import aggregate_weekly, build_windows, load_raw_m5, temporal_split
from demand_vae.eval.metrics import evaluate_two_level

REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINES = {
    "empirical": EmpiricalResampler,
    "gaussian": GaussianSampler,
    "poisson": PoissonSampler,
    "negative_binomial": NegativeBinomialSampler,
}


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
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "baseline_comparison.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.training.seeds[0]
    horizon = config.data.horizon_weeks

    print("Loading M5 and building the weekly panel + windows ...")
    t0 = time.time()
    panel = aggregate_weekly(load_raw_m5(REPO_ROOT / "data"), config)
    spans = temporal_split(panel, config)
    splits = build_windows(panel, config, spans)
    split = splits[args.split]
    # Classical baselines ignore continuous context features, so the scaler is
    # irrelevant here; per-series fits below use the train span only.
    print(
        f"  {panel.n_series} series; evaluating on {args.split}: {len(split):,} windows "
        f"(S={config.decision.n_scenarios}, seed={seed})  [{time.time() - t0:.1f}s]"
    )

    rows = []
    for name, cls in BASELINES.items():
        t1 = time.time()
        sampler = cls(horizon=horizon, seed=seed).fit(panel, spans["train"])
        result = evaluate_two_level(sampler, split, config, pit_seed=seed)
        elapsed = time.time() - t1
        level1 = result["level1"]
        print(
            f"  {name:>18}: CRPS={level1['crps']:.3f}  cov50={level1['coverage_50']:.3f}  "
            f"cov90={level1['coverage_90']:.3f}  [{elapsed:.0f}s]"
        )
        for l2 in result["level2"]:
            rows.append(
                {
                    "model": name,
                    "split": args.split,
                    "c_u": l2["c_u"],
                    "c_o": l2["c_o"],
                    "mean_realized_cost": l2["mean_realized_cost"],
                    "service_level": l2["service_level"],
                    "fill_rate": l2["fill_rate"],
                    "mean_overstock": l2["mean_overstock"],
                    "crps": level1["crps"],
                    "coverage_50": level1["coverage_50"],
                    "coverage_90": level1["coverage_90"],
                    "n_windows": result["n_windows"],
                    "n_scenarios": result["n_scenarios"],
                    "seed": seed,
                    "commit": git_commit_hash(),
                }
            )

    table = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\n=== Level 2: mean realized cost (lower is better) ===")
    cost = table.pivot(index="model", columns=["c_u", "c_o"], values="mean_realized_cost")
    print(cost.round(2).to_string())
    print("\n=== Level 2: service level ===")
    service = table.pivot(index="model", columns=["c_u", "c_o"], values="service_level")
    print(service.round(3).to_string())
    print("\n=== Level 1 (on aggregate-horizon totals) ===")
    level1_table = table.drop_duplicates("model").set_index("model")[
        ["crps", "coverage_50", "coverage_90"]
    ]
    print(level1_table.round(3).to_string())

    best = cost.mean(axis=1).idxmin()
    print(f"\nBest mean realized cost across ratios: {best} — the number the CVAE must beat.")


if __name__ == "__main__":
    main()

"""Phase 8 worker: evaluate ONE model at ONE seed on the full test split.

Runs the identical two-level harness used for the Phase 5 baselines. For the
CVAE it additionally runs the RQ4 marginal decision protocol from the same
scenario draws. Writes per-run artifacts to results/full_eval/:

- {model}_seed{seed}.json  — Level-1 + Level-2 (+ marginal) metrics,
  PIT histogram (10 bins), run metadata.
- {model}_seed{seed}_costs.npz — per-window aggregate-protocol realized
  costs per cost ratio (the paired-bootstrap input; git-ignored, ~8 MB).

Designed to be launched in parallel, one process per (model, seed); the
aggregator (scripts/aggregate_results.py) collates every run afterwards.

Usage: python scripts/evaluate_full.py --model cvae --seed 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from demand_vae.baselines.classical import (
    EmpiricalResampler,
    GaussianSampler,
    NegativeBinomialSampler,
    PoissonSampler,
)
from demand_vae.config import load_config
from demand_vae.data.pipeline import build_dataset, temporal_split
from demand_vae.eval.metrics import evaluate_two_level
from demand_vae.models.sampler import CVAESampler

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "full_eval"
BASELINES = {
    "empirical": EmpiricalResampler,
    "gaussian": GaussianSampler,
    "poisson": PoissonSampler,
    "negative_binomial": NegativeBinomialSampler,
}
PIT_BINS = 10


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
    parser.add_argument("--model", required=True, choices=[*BASELINES, "cvae"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "best.pt"))
    args = parser.parse_args()
    config = load_config(args.config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    print(f"[{args.model} seed {args.seed}] building dataset ...", flush=True)
    splits, _, panel = build_dataset(config, REPO_ROOT / "data")
    test = splits["test"]

    if args.model == "cvae":
        sampler = CVAESampler.from_checkpoint(args.checkpoint, seed=args.seed)
        protocols = ("aggregate", "marginal")  # RQ4 from the same draws
    else:
        spans = temporal_split(panel, config)
        sampler = BASELINES[args.model](horizon=config.data.horizon_weeks, seed=args.seed).fit(
            panel, spans["train"]
        )
        protocols = ("aggregate",)

    print(f"[{args.model} seed {args.seed}] evaluating {len(test):,} windows ...", flush=True)
    result = evaluate_two_level(
        sampler,
        test,
        config,
        pit_seed=args.seed,
        protocols=protocols,
        collect_per_window=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_seed{args.seed}"
    pit_hist = np.histogram(result["pit"], bins=PIT_BINS, range=(0.0, 1.0))[0]
    payload = {
        "model": args.model,
        "seed": args.seed,
        "commit": git_commit_hash(),
        "n_windows": result["n_windows"],
        "n_scenarios": result["n_scenarios"],
        "level1": result["level1"],
        "level2": result["level2"],
        "pit_hist": pit_hist.tolist(),
        "seconds": round(time.time() - t0, 1),
    }
    if "level2_marginal" in result:
        payload["level2_marginal"] = result["level2_marginal"]
    with open(OUT_DIR / f"{stem}.json", "w") as f:
        json.dump(payload, f, indent=1)
    np.savez_compressed(
        OUT_DIR / f"{stem}_costs.npz",
        **{key.replace(":", "_"): arr for key, arr in result["per_window_cost"].items()},
    )
    print(
        f"[{args.model} seed {args.seed}] DONE in {(time.time() - t0) / 60:.1f} min -> {stem}.json",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Phase 8 aggregator: collate all evaluation runs into the results tables.

Inputs: results/baseline_comparison.csv (Phase 5 baseline seed-0 run, reused
per the no-rerun rule) + results/full_eval/*.json / *_costs.npz (Phase 8
worker runs). Outputs:

- results/full_comparison.csv          — model x protocol x seed x ratio rows
- results/full_comparison_summary.csv  — seed-averaged mean ± std
- results/paired_bootstrap.csv         — CVAE vs each baseline, per ratio,
  paired on windows at PAIRING_SEED (the seed where every model has
  per-window costs; Phase 5 stored none for seed 0)
- results/calibration_summary.csv      — PIT histogram + coverage per model

Also prints the RQ3 Spearman rank agreement (CRPS ranking vs cost ranking)
per cost ratio to stdout.

Usage: python scripts/aggregate_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from demand_vae.eval.metrics import paired_bootstrap_ci, spearman_rank_correlation

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
EVAL_DIR = RESULTS / "full_eval"
BASELINE_MODELS = ["empirical", "gaussian", "poisson", "negative_binomial"]
ALL_SEEDS = [0, 1, 2]
PAIRING_SEED = 1  # every model (incl. baselines) has per-window costs here
N_BOOT = 10_000
PIT_BINS = 10

METRICS = [
    "mean_realized_cost",
    "service_level",
    "fill_rate",
    "mean_overstock",
    "crps",
    "coverage_90",
]


def rows_from_phase5_csv() -> list[dict]:
    """Baseline seed-0 rows, reused from the Phase 5 table (no rerun)."""
    table = pd.read_csv(RESULTS / "baseline_comparison.csv")
    rows = []
    for _, r in table.iterrows():
        rows.append(
            {
                "model": r["model"],
                "protocol": "aggregate",
                "seed": int(r["seed"]),
                "c_u": int(r["c_u"]),
                "c_o": int(r["c_o"]),
                "mean_realized_cost": r["mean_realized_cost"],
                "service_level": r["service_level"],
                "fill_rate": r["fill_rate"],
                "mean_overstock": r["mean_overstock"],
                "crps": r["crps"],
                "coverage_90": r["coverage_90"],
            }
        )
    return rows


def rows_from_json(payload: dict) -> list[dict]:
    rows = []
    for protocol, key in (("aggregate", "level2"), ("marginal", "level2_marginal")):
        for l2 in payload.get(key, []):
            rows.append(
                {
                    "model": payload["model"],
                    "protocol": protocol,
                    "seed": payload["seed"],
                    "c_u": l2["c_u"],
                    "c_o": l2["c_o"],
                    "mean_realized_cost": l2["mean_realized_cost"],
                    "service_level": l2["service_level"],
                    "fill_rate": l2["fill_rate"],
                    "mean_overstock": l2["mean_overstock"],
                    "crps": payload["level1"]["crps"],
                    "coverage_90": payload["level1"]["coverage_90"],
                }
            )
    return rows


def main() -> None:
    payloads = {}
    for path in sorted(EVAL_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        payloads[(payload["model"], payload["seed"])] = payload

    rows = rows_from_phase5_csv()
    for payload in payloads.values():
        rows.extend(rows_from_json(payload))
    table = pd.DataFrame(rows)
    # Row identity: (model, protocol) — the CVAE's marginal protocol reports
    # as model "cvae" protocol "marginal" (aka cvae_marginal in the report).
    table["model_protocol"] = np.where(
        table["protocol"] == "marginal", table["model"] + "_marginal", table["model"]
    )
    table = table.sort_values(["model_protocol", "c_u", "c_o", "seed"], ignore_index=True)
    table.to_csv(RESULTS / "full_comparison.csv", index=False)

    summary = table.groupby(["model_protocol", "c_u", "c_o"])[METRICS].agg(["mean", "std"]).round(4)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "full_comparison_summary.csv", index=False)

    # Paired bootstrap: CVAE (aggregate) vs each baseline, same windows, same seed.
    boot_rows = []
    cvae_costs = np.load(EVAL_DIR / f"cvae_seed{PAIRING_SEED}_costs.npz")
    for baseline in BASELINE_MODELS:
        base_costs = np.load(EVAL_DIR / f"{baseline}_seed{PAIRING_SEED}_costs.npz")
        for key in cvae_costs.files:  # e.g. "3_1"
            c_u, c_o = key.split("_")
            ci = paired_bootstrap_ci(cvae_costs[key], base_costs[key], n_boot=N_BOOT, seed=0)
            boot_rows.append(
                {
                    "comparison": f"cvae_minus_{baseline}",
                    "c_u": int(c_u),
                    "c_o": int(c_o),
                    "pairing_seed": PAIRING_SEED,
                    "n_boot": N_BOOT,
                    **{k: round(v, 4) for k, v in ci.items()},
                }
            )
    pd.DataFrame(boot_rows).to_csv(RESULTS / "paired_bootstrap.csv", index=False)

    # Calibration summary: PIT histogram (density) + coverage, seed-averaged
    # over the runs that stored PIT (Phase 8 runs; Phase 5 seed 0 stored none).
    calib_rows = []
    models = BASELINE_MODELS + ["cvae"]
    for model in models:
        hists, cov50, cov90 = [], [], []
        for seed in ALL_SEEDS:
            payload = payloads.get((model, seed))
            if payload is None:
                continue
            hist = np.asarray(payload["pit_hist"], dtype=float)
            hists.append(hist / hist.sum() * PIT_BINS)  # density: flat == 1.0
            cov50.append(payload["level1"]["coverage_50"])
            cov90.append(payload["level1"]["coverage_90"])
        row = {
            "model": model,
            "n_seeds": len(hists),
            "coverage_50": round(float(np.mean(cov50)), 4),
            "coverage_90": round(float(np.mean(cov90)), 4),
        }
        mean_hist = np.mean(hists, axis=0)
        row.update({f"pit_bin_{i}": round(float(v), 4) for i, v in enumerate(mean_hist)})
        calib_rows.append(row)
    pd.DataFrame(calib_rows).to_csv(RESULTS / "calibration_summary.csv", index=False)

    # RQ3: does the statistical ranking (CRPS) match the decision ranking (cost)?
    print("\n=== RQ3: Spearman rank agreement (CRPS vs realized cost), aggregate protocol ===")
    agg = summary[~summary["model_protocol"].str.endswith("_marginal")]
    for (c_u, c_o), group in agg.groupby(["c_u", "c_o"]):
        rho = spearman_rank_correlation(
            group["crps_mean"].to_numpy(), group["mean_realized_cost_mean"].to_numpy()
        )
        print(f"  {c_u}:{c_o} -> rho = {rho:+.3f}   (1.0 = identical model rankings)")

    print("\n=== Level 2: mean realized cost (seed mean +/- std) ===")
    pivot = summary.pivot(
        index="model_protocol", columns=["c_u", "c_o"], values="mean_realized_cost_mean"
    )
    print(pivot.round(3).to_string())
    print(f"\nWrote 4 tables to {RESULTS}")


if __name__ == "__main__":
    main()

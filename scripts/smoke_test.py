"""End-to-end smoke test of the vertical slice — no data, no torch, no fitting.

Proves the pipeline connects: synthetic demand -> trivial sampler ->
aggregate-horizon SAA newsvendor -> realized cost. This is the thin slice the
roadmap requires to exist before any real modeling ("data -> baseline ->
newsvendor -> cost number"); real data and models replace the fakes in
Phases 4-6 without changing the wiring.

Usage: python scripts/smoke_test.py  (exits 0 on success)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from demand_vae.config import load_config
from demand_vae.eval.metrics import evaluate_decisions
from demand_vae.models.trivial import FixedRatePoissonSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUE_RATE = 20.0  # weekly demand rate of the synthetic "world"
N_TEST_CONTEXTS = 50


def main() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    horizon = config.data.horizon_weeks
    n_scenarios = config.decision.n_scenarios
    rng = np.random.default_rng(0)

    # Fake world: actual demand is Poisson(TRUE_RATE) per week; contexts are
    # placeholders until the Phase 4 pipeline defines the real feature vector.
    contexts = [None] * N_TEST_CONTEXTS
    actual_windows = rng.poisson(TRUE_RATE, size=(N_TEST_CONTEXTS, horizon))

    # Trivial "model": Poisson at the true rate (an oracle-rate sampler, so
    # the numbers below should look sane, not impressive).
    sampler = FixedRatePoissonSampler(horizon=horizon, rate=TRUE_RATE, seed=1)

    print(
        f"Smoke test: H={horizon}, S={n_scenarios}, {N_TEST_CONTEXTS} decisions, "
        f"aggregate-horizon newsvendor\n"
    )
    columns = ["c_u:c_o", "fractile", "mean cost", "service", "overstock"]
    widths = [8, 8, 10, 8, 9]
    print(" | ".join(f"{c:>{w}}" for c, w in zip(columns, widths, strict=True)))
    print("-" * 56)
    for c_u, c_o in config.decision.cost_ratios:
        result = evaluate_decisions(
            sampler, contexts, actual_windows, c_u=c_u, c_o=c_o, n_scenarios=n_scenarios
        )
        fractile = c_u / (c_u + c_o)
        print(
            f"{c_u:>4}:{c_o:<3} | {fractile:>8.3f} | {result['mean_realized_cost']:>10.2f} | "
            f"{result['service_level']:>8.2f} | {result['mean_overstock']:>9.2f}"
        )

    print("\nVertical slice OK: sampler -> SAA newsvendor -> realized cost.")


if __name__ == "__main__":
    main()

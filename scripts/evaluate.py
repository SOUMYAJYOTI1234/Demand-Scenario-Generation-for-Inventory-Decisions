"""Produce the full two-level evaluation table for any model (roadmap Phase 8) — stub CLI.

Will become: load config + model (checkpoint or fitted baseline) -> generate
S scenarios per test context -> Level 1 (NLL, CRPS, PIT, coverage) and
Level 2 (realized cost, service level, fill rate at all pre-registered cost
ratios, paired bootstrap CIs) -> write tables to results/.

Usage: python scripts/evaluate.py --config configs/default.yaml --model <name-or-checkpoint>
"""

from __future__ import annotations

import argparse

from demand_vae.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config")
    parser.add_argument("--model", default=None, help="Model name or checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    ratios = [tuple(r) for r in config.decision.cost_ratios]
    print(f"Loaded config: S={config.decision.n_scenarios}, cost ratios={ratios}")
    raise NotImplementedError(
        "TODO(Phase 8): two-level evaluation harness over the test split "
        "(validated on baselines before the CVAE exists — design doc §3.3)."
    )


if __name__ == "__main__":
    main()

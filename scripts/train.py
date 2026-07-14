"""Train a demand model from a config file (roadmap Phases 6-7) — stub CLI.

Will become: load config -> build data splits (Phase 4 pipeline) -> construct
the model (CVAE or ablation variant per config) -> train with ELBO objective,
KL monitoring, early stopping on validation ELBO -> save best/last checkpoints
with the config alongside.

Usage: python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from demand_vae.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    print(
        f"Loaded config: H={config.data.horizon_weeks}, K={config.model.latent_dim}, "
        f"decoder={config.model.decoder}"
    )
    raise NotImplementedError(
        "TODO(Phase 6/7): training loop — data pipeline, CVAE, ELBO optimization, "
        "KL monitoring, checkpoints."
    )


if __name__ == "__main__":
    main()

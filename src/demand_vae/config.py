"""Configuration loading.

Every experiment is fully described by a YAML file under ``configs/`` (roadmap
Phase 3: no hyperparameter lives hard-coded in a script). This module loads
such a file into a dot-accessible namespace and sanity-checks the values the
rest of the pipeline relies on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Dot-accessible, read-mostly view over a nested dict.

    Nested mappings become nested ``Config`` objects; everything else is left
    as loaded by ``yaml.safe_load``.
    """

    def __init__(self, data: dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value.to_dict() if isinstance(value, Config) else value
            for key, value in vars(self).items()
        }

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"


def load_config(path: str | Path) -> Config:
    """Load a YAML experiment config and validate its core decision values.

    Parameters
    ----------
    path:
        Path to a YAML file shaped like ``configs/default.yaml``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level")
    config = Config(raw)
    validate(config)
    return config


def validate(config: Config) -> None:
    """Check the invariants the decision/evaluation layers depend on.

    Raises ``ValueError`` with a specific message on the first violation.
    """
    horizon = config.data.horizon_weeks
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"data.horizon_weeks must be a positive integer, got {horizon!r}")

    n_scenarios = config.decision.n_scenarios
    if not isinstance(n_scenarios, int) or n_scenarios < 1:
        raise ValueError(f"decision.n_scenarios must be a positive integer, got {n_scenarios!r}")

    ratios = config.decision.cost_ratios
    if not ratios:
        raise ValueError("decision.cost_ratios must list at least one [c_u, c_o] pair")
    for ratio in ratios:
        if len(ratio) != 2 or any(c <= 0 for c in ratio):
            raise ValueError(f"Each cost ratio must be a pair of positive costs, got {ratio!r}")

    latent_dim = config.model.latent_dim
    if not isinstance(latent_dim, int) or latent_dim < 1:
        raise ValueError(f"model.latent_dim must be a positive integer, got {latent_dim!r}")

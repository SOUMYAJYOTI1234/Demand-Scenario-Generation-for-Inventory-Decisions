"""Tests for the config loader and the frozen design decisions in default.yaml."""

from pathlib import Path

import pytest

from demand_vae.config import Config, load_config, validate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class TestLoadDefaultConfig:
    def test_loads(self):
        config = load_config(DEFAULT_CONFIG)
        assert isinstance(config, Config)

    def test_frozen_design_decisions(self):
        """The binding decisions from the design document §10 must be present
        and unchanged unless the design log says otherwise."""
        config = load_config(DEFAULT_CONFIG)
        assert config.data.horizon_weeks == 4  # H = 4
        assert config.decision.n_scenarios == 1000  # S = 1000
        assert [tuple(r) for r in config.decision.cost_ratios] == [(3, 1), (1, 1), (1, 3)]
        assert config.model.latent_dim == 16  # K, within the design range 8-16
        assert config.model.encoder_hidden == [128, 64]
        assert config.model.decoder_hidden == [64, 128]
        assert config.model.decoder == "negative_binomial"
        assert config.decision.protocol == "aggregate_horizon"
        assert config.data.category == "FOODS"
        assert config.training.seeds == [0, 1, 2]

    def test_dot_and_item_access(self):
        config = load_config(DEFAULT_CONFIG)
        assert config["data"]["horizon_weeks"] == config.data.horizon_weeks
        assert "decision" in config

    def test_to_dict_roundtrip(self):
        config = load_config(DEFAULT_CONFIG)
        as_dict = config.to_dict()
        assert as_dict["data"]["horizon_weeks"] == 4
        assert isinstance(as_dict["decision"], dict)


class TestLoadErrors:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config(REPO_ROOT / "configs" / "does-not-exist.yaml")

    def test_non_mapping_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(bad)


class TestValidate:
    def _valid(self) -> Config:
        return load_config(DEFAULT_CONFIG)

    def test_rejects_bad_horizon(self):
        config = self._valid()
        config.data.horizon_weeks = 0
        with pytest.raises(ValueError, match="horizon_weeks"):
            validate(config)

    def test_rejects_bad_scenario_count(self):
        config = self._valid()
        config.decision.n_scenarios = -5
        with pytest.raises(ValueError, match="n_scenarios"):
            validate(config)

    def test_rejects_bad_cost_ratio(self):
        config = self._valid()
        config.decision.cost_ratios = [[3, 0]]
        with pytest.raises(ValueError, match="cost ratio"):
            validate(config)

    def test_rejects_bad_latent_dim(self):
        config = self._valid()
        config.model.latent_dim = 0
        with pytest.raises(ValueError, match="latent_dim"):
            validate(config)

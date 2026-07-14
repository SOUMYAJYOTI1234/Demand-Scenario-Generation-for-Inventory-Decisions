# Experiments

Per-run configs and small result JSONs (roadmap Phases 7 and 9). Every
reported number must trace to (config file, seed, commit hash). Large
artifacts (checkpoints, wandb runs) are git-ignored; only small config/result
files live here.

Planned layout:
- `experiments/<run-name>/config.yaml` + `results.json`
- `experiments/ablations/` — decoder likelihood, conditioning on/off,
  KL-annealing/β, latent-dim sweep (Phase 9).

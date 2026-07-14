# demand-scenario-vae

**Conditional VAE demand scenario generation for inventory decisions — evaluated on realized newsvendor cost, not just forecast accuracy.**

> **Status: baselines complete (Phases 1–5).** The M5 FOODS weekly pipeline
> (temporal splits, leakage guards), the four classical baselines, the SAA
> newsvendor, and the two-level evaluation harness are implemented and
> tested; the first realized-cost table lives in
> `results/baseline_comparison.csv`. The CVAE itself (Phase 6) is next —
> its stubs name the roadmap phase that fills them in.

## Project blueprint

Learn the conditional distribution of 4-week retail demand windows given
context (category, store, week-of-year/events, relative price, recent lags)
with a conditional VAE trained on the ELBO, using a Negative Binomial decoder
to respect discrete, over-dispersed demand. Generate 1,000-scenario sets per
test context, solve the aggregate-horizon newsvendor by sample average
approximation at three cost ratios, and evaluate every model — CVAE and four
classical baselines sharing one sampler interface — on both distributional
metrics (NLL, CRPS, calibration) and realized decision cost with paired
statistics over three seeds. The contribution is the decision-level empirical
answer, not model novelty; a rigorous negative result is an acceptable outcome
by design. *(Design document, Section 10.)*

## Quickstart

```bash
pip install -e ".[dev]"        # install package + dev tools
pytest                          # unit tests (newsvendor math, config loader)
python scripts/smoke_test.py    # end-to-end vertical slice, no data needed
```

The smoke test wires synthetic demand through a trivial sampler → SAA
newsvendor → realized cost at all three pre-registered cost ratios, proving
the pipeline connects before any real model exists.

For later phases (requires Kaggle credentials — see [data/README.md](data/README.md)):

```bash
python scripts/download_data.py                     # M5 dataset -> data/raw/
python scripts/train.py --config configs/default.yaml     # Phase 6/7 (stub)
python scripts/evaluate.py --config configs/default.yaml  # Phase 8 (stub)
```

## Repository map

| Path | Purpose |
|---|---|
| `configs/` | One YAML per experiment; `default.yaml` holds the frozen design decisions (H=4, K=8, S=1000, cost ratios 3:1/1:1/1:3) |
| `src/demand_vae/data/` | M5 pipeline: weekly aggregation, features, windowing, temporal splits + leakage validation — **implemented + tested** |
| `src/demand_vae/models/` | `BaseSampler` interface (`sample` / vectorized `sample_batch`), trivial samplers, CVAE *(Phase 6, stub)* |
| `src/demand_vae/baselines/` | Empirical resampling, Gaussian, Poisson, fitted NB (the pivotal baseline) — **implemented + tested** |
| `src/demand_vae/opt/` | SAA newsvendor: critical fractile, scalar + batch solver, realized cost — **implemented + tested** |
| `src/demand_vae/eval/` | Two-level harness: CRPS, PIT, coverage + realized cost/service/fill/overstock — **implemented + tested** (NLL in Phase 8) |
| `scripts/` | Thin CLIs: download, train, evaluate, smoke test |
| `tests/` | pytest: SAA quantile vs closed form, config loader |
| `notebooks/` | Exploratory only; numbered (`01-eda.ipynb`, Phase 4) |
| `experiments/`, `results/`, `figures/` | Per-run records, final tables, final figures |
| `reports/` | LaTeX report source (Phase 10) |
| `docs/` | [Design document](docs/project-design-document.md) (source of truth) · [Roadmap](docs/full-project-roadmap.md) · [Literature review](docs/literature-review.md) · [Design-decisions log](docs/design-decisions.md) |

## Method (one line per stage)

1. **Data** — M5 Walmart unit sales, FOODS category, aggregated to weekly, temporal train/val/test split.
2. **Model** — CVAE: encoder q(z|x,y), NB decoder p(x|z,y), context y in both networks; classical baselines behind the same `sample(context, S)` interface.
3. **Decision** — aggregate-horizon newsvendor: order Q = empirical critical-fractile quantile of S=1000 scenario sums (SAA).
4. **Evaluation** — Level 1: NLL, CRPS, calibration; Level 2: realized cost, service level, fill rate across cost ratios, paired bootstrap over 3 seeds.

## License

MIT — see [LICENSE](LICENSE).

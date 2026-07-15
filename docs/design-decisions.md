# Design Decisions Log

Running log of decisions made during implementation, with reasons. The frozen
pre-implementation decisions live in
[project-design-document.md](project-design-document.md) §10; this file records
everything decided *after* that document, in chronological order. Any scope
addition must displace something explicitly here (anti-scope-creep rule,
design doc §4).

---

## 2026-07-09 — Scaffold (roadmap Phases 1–3, thin slice)

- **Repo layout** follows roadmap Phase 3 exactly; `requirements.txt` omitted
  (roadmap marks it optional) — `pyproject.toml` is the single source of
  dependencies.
- **SAA quantile convention:** `solve_newsvendor_saa` returns the
  ⌈S·τ⌉-th order statistic (`np.quantile(..., method="inverted_cdf")`),
  which is the exact optimizer of the discrete SAA newsvendor objective —
  not an interpolated quantile. Tested against closed-form Normal and Poisson
  fractiles in `tests/test_newsvendor.py`.
- **Trivial samplers** (`ConstantSampler`, `FixedRatePoissonSampler`) exist
  only to make the vertical slice runnable before Phase 5; they are *not*
  baselines and take no fitted parameters.
- **`evaluate_decisions` implemented early** (Phase 8 territory) but only the
  realized-cost/service-level/overstock arithmetic — the smoke test needs a
  number. CRPS/PIT/coverage/NLL remain stubs until Phase 8.
- **Split boundary dates in `configs/default.yaml` are placeholders**, to be
  pinned in Phase 4 once the weekly calendar exists (marked PLACEHOLDER in
  the file).
- **`latent_dim = 8`** chosen as the config default from the design doc's
  8–16 range; the sweep is a Phase 9 ablation.
- **CI runs the smoke test** in addition to ruff + pytest, so the vertical
  slice can never silently break.

---

## 2026-07-10 — Data pipeline (roadmap Phase 4)

- **Week definition:** Walmart calendar weeks (Saturday–Friday, `wm_yr_wk`).
  The evaluation file spans 277 complete weeks, 2011-01-29 → 2016-05-20; the
  trailing 2-day partial week is dropped.
- **Split boundaries pinned:** train ends **2014-07-25** (weeks 1–182,
  ~3.5 y), val ends **2015-04-24** (weeks 183–221, ~0.75 y), test = weeks
  222–277 (~1 y). A week belongs to a split iff its full 7 days end on or
  before the boundary; a window belongs to a split iff all its L+H weeks lie
  inside it (straddlers dropped).
- **Pre-launch / availability:** a series is available from its first week
  with a recorded sell price (M5 prices exist exactly when the item is
  offered). Windows touching unavailable weeks are dropped. **1,076 of
  14,370 FOODS series (7.5%) have no train-period price** (launched later)
  and are dropped entirely — their relative-price feature would otherwise
  require val/test statistics (leakage). 13,294 series remain.
- **Feature operationalization** (design doc §5 advisor challenge):
  season = sin/cos of ISO week-of-year of the first target week; events and
  SNAP = fraction of event/SNAP days over the target window; price =
  window-mean sell price ÷ series' **train-period** mean price. Lags =
  log1p of the L=4 weeks preceding the target window.
- **Scaling:** `ContextScaler` standardizes all continuous features except
  sin/cos; fit on train windows only; **statistics accumulated in float64** —
  numpy's axis-0 reduction over a float32 array drifts ~1% at 1.7M rows,
  which silently broke standardization until `assert_no_leakage` caught it.
  Zero-variance columns get scale 1 (sklearn convention).
- **Resulting dataset:** 1,681,819 train / 425,408 val / 651,406 test
  windows. `assert_no_leakage` (temporal + scaler checks) passes on the real
  data and is unit-tested to reject a scaler fit on train+val+test.
- **EDA headline** (notebooks/01-eda.ipynb, figures/): median dispersion
  var/mean ≈ **4.0**, 99.9% of series over-dispersed → the NB-decoder
  argument is empirically grounded; median weekly zero-fraction 0.17 with
  6.7% of series majority-zero.

---

## 2026-07-13 — Classical baselines + two-level harness (roadmap Phase 5)

- **Batched sampler interface:** `BaseSampler.sample_batch(context_batch, S)`
  added (default: loop over `sample`; classical baselines override with
  vectorized draws). At 651K test windows × S=1000 a per-window Python loop
  is hours; the batched path evaluates a baseline in minutes. A vectorized
  `solve_newsvendor_saa_batch` mirrors the scalar solver (unit-tested equal).
- **Context contract fixed:** a mapping with `series_row`, `context_cat`,
  `context_cont`. Classical per-series baselines key on `series_row`; the
  CVAE will consume cat/cont through the same harness.
- **Baseline fitting:** all statistics from each series' *available train
  weeks* only. NB via method of moments (r = m²/(v−m), p = r/(r+m); exact
  moment match, unit-tested); var ≤ mean degrades to ~Poisson (r = 1e6);
  Gaussian samples clipped at 0 (charitable to the strawman; documented).
  Empirical resampler draws contiguous H-week train windows (preserves
  within-window correlation); series with no full train window fall back to
  independent single-week resamples.
- **Metric conventions:** Level-1 and Level-2 computed on the
  **aggregate-horizon total** (the decision quantity). CRPS via the energy
  form with the S²-pair estimator in O(S log S); PIT randomized (correct for
  discrete demand); coverage from central empirical quantiles. NLL deferred
  to Phase 8.
- **Harness validated on baselines** (design doc §8): Poisson's predicted
  under-coverage is clearly visible (0.374 at nominal 90%) — the harness
  detects what it must detect.
- **First real cost table** (test split, 651,406 windows, S=1000, seed 0,
  results/baseline_comparison.csv): mean realized cost —
  3:1 → Gaussian 50.97 < NB 51.20 < Poisson 51.61 < empirical 52.48;
  1:1 → **NB 29.17** best; 1:3 → **empirical 45.15** best.
  CRPS ranks empirical (21.89) < NB (22.66) < Gaussian (23.25) < Poisson
  (26.65). Two pre-registered findings already visible in baselines alone:
  (i) statistical and decision rankings disagree (RQ3), and per-ratio
  winners differ (RQ5: ratios probe different quantiles); (ii) every
  per-week-independent parametric model **under-covers the 4-week totals**
  (NB 0.623, Gaussian 0.567 at nominal 90%) while the empirical resampler —
  which keeps within-window correlation — covers 0.789: the RQ4 premise
  (correlations matter for the sum) has empirical support before any deep
  model exists.
- **Runtime note:** numpy's `negative_binomial`/`poisson` with broadcast
  per-element parameters dominate the evaluation wall-clock (~87 min for NB
  vs ~2 min for empirical/Gaussian). Acceptable at current scale; if 3-seed
  full sweeps get slow, sample NB as Gamma–Poisson with vectorized rates.

---

## 2026-07-13 — CVAE Milestone 1: plain autoencoder (roadmap Phase 6.1)

- **latent_dim pinned to 16** (upper end of the design range 8–16; the
  4/8/16/32 sweep remains a Phase 9 ablation). Architecture in config:
  encoder [128, 64], decoder [64, 128], item embedding 32, store embedding 4
  — MLPs per the roadmap ("resist transformer temptation").
- **Reconstruction operates in log1p space** (raw counts span orders of
  magnitude across series; raw-space MSE would be dominated by high-volume
  items). This convention carries into the VAE's Gaussian-decoder milestone;
  the NB decoder (Milestone 4) works on raw counts natively.
- **`ContextEncoder` is the shared conditioning block** (embeddings + scaled
  continuous features). Milestone 1 feeds it to the encoder only; the CVAE
  must feed it to *both* networks (design doc §10.3 checklist item).
- Smoke training on the first 1,000 real train windows: MSE(log1p)
  5.15 → 4.22 → 1.72 over 3 epochs — tensors flow, gradients descend.

---

## 2026-07-13 — CVAE Milestone 2: unconditional VAE, Gaussian decoder (Phase 6.2)

- **ELBO conventions fixed for all later milestones:** reconstruction term =
  unit-variance Gaussian NLL in log1p space, *summed over the H weeks*
  (constant dropped); KL summed over latent dims; both averaged over the
  batch — sum-vs-mean consistency is what keeps the two terms commensurate.
  Training loss = recon + β·KL with β = 1 (annealing is Milestone 5).
- **Closed-form diagonal-Gaussian KL implemented by hand** and unit-tested
  against `torch.distributions.kl_divergence` (≤1e-5); reparameterization
  unit-tested by sample moments (mean ≈ μ, var ≈ exp(log σ²)) and gradient
  flow. These are the two silent-failure pieces of every later milestone.
- **Per-dimension KL monitoring live from run one** (design doc §8):
  `gaussian_kl_per_dim` + the < 0.1 nats/dim warning in scripts/train_vae.py.
- Smoke run (1,000 real windows, 3 epochs, 20,900 params): ELBO
  −9.84 → −7.17 → −4.29; KL 0.33 → 0.24 → 0.75 nats with 5/16 dimensions
  ≥ 0.1 — the encoder carries information; no early collapse. (The epoch-2
  KL dip then rise is the familiar compress-then-use trajectory.)

---

## 2026-07-14 — CVAE Milestone 3: conditioning both networks (Phase 6.3)

- **Context embedding concatenated into BOTH encoder and decoder inputs**
  (Sohn et al. 2015). The design doc's checklist item — conditioning only
  the decoder is the documented error — is enforced structurally: neither
  network has a context-free path, and two unit tests pin that the posterior
  mean and the decoder output each respond to a context change.
- **Conditioning verified two ways:** (i) unit test — on synthetic data
  where context IS the signal, zeroing the context embedding after 10
  training steps degrades the ELBO; (ii) real-data inline ablation — a
  fresh model trained 1 epoch with zeroed context ends at ELBO −9.64 vs
  −9.15 with real context.
- **Predicted pathology observed, on schedule:** with context active, total
  KL falls to ~0.09 nats (0/16 dims ≥ 0.1) vs 0.75 nats for the
  unconditional VAE — the informative context (lags above all) explains most
  of the window, so the ELBO pushes the posterior toward the prior. This is
  the early-collapse pressure the design doc rates "likelihood: high" and is
  precisely what Milestone 5's KL annealing + free bits are for. Not fixed
  here by design; the per-dimension monitor caught it, which is the point.
- Smoke run: ELBO −9.15 → −6.22 → −3.05 over 3 epochs (73,804 params).

---

## 2026-07-15 — CVAE Milestone 4: Negative Binomial decoder (Phase 6.4)

- **Config switch `model.decoder_likelihood: nb | gaussian`** (replaces the
  scaffold-era `decoder:` key). Both decoders live in one `ConditionalVAE`:
  a shared MLP trunk feeds either two NB heads or one Gaussian head, so the
  RQ1 ablation swaps a config value, nothing else.
- **NB parameterization** (mean μ, dispersion r; Var = μ + μ²/r): heads pass
  through softplus with floors μ ≥ 0.01, r ≥ 0.1 — the design doc §8
  stability guard against NaNs. The log-pmf is implemented by hand in
  `models/distributions.py` and unit-tested against
  `scipy.stats.nbinom.logpmf` (≤1e-4), at x = 0 (common in M5), and shown to
  beat Poisson in total log-likelihood on over-dispersed synthetic data.
- **Scale convention:** the NB decoder scores **raw counts** (its ELBO is an
  exact bound on the discrete log-likelihood — directly comparable to the
  classical baselines' NLL in Phase 8); the encoder input remains log1p'd;
  the Gaussian decoder keeps its log1p reconstruction space.
- **Smoke run** (1,000 real windows, 3 epochs, 74,320 params): NB ELBO
  finite throughout, −42.75 → −31.20 → −20.83. **KL reversal observed:**
  under the NB decoder the KL *grows* (0.16 → 0.45 → 1.45 nats, 4/16 dims
  active) where the Gaussian-decoder CVAE had collapsed to 0.09 — with an
  exact count likelihood, residual per-week stochasticity is apparently
  worth encoding in z. Early evidence, small slice; Milestone 5's
  annealing/free-bits still get implemented as planned, with the β=1
  no-annealing run as the documented comparison point.

---

## 2026-07-15 — CVAE Milestone 5: KL annealing + free bits (Phase 6.5)

- **Annealing:** linear β = min(1, epoch / n_anneal_epochs) (Bowman et al.
  2016), `n_anneal_epochs: 10` in config; `<= 0` disables (β = 1), which is
  the Phase-9 annealing-off ablation for free.
- **Free bits** (Kingma et al. 2016): the floor is applied to the
  **batch-mean per-dimension KL** — `clamp(kl_per_dim, min=lambda_fb)` with
  `lambda_fb: 0.5` nats — before the β-weighted sum. Dimensions below the
  floor contribute a constant, so the optimizer gains nothing by collapsing
  them. **Reported `kl` and `elbo` always use the raw (unfloored) KL** — the
  floor shapes gradients, never the reported bound. `lambda_fb: 0`
  reproduces the Milestone-2/4 objective exactly (unit-tested).
- **Collapse detector:** warns iff mean per-dim KL < 0.1 nats AND β ≥ 1 —
  low KL during the ramp is expected, not pathological.
- Training defaults pinned: `n_epochs: 30`, `batch_size: 256`, `lr: 1e-3`.
- **Smoke run** (1,000 real windows, 5 epochs, β 0.1 → 0.5): raw KL grows
  0.23 → **6.53 nats** (vs 0.09 collapsed in Milestone 3), min-dim KL rising
  every epoch (0.004 → 0.066), 7/16 dims already at/above the 0.5 floor
  mid-ramp; no warning fired. Phase 6 model stack complete — Phase 7 trains
  it on all 1.68M windows.

---

## 2026-07-15 — Phase 7: full training run + CVAE sampler

- **Engine in `src/demand_vae/training.py`**, `scripts/train.py` is the thin
  CLI (roadmap Phase 3 discipline). Batching = per-epoch permutation with
  contiguous tensor slices (equivalent to a shuffling DataLoader, several
  times faster on CPU). Validation ELBO uses raw KL and a fixed noise seed
  so epochs share one estimator; "best" = **highest** val ELBO (the prompt's
  "lowest" read as a slip — ELBO is maximized).
- **Full run** (1,681,819 train / 425,408 val windows, NB decoder, K=16,
  batch 256, lr 1e-3, anneal 10 epochs, λ_fb=0.5, patience 5, seed 0):
  early-stopped at epoch 18 in **9.8 min CPU** — no epoch-count reduction
  needed. Best val ELBO **−13.4097** (epoch 13). **16/16 latent dimensions
  active every epoch** (mean per-dim KL ~0.42, min ~0.22 nats); the
  Milestone-3 collapse never reappeared at scale. Log:
  results/training_log.csv; checkpoints/best.pt + last.pt (git-ignored)
  carry model weights, optimizer state, epoch, val ELBO, full config, and
  model constructor kwargs.
- **`CVAESampler`** (`models/sampler.py`): the trained CVAE behind the
  Phase 5 `sample(context, n_scenarios)` contract — z ~ N(0,I) → decode →
  NB draw via numpy **Gamma–Poisson** (the runtime fix flagged in the
  Phase 5 log; identical distribution, far faster than direct
  negative_binomial), memory-capped decode chunks, integer ≥ 0 outputs.
  Verified from checkpoint on real test contexts.
- Phase 8 next: the trained sampler enters the two-level harness against
  the four baselines (S=1000, 3 ratios, 3 seeds, paired bootstrap).

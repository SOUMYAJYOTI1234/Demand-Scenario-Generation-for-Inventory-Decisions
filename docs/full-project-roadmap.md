# Roadmap — Demand Scenario Generation for Inventory Decisions

Conditional VAE for retail demand scenarios → newsvendor decisions → decision-level evaluation.
Two deliverables: (1) a professional GitHub repository, (2) a LaTeX project report.

A note on time: the phases below are written for a semester-scale effort. If you are working against the end-of-July course deadline, the compression is: Phases 1–3 in week 1, Phase 4–5 in week 2, Phase 6–7 in weeks 3–4, Phase 8–9 in week 5, Phase 10–11 in the final week. The phase content is identical; only the depth of each pass changes. The rule that protects you either way: **have a thin end-to-end pipeline (data → baseline → newsvendor → cost number) running by the end of the first week.** Everything after that is upgrading components, never wondering whether the whole thing connects.

---

## Phase 1 — Understanding the Problem

**Why this phase exists.** Every downstream decision (decoder likelihood, aggregation level, evaluation metric) is forced by things you learn here. Students who skip this phase make those decisions by copying tutorials, and it shows in the viva.

**What to study, specifically:**

*The VAE line.*
- Kingma & Welling (2014), "Auto-Encoding Variational Bayes" — the original paper. Read for: the intractable-posterior problem statement, the reparameterization trick, the ELBO.
- Sohn, Lee & Yan (2015), "Learning Structured Output Representation using Deep Conditional Generative Models" — the CVAE. This is your actual model: everything conditions on y.
- Doersch, "Tutorial on Variational Autoencoders" (arXiv 2016) — the most readable derivation walkthrough; read alongside the original paper.
- Bowman et al. (2016), "Generating Sentences from a Continuous Space" — where KL annealing and posterior collapse are described in practice. You will hit this problem; read it before you do.
- Optional depth: Hoffman & Johnson, "ELBO surgery" — alternative decompositions of the ELBO; useful for the report's methodology discussion.

*The demand/forecasting line.*
- Salinas et al., "DeepAR" (2020) — the key precedent for using a **Negative Binomial likelihood** for retail count demand. Your decoder-likelihood argument leans on this paper; cite it.
- Makridakis et al., the M5 competition papers ("The M5 competition: Background, organization, and implementation" and the results/findings papers) — what the dataset is, what won, why intermittency is the central difficulty.
- Croston's method / intermittent-demand literature (a survey skim is enough) — so you can name the phenomenon precisely.

*The inventory/decision line.*
- Any treatment of the **newsvendor problem** — Snyder & Shen, *Fundamentals of Supply Chain Theory*, ch. on single-period problems is clean. You need: the cost function, the critical fractile solution Q* = F⁻¹(cᵤ/(cᵤ+cₒ)), and why the answer is a *quantile* of the demand distribution.
- **Sample Average Approximation (SAA)** — the bridge from "I have scenarios" to "I have a decision." Shapiro's SAA survey or any stochastic-programming lecture notes.
- Gneiting & Raftery (2007), "Strictly Proper Scoring Rules" — where CRPS comes from; needed for Phase 8.
- Optional (for the extension discussion): Elmachtoub & Grigas, "Smart Predict, then Optimize" — decision-focused learning, which is your future-work story.

**Concepts you must be able to explain before moving on:** why the marginal likelihood is intractable in a latent-variable model; what the ELBO bounds and why maximizing it is legitimate; why the reparameterization trick is needed (gradient of an expectation w.r.t. the distribution's own parameters); why a GAN gives you samples but no explicit likelihood or usable encoder; why Poisson (mean = variance) is usually wrong for retail demand and Negative Binomial (over-dispersion) is usually right; why the newsvendor solution is a quantile, which is the deep reason a *distributional* model matters at all.

**Common mistakes:** reading survey blogs instead of the three or four primary papers; treating the newsvendor as an afterthought (it is half the project); not understanding *why* the quantile insight makes distribution quality decision-relevant.

**Deliverables / GitHub:** `docs/literature-notes.md` — one paragraph per paper: claim, method, what this project takes from it. This becomes your report's related-work section almost verbatim. Initial repo with README stub and LICENSE.

---

## Phase 2 — Project Planning

**Why.** Scope creep is the number-one killer of student projects. Writing the scope down converts "I hope I finish" into a checklist.

**Define, in writing (`docs/project-plan.md`):**
- **Scope:** one product category (suggest FOODS), weekly aggregation, single-item newsvendor. Everything else is explicitly out of scope or an extension.
- **Assumptions:** demand is exogenous (your order doesn't change demand); costs cᵤ, cₒ are fixed and known (report results for 2–3 cost ratios); no lead time; scenarios are i.i.d. samples from the learned conditional.
- **Evaluation criteria (decide now, before any model exists):** Level 1 — CRPS, calibration, coverage on held-out demand. Level 2 — realized newsvendor cost and service level vs baselines. Pre-committing to metrics prevents the classic mistake of choosing metrics after seeing results.
- **Milestones:** M1 pipeline + baselines end-to-end; M2 CVAE training stably; M3 count-likelihood decoder + collapse handling; M4 full evaluation table; M5 ablations; M6 report + polish.
- **Success criteria (honest version):** primary — a correct, reproducible pipeline with a defensible comparison. The CVAE beating baselines is a *hoped-for* outcome, not the success criterion; a well-analyzed negative result passes.

**Common mistakes:** defining success as "the model wins"; leaving evaluation design until after training; planning no buffer week.

**GitHub:** `docs/project-plan.md`, a GitHub Project board with the milestones as columns, issues for M1 tasks.

---

## Phase 3 — Repository Design

**Why.** Recruiters and interviewers open the repo before they read the report. Structure signals engineering maturity faster than any model does.

**Structure:**

```
demand-scenario-vae/
├── README.md              # see below
├── LICENSE                # MIT is fine for coursework
├── .gitignore             # python template + data/, *.ckpt, wandb/
├── pyproject.toml         # single source of deps + tool config (preferred over bare requirements.txt)
├── requirements.txt       # optional pinned export for quick pip install
├── .pre-commit-config.yaml
├── configs/               # yaml per experiment: model, training, data, costs
├── data/                  # NOT committed; README explains how to download M5
│   └── README.md
├── notebooks/             # numbered, exploratory only: 01-eda.ipynb, 02-...
├── src/demand_vae/        # installable package: data/, models/, eval/, opt/
├── scripts/               # thin CLIs: train.py, evaluate.py, run_newsvendor.py
├── tests/                 # pytest: likelihoods, newsvendor math, data shapes
├── experiments/           # per-run configs + result jsons (small files only)
├── results/               # final tables/csvs used in the report
├── figures/               # final figures used in the report/README
├── reports/               # LaTeX source of the final report
└── docs/                  # plan, literature notes, design decisions
```

**Purposes worth stating:** `src/` vs `notebooks/` separation is the single biggest signal — notebooks explore, the package implements; anything used twice gets promoted from notebook to `src/`. `configs/` means no hyperparameter lives hard-coded in a script. `tests/` even with only 10 tests (does the NB log-likelihood match scipy? does the newsvendor quantile match the closed form on a known distribution?) puts you ahead of 95% of student repos.

**README structure:** one-line pitch → the money figure (cost comparison plot) → problem statement (3 sentences) → method diagram → results table → quickstart (install, download data, train, evaluate — each one command) → repo map → citation/report link.

**Tooling:** pre-commit with ruff (lint+format); GitHub Actions running lint + pytest on push (a green badge is cheap and looks professional); issue templates optional — the Project board matters more.

**Common mistakes:** committing the dataset (M5 is large; .gitignore it and script the download); one giant `main.py`; notebooks as the only implementation; requirements.txt with 200 unpinned transitive deps.

**GitHub:** the full skeleton, CI passing on an empty test, README stub with the plan.

---

## Phase 4 — Data Pipeline

**Why.** M5 is large and long-format-awkward; the pipeline decisions (aggregation, split, features) silently determine every result downstream.

**Steps:**
1. **Download via script** (`scripts/download_data.py`, Kaggle API), never manually. Reproducibility starts here.
2. **EDA notebook:** distribution of weekly demand per series (expect right skew, many zeros); zero-fraction per series; seasonality plots (week-of-year averages); price/promo effects; a few representative series plotted over time. Figures to keep: demand histogram with NB/Poisson/Gaussian fits overlaid (this single figure justifies the decoder choice in the report); an intermittency plot; a seasonal profile.
3. **Scope + aggregate:** one category, sum daily → weekly per item-store. Weekly reduces intermittency and matches ordering cadence — say both reasons in the report.
4. **Windowing:** each sample = (context features y, demand window x ∈ ℝ^H), H = 4–8 weeks. Decide whether x is the *next* H weeks given y containing recent lags (recommended — matches the decision use-case).
5. **Conditioning features y:** item/category embedding index, store index, week-of-year (as sin/cos or index), holiday/SNAP indicators, price level, promo flag, recent-demand lags (e.g., last 4 weeks, log1p-scaled).
6. **Split: temporal.** Train on the earliest span, validate on the next, test on the final year. Never random-split time series — it leaks the future. State the exact date boundaries in a config.
7. **Missing/zero handling:** M5 has no missing values proper, but items appear mid-history (zeros before launch). Either drop pre-launch periods per item or add an availability flag; document the choice.

**Common mistakes:** random splits (instant disqualification of results); normalizing counts with statistics computed on the full dataset (leakage — fit scalers on train only); silently dropping zero-heavy series and then claiming the method handles intermittency.

**GitHub:** `src/demand_vae/data/` with the pipeline, `notebooks/01-eda.ipynb`, EDA figures in `figures/`, a `configs/data.yaml` pinning every choice.

---

## Phase 5 — Baseline Models

**Why.** Your entire claim is comparative. Baselines are not a chore before the real model — they *are* the measuring stick, and they must be strong and fair.

**Implement, in this order:**
1. **Empirical historical resampling.** For each context, resample from that series' (or that series-season's) historical demand. Why: the strongest cheap baseline; if the CVAE can't beat resampling history, conditioning has added nothing.
2. **Gaussian fitted to historical demand** (per series or per series-season). Why: it is *the* textbook assumption in inventory theory; beating it is the headline claim.
3. **Poisson** (mean from history). Why: the naive count model; its failure (under-dispersion) motivates NB.
4. **Negative Binomial fitted directly** (per series, method of moments or MLE). Why: this is the sneaky-important one — a *non-conditional* NB. If your CVAE beats Gaussian but not plain NB, the win came from the likelihood family, not the deep model. Including this baseline makes your eventual claim honest and much stronger.

Each baseline must expose the same interface as the CVAE: `sample(context, n_scenarios) → array`. Then the entire evaluation stack is model-agnostic.

**Run the full Level-2 evaluation on baselines now.** You get your first realized-cost table with zero deep learning — the thin slice is complete, and you now know the number to beat.

**Common mistakes:** weak strawman baselines (Gaussian with a single global mean); baselines evaluated under different conditions than the model; skipping the plain-NB baseline and later being unable to attribute the win.

**GitHub:** `src/demand_vae/baselines/`, tests comparing fitted likelihoods to scipy, first results table in `results/`.

---

## Phase 6 — Building the Conditional VAE

**Why.** The core modeling contribution. Build it in milestones so a bug is always localized to the last small change.

**Milestones:**
1. **Plain autoencoder** on demand windows (no sampling, MSE). Verifies data flow and reconstruction plumbing.
2. **Unconditional VAE, Gaussian decoder.** Adds: encoder outputting (μ_φ, log σ²_φ), reparameterization z = μ + σ⊙ε, ELBO = reconstruction + KL. The KL between diagonal Gaussians is closed-form — implement it yourself and test against a library.
3. **Conditioning.** Concatenate an embedding of y into *both* encoder input and decoder input (this is Sohn et al.'s CVAE). Categorical features → learned embeddings; continuous → normalized and concatenated.
4. **Count decoder.** Decoder outputs NB parameters per time step — e.g., total_count/dispersion and logits, or (mean, dispersion) with softplus positivity. Reconstruction term becomes the NB negative log-likelihood. Keep the Gaussian decoder switchable by config: that's an ablation for free.
5. **Collapse handling.** Monitor per-dimension KL from the first run. If KL → 0 (posterior collapse: encoder ignored, decoder becomes an unconditional model), apply **KL annealing** (β from 0 → 1 over the first N epochs) and/or **free bits** (floor per-dim KL). Have both implemented before you need them.

**Design decisions to make deliberately (and record in `docs/design-decisions.md`):**
- **Latent dimension:** start small (8–16) for weekly windows of length 4–8; the data is low-dimensional compared to images. Sweep later (Phase 9).
- **Architecture:** MLPs are entirely sufficient at H ≤ 8; a GRU/1-D conv encoder is an ablation, not the default. Resist transformer temptation — nothing here needs it and it costs your timeline.
- **Prior:** standard N(0, I). Learned priors are out of scope.
- **Output horizon:** decoder emits parameters for all H steps jointly (captures within-window correlation through z — worth one sentence in the report: independence *given z*, dependence marginally).
- **β (KL weight):** default 1 after annealing; β ≠ 1 is a Phase-9 ablation.

**Common mistakes:** implementing everything at once and debugging a five-way interaction; not monitoring KL until generation looks suspiciously generic; forgetting positivity constraints on NB parameters (NaNs at 3 a.m.); conditioning only the decoder (encoder must see y too, or the posterior has to smuggle context through z).

**GitHub:** `src/demand_vae/models/` with autoencoder → VAE → CVAE progression visible in history, unit tests for the KL formula and NB likelihood, config-switchable decoder.

---

## Phase 7 — Training Strategy

**Why.** Undisciplined training wastes the two scarcest resources you have: time and belief in your own results.

**Practices:**
- **Batching:** shuffle windows within train; batch 128–512 (this model is tiny; CPU/Colab is fine).
- **Optimizer:** Adam, lr 1e-3, with reduce-on-plateau or cosine decay. Do not tune the optimizer before the model is correct.
- **Early stopping** on validation ELBO (or validation NLL), patience ~10–20 epochs.
- **Monitoring, every run:** total ELBO, reconstruction term, KL term (total and per-dimension), plus a fixed grid of sampled scenarios vs actuals for 3–4 held-out contexts — the qualitative canary.
- **Experiment tracking:** Weights & Biases (free tier) or MLflow (local, zero-account). Log config, metrics, and figures per run; name runs after config files.
- **Checkpoints:** best-on-validation plus last; save the config *inside* the checkpoint dir.
- **Seeds:** fix and log; run the final model with 3 seeds so the report can carry a ±std.

**Common mistakes:** comparing runs whose configs are reconstructed from memory; watching only total loss (collapse hides inside a healthy-looking total: reconstruction improves as KL dies); "it works on my machine" — the config + seed + commit hash should reproduce any number in the report.

**GitHub:** `scripts/train.py` reading a config path, tracked-experiment links or `experiments/` records, checkpoint loading tested.

---

## Phase 8 — Evaluation (the heart of the project)

**Why downstream evaluation is the point.** Likelihood measures a model's fit everywhere; the newsvendor cares about a specific *quantile* (the critical fractile). Two models with similar NLL can imply very different order quantities. Evaluating on realized cost measures what the model is *for* — that argument, made explicitly, is the intellectual core of your report and the line that distinguishes this project.

**Level 1 — the generative model itself (held-out test set):**
- **NLL** of held-out demand under each model (computable for every method here — one advantage of likelihood-based models; note the GAN could not join this table).
- **CRPS** per prediction, averaged — the standard proper score for distributional forecasts; estimate from samples.
- **Calibration:** PIT histogram (should be flat) and quantile reliability (nominal q vs empirical coverage).
- **Coverage:** does the 90% interval cover ~90% of actuals?
- **Diversity sanity check:** per-context scenario spread vs historical variability — catches collapse that metrics can miss.

**Level 2 — the decisions:**
- For each test (context, actual-demand) pair and each method: generate S scenarios (e.g., S = 1000), solve the newsvendor by SAA (order Q = the critical-fractile empirical quantile of the scenarios), then compute **realized cost** cᵤ(D−Q)⁺ + cₒ(Q−D)⁺ against the *actual* demand D.
- Report: mean realized cost (with std over seeds), **service level** (fraction of periods without stockout), **fill rate** (fraction of demand met), average **overstock** units.
- Run at 2–3 cost ratios (e.g., cᵤ:cₒ = 3:1, 1:1, 1:3) — different ratios probe different quantiles of the learned distribution, so this doubles as a distribution-quality probe and is a genuinely informative table.
- **Statistical honesty:** paired comparisons per test point; report a paired t-test or bootstrap CI on the cost difference, not just means.

**Common mistakes:** reporting only the cost ratio where the model wins; unpaired comparisons; S too small (SAA noise swamps model differences); forgetting that CRPS and cost can disagree — when they do, *that disagreement is a finding*, write it up.

**GitHub:** `src/demand_vae/eval/` (metrics) and `src/demand_vae/opt/` (newsvendor/SAA, unit-tested against the closed form), `scripts/evaluate.py` producing the full table for any model from its config, main results table + money plot committed to `results/` and `figures/`.

---

## Phase 9 — Ablation Studies

**Why.** Ablations convert "it works" into "I know why it works," which is the difference between a course project and research training.

**Priority order (do the top ones; the rest as time allows):**
1. **Decoder likelihood: NB vs Poisson vs Gaussian.** Supports the central statistical claim (over-dispersion matters). Expect Gaussian to calibrate poorly on low-volume items.
2. **Conditioning on vs off** (CVAE vs plain VAE). Isolates the value of context — the "conditional" in your title.
3. **KL weight β / annealing on vs off.** Documents the collapse story with evidence.
4. **Latent dimension sweep** (e.g., 4/8/16/32). Shows capacity sensitivity; expect a plateau.
5. **Aggregation level** (weekly vs biweekly) — only if time; it requires re-running the pipeline.

Each ablation reports the *same* Level-1 + Level-2 table. One config file per ablation; the evaluation script does the rest — this is where the Phase-3 architecture pays for itself.

**GitHub:** `experiments/ablations/` configs + results, an ablation summary table in `results/`.

---

## Phase 10 — Final Report (LaTeX)

**Structure and what each chapter actually contains:**
1. **Introduction** — the decision problem (ordering under demand uncertainty), the gap (inventory practice assumes Gaussian; demand is over-dispersed, intermittent, context-dependent), the proposal (learn the conditional demand distribution with a CVAE; evaluate at the decision level), contributions as 3 bullets.
2. **Literature Review** — three threads from Phase 1: VAEs/CVAEs, probabilistic demand forecasting (DeepAR, M5 findings), newsvendor/SAA. End by positioning: this project connects thread 2's models to thread 3's decisions.
3. **Problem Formulation** — the mathematical objects: the conditional demand distribution p(x|y), the ELBO (derive it — one page, from Jensen), the newsvendor cost and critical fractile, SAA. This chapter proves you own the math.
4. **Dataset** — M5 description, scoping decisions with reasons, the EDA figures (histogram-with-fits earns its place here), split protocol.
5. **Methodology** — architecture diagram, conditioning mechanism, decoder likelihood with the NB parameterization, collapse handling, training protocol, and one honest paragraph of design decisions considered-and-rejected (from `docs/design-decisions.md`).
6. **Experiments** — baselines, metrics (define CRPS, calibration, realized cost precisely), protocol (seeds, S, cost ratios).
7. **Results** — Level-1 table, Level-2 table across cost ratios, the money plot, ablations. Report the negative cells too.
8. **Discussion** — *why* the results look as they do; where likelihood and cost rankings disagreed and what that means; when the CVAE helps (which items/contexts) and when it doesn't.
9. **Limitations** — single-item newsvendor, no lead time, exogenous demand, one category, likelihood-trained (not decision-trained) generator.
10. **Future Work** — from Phase 12; decision-focused training is the natural headline.
11. **References**, **Appendix** (extra tables, derivation details, reproducibility: commit hash + configs per reported number).

**Common mistakes:** writing the report in the last three days (start chapters 3–5 the moment those phases finish — they don't depend on results); figures made in a rush that don't match table numbers; no limitations section (examiners *always* probe what you didn't say).

**GitHub:** `reports/` with LaTeX source, a `make report` or compile script, the PDF in releases (not the repo tree).

---

## Phase 11 — GitHub Polish

**Why.** This is the deliverable a hiring manager sees in 90 seconds. Optimize for that skim.

- **README top:** the money plot + one-sentence result. A recruiter should get the point without scrolling.
- **Architecture diagram** (draw.io/excalidraw is fine) in README and report.
- **Reproducibility block:** exact commands from clean clone → main results table. Test it in a fresh environment; this is the credibility maker.
- **Badges:** CI status, python version, license — cheap and professional.
- **Release tag** `v1.0` matching the report submission, with the PDF attached to the release.
- **Example artifacts:** a small `examples/` notebook that loads the trained checkpoint and generates scenarios for one context — the 2-minute demo path.
- **Commit hygiene pass:** squash embarrassing WIP chains if needed; the history should show the milestone progression.
- Link the repo from your resume/LinkedIn with the one-line pitch from the project sheet.

---

## Phase 12 — Extensions

**Natural future work (write these into the report):**
- **Conditional diffusion generator** — same interface, swap the model; the course covers DDPMs, so this is the most in-scope extension and the one to attempt first if time remains.
- **Decision-focused training** — train the generator through the newsvendor objective (Smart Predict-then-Optimize lineage). Intellectually the deepest extension and the strongest "future work" paragraph; a genuinely publishable direction.
- **Time-series VAEs** (recurrent latent models) — better temporal structure; moderate effort.

**Explicitly beyond scope (say so in the report — scoping discipline reads as maturity):**
- Multi-echelon inventory, RL-based ordering policies, normalizing flows / flow matching, Bayesian optimization over policies, multi-product with substitution effects. Each is a project of its own; naming them shows you see the landscape without pretending to cover it.

---

## Where to spend your time (the honest allocation)

Most effort: **Phase 4 (data), Phase 6 milestone 4–5 (count decoder + collapse), Phase 8 (evaluation).** These carry the project.
Keep deliberately simple: architecture (MLPs), optimizer tuning, repo automation beyond lint+test.
Constant background thread: `docs/design-decisions.md` and report chapters 3–5, written as you go — the report should be 60% drafted before Phase 10 officially starts.

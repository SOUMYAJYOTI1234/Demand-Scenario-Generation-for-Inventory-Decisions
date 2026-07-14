# Project Design Document — Demand Scenario Generation for Inventory Decisions

*Pre-implementation planning document. Prepared before any code is written. Companion to the literature review (literature-review.md) and the phase roadmap (full-project-roadmap.md).*

---

## Section 1 — Problem Formulation

### 1.1 Four nested problems

This project is best understood as four problems stacked on top of each other. Keeping them separate in notation and in thought prevents the most common design confusion in estimate-then-optimize work: evaluating one problem with the metrics of another.

**(a) The machine learning problem.** Given a dataset of context–demand pairs, learn a function-like object that, presented with a new context, produces samples of plausible demand. Formally: learn a conditional sampler for demand given context, trained on historical (context, demand) pairs from many item–store series jointly.

**(b) The probabilistic modeling problem.** Estimate the conditional distribution

\[ p^{*}(x \mid y) \]

where \(x \in \mathbb{Z}_{\geq 0}^{H}\) is a window of \(H\) consecutive weeks of unit demand for one item–store series, and \(y\) is the observable context for that window. The model is a conditional latent-variable model

\[ p_{\theta}(x \mid y) = \int p_{\theta}(x \mid z, y)\, p(z)\, dz, \qquad z \in \mathbb{R}^{K},\ p(z) = \mathcal{N}(0, I_K), \]

trained by maximizing the conditional ELBO

\[ \mathcal{L}(\theta, \phi; x, y) = \mathbb{E}_{q_{\phi}(z \mid x, y)}\!\big[\log p_{\theta}(x \mid z, y)\big] - D_{\mathrm{KL}}\!\big(q_{\phi}(z \mid x, y)\,\Vert\, p(z)\big) \;\leq\; \log p_{\theta}(x \mid y). \]

**(c) The optimization problem.** Given a demand distribution (represented by \(S\) scenarios \(\hat{x}^{(1)}, \dots, \hat{x}^{(S)}\) drawn from the model), choose an order quantity minimizing expected newsvendor cost. With underage cost \(c_u\) and overage cost \(c_o\), and writing \(D\) for the (scalar) demand quantity the decision covers:

\[ Q^{*} = \arg\min_{Q}\ \mathbb{E}\big[c_u (D - Q)^{+} + c_o (Q - D)^{+}\big] = F_D^{-1}\!\left(\frac{c_u}{c_u + c_o}\right). \]

Under sample average approximation, \(Q^{*}\) is the empirical critical-fractile quantile of the scenario set.

**(d) The decision-making problem.** Evaluate, on held-out data, the *realized* cost of the decisions each demand model induces:

\[ \text{RealizedCost}(Q; D_{\text{actual}}) = c_u (D_{\text{actual}} - Q)^{+} + c_o (Q - D_{\text{actual}})^{+}, \]

aggregated over test contexts, compared across models, at several cost ratios.

### 1.2 A design decision that must be made now: what does the decision consume?

**Advisor challenge.** The model generates an \(H\)-week window \(x\); the newsvendor is a single-period model with a scalar demand \(D\). These do not match automatically, and the project description as written leaves the mapping undefined. There are three coherent options:

1. **Single-week decision (\(H = 1\)).** Model next-week demand only; the newsvendor covers one week. Simplest, but it discards the joint-trajectory argument made in the literature review (Section 6 there) — a single-step marginal needs no latent-variable machinery beyond what DeepAR-style models already provide.
2. **Repeated weekly decisions from the window's marginals.** Generate \(H\)-week scenarios, but make \(H\) separate weekly newsvendor decisions using each week's marginal scenario distribution. This uses the model's conditional marginals but never its correlations — again undercutting the joint-modeling motivation.
3. **Aggregate-horizon decision.** Define the decision quantity as total demand over the horizon, \(D = \sum_{h=1}^{H} x_h\) — an order placed to cover the next \(H\) weeks. The distribution of a *sum* depends on the within-window correlations, so this decision genuinely consumes the joint distribution the CVAE is built to capture.

**Recommendation: option 3 as the primary decision task, with option 2 as a secondary analysis.** The aggregate-horizon newsvendor is operationally meaningful (order-up-to decisions over a review period), and it is the only variant in which the CVAE's central technical advantage — joint scenario generation — can show up in the results at all. Option 2 is retained as a secondary experiment precisely because comparing it against option 3 isolates the value of modeling correlations. This choice should be recorded as final before implementation.

### 1.3 Objects and notation

| Object | Symbol | Definition |
|---|---|---|
| Demand window (target) | \(x \in \mathbb{Z}_{\geq 0}^{H}\) | Next \(H\) weeks of unit sales for one item–store series; \(H = 4\) (design default) |
| Context (conditioning) | \(y\) | Concatenation of: item/category embedding index, store index, week-of-year encoding (sin/cos), event/SNAP indicators for the window, relative price level, and \(L\) lagged weeks of recent demand (log1p-scaled), \(L = 4\text{–}8\) |
| Latent variable | \(z \in \mathbb{R}^{K}\) | \(K = 8\text{–}16\) design default; prior \(\mathcal{N}(0, I)\) |
| Encoder (inference network) | \(q_{\phi}(z \mid x, y)\) | Diagonal Gaussian; network outputs \((\mu_{\phi}, \log \sigma^{2}_{\phi})\) |
| Decoder (generator) | \(p_{\theta}(x \mid z, y)\) | Per-week Negative Binomial parameters, conditionally independent across weeks given \((z, y)\) |
| Decision quantity | \(D = \sum_h x_h\) | Aggregate demand over the horizon (primary task) |
| Order quantity | \(Q\) | Chosen by SAA over \(S\) scenarios; \(S = 1000\) default |
| Costs | \(c_u, c_o\) | Evaluated at ratios \(c_u{:}c_o \in \{3{:}1,\ 1{:}1,\ 1{:}3\}\) |

**Inputs at training time:** (x, y) pairs windowed from historical series. **Inputs at decision time:** y only. **Outputs:** scenario sets \(\{\hat{x}^{(s)}\}\), the induced order \(Q\), and the realized cost against actual demand.

**A note on "conditional independence across weeks given \((z, y)\)."** The decoder factorizes over weeks *given the latent*; marginally, weeks remain dependent because they share \(z\). This is the standard latent-variable mechanism for capturing correlation cheaply, and it is exactly what makes the aggregate-horizon decision a fair test of the model: if \(z\) carries no information (posterior collapse), the implied sum-distribution degenerates toward independence and the decision quality should measurably suffer. This link between a training pathology and a decision-level symptom is worth a paragraph in the final report.

---

## Section 2 — Research Questions

**Primary research question (RQ0).**
*Does a conditional VAE that learns the conditional demand distribution produce demand scenarios that lead to lower realized newsvendor cost than classical probabilistic baselines (Gaussian, Poisson, fitted Negative Binomial, empirical resampling) on held-out retail data?*
Measured by: mean realized cost and service level over the test period, per cost ratio, with paired statistical comparison.

**Secondary research questions.**

- **RQ1 (likelihood family).** How much of any decision-level improvement is attributable to the count likelihood rather than the deep conditional model? Measured by: CVAE-NB vs the *non-conditional* fitted NB baseline, and CVAE-NB vs CVAE-Gaussian ablation.
- **RQ2 (conditioning).** Does conditioning on context (category, store, season, price) improve decisions relative to an unconditional VAE of the same capacity? Measured by: CVAE vs VAE ablation on identical metrics.
- **RQ3 (statistical vs decision metrics).** Do distributional metrics (CRPS, calibration, coverage) and realized decision cost rank the candidate models identically? Measured by: rank correlation between Level-1 and Level-2 model rankings; disagreements analyzed by quantile region.
- **RQ4 (correlation value).** Does modeling within-window correlation matter for decisions? Measured by: aggregate-horizon decisions (which consume the joint) vs repeated-marginal decisions (which do not), for the same trained model.
- **RQ5 (cost-ratio sensitivity).** Is the CVAE's advantage (if any) concentrated at particular critical fractiles? Measured by: realized-cost gaps across \(c_u{:}c_o \in \{3{:}1, 1{:}1, 1{:}3\}\), interpreted as probing upper, central, and lower quantiles respectively.

Each question maps to a specific experiment in Section 7; none requires apparatus beyond what the pipeline already contains.

---

## Section 3 — Objectives

**Overall objective.** Build and evaluate an end-to-end pipeline — conditional generative demand model → scenario set → newsvendor decision → realized-cost evaluation — that answers RQ0 with statistical honesty on public retail data.

**Mandatory objectives.**
1. A reproducible M5 data pipeline: one category, weekly aggregation, temporal splits, documented feature construction.
2. Four classical baselines behind a common sampler interface: empirical resampling, Gaussian fit, Poisson fit, Negative Binomial fit (all per-series or per-series-season).
3. The SAA newsvendor decision layer and the two-level evaluation harness (Level 1: NLL where defined, CRPS, PIT calibration, coverage; Level 2: realized cost, service level, fill rate) — built and validated on baselines *before* the deep model exists.
4. A conditional VAE with (i) Gaussian and (ii) Negative Binomial decoders, with KL-annealing/free-bits collapse handling and per-dimension KL monitoring.
5. The core comparison answering RQ0–RQ3, with paired significance analysis over ≥3 seeds.
6. A written report and a clean repository meeting the standards in the phase roadmap.

**Optional extensions (attempt only after all mandatory items are complete).**
7. RQ4's marginal-vs-aggregate analysis (cheap: same model, second decision protocol) — this is the first extension to do, and is close to mandatory in value.
8. Zero-inflated NB decoder ablation.
9. Conditional diffusion generator behind the same sampler interface.
10. Decision-aware fine-tuning of the generator (Smart Predict-then-Optimize direction) — explicitly a stretch; framed as future work if untouched.

**Expected outcomes.** A defensible answer to RQ0 in either direction; a quantified decomposition of where any improvement comes from (likelihood family vs conditioning vs correlation); at least one documented case where statistical and decision rankings diverge or provably coincide; a reusable evaluation harness.

---

## Section 4 — Scope

**Included.** Single-item, single-location newsvendor decisions; one M5 product category (FOODS recommended: highest volume, least sparsity) at weekly aggregation; the four classical baselines; CVAE with two decoder likelihoods; two decision protocols (aggregate-horizon primary, repeated-marginal secondary); three cost ratios; the full two-level evaluation.

**Excluded, with reasons.**
- **Multi-item / assortment decisions and substitution effects.** Requires joint cross-item modeling and a fundamentally larger optimization; a project of its own.
- **Lead times, multi-echelon structure, capacity constraints.** Each converts the newsvendor into a different (harder) inventory model and dilutes the central question, which is about the *demand model*, not the inventory model.
- **Demand censoring correction.** M5 records *sales*, not demand; stockouts censor observations (see Assumption A6). Correcting for censoring is a research literature in itself; we acknowledge and proceed.
- **Price optimization / endogenous demand.** Price is a conditioning input, never a decision variable here.
- **Normalizing flows, autoregressive deep baselines, EBMs.** The baseline set is classical by design (RQ1 needs the fitted NB); adding deep baselines expands compute and analysis beyond the timeline. The fitted-NB baseline stands in for "parametric likelihood without deep conditioning."
- **Hyperparameter search at scale.** A small manual sweep over \(K\) and β-schedule only; the claim is not "best possible CVAE" but "a competently trained CVAE."

**Anti-scope-creep rule.** Any addition must displace something on this list explicitly, in writing, in the design log. Nothing is added silently.

---

## Section 5 — Assumptions

**A1 — Sales approximate demand (censoring ignored).**
*Why:* M5 contains no inventory or stockout records, so uncensoring is impossible without external data.
*Consequence:* demand is understated exactly in high-demand periods, biasing all models' upper quantiles downward — note this affects every compared method equally, so *relative* comparisons remain meaningful even though absolute service levels are optimistic.
*If relaxed:* requires censored-demand estimation (EM-style corrections as in the Agrawal–Smith tradition); a different project.

**A2 — Weekly aggregation.**
*Why:* daily M5 series are dominated by zeros (majority of item–store–days), which makes likelihood estimation and calibration analysis fragile; weekly buckets reduce intermittency while matching realistic ordering cadence.
*Consequence:* within-week timing information is lost; "season" must be encoded at weekly resolution (week-of-year, not day-of-week).
*If relaxed:* daily modeling forces zero-inflation machinery to the foreground and multiplies data volume ~7×; feasible but shifts the project's center of gravity to intermittency modeling.

**A3 — Fixed, known, stationary costs \((c_u, c_o)\).**
*Why:* M5 has prices but not cost structures; costs are the analyst's lever, and sweeping ratios doubles as a quantile probe (RQ5).
*Consequence:* results are conditional on the assumed ratios; no claim about any retailer's true economics.
*If relaxed:* cost estimation from margins is possible in principle but unidentifiable from this data.

**A4 — No lead time; single-period, independent decisions.**
*Why:* keeps the optimization layer analytically transparent (critical fractile), so decision differences are attributable to the demand model alone.
*Consequence:* no inventory carry-over between decisions; each test window is an independent trial (which is also what makes paired statistics clean).
*If relaxed:* base-stock/multi-period models introduce state, simulation, and policy evaluation — the demand model's effect becomes entangled with policy dynamics.

**A5 — Scenario exchangeability within a decision (i.i.d. draws from the model).**
*Why:* required by SAA.
*Consequence:* model misspecification propagates directly into decisions — which is the point being measured, not a nuisance.

**A6 — Contexts observed at decision time.** Price, calendar, and events for the upcoming window are known when ordering.
*Why:* calendar and events genuinely are known ahead; prices are set by the retailer.
*Consequence:* mild optimism if real prices are uncertain ahead of time; uniform across methods.

**A7 — Train-period relationships persist into the test period (no structural break).**
*Why:* unavoidable for any learned model; M5 (2011–2016) contains no catastrophic regime change.
*Consequence:* temporal split (Section 7) makes this assumption *testable in effect* — degradation from validation to test is observable.

**Advisor challenge on "season" and "price" as stated in the project description.** "Season" is not a variable; it must be operationalized — the design choice here is week-of-year as sin/cos plus explicit event/SNAP flags (M5's event calendar is unusually rich; use it, it is free signal). "Price" in M5 varies slowly and is confounded with promotions; the defensible encoding is *relative price* (current price ÷ item's training-period average), which captures promotional deviation rather than absolute price level. Both operationalizations should be stated in the report as decisions, not defaults.

---

## Section 6 — Success Criteria

**Minimum success (must achieve; failure here is project failure).**
A correct, reproducible end-to-end pipeline: data → four baselines → SAA newsvendor → two-level evaluation, plus a CVAE that trains stably (no collapse, monitored KL) and enters the comparison. All numbers reproducible from configs + seeds + commit hash. *Measured by:* the pipeline running from a clean clone to the main results table; unit tests passing on the likelihood and newsvendor math.

**Expected success.**
Minimum success, plus: a statistically defensible answer to RQ0 *in either direction* at all three cost ratios; the RQ1–RQ3 decompositions completed; the report's discussion explaining *why* the observed ranking occurred (e.g., which quantile regions drive cost differences). *Measured by:* paired bootstrap confidence intervals on cost differences; completed ablation table; calibration diagnostics per model.

**Excellent success.**
Expected success, plus: the CVAE demonstrating a significant realized-cost improvement over the fitted-NB baseline (the hard comparison, not just the Gaussian strawman) at ≥1 cost ratio, *or* an equally rigorous negative/partial result with a mechanistic explanation (e.g., "conditioning helps only for high-velocity items; for intermittent items the per-series NB is unbeatable at this data scale"); plus at least one optional extension (RQ4 analysis or ZINB ablation) completed. *Measured by:* the same paired statistics; the mechanism analysis surviving an advisor's cross-examination.

**Deliberate framing note.** Success is defined by *rigor of the answer*, not direction of the result. This is stated in the report's introduction so the evaluation cannot be read as post-hoc rationalization.

---

## Section 7 — Experimental Design

**Dataset.** M5 (Walmart): unit sales for 3,049 items × 10 stores, 2011–2016, with calendar, events, SNAP, and prices. Scoped to the FOODS category (rationale: highest volumes, most usable weekly series). After weekly aggregation and windowing (\(H=4\), stride 1), the sample count is in the hundreds of thousands — ample for the model sizes planned.

**Splits — temporal, fixed, and stated in the config.** Train: first ~3.5 years. Validation: next ~0.75 year (model selection, early stopping, β-schedule tuning). Test: final ~1 year, touched once. *Why temporal:* random splits leak future statistics into training through overlapping windows and shared seasonal states; a temporal split is the only honest protocol and simultaneously stress-tests Assumption A7. Additional leakage guards: all normalization/scaling statistics and per-series baseline fits use train data only; windows straddling a split boundary are dropped.

**Baselines and their reasons.**
1. *Empirical resampling (per series-season):* the strongest assumption-free baseline; beating it shows the model adds structure beyond memory.
2. *Gaussian fit:* the inventory-theory default; the headline strawman, but a fair one because it is what practice assumes.
3. *Poisson fit:* isolates the over-dispersion argument — its predicted failure is part of the narrative.
4. *Fitted Negative Binomial (per series):* **the pivotal baseline.** If the CVAE beats Gaussian but not this, the win came from the likelihood family, not deep conditioning. Any claim of "deep model value" must survive this comparison.
All baselines implement the same `sample(context, S)` interface so the decision and evaluation layers are model-agnostic.

**Evaluation metrics.**
*Level 1 (distributional):* held-out NLL (for likelihood-based models), CRPS (sample-estimated), PIT histograms, central-interval coverage (50%, 90%). *Why:* proper scoring rules and calibration are the accepted standards; they also localize *where* distributions differ.
*Level 2 (decision):* realized cost (mean ± paired CI), service level, fill rate, average overstock; per cost ratio; both decision protocols (aggregate-horizon primary, repeated-marginal secondary). *Why:* this is the project's thesis (Section 9 of the literature review).

**Ablations (each answers a named RQ).**
- Decoder likelihood NB vs Gaussian vs Poisson (RQ1) — same architecture, swap likelihood.
- Conditional vs unconditional (RQ2) — remove \(y\) from both networks.
- β/KL-annealing on vs off (training-pathology documentation).
- Latent dimension \(K \in \{4, 8, 16, 32\}\) (capacity sensitivity).
- Aggregate vs marginal decision protocol (RQ4) — no retraining required.

**Statistical significance.** Decisions on the same test windows are paired across methods; report paired bootstrap CIs (10⁴ resamples) on mean cost differences, and repeat the full pipeline over 3 seeds reporting mean ± sd. *Why paired:* between-window demand variance dwarfs between-method differences; pairing removes it. Avoid declaring victory from unpaired means — this is the most common statistical error in comparative ML work.

**Reproducibility.** Every reported number traces to (config file, seed, commit hash); scaler/fit artifacts versioned; data acquisition scripted; the results table regenerable by one command. Scenario count \(S = 1000\) with a one-off sensitivity check at \(S \in \{100, 1000, 5000\}\) to confirm SAA noise is negligible relative to model differences.

---

## Section 8 — Risks and Mitigations

**Technical risks.**
- *Posterior collapse* (encoder ignored; latent KL → 0). Likelihood: high, especially with a strong NB decoder. Mitigation: per-dimension KL monitored from run one; KL annealing and free bits implemented before first CVAE training; RQ4's decision-level symptom (aggregate decisions degrade) as a second detector.
- *NB parameterization instability* (dispersion → 0 or ∞; NaNs). Mitigation: softplus with floors on both parameters; gradient clipping; unit tests of the NB log-likelihood against a reference implementation before training.
- *Conditioning leakage design error* (context fed only to decoder). Mitigation: architecture checklist; the RQ2 ablation would also expose it (conditional ≈ unconditional is a red flag).

**Research risks.**
- *The pivotal comparison is null* — CVAE ≈ fitted NB. Mitigation: pre-framed as an acceptable outcome (Section 6); the analysis plan (which items/quantiles/contexts differ) turns a null into content. This is the single most likely "disappointment" and it is planned for.
- *Metrics disagree confusingly* (CRPS says A, cost says B). Mitigation: this is RQ3 — a finding, not a failure; the quantile-region analysis explains it.

**Implementation risks.**
- *Timeline compression* (see Section 9 — the real deadline is tight). Mitigation: harness-before-model ordering; mandatory/optional split in Section 3; the thin-slice rule (a submittable pipeline exists from week one).
- *Silent evaluation bugs* (the most dangerous class: wrong numbers that look plausible). Mitigation: unit tests on the newsvendor quantile against the closed form on synthetic Gaussians; evaluation validated on baselines whose behaviour is analytically predictable (Poisson under-coverage should be visible — if it is not, the harness is broken).

**Data risks.**
- *Censoring (A1) challenged by examiners.* Mitigation: acknowledged in assumptions and limitations; the relative-comparison argument stated explicitly.
- *Pre-launch zeros and unavailable weeks.* Mitigation: availability handling decided in the data pipeline (drop pre-launch periods; availability flag thereafter) and documented.
- *Chosen category unrepresentative.* Mitigation: scope is explicit; generalization across categories named as future work, not claimed.

**Evaluation risks.**
- *SAA noise mistaken for model differences.* Mitigation: the \(S\)-sensitivity check; paired statistics.
- *Cherry-picking cost ratios.* Mitigation: all three ratios pre-registered in this document; all reported.

---

## Section 9 — Timeline

Anchored to the actual course deadline (end of July 2026; ~4 working weeks from now). The semester-scale version of each milestone appears in the phase roadmap; this is the binding schedule.

**Week 1 (Jul 6–12) — Data + baselines + harness (the thin slice).**
Deliverables: scripted M5 download; FOODS weekly pipeline with temporal splits and features; all four baselines behind the sampler interface; SAA newsvendor + Level-2 harness with unit tests; first realized-cost table (baselines only).
Dependencies: none. Effort: heavy (this is the most labour-intensive week by design).
Blockers: Kaggle access; feature-engineering rabbit holes (time-box EDA to two days).

**Week 2 (Jul 13–19) — CVAE core.**
Deliverables: autoencoder → unconditional VAE (Gaussian decoder) → conditional VAE, in that order; KL monitoring live; Level-1 metrics added to the harness; first CVAE-Gaussian entry in the comparison table.
Dependencies: Week 1 harness. Effort: moderate-heavy.
Blockers: collapse appearing early (annealing/free-bits ready as per Section 8); training-time surprises (model is small — Colab suffices; if not, reduce \(L\) and network width, not the experiment set).

**Week 3 (Jul 20–26) — Count decoder + full comparison.**
Deliverables: NB decoder with stability guards; collapse handling tuned on validation; the complete RQ0 comparison (all models × 3 cost ratios × 3 seeds, paired CIs); RQ1/RQ2 ablations; RQ4 protocol comparison (no retraining, cheap).
Dependencies: Week 2 model. Effort: heavy on experiments, light on new code.
Blockers: NB instability (unit-tested in advance); seed-variance too high (report it honestly; do not tune it away).

**Week 4 (Jul 27–31) — Report + repository.**
Deliverables: report (formulation and methodology chapters drafted during Weeks 1–3 as per the running design-log habit; this week is results, discussion, limitations); repository polish per roadmap Phase 11; release tag; slides.
Dependencies: Week 3 results frozen by Jul 27. Effort: writing-dominant.
Blockers: late experiment reruns — rule: no new experiments after Jul 27 except bug-fix reruns.

**Buffer policy.** There is no explicit buffer week; the buffer is the mandatory/optional split. If Week 2 slips, extensions 7–10 are cancelled in reverse order before any mandatory item is compressed.

---

## Section 10 — Final Project Blueprint

**One paragraph.** Learn the conditional distribution of 4-week retail demand windows given context (category, store, week-of-year/events, relative price, recent lags) with a conditional VAE trained on the ELBO, using a Negative Binomial decoder to respect discrete, over-dispersed demand. Generate 1,000-scenario sets per test context, solve the aggregate-horizon newsvendor by sample average approximation at three cost ratios, and evaluate every model — CVAE and four classical baselines sharing one sampler interface — on both distributional metrics (NLL, CRPS, calibration) and realized decision cost with paired statistics over three seeds. The contribution is the decision-level empirical answer, not model novelty; a rigorous negative result is an acceptable outcome by design.

**Binding design decisions (frozen by this document).**
1. Decision task: aggregate-horizon newsvendor over \(H = 4\) weeks (primary); repeated-marginal (secondary, RQ4).
2. Data: M5 FOODS, weekly, temporal splits, train-only scaling, availability handling documented.
3. Context encoding: embeddings (item/category, store) + sin/cos week-of-year + event/SNAP flags + relative price + log1p lags; context fed to *both* encoder and decoder.
4. Model: diagonal-Gaussian encoder, NB decoder (Gaussian decoder retained as ablation), \(K = 8\text{–}16\), \(\mathcal{N}(0,I)\) prior, KL annealing + free bits available from run one.
5. Decision layer: SAA with \(S = 1000\); ratios 3:1, 1:1, 1:3, all pre-registered.
6. Evaluation: two-level; paired bootstrap; 3 seeds; harness validated on baselines before the CVAE exists.
7. Success: defined by rigor (Section 6), not by the CVAE winning.
8. Timeline: four weeks, thin-slice-first, extensions cancelled before mandatory items are compressed.

**Open items intentionally deferred to implementation (with owners = you, deadline = the week they arise).**
- Exact lag length \(L\) and network widths (Week 2, chosen on validation).
- Whether zero-inflation is needed on top of NB at weekly aggregation (Week 3 ablation, data will answer).
- Whether the diffusion extension is attempted (decision point: end of Week 3, only if all mandatory items are green).

*This document, together with the literature review and the phase roadmap, constitutes the complete pre-implementation plan. Implementation begins with Week 1's data pipeline and evaluation harness.*

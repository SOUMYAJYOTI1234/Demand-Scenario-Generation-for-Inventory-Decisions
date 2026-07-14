# Literature Review — Demand Scenario Generation for Inventory Decisions

*A critical survey of generative modeling, probabilistic demand forecasting, and inventory optimization, motivating a conditional VAE with decision-level evaluation.*

---

## 1. Introduction

Inventory management is, at its core, decision-making under uncertainty. A retailer must commit to an order quantity before demand is realized; if the order is too small, sales and goodwill are lost, and if it is too large, capital is tied up in stock that may spoil, depreciate, or require markdowns. The economic consequences of this asymmetry have been studied for well over a century — the essential problem appears as early as Edgeworth's analysis of bank cash reserves (Edgeworth, 1888) — and remain central to modern supply chain operations, where thin retail margins amplify the cost of both stockouts and overstock.

The dominant industrial practice couples a *point forecast* of demand with a safety-stock buffer. This practice embeds two assumptions that the forecasting literature has repeatedly shown to be fragile. First, it assumes that a single summary statistic (typically the conditional mean) is a sufficient input to the inventory decision. It is not: the classical newsvendor analysis shows that the optimal order quantity is a *quantile* of the demand distribution, with the quantile level determined by the ratio of underage to overage costs (Arrow, Harris and Marschak, 1951; Porteus, 2002). A model that estimates the mean well but the tails poorly can therefore produce systematically bad decisions even while scoring well on standard point-forecast accuracy metrics. Second, the safety-stock correction typically assumes Gaussian demand, an assumption that fails visibly for retail unit sales, which are non-negative, integer-valued, frequently intermittent, and over-dispersed (Croston, 1972; Syntetos and Boylan, 2005; Agrawal and Smith, 1996).

These observations motivate *probabilistic* demand modeling: estimating the full conditional distribution of demand given observable context, rather than a point summary. Once a distributional estimate is available, there are two ways to consume it downstream. If the distribution has a tractable analytical form, some inventory problems can be solved in closed form. More generally — and more usefully for modern optimization pipelines — one draws *scenarios* from the estimated distribution and solves the decision problem over the scenario set, an approach formalized in stochastic programming as sample average approximation (Kleywegt, Shapiro and Homem-de-Mello, 2002; Shapiro, Dentcheva and Ruszczyński, 2009). Scenario generation is therefore the bridge between distribution estimation and decision-making, and the quality of the generated scenarios directly bounds the quality of the resulting decisions (Høyland and Wallace, 2001; Kaut and Wallace, 2007).

Deep generative models are natural candidates for this scenario-generation role. They are trained explicitly to estimate and sample from complex, high-dimensional, conditional data distributions, and the last decade has produced a family of such models — variational autoencoders, generative adversarial networks, normalizing flows, autoregressive models, and diffusion models — with well-understood trade-offs (Kingma and Welling, 2019; Bond-Taylor et al., 2022). This review surveys the relevant literature across generative modeling, retail demand forecasting, count-data statistics, and inventory optimization, and argues that a conditional variational autoencoder (CVAE) with a count-appropriate decoder likelihood, evaluated at the level of realized inventory cost, occupies a well-motivated and comparatively under-explored position at their intersection.

## 2. Research Landscape

The project sits at the confluence of several research communities that developed largely in parallel and have only recently begun to interact systematically.

**Variational inference and probabilistic machine learning.** Variational methods for approximate Bayesian inference matured in the graphical-models era (Jordan et al., 1999) and were consolidated for the statistics community by Blei, Kucukelbir and McAuliffe (2017). The variational autoencoder (Kingma and Welling, 2014; Rezende, Mohamed and Wierstra, 2014) fused this tradition with deep learning by amortizing inference in a neural network, making latent-variable modeling scalable to large datasets.

**Deep generative modeling.** In roughly the same period, generative adversarial networks (Goodfellow et al., 2014), normalizing flows (Rezende and Mohamed, 2015; Dinh, Sohl-Dickstein and Bengio, 2017), deep autoregressive models (van den Oord, Kalchbrenner and Kavukcuoglu, 2016), and later denoising diffusion models (Sohl-Dickstein et al., 2015; Ho, Jain and Abbeel, 2020) established a spectrum of mechanisms for learning to sample from data distributions, differing principally in whether they admit explicit likelihoods, how stable they are to train, and how expensive sampling is.

**Probabilistic time-series forecasting.** The forecasting community moved from point forecasts toward full predictive distributions, with proper scoring rules providing the evaluation foundation (Gneiting and Raftery, 2007; Gneiting and Katzfuss, 2014). Deep probabilistic forecasters — most prominently DeepAR (Salinas et al., 2020) — demonstrated that neural networks trained with explicit parametric likelihoods (including the negative binomial, specifically motivated by retail demand) could outperform classical methods at scale. The M5 competitions on Walmart data marked a watershed: the accuracy track established the competitiveness of machine-learning methods on retail unit sales (Makridakis, Spiliotis and Assimakopoulos, 2022a), and the uncertainty track made *distributional* forecast quality a first-class objective (Makridakis et al., 2022b).

**Stochastic programming and inventory theory.** Independently, operations research developed the machinery for decisions under distributional uncertainty: the newsvendor model and its extensions (Arrow, Harris and Marschak, 1951; Porteus, 2002; Snyder and Shen, 2019), scenario-based stochastic programming (Birge and Louveaux, 2011), sample average approximation (Kleywegt, Shapiro and Homem-de-Mello, 2002), and, where distributional knowledge is itself unreliable, distributionally robust formulations tracing back to Scarf (1958) and modernized by Delage and Ye (2010).

**Decision-focused learning.** Most recently, a line of work at the ML–OR interface has questioned the traditional two-stage separation of "estimate, then optimize," proposing instead to train predictive models directly against downstream decision loss (Donti, Amos and Kolter, 2017; Elmachtoub and Grigas, 2022; Ban and Rudin, 2019; Bertsimas and Kallus, 2020). This literature supplies the conceptual justification for the evaluation philosophy adopted in this project: the value of a demand model is the quality of the decisions it induces.

The connecting thread is that each community solved a piece of the same pipeline — estimate a conditional distribution, sample from it, decide against the samples, and evaluate the decision — but the pieces are rarely assembled end-to-end on retail data with a deep generative model in the estimation slot. That assembly is the territory of this project.

## 3. Variational Autoencoders

### 3.1 Historical motivation

Latent-variable models posit that observed data \(x\) are generated from unobserved variables \(z\) through a conditional distribution \(p_\theta(x\mid z)\), with the model distribution given by the marginal

\[ p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz. \tag{3.1} \]

The appeal is twofold: the latent space provides a compressed, structured representation of data, and sampling is straightforward (draw \(z\), decode). The classical obstacle is inference: maximum-likelihood training requires the intractable marginal (3.1), and the posterior \(p_\theta(z\mid x)\) needed by the expectation–maximization algorithm (Dempster, Laird and Rubin, 1977) is available in closed form only for restricted families such as Gaussian mixtures. Variational inference addresses this by introducing an approximating distribution \(q(z\mid x)\) and optimizing a bound on the likelihood (Jordan et al., 1999; Blei, Kucukelbir and McAuliffe, 2017), but classical variational methods required per-datapoint optimization and conjugate model structure.

### 3.2 The VAE and the ELBO

The variational autoencoder (Kingma and Welling, 2014; concurrently Rezende, Mohamed and Wierstra, 2014) resolved both limitations with two moves. First, *amortized inference*: a single neural network (the encoder, or inference network) maps each \(x\) to the parameters of \(q_\phi(z\mid x)\), replacing per-datapoint optimization with a forward pass. Second, a low-variance, unbiased gradient estimator for the resulting objective. The objective is the evidence lower bound (ELBO), obtained via Jensen's inequality:

\[ \log p_\theta(x) \;\geq\; \mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big] \;-\; D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big) \;=:\; \mathcal{L}(\theta,\phi;x). \tag{3.2} \]

The first term rewards accurate reconstruction of \(x\) from latent codes drawn from the approximate posterior; the second regularizes the posterior toward the prior, typically \(p(z)=\mathcal{N}(0,I)\). The gap between the ELBO and the true log-likelihood equals \(D_{\mathrm{KL}}(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x))\), so maximizing the ELBO jointly fits the generative model and improves the posterior approximation.

### 3.3 The reparameterization trick

The gradient of (3.2) with respect to \(\phi\) is a gradient of an expectation taken over a distribution that itself depends on \(\phi\); naive differentiation yields terms that are not expectations and cannot be estimated by sampling. The reparameterization trick expresses the latent variable as a deterministic transformation of parameter-free noise, \(z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon\) with \(\varepsilon \sim \mathcal{N}(0, I)\), so that the expectation is taken over \(\varepsilon\) and the gradient passes inside (Kingma and Welling, 2014). Compared with the score-function (REINFORCE) estimator available for arbitrary distributions, the reparameterized estimator has dramatically lower variance, which is the practical reason VAEs train stably.

### 3.4 Strengths, limitations, and the posterior-collapse literature

The VAE's principal advantages for the present application are: (i) an explicit probabilistic model with a computable objective, enabling likelihood-based comparison against classical statistical baselines; (ii) an encoder, providing representations of observed data; (iii) stable, standard gradient-based training; and (iv) complete freedom in the choice of decoder likelihood \(p_\theta(x\mid z)\), which is what allows a count distribution to be used for demand data.

Its best-documented weaknesses are also relevant. First, samples from Gaussian-decoder VAEs on images are characteristically blurry — commonly attributed to the mode-covering behaviour of the maximum-likelihood objective and the restrictive decoder family. Notably, this criticism carries far less weight for low-dimensional demand vectors than for natural images, where perceptual sharpness matters; for scenario generation, calibrated spread is the desideratum, not visual fidelity. Second, *posterior collapse*: the optimizer can drive \(D_{\mathrm{KL}}(q_\phi\Vert p)\) to zero, at which point the latent code carries no information and the model degenerates to an unconditional decoder. The phenomenon was documented in sequence modeling by Bowman et al. (2016), who introduced KL annealing (gradually increasing the weight on the KL term) as a mitigation; subsequent analyses and remedies include free bits (Kingma et al., 2016), aggressive inference-network training (He et al., 2019), constrained-rate formulations (Razavi et al., 2019), and the rate–distortion perspective of Alemi et al. (2018), which reframes the ELBO's apparent pathologies as movement along a rate–distortion curve. The β-VAE of Higgins et al. (2017) generalizes the objective by weighting the KL term, trading reconstruction against latent regularity. For this project, KL annealing and free bits are the directly applicable tools, and monitoring per-dimension KL during training is the standard diagnostic.

A fair critical summary of the VAE literature is that the model's theory is unusually clean — the ELBO is a principled, interpretable objective with a precise relationship to maximum likelihood — while its practical behaviour depends delicately on the balance between the two ELBO terms, a balance that must be actively managed rather than assumed.

## 4. Conditional Variational Autoencoders

The standard VAE models the marginal \(p(x)\). Many tasks instead require the conditional \(p(x\mid y)\) for context \(y\) — in this project, demand given product category, store, seasonality, and price. Sohn, Lee and Yan (2015) introduced the conditional VAE for structured prediction, deriving the conditional ELBO

\[ \log p_\theta(x\mid y) \;\geq\; \mathbb{E}_{q_\phi(z\mid x,y)}\big[\log p_\theta(x\mid z,y)\big] \;-\; D_{\mathrm{KL}}\big(q_\phi(z\mid x,y)\,\Vert\,p_\theta(z\mid y)\big), \tag{4.1} \]

in which both the encoder and the decoder receive the conditioning information, and the prior may itself be conditioned on \(y\) (in practice a fixed \(\mathcal{N}(0,I)\) prior is common and adequate). The essential division of labour is that \(y\) explains the *systematic* variation in \(x\) (seasonal level, price response), while \(z\) captures residual, unexplained variability — precisely the decomposition wanted for demand: the context determines the expected demand regime, the latent variable generates the stochastic scenario around it.

Mechanically, conditioning is implemented by concatenating an embedding of \(y\) (learned embeddings for categorical variables; normalized values for continuous ones) to the inputs of both networks. A recurrent design error, noted throughout the applied literature, is to condition only the decoder; the encoder must also observe \(y\), otherwise the posterior is forced to encode context information into \(z\), corrupting the intended decomposition.

CVAEs have been applied to structured output prediction (Sohn, Lee and Yan, 2015), controllable generation, and trajectory or scenario prediction in domains such as autonomous driving. Their strengths mirror the VAE's — explicit likelihood, cheap sampling, stable training — with the addition of controllable, context-aware generation. Their weaknesses likewise carry over: latent capacity must be balanced against collapse, and the conditional decomposition is only as good as the conditioning features. For conditional demand generation, where each context requires an entire distribution of plausible outcomes and where thousands of scenario draws must be cheap, the CVAE's single-forward-pass sampling is a decisive practical advantage.

## 5. Alternative Generative Models

A defensible model choice requires an honest survey of the alternatives.

**Generative adversarial networks.** GANs (Goodfellow et al., 2014) train a generator against a discriminator in a min–max game; the f-GAN framework (Nowozin, Cseke and Tomioka, 2016) later showed the objective to be a variational estimate of an f-divergence. Conditional GANs (Mirza and Osindero, 2014) admit context in both networks. GANs produce sharp samples and have been used for scenario generation in the energy domain (Chen et al., 2018). Their liabilities are well documented: no explicit likelihood (precluding direct comparison with statistical baselines and complicating calibration analysis), no encoder, and notoriously unstable saddle-point training with mode collapse as a characteristic failure — a particularly serious defect for scenario generation, where losing modes means understating risk. Stabilization via Wasserstein objectives and gradient penalties (Arjovsky, Chintala and Bottou, 2017; Gulrajani et al., 2017) mitigates but does not remove these issues.

**Denoising diffusion models.** Diffusion models (Sohl-Dickstein et al., 2015; Ho, Jain and Abbeel, 2020), closely related to score-based models (Song and Ermon, 2019), define a fixed forward noising process and learn its reverse, and can be read as deep hierarchical VAEs with a fixed inference chain. They currently set the state of the art in sample quality across many domains, and TimeGrad demonstrated their effectiveness for multivariate probabilistic time-series forecasting (Rasul et al., 2021). Their costs are an iterative, comparatively expensive sampling procedure (a real consideration when SAA requires thousands of scenarios per decision), only a bound on the likelihood, and greater implementation complexity. For this project, diffusion is the natural *extension* rather than the starting point: the pipeline is model-agnostic at the sampler interface, so a conditional diffusion generator can later be dropped into the CVAE's slot and compared on identical decision metrics.

**Normalizing flows.** Flows (Rezende and Mohamed, 2015; Dinh, Sohl-Dickstein and Bengio, 2017; Kingma and Dhariwal, 2018; survey: Papamakarios et al., 2021) compose invertible transformations to give exact likelihoods and exact sampling. Their constraint is architectural: invertibility with tractable Jacobians restricts the transformation family, and standard flows model continuous densities, making integer-valued, zero-inflated demand awkward without dequantization or discrete-flow machinery. They remain a credible alternative and are noted as future work.

**Autoregressive models.** Deep autoregressive models factorize the joint distribution over dimensions and achieve exact likelihoods (van den Oord, Kalchbrenner and Kavukcuoglu, 2016; van den Oord et al., 2016). In forecasting, DeepAR (Salinas et al., 2020) is exactly such a model over time steps with a parametric emission. They are strong, well-understood baselines; relative to a latent-variable design they offer no compact scenario-level representation, and sampling is sequential. The distinction from this project is architectural rather than philosophical, and Section 6 returns to it.

**Energy-based models.** EBMs define unnormalized densities and are trained via approximations to the likelihood gradient (LeCun et al., 2006; Du and Mordatch, 2019). Their sampling requires MCMC, which is difficult to justify inside an SAA loop; they are noted for completeness only.

**Comparison.**

| Criterion | CVAE | cGAN | Diffusion | Flows | Autoregressive | EBM |
|---|---|---|---|---|---|---|
| Explicit likelihood | Bound (ELBO) | No | Bound | Exact | Exact | Unnormalized |
| Training stability | High | Low | High | High | High | Low–medium |
| Sampling cost | One pass | One pass | Many passes | One pass | Sequential | MCMC |
| Encoder / representation | Yes | No | No (native) | Bijective | No | No |
| Flexible count likelihood | Yes (decoder choice) | Indirect | Nontrivial | Awkward | Yes (emission choice) | Possible |
| Mode coverage tendency | Covering | Dropping risk | Covering | Covering | Covering | Varies |
| Implementation complexity | Low | Medium | High | Medium | Low–medium | High |

**Justification of the CVAE choice.** For demand scenario generation the requirements are: cheap conditional sampling at SAA scale; an explicit training objective comparable against statistical baselines; direct support for count likelihoods; stable training within a constrained project timeline; and mode-covering behaviour, since underrepresenting demand modes corrupts tail quantiles and hence decisions. The CVAE satisfies all five. GANs fail the likelihood, stability, and mode-coverage criteria; diffusion fails sampling cost and simplicity (while remaining the best-motivated extension); flows fail count-data convenience; autoregressive models are the nearest competitor and are effectively represented in this project's baseline set through parametric likelihood models in the DeepAR tradition.

## 6. Retail Demand Forecasting

Classical retail forecasting rests on exponential smoothing and ARIMA-class models (consolidated in Hyndman and Athanasopoulos, 2021), extended to intermittent series by Croston (1972), whose method separately smooths demand sizes and inter-demand intervals, and corrected for bias by Syntetos and Boylan (2005). These methods remain strong baselines — a recurring and humbling finding of the M-competitions — but they produce per-series models that cannot pool information across thousands of related products.

Deep learning changed the economics of cross-series learning. DeepAR (Salinas et al., 2020) trains a single autoregressive RNN across all series, emitting parameters of a per-step likelihood — Gaussian for continuous magnitudes, *negative binomial for retail unit sales* — and demonstrated consistent gains on retail data. This paper is the closest methodological precedent for the present project's decoder-likelihood argument. Subsequent architectures include MQ-RNN's direct multi-horizon quantile forecasts (Wen et al., 2017), deep state-space hybrids (Rangapuram et al., 2018), N-BEATS for point forecasting (Oreshkin et al., 2020), and the Temporal Fusion Transformer, which couples attention-based multi-horizon forecasting with quantile outputs and interpretability mechanisms (Lim et al., 2021).

The M5 competitions provide both the dataset used in this project and the definitive empirical reference on it. The accuracy track (Makridakis, Spiliotis and Assimakopoulos, 2022a) established that gradient-boosted trees and deep models dominate on Walmart unit sales, and documented the dataset's severe intermittency (a majority of item–store–day observations are zero). The uncertainty track (Makridakis et al., 2022b) evaluated quantile forecasts with pinball loss, foregrounding distributional quality.

The distinction between this forecasting literature and *scenario generation* deserves emphasis, because the two are frequently conflated. Probabilistic forecasters output per-step predictive distributions or quantiles; scenario generation requires *joint* samples of coherent demand trajectories, preserving temporal dependence within the horizon — a stockout decision depends on whether high-demand weeks cluster, which marginal quantiles do not reveal. Latent-variable models produce such joint samples natively: a single draw of \(z\) generates an entire correlated trajectory. Quantile-based forecasters require additional machinery (e.g., copula-style post-processing) to yield joint scenarios. This distinction is a substantive part of the motivation for a generative-model approach rather than a purely forecasting one.

## 7. Probability Models for Retail Demand

The choice of observation model is not a technicality; it determines which aspects of demand the model can represent at all.

**Gaussian.** The default in classical inventory theory, largely for tractability of the critical-fractile formula. Its failures for unit sales are structural: it assigns mass to negative demand, is continuous rather than integer-valued, is symmetric where demand is right-skewed, and ties tail thickness to the same parameter as central spread.

**Poisson.** The canonical count model, with a single rate parameter and the defining property that the variance equals the mean. Empirical retail demand almost universally violates this equality in the direction of *over-dispersion* — variance exceeding the mean — driven by heterogeneity across customers, promotions, and unobserved factors (Cameron and Trivedi, 2013).

**Negative binomial.** The standard remedy for over-dispersion, expressible as a Gamma–Poisson mixture: a Poisson rate that is itself Gamma-distributed yields marginal variance \(\mu + \mu^2/r\), strictly exceeding the mean, with dispersion \(r\) learned from data. Its suitability for retail demand is not merely asymptotic folklore: Agrawal and Smith (1996) fitted count families to retail sales data and found the negative binomial markedly superior to Poisson and normal fits, and DeepAR adopted it as the recommended likelihood for unit sales (Salinas et al., 2020). It is the primary decoder candidate in this project.

**Zero-inflated models.** When zeros arise from a distinct structural process (item unavailable, no store traffic) rather than purely from low demand rates, a zero-inflated formulation mixes a point mass at zero with a count distribution (Lambert, 1992). Whether M5's weekly-aggregated series require zero inflation beyond what a negative binomial captures is an empirical question flagged for the ablation phase rather than assumed either way.

The critical point for the project's architecture is that in a VAE the observation model is simply the decoder likelihood: swapping Gaussian for negative binomial changes the reconstruction term of the ELBO and nothing else. The generative-model literature and the count-data literature compose cleanly here, and making that composition explicit is one of the project's contributions in exposition if not in technique.

## 8. Inventory Optimization

**The newsvendor problem.** A single-period order quantity \(Q\) is chosen against random demand \(D\) with underage cost \(c_u\) per unsold unit of demand and overage cost \(c_o\) per unit of excess stock, minimizing \(\mathbb{E}[c_u (D-Q)^+ + c_o (Q-D)^+]\). The optimal solution is the *critical fractile*:

\[ Q^{*} = F^{-1}\!\left(\frac{c_u}{c_u + c_o}\right), \tag{8.1} \]

a specific quantile of the demand distribution \(F\) (Arrow, Harris and Marschak, 1951; textbook treatments in Porteus, 2002 and Snyder and Shen, 2019). Equation (8.1) is the mathematical hinge of this entire project: it says the decision consumes a quantile, so a demand model's value depends on the accuracy of the *particular quantile the cost structure selects* — which likelihood-based training does not specifically target, and which point-forecast metrics do not measure at all. Different cost ratios probe different quantiles, so evaluating decisions across several ratios amounts to probing several regions of the learned distribution.

**Stochastic programming and SAA.** When \(F\) is known only through samples — precisely the situation with a generative model — the expectation is replaced by an average over \(S\) scenarios and the resulting deterministic problem is solved; for the newsvendor this reduces to an empirical quantile. Sample average approximation carries convergence guarantees as \(S\) grows (Kleywegt, Shapiro and Homem-de-Mello, 2002; Shapiro, Dentcheva and Ruszczyński, 2009; Birge and Louveaux, 2011). The scenario-generation literature within stochastic programming (Høyland and Wallace, 2001; Kaut and Wallace, 2007) established two principles adopted wholesale here: scenario quality should be judged by the *stability and quality of the decisions* it induces, and scenario-generation methods are legitimately compared on downstream objective value. This project transplants exactly that evaluation philosophy to deep generative models.

**Data-driven and robust variants.** Ban and Rudin (2019) solve the newsvendor directly from feature data, bypassing distribution estimation; Bertsimas and Kallus (2020) develop the general predictive-to-prescriptive framework; Elmachtoub and Grigas (2022) train predictors against decision loss ("Smart Predict-then-Optimize"); Donti, Amos and Kolter (2017) differentiate through optimization layers end-to-end. Where the estimated distribution itself is distrusted, distributionally robust optimization hedges over an ambiguity set, from Scarf's (1958) min–max newsvendor to moment-based DRO (Delage and Ye, 2010). These lines matter to this review for two reasons: they justify decision-level evaluation as the appropriate metric, and they define the most natural extension — training the generator through the decision objective — while the present project deliberately retains the two-stage estimate-then-optimize design for interpretability and scope control.

## 9. Evaluation of Generative Models

Evaluation is where this project stakes its main claim, so the metric hierarchy deserves careful treatment.

**Likelihood-family metrics.** The ELBO and held-out negative log-likelihood measure global distributional fit and permit direct comparison among likelihood-based models (a comparison from which GANs are structurally excluded). Reconstruction error isolates the decoder's fidelity but ignores the prior-sampling path actually used in deployment. These metrics are necessary diagnostics but answer "does the model fit the data on average," not "does it support good decisions."

**Distributional forecast metrics.** The proper-scoring-rule framework (Gneiting and Raftery, 2007) supplies the continuous ranked probability score (CRPS), estimable from samples and sensitive to both calibration and sharpness; the guiding paradigm is "maximizing sharpness subject to calibration" (Gneiting and Katzfuss, 2014). Calibration is checked via probability integral transform histograms and quantile coverage (does the nominal 90% interval cover ~90% of outcomes). These metrics evaluate the predictive distribution on its own terms and form this project's Level-1 evaluation.

**Decision-based metrics.** Level-2 evaluation feeds each model's scenarios through the identical SAA newsvendor and records *realized* cost against actual held-out demand, along with service level, fill rate, stockout frequency, and overstock. The argument that this level is the more meaningful one for this project has three independent supports. First, the critical-fractile structure (8.1) means the decision depends on specific quantiles, and models with similar aggregate scores can differ materially exactly there. Second, the stochastic-programming tradition already evaluates scenario methods by induced decision quality (Kaut and Wallace, 2007), so the standard is established, not invented. Third, the decision-focused learning literature demonstrates empirically that prediction loss and decision loss can rank models differently (Elmachtoub and Grigas, 2022; Donti, Amos and Kolter, 2017). A candid corollary follows: where CRPS and realized cost *disagree* in this project's results, the disagreement is itself a finding about the insufficiency of purely statistical evaluation, and will be reported as such.

## 10. Research Gap

Assembled, the literature exhibits the following configuration. Deep probabilistic forecasting of retail demand is mature (Salinas et al., 2020; Lim et al., 2021; Makridakis et al., 2022b), but is evaluated almost exclusively with statistical scores — pinball loss, CRPS, weighted quantile losses — and stops short of the decision. Scenario generation with deep generative models is established in adjacent domains, notably energy systems (Chen et al., 2018), but retail demand has not been a primary target, and VAE-based conditional scenario generation with count likelihoods is thinly represented. The data-driven newsvendor literature connects features to decisions (Ban and Rudin, 2019; Bertsimas and Kallus, 2020) but typically without a deep generative model in the loop, and decision-focused learning trains discriminative predictors rather than scenario generators. Finally, the scenario-evaluation principles of stochastic programming (Kaut and Wallace, 2007) predate deep generative models and have not been systematically applied to them.

The gap this project addresses is therefore the *composition*: a conditional deep generative model, with a count-appropriate likelihood, generating demand scenarios for a canonical inventory decision, evaluated head-to-head against classical probabilistic baselines on realized decision cost. Stated honestly, the expected contribution is not a new model class or algorithm; it is a careful, reproducible empirical study of whether deep conditional generation adds *decision value* over well-fitted classical distributions on public retail data — a question the surveyed literatures each imply but none directly answers. A well-executed negative result (the CVAE failing to beat a per-series negative binomial on cost) would itself be informative, delimiting where deep generative machinery earns its complexity.

## 11. Critical Discussion

Several cross-cutting judgments emerge from the survey.

**Realism of assumptions.** The Gaussian demand assumption pervading textbook inventory theory is analytically motivated, not empirically; the count-data literature is unambiguous that over-dispersed count models fit retail sales better (Agrawal and Smith, 1996; Cameron and Trivedi, 2013). Conversely, the i.i.d.-scenario assumption underlying SAA is reasonable within a short horizon but strains under regime changes — an honest limitation of the two-stage design shared by nearly all surveyed work.

**Data efficiency.** Classical per-series models (Croston variants, fitted NB) need little data but cannot pool across items; deep cross-series models pool aggressively but need the scale that M5 provides. This is precisely why the plain fitted negative binomial is the pivotal baseline: it isolates whether the deep model's cross-series conditioning adds value beyond the likelihood family itself.

**Trainability.** On stability, the ordering is clear: likelihood-based models (VAE, flows, autoregressive) train predictably; GANs do not; diffusion trains stably but samples slowly. For a timeline-constrained project whose scientific weight rests on the *evaluation* rather than the generator, choosing the most controllable adequate generator is sound methodology, not conservatism.

**Fit to inventory optimization.** The decision layer requires many cheap joint samples per context and rewards calibrated tails. This favours one-pass samplers with mode-covering objectives (CVAE) over mode-dropping ones (GANs) and over expensive iterative samplers (diffusion) — while acknowledging that diffusion's distributional quality may ultimately justify its cost, which is exactly what the planned extension would test.

**Evaluation philosophy.** The strongest single lesson across communities is convergent: stochastic programming (Kaut and Wallace, 2007), decision-focused learning (Elmachtoub and Grigas, 2022), and the sharpness-subject-to-calibration doctrine (Gneiting and Katzfuss, 2014) all point away from evaluating generative demand models purely on statistical fit. This project's two-level evaluation operationalizes that convergence.

## 12. Summary

The review supports the project's design decisions as follows. Demand scenario generation matters because inventory decisions consume distributions — specifically quantiles — not point forecasts (Section 8). Probabilistic generative models are the appropriate tool because they estimate and sample conditional distributions natively, producing the joint trajectories that scenario-based optimization requires (Sections 1, 6). Among generative families, the conditional VAE offers the combination this task needs: explicit likelihood-based training, one-pass conditional sampling at SAA scale, a freely chosen decoder likelihood, stable optimization, and mode-covering behaviour (Sections 3–5). The negative binomial decoder follows from the count-data literature's consistent finding of over-dispersion in retail sales (Section 7; Agrawal and Smith, 1996; Salinas et al., 2020). The M5 dataset is the canonical public testbed with exactly the required conditioning structure and documented intermittency (Section 6). Finally, newsvendor-based, decision-level evaluation — realized cost against classical baselines across multiple cost ratios — implements the scenario-evaluation standard of stochastic programming and the central insight of decision-focused learning within a deliberately interpretable two-stage design (Sections 8–9), and it is at this evaluative level, rather than in model novelty, that the project's contribution honestly lies.

---

## References

Agrawal, N. and Smith, S. A. (1996). Estimating negative binomial demand for retail inventory management with unobservable lost sales. *Naval Research Logistics*, 43(6), 839–861.

Alemi, A. A., Poole, B., Fischer, I., Dillon, J. V., Saurous, R. A. and Murphy, K. (2018). Fixing a broken ELBO. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.

Arjovsky, M., Chintala, S. and Bottou, L. (2017). Wasserstein generative adversarial networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*.

Arrow, K. J., Harris, T. and Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250–272.

Ban, G.-Y. and Rudin, C. (2019). The big data newsvendor: Practical insights from machine learning. *Operations Research*, 67(1), 90–108.

Bertsimas, D. and Kallus, N. (2020). From predictive to prescriptive analytics. *Management Science*, 66(3), 1025–1044.

Birge, J. R. and Louveaux, F. (2011). *Introduction to Stochastic Programming* (2nd ed.). Springer.

Blei, D. M., Kucukelbir, A. and McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877.

Bond-Taylor, S., Leach, A., Long, Y. and Willcocks, C. G. (2022). Deep generative modelling: A comparative review of VAEs, GANs, normalizing flows, energy-based and autoregressive models. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(11), 7327–7347.

Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R. and Bengio, S. (2016). Generating sentences from a continuous space. *Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning (CoNLL)*.

Cameron, A. C. and Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.

Chen, Y., Wang, Y., Kirschen, D. and Zhang, B. (2018). Model-free renewable scenario generation using generative adversarial networks. *IEEE Transactions on Power Systems*, 33(3), 3265–3275.

Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Operational Research Quarterly*, 23(3), 289–303.

Delage, E. and Ye, Y. (2010). Distributionally robust optimization under moment uncertainty with application to data-driven problems. *Operations Research*, 58(3), 595–612.

Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38.

Dinh, L., Sohl-Dickstein, J. and Bengio, S. (2017). Density estimation using Real NVP. *International Conference on Learning Representations (ICLR)*.

Donti, P., Amos, B. and Kolter, J. Z. (2017). Task-based end-to-end model learning in stochastic optimization. *Advances in Neural Information Processing Systems (NeurIPS)*.

Du, Y. and Mordatch, I. (2019). Implicit generation and modeling with energy based models. *Advances in Neural Information Processing Systems (NeurIPS)*.

Edgeworth, F. Y. (1888). The mathematical theory of banking. *Journal of the Royal Statistical Society*, 51(1), 113–127.

Elmachtoub, A. N. and Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science*, 68(1), 9–26.

Gneiting, T. and Katzfuss, M. (2014). Probabilistic forecasting. *Annual Review of Statistics and Its Application*, 1, 125–151.

Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems (NeurIPS)*.

Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. and Courville, A. (2017). Improved training of Wasserstein GANs. *Advances in Neural Information Processing Systems (NeurIPS)*.

He, J., Spokoyny, D., Neubig, G. and Berg-Kirkpatrick, T. (2019). Lagging inference networks and posterior collapse in variational autoencoders. *International Conference on Learning Representations (ICLR)*.

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S. and Lerchner, A. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. *International Conference on Learning Representations (ICLR)*.

Ho, J., Jain, A. and Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems (NeurIPS)*.

Høyland, K. and Wallace, S. W. (2001). Generating scenario trees for multistage decision problems. *Management Science*, 47(2), 295–307.

Hyndman, R. J. and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

Jordan, M. I., Ghahramani, Z., Jaakkola, T. S. and Saul, L. K. (1999). An introduction to variational methods for graphical models. *Machine Learning*, 37, 183–233.

Kaut, M. and Wallace, S. W. (2007). Evaluation of scenario-generation methods for stochastic programming. *Pacific Journal of Optimization*, 3(2), 257–271.

Kingma, D. P. and Dhariwal, P. (2018). Glow: Generative flow with invertible 1×1 convolutions. *Advances in Neural Information Processing Systems (NeurIPS)*.

Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I. and Welling, M. (2016). Improved variational inference with inverse autoregressive flow. *Advances in Neural Information Processing Systems (NeurIPS)*.

Kingma, D. P. and Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations (ICLR)*.

Kingma, D. P. and Welling, M. (2019). An introduction to variational autoencoders. *Foundations and Trends in Machine Learning*, 12(4), 307–392.

Kleywegt, A. J., Shapiro, A. and Homem-de-Mello, T. (2002). The sample average approximation method for stochastic discrete optimization. *SIAM Journal on Optimization*, 12(2), 479–502.

Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics*, 34(1), 1–14.

LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. and Huang, F. (2006). A tutorial on energy-based learning. In *Predicting Structured Data*. MIT Press.

Lim, B., Arık, S. Ö., Loeff, N. and Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764.

Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2022a). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364.

Makridakis, S., Spiliotis, E., Assimakopoulos, V., Chen, Z., Gaba, A., Tsetlin, I. and Winkler, R. L. (2022b). The M5 uncertainty competition: Results, findings and conclusions. *International Journal of Forecasting*, 38(4), 1365–1385.

Mirza, M. and Osindero, S. (2014). Conditional generative adversarial nets. *arXiv preprint arXiv:1411.1784*.

Nowozin, S., Cseke, B. and Tomioka, R. (2016). f-GAN: Training generative neural samplers using variational divergence minimization. *Advances in Neural Information Processing Systems (NeurIPS)*.

Oreshkin, B. N., Carpov, D., Chapados, N. and Bengio, Y. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *International Conference on Learning Representations (ICLR)*.

Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S. and Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and inference. *Journal of Machine Learning Research*, 22(57), 1–64.

Porteus, E. L. (2002). *Foundations of Stochastic Inventory Theory*. Stanford University Press.

Rangapuram, S. S., Seeger, M., Gasthaus, J., Stella, L., Wang, Y. and Januschowski, T. (2018). Deep state space models for time series forecasting. *Advances in Neural Information Processing Systems (NeurIPS)*.

Rasul, K., Seward, C., Schuster, I. and Vollgraf, R. (2021). Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.

Razavi, A., van den Oord, A., Poole, B. and Vinyals, O. (2019). Preventing posterior collapse with δ-VAEs. *International Conference on Learning Representations (ICLR)*.

Rezende, D. J. and Mohamed, S. (2015). Variational inference with normalizing flows. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.

Rezende, D. J., Mohamed, S. and Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. *Proceedings of the 31st International Conference on Machine Learning (ICML)*.

Salinas, D., Flunkert, V., Gasthaus, J. and Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181–1191.

Scarf, H. (1958). A min-max solution of an inventory problem. In *Studies in the Mathematical Theory of Inventory and Production*. Stanford University Press.

Shapiro, A., Dentcheva, D. and Ruszczyński, A. (2009). *Lectures on Stochastic Programming: Modeling and Theory*. SIAM.

Snyder, L. V. and Shen, Z.-J. M. (2019). *Fundamentals of Supply Chain Theory* (2nd ed.). Wiley.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. and Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.

Sohn, K., Lee, H. and Yan, X. (2015). Learning structured output representation using deep conditional generative models. *Advances in Neural Information Processing Systems (NeurIPS)*.

Song, Y. and Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *Advances in Neural Information Processing Systems (NeurIPS)*.

Syntetos, A. A. and Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303–314.

van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A. and Kavukcuoglu, K. (2016). WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*.

van den Oord, A., Kalchbrenner, N. and Kavukcuoglu, K. (2016). Pixel recurrent neural networks. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*.

"""Decoder likelihoods for count demand — CVAE Milestone 4 (roadmap Phase 6.4).

The Negative Binomial is the primary decoder likelihood (design doc §10.4):
discrete, non-negative, and over-dispersed (variance mu + mu^2/r > mu), which
is what M5 FOODS demand empirically is (median var/mean ~ 4, EDA Phase 4).
DeepAR (Salinas et al. 2020) is the precedent for NB emissions on retail
unit sales.

Parameterization (mean mu > 0, dispersion r > 0):

    log p(x | mu, r) = lgamma(x + r) - lgamma(r) - lgamma(x + 1)
                       + r * log(r / (r + mu)) + x * log(mu / (r + mu))

Var = mu + mu^2 / r; r -> inf recovers Poisson. Unit-tested against
scipy.stats.nbinom.logpmf with n = r, p = r / (r + mu).

Stability guards (design doc §8, "NB parameterization instability"): decoder
heads pass through softplus and are floored at MU_FLOOR / R_FLOOR, so every
log/lgamma argument stays strictly positive — no NaNs at 3 a.m.
"""

from __future__ import annotations

import torch

MU_FLOOR = 0.01  # smallest decoder mean: keeps log(mu/(r+mu)) finite
R_FLOOR = 0.1  # smallest dispersion: keeps lgamma(r) and the r*log term sane


def nb_log_likelihood(x: torch.Tensor, mu: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Elementwise Negative Binomial log-pmf log p(x | mu, r); shape-preserving.

    ``x`` are raw demand counts (the NB decoder works on the original count
    scale, unlike the Gaussian decoder's log1p space). Finite for x = 0 —
    zero weeks are common in M5 — provided mu, r respect the floors.
    """
    x = x.to(mu.dtype)
    log_r_ratio = torch.log(r) - torch.log(r + mu)  # log(r / (r + mu))
    log_mu_ratio = torch.log(mu) - torch.log(r + mu)  # log(mu / (r + mu))
    return (
        torch.lgamma(x + r)
        - torch.lgamma(r)
        - torch.lgamma(x + 1.0)
        + r * log_r_ratio
        + x * log_mu_ratio
    )

"""Classical probabilistic baselines (roadmap Phase 5).

The project's entire claim is comparative, so these are the measuring stick.
All four implement the ``BaseSampler`` contract — ``sample(context,
n_scenarios)`` / vectorized ``sample_batch`` — so the decision and evaluation
layers are model-agnostic. Every statistic is fitted on each series' **train
span only** (leakage guard, roadmap Phase 4): per-series moments use the
available (priced) weeks inside the train span, exactly the data a practitioner
would have at decision time.

Context contract: classical samplers key on ``context["series_row"]`` — the
row of the item-store series in the :class:`~demand_vae.data.pipeline.WeeklyPanel`.

The four baselines and why each exists (design doc §7):
1. :class:`EmpiricalResampler` — the strongest assumption-free baseline.
2. :class:`GaussianSampler` — the textbook inventory-theory assumption.
3. :class:`PoissonSampler` — the naive count model; its under-dispersion is
   part of the narrative.
4. :class:`NegativeBinomialSampler` — **the pivotal baseline**: a
   non-conditional NB. If the CVAE beats Gaussian but not this, the win came
   from the likelihood family, not deep conditioning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from demand_vae.data.pipeline import WeeklyPanel
from demand_vae.models.base import BaseSampler


def nb_params_from_moments(mean: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Method-of-moments Negative Binomial parameters (r, p) from mean/variance.

    Solves var = mean + mean^2 / r, so r = mean^2 / (var - mean) and
    p = r / (r + mean); then E[NB(r, p)] = mean and Var = var exactly.

    Degenerate cases: series with var <= mean (not over-dispersed) get a huge
    r, making the NB numerically a Poisson; all-zero series get p = 1, which
    samples exact zeros.
    """
    mean = np.asarray(mean, dtype=float)
    var = np.asarray(var, dtype=float)
    overdispersed = var > mean + 1e-9
    r = np.where(overdispersed, mean**2 / np.maximum(var - mean, 1e-9), 1e6)
    r = np.maximum(r, 1e-6)  # numpy requires n > 0 even when p = 1
    p = r / (r + mean)
    return r, p


def _train_moments(
    panel: WeeklyPanel, train_span: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-series (mean, var) of weekly demand over available train weeks."""
    lo, hi = train_span
    sales = panel.sales[:, lo : hi + 1].astype(float)
    avail = panel.available[:, lo : hi + 1]
    n = np.maximum(avail.sum(axis=1), 1)
    mean = np.where(avail, sales, 0.0).sum(axis=1) / n
    var = (np.where(avail, (sales - mean[:, None]) ** 2, 0.0)).sum(axis=1) / n
    return mean, var


class _PerSeriesSampler(BaseSampler):
    """Shared scaffolding: fit-on-train-span bookkeeping and context handling."""

    def __init__(self, horizon: int, seed: int | None = None):
        super().__init__(horizon)
        self._rng = np.random.default_rng(seed)
        self.fitted_ = False

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{type(self).__name__} must be fit(panel, train_span) first")

    @staticmethod
    def _rows(context_batch: dict[str, np.ndarray]) -> np.ndarray:
        return np.atleast_1d(np.asarray(context_batch["series_row"]))

    def sample(self, context: Any, n_scenarios: int) -> np.ndarray:
        batch = {"series_row": np.atleast_1d(np.asarray(context["series_row"]))}
        return self.sample_batch(batch, n_scenarios)[0]


class EmpiricalResampler(_PerSeriesSampler):
    """Resample contiguous H-week demand windows from the series' own train history.

    The strongest cheap baseline: if the CVAE cannot beat resampling history,
    conditioning has added nothing. Scenarios are actual historical windows,
    so within-window correlation is preserved for free. Series too short for
    any full H-week train window fall back to H independently resampled
    single weeks (documented; affects only fringe series).
    """

    def fit(self, panel: WeeklyPanel, train_span: tuple[int, int]) -> EmpiricalResampler:
        lo, hi = train_span
        horizon = self.horizon
        self._sales = panel.sales  # (n_series, n_weeks); windows gathered lazily

        swv = np.lib.stride_tricks.sliding_window_view
        avail_win = swv(panel.available, horizon, axis=1).all(axis=2)
        n_starts_total = avail_win.shape[1]
        start_ok = np.zeros_like(avail_win)
        last_valid_start = min(hi - horizon + 1, n_starts_total - 1)
        if last_valid_start >= lo:
            start_ok[:, lo : last_valid_start + 1] = avail_win[:, lo : last_valid_start + 1]

        counts = start_ok.sum(axis=1)
        max_starts = max(int(counts.max()), 1)
        starts = np.zeros((panel.n_series, max_starts), dtype=np.int64)
        for row in np.flatnonzero(counts):
            idx = np.flatnonzero(start_ok[row])
            starts[row, : idx.size] = idx
        self._starts, self._start_counts = starts, counts

        # Fallback pool for series with no full train window: single available weeks.
        week_ok = panel.available[:, lo : hi + 1]
        week_counts = week_ok.sum(axis=1)
        weeks = np.zeros((panel.n_series, max(int(week_counts.max()), 1)), dtype=np.int64)
        for row in range(panel.n_series):
            idx = np.flatnonzero(week_ok[row]) + lo
            weeks[row, : idx.size] = idx
        self._weeks, self._week_counts = weeks, week_counts
        self.fitted_ = True
        return self

    def sample_batch(self, context_batch: dict[str, np.ndarray], n_scenarios: int) -> np.ndarray:
        self._require_fitted()
        rows = self._rows(context_batch)
        n = rows.size
        out = np.empty((n, n_scenarios, self.horizon), dtype=float)

        counts = self._start_counts[rows]
        u = self._rng.random((n, n_scenarios))
        has_window = counts > 0
        if has_window.any():
            r = rows[has_window]
            pick = (u[has_window] * counts[has_window][:, None]).astype(np.int64)
            t = self._starts[r[:, None], pick]  # (m, S) window starts
            cols = t[..., None] + np.arange(self.horizon)  # (m, S, H)
            out[has_window] = self._sales[r[:, None, None], cols]
        if (~has_window).any():
            r = rows[~has_window]
            wk_counts = np.maximum(self._week_counts[r], 1)
            u_weeks = self._rng.random((r.size, n_scenarios, self.horizon))
            pick = (u_weeks * wk_counts[:, None, None]).astype(np.int64)
            cols = self._weeks[r[:, None, None], pick]
            out[~has_window] = self._sales[r[:, None, None], cols]
        return out


class GaussianSampler(_PerSeriesSampler):
    """Per-series Gaussian fit — the textbook inventory-theory assumption.

    The headline strawman, but a fair one: it is what safety-stock practice
    assumes. Weeks are drawn independently from N(mean, std^2) and clipped at
    zero (the clip is charitable to the Gaussian — unclipped it would order
    against negative-demand mass; the structural mis-shape remains).
    """

    def fit(self, panel: WeeklyPanel, train_span: tuple[int, int]) -> GaussianSampler:
        self.mean_, var = _train_moments(panel, train_span)
        self.std_ = np.sqrt(var)
        self.fitted_ = True
        return self

    def sample_batch(self, context_batch: dict[str, np.ndarray], n_scenarios: int) -> np.ndarray:
        self._require_fitted()
        rows = self._rows(context_batch)
        size = (rows.size, n_scenarios, self.horizon)
        draws = self._rng.normal(self.mean_[rows, None, None], self.std_[rows, None, None], size)
        return np.clip(draws, 0.0, None)


class PoissonSampler(_PerSeriesSampler):
    """Per-series Poisson fit (rate = train mean).

    The naive count model. Its defining constraint variance = mean is
    violated by ~99.9% of FOODS series (EDA: median var/mean ~ 4), so its
    predicted failure mode is too-narrow intervals and over-confident orders.
    """

    def fit(self, panel: WeeklyPanel, train_span: tuple[int, int]) -> PoissonSampler:
        self.rate_, _ = _train_moments(panel, train_span)
        self.fitted_ = True
        return self

    def sample_batch(self, context_batch: dict[str, np.ndarray], n_scenarios: int) -> np.ndarray:
        self._require_fitted()
        rows = self._rows(context_batch)
        size = (rows.size, n_scenarios, self.horizon)
        return self._rng.poisson(self.rate_[rows, None, None], size).astype(float)


class NegativeBinomialSampler(_PerSeriesSampler):
    """Per-series Negative Binomial via method of moments — the pivotal baseline.

    A *non-conditional* NB: same likelihood family the CVAE decoder will use,
    with zero deep learning. Any claim of deep-model value must survive this
    comparison (design doc §7). Parameterization: NB(r, p) with
    r = mean^2/(var - mean), p = r/(r + mean), matching the fitted moments
    exactly; non-over-dispersed series degrade gracefully to ~Poisson.
    """

    def fit(self, panel: WeeklyPanel, train_span: tuple[int, int]) -> NegativeBinomialSampler:
        self.mean_, self.var_ = _train_moments(panel, train_span)
        self.r_, self.p_ = nb_params_from_moments(self.mean_, self.var_)
        self.fitted_ = True
        return self

    def sample_batch(self, context_batch: dict[str, np.ndarray], n_scenarios: int) -> np.ndarray:
        self._require_fitted()
        rows = self._rows(context_batch)
        size = (rows.size, n_scenarios, self.horizon)
        # NB drawn via its Gamma-Poisson mixture: identical distribution to
        # rng.negative_binomial(r, p) but ~2x faster at Phase 8 scale (the
        # runtime fix pre-registered in the Phase 5 design log). mu = r(1-p)/p.
        r = np.broadcast_to(self.r_[rows, None, None], size)
        mu = np.broadcast_to((self.r_ * (1.0 - self.p_) / self.p_)[rows, None, None], size)
        rate = self._rng.gamma(shape=r, scale=np.divide(mu, r, where=r > 0, out=np.zeros(size)))
        return self._rng.poisson(rate).astype(float)

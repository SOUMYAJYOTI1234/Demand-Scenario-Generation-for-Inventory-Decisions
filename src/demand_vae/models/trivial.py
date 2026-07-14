"""Trivial samplers that make the pipeline runnable before any model exists.

These are scaffolding, not baselines: the real classical baselines (empirical
resampling, Gaussian, Poisson, fitted Negative Binomial — roadmap Phase 5)
live in ``demand_vae.baselines``. These two exist so the vertical slice
(sampler -> SAA newsvendor -> realized cost) can be exercised end-to-end with
zero data and zero fitting (scripts/smoke_test.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from demand_vae.models.base import BaseSampler


class ConstantSampler(BaseSampler):
    """Every scenario is the same constant demand in every week.

    Degenerate on purpose: the SAA order equals ``horizon * value`` for any
    cost ratio, which makes it useful for wiring checks.
    """

    def __init__(self, horizon: int, value: float):
        super().__init__(horizon)
        if value < 0:
            raise ValueError(f"value must be non-negative, got {value}")
        self.value = value

    def sample(self, context: Any, n_scenarios: int) -> np.ndarray:
        return np.full((n_scenarios, self.horizon), self.value, dtype=float)


class FixedRatePoissonSampler(BaseSampler):
    """I.i.d. Poisson demand at a fixed rate, ignoring the context.

    Not a fitted model — the rate is supplied, not estimated. The fitted
    Poisson baseline (rate from each series' history) is Phase 5.
    """

    def __init__(self, horizon: int, rate: float, seed: int | None = None):
        super().__init__(horizon)
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self._rng = np.random.default_rng(seed)

    def sample(self, context: Any, n_scenarios: int) -> np.ndarray:
        return self._rng.poisson(self.rate, size=(n_scenarios, self.horizon)).astype(float)

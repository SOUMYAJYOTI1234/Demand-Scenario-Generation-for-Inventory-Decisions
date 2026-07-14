"""The sampler interface every demand model implements.

The whole comparison rests on one contract (roadmap Phase 5): every model —
classical baseline or CVAE — exposes ``sample(context, n_scenarios)`` and
returns an ``(n_scenarios, H)`` array of demand windows. The decision and
evaluation layers are model-agnostic against this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseSampler(ABC):
    """A conditional demand scenario generator.

    Parameters of the generative process are learned/fitted elsewhere; a
    sampler only needs to draw scenario windows given a decision-time context.

    Context representation (fixed by the Phase 4 pipeline): a mapping with
    keys ``series_row`` (panel row index), ``context_cat`` (item/store
    indices), and ``context_cont`` (continuous features). Classical per-series
    baselines key on ``series_row``; the CVAE consumes cat/cont. Trivial
    samplers may ignore the context entirely.
    """

    def __init__(self, horizon: int):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon

    def sample_batch(self, context_batch: dict[str, np.ndarray], n_scenarios: int) -> np.ndarray:
        """Draw scenarios for a batch of contexts: shape (n_windows, n_scenarios, H).

        ``context_batch`` maps each context key to an array whose first axis
        indexes the windows. The default implementation loops over
        :meth:`sample`; subclasses override it with vectorized draws — at SAA
        scale (~10^5 windows x S=1000) the loop is the difference between
        seconds and hours.
        """
        keys = list(context_batch)
        n_windows = len(context_batch[keys[0]])
        return np.stack(
            [
                self.sample({k: context_batch[k][i] for k in keys}, n_scenarios)
                for i in range(n_windows)
            ]
        )

    @abstractmethod
    def sample(self, context: Any, n_scenarios: int) -> np.ndarray:
        """Draw demand scenario windows for one decision context.

        Parameters
        ----------
        context:
            Decision-time conditioning information ``y`` (design doc §1.3):
            item/store identifiers, week-of-year, event/SNAP flags, relative
            price, recent-demand lags. Trivial samplers may ignore it. The
            concrete context representation is fixed by the Phase 4 data
            pipeline.
        n_scenarios:
            Number of scenario draws S (S = 1000 default, config
            ``decision.n_scenarios``).

        Returns
        -------
        np.ndarray of shape ``(n_scenarios, horizon)`` — non-negative demand
        windows, one row per scenario.
        """

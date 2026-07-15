"""CVAESampler: the trained CVAE behind the common sampler interface (Phase 7).

This is the adapter that plugs the deep model into the Phase 5 evaluation
harness with zero harness changes: it implements the same
``sample(context, n_scenarios) -> (n_scenarios, H)`` contract as the
classical baselines (design doc §7).

Scenario generation follows the design doc's generative process: draw
z ~ N(0, I_K), decode with the context embedding y, then sample from the
decoder likelihood — for the NB decoder via its Gamma–Poisson mixture
(rate ~ Gamma(shape=r, scale=mu/r), x ~ Poisson(rate)), drawn with numpy for
speed and seedability (the design-log optimization over numpy's direct
negative_binomial). Gaussian decoder: unit-variance draw in log1p space,
mapped back with expm1. Outputs are rounded to the nearest integer and
clipped at 0 (a no-op for the NB path, which is integer-valued by
construction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from demand_vae.models.base import BaseSampler
from demand_vae.models.cvae import ConditionalVAE

MAX_DECODE_ROWS = 500_000  # decoder forward sub-chunk: caps peak memory


class CVAESampler(BaseSampler):
    """Wraps a trained :class:`ConditionalVAE` as a scenario sampler."""

    def __init__(self, model: ConditionalVAE, seed: int | None = None):
        super().__init__(model.horizon)
        self.model = model.eval()
        self._torch_gen = torch.Generator().manual_seed(0 if seed is None else seed)
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_checkpoint(cls, path: str | Path, seed: int | None = None) -> CVAESampler:
        """Rebuild the model from a Phase 7 checkpoint (uses stored model_kwargs)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ConditionalVAE(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, seed=seed)

    def sample(self, context, n_scenarios: int) -> np.ndarray:
        """Single context -> (n_scenarios, H) non-negative integer demand."""
        batch = {
            "context_cat": np.asarray(context["context_cat"])[None, :],
            "context_cont": np.asarray(context["context_cont"])[None, :],
        }
        return self.sample_batch(batch, n_scenarios)[0]

    @torch.no_grad()
    def sample_batch(self, context_batch, n_scenarios: int) -> np.ndarray:
        """(N contexts, S scenarios) -> (N, S, H); decoder run in memory-capped chunks."""
        cat = torch.as_tensor(np.asarray(context_batch["context_cat"]), dtype=torch.long)
        cont = torch.as_tensor(np.asarray(context_batch["context_cont"]), dtype=torch.float32)
        n_windows = cat.shape[0]
        ctx = self.model.embed_context(cat, cont)  # (N, C)
        ctx_rows = ctx.repeat_interleave(n_scenarios, dim=0)  # (N*S, C)
        total_rows = n_windows * n_scenarios

        out = np.empty((total_rows, self.horizon), dtype=np.float32)
        for start in range(0, total_rows, MAX_DECODE_ROWS):
            stop = min(start + MAX_DECODE_ROWS, total_rows)
            z = torch.randn(stop - start, self.model.latent_dim, generator=self._torch_gen)
            decoded = self.model.decode(z, ctx_rows[start:stop])
            if self.model.decoder_likelihood == "nb":
                mu = decoded[0].numpy().astype(np.float64)
                r = decoded[1].numpy().astype(np.float64)
                # NB as Gamma-Poisson: much faster than rng.negative_binomial
                # at this scale, identical distribution.
                rate = self._rng.gamma(shape=r, scale=mu / r)
                draws = self._rng.poisson(rate).astype(np.float32)
            else:  # gaussian decoder: unit-variance draw in log1p space
                recon = decoded.numpy().astype(np.float64)
                log1p_draw = recon + self._rng.standard_normal(recon.shape)
                draws = np.expm1(log1p_draw).astype(np.float32)
            out[start:stop] = draws

        counts = np.clip(np.rint(out), 0, None)  # integers >= 0 (no-op for NB)
        return counts.reshape(n_windows, n_scenarios, self.horizon)

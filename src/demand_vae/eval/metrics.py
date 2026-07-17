"""Two-level evaluation (roadmap Phase 8; harness built in Phase 5 so it is
validated on baselines before the CVAE exists — design doc §3.3).

Level 1 (distributional): CRPS, randomized PIT, central-interval coverage.
Level 2 (decision): realized newsvendor cost, service level, fill rate,
overstock — per pre-registered cost ratio.

Both levels are computed on the **aggregate-horizon demand total**
D = sum of the H target weeks: that is the quantity the primary decision
consumes (design doc §1.2), so distributional quality is measured exactly
where it matters for the decision. Held-out NLL (Phase 8) will complete the
Level-1 table for likelihood-based models.

The harness streams over the split in chunks and calls the sampler's
vectorized ``sample_batch`` — at 651K test windows x S=1000 scenarios the
full tensor would be ~10 GB and a per-window Python loop would take hours.
"""

from __future__ import annotations

import numpy as np

from demand_vae.config import Config
from demand_vae.data.pipeline import WindowedSplit
from demand_vae.models.base import BaseSampler
from demand_vae.opt.newsvendor import (
    aggregate_horizon_demand,
    realized_cost,
    solve_newsvendor_saa,
    solve_newsvendor_saa_batch,
)

# ---------------------------------------------------------------------------
# Level-1 primitives (sample-based, vectorized over windows)
# ---------------------------------------------------------------------------


def crps_from_samples(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """CRPS per window, estimated from samples (Gneiting & Raftery 2007).

    Uses the energy form CRPS = E|X - y| - 0.5 E|X - X'| with the classic
    S^2-pair estimator, computed in O(S log S) per row via the sorted-sample
    identity  E|X - X'| = (2/S^2) * sum_i (2i - S - 1) x_(i).

    Parameters: ``samples`` (n_windows, S), ``actual`` (n_windows,).
    """
    samples = np.asarray(samples, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n_s = samples.shape[1]
    sorted_s = np.sort(samples, axis=1)
    term_abs = np.abs(sorted_s - actual[:, None]).mean(axis=1)
    weights = 2.0 * np.arange(1, n_s + 1) - n_s - 1  # (2i - S - 1), i = 1..S
    spread = (sorted_s @ weights) / (n_s * n_s)  # = 0.5 * E|X - X'|
    return term_abs - spread


def pit_from_samples(
    samples: np.ndarray, actual: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomized PIT per window; uniform on [0, 1) iff the forecast is calibrated.

    For discrete demand the plain PIT F(y) is non-uniform even for a perfect
    model; the standard fix is the randomized PIT — the rank of the actual
    among the samples with uniform tie-breaking:
    pit = (#[x < y] + U * (#[x = y] + 1)) / (S + 1),  U ~ Uniform[0, 1).
    """
    samples = np.asarray(samples)
    actual = np.asarray(actual)
    n_less = (samples < actual[:, None]).sum(axis=1)
    n_equal = (samples == actual[:, None]).sum(axis=1)
    u = rng.random(actual.shape[0])
    return (n_less + u * (n_equal + 1)) / (samples.shape[1] + 1)


def interval_coverage(samples: np.ndarray, actual: np.ndarray, level: float) -> float:
    """Empirical coverage of the central ``level`` interval of the samples.

    A calibrated model covers ~``level`` of actuals; Poisson's predicted
    failure is visible under-coverage here (design doc §8: if it is not
    visible, the harness is broken).
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    alpha = (1.0 - level) / 2.0
    lo = np.quantile(samples, alpha, axis=1)
    hi = np.quantile(samples, 1.0 - alpha, axis=1)
    actual = np.asarray(actual, dtype=float)
    return float(((actual >= lo) & (actual <= hi)).mean())


def held_out_nll(model, windows: np.ndarray, contexts: list) -> float:
    """Held-out negative log-likelihood for likelihood-based models."""
    raise NotImplementedError(
        "TODO(Phase 8): NLL table (ELBO bound for the CVAE; exact for classical fits)."
    )


# ---------------------------------------------------------------------------
# Two-level harness
# ---------------------------------------------------------------------------

COVERAGE_LEVELS = (0.5, 0.9)


def make_context_batch(split: WindowedSplit, start: int, stop: int) -> dict[str, np.ndarray]:
    """Slice a WindowedSplit into the context-batch mapping samplers consume."""
    return {
        "series_row": split.series_row[start:stop],
        "context_cat": split.context_cat[start:stop],
        "context_cont": split.context_cont[start:stop],
    }


def evaluate_two_level(
    sampler: BaseSampler,
    split: WindowedSplit,
    config: Config,
    pit_seed: int = 0,
    chunk_size: int = 4096,
    protocols: tuple[str, ...] = ("aggregate",),
    collect_per_window: bool = False,
) -> dict:
    """Full two-level evaluation of one sampler on one split.

    For each window: draw S scenario windows, reduce to aggregate-horizon
    totals, score the predictive distribution (Level 1), then solve the SAA
    newsvendor at every pre-registered cost ratio and score the realized
    decisions against the actual totals (Level 2). Streaming: O(chunk) memory.

    ``protocols``: "aggregate" (primary, design doc §1.2 option 3) is always
    required; adding "marginal" also runs the RQ4 secondary protocol — H
    separate weekly newsvendor decisions from each week's marginal scenario
    distribution, using the *same* scenario draws (per-window cost = sum of
    the H weekly costs; service level counted per week-decision).

    ``collect_per_window=True`` additionally returns the aggregate-protocol
    per-window realized costs per ratio — the pairing unit for the Phase 8
    paired bootstrap.

    Returns a dict with ``n_windows``, ``level1`` (mean CRPS, coverage at 50%
    and 90%), ``pit`` (per-window randomized PIT values, for histograms),
    ``level2`` (aggregate protocol; one dict per cost ratio with
    mean_realized_cost, service_level, fill_rate, mean_overstock), plus
    optional ``level2_marginal`` and ``per_window_cost``.
    """
    unknown = set(protocols) - {"aggregate", "marginal"}
    if unknown or "aggregate" not in protocols:
        raise ValueError(f"protocols must include 'aggregate' (got {protocols})")
    run_marginal = "marginal" in protocols
    n_scenarios = config.decision.n_scenarios
    ratios = [tuple(r) for r in config.decision.cost_ratios]
    n_windows = len(split)
    horizon = split.x.shape[1]
    pit_rng = np.random.default_rng(pit_seed)

    crps_sum = 0.0
    covered = dict.fromkeys(COVERAGE_LEVELS, 0.0)
    pit_values = np.empty(n_windows, dtype=np.float32)
    acc = {
        ratio: {"cost": 0.0, "stockouts": 0.0, "overstock": 0.0, "served": 0.0} for ratio in ratios
    }
    acc_marginal = {
        ratio: {"cost": 0.0, "stockouts": 0.0, "overstock": 0.0, "served": 0.0} for ratio in ratios
    }
    per_window_cost = (
        {ratio: np.empty(n_windows, dtype=np.float32) for ratio in ratios}
        if collect_per_window
        else None
    )
    total_demand = 0.0

    for start in range(0, n_windows, chunk_size):
        stop = min(start + chunk_size, n_windows)
        batch = make_context_batch(split, start, stop)
        scenarios = sampler.sample_batch(batch, n_scenarios)  # (n, S, H)
        totals = scenarios.sum(axis=2)  # (n, S)
        actual = split.x[start:stop].sum(axis=1).astype(float)  # (n,)
        actual_weekly = split.x[start:stop].astype(float)  # (n, H)

        crps_sum += crps_from_samples(totals, actual).sum()
        pit_values[start:stop] = pit_from_samples(totals, actual, pit_rng)
        n_chunk = stop - start
        for level in COVERAGE_LEVELS:
            covered[level] += interval_coverage(totals, actual, level) * n_chunk

        total_demand += actual.sum()
        for c_u, c_o in ratios:
            order = solve_newsvendor_saa_batch(totals, c_u, c_o)
            under = np.maximum(actual - order, 0.0)
            over = np.maximum(order - actual, 0.0)
            cost = c_u * under + c_o * over
            a = acc[(c_u, c_o)]
            a["cost"] += cost.sum()
            a["stockouts"] += (actual > order).sum()
            a["overstock"] += over.sum()
            a["served"] += np.minimum(order, actual).sum()
            if per_window_cost is not None:
                per_window_cost[(c_u, c_o)][start:stop] = cost

            if run_marginal:
                tau = c_u / (c_u + c_o)
                weekly_orders = np.quantile(scenarios, tau, axis=1, method="inverted_cdf")  # (n, H)
                w_under = np.maximum(actual_weekly - weekly_orders, 0.0)
                w_over = np.maximum(weekly_orders - actual_weekly, 0.0)
                m = acc_marginal[(c_u, c_o)]
                m["cost"] += (c_u * w_under + c_o * w_over).sum()
                m["stockouts"] += (actual_weekly > weekly_orders).sum()
                m["overstock"] += w_over.sum()
                m["served"] += np.minimum(weekly_orders, actual_weekly).sum()

    def level2_rows(bucket: dict, decisions_per_window: int) -> list[dict]:
        return [
            {
                "c_u": c_u,
                "c_o": c_o,
                "mean_realized_cost": bucket[(c_u, c_o)]["cost"] / n_windows,
                "service_level": 1.0
                - bucket[(c_u, c_o)]["stockouts"] / (n_windows * decisions_per_window),
                "fill_rate": bucket[(c_u, c_o)]["served"] / max(total_demand, 1.0),
                "mean_overstock": bucket[(c_u, c_o)]["overstock"] / n_windows,
            }
            for c_u, c_o in ratios
        ]

    result = {
        "n_windows": n_windows,
        "n_scenarios": n_scenarios,
        "level1": {
            "crps": crps_sum / n_windows,
            **{
                f"coverage_{int(level * 100)}": covered[level] / n_windows
                for level in COVERAGE_LEVELS
            },
        },
        "pit": pit_values,
        "level2": level2_rows(acc, 1),
    }
    if run_marginal:
        result["level2_marginal"] = level2_rows(acc_marginal, horizon)
    if per_window_cost is not None:
        result["per_window_cost"] = {
            f"{c_u}:{c_o}": per_window_cost[(c_u, c_o)] for c_u, c_o in ratios
        }
    return result


# ---------------------------------------------------------------------------
# Phase 8 statistics: paired bootstrap and rank agreement
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    cost_a: np.ndarray,
    cost_b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    batch: int = 20,
) -> dict[str, float]:
    """Paired bootstrap on the mean cost difference (a - b), pairing on windows.

    Pairing removes the between-window demand variance that dwarfs
    between-method differences (design doc §7). Returns the observed mean
    difference, the 95% percentile CI over ``n_boot`` resamples, and
    ``p_gt_zero`` — the fraction of bootstrap means > 0 (the design's
    pre-registered one-sided summary; a negative mean difference with
    ``p_gt_zero`` near 0 means model a is cheaper).
    """
    diff = np.asarray(cost_a, dtype=np.float64) - np.asarray(cost_b, dtype=np.float64)
    if diff.shape != np.asarray(cost_b).shape or diff.ndim != 1:
        raise ValueError("cost_a and cost_b must be 1-D arrays of equal length (paired)")
    n = diff.size
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, batch):
        stop = min(start + batch, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean_diff": float(diff.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "p_gt_zero": float((means > 0).mean()),
    }


def spearman_rank_correlation(a, b) -> float:
    """Spearman rank correlation between two metric vectors (RQ3 helper)."""
    from scipy import stats

    rho = stats.spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float)).statistic
    return float(rho)


# ---------------------------------------------------------------------------
# Small per-context harness (kept for the smoke test and tiny examples)
# ---------------------------------------------------------------------------


def evaluate_decisions(
    sampler: BaseSampler,
    contexts: list,
    actual_windows: np.ndarray,
    c_u: float,
    c_o: float,
    n_scenarios: int,
) -> dict[str, float]:
    """Aggregate-horizon newsvendor over a list of contexts, one at a time.

    The readable reference implementation used by scripts/smoke_test.py;
    :func:`evaluate_two_level` is the streaming version for real splits.
    """
    actual_windows = np.asarray(actual_windows, dtype=float)
    if len(contexts) != actual_windows.shape[0]:
        raise ValueError(
            f"Got {len(contexts)} contexts but {actual_windows.shape[0]} actual windows"
        )

    costs, stockouts, overstocks = [], [], []
    for context, actual_window in zip(contexts, actual_windows, strict=True):
        scenarios = sampler.sample(context, n_scenarios)
        demand_totals = aggregate_horizon_demand(scenarios)
        order = solve_newsvendor_saa(demand_totals, c_u, c_o)
        actual_total = float(np.sum(actual_window))
        costs.append(realized_cost(order, actual_total, c_u, c_o))
        stockouts.append(actual_total > order)
        overstocks.append(max(order - actual_total, 0.0))

    return {
        "mean_realized_cost": float(np.mean(costs)),
        "service_level": float(1.0 - np.mean(stockouts)),
        "mean_overstock": float(np.mean(overstocks)),
        "n_decisions": len(costs),
    }

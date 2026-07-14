"""Newsvendor decision layer: SAA critical-fractile solver and realized cost.

This is the optimization problem (c) and decision-evaluation problem (d) of
the design document, Section 1.1. The single-period newsvendor with underage
cost ``c_u`` and overage cost ``c_o`` is solved by its critical fractile

    Q* = F^{-1}( c_u / (c_u + c_o) ),

and under sample average approximation (SAA) over S scenarios, Q* is the
corresponding empirical quantile of the scenario set (design doc §1.1c).

The primary decision task is the *aggregate-horizon* newsvendor (design doc
§1.2, option 3): the decision quantity is D = sum of the H-week demand window,
so the scenario windows are reduced to per-scenario sums before solving.
"""

from __future__ import annotations

import numpy as np


def critical_fractile(c_u: float, c_o: float) -> float:
    """Return the newsvendor critical fractile tau = c_u / (c_u + c_o)."""
    if c_u <= 0 or c_o <= 0:
        raise ValueError(f"Costs must be positive, got c_u={c_u}, c_o={c_o}")
    return c_u / (c_u + c_o)


def solve_newsvendor_saa(scenario_demands: np.ndarray, c_u: float, c_o: float) -> float:
    """Solve the newsvendor by SAA: the empirical critical-fractile quantile.

    Parameters
    ----------
    scenario_demands:
        1-D array of S scalar demand scenarios (for the aggregate-horizon
        task, per-scenario window sums — see :func:`aggregate_horizon_demand`).
    c_u, c_o:
        Underage and overage cost per unit.

    Returns
    -------
    The SAA-optimal order quantity Q: the ceil(S * tau)-th order statistic of
    the scenarios, i.e. the smallest scenario value whose empirical CDF
    reaches tau. Computed with ``np.quantile(..., method="inverted_cdf")``,
    which is exactly that order statistic and matches the discrete SAA
    optimum (tested against the closed form in tests/test_newsvendor.py).
    """
    scenario_demands = np.asarray(scenario_demands, dtype=float)
    if scenario_demands.ndim != 1 or scenario_demands.size == 0:
        raise ValueError(
            f"scenario_demands must be a non-empty 1-D array, got shape {scenario_demands.shape}"
        )
    tau = critical_fractile(c_u, c_o)
    return float(np.quantile(scenario_demands, tau, method="inverted_cdf"))


def solve_newsvendor_saa_batch(scenario_demands: np.ndarray, c_u: float, c_o: float) -> np.ndarray:
    """Vectorized :func:`solve_newsvendor_saa` over many decisions at once.

    Parameters
    ----------
    scenario_demands:
        2-D array (n_decisions, S) of scalar demand scenarios per decision.

    Returns the (n_decisions,) SAA-optimal orders — the same ceil(S*tau)-th
    order statistic per row (tested to match the scalar solver exactly).
    """
    scenario_demands = np.asarray(scenario_demands, dtype=float)
    if scenario_demands.ndim != 2 or scenario_demands.shape[1] == 0:
        raise ValueError(
            f"scenario_demands must have shape (n_decisions, S), got {scenario_demands.shape}"
        )
    tau = critical_fractile(c_u, c_o)
    return np.quantile(scenario_demands, tau, axis=1, method="inverted_cdf")


def aggregate_horizon_demand(scenario_windows: np.ndarray) -> np.ndarray:
    """Reduce (n_scenarios, H) demand windows to per-scenario horizon totals.

    The distribution of this sum depends on within-window correlations, which
    is what makes the aggregate-horizon decision consume the *joint*
    distribution the CVAE is built to capture (design doc §1.2).
    """
    scenario_windows = np.asarray(scenario_windows, dtype=float)
    if scenario_windows.ndim != 2:
        raise ValueError(
            f"scenario_windows must have shape (n_scenarios, H), got {scenario_windows.shape}"
        )
    return scenario_windows.sum(axis=1)


def realized_cost(order: float, actual_demand: float, c_u: float, c_o: float) -> float:
    """Realized newsvendor cost of an order against the actual demand.

    RealizedCost(Q; D) = c_u * (D - Q)^+ + c_o * (Q - D)^+   (design doc §1.1d)
    """
    if c_u <= 0 or c_o <= 0:
        raise ValueError(f"Costs must be positive, got c_u={c_u}, c_o={c_o}")
    underage = max(actual_demand - order, 0.0)
    overage = max(order - actual_demand, 0.0)
    return c_u * underage + c_o * overage

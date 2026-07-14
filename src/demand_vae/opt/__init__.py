"""Optimization layer: SAA newsvendor solver (roadmap Phase 8)."""

from demand_vae.opt.newsvendor import (
    aggregate_horizon_demand,
    critical_fractile,
    realized_cost,
    solve_newsvendor_saa,
    solve_newsvendor_saa_batch,
)

__all__ = [
    "aggregate_horizon_demand",
    "critical_fractile",
    "realized_cost",
    "solve_newsvendor_saa",
    "solve_newsvendor_saa_batch",
]

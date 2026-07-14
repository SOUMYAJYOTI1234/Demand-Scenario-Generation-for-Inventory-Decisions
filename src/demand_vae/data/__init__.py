"""M5 data pipeline: weekly aggregation, features, windowing, temporal splits (Phase 4)."""

from demand_vae.data.pipeline import (
    ContextScaler,
    WeeklyPanel,
    WindowedSplit,
    aggregate_weekly,
    build_dataset,
    build_windows,
    continuous_feature_names,
    load_raw_m5,
    temporal_split,
)
from demand_vae.data.validation import assert_no_leakage, check_schema, zero_fraction_report

__all__ = [
    "ContextScaler",
    "WeeklyPanel",
    "WindowedSplit",
    "aggregate_weekly",
    "assert_no_leakage",
    "build_dataset",
    "build_windows",
    "check_schema",
    "continuous_feature_names",
    "load_raw_m5",
    "temporal_split",
    "zero_fraction_report",
]

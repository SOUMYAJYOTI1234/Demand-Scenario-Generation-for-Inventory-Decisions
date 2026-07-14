"""Data-quality and leakage checks for the Phase 4 pipeline.

Silent evaluation bugs are the most dangerous failure class in this project
(design doc §8), and the data pipeline is where they enter: a random split, a
scaler fit on the full dataset, or a malformed raw file all produce plausible
wrong numbers. These checks raise loudly instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand_vae.config import Config
from demand_vae.data.pipeline import (
    DAYS_PER_WEEK,
    N_UNSCALED,
    ContextScaler,
    WeeklyPanel,
    WindowedSplit,
)

REQUIRED_COLUMNS = {
    "sales": ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d_1"],
    "calendar": [
        "date",
        "wm_yr_wk",
        "d",
        "event_name_1",
        "event_name_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ],
    "prices": ["store_id", "item_id", "wm_yr_wk", "sell_price"],
}


def check_schema(raw: dict[str, pd.DataFrame]) -> None:
    """Assert the raw M5 frames have the columns the pipeline relies on."""
    for name, required in REQUIRED_COLUMNS.items():
        if name not in raw:
            raise ValueError(f"Raw data is missing the {name!r} frame")
        frame = raw[name]
        if frame.empty:
            raise ValueError(f"Raw {name!r} frame is empty")
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"Raw {name!r} frame is missing columns: {missing}")
    if raw["prices"]["sell_price"].le(0).any():
        raise ValueError("Raw prices contain non-positive sell_price values")


def zero_fraction_report(panel: WeeklyPanel) -> pd.DataFrame:
    """Per-series intermittency report over each series' available span.

    Columns: zero_frac (share of available weeks with zero sales), mean, var,
    and dispersion ratio var/mean (> 1 = over-dispersed vs Poisson — the
    statistic the NB-decoder argument rests on). Sorted by zero_frac.
    """
    import warnings

    sales = np.where(panel.available, panel.sales, np.nan).astype(float)
    n_avail = panel.available.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        mean = np.nanmean(sales, axis=1)
        var = np.nanvar(sales, axis=1)
        zero_frac = np.nansum(sales == 0, axis=1) / np.maximum(n_avail, 1)
        dispersion = np.divide(var, mean, out=np.full_like(var, np.nan), where=mean > 0)
    return pd.DataFrame(
        {
            "series_id": panel.series_ids,
            "n_available_weeks": n_avail,
            "zero_frac": zero_frac,
            "mean": mean,
            "var": var,
            "dispersion": dispersion,
        }
    ).sort_values("zero_frac", ascending=False, ignore_index=True)


def assert_no_leakage(
    splits: dict[str, WindowedSplit],
    scaler: ContextScaler,
    panel: WeeklyPanel,
    config: Config,
    unscaled_train_cont: np.ndarray | None = None,
) -> None:
    """Assert the two leakage guarantees the evaluation depends on.

    1. **Temporal:** every window's full span (first lag week through the last
       day of the last target week) lies inside its split's date range —
       train windows never touch data after ``train_end``, val windows stay in
       (``train_end``, ``val_end``], test windows after ``val_end``.
    2. **Scaler:** the scaler's statistics were fit on the train split only —
       train's scaled columns have mean ~0 / std ~1, and the recorded fit-set
       size equals the train split size. If the *unscaled* train features are
       provided, the scaler's stored mean/std must match them exactly.

    Raises AssertionError with a specific message on the first violation.
    """
    lags, horizon = config.data.n_lags, config.data.horizon_weeks
    train_end = pd.Timestamp(config.data.split.train_end)
    val_end = pd.Timestamp(config.data.split.val_end)
    panel_start = panel.week_start[0]
    panel_end = panel.week_end()[-1]
    bounds = {
        "train": (panel_start, train_end),
        "val": (train_end, val_end),  # exclusive lower bound, handled below
        "test": (val_end, panel_end),
    }

    for name, split in splits.items():
        first_lag_start = panel.week_start[split.first_target_week - lags]
        last_target_end = panel.week_start[split.first_target_week + horizon - 1] + pd.Timedelta(
            days=DAYS_PER_WEEK - 1
        )
        lower, upper = bounds[name]
        if name == "train":
            if not (first_lag_start >= lower).all():
                raise AssertionError("train window begins before the panel start")
        else:
            if not (first_lag_start > lower).all():
                raise AssertionError(f"{name} window reaches back across its split boundary")
        if not (last_target_end <= upper).all():
            raise AssertionError(f"{name} window extends past its split boundary ({upper.date()})")

    train = splits["train"]
    if scaler.n_fit_samples_ != len(train):
        raise AssertionError(
            f"Scaler was fit on {scaler.n_fit_samples_} samples but the train split has "
            f"{len(train)} — it was not fit on (exactly) the train split"
        )
    # float64 accumulation: float32 axis-0 reductions drift at millions of rows.
    scaled = train.context_cont[:, N_UNSCALED:]
    if not (np.abs(scaled.mean(axis=0, dtype=np.float64)) < 1e-3).all():
        raise AssertionError("Scaled train features do not have ~zero mean — scaler not train-fit")
    stds = scaled.std(axis=0, dtype=np.float64)
    if not ((np.abs(stds - 1.0) < 1e-3) | (stds < 1e-6)).all():  # constant cols scale to ~0 std
        raise AssertionError("Scaled train features do not have ~unit std — scaler not train-fit")
    if unscaled_train_cont is not None:
        expect_mean = unscaled_train_cont[:, N_UNSCALED:].mean(axis=0, dtype=np.float64)
        expect_std = unscaled_train_cont[:, N_UNSCALED:].std(axis=0, dtype=np.float64)
        expect_std = np.where(expect_std < 1e-8, 1.0, expect_std)  # ContextScaler convention
        if not np.allclose(scaler.mean_, expect_mean, atol=1e-6) or not np.allclose(
            scaler.std_, expect_std, atol=1e-6
        ):
            raise AssertionError("Scaler statistics do not match the train split's features")

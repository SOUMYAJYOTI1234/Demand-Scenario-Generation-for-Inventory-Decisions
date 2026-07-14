"""M5 data pipeline (roadmap Phase 4).

Flow: raw M5 csvs -> FOODS filter -> weekly aggregation per item-store series
-> temporal split -> windowing into (context y, demand window x) samples with
stride 1 -> train-only feature scaling.

Binding choices implemented here (design doc §10; see docs/design-decisions.md):

- **Weeks** are Walmart calendar weeks (Saturday-Friday, the `wm_yr_wk`
  column); only complete 7-day weeks are kept, so the partial final week of
  the evaluation file is dropped.
- **Pre-launch handling:** a series is considered available from its first
  week with a recorded sell price (M5 records a price exactly when the item
  is offered). Windows touching unavailable weeks are dropped. Series with no
  price at all during the train span (launched later) are dropped entirely,
  because their relative-price feature would be undefined without leaking
  val/test statistics.
- **"Season"** is operationalized as sin/cos of the ISO week-of-year of the
  first target week, plus explicit event/SNAP fractions over the target
  window (assumption A6: calendar, events, and prices are known at decision
  time).
- **"Price"** is the *relative* price: mean sell price over the target window
  divided by the series' train-period mean price (captures promotional
  deviation, not absolute level).
- **Splits are temporal, never random.** A week belongs to a split iff its
  full 7 days end on or before the boundary date; a window belongs to a split
  iff all its L lag weeks and H target weeks lie inside that split
  (boundary-straddling windows are dropped).
- **Scaling:** the context scaler is fit on train windows only and applied
  unchanged to val/test (leakage guard, checked by
  `demand_vae.data.validation.assert_no_leakage`).

Window layout: a window starting at week index t uses weeks
[t, t+L) as lags (log1p-scaled, part of context y) and [t+L, t+L+H) as the
demand target x. Context y = categorical (item_idx, store_idx) + continuous
(sin/cos week-of-year, event fraction, SNAP fraction, relative price, L lags).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from demand_vae.config import Config

# Continuous context feature layout (order matters; tests pin it).
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "woy_sin",
    "woy_cos",
    "event_frac",
    "snap_frac",
    "rel_price",
)  # followed by n_lags log1p lag features named lag_1 (oldest) .. lag_L (most recent)
CATEGORICAL_FEATURES: tuple[str, ...] = ("item_idx", "store_idx")
# sin/cos are already in [-1, 1]; everything from event_frac onward is standardized.
N_UNSCALED = 2

DAYS_PER_WEEK = 7


def continuous_feature_names(n_lags: int) -> list[str]:
    return list(CONTINUOUS_FEATURES) + [f"lag_{i + 1}" for i in range(n_lags)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_m5(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the raw M5 csvs downloaded by scripts/download_data.py.

    Returns a dict with keys ``sales`` (wide daily unit sales), ``calendar``,
    and ``prices``. Schema-checked by :func:`demand_vae.data.validation.check_schema`.
    """
    from demand_vae.data.validation import check_schema

    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw" if (data_dir / "raw").is_dir() else data_dir
    paths = {
        "sales": raw_dir / "sales_train_evaluation.csv",
        "calendar": raw_dir / "calendar.csv",
        "prices": raw_dir / "sell_prices.csv",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing M5 files: {missing}. Run `python scripts/download_data.py` first "
            "(see data/README.md for Kaggle setup)."
        )
    raw = {name: pd.read_csv(path) for name, path in paths.items()}
    check_schema(raw)
    return raw


# ---------------------------------------------------------------------------
# Weekly aggregation
# ---------------------------------------------------------------------------


@dataclass
class WeeklyPanel:
    """The weekly-aggregated FOODS panel: everything downstream stages need.

    All per-week arrays are aligned to ``week_start`` (index 0 = the first
    complete Walmart week in the data).
    """

    sales: np.ndarray  # (n_series, n_weeks) int32 weekly unit sales
    rel_price: np.ndarray  # (n_series, n_weeks) float32; NaN where unavailable
    available: np.ndarray  # (n_series, n_weeks) bool: price recorded that week
    series_ids: np.ndarray  # (n_series,) e.g. "FOODS_3_090_CA_1"
    item_idx: np.ndarray  # (n_series,) int contiguous item codes
    store_idx: np.ndarray  # (n_series,) int contiguous store codes
    item_categories: list[str] = field(repr=False)  # code -> item_id
    store_categories: list[str] = field(repr=False)  # code -> store_id
    week_start: pd.DatetimeIndex = field(repr=False)  # (n_weeks,) Saturdays
    woy_sin: np.ndarray = field(repr=False)  # (n_weeks,)
    woy_cos: np.ndarray = field(repr=False)  # (n_weeks,)
    event_frac: np.ndarray = field(repr=False)  # (n_weeks,) fraction of event days
    snap_frac: np.ndarray = field(repr=False)  # (n_series, n_weeks) via store's state
    n_dropped_series: int = 0  # series without any train-period price

    @property
    def n_series(self) -> int:
        return self.sales.shape[0]

    @property
    def n_weeks(self) -> int:
        return self.sales.shape[1]

    def week_end(self) -> pd.DatetimeIndex:
        return self.week_start + pd.Timedelta(days=DAYS_PER_WEEK - 1)


def aggregate_weekly(raw: dict[str, pd.DataFrame], config: Config) -> WeeklyPanel:
    """Filter to the scoped category and aggregate daily sales to complete weeks.

    Weekly buckets reduce intermittency and match ordering cadence
    (assumption A2). Relative price is computed against the series'
    *train-period* mean price (train-only statistic; leakage guard).
    """
    sales, calendar, prices = raw["sales"], raw["calendar"], raw["prices"]

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    calendar = calendar.iloc[: len(day_cols)].copy()  # sales file may end mid-calendar
    calendar["date"] = pd.to_datetime(calendar["date"])

    # Keep only complete 7-day Walmart weeks.
    week_sizes = calendar.groupby("wm_yr_wk")["d"].size()
    complete_weeks = week_sizes.index[week_sizes == DAYS_PER_WEEK].to_numpy()
    n_weeks = len(complete_weeks)
    n_days = n_weeks * DAYS_PER_WEEK
    calendar = calendar.iloc[:n_days]
    if not (calendar["wm_yr_wk"].to_numpy() == np.repeat(complete_weeks, DAYS_PER_WEEK)).all():
        raise ValueError("Calendar days are not contiguous complete weeks from the start")

    cat = sales[sales["cat_id"] == config.data.category].reset_index(drop=True)
    if cat.empty:
        raise ValueError(f"No series found for category {config.data.category!r}")

    daily = cat[day_cols[:n_days]].to_numpy(dtype=np.int32)
    weekly_sales = daily.reshape(len(cat), n_weeks, DAYS_PER_WEEK).sum(axis=2, dtype=np.int32)

    # Per-week calendar features.
    week_start = pd.DatetimeIndex(calendar["date"].iloc[::DAYS_PER_WEEK])
    woy = week_start.isocalendar().week.to_numpy(dtype=float)
    woy_sin = np.sin(2 * np.pi * woy / 52.0).astype(np.float32)
    woy_cos = np.cos(2 * np.pi * woy / 52.0).astype(np.float32)
    has_event = (calendar["event_name_1"].notna() | calendar["event_name_2"].notna()).to_numpy()
    event_frac = has_event.reshape(n_weeks, DAYS_PER_WEEK).mean(axis=1).astype(np.float32)

    # SNAP fraction per state per week, mapped to each series via its store's state.
    states = sorted({s.split("_")[0] for s in cat["store_id"].unique()})
    snap_by_state = np.stack(
        [
            calendar[f"snap_{st}"].to_numpy(dtype=float).reshape(n_weeks, DAYS_PER_WEEK).mean(1)
            for st in states
        ]
    ).astype(np.float32)  # (n_states, n_weeks)
    state_idx = cat["store_id"].str.split("_").str[0].map({s: i for i, s in enumerate(states)})
    snap_frac = snap_by_state[state_idx.to_numpy()]  # (n_series, n_weeks)

    # Weekly prices: pivot to (n_series, n_weeks); NaN = not offered that week.
    prices = prices[prices["item_id"].isin(cat["item_id"].unique())]
    prices = prices[prices["wm_yr_wk"].isin(complete_weeks)]
    price_wide = prices.pivot_table(
        index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price"
    ).reindex(
        pd.MultiIndex.from_frame(cat[["store_id", "item_id"]]),
        columns=complete_weeks,
    )
    price = price_wide.to_numpy(dtype=np.float32)
    available = ~np.isnan(price)

    # Relative price against the train-period mean (train-only statistic).
    train_end = pd.Timestamp(config.data.split.train_end)
    week_end = week_start + pd.Timedelta(days=DAYS_PER_WEEK - 1)
    train_weeks = np.asarray(week_end <= train_end)
    if not train_weeks.any():
        raise ValueError(f"No complete weeks end on or before train_end={train_end.date()}")
    train_price = np.where(train_weeks[None, :], price, np.nan)
    n_train_priced = (~np.isnan(train_price)).sum(axis=1)
    train_mean_price = np.where(
        n_train_priced > 0,
        np.nansum(np.nan_to_num(train_price), axis=1) / np.maximum(n_train_priced, 1),
        np.nan,
    )
    has_train_price = ~np.isnan(train_mean_price)
    n_dropped = int((~has_train_price).sum())

    keep = has_train_price
    rel_price = price[keep] / train_mean_price[keep, None]

    kept = cat.loc[keep].reset_index(drop=True)
    item_codes, item_categories = pd.factorize(kept["item_id"])
    store_codes, store_categories = pd.factorize(kept["store_id"])

    return WeeklyPanel(
        sales=weekly_sales[keep],
        rel_price=rel_price.astype(np.float32),
        available=available[keep],
        series_ids=kept["id"].to_numpy(),
        item_idx=item_codes.astype(np.int64),
        store_idx=store_codes.astype(np.int64),
        item_categories=list(item_categories),
        store_categories=list(store_categories),
        week_start=week_start,
        woy_sin=woy_sin,
        woy_cos=woy_cos,
        event_frac=event_frac,
        snap_frac=snap_frac,
        n_dropped_series=n_dropped,
    )


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def temporal_split(panel: WeeklyPanel, config: Config) -> dict[str, tuple[int, int]]:
    """Assign week indices to contiguous train/val/test spans. Never random.

    Returns ``{split: (first_week_idx, last_week_idx)}`` (inclusive). A week
    belongs to a split iff its full 7 days end on or before the boundary date.
    """
    week_end = panel.week_end()
    train_end = pd.Timestamp(config.data.split.train_end)
    val_end = pd.Timestamp(config.data.split.val_end)
    if not (train_end < val_end):
        raise ValueError(f"train_end ({train_end.date()}) must precede val_end ({val_end.date()})")

    in_train = week_end <= train_end
    in_val = (week_end > train_end) & (week_end <= val_end)
    in_test = week_end > val_end
    spans: dict[str, tuple[int, int]] = {}
    for name, mask in (("train", in_train), ("val", in_val), ("test", in_test)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError(f"Temporal split produced an empty {name!r} span")
        spans[name] = (int(idx[0]), int(idx[-1]))
    return spans


# ---------------------------------------------------------------------------
# Windowing + scaling
# ---------------------------------------------------------------------------


@dataclass
class WindowedSplit:
    """Windowed samples for one split, ready for models and the decision layer."""

    x: np.ndarray  # (N, H) int32 target demand windows
    context_cat: np.ndarray  # (N, 2) int64: item_idx, store_idx
    context_cont: np.ndarray  # (N, 5 + L) float32, scaled (train-fit scaler)
    series_row: np.ndarray  # (N,) row into the panel arrays
    first_target_week: np.ndarray  # (N,) week index of the first target week

    def __len__(self) -> int:
        return self.x.shape[0]


class ContextScaler:
    """Standardizes the continuous context features, fit on train only.

    sin/cos week-of-year (the first ``N_UNSCALED`` columns) are already in
    [-1, 1] and pass through unchanged; the remaining columns are
    standardized to train mean 0 / std 1.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_fit_samples_: int = 0

    def fit(self, context_cont: np.ndarray) -> ContextScaler:
        # Accumulate in float64: numpy's axis-0 reduction over a C-ordered
        # float32 array sums sequentially in float32, which loses ~1% accuracy
        # at millions of rows — enough to silently break standardization.
        scaled_part = context_cont[:, N_UNSCALED:]
        self.mean_ = scaled_part.mean(axis=0, dtype=np.float64)
        std = scaled_part.std(axis=0, dtype=np.float64)
        # Constant columns get scale 1 (sklearn convention): they transform to
        # ~0 instead of amplifying float rounding by a tiny divisor.
        self.std_ = np.where(std < 1e-8, 1.0, std)
        self.n_fit_samples_ = context_cont.shape[0]
        return self

    def transform(self, context_cont: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ContextScaler.transform called before fit")
        out = context_cont.astype(np.float32, copy=True)
        out[:, N_UNSCALED:] = (out[:, N_UNSCALED:] - self.mean_) / self.std_
        return out


def build_windows(
    panel: WeeklyPanel, config: Config, spans: dict[str, tuple[int, int]]
) -> dict[str, WindowedSplit]:
    """Window every series with stride 1 and assign windows to splits.

    A window starting at week t uses weeks [t, t+L) as log1p lags and
    [t+L, t+L+H) as the target x. It enters a split only if all L+H weeks lie
    inside that split's span AND the series is available (priced) in all of
    them — this drops boundary-straddling and pre-launch windows.
    """
    lags, horizon = config.data.n_lags, config.data.horizon_weeks
    total = lags + horizon
    if panel.n_weeks < total:
        raise ValueError(f"Panel has {panel.n_weeks} weeks; windows need at least {total}")

    swv = np.lib.stride_tricks.sliding_window_view
    sales_win = swv(panel.sales, total, axis=1)  # (n_series, n_starts, L+H)
    avail_win = swv(panel.available, total, axis=1).all(axis=2)  # (n_series, n_starts)
    n_starts = sales_win.shape[1]
    starts = np.arange(n_starts)

    splits: dict[str, WindowedSplit] = {}
    for name, (span_start, span_end) in spans.items():
        in_span = (starts >= span_start) & (starts + total - 1 <= span_end)
        series_row, t = np.nonzero(avail_win[:, in_span])
        t = starts[in_span][t]
        if series_row.size == 0:
            raise ValueError(f"Split {name!r} produced zero windows")

        window = sales_win[series_row, t]  # (N, L+H)
        x = np.ascontiguousarray(window[:, lags:], dtype=np.int32)
        lag_feats = np.log1p(window[:, :lags].astype(np.float32))

        first_target = t + lags
        # Target-window aggregates (all known at decision time, assumption A6).
        target_cols = first_target[:, None] + np.arange(horizon)[None, :]
        event = panel.event_frac[target_cols].mean(axis=1)
        snap = panel.snap_frac[series_row[:, None], target_cols].mean(axis=1)
        rel_price = panel.rel_price[series_row[:, None], target_cols].mean(axis=1)

        context_cont = np.column_stack(
            [
                panel.woy_sin[first_target],
                panel.woy_cos[first_target],
                event,
                snap,
                rel_price,
                lag_feats,
            ]
        ).astype(np.float32)
        context_cat = np.column_stack(
            [panel.item_idx[series_row], panel.store_idx[series_row]]
        ).astype(np.int64)

        splits[name] = WindowedSplit(
            x=x,
            context_cat=context_cat,
            context_cont=context_cont,
            series_row=series_row.astype(np.int64),
            first_target_week=first_target.astype(np.int64),
        )
    return splits


def build_dataset(
    config: Config, data_dir: str | Path = "data"
) -> tuple[dict[str, WindowedSplit], ContextScaler, WeeklyPanel]:
    """End-to-end Phase 4 pipeline: raw csvs -> scaled windowed splits.

    Returns (splits, scaler, panel). The scaler is fit on the train split
    only and already applied to every split's ``context_cont``.
    """
    raw = load_raw_m5(data_dir)
    panel = aggregate_weekly(raw, config)
    spans = temporal_split(panel, config)
    splits = build_windows(panel, config, spans)

    scaler = ContextScaler().fit(splits["train"].context_cont)
    for split in splits.values():
        split.context_cont = scaler.transform(split.context_cont)
    return splits, scaler, panel

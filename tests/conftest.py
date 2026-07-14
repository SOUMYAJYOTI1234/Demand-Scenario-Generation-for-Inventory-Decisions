"""Shared fixtures: a small synthetic M5-shaped dataset for pipeline tests.

28 complete Walmart weeks (Sat-Fri) from 2011-01-29. Split: train = weeks
1-12 (ends 2011-04-22), val = weeks 13-20 (ends 2011-06-17), test = 21-28.
With L=4 lags + H=4 targets (8 consecutive weeks per window) each fully
available series yields 5 train windows, 1 val window, 1 test window.

Series (all in dept FOODS_1 except the HOBBIES decoy):
- FOODS_1_001 @ CA_1 - daily sales constantly 1 (weekly sum = 7), priced all
  weeks at 2.5. The aggregation oracle.
- FOODS_1_001 @ TX_1 - random counts, priced all weeks; exercises the TX SNAP
  mapping.
- FOODS_1_002 @ CA_1 - priced only from week 5 onward (pre-launch weeks 1-4
  unavailable -> exactly 1 train window survives), price drops from 4.0 to
  3.0 at week 11 (relative-price variation).
- FOODS_1_003 @ CA_1 - priced only from week 21 (no train-period price ->
  the whole series must be dropped).
- HOBBIES_1_001 @ CA_1 - wrong category, must be filtered out.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand_vae.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

N_WEEKS = 28
N_DAYS = N_WEEKS * 7
START = pd.Timestamp("2011-01-29")  # a Saturday, matching real M5 week alignment
TRAIN_END = "2011-04-22"  # end of week 12
VAL_END = "2011-06-17"  # end of week 20


def _make_calendar() -> pd.DataFrame:
    dates = pd.date_range(START, periods=N_DAYS)
    day_of_series = np.arange(N_DAYS)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + day_of_series // 7,
            "weekday": dates.day_name(),
            "wday": day_of_series % 7 + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in day_of_series],
            # one event day every 4 weeks (on the last day of the week)
            "event_name_1": [("FakeFest" if i % 28 == 6 else None) for i in day_of_series],
            "event_type_1": [("Cultural" if i % 28 == 6 else None) for i in day_of_series],
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": (dates.day <= 10).astype(int),
            "snap_TX": ((dates.day >= 5) & (dates.day <= 15)).astype(int),
            "snap_WI": 0,
        }
    )


def _make_sales(rng: np.random.Generator) -> pd.DataFrame:
    def row(item: str, dept: str, cat: str, store: str, daily: np.ndarray) -> dict:
        state = store.split("_")[0]
        record = {
            "id": f"{item}_{store}_evaluation",
            "item_id": item,
            "dept_id": dept,
            "cat_id": cat,
            "store_id": store,
            "state_id": state,
        }
        record.update({f"d_{i + 1}": int(v) for i, v in enumerate(daily)})
        return record

    ones = np.ones(N_DAYS, dtype=int)
    return pd.DataFrame(
        [
            row("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", ones),
            row("FOODS_1_001", "FOODS_1", "FOODS", "TX_1", rng.poisson(3.0, N_DAYS)),
            row("FOODS_1_002", "FOODS_1", "FOODS", "CA_1", rng.poisson(2.0, N_DAYS)),
            row("FOODS_1_003", "FOODS_1", "FOODS", "CA_1", rng.poisson(1.0, N_DAYS)),
            row("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", rng.poisson(5.0, N_DAYS)),
        ]
    )


def _make_prices() -> pd.DataFrame:
    weeks = 11101 + np.arange(N_WEEKS)
    rows = []
    for store in ("CA_1", "TX_1"):
        rows += [(store, "FOODS_1_001", int(w), 2.5) for w in weeks]
    # FOODS_1_002: launched week 5 (index 4); price cut at week 13 (index 12,
    # the first val week) so the whole train period sits at 4.0
    rows += [("CA_1", "FOODS_1_002", int(w), 4.0) for w in weeks[4:12]]
    rows += [("CA_1", "FOODS_1_002", int(w), 3.0) for w in weeks[12:]]
    # FOODS_1_003: launched week 21 (index 20) - after train_end
    rows += [("CA_1", "FOODS_1_003", int(w), 1.5) for w in weeks[20:]]
    return pd.DataFrame(rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])


@pytest.fixture(scope="session")
def synthetic_raw() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    return {"sales": _make_sales(rng), "calendar": _make_calendar(), "prices": _make_prices()}


@pytest.fixture()
def synthetic_config() -> Config:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config.data.split.train_end = TRAIN_END
    config.data.split.val_end = VAL_END
    return config

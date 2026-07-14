# Data

This directory is **git-ignored** (except this README). The M5 dataset is large
and its license does not permit redistribution — it is downloaded, never
committed.

## Getting the M5 dataset

1. Install the Kaggle CLI: `pip install kaggle`
2. Create an API token: kaggle.com → Settings → API → **Create New Token**,
   and place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
   (Windows: `C:/Users/<you>/.kaggle/kaggle.json`).
3. Accept the competition rules once in the browser:
   https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
4. From the repo root:

   ```
   python scripts/download_data.py
   ```

This places the raw csvs in `data/raw/`:

| File | Contents |
|---|---|
| `sales_train_evaluation.csv` | Daily unit sales, 3,049 items × 10 stores, 2011–2016 |
| `calendar.csv` | Dates, events, SNAP indicators |
| `sell_prices.csv` | Weekly sell prices per item-store |

Processed artifacts (weekly FOODS aggregation, windows, splits — roadmap
Phase 4) will be written to `data/processed/` by the pipeline and are likewise
git-ignored.

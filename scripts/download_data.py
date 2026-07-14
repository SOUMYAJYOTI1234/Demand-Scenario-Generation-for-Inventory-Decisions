"""Download the M5 (m5-forecasting-accuracy) dataset via the Kaggle API.

Reproducibility starts here (roadmap Phase 4, step 1): the data is never
committed and never downloaded manually. Files land in data/raw/.

Prerequisites:
  1. pip install kaggle
  2. A Kaggle account + API token: kaggle.com -> Settings -> API ->
     "Create New Token", then place the downloaded kaggle.json at
     ~/.kaggle/kaggle.json (Windows: C:/Users/<you>/.kaggle/kaggle.json).
  3. Accept the competition rules once in the browser:
     https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules

Usage: python scripts/download_data.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

COMPETITION = "m5-forecasting-accuracy"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

EXPECTED_FILES = [
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
]


def main() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "The 'kaggle' package is not installed.\n"
            "  Fix: pip install kaggle\n"
            "Then re-run: python scripts/download_data.py"
        )
        return 1

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # kaggle raises bare/varied exceptions on missing creds
        print(
            f"Kaggle authentication failed ({exc}).\n"
            "  Fix: create an API token at kaggle.com -> Settings -> API -> "
            "'Create New Token',\n"
            "  and place kaggle.json in ~/.kaggle/ "
            "(Windows: C:/Users/<you>/.kaggle/kaggle.json).\n"
            "  Also accept the competition rules once at\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules"
        )
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    present = [f for f in EXPECTED_FILES if (RAW_DIR / f).is_file()]
    if len(present) == len(EXPECTED_FILES):
        print(f"All M5 files already present in {RAW_DIR}; nothing to do.")
        return 0

    print(f"Downloading {COMPETITION} to {RAW_DIR} ...")
    try:
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
    except Exception as exc:
        print(
            f"Download failed ({exc}).\n"
            "  Most common cause: competition rules not yet accepted. Visit\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules and accept them, "
            "then re-run."
        )
        return 1

    archive = RAW_DIR / f"{COMPETITION}.zip"
    if archive.is_file():
        print(f"Extracting {archive.name} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW_DIR)
        archive.unlink()

    missing = [f for f in EXPECTED_FILES if not (RAW_DIR / f).is_file()]
    if missing:
        print(f"Warning: download finished but expected files are missing: {missing}")
        return 1
    print(f"Done. M5 files in {RAW_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

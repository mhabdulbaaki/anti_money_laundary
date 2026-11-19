"""Paths and fixed settings shared by all the code. Every path is relative to the aml/ folder."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the aml/ folder

# data/raw: the three Kaggle files (download them yourself; never committed)
DATA_RAW = ROOT / "data" / "raw"
TRANS_CSV = DATA_RAW / "HI-Small_Trans.csv"
PATTERNS_TXT = DATA_RAW / "HI-Small_Patterns.txt"
ACCOUNTS_CSV = DATA_RAW / "HI-Small_accounts.csv"

# data/processed: large files the pipeline writes (never committed)
DATA_PROCESSED = ROOT / "data" / "processed"
PREPARED = DATA_PROCESSED / "transactions.parquet"  # one row per transfer: features, label, split
ACCOUNT_KEYS = DATA_PROCESSED / "account_keys.parquet"  # integer account id -> "bank_account" key
SCORES_TEST = DATA_PROCESSED / "scores_test.parquet"  # test-day transfers with model and rule scores
TEST_REASONS = DATA_PROCESSED / "test_reasons.parquet"  # top reasons behind each alert
ACCOUNT_SCORES_TEST = DATA_PROCESSED / "account_scores_test.parquet"  # account-day risk scores

# model: the trained model, its alert threshold and the settings it was trained with
MODEL_DIR = ROOT / "model"
MODEL_FILE = MODEL_DIR / "final_model.joblib"

# outputs: small results (committed) that the notebook and the course report read
OUTPUTS = ROOT / "outputs"
CV_DIR = OUTPUTS / "cv"
FIGURES = OUTPUTS / "figures"
FINTRAC_DIR = OUTPUTS / "fintrac_reports"
TRAIN_METRICS = OUTPUTS / "train_metrics.json"
TEST_METRICS = OUTPUTS / "test_metrics.json"

SEED = 42
# Days 1-10 hold 99.98% of transfers; after that the simulator only finishes laundering
# schemes, so 57-73% of the few remaining transfers are laundering. We report them apart.
LAST_CORE_DAY = "2022-09-10"


def ensure_dirs() -> None:
    for d in (DATA_PROCESSED, MODEL_DIR, OUTPUTS, CV_DIR, FIGURES, FINTRAC_DIR):
        d.mkdir(parents=True, exist_ok=True)

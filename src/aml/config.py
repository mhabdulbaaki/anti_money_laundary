"""Paths and fixed settings shared by every script."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"  # large intermediate files (git-ignored)
RESULTS = ROOT / "results"  # small tables / json that the report quotes
FIGURES = ROOT / "figures"

TRANS_CSV = DATA / "HI-Small_Trans.csv"
PATTERNS_TXT = DATA / "HI-Small_Patterns.txt"
ACCOUNTS_CSV = DATA / "HI-Small_accounts.csv"

PREPARED = ARTIFACTS / "transactions.parquet"
ACCOUNT_KEYS = ARTIFACTS / "account_keys.parquet"

SEED = 42
# Days 1-10 hold 99.98% of transfers; after that the simulator only finishes laundering
# schemes, so 57-73% of the few remaining transfers are laundering. We report them apart.
LAST_CORE_DAY = "2022-09-10"


def ensure_dirs() -> None:
    for d in (ARTIFACTS, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)

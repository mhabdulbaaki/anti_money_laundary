"""Prepare the data (first half of `make train`): read the raw files, check data quality,
convert amounts to US dollars, build the features and the 60/20/20 time split.

Run: python -m src.prepare
Writes data/processed/transactions.parquet (one row per transfer: features, label, split)
and outputs/data_quality.json.
"""

from __future__ import annotations

import json
import time

import polars as pl

from . import config
from .data import KEY_COLUMNS, attach_typology, index_accounts, parse_patterns, read_accounts, read_transactions
from .features import build_features
from .fx import add_amount_usd, estimate_usd_rates, max_conversion_error
from .split import chronological_split


def main() -> None:
    config.ensure_dirs()
    t0 = time.time()
    raw = pl.read_csv(config.TRANS_CSV, infer_schema=False)
    n_raw = raw.height
    n_null = int(sum(raw.null_count().row(0)))
    del raw

    tx = read_transactions(config.TRANS_CSV)
    print(f"read {tx.height:,} rows ({n_raw - tx.height} exact duplicates dropped) in {time.time()-t0:.0f}s")

    rates = estimate_usd_rates(tx, min_usd=100)
    tx = add_amount_usd(tx, rates)
    conv_err = max_conversion_error(tx, rates, min_usd=100)

    tx, keys = index_accounts(tx)
    patterns = parse_patterns(config.PATTERNS_TXT)
    tx = attach_typology(tx, patterns)
    tx = tx.with_columns(
        pl.Series("split", chronological_split(tx["ts"])),
        pl.col("ts").dt.date().alias("day"),
        (pl.col("ts").dt.date() > pl.lit(config.LAST_CORE_DAY).str.to_date()).alias("tail_day"),
    )

    t1 = time.time()
    feats = build_features(tx)
    print(f"features {feats.width} cols in {time.time()-t1:.0f}s")

    keep = [
        "ts", "day", "tail_day", "split", "src", "dst", "src_key", "dst_key", "from_bank", "to_bank",
        "amount_usd", "amt_paid", "amt_recv", "is_laundering", "typology", "attempt_id",
    ]
    out = pl.concat([tx.select(keep), feats], how="horizontal")
    out.write_parquet(config.PREPARED, compression="zstd")
    pl.DataFrame({"key": keys}).write_parquet(config.ACCOUNT_KEYS)

    accounts = read_accounts(config.ACCOUNTS_CSV)
    matched = patterns.join(tx.select(KEY_COLUMNS).unique(), on=KEY_COLUMNS, how="semi").height
    quality = {
        "rows_raw": n_raw,
        "exact_duplicates_dropped": n_raw - tx.height,
        "null_cells": n_null,
        "ts_min": str(tx["ts"].min()),
        "ts_max": str(tx["ts"].max()),
        "non_positive_amounts": int((tx["amt_paid"] <= 0).sum()),
        "amount_usd_min": float(tx["amount_usd"].min()),  # type: ignore[arg-type]  # polars types min/mean loosely
        "amount_usd_max": float(tx["amount_usd"].max()),  # type: ignore[arg-type]  # polars types min/mean loosely
        "amount_usd_median": float(tx["amount_usd"].median()),  # type: ignore[arg-type]  # polars types min/mean loosely
        "usd_rates": dict(sorted(rates.items())),  # sorted so the file is stable across runs
        "max_conversion_error_rel": conv_err,
        "accounts": int(keys.len()),
        "laundering": int(tx["is_laundering"].sum()),
        "laundering_rate": float(tx["is_laundering"].mean()),  # type: ignore[arg-type]  # polars types min/mean loosely
        "pattern_attempts": int(patterns["attempt_id"].n_unique()),
        "pattern_transactions": patterns.height,
        "pattern_transactions_matched": matched,
        "laundering_in_named_pattern": int(tx["typology"].is_not_null().sum()),
        "cross_currency_rows": int((tx["cur_paid"] != tx["cur_recv"]).sum()),
        "cross_currency_laundering": int(tx.filter(pl.col("cur_paid") != pl.col("cur_recv"))["is_laundering"].sum()),
        "accounts_file_rows": accounts.height,
        "accounts_file_entity_types": dict(sorted(accounts["entity_type"].value_counts().iter_rows())),
        "split": {
            s: {
                "rows": int((out["split"] == s).sum()),
                "laundering": int(out.filter(pl.col("split") == s)["is_laundering"].sum()),
                "start": str(out.filter(pl.col("split") == s)["ts"].min()),
                "end": str(out.filter(pl.col("split") == s)["ts"].max()),
            }
            for s in ("train", "val", "test")
        },
        "tail_rows": int(out["tail_day"].sum()),
        "tail_laundering": int(out.filter(pl.col("tail_day"))["is_laundering"].sum()),
    }
    (config.OUTPUTS / "data_quality.json").write_text(json.dumps(quality, indent=2, default=str))
    print(json.dumps({k: v for k, v in quality.items() if k not in ("usd_rates",)}, indent=2, default=str))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

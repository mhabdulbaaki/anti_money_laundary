"""Reading the IBM AMLworld files (HI-Small_Trans.csv, HI-Small_Patterns.txt, HI-Small_accounts.csv)."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import IO

import polars as pl

COLUMNS = [
    "ts",
    "from_bank",
    "from_acct",
    "to_bank",
    "to_acct",
    "amt_recv",
    "cur_recv",
    "amt_paid",
    "cur_paid",
    "fmt",
    "is_laundering",
]
KEY_COLUMNS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt"]

Source = str | Path | IO[str] | IO[bytes]


def _parse(source: Source, has_header: bool) -> pl.DataFrame:
    # Read everything as text first: bank codes and some account numbers are all digits
    # with leading zeros, which numeric inference would destroy.
    raw = pl.read_csv(source, has_header=has_header, infer_schema=False, new_columns=COLUMNS)
    return raw.with_columns(
        pl.col("ts").str.strptime(pl.Datetime("us"), "%Y/%m/%d %H:%M"),
        pl.col("amt_recv", "amt_paid").cast(pl.Float64),
        pl.col("is_laundering").cast(pl.Int8),
    )


def read_transactions(source: Source) -> pl.DataFrame:
    """Parse the transactions file, drop exact duplicate rows, sort by time (stable) and add account keys."""
    tx = _parse(source, has_header=True).unique(maintain_order=True).sort("ts", maintain_order=True)
    return tx.with_columns(
        (pl.col("from_bank") + "_" + pl.col("from_acct")).alias("src_key"),
        (pl.col("to_bank") + "_" + pl.col("to_acct")).alias("dst_key"),
    )


def index_accounts(tx: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    """Add integer ``src`` / ``dst`` ids. Returns the frame and the id -> account key lookup."""
    keys = pl.concat([tx["src_key"], tx["dst_key"]]).unique().sort()
    lookup = pl.DataFrame({"key": keys, "id": pl.int_range(0, keys.len(), eager=True, dtype=pl.Int32)})
    out = tx.join(lookup.rename({"key": "src_key", "id": "src"}), on="src_key", how="left")
    out = out.join(lookup.rename({"key": "dst_key", "id": "dst"}), on="dst_key", how="left")
    return out, keys


_BEGIN = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z-]+):?\s*(.*)")


def parse_patterns(source: Source) -> pl.DataFrame:
    """One row per transaction listed in the patterns file, with ``attempt_id``, ``typology`` and ``detail``."""
    if isinstance(source, (str, Path)):
        text = Path(source).read_text()
    else:
        content = source.read()
        text = content.decode() if isinstance(content, bytes) else content

    lines: list[str] = []
    meta: list[tuple[int, str, str]] = []
    attempt = -1
    typology, detail = "", ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("END"):
            continue
        m = _BEGIN.match(line)
        if m:
            attempt += 1
            typology, detail = m.group(1), m.group(2).strip()
            continue
        lines.append(line)
        meta.append((attempt, typology, detail))

    tx = _parse(io.StringIO("\n".join(lines) + "\n"), has_header=False)
    return tx.with_columns(
        pl.Series("attempt_id", [m[0] for m in meta], dtype=pl.Int32),
        pl.Series("typology", [m[1] for m in meta]),
        pl.Series("detail", [m[2] for m in meta]),
    )


def attach_typology(tx: pl.DataFrame, patterns: pl.DataFrame) -> pl.DataFrame:
    """Add ``typology`` and ``attempt_id`` (null for transfers that are not part of a listed scheme)."""
    p = patterns.select(*KEY_COLUMNS, "typology", "attempt_id").unique(subset=KEY_COLUMNS, keep="first")
    out = tx.join(p, on=KEY_COLUMNS, how="left")
    if out.height != tx.height:  # pragma: no cover - guarded by the unique() above
        raise RuntimeError("pattern join duplicated rows")
    return out


def read_accounts(source: Source) -> pl.DataFrame:
    """Account owner file: bank name, bank id, account number, entity id and entity type."""
    raw = pl.read_csv(source, infer_schema=False)
    raw.columns = ["bank_name", "bank_id", "account", "entity_id", "entity_name"]
    return raw.with_columns(
        pl.col("entity_name").str.replace(r"\s*#\d+$", "").alias("entity_type"),
        pl.col("bank_name").str.replace(r"\s*#\d+$", "").str.replace(r"\s+Bank$", "").alias("bank_country"),
    )


def owner_types(keys: pl.Series, accounts: pl.DataFrame) -> pl.Series:
    """Owner type for each ``bank_account`` key (null when unknown).

    The accounts file writes bank ids without leading zeros, so banks are matched as integers;
    account numbers alone are not unique across banks.
    """
    lookup = accounts.select(
        pl.col("bank_id").cast(pl.Int64).alias("_bank"), pl.col("account").alias("_acct"), "entity_type"
    ).unique(subset=["_bank", "_acct"], keep="first")
    parts = pl.DataFrame({"key": keys}).with_columns(
        pl.col("key").str.split("_").list.get(0).cast(pl.Int64, strict=False).alias("_bank"),
        pl.col("key").str.split("_").list.get(1).alias("_acct"),
    )
    return parts.join(lookup, on=["_bank", "_acct"], how="left")["entity_type"]

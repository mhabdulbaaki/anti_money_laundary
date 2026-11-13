"""Roll transfer scores up into one risk score per account per day."""

from __future__ import annotations

import polars as pl


def account_day_scores(tx: pl.DataFrame, score_col: str = "score") -> pl.DataFrame:
    """For each (day, account): the highest transfer score the account took part in (as payer
    or payee), whether it touched any laundering transfer that day, and how many transfers.

    Needs ``day``, ``src``, ``dst``, ``is_laundering`` and the score column.
    """
    base = tx.select("day", "src", "dst", pl.col(score_col).alias("score"), "is_laundering")
    sides = pl.concat(
        [
            base.select("day", pl.col("src").alias("account"), "score", "is_laundering"),
            base.filter(pl.col("src") != pl.col("dst")).select(
                "day", pl.col("dst").alias("account"), "score", "is_laundering"
            ),
        ]
    )
    return sides.group_by("day", "account").agg(
        pl.col("score").max(),
        pl.col("is_laundering").max().cast(pl.Int8).alias("involved"),
        pl.len().alias("n_tx"),
    )

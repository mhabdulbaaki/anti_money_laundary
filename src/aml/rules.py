"""Warning-sign rules of the kind many banks run, used as a baseline."""

from __future__ import annotations

import polars as pl


def red_flags(f: pl.DataFrame, fan_k: int = 5, pass_band: tuple[float, float] = (0.9, 1.1)) -> pl.DataFrame:
    """0/1 flag per rule and ``rule_score`` (number of rules hit), one row per transfer.

    * fan_out: the sender had already paid at least ``fan_k`` different accounts in the previous 24 hours;
    * fan_in: the receiver had already been paid by at least ``fan_k`` different accounts in the previous 24 hours;
    * pass_through: the sender received money in the last 24 hours and this payment is
      about the same size (``pass_band`` of what came in);
    * structuring: the amount is just below the US$10,000 reporting limit.
    Needs src_out_uniq_24h, dst_in_uniq_24h, src_in_usd_24h, amount_usd, just_below_limit.
    """
    ratio = pl.col("amount_usd") / pl.col("src_in_usd_24h")
    out = f.select(
        (pl.col("src_out_uniq_24h") >= fan_k).cast(pl.Int8).alias("fan_out"),
        (pl.col("dst_in_uniq_24h") >= fan_k).cast(pl.Int8).alias("fan_in"),
        ((pl.col("src_in_usd_24h") > 0) & ratio.is_between(*pass_band)).fill_null(False).cast(pl.Int8).alias("pass_through"),
        pl.col("just_below_limit").cast(pl.Int8).alias("structuring"),
    )
    return out.with_columns(pl.sum_horizontal(pl.all()).cast(pl.Int8).alias("rule_score"))

"""Currency normalisation.

The simulator uses one fixed exchange rate per currency, so we can recover
"US dollars per unit" from transfers that change currency. Very small amounts
are rounded to the cent, which distorts their ratio, so we only use rows whose
US-dollar side is at least ``min_usd``.
"""

from __future__ import annotations

import polars as pl

USD = "US Dollar"


def estimate_usd_rates(tx: pl.DataFrame, min_usd: float = 100.0) -> dict[str, float]:
    """Return {currency: US dollars per unit}, estimated as a median over cross-currency rows."""
    to_usd = tx.filter(
        (pl.col("cur_recv") == USD) & (pl.col("cur_paid") != USD) & (pl.col("amt_recv") >= min_usd)
    ).select(pl.col("cur_paid").alias("cur"), (pl.col("amt_recv") / pl.col("amt_paid")).alias("rate"))
    from_usd = tx.filter(
        (pl.col("cur_paid") == USD) & (pl.col("cur_recv") != USD) & (pl.col("amt_paid") >= min_usd)
    ).select(pl.col("cur_recv").alias("cur"), (pl.col("amt_paid") / pl.col("amt_recv")).alias("rate"))
    med = pl.concat([to_usd, from_usd]).group_by("cur").agg(pl.col("rate").median())
    rates = {USD: 1.0, **dict(zip(med["cur"].to_list(), med["rate"].to_list()))}

    seen = set(tx["cur_paid"].unique().to_list()) | set(tx["cur_recv"].unique().to_list())
    missing = sorted(seen - rates.keys())
    if missing:
        raise ValueError(f"No US-dollar exchange pair to estimate a rate for: {missing}")
    return rates


def add_amount_usd(tx: pl.DataFrame, rates: dict[str, float]) -> pl.DataFrame:
    """Add ``amount_usd``: the amount paid, converted to US dollars."""
    rate = pl.col("cur_paid").replace_strict(rates, return_dtype=pl.Float64)
    return tx.with_columns((pl.col("amt_paid") * rate).alias("amount_usd"))


def max_conversion_error(tx: pl.DataFrame, rates: dict[str, float], min_usd: float = 100.0) -> float:
    """Largest relative gap between the paid and received sides once both are in US dollars."""
    paid = pl.col("amt_paid") * pl.col("cur_paid").replace_strict(rates, return_dtype=pl.Float64)
    recv = pl.col("amt_recv") * pl.col("cur_recv").replace_strict(rates, return_dtype=pl.Float64)
    gaps = tx.filter(pl.col("cur_paid") != pl.col("cur_recv")).select(
        paid.alias("paid"), ((paid - recv).abs() / paid).alias("gap")
    ).filter(pl.col("paid") >= min_usd)
    worst = gaps["gap"].max()
    return float(worst) if isinstance(worst, (int, float)) else 0.0

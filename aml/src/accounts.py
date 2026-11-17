"""Roll transfer scores up into one risk score per account per day, and build the daily review list."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import config
from .metrics import precision_at_k_by_day
from .viz import LAUNDERING, NORMAL, TEXT2


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


REVIEW_LIST_SIZE = 100  # accounts an analyst reviews per day


def daily_review_lists(test: pl.DataFrame) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    """Precision of each day's top-``REVIEW_LIST_SIZE`` accounts, ranked by the model and by the rules.

    Returns (summary, per-day precision table, model account-day scores)."""
    model_acc = account_day_scores(test, "score")
    rule_acc = account_day_scores(test, "rule_score")
    # rules give ties (scores 0-4): break them by the account's total model-free activity
    rule_acc = rule_acc.with_columns((pl.col("score") + pl.col("n_tx") * 1e-6).alias("score"))
    tables = []
    for name, acc in (("model", model_acc), ("rules", rule_acc)):
        p = precision_at_k_by_day(acc.rename({"involved": "label"}), k=REVIEW_LIST_SIZE)
        tables.append(p.with_columns(pl.lit(name).alias("ranker")))
    table = pl.concat(tables).sort("day", "ranker")
    core = table.filter(pl.col("day") <= pl.lit(config.LAST_CORE_DAY).str.to_date())
    out: dict[str, Any] = {}
    for name in ("model", "rules"):
        c = core.filter(pl.col("ranker") == name)
        out[name] = {
            "core_days_mean_precision_at_100": float(c["precision"].mean()),  # type: ignore[arg-type]  # polars types min/mean loosely
            "per_day": c.with_columns(pl.col("day").cast(pl.Utf8)).to_dicts(),
        }
    out["involved_accounts_per_day"] = (
        model_acc.group_by("day")
        .agg(pl.col("involved").sum(), pl.len().alias("active_accounts"))
        .sort("day")
        .with_columns(pl.col("day").cast(pl.Utf8))
        .to_dicts()
    )
    return out, table, model_acc


def fig_accounts(table: pl.DataFrame) -> plt.Figure:
    core = table.filter(pl.col("day") <= pl.lit(config.LAST_CORE_DAY).str.to_date())
    days = sorted(core["day"].unique().to_list())
    x = np.arange(len(days))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    w = 0.32
    for off, name, colour, label in ((-w / 2 - 0.02, "model", NORMAL, "Model risk score"), (w / 2 + 0.02, "rules", LAUNDERING, "Warning-sign rules")):
        vals = [float(core.filter((pl.col("day") == d) & (pl.col("ranker") == name))["precision"][0]) for d in days]
        ax.bar(x + off, vals, width=w, color=colour, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi + off, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.5, color=TEXT2)
    ax.set_xticks(x, [d.strftime("%b %d") + (" (from 16:12)" if i == 0 else "") for i, d in enumerate(days)])
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Top-100 accounts truly involved")
    ax.set_title("Daily review list: precision of the top 100 accounts")
    ax.legend(loc="upper center", ncol=2, fontsize=7)
    return fig

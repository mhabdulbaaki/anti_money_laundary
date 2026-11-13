"""Step 3: exploratory charts (figures/eda_*.png) and the numbers behind them (results/eda_summary.json)."""

from __future__ import annotations

import json
import time

import polars as pl

from aml import config, eda
from aml.data import parse_patterns, read_accounts
from aml.viz import apply_style, save


def main() -> None:
    config.ensure_dirs()
    apply_style()
    t = time.time()
    df = pl.read_parquet(config.PREPARED)
    patterns = parse_patterns(config.PATTERNS_TXT)
    accounts = read_accounts(config.ACCOUNTS_CSV)
    F = config.FIGURES

    daily = eda.daily_profile(df)
    save(eda.fig_daily(daily, config.LAST_CORE_DAY), F / "eda_01_daily.png")
    fmt = eda.category_profile(df, "fmt")
    save(eda.fig_shares(fmt, "fmt", "ACH carries 12% of transfers but 87% of laundering"), F / "eda_02_payment_format.png")
    pairs = eda.currency_pairs(df)
    cur = eda.category_profile(df, "cur_paid")
    save(eda.fig_shares(cur, "cur_paid", "Payment currency: laundering mirrors overall use"), F / "eda_03_currency.png")
    save(eda.fig_amount(df), F / "eda_04_amount.png")
    save(eda.fig_hour(df), F / "eda_05_hour.png")
    stats = eda.account_graph_stats(df)
    hubs = eda.hub_summary(stats)
    save(eda.fig_hubs(stats), F / "eda_06_hubs.png")
    save(eda.fig_typology_counts(patterns, int(df["is_laundering"].sum())), F / "eda_07_typology_counts.png")
    save(eda.fig_typology_examples(patterns), F / "eda_08_typology_examples.png")
    lift = eda.signal_lift(df)
    save(eda.fig_signal_lift(lift), F / "eda_09_signal_lift.png")
    corr, names = eda.spearman_matrix(df)
    save(eda.fig_correlation(corr, names), F / "eda_10_correlation.png")
    ent = eda.entity_profile(df, accounts)
    save(eda.fig_entity(ent), F / "eda_11_entity_type.png")

    top_hubs = stats.sort("pagerank", descending=True).head(10).join(
        pl.read_parquet(config.ACCOUNT_KEYS).with_row_index("account").with_columns(pl.col("account").cast(pl.Int32)),
        on="account",
        how="left",
    )
    summary = {
        "daily": daily.with_columns(pl.col("day").cast(pl.Utf8)).to_dicts(),
        "payment_format": fmt.to_dicts(),
        "currency": cur.to_dicts(),
        "currency_pairs": {
            "pairs": pairs.height,
            "cross_currency_pairs": int(pairs["changes_currency"].sum()),
            "cross_currency_transfers": int(pairs.filter(pl.col("changes_currency"))["transfers"].sum()),
            "cross_currency_laundering": int(pairs.filter(pl.col("changes_currency"))["laundering"].sum()),
            "top_cross_pairs": pairs.filter(pl.col("changes_currency")).head(10).to_dicts(),
        },
        "hubs": hubs,
        "top_pagerank_accounts": top_hubs.to_dicts(),
        "signal_lift": lift.to_dicts(),
        "entity_type": ent.to_dicts(),
        "typology": patterns.group_by("typology").agg(pl.col("attempt_id").n_unique().alias("attempts"), pl.len().alias("transfers")).to_dicts(),
        "corr_with_label": dict(zip(names, [float(c) for c in corr[-1]])),
        **eda.summary_numbers(df),
    }
    (config.RESULTS / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"EDA done in {time.time()-t:.0f}s")
    print(json.dumps(hubs, indent=1))
    print(lift)
    print(ent)


if __name__ == "__main__":
    main()

"""Exploratory analysis: the tables behind each chart, and the charts themselves.

Every function takes the prepared transfer table (artifacts/transactions.parquet).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .data import owner_types
from .graph import pagerank
from .rules import red_flags
from .viz import BLUE, DIVERGING, GRID, LAUNDERING, MUTED, NORMAL, TEXT, TEXT2, note

TYPOLOGY_ORDER = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM", "BIPARTITE", "STACK"]


def _compact(x: float, _pos: int | None = None) -> str:
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:g}{suf}"
    return f"{x:g}"


# ---------------------------------------------------------------- time
def daily_profile(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("day")
        .agg(pl.len().alias("transfers"), pl.col("is_laundering").sum().alias("laundering"))
        .sort("day")
        .with_columns((pl.col("laundering") / pl.col("transfers")).alias("rate"))
    )


def fig_daily(daily: pl.DataFrame, last_core_day: str) -> Figure:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"hspace": 0.35})
    days = daily["day"].to_list()
    x = np.arange(len(days))
    a1.bar(x, daily["transfers"], width=0.6, color=NORMAL)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(FuncFormatter(_compact))
    a1.set_title("Transfers per day (log scale)")
    a2.plot(x, daily["rate"].to_numpy() * 100, color=LAUNDERING, marker="o", markersize=4)
    a2.set_yscale("log")
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}%"))
    a2.set_title("Share of transfers that are laundering (log scale)")
    tail_start = next(i for i, d in enumerate(days) if str(d) > last_core_day)
    for ax in (a1, a2):
        ax.axvspan(tail_start - 0.5, len(days) - 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    a1.text(tail_start, a1.get_ylim()[1] * 0.5, "simulation tail:\nonly unfinished schemes", fontsize=7.5, color=TEXT2, va="top")
    a2.set_xticks(x, [d.strftime("%b %d") for d in days], rotation=45, ha="right")
    return fig


# ---------------------------------------------------------------- payment format / currency
def category_profile(df: pl.DataFrame, col: str) -> pl.DataFrame:
    total, launder = df.height, int(df["is_laundering"].sum())
    return (
        df.group_by(col)
        .agg(pl.len().alias("transfers"), pl.col("is_laundering").sum().alias("laundering"))
        .with_columns(
            (pl.col("transfers") / total).alias("share_of_transfers"),
            (pl.col("laundering") / launder).alias("share_of_laundering"),
            (pl.col("laundering") / pl.col("transfers")).alias("rate"),
        )
        .sort("transfers", descending=True)
    )


def fig_shares(prof: pl.DataFrame, col: str, title: str) -> Figure:
    prof = prof.sort("share_of_transfers")
    labels = prof[col].to_list()
    y = np.arange(len(labels))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(labels) + 1.2))
    ax.barh(y + h / 2 + 0.02, prof["share_of_transfers"] * 100, height=h, color=NORMAL, label="% of all transfers")
    ax.barh(y - h / 2 - 0.02, prof["share_of_laundering"] * 100, height=h, color=LAUNDERING, label="% of laundering transfers")
    for yi, v in zip(y, prof["share_of_laundering"] * 100):
        if v >= 5:
            ax.text(v + 0.8, yi - h / 2, f"{v:.0f}%", va="center", fontsize=7.5, color=TEXT2)
    ax.set_yticks(y, labels)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig


def currency_pairs(df: pl.DataFrame) -> pl.DataFrame:
    """Transfers and laundering per (payment currency, receiving currency) pair."""
    return (
        df.group_by("cur_paid", "cur_recv")
        .agg(pl.len().alias("transfers"), pl.col("is_laundering").sum().alias("laundering"))
        .with_columns((pl.col("cur_paid") != pl.col("cur_recv")).alias("changes_currency"))
        .sort("transfers", descending=True)
    )


# ---------------------------------------------------------------- amounts and hours
def fig_amount(df: pl.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bins = np.linspace(-4, 11, 76)
    for label, colour, name in ((0, NORMAL, "Normal"), (1, LAUNDERING, "Laundering")):
        v = np.log10(df.filter(pl.col("is_laundering") == label)["amount_usd"].to_numpy())
        ax.hist(v, bins=list(bins), density=True, histtype="step", lw=1.6, color=colour, label=name)
    ax.axvline(4, color=TEXT2, lw=0.8)
    ax.text(4.1, ax.get_ylim()[1] * 0.92, "US$10,000\nreporting limit", fontsize=7.5, color=TEXT2, va="top")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "$" + _compact(10**v) if v >= 0 else f"${10**v:g}"))
    ax.set_xlabel("Amount in US dollars (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Laundering transfers are larger than normal ones")
    ax.legend(loc="upper left")
    return fig


def fig_hour(df: pl.DataFrame) -> Figure:
    h = (
        df.group_by("hour")
        .agg(pl.len().alias("n"), pl.col("is_laundering").mean().alias("rate"))
        .sort("hour")
    )
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.3})
    a1.bar(h["hour"], h["n"], width=0.6, color=NORMAL)
    a1.yaxis.set_major_formatter(FuncFormatter(_compact))
    a1.set_title("Transfers by hour of day")
    a2.plot(h["hour"], h["rate"] * 100, color=LAUNDERING, marker="o", markersize=3.5)
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}%"))
    a2.set_title("Laundering share by hour of day")
    for ax in (a1, a2):
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Hour")
    return fig


# ---------------------------------------------------------------- hubs (whole-period graph)
def account_graph_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Distinct counterparties in/out, transfer counts, PageRank and laundering involvement per account."""
    nl = df.filter(pl.col("src") != pl.col("dst")).select("src", "dst", "is_laundering")
    pairs = nl.select("src", "dst").unique()
    n = int(max(df["src"].to_numpy().max(), df["dst"].to_numpy().max())) + 1
    pr = pagerank(pairs["src"].to_numpy(), pairs["dst"].to_numpy(), n)
    out_deg = pairs.group_by("src").len().rename({"src": "account", "len": "out_degree"})
    in_deg = pairs.group_by("dst").len().rename({"dst": "account", "len": "in_degree"})
    sides = pl.concat(
        [nl.select(pl.col("src").alias("account"), "is_laundering"), nl.select(pl.col("dst").alias("account"), "is_laundering")]
    )
    inv = sides.group_by("account").agg(pl.len().alias("transfers"), pl.col("is_laundering").max().alias("involved"))
    stats = (
        inv.join(out_deg, on="account", how="left")
        .join(in_deg, on="account", how="left")
        .with_columns(pl.col("out_degree").fill_null(0), pl.col("in_degree").fill_null(0))
    )
    return stats.with_columns(pl.Series("pagerank", pr[stats["account"].to_numpy()]))


def hub_summary(stats: pl.DataFrame, top_frac: float = 0.001) -> dict[str, float]:
    k = max(int(stats.height * top_frac), 1)
    out: dict[str, float] = {"accounts": stats.height, "top_k": k}
    for col in ("out_degree", "in_degree", "pagerank"):
        top = stats.sort(col, descending=True).head(k)
        rest = stats.sort(col, descending=True).slice(k)
        out[f"{col}_top_involved_rate"] = float(top["involved"].mean())  # type: ignore[arg-type]
        out[f"{col}_rest_involved_rate"] = float(rest["involved"].mean())  # type: ignore[arg-type]
        out[f"{col}_top_share_of_transfers"] = float(top["transfers"].sum() / stats["transfers"].sum())
    return out


def fig_hubs(stats: pl.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    rng = np.random.default_rng(0)
    for flag, colour, name, size, alpha in ((0, NORMAL, "Never touched laundering", 3, 0.25), (1, LAUNDERING, "Touched laundering", 7, 0.8)):
        s = stats.filter(pl.col("involved") == flag)
        jx = rng.uniform(-0.15, 0.15, s.height)
        jy = rng.uniform(-0.15, 0.15, s.height)
        ax.scatter(
            np.log10(s["out_degree"].to_numpy() + 1) + jx * 0.1,
            np.log10(s["in_degree"].to_numpy() + 1) + jy * 0.1,
            s=size,
            color=colour,
            alpha=alpha,
            lw=0,
            label=name,
            rasterized=True,
        )
    top = stats.sort("out_degree", descending=True).head(3)
    for row in top.iter_rows(named=True):
        ax.annotate(
            f"hub: pays {row['out_degree']:,} accounts",
            (np.log10(row["out_degree"] + 1), np.log10(row["in_degree"] + 1)),
            xytext=(-120, -18),
            textcoords="offset points",
            fontsize=7.5,
            color=TEXT2,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6},
        )
    vals = [0, 10, 100, 1000, 10000]
    ax.set_xticks([np.log10(v + 1) for v in vals], [f"{v:,}" for v in vals])
    ax.set_yticks([np.log10(v + 1) for v in vals[:4]], [f"{v:,}" for v in vals[:4]])
    ax.set_xlabel("Distinct accounts paid (log scale)")
    ax.set_ylabel("Distinct accounts paid by (log scale)")
    ax.set_title("Accounts by number of counterparties")
    ax.legend(loc="lower right", markerscale=2)
    return fig


# ---------------------------------------------------------------- typologies
def fig_typology_counts(patterns: pl.DataFrame, total_laundering: int) -> Figure:
    t = (
        patterns.group_by("typology")
        .agg(pl.col("attempt_id").n_unique().alias("attempts"), pl.len().alias("transfers"))
        .with_columns(pl.col("typology").replace_strict({k: i for i, k in enumerate(TYPOLOGY_ORDER)}).alias("o"))
        .sort("o", descending=True)
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    y = np.arange(t.height)
    ax.barh(y, t["transfers"], height=0.55, color=LAUNDERING)
    for yi, n, a in zip(y, t["transfers"], t["attempts"]):
        ax.text(n + 8, yi, f"{n:,} transfers · {a} schemes", va="center", fontsize=7.5, color=TEXT2)
    ax.set_yticks(y, t["typology"].to_list())
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, float(t["transfers"].to_numpy().max()) * 1.45)
    named = patterns.height
    ax.set_title(f"Named laundering schemes: {named:,} of {total_laundering:,} laundering transfers ({named / total_laundering:.0%})")
    ax.set_xlabel("Laundering transfers in the patterns file")
    return fig


def _example_attempt(patterns: pl.DataFrame, typology: str, lo: int = 5, hi: int = 12) -> pl.DataFrame:
    sizes = patterns.filter(pl.col("typology") == typology).group_by("attempt_id").len().sort("attempt_id")
    ok = sizes.filter(pl.col("len").is_between(lo, hi))
    attempt = (ok if ok.height else sizes)["attempt_id"][0]
    return patterns.filter(pl.col("attempt_id") == attempt)


def fig_typology_examples(patterns: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0))
    for ax, typ in zip(axes.ravel(), TYPOLOGY_ORDER):
        ex = _example_attempt(patterns, typ)
        g = nx.MultiDiGraph()
        for r in ex.iter_rows(named=True):
            g.add_edge(r["from_bank"] + r["from_acct"], r["to_bank"] + r["to_acct"])
        if typ in ("FAN-OUT", "FAN-IN", "GATHER-SCATTER"):
            hub = max(g.degree, key=lambda kv: kv[1])[0]
            others = [n for n in g.nodes if n != hub]
            pos = nx.shell_layout(g, nlist=[[hub], others])
        elif typ in ("CYCLE", "RANDOM"):
            pos = nx.circular_layout(g)
        else:
            pos = nx.kamada_kawai_layout(nx.Graph(g))
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color=TEXT2, width=0.9, arrowsize=8, node_size=90, connectionstyle="arc3,rad=0.08")
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=90, node_color=BLUE, edgecolors="white", linewidths=1.0)
        ax.set_title(typ.title(), fontsize=9)
        ax.text(0.5, -0.04, f"{ex.height} transfers, {g.number_of_nodes()} accounts", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=TEXT2)
        ax.set_axis_off()
    fig.suptitle("One real example of each laundering typology (arrows = transfers)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- relations between features and the label
FAN_K = 5  # the warning-sign rules' fan threshold used for exploration
SIGNALS: list[tuple[str, pl.Expr]] = [
    ("Sender pays this receiver for the first time", pl.col("pair_seen_before") == 0),
    ("Receiver had already paid the sender", pl.col("reciprocal_before") == 1),
    ("Transfer closes a three-step loop", pl.col("cycle3_before") == 1),
    # the next four are exactly the warning-sign rules (aml.rules.red_flags)
    (f"Sender already paid {FAN_K}+ accounts in previous 24 h", pl.col("fan_out") == 1),
    (f"Receiver already paid by {FAN_K}+ accounts in previous 24 h", pl.col("fan_in") == 1),
    ("Pays on about what came in (24 h)", pl.col("pass_through") == 1),
    ("Just below US$10,000", pl.col("structuring") == 1),
    ("ACH payment", pl.col("fmt") == "ACH"),
    ("Same bank on both sides", pl.col("same_bank") == 1),
    ("Changes currency", pl.col("cross_currency") == 1),
]


def signal_lift(df: pl.DataFrame) -> pl.DataFrame:
    base = float(df["is_laundering"].mean())  # type: ignore[arg-type]
    total_pos = int(df["is_laundering"].sum())
    df = pl.concat([df, red_flags(df, fan_k=FAN_K).drop("rule_score")], how="horizontal")
    rows = []
    for name, cond in SIGNALS:
        sub = df.filter(cond)
        rate = float(sub["is_laundering"].mean()) if sub.height else 0.0  # type: ignore[arg-type]
        rows.append(
            {
                "signal": name,
                "transfers": sub.height,
                "laundering_rate": rate,
                "lift": rate / base,
                "coverage": int(sub["is_laundering"].sum()) / total_pos,
            }
        )
    return pl.DataFrame(rows).sort("lift")


def fig_signal_lift(lift: pl.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(lift.height)
    vals = np.maximum(lift["lift"].to_numpy(), 1e-2)
    ax.barh(y, vals, height=0.55, color=[LAUNDERING if v >= 1 else NORMAL for v in vals])
    ax.set_xscale("log")
    ax.axvline(1, color=TEXT2, lw=0.8)
    for yi, v, cov, n in zip(y, vals, lift["coverage"], lift["transfers"]):
        label = f"{v:.1f}× · covers {cov:.0%}" if v > 1e-2 else f"never laundering ({n:,} transfers)"
        ax.text(v * 1.15, yi, label, va="center", ha="left", fontsize=7.5, color=TEXT2)
    ax.set_yticks(y, lift["signal"].to_list())
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.01, vals.max() * 12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}×"))
    ax.set_xlabel("Laundering rate when the signal is present ÷ overall rate (log scale)")
    ax.set_title("How strongly each warning sign relates to laundering")
    note(ax, "'covers' = share of all laundering transfers that show the sign. Orange = above the base rate.", (0.0, -0.2))
    return fig


CORR_FEATURES = [
    "log_amount_usd",
    "src_out_cnt_24h",
    "src_out_uniq_24h",
    "src_out_usd_24h",
    "src_in_cnt_24h",
    "src_in_usd_24h",
    "dst_in_cnt_24h",
    "dst_in_uniq_24h",
    "dst_in_usd_24h",
    "dst_out_cnt_24h",
    "src_out_uniq_7d",
    "dst_in_uniq_7d",
    "src_in_out_ratio_24h",
    "amount_vs_src_in_24h",
    "amount_vs_src_avg_out_7d",
    "pair_seen_before",
    "reciprocal_before",
    "cycle3_before",
    "is_laundering",
]


def spearman_matrix(df: pl.DataFrame, n: int = 300_000, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    s = df.select(CORR_FEATURES).sample(n=min(n, df.height), seed=seed)
    ranks = s.select([pl.col(c).cast(pl.Float64).rank("average") for c in CORR_FEATURES]).to_numpy()
    return np.corrcoef(ranks, rowvar=False), CORR_FEATURES


def fig_correlation(corr: np.ndarray, names: list[str]) -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and abs(corr[i, j]) >= 0.6:
                ax.text(j, i, f"{corr[i, j]:.1f}", ha="center", va="center", fontsize=5.5, color="white" if abs(corr[i, j]) > 0.75 else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    getattr(cb, "outline").set_visible(False)
    cb.set_label("Spearman correlation", color=TEXT2)
    ax.set_title("Engineered features move together in families")
    return fig


# ---------------------------------------------------------------- owners (accounts file)
def entity_profile(df: pl.DataFrame, accounts: pl.DataFrame) -> pl.DataFrame:
    owners = owner_types(df["src_key"], accounts).fill_null("Unknown")
    joined = df.select("is_laundering").with_columns(owners.alias("entity_type"))
    return (
        joined.group_by("entity_type")
        .agg(pl.len().alias("transfers"), pl.col("is_laundering").sum().alias("laundering"))
        .with_columns((pl.col("laundering") / pl.col("transfers")).alias("rate"))
        .sort("transfers", descending=True)
    )


def fig_entity(prof: pl.DataFrame) -> Figure:
    prof = prof.sort("rate")
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    y = np.arange(prof.height)
    ax.barh(y, prof["rate"] * 100, height=0.55, color=NORMAL)
    for yi, r, n in zip(y, prof["rate"] * 100, prof["transfers"]):
        ax.text(r, yi, f"  {r:.3f}%  ({n:,} transfers)", va="center", fontsize=7.5, color=TEXT2)
    ax.set_yticks(y, prof["entity_type"].to_list())
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, float(prof["rate"].max()) * 100 * 1.8)  # type: ignore[arg-type]
    ax.set_xlabel("Laundering share of transfers sent (%)")
    ax.set_title("Laundering share by sender's owner type")
    return fig


def summary_numbers(df: pl.DataFrame) -> dict[str, object]:
    pos = df.filter(pl.col("is_laundering") == 1)
    return {
        "ach_share_of_laundering": float((pos["fmt"] == "ACH").mean()),  # type: ignore[arg-type]
        "ach_share_of_transfers": float((df["fmt"] == "ACH").mean()),  # type: ignore[arg-type]
        "laundering_formats_with_zero": [f for f in df["fmt"].unique().to_list() if pos.filter(pl.col("fmt") == f).height == 0],
        "median_usd_normal": float(df.filter(pl.col("is_laundering") == 0)["amount_usd"].median()),  # type: ignore[arg-type]
        "median_usd_laundering": float(pos["amount_usd"].median()),  # type: ignore[arg-type]
        "self_transfer_share": float(df["self_transfer"].mean()),  # type: ignore[arg-type]
        "self_transfer_laundering": int(pos["self_transfer"].sum()),
    }

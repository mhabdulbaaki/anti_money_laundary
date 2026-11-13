"""Step 5: explanations (do the top reasons match the true scheme?) and the daily account review list.

Reads artifacts/final_model.joblib and artifacts/scores_test.parquet.
Writes results/explanations.json, results/accounts.json, artifacts/test_reasons.parquet
and figures/explain_*.png, figures/accounts_*.png.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from aml import config
from aml.accounts import account_day_scores
from aml.explain import (
    TYPOLOGY_FAMILIES,
    feature_family,
    permutation_baseline,
    random_reason_baseline,
    top_reasons,
    typology_hit_rate,
)
from aml.features import SHORTCUT_FEATURES
from aml.metrics import precision_at_k_by_day
from aml.models import contributions
from aml.viz import BLUE, GRID, LAUNDERING, MUTED, NORMAL, TEXT2, apply_style, save

warnings.filterwarnings("ignore", category=UserWarning)
K_REASONS = 3
K_ALERTS = 100


def explain(test: pl.DataFrame, bundle: dict[str, Any]) -> tuple[dict[str, Any], pl.DataFrame, np.ndarray, list[str]]:
    model, thr, features = bundle["model"], bundle["threshold"], bundle["features"]
    flagged = test.filter(pl.col("score") >= thr)
    contrib, names = contributions(model, flagged.select(features).to_pandas())
    reasons = top_reasons(contrib, names, k=K_REASONS)
    flagged = flagged.with_columns(pl.Series("reasons", reasons, dtype=pl.List(pl.Utf8)))

    tp = flagged.filter((pl.col("is_laundering") == 1) & pl.col("typology").is_not_null())
    typ = tp["typology"].to_list()
    rs = tp["reasons"].to_list()
    observed = typology_hit_rate(rs, typ)
    perm_mean, p_value = permutation_baseline(rs, typ, n_perm=2000, seed=config.SEED)
    rand = random_reason_baseline(typ, names, k=K_REASONS)
    per_typ = []
    for t in TYPOLOGY_FAMILIES:
        idx = [i for i, x in enumerate(typ) if x == t]
        if idx:
            per_typ.append(
                {
                    "typology": t,
                    "n": len(idx),
                    "hit_rate": typology_hit_rate([rs[i] for i in idx], [t] * len(idx)),
                    "random_reasons": random_reason_baseline([t], names, k=K_REASONS),
                }
            )
    fam_counts: dict[str, int] = {}
    for r in flagged.filter(pl.col("is_laundering") == 1)["reasons"].to_list():
        for f in r:
            fam_counts[feature_family(f)] = fam_counts.get(feature_family(f), 0) + 1
    result = {
        "alerts_explained": flagged.height,
        "true_positives_with_known_scheme": tp.height,
        "hit_rate": observed,
        "permutation_baseline": perm_mean,
        "permutation_p_value": p_value,
        "random_reason_baseline": rand,
        "per_typology": per_typ,
        "reason_families_in_true_alerts": fam_counts,
    }
    return result, flagged, contrib, names


def fig_importance(contrib: np.ndarray, names: list[str]) -> plt.Figure:
    imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(order))
    colours = [LAUNDERING if names[i] in SHORTCUT_FEATURES else NORMAL for i in order]
    ax.barh(y, imp[order], height=0.55, color=colours)
    ax.set_yticks(y, [names[i] for i in order], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |contribution| to the log-odds, over all alerts on test days")
    ax.set_title("What drives the alerts (TreeSHAP)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LAUNDERING, label="simulator shortcut (format/currency)"),
        plt.Rectangle((0, 0), 1, 1, color=NORMAL, label="behaviour feature"),
    ]
    ax.legend(handles=handles, loc="lower right")
    return fig


def fig_hit_rates(res: dict[str, Any]) -> plt.Figure:
    rows = sorted(res["per_typology"], key=lambda r: r["hit_rate"])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        ax.plot([r["random_reasons"], r["hit_rate"]], [yi, yi], color=GRID, lw=2.5)
        ax.plot(r["random_reasons"], yi, "o", color=MUTED, markersize=5, mec="white", mew=1)
        ax.plot(r["hit_rate"], yi, "o", color=BLUE, markersize=6, mec="white", mew=1)
        ax.text(1.02, yi, f"n={r['n']}", va="center", fontsize=7, color=TEXT2)
    ax.set_yticks(y, [r["typology"] for r in rows], fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, label="model's top-3 reasons"),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, label="3 random features"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    ax.set_xlabel("Share of caught laundering whose top-3 reasons match its scheme")
    ax.set_title("Do the reasons point to the true laundering pattern?")
    return fig


def accounts(test: pl.DataFrame) -> tuple[dict[str, Any], pl.DataFrame]:
    model_acc = account_day_scores(test, "score")
    rule_acc = account_day_scores(test, "rule_score")
    # rules give ties (scores 0-4): break them by the account's total model-free activity
    rule_acc = rule_acc.with_columns((pl.col("score") + pl.col("n_tx") * 1e-6).alias("score"))
    out: dict[str, Any] = {}
    tables = []
    for name, acc in (("model", model_acc), ("rules", rule_acc)):
        p = precision_at_k_by_day(acc.rename({"involved": "label"}), k=K_ALERTS).with_columns(pl.lit(name).alias("ranker"))
        tables.append(p)
    table = pl.concat(tables).sort("day", "ranker")
    tail = config.LAST_CORE_DAY
    core = table.filter(pl.col("day") <= pl.lit(tail).str.to_date())
    for name in ("model", "rules"):
        c = core.filter(pl.col("ranker") == name)
        out[name] = {
            "core_days_mean_precision_at_100": float(c["precision"].mean()),
            "per_day": c.with_columns(pl.col("day").cast(pl.Utf8)).to_dicts(),
        }
    out["involved_accounts_per_day"] = (
        model_acc.group_by("day").agg(pl.col("involved").sum(), pl.len().alias("active_accounts")).sort("day").with_columns(pl.col("day").cast(pl.Utf8)).to_dicts()
    )
    model_acc.write_parquet(config.ARTIFACTS / "account_scores_test.parquet")
    return out, table


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


def main() -> None:
    apply_style()
    bundle = joblib.load(config.ARTIFACTS / "final_model.joblib")
    test = pl.read_parquet(config.ARTIFACTS / "scores_test.parquet")

    exp, flagged, contrib, names = explain(test, bundle)
    save(fig_importance(contrib, names), config.FIGURES / "explain_importance.png")
    save(fig_hit_rates(exp), config.FIGURES / "explain_typology_hits.png")
    # keep contributions of flagged transfers for the report drafter
    flagged.select("row", "score", "reasons").with_columns(
        pl.Series("contrib", contrib.astype(np.float32).tolist(), dtype=pl.List(pl.Float32))
    ).write_parquet(config.ARTIFACTS / "test_reasons.parquet")
    (config.RESULTS / "explanations.json").write_text(json.dumps({**exp, "feature_names": names}, indent=2))

    acc, table = accounts(test)
    save(fig_accounts(table), config.FIGURES / "accounts_precision_at_100.png")
    (config.RESULTS / "accounts.json").write_text(json.dumps(acc, indent=2, default=str))
    print(json.dumps({k: v for k, v in exp.items() if k != "per_typology"}, indent=1))
    print(json.dumps(exp["per_typology"], indent=1))
    print(json.dumps({k: v["core_days_mean_precision_at_100"] for k, v in acc.items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()

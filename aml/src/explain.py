"""Do the model's reasons match the true laundering pattern?

For each alert we take the features that pushed its score up the most (TreeSHAP
contributions) and ask whether any of them belongs to the warning-sign family that
the scheme's typology should trigger. We compare that hit rate with two baselines:
the same reasons matched against shuffled typologies (permutation), and reasons
drawn at random from all features.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from math import comb
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import config
from .features import SHORTCUT_FEATURES
from .models import contributions
from .viz import BLUE, GRID, LAUNDERING, MUTED, NORMAL, TEXT2

# Which warning-sign families each typology should light up (fixed before looking at results).
TYPOLOGY_FAMILIES: dict[str, set[str]] = {
    "FAN-OUT": {"fan_out"},
    "FAN-IN": {"fan_in"},
    "GATHER-SCATTER": {"fan_in", "fan_out", "pass_through"},
    "SCATTER-GATHER": {"fan_out", "fan_in", "parallel_paths"},
    "CYCLE": {"loop", "pass_through"},
    "RANDOM": {"pass_through"},
    "BIPARTITE": {"fan_out", "fan_in"},
    "STACK": {"fan_out", "fan_in", "parallel_paths", "pass_through"},
}

_FAMILY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("fan_out", re.compile(r"^src_out_")),
    ("fan_in", re.compile(r"^dst_in_")),
    ("pass_through", re.compile(r"^(src_in_|dst_out_|src_in_out_ratio|amount_vs_src_in|mins_since_src_in)")),
    ("loop", re.compile(r"^(reciprocal_before|cycle3_before)$")),
    ("parallel_paths", re.compile(r"^paths2_before$")),
]


def feature_family(name: str) -> str:
    for family, pattern in _FAMILY_RULES:
        if pattern.search(name):
            return family
    return "other"


def top_reasons(contrib: np.ndarray, names: Sequence[str], k: int = 3) -> list[list[str]]:
    """Names of the k features with the largest POSITIVE contribution in each row."""
    order = np.argsort(-contrib, axis=1)[:, :k]
    out = []
    for row, idx in zip(contrib, order):
        out.append([names[i] for i in idx if row[i] > 0])
    return out


def _hits(reasons: Sequence[Sequence[str]], typologies: Sequence[str]) -> np.ndarray:
    return np.array(
        [any(feature_family(f) in TYPOLOGY_FAMILIES[t] for f in r) for r, t in zip(reasons, typologies)],
        dtype=float,
    )


def typology_hit_rate(reasons: Sequence[Sequence[str]], typologies: Sequence[str]) -> float:
    return float(_hits(reasons, typologies).mean())


def permutation_baseline(
    reasons: Sequence[Sequence[str]], typologies: Sequence[str], n_perm: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """Mean hit rate when typologies are shuffled, and the one-sided p-value of the observed rate."""
    rng = np.random.default_rng(seed)
    observed = typology_hit_rate(reasons, typologies)
    t = np.array(typologies)
    rates = np.array([typology_hit_rate(reasons, list(rng.permutation(t))) for _ in range(n_perm)])
    p = (1 + int(np.sum(rates >= observed))) / (n_perm + 1)
    return float(rates.mean()), float(p)


def random_reason_baseline(typologies: Sequence[str], features: Sequence[str], k: int = 3) -> float:
    """Expected hit rate if the k reasons were k distinct features drawn at random."""
    n = len(features)
    probs = []
    for t in typologies:
        g = sum(feature_family(f) in TYPOLOGY_FAMILIES[t] for f in features)
        probs.append(1 - comb(n - g, k) / comb(n, k))
    return float(np.mean(probs))


# ---------------------------------------------------------------- applied to the test-day alerts
K_REASONS = 3


def explain_alerts(test: pl.DataFrame, bundle: dict[str, Any]) -> tuple[dict[str, Any], pl.DataFrame, np.ndarray, list[str]]:
    """Top-``K_REASONS`` TreeSHAP reasons for every alert, and how often they match the true scheme.

    Returns (summary, alerts with a ``reasons`` column, contributions, feature names)."""
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

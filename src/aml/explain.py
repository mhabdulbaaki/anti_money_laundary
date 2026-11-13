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

import numpy as np

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

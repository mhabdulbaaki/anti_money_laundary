"""Time-ordered splits.

Cuts are placed at timestamps (not row positions), so all transfers from the
same minute land in the same part and no later information sits in an earlier part.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl


def _as_int(ts: pl.Series | np.ndarray) -> np.ndarray:
    if isinstance(ts, pl.Series):
        return ts.dt.epoch("s").to_numpy().astype(np.int64)
    return np.asarray(ts).astype(np.int64)


def _cut_points(t: np.ndarray, fractions: Sequence[float]) -> list[int]:
    s = np.sort(t)
    n = len(s)
    return [int(s[min(int(np.floor(f * n)), n - 1)]) for f in np.cumsum(fractions)[:-1]]


def chronological_split(
    ts: pl.Series | np.ndarray, fractions: Sequence[float] = (0.6, 0.2, 0.2)
) -> np.ndarray:
    """Label every row "train", "val" or "test" by time (≈60/20/20 of rows by default)."""
    t = _as_int(ts)
    c1, c2 = _cut_points(t, fractions)
    return np.where(t < c1, "train", np.where(t < c2, "val", "test"))


def expanding_time_folds(
    ts: pl.Series | np.ndarray, n_splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: cut time into n_splits + 1 blocks of about equal size;
    fold k trains on blocks 1..k and validates on block k + 1. Indices are positions in ``ts``."""
    t = _as_int(ts)
    cuts = _cut_points(t, [1 / (n_splits + 1)] * (n_splits + 1))
    edges = [np.iinfo(np.int64).min, *cuts, np.iinfo(np.int64).max]
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    folds = []
    for k in range(1, n_splits + 1):
        train = order[t_sorted < edges[k]]
        val = order[(t_sorted >= edges[k]) & (t_sorted < edges[k + 1])]
        folds.append((train, val))
    return folds

"""Whole-graph statistics for exploration (not used as model features, since they see the future)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def pagerank(src: np.ndarray, dst: np.ndarray, n: int, alpha: float = 0.85, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """PageRank of a directed graph given as edge lists (duplicate edges collapse to one).

    Matches networkx.pagerank: dangling nodes spread their rank uniformly.
    """
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A.data[:] = 1.0  # collapse repeated edges
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, out_deg, out=np.zeros(n), where=out_deg > 0)
    P = sp.diags(inv) @ A  # row-stochastic for non-dangling rows
    dangling = out_deg == 0
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        x_new = alpha * (P.T @ x + x[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(x_new - x).sum() < n * tol:
            return x_new
        x = x_new
    return x

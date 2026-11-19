"""Scores that match the cost of errors on a 0.1%-positive problem."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Alert threshold with the highest F1 on (y, score), and that F1."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    p, r = precision[:-1], recall[:-1]
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def f1_at(y: np.ndarray, score: np.ndarray, threshold: float) -> float:
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def confusion(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, int]:
    pred = score >= threshold
    return {
        "tp": int(np.sum(pred & (y == 1))),
        "fp": int(np.sum(pred & (y == 0))),
        "fn": int(np.sum(~pred & (y == 1))),
        "tn": int(np.sum(~pred & (y == 0))),
    }


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


def summary(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    c = confusion(y, score, threshold)
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    return {
        "f1": f1_at(y, score, threshold),
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc(y, score) if y.sum() > 0 else float("nan"),
        "threshold": float(threshold),
        **c,
    }


def precision_at_k_by_day(df: pl.DataFrame, k: int = 100) -> pl.DataFrame:
    """Share of each day's top-k scored rows whose label is 1. Needs ``day``, ``score``, ``label``."""
    top = df.sort("score", descending=True).group_by("day", maintain_order=True).head(k)
    hits = top.group_by("day").agg(pl.col("label").mean().alias("precision"), pl.len().alias("alerts"))
    pos = df.group_by("day").agg(pl.col("label").sum().alias("positives_in_day"))
    return hits.join(pos, on="day").sort("day")

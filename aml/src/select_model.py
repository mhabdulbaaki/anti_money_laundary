"""Model selection (`make select`, 1-2 hours): time-ordered cross-validation of every candidate
setting on the training days only. All candidates use the SAME three folds.

Run: python -m src.select_model [model ...]   (default: xgboost lightgbm logreg)
Appends one row per (model, setting, fold) to outputs/cv/<model>.csv and skips settings already
there, so it can be stopped and resumed. `make train` then picks the best setting from these files
(see `summarise`), so this only needs re-running when the candidates or the features change.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from . import config
from .features import feature_columns
from .models import cross_validate, grid
from .split import expanding_time_folds
from .viz import BLUE, GRID, MUTED, ORANGE, TEXT2, save

warnings.filterwarnings("ignore", category=UserWarning)
N_FOLDS = 3
FAMILY_LABEL = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "logreg": "Logistic regression"}


def run_cv(models: list[str]) -> None:
    config.ensure_dirs()
    train = pl.read_parquet(config.PREPARED).filter(pl.col("split") == "train")
    features = feature_columns()
    X = train.select(features).to_pandas()
    y = train["is_laundering"].to_numpy()
    folds = expanding_time_folds(train["ts"], N_FOLDS)
    fold_info = [
        {"fold": k, "train_rows": len(tr), "val_rows": len(va), "val_laundering": int(y[va].sum()),
         "val_start": str(train["ts"][int(va.min())]), "val_end": str(train["ts"][int(va.max())])}
        for k, (tr, va) in enumerate(folds)
    ]
    (config.OUTPUTS / "cv_folds_info.json").write_text(json.dumps(fold_info, indent=2))

    for name in models:
        out = config.CV_DIR / f"{name}.csv"
        done = set(pd.read_csv(out)["params"]) if out.exists() else set()
        for params in grid(name):
            key = json.dumps(params, sort_keys=True)
            if key in done:
                continue
            t = time.time()
            rows = cross_validate(name, params, X, y, folds, features, seed=config.SEED)
            frame = pd.DataFrame(rows).assign(model=name, params=key)
            frame.to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{name} {key}: F1 {frame.f1.mean():.3f}±{frame.f1.std():.3f} "
                  f"PR-AUC {frame.pr_auc.mean():.3f} ({time.time()-t:.0f}s)", flush=True)


def cv_summary() -> pd.DataFrame:
    """Mean and SD over folds for every setting that finished all folds, best first."""
    paths = sorted(config.CV_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No cross-validation results in {config.CV_DIR}: run `make select` first.")
    folds = pd.concat([pd.read_csv(p) for p in paths])
    s = (
        folds.groupby(["model", "params"])
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            folds=("fold", "count"),
            seconds=("seconds", "sum"),
        )
        .reset_index()
        .sort_values(["f1_mean", "pr_auc_mean"], ascending=False)
    )
    return s[s.folds == s.folds.max()]


def short_params(p: dict[str, Any]) -> str:
    keys = {"max_depth": "depth", "learning_rate": "lr", "scale_pos_weight": "spw", "num_leaves": "leaves", "is_unbalance": "unbal", "penalty": "", "C": "C"}
    return ", ".join(f"{keys.get(k, k)}={v}".lstrip("=") for k, v in p.items())


def fig_cv(s: pd.DataFrame, chosen: str) -> plt.Figure:
    s = s.copy()
    s["label"] = s.apply(lambda r: f"{FAMILY_LABEL[r.model]} {short_params(json.loads(r.params))}", axis=1)
    s = s.sort_values("f1_mean")
    fig, ax = plt.subplots(figsize=(7.2, 0.17 * len(s) + 1.2))
    y = np.arange(len(s))
    colours = {"xgboost": BLUE, "lightgbm": ORANGE, "logreg": MUTED}
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.plot([r.f1_mean - r.f1_std, r.f1_mean + r.f1_std], [yi, yi], color=GRID, lw=2.5, solid_capstyle="round")
        ax.plot(r.f1_mean, yi, "o", color=colours[r.model], markersize=4.5, mec="white", mew=1)
    ax.set_yticks(y, s["label"], fontsize=6.5)
    best_y = int(np.where(s["label"].to_numpy() == chosen)[0][0])
    ax.annotate("selected", (s.f1_mean.iloc[best_y], best_y), xytext=(14, -3), textcoords="offset points", fontsize=7.5, color=TEXT2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cross-validated F1 on the laundering class (mean ± SD over 3 time-ordered folds)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=FAMILY_LABEL[m]) for m, c in colours.items()]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("Every candidate on the same three folds")
    return fig


def summarise() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write outputs/cv_summary.csv and the comparison chart; return (all settings, best per family)."""
    s = cv_summary()
    s.to_csv(config.OUTPUTS / "cv_summary.csv", index=False)
    best = s.groupby("model").head(1).set_index("model")
    winner = str(best.f1_mean.idxmax())
    chosen = f"{FAMILY_LABEL[winner]} {short_params(json.loads(best.loc[winner, 'params']))}"
    save(fig_cv(s, chosen), config.FIGURES / "model_cv_comparison.png")
    return s, best


if __name__ == "__main__":
    from .viz import apply_style

    apply_style()
    run_cv(sys.argv[1:] or ["xgboost", "lightgbm", "logreg"])
    print(summarise()[1][["params", "f1_mean", "f1_std", "pr_auc_mean"]])

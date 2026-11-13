"""Step 2: time-ordered cross-validation of every candidate on the training days only.

Usage: python scripts/02_cv.py [model ...]   (default: xgboost lightgbm logreg)
Appends one row per (model, setting, fold) to results/cv/<model>.csv and skips settings
already there, so it can be stopped and resumed (and models can run in parallel).
All candidates use the SAME three folds.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import pandas as pd
import polars as pl

from aml import config
from aml.features import feature_columns
from aml.models import cross_validate, grid
from aml.split import expanding_time_folds

warnings.filterwarnings("ignore", category=UserWarning)
N_FOLDS = 3


def main(models: list[str]) -> None:
    config.ensure_dirs()
    cv_dir = config.RESULTS / "cv"
    cv_dir.mkdir(exist_ok=True)

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
    (config.RESULTS / "cv_folds_info.json").write_text(json.dumps(fold_info, indent=2))

    for name in models:
        out = cv_dir / f"{name}.csv"
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


if __name__ == "__main__":
    main(sys.argv[1:] or ["xgboost", "lightgbm", "logreg"])

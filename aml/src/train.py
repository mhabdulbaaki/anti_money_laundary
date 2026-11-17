"""Train the model (`make train`, after src.prepare).

Run: python -m src.train
1. take the best setting of each model family from cross-validation on the training days
   (outputs/cv/*.csv, written by `make select`) and pick the family with the highest mean F1;
2. refit that setting on all training days and set its alert threshold to the best F1 on the
   validation days;
3. tune the warning-sign rules on the same validation days, as a fair baseline;
4. shortcut test: the same setting without payment format and currency (validation days only).
The test days are not touched here; `make evaluate` scores them once.

Writes model/final_model.joblib, outputs/train_metrics.json, outputs/cv_summary.csv and
outputs/figures/model_cv_comparison.png.
"""

from __future__ import annotations

import json
import time
import warnings
from typing import Any

import joblib
import polars as pl

from . import config
from .features import feature_columns
from .metrics import best_f1_threshold, summary
from .models import make_model, n_trees
from .rules import tune_rules
from .select_model import summarise
from .viz import apply_style

warnings.filterwarnings("ignore", category=UserWarning)
PUBLISHED_F1 = {"lightgbm_no_graph_features": 0.21, "gfp_lightgbm": 0.6286, "gfp_xgboost": 0.6323}


def main() -> None:
    apply_style()
    config.ensure_dirs()
    t0 = time.time()
    df = pl.read_parquet(config.PREPARED)
    features = feature_columns()
    train = df.filter(pl.col("split") == "train")
    val = df.filter(pl.col("split") == "val")
    X_tr, y_tr = train.select(features).to_pandas(), train["is_laundering"].to_numpy()
    X_va, y_va = val.select(features).to_pandas(), val["is_laundering"].to_numpy()

    # ---- 1. model selection from cross-validation (training days only)
    _, best = summarise()
    winner = str(best.f1_mean.idxmax())
    families: dict[str, Any] = {
        str(fam): {
            "params": json.loads(row.params),
            "cv_f1_mean": row.f1_mean,
            "cv_f1_std": row.f1_std,
            "cv_pr_auc_mean": row.pr_auc_mean,
            "cv_pr_auc_std": row.pr_auc_std,
        }
        for fam, row in best.iterrows()
    }
    params = families[winner]["params"]
    print(f"selected {winner} {params} (CV F1 {families[winner]['cv_f1_mean']:.3f})")

    # ---- 2. refit on all training days; alert threshold tuned on the validation days
    t = time.time()
    model = make_model(winner, params, features, seed=config.SEED).fit(X_tr, y_tr)
    s_val = model.predict_proba(X_va)[:, 1]
    thr, _ = best_f1_threshold(y_va, s_val)
    families[winner].update(
        trees=n_trees(model.named_steps["clf"].estimator_), val=summary(y_va, s_val, thr), fit_seconds=time.time() - t
    )
    print("validation", families[winner]["val"])

    # ---- 3. warning-sign rules, tuned on the validation days
    rules = tune_rules(val)

    # ---- 4. shortcut test: same setting without the simulator shortcuts (validation days only)
    no_short = feature_columns(drop_shortcuts=True)
    abl = make_model(winner, params, no_short, seed=config.SEED).fit(train.select(no_short).to_pandas(), y_tr)
    s_abl = abl.predict_proba(val.select(no_short).to_pandas())[:, 1]
    abl_thr, _ = best_f1_threshold(y_va, s_abl)
    ablation = {"dropped": [c for c in features if c not in no_short], "val": summary(y_va, s_abl, abl_thr)}

    joblib.dump(
        {"model": model, "threshold": thr, "features": features, "family": winner, "params": params, "rules": rules},
        config.MODEL_FILE,
    )
    out = {
        "winner_family": winner,
        "threshold": thr,
        "families": families,
        "rules": {"tuned_on_val": rules},
        "ablation_no_shortcuts": ablation,
        "published_reference_f1": PUBLISHED_F1,
        "seconds": time.time() - t0,
    }
    config.TRAIN_METRICS.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved {config.MODEL_FILE.relative_to(config.ROOT)} (threshold {thr:.3f}) in {time.time() - t0:.0f}s")
    print("rules", json.dumps(rules, default=float), "| without shortcuts", json.dumps(ablation["val"], default=float))


if __name__ == "__main__":
    main()

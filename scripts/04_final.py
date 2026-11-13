"""Step 4: choose the winner from cross-validation, tune the alert threshold on the validation
days, and score the untouched test days ONCE. Also: baselines and the shortcut ablation.

Writes results/cv_summary.csv, results/final_metrics.json, artifacts/final_model.joblib,
artifacts/scores_{val,test}.parquet and figures/model_*.png.
"""

from __future__ import annotations

import json
import time
import warnings
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import precision_recall_curve

from aml import config
from aml.features import feature_columns
from aml.metrics import best_f1_threshold, summary
from aml.models import make_model, n_trees
from aml.rules import red_flags
from aml.viz import BLUE, GRID, LAUNDERING, MUTED, NORMAL, ORANGE, TEXT, TEXT2, apply_style, save

warnings.filterwarnings("ignore", category=UserWarning)
FAMILY_LABEL = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "logreg": "Logistic regression"}


def cv_summary() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((config.RESULTS / "cv").glob("*.csv"))]
    folds = pd.concat(frames)
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


def rule_baseline(val: pl.DataFrame) -> dict[str, Any]:
    y = val["is_laundering"].to_numpy()
    best: dict[str, Any] = {"f1": -1.0}
    for k in (3, 5, 10, 20):
        score = red_flags(val, fan_k=k)["rule_score"].to_numpy().astype(float)
        for min_hits in (1, 2):
            m = summary(y, score, min_hits)
            if m["f1"] > best["f1"]:
                best = {"fan_k": k, "min_hits": min_hits, **m}
    return best


def evaluate_parts(frame: pl.DataFrame, score: np.ndarray, threshold: float) -> dict[str, Any]:
    out = {}
    for name, mask in (
        ("all", np.ones(frame.height, bool)),
        ("core_days", ~frame["tail_day"].to_numpy()),
        ("tail_days", frame["tail_day"].to_numpy()),
    ):
        y = frame["is_laundering"].to_numpy()[mask]
        out[name] = summary(y, score[mask], threshold) if mask.any() else {}
    return out


def fig_cv(s: pd.DataFrame, chosen: str) -> plt.Figure:
    s = s.copy()
    s["label"] = s.apply(lambda r: f"{FAMILY_LABEL[r.model]} {_short(json.loads(r.params))}", axis=1)
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


def _short(p: dict[str, Any]) -> str:
    keys = {"max_depth": "depth", "learning_rate": "lr", "scale_pos_weight": "spw", "num_leaves": "leaves", "is_unbalance": "unbal", "penalty": "", "C": "C"}
    return ", ".join(f"{keys.get(k, k)}={v}".lstrip("=") for k, v in p.items())


def fig_result(
    y: np.ndarray, s: np.ndarray, thr: float, rule_point: tuple[float, float], conf: dict[str, int], label: str
) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.4})
    p, r, _ = precision_recall_curve(y, s)
    a1.plot(r, p, color=NORMAL, lw=1.6, label=label)
    op_p = conf["tp"] / max(conf["tp"] + conf["fp"], 1)
    op_r = conf["tp"] / max(conf["tp"] + conf["fn"], 1)
    a1.plot(op_r, op_p, "o", color=NORMAL, markersize=6, mec="white", mew=1.5)
    a1.annotate(
        f"alert threshold\nP = {op_p:.2f}, R = {op_r:.2f}",
        (op_r, op_p),
        xytext=(0.05, 0.42),
        textcoords="data",
        ha="left",
        fontsize=7,
        color=TEXT2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6},
    )
    a1.plot(rule_point[1], rule_point[0], "s", color=LAUNDERING, markersize=6, mec="white", mew=1.5, label="Warning-sign rules", clip_on=False)
    a1.set_xlabel("Recall")
    a1.set_ylabel("Precision")
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.02)
    a1.legend(loc="upper right", fontsize=7)
    a1.set_title("(a) Precision-recall, normal test days", fontsize=9)
    m = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    a2.imshow(np.log10(m + 1), cmap="Blues", vmin=0, vmax=np.log10(m.max() + 1) * 1.3)
    for i in range(2):
        for j in range(2):
            a2.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=8.5, color=TEXT)
    a2.set_xticks([0, 1], ["Normal", "Laundering"], fontsize=7.5)
    a2.set_yticks([0, 1], ["Normal", "Laundering"], fontsize=7.5)
    a2.set_xlabel("Predicted")
    a2.set_ylabel("True")
    a2.grid(False)
    a2.set_title("(b) Confusion matrix", fontsize=9)
    return fig


def main() -> None:
    apply_style()
    t0 = time.time()
    df = pl.read_parquet(config.PREPARED)
    features = feature_columns()
    train = df.filter(pl.col("split") == "train")
    val = df.filter(pl.col("split") == "val")
    test = df.filter(pl.col("split") == "test")
    X_tr, y_tr = train.select(features).to_pandas(), train["is_laundering"].to_numpy()
    X_va, y_va = val.select(features).to_pandas(), val["is_laundering"].to_numpy()

    # ---- 1. model selection from cross-validation (training days only)
    s = cv_summary()
    s.to_csv(config.RESULTS / "cv_summary.csv", index=False)
    best = s.groupby("model").head(1).set_index("model")
    winner_family = str(best.f1_mean.idxmax())
    print(best[["params", "f1_mean", "f1_std", "pr_auc_mean"]])

    # ---- 2. refit the best setting of each family on all training days; tune the threshold on validation
    families: dict[str, Any] = {}
    fitted = {}
    for fam, row in best.iterrows():
        params = json.loads(row.params)
        t = time.time()
        model = make_model(str(fam), params, features, seed=config.SEED).fit(X_tr, y_tr)
        sv = model.predict_proba(X_va)[:, 1]
        thr, _ = best_f1_threshold(y_va, sv)
        families[str(fam)] = {
            "params": params,
            "cv_f1_mean": row.f1_mean,
            "cv_f1_std": row.f1_std,
            "cv_pr_auc_mean": row.pr_auc_mean,
            "cv_pr_auc_std": row.pr_auc_std,
            "trees": n_trees(model.named_steps["clf"].estimator_),
            "val": summary(y_va, sv, thr),
            "fit_seconds": time.time() - t,
        }
        fitted[str(fam)] = (model, sv, thr)
        print(fam, families[str(fam)]["val"])

    model, s_val, thr = fitted[winner_family]

    # ---- 3. baselines (rules tuned on validation days)
    rules = rule_baseline(val)
    rule_test = red_flags(test, fan_k=rules["fan_k"])["rule_score"].to_numpy().astype(float)

    # ---- 4. the one look at the test days
    X_te, y_te = test.select(features).to_pandas(), test["is_laundering"].to_numpy()
    s_test = model.predict_proba(X_te)[:, 1]
    test_metrics = evaluate_parts(test, s_test, thr)
    rule_metrics = evaluate_parts(test, rule_test, rules["min_hits"])
    nothing = evaluate_parts(test, np.zeros(test.height), 0.5)
    print("TEST", json.dumps(test_metrics, indent=1))

    # ---- 5. ablation: same setting without the simulator shortcuts (validation days only)
    no_short = feature_columns(drop_shortcuts=True)
    abl = make_model(winner_family, families[winner_family]["params"], no_short, seed=config.SEED).fit(train.select(no_short).to_pandas(), y_tr)
    s_abl = abl.predict_proba(val.select(no_short).to_pandas())[:, 1]
    abl_thr, _ = best_f1_threshold(y_va, s_abl)
    ablation = {"dropped": [c for c in features if c not in no_short], "val": summary(y_va, s_abl, abl_thr)}

    # ---- save
    joblib.dump({"model": model, "threshold": thr, "features": features, "family": winner_family}, config.ARTIFACTS / "final_model.joblib")
    keep = ["ts", "day", "tail_day", "src", "dst", "src_key", "dst_key", "amount_usd", "is_laundering", "typology", "attempt_id", *features]
    # "row" identifies a transfer uniquely (pair + minute + score is not unique), for joins in steps 5-6
    val.select(keep).with_columns(pl.Series("score", s_val)).with_row_index("row").write_parquet(config.ARTIFACTS / "scores_val.parquet")
    test.select(keep).with_columns(pl.Series("score", s_test), pl.Series("rule_score", rule_test)).with_row_index("row").write_parquet(
        config.ARTIFACTS / "scores_test.parquet"
    )

    core = ~test["tail_day"].to_numpy()
    conf = test_metrics["core_days"]
    rule_core = rule_metrics["core_days"]
    save(
        fig_result(y_te[core], s_test[core], thr, (rule_core["precision"], rule_core["recall"]), conf, f"Selected {FAMILY_LABEL[winner_family]}"),
        config.FIGURES / "model_result_test.png",
    )
    chosen_label = f"{FAMILY_LABEL[winner_family]} {_short(families[winner_family]['params'])}"
    save(fig_cv(s, chosen_label), config.FIGURES / "model_cv_comparison.png")

    out = {
        "winner_family": winner_family,
        "threshold": thr,
        "families": families,
        "test": test_metrics,
        "rules": {"tuned_on_val": rules, "test": rule_metrics},
        "do_nothing": {"test": nothing},
        "ablation_no_shortcuts": ablation,
        "published_reference_f1": {"lightgbm_no_graph_features": 0.21, "gfp_lightgbm": 0.6286, "gfp_xgboost": 0.6323},
        "seconds": time.time() - t0,
    }
    (config.RESULTS / "final_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("winner_family", "threshold", "ablation_no_shortcuts")}, indent=1, default=float))
    print("rules", json.dumps(out["rules"], indent=1, default=float))


if __name__ == "__main__":
    main()

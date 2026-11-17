"""Test the trained model (`make evaluate`, before the AI drafts).

Run: python -m src.evaluate
1. score every test-day transfer with the saved model and its validation-tuned threshold, and
   compare with the warning-sign rules and with doing nothing -> outputs/test_metrics.json;
2. explain each alert: top-3 TreeSHAP reasons vs the true scheme -> outputs/explanations.json;
3. daily account review list: precision of each day's top 100 accounts -> outputs/accounts.json.
Nothing here changes the model or its threshold. Also writes the scored test days, alert reasons
and account scores to data/processed/ for the AI drafts, and charts to outputs/figures/.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve

from . import config
from .accounts import daily_review_lists, fig_accounts
from .explain import explain_alerts, fig_hit_rates, fig_importance
from .metrics import summary
from .rules import red_flags
from .select_model import FAMILY_LABEL
from .viz import LAUNDERING, MUTED, NORMAL, TEXT, TEXT2, apply_style, save

warnings.filterwarnings("ignore", category=UserWarning)
KEEP = ["ts", "day", "tail_day", "src", "dst", "src_key", "dst_key", "amount_usd", "is_laundering", "typology", "attempt_id"]


def load_model() -> dict[str, Any]:
    """The bundle `make train` saved: model, threshold, features, family, params, tuned rules."""
    if not config.MODEL_FILE.exists():
        raise SystemExit(f"No trained model at {config.MODEL_FILE}: run `make train` first.")
    bundle: dict[str, Any] = joblib.load(config.MODEL_FILE)
    return bundle


def evaluate_parts(frame: pl.DataFrame, score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Metrics on all test days, on the normal days only, and on the tail days only."""
    out = {}
    for name, mask in (
        ("all", np.ones(frame.height, bool)),
        ("core_days", ~frame["tail_day"].to_numpy()),
        ("tail_days", frame["tail_day"].to_numpy()),
    ):
        y = frame["is_laundering"].to_numpy()[mask]
        out[name] = summary(y, score[mask], threshold) if mask.any() else {}
    return out


def score_test_days(bundle: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score the test days once. Returns (scored transfers with a ``row`` id, metrics)."""
    features, thr, rules = bundle["features"], bundle["threshold"], bundle["rules"]
    test = pl.read_parquet(config.PREPARED).filter(pl.col("split") == "test")
    score = bundle["model"].predict_proba(test.select(features).to_pandas())[:, 1]
    rule_score = red_flags(test, fan_k=rules["fan_k"])["rule_score"].to_numpy().astype(float)
    metrics = {
        "family": bundle["family"],
        "threshold": thr,
        "test": evaluate_parts(test, score, thr),
        "rules": {"test": evaluate_parts(test, rule_score, rules["min_hits"])},
        "do_nothing": {"test": evaluate_parts(test, np.zeros(test.height), 0.5)},
    }
    # "row" identifies a transfer uniquely (pair + minute + score is not unique), for later joins
    scored = (
        test.select(*KEEP, *features)
        .with_columns(pl.Series("score", score), pl.Series("rule_score", rule_score))
        .with_row_index("row")
    )
    return scored, metrics


def fig_result(
    y: np.ndarray, s: np.ndarray, rule_point: tuple[float, float], conf: dict[str, int], label: str
) -> plt.Figure:
    """(a) precision-recall curve with the alert threshold and the rules marked; (b) confusion matrix."""
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


def result_figure(scored: pl.DataFrame, metrics: dict[str, Any]) -> plt.Figure:
    """Figure 1 of the course report, on the normal test days."""
    core = scored.filter(~pl.col("tail_day"))
    rules = metrics["rules"]["test"]["core_days"]
    return fig_result(
        core["is_laundering"].to_numpy(),
        core["score"].to_numpy(),
        (rules["precision"], rules["recall"]),
        metrics["test"]["core_days"],
        f"Selected {FAMILY_LABEL[metrics['family']]}",
    )


def main() -> None:
    apply_style()
    config.ensure_dirs()
    bundle = load_model()

    # ---- 1. the one look at the test days
    scored, metrics = score_test_days(bundle)
    scored.write_parquet(config.SCORES_TEST)
    config.TEST_METRICS.write_text(json.dumps(metrics, indent=2, default=float))
    save(result_figure(scored, metrics), config.FIGURES / "model_result_test.png")

    # ---- 2. why each alert was raised, and does it match the true scheme?
    exp, flagged, contrib, names = explain_alerts(scored, bundle)
    save(fig_importance(contrib, names), config.FIGURES / "explain_importance.png")
    save(fig_hit_rates(exp), config.FIGURES / "explain_typology_hits.png")
    flagged.select("row", "score", "reasons").with_columns(
        pl.Series("contrib", contrib.astype(np.float32).tolist(), dtype=pl.List(pl.Float32))
    ).write_parquet(config.TEST_REASONS)
    (config.OUTPUTS / "explanations.json").write_text(json.dumps({**exp, "feature_names": names}, indent=2))

    # ---- 3. the daily account review list
    acc, table, account_scores = daily_review_lists(scored)
    account_scores.write_parquet(config.ACCOUNT_SCORES_TEST)
    save(fig_accounts(table), config.FIGURES / "accounts_precision_at_100.png")
    (config.OUTPUTS / "accounts.json").write_text(json.dumps(acc, indent=2, default=str))

    core = metrics["test"]["core_days"]
    print(
        f"normal test days: F1 {core['f1']:.3f} (P {core['precision']:.2f}, R {core['recall']:.2f}); "
        f"with tail days: F1 {metrics['test']['all']['f1']:.3f}; rules: F1 {metrics['rules']['test']['core_days']['f1']:.3f}"
    )
    print(f"explanations: hit rate {exp['hit_rate']:.0%} vs shuffled {exp['permutation_baseline']:.0%}, random {exp['random_reason_baseline']:.0%}")
    print(f"daily top-100 accounts truly involved: model {acc['model']['core_days_mean_precision_at_100']:.0%}, rules {acc['rules']['core_days_mean_precision_at_100']:.0%}")


if __name__ == "__main__":
    main()

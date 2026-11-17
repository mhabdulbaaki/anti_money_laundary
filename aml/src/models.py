"""Candidate models, their hyper-parameter grids and a time-ordered cross-validation runner.

Every candidate is a scikit-learn Pipeline (encoders + classifier), so encoders are fitted
only on the training rows of each fold. The classifier is wrapped in ``HoldoutTuned``: it
keeps the most recent ``holdout_frac`` of its training rows aside, uses them for early
stopping (boosted trees) and to pick its own alert threshold, and never looks further ahead.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import CATEGORICAL
from .metrics import best_f1_threshold, f1_at, pr_auc

GRIDS: dict[str, dict[str, list[Any]]] = {
    # scale_pos_weight = 1 (no re-weighting) is added as a reference point next to 500 and 1000.
    "xgboost": {"max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1], "scale_pos_weight": [1, 500, 1000]},
    "lightgbm": {"num_leaves": [31, 63, 127], "learning_rate": [0.05, 0.1], "is_unbalance": [True, False]},
    "logreg": {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1.0, 10.0]},
}
MAX_TREES = 1000
EARLY_STOP = 50


def grid(name: str) -> list[dict[str, Any]]:
    g = GRIDS[name]
    return [dict(zip(g, values)) for values in itertools.product(*g.values())]


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class HoldoutTuned(ClassifierMixin, BaseEstimator):
    """Fit ``estimator`` on all but the latest ``holdout_frac`` rows (rows must be in time order);
    use the held-out tail for early stopping and to choose ``threshold_`` (max F1).

    ``neg_frac`` < 1 fits on every positive but only that random share of the negatives in the
    fitting part (the held-out tail is never sampled), for solvers too slow on millions of rows."""

    def __init__(self, estimator: Any = None, holdout_frac: float = 0.15, neg_frac: float = 1.0, random_state: int = 42):
        self.estimator = estimator
        self.holdout_frac = holdout_frac
        self.neg_frac = neg_frac
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> HoldoutTuned:
        y = np.asarray(y)
        cut = int(len(y) * (1 - self.holdout_frac))
        X_fit, y_fit, X_hold, y_hold = X[:cut], y[:cut], X[cut:], y[cut:]
        if self.neg_frac < 1.0:
            rng = np.random.default_rng(self.random_state)
            keep = (y_fit == 1) | (rng.random(len(y_fit)) < self.neg_frac)
            X_fit, y_fit = X_fit[keep], y_fit[keep]
        est = clone(self.estimator)
        if isinstance(est, XGBClassifier):
            est.set_params(early_stopping_rounds=EARLY_STOP)
            est.fit(X_fit, y_fit, eval_set=[(X_hold, y_hold)], verbose=False)
        elif isinstance(est, lgb.LGBMClassifier):
            est.fit(
                X_fit,
                y_fit,
                eval_set=[(X_hold, y_hold)],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
            )
        else:
            est.fit(X_fit, y_fit)
        self.estimator_ = est
        self.classes_ = est.classes_
        hold_score = est.predict_proba(X_hold)[:, 1]
        self.threshold_, self.holdout_f1_ = best_f1_threshold(y_hold, hold_score)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


def make_preprocessor(kind: str, features: list[str]) -> ColumnTransformer:
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in CATEGORICAL]
    if kind == "linear":
        num_pipe = Pipeline([("log", FunctionTransformer(_signed_log1p)), ("scale", StandardScaler())])
        cat_enc: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        num_pipe = "passthrough"
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    return ColumnTransformer(
        [("num", num_pipe, nums), ("cat", cat_enc, cats)], verbose_feature_names_out=False
    )


def make_model(name: str, params: dict[str, Any], features: list[str], seed: int = 42) -> Pipeline:
    if name == "xgboost":
        clf: Any = XGBClassifier(
            n_estimators=MAX_TREES,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=8,
            random_state=seed,
            **params,
        )
        kind = "tree"
    elif name == "lightgbm":
        # min_child_weight=1 (XGBoost's default) stops leaf values exploding on 1:1000 data;
        # LightGBM's default (1e-3) made the model saturate after one or two trees.
        clf = lgb.LGBMClassifier(
            n_estimators=MAX_TREES,
            min_child_weight=1.0,
            metric="average_precision",
            n_jobs=8,
            random_state=seed,
            verbose=-1,
            **params,
        )
        kind = "tree"
    elif name == "logreg":
        # liblinear's L1 solver did not finish within an hour on 2.3M rows, so L1 uses saga, and
        # logistic regression sees every positive plus 10% of negatives (class weights rebalance).
        solver = "saga" if params.get("penalty") == "l1" else "liblinear"
        clf = LogisticRegression(
            solver=solver, class_weight="balanced", max_iter=300, tol=1e-3, random_state=seed, **params
        )
        return Pipeline(
            [("prep", make_preprocessor("linear", features)), ("clf", HoldoutTuned(clf, neg_frac=0.1, random_state=seed))]
        )
    else:
        raise ValueError(name)
    return Pipeline([("prep", make_preprocessor(kind, features)), ("clf", HoldoutTuned(clf))])


def cross_validate(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    features: list[str],
    seed: int = 42,
) -> list[dict[str, float]]:
    """Fit on each fold's past, score the fold's next block with the threshold the model chose itself."""
    rows = []
    for k, (tr, va) in enumerate(folds):
        t = time.time()
        model = make_model(name, params, features, seed)
        model.fit(X.iloc[tr], y[tr])
        s = model.predict_proba(X.iloc[va])[:, 1]
        clf = model.named_steps["clf"]
        rows.append(
            {
                "fold": k,
                "f1": f1_at(y[va], s, clf.threshold_),
                "pr_auc": pr_auc(y[va], s),
                "threshold": float(clf.threshold_),
                "trees": float(n_trees(clf.estimator_)),
                "seconds": time.time() - t,
            }
        )
    return rows


def n_trees(est: Any) -> int:
    """Trees kept after early stopping (0 for non-tree models)."""
    if isinstance(est, XGBClassifier):
        return int(est.best_iteration) + 1
    if isinstance(est, lgb.LGBMClassifier):
        return int(est.best_iteration_ or est.n_estimators)
    return 0


def contributions(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Per-feature contributions to the log-odds (TreeSHAP) for a fitted tree pipeline.

    Returns (n_rows x n_features) array without the bias column, and the feature names.
    """
    prep = model.named_steps["prep"]
    est = model.named_steps["clf"].estimator_
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())
    if isinstance(est, XGBClassifier):
        import xgboost as xgb

        booster = est.get_booster()
        contrib = booster.predict(
            xgb.DMatrix(Xt), pred_contribs=True, iteration_range=(0, est.best_iteration + 1)
        )
    elif isinstance(est, lgb.LGBMClassifier):
        contrib = est.predict(Xt, pred_contrib=True, num_iteration=est.best_iteration_ or None)
    else:
        raise TypeError("contributions() needs a tree model")
    return np.asarray(contrib)[:, :-1], names

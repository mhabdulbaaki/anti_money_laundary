import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models import HoldoutTuned


class Recorder(ClassifierMixin, BaseEstimator):
    """Stub classifier that remembers the labels it was trained on and scores by the first column."""

    def fit(self, X, y):
        self.seen_y_ = np.asarray(y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X)[:, 0].astype(float)
        return np.column_stack([1 - p, p])


def test_negative_sampling_keeps_every_positive_and_only_some_negatives():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.01).astype(int)
    X = y.reshape(-1, 1).astype(float)
    model = HoldoutTuned(Recorder(), holdout_frac=0.2, neg_frac=0.1).fit(X, y)
    seen = model.estimator_.seen_y_
    head = y[:16_000]
    assert seen.sum() == head.sum()  # all positives in the fitting part are kept
    assert 0.07 < (seen == 0).sum() / (head == 0).sum() < 0.13


def test_threshold_is_chosen_on_the_latest_rows_only():
    y = np.array([0] * 80 + [0, 1] * 10)
    X = np.zeros((100, 1))
    X[80:, 0] = np.where(y[80:] == 1, 0.9, 0.2)  # held-out tail is perfectly separable at 0.9
    model = HoldoutTuned(Recorder(), holdout_frac=0.2).fit(X, y)
    assert model.threshold_ == 0.9
    assert model.holdout_f1_ == 1.0

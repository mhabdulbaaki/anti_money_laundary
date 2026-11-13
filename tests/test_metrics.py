from datetime import date

import numpy as np
import polars as pl
import pytest

from aml.accounts import account_day_scores
from aml.metrics import best_f1_threshold, f1_at, precision_at_k_by_day


def test_best_threshold_maximises_f1():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    thr, f1 = best_f1_threshold(y, s)
    assert thr == pytest.approx(0.35)
    assert f1 == pytest.approx(0.8)
    assert f1_at(y, s, thr) == pytest.approx(0.8)


def test_f1_is_zero_when_nothing_is_flagged():
    assert f1_at(np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]), 0.9) == 0.0


def test_precision_at_k_is_computed_separately_for_each_day():
    d1, d2 = date(2022, 9, 9), date(2022, 9, 10)
    df = pl.DataFrame(
        {
            "day": [d1, d1, d1, d2, d2, d2],
            "score": [0.9, 0.8, 0.1, 0.7, 0.6, 0.5],
            "label": [1, 0, 1, 1, 1, 0],
        }
    )
    out = precision_at_k_by_day(df, k=2)
    assert out.sort("day")["precision"].to_list() == [0.5, 1.0]
    assert out.sort("day")["positives_in_day"].to_list() == [2, 2]


def test_account_scores_take_the_riskiest_transfer_the_account_touched_that_day():
    d = date(2022, 9, 9)
    tx = pl.DataFrame(
        {
            "day": [d, d, d],
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "score": [0.2, 0.9, 0.1],
            "is_laundering": [0, 1, 0],
        }
    )
    acc = account_day_scores(tx).sort("account")
    assert acc["account"].to_list() == [0, 1, 2, 3]
    assert acc["score"].to_list() == pytest.approx([0.2, 0.9, 0.9, 0.1])
    assert acc["involved"].to_list() == [0, 1, 1, 0]
    assert acc["n_tx"].to_list() == [1, 2, 2, 1]


def test_self_transfers_count_once_for_the_account():
    d = date(2022, 9, 9)
    tx = pl.DataFrame({"day": [d], "src": [5], "dst": [5], "score": [0.3], "is_laundering": [0]})
    acc = account_day_scores(tx)
    assert acc.height == 1 and acc["n_tx"][0] == 1

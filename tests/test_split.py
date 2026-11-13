from datetime import datetime, timedelta

import numpy as np
import polars as pl

from aml.split import chronological_split, expanding_time_folds

T0 = datetime(2022, 9, 1)


def _ts(minutes):
    return pl.Series([T0 + timedelta(minutes=m) for m in minutes])


def test_split_is_60_20_20_by_time_and_never_shares_a_minute():
    ts = _ts(list(range(10)))
    assert chronological_split(ts).tolist() == ["train"] * 6 + ["val"] * 2 + ["test"] * 2


def test_rows_in_the_cut_minute_all_go_to_the_later_part():
    # Ten rows; the row at the 60% mark shares minute 5 with the row before it.
    ts = _ts([0, 1, 2, 3, 4, 5, 5, 7, 8, 9])
    labels = chronological_split(ts)
    assert labels.tolist() == ["train"] * 5 + ["val"] * 3 + ["test"] * 2


def test_split_does_not_depend_on_row_order():
    ts = _ts([9, 0, 8, 1, 7, 2, 6, 3, 5, 4])
    labels = chronological_split(ts)
    assert labels[1] == "train" and labels[0] == "test" and labels[6] == "val"


def test_expanding_folds_train_on_the_past_and_validate_on_the_next_block():
    ts = _ts(list(range(8)))
    folds = expanding_time_folds(ts, n_splits=3)
    assert [(tr.tolist(), va.tolist()) for tr, va in folds] == [
        ([0, 1], [2, 3]),
        ([0, 1, 2, 3], [4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7]),
    ]


def test_fold_indices_refer_to_original_positions_when_unsorted():
    ts = _ts([7, 6, 5, 4, 3, 2, 1, 0])
    tr, va = expanding_time_folds(ts, n_splits=3)[0]
    assert sorted(tr.tolist()) == [6, 7] and sorted(va.tolist()) == [4, 5]
    assert np.intersect1d(tr, va).size == 0

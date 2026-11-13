from datetime import datetime, timedelta

import polars as pl
import pytest

from aml.features import loop_features, rolling_account_features

T0 = datetime(2022, 9, 1)
A, B, C, D = 0, 1, 2, 3


def _tx(rows):
    """rows: (minutes after T0, src, dst, amount_usd)."""
    return pl.DataFrame(
        {
            "ts": [T0 + timedelta(minutes=m) for m, *_ in rows],
            "src": [r[1] for r in rows],
            "dst": [r[2] for r in rows],
            "amount_usd": [float(r[3]) for r in rows],
        }
    )


TX = _tx(
    [
        (0, A, B, 100),
        (10, A, C, 200),
        (10, A, B, 50),  # same minute as the row above: neither may see the other
        (30, B, A, 70),
        (2000, A, D, 10),  # more than a day later
    ]
)


def test_sender_outgoing_history_uses_only_strictly_earlier_transfers():
    f = rolling_account_features(TX, windows={"24h": 24 * 60})
    assert f["src_out_cnt_24h"].to_list() == [0, 1, 1, 0, 0]
    assert f["src_out_usd_24h"].to_list() == pytest.approx([0, 100, 100, 0, 0])
    assert f["src_out_uniq_24h"].to_list() == [0, 1, 1, 0, 0]


def test_incoming_history_of_sender_and_receiver():
    f = rolling_account_features(TX, windows={"24h": 24 * 60})
    # Row 3 (B pays A): B had received 100 + 50 from A earlier; A had paid out 350 to B and C.
    assert f["src_in_cnt_24h"][3] == 2
    assert f["src_in_usd_24h"][3] == pytest.approx(150)
    assert f["src_in_uniq_24h"][3] == 1
    assert f["dst_out_cnt_24h"][3] == 3
    assert f["dst_out_usd_24h"][3] == pytest.approx(350)
    assert f["dst_out_uniq_24h"][3] == 2
    # Row 1 (A pays C): C had received nothing before.
    assert f["dst_in_cnt_24h"][1] == 0


def test_window_length_controls_how_far_back_we_look():
    f = rolling_account_features(TX, windows={"1h": 60, "7d": 7 * 24 * 60})
    assert f["src_out_cnt_1h"][4] == 0
    assert f["src_out_cnt_7d"][4] == 3
    assert f["src_out_uniq_7d"][4] == 2
    assert f["src_in_usd_7d"][4] == pytest.approx(70)


def test_output_keeps_the_input_row_order_when_input_is_unsorted():
    shuffled = TX.reverse()
    f = rolling_account_features(shuffled, windows={"24h": 24 * 60})
    assert f["src_out_cnt_24h"].to_list() == [0, 0, 1, 1, 0]


def test_reciprocal_flag_is_set_only_if_the_receiver_paid_the_sender_before():
    tx = _tx([(0, A, B, 1), (5, B, A, 1), (5, A, B, 1), (9, A, B, 1)])
    f = loop_features(tx)
    # Row 2 happens in the same minute as B->A, so it cannot know about it yet.
    assert f["reciprocal_before"].to_list() == [0, 1, 0, 1]


def test_three_step_cycle_flag_closes_a_loop_back_to_the_sender():
    # A->B, then B->C, then C->A closes the loop A->B->C->A.
    tx = _tx([(0, A, B, 1), (1, B, C, 1), (2, C, A, 1), (3, D, A, 1)])
    f = loop_features(tx)
    assert f["cycle3_before"].to_list() == [0, 0, 1, 0]


def test_self_transfers_never_count_as_loops():
    tx = _tx([(0, A, A, 1), (1, A, A, 1)])
    f = loop_features(tx)
    assert f["reciprocal_before"].to_list() == [0, 0]
    assert f["cycle3_before"].to_list() == [0, 0]


def test_pair_seen_before_marks_repeat_payments_to_the_same_receiver():
    tx = _tx([(0, A, B, 1), (0, A, B, 1), (1, A, B, 1), (2, A, C, 1)])
    assert loop_features(tx)["pair_seen_before"].to_list() == [0, 0, 1, 0]


def test_two_step_paths_count_intermediaries_already_linking_sender_to_receiver():
    # A->B, B->D and A->C, C->D happened earlier, so A->D has two parallel two-step paths.
    tx = _tx([(0, A, B, 1), (0, A, C, 1), (1, B, D, 1), (1, C, D, 1), (2, A, D, 1), (2, B, D, 1)])
    f = loop_features(tx)
    assert f["paths2_before"].to_list() == [0, 0, 0, 0, 2, 0]


def test_minutes_since_sender_last_received_money():
    # B receives at minute 0 and 50; B pays out at 50 (same minute: must not see it) and at 80.
    tx = _tx([(0, A, B, 1), (50, C, B, 1), (50, B, D, 1), (80, B, D, 1), (90, D, A, 1)])
    f = loop_features(tx)
    assert f["mins_since_src_in"].to_list()[2:4] == [50, 30]
    # A never received anything before minute 0: the value is capped at one week.
    assert f["mins_since_src_in"][0] == 7 * 24 * 60
    # D last received at minute 80 (from B), so D paying at 90 is 10 minutes later.
    assert f["mins_since_src_in"][4] == 90 - 80

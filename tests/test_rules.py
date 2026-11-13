import polars as pl

from aml.rules import red_flags


def _f(**cols):
    base = {
        "src_out_uniq_24h": [0],
        "dst_in_uniq_24h": [0],
        "src_in_usd_24h": [0.0],
        "amount_usd": [100.0],
        "just_below_limit": [0],
    }
    base.update({k: [v] for k, v in cols.items()})
    return pl.DataFrame(base)


def test_fan_out_fires_when_the_sender_already_paid_many_different_accounts_in_the_last_day():
    assert red_flags(_f(src_out_uniq_24h=5), fan_k=5)["fan_out"][0] == 1
    assert red_flags(_f(src_out_uniq_24h=4), fan_k=5)["fan_out"][0] == 0


def test_fan_in_fires_when_the_receiver_was_already_paid_by_many_different_accounts():
    assert red_flags(_f(dst_in_uniq_24h=5), fan_k=5)["fan_in"][0] == 1
    assert red_flags(_f(dst_in_uniq_24h=4), fan_k=5)["fan_in"][0] == 0


def test_pass_through_fires_when_money_out_roughly_equals_money_just_received():
    assert red_flags(_f(src_in_usd_24h=1000.0, amount_usd=950.0))["pass_through"][0] == 1
    assert red_flags(_f(src_in_usd_24h=1000.0, amount_usd=300.0))["pass_through"][0] == 0
    assert red_flags(_f(src_in_usd_24h=0.0, amount_usd=300.0))["pass_through"][0] == 0


def test_structuring_fires_just_below_the_reporting_limit_and_rule_score_counts_hits():
    out = red_flags(_f(just_below_limit=1, src_out_uniq_24h=10), fan_k=5)
    assert out["structuring"][0] == 1
    assert out["rule_score"][0] == 2

import polars as pl
import pytest

from aml.fx import add_amount_usd, estimate_usd_rates, max_conversion_error


def _tx(rows):
    return pl.DataFrame(rows, schema=["amt_paid", "cur_paid", "amt_recv", "cur_recv"], orient="row")


def test_rates_come_from_cross_currency_rows_in_both_directions():
    tx = _tx(
        [
            (1000.0, "Euro", 1171.78, "US Dollar"),  # 1 EUR = 1.17178 USD
            (2000.0, "US Dollar", 1706.80, "Euro"),  # 1 EUR = 2000/1706.8 = 1.17178 USD
            (1000.0, "US Dollar", 6697.6, "Yuan"),  # 1 CNY = 0.149308 USD
            (500.0, "US Dollar", 500.0, "US Dollar"),
        ]
    )
    rates = estimate_usd_rates(tx, min_usd=100)
    assert rates["US Dollar"] == 1.0
    assert rates["Euro"] == pytest.approx(1.17178, rel=1e-4)
    assert rates["Yuan"] == pytest.approx(0.149308, rel=1e-4)


def test_tiny_amounts_are_ignored_because_rounding_distorts_their_ratio():
    tx = _tx(
        [
            (1000.0, "Euro", 1171.78, "US Dollar"),
            (0.03, "US Dollar", 0.02, "Euro"),  # ratio 1.5, a rounding artefact
            (0.01, "US Dollar", 0.01, "Euro"),  # ratio 1.0, a rounding artefact
        ]
    )
    assert estimate_usd_rates(tx, min_usd=100)["Euro"] == pytest.approx(1.17178, rel=1e-4)


def test_currency_without_any_usd_pair_raises():
    tx = _tx([(1000.0, "Euro", 1171.78, "US Dollar"), (500.0, "Yen", 500.0, "Yen")])
    with pytest.raises(ValueError, match="Yen"):
        estimate_usd_rates(tx, min_usd=100)


def test_amount_usd_uses_the_paid_side():
    tx = _tx([(1000.0, "Euro", 1171.78, "US Dollar"), (10.0, "Euro", 10.0, "Euro")])
    out = add_amount_usd(tx, {"US Dollar": 1.0, "Euro": 1.17178})
    assert out["amount_usd"].to_list() == pytest.approx([1171.78, 11.7178])


def test_conversion_error_is_small_when_both_sides_agree():
    tx = _tx([(1000.0, "Euro", 1171.78, "US Dollar"), (2000.0, "US Dollar", 1706.80, "Euro")])
    rates = {"US Dollar": 1.0, "Euro": 1.17178}
    assert max_conversion_error(tx, rates, min_usd=100) < 1e-4

import io

import polars as pl

from src.data import attach_typology, parse_patterns, read_accounts, read_transactions

TRANS_CSV = """Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,010,8000EBD30,010,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
2022/09/01 00:03,01467,8013C4030,020,80BC62F10,58702.10,Yuan,58702.10,Yuan,ACH,1
2022/09/02 00:03,0071,80BC62F10,0071,80BC62F10,1.00,Yuan,1.00,Yuan,ACH,0
"""

PATTERNS = """BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops
2022/09/01 00:03,01467,8013C4030,020,80BC62F10,58702.10,Yuan,58702.10,Yuan,ACH,1
END LAUNDERING ATTEMPT - CYCLE
"""


def test_read_transactions_sorts_by_time_drops_exact_duplicates_and_keeps_bank_codes():
    tx = read_transactions(io.StringIO(TRANS_CSV))
    assert tx.height == 4
    assert tx["ts"].is_sorted()
    assert tx["from_bank"][0] == "01467"  # leading zero kept
    assert tx["is_laundering"].to_list() == [1, 1, 0, 0]


def test_accounts_are_identified_by_bank_and_account_number():
    tx = read_transactions(io.StringIO(TRANS_CSV))
    # Account 80BC62F10 at bank 020 and at bank 0071 are different accounts.
    assert tx["dst_key"][0] == "020_80BC62F10"
    assert tx["src_key"][3] == "0071_80BC62F10"


def test_patterns_are_parsed_into_attempts_with_their_typology():
    p = parse_patterns(io.StringIO(PATTERNS))
    assert p["typology"].to_list() == ["FAN-OUT", "CYCLE"]
    assert p["attempt_id"].to_list() == [0, 1]
    assert p["detail"][0] == "Max 16-degree Fan-Out"


def test_typology_is_attached_to_matching_transactions_only():
    tx = read_transactions(io.StringIO(TRANS_CSV))
    out = attach_typology(tx, parse_patterns(io.StringIO(PATTERNS)))
    assert out.height == tx.height
    assert out["typology"].to_list() == ["CYCLE", "FAN-OUT", None, None]


ACCOUNTS_CSV = """Bank Name,Bank ID,Account Number,Entity ID,Entity Name
Canada Bank #27,20,80BC62F10,800C998A0,Corporation #33520
UK Bank #33,71,80BC62F10,800C47F50,Partnership #35397
"""


def test_owner_type_is_looked_up_by_bank_and_account_not_account_alone():
    from src.data import owner_types

    accounts = read_accounts(io.StringIO(ACCOUNTS_CSV))
    keys = pl.Series(["020_80BC62F10", "0071_80BC62F10", "0099_UNKNOWN"])
    assert owner_types(keys, accounts).to_list() == ["Corporation", "Partnership", None]

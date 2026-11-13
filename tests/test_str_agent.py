from aml.str_agent import REQUIRED_SECTIONS, draft_report_template, validate_report

EVIDENCE = {
    "case_id": "CASE-0007",
    "review_date": "2022-09-09",
    "account": {"account_id": "0071_80BC62F10", "owner_type": "Corporation"},
    "risk": {"score": 0.97, "rank_today": 3},
    "flagged_transfers": [
        {"time": "2022-09-09 10:03", "direction": "sent", "amount_usd": 12345.67, "payment_format": "ACH"}
    ],
    "reasons": ["the sender had paid 6 different accounts in the previous 24 hours"],
}


def _report(body_numbers="US$12,345.67 on 2022-09-09 10:03"):
    return "\n".join(
        [
            "# DRAFT Suspicious Transaction Report - for analyst review",
            "## 1. Subject",
            "Account 0071_80BC62F10 (Corporation), case CASE-0007.",
            "## 2. Transactions of concern",
            f"One ACH payment of {body_numbers}.",
            "## 3. Grounds for suspicion",
            "The sender had paid 6 different accounts in the previous 24 hours.",
            "## 4. Model explanation",
            "Risk score 0.97, ranked 3 today.",
            "## 5. Recommended action",
            "Analyst to review the account.",
        ]
    )


def test_a_well_formed_grounded_report_passes():
    v = validate_report(_report(), EVIDENCE)
    assert v.format_ok and v.numbers_ok
    assert v.unsupported_numbers == []


def test_missing_section_fails_the_format_check():
    text = _report().replace("## 4. Model explanation\n", "")
    v = validate_report(text, EVIDENCE)
    assert not v.format_ok
    assert "## 4. Model explanation" in v.missing_sections


def test_sections_out_of_order_fail_the_format_check():
    text = _report()
    a, b = REQUIRED_SECTIONS[1], REQUIRED_SECTIONS[2]
    swapped = text.replace(a, "@@").replace(b, a).replace("@@", b)
    assert not validate_report(swapped, EVIDENCE).format_ok


def test_an_empty_section_fails_the_format_check():
    text = _report().replace("Analyst to review the account.", "")
    assert not validate_report(text, EVIDENCE).format_ok


def test_a_number_not_in_the_evidence_is_flagged():
    v = validate_report(_report("US$12,345.67, about 40% of the account's inflow"), EVIDENCE)
    assert not v.numbers_ok
    assert v.unsupported_numbers == ["40"]


def test_rounded_quotes_of_evidence_numbers_are_accepted():
    v = validate_report(_report("about US$12,346 on 2022-09-09 10:03"), EVIDENCE)
    assert v.numbers_ok


def test_the_template_drafter_always_passes_its_own_checks():
    v = validate_report(draft_report_template(EVIDENCE), EVIDENCE)
    assert v.format_ok and v.numbers_ok


def test_reasons_are_described_in_plain_words_for_every_kind_of_feature():
    from aml.str_agent import describe_reason

    row = {"fmt": "ACH", "amount_usd": 950.0, "src_out_uniq_24h": 6, "mins_since_src_in": 12, "cycle3_before": 1}
    assert describe_reason("fmt", row) == "it was paid by ACH"
    assert describe_reason("src_out_uniq_24h", row) == "the sender had paid 6 different accounts in the previous 24 hours"
    assert describe_reason("mins_since_src_in", row) == "the sender last received money 12 minutes before this payment"
    assert describe_reason("cycle3_before", row) == "this payment closes a three-step loop back to the sender"


def test_digits_inside_identifiers_do_not_count_as_evidence_numbers():
    # "62" only appears inside the account id, so a claim of 62 transfers is unsupported.
    text = _report().replace("Analyst to review the account.", "Analyst to review all 62 transfers.")
    v = validate_report(text, EVIDENCE)
    assert v.unsupported_numbers == ["62"]


def test_small_numbers_cannot_be_justified_by_rounding():
    # The risk score 0.97 must not make a bare "1" look supported.
    text = _report().replace("Analyst to review the account.", "Analyst to review 1 account.")
    assert validate_report(text, EVIDENCE).unsupported_numbers == ["1"]

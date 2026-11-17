from types import SimpleNamespace

from ai_agent.checks import REQUIRED_SECTIONS, validate_report
from ai_agent.drafting import (
    CLAUDE_MODEL,
    GEMINI_MODEL,
    SYSTEM_PROMPT,
    draft_report_claude,
    draft_report_gemini,
    draft_report_template,
    model_display_name,
    pick_drafter,
)

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
    from ai_agent.evidence import describe_reason

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


# ---------------------------------------------------------------- Claude call (mocked, no network)
class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.beta = SimpleNamespace(messages=_FakeMessages(response))


def _response(stop_reason="end_turn", text="draft"):
    return SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=900, output_tokens=300),
        content=[SimpleNamespace(type="thinking", text="ignored"), SimpleNamespace(type="text", text=text)]
        if stop_reason != "refusal"
        else [],
    )


def test_llm_draft_sends_evidence_with_fallbacks_and_returns_text():
    client = _FakeClient(_response(text=_report()))
    text, meta = draft_report_claude(EVIDENCE, client=client)
    sent = client.beta.messages.kwargs
    assert sent["model"] == CLAUDE_MODEL
    assert sent["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in sent["betas"]
    assert sent["system"][0]["text"] == SYSTEM_PROMPT
    assert "CASE-0007" in sent["messages"][0]["content"]
    assert text == _report() and validate_report(text, EVIDENCE).ok
    assert meta["stop_reason"] == "end_turn" and meta["output_tokens"] == 300


def test_llm_draft_refusal_returns_empty_text():
    text, meta = draft_report_claude(EVIDENCE, client=_FakeClient(_response(stop_reason="refusal")))
    assert text == "" and meta["stop_reason"] == "refusal"


# ---------------------------------------------------------------- Gemini call (mocked, no network)
class _FakeInteractions:
    def __init__(self, interaction):
        self.interaction = interaction
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.interaction


def _interaction(status="completed", text="draft"):
    return SimpleNamespace(
        id="int-1",
        model="gemini-3.8-flash",
        status=status,
        output_text=text,
        usage=SimpleNamespace(total_input_tokens=800, total_output_tokens=250),
    )


def test_gemini_draft_sends_same_prompt_and_evidence():
    client = SimpleNamespace(interactions=_FakeInteractions(_interaction(text=_report())))
    text, meta = draft_report_gemini(EVIDENCE, client=client)
    sent = client.interactions.kwargs
    assert sent["model"] == GEMINI_MODEL
    assert sent["system_instruction"] == SYSTEM_PROMPT
    assert "CASE-0007" in sent["input"]
    assert text == _report() and validate_report(text, EVIDENCE).ok
    assert meta == {
        "model": "gemini-3.8-flash",
        "stop_reason": "completed",
        "input_tokens": 800,
        "output_tokens": 250,
        "request_id": "int-1",
    }


def test_gemini_draft_unfinished_returns_empty_text():
    client = SimpleNamespace(interactions=_FakeInteractions(_interaction(status="failed", text="partial")))
    text, meta = draft_report_gemini(EVIDENCE, client=client)
    assert text == "" and meta["stop_reason"] == "failed"


def test_model_display_names():
    assert model_display_name("claude-opus-5-5") == "Claude Opus 5.5"
    assert model_display_name("gemini-3.8-flash") == "Gemini 3.8 Flash"
    assert model_display_name("gemini-4-hydrogen") == "Gemini 4 Hydrogen"


def test_pick_drafter_from_environment():
    assert pick_drafter({}) == "template"
    assert pick_drafter({"ANTHROPIC_API_KEY": "k"}) == "claude"
    assert pick_drafter({"GEMINI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}) == "gemini"
    assert pick_drafter({"GOOGLE_API_KEY": "k"}) == "gemini"
    assert pick_drafter({"GEMINI_API_KEY": ""}) == "template"

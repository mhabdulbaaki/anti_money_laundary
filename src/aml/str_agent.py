"""AI-drafted suspicious transaction reports (STRs), checked automatically.

The pipeline computes every fact first (the *evidence*), the language model only turns
those facts into a FINTRAC-style narrative, and ``validate_report`` then checks that the
draft (1) follows the required format and (2) uses only numbers found in the evidence.
Drafts are advisory: an analyst decides whether anything is filed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .features import MINUTES_CAP

DEFAULT_MODEL = "claude-opus-5-5"
GEMINI_MODEL = "gemini-3.8-flash"  # free tier on the Gemini API (Google AI Studio), Nov 2026
TITLE = "# DRAFT Suspicious Transaction Report - for analyst review"
REQUIRED_SECTIONS = [
    "## 1. Subject",
    "## 2. Transactions of concern",
    "## 3. Grounds for suspicion",
    "## 4. Model explanation",
    "## 5. Recommended action",
]

SYSTEM_PROMPT = f"""You draft suspicious transaction report (STR) narratives for an anti-money-laundering \
analyst at a Canadian bank, in the style of the "description of suspicious activity" part of a FINTRAC STR. \
The analyst, not you, decides whether anything is filed; your draft saves them writing time.

Rules:
- Use only facts in the evidence JSON. Do not invent names, places, motives, links or history.
- Every number you write must appear in the evidence as given (you may round a money amount to whole \
dollars). Do not add, subtract, average, or turn anything into a percentage or a new total.
- Write plain, neutral, factual English for a compliance reader. Say what the facts "may indicate", \
never that a crime happened.
- Output exactly this template: the title line, then the five headings in this order, each followed by \
one to four sentences or short bullet points. No other headings, no preamble, no closing remarks.

{TITLE}
{chr(10).join(REQUIRED_SECTIONS)}"""

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class Validation:
    format_ok: bool
    numbers_ok: bool
    missing_sections: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.format_ok and self.numbers_ok


def _numbers(text: str) -> list[str]:
    return [m.group(0).rstrip(",") for m in _NUMBER.finditer(text)]


# Evidence fields that are identifiers: their digits are labels, not numbers a report may quote.
ID_KEYS = frozenset({"case_id", "account_id", "counterparty", "bank_code"})
# Only money-sized values may be quoted rounded (US$12,345.67 -> US$12,346); a score of 0.97
# must not make a bare "1" look supported.
ROUNDING_MIN = 100.0


def _evidence_numbers(evidence: Any) -> tuple[list[float], list[str]]:
    """Numbers stated in the evidence, and the identifier strings whose digits do not count."""
    values: list[float] = []
    ids: list[str] = []

    def walk(x: Any, key: str | None) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, bool) or x is None:
            return
        elif isinstance(x, (int, float)):
            values.append(float(x))
        elif isinstance(x, str):
            if key in ID_KEYS:
                ids.append(x)
            else:
                values.extend(float(t.replace(",", "")) for t in _numbers(x))

    walk(evidence, None)
    return values, ids


def _supported(token: str, evidence_values: list[float]) -> bool:
    value = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    return any(
        abs(e - value) < 1e-9 or (abs(e) >= ROUNDING_MIN and abs(round(e, decimals) - value) < 1e-9)
        for e in evidence_values
    )


def validate_report(text: str, evidence: dict[str, Any]) -> Validation:
    """Format: the five headings, each exactly once, in order, each with some content.
    Grounding: every number outside heading lines equals a number stated in the evidence (money
    amounts may be rounded to the decimals written); digits inside identifiers do not count."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    stripped = [ln.strip() for ln in lines]
    missing = [s for s in REQUIRED_SECTIONS if stripped.count(s) != 1]
    format_ok = not missing
    if format_ok:
        pos = [stripped.index(s) for s in REQUIRED_SECTIONS]
        ends = pos[1:] + [len(stripped)]
        in_order = pos == sorted(pos)
        filled = all(any(stripped[i] for i in range(a + 1, b)) for a, b in zip(pos, ends)) if in_order else False
        extra_headings = any(ln.startswith("#") and ln not in REQUIRED_SECTIONS and ln != TITLE for ln in stripped)
        format_ok = in_order and filled and not extra_headings

    evidence_values, ids = _evidence_numbers(evidence)
    body = "\n".join(ln for ln in stripped if not ln.startswith("#"))
    for ident in sorted(ids, key=len, reverse=True):
        body = body.replace(ident, " ")
    unsupported = [t for t in _numbers(body) if not _supported(t, evidence_values)]
    return Validation(format_ok, not unsupported, missing, unsupported)


# ---------------------------------------------------------------- plain-English reasons
_WINDOWS = {"1h": "previous hour", "24h": "previous 24 hours", "7d": "previous 7 days"}
_ROLE = {"src": "the sender", "dst": "the receiver"}


def describe_reason(feature: str, row: dict[str, Any]) -> str:
    """Turn one model input (and this transfer's value for it) into a short factual phrase."""
    v: Any = row.get(feature)
    m = re.match(r"^(src|dst)_(out|in)_(cnt|usd|uniq)_(1h|24h|7d)$", feature)
    if m:
        who, direction, stat, w = _ROLE[m[1]], m[2], m[3], _WINDOWS[m[4]]
        if stat == "cnt":
            return f"{who} had {'made' if direction == 'out' else 'received'} {int(v)} payments in the {w}"
        if stat == "uniq":
            verb = "paid" if direction == "out" else "been paid by"
            return f"{who} had {verb} {int(v)} different accounts in the {w}"
        verb = "paid out" if direction == "out" else "received"
        return f"{who} had {verb} US${float(v):,.2f} in the {w}"
    amount = float(row.get("amount_usd", 0.0))

    def flag(yes: str, no: str) -> Callable[[], str]:
        return lambda: yes if v else no

    # Lambdas, so only the requested feature's value is formatted (values differ in type).
    simple: dict[str, Callable[[], str]] = {
        "log_amount_usd": lambda: f"the amount was US${amount:,.2f}",
        "hour": lambda: f"it was made in hour {row.get('hour')} of the day",
        "fmt": lambda: f"it was paid by {v}",
        "cur_paid": lambda: f"it was paid in {v}",
        "cur_recv": lambda: f"it was received in {v}",
        "self_transfer": flag("it moved money from the account to itself", "it went to a different account"),
        "same_bank": flag("both accounts are at the same bank", "the two accounts are at different banks"),
        "cross_currency": flag("it changed currency", "it did not change currency"),
        "just_below_limit": flag(
            "the amount is just below the US$10,000 reporting limit",
            "the amount is not just below the US$10,000 reporting limit",
        ),
        "round_hundred": flag("the amount is a round multiple of US$100", "the amount is not a round number"),
        "pair_seen_before": flag("the sender had paid this receiver before", "the sender had never paid this receiver before"),
        "reciprocal_before": flag(
            "the receiver had earlier paid the sender (money going back and forth)",
            "the receiver had never paid the sender",
        ),
        "cycle3_before": flag(
            "this payment closes a three-step loop back to the sender", "this payment does not close a three-step loop"
        ),
        "paths2_before": lambda: f"{int(v or 0)} accounts already linked the sender to the receiver in two steps",
        "mins_since_src_in": lambda: "the sender had not received money in the previous 7 days"
        if int(v or 0) >= MINUTES_CAP
        else f"the sender last received money {int(v or 0)} minutes before this payment",
        "src_in_out_ratio_24h": lambda: f"what the sender received in the previous 24 hours was {float(v or 0):.2f} times what it paid out",
        "src_in_out_ratio_7d": lambda: f"what the sender received in the previous 7 days was {float(v or 0):.2f} times what it paid out",
        "amount_vs_src_in_24h": lambda: f"this payment was {float(v or 0):.2f} times what the sender received in the previous 24 hours",
        "amount_vs_src_avg_out_7d": lambda: f"this payment was {float(v or 0):.2f} times the sender's average payment in the previous 7 days",
    }
    describe = simple.get(feature)
    return describe() if describe else f"{feature} = {v}"


# ---------------------------------------------------------------- drafting
def draft_report_template(evidence: dict[str, Any]) -> str:
    """Deterministic fill-in-the-blanks draft (no language model). Used as a format baseline
    and when no API credentials are available."""
    acct = evidence.get("account", {})
    risk = evidence.get("risk", {})
    transfers = evidence.get("flagged_transfers", [])
    reasons = evidence.get("reasons", [])
    rules = evidence.get("red_flag_rules", [])
    lines = [
        TITLE,
        REQUIRED_SECTIONS[0],
        f"Account {acct.get('account_id')} ({acct.get('owner_type', 'owner type unknown')}), "
        f"case {evidence.get('case_id')}, review date {evidence.get('review_date')}.",
        REQUIRED_SECTIONS[1],
    ]
    for t in transfers:
        lines.append(
            f"- {t.get('time')}: {t.get('direction')} US${t.get('amount_usd'):,.2f} by {t.get('payment_format')}"
            + (f" (counterparty {t['counterparty']})" if t.get("counterparty") else "")
        )
    lines += [REQUIRED_SECTIONS[2]]
    lines += [f"- {r[0].upper()}{r[1:]}." for r in reasons] or ["- No single indicator stands out."]
    if rules:
        lines.append(f"- Red-flag rules triggered: {', '.join(rules)}.")
    lines += [
        REQUIRED_SECTIONS[3],
        f"The model gave this account a risk score of {risk.get('score')}, rank {risk.get('rank_today')} of the day's alerts.",
        REQUIRED_SECTIONS[4],
        "An analyst should review the account's recent activity and decide whether a report to FINTRAC is warranted.",
    ]
    return "\n".join(lines)


def model_display_name(model_id: str) -> str:
    """'claude-opus-5-5' -> 'Claude Opus 5.5', 'gemini-3.8-flash' -> 'Gemini 3.8 Flash'."""
    vendor, *rest = model_id.split("-")
    words: list[str] = []
    for part in rest:
        if part.isdigit() and words and words[-1].replace(".", "").isdigit():
            words[-1] += f".{part}"  # version written with dashes: 5-5 -> 5.5
        else:
            words.append(part if part[0].isdigit() else part.capitalize())
    return " ".join([vendor.capitalize(), *words])


def pick_drafter(env: Mapping[str, str]) -> str:
    """Which drafter the environment's API keys allow: Gemini first, then Claude, else the template."""
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        return "gemini"
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    return "template"


def _user_message(evidence: dict[str, Any]) -> str:
    return "Evidence for one alert:\n```json\n" + json.dumps(evidence, indent=2) + "\n```\nDraft the report."


def draft_report_llm(evidence: dict[str, Any], client: Any = None, model: str = DEFAULT_MODEL) -> tuple[str, dict[str, Any]]:
    """Ask Claude for a draft. Returns (text, metadata). Text is empty if the model declined."""
    import anthropic

    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on a fallback model if a safety classifier declines
        output_config={"effort": "medium"},
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _user_message(evidence)}],
    )
    meta = {
        "model": response.model,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "request_id": getattr(response, "_request_id", None),
    }
    if response.stop_reason == "refusal":
        return "", meta
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, meta


def draft_report_gemini(evidence: dict[str, Any], client: Any = None, model: str = GEMINI_MODEL) -> tuple[str, dict[str, Any]]:
    """Ask Gemini (Interactions API, free tier) for a draft, with the same prompt and checks as Claude.
    Returns (text, metadata). Text is empty unless the interaction completed."""
    if client is None:
        from google import genai
        from google.genai import types

        # The key comes from GEMINI_API_KEY (or GOOGLE_API_KEY). Free-tier rate limits are low,
        # so rate-limit (429) and server errors are retried with backoff.
        retry = types.HttpRetryOptions(attempts=6, initial_delay=10.0, max_delay=120.0)
        client = genai.Client(http_options=types.HttpOptions(retry_options=retry))
    interaction = client.interactions.create(model=model, system_instruction=SYSTEM_PROMPT, input=_user_message(evidence))
    usage = interaction.usage
    meta = {
        "model": str(interaction.model or model),
        "stop_reason": str(interaction.status),
        "input_tokens": int(getattr(usage, "total_input_tokens", None) or 0),
        "output_tokens": int(getattr(usage, "total_output_tokens", None) or 0),
        "request_id": interaction.id,
    }
    if interaction.status != "completed":
        return "", meta
    return interaction.output_text or "", meta

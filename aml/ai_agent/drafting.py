"""Draft a FINTRAC-style suspicious transaction report (STR) from one evidence packet.

Three drafters share one format: Gemini (free tier), Claude, or a deterministic template when no API
key is set. The language model only turns computed facts into a narrative; ai_agent.checks then
tests every draft. Drafts are advisory: an analyst decides whether anything is filed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .checks import REQUIRED_SECTIONS, TITLE

CLAUDE_MODEL = "claude-opus-5-5"
GEMINI_MODEL = "gemini-3.8-flash"  # free tier on the Gemini API (Google AI Studio), Nov 2026

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


def draft_report_claude(evidence: dict[str, Any], client: Any = None, model: str = CLAUDE_MODEL) -> tuple[str, dict[str, Any]]:
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

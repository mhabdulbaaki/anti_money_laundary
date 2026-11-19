"""Automatic checks on every draft report, whoever wrote it (Gemini, Claude or the template).

1. Format: the title, then the five FINTRAC-style headings, each exactly once, in order, each with content.
2. Grounding: every number in the text must be stated in the evidence (money amounts may be rounded);
   a number the model worked out itself (a total, a time gap, a count) fails the check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TITLE = "# DRAFT Suspicious Transaction Report - for analyst review"
REQUIRED_SECTIONS = [
    "## 1. Subject",
    "## 2. Transactions of concern",
    "## 3. Grounds for suspicion",
    "## 4. Model explanation",
    "## 5. Recommended action",
]

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

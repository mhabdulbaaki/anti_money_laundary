"""Draft and check FINTRAC-style reports for each test day's top accounts (`make evaluate`, last step).

Run: python -m ai_agent.draft_reports [--drafter auto|gemini|claude|template] [--model ID] [--per-day 10]
  auto (default): Gemini if GEMINI_API_KEY (or GOOGLE_API_KEY) is set, else Claude if
  ANTHROPIC_API_KEY is set, else the deterministic template. Keys are read from ai_agent/.env
  (copy .env.example); a key already set in the shell wins.
Needs `python -m src.evaluate` first. Writes, per drafter, outputs/fintrac_reports/<drafter>/:
<case>.json (the evidence packet), <case>.md (the draft) and summary.json (the checks). Each drafter
keeps the evidence it was given, so its drafts can always be re-checked against it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from src import config

from .checks import validate_report
from .drafting import (
    CLAUDE_MODEL,
    GEMINI_MODEL,
    draft_report_claude,
    draft_report_gemini,
    draft_report_template,
    pick_drafter,
)
from .evidence import build_cases

ENV_FILE = Path(__file__).with_name(".env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", choices=["auto", "gemini", "claude", "template"], default="auto")
    ap.add_argument("--model", help=f"model id (default: {GEMINI_MODEL} for Gemini, {CLAUDE_MODEL} for Claude)")
    ap.add_argument("--per-day", type=int, default=10, help="accounts per test day to draft for (top of the review list)")
    args = ap.parse_args()
    load_dotenv(ENV_FILE)  # does not override keys already set in the shell
    drafter = args.drafter if args.drafter != "auto" else pick_drafter(os.environ)
    model = args.model or {"gemini": GEMINI_MODEL, "claude": CLAUDE_MODEL}.get(drafter)
    print(f"drafter: {drafter}" + (f" ({model})" if model else " (no API key found in ai_agent/.env or the shell)"))

    cases = build_cases(args.per_day)
    out_dir = config.FINTRAC_DIR / drafter
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for c in cases:
        ev = c["evidence"]
        (out_dir / f"{ev['case_id']}.json").write_text(json.dumps(ev, indent=2))
        t = time.time()
        meta: dict[str, Any] = {}
        if drafter == "template":
            text = draft_report_template(ev)
        else:
            assert model is not None
            ask = draft_report_gemini if drafter == "gemini" else draft_report_claude
            text, meta = ask(ev, model=model)
            usage["input_tokens"] += meta.get("input_tokens", 0)
            usage["output_tokens"] += meta.get("output_tokens", 0)
        v = validate_report(text, ev)
        (out_dir / f"{ev['case_id']}.md").write_text(text)
        rows.append(
            {
                "case_id": ev["case_id"],
                "format_ok": v.format_ok,
                "numbers_ok": v.numbers_ok,
                "missing_sections": v.missing_sections,
                "unsupported_numbers": v.unsupported_numbers,
                "truly_involved": c["truly_involved"],
                "seconds": round(time.time() - t, 2),
                **{k: meta[k] for k in ("model", "stop_reason") if k in meta},
            }
        )
        print(ev["case_id"], "format", v.format_ok, "numbers", v.numbers_ok, v.unsupported_numbers)
    summary = {
        "drafter": drafter,
        "model": model,
        "reports": len(rows),
        "format_ok_rate": float(np.mean([r["format_ok"] for r in rows])),
        "numbers_ok_rate": float(np.mean([r["numbers_ok"] for r in rows])),
        "both_ok_rate": float(np.mean([r["format_ok"] and r["numbers_ok"] for r in rows])),
        "cases_truly_involved": int(sum(r["truly_involved"] for r in rows)),
        "usage": usage,
        "cases": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()

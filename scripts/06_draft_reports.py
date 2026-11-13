"""Step 6: evidence packets for the top accounts of each test day, AI-drafted STRs, and the checks.

Usage: python scripts/06_draft_reports.py [--drafter auto|gemini|claude|template] [--model ID] [--per-day 10]
  auto (default): Gemini if GEMINI_API_KEY / GOOGLE_API_KEY is set, else Claude if ANTHROPIC_API_KEY /
  ANTHROPIC_AUTH_TOKEN is set, else the deterministic template.
Writes results/reports/<drafter>/<case>.md, results/reports/evidence/<case>.json and
results/reports_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import joblib
import numpy as np
import polars as pl

from aml import config
from aml.data import owner_types, read_accounts
from aml.rules import red_flags
from aml.str_agent import (
    DEFAULT_MODEL,
    GEMINI_MODEL,
    describe_reason,
    draft_report_gemini,
    draft_report_llm,
    draft_report_template,
    pick_drafter,
    validate_report,
)

REVIEW_LIST_SIZE = 100  # the daily account review list (step 5); we draft for the top of it


def rule_text(fan_k: int) -> dict[str, str]:
    return {
        "fan_out": f"fan-out (had already paid {fan_k} or more different accounts in the previous 24 hours)",
        "fan_in": f"fan-in (had already been paid by {fan_k} or more different accounts in the previous 24 hours)",
        "pass_through": "pass-through (paid on about what it had received in the previous 24 hours)",
        "structuring": "structuring (amount just below the US$10,000 reporting limit)",
    }


def build_cases(per_day: int) -> list[dict[str, Any]]:
    bundle = joblib.load(config.ARTIFACTS / "final_model.joblib")
    thr = float(bundle["threshold"])
    fan_k = int(json.loads((config.RESULTS / "final_metrics.json").read_text())["rules"]["tuned_on_val"]["fan_k"])
    tx = pl.read_parquet(config.ARTIFACTS / "scores_test.parquet").filter(~pl.col("tail_day"))
    reasons = pl.read_parquet(config.ARTIFACTS / "test_reasons.parquet").select("row", "reasons")
    acc = pl.read_parquet(config.ARTIFACTS / "account_scores_test.parquet").filter(
        pl.col("day") <= pl.lit(config.LAST_CORE_DAY).str.to_date()
    )
    keys = pl.read_parquet(config.ACCOUNT_KEYS)["key"]
    owner_of = dict(zip(keys.to_list(), owner_types(keys, read_accounts(config.ACCOUNTS_CSV)).to_list()))
    keys_list = keys.to_list()
    tx = tx.join(reasons, on="row", how="left")
    tx = pl.concat([tx, red_flags(tx, fan_k=fan_k).drop("rule_score")], how="horizontal")
    rules = rule_text(fan_k)

    cases = []
    for day in sorted(acc["day"].unique().to_list()):
        top = acc.filter(pl.col("day") == day).sort("score", descending=True).head(per_day)
        for rank, row in enumerate(top.iter_rows(named=True), start=1):
            a = int(row["account"])
            key = keys_list[a]
            mine = tx.filter((pl.col("day") == day) & ((pl.col("src") == a) | (pl.col("dst") == a)))
            sent = mine.filter(pl.col("src") == a)
            recv = mine.filter((pl.col("dst") == a) & (pl.col("src") != a))
            counterparties = set(sent["dst"].to_list()) | set(recv["src"].to_list())
            counterparties.discard(a)
            flagged = mine.filter(pl.col("score") >= thr).sort("score", descending=True)
            transfers = []
            for t in flagged.head(5).iter_rows(named=True):
                out = t["src"] == a
                transfers.append(
                    {
                        "time": t["ts"].strftime("%Y-%m-%d %H:%M"),
                        "direction": "sent" if out else "received",
                        "counterparty": keys_list[t["dst"] if out else t["src"]],
                        "amount_usd": round(float(t["amount_usd"]), 2),
                        "currency": t["cur_paid"],
                        "payment_format": t["fmt"],
                        "transfer_score": round(float(t["score"]), 3),
                    }
                )
            lead = (flagged if flagged.height else mine.sort("score", descending=True)).row(0, named=True)
            lead_reasons = [describe_reason(f, lead) for f in (lead.get("reasons") or [])]
            rules_hit = [text for r, text in rules.items() if int(mine[r].sum()) > 0]
            case_id = f"STR-{day.strftime('%Y%m%d')}-{rank:03d}"
            evidence = {
                "case_id": case_id,
                "review_date": str(day),
                "account": {
                    "account_id": key,
                    "bank_code": key.split("_")[0],
                    "owner_type": owner_of.get(key) or "unknown",
                },
                "risk": {
                    "score": round(float(row["score"]), 3),
                    "rank_today": rank,
                    "daily_review_list_size": REVIEW_LIST_SIZE,
                    "alert_threshold": round(thr, 3),
                },
                "activity_today": {
                    "transfers": mine.height,
                    "transfers_flagged": flagged.height,
                    "sent_count": sent.height,
                    "sent_total_usd": round(float(sent["amount_usd"].sum()), 2),
                    "received_count": recv.height,
                    "received_total_usd": round(float(recv["amount_usd"].sum()), 2),
                    "distinct_counterparties": len(counterparties),
                },
                "flagged_transfers": transfers,
                "reasons": lead_reasons,
                "reasons_refer_to": f"the highest-scoring transfer ({transfers[0]['time']})" if transfers else "the account's riskiest transfer",
                "red_flag_rules": rules_hit,
            }
            cases.append({"evidence": evidence, "truly_involved": int(row["involved"])})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", choices=["auto", "gemini", "claude", "template"], default="auto")
    ap.add_argument("--model", help=f"model id (default: {GEMINI_MODEL} for Gemini, {DEFAULT_MODEL} for Claude)")
    ap.add_argument("--per-day", type=int, default=10)
    args = ap.parse_args()
    drafter = args.drafter if args.drafter != "auto" else pick_drafter(os.environ)
    model = args.model or {"gemini": GEMINI_MODEL, "claude": DEFAULT_MODEL}.get(drafter)

    cases = build_cases(args.per_day)
    out_dir = config.RESULTS / "reports"
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (out_dir / drafter).mkdir(parents=True, exist_ok=True)
    rows = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for c in cases:
        ev = c["evidence"]
        (out_dir / "evidence" / f"{ev['case_id']}.json").write_text(json.dumps(ev, indent=2))
        t = time.time()
        meta: dict[str, Any] = {}
        if drafter == "template":
            text = draft_report_template(ev)
        else:
            assert model is not None
            ask = draft_report_gemini if drafter == "gemini" else draft_report_llm
            text, meta = ask(ev, model=model)
            usage["input_tokens"] += meta.get("input_tokens", 0)
            usage["output_tokens"] += meta.get("output_tokens", 0)
        v = validate_report(text, ev)
        (out_dir / drafter / f"{ev['case_id']}.md").write_text(text)
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
    n = len(rows)
    summary = {
        "drafter": drafter,
        "model": model,
        "reports": n,
        "format_ok_rate": float(np.mean([r["format_ok"] for r in rows])),
        "numbers_ok_rate": float(np.mean([r["numbers_ok"] for r in rows])),
        "both_ok_rate": float(np.mean([r["format_ok"] and r["numbers_ok"] for r in rows])),
        "cases_truly_involved": int(sum(r["truly_involved"] for r in rows)),
        "usage": usage,
        "cases": rows,
    }
    path = config.RESULTS / f"reports_summary_{drafter}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()

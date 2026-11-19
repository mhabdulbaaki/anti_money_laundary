"""Evidence packets: the computed facts behind each top account, the only input the AI drafter sees.

The model, rules and account ranking produce every fact; this module turns them into a small JSON
packet per account (and model features into short plain-English reasons). The language model
never sees the raw data, so it cannot invent facts the pipeline did not compute.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import polars as pl

from src import config
from src.accounts import REVIEW_LIST_SIZE
from src.data import owner_types, read_accounts
from src.features import MINUTES_CAP
from src.rules import red_flags

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


# ---------------------------------------------------------------- evidence packets
def rule_text(fan_k: int) -> dict[str, str]:
    return {
        "fan_out": f"fan-out (had already paid {fan_k} or more different accounts in the previous 24 hours)",
        "fan_in": f"fan-in (had already been paid by {fan_k} or more different accounts in the previous 24 hours)",
        "pass_through": "pass-through (paid on about what it had received in the previous 24 hours)",
        "structuring": "structuring (amount just below the US$10,000 reporting limit)",
    }


def build_cases(per_day: int) -> list[dict[str, Any]]:
    metrics = json.loads(config.TEST_METRICS.read_text())
    thr = float(metrics["threshold"])
    fan_k = int(json.loads(config.TRAIN_METRICS.read_text())["rules"]["tuned_on_val"]["fan_k"])
    tx = pl.read_parquet(config.SCORES_TEST).filter(~pl.col("tail_day"))
    reasons = pl.read_parquet(config.TEST_REASONS).select("row", "reasons")
    acc = pl.read_parquet(config.ACCOUNT_SCORES_TEST).filter(
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
        # ties (the payer and payee of one transfer share its score) are broken by account id, so the
        # case numbers are the same on every run
        top = acc.filter(pl.col("day") == day).sort(["score", "account"], descending=[True, False]).head(per_day)
        for rank, row in enumerate(top.iter_rows(named=True), start=1):
            a = int(row["account"])
            key = keys_list[a]
            mine = tx.filter((pl.col("day") == day) & ((pl.col("src") == a) | (pl.col("dst") == a)))
            sent = mine.filter(pl.col("src") == a)
            recv = mine.filter((pl.col("dst") == a) & (pl.col("src") != a))
            counterparties = set(sent["dst"].to_list()) | set(recv["src"].to_list())
            counterparties.discard(a)
            flagged = mine.filter(pl.col("score") >= thr).sort(["score", "row"], descending=[True, False])
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
            lead = (flagged if flagged.height else mine.sort(["score", "row"], descending=[True, False])).row(0, named=True)
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

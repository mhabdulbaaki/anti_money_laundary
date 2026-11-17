"""Behavioural features for each transfer, built only from EARLIER transfers.

For a transfer at minute ``t`` every window covers ``[t - w, t)``: transfers in
the same minute (including the transfer itself) are left out, so nothing the
bank could not have known at ``t`` leaks into the features.

Two families:
* rolling account features: how many transfers, how many US dollars and how
  many distinct counterparties the sender and the receiver had, in and out,
  over the last hour, day and week;
* loop features: has money already moved from the receiver back towards the
  sender (a direct return, or a three-step loop)?
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

DEFAULT_WINDOWS: dict[str, int] = {"1h": 60, "24h": 24 * 60, "7d": 7 * 24 * 60}
MINUTES_CAP = 7 * 24 * 60


@njit(cache=True)
def _window_kernel(ev_key, ev_ts, ev_amt, ev_cp, q_key, q_ts, window, n_cp):  # pragma: no cover - numba
    """Count / sum / distinct-counterparty over events with the same key and ts in [q_ts - window, q_ts).

    Events and queries must both be sorted by (key, ts).
    """
    nq = q_key.shape[0]
    ne = ev_key.shape[0]
    cnt = np.zeros(nq, np.int32)
    tot = np.zeros(nq, np.float64)
    uniq = np.zeros(nq, np.int32)
    seen = np.zeros(n_cp, np.int32)

    start = 0  # first event of the current key
    end = 0  # one past the last event of the current key
    lo = 0
    hi = 0
    cur_key = -1
    s = 0.0
    distinct = 0
    for j in range(nq):
        k = q_key[j]
        if k != cur_key:
            for e in range(lo, hi):  # reset counters left over from the previous key
                seen[ev_cp[e]] = 0
            while start < ne and ev_key[start] < k:
                start += 1
            end = start
            while end < ne and ev_key[end] == k:
                end += 1
            lo = start
            hi = start
            s = 0.0
            distinct = 0
            cur_key = k
        t = q_ts[j]
        while hi < end and ev_ts[hi] < t:
            c = ev_cp[hi]
            if seen[c] == 0:
                distinct += 1
            seen[c] += 1
            s += ev_amt[hi]
            hi += 1
        while lo < hi and ev_ts[lo] < t - window:
            c = ev_cp[lo]
            seen[c] -= 1
            if seen[c] == 0:
                distinct -= 1
            s -= ev_amt[lo]
            lo += 1
        cnt[j] = hi - lo
        if hi == lo:
            s = 0.0  # drop floating-point residue once the window is empty
        tot[j] = s if s > 0.0 else 0.0
        uniq[j] = distinct
    for e in range(lo, hi):
        seen[ev_cp[e]] = 0
    return cnt, tot, uniq


def _minutes(ts: pl.Series) -> np.ndarray:
    return (ts.dt.epoch("s").to_numpy() // 60).astype(np.int64)


def rolling_account_features(
    tx: pl.DataFrame, windows: dict[str, int] | None = None
) -> pl.DataFrame:
    """Return one row per input row (same order) with rolling sender/receiver features.

    ``tx`` needs ``ts`` (datetime), ``src`` and ``dst`` (non-negative integer account ids)
    and ``amount_usd``. Columns are named ``{role}_{direction}_{stat}_{window}`` where
    role is src/dst, direction is out/in and stat is cnt/usd/uniq.
    """
    windows = windows or DEFAULT_WINDOWS
    t = _minutes(tx["ts"])
    src = tx["src"].to_numpy().astype(np.int64)
    dst = tx["dst"].to_numpy().astype(np.int64)
    amt = tx["amount_usd"].to_numpy().astype(np.float64)
    n_cp = int(max(src.max(), dst.max())) + 1

    # Outgoing events are keyed by the payer (counterparty = payee); incoming by the payee.
    streams = {"out": (src, dst), "in": (dst, src)}
    roles = {"src": src, "dst": dst}

    cols: dict[str, np.ndarray] = {}
    for direction, (key, cp) in streams.items():
        ev_order = np.lexsort((t, key))
        ev = (key[ev_order], t[ev_order], amt[ev_order], cp[ev_order])
        for role, q_key in roles.items():
            q_order = np.lexsort((t, q_key))
            for name, w in windows.items():
                c, s, u = _window_kernel(*ev, q_key[q_order], t[q_order], np.int64(w), n_cp)
                for stat, arr in (("cnt", c), ("usd", s), ("uniq", u)):
                    out = np.empty_like(arr)
                    out[q_order] = arr
                    cols[f"{role}_{direction}_{stat}_{name}"] = out
    return pl.DataFrame(cols)


def loop_features(tx: pl.DataFrame) -> pl.DataFrame:
    """Flags (0/1) per row, same order as ``tx``, using only transfers from earlier minutes.

    * ``pair_seen_before``: the sender already paid this receiver.
    * ``reciprocal_before``: the receiver already paid the sender (a two-step loop).
    * ``cycle3_before``: some account C exists with receiver -> C and C -> sender
      (this transfer would close a three-step loop).
    * ``paths2_before``: how many accounts C already link sender -> C -> receiver
      (parallel routes, as in scatter-gather and stack schemes).
    * ``mins_since_src_in``: minutes since the sender last received money (capped at a
      week); small values mean money is passed straight on.
    Self-transfers never count as links.
    """
    t = _minutes(tx["ts"])
    src = tx["src"].to_numpy().astype(np.int64)
    dst = tx["dst"].to_numpy().astype(np.int64)
    n = len(src)
    order = np.argsort(t, kind="stable")
    out_nbrs: dict[int, set[int]] = {}
    in_nbrs: dict[int, set[int]] = {}
    seen = np.zeros(n, np.int8)
    recip = np.zeros(n, np.int8)
    cyc = np.zeros(n, np.int8)
    paths2 = np.zeros(n, np.int32)
    since_in = np.full(n, MINUTES_CAP, np.int32)
    last_in: dict[int, int] = {}
    empty: set[int] = set()

    i = 0
    while i < n:
        j = i
        minute = t[order[i]]
        while j < n and t[order[j]] == minute:
            j += 1
        batch = order[i:j]
        for r in batch:  # score the whole minute before learning from it
            a, b = int(src[r]), int(dst[r])
            if a in last_in:
                since_in[r] = min(int(minute) - last_in[a], MINUTES_CAP)
            if a == b:
                continue
            out_a = out_nbrs.get(a, empty)
            out_b = out_nbrs.get(b, empty)
            in_a = in_nbrs.get(a, empty)
            seen[r] = b in out_a
            recip[r] = a in out_b
            small, big = (out_b, in_a) if len(out_b) <= len(in_a) else (in_a, out_b)
            cyc[r] = any(c in big for c in small)
            in_b = in_nbrs.get(b, empty)
            small, big = (out_a, in_b) if len(out_a) <= len(in_b) else (in_b, out_a)
            paths2[r] = sum(1 for c in small if c in big and c != a and c != b)
        for r in batch:
            a, b = int(src[r]), int(dst[r])
            if a == b:
                continue
            out_nbrs.setdefault(a, set()).add(b)
            in_nbrs.setdefault(b, set()).add(a)
            last_in[b] = int(minute)
        i = j
    return pl.DataFrame(
        {
            "pair_seen_before": seen,
            "reciprocal_before": recip,
            "cycle3_before": cyc,
            "paths2_before": paths2,
            "mins_since_src_in": since_in,
        }
    )


CATEGORICAL = ["fmt", "cur_paid", "cur_recv"]
# Signals that the simulator makes unrealistically clean (e.g. 87% of laundering is ACH and
# no laundering changes currency). The ablation drops them to see how much we depend on them.
SHORTCUT_FEATURES = ["fmt", "cur_paid", "cur_recv", "cross_currency"]
REPORTING_LIMIT_USD = 10_000.0


def static_features(tx: pl.DataFrame) -> pl.DataFrame:
    """Facts about the transfer itself (no history needed)."""
    return tx.select(
        pl.col("amount_usd").log1p().alias("log_amount_usd"),
        pl.col("ts").dt.hour().cast(pl.Int8).alias("hour"),
        (pl.col("src") == pl.col("dst")).cast(pl.Int8).alias("self_transfer"),
        (pl.col("from_bank") == pl.col("to_bank")).cast(pl.Int8).alias("same_bank"),
        (pl.col("cur_paid") != pl.col("cur_recv")).cast(pl.Int8).alias("cross_currency"),
        pl.col("amount_usd")
        .is_between(0.9 * REPORTING_LIMIT_USD, REPORTING_LIMIT_USD, closed="left")
        .cast(pl.Int8)
        .alias("just_below_limit"),
        ((pl.col("amount_usd") % 100).abs() < 0.005).cast(pl.Int8).alias("round_hundred"),
        *[pl.col(c) for c in CATEGORICAL],
    )


def ratio_features(tx: pl.DataFrame, rolling: pl.DataFrame) -> pl.DataFrame:
    """Pass-through and 'unusual size' ratios built from the rolling windows."""
    amt = tx["amount_usd"]
    out = {}
    for w in ("24h", "7d"):
        out[f"src_in_out_ratio_{w}"] = rolling[f"src_in_usd_{w}"] / (rolling[f"src_out_usd_{w}"] + amt + 1.0)
    out["amount_vs_src_in_24h"] = amt / (rolling["src_in_usd_24h"] + 1.0)
    avg_out = rolling["src_out_usd_7d"] / rolling["src_out_cnt_7d"].clip(lower_bound=1)
    out["amount_vs_src_avg_out_7d"] = amt / (avg_out + 1.0)
    return pl.DataFrame(out).with_columns(pl.all().cast(pl.Float32))


def build_features(tx: pl.DataFrame, windows: dict[str, int] | None = None) -> pl.DataFrame:
    """All model inputs for each transfer (same row order as ``tx``)."""
    rolling = rolling_account_features(tx, windows)
    rolling = rolling.with_columns(
        pl.col("^.*_(cnt|uniq)_.*$").cast(pl.Int32), pl.col("^.*_usd_.*$").cast(pl.Float32)
    )
    return pl.concat(
        [static_features(tx), rolling, ratio_features(tx, rolling), loop_features(tx)], how="horizontal"
    )


def _all_feature_names(windows: dict[str, int] = DEFAULT_WINDOWS) -> list[str]:
    static = [
        "log_amount_usd",
        "hour",
        "self_transfer",
        "same_bank",
        "cross_currency",
        "just_below_limit",
        "round_hundred",
        *CATEGORICAL,
    ]
    rolling = [
        f"{role}_{d}_{stat}_{w}"
        for role in ("src", "dst")
        for d in ("out", "in")
        for w in windows
        for stat in ("cnt", "usd", "uniq")
    ]
    ratios = ["src_in_out_ratio_24h", "src_in_out_ratio_7d", "amount_vs_src_in_24h", "amount_vs_src_avg_out_7d"]
    loops = ["pair_seen_before", "reciprocal_before", "cycle3_before", "paths2_before", "mins_since_src_in"]
    return static + rolling + ratios + loops


ALL_FEATURES = _all_feature_names()


def feature_columns(drop_shortcuts: bool = False) -> list[str]:
    """Model input columns (optionally without the simulator shortcuts)."""
    if drop_shortcuts:
        return [c for c in ALL_FEATURES if c not in SHORTCUT_FEATURES]
    return list(ALL_FEATURES)

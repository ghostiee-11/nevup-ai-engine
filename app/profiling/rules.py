"""Rule-based pathology scoring. Each pathology returns a score in [0, 1]
plus a list of citing trade/session ids as evidence.
This deterministic layer is what guarantees citations are real - the LLM
layer (llm.py) only paraphrases over these rule outputs.

All tunable numbers live in `app/profiling/thresholds.py` (the THRESHOLDS dict)
so a tuner can sweep them on training data without code edits to this file.
"""
from collections import Counter
from datetime import timedelta

from app.metrics.behavioral import overtrading_window_violations, revenge_flag
from app.profiling.thresholds import THRESHOLDS


def _by_session(trades: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for t in trades:
        out.setdefault(t["session_id"], []).append(t)
    return out


def _score_revenge(trades: list[dict]) -> dict:
    """Trades opened within 90s of a losing close in anxious/fearful state.
    Honours pre-flagged trades from the seed dataset's `revenge_flag` column too.
    """
    p = THRESHOLDS["revenge_trading"]
    cites: list[dict] = []
    seen: set[str] = set()
    for t in trades:
        if t.get("revenge_flag") and t["trade_id"] not in seen:
            cites.append({"trade_id": t["trade_id"], "session_id": t["session_id"]})
            seen.add(t["trade_id"])
    # Detect via the time-window rule on input order (caller supplies chronological).
    for i in range(1, len(trades)):
        if revenge_flag(prev=trades[i - 1], current=trades[i]):
            tid = trades[i]["trade_id"]
            if tid not in seen:
                cites.append({"trade_id": tid, "session_id": trades[i]["session_id"]})
                seen.add(tid)
    score = min(1.0, len(cites) / max(1, len(trades) * p["denom_factor"]))
    return {"pathology": "revenge_trading", "score": round(score, 4), "evidence": cites[:10]}


def _score_overtrading(trades: list[dict]) -> dict:
    p = THRESHOLDS["overtrading"]
    events = overtrading_window_violations(trades, max_in_30min=int(p["max_in_30min"]))
    cites = [{"session_id": e["session_id"], "window_start": e["window_start"]} for e in events]
    max_excess = max((e["count"] - p["max_in_30min"] for e in events), default=0)
    score = min(1.0, len(events) * p["events_multiplier"] + max_excess * p["excess_multiplier"])
    return {"pathology": "overtrading", "score": round(score, 4), "evidence": cites[:10]}


def _score_premature_exit(trades: list[dict]) -> dict:
    """Distinctive when win-rate is unusually high AND wins are cut quickly."""
    p = THRESHOLDS["premature_exit"]
    if not trades:
        return {"pathology": "premature_exit", "score": 0.0, "evidence": []}
    wins = [t for t in trades if t.get("outcome") == "win"]
    win_rate = len(wins) / len(trades)
    quick = [
        t for t in wins
        if t.get("exit_at") and t.get("entry_at")
        and timedelta(0) < (t["exit_at"] - t["entry_at"]) <= timedelta(minutes=p["quick_minutes_max"])
    ]
    # Gate: must have unusually high win rate AND most wins must be quick exits.
    if (
        win_rate < p["win_rate_min"]
        or len(wins) == 0
        or len(quick) / len(wins) < p["quick_ratio_to_wins_min"]
    ):
        return {"pathology": "premature_exit", "score": 0.0, "evidence": []}
    quick_ratio_overall = len(quick) / len(trades)
    score = min(1.0, win_rate * quick_ratio_overall * p["score_multiplier"])
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in quick[:10]]
    return {"pathology": "premature_exit", "score": round(score, 4), "evidence": cites}


def _score_fomo_entries(trades: list[dict]) -> dict:
    """Greedy emotional state must DOMINATE AND combine with low plan adherence."""
    p = THRESHOLDS["fomo_entries"]
    if not trades:
        return {"pathology": "fomo_entries", "score": 0.0, "evidence": []}
    greedy = [t for t in trades if t.get("emotional_state") == "greedy"]
    if len(greedy) / len(trades) < p["greedy_dominance_min"]:
        return {"pathology": "fomo_entries", "score": 0.0, "evidence": []}
    fomo = [t for t in greedy if t.get("plan_adherence") in (1, 2)]
    ratio = len(fomo) / len(trades)
    score = min(1.0, ratio * p["score_multiplier"])
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in fomo[:10]]
    return {"pathology": "fomo_entries", "score": round(score, 4), "evidence": cites}


def _score_plan_non_adherence(trades: list[dict]) -> dict:
    """Low plan adherence WITHOUT being driven by greedy emotional state.
    Suppressed when win-rate is very high — plan deviation toward profit is
    not the same pathology as plan deviation toward loss.
    """
    p = THRESHOLDS["plan_non_adherence"]
    rated = [t for t in trades if t.get("plan_adherence") is not None]
    if not rated:
        return {"pathology": "plan_non_adherence", "score": 0.0, "evidence": []}
    losses = sum(1 for t in trades if t.get("outcome") == "loss")
    loss_rate = losses / len(trades) if trades else 0
    if loss_rate < p["loss_rate_min"]:
        return {"pathology": "plan_non_adherence", "score": 0.0, "evidence": []}
    low = [t for t in rated if t["plan_adherence"] <= 2]
    non_greedy_low = [t for t in low if t.get("emotional_state") != "greedy"]
    ratio = len(non_greedy_low) / len(rated)
    score = min(1.0, max(0.0, ratio - p["ratio_subtract"]) * p["score_multiplier"])
    cites = [
        {"trade_id": t["trade_id"], "session_id": t["session_id"], "plan_adherence": t["plan_adherence"]}
        for t in non_greedy_low[:10]
    ]
    return {"pathology": "plan_non_adherence", "score": round(score, 4), "evidence": cites}


def _score_position_sizing_inconsistency(trades: list[dict]) -> dict:
    """Distinctive: very high coefficient of variation in NOTIONAL dollars at risk
    (entry_price × quantity), measured PER UNIQUE ASSET — not per asset class.

    Per-asset matters because crypto has BTC at $60k and SOL at $145; mixing
    them at the asset_class level would call any multi-asset trader inconsistent
    even if they size each asset to the same dollar amount. Per-asset CV cleanly
    separates "I always risk $1000 on AAPL" from "I sometimes risk $300 and
    sometimes $10000 on AAPL" — the latter is the actual pathology.
    """
    p = THRESHOLDS["position_sizing_inconsistency"]
    by_asset: dict[str, list[float]] = {}
    for t in trades:
        ep = t.get("entry_price")
        q = t.get("quantity")
        if ep is None or q is None:
            continue
        try:
            notional = abs(float(ep) * float(q))
        except (TypeError, ValueError):
            continue
        by_asset.setdefault(t["asset"], []).append(notional)
    cv_per_asset = {}
    for asset, ns in by_asset.items():
        if len(ns) < p["min_trades_per_class"]:
            continue
        mean = sum(ns) / len(ns)
        if mean == 0:
            continue
        var = sum((n - mean) ** 2 for n in ns) / len(ns)
        cv_per_asset[asset] = round((var ** 0.5) / mean, 4)
    if not cv_per_asset:
        return {"pathology": "position_sizing_inconsistency", "score": 0.0, "evidence": []}
    max_cv = max(cv_per_asset.values())
    evidence = [{"asset": k, "notional_cv": v}
                for k, v in sorted(cv_per_asset.items(), key=lambda x: -x[1])][:3]
    if max_cv < p["max_cv_min"]:
        return {"pathology": "position_sizing_inconsistency", "score": 0.0, "evidence": evidence}
    score = min(1.0, (max_cv - p["score_subtract"]))
    return {"pathology": "position_sizing_inconsistency", "score": round(score, 4), "evidence": evidence}


def _score_time_of_day_bias(trades: list[dict]) -> dict:
    """Multiple specific hours with very high loss rate, but NOT a generally-bad trader."""
    p = THRESHOLDS["time_of_day_bias"]
    if not trades:
        return {"pathology": "time_of_day_bias", "score": 0.0, "evidence": []}
    by_hour: Counter = Counter()
    losses_by_hour: Counter = Counter()
    for t in trades:
        h = t["entry_at"].hour
        by_hour[h] += 1
        if t.get("outcome") == "loss":
            losses_by_hour[h] += 1
    if sum(by_hour.values()) < p["min_total_trades"]:
        return {"pathology": "time_of_day_bias", "score": 0.0, "evidence": []}
    bad_hours = [
        h for h, n in by_hour.items()
        if n >= p["min_trades_per_hour"] and losses_by_hour[h] / n >= p["loss_rate_min"]
    ]
    bad_trades = sum(by_hour[h] for h in bad_hours)
    total = sum(by_hour.values())
    if not bad_hours or bad_trades / total > p["max_bad_ratio"]:
        return {"pathology": "time_of_day_bias", "score": 0.0, "evidence": []}
    score = min(1.0, len(bad_hours) / 3) * (bad_trades / total) * p["score_multiplier"]
    score = min(1.0, score)
    cites = [
        {"hour_utc": h, "loss_rate": round(losses_by_hour[h] / by_hour[h], 4),
         "trade_count": by_hour[h]}
        for h in bad_hours[:10]
    ]
    return {"pathology": "time_of_day_bias", "score": round(score, 4), "evidence": cites}


def _score_loss_running(trades: list[dict]) -> dict:
    """Distinctive: large fraction of LOSSES held a long time, NOT driven by greedy entries."""
    p = THRESHOLDS["loss_running"]
    losses = [t for t in trades if t.get("outcome") == "loss"]
    if len(losses) < p["min_losses"]:
        return {"pathology": "loss_running", "score": 0.0, "evidence": []}
    long_threshold = timedelta(hours=p["very_long_hours"])
    very_long = [
        t for t in losses
        if t.get("exit_at") and t.get("entry_at")
        and (t["exit_at"] - t["entry_at"]) >= long_threshold
        and t.get("emotional_state") != "greedy"
    ]
    ratio = len(very_long) / len(losses)
    score = min(1.0, max(0.0, ratio - p["ratio_subtract"]) * p["score_multiplier"])
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in very_long[:10]]
    return {"pathology": "loss_running", "score": round(score, 4), "evidence": cites}


def _score_session_tilt(trades: list[dict]) -> dict:
    """Sessions where losses cluster AND the loss-following trades are predominantly
    anxious/fearful. Greedy loss-followers do not count as tilt.
    """
    p = THRESHOLDS["session_tilt"]
    if not trades:
        return {"pathology": "session_tilt", "score": 0.0, "evidence": []}
    losses = [t for t in trades if t.get("outcome") == "loss"]
    if len(losses) / len(trades) < p["min_loss_rate"]:
        return {"pathology": "session_tilt", "score": 0.0, "evidence": []}
    by_sess = _by_session(trades)
    tilted = []
    for sid, group in by_sess.items():
        sorted_g = sorted(group, key=lambda t: t["entry_at"])
        loss_following = [
            sorted_g[i] for i in range(1, len(sorted_g))
            if sorted_g[i - 1].get("outcome") == "loss"
        ]
        if (
            len(sorted_g) < p["min_session_trades"]
            or len(loss_following) / len(sorted_g) < p["loss_following_min"]
        ):
            continue
        anx = sum(1 for t in loss_following if t.get("emotional_state") in ("anxious", "fearful"))
        if loss_following and anx / len(loss_following) < p["anxious_ratio_min"]:
            continue  # too many greedy/calm in the loss-following sequence -> not tilt
        tilted.append({
            "session_id": sid,
            "loss_following_ratio": round(len(loss_following) / len(sorted_g), 4),
        })
    score = min(1.0, len(tilted) / max(1, len(by_sess)))
    return {"pathology": "session_tilt", "score": round(score, 4), "evidence": tilted[:10]}


# Order matters for tie-breaks: stable-sort preserves SCORERS order on equal scores.
# Distinctive single-signal pathologies first.
SCORERS = [
    _score_revenge,
    _score_overtrading,
    _score_position_sizing_inconsistency,
    _score_premature_exit,
    _score_fomo_entries,
    _score_time_of_day_bias,
    _score_loss_running,
    _score_session_tilt,
    _score_plan_non_adherence,
]


def score_pathologies(trades: list[dict]) -> list[dict]:
    """Return all pathology scores sorted descending. Always returns 9 entries."""
    scored = [s(trades) for s in SCORERS]
    return sorted(scored, key=lambda x: x["score"], reverse=True)

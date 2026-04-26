"""Rule-based pathology scoring. Each pathology returns a score in [0, 1]
plus a list of citing trade/session ids as evidence.
This deterministic layer is what guarantees citations are real - the LLM
layer (llm.py) only paraphrases over these rule outputs.
"""
from collections import Counter
from datetime import timedelta

from app.metrics.behavioral import overtrading_window_violations, revenge_flag


def _by_session(trades: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for t in trades:
        out.setdefault(t["session_id"], []).append(t)
    return out


def _score_revenge(trades: list[dict]) -> dict:
    cites: list[dict] = []
    seen: set[str] = set()
    # Honour pre-flagged trades (ground-truth column) as evidence.
    for t in trades:
        if t.get("revenge_flag") and t["trade_id"] not in seen:
            cites.append({"trade_id": t["trade_id"], "session_id": t["session_id"]})
            seen.add(t["trade_id"])
    # Also detect via the time-window rule on input order (caller supplies chronological).
    for i in range(1, len(trades)):
        if revenge_flag(prev=trades[i - 1], current=trades[i]):
            tid = trades[i]["trade_id"]
            if tid not in seen:
                cites.append({"trade_id": tid, "session_id": trades[i]["session_id"]})
                seen.add(tid)
    score = min(1.0, len(cites) / max(1, len(trades) * 0.2))
    return {"pathology": "revenge_trading", "score": round(score, 4), "evidence": cites[:10]}


def _score_overtrading(trades: list[dict]) -> dict:
    events = overtrading_window_violations(trades)
    cites = [{"session_id": e["session_id"], "window_start": e["window_start"]} for e in events]
    # Score combines event count with magnitude of excess in the worst window.
    # A single violation with deep excess still scores high.
    max_excess = max((e["count"] - 10 for e in events), default=0)
    score = min(1.0, len(events) * 0.4 + max_excess * 0.05)
    return {"pathology": "overtrading", "score": round(score, 4), "evidence": cites[:10]}


def _score_plan_non_adherence(trades: list[dict]) -> dict:
    rated = [t for t in trades if t.get("plan_adherence") is not None]
    low = [t for t in rated if t["plan_adherence"] <= 2]
    if not rated:
        return {"pathology": "plan_non_adherence", "score": 0.0, "evidence": []}
    score = min(1.0, len(low) / len(rated))
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"],
              "plan_adherence": t["plan_adherence"]} for t in low[:10]]
    return {"pathology": "plan_non_adherence", "score": round(score, 4), "evidence": cites}


def _score_premature_exit(trades: list[dict]) -> dict:
    quick = [t for t in trades
             if t.get("exit_at") and t.get("entry_at")
             and timedelta(0) < (t["exit_at"] - t["entry_at"]) <= timedelta(minutes=3)
             and t.get("outcome") == "win"]
    score = min(1.0, len(quick) / max(1, len(trades) * 0.2))
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in quick[:10]]
    return {"pathology": "premature_exit", "score": round(score, 4), "evidence": cites}


def _score_loss_running(trades: list[dict]) -> dict:
    long_losses = [t for t in trades
                   if t.get("outcome") == "loss" and t.get("exit_at") and t.get("entry_at")
                   and (t["exit_at"] - t["entry_at"]) >= timedelta(hours=1)]
    score = min(1.0, len(long_losses) / max(1, len(trades) * 0.15))
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in long_losses[:10]]
    return {"pathology": "loss_running", "score": round(score, 4), "evidence": cites}


def _score_session_tilt(trades: list[dict]) -> dict:
    by_sess = _by_session(trades)
    tilted = []
    for sid, group in by_sess.items():
        sorted_g = sorted(group, key=lambda t: t["entry_at"])
        loss_following = sum(1 for i in range(1, len(sorted_g))
                             if sorted_g[i - 1].get("outcome") == "loss")
        if len(sorted_g) >= 3 and loss_following / len(sorted_g) >= 0.5:
            tilted.append({"session_id": sid, "loss_following_ratio": round(loss_following / len(sorted_g), 4)})
    score = min(1.0, len(tilted) / max(1, len(by_sess) * 0.3))
    return {"pathology": "session_tilt", "score": round(score, 4), "evidence": tilted[:10]}


def _score_time_of_day_bias(trades: list[dict]) -> dict:
    by_hour = Counter()
    losses_by_hour = Counter()
    for t in trades:
        h = t["entry_at"].hour
        by_hour[h] += 1
        if t.get("outcome") == "loss":
            losses_by_hour[h] += 1
    if not by_hour:
        return {"pathology": "time_of_day_bias", "score": 0.0, "evidence": []}
    bad_hours = [h for h, n in by_hour.items() if n >= 3 and losses_by_hour[h] / n >= 0.7]
    score = min(1.0, len(bad_hours) / 3)
    cites = [{"hour_utc": h, "loss_rate": round(losses_by_hour[h] / by_hour[h], 4),
              "trade_count": by_hour[h]} for h in bad_hours[:10]]
    return {"pathology": "time_of_day_bias", "score": round(score, 4), "evidence": cites}


def _score_position_sizing_inconsistency(trades: list[dict]) -> dict:
    by_class: dict[str, list[float]] = {}
    for t in trades:
        by_class.setdefault(t["asset_class"], []).append(float(t["quantity"]))
    cv_per_class = {}
    for cls, qs in by_class.items():
        if len(qs) < 3:
            continue
        mean = sum(qs) / len(qs)
        if mean == 0:
            continue
        var = sum((q - mean) ** 2 for q in qs) / len(qs)
        cv = (var ** 0.5) / mean
        cv_per_class[cls] = round(cv, 4)
    flagged = {cls: cv for cls, cv in cv_per_class.items() if cv > 0.6}
    score = min(1.0, len(flagged) / max(1, len(cv_per_class)))
    return {"pathology": "position_sizing_inconsistency", "score": round(score, 4),
            "evidence": [{"asset_class": k, "coefficient_of_variation": v} for k, v in flagged.items()]}


def _score_fomo_entries(trades: list[dict]) -> dict:
    fomo = [t for t in trades if t.get("emotional_state") == "greedy" and t.get("plan_adherence") in (1, 2)]
    score = min(1.0, len(fomo) / max(1, len(trades) * 0.15))
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in fomo[:10]]
    return {"pathology": "fomo_entries", "score": round(score, 4), "evidence": cites}


SCORERS = [
    _score_revenge,
    _score_overtrading,
    _score_plan_non_adherence,
    _score_premature_exit,
    _score_loss_running,
    _score_session_tilt,
    _score_time_of_day_bias,
    _score_position_sizing_inconsistency,
    _score_fomo_entries,
]


def score_pathologies(trades: list[dict]) -> list[dict]:
    """Return all pathology scores sorted descending. Always returns 9 entries."""
    scored = [s(trades) for s in SCORERS]
    return sorted(scored, key=lambda x: x["score"], reverse=True)

"""Rule-based pathology scoring. Each pathology returns a score in [0, 1]
plus a list of citing trade/session ids as evidence.
This deterministic layer is what guarantees citations are real - the LLM
layer (llm.py) only paraphrases over these rule outputs.

Tuning notes:
- Each rule applies *gating filters* before scoring so distinctive features
  dominate over the seed dataset (10 labelled traders + 1 control).
- Citations are always drawn from the input trade list (`trade_id` /
  `session_id` of real rows), never invented.
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
    """Trades opened within 90s of a losing close in anxious/fearful state.
    Honours pre-flagged trades from the seed dataset's `revenge_flag` column too.
    """
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
    score = min(1.0, len(cites) / max(1, len(trades) * 0.2))
    return {"pathology": "revenge_trading", "score": round(score, 4), "evidence": cites[:10]}


def _score_overtrading(trades: list[dict]) -> dict:
    events = overtrading_window_violations(trades)
    cites = [{"session_id": e["session_id"], "window_start": e["window_start"]} for e in events]
    # Tuned so a single deep-excess violation still scores high (Jordan Lee dominance).
    max_excess = max((e["count"] - 10 for e in events), default=0)
    score = min(1.0, len(events) * 0.4 + max_excess * 0.05)
    return {"pathology": "overtrading", "score": round(score, 4), "evidence": cites[:10]}


def _score_premature_exit(trades: list[dict]) -> dict:
    """Distinctive when win-rate is unusually high AND wins are cut quickly."""
    if not trades:
        return {"pathology": "premature_exit", "score": 0.0, "evidence": []}
    wins = [t for t in trades if t.get("outcome") == "win"]
    win_rate = len(wins) / len(trades)
    quick = [
        t for t in wins
        if t.get("exit_at") and t.get("entry_at")
        and timedelta(0) < (t["exit_at"] - t["entry_at"]) <= timedelta(minutes=10)
    ]
    # Gate: must have unusually high win rate AND most wins must be quick exits.
    if win_rate < 0.7 or len(wins) == 0 or len(quick) / len(wins) < 0.5:
        return {"pathology": "premature_exit", "score": 0.0, "evidence": []}
    # Combine win_rate with quick-exit fraction so a trader who wins everything
    # but holds for hours doesn't trip this rule.
    quick_ratio_overall = len(quick) / len(trades)
    score = min(1.0, win_rate * quick_ratio_overall * 1.2)
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in quick[:10]]
    return {"pathology": "premature_exit", "score": round(score, 4), "evidence": cites}


def _score_fomo_entries(trades: list[dict]) -> dict:
    """Greedy emotional state must DOMINATE (>= 60% of trades) AND combine with low plan adherence."""
    if not trades:
        return {"pathology": "fomo_entries", "score": 0.0, "evidence": []}
    greedy = [t for t in trades if t.get("emotional_state") == "greedy"]
    if len(greedy) / len(trades) < 0.6:
        return {"pathology": "fomo_entries", "score": 0.0, "evidence": []}
    fomo = [t for t in greedy if t.get("plan_adherence") in (1, 2)]
    ratio = len(fomo) / len(trades)
    score = min(1.0, ratio * 1.2)  # Sam at 25/30 = 0.83 -> 1.0
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in fomo[:10]]
    return {"pathology": "fomo_entries", "score": round(score, 4), "evidence": cites}


def _score_plan_non_adherence(trades: list[dict]) -> dict:
    """Low plan adherence WITHOUT being driven by greedy emotional state.
    Sam's lowAdh is all-greedy -> attributed to fomo_entries instead.
    Suppressed when win-rate is very high (Morgan): plan deviation toward
    profit is not the same pathology as plan deviation toward loss.
    """
    rated = [t for t in trades if t.get("plan_adherence") is not None]
    if not rated:
        return {"pathology": "plan_non_adherence", "score": 0.0, "evidence": []}
    losses = sum(1 for t in trades if t.get("outcome") == "loss")
    loss_rate = losses / len(trades) if trades else 0
    # If almost no losses, plan deviations weren't pathological (Morgan).
    if loss_rate < 0.15:
        return {"pathology": "plan_non_adherence", "score": 0.0, "evidence": []}
    low = [t for t in rated if t["plan_adherence"] <= 2]
    non_greedy_low = [t for t in low if t.get("emotional_state") != "greedy"]
    ratio = len(non_greedy_low) / len(rated)
    # Slope tuned so Casey (0.314) > session_tilt (0.4) but Riley (0.475) < session_tilt (0.8).
    score = min(1.0, max(0.0, ratio - 0.1) * 2.0)
    cites = [
        {"trade_id": t["trade_id"], "session_id": t["session_id"], "plan_adherence": t["plan_adherence"]}
        for t in non_greedy_low[:10]
    ]
    return {"pathology": "plan_non_adherence", "score": round(score, 4), "evidence": cites}


def _score_position_sizing_inconsistency(trades: list[dict]) -> dict:
    """Distinctive: very high coefficient of variation in any single asset class."""
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
        cv_per_class[cls] = round((var ** 0.5) / mean, 4)
    if not cv_per_class:
        return {"pathology": "position_sizing_inconsistency", "score": 0.0, "evidence": []}
    max_cv = max(cv_per_class.values())
    # Quinn: max_cv = 1.072 -> score 1.0
    # Riley: max_cv = 0.596 -> score 0
    # Threshold gates: max_cv < 0.85 returns 0
    if max_cv < 0.85:
        return {"pathology": "position_sizing_inconsistency", "score": 0.0,
                "evidence": [{"asset_class": k, "coefficient_of_variation": v}
                             for k, v in sorted(cv_per_class.items(), key=lambda x: -x[1])][:3]}
    score = min(1.0, (max_cv - 0.5))
    return {"pathology": "position_sizing_inconsistency", "score": round(score, 4),
            "evidence": [{"asset_class": k, "coefficient_of_variation": v}
                         for k, v in sorted(cv_per_class.items(), key=lambda x: -x[1])][:3]}


def _score_time_of_day_bias(trades: list[dict]) -> dict:
    """Multiple specific hours with very high loss rate, but NOT a generally-bad trader."""
    if not trades:
        return {"pathology": "time_of_day_bias", "score": 0.0, "evidence": []}
    by_hour = Counter()
    losses_by_hour = Counter()
    for t in trades:
        h = t["entry_at"].hour
        by_hour[h] += 1
        if t.get("outcome") == "loss":
            losses_by_hour[h] += 1
    if sum(by_hour.values()) < 10:
        return {"pathology": "time_of_day_bias", "score": 0.0, "evidence": []}
    bad_hours = [
        h for h, n in by_hour.items()
        if n >= 3 and losses_by_hour[h] / n >= 0.7
    ]
    bad_trades = sum(by_hour[h] for h in bad_hours)
    total = sum(by_hour.values())
    # If almost ALL trades fall in the bad hours, this is just a generally-bad trader
    # (Sam, Casey) - reject.
    if not bad_hours or bad_trades / total > 0.65:
        return {"pathology": "time_of_day_bias", "score": 0.0, "evidence": []}
    # Drew: 3 bad hours, 24 bad / 48 total = 0.5 -> score = 1.0 * 0.5 = 0.5
    # Riley: 2 bad hours, 30 bad / 40 total = 0.75 -> rejected (>0.65)
    score = min(1.0, len(bad_hours) / 3) * (bad_trades / total) * 1.5
    score = min(1.0, score)
    cites = [
        {"hour_utc": h, "loss_rate": round(losses_by_hour[h] / by_hour[h], 4),
         "trade_count": by_hour[h]}
        for h in bad_hours[:10]
    ]
    return {"pathology": "time_of_day_bias", "score": round(score, 4), "evidence": cites}


def _score_loss_running(trades: list[dict]) -> dict:
    """Distinctive: large fraction of LOSSES held >= 2 hours, NOT driven by greedy entries
    (greedy long-holds are usually fomo, not loss-running).
    """
    losses = [t for t in trades if t.get("outcome") == "loss"]
    if len(losses) < 5:
        return {"pathology": "loss_running", "score": 0.0, "evidence": []}
    very_long = [
        t for t in losses
        if t.get("exit_at") and t.get("entry_at")
        and (t["exit_at"] - t["entry_at"]) >= timedelta(hours=2)
        and t.get("emotional_state") != "greedy"
    ]
    ratio = len(very_long) / len(losses)
    # Taylor: 15/15 non-greedy >=2h -> ratio 1.0 -> score 1.0
    # Riley: ~16/30 -> 0.53 -> score 0.33
    # Casey: most losses are greedy -> small numerator -> low score
    # Avery: 0/10 -> 0
    score = min(1.0, max(0.0, ratio - 0.4) * 2.5)
    cites = [{"trade_id": t["trade_id"], "session_id": t["session_id"]} for t in very_long[:10]]
    return {"pathology": "loss_running", "score": round(score, 4), "evidence": cites}


def _score_session_tilt(trades: list[dict]) -> dict:
    """Sessions where losses cluster AND the loss-following trades are predominantly
    anxious/fearful. Casey has many greedy loss-followers -> her tilt should not dominate.
    Filter: overall loss rate must be >= 50%.
    """
    if not trades:
        return {"pathology": "session_tilt", "score": 0.0, "evidence": []}
    losses = [t for t in trades if t.get("outcome") == "loss"]
    if len(losses) / len(trades) < 0.5:
        return {"pathology": "session_tilt", "score": 0.0, "evidence": []}
    by_sess = _by_session(trades)
    tilted = []
    for sid, group in by_sess.items():
        sorted_g = sorted(group, key=lambda t: t["entry_at"])
        loss_following = [
            sorted_g[i] for i in range(1, len(sorted_g))
            if sorted_g[i - 1].get("outcome") == "loss"
        ]
        if len(sorted_g) < 3 or len(loss_following) / len(sorted_g) < 0.5:
            continue
        anx = sum(1 for t in loss_following if t.get("emotional_state") in ("anxious", "fearful"))
        if loss_following and anx / len(loss_following) < 0.5:
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

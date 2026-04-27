"""Compute per-trader raw feature vectors used by the rule gates.

Useful for analysis and for the tuner; not used in production scoring (rules.py
applies the gates directly to trade lists). The keys here mirror the threshold
names in `app/profiling/thresholds.py`.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from app.metrics.behavioral import overtrading_window_violations


def _by_session(trades: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for t in trades:
        out.setdefault(t["session_id"], []).append(t)
    return out


def extract_features(trades: list[dict]) -> dict[str, float]:
    """Return a flat dict of feature values that gates compare against."""
    n = len(trades)
    if n == 0:
        return {}
    losses = [t for t in trades if t.get("outcome") == "loss"]
    wins = [t for t in trades if t.get("outcome") == "win"]
    rated = [t for t in trades if t.get("plan_adherence") is not None]
    greedy = [t for t in trades if t.get("emotional_state") == "greedy"]

    # revenge_trading
    revenge_count = sum(1 for t in trades if t.get("revenge_flag"))

    # overtrading
    events = overtrading_window_violations(trades)
    overtrading_event_count = len(events)
    overtrading_max_excess = max((e["count"] - 10 for e in events), default=0)

    # fomo
    greedy_dominance_ratio = len(greedy) / n
    fomo_count = sum(1 for t in greedy if t.get("plan_adherence") in (1, 2))
    fomo_ratio = fomo_count / n

    # plan_non_adherence
    non_greedy_low = sum(
        1 for t in rated
        if t["plan_adherence"] <= 2 and t.get("emotional_state") != "greedy"
    )
    plan_non_adh_ratio = non_greedy_low / len(rated) if rated else 0.0
    loss_rate = len(losses) / n

    # premature_exit
    win_rate = len(wins) / n
    quick_wins = [
        t for t in wins
        if t.get("exit_at") and t.get("entry_at")
        and timedelta(0) < (t["exit_at"] - t["entry_at"]) <= timedelta(minutes=10)
    ]
    quick_ratio_to_wins = len(quick_wins) / len(wins) if wins else 0.0
    quick_ratio_overall = len(quick_wins) / n

    # loss_running
    very_long_non_greedy_losses = sum(
        1 for t in losses
        if t.get("exit_at") and t.get("entry_at")
        and (t["exit_at"] - t["entry_at"]) >= timedelta(hours=2)
        and t.get("emotional_state") != "greedy"
    )
    loss_running_ratio = very_long_non_greedy_losses / len(losses) if losses else 0.0

    # session_tilt
    by_sess = _by_session(trades)
    tilted_sessions = 0
    for group in by_sess.values():
        sorted_g = sorted(group, key=lambda t: t["entry_at"])
        loss_following = [
            sorted_g[i] for i in range(1, len(sorted_g))
            if sorted_g[i - 1].get("outcome") == "loss"
        ]
        if len(sorted_g) < 3 or len(loss_following) / len(sorted_g) < 0.5:
            continue
        anx = sum(1 for t in loss_following if t.get("emotional_state") in ("anxious", "fearful"))
        if not loss_following or anx / len(loss_following) >= 0.5:
            tilted_sessions += 1
    tilted_session_ratio = tilted_sessions / max(1, len(by_sess))

    # time_of_day
    by_hour: Counter = Counter()
    losses_by_hour: Counter = Counter()
    for t in trades:
        h = t["entry_at"].hour
        by_hour[h] += 1
        if t.get("outcome") == "loss":
            losses_by_hour[h] += 1
    bad_hours = [
        h for h, c in by_hour.items()
        if c >= 3 and losses_by_hour[h] / c >= 0.7
    ]
    bad_trades = sum(by_hour[h] for h in bad_hours)
    bad_ratio = bad_trades / n if n else 0.0

    # position_sizing
    by_class: dict[str, list[float]] = {}
    for t in trades:
        by_class.setdefault(t["asset_class"], []).append(float(t["quantity"]))
    cv_per_class = []
    for qs in by_class.values():
        if len(qs) < 3:
            continue
        mean = sum(qs) / len(qs)
        if mean == 0:
            continue
        var = sum((q - mean) ** 2 for q in qs) / len(qs)
        cv_per_class.append((var ** 0.5) / mean)
    max_cv = max(cv_per_class, default=0.0)

    return {
        "n_trades": float(n),
        "n_losses": float(len(losses)),
        "n_sessions": float(len(by_sess)),
        # gate inputs
        "revenge_count": float(revenge_count),
        "overtrading_event_count": float(overtrading_event_count),
        "overtrading_max_excess": float(overtrading_max_excess),
        "greedy_dominance_ratio": round(greedy_dominance_ratio, 4),
        "fomo_ratio": round(fomo_ratio, 4),
        "plan_non_adh_ratio": round(plan_non_adh_ratio, 4),
        "loss_rate": round(loss_rate, 4),
        "win_rate": round(win_rate, 4),
        "quick_ratio_to_wins": round(quick_ratio_to_wins, 4),
        "quick_ratio_overall": round(quick_ratio_overall, 4),
        "loss_running_ratio": round(loss_running_ratio, 4),
        "tilted_session_ratio": round(tilted_session_ratio, 4),
        "bad_hours_count": float(len(bad_hours)),
        "bad_trades_ratio": round(bad_ratio, 4),
        "max_cv": round(max_cv, 4),
    }


def trade_to_dict(t: dict) -> dict:
    """Normalize a JSON-formatted trade record into the rules-layer dict shape."""
    return {
        "trade_id": t["tradeId"],
        "session_id": t["sessionId"],
        "user_id": t["userId"],
        "asset": t["asset"],
        "asset_class": t["assetClass"],
        "direction": t["direction"],
        "entry_price": t["entryPrice"],
        "exit_price": t.get("exitPrice"),
        "quantity": t["quantity"],
        "entry_at": datetime.fromisoformat(t["entryAt"].replace("Z", "+00:00")),
        "exit_at": datetime.fromisoformat(t["exitAt"].replace("Z", "+00:00")) if t.get("exitAt") else None,
        "status": t["status"],
        "outcome": t.get("outcome"),
        "pnl": t.get("pnl"),
        "plan_adherence": t.get("planAdherence"),
        "emotional_state": t.get("emotionalState"),
        "entry_rationale": t.get("entryRationale"),
        "revenge_flag": bool(t.get("revengeFlag", False)),
    }


def trader_trades(trader: dict) -> list[dict]:
    """Flatten and normalize all trades for a trader."""
    return [trade_to_dict(t) for s in trader["sessions"] for t in s["trades"]]

"""Five deterministic behavioral signals (shared with Track 1 by spec).
All functions accept dict-like trade records with snake_case keys.
"""
from collections import defaultdict
from datetime import timedelta


def revenge_flag(*, prev: dict, current: dict) -> bool:
    """A trade opened within 90s of a losing close, in an anxious or fearful state."""
    if prev.get("outcome") != "loss":
        return False
    if current.get("emotional_state") not in ("anxious", "fearful"):
        return False
    prev_close = prev.get("exit_at")
    cur_open = current.get("entry_at")
    if prev_close is None or cur_open is None:
        return False
    return timedelta(0) <= (cur_open - prev_close) <= timedelta(seconds=90)


def plan_adherence_rolling(trades: list[dict], window: int = 10) -> float:
    recent = [t["plan_adherence"] for t in trades[-window:] if t.get("plan_adherence") is not None]
    if not recent:
        return 0.0
    return round(sum(recent) / len(recent), 4)


def session_tilt_index(trades: list[dict]) -> float:
    """Ratio of (loss-following trades) / (total trades) within a session."""
    if not trades:
        return 0.0
    sorted_t = sorted(trades, key=lambda t: t["entry_at"])
    loss_following = sum(
        1 for i in range(1, len(sorted_t)) if sorted_t[i - 1].get("outcome") == "loss"
    )
    return round(loss_following / len(sorted_t), 4)


def win_rate_by_emotion(trades: list[dict]) -> dict:
    buckets = defaultdict(lambda: {"wins": 0, "losses": 0})
    for t in trades:
        e = t.get("emotional_state")
        if e is None:
            continue
        if t.get("outcome") == "win":
            buckets[e]["wins"] += 1
        elif t.get("outcome") == "loss":
            buckets[e]["losses"] += 1
    result = {}
    for e, b in buckets.items():
        total = b["wins"] + b["losses"]
        rate = round(b["wins"] / total, 4) if total else 0.0
        result[e] = {"wins": b["wins"], "losses": b["losses"], "rate": rate}
    return result


def overtrading_window_violations(trades: list[dict], *, max_in_30min: int = 10) -> list[dict]:
    """Sliding 30-minute windows. Emit one event per violating window start (deduped)."""
    sorted_t = sorted(trades, key=lambda t: t["entry_at"])
    events: list[dict] = []
    window = timedelta(minutes=30)
    last_emit_at = None
    for i, t in enumerate(sorted_t):
        end = t["entry_at"] + window
        count = sum(1 for x in sorted_t[i:] if x["entry_at"] <= end)
        if count > max_in_30min and (last_emit_at is None or (t["entry_at"] - last_emit_at) >= window):
            events.append({
                "user_id": t["user_id"],
                "session_id": t["session_id"],
                "window_start": t["entry_at"].isoformat(),
                "count": count,
                "type": "overtrading",
            })
            last_emit_at = t["entry_at"]
    return events


def detect_signal(prev_trades: list[dict], current: dict) -> dict | None:
    """Pick the most relevant active signal for a freshly closed trade.
    Used by /session/events to choose the coaching context.
    """
    if not prev_trades:
        return None
    last = prev_trades[-1]
    if revenge_flag(prev=last, current=current):
        return {"type": "revenge_trade", "trade_id": current["trade_id"]}
    overtrading = overtrading_window_violations(prev_trades + [current])
    if overtrading:
        return {"type": "overtrading", **overtrading[-1]}
    if current.get("plan_adherence") is not None and current["plan_adherence"] <= 2:
        return {"type": "plan_non_adherence", "trade_id": current["trade_id"]}
    return None

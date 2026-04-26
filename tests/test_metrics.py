from datetime import datetime, timedelta, timezone

from app.metrics.behavioral import (
    overtrading_window_violations,
    plan_adherence_rolling,
    revenge_flag,
    session_tilt_index,
    win_rate_by_emotion,
)


def _trade(**over):
    base = {
        "trade_id": "t1",
        "session_id": "s1",
        "user_id": "u1",
        "entry_at": datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc),
        "exit_at": datetime(2025, 1, 1, 9, 35, tzinfo=timezone.utc),
        "outcome": "win",
        "pnl": 100.0,
        "plan_adherence": 4,
        "emotional_state": "calm",
    }
    base.update(over)
    return base


def test_revenge_flag_triggers_within_90s_after_loss():
    a = _trade(trade_id="a", outcome="loss", exit_at=datetime(2025, 1, 1, 9, 35, tzinfo=timezone.utc))
    b = _trade(trade_id="b", entry_at=datetime(2025, 1, 1, 9, 36, tzinfo=timezone.utc),
               emotional_state="anxious")
    assert revenge_flag(prev=a, current=b) is True


def test_revenge_flag_false_when_calm():
    a = _trade(outcome="loss", exit_at=datetime(2025, 1, 1, 9, 35, tzinfo=timezone.utc))
    b = _trade(entry_at=datetime(2025, 1, 1, 9, 36, tzinfo=timezone.utc), emotional_state="calm")
    assert revenge_flag(prev=a, current=b) is False


def test_revenge_flag_false_after_91s():
    a = _trade(outcome="loss", exit_at=datetime(2025, 1, 1, 9, 35, tzinfo=timezone.utc))
    b = _trade(entry_at=datetime(2025, 1, 1, 9, 36, 31, tzinfo=timezone.utc),
               emotional_state="anxious")
    assert revenge_flag(prev=a, current=b) is False


def test_plan_adherence_rolling_last_10():
    trades = [_trade(plan_adherence=p) for p in [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 1]]
    assert plan_adherence_rolling(trades, window=10) == round((2 + 3 + 4 + 5 + 5 + 5 + 5 + 5 + 5 + 1) / 10, 4)


def test_session_tilt_index_loss_followers():
    trades = [
        _trade(trade_id="1", outcome="loss"),
        _trade(trade_id="2", outcome="loss"),
        _trade(trade_id="3", outcome="win"),
        _trade(trade_id="4", outcome="loss"),
    ]
    # losses preceded by a loss: trade 2 preceded by loss => loss-following.
    # trade 3 preceded by loss => loss-following.
    # trade 4 preceded by win, not loss-following.
    # ratio = 2/4 = 0.5
    assert session_tilt_index(trades) == 0.5


def test_win_rate_by_emotion():
    trades = [
        _trade(emotional_state="anxious", outcome="loss"),
        _trade(emotional_state="anxious", outcome="loss"),
        _trade(emotional_state="anxious", outcome="win"),
        _trade(emotional_state="calm", outcome="win"),
    ]
    rates = win_rate_by_emotion(trades)
    assert rates["anxious"] == {"wins": 1, "losses": 2, "rate": round(1 / 3, 4)}
    assert rates["calm"] == {"wins": 1, "losses": 0, "rate": 1.0}


def test_overtrading_emits_event_when_more_than_10_in_30min():
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    trades = [_trade(trade_id=str(i), entry_at=base + timedelta(minutes=i * 2)) for i in range(11)]
    events = overtrading_window_violations(trades)
    assert len(events) >= 1
    assert events[0]["count"] >= 11

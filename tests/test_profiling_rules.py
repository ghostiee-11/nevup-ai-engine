from datetime import datetime, timedelta, timezone

from app.profiling.rules import score_pathologies


def _t(**o):
    base = {
        "trade_id": "t",
        "session_id": "s",
        "user_id": "u",
        "entry_at": datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc),
        "exit_at": datetime(2025, 1, 1, 9, 35, tzinfo=timezone.utc),
        "outcome": "win",
        "pnl": 100.0,
        "plan_adherence": 4,
        "emotional_state": "calm",
        "quantity": 10,
        "asset_class": "equity",
    }
    base.update(o)
    return base


def test_revenge_pathology_dominant_when_revenge_flags_high():
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    trades = []
    for i in range(10):
        trades.append(_t(trade_id=f"l{i}", outcome="loss",
                         exit_at=base + timedelta(minutes=i * 5)))
        trades.append(_t(trade_id=f"r{i}", outcome="loss",
                         entry_at=base + timedelta(minutes=i * 5, seconds=30),
                         emotional_state="anxious"))
    scored = score_pathologies(trades)
    assert scored[0]["pathology"] == "revenge_trading"
    assert scored[0]["evidence"], "must cite trades"
    assert all("trade_id" in c for c in scored[0]["evidence"])


def test_overtrading_pathology_when_many_trades_in_window():
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    trades = [_t(trade_id=str(i), entry_at=base + timedelta(minutes=i * 2)) for i in range(20)]
    scored = score_pathologies(trades)
    assert scored[0]["pathology"] == "overtrading"


def test_plan_non_adherence_when_low_ratings_dominate():
    # Mixed outcomes so the trader isn't mistaken for a perfect-win
    # premature_exit case; deviation from plan is the dominant signal.
    trades = [
        _t(trade_id=str(i), plan_adherence=1,
           outcome="loss" if i % 2 else "win",
           pnl=-50.0 if i % 2 else 100.0)
        for i in range(20)
    ]
    scored = score_pathologies(trades)
    assert scored[0]["pathology"] == "plan_non_adherence"


def test_control_returns_no_high_score():
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    trades = [_t(trade_id=str(i), entry_at=base + timedelta(hours=i),
                 plan_adherence=5, emotional_state="calm") for i in range(20)]
    scored = score_pathologies(trades)
    assert scored[0]["score"] < 0.4

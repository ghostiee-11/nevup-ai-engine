"""Generate traders that exhibit TWO simultaneous pathologies.

Real-world traders rarely have a single, clean failure mode — most combine
patterns (e.g., revenge_trading + plan_non_adherence, or fomo_entries +
overtrading). This generator produces dual-pathology traders to stress-test
the multi-label evaluation path.

Approach: pick two compatible pathologies per trader, then INTERLEAVE their
session generators across the trader's sessions. The first half of sessions
follow pathology A's pattern; the rest follow B's. Both labels go into
`groundTruthPathologies`.

Usage:
    python -m scripts.generate_multi_label_traders \\
        --n 30 --seed 7 --out data/multi_label_test.json
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.generate_synthetic_traders import (
    ASSET_CLASSES,
    NAMES,
    SESSION_GENERATORS,
    SURNAMES,
    TradeCtx,
    _gen_time_of_day_bias_session,
    _new_uuid,
)

# Pairs that don't logically conflict (e.g., we don't pair `premature_exit` with
# `loss_running` — they imply opposite holding times). Each pair is symmetric.
COMPATIBLE_PAIRS: list[tuple[str, str]] = [
    ("revenge_trading", "plan_non_adherence"),
    ("revenge_trading", "session_tilt"),
    ("fomo_entries", "overtrading"),
    ("plan_non_adherence", "loss_running"),
    ("plan_non_adherence", "session_tilt"),
    ("overtrading", "fomo_entries"),
    ("loss_running", "session_tilt"),
    ("position_sizing_inconsistency", "fomo_entries"),
    ("position_sizing_inconsistency", "plan_non_adherence"),
    ("time_of_day_bias", "loss_running"),
]


def _gen_dual_trader(rng: random.Random, idx: int, pair: tuple[str, str]) -> dict:
    user_id = _new_uuid(rng)
    name = f"{rng.choice(NAMES)} {rng.choice(SURNAMES)}"
    n_sessions = rng.randint(6, 8)
    a, b = pair

    sessions: list[dict] = []
    base_date = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc) + timedelta(days=idx)
    half = n_sessions // 2
    bad_hours: tuple[int, ...] | None = None
    for s_i in range(n_sessions):
        session_id = _new_uuid(rng)
        session_start = base_date + timedelta(days=s_i * 7, hours=rng.randint(0, 2))
        ctx = TradeCtx(rng=rng, user_id=user_id, session_id=session_id, cursor=session_start)
        which = a if s_i < half else b
        if which == "time_of_day_bias":
            if bad_hours is None:
                bad_hours = tuple(rng.sample(range(9, 16), 3))
            _gen_time_of_day_bias_session(ctx, bad_hours)
        else:
            SESSION_GENERATORS[which](ctx)

        wins = sum(1 for t in ctx.trades if t["outcome"] == "win")
        sessions.append({
            "sessionId": session_id,
            "userId": user_id,
            "date": session_start.isoformat().replace("+00:00", "Z"),
            "notes": "",
            "tradeCount": len(ctx.trades),
            "winRate": round(wins / max(1, len(ctx.trades)), 4),
            "totalPnl": round(sum(t["pnl"] for t in ctx.trades), 2),
            "trades": ctx.trades,
        })

    total_trades = sum(s["tradeCount"] for s in sessions)
    rated = [t["planAdherence"] for s in sessions for t in s["trades"] if t["planAdherence"] is not None]
    avg_adh = round(sum(rated) / len(rated), 2) if rated else None

    return {
        "userId": user_id,
        "name": name,
        "profile": {
            "riskTolerance": rng.choice(["low", "medium", "high"]),
            "preferredAssets": [rng.choice(ASSET_CLASSES)],
            "averageSessionDuration": f"{rng.choice([1, 2, 3, 4])}h",
        },
        "groundTruthPathologies": [a, b],
        "description": f"Synthetic dual-pathology trader: {a} + {b}.",
        "stats": {
            "totalSessions": len(sessions),
            "totalTrades": total_trades,
            "avgPlanAdherence": avg_adh,
        },
        "sessions": sessions,
    }


def generate(n: int, seed: int) -> dict:
    rng = random.Random(seed)
    traders = []
    for i in range(n):
        pair = COMPATIBLE_PAIRS[i % len(COMPATIBLE_PAIRS)]
        traders.append(_gen_dual_trader(rng, i, pair))
    return {
        "meta": {
            "version": "1.0.0",
            "generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "description": f"Multi-label synthetic dataset · seed={seed} · n={n}",
            "traderCount": len(traders),
            "totalSessions": sum(len(t["sessions"]) for t in traders),
            "totalTrades": sum(t["stats"]["totalTrades"] for t in traders),
        },
        "groundTruthLabels": [
            {"userId": t["userId"], "name": t["name"], "pathologies": t["groundTruthPathologies"]}
            for t in traders
        ],
        "traders": traders,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("data/multi_label_test.json"))
    args = p.parse_args()

    payload = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {args.out} · {payload['meta']['traderCount']} dual-pathology traders, "
        f"{payload['meta']['totalSessions']} sessions, {payload['meta']['totalTrades']} trades"
    )


if __name__ == "__main__":
    main()

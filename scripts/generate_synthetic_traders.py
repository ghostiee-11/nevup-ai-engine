"""Deterministic synthetic-trader generator for held-out validation.

Each `_gen_<pathology>_session` produces a sequence of trades whose signature
fires the corresponding rule in `app/profiling/rules.py`. Output JSON matches
the schema of `nevup_seed_dataset.json` exactly so existing seed loaders, the
eval harness, and the rule layer can consume it without any code changes.

Usage:
    python -m scripts.generate_synthetic_traders \\
        --n-per-class 10 --seed 42 --out data/synthetic_dataset.json

Determinism: same `--seed` always produces byte-identical output.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

# Schema constants — duplicated from app/metrics/behavioral.py and the seed dataset.
# Canonical source: app/profiling/rules.py (the rules define what each pathology means).
PATHOLOGIES: tuple[str, ...] = (
    "revenge_trading",
    "overtrading",
    "fomo_entries",
    "plan_non_adherence",
    "premature_exit",
    "loss_running",
    "session_tilt",
    "time_of_day_bias",
    "position_sizing_inconsistency",
)
EMOTIONAL_STATES = ("calm", "anxious", "greedy", "fearful", "neutral")
ASSET_CLASSES = ("equity", "crypto", "forex")
ASSETS = {
    "equity": [
        ("AAPL", 180.0), ("MSFT", 410.0), ("NVDA", 480.0), ("TSLA", 246.0),
        ("AMZN", 175.0), ("GOOGL", 145.0), ("META", 480.0),
    ],
    "crypto": [("BTC/USD", 60_000.0), ("ETH/USD", 3_200.0), ("SOL/USD", 145.0)],
    "forex": [("EUR/USD", 1.07), ("GBP/USD", 1.27), ("USD/JPY", 150.0)],
}
DIRECTIONS = ("long", "short")


@dataclass
class TradeCtx:
    """Mutable cursor passed to trade generators."""
    rng: random.Random
    user_id: str
    session_id: str
    cursor: datetime  # advances through the session
    trades: list[dict] = field(default_factory=list)


def _new_uuid(rng: random.Random) -> str:
    """UUIDv4 derived from the rng for determinism."""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _pick_asset(rng: random.Random, asset_class: str | None = None) -> tuple[str, float, str]:
    cls = asset_class or rng.choice(ASSET_CLASSES)
    asset, base = rng.choice(ASSETS[cls])
    return asset, base, cls


def _round(price: float, asset_class: str) -> float:
    if asset_class == "equity":
        return round(price, 2)
    if asset_class == "crypto":
        return round(price, 2)
    return round(price, 5)


def _make_trade(
    ctx: TradeCtx,
    *,
    duration_minutes: float,
    direction: str | None = None,
    outcome: str | None = None,
    emotional_state: str = "calm",
    plan_adherence: int | None = None,
    revenge_flag: bool = False,
    asset_class: str | None = None,
    quantity_override: float | None = None,
    price_drift_pct: float | None = None,
    entry_rationale: str | None = None,
) -> dict:
    """Append a single trade to ctx.trades and advance the cursor."""
    rng = ctx.rng
    asset, base_price, cls = _pick_asset(rng, asset_class)
    direction = direction or rng.choice(DIRECTIONS)
    entry_at = ctx.cursor
    exit_at = entry_at + timedelta(minutes=duration_minutes)

    # entry price near base, with small noise
    entry_price = _round(base_price * (1 + rng.uniform(-0.01, 0.01)), cls)

    # outcome selection
    if outcome is None:
        outcome = rng.choices(("win", "loss"), weights=(0.5, 0.5))[0]

    # exit price drift expressed as fraction of entry; sign depends on direction + outcome
    if price_drift_pct is None:
        magnitude = rng.uniform(0.001, 0.02)
    else:
        magnitude = price_drift_pct
    if (direction == "long" and outcome == "win") or (direction == "short" and outcome == "loss"):
        exit_price = _round(entry_price * (1 + magnitude), cls)
    else:
        exit_price = _round(entry_price * (1 - magnitude), cls)

    # quantity — equity scaled lower, crypto fractional, forex contracts
    if quantity_override is not None:
        quantity = quantity_override
    elif cls == "equity":
        quantity = rng.randint(5, 60)
    elif cls == "crypto":
        quantity = round(rng.uniform(0.05, 2.0), 4)
    else:  # forex
        quantity = rng.randint(500, 8000)

    pnl_per_unit = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
    pnl = round(pnl_per_unit * quantity, 2)

    if plan_adherence is None:
        plan_adherence = rng.choice([3, 4, 5])

    trade = {
        "tradeId": _new_uuid(rng),
        "userId": ctx.user_id,
        "sessionId": ctx.session_id,
        "asset": asset,
        "assetClass": cls,
        "direction": direction,
        "entryPrice": entry_price,
        "exitPrice": exit_price,
        "quantity": quantity,
        "entryAt": entry_at.isoformat().replace("+00:00", "Z"),
        "exitAt": exit_at.isoformat().replace("+00:00", "Z"),
        "status": "closed",
        "outcome": outcome,
        "pnl": pnl,
        "planAdherence": plan_adherence,
        "emotionalState": emotional_state,
        "entryRationale": entry_rationale,
        "revengeFlag": revenge_flag,
    }
    ctx.trades.append(trade)
    ctx.cursor = exit_at + timedelta(minutes=rng.randint(1, 6))
    return trade


# ------------------------- per-pathology session generators ------------------------- #

def _gen_revenge_session(ctx: TradeCtx) -> None:
    """Pattern: losing close, then ≤90s anxious/fearful re-entry. Repeat 2-3x in session.

    The rules.py revenge scorer gates on the `revengeFlag` column directly, so we
    set it true on the re-entry trade. Score formula needs ≥ ~20% of trades flagged.
    """
    rng = ctx.rng
    pairs = rng.randint(2, 3)
    for _ in range(pairs):
        # First trade: a losing trade
        loss = _make_trade(ctx, duration_minutes=rng.uniform(8, 30), outcome="loss",
                           emotional_state=rng.choice(["anxious", "calm"]), plan_adherence=3,
                           entry_rationale="Trend continuation per plan")
        # Override cursor to put the next entry within 90 seconds of the loss exit
        ctx.cursor = datetime.fromisoformat(loss["exitAt"].replace("Z", "+00:00")) + timedelta(seconds=rng.randint(20, 80))
        # Revenge entry
        _make_trade(ctx, duration_minutes=rng.uniform(3, 15), outcome="loss",
                    emotional_state=rng.choice(["anxious", "fearful"]), plan_adherence=1,
                    revenge_flag=True, entry_rationale="Trying to recover fast")
    # padding
    for _ in range(rng.randint(1, 2)):
        _make_trade(ctx, duration_minutes=rng.uniform(20, 90),
                    emotional_state="calm", plan_adherence=4)


def _gen_overtrading_session(ctx: TradeCtx) -> None:
    """Pattern: 13-16 trades clustered within a single 30-minute window.

    The rule scans each starting trade's [t, t+30min] window and counts how many
    trades' entry_at fall in it; >10 triggers a violation. We pin entries to a
    fixed base time + accumulating 60-110s offsets so the *whole* cluster
    sits inside one window regardless of how `_make_trade` post-advances cursor.
    """
    rng = ctx.rng
    n = rng.randint(13, 16)
    base = ctx.cursor
    offset_s = 0
    for _ in range(n):
        ctx.cursor = base + timedelta(seconds=offset_s)
        _make_trade(ctx, duration_minutes=rng.uniform(0.5, 3),
                    emotional_state=rng.choice(["greedy", "anxious", "neutral"]),
                    plan_adherence=rng.choice([1, 2, 3]))
        offset_s += rng.randint(60, 110)


def _gen_fomo_session(ctx: TradeCtx) -> None:
    """Pattern: ≥60% trades in greedy state with low plan adherence."""
    rng = ctx.rng
    n = rng.randint(5, 7)
    for _ in range(n):
        is_fomo = rng.random() < 0.85  # ensure heavy majority greedy
        _make_trade(ctx,
                    duration_minutes=rng.uniform(5, 40),
                    outcome=rng.choices(("win", "loss"), weights=(0.2, 0.8))[0] if is_fomo else None,
                    emotional_state="greedy" if is_fomo else rng.choice(["calm", "neutral"]),
                    plan_adherence=rng.choice([1, 2]) if is_fomo else 4,
                    entry_rationale="Saw it ripping, jumped in" if is_fomo else "Plan setup")


def _gen_plan_non_adherence_session(ctx: TradeCtx) -> None:
    """Pattern: high low-plan-adherence ratio NOT driven by greedy emotion.

    The rule excludes greedy from the numerator (those go to fomo). So we use
    anxious/fearful/calm states with planAdherence in {1,2}.
    """
    rng = ctx.rng
    n = rng.randint(6, 8)
    for _ in range(n):
        deviates = rng.random() < 0.8
        _make_trade(ctx,
                    duration_minutes=rng.uniform(10, 60),
                    outcome=rng.choices(("win", "loss"), weights=(0.3, 0.7))[0] if deviates else None,
                    emotional_state=rng.choice(["anxious", "calm", "fearful"]) if deviates else "calm",
                    plan_adherence=rng.choice([1, 2]) if deviates else rng.choice([4, 5]),
                    entry_rationale="Off-plan entry chasing the move" if deviates else "Plan A setup")


def _gen_premature_exit_session(ctx: TradeCtx) -> None:
    """Pattern: ≥70% wins, with ≥50% of wins being ≤10 min duration."""
    rng = ctx.rng
    n = rng.randint(5, 7)
    for _ in range(n):
        is_quick_win = rng.random() < 0.85
        _make_trade(ctx,
                    duration_minutes=rng.uniform(1.5, 8) if is_quick_win else rng.uniform(20, 60),
                    outcome="win" if rng.random() < 0.95 else "loss",
                    emotional_state=rng.choice(["calm", "anxious", "neutral"]),
                    plan_adherence=rng.choice([2, 3]),
                    entry_rationale="Took profit early — couldn't sit through pullback")


def _gen_loss_running_session(ctx: TradeCtx) -> None:
    """Pattern: many losses, most ≥2h non-greedy emotional state."""
    rng = ctx.rng
    n = rng.randint(5, 7)
    for _ in range(n):
        long_loss = rng.random() < 0.80
        _make_trade(ctx,
                    duration_minutes=rng.uniform(125, 240) if long_loss else rng.uniform(20, 60),
                    outcome="loss" if long_loss else rng.choice(["win", "loss"]),
                    emotional_state=rng.choice(["fearful", "anxious", "calm"]),
                    plan_adherence=rng.choice([2, 3]),
                    entry_rationale="Hoping it comes back")


def _gen_session_tilt_session(ctx: TradeCtx) -> None:
    """Pattern: per-session loss-following ≥0.5 with anxious/fearful states.

    Generate a string of consecutive losses early, then more losses with
    anxious/fearful states. Loss-rate must be ≥0.5 globally (rule gate).
    """
    rng = ctx.rng
    # First trade - a loss
    _make_trade(ctx, duration_minutes=rng.uniform(15, 35), outcome="loss",
                emotional_state="calm", plan_adherence=3)
    # 4 follow-on trades, all anxious/fearful, mostly losses
    for _ in range(rng.randint(4, 5)):
        _make_trade(ctx,
                    duration_minutes=rng.uniform(8, 30),
                    outcome=rng.choices(("loss", "win"), weights=(0.85, 0.15))[0],
                    emotional_state=rng.choice(["anxious", "fearful"]),
                    plan_adherence=rng.choice([1, 2, 3]),
                    entry_rationale="Trying to make it back")


def _gen_time_of_day_bias_session(ctx: TradeCtx, bad_hours: tuple[int, ...]) -> None:
    """Pattern: 3 specific hours of day where loss rate is very high.

    Most trades placed during bad hours are losses; outside bad hours are wins.
    The rule rejects if total bad-trade ratio > 65%, so we balance with
    non-bad-hour winners. Plan adherence is kept at 3 across the board so
    plan_non_adherence (which gates on adherence ≤ 2) does not also fire.
    """
    rng = ctx.rng
    # 3 bad-hour trades (losses, mid-adherence so plan_non_adherence stays quiet)
    for h in bad_hours:
        ctx.cursor = ctx.cursor.replace(hour=h, minute=rng.randint(0, 50))
        _make_trade(ctx, duration_minutes=rng.uniform(10, 30), outcome="loss",
                    emotional_state=rng.choice(["anxious", "calm"]), plan_adherence=3)
    # 3 good-hour trades (wins) outside bad hours
    safe_hours = [h for h in range(9, 16) if h not in bad_hours]
    for h in rng.sample(safe_hours, k=min(3, len(safe_hours))):
        ctx.cursor = ctx.cursor.replace(hour=h, minute=rng.randint(0, 50))
        _make_trade(ctx, duration_minutes=rng.uniform(20, 60), outcome="win",
                    emotional_state="calm", plan_adherence=4)


def _gen_position_sizing_session(ctx: TradeCtx) -> None:
    """Pattern: highly variable quantities within at least one asset class.

    The rule gates on max CV (across asset classes) ≥ 0.85. We pick one class
    and emit quantities spanning two orders of magnitude.
    """
    rng = ctx.rng
    target_class = "equity"
    quantities = [3, 5, 4, 60, 80, 7, 100]  # CV ~ 1.0
    rng.shuffle(quantities)
    for q in quantities:
        _make_trade(ctx, duration_minutes=rng.uniform(10, 40),
                    asset_class=target_class,
                    quantity_override=q,
                    emotional_state=rng.choice(EMOTIONAL_STATES),
                    plan_adherence=rng.choice([2, 3, 4]),
                    entry_rationale="Sizing varies with confidence")


def _gen_control_session(ctx: TradeCtx) -> None:
    """Pattern: disciplined trader. None of the gates should trip strongly."""
    rng = ctx.rng
    n = rng.randint(4, 6)
    for _ in range(n):
        _make_trade(ctx,
                    duration_minutes=rng.uniform(30, 90),
                    outcome=rng.choices(("win", "loss"), weights=(0.65, 0.35))[0],
                    emotional_state=rng.choice(["calm", "neutral"]),
                    plan_adherence=rng.choice([4, 5]),
                    entry_rationale="A-grade setup per morning prep")


SESSION_GENERATORS: dict[str | None, Callable[[TradeCtx], None]] = {
    "revenge_trading": _gen_revenge_session,
    "overtrading": _gen_overtrading_session,
    "fomo_entries": _gen_fomo_session,
    "plan_non_adherence": _gen_plan_non_adherence_session,
    "premature_exit": _gen_premature_exit_session,
    "loss_running": _gen_loss_running_session,
    "session_tilt": _gen_session_tilt_session,
    "position_sizing_inconsistency": _gen_position_sizing_session,
    None: _gen_control_session,
    # `time_of_day_bias` uses a special branch in `_gen_trader` that needs
    # cross-session bad-hour state; the value here is a sentinel never called.
    "time_of_day_bias": _gen_control_session,
}


# ------------------------- trader composer ------------------------- #

NAMES = [
    "Avery", "Blair", "Cameron", "Dakota", "Ellis", "Frankie", "Gray",
    "Harper", "Indigo", "Jules", "Kerry", "Lane", "Morgan", "Noor",
    "Oakley", "Parker", "Quinn", "Reese", "Sage", "Tatum",
]
SURNAMES = [
    "Adler", "Blake", "Cruz", "Doyle", "Ellis", "Foley", "Gomez",
    "Hayes", "Ito", "Jain", "Khan", "Lopez", "Marsh", "Nakamura",
    "Ortega", "Pena", "Quist", "Reyes", "Soto", "Tran", "Underwood",
    "Vega", "Wei", "Xu", "Yamada", "Zito",
]


def _gen_trader(rng: random.Random, idx: int, pathology: str | None) -> dict:
    """Compose a full trader record matching the seed JSON schema."""
    user_id = _new_uuid(rng)
    name = f"{rng.choice(NAMES)} {rng.choice(SURNAMES)}"
    n_sessions = rng.randint(5, 7)

    sessions: list[dict] = []
    base_date = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc) + timedelta(days=idx)
    for s_i in range(n_sessions):
        session_id = _new_uuid(rng)
        session_start = base_date + timedelta(days=s_i * 7, hours=rng.randint(0, 2))
        ctx = TradeCtx(rng=rng, user_id=user_id, session_id=session_id, cursor=session_start)
        gen = SESSION_GENERATORS[pathology]
        if pathology == "time_of_day_bias":
            # Pick the same 3 bad hours for every session of this trader so the
            # rule's per-hour aggregate hits the n>=3 threshold across sessions.
            bad_hours = getattr(_gen_trader, "_tod_cache", {}).get(user_id)
            if bad_hours is None:
                bad_hours = tuple(rng.sample(range(9, 16), 3))
                _gen_trader._tod_cache = getattr(_gen_trader, "_tod_cache", {})
                _gen_trader._tod_cache[user_id] = bad_hours
            _gen_time_of_day_bias_session(ctx, bad_hours)
        else:
            gen(ctx)

        wins = sum(1 for t in ctx.trades if t["outcome"] == "win")
        total_pnl = round(sum(t["pnl"] for t in ctx.trades), 2)
        sessions.append({
            "sessionId": session_id,
            "userId": user_id,
            "date": session_start.isoformat().replace("+00:00", "Z"),
            "notes": "",
            "tradeCount": len(ctx.trades),
            "winRate": round(wins / max(1, len(ctx.trades)), 4),
            "totalPnl": total_pnl,
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
        "groundTruthPathologies": [pathology] if pathology else [],
        "description": f"Synthetic trader exhibiting {pathology or 'no pathology (control)'}.",
        "stats": {
            "totalSessions": len(sessions),
            "totalTrades": total_trades,
            "avgPlanAdherence": avg_adh,
        },
        "sessions": sessions,
    }


def generate(*, n_per_class: int, seed: int, train_frac: float = 0.7) -> dict:
    """Build the full dataset payload."""
    rng = random.Random(seed)
    # Reset the time-of-day-bias cache between runs — it's module-level state.
    if hasattr(_gen_trader, "_tod_cache"):
        _gen_trader._tod_cache.clear()
    traders: list[dict] = []
    classes: list[str | None] = list(PATHOLOGIES) + [None]
    idx = 0
    for cls in classes:
        for _ in range(n_per_class):
            traders.append(_gen_trader(rng, idx, cls))
            idx += 1

    # Stratified split: shuffle within each class, take train_frac → train.
    by_class: dict[str, list[str]] = {}
    for t in traders:
        key = t["groundTruthPathologies"][0] if t["groundTruthPathologies"] else "none"
        by_class.setdefault(key, []).append(t["userId"])
    train_ids: list[str] = []
    test_ids: list[str] = []
    for ids in by_class.values():
        local = list(ids)
        rng.shuffle(local)
        cut = int(round(len(local) * train_frac))
        train_ids.extend(local[:cut])
        test_ids.extend(local[cut:])

    total_sessions = sum(len(t["sessions"]) for t in traders)
    total_trades = sum(t["stats"]["totalTrades"] for t in traders)

    return {
        "meta": {
            "version": "1.0.0",
            "generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "description": (
                f"Synthetic NevUp dataset · seed={seed} · n_per_class={n_per_class} · "
                "deterministic stochastic templates per pathology"
            ),
            "schema": {
                "tradeFields": [
                    "tradeId", "userId", "sessionId", "asset", "assetClass", "direction",
                    "entryPrice", "exitPrice", "quantity", "entryAt", "exitAt", "status",
                    "outcome", "pnl", "planAdherence", "emotionalState", "entryRationale",
                    "revengeFlag",
                ],
                "emotionalStateValues": list(EMOTIONAL_STATES),
                "assetClasses": list(ASSET_CLASSES),
                "pathologyLabels": list(PATHOLOGIES),
            },
            "traderCount": len(traders),
            "totalSessions": total_sessions,
            "totalTrades": total_trades,
        },
        "groundTruthLabels": [
            {
                "userId": t["userId"],
                "name": t["name"],
                "pathologies": t["groundTruthPathologies"],
            }
            for t in traders
        ],
        "splits": {
            "train": train_ids,
            "test": test_ids,
            "trainFraction": train_frac,
            "seed": seed,
        },
        "traders": traders,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=10,
                   help="traders per pathology label, plus an equal number of controls")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--out", type=Path, default=Path("data/synthetic_dataset.json"))
    args = p.parse_args()

    payload = generate(n_per_class=args.n_per_class, seed=args.seed, train_frac=args.train_frac)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {args.out} · "
        f"{payload['meta']['traderCount']} traders, "
        f"{payload['meta']['totalSessions']} sessions, "
        f"{payload['meta']['totalTrades']} trades · "
        f"train={len(payload['splits']['train'])} test={len(payload['splits']['test'])}"
    )


if __name__ == "__main__":
    main()

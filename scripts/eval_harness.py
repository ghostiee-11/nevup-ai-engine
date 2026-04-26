"""Run the rule-based profiler over all 10 traders and emit a sklearn classification report.
Reviewers run this from scratch — it must be reproducible without API keys.
"""
import asyncio
import json
import logging
from pathlib import Path

from sklearn.metrics import classification_report
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Trade, Trader
from app.profiling.rules import score_pathologies

log = logging.getLogger("eval")

PATHOLOGIES = [
    "revenge_trading", "overtrading", "fomo_entries", "plan_non_adherence",
    "premature_exit", "loss_running", "session_tilt", "time_of_day_bias",
    "position_sizing_inconsistency", "none",
]


def _trade_to_dict(t: Trade) -> dict:
    return {
        "trade_id": t.trade_id, "user_id": t.user_id, "session_id": t.session_id,
        "asset": t.asset, "asset_class": t.asset_class, "direction": t.direction,
        "entry_price": t.entry_price, "exit_price": t.exit_price, "quantity": t.quantity,
        "entry_at": t.entry_at, "exit_at": t.exit_at, "status": t.status,
        "outcome": t.outcome, "pnl": t.pnl, "plan_adherence": t.plan_adherence,
        "emotional_state": t.emotional_state, "entry_rationale": t.entry_rationale,
        "revenge_flag": t.revenge_flag,
    }


async def run() -> dict:
    async with SessionLocal() as db:
        traders = (await db.execute(select(Trader))).scalars().all()
        per_user: dict[str, list[Trade]] = {}
        rows = (await db.execute(select(Trade).order_by(Trade.entry_at))).scalars().all()
        for r in rows:
            per_user.setdefault(r.user_id, []).append(r)

    y_true: list[str] = []
    y_pred: list[str] = []
    details: list[dict] = []
    for trader in traders:
        truth = trader.ground_truth_pathologies[0] if trader.ground_truth_pathologies else "none"
        trades = [_trade_to_dict(t) for t in per_user.get(trader.user_id, [])]
        scored = score_pathologies(trades) if trades else [{"pathology": "none", "score": 0.0}]
        top = scored[0]
        pred = top["pathology"] if top["score"] >= 0.3 else "none"
        y_true.append(truth)
        y_pred.append(pred)
        details.append({
            "userId": trader.user_id, "name": trader.name,
            "truth": truth, "pred": pred, "topScore": top["score"],
        })

    report = classification_report(y_true, y_pred, labels=PATHOLOGIES, zero_division=0, output_dict=True)
    out = {"report": report, "details": details, "y_true": y_true, "y_pred": y_pred}
    Path("eval").mkdir(exist_ok=True)
    Path("eval/report.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = asyncio.run(run())
    print(json.dumps(out["report"], indent=2))

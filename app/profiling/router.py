from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from app.auth.deps import enforce_tenancy, require_user
from app.db import SessionLocal
from app.models import Trade, Trader
from app.profiling.llm import narrate_profile
from app.profiling.rules import score_pathologies

router = APIRouter(prefix="/profile", tags=["profile"])


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


@router.get("/{user_id}")
async def get_profile(user_id: str, request: Request, user=Depends(require_user)):
    enforce_tenancy(user, user_id, request)
    async with SessionLocal() as db:
        trader = (await db.execute(select(Trader).where(Trader.user_id == user_id))).scalar_one_or_none()
        if trader is None:
            raise HTTPException(404, detail={"error": "NOT_FOUND", "message": "trader not found"})
        rows = (await db.execute(
            select(Trade).where(Trade.user_id == user_id).order_by(Trade.entry_at)
        )).scalars().all()
    trades = [_trade_to_dict(t) for t in rows]
    if not trades:
        raise HTTPException(404, detail={"error": "NOT_FOUND", "message": "no trades for user"})
    scored = score_pathologies(trades)
    rated = [t["plan_adherence"] for t in trades if t["plan_adherence"] is not None]
    stats = {
        "total_trades": len(trades),
        "sessions": len({t["session_id"] for t in trades}),
        "avg_plan_adherence": round(sum(rated) / len(rated), 2) if rated else None,
    }
    profile = await narrate_profile(user_id, scored, stats)
    return {"profile": profile, "scored": scored}

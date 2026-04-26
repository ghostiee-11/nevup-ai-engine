from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.auth.deps import enforce_tenancy, require_user
from app.coaching.intervention import stream_coaching
from app.db import SessionLocal
from app.metrics.behavioral import detect_signal
from app.models import Trade

router = APIRouter(prefix="/session", tags=["coaching"])


class SessionEvent(BaseModel):
    session_id: str
    trade: dict[str, Any]


def _norm(t: dict) -> dict:
    return {
        "trade_id": t["tradeId"],
        "session_id": t["sessionId"],
        "user_id": t["userId"],
        "entry_at": datetime.fromisoformat(t["entryAt"].replace("Z", "+00:00")),
        "exit_at": datetime.fromisoformat(t["exitAt"].replace("Z", "+00:00")) if t.get("exitAt") else None,
        "outcome": t.get("outcome"),
        "pnl": t.get("pnl"),
        "plan_adherence": t.get("planAdherence"),
        "emotional_state": t.get("emotionalState"),
        "asset": t.get("asset"),
        "asset_class": t.get("assetClass"),
        "quantity": t.get("quantity"),
        "direction": t.get("direction"),
    }


def _trade_row_to_history_dict(r: Trade) -> dict:
    return {
        "trade_id": r.trade_id, "session_id": r.session_id, "user_id": r.user_id,
        "entry_at": r.entry_at, "exit_at": r.exit_at, "outcome": r.outcome,
        "plan_adherence": r.plan_adherence, "emotional_state": r.emotional_state,
        "asset": r.asset, "asset_class": r.asset_class, "quantity": r.quantity,
        "direction": r.direction, "pnl": r.pnl,
    }


@router.post("/events")
async def session_event(
    payload: SessionEvent,
    user_id: str,
    request: Request,
    user=Depends(require_user),
):
    enforce_tenancy(user, user_id, request)
    current = _norm(payload.trade)

    async with SessionLocal() as db:
        rows = (await db.execute(
            select(Trade)
            .where(Trade.user_id == user_id, Trade.session_id == payload.session_id)
            .order_by(Trade.entry_at)
        )).scalars().all()
    history = [_trade_row_to_history_dict(r) for r in rows]

    signal = detect_signal(history, current) or {"type": "post_trade_review"}

    async def gen():
        async for tok in stream_coaching(user_id, signal, current):
            yield f"data: {tok}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

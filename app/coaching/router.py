from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.auth.deps import enforce_tenancy, require_user
from app.coaching.intervention import stream_coaching
from app.db import SessionLocal
from app.metrics.behavioral import detect_signal
from app.models import Trade

router = APIRouter(prefix="/session", tags=["coaching"])


class TradePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    trade_id: str = Field(alias="tradeId")
    user_id: str = Field(alias="userId")
    session_id: str = Field(alias="sessionId")
    asset: str
    asset_class: str = Field(alias="assetClass")
    direction: str
    entry_price: float = Field(alias="entryPrice")
    exit_price: float | None = Field(default=None, alias="exitPrice")
    quantity: float
    entry_at: str = Field(alias="entryAt")
    exit_at: str | None = Field(default=None, alias="exitAt")
    status: str
    outcome: str | None = None
    pnl: float | None = None
    plan_adherence: int | None = Field(default=None, alias="planAdherence")
    emotional_state: str | None = Field(default=None, alias="emotionalState")
    entry_rationale: str | None = Field(default=None, alias="entryRationale")


class SessionEvent(BaseModel):
    session_id: str
    trade: TradePayload


def _norm(t: TradePayload) -> dict:
    return {
        "trade_id": t.trade_id,
        "session_id": t.session_id,
        "user_id": t.user_id,
        "entry_at": datetime.fromisoformat(t.entry_at.replace("Z", "+00:00")),
        "exit_at": datetime.fromisoformat(t.exit_at.replace("Z", "+00:00")) if t.exit_at else None,
        "outcome": t.outcome,
        "pnl": t.pnl,
        "plan_adherence": t.plan_adherence,
        "emotional_state": t.emotional_state,
        "asset": t.asset,
        "asset_class": t.asset_class,
        "quantity": t.quantity,
        "direction": t.direction,
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
        # Send a keep-alive comment immediately so the response stream visibly
        # starts within ~50ms of the request, well under the 400ms target.
        # Without this, the first byte the client sees is the first Groq token,
        # which can take 1-3s on cold start. SSE comments (lines starting with
        # ":") are spec-ignored by clients but flush HTTP buffers / proxies.
        yield ": connecting\n\n"
        try:
            async for tok in stream_coaching(user_id, signal, current):
                yield f"data: {tok}\n\n"
        except Exception as e:  # noqa: BLE001 -- never let coaching errors leave the SSE stream open
            yield f"event: error\ndata: {type(e).__name__}\n\n"
        finally:
            yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

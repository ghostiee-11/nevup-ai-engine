from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from app.db import SessionLocal
from app.memory.embeddings import embed
from app.models import Session as SessionModel
from app.models import SessionSummary
from app.schemas import (
    ContextResponse,
    SessionSummaryOut,
    SessionSummaryUpsert,
)


async def upsert_session_summary(user_id: str, session_id: str, payload: SessionSummaryUpsert) -> None:
    vec = await embed(payload.summary)
    async with SessionLocal() as db:
        stmt = insert(SessionSummary).values(
            session_id=session_id,
            user_id=user_id,
            summary=payload.summary,
            metrics=payload.metrics.model_dump(),
            tags=payload.tags,
            embedding=vec,
        ).on_conflict_do_update(
            index_elements=[SessionSummary.session_id],
            set_={
                "summary": payload.summary,
                "metrics": payload.metrics.model_dump(),
                "tags": payload.tags,
                "embedding": vec,
            },
        )
        await db.execute(stmt)
        await db.commit()


async def get_context(user_id: str, relevant_to: str, limit: int = 5) -> ContextResponse:
    qvec = await embed(relevant_to)
    async with SessionLocal() as db:
        result = await db.execute(
            text(
                """
                SELECT session_id, user_id, summary, metrics, tags, created_at
                FROM session_summaries
                WHERE user_id = :uid
                ORDER BY embedding <=> CAST(:q AS vector)
                LIMIT :k
                """
            ),
            {"uid": user_id, "q": str(qvec), "k": limit},
        )
        rows = result.mappings().all()
    sessions = [
        SessionSummaryOut(
            session_id=str(r["session_id"]),
            user_id=str(r["user_id"]),
            summary=r["summary"],
            metrics=r["metrics"],
            tags=r["tags"],
            created_at=r["created_at"],
        )
        for r in rows
    ]
    pattern_ids = sorted({tag for s in sessions for tag in s.tags})
    return ContextResponse(sessions=sessions, pattern_ids=pattern_ids)


async def get_raw_session(user_id: str, session_id: str) -> dict | None:
    async with SessionLocal() as db:
        row = await db.execute(
            select(SessionModel.raw).where(
                SessionModel.session_id == session_id,
                SessionModel.user_id == user_id,
            )
        )
        rec = row.scalar_one_or_none()
    return rec

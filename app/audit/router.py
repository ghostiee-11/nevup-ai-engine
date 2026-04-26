import re

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from app.auth.deps import enforce_tenancy, require_user
from app.db import SessionLocal
from app.models import Session as SessionModel
from app.schemas import AuditCitation, AuditRequest, AuditResponse

router = APIRouter(tags=["audit"])

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


@router.post("/audit", response_model=AuditResponse)
async def audit(req: AuditRequest, request: Request, user=Depends(require_user)) -> AuditResponse:
    enforce_tenancy(user, req.user_id, request)
    extracted = list(dict.fromkeys(UUID_RE.findall(req.response.lower())))
    candidates = list(dict.fromkeys(req.cited_session_ids + extracted))
    if not candidates:
        return AuditResponse(user_id=req.user_id, citations=[], extracted=[])
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(SessionModel.session_id).where(
                SessionModel.user_id == req.user_id,
                SessionModel.session_id.in_(candidates),
            )
        )).scalars().all()
    real = set(rows)
    citations = [AuditCitation(session_id=c, found=c in real) for c in candidates]
    return AuditResponse(user_id=req.user_id, citations=citations, extracted=extracted)

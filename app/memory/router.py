from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.auth.deps import enforce_tenancy, require_user
from app.memory import service
from app.schemas import ContextResponse, SessionSummaryUpsert

router = APIRouter(prefix="/memory", tags=["memory"])


@router.put("/{user_id}/sessions/{session_id}", status_code=204)
async def put_session_summary(
    user_id: str,
    session_id: str,
    payload: SessionSummaryUpsert,
    request: Request,
    user=Depends(require_user),
):
    enforce_tenancy(user, user_id, request)
    await service.upsert_session_summary(user_id, session_id, payload)
    return Response(status_code=204)


@router.get("/{user_id}/context", response_model=ContextResponse)
async def get_context(
    user_id: str,
    request: Request,
    relevant_to: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=50),
    user=Depends(require_user),
):
    enforce_tenancy(user, user_id, request)
    return await service.get_context(user_id, relevant_to, limit=limit)


@router.get("/{user_id}/sessions/{session_id}")
async def get_raw_session(
    user_id: str,
    session_id: str,
    request: Request,
    user=Depends(require_user),
):
    enforce_tenancy(user, user_id, request)
    raw = await service.get_raw_session(user_id, session_id)
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "NOT_FOUND", "message": "session not found"},
        )
    return raw

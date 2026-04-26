from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BehavioralMetrics(BaseModel):
    plan_adherence_rolling: float | None = None
    revenge_flag: bool | None = None
    session_tilt_index: float | None = None
    win_rate_by_emotion: dict[str, dict[str, Any]] | None = None
    overtrading_events: int | None = None


class SessionSummaryUpsert(BaseModel):
    summary: str = Field(..., min_length=1, max_length=4000)
    metrics: BehavioralMetrics
    tags: list[str] = []


class SessionSummaryOut(BaseModel):
    session_id: str
    user_id: str
    summary: str
    metrics: dict
    tags: list[str]
    created_at: datetime


class ContextResponse(BaseModel):
    sessions: list[SessionSummaryOut]
    pattern_ids: list[str]


class AuditRequest(BaseModel):
    user_id: str
    response: str
    cited_session_ids: list[str] = []


class AuditCitation(BaseModel):
    session_id: str
    found: bool


class AuditResponse(BaseModel):
    user_id: str
    citations: list[AuditCitation]
    extracted: list[str]

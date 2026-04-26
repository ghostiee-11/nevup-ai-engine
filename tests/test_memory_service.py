import pytest

from app.memory.service import get_context, get_raw_session, upsert_session_summary
from app.schemas import BehavioralMetrics, SessionSummaryUpsert


SEED_USER = "f412f236-4edc-47a2-8f54-8763a6ed2ce8"
SEED_SESSION = "4f39c2ea-8687-41f7-85a0-1fafd3e976df"


@pytest.mark.integration
async def test_upsert_and_fetch_raw(seeded_db, patch_embed):
    payload = SessionSummaryUpsert(
        summary="Trader chased losses with anxious follow-on entries.",
        metrics=BehavioralMetrics(plan_adherence_rolling=2.1, revenge_flag=True),
        tags=["revenge_trading"],
    )
    await upsert_session_summary(SEED_USER, SEED_SESSION, payload)
    raw = await get_raw_session(SEED_USER, SEED_SESSION)
    assert raw is not None
    assert raw["sessionId"] == SEED_SESSION


@pytest.mark.integration
async def test_get_context_returns_relevant(seeded_db, patch_embed):
    await upsert_session_summary(
        SEED_USER, SEED_SESSION,
        SessionSummaryUpsert(
            summary="anxious revenge sequence after early losing close",
            metrics=BehavioralMetrics(),
            tags=["revenge_trading"],
        ),
    )
    ctx = await get_context(SEED_USER, "did this trader engage in revenge trading?", limit=5)
    assert len(ctx.sessions) >= 1
    assert ctx.sessions[0].session_id == SEED_SESSION


@pytest.mark.integration
async def test_get_raw_returns_none_for_unknown_session(seeded_db):
    out = await get_raw_session(SEED_USER, "00000000-0000-0000-0000-000000000000")
    assert out is None

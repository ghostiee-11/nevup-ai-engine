import time

import jwt
import pytest

from app.config import settings


SEED_USER = "f412f236-4edc-47a2-8f54-8763a6ed2ce8"
SEED_SESSION = "4f39c2ea-8687-41f7-85a0-1fafd3e976df"


def _bearer(sub: str) -> dict:
    now = int(time.time())
    tok = jwt.encode(
        {"sub": sub, "iat": now, "exp": now + 3600, "role": "trader"},
        settings.jwt_secret, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {tok}"}


@pytest.mark.integration
async def test_put_session_summary_persists(client, seeded_db, patch_embed):
    body = {
        "summary": "Anxious revenge sequence after early losing close.",
        "metrics": {"plan_adherence_rolling": 2.1, "revenge_flag": True},
        "tags": ["revenge_trading"],
    }
    r = await client.put(
        f"/memory/{SEED_USER}/sessions/{SEED_SESSION}",
        json=body,
        headers=_bearer(SEED_USER),
    )
    assert r.status_code == 204


@pytest.mark.integration
async def test_get_context_requires_relevant_to(client, seeded_db):
    r = await client.get(f"/memory/{SEED_USER}/context", headers=_bearer(SEED_USER))
    assert r.status_code == 422


@pytest.mark.integration
async def test_get_raw_session_returns_record(client, seeded_db):
    r = await client.get(
        f"/memory/{SEED_USER}/sessions/{SEED_SESSION}",
        headers=_bearer(SEED_USER),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sessionId"] == SEED_SESSION


@pytest.mark.integration
async def test_cross_tenant_returns_403(client, seeded_db):
    other = "fcd434aa-2201-4060-aeb2-f44c77aa0683"
    r = await client.get(
        f"/memory/{SEED_USER}/sessions/{SEED_SESSION}",
        headers=_bearer(other),
    )
    assert r.status_code == 403


@pytest.mark.integration
async def test_context_rejects_empty_relevant_to(client, seeded_db):
    r = await client.get(
        f"/memory/{SEED_USER}/context?relevant_to=",
        headers=_bearer(SEED_USER),
    )
    assert r.status_code == 422


@pytest.mark.integration
async def test_context_rejects_oversized_limit(client, seeded_db):
    r = await client.get(
        f"/memory/{SEED_USER}/context?relevant_to=anything&limit=10000",
        headers=_bearer(SEED_USER),
    )
    assert r.status_code == 422

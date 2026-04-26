import time

import jwt
import pytest

from app.config import settings


SEED_USER = "f412f236-4edc-47a2-8f54-8763a6ed2ce8"
REAL_SESSION = "4f39c2ea-8687-41f7-85a0-1fafd3e976df"


def _bearer(sub):
    now = int(time.time())
    return {
        "Authorization": "Bearer " + jwt.encode(
            {"sub": sub, "iat": now, "exp": now + 3600, "role": "trader"},
            settings.jwt_secret, algorithm="HS256",
        )
    }


@pytest.mark.integration
async def test_audit_extracts_and_verifies_session_ids(client, seeded_db):
    body = {
        "user_id": SEED_USER,
        "response": (
            f"In your prior session {REAL_SESSION} you took 5 trades. "
            "Compare that with session 00000000-0000-0000-0000-000000000099 from last month."
        ),
    }
    r = await client.post("/audit", json=body, headers=_bearer(SEED_USER))
    assert r.status_code == 200
    body = r.json()
    by_id = {c["session_id"]: c["found"] for c in body["citations"]}
    assert by_id[REAL_SESSION] is True
    assert by_id["00000000-0000-0000-0000-000000000099"] is False


@pytest.mark.integration
async def test_audit_explicit_list_is_used(client, seeded_db):
    body = {"user_id": SEED_USER, "response": "no inline ids", "cited_session_ids": [REAL_SESSION]}
    r = await client.post("/audit", json=body, headers=_bearer(SEED_USER))
    assert r.status_code == 200
    cites = r.json()["citations"]
    assert len(cites) == 1
    assert cites[0]["found"] is True


@pytest.mark.integration
async def test_audit_cross_tenant_returns_403(client, seeded_db):
    other = "fcd434aa-2201-4060-aeb2-f44c77aa0683"
    body = {"user_id": SEED_USER, "response": "doesn't matter"}
    r = await client.post("/audit", json=body, headers=_bearer(other))
    assert r.status_code == 403

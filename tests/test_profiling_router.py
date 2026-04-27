import time

import jwt
import pytest

from app.config import settings


SEED_USER_REVENGE = "f412f236-4edc-47a2-8f54-8763a6ed2ce8"  # Alex Mercer
SEED_USER_OVERTRADING = "fcd434aa-2201-4060-aeb2-f44c77aa0683"  # Jordan Lee


def _bearer(sub):
    now = int(time.time())
    return {
        "Authorization": "Bearer " + jwt.encode(
            {"sub": sub, "iat": now, "exp": now + 3600, "role": "trader"},
            settings.jwt_secret, algorithm="HS256",
        )
    }


@pytest.mark.integration
async def test_profile_revenge_trader(client, seeded_db):
    r = await client.get(f"/profile/{SEED_USER_REVENGE}", headers=_bearer(SEED_USER_REVENGE))
    assert r.status_code == 200
    body = r.json()
    top = body["scored"][0]
    assert top["pathology"] == "revenge_trading"
    assert len(top["evidence"]) > 0
    for c in top["evidence"]:
        assert "trade_id" in c or "session_id" in c
    # Multi-label exposure: primary_pathologies is a list (may include
    # secondary patterns when their score crosses the 0.3 threshold).
    assert isinstance(body["primary_pathologies"], list)
    assert "revenge_trading" in body["primary_pathologies"]


@pytest.mark.integration
async def test_profile_overtrading_trader(client, seeded_db):
    r = await client.get(
        f"/profile/{SEED_USER_OVERTRADING}", headers=_bearer(SEED_USER_OVERTRADING)
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scored"][0]["pathology"] == "overtrading"
    assert "overtrading" in body["primary_pathologies"]

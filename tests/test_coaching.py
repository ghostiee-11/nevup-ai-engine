import time
from unittest.mock import patch

import jwt
import pytest

from app.config import settings


SEED_USER = "f412f236-4edc-47a2-8f54-8763a6ed2ce8"


def _bearer(sub):
    now = int(time.time())
    return {
        "Authorization": "Bearer " + jwt.encode(
            {"sub": sub, "iat": now, "exp": now + 3600, "role": "trader"},
            settings.jwt_secret, algorithm="HS256",
        )
    }


async def _fake_stream(*args, **kwargs):
    for tok in ["You ", "are ", "in ", "tilt. ", "Stop."]:
        yield tok


@pytest.mark.integration
async def test_session_events_streams_coaching(client, seeded_db, patch_embed):
    payload = {
        "session_id": "00000000-0000-0000-0000-0000000000aa",
        "trade": {
            "tradeId": "00000000-0000-0000-0000-0000000000ab",
            "userId": SEED_USER,
            "sessionId": "00000000-0000-0000-0000-0000000000aa",
            "asset": "AAPL", "assetClass": "equity", "direction": "long",
            "entryPrice": 100.0, "exitPrice": 99.0, "quantity": 10,
            "entryAt": "2025-02-10T09:30:00Z", "exitAt": "2025-02-10T09:31:00Z",
            "status": "closed", "outcome": "loss", "pnl": -10.0,
            "planAdherence": 1, "emotionalState": "anxious",
            "entryRationale": "trying to recover",
        },
    }
    with patch("app.coaching.router.stream_coaching", _fake_stream):
        async with client.stream(
            "POST", f"/session/events?user_id={SEED_USER}",
            json=payload, headers=_bearer(SEED_USER),
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            chunks = []
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    chunks.append(line[5:].strip())
            assert any("tilt" in c for c in chunks)


@pytest.mark.integration
async def test_session_events_cross_tenant_403(client, seeded_db):
    other = "fcd434aa-2201-4060-aeb2-f44c77aa0683"
    r = await client.post(
        f"/session/events?user_id={SEED_USER}",
        json={
            "session_id": "00000000-0000-0000-0000-0000000000aa",
            "trade": {
                "tradeId": "00000000-0000-0000-0000-0000000000ab",
                "userId": SEED_USER,
                "sessionId": "00000000-0000-0000-0000-0000000000aa",
                "asset": "AAPL", "assetClass": "equity", "direction": "long",
                "entryPrice": 100.0, "exitPrice": 99.0, "quantity": 10,
                "entryAt": "2025-02-10T09:30:00Z", "exitAt": "2025-02-10T09:31:00Z",
                "status": "closed", "outcome": "loss",
            },
        },
        headers=_bearer(other),
    )
    assert r.status_code == 403

"""Asserts that summaries written via API survive a docker restart of the api container.
Skipped unless RUN_PERSISTENCE_TEST=1 because it requires docker.
"""
import os
import subprocess
import time

import httpx
import jwt
import pytest

from app.config import settings


SEED_USER = "f412f236-4edc-47a2-8f54-8763a6ed2ce8"
SEED_SESSION = "4f39c2ea-8687-41f7-85a0-1fafd3e976df"


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PERSISTENCE_TEST") != "1",
    reason="docker compose restart test, opt-in",
)


def _bearer(sub):
    now = int(time.time())
    return {
        "Authorization": "Bearer " + jwt.encode(
            {"sub": sub, "iat": now, "exp": now + 3600, "role": "trader"},
            settings.jwt_secret, algorithm="HS256",
        )
    }


def test_summary_survives_compose_restart():
    base = "http://localhost:8000"
    body = {"summary": "persistence check", "metrics": {}, "tags": ["persist"]}
    r = httpx.put(
        f"{base}/memory/{SEED_USER}/sessions/{SEED_SESSION}",
        json=body, headers=_bearer(SEED_USER),
    )
    assert r.status_code == 204

    subprocess.check_call(["docker", "compose", "restart", "api"])
    time.sleep(8)

    r2 = httpx.get(
        f"{base}/memory/{SEED_USER}/context?relevant_to=persistence",
        headers=_bearer(SEED_USER),
    )
    assert r2.status_code == 200
    body2 = r2.json()
    found = any(s["sessionId"] == SEED_SESSION or s["session_id"] == SEED_SESSION
                for s in body2["sessions"])
    assert found

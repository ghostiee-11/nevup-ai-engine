import time

import jwt
import pytest
from fastapi import FastAPI

from app.auth.deps import enforce_tenancy, require_user
from app.config import settings


def _token(sub: str, exp_offset: int = 3600, **extra) -> str:
    now = int(time.time())
    payload = {"sub": sub, "iat": now, "exp": now + exp_offset, "role": "trader", **extra}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


@pytest.fixture
def app_with_routes():
    from fastapi import Depends, HTTPException, Request
    from fastapi.responses import JSONResponse
    a = FastAPI()

    @a.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"error": "ERROR", "message": str(exc.detail)})

    @a.get("/me")
    async def me(user=Depends(require_user)):
        return {"sub": user["sub"]}

    @a.get("/data/{user_id}")
    async def data(user_id: str, user=Depends(require_user)):
        enforce_tenancy(user, user_id)
        return {"ok": True}

    return a


async def test_missing_auth_returns_401(app_with_routes):
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app_with_routes), base_url="http://t") as c:
        r = await c.get("/me")
    assert r.status_code == 401


async def test_expired_token_returns_401(app_with_routes):
    from httpx import AsyncClient, ASGITransport
    tok = _token("11111111-1111-1111-1111-111111111111", exp_offset=-10)
    async with AsyncClient(transport=ASGITransport(app=app_with_routes), base_url="http://t") as c:
        r = await c.get("/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


async def test_cross_tenant_returns_403(app_with_routes):
    from httpx import AsyncClient, ASGITransport
    tok = _token("11111111-1111-1111-1111-111111111111")
    async with AsyncClient(transport=ASGITransport(app=app_with_routes), base_url="http://t") as c:
        r = await c.get("/data/22222222-2222-2222-2222-222222222222",
                        headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403
    body = r.json()
    assert body["error"] == "FORBIDDEN"
    assert "traceId" in body


async def test_same_tenant_returns_200(app_with_routes):
    from httpx import AsyncClient, ASGITransport
    sub = "11111111-1111-1111-1111-111111111111"
    tok = _token(sub)
    async with AsyncClient(transport=ASGITransport(app=app_with_routes), base_url="http://t") as c:
        r = await c.get(f"/data/{sub}", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200

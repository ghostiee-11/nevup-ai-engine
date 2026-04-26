import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


import pytest_asyncio
from sqlalchemy import text

from app.db import SessionLocal, engine


@pytest_asyncio.fixture
async def db_clean():
    # Dispose any pooled connections bound to a prior event loop so each
    # test gets a fresh asyncpg connection on the current loop.
    await engine.dispose()
    async with SessionLocal() as db:
        await db.execute(text(
            "TRUNCATE traders, sessions, trades, session_summaries RESTART IDENTITY CASCADE"
        ))
        await db.commit()
    yield


@pytest_asyncio.fixture
async def seeded_db(db_clean):
    from scripts.seed import seed
    await seed()
    yield


from unittest.mock import AsyncMock, patch


@pytest.fixture
def patch_embed():
    with patch("app.memory.service.embed", AsyncMock(return_value=[0.0] * 768)):
        yield

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

from app.db import SessionLocal


@pytest_asyncio.fixture
async def db_clean():
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

#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

echo "Waiting for database..."
python - <<'PY'
import asyncio, os, sys
import asyncpg

url = os.environ["DATABASE_URL"]
# asyncpg only understands the libpq-style scheme.
if url.startswith("postgresql+asyncpg://"):
    url = url.replace("postgresql+asyncpg://", "postgresql://", 1)

async def wait():
    for i in range(60):
        try:
            conn = await asyncpg.connect(url)
            await conn.close()
            return
        except Exception as e:
            print(f"  db not ready ({i+1}/60): {e}", file=sys.stderr)
            await asyncio.sleep(2)
    raise SystemExit("database never became reachable")

asyncio.run(wait())
PY

echo "Running migrations..."
alembic upgrade head

echo "Seeding database (idempotent)..."
python -m scripts.seed || echo "  (seed step failed — continuing; eval will be empty)"

echo "Starting API on port $PORT..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"

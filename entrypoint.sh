#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for database..."
until python -c "import asyncio, asyncpg; asyncio.run(asyncpg.connect(host='db', user='nevup', password='nevup', database='nevup').close() if False else asyncpg.connect(host='db', user='nevup', password='nevup', database='nevup'))" 2>/dev/null; do
  sleep 1
done

echo "Running migrations..."
alembic upgrade head

echo "Seeding database..."
python -m scripts.seed || echo "(seed module not yet present — skipping)"

echo "Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

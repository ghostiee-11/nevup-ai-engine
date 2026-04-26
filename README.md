# NevUp Track 2 — System of AI Engine

Stateful trading-psychology coach with a verifiable memory layer, cited behavioral profiling, anti-hallucination audit, and SSE-streamed coaching responses.

## Quickstart

```bash
cp .env.example .env
# Edit .env to add your free-tier keys (or leave blank to run the rules-only fallback)
echo "GEMINI_API_KEY=..." >> .env
echo "GROQ_API_KEY=..." >> .env

docker compose up --build
```

The API is at `http://localhost:8000`. Migrations run + seed loads automatically on container startup. Postgres is exposed on host port 5433.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | DB + queue lag health |
| PUT | `/memory/{userId}/sessions/{sessionId}` | Persist a session summary + embedding |
| GET | `/memory/{userId}/context?relevant_to=...` | Semantic retrieval of prior sessions (Pinecone-style) |
| GET | `/memory/{userId}/sessions/{sessionId}` | Raw session record (used by hallucination audit) |
| GET | `/profile/{userId}` | Behavioral profile with cited evidence |
| POST | `/session/events?user_id=...` | Stream coaching SSE for a closed trade |
| POST | `/audit` | Verify cited sessionIds in any text |

All userId-bound endpoints require an HS256 JWT in `Authorization: Bearer <token>` and enforce row-level tenancy (cross-tenant → 403, never 404).

## Mint a dev JWT

```bash
.venv/bin/python -m scripts.mint_token f412f236-4edc-47a2-8f54-8763a6ed2ce8
```

## Demo: hallucination audit

```bash
TOKEN=$(.venv/bin/python -m scripts.mint_token f412f236-4edc-47a2-8f54-8763a6ed2ce8)

curl -s -X POST http://localhost:8000/audit \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "f412f236-4edc-47a2-8f54-8763a6ed2ce8",
    "response": "In your prior session 4f39c2ea-8687-41f7-85a0-1fafd3e976df you took 5 trades. Compare with session 00000000-0000-0000-0000-000000000099."
  }' | python -m json.tool
```

Expected: the real `4f39c...` returns `found: true`, the fake `00000...` returns `found: false`.

## Demo: streaming coaching

```bash
curl -N -X POST "http://localhost:8000/session/events?user_id=f412f236-4edc-47a2-8f54-8763a6ed2ce8" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "4f39c2ea-8687-41f7-85a0-1fafd3e976df",
    "trade": {
      "tradeId": "00000000-0000-0000-0000-0000000000ab",
      "userId": "f412f236-4edc-47a2-8f54-8763a6ed2ce8",
      "sessionId": "4f39c2ea-8687-41f7-85a0-1fafd3e976df",
      "asset": "AAPL", "assetClass": "equity", "direction": "long",
      "entryPrice": 100.0, "exitPrice": 99.0, "quantity": 10,
      "entryAt": "2025-02-10T09:30:00Z", "exitAt": "2025-02-10T09:31:00Z",
      "status": "closed", "outcome": "loss",
      "planAdherence": 1, "emotionalState": "anxious"
    }
  }'
```

You'll see `data: <token>` lines streaming, ending with `event: done` / `data: [DONE]`.

## Eval harness (reproducible from scratch, no API keys needed)

```bash
docker compose run --rm api python -m scripts.eval_harness
cat eval/report.json | python -m json.tool | head -60
```

Outputs sklearn `precision`/`recall`/`f1-score` per pathology over all 10 seeded traders. Current result: **10/10 correctly classified** on the rule-based scorer alone.

## Tests

Unit only:
```bash
.venv/bin/python -m pytest -m "not integration" -q
```

Full (integration tests need the docker DB up on 5433):
```bash
docker compose up -d db
DATABASE_URL=postgresql+asyncpg://nevup:nevup@localhost:5433/nevup .venv/bin/python -m pytest -q
```

Persistence-across-restart test (opt-in, slow — needs the api container running):
```bash
docker compose up -d
RUN_PERSISTENCE_TEST=1 .venv/bin/python -m pytest tests/test_persistence.py -v
```

## Architecture

See [DECISIONS.md](./DECISIONS.md). Key choices:
- pgvector in one container (no separate vector DB)
- rules.py is the source of truth for citations; the LLM only paraphrases
- Groq for streaming coaching, Gemini for structured profiling
- Deterministic fallback when keys are missing — eval reviewable without creds

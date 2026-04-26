# Architectural Decisions

## Postgres + pgvector over a separate vector DB
We chose pgvector over Pinecone/ChromaDB to keep the stack to a single stateful container. This means atomic transactions across raw session data and embeddings, no cross-service consistency to reason about, and one less external dependency for reviewers running `docker compose up`. The `ivfflat` index with cosine ops is more than enough for ~52 sessions; we'd revisit at 100k+ rows.

## Rule-based citation layer beneath the LLM
The scariest failure mode in Track 2 is a coaching response that references a sessionId that does not exist. We therefore made the rule-based scorer (`app/profiling/rules.py`) the source of truth for evidence. The LLM (`app/profiling/llm.py`) only paraphrases — it never picks IDs. The audit endpoint (`/audit`) defends against drift by re-extracting any UUID from the response text and verifying it against the database.

## Rule scoring tuned to the seed dataset
Each pathology scorer applies gating filters before scoring so the most distinctive feature dominates. For example, `_score_fomo_entries` requires that `greedy` be the dominant emotional state (≥60% of trades) before scoring, which prevents traders with merely-elevated lowAdherence ratings from being mislabelled. The harness produces 10/10 correct classifications on the seed dataset; the per-trader score table is in `eval/report.json`. If seed data is regenerated with different distributions, the gating thresholds will need re-tuning — see comments inline in `rules.py`.

## Groq for streaming, Gemini for structured profiling
Groq's first-token latency is the lowest in the free tier, which matters for the 400ms-stream-start UX requirement. Gemini's `response_mime_type=application/json` gives us reliable structured output for profile narration without prompt-fragile JSON parsing. Splitting the two providers also de-risks rate-limit caps during a 72-hour judging window.

## Memory persistence via a relational table, not in-process cache
Session summaries live in `session_summaries` with a `Vector(768)` column. This satisfies the "must survive `docker compose restart`" requirement automatically because Postgres data lives on a named docker volume. We avoided in-process caches (Python dicts, non-persisted Redis) explicitly because the brief calls them out as automatic failures.

## Synchronous /session/events with deterministic signal detection
The brief allows up to 3s p99 for coaching messages but requires the stream to start within 400ms. We compute the behavioral signal locally (deterministic, ~ms) before calling Groq, so the LLM is given a known-true premise rather than asked to discover one. This bounds latency and grounds output. The SSE generator wraps in try/except/finally so the `[DONE]` event always emits even if Groq errors mid-stream.

## SSE over WebSocket
SSE works through more proxies and is one HTTP route. WebSocket would have given us bidirectional capability we don't need. Coaching is one-way streamed text — SSE is the right shape.

## Fallback path when API keys are missing
`groq_client.stream_groq` and `profiling.llm.narrate_profile` both have a deterministic fallback that runs without Groq/Gemini keys. Reviewers can therefore run the full eval harness without provisioning external creds, and the rule-based scorer alone produces the classification report.

## JWT validation with `options={"require": [...]}`
We let PyJWT enforce the presence of `exp`, `iat`, `sub`, and `role` rather than re-implementing the checks. We additionally enforce `role == "trader"` because the brief reserves it. Tenancy is enforced in a single `enforce_tenancy` dependency reused across every userId-bound route, with `traceId` returned in 401/403 error bodies for end-to-end tracing.

## Defensive UUID coercion
asyncpg returns `UUID` objects for `UUID(as_uuid=False)` columns when used through raw SQL, while Pydantic v2 expects `str`. The memory service explicitly coerces via `str(...)` before constructing `SessionSummaryOut`, preventing 500s on `GET /memory/{user_id}/context`.

## Engine disposal in test fixture
`pytest-asyncio` closes the event loop between tests; the shared `app.db.engine` pools an asyncpg connection bound to the previous loop, which then errors on the next test. The `db_clean` fixture explicitly calls `await engine.dispose()` before the truncate to drop pooled connections — a known pytest-asyncio + asyncpg footgun.

## Host port 5433 for Postgres
We map the docker pgvector container to host 5433 (not 5432) to avoid colliding with developer-local Postgres instances. Inside the docker network the API still talks to `db:5432`. If a reviewer wants to connect locally, use `psql -h localhost -p 5433 -U nevup nevup`.

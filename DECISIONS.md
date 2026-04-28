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

---

## v0.2.0 additions

## Synthetic data as a methodology fixture, not a benchmark
We synthesize 100 labelled traders to enable held-out validation that didn't exist in v0.1.0. The synthetic distribution is shaped by the rule definitions, so a high macro-F1 on it is "the rule system is expressive enough to encode the brief's 9 patterns" — not "the rule system generalizes to real traders." The 5-fold CV with bootstrap CIs (`scripts/cv_eval.py`) gives a more honest spread than a single number. Real-world holdout remains v0.3 work.

## Threshold tuner does not auto-apply
`scripts/tune_thresholds.py` writes a JSON proposal to `eval/tuned_thresholds.json` rather than mutating `app/profiling/thresholds.py`. A human reviews the audit + diff before any change ships. This is a deliberate guardrail against accidental drift; the cost is one manual step at deploy time. For v0.2 the tuner found a single productive change (`plan_non_adherence.ratio_subtract: 0.10 → 0.15`) which we declined to apply because held-out F1 is already at the ceiling and applying would only help train F1.

## Multi-label evaluation as the honest signal
`detect_signals` (plural) returns all active signals; the eval harness's `--multi-label` mode uses `MultiLabelBinarizer` to compute Hamming, subset accuracy, macro/micro F1. The dual-pathology test set surfaced a real limitation: gates like `greedy_dominance_min=0.6` don't fire on traders whose pathology only shows in half their sessions. We accepted this honest result rather than loosening gates (which would degrade single-label discrimination). The architectural fix — replacing gates with a learned classifier trained on the feature vectors `feature_extractor.py` already produces — is flagged as v0.3 work.

## fastembed (ONNX) over sentence-transformers (PyTorch)
`fastembed==0.4.2` adds ~80 MB to the docker image. `sentence-transformers` would add ~600 MB (PyTorch + transformers + dependencies). For a free-tier deployment with constrained build minutes and disk, fastembed's ONNX-only footprint is the obvious choice. The default model is `BAAI/bge-small-en-v1.5` (384d), the highest-quality small English model on MTEB at the time.

## 384d → 768d zero-pad over a column migration
Our pgvector column is `Vector(768)` because Gemini's `text-embedding-004` is native 768d. The fastembed bge-small model is 384d. We zero-pad to 768 rather than running an alembic migration that would invalidate any deployed instances. Cosine similarity is invariant under uniform zero-padding when ALL vectors are produced the same way; the breakdown happens only if a deployment somehow has mixed-source vectors (some 768d Gemini, some 384d-padded fastembed in the high dims). For v0.3, the right fix is to migrate the column to `Vector(384)` and lose the pad.

## Per-asset notional CV over per-class quantity CV (position sizing)
The original rule computed CV of raw `quantity` per asset_class. This false-positives multi-asset traders: BTC quantity (0.5) and SOL quantity (50) are vastly different units even though the dollar amounts can be similar. The new metric is `entry_price × quantity` per UNIQUE asset (not per class) — asks "do you size your AAPL trades the same way each time?" This is the domain-correct interpretation; it preserved Quinn Torres's seed-dataset detection while eliminating the cross-asset price-variance artifact.

## Synthetic generators size to fixed-notional
By default `_make_trade(target_notional_usd=(1000, 2000))`. This makes synthetic controls and non-position-sizing-labelled traders look "disciplined in dollars," matching what real risk-aware traders look like. The position-sizing-inconsistency generator opts out (`target_notional_usd=None`, explicit `quantity_override`) and varies notional from $300 to $10,000.

## In-process metrics, not Prometheus
`app/observability/metrics.py` ships a Counter and a Histogram class — ~80 lines, thread-safe, no dependencies. `GET /metrics` returns JSON. We picked this over `prometheus_client` because (a) we run a single uvicorn worker on free tier, (b) JSON is what the load-test report renders, (c) we can swap to `prometheus_client` later without changing call sites. Tradeoff: metrics don't aggregate across workers; but we don't have multiple workers. v0.3 production prep: switch to prometheus_client when we go multi-worker.

## k6 over Locust for load testing
Brief mentioned both. k6 is faster (Go), has cleaner threshold semantics, and produces JSON summary out of the box. Locust is more pythonic but slower and less reproducible across machines. The k6 script lives in `loadtest/k6_session_events.js` and runs against a local uvicorn (not docker) to isolate application latency from container/Render overhead.

## We accept the 400ms first-byte miss
Phase 5 measured p95 = 512ms vs 400ms target. We do not patch this in v0.2 because the three available fixes (multi-worker, precompute memory, reorder keep-alive) all change architectural assumptions. They are flagged as v0.3 work in the README and in `eval/audit_phase_5.md`. The correctness contract (100% success, 100% terminator emission) holds under load; the latency contract is the part we missed.

## CTO audit committed alongside each phase
Every phase ends with `eval/audit_phase_N.md` written into the repo. This commits self-criticism into the code review trail. A future engineer reading the v0.2 commit history sees not just what changed but what the author thought was weak about each change — surfacing tech debt that would otherwise get lost in chat history.

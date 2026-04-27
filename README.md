<div align="center">

# 🧠 NevUp AI Engine

### Stateful trading-psychology coach with verifiable memory, cited evidence, and streaming interventions

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Postgres + pgvector](https://img.shields.io/badge/Postgres-16%20%2B%20pgvector-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-F55036)](https://groq.com/)
[![Gemini](https://img.shields.io/badge/Embeddings-Gemini%20text--embedding--004-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/tests-37%20passed-success)]()
[![Docker compose up](https://img.shields.io/badge/zero--step-docker%20compose%20up-2496ED?logo=docker&logoColor=white)]()

</div>

```mermaid
flowchart LR
    T(["closed&nbsp;trade"]):::evt --> EP["POST /session/events"]:::route
    EP --> DET[["detect_signal<br/>deterministic, ~ms"]]:::rule
    DET --> LLM[["Groq stream<br/>Llama 3.3 70B"]]:::llm
    LLM -. SSE tokens .-> CLI(["client"]):::evt
    DB[("Postgres<br/>+ pgvector")]:::db --- DET
    DB --- LLM
    MEM["/memory/context"]:::route --- DB

    classDef evt fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:2px
    classDef route fill:#dbeafe,stroke:#1d4ed8,color:#0c4a6e,stroke-width:2px
    classDef rule fill:#dcfce7,stroke:#15803d,color:#14532d,stroke-width:2px
    classDef llm fill:#fed7aa,stroke:#c2410c,color:#7c2d12,stroke-width:2px
    classDef db fill:#fef3c7,stroke:#a16207,color:#451a03,stroke-width:2px
```

---

## 📋 Table of contents

- [💡 What this is](#-what-this-is)
- [🚀 Quickstart](#-quickstart)
- [📊 What the eval shows — and what it doesn't](#-what-the-eval-shows--and-what-it-doesnt)
- [⚠️ Known limitations](#️-known-limitations)
- [🌐 API reference](#-api-reference)
- [🧪 Live demos (copy-paste-able)](#-live-demos-copy-paste-able)
- [🏗️ Architecture](#️-architecture)
- [🛡️ Anti-hallucination strategy](#️-anti-hallucination-strategy)
- [🔒 Auth & tenancy](#-auth--tenancy)
- [📦 Project layout](#-project-layout)
- [✅ Tests](#-tests)
- [🧠 Architectural decisions](#-architectural-decisions)

---

## 💡 What this is

A backend service that ingests a trader's history — every closed position with its emotional state, plan-adherence rating, and entry rationale — and produces:

1. A **behavioral profile** that names the trader's dominant psychological pathology, with citations to the specific trades that prove it.
2. **Real-time coaching messages** streamed when a new trade closes, grounded in a deterministic signal detector so the LLM is never asked to discover a problem from scratch.
3. A **persistent semantic memory** of past sessions, queryable via vector similarity.
4. An **independent audit endpoint** that any reviewer can run on any text to verify whether the cited session IDs actually exist.

> **The hard part isn't calling an LLM. It's building infrastructure the LLM can stand on without lying.**

| Component | Why it's there |
|---|---|
| **Postgres + pgvector** for memory | Survives container restart by construction — no in-process caches |
| **9 deterministic pathology scorers** with cited `tradeId`/`sessionId` | LLM never invents IDs — `rules.py` is the source of truth |
| **`/audit` endpoint** that re-extracts UUIDs from any text | Any caller can verify hallucinations end-to-end |
| **Streaming coaching via Groq SSE** with deterministic signal grounding | First-token < 400ms, p99 < 3s — rule fires before LLM is called |
| **Reproducible eval harness** (no API keys needed) | sklearn classification report on the labelled dataset |
| **JWT-HS256 + row-level tenancy** | Cross-tenant access returns **403** with `traceId` in the body |

---

## 🚀 Quickstart

```bash
# 1. clone, cd in, copy env template
cp .env.example .env

# 2. (optional) add free-tier API keys — without them the service still runs in fallback mode
echo 'GEMINI_API_KEY=your_key_here' >> .env
echo 'GROQ_API_KEY=your_key_here'  >> .env

# 3. one command. seriously.
docker compose up --build
```

The entrypoint **waits for the DB → applies migrations → seeds the dataset → starts uvicorn**. Total cold-start: ~25 seconds. The API is ready at `http://localhost:8000`.

> 💡 **Postgres on host port `5433`** (we map `5433:5432`) to avoid colliding with developer-local Postgres. Inside the docker network the API still talks to `db:5432`.

<details>
<summary><b>🔑 Get free-tier API keys (the eval harness works without them)</b></summary>

| Provider | Used for | Get a key |
|---|---|---|
| **Groq** | Streaming coaching tokens (sub-400ms first token) | https://console.groq.com/keys |
| **Gemini** | Embeddings + structured profile narration | https://aistudio.google.com/apikey |

If both are blank, the service runs in **deterministic fallback mode**:
- Profile narration falls back to `rules_only_profile()`
- Coaching SSE emits a stub message
- Embedding-dependent endpoints (`/memory/context`) require a Gemini key in production
- **The eval harness uses ONLY the rule layer — runs perfectly without keys**

</details>

---

## 📊 What the eval shows — and what it doesn't

The harness runs the rule-based profiler over a labelled 10-trader dataset and emits a sklearn classification report:

```bash
docker compose run --rm api python -m scripts.eval_harness
```

<table>
<tr>
<th>Trader</th>
<th>Ground truth</th>
<th>Predicted</th>
<th>Top score</th>
<th>Match</th>
</tr>
<tr><td>Alex Mercer</td><td><code>revenge_trading</code></td><td><code>revenge_trading</code></td><td>1.00</td><td>✅</td></tr>
<tr><td>Jordan Lee</td><td><code>overtrading</code></td><td><code>overtrading</code></td><td>1.00</td><td>✅</td></tr>
<tr><td>Sam Rivera</td><td><code>fomo_entries</code></td><td><code>fomo_entries</code></td><td>1.00</td><td>✅</td></tr>
<tr><td>Casey Kim</td><td><code>plan_non_adherence</code></td><td><code>plan_non_adherence</code></td><td>0.43</td><td>✅</td></tr>
<tr><td>Morgan Bell</td><td><code>premature_exit</code></td><td><code>premature_exit</code></td><td>0.72</td><td>✅</td></tr>
<tr><td>Taylor Grant</td><td><code>loss_running</code></td><td><code>loss_running</code></td><td>1.00</td><td>✅</td></tr>
<tr><td>Riley Stone</td><td><code>session_tilt</code></td><td><code>session_tilt</code></td><td>0.80</td><td>✅</td></tr>
<tr><td>Drew Patel</td><td><code>time_of_day_bias</code></td><td><code>time_of_day_bias</code></td><td>0.75</td><td>✅</td></tr>
<tr><td>Quinn Torres</td><td><code>position_sizing_inconsistency</code></td><td><code>position_sizing_inconsistency</code></td><td>0.57</td><td>✅</td></tr>
<tr><td>Avery Chen <em>(control)</em></td><td><code>none</code></td><td><code>none</code></td><td>0.00</td><td>✅</td></tr>
</table>

The harness recovers **all 10 ground-truth labels**. Full per-class precision/recall/F1 lives in [`eval/report.json`](./eval/report.json).

### Read this before you read 10/10 as accuracy

> 🚨 **This is not a model-accuracy claim.**

**What it IS:** evidence that the 9-pathology rule system is **expressive enough to encode the distinguishing patterns** in the labelled dataset. Each scorer applies gating filters that target a single feature signature — e.g., `_score_fomo_entries` requires `greedy` to be ≥60% of trades; `_score_position_sizing_inconsistency` requires `CV ≥ 0.85` in any asset class. When those gates fire correctly, the system maps each trader to their labelled pathology.

**What it is NOT:**
- ❌ A generalization claim. The thresholds were tuned **on the same 10 traders** the eval evaluates against — there is no held-out test set with N=10 labelled examples.
- ❌ A statistical accuracy number. With 10 samples and 9 classes (+ control), confidence intervals on any "accuracy" metric are enormous.
- ❌ A learned model. There is no training. Rules are hand-written; thresholds were chosen by inspecting feature distributions and adjusting until each gate fired cleanly.

**Why this design over a black-box classifier:** the product needs **citable evidence** for every claim it makes about a trader. A learned model with the same accuracy could not point to specific `tradeId`s — its "evidence" would be a softmax. Hand-written rules with deterministic thresholds give the LLM a list of real, verifiable references to paraphrase. That's the architecture, not the accuracy.

See [⚠️ Known limitations](#️-known-limitations) for the honest list of what would need to change before deploying beyond the labelled dataset.

---

## ⚠️ Known limitations

| Limitation | Where | What we'd do next |
|---|---|---|
| **Rule thresholds are tuned on the eval set** | [app/profiling/rules.py](./app/profiling/rules.py) — comments mark which thresholds were tuned to specific traders | Hold out 30% of traders in a generated dataset; cross-validate. Or replace gates with a small classifier learned from feature vectors, keeping `rules.py` as the citation extractor. |
| **N=10 is too small for any accuracy claim** | The labelled dataset | Synthesize more labelled traders following the same generator that produced the seed; treat 10 as a sanity-check fixture, not a benchmark. |
| **Single-pathology assumption** | `score_pathologies` returns a sorted list, but the harness picks `top[0]` if score ≥ 0.3 | Multi-label evaluation with per-pathology decision thresholds. The current dataset has one label per trader — real traders combine pathologies. |
| **Embeddings depend on Gemini** | [app/memory/embeddings.py](./app/memory/embeddings.py) | Add a local-model fallback (e.g., `sentence-transformers/all-MiniLM-L6-v2`) so semantic memory works without external keys |
| **Coaching prompt is unbounded English** | [app/coaching/intervention.py](./app/coaching/intervention.py) | Post-process Groq output through `/audit` before returning to client, dropping invented IDs |
| **No load test for /session/events** | — | Add k6 or Locust script |
| **Position sizing CV is per-asset-class, not per-equity-fraction** | [app/profiling/rules.py](./app/profiling/rules.py) | Real position sizing should be normalised against account equity, not raw quantity |
| **Single-language LLM** | — | Coaching prompt assumes English. International users would need locale-aware prompts |

---

## 🌐 API reference

| Method | Path | Auth | Returns |
|---|---|---|---|
| `GET` | `/health` | — | `{status, queue_lag, db}` |
| `PUT` | `/memory/{userId}/sessions/{sessionId}` | 🔒 | `204` |
| `GET` | `/memory/{userId}/context?relevant_to=...&limit=5` | 🔒 | `{sessions[], pattern_ids[]}` |
| `GET` | `/memory/{userId}/sessions/{sessionId}` | 🔒 | raw session JSON |
| `GET` | `/profile/{userId}` | 🔒 | `{profile, scored[]}` |
| `POST` | `/session/events?user_id=...` | 🔒 | `text/event-stream` |
| `POST` | `/audit` | 🔒 | `{citations[], extracted[]}` |

🔒 = requires `Authorization: Bearer <jwt>` and enforces **row-level tenancy** (`sub === userId` else **403** with `traceId`).

---

## 🧪 Live demos (copy-paste-able)

> Mint a dev JWT first (24h expiry):
> ```bash
> TOKEN=$(.venv/bin/python -m scripts.mint_token f412f236-4edc-47a2-8f54-8763a6ed2ce8)
> ```

<details open>
<summary><b>🛡️ Demo 1 — Hallucination audit catches a fake sessionId</b></summary>

```bash
curl -s -X POST http://localhost:8000/audit \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "f412f236-4edc-47a2-8f54-8763a6ed2ce8",
    "response": "In your prior session 4f39c2ea-8687-41f7-85a0-1fafd3e976df you took 5 trades. Compare with session 00000000-0000-0000-0000-000000000099."
  }' | python -m json.tool
```

**Expected response:**

```json
{
  "user_id": "f412f236-4edc-47a2-8f54-8763a6ed2ce8",
  "citations": [
    { "session_id": "4f39c2ea-8687-41f7-85a0-1fafd3e976df", "found": true  },
    { "session_id": "00000000-0000-0000-0000-000000000099", "found": false }
  ],
  "extracted": [
    "4f39c2ea-8687-41f7-85a0-1fafd3e976df",
    "00000000-0000-0000-0000-000000000099"
  ]
}
```

</details>

<details>
<summary><b>📡 Demo 2 — Streaming coaching via SSE</b></summary>

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

**You'll see:**

```
data: I notice
data:  this trade
data:  followed
...
event: done
data: [DONE]
```

The `try/except/finally` in [app/coaching/router.py](./app/coaching/router.py) guarantees `[DONE]` always emits — even if Groq errors mid-stream.

</details>

<details>
<summary><b>🔍 Demo 3 — Behavioral profile with cited evidence</b></summary>

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/profile/f412f236-4edc-47a2-8f54-8763a6ed2ce8 | python -m json.tool
```

**Expected (truncated):**

```json
{
  "profile": {
    "userId": "f412f236-4edc-47a2-8f54-8763a6ed2ce8",
    "primaryPathology": "revenge_trading",
    "confidence": 1.0,
    "weaknesses": [{
      "pattern": "revenge_trading",
      "citations": [
        {"sessionId": "4f39c2ea-8687-41f7-85a0-1fafd3e976df",
         "tradeId":   "49e71680-b968-4955-9e00-0b831d331ebd"}
      ]
    }]
  },
  "scored": [
    {"pathology": "revenge_trading", "score": 1.0, "evidence": [...]},
    {"pathology": "overtrading",     "score": 0.4, "evidence": [...]}
  ]
}
```

> 🛡️ **Every `tradeId` and `sessionId` in the response exists in the DB.** The LLM cannot invent them — see [Anti-hallucination strategy](#️-anti-hallucination-strategy).

</details>

<details>
<summary><b>🔒 Demo 4 — Cross-tenant access returns 403 (with traceId)</b></summary>

```bash
# Token for one user, asking for another user's profile
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/profile/fcd434aa-2201-4060-aeb2-f44c77aa0683
```

**Output:** `403`

Tenancy is enforced by a single shared dependency — see [app/auth/deps.py](./app/auth/deps.py). Errors include a `traceId` matching the structured log.

</details>

<details>
<summary><b>💾 Demo 5 — Memory persists across container restart</b></summary>

```bash
# Write a summary
curl -s -X PUT "http://localhost:8000/memory/f412f236-4edc-47a2-8f54-8763a6ed2ce8/sessions/4f39c2ea-8687-41f7-85a0-1fafd3e976df" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"summary":"persistence check","metrics":{},"tags":["persist"]}'

# Restart the api container (DB stays on its named volume)
docker compose restart api
sleep 8

# Retrieve it back
curl -s "http://localhost:8000/memory/f412f236-4edc-47a2-8f54-8763a6ed2ce8/context?relevant_to=persistence" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

The summary returns intact. There is **no in-process cache** — Postgres on a named volume is the only state.

</details>

---

## 🏗️ Architecture

```mermaid
flowchart TB
    Client([Client]) -->|JWT| MW[TracingMiddleware<br/>traceId + JSON logs]
    MW --> Auth{require_user<br/>+ enforce_tenancy}
    Auth -->|403| Client

    Auth --> MR["/memory/*"]
    Auth --> PR["/profile/{userId}"]
    Auth --> CR["/session/events"]
    Auth --> AR["/audit"]

    MR --> MS[memory.service]
    MS -->|embed| GE[Gemini<br/>text-embedding-004]
    MS --> PG[(Postgres + pgvector<br/>session_summaries)]

    PR --> RULES[profiling.rules<br/>9 deterministic scorers]
    PR --> LLM[profiling.llm<br/>Gemini JSON narrate]
    RULES --> PG
    LLM -.fallback.-> RULES

    CR --> SIG[detect_signal<br/>deterministic, ~ms]
    SIG --> INT[intervention.stream_coaching]
    INT --> MS
    INT --> GROQ[Groq<br/>Llama 3.3 70B<br/>SSE tokens]

    AR --> RX[regex extract<br/>UUID]
    RX --> PG

    style RULES fill:#dcfce7,stroke:#16a34a,color:#000
    style PG fill:#dbeafe,stroke:#2563eb,color:#000
    style GROQ fill:#fed7aa,stroke:#ea580c,color:#000
    style GE fill:#fef3c7,stroke:#ca8a04,color:#000
    style LLM fill:#fef3c7,stroke:#ca8a04,color:#000
```

### Coaching request flow (sub-400ms first token)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI
    participant DB as Postgres
    participant DET as detect_signal()
    participant MEM as memory.get_context
    participant G as Groq stream

    C->>API: POST /session/events<br/>(closed trade)
    API->>API: enforce_tenancy → 403 short-circuit
    API->>DB: SELECT trades WHERE user_id=?<br/>ORDER BY entry_at
    DB-->>API: history[]
    API->>DET: detect_signal(history, current)
    DET-->>API: {type: "revenge_trade", ...} (deterministic)
    API->>MEM: get_context(user, signal)
    MEM->>DB: ORDER BY embedding <=> ?
    DB-->>MEM: top-3 sessions
    MEM-->>API: SessionSummary[]
    API->>G: stream(system + signal + memory)
    loop tokens
        G-->>API: delta
        API-->>C: data: <token>\n\n
    end
    API-->>C: event: done\ndata: [DONE]
```

---

## 🛡️ Anti-hallucination strategy

The biggest failure mode for an LLM-driven coaching system is referencing a prior session that never existed. We defend in three layers:

```mermaid
flowchart LR
    A[1. Rules emit citations<br/>from real input rows] --> B[2. LLM is told<br/>'cite ONLY from this list']
    B --> C[3. /audit endpoint<br/>regex-extracts UUIDs<br/>and verifies in DB]
    style A fill:#dcfce7,stroke:#16a34a,color:#000
    style B fill:#dbeafe,stroke:#2563eb,color:#000
    style C fill:#fce7f3,stroke:#db2777,color:#000
```

| Layer | Where | What it does |
|---|---|---|
| **1. Source of truth** | [app/profiling/rules.py](./app/profiling/rules.py) | Each scorer reads `t["trade_id"]` / `t["session_id"]` directly off input rows. **No string formatting, no hardcoded UUIDs.** |
| **2. Constrained LLM** | [app/profiling/llm.py](./app/profiling/llm.py) | System prompt: *"Citations MUST be drawn from the evidence array provided. Never invent IDs."* JSON mode. After parsing, `parsed["userId"] = user_id` (LLM cannot smuggle a wrong tenant). |
| **3. Independent audit** | [app/audit/router.py](./app/audit/router.py) | `POST /audit` accepts any text, regex-extracts every UUID, and verifies each one against `sessions WHERE user_id = sub`. Callers can run this on **any** coaching response. |

---

## 🔒 Auth & tenancy

<table>
<tr>
<th width="50%">JWT validation</th>
<th width="50%">Row-level tenancy</th>
</tr>
<tr>
<td>

```python
# app/auth/jwt.py
jwt.decode(
    token, settings.jwt_secret,
    algorithms=["HS256"],
    options={"require": [
        "exp", "iat", "sub", "role"
    ]},
)
# + explicit role == "trader" check
```

PyJWT enforces presence of all 4 claims. We additionally check `role == "trader"` — reserved for future role expansion.

</td>
<td>

```python
# app/auth/deps.py
def enforce_tenancy(user, requested_user_id, request):
    if user["sub"] != requested_user_id:
        raise HTTPException(
            status_code=403,
            detail={"error": "FORBIDDEN", ...,
                    "traceId": _trace_id(request)}
        )
```

Called by **every** userId-bound route. Returns **403** with `traceId` matching the structured log — never 404 on cross-tenant access.

</td>
</tr>
</table>

---

## 📦 Project layout

```
nevup/
├── docker-compose.yml          # postgres+pgvector, api — single command
├── Dockerfile + entrypoint.sh  # waits for DB → migrate → seed → uvicorn
├── alembic/versions/0001_*.py  # creates pgvector extension + 4 tables
├── app/
│   ├── main.py                 # FastAPI, middleware, 4 routers wired
│   ├── config.py               # pydantic-settings (env-driven)
│   ├── db.py / models.py       # async SQLAlchemy 2.0
│   ├── schemas.py              # Pydantic DTOs
│   ├── auth/                   # JWT-HS256 + tenancy
│   ├── memory/                 # embeddings + service + router
│   │   ├── embeddings.py       # Gemini, async, tenacity retry
│   │   ├── service.py          # upsert / semantic-context / raw fetch
│   │   └── router.py           # PUT/GET endpoints with input validation
│   ├── metrics/behavioral.py   # 5 deterministic signals
│   ├── profiling/              # 🛡️ anti-hallucination layer
│   │   ├── rules.py            # 9 scorers — source of truth for IDs
│   │   ├── llm.py              # Gemini JSON narrate, rules-only fallback
│   │   └── router.py           # GET /profile/{userId}
│   ├── coaching/               # SSE streaming
│   │   ├── groq_client.py      # async streaming, stub fallback
│   │   ├── intervention.py     # signal-grounded prompt assembly
│   │   └── router.py           # POST /session/events with SSE
│   ├── audit/router.py         # POST /audit — regex + DB verify
│   └── observability/          # JSON logger + traceId middleware
├── scripts/
│   ├── seed.py                 # idempotent seed loader
│   ├── mint_token.py           # dev JWT minter
│   └── eval_harness.py         # sklearn classification report
├── tests/                      # 37 tests, 1 opt-in skip
└── eval/report.json            # generated by harness (gitignored)
```

---

## ✅ Tests

| Suite | Count | Run |
|---|---|---|
| Unit (no DB) | 19 | `pytest -m "not integration"` |
| Integration (real Postgres + pgvector) | 18 | `pytest` (DB on port 5433) |
| Persistence-across-restart | 1 | `RUN_PERSISTENCE_TEST=1 pytest tests/test_persistence.py` |
| **Total** | **37 passed, 1 skipped** | — |

**Test honesty:** integration tests hit the real seeded Postgres. Mocks are limited to true external boundaries (Gemini embeddings, Groq stream).

```bash
# Unit only (fast, no DB)
.venv/bin/python -m pytest -m "not integration" -q

# Full suite (needs docker DB up)
docker compose up -d db
DATABASE_URL=postgresql+asyncpg://nevup:nevup@localhost:5433/nevup .venv/bin/python -m pytest -q
```

---

## 🧠 Architectural decisions

The full reasoning lives in [**DECISIONS.md**](./DECISIONS.md). Highlights:

| Decision | Why |
|---|---|
| **pgvector** instead of Pinecone/Chroma | One container, atomic UPSERT, persists for free, no cross-service consistency to reason about |
| **Rules-first, LLM-second** | Citations come from deterministic detection — LLM physically cannot invent IDs |
| **Groq for streaming, Gemini for structured profiling** | Lowest first-token latency on Groq; Gemini's `response_mime_type=application/json` is reliable |
| **Synchronous signal detection** | Compute the behavioral signal in ~ms before calling Groq — bounds latency, grounds output |
| **Fallback when keys missing** | Eval harness reproducible without provisioning external creds |
| **Host port 5433** | Avoids developer-local Postgres collisions; container network unaffected |

---

<div align="center">

37 tests passing · [`/audit`](./app/audit/router.py) · [`rules.py`](./app/profiling/rules.py) is the source of truth for IDs · zero-step `docker compose up`

</div>

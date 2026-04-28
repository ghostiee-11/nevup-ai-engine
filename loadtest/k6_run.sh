#!/usr/bin/env bash
# Convenience wrapper: starts a local uvicorn (with stub coaching), mints a JWT,
# runs k6, then renders the JSON summary to HTML.
#
# Prereqs:
#   - docker compose db is up (port 5433)
#   - .venv populated via `pip install -e ".[dev]"`
#   - k6 in PATH
#
# The script intentionally runs uvicorn DIRECTLY (not docker) so we measure the
# Python application's behavior without containerisation overhead. Docker numbers
# would be dominated by the API container's free-tier-Render constraints which
# don't reflect the application's own latency.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p loadtest/results
export DATABASE_URL=postgresql+asyncpg://nevup:nevup@localhost:5433/nevup
export GEMINI_API_KEY=""   # ⇒ embeddings use SHA fallback (fast, deterministic)
export GROQ_API_KEY=""     # ⇒ stub coaching (fast, deterministic)

# Boot uvicorn in the background; tear down on exit.
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765 \
  > loadtest/results/uvicorn.log 2>&1 &
UVPID=$!
trap 'kill ${UVPID} 2>/dev/null || true' EXIT
echo "started uvicorn pid=${UVPID}"

# Wait for /health
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8765/health > /dev/null 2>&1; then
    echo "uvicorn ready (after ${i}s)"
    break
  fi
  sleep 1
done

# Mint a JWT
TOKEN=$(.venv/bin/python -m scripts.mint_token f412f236-4edc-47a2-8f54-8763a6ed2ce8)
export API_URL=http://127.0.0.1:8765
export TOKEN

# Run k6 — note `|| true` so a threshold failure doesn't abort the post-processing.
# (k6 returns non-zero when any threshold crosses; we still want the report.)
echo "starting k6…"
k6 run loadtest/k6_session_events.js || echo "(k6 exited non-zero — threshold crossed; results still valid)"

# Render HTML summary
.venv/bin/python loadtest/summary_to_html.py \
  loadtest/results/summary.json loadtest/results/results.html

# Snapshot /metrics
curl -s http://127.0.0.1:8765/metrics > loadtest/results/metrics.json
echo "wrote loadtest/results/{summary.json,results.html,metrics.json,uvicorn.log}"

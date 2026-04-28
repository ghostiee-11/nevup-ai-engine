// k6 load test for /session/events SSE endpoint.
//
// Goal: measure the *time-to-first-byte* — the SSE spec's "stream starts within
// 400ms" requirement is about the HTTP response getting flowing, not Groq
// finishing inference. We run with GROQ_API_KEY="" so the stub fallback fires
// (deterministic, fast tokens) and we measure server-side handling.
//
// Usage:
//   loadtest/k6_run.sh
//   # or directly:
//   API_URL=http://localhost:8000 \
//     TOKEN=$(.venv/bin/python -m scripts.mint_token f412f236-4edc-47a2-8f54-8763a6ed2ce8) \
//     k6 run loadtest/k6_session_events.js
//
// Output:
//   - terminal summary
//   - loadtest/results/summary.json (machine-readable)
//   - loadtest/results/results.html (rendered by summary_to_html.py)

import http from "k6/http";
import { check } from "k6";
import { Trend, Rate } from "k6/metrics";
import { SharedArray } from "k6/data";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.4/index.js";

const API_URL = __ENV.API_URL || "http://localhost:8000";
const TOKEN = __ENV.TOKEN;
if (!TOKEN) {
  throw new Error("TOKEN env var required (mint via scripts.mint_token)");
}

// Custom metrics
const sseFirstByteMs = new Trend("sse_first_byte_ms", true);
const sseSuccessRate = new Rate("sse_success_rate");

export const options = {
  scenarios: {
    ramping_load: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "10s", target: 10 },
        { duration: "30s", target: 30 },
        { duration: "20s", target: 30 },
      ],
      gracefulRampDown: "5s",
    },
  },
  thresholds: {
    // Headline target from the brief: stream should start within 400ms.
    "sse_first_byte_ms": ["p(95)<400", "p(99)<3000"],
    "http_req_failed":   ["rate<0.01"],
    "sse_success_rate":  ["rate>0.99"],
  },
  summaryTrendStats: ["min", "med", "avg", "p(95)", "p(99)", "max"],
};

const TRADE_TEMPLATE = {
  asset: "AAPL",
  assetClass: "equity",
  direction: "long",
  entryPrice: 180.0,
  exitPrice: 179.0,
  quantity: 10,
  entryAt: "2025-02-10T09:30:00Z",
  exitAt: "2025-02-10T09:31:00Z",
  status: "closed",
  outcome: "loss",
  pnl: -10.0,
  planAdherence: 1,
  emotionalState: "anxious",
  entryRationale: "load test",
};

const USER_ID = "f412f236-4edc-47a2-8f54-8763a6ed2ce8";

export default function () {
  const sessionId = `00000000-0000-0000-0000-${("000000000000" + __ITER).slice(-12)}`;
  const tradeId = `11111111-1111-1111-1111-${("000000000000" + __ITER).slice(-12)}`;
  const body = JSON.stringify({
    session_id: sessionId,
    trade: { ...TRADE_TEMPLATE, tradeId, userId: USER_ID, sessionId },
  });

  const params = {
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    timeout: "10s",
    // We want the response body to be readable as SSE; k6 will buffer until
    // the request completes. The HTTP-level metric `http_req_waiting` is
    // what we treat as time-to-first-byte for an SSE stream.
  };

  const res = http.post(`${API_URL}/session/events?user_id=${USER_ID}`, body, params);

  // http_req_waiting ≈ TTFB. For SSE, this is the moment the keep-alive
  // ": connecting\n\n" comment frame (or first data frame) hits the wire.
  if (res.timings && res.timings.waiting !== undefined) {
    sseFirstByteMs.add(res.timings.waiting);
  }

  const ok = check(res, {
    "status 200":         (r) => r.status === 200,
    "content-type SSE":   (r) => /text\/event-stream/.test(r.headers["Content-Type"] || ""),
    "stream terminator":  (r) => /event:\s*done/.test(r.body || ""),
  });
  sseSuccessRate.add(ok);
}

export function handleSummary(data) {
  return {
    "loadtest/results/summary.json": JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}

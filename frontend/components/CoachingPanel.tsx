"use client";

import { useRef, useState } from "react";
import { streamSSE, ApiError } from "@/lib/api";
import { Card, ErrorBox, EmptyState } from "./Card";

const PRESETS = {
  revenge: {
    label: "Anxious revenge sequence",
    trade: {
      asset: "AAPL",
      assetClass: "equity",
      direction: "long",
      entryPrice: 100.0,
      exitPrice: 99.0,
      quantity: 10,
      entryAt: "2025-02-10T09:30:00Z",
      exitAt: "2025-02-10T09:31:00Z",
      status: "closed",
      outcome: "loss",
      planAdherence: 1,
      emotionalState: "anxious",
    },
  },
  greedy_fomo: {
    label: "Greedy FOMO entry",
    trade: {
      asset: "BTC/USD",
      assetClass: "crypto",
      direction: "long",
      entryPrice: 65000,
      exitPrice: 64200,
      quantity: 0.5,
      entryAt: "2025-02-10T11:15:00Z",
      exitAt: "2025-02-10T11:18:00Z",
      status: "closed",
      outcome: "loss",
      planAdherence: 2,
      emotionalState: "greedy",
    },
  },
  calm_win: {
    label: "Calm planned win",
    trade: {
      asset: "MSFT",
      assetClass: "equity",
      direction: "long",
      entryPrice: 410,
      exitPrice: 414,
      quantity: 5,
      entryAt: "2025-02-10T14:00:00Z",
      exitAt: "2025-02-10T15:30:00Z",
      status: "closed",
      outcome: "win",
      planAdherence: 5,
      emotionalState: "calm",
    },
  },
} as const;

type PresetKey = keyof typeof PRESETS;

export function CoachingPanel({ token, userId }: { token: string; userId: string }) {
  const [preset, setPreset] = useState<PresetKey>("revenge");
  const [streaming, setStreaming] = useState(false);
  const [tokens, setTokens] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [firstTokenMs, setFirstTokenMs] = useState<number | null>(null);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const t0Ref = useRef<number>(0);

  const start = async () => {
    if (!token || !userId) return;
    setTokens([]);
    setError(null);
    setFirstTokenMs(null);
    setTotalMs(null);
    setStreaming(true);
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    t0Ref.current = performance.now();
    let firstSet = false;

    const tradeId = crypto.randomUUID();
    const sessionId = crypto.randomUUID();
    const body = {
      session_id: sessionId,
      trade: {
        ...PRESETS[preset].trade,
        tradeId,
        userId,
        sessionId,
      },
    };

    await streamSSE(
      `/session/events?user_id=${encodeURIComponent(userId)}`,
      body,
      token,
      {
        onToken: (t) => {
          if (!firstSet) {
            setFirstTokenMs(Math.round(performance.now() - t0Ref.current));
            firstSet = true;
          }
          setTokens((prev) => [...prev, t]);
        },
        onDone: () => {
          setTotalMs(Math.round(performance.now() - t0Ref.current));
          setStreaming(false);
        },
        onError: (e) => {
          setStreaming(false);
          if (e instanceof ApiError) setError(`HTTP ${e.status}: ${e.body.slice(0, 200)}`);
          else setError(e.message);
        },
      },
      ctrl.signal,
    );
  };

  const stop = () => {
    abortRef.current?.abort();
    setStreaming(false);
  };

  return (
    <Card
      title="Live coaching stream"
      subtitle="Submit a closed trade · deterministic signal grounds the prompt · SSE tokens from Groq"
      endpoint="POST /session/events"
    >
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {(Object.keys(PRESETS) as PresetKey[]).map((k) => (
            <button
              key={k}
              onClick={() => setPreset(k)}
              disabled={streaming}
              className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
                preset === k
                  ? "border-accent-blue bg-accent-blue/15 text-accent-blue"
                  : "border-ink-700 bg-ink-800/40 text-ink-300 hover:border-ink-600 hover:text-ink-100"
              } disabled:opacity-50`}
            >
              {PRESETS[k].label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={streaming ? stop : start}
            disabled={!token}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              streaming
                ? "bg-accent-rose/20 text-accent-rose hover:bg-accent-rose/30"
                : "bg-accent-green/20 text-accent-green hover:bg-accent-green/30"
            } disabled:opacity-50`}
          >
            {streaming ? "Stop stream" : "Start coaching →"}
          </button>
          {firstTokenMs !== null ? (
            <span className="font-mono text-xs text-ink-300">
              first token <b className="text-accent-green">{firstTokenMs}ms</b>
              {totalMs !== null ? (
                <>
                  {" · "}total <b className="text-ink-100">{totalMs}ms</b>
                </>
              ) : null}
            </span>
          ) : null}
        </div>

        {error ? <ErrorBox>{error}</ErrorBox> : null}

        <div className="min-h-[140px] max-h-[260px] overflow-y-auto rounded-xl border border-ink-700 bg-ink-950/70 p-4 font-mono text-sm leading-relaxed text-ink-100">
          {tokens.length === 0 ? (
            <span className="text-ink-400">
              {streaming ? "Waiting for first token…" : "Click ‘Start coaching’ to stream"}
            </span>
          ) : (
            tokens.map((t, i) => (
              <span key={i} className="animate-token_in">
                {t}
              </span>
            ))
          )}
          {streaming ? <span className="ml-1 inline-block h-3 w-1.5 animate-pulse_dot bg-accent-blue align-middle" /> : null}
        </div>
      </div>
    </Card>
  );
}

"use client";

import { useState } from "react";
import { apiJson, ApiError } from "@/lib/api";
import { Card, ErrorBox, EmptyState } from "./Card";

type SessionSummaryOut = {
  session_id: string;
  user_id: string;
  summary: string;
  metrics: Record<string, unknown>;
  tags: string[];
  created_at: string;
};
type ContextResponse = {
  sessions: SessionSummaryOut[];
  pattern_ids: string[];
};

export function MemoryPanel({ token, userId }: { token: string; userId: string }) {
  const [query, setQuery] = useState("revenge trading after a losing close");
  const [data, setData] = useState<ContextResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [seedDone, setSeedDone] = useState(false);
  const [seeding, setSeeding] = useState(false);

  const search = async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const r = await apiJson<ContextResponse>(
        `/memory/${encodeURIComponent(userId)}/context?relevant_to=${encodeURIComponent(query)}&limit=5`,
        { token },
      );
      setData(r);
    } catch (e) {
      if (e instanceof ApiError) setError(`HTTP ${e.status}: ${e.body.slice(0, 200)}`);
      else setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const seedSummary = async () => {
    setSeeding(true);
    setError(null);
    try {
      // Pick a known seeded session for the demo. The backend will accept any
      // sessionId UUID; using a real one keeps the audit endpoint happy too.
      const sessionId =
        userId === "f412f236-4edc-47a2-8f54-8763a6ed2ce8"
          ? "4f39c2ea-8687-41f7-85a0-1fafd3e976df"
          : crypto.randomUUID();
      const summary = `Demo summary for ${userId.slice(0, 8)}: anxious follow-on entries after losing closes; ${query}.`;
      const r = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/memory/${encodeURIComponent(userId)}/sessions/${sessionId}`,
        {
          method: "PUT",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            summary,
            metrics: { revenge_flag: true, plan_adherence_rolling: 2.1 },
            tags: ["revenge_trading", "demo"],
          }),
        },
      );
      if (!r.ok) throw new ApiError(r.status, await r.text());
      setSeedDone(true);
    } catch (e) {
      if (e instanceof ApiError) setError(`HTTP ${e.status}: ${e.body.slice(0, 200)}`);
      else setError((e as Error).message);
    } finally {
      setSeeding(false);
    }
  };

  return (
    <Card
      title="Semantic memory search"
      subtitle="pgvector cosine similarity over session_summaries · scoped by userId"
      endpoint="GET /memory/{userId}/context"
    >
      <div className="space-y-3">
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search prior sessions…"
            className="flex-1 rounded-lg border border-ink-700 bg-ink-950/70 px-3 py-2 text-sm text-ink-100 placeholder-ink-400 focus:border-accent-blue focus:outline-none"
          />
          <button
            onClick={search}
            disabled={loading || !token}
            className="rounded-lg bg-accent-blue/20 px-4 py-2 text-sm font-medium text-accent-blue transition hover:bg-accent-blue/30 disabled:opacity-50"
          >
            {loading ? "Searching…" : "Search →"}
          </button>
        </div>

        {error ? <ErrorBox>{error}</ErrorBox> : null}

        {!data ? (
          <EmptyState>
            No summaries indexed for this trader yet.
            <button
              onClick={seedSummary}
              disabled={seeding || !token}
              className="ml-2 underline decoration-dotted hover:text-ink-200 disabled:opacity-50"
            >
              {seeding ? "Writing…" : seedDone ? "✓ wrote demo summary" : "Write a demo summary"}
            </button>
            {" "}then search again.
          </EmptyState>
        ) : data.sessions.length === 0 ? (
          <EmptyState>
            No matching summaries. Try writing one first via PUT /memory/.../sessions/...
            <button
              onClick={seedSummary}
              disabled={seeding || !token}
              className="ml-2 underline decoration-dotted hover:text-ink-200 disabled:opacity-50"
            >
              {seeding ? "Writing…" : seedDone ? "✓ wrote demo summary" : "Write demo summary"}
            </button>
          </EmptyState>
        ) : (
          <div className="space-y-3">
            {data.pattern_ids.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {data.pattern_ids.map((p) => (
                  <span
                    key={p}
                    className="rounded-full border border-accent-violet/40 bg-accent-violet/10 px-2 py-0.5 font-mono text-[10px] text-accent-violet"
                  >
                    {p}
                  </span>
                ))}
              </div>
            ) : null}
            <ul className="space-y-2">
              {data.sessions.map((s) => (
                <li
                  key={s.session_id}
                  className="rounded-lg border border-ink-700 bg-ink-800/40 p-3"
                >
                  <div className="flex items-center justify-between gap-2">
                    <code className="truncate font-mono text-[11px] text-ink-300">
                      {s.session_id}
                    </code>
                    <span className="shrink-0 text-[10px] text-ink-400">
                      {new Date(s.created_at).toLocaleString()}
                    </span>
                  </div>
                  <p className="mt-1.5 text-xs text-ink-200">{s.summary}</p>
                  {s.tags.length > 0 ? (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {s.tags.map((t) => (
                        <span
                          key={t}
                          className="rounded-md border border-ink-700 bg-ink-900/60 px-1.5 py-0.5 font-mono text-[10px] text-ink-300"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}

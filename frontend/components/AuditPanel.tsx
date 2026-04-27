"use client";

import { useState } from "react";
import { apiJson, ApiError } from "@/lib/api";
import { Card, ErrorBox } from "./Card";

type Citation = { session_id: string; found: boolean };
type AuditResponse = {
  user_id: string;
  citations: Citation[];
  extracted: string[];
};

const DEFAULT_TEXT = `In your prior session 4f39c2ea-8687-41f7-85a0-1fafd3e976df you took 5 trades with anxious follow-on entries. Compare with session 00000000-0000-0000-0000-000000000099 from last month — the patterns differ.`;

export function AuditPanel({ token, userId }: { token: string; userId: string }) {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [data, setData] = useState<AuditResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const r = await apiJson<AuditResponse>("/audit", {
        method: "POST",
        token,
        body: JSON.stringify({ user_id: userId, response: text }),
      });
      setData(r);
    } catch (e) {
      if (e instanceof ApiError) setError(`HTTP ${e.status}: ${e.body.slice(0, 200)}`);
      else setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const real = data?.citations.filter((c) => c.found).length ?? 0;
  const fake = data?.citations.filter((c) => !c.found).length ?? 0;

  return (
    <Card
      title="Hallucination audit"
      subtitle="Paste any text · regex extracts UUIDs · DB verifies they belong to this user"
      endpoint="POST /audit"
    >
      <div className="space-y-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          className="w-full resize-y rounded-lg border border-ink-700 bg-ink-950/70 p-3 font-mono text-xs text-ink-100 placeholder-ink-400 focus:border-accent-blue focus:outline-none"
          placeholder="Paste a coaching response containing session UUIDs…"
        />
        <div className="flex items-center gap-3">
          <button
            onClick={run}
            disabled={loading || !token}
            className="rounded-lg bg-accent-violet/20 px-4 py-2 text-sm font-medium text-accent-violet transition hover:bg-accent-violet/30 disabled:opacity-50"
          >
            {loading ? "Verifying…" : "Audit citations →"}
          </button>
          {data ? (
            <div className="flex items-center gap-3 font-mono text-xs">
              <span className="text-accent-green">{real} real</span>
              <span className="text-ink-400">/</span>
              <span className="text-accent-rose">{fake} fake</span>
            </div>
          ) : null}
        </div>

        {error ? <ErrorBox>{error}</ErrorBox> : null}

        {data ? (
          <ul className="space-y-1.5">
            {data.citations.map((c) => (
              <li
                key={c.session_id}
                className={`flex items-center gap-3 rounded-md border px-3 py-2 ${
                  c.found
                    ? "border-accent-green/40 bg-accent-green/5"
                    : "border-accent-rose/40 bg-accent-rose/5"
                }`}
              >
                <span
                  className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                    c.found
                      ? "bg-accent-green/20 text-accent-green"
                      : "bg-accent-rose/20 text-accent-rose"
                  }`}
                >
                  {c.found ? "real" : "fake"}
                </span>
                <code className="truncate font-mono text-xs text-ink-200">
                  {c.session_id}
                </code>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </Card>
  );
}

"use client";

import { useEffect, useState } from "react";
import { apiJson, ApiError } from "@/lib/api";
import { PATHOLOGY_COLORS } from "@/lib/traders";
import { Card, ErrorBox, EmptyState } from "./Card";

type Citation = { trade_id?: string; session_id?: string; [k: string]: unknown };
type Scored = { pathology: string; score: number; evidence: Citation[] };
type ProfileResponse = {
  profile: {
    userId: string;
    primaryPathology: string;
    confidence: number;
    weaknesses?: { pattern: string; citations: Citation[] }[];
    narrative?: string;
  };
  scored: Scored[];
};

export function ProfilePanel({ token, userId }: { token: string; userId: string }) {
  const [data, setData] = useState<ProfileResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token || !userId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    apiJson<ProfileResponse>(`/profile/${userId}`, { token })
      .then((r) => !cancelled && setData(r))
      .catch((e: ApiError) =>
        !cancelled &&
        setError(`HTTP ${e.status}: ${e.body.slice(0, 200) || "request failed"}`),
      )
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [token, userId]);

  return (
    <Card
      title="Behavioral profile"
      subtitle="Top scored pathology with citations to real trades"
      endpoint="GET /profile/{userId}"
    >
      {loading ? (
        <EmptyState>Computing rules over trade history…</EmptyState>
      ) : error ? (
        <ErrorBox>{error}</ErrorBox>
      ) : !data ? (
        <EmptyState>No data yet.</EmptyState>
      ) : (
        <div className="space-y-4">
          <Top top={data.scored[0]} />
          <ScoredList scored={data.scored} />
          {data.profile.narrative ? (
            <p className="rounded-lg border border-ink-700 bg-ink-800/40 p-3 text-xs leading-relaxed text-ink-200">
              <span className="mr-2 font-mono text-[10px] uppercase tracking-wider text-ink-400">
                narrative
              </span>
              {data.profile.narrative}
            </p>
          ) : null}
        </div>
      )}
    </Card>
  );
}

function Top({ top }: { top: Scored }) {
  const colors = PATHOLOGY_COLORS[top.pathology] ?? PATHOLOGY_COLORS.none;
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-ink-700 bg-ink-800/40 p-4 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <div className="text-[11px] uppercase tracking-wider text-ink-400">primary pathology</div>
        <div className="mt-1 flex items-center gap-3">
          <span
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-medium ${colors}`}
          >
            {top.pathology}
          </span>
          <span className="font-mono text-sm text-ink-200">
            score {top.score.toFixed(2)}
          </span>
        </div>
      </div>
      <div className="text-[11px] uppercase tracking-wider text-ink-400">
        {top.evidence.length} citations
      </div>
    </div>
  );
}

function ScoredList({ scored }: { scored: Scored[] }) {
  const max = Math.max(...scored.map((s) => s.score), 0.0001);
  return (
    <div className="space-y-2">
      <div className="text-[11px] uppercase tracking-wider text-ink-400">
        all 9 pathologies (sorted)
      </div>
      <ul className="space-y-1.5">
        {scored.map((s) => {
          const colors = PATHOLOGY_COLORS[s.pathology] ?? PATHOLOGY_COLORS.none;
          const widthPct = Math.max(2, (s.score / max) * 100);
          return (
            <li
              key={s.pathology}
              className="grid grid-cols-[1fr_auto] items-center gap-3 rounded-md border border-ink-700/70 bg-ink-800/40 px-3 py-1.5"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate font-mono text-xs text-ink-100">
                    {s.pathology}
                  </span>
                  <span className="text-[10px] text-ink-400">
                    {s.evidence.length} cites
                  </span>
                </div>
                <div className="mt-1 h-1 overflow-hidden rounded-full bg-ink-700">
                  <div
                    className={`h-full ${colors.split(" ")[0].replace("text-", "bg-")}`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
              <span className="shrink-0 font-mono text-xs tabular-nums text-ink-300">
                {s.score.toFixed(3)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

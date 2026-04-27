"use client";

import { useEffect, useMemo, useState } from "react";
import { mintToken, apiFetch, ApiError } from "@/lib/api";
import { TRADERS, PATHOLOGY_COLORS, type Trader } from "@/lib/traders";
import { HealthBadge } from "@/components/HealthBadge";
import { ProfilePanel } from "@/components/ProfilePanel";
import { CoachingPanel } from "@/components/CoachingPanel";
import { AuditPanel } from "@/components/AuditPanel";
import { MemoryPanel } from "@/components/MemoryPanel";

export default function Home() {
  const [trader, setTrader] = useState<Trader>(TRADERS[0]);
  const [token, setToken] = useState<string>("");
  const [tenancyResult, setTenancyResult] = useState<{ status: number; ok: boolean } | null>(null);
  const [tenancyLoading, setTenancyLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setToken("");
    setTenancyResult(null);
    mintToken(trader.userId, trader.name).then((t) => !cancelled && setToken(t));
    return () => {
      cancelled = true;
    };
  }, [trader]);

  const otherUserId = useMemo(
    () => TRADERS.find((t) => t.userId !== trader.userId)?.userId ?? trader.userId,
    [trader],
  );

  const runTenancyCheck = async () => {
    setTenancyLoading(true);
    setTenancyResult(null);
    try {
      const r = await apiFetch(`/profile/${otherUserId}`, { token });
      setTenancyResult({ status: r.status, ok: r.status === 403 });
    } catch (e) {
      if (e instanceof ApiError) setTenancyResult({ status: e.status, ok: e.status === 403 });
    } finally {
      setTenancyLoading(false);
    }
  };

  const pathologyColor =
    PATHOLOGY_COLORS[trader.pathology ?? "none"] ?? PATHOLOGY_COLORS.none;

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <header className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <span className="inline-block h-8 w-8 rounded-lg bg-gradient-to-br from-accent-green via-accent-blue to-accent-violet" />
            <h1 className="text-2xl font-semibold tracking-tight text-ink-100">
              NevUp <span className="text-ink-400">·</span> AI Engine
            </h1>
          </div>
          <p className="mt-1 max-w-2xl text-sm text-ink-300">
            Stateful trading-psychology coach with verifiable memory, cited evidence,
            and streaming interventions. Every endpoint exercised below talks to the live
            FastAPI service over the public internet.
          </p>
        </div>
        <HealthBadge />
      </header>

      {/* Trader picker */}
      <section className="mb-8 rounded-2xl border border-ink-700/80 bg-ink-900/60 p-5 backdrop-blur">
        <div className="grid gap-5 lg:grid-cols-[1fr_auto] lg:items-end">
          <div className="space-y-3">
            <div>
              <div className="text-[11px] uppercase tracking-wider text-ink-400">
                authenticated trader
              </div>
              <div className="mt-1 flex items-center gap-3">
                <h2 className="text-xl font-semibold tracking-tight text-ink-100">
                  {trader.name}
                </h2>
                <span
                  className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${pathologyColor}`}
                >
                  {trader.pathology ?? "control · no pathology"}
                </span>
                <span className="font-mono text-xs text-ink-400">
                  {trader.totalTrades} trades
                </span>
              </div>
              <code className="mt-1 block truncate font-mono text-[11px] text-ink-400">
                sub: {trader.userId}
              </code>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {TRADERS.map((t) => (
                <button
                  key={t.userId}
                  onClick={() => setTrader(t)}
                  className={`rounded-md border px-2.5 py-1 text-xs transition ${
                    t.userId === trader.userId
                      ? "border-ink-300 bg-ink-700 text-ink-100"
                      : "border-ink-700 bg-ink-800/40 text-ink-300 hover:border-ink-600 hover:text-ink-100"
                  }`}
                >
                  {t.name}
                </button>
              ))}
            </div>
          </div>

          {/* Cross-tenant test */}
          <div className="rounded-xl border border-ink-700/80 bg-ink-800/40 p-4 lg:max-w-xs">
            <div className="text-[11px] uppercase tracking-wider text-ink-400">
              tenancy enforcement
            </div>
            <p className="mt-1 text-xs text-ink-300">
              Use {trader.name}'s JWT to ask for{" "}
              <span className="font-mono text-ink-200">
                {otherUserId.slice(0, 8)}…
              </span>
              's profile. Should return 403.
            </p>
            <div className="mt-3 flex items-center gap-2">
              <button
                onClick={runTenancyCheck}
                disabled={!token || tenancyLoading}
                className="rounded-md bg-ink-700 px-3 py-1.5 text-xs font-medium text-ink-100 transition hover:bg-ink-600 disabled:opacity-50"
              >
                {tenancyLoading ? "checking…" : "run cross-tenant check"}
              </button>
              {tenancyResult ? (
                <span
                  className={`font-mono text-xs ${
                    tenancyResult.ok ? "text-accent-green" : "text-accent-rose"
                  }`}
                >
                  {tenancyResult.ok ? "✓" : "✗"} HTTP {tenancyResult.status}
                </span>
              ) : null}
            </div>
          </div>
        </div>
      </section>

      {/* Main grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ProfilePanel token={token} userId={trader.userId} />
        <CoachingPanel token={token} userId={trader.userId} />
        <MemoryPanel token={token} userId={trader.userId} />
        <AuditPanel token={token} userId={trader.userId} />
      </div>

      {/* Footer */}
      <footer className="mt-10 flex flex-col items-start justify-between gap-3 border-t border-ink-700/60 pt-6 text-xs text-ink-400 sm:flex-row sm:items-center">
        <div>
          API:{" "}
          <a
            href={process.env.NEXT_PUBLIC_API_URL}
            target="_blank"
            rel="noreferrer"
            className="font-mono text-ink-300 hover:text-ink-100"
          >
            {process.env.NEXT_PUBLIC_API_URL}
          </a>
        </div>
        <div className="flex gap-3">
          <a
            href="https://github.com/ghostiee-11/nevup-ai-engine"
            target="_blank"
            rel="noreferrer"
            className="hover:text-ink-100"
          >
            github →
          </a>
          <a
            href={`${process.env.NEXT_PUBLIC_API_URL}/docs`}
            target="_blank"
            rel="noreferrer"
            className="hover:text-ink-100"
          >
            openapi →
          </a>
        </div>
      </footer>
    </main>
  );
}

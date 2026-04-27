"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";

type Status = "checking" | "ok" | "down";

export function HealthBadge() {
  const [status, setStatus] = useState<Status>("checking");
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const t0 = performance.now();
      try {
        const r = await apiFetch("/health");
        if (!cancelled) {
          setLatency(Math.round(performance.now() - t0));
          setStatus(r.ok ? "ok" : "down");
        }
      } catch {
        if (!cancelled) setStatus("down");
      }
    };
    check();
    const id = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const dot =
    status === "ok"
      ? "bg-accent-green"
      : status === "down"
        ? "bg-accent-rose"
        : "bg-accent-amber animate-pulse_dot";
  const label =
    status === "ok"
      ? `live · ${latency}ms`
      : status === "down"
        ? "down"
        : "checking…";

  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-ink-700 bg-ink-800/60 px-3 py-1 text-xs font-medium text-ink-200 backdrop-blur">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      <span className="font-mono">{label}</span>
    </span>
  );
}

import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  endpoint,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  endpoint?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-2xl border border-ink-700/80 bg-ink-900/60 p-5 backdrop-blur transition hover:border-ink-600 ${className}`}
    >
      <header className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold tracking-tight text-ink-100">{title}</h2>
          {subtitle ? (
            <p className="mt-0.5 text-xs text-ink-400">{subtitle}</p>
          ) : null}
        </div>
        {endpoint ? (
          <code className="shrink-0 rounded-md border border-ink-700 bg-ink-800/80 px-2 py-1 font-mono text-[11px] text-ink-300">
            {endpoint}
          </code>
        ) : null}
      </header>
      {children}
    </section>
  );
}

export function ErrorBox({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-accent-rose/40 bg-accent-rose/10 px-3 py-2 text-xs text-accent-rose">
      {children}
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-ink-700 bg-ink-800/30 p-4 text-center text-xs text-ink-400">
      {children}
    </div>
  );
}

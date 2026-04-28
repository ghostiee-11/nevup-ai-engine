"""Render k6's summary.json into a static HTML report.

Trivial template — k6's terminal output is already excellent; this just makes
the numbers shareable as a single file artifact (e.g. attach to a PR).
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

TEMPLATE = """<!doctype html>
<meta charset="utf-8" />
<title>NevUp · k6 SSE load test</title>
<style>
  body {{ font: 14px/1.5 ui-sans-serif, system-ui, -apple-system; background: #0d0c12;
         color: #ecebef; padding: 32px; max-width: 920px; margin: 0 auto; }}
  h1, h2 {{ margin-top: 1.6em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid #2c2a36; font-variant-numeric: tabular-nums; }}
  th {{ font-weight: 600; color: #74717f; text-transform: uppercase; font-size: 11px; letter-spacing: .05em; }}
  .ok {{ color: #21c98a; }}
  .bad {{ color: #ff5d6c; }}
  .meta {{ color: #74717f; font-size: 12px; margin-top: 4px; }}
  pre {{ background: #16151c; padding: 12px; border-radius: 8px; overflow-x: auto; }}
</style>

<h1>NevUp · k6 SSE load test</h1>
<p class="meta">Source: <code>{src}</code> · iterations: {iters} · vus_max: {vus_max}</p>

<h2>Threshold checks</h2>
<table><tr><th>metric</th><th>threshold</th><th>status</th></tr>
{threshold_rows}
</table>

<h2>Headline metrics</h2>
<table><tr><th>metric</th><th>min</th><th>med</th><th>avg</th><th>p(95)</th><th>p(99)</th><th>max</th></tr>
{metric_rows}
</table>

<h2>Counters</h2>
<table><tr><th>metric</th><th>count</th><th>rate</th></tr>
{counter_rows}
</table>

<h2>Notes</h2>
<ul>
  <li><code>sse_first_byte_ms</code> uses k6's <code>http_req_waiting</code> — the
      time between sending the request and receiving the first byte of the body.
      For our SSE endpoint, that first byte is the <code>:&nbsp;connecting</code>
      keep-alive comment we send before any Groq output.</li>
  <li><code>GEMINI_API_KEY</code> and <code>GROQ_API_KEY</code> were unset for
      this run, so embeddings used the SHA fallback and coaching used the stub
      stream. This isolates server-side handling from external API latency.</li>
</ul>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()

    data = json.loads(args.summary.read_text())
    metrics = data.get("metrics", {})
    state = data.get("state", {})

    def fmt_threshold(name: str, t: dict) -> str:
        ok = all(not c.get("ok", False) is False for c in [t]) and not t.get("ok", True) is False
        rows = []
        for src, info in (t.get("thresholds") or {}).items():
            ok = info.get("ok", False)
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(src)}</code></td>"
                f"<td class={'ok' if ok else 'bad'!r}>{'PASS' if ok else 'FAIL'}</td></tr>"
            )
        return "\n".join(rows)

    threshold_rows = "\n".join(
        fmt_threshold(name, m) for name, m in metrics.items() if m.get("thresholds")
    ) or "<tr><td colspan=3>(no thresholds defined)</td></tr>"

    headline_keys = ("sse_first_byte_ms", "http_req_duration", "http_req_waiting", "iteration_duration")
    metric_rows = []
    for k in headline_keys:
        m = metrics.get(k)
        if not m:
            continue
        v = m.get("values", {})
        cells = [
            f"{v.get(stat, 0):.1f}" for stat in ("min", "med", "avg", "p(95)", "p(99)", "max")
        ]
        metric_rows.append(
            f"<tr><td><code>{k}</code></td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        )
    metric_rows = "\n".join(metric_rows) or "<tr><td colspan=7>(no values)</td></tr>"

    counter_keys = ("http_reqs", "http_req_failed", "iterations", "sse_success_rate")
    counter_rows = []
    for k in counter_keys:
        m = metrics.get(k)
        if not m:
            continue
        v = m.get("values", {})
        count = v.get("count") or v.get("passes") or v.get("rate")
        rate = v.get("rate")
        counter_rows.append(
            f"<tr><td><code>{k}</code></td><td>{count}</td><td>{rate or ''}</td></tr>"
        )
    counter_rows = "\n".join(counter_rows) or "<tr><td colspan=3>(no counters)</td></tr>"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(TEMPLATE.format(
        src=str(args.summary),
        iters=int(metrics.get("iterations", {}).get("values", {}).get("count", 0)),
        vus_max=int(state.get("testRunDurationMs", 0) / 1000) if not metrics.get("vus_max") else
                int(metrics["vus_max"]["values"].get("max", 0)),
        threshold_rows=threshold_rows,
        metric_rows=metric_rows,
        counter_rows=counter_rows,
    ))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

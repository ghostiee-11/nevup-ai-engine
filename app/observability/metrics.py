"""Minimal in-process metrics. Replace with prometheus_client when we move
beyond a single uvicorn worker. For now this is enough to expose JSON
counters at GET /metrics for the load test and ops dashboards.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict


class Counter:
    """Labelled, monotonic counter. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple, float] = defaultdict(float)

    def inc(self, *, by: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] += by

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {"labels": dict(k), "value": v}
                for k, v in self._values.items()
            ]


class Histogram:
    """Bucketed latency histogram with tracked sum/count for averages.
    Buckets are in milliseconds; +Inf is implicit.
    """

    DEFAULT_BUCKETS_MS = (5, 10, 25, 50, 100, 200, 400, 800, 1500, 3000, 10000)

    def __init__(self, buckets_ms: tuple[float, ...] = DEFAULT_BUCKETS_MS) -> None:
        self._lock = threading.Lock()
        self._buckets_ms = buckets_ms
        self._counts: dict[tuple, list[int]] = defaultdict(lambda: [0] * (len(buckets_ms) + 1))
        self._sums: dict[tuple, float] = defaultdict(float)
        self._observations: dict[tuple, int] = defaultdict(int)

    def observe(self, latency_ms: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._sums[key] += latency_ms
            self._observations[key] += 1
            for i, b in enumerate(self._buckets_ms):
                if latency_ms <= b:
                    self._counts[key][i] += 1
                    return
            self._counts[key][-1] += 1  # +Inf bucket

    def snapshot(self) -> list[dict]:
        out = []
        with self._lock:
            for key in self._observations:
                bucket_dict = {f"≤{b}ms": self._counts[key][i] for i, b in enumerate(self._buckets_ms)}
                bucket_dict["+Inf"] = self._counts[key][-1]
                count = self._observations[key]
                out.append({
                    "labels": dict(key),
                    "buckets": bucket_dict,
                    "count": count,
                    "sum_ms": round(self._sums[key], 2),
                    "mean_ms": round(self._sums[key] / count, 2) if count else 0,
                })
        return out


# Module-level singletons referenced by middleware and embedders.
requests_total = Counter()
request_latency_ms = Histogram()
embedding_fallback_total = Counter()


def all_metrics() -> dict:
    return {
        "requests_total": requests_total.snapshot(),
        "request_latency_ms": request_latency_ms.snapshot(),
        "embedding_fallback_total": embedding_fallback_total.snapshot(),
        "uptime_seconds": round(time.monotonic() - _STARTED_AT, 1),
    }


_STARTED_AT = time.monotonic()

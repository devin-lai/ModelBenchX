"""Latency statistics from a vector of per-iteration timings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..results import TimingStats


def cv_pct(t: TimingStats | None) -> float | None:
    """Coefficient of variation (%): steady-state jitter/stability. Lower is steadier."""
    if t is None or not t.mean_ms:
        return None
    return 100.0 * t.std_ms / t.mean_ms


def cold_start_ms(t: TimingStats | None) -> float | None:
    """Time-to-first-inference: model load/compile + the first (cold) inference."""
    if t is None or t.load_ms is None:
        return None
    return t.load_ms + (t.first_call_ms or 0.0)


def summarize(
    raw_ms: Sequence[float],
    *,
    load_ms: float | None = None,
    first_call_ms: float | None = None,
    keep_raw: bool = True,
) -> TimingStats:
    """Reduce per-iteration latencies (ms) to a :class:`TimingStats`.

    ``std`` is the sample standard deviation (ddof=1) when more than one sample
    is present. Percentiles use linear interpolation. Throughput is derived from
    the mean (``1000 / mean_ms``).
    """
    a = np.asarray(list(raw_ms), dtype=np.float64)
    if a.size == 0:
        raise ValueError("summarize() requires at least one timing sample")
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    p90, p95, p99 = (float(x) for x in np.percentile(a, [90, 95, 99]))
    return TimingStats(
        iters=int(a.size),
        mean_ms=mean,
        median_ms=float(np.median(a)),
        std_ms=std,
        min_ms=float(a.min()),
        max_ms=float(a.max()),
        p90_ms=p90,
        p95_ms=p95,
        p99_ms=p99,
        throughput_ips=(1000.0 / mean) if mean > 0 else float("inf"),
        load_ms=load_ms,
        first_call_ms=first_call_ms,
        raw_ms=[float(x) for x in a] if keep_raw else None,
    )

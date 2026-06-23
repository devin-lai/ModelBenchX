"""Shared steady-state timing harness used by every worker.

Sampling policy (one place, so all backends agree):

* warm up first: at least ``warmup`` calls, then continue until the latency
  flattens (so a lazy-specializing backend's ramp does not leak into timing),
  bounded by ``warmup_budget_s``/``_WARMUP_MAX`` so it never runs away. Warmup
  calls are discarded, but the first call's latency is recorded; it carries the
  lazy compile/specialize cost.
* then time calls until ``min_iters`` is reached and the time budget is spent,
  never exceeding ``max_iters``. Fast models get the full ``max_iters`` (tight
  statistics); slow models stop near ``min_iters`` (bounded wall-clock).

Cyclic GC is frozen for the whole measurement (warmup + timed) so a collection
pause cannot land inside a sample and distort the tail percentiles / CV.

Provides a synchronous and an async variant sharing the same stop rule, so the
async Core AI runtime is timed in its own event loop rather than paying
``asyncio.run`` per call.
"""

from __future__ import annotations

import gc
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from statistics import median
from time import perf_counter_ns
from typing import Any

_WARMUP_BUDGET_S = 8.0
_MS = 1_000_000.0

# Adaptive warmup: ``warmup`` is a floor, after which calls continue until the
# latency flattens. This matters because lazy-specializing backends (Core ML on
# ANE/GPU) keep getting faster for the first several calls; a fixed tiny warmup
# leaves that ramp inside the timed window and inflates the reported latency.
# Steady state = median of the last ``_WARMUP_WINDOW`` calls within
# ``_WARMUP_TOL`` (relative) of the preceding window. Bounded by
# ``_WARMUP_BUDGET_S`` and a hard ``_WARMUP_MAX`` so warmup never runs away.
_WARMUP_WINDOW = 5
_WARMUP_TOL = 0.05
_WARMUP_MAX = 200


def _warmup_stabilized(times: list[float], window: int = _WARMUP_WINDOW, tol: float = _WARMUP_TOL) -> bool:
    """True once latency has flattened across two consecutive windows.

    Pure and deterministic (no clock), so the convergence rule is unit-tested
    without running a real backend. Needs at least ``2*window`` samples.
    """
    if len(times) < 2 * window:
        return False
    prev = median(times[-2 * window:-window])
    cur = median(times[-window:])
    if cur <= 0:
        return True
    return abs(cur - prev) / cur <= tol

# Slow-model wall: once a run has spent this long with at least this many
# samples, stop even if ``min_iters`` is not reached. Bounds wall-clock for
# multi-second-per-call paths (e.g. Core AI cpu_only on large graphs) while
# still collecting a few samples. The report flags low sample counts.
_SLOW_WALL_S = 15.0
_SLOW_FLOOR = 3


def _should_stop(i: int, elapsed_s: float, min_iters: int, max_iters: int, budget_s: float) -> bool:
    if i >= max_iters:
        return True
    if i >= _SLOW_FLOOR and elapsed_s >= _SLOW_WALL_S:
        return True
    return i >= min_iters and elapsed_s >= budget_s


@contextmanager
def _gc_disabled():
    """Freeze cyclic GC for the duration of a measurement.

    A GC pause that lands inside a timed call inflates that one sample and
    pollutes std/max/p95/p99/CV. Collect once for a clean heap, disable
    automatic collection while sampling, then restore the prior state (even on
    error). Mirrors what ``timeit`` does.
    """
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def run_timed(
    call: Callable[[], Any],
    *,
    warmup: int,
    min_iters: int,
    max_iters: int,
    time_budget_s: float,
) -> tuple[list[float], float | None, Any]:
    """Synchronous timing loop. Returns ``(raw_ms, first_call_ms, last_output)``."""
    first_call_ms: float | None = None
    last_out: Any = None
    w_times: list[float] = []
    raw: list[float] = []

    with _gc_disabled():
        w_start = perf_counter_ns()
        i = 0
        while warmup > 0:
            t0 = perf_counter_ns()
            last_out = call()
            dt = (perf_counter_ns() - t0) / _MS
            if i == 0:
                first_call_ms = dt
            w_times.append(dt)
            i += 1
            if (perf_counter_ns() - w_start) / 1e9 > _WARMUP_BUDGET_S or i >= _WARMUP_MAX:
                break
            if i >= warmup and _warmup_stabilized(w_times):
                break

        start = perf_counter_ns()
        i = 0
        while True:
            t0 = perf_counter_ns()
            last_out = call()
            t1 = perf_counter_ns()
            raw.append((t1 - t0) / _MS)
            i += 1
            if _should_stop(i, (perf_counter_ns() - start) / 1e9, min_iters, max_iters, time_budget_s):
                break
    return raw, first_call_ms, last_out


async def arun_timed(
    call: Callable[[], Awaitable[Any]],
    *,
    warmup: int,
    min_iters: int,
    max_iters: int,
    time_budget_s: float,
) -> tuple[list[float], float | None, Any]:
    """Async timing loop (same stop rule). Returns ``(raw_ms, first_call_ms, last_output)``."""
    first_call_ms: float | None = None
    last_out: Any = None
    w_times: list[float] = []
    raw: list[float] = []

    with _gc_disabled():
        w_start = perf_counter_ns()
        i = 0
        while warmup > 0:
            t0 = perf_counter_ns()
            last_out = await call()
            dt = (perf_counter_ns() - t0) / _MS
            if i == 0:
                first_call_ms = dt
            w_times.append(dt)
            i += 1
            if (perf_counter_ns() - w_start) / 1e9 > _WARMUP_BUDGET_S or i >= _WARMUP_MAX:
                break
            if i >= warmup and _warmup_stabilized(w_times):
                break

        start = perf_counter_ns()
        i = 0
        while True:
            t0 = perf_counter_ns()
            last_out = await call()
            t1 = perf_counter_ns()
            raw.append((t1 - t0) / _MS)
            i += 1
            if _should_stop(i, (perf_counter_ns() - start) / 1e9, min_iters, max_iters, time_budget_s):
                break
    return raw, first_call_ms, last_out

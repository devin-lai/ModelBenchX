import asyncio
import gc
from pathlib import Path

from modelbenchx.config import BenchmarkConfig
from modelbenchx.workers._bench import _should_stop, _warmup_stabilized, arun_timed, run_timed


def _cfg(**kw):
    kw.setdefault("test_model_dir", Path("/x"))
    kw.setdefault("results_dir", Path("/y"))
    return BenchmarkConfig(**kw)


# ---- timing stop rule ------------------------------------------------------

def test_stop_at_max_iters():
    assert _should_stop(50, 0.1, min_iters=10, max_iters=50, budget_s=3.0)


def test_stop_after_min_and_budget():
    # below min: keep going even past budget
    assert not _should_stop(5, 5.0, min_iters=10, max_iters=50, budget_s=3.0)
    # at min and past budget: stop
    assert _should_stop(10, 3.1, min_iters=10, max_iters=50, budget_s=3.0)


def test_fast_model_runs_to_max_not_budget():
    # 0.5ms model: 20 iters in 0.01s, below budget -> keep going (not stopped)
    assert not _should_stop(20, 0.01, min_iters=10, max_iters=50, budget_s=3.0)


def test_slow_model_wall_bounds_below_min_iters():
    # 5s/call model: at i=3, elapsed 15s -> stop even though min_iters=10
    assert _should_stop(3, 15.0, min_iters=10, max_iters=50, budget_s=3.0)
    assert not _should_stop(2, 10.0, min_iters=10, max_iters=50, budget_s=3.0)


# ---- adaptive warmup (steady-state detection) ------------------------------

def test_warmup_not_stabilized_without_enough_samples():
    # Fewer than 2*window samples: cannot judge yet.
    assert not _warmup_stabilized([1.0, 1.0, 1.0])


def test_warmup_stabilized_when_flat():
    assert _warmup_stabilized([1.0] * 10)


def test_warmup_not_stabilized_while_ramping_down():
    # Still descending fast (lazy ANE/GPU specialization): last window much
    # lower than the preceding one -> not yet steady.
    ramp = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.1, 1.0]
    assert not _warmup_stabilized(ramp)


def test_warmup_stabilized_after_ramp_settles():
    # Once enough steady calls accumulate that both trailing windows are flat,
    # convergence is detected even though the run began with a ramp.
    settled = [5, 4, 3, 2, 1.5] + [1.01, 1.0, 1.0, 1.0, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert _warmup_stabilized(settled)


def test_run_timed_adaptive_warmup_skips_ramp_then_times_steady():
    # A callable that is slow for its first 12 calls then steady at ~1.0.
    # Adaptive warmup must absorb the ramp so the timed window is steady.
    seq = iter([8.0, 6.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.3, 1.1, 1.05, 1.02, 1.01]
               + [1.0] * 200)
    now = {"t": 0}

    # Drive the monotonic clock off the simulated per-call cost (ms -> ns).
    import modelbenchx.workers._bench as b
    real = b.perf_counter_ns
    pending = {"dt_ms": 0.0}

    def fake_now():
        return int(now["t"])

    def call():
        dt = next(seq)
        pending["dt_ms"] = dt
        now["t"] += dt * 1_000_000  # advance clock by the call's cost
        return dt

    b.perf_counter_ns = fake_now
    try:
        raw, first_call_ms, _ = run_timed(
            call, warmup=3, min_iters=10, max_iters=30, time_budget_s=0.0
        )
    finally:
        b.perf_counter_ns = real

    assert first_call_ms == 8.0  # first (cold) call recorded
    # Every timed sample is the steady value, not a ramp sample.
    assert raw and all(abs(x - 1.0) < 1e-9 for x in raw)


# ---- async timing loop (arun_timed: same stop/warmup rule as run_timed) -----

def test_arun_timed_adaptive_warmup_skips_ramp_then_times_steady():
    # Async mirror of the sync test. arun_timed is a separate (hand-duplicated)
    # loop, so it gets its own coverage to catch drift from run_timed.
    seq = iter([8.0, 6.0, 4.0, 3.0, 2.5, 2.0, 1.6, 1.3, 1.1, 1.05, 1.02, 1.01]
               + [1.0] * 200)
    now = {"t": 0}

    import modelbenchx.workers._bench as b
    real = b.perf_counter_ns

    def fake_now():
        return int(now["t"])

    async def call():
        dt = next(seq)
        now["t"] += dt * 1_000_000  # advance clock by the call's cost
        return dt

    b.perf_counter_ns = fake_now
    try:
        raw, first_call_ms, _ = asyncio.run(
            arun_timed(call, warmup=3, min_iters=10, max_iters=30, time_budget_s=0.0)
        )
    finally:
        b.perf_counter_ns = real

    assert first_call_ms == 8.0  # first (cold) call recorded
    assert raw and all(abs(x - 1.0) < 1e-9 for x in raw)


# ---- GC frozen during measurement (a collection pause must not hit a sample) -

def test_run_timed_disables_gc_during_calls_and_restores():
    import modelbenchx.workers._bench as b
    gc.enable()  # known starting state
    real = b.perf_counter_ns
    now = {"t": 0}
    seen: list[bool] = []

    def fake_now():
        return now["t"]

    def call():
        seen.append(gc.isenabled())
        now["t"] += 1_000_000  # 1 ms/call so the loop terminates
        return None

    b.perf_counter_ns = fake_now
    try:
        run_timed(call, warmup=2, min_iters=3, max_iters=3, time_budget_s=0.0)
    finally:
        b.perf_counter_ns = real

    assert seen, "call() was never invoked"
    assert not any(seen), "GC must be disabled during every measured call"
    assert gc.isenabled(), "GC must be restored to enabled after timing"


def test_arun_timed_disables_gc_during_calls_and_restores():
    import modelbenchx.workers._bench as b
    gc.enable()
    real = b.perf_counter_ns
    now = {"t": 0}
    seen: list[bool] = []

    def fake_now():
        return now["t"]

    async def call():
        seen.append(gc.isenabled())
        now["t"] += 1_000_000
        return None

    b.perf_counter_ns = fake_now
    try:
        asyncio.run(arun_timed(call, warmup=2, min_iters=3, max_iters=3, time_budget_s=0.0))
    finally:
        b.perf_counter_ns = real

    assert seen
    assert not any(seen), "GC must be disabled during every measured call (async)"
    assert gc.isenabled()


# ---- model selection -------------------------------------------------------

def test_selects_all_when_none():
    c = _cfg()
    assert c.selects_model("resnet50__resnet50")


def test_selects_by_model_prefix_and_exact_key():
    c = _cfg(models=("resnet50",))
    assert c.selects_model("resnet50__resnet50")
    assert not c.selects_model("resnet101__resnet101")

    c2 = _cfg(models=("sam2__encoder",))
    assert c2.selects_model("sam2__encoder")
    assert not c2.selects_model("sam2__decoder")


def test_results_subpaths():
    c = _cfg(results_dir=Path("/tmp/r"))
    assert c.runs_dir == Path("/tmp/r/runs")
    assert c.cache_dir == Path("/tmp/r/cache")
    assert c.reports_dir == Path("/tmp/r/reports")

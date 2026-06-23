"""Orchestrator feed/accuracy decoupling (end-to-end via the synth backend).

The synth backend is the only framework available dep-free, so it is the real
end-to-end check that orchestrator -> harness -> worker -> RunResult works. With
no onnx reference present the run is latency-only: timing is measured, accuracy is
None. The reference/consumer-with-models paths need real models + runtimes (not
available here) and are correct by construction.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import platform

import numpy as np
import pytest


def _write_synth_graph(root, key="m__m"):
    synth = root / "synth"
    synth.mkdir(parents=True, exist_ok=True)
    npz = synth / f"{key}.npmodel.npz"
    np.savez(npz, w=np.ones((3, 2), np.float32))
    # The discovered suffix is ``.npmodel``; the worker np.load()s it as npz.
    (synth / f"{key}.npmodel").write_bytes(npz.read_bytes())


def test_latency_only_when_no_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELBENCHX_SYNTH", "1")
    from modelbenchx import backends, config, orchestrator, registry

    for m in (backends.base, registry, orchestrator):
        importlib.reload(m)
    try:
        _write_synth_graph(tmp_path)
        cfg = config.BenchmarkConfig(
            test_model_dir=tmp_path,
            results_dir=tmp_path / "out",
            reference_backend="onnxruntime",
            warmup=1,
            min_iters=1,
            max_iters=1,
        )
        results = orchestrator.Orchestrator(cfg).run()
        synth = [r for r in results if r.backend == "synth"]
        assert synth and synth[0].status == "ok"
        assert synth[0].timing is not None  # latency is always measured
        assert synth[0].accuracy is None  # latency-only: no onnx reference present
    finally:
        monkeypatch.delenv("MODELBENCHX_SYNTH", raising=False)
        for m in (backends.base, registry, orchestrator):
            importlib.reload(m)


def test_feed_source_backend_runs_all_its_modes(tmp_path, monkeypatch):
    """A provides_feed backend that is the feed source (no onnx reference) must
    still benchmark every planned mode, not only the mode that generated the
    shared feed. The feed worker runs modes[0]; the remaining modes must run as
    ordinary consumers. Regression: the consumer loop used to skip the whole
    feed backend, silently dropping its other modes (e.g. tflite xnnpack)."""
    monkeypatch.setenv("MODELBENCHX_SYNTH", "1")
    from modelbenchx import backends, config, orchestrator, registry

    for m in (backends.base, registry, orchestrator):
        importlib.reload(m)
    try:
        base = backends.base
        # Give the dep-free synth feed-source backend a second mode so the
        # multi-mode feed-source path is exercised without a real runtime.
        synth = base.get_backend("synth")
        synth2 = dataclasses.replace(
            synth, modes=synth.modes + (config.Mode("alt", "CPU (alt)", config.FP32),)
        )
        patched = tuple(synth2 if b.name == "synth" else b for b in base.BACKENDS)
        monkeypatch.setattr(base, "BACKENDS", patched)
        monkeypatch.setattr(base, "_BY_NAME", {b.name: b for b in patched})

        _write_synth_graph(tmp_path)
        cfg = config.BenchmarkConfig(
            test_model_dir=tmp_path, results_dir=tmp_path / "out",
            warmup=1, min_iters=1, max_iters=1,
        )
        results = orchestrator.Orchestrator(cfg).run()
        modes = {r.mode_id for r in results if r.backend == "synth" and r.status == "ok"}
        assert modes == {"cpu", "alt"}, f"both feed-source modes must be benchmarked, got {modes}"
    finally:
        monkeypatch.delenv("MODELBENCHX_SYNTH", raising=False)
        for m in (backends.base, registry, orchestrator):
            importlib.reload(m)


def _reload(*mods):
    for m in mods:
        importlib.reload(m)


def test_resumable_cache_reuse_force_and_feed_change(tmp_path, monkeypatch):
    """The headline 'resumable + cached' guarantee, end-to-end: a committed cache
    is reused (worker not re-run), --force re-runs, and a changed feed fingerprint
    forces regeneration. Exercises the orchestrator.run() wiring that the
    predicate-only cache unit tests never reach."""
    monkeypatch.setenv("MODELBENCHX_SYNTH", "1")
    from modelbenchx import backends, config, orchestrator, registry

    _reload(backends.base, registry, orchestrator)
    try:
        _write_synth_graph(tmp_path)
        cfg = config.BenchmarkConfig(
            test_model_dir=tmp_path, results_dir=tmp_path / "out",
            warmup=1, min_iters=1, max_iters=1,
        )
        calls: list[int] = []
        real_execute = orchestrator.P.execute

        def counting_execute(*a, **k):
            calls.append(1)
            return real_execute(*a, **k)

        monkeypatch.setattr(orchestrator.P, "execute", counting_execute)

        orchestrator.Orchestrator(cfg).run()
        assert len(calls) == 1  # the synth feed worker ran once

        orchestrator.Orchestrator(cfg).run()
        assert len(calls) == 1, "a committed cache must be reused, not re-run"

        orchestrator.Orchestrator(dataclasses.replace(cfg, force=True)).run()
        assert len(calls) == 2, "--force must re-run"

        fp = cfg.cache_dir / "m__m" / orchestrator.FEED_FINGERPRINT
        data = json.loads(fp.read_text())
        data["seed"] = 999  # a changed feed parameter
        fp.write_text(json.dumps(data))
        orchestrator.Orchestrator(cfg).run()
        assert len(calls) == 3, "a stale feed fingerprint must force regeneration"
    finally:
        monkeypatch.delenv("MODELBENCHX_SYNTH", raising=False)
        _reload(backends.base, registry, orchestrator)


def test_resume_survives_corrupt_cached_result(tmp_path, monkeypatch):
    """A corrupt cached run JSON must be treated as a cache miss (regenerate), not
    raise json.JSONDecodeError and abort the whole resumable sweep."""
    monkeypatch.setenv("MODELBENCHX_SYNTH", "1")
    from modelbenchx import backends, config, orchestrator, registry

    _reload(backends.base, registry, orchestrator)
    try:
        _write_synth_graph(tmp_path)
        cfg = config.BenchmarkConfig(
            test_model_dir=tmp_path, results_dir=tmp_path / "out",
            warmup=1, min_iters=1, max_iters=1,
        )
        orchestrator.Orchestrator(cfg).run()  # populate the cache
        result_path = cfg.runs_dir / "m__m" / "synth__cpu.json"
        result_path.write_text("{ truncated json")  # simulate a corrupt commit marker

        results = orchestrator.Orchestrator(cfg).run()  # must not raise
        synth = [r for r in results if r.backend == "synth"]
        assert synth and synth[0].status == "ok"  # regenerated cleanly
    finally:
        monkeypatch.delenv("MODELBENCHX_SYNTH", raising=False)
        _reload(backends.base, registry, orchestrator)


def test_failed_feed_commit_marker_forces_regeneration(tmp_path, monkeypatch):
    """The atomically-written run result is the feed/baseline commit marker. If it
    records STATUS_FAILED (a worker killed mid-write, npz files left partial), the
    cache must NOT be reused — even though the feed files still exist — so fresh
    inputs are never paired with a stale baseline (silently wrong accuracy)."""
    monkeypatch.setenv("MODELBENCHX_SYNTH", "1")
    from modelbenchx import backends, config, orchestrator, registry
    from modelbenchx.results import STATUS_FAILED, RunResult

    _reload(backends.base, registry, orchestrator)
    try:
        _write_synth_graph(tmp_path)
        cfg = config.BenchmarkConfig(
            test_model_dir=tmp_path, results_dir=tmp_path / "out",
            warmup=1, min_iters=1, max_iters=1,
        )
        orchestrator.Orchestrator(cfg).run()  # populate a committed cache
        result_path = cfg.runs_dir / "m__m" / "synth__cpu.json"
        rr = RunResult.load(result_path)
        rr.status = STATUS_FAILED  # demote the commit marker
        rr.save(result_path)

        calls: list[int] = []
        real_execute = orchestrator.P.execute

        def counting_execute(*a, **k):
            calls.append(1)
            return real_execute(*a, **k)

        monkeypatch.setattr(orchestrator.P, "execute", counting_execute)
        orchestrator.Orchestrator(cfg).run()  # plain resume, not --force
        assert len(calls) == 1, "an uncommitted (failed) feed must be regenerated, not reused"
    finally:
        monkeypatch.delenv("MODELBENCHX_SYNTH", raising=False)
        _reload(backends.base, registry, orchestrator)


def test_result_from_outcome_accuracy_gating(tmp_path, monkeypatch):
    """The reference-decoupling core: a non-reference consumer gets PSNR vs the
    baseline, while the reference backend itself stays latency-only (accuracy
    None). This branch is otherwise only 'correct by construction'."""
    monkeypatch.setenv("MODELBENCHX_SYNTH", "1")
    from modelbenchx import backends, config, orchestrator, registry
    from modelbenchx.workers import _io as npio
    from modelbenchx.workers import _protocol as P

    _reload(backends.base, registry, orchestrator)
    try:
        synth_backend = backends.base.get_backend("synth")
        mode = synth_backend.modes[0]
        model_file = tmp_path / "m.synth"
        model_file.write_bytes(b"x")  # only needs to exist (for _path_signature)
        record = registry.GraphRecord(
            key="m__m", model="m", component="m",
            sources={"synth": registry.GraphSource(fmt="synth", path=str(model_file))},
        )
        jobdir = tmp_path / "job"
        jobdir.mkdir()
        npio.save_named(jobdir / P.OUTPUTS, ["y"], {"y": np.array([1.0, 2.0, 3.0])})
        outcome = P.WorkerOutcome(
            ok=True, crashed=False, returncode=0,
            result={"raw_ms": [1.0, 1.1], "produced_output_names": ["y"],
                    "note": "", "realized_device": None},
        )
        baseline = {"y": np.array([1.0, 2.0, 3.0])}

        # backend IS the reference -> never scored against itself.
        orch_ref = orchestrator.Orchestrator(config.BenchmarkConfig(
            test_model_dir=tmp_path, results_dir=tmp_path / "out", reference_backend="synth"))
        rr_ref = orch_ref._result_from_outcome(
            record, synth_backend, mode, outcome,
            duration_s=0.1, baseline_outputs=baseline, jobdir=jobdir)
        assert rr_ref.timing is not None
        assert rr_ref.accuracy is None

        # backend is a CONSUMER (reference is a different backend) -> accuracy attached.
        orch_con = orchestrator.Orchestrator(config.BenchmarkConfig(
            test_model_dir=tmp_path, results_dir=tmp_path / "out", reference_backend="onnxruntime"))
        rr_con = orch_con._result_from_outcome(
            record, synth_backend, mode, outcome,
            duration_s=0.1, baseline_outputs=baseline, jobdir=jobdir)
        assert rr_con.accuracy is not None
        assert rr_con.accuracy.min_psnr_db == float("inf")  # identical outputs -> bit-exact
    finally:
        monkeypatch.delenv("MODELBENCHX_SYNTH", raising=False)
        _reload(backends.base, registry, orchestrator)


@pytest.mark.skipif(platform.system() != "Darwin", reason="coreml-mlmodel backend is Darwin-only")
def test_graph_skipped_when_no_feed_source(tmp_path):
    """A graph whose only source is a non-feed-capable format with no reference
    is skipped (no worker runs), not crashed. coreml-mlmodel cannot generate a
    feed and there is no onnx reference, so the run is recorded as skipped."""
    from modelbenchx import config, orchestrator

    mlmodel = tmp_path / "mlmodel"
    mlmodel.mkdir()
    (mlmodel / "m__m.mlmodel").write_bytes(b"x")  # never opened: skipped before exec
    cfg = config.BenchmarkConfig(
        test_model_dir=tmp_path, results_dir=tmp_path / "out",
        warmup=1, min_iters=1, max_iters=1,
    )
    results = orchestrator.Orchestrator(cfg).run()
    assert results, "expected skipped results, not an empty run"
    assert all(r.status == "skipped" for r in results)
    assert all("no feed source" in r.note for r in results)


def test_skipped_run_is_reattempted_on_resume(tmp_path, monkeypatch):
    """A cached SKIPPED result must not be reused on resume. A skip records only
    that no feed was available at the time (a precondition outside this run's
    identity); reusing it would leave a backend that was skipped before its
    reference existed permanently skipped until --force. It must be re-attempted —
    and, now that a feed exists, succeed."""
    monkeypatch.setenv("MODELBENCHX_SYNTH", "1")
    from modelbenchx import backends, config, orchestrator, registry
    from modelbenchx.results import STATUS_OK

    _reload(backends.base, registry, orchestrator)
    try:
        _write_synth_graph(tmp_path)
        cfg = config.BenchmarkConfig(
            test_model_dir=tmp_path, results_dir=tmp_path / "out",
            warmup=1, min_iters=1, max_iters=1,
        )
        orch = orchestrator.Orchestrator(cfg)
        synth = backends.base.get_backend("synth")
        mode = synth.modes[0]
        record = registry.discover(tmp_path).get("m__m")

        # A real shared feed now exists, so a re-attempt can actually run.
        state = orch._ensure_feed(record, [synth], None)
        assert state.ok

        # Simulate an earlier run that recorded this (graph, backend, mode) as
        # skipped (no feed back then), WITH a matching model_sig so the freshness
        # check alone would treat it as reusable.
        rp = orch._result_path(record.key, synth.name, mode.id)
        skip = orch._skipped(record, synth, mode, "no feed available")
        skip.model_sig = orch._path_signature(record.source(synth.fmt).path)
        skip.save(rp)

        rr = orch._run_one(record, synth, mode, state, prefix="[x]", force=False)
        assert rr.status == STATUS_OK, "a cached SKIPPED result must be re-attempted, not reused"
    finally:
        monkeypatch.delenv("MODELBENCHX_SYNTH", raising=False)
        _reload(backends.base, registry, orchestrator)


def test_load_result_tolerates_shape_mismatch(tmp_path):
    """_load_result treats a JSON that parses but whose shape no longer matches
    the dataclasses (here a truncated per-output object -> TypeError on construct)
    as a cache miss, so one bad cache file cannot abort the resumable sweep."""
    from modelbenchx import orchestrator
    from modelbenchx.results import AccuracyStats, OutputAccuracy, RunResult

    p = tmp_path / "r.json"
    RunResult(
        graph_key="m__m", model="m", component="m", backend="synth", fmt=".synth",
        mode_id="cpu", mode_label="CPU", precision="fp32", status="ok", model_path="/x",
        accuracy=AccuracyStats(
            per_output=[OutputAccuracy("y", [3], "float32", 60.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
            min_psnr_db=60.0, max_abs_err=0.0, max_rel_err=0.0, mean_cosine=1.0,
            all_finite_match=True),
    ).save(p)
    d = json.loads(p.read_text())
    d["accuracy"]["per_output"][0] = {"name": "y"}  # truncated: missing required fields
    p.write_text(json.dumps(d))
    assert orchestrator.Orchestrator._load_result(p) is None

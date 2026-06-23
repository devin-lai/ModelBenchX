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

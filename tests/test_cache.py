"""Feed-fingerprint cache invalidation in the orchestrator.

The orchestrator must not silently reuse a cached feed/baseline when a
feed-affecting parameter (seed, dynamic dim size, ORT optimization policy)
changes. Otherwise a re-run reports numbers that contradict its own config.
These checks exercise that logic without spawning worker subprocesses.
"""

from pathlib import Path

from modelbenchx.config import BenchmarkConfig
from modelbenchx.orchestrator import _LEGACY_FINGERPRINT, FEED_FINGERPRINT, Orchestrator


def _orch(**kw) -> Orchestrator:
    kw.setdefault("test_model_dir", Path("/x"))
    kw.setdefault("results_dir", Path("/y"))
    return Orchestrator(BenchmarkConfig(**kw))


def test_fingerprint_tracks_feed_parameters():
    base = _orch()._feed_fingerprint()
    assert base == _LEGACY_FINGERPRINT  # defaults match the legacy reading
    assert _orch(seed=7)._feed_fingerprint() != base
    assert _orch(dynamic_dim_size=4)._feed_fingerprint() != base
    assert _orch(ort_disable_optimizations=False)._feed_fingerprint() != base
    assert _orch(input_samples=2)._feed_fingerprint() != base  # changes the shared feed


def test_missing_fingerprint_reads_as_legacy_defaults(tmp_path):
    # A cache written before fingerprinting existed: a default run is not stale,
    # so the existing cache is preserved (no surprise mass re-run on upgrade).
    assert Orchestrator._read_fingerprint(tmp_path / FEED_FINGERPRINT) == _LEGACY_FINGERPRINT
    assert _orch()._feed_fingerprint() == Orchestrator._read_fingerprint(tmp_path / FEED_FINGERPRINT)


def test_changed_seed_is_detected_as_stale(tmp_path):
    fp = tmp_path / FEED_FINGERPRINT
    orch0 = _orch(seed=0)
    fp.write_text(__import__("json").dumps(orch0._feed_fingerprint(), sort_keys=True))

    # Same config -> not stale; changed seed -> stale.
    assert Orchestrator._read_fingerprint(fp) == orch0._feed_fingerprint()
    assert Orchestrator._read_fingerprint(fp) != _orch(seed=1)._feed_fingerprint()


def test_corrupt_fingerprint_falls_back_to_legacy(tmp_path):
    fp = tmp_path / FEED_FINGERPRINT
    fp.write_text("{ not json")
    assert Orchestrator._read_fingerprint(fp) == _LEGACY_FINGERPRINT


# ---- model-file staleness (a re-exported model of the same canonical name) ----

def test_path_signature_detects_file_change(tmp_path):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"a" * 100)
    s1 = Orchestrator._path_signature(f)
    f.write_bytes(b"b" * 200)  # re-exported: different bytes/size
    assert Orchestrator._path_signature(f) != s1


def test_path_signature_detects_dir_bundle_change(tmp_path):
    # .mlpackage / .aimodel are directories: the signature must reflect their
    # contents, not just the directory's own mtime.
    d = tmp_path / "m.mlpackage"
    (d / "data").mkdir(parents=True)
    (d / "data" / "weight.bin").write_bytes(b"x" * 10)
    s1 = Orchestrator._path_signature(d)
    (d / "data" / "weight.bin").write_bytes(b"y" * 99)
    assert Orchestrator._path_signature(d) != s1


def test_path_signature_missing_is_stable(tmp_path):
    missing = tmp_path / "nope.onnx"
    assert Orchestrator._path_signature(missing) == Orchestrator._path_signature(missing)


def test_path_signature_distinct_missing_paths_differ(tmp_path):
    # Two different absent models must not be conflated into one signature.
    a = Orchestrator._path_signature(tmp_path / "a.onnx")
    b = Orchestrator._path_signature(tmp_path / "b.onnx")
    assert a != b


def test_graph_fingerprint_tracks_feed_model(tmp_path):
    f = tmp_path / "ref.onnx"
    f.write_bytes(b"a" * 50)
    orch = _orch()
    fp1 = orch._graph_fingerprint(f)
    assert fp1["seed"] == 0  # still carries the feed parameters
    f.write_bytes(b"a" * 51)  # re-exported, same name
    assert orch._graph_fingerprint(f) != fp1  # model change invalidates the cached feed


def test_fingerprint_stale_ignores_model_for_legacy_cache():
    # A cache from before model tracking (no 'feed_model') must not be forced to
    # re-run merely because the current fingerprint now tracks the model file.
    legacy = dict(_LEGACY_FINGERPRINT)
    current = {**_LEGACY_FINGERPRINT, "feed_model": "sigX"}
    assert Orchestrator._fingerprint_stale(legacy, current) is False
    # But a real feed-parameter change is still detected.
    assert Orchestrator._fingerprint_stale(legacy, {**current, "seed": 9}) is True


def test_fingerprint_stale_ignores_keys_absent_from_existing_cache():
    """A fingerprint file written by older code (lacking newer keys like
    'input_samples' or 'feed_model') must not be judged stale under default
    settings, else upgrading silently re-runs every cached graph."""
    pre_input_samples = {
        "seed": 0, "dynamic_dim_size": 1,
        "ort_disable_optimizations": True, "reference_backend": "onnxruntime",
    }  # no 'input_samples', no 'feed_model'
    current = _orch()._graph_fingerprint("/some/model")  # has both new keys
    assert Orchestrator._fingerprint_stale(pre_input_samples, current) is False
    # But a genuine change to a key the cache did record is still detected.
    changed = _orch(seed=5)._graph_fingerprint("/some/model")
    assert Orchestrator._fingerprint_stale(pre_input_samples, changed) is True


def test_fingerprint_stale_detects_model_change_for_tracked_cache():
    read = {**_LEGACY_FINGERPRINT, "feed_model": "old"}
    assert Orchestrator._fingerprint_stale(read, {**_LEGACY_FINGERPRINT, "feed_model": "new"}) is True
    assert Orchestrator._fingerprint_stale(read, {**_LEGACY_FINGERPRINT, "feed_model": "old"}) is False


def test_cached_run_is_stale_when_consumer_model_changes():
    from modelbenchx.results import STATUS_OK, RunResult
    rr = RunResult(
        graph_key="m__m", model="m", component="m", backend="coreml-mlpackage",
        fmt=".mlpackage", mode_id="all", mode_label="x", precision="fp16",
        status=STATUS_OK, model_path="/x", model_sig="sigA",
    )
    assert Orchestrator._cached_is_fresh(rr, "sigA") is True
    assert Orchestrator._cached_is_fresh(rr, "sigB") is False
    rr.model_sig = None  # legacy result: treated as fresh (no surprise re-run)
    assert Orchestrator._cached_is_fresh(rr, "sigB") is True

import json
from pathlib import Path

import numpy as np

from modelbenchx.workers import _harness
from modelbenchx.workers import _protocol as P
from tests._synth_backend import SynthWorker


def _job(tmp_path, **meta):
    P.write_meta(tmp_path, {"warmup": 1, "min_iters": 2, "max_iters": 2,
                            "time_budget_s": 0.0, "seed": 0, **meta})
    return tmp_path


def test_generate_feed_mode_writes_shared_and_outputs(tmp_path):
    model = tmp_path / "m.npz"
    np.savez(model, w=np.ones((3, 2), dtype=np.float32))
    shared = tmp_path / "inputs.npz"
    jd = _job(tmp_path, model_path=str(model), generate_feed=True,
              shared_inputs_npz=str(shared), shared_input_names_json=str(tmp_path / "names.json"))
    rc = _harness.run_worker(jd, SynthWorker())
    assert rc == 0
    res = json.loads((Path(jd) / P.RESULT).read_text())
    assert res["produced_output_names"] == ["y"] and len(res["raw_ms"]) == 2
    assert shared.exists()  # feed shared for other backends


def test_consume_shared_feed_mode(tmp_path):
    model = tmp_path / "m.npz"
    np.savez(model, w=np.ones((3, 2), dtype=np.float32))
    inputs = tmp_path / "inputs.npz"
    from modelbenchx.workers import _io as npio
    npio.save_named(inputs, ["x"], {"x": np.ones((1, 3), dtype=np.float32)})
    jd = _job(tmp_path, model_path=str(model), inputs_npz=str(inputs), input_names=["x"])
    rc = _harness.run_worker(jd, SynthWorker())
    assert rc == 0
    out = npio.load_named(Path(jd) / P.OUTPUTS, ["y"])
    assert np.allclose(out["y"], np.full((1, 2), 3.0))


def test_exception_writes_error_json(tmp_path):
    jd = _job(tmp_path, model_path=str(tmp_path / "missing.npz"), generate_feed=True,
              shared_inputs_npz=str(tmp_path / "i.npz"),
              shared_input_names_json=str(tmp_path / "n.json"))
    rc = _harness.run_worker(jd, SynthWorker())
    assert rc == 1
    assert (Path(jd) / P.ERROR).exists()


def test_input_samples_rotates_feeds_and_captures_sample0(tmp_path):
    """With input_samples>1 the timed loop rotates distinct inputs (representative
    latency), while the saved outputs come from sample 0 (deterministic accuracy)."""
    from modelbenchx.workers import _io as npio
    from modelbenchx.workers._feedgen import InputSpec

    seen: list[float] = []

    class Rec(_harness.Worker):
        def load(self, meta): ...
        def input_spec(self): return [InputSpec("x", (1,), np.dtype(np.float32))]
        def build_feed(self, shared, meta): return shared
        def infer(self, feed):
            seen.append(float(feed["x"][0]))
            return [feed["x"]]
        def output_names(self): return ["y"]

    P.write_meta(tmp_path, {"warmup": 1, "min_iters": 6, "max_iters": 6, "time_budget_s": 0.0,
                            "seed": 0, "input_samples": 3, "generate_feed": True,
                            "shared_inputs_npz": str(tmp_path / "i.npz"),
                            "shared_input_names_json": str(tmp_path / "n.json")})
    rc = _harness.run_worker(tmp_path, Rec())
    assert rc == 0
    assert len(set(seen)) >= 2  # rotated through >1 distinct input, not a single feed
    sample0 = npio.load_samples(tmp_path / "i.npz", ["x"], 3)[0]["x"]
    out = npio.load_named(tmp_path / P.OUTPUTS, ["y"])
    assert np.allclose(out["y"], sample0)  # accuracy output is sample 0's


def test_input_samples_consumer_path_loads_and_rotates(tmp_path):
    """The consumer branch (inputs_npz + input_samples>1) must load_samples and
    rotate them; accuracy output is sample 0. Complements the feed-generator test."""
    from modelbenchx.workers import _io as npio

    samples = [{"x": np.array([float(i)], dtype=np.float32)} for i in range(3)]
    npio.save_samples(tmp_path / "inputs.npz", ["x"], samples)
    seen: list[float] = []

    class Cons(_harness.Worker):
        def load(self, meta): ...
        def build_feed(self, shared, meta): return shared
        def infer(self, feed):
            seen.append(float(feed["x"][0]))
            return [feed["x"]]
        def output_names(self): return ["y"]

    P.write_meta(tmp_path, {"warmup": 1, "min_iters": 6, "max_iters": 6, "time_budget_s": 0.0,
                            "seed": 0, "input_samples": 3,
                            "inputs_npz": str(tmp_path / "inputs.npz"), "input_names": ["x"]})
    rc = _harness.run_worker(tmp_path, Cons())
    assert rc == 0
    assert set(seen) == {0.0, 1.0, 2.0}  # rotated through all loaded samples
    out = npio.load_named(tmp_path / P.OUTPUTS, ["y"])
    assert np.allclose(out["y"], samples[0]["x"])  # accuracy uses sample 0


def test_materialize_runs_inside_timed_region(tmp_path):
    """Outputs that are lazy handles (Core AI) must be host-materialized inside
    the timed region, once per inference, so latency includes the device->host
    copy (parity with Core ML's predict(), which already returns host arrays).
    If materialize ran only once after timing, its count would be 1, not == infer.
    """
    from modelbenchx.workers._feedgen import InputSpec

    class _Handle:
        def __init__(self, arr): self._arr = arr
        def numpy(self): return self._arr

    class Lazy(_harness.AsyncWorker):
        def __init__(self):
            self.infer_calls = 0
            self.materialize_calls = 0
        def load(self, meta): ...
        def input_spec(self): return [InputSpec("x", (1,), np.dtype(np.float32))]
        def build_feed(self, shared, meta): return shared
        async def infer(self, feed):
            self.infer_calls += 1
            return {"y": _Handle(np.array([1.0, 2.0], dtype=np.float32))}
        def materialize(self, out):
            self.materialize_calls += 1
            return {k: np.asarray(v.numpy()) for k, v in out.items()}
        def output_names(self): return ["y"]
        def extract_outputs(self, last): return last  # already materialized

    P.write_meta(tmp_path, {"warmup": 1, "min_iters": 3, "max_iters": 3, "time_budget_s": 0.0,
                            "seed": 0, "generate_feed": True,
                            "shared_inputs_npz": str(tmp_path / "i.npz"),
                            "shared_input_names_json": str(tmp_path / "n.json")})
    w = Lazy()
    rc = _harness.arun_worker(tmp_path, w)
    assert rc == 0
    assert w.infer_calls >= 4  # 1 warmup + 3 timed
    assert w.materialize_calls == w.infer_calls  # materialized every call, inside timing
    from modelbenchx.workers import _io as npio
    out = npio.load_named(tmp_path / P.OUTPUTS, ["y"])
    assert np.allclose(out["y"], [1.0, 2.0])  # extract_outputs received host arrays


def test_arun_worker_runs_and_calls_aclose(tmp_path):
    from modelbenchx.workers._feedgen import InputSpec

    closed = {"v": False}

    class Echo(_harness.AsyncWorker):
        def load(self, meta): ...
        def input_spec(self): return [InputSpec("x", (1,), np.dtype(np.float32))]
        def build_feed(self, shared, meta): return shared
        async def infer(self, feed): return [feed["x"]]
        def output_names(self): return ["y"]
        async def aclose(self): closed["v"] = True

    P.write_meta(tmp_path, {"warmup": 1, "min_iters": 1, "max_iters": 1, "time_budget_s": 0.0,
                            "seed": 0, "generate_feed": True,
                            "shared_inputs_npz": str(tmp_path / "i.npz"),
                            "shared_input_names_json": str(tmp_path / "n.json")})
    rc = _harness.arun_worker(tmp_path, Echo())
    assert rc == 0
    assert closed["v"] is True
    assert (tmp_path / P.RESULT).exists()


def test_aclose_exception_does_not_mask_original_error(tmp_path):
    """If infer fails and aclose ALSO fails during cleanup, the worker must report
    the original failure (the root cause), not the secondary cleanup error — the
    native-abort path depends on this for diagnosis."""
    from modelbenchx.workers._feedgen import InputSpec

    attempted = {"v": False}

    class Faulty(_harness.AsyncWorker):
        def load(self, meta): ...
        def input_spec(self): return [InputSpec("x", (1,), np.dtype(np.float32))]
        def build_feed(self, shared, meta): return shared
        async def infer(self, feed): raise ValueError("root cause")
        def output_names(self): return ["y"]
        async def aclose(self):
            attempted["v"] = True
            raise RuntimeError("secondary cleanup failure")

    P.write_meta(tmp_path, {"warmup": 1, "min_iters": 1, "max_iters": 1, "time_budget_s": 0.0,
                            "seed": 0, "generate_feed": True,
                            "shared_inputs_npz": str(tmp_path / "i.npz"),
                            "shared_input_names_json": str(tmp_path / "n.json")})
    rc = _harness.arun_worker(tmp_path, Faulty())
    assert rc == 1
    assert attempted["v"] is True  # aclose WAS attempted; only its exception was suppressed
    err = json.loads((tmp_path / P.ERROR).read_text())
    assert err["type"] == "ValueError" and "root cause" in err["message"]

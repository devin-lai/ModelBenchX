"""Tests for the TFLite backend registration and worker.

The registration test runs unconditionally (no tflite runtime needed).
The end-to-end test is gated: it builds a tiny .tflite model at runtime using
tensorflow (which is present in this environment) and exercises TFLiteWorker
through the full harness. Both model-building and inference run in subprocesses
to avoid the onnx+tensorflow protobuf crash that occurs when both are loaded
in the same pytest process.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

import modelbenchx.backends.base as base
from modelbenchx.workers import _protocol as P

# ---------------------------------------------------------------------------
# Registration test: always runs, no tflite runtime required
# ---------------------------------------------------------------------------

def test_tflite_backend_registered():
    names = {b.name for b in base.all_backends()}
    assert "tflite" in names


def test_tflite_backend_provides_feed():
    b = base.get_backend("tflite")
    assert b.provides_feed is True


def test_tflite_backend_discovery():
    b = base.get_backend("tflite")
    assert b.discovery is not None
    assert b.discovery.suffix == ".tflite"
    # key_fn strips the .tflite suffix
    assert b.discovery.key_fn("mymodel.tflite") == "mymodel"
    assert b.discovery.key_fn("mymodel") == "mymodel"


def test_tflite_backend_cross_platform():
    b = base.get_backend("tflite")
    assert b.platforms is None


# ---------------------------------------------------------------------------
# Structural import test: always runs, verifies lazy-import design
# ---------------------------------------------------------------------------

def test_tflite_worker_imports_without_runtime(monkeypatch):
    """The worker module must import even when no tflite runtime is installed."""
    # Temporarily remove all three runtimes from sys.modules and builtins
    saved = {}
    for key in list(sys.modules):
        if key.startswith(("ai_edge_litert", "tflite_runtime", "tensorflow")):
            saved[key] = sys.modules.pop(key)

    # Re-import the worker with a clean module cache and blocked imports
    import builtins
    original_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name in ("ai_edge_litert", "tflite_runtime", "tensorflow"):
            raise ImportError(f"blocked: {name}")
        if name.startswith(("ai_edge_litert.", "tflite_runtime.", "tensorflow.")):
            raise ImportError(f"blocked: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    # Remove cached worker module so it re-executes
    worker_key = "modelbenchx.workers.tflite_worker"
    saved_worker = sys.modules.pop(worker_key, None)
    try:
        import modelbenchx.workers.tflite_worker  # must not raise  # noqa: F401
    finally:
        # Restore everything
        if saved_worker is not None:
            sys.modules[worker_key] = saved_worker
        sys.modules.update(saved)


def test_tflite_input_spec_resolves_dynamic_dims():
    """Dynamic dims are reported as -1 (LiteRT) or 0 (older tf.lite); both must
    resolve to 1 so the worker can serve as a concrete feed source."""
    from modelbenchx.workers.tflite_worker import TFLiteWorker

    w = TFLiteWorker()
    w._in = [{"name": "x", "shape": np.array([0, 3, -1, 5]), "dtype": np.float32}]
    specs = w.input_spec()
    assert specs[0].shape == (1, 3, 1, 5)


def test_tflite_build_feed_rejects_underfilled_feed():
    """If the shared feed has fewer arrays than the model has inputs, build_feed
    must fail loud (recorded as a failed run) rather than silently leave inputs at
    their zero default and report meaningless numbers as success."""
    from modelbenchx.workers.tflite_worker import TFLiteWorker

    w = TFLiteWorker()
    # Model expects two inputs; the shared feed only carries one.
    w._in = [
        {"name": "a", "index": 0, "shape": np.array([1, 3]), "dtype": np.float32},
        {"name": "b", "index": 1, "shape": np.array([1, 3]), "dtype": np.float32},
    ]
    with pytest.raises(ValueError, match="needs 2 input"):
        w.build_feed({"a": np.zeros((1, 3), np.float32)}, {})


# ---------------------------------------------------------------------------
# End-to-end test: gated on a tflite runtime being available
# ---------------------------------------------------------------------------

_HAS_RUNTIME = (
    importlib.util.find_spec("ai_edge_litert") is not None
    or importlib.util.find_spec("tflite_runtime") is not None
    or importlib.util.find_spec("tensorflow") is not None
)


_BUILDER_SCRIPT = """
import sys, tempfile, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf

# Build a trivial tf.Module: y = x @ ones(3,2)
class LinearModule(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 3], tf.float32)])
    def __call__(self, x):
        w = tf.constant(np.ones((3, 2), dtype=np.float32))
        return tf.matmul(x, w)

module = LinearModule()
concrete = module.__call__.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
out = pathlib.Path(sys.argv[1])
out.write_bytes(converter.convert())
"""


@pytest.mark.skipif(not _HAS_RUNTIME, reason="no tflite runtime installed")
def test_tflite_e2e_generate_feed(tmp_path):
    """Build a tiny .tflite in a subprocess, then run TFLiteWorker through the harness.

    Both the model-building step and the inference step run in separate subprocesses
    to avoid the onnx/tensorflow protobuf conflict that arises when both are loaded in
    the same pytest process (test_inputs.py imports onnx; loading tensorflow in the
    same process causes a segfault on some platforms/versions). Inference runs via
    P.execute, which is the production subprocess path.
    """
    model_path = tmp_path / "m.tflite"
    builder = tmp_path / "_build.py"
    builder.write_text(_BUILDER_SCRIPT)

    result = subprocess.run(
        [sys.executable, str(builder), str(model_path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"TFLite model builder failed (converter API issue): {result.stderr[-300:]}")

    shared_npz = tmp_path / "inputs.npz"
    names_json = tmp_path / "names.json"

    P.write_meta(tmp_path, {
        "model_path": str(model_path),
        "mode": "cpu",
        "generate_feed": True,
        "shared_inputs_npz": str(shared_npz),
        "shared_input_names_json": str(names_json),
        "warmup": 1,
        "min_iters": 1,
        "max_iters": 1,
        "time_budget_s": 0.0,
        "seed": 0,
    })

    # Run inference via subprocess to avoid onnx+tensorflow protobuf conflict.
    # (test_inputs.py imports onnx into the pytest process; loading the tflite
    # runtime in the same process then segfaults on macOS with TF 2.21.)
    outcome = P.execute(sys.executable, "modelbenchx.workers.tflite_worker", tmp_path, timeout_s=60)
    assert outcome.ok, outcome.error or "unknown failure"

    result_path = tmp_path / P.RESULT
    assert result_path.exists()
    res = json.loads(result_path.read_text())
    assert "raw_ms" in res
    assert len(res["raw_ms"]) >= 1

    outputs_path = tmp_path / P.OUTPUTS
    assert outputs_path.exists()

    # Verify correctness: load shared feed and compute expected y = x @ ones(3,2)
    from modelbenchx.workers import _io as npio

    output_names = res["produced_output_names"]
    outputs = npio.load_named(outputs_path, output_names)
    assert len(outputs) == 1
    arr = next(iter(outputs.values()))

    shared_names = json.loads(names_json.read_text())
    shared = npio.load_named(shared_npz, shared_names)
    x = next(iter(shared.values())).astype(np.float32)
    expected = x @ np.ones((3, 2), dtype=np.float32)
    assert np.allclose(arr, expected, atol=1e-5), f"output {arr!r} != expected {expected!r}"

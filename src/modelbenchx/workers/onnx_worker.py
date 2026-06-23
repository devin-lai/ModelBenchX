"""ONNX Runtime worker (CPU FP32). The accuracy baseline + the shared feed.

Runs first per graph. It extracts the requested ``.onnx`` from its zip, generates
the seeded input feed, runs ONNX Runtime on CPU, and writes the shared
``inputs.npz`` + baseline ``outputs`` into the per-graph cache for the other
backends to consume. Imports onnx + onnxruntime + numpy only (never coremltools).

Run as: ``python -m modelbenchx.workers.onnx_worker <jobdir>``
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from . import _inputs
from . import _io as npio
from ._harness import Worker, run_worker


def _extract(zip_path: str, member: str, dest: Path) -> Path:
    """Extract the whole archive once (so external .data weights sit beside the
    .onnx) and return the path to the requested member."""
    onnx_path = dest / member
    if not onnx_path.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    if not onnx_path.exists():
        raise FileNotFoundError(f"member {member!r} not found in {zip_path}")
    return onnx_path


def _build_session(onnx_path: Path, disable_opt: bool):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 4  # FATAL only; kernel failures still raise
    if disable_opt:
        # Spec-faithful baseline: ORT's graph rewrites and KleidiAI SGEMM have
        # produced wrong outputs on macOS arm64 for several of these models.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("mlas.disable_kleidiai", "1")
    return ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


class OnnxWorker(Worker):
    def load(self, meta):
        import onnx

        cache = Path(meta["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        extract_dir = Path(meta.get("onnx_extract_dir") or (cache / "onnx_src"))
        self._onnx_path = _extract(meta["model_path"], meta["onnx_member"], extract_dir)
        model = onnx.load(str(self._onnx_path))
        input_meta = _inputs.load_input_metadata(extract_dir, meta["onnx_member"])
        self._specs = _inputs.spec_from_onnx(model, input_meta, dynamic_dim_size=meta.get("dynamic_dim_size", 1))
        self._sess = _build_session(self._onnx_path, meta.get("ort_disable_optimizations", True))
        self._out_names = [o.name for o in self._sess.get_outputs()]
        self._cache = cache

    def input_spec(self):
        return self._specs  # onnx is the feed source

    def build_feed(self, shared, meta):
        if shared is None:
            raise RuntimeError("OnnxWorker requires generate_feed=True (shared feed not provided)")
        self._feed = shared  # generate_feed mode -> shared is our generated feed
        return shared

    def infer(self, feed):
        return self._sess.run(self._out_names, feed)

    def output_names(self):
        return self._out_names

    def realized_device(self):
        return "CPU"

    def extract_outputs(self, last):
        outputs = {n: np.asarray(a) for n, a in zip(self._out_names, last, strict=True)}
        # Reference extras for accuracy + downstream backends:
        npio.save_named(self._cache / "baseline_outputs.npz", self._out_names, outputs)
        (self._cache / "baseline_meta.json").write_text(
            json.dumps({"input_names": list(self._feed), "output_names": self._out_names})
        )
        return outputs


if __name__ == "__main__":
    sys.exit(run_worker(sys.argv[1], OnnxWorker()))

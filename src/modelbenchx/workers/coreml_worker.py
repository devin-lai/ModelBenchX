"""Core ML worker. Runs a ``.mlpackage`` or ``.mlmodel`` via coremltools.

One compute unit per invocation (``cpu_only`` / ``cpu_and_gpu`` / ``all``).
Imports coremltools + numpy only (never onnx, because that pair aborts the process).
Consumes the shared ``inputs.npz`` produced by the onnx worker.

Run as: ``python -m modelbenchx.workers.coreml_worker <jobdir>``
"""

from __future__ import annotations

import sys

import numpy as np

from .. import naming
from ._harness import Worker, run_worker

_MODE_TO_UNIT = {
    "cpu_only": "CPU_ONLY",
    "cpu_and_gpu": "CPU_AND_GPU",
    "all": "ALL",
}

# Core ML multiarray dtype name -> numpy dtype for the feed.
_CT_DTYPE = {
    "FLOAT32": np.float32,
    "FLOAT16": np.float16,
    "DOUBLE": np.float64,
    "INT32": np.int32,
}


def _spec_input_dtype(ml_input) -> np.dtype:
    mat = ml_input.type.multiArrayType
    enum_desc = mat.DESCRIPTOR.fields_by_name["dataType"].enum_type
    name = enum_desc.values_by_number[mat.dataType].name
    return np.dtype(_CT_DTYPE.get(name, np.float32))


def _build_feed(spec, feed_by_onnx_name: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    inputs = list(spec.description.input)
    for inp in inputs:
        if inp.type.HasField("imageType"):
            raise NotImplementedError(
                f"Core ML image input '{inp.name}' is not supported by the harness"
            )
    ml_names = [i.name for i in inputs]
    mapping = naming.match_output_names(list(feed_by_onnx_name), ml_names)
    out: dict[str, np.ndarray] = {}
    for inp in inputs:
        onnx_name = next((o for o, m in mapping.items() if m == inp.name), None)
        if onnx_name is None:
            raise KeyError(f"no feed array maps to Core ML input '{inp.name}'")
        out[inp.name] = np.ascontiguousarray(
            feed_by_onnx_name[onnx_name].astype(_spec_input_dtype(inp))
        )
    return out


class CoreMLWorker(Worker):
    def load(self, meta):
        import coremltools as ct

        unit = getattr(ct.ComputeUnit, _MODE_TO_UNIT[meta["mode"]])
        self._model = ct.models.MLModel(meta["model_path"], compute_units=unit)
        self._spec = self._model.get_spec()
        self._out_names = [o.name for o in self._spec.description.output]

    def build_feed(self, shared, meta):
        return _build_feed(self._spec, shared)

    def infer(self, feed):
        return self._model.predict(feed)

    def output_names(self):
        return self._out_names

    def extract_outputs(self, last):
        return {n: np.asarray(last[n]) for n in self._out_names}


if __name__ == "__main__":
    sys.exit(run_worker(sys.argv[1], CoreMLWorker()))

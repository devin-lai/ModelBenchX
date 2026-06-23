"""Deterministic, seeded input generation for an ONNX graph (worker-side).

Imports ``onnx`` + ``numpy`` only, never ``coremltools``, so it is safe inside
the onnx worker. The same generated feed is saved and reused by every backend so
accuracy is compared on identical inputs.

Generation honors, in order of authority: the ONNX graph dtype (so ONNX Runtime
accepts the feed), the ``metadata.json`` concrete shape (to resolve dynamic
dims) and its ``value_range``/``io_type`` (so image-like inputs land in a valid
range instead of arbitrary normal noise).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from onnx import TensorProto

from . import _feedgen
from ._feedgen import InputSpec

_ONNX_TO_NUMPY: dict[int, np.dtype] = {
    TensorProto.FLOAT: np.dtype(np.float32),
    TensorProto.FLOAT16: np.dtype(np.float16),
    TensorProto.DOUBLE: np.dtype(np.float64),
    TensorProto.INT8: np.dtype(np.int8),
    TensorProto.INT16: np.dtype(np.int16),
    TensorProto.INT32: np.dtype(np.int32),
    TensorProto.INT64: np.dtype(np.int64),
    TensorProto.UINT8: np.dtype(np.uint8),
    TensorProto.UINT16: np.dtype(np.uint16),
    TensorProto.UINT32: np.dtype(np.uint32),
    TensorProto.UINT64: np.dtype(np.uint64),
    TensorProto.BOOL: np.dtype(np.bool_),
}


def load_input_metadata(extract_dir: Path, member: str) -> dict:
    """Return ``{input_name: spec}`` from the zip's metadata.json, if present.

    ``member`` is the onnx arcname (e.g. ``sam2-onnx-float/encoder.onnx``); the
    metadata keys models by basename (``encoder.onnx``).
    """
    meta_path = Path(extract_dir) / Path(member).parent / "metadata.json"
    if not meta_path.exists():
        # single-folder archives keep metadata.json beside the onnx
        alt = next(Path(extract_dir).rglob("metadata.json"), None)
        if alt is None:
            return {}
        meta_path = alt
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    basename = Path(member).name
    files = meta.get("model_files", {})
    entry = files.get(basename) or (next(iter(files.values())) if len(files) == 1 else {})
    return entry.get("inputs", {}) if isinstance(entry, dict) else {}


def _resolve_shape(vi_shape, meta_shape, dynamic_dim_size: int) -> tuple[int, ...]:
    dims: list[int] = []
    for i, d in enumerate(vi_shape.dim):
        if d.HasField("dim_value") and d.dim_value > 0:
            dims.append(int(d.dim_value))
        elif meta_shape is not None and i < len(meta_shape) and int(meta_shape[i]) > 0:
            dims.append(int(meta_shape[i]))
        else:
            dims.append(int(dynamic_dim_size))
    return tuple(dims)


def spec_from_onnx(model, input_meta: dict, *, dynamic_dim_size: int = 1) -> list[InputSpec]:
    """Build InputSpecs for every graph input not shadowed by an initializer."""
    initializer_names = {init.name for init in model.graph.initializer}
    specs: list[InputSpec] = []
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue
        tt = vi.type.tensor_type
        dtype = _ONNX_TO_NUMPY.get(tt.elem_type)
        if dtype is None:
            name = TensorProto.DataType.Name(tt.elem_type)
            raise ValueError(f"input '{vi.name}': unsupported ONNX dtype {name}")
        spec = input_meta.get(vi.name, {})
        shape = _resolve_shape(tt.shape, spec.get("shape"), dynamic_dim_size)
        vr = spec.get("value_range")
        value_range = (float(vr[0]), float(vr[1])) if vr and len(vr) == 2 else None
        specs.append(InputSpec(vi.name, shape, dtype, value_range, spec.get("io_type")))
    return specs


def generate_inputs(model, input_meta, *, seed: int = 0, dynamic_dim_size: int = 1):
    """Back-compat wrapper: ONNX spec -> seeded feed."""
    specs = spec_from_onnx(model, input_meta, dynamic_dim_size=dynamic_dim_size)
    return _feedgen.generate_from_spec(specs, seed=seed)

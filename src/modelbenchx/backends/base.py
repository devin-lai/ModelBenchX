"""Backend plugins: declarative descriptions of each runtime and its modes.

A backend is intentionally thin: it declares its source format, the worker
module that runs it, and its modes. All execution lives in the worker; all
scheduling/accuracy/reporting lives in the orchestrator. Adding a framework is
therefore: add one ``Backend`` here + one worker module, then register it.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Callable
from dataclasses import dataclass

from .. import naming
from ..config import AUTO, FP16, FP32, Mode


@dataclass(frozen=True)
class FormatSpec:
    suffix: str                               # ".tflite"; ".zip" for onnx
    key_fn: Callable[[str], str]              # filename -> canonical key
    archive_member_suffix: str | None = None  # ".onnx" -> scan zip members; None = file-per-graph


@dataclass(frozen=True)
class Backend:
    name: str  # unique backend id, e.g. "coreml-mlpackage"
    fmt: str  # registry source key: onnx | mlpackage | mlmodel | aimodel
    kind: str  # worker kind: onnx | coreml | coreai
    worker_module: str
    modes: tuple[Mode, ...]
    label: str  # human framework label for reports
    is_baseline: bool = False
    entrypoint: str = "main"
    discovery: FormatSpec | None = None  # set per backend below
    platforms: tuple[str, ...] | None = None
    provides_feed: bool = False  # worker implements input_spec() -> can generate the shared feed

    def mode(self, mode_id: str) -> Mode:
        for m in self.modes:
            if m.id == mode_id:
                return m
        raise KeyError(f"{self.name} has no mode {mode_id!r}")


# Core ML's three compute-unit choices, shared by mlpackage and mlmodel
# (both are fp16 as exported by onnx2coreml).
_COREML_MODES = (
    Mode("cpu_only", "CPU only", FP16),
    Mode("cpu_and_gpu", "CPU + GPU", FP16),
    Mode("all", "ANE + GPU + CPU", FP16),
)

BACKENDS: tuple[Backend, ...] = (
    Backend(
        name="onnxruntime",
        fmt="onnx",
        kind="onnx",
        worker_module="modelbenchx.workers.onnx_worker",
        modes=(Mode("cpu", "CPU", FP32),),
        label="ONNX Runtime",
        is_baseline=True,
        discovery=FormatSpec(".zip", naming.model_from_onnx_zip, archive_member_suffix=".onnx"),
        platforms=None,
        provides_feed=True,
    ),
    Backend(
        name="coreml-mlpackage",
        fmt="mlpackage",
        kind="coreml",
        worker_module="modelbenchx.workers.coreml_worker",
        modes=_COREML_MODES,
        label="Core ML (ML Program / .mlpackage)",
        discovery=FormatSpec(".mlpackage", naming.key_from_coreml_filename),
        platforms=("Darwin",),
    ),
    Backend(
        name="coreml-mlmodel",
        fmt="mlmodel",
        kind="coreml",
        worker_module="modelbenchx.workers.coreml_worker",
        modes=_COREML_MODES,
        label="Core ML (Neural Network / .mlmodel)",
        discovery=FormatSpec(".mlmodel", naming.key_from_coreml_filename),
        platforms=("Darwin",),
    ),
    Backend(
        name="coreai",
        fmt="aimodel",
        kind="coreai",
        worker_module="modelbenchx.workers.coreai_worker",
        modes=(
            Mode("cpu_only", "CPU only", FP32),
            Mode("gpu", "GPU", FP16),
            Mode("ane", "ANE", FP16),
            Mode("all", "Auto (all units)", AUTO),
        ),
        label="Core AI (.aimodel)",
        discovery=FormatSpec(".aimodel", naming.key_from_aimodel_dirname),
        platforms=("Darwin",),
    ),
    Backend(
        name="tflite",
        fmt="tflite",
        kind="tflite",
        worker_module="modelbenchx.workers.tflite_worker",
        modes=(Mode("cpu", "CPU", FP32), Mode("xnnpack", "XNNPACK", FP32)),
        label="TFLite",
        discovery=FormatSpec(".tflite", lambda n: n[:-7] if n.endswith(".tflite") else n),
        platforms=None,          # cross-platform
        provides_feed=True,      # enables latency-only benchmark of a standalone .tflite
    ),
)

_BY_NAME = {b.name: b for b in BACKENDS}

if os.environ.get("MODELBENCHX_SYNTH"):
    BACKENDS = BACKENDS + (
        Backend(
            name="synth", fmt="synth", kind="synth",
            worker_module="modelbenchx.workers.synth_worker",
            modes=(Mode("cpu", "CPU", FP32),), label="Synthetic (numpy)",
            discovery=FormatSpec(".npmodel",
                                 lambda n: n[: -len(".npmodel")] if n.endswith(".npmodel") else n),
            platforms=None,
            provides_feed=True,
        ),
    )
    _BY_NAME["synth"] = BACKENDS[-1]


def all_backends() -> tuple[Backend, ...]:
    return BACKENDS


def get_backend(name: str) -> Backend:
    return _BY_NAME[name]


def format_specs() -> dict[str, FormatSpec]:
    return {b.fmt: b.discovery for b in BACKENDS if b.discovery is not None}


def select_backends(names: tuple[str, ...] | None, *, system: str | None = None) -> list[Backend]:
    """Chosen backends, baseline first, dropping those unsupported on this OS."""
    system = system or platform.system()
    if names is not None:
        unknown = [n for n in names if n not in _BY_NAME]
        if unknown:
            raise ValueError(
                f"unknown backend(s): {', '.join(unknown)}; "
                f"choices: {', '.join(sorted(_BY_NAME))}"
            )
    chosen = list(BACKENDS) if names is None else [get_backend(n) for n in names]
    chosen = [b for b in chosen if b.platforms is None or system in b.platforms]
    chosen.sort(key=lambda b: (not b.is_baseline, b.name))
    return chosen

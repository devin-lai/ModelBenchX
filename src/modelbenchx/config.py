"""Configuration objects: compute modes and the top-level benchmark config."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Precision tags used in labels/reports.
FP32 = "fp32"
FP16 = "fp16"
AUTO = "auto"  # runtime chooses the unit (and thus the precision)


@dataclass(frozen=True)
class Mode:
    """A single execution mode of a backend (a compute-unit choice).

    ``id`` is the stable token used in result filenames and the worker meta;
    ``label`` is what the report shows; ``precision`` documents the numeric
    format the framework uses in this mode.
    """

    id: str
    label: str
    precision: str


@dataclass
class BenchmarkConfig:
    """Everything the orchestrator needs for a run.

    Filters (``backends``/``modes``/``models``) are ``None`` to mean "all".
    ``models`` entries match either an exact graph key or a model prefix
    (``"resnet50"`` selects ``resnet50__resnet50``).
    """

    test_model_dir: Path
    results_dir: Path

    # Sampling.
    seed: int = 0
    warmup: int = 5
    max_iters: int = 50
    min_iters: int = 10
    time_budget_s: float = 3.0
    dynamic_dim_size: int = 1
    # Number of distinct seeded inputs to rotate through the timed loop. 1 (the
    # default) reproduces single-feed behavior exactly; >1 spreads latency across
    # representative inputs for data-dependent graphs. Accuracy always uses
    # sample 0, so the reference comparison is unchanged.
    input_samples: int = 1

    # Reference backend: when present in a graph's sources, it generates the
    # shared feed + reference outputs and accuracy is computed against it. When
    # absent for a graph, that graph's runs are latency-only.
    reference_backend: str = "onnxruntime"

    # Selection filters.
    backends: tuple[str, ...] | None = None
    modes: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    smoke: int = 0  # >0 limits to that many (small) graphs

    # Behaviour.
    force: bool = False
    keep_raw_timings: bool = True
    python_executable: str = field(default_factory=lambda: sys.executable)
    worker_timeout_s: float = 1800.0

    # Thermal gating (Darwin). When enabled, the orchestrator pauses before each
    # run until the SoC's CPU speed limit recovers, so a long serial sweep does
    # not record throttled latencies. Off by default (no behaviour change).
    thermal_gate: bool = False
    thermal_min_speed: int = 100      # require full CPU speed (pmset -g therm %)
    thermal_max_wait_s: float = 120.0  # bounded per-run cooldown
    thermal_poll_s: float = 5.0

    # Best-effort scheduling QoS for worker subprocesses (Darwin). When set, the
    # worker command is wrapped with ``taskpolicy -c <class>`` (e.g. "utility").
    # None = inherit the orchestrator's QoS (current behaviour). NB: macOS QoS,
    # not CPU affinity, governs P/E-core placement; this is a hint, not a pin.
    worker_qos: str | None = None

    # ONNX Runtime baseline fidelity (see docs/design.md §5).
    ort_disable_optimizations: bool = True

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "runs"

    @property
    def cache_dir(self) -> Path:
        return self.results_dir / "cache"

    @property
    def reports_dir(self) -> Path:
        return self.results_dir / "reports"

    def selects_model(self, graph_key: str) -> bool:
        if self.models is None:
            return True
        model = graph_key.split("__", 1)[0]
        return any(graph_key == m or model == m or graph_key.startswith(m) for m in self.models)

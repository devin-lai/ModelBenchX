"""Load cached run results from disk and index them for reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..backends.base import all_backends
from ..results import RunResult


def collect_results(runs_dir: str | Path) -> list[RunResult]:
    runs_dir = Path(runs_dir)
    out: list[RunResult] = []
    if not runs_dir.exists():
        return out
    for path in sorted(runs_dir.glob("*/*.json")):
        try:
            out.append(RunResult.load(path))
        except (OSError, ValueError, KeyError, TypeError):
            # A corrupt, truncated, or shape-mismatched result file is skipped, not
            # fatal: one bad cache file must never abort report generation.
            continue
    return out


def index(results: list[RunResult]) -> dict[tuple[str, str, str], RunResult]:
    """Index by ``(graph_key, backend, mode_id)``."""
    return {(r.graph_key, r.backend, r.mode_id): r for r in results}


@dataclass(frozen=True)
class Column:
    backend: str
    mode_id: str
    short: str
    full: str
    precision: str
    is_baseline: bool
    framework: str = ""   # framework label, e.g. "Core ML (ML Program / .mlpackage)"
    mode_label: str = ""  # human mode label, e.g. "ANE + GPU + CPU"


_BACKEND_ABBR = {
    "onnxruntime": "ORT",
    "coreml-mlpackage": "MLPkg",
    "coreml-mlmodel": "MLMdl",
    "coreai": "AI",
}
_MODE_ABBR = {
    "cpu": "cpu",
    "cpu_only": "cpu",
    "cpu_and_gpu": "cpu+gpu",
    "all": "all",
    "gpu": "gpu",
    "ane": "ane",
}


def column_spec(results: list[RunResult]) -> list[Column]:
    """Stable, canonical column order for the matrices (only columns present)."""
    present = {(r.backend, r.mode_id) for r in results}
    cols: list[Column] = []
    for b in all_backends():
        for m in b.modes:
            if (b.name, m.id) in present:
                cols.append(
                    Column(
                        backend=b.name,
                        mode_id=m.id,
                        short=f"{_BACKEND_ABBR.get(b.name, b.name)}/{_MODE_ABBR.get(m.id, m.id)}",
                        full=f"{b.label} — {m.label} ({m.precision})",
                        precision=m.precision,
                        is_baseline=b.is_baseline,
                        framework=b.label,
                        mode_label=m.label,
                    )
                )
    return cols

"""Discover models across registered format folders and build the benchmarkable union.

Pure stdlib (``zipfile`` only). This module never imports ``onnx`` or
``coremltools``; it only inspects filenames and zip listings, so it is safe to
call from the orchestrator process.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import naming
from .backends.base import FormatSpec, format_specs


@dataclass(frozen=True)
class GraphSource:
    """Where one format stores a given graph."""

    fmt: str
    path: str  # zip path (onnx) or bundle path (everything else)
    member: str | None = None  # arcname of the .onnx inside the zip (onnx only)


@dataclass
class GraphRecord:
    """A graph available in at least one registered format."""

    key: str
    model: str
    component: str
    sources: dict[str, GraphSource] = field(default_factory=dict)

    def source(self, fmt: str) -> GraphSource:
        return self.sources[fmt]


@dataclass
class SkippedGraph:
    key: str
    present_in: list[str]
    missing_from: list[str]


@dataclass
class Registry:
    benchmarkable: list[GraphRecord]
    skipped: list[SkippedGraph]
    per_format_counts: dict[str, int]

    def get(self, key: str) -> GraphRecord:
        for r in self.benchmarkable:
            if r.key == key:
                return r
        raise KeyError(key)


def _discover_format(folder: Path, fmt: str, spec: FormatSpec) -> dict[str, GraphSource]:
    out: dict[str, GraphSource] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob(f"*{spec.suffix}")):
        if spec.archive_member_suffix is not None:        # archive: scan members
            model = spec.key_fn(path.name)
            try:
                with zipfile.ZipFile(path) as zf:
                    members = [n for n in zf.namelist() if n.endswith(spec.archive_member_suffix)]
            except (zipfile.BadZipFile, OSError):
                continue
            for member in members:
                key = naming.onnx_member_key(model, member)
                out[key] = GraphSource(fmt=fmt, path=str(path), member=member)
        else:                                             # file/bundle per graph
            out[spec.key_fn(path.name)] = GraphSource(fmt=fmt, path=str(path))
    return out


def discover(test_model_dir: str | Path) -> Registry:
    """Union of every graph present in any registered format's folder.

    A graph is benchmarkable if present in >=1 format; ``sources`` records which.
    """
    root = Path(test_model_dir)
    specs = format_specs()
    per_format = {fmt: _discover_format(root / fmt, fmt, spec) for fmt, spec in specs.items()}

    all_keys: set[str] = set()
    for d in per_format.values():
        all_keys |= set(d)

    benchmarkable: list[GraphRecord] = []
    for key in sorted(all_keys):
        present = {fmt: per_format[fmt][key] for fmt in specs if key in per_format[fmt]}
        # A standalone file whose name carries no "<model>__<component>" split
        # (e.g. a lone foo.tflite benchmarked latency-only) has no distinct
        # component; treat the whole key as both model and component rather than
        # rejecting it, which would abort discovery of every format folder.
        model, component = naming.split_key(key) if naming.SEP in key else (key, key)
        benchmarkable.append(GraphRecord(key=key, model=model, component=component, sources=present))

    return Registry(
        benchmarkable=benchmarkable,
        skipped=[],  # union model: nothing is excluded; coverage is per-graph via sources
        per_format_counts={fmt: len(d) for fmt, d in per_format.items()},
    )

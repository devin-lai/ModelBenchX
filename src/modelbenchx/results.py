"""Result schema: plain dataclasses that round-trip to/from JSON.

Numpy-free so the orchestrator and reporters never need a heavy import. Workers
emit the timing/accuracy numbers as plain floats.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = 2

# Non-finite floats (±inf = hard-failure / bit-exact accuracy sentinels, NaN)
# are not representable in strict JSON. We serialize them as string tokens so
# the output parses everywhere (JS/Go/Rust), preserving the +inf vs -inf
# distinction (null would lose it), and revive them (per numeric field) on load.
_STR_TO_NONFINITE = {"Infinity": float("inf"), "-Infinity": float("-inf"), "NaN": float("nan")}


def finitize(obj):
    """Recursively replace non-finite reals with JSON-safe string tokens.

    Accepts any ``numbers.Real`` (so ``numpy`` scalars are handled too; they are
    not Python ``float`` subclasses) and normalizes them to ``float`` so the
    strict ``allow_nan=False`` dump never trips over a stray non-finite. Bools and
    ints pass through unchanged.
    """
    if isinstance(obj, numbers.Real) and not isinstance(obj, (bool, int)):
        f = float(obj)
        if math.isnan(f):
            return "NaN"
        if f == float("inf"):
            return "Infinity"
        if f == float("-inf"):
            return "-Infinity"
        return f
    if isinstance(obj, dict):
        return {k: finitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finitize(v) for v in obj]
    return obj


def _revive(v):
    """Revive a single non-finite string token back to a float; pass anything
    else through. Applied only to numeric fields, so a string field that happens
    to equal a token (e.g. an ONNX tensor literally named ``"NaN"``) is safe."""
    return _STR_TO_NONFINITE.get(v, v) if isinstance(v, str) else v


def _revive_numeric(d: dict, float_keys: set[str]) -> dict:
    return {k: (_revive(v) if k in float_keys else v) for k, v in d.items()}


def dumps(obj, **kw) -> str:
    """``json.dumps`` that emits strict-valid JSON for non-finite floats."""
    return json.dumps(finitize(obj), allow_nan=False, **kw)

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass
class TimingStats:
    """Steady-state latency statistics, all in milliseconds unless noted."""

    iters: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ips: float  # inferences per second (1000 / mean_ms)
    load_ms: float | None = None  # load + compile/specialize (cold)
    first_call_ms: float | None = None  # first warmup call (lazy compile cost)
    raw_ms: list[float] | None = None  # per-iteration timings (audit trail)


@dataclass
class OutputAccuracy:
    """Accuracy of one output tensor versus the baseline."""

    name: str
    shape: list[int]
    dtype: str
    psnr_db: float
    max_abs_err: float
    max_rel_err: float
    rmse: float
    mae: float
    cosine: float
    expected_nonfinite: int = 0
    nonfinite_match: bool = True


@dataclass
class AccuracyStats:
    """Per-output accuracy plus run-level aggregates (worst case)."""

    per_output: list[OutputAccuracy]
    min_psnr_db: float
    max_abs_err: float
    max_rel_err: float
    mean_cosine: float
    all_finite_match: bool
    note: str = ""


@dataclass
class RunResult:
    """One ``(graph, backend, mode)`` measurement."""

    graph_key: str
    model: str
    component: str
    backend: str
    fmt: str
    mode_id: str
    mode_label: str
    precision: str
    status: str
    model_path: str
    is_baseline: bool = False
    note: str = ""
    iters_requested: int = 0
    warmup_requested: int = 0
    timing: TimingStats | None = None
    accuracy: AccuracyStats | None = None
    realized_device: str | None = None
    timestamp: float | None = None
    duration_s: float | None = None
    worker_returncode: int | None = None
    # Content-identity of the model file at measurement time. Lets a resumed run
    # detect a re-exported model of the same canonical name and re-measure it
    # instead of silently reusing a stale cached result. ``None`` = legacy result
    # (predates tracking) -> treated as fresh.
    model_sig: str | None = None
    schema_version: int = SCHEMA_VERSION

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunResult:
        d = dict(d)
        timing = d.pop("timing", None)
        accuracy = d.pop("accuracy", None)
        obj = cls(**{k: v for k, v in d.items() if k in _RUN_FIELDS})
        if timing is not None:
            fields = _revive_numeric(
                {k: v for k, v in timing.items() if k in _TIMING_FIELDS}, _TIMING_FLOATS)
            # raw_ms is a list of floats: revive each element from its non-finite
            # token, mirroring how scalar float fields are revived above.
            if isinstance(fields.get("raw_ms"), list):
                fields["raw_ms"] = [_revive(v) for v in fields["raw_ms"]]
            obj.timing = TimingStats(**fields)
        if accuracy is not None:
            per = [OutputAccuracy(**_revive_numeric(o, _OUTPUT_FLOATS))
                   for o in accuracy.get("per_output", [])]
            obj.accuracy = AccuracyStats(
                per_output=per,
                min_psnr_db=_revive(accuracy["min_psnr_db"]),
                max_abs_err=_revive(accuracy["max_abs_err"]),
                max_rel_err=_revive(accuracy["max_rel_err"]),
                mean_cosine=_revive(accuracy["mean_cosine"]),
                all_finite_match=accuracy["all_finite_match"],
                note=accuracy.get("note", ""),
            )
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(dumps(self.to_dict(), indent=2))
        tmp.replace(path)  # atomic: a crash mid-write never leaves a partial json

    @classmethod
    def load(cls, path: Path) -> RunResult:
        return cls.from_dict(json.loads(Path(path).read_text()))


_RUN_FIELDS = set(RunResult.__dataclass_fields__)  # type: ignore[attr-defined]
_TIMING_FIELDS = set(TimingStats.__dataclass_fields__)  # type: ignore[attr-defined]
# Numeric fields that may hold a non-finite token on load (revived to float).
# Excludes string/int/bool/list fields, so a tensor named "NaN" is never coerced.
_TIMING_FLOATS = {
    "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms",
    "p90_ms", "p95_ms", "p99_ms", "throughput_ips", "load_ms", "first_call_ms",
}
_OUTPUT_FLOATS = {"psnr_db", "max_abs_err", "max_rel_err", "rmse", "mae", "cosine"}

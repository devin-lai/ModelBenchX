"""Flat one-row-per-run CSV for spreadsheets / further analysis."""

from __future__ import annotations

import csv
from pathlib import Path

from ..metrics import timing as tmetrics
from ..results import RunResult

_FIELDS = [
    "graph_key", "model", "component", "backend", "fmt", "mode_id", "mode_label",
    "precision", "is_baseline", "status",
    "mean_ms", "median_ms", "std_ms", "cv_pct", "min_ms", "max_ms", "p90_ms", "p95_ms",
    "p99_ms", "throughput_ips", "load_ms", "first_call_ms", "cold_start_ms", "iters",
    "min_psnr_db", "max_abs_err", "max_rel_err", "mean_cosine", "all_finite_match",
    "realized_device", "duration_s", "note",
]


def _row(r: RunResult) -> dict:
    d = {
        "graph_key": r.graph_key, "model": r.model, "component": r.component,
        "backend": r.backend, "fmt": r.fmt, "mode_id": r.mode_id,
        "mode_label": r.mode_label, "precision": r.precision,
        "is_baseline": r.is_baseline, "status": r.status,
        "realized_device": r.realized_device or "", "duration_s": r.duration_s,
        "note": (r.note or "").replace("\n", " "),
    }
    if r.timing is not None:
        t = r.timing
        d.update(
            mean_ms=t.mean_ms, median_ms=t.median_ms, std_ms=t.std_ms,
            cv_pct=tmetrics.cv_pct(t), min_ms=t.min_ms,
            max_ms=t.max_ms, p90_ms=t.p90_ms, p95_ms=t.p95_ms, p99_ms=t.p99_ms,
            throughput_ips=t.throughput_ips, load_ms=t.load_ms,
            first_call_ms=t.first_call_ms, cold_start_ms=tmetrics.cold_start_ms(t),
            iters=t.iters,
        )
    if r.accuracy is not None:
        a = r.accuracy
        d.update(
            min_psnr_db=a.min_psnr_db, max_abs_err=a.max_abs_err,
            max_rel_err=a.max_rel_err, mean_cosine=a.mean_cosine,
            all_finite_match=a.all_finite_match,
        )
    return d


def write(path: str | Path, results: list[RunResult]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(_row(r))

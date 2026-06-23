"""Full machine-readable JSON dump of a benchmark run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..results import RunResult, dumps


def build(results: list[RunResult], registry=None, env=None, config=None) -> dict:
    doc: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": (env.to_dict() if hasattr(env, "to_dict") else env),
        "results": [r.to_dict() for r in results],
    }
    if config is not None:
        doc["config"] = {
            "seed": config.seed,
            "warmup": config.warmup,
            "min_iters": config.min_iters,
            "max_iters": config.max_iters,
            "time_budget_s": config.time_budget_s,
            "input_samples": config.input_samples,
            "ort_disable_optimizations": config.ort_disable_optimizations,
        }
    if registry is not None:
        doc["per_format_counts"] = registry.per_format_counts
        doc["skipped"] = [
            {"key": s.key, "present_in": s.present_in, "missing_from": s.missing_from}
            for s in registry.skipped
        ]
    return doc


def write(path: str | Path, results, registry=None, env=None, config=None) -> None:
    Path(path).write_text(dumps(build(results, registry, env, config), indent=2))

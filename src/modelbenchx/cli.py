"""Command-line interface: ``modelbenchx discover | run | report``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import environment
from .config import BenchmarkConfig
from .orchestrator import Orchestrator
from .registry import discover
from .report import _collect, csv_report, json_report, markdown


def _csv_tuple(s: str) -> tuple[str, ...] | None:
    # "" → None ("all"); an empty tuple would override the default and select nothing.
    items = tuple(x.strip() for x in s.split(",") if x.strip())
    return items or None


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--test-model", default="test_model", type=Path,
                   help="directory holding onnx/ mlmodel/ mlpackage/ aimodel/ (default: ./test_model)")
    p.add_argument("--results", default="results", type=Path,
                   help="output directory for run cache + reports (default: ./results)")


def _build_config(args) -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        test_model_dir=args.test_model.resolve(),
        results_dir=args.results.resolve(),
    )
    for attr in ("seed", "warmup", "max_iters", "min_iters", "time_budget_s",
                 "input_samples", "smoke", "force", "thermal_gate", "worker_qos"):
        if getattr(args, attr, None) is not None:
            setattr(cfg, attr, getattr(args, attr))
    cfg.backends = getattr(args, "backends", None)
    cfg.modes = getattr(args, "modes", None)
    cfg.models = getattr(args, "models", None)
    if getattr(args, "reference_backend", None):
        cfg.reference_backend = args.reference_backend
    return cfg


def cmd_discover(args) -> int:
    root = args.test_model.resolve()
    if not root.exists():
        print(f"error: test-model dir not found: {root}", file=sys.stderr)
        return 2
    reg = discover(root)
    print("Per-format graph counts:")
    for fmt, n in reg.per_format_counts.items():
        print(f"  {fmt:10s} {n}")
    print(f"\nBenchmarkable (present in >=1 format): {len(reg.benchmarkable)}")
    for r in reg.benchmarkable[:10]:
        print(f"  - {r.key}")
    if len(reg.benchmarkable) > 10:
        print(f"  ... (+{len(reg.benchmarkable) - 10} more)")
    return 0


def _generate_reports(cfg: BenchmarkConfig) -> Path:
    reg = discover(cfg.test_model_dir)
    env = environment.capture()
    results = _collect.collect_results(cfg.runs_dir)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = cfg.reports_dir / "report.md"
    md_path.write_text(markdown.render(results, registry=reg, env=env, config=cfg))
    json_report.write(cfg.reports_dir / "report.json", results, registry=reg, env=env, config=cfg)
    csv_report.write(cfg.reports_dir / "report.csv", results)
    return md_path


def cmd_run(args) -> int:
    cfg = _build_config(args)
    if not cfg.test_model_dir.exists():
        print(f"error: test-model dir not found: {cfg.test_model_dir}", file=sys.stderr)
        return 2
    orch = Orchestrator(cfg, progress=lambda m: print(m, flush=True))
    orch.run()
    md_path = _generate_reports(cfg)
    print(f"\nReports written to {cfg.reports_dir}/ (report.md, report.json, report.csv)")
    print(f"  -> {md_path}")
    return 0


def cmd_report(args) -> int:
    cfg = _build_config(args)
    md_path = _generate_reports(cfg)
    print(f"Reports regenerated in {cfg.reports_dir}/")
    print(f"  -> {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="modelbenchx", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("discover", help="list benchmarkable + skipped graphs")
    _add_common(pd)
    pd.set_defaults(func=cmd_discover)

    pr = sub.add_parser("run", help="run the benchmark matrix (resumable)")
    _add_common(pr)
    pr.add_argument("--smoke", type=int, default=0, help="limit to N smallest graphs (pipeline check)")
    pr.add_argument("--backends", type=_csv_tuple, default=None, help="comma-separated backend names")
    pr.add_argument("--modes", type=_csv_tuple, default=None, help="comma-separated mode ids to include")
    pr.add_argument("--models", type=_csv_tuple, default=None, help="comma-separated model/graph filters")
    pr.add_argument("--reference-backend", dest="reference_backend", default=None,
                    help="backend whose outputs accuracy is measured against (default: onnxruntime)")
    pr.add_argument("--warmup", type=int, default=None,
                    help="warmup calls to discard before timing (a floor; 0 times the cold first call)")
    pr.add_argument("--min-iters", dest="min_iters", type=int, default=None)
    pr.add_argument("--max-iters", dest="max_iters", type=int, default=None)
    pr.add_argument("--time-budget", dest="time_budget_s", type=float, default=None)
    pr.add_argument("--input-samples", dest="input_samples", type=int, default=None,
                    help="distinct seeded inputs to rotate through timing (default 1; "
                         "accuracy always uses sample 0)")
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--force", action="store_true", default=None, help="ignore cached results and re-run")
    pr.add_argument("--thermal-gate", dest="thermal_gate", action="store_true", default=None,
                    help="pause before each run until the SoC's CPU speed limit recovers (Darwin)")
    pr.add_argument("--worker-qos", dest="worker_qos", default=None,
                    help="taskpolicy QoS class for worker subprocesses, e.g. 'utility' (Darwin; best-effort)")
    pr.set_defaults(func=cmd_run)

    prep = sub.add_parser("report", help="regenerate reports from cached results")
    _add_common(prep)
    prep.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        # Config-level errors (e.g. an unknown --backends name) are reported as a
        # clean message + exit 2, consistent with the missing-dir handling above,
        # rather than a bare traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""Render the formal Markdown performance + accuracy report."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime

from ..metrics import timing as tmetrics
from ..metrics.accuracy import PSNR_EXACT, PSNR_FAIL, PSNR_OK, classify_psnr
from ..results import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, RunResult
from ._collect import Column, column_spec, index


def _med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _esc(s: str) -> str:
    """Escape Markdown table metacharacters in free-form / filename-derived cell
    text (graph keys, failure notes) so a literal ``|`` cannot inject columns and
    corrupt the rendered table."""
    return str(s).replace("\\", "\\\\").replace("|", "\\|")


def _is_ref(col: Column, ref_name: str) -> bool:
    """The accuracy reference is the configured ``reference_backend``, not the
    static ``is_baseline`` flag — they differ whenever ``--reference-backend`` is
    set to anything other than the default, and the labels must follow the
    backend the PSNR was actually computed against."""
    return col.backend == ref_name


def _ref_label(cols: list[Column], ref_name: str) -> str:
    col = next((c for c in cols if c.backend == ref_name), None)
    return f"{col.framework} ({col.mode_label}, {col.precision.upper()})" if col else ref_name


def _ref_short(cols: list[Column], ref_name: str) -> str:
    col = next((c for c in cols if c.backend == ref_name), None)
    return f"{col.framework} {col.precision.upper()}" if col else ref_name


def _fmt_psnr(v: float) -> str:
    cls = classify_psnr(v)
    if cls == PSNR_EXACT:
        return "∞"
    if cls == PSNR_FAIL:
        return "−∞"
    return f"{v:.1f}"


def _cell_latency(r: RunResult | None) -> str:
    if r is None:
        return "—"
    if r.status == STATUS_OK and r.timing is not None:
        return f"{r.timing.mean_ms:.2f}"
    return "fail" if r.status == STATUS_FAILED else "skip"


def _cell_accuracy(r: RunResult | None, col: Column, ref_name: str) -> str:
    if _is_ref(col, ref_name):
        return "ref"
    if r is None:
        return "—"
    if r.status == STATUS_OK:
        return _fmt_psnr(r.accuracy.min_psnr_db) if r.accuracy is not None else "n/a"
    return "fail" if r.status == STATUS_FAILED else "skip"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def _summary(results: list[RunResult]) -> dict:
    graphs = {r.graph_key for r in results}
    return {
        "graphs": len(graphs),
        "runs": len(results),
        "ok": sum(r.status == STATUS_OK for r in results),
        "failed": sum(r.status == STATUS_FAILED for r in results),
        "skipped": sum(r.status == STATUS_SKIPPED for r in results),
    }


def _aggregate(results: list[RunResult], cols: list[Column], ref_name: str, min_iters: int = 1) -> list[dict]:
    """Per backend×mode aggregates (median over a backend's ok runs)."""
    by_col: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        by_col.setdefault((r.backend, r.mode_id), []).append(r)
    agg: list[dict] = []
    for c in cols:
        rs = by_col.get((c.backend, c.mode_id), [])
        ok = [r for r in rs if r.status == STATUS_OK and r.timing is not None]
        # Tail percentiles (p90/p99) are only meaningful with enough samples; the
        # slow-model wall can stop a run at ~3 iterations, where np.percentile is
        # pure interpolation toward the max and carries no real tail. Aggregate
        # them over runs that met the iteration floor so the breakdown does not
        # present an authoritative-looking tail built from 3 points.
        reliable = [r for r in ok if r.timing.iters >= min_iters]
        means = [r.timing.mean_ms for r in ok]
        mins = [r.timing.min_ms for r in ok]
        thru = [r.timing.throughput_ips for r in ok]
        # ±inf are kept out of the median so they can't skew it (a +inf/-inf pair
        # would also yield NaN). When every PSNR is infinite: all +inf is
        # bit-exact (∞); any -inf is a real failure (−∞), never bit-exact.
        psnrs = [r.accuracy.min_psnr_db for r in ok if r.accuracy is not None]
        finite = [p for p in psnrs if classify_psnr(p) == PSNR_OK]
        # Hard accuracy failures (−∞) among STATUS_OK runs: the inference ran but
        # the output was incomparable (shape mismatch / missing / NaN divergence).
        # These are dropped from the central-tendency median, so they must be
        # counted and surfaced, otherwise a clean finite median hides them.
        n_acc_fail = sum(1 for p in psnrs if classify_psnr(p) == PSNR_FAIL)
        if _is_ref(c, ref_name):
            median_psnr = None
        elif finite:
            median_psnr = statistics.median(finite)
        elif not psnrs:
            median_psnr = None  # latency-only
        else:
            all_exact = all(classify_psnr(p) == PSNR_EXACT for p in psnrs)
            median_psnr = float("inf") if all_exact else float("-inf")
        agg.append({
            "n_acc_fail": n_acc_fail,
            "col": c,
            "n_ok": sum(r.status == STATUS_OK for r in rs),
            "n_fail": sum(r.status == STATUS_FAILED for r in rs),
            "n_skip": sum(r.status == STATUS_SKIPPED for r in rs),
            "median_latency": _med(means),
            "median_min_latency": _med(mins),
            # Best-case observed single-inference latency across all graphs. For the
            # GPU/ANE/auto modes this approximates each binding's per-call dispatch
            # floor (cross-checked against a controlled micro-benchmark).
            "floor_latency": min(mins) if mins else None,
            "median_throughput": _med(thru),
            "median_psnr": median_psnr,
            # Cold-start & distribution (professional latency profile).
            "median_load": _med([r.timing.load_ms for r in ok if r.timing.load_ms is not None]),
            "median_first": _med([r.timing.first_call_ms for r in ok if r.timing.first_call_ms is not None]),
            "median_cold_start": _med([v for r in ok if (v := tmetrics.cold_start_ms(r.timing)) is not None]),
            "median_p50": _med([r.timing.median_ms for r in ok]),
            "median_p90": _med([r.timing.p90_ms for r in reliable]),
            "median_p99": _med([r.timing.p99_ms for r in reliable]),
            "median_cv": _med([v for r in ok if (v := tmetrics.cv_pct(r.timing)) is not None]),
        })
    return agg


def _aggregate_rows(agg: list[dict], ref_name: str) -> list[list[str]]:
    rows = []
    for a in agg:
        c = a["col"]
        if _is_ref(c, ref_name):
            acc = "ref"
        elif a["median_psnr"] is None:
            acc = "—"
        else:
            acc = _fmt_psnr(a["median_psnr"])
            # Surface dropped hard failures next to the median they would skew
            # (only when the median itself is finite; an all-failed column is
            # already shown as −∞).
            if a["n_acc_fail"] and classify_psnr(a["median_psnr"]) == PSNR_OK:
                acc += f" ({a['n_acc_fail']}✗)"
        rows.append([
            c.full, str(a["n_ok"]), str(a["n_fail"]), str(a["n_skip"]),
            f"{a['median_latency']:.2f}" if a["median_latency"] is not None else "—",
            f"{a['median_min_latency']:.2f}" if a["median_min_latency"] is not None else "—",
            f"{a['floor_latency']:.2f}" if a["floor_latency"] is not None else "—",
            f"{a['median_throughput']:.0f}" if a["median_throughput"] is not None else "—",
            acc,
        ])
    return rows


def _num(v: float | None, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if v is not None else "—"


def _latency_breakdown_rows(agg: list[dict]) -> list[list[str]]:
    rows = []
    for a in agg:
        rows.append([
            a["col"].full,
            _num(a["median_load"], 1),
            _num(a["median_first"]),
            _num(a["median_cold_start"], 1),
            _num(a["median_p50"]),
            _num(a["median_p90"]),
            _num(a["median_p99"]),
            _num(a["median_cv"], 1),
        ])
    return rows


def _key_findings(agg: list[dict], ref_name: str, ref_label: str) -> str:
    base = next((a for a in agg if _is_ref(a["col"], ref_name)), None)
    base_lat = base["median_latency"] if base else None
    bullets: list[str] = []
    if base_lat:
        bullets.append(
            f"- **Baseline** — {ref_label}: median **{base_lat:.1f} ms** "
            f"({base['median_throughput']:.0f} inf/s) per inference."
        )
    ranked = sorted(
        (a for a in agg if not _is_ref(a["col"], ref_name) and a["median_latency"] is not None),
        key=lambda a: a["median_latency"],
    )
    if ranked:
        f = ranked[0]
        sp = f" — **{base_lat / f['median_latency']:.1f}× faster** than the baseline" if base_lat else ""
        bullets.append(
            f"- **Fastest configuration** (median latency): {f['col'].full} at "
            f"**{f['median_latency']:.2f} ms**{sp}."
        )
    # Only columns with a genuine accuracy number and no hard failures are
    # eligible: a column that hard-fails on some graphs must not be crowned
    # "most accurate" just because those −∞ graphs were dropped from its median.
    fp16 = [
        a for a in agg
        if a["col"].precision == "fp16"
        and a["median_psnr"] is not None
        and classify_psnr(a["median_psnr"]) != PSNR_FAIL
        and a["n_acc_fail"] == 0
    ]
    if fp16:
        best = max(fp16, key=lambda a: a["median_psnr"])
        bullets.append(
            f"- **Most accurate FP16 mode** (median worst-case PSNR): {best['col'].full} at "
            f"**{_fmt_psnr(best['median_psnr'])} dB**."
        )
    total_fail = sum(a["n_fail"] for a in agg)
    if total_fail:
        worst = max(agg, key=lambda a: a["n_fail"])
        bullets.append(
            f"- **{total_fail} run(s) failed** and are recorded with notes (models left unmodified); "
            f"most affected: {worst['col'].full} ({worst['n_fail']})."
        )
    bullets.append(
        "- **Read latency with the per-call overhead caveat** (see Methodology → *Interpreting "
        "latency*): each framework's Python entry point carries a different fixed dispatch floor "
        "(Core ML `predict` ~0.1 ms vs Core AI `await fn()` ~1 ms), so latency ratios on "
        "sub-millisecond models reflect binding overhead more than engine speed."
    )
    return "\n".join(bullets) + "\n"


def render(results, registry=None, env=None, config=None) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    cols = column_spec(results)
    idx = index(results)
    graph_keys = sorted({r.graph_key for r in results})
    s = _summary(results)
    ref_name = getattr(config, "reference_backend", "onnxruntime") if config is not None else "onnxruntime"
    ref_label = _ref_label(cols, ref_name)
    ref_short = _ref_short(cols, ref_name)
    min_iters = getattr(config, "min_iters", 1) if config is not None else 1
    agg = _aggregate(results, cols, ref_name, min_iters)

    parts: list[str] = []
    parts.append("# ModelBenchX — Inference Framework Benchmark Report\n")
    parts.append(f"_Generated {now}_\n")

    # Summary.
    parts.append("## Summary\n")
    parts.append(
        f"- **{s['graphs']}** graph(s) benchmarked across **{len(cols)}** backend×mode "
        f"configurations — **{s['runs']}** runs total.\n"
        f"- Outcomes: **{s['ok']} ok**, **{s['failed']} failed**, **{s['skipped']} skipped**.\n"
        f"- Accuracy baseline: **{ref_label}** (configurable via `--reference-backend`). Accuracy is "
        f"the worst-case (minimum) PSNR across a graph's outputs, in dB; higher is better, ∞ = bit-exact. "
        f"Graphs benchmarked without a reference show `n/a` in accuracy columns (latency-only).\n"
    )
    if config is not None:
        low = sum(
            1 for r in results
            if r.status == STATUS_OK and r.timing is not None and r.timing.iters < config.min_iters
        )
        if low:
            parts.append(
                f"- **{low} run(s)** were timed below the {config.min_iters}-iteration floor "
                f"(the slow-model wall stops long runs early); their latency stats rest on few "
                f"samples — see the per-run `iters` column in `report.csv`.\n"
            )

    # Key findings.
    parts.append("\n## Key findings\n")
    parts.append(_key_findings(agg, ref_name, ref_label))

    # Environment.
    if env is not None:
        e = env.to_dict() if hasattr(env, "to_dict") else env
        rv = e.get("runtime_versions", {})
        cores = str(e.get("cpu_cores"))
        p, ecore = e.get("performance_cores"), e.get("efficiency_cores")
        if p is not None and ecore is not None:
            cores += f" ({p}P + {ecore}E)"
        env_rows = [
            ["Chip", str(e.get("chip"))],
            ["CPU cores", cores],
            ["Memory", f"{e.get('memory_gb')} GB"],
            ["OS", f"{e.get('os')} {e.get('os_version')} ({e.get('machine')})"],
            ["Python", str(e.get("python_version"))],
            ["onnxruntime", str(rv.get("onnxruntime"))],
            ["coremltools", str(rv.get("coremltools"))],
            ["coreai", str(rv.get("coreai"))],
            ["numpy", str(rv.get("numpy"))],
        ]
        if e.get("tool_version") is not None:
            env_rows.append(["ModelBenchX", str(e.get("tool_version"))])
        # Power/thermal state (Darwin), only shown when captured, so off-Darwin
        # reports stay clean.
        speed = e.get("cpu_speed_limit")
        if e.get("power_source") is not None:
            env_rows.append(["Power source", str(e.get("power_source"))])
        if e.get("low_power_mode") is not None:
            env_rows.append(["Low Power Mode", "on" if e.get("low_power_mode") else "off"])
        if speed is not None:
            env_rows.append(["CPU speed limit", f"{speed}%" + ("" if speed >= 100 else "  ⚠ throttled")])
        parts.append("## Environment\n")
        parts.append(_table(["property", "value"], env_rows) + "\n")
        throttled = (speed is not None and speed < 100)
        on_battery = (e.get("power_source") == "Battery Power")
        if throttled or on_battery or e.get("low_power_mode"):
            parts.append(
                "> **Latency caveat:** the host was "
                + ", ".join(
                    filter(None, [
                        "thermally/power throttled (CPU speed < 100%)" if throttled else "",
                        "on battery power" if on_battery else "",
                        "in Low Power Mode" if e.get("low_power_mode") else "",
                    ])
                )
                + " during capture — measured latencies may understate the SoC's "
                "sustained capability. Re-run on AC power, Low Power Mode off, and a cool SoC.\n"
            )

    # Methodology.
    parts.append("## Methodology\n")
    if config is not None:
        n_samples = getattr(config, "input_samples", 1)
        if n_samples > 1:
            input_bullet = (
                f"- **{n_samples} distinct seeded inputs** (seeds {config.seed}…"
                f"{config.seed + n_samples - 1}) are generated per graph and rotated through the "
                f"timed loop, so latency reflects more than one input; **accuracy uses sample 0** "
                f"(seed {config.seed}), compared on an identical feed across backends.\n"
            )
        else:
            input_bullet = (
                f"- A single **seeded, deterministic input** (seed {config.seed}) is generated per "
                f"graph and reused by every backend, so latency and accuracy are measured on an "
                f"identical feed.\n"
            )
        parts.append(
            f"- Each run executes in an **isolated subprocess** (one runtime per process; the "
            f"native runtimes cannot share an interpreter). Runs are **serial** so they never "
            f"contend for the CPU/GPU/ANE.\n"
            f"- Per run: model load/compile is timed separately; **{config.warmup} warmup** "
            f"call(s) are discarded (the first carries lazy compile cost); then up to "
            f"**{config.max_iters} iterations** (min {config.min_iters}, ~{config.time_budget_s:.0f}s "
            f"budget, with a slow-model wall) are timed with a nanosecond clock.\n"
            + input_bullet
        )
    parts.append(
        "- **Mode labels** report the *requested* compute unit. Requesting ANE/auto selects a "
        "*preferred* placement; the OS may still place some ops elsewhere.\n"
        "- **Precision:** ONNX Runtime CPU and Core AI `cpu_only` run **FP32**; Core ML "
        "(`.mlpackage`/`.mlmodel`) and Core AI GPU/ANE run **FP16** — larger numerical error "
        "in those modes is expected, not a defect.\n"
        "- **Load times** include a one-time on-disk device specialization (Core AI / Core ML "
        "compile-and-cache); steady-state latency excludes it via warmup.\n"
        "- The ONNX Runtime baseline runs with graph optimizations and KleidiAI **disabled** for "
        "a spec-faithful reference (these have produced incorrect outputs on macOS arm64 for "
        "some of these models).\n"
        "- **Reference backend** defaults to `onnxruntime` and is configurable via `--reference-backend`. "
        "When no reference is present for a graph, backends are benchmarked **latency-only** and "
        "accuracy columns show `n/a`.\n"
    )

    # Per-call overhead caveat. Needed to interpret fast-model latency fairly.
    parts.append(
        "\n### Interpreting latency — per-call dispatch overhead\n"
        "Each latency is the **end-to-end time of one inference as invoked through that "
        "framework's standard Python API** — `session.run` (ONNX Runtime), `MLModel.predict` "
        "(Core ML), `await fn(feed)` (Core AI). These Python entry points carry *different fixed "
        "per-call dispatch overheads*, which is visible in the **best-case floor** column above "
        "and matters when comparing fast models:\n"
        "- Core ML `predict` floors around **~0.04–0.2 ms**; Core AI's async `await fn(feed)` "
        "floors around **~0.5–1 ms** (a fixed cost of the Python binding's command submit/"
        "synchronize round-trip — measured to be independent of model size, and *not* asyncio "
        "scheduling or input boxing). Output host-materialization is timed for **every** backend "
        "(Core ML/ORT already return host arrays; Core AI's `.numpy()` copy runs inside the timed "
        "window too), so the latencies are compared like-for-like.\n"
        "- For sub-millisecond models this fixed floor dominates, so a 3–5× latency ratio there "
        "reflects **binding dispatch overhead, not engine speed**. For compute-heavy models the "
        "floor is negligible and a residual ~2–2.5× gap remains (a Core AI compiler/runtime "
        "maturity difference vs Apple's production Core ML engine; Core AI is faster on some "
        "graphs, e.g. `yolox`, `cvt`).\n"
        "- `all`/auto modes let the system place ops across **ANE+GPU+CPU**; single-unit modes "
        "(`ane`, `gpu`, `cpu_only`) force one preferred unit — compare like-for-like.\n"
        "- The latencies reflect each framework's **Python** path. Core AI's Swift `axon` runtime "
        "offers a zero-allocation, pre-bound-I/O path that is not exposed to Python and would "
        "remove its per-call floor; these numbers do not capture that.\n"
    )

    # Mode legend.
    parts.append("\n## Backends & modes\n")
    parts.append(
        _table(
            ["column", "framework", "format", "mode", "precision"],
            [
                [c.short, c.framework, _fmt_for(c.backend), c.mode_label, c.precision]
                for c in cols
            ],
        )
        + "\n"
    )

    # Aggregate.
    parts.append("\n## Aggregate performance (median across graphs)\n")
    parts.append(
        _table(
            ["backend × mode", "ok", "fail", "skip", "median mean-lat (ms)",
             "median min-lat (ms)", "best-case floor (ms)", "median throughput (inf/s)", "median PSNR (dB)"],
            _aggregate_rows(agg, ref_name),
        )
        + "\n"
    )

    # Professional latency breakdown: cold start -> steady state -> tail.
    parts.append("\n## Latency breakdown — cold start → steady state (median across graphs)\n")
    parts.append(
        "- **load** = model load + one-time device specialization/compile (cold). "
        "**1st inference** = first (cold) inference call, separate from load. "
        "**cold start** = load + 1st inference (time-to-first-result).\n"
        "- **p50/p90/p99** = steady-state percentiles (warmup excluded). "
        "**CV%** = coefficient of variation (std/mean) — measurement stability; lower is steadier.\n"
        "- *Cache caveat:* Core ML and Core AI keep a **persistent on-disk specialization cache** "
        "keyed by model+device, so `load` for a mode that runs after another mode of the same model "
        "may be a warm-cache load (e.g. Core AI `auto`, run last, loads far faster than its first-run "
        "`ane`/`gpu`). The first mode evaluated per model pays the full one-time compilation.\n"
    )
    parts.append(
        _table(
            ["backend × mode", "load (ms)", "1st inference (ms)", "cold start (ms)",
             "steady p50 (ms)", "p90 (ms)", "p99 (ms)", "CV %"],
            _latency_breakdown_rows(agg),
        )
        + "\n"
    )

    # Latency matrix.
    parts.append("\n## Latency matrix — mean latency per inference (ms)\n")
    parts.append(_matrix(graph_keys, cols, idx, _cell_latency) + "\n")

    # Accuracy matrix.
    parts.append(f"\n## Accuracy matrix — worst-case PSNR vs {ref_short} (dB)\n")
    parts.append(
        _matrix(graph_keys, cols, idx, lambda r, c=None: _cell_accuracy(r, c, ref_name), pass_col=True) + "\n"
    )

    # Failures & notes.
    fails = [r for r in results if r.status in (STATUS_FAILED, STATUS_SKIPPED) and not r.is_baseline]
    fails += [r for r in results if r.status == STATUS_FAILED and r.is_baseline]
    if fails:
        parts.append("\n## Failures & notes\n")
        parts.append(
            _table(
                ["graph", "backend", "mode", "status", "note"],
                [[_esc(r.graph_key), r.backend, r.mode_id, r.status,
                  _esc((r.note or "").replace("\n", " ")[:140])] for r in fails],
            )
            + "\n"
        )

    return "\n".join(parts)


_FMT_BY_BACKEND = {
    "onnxruntime": ".onnx",
    "coreml-mlpackage": ".mlpackage",
    "coreml-mlmodel": ".mlmodel",
    "coreai": ".aimodel",
}


def _fmt_for(backend: str) -> str:
    return _FMT_BY_BACKEND.get(backend, "")


def _matrix(graph_keys, cols, idx, cell_fn, pass_col: bool = False) -> str:
    headers = ["graph"] + [c.short for c in cols]
    rows = []
    for key in graph_keys:
        row = [_esc(key)]
        for c in cols:
            r = idx.get((key, c.backend, c.mode_id))
            row.append(cell_fn(r, c) if pass_col else cell_fn(r))
        rows.append(row)
    return _table(headers, rows)

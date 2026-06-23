from modelbenchx.report import _collect, csv_report, json_report, markdown
from modelbenchx.results import (
    STATUS_FAILED,
    STATUS_OK,
    AccuracyStats,
    OutputAccuracy,
    RunResult,
    TimingStats,
)


def _timing(mean):
    return TimingStats(
        iters=10, mean_ms=mean, median_ms=mean, std_ms=0.1, min_ms=mean, max_ms=mean,
        p90_ms=mean, p95_ms=mean, p99_ms=mean, throughput_ips=1000.0 / mean, load_ms=5.0,
    )


def _acc(psnr):
    return AccuracyStats(
        per_output=[OutputAccuracy("y", [1, 2], "float32", psnr, 0.01, 0.02, 0.001, 0.0005, 0.999)],
        min_psnr_db=psnr, max_abs_err=0.01, max_rel_err=0.02, mean_cosine=0.999, all_finite_match=True,
    )


def _results():
    base = RunResult(
        graph_key="m__m", model="m", component="m", backend="onnxruntime", fmt=".onnx",
        mode_id="cpu", mode_label="CPU", precision="fp32", status=STATUS_OK,
        model_path="/x.onnx", is_baseline=True, timing=_timing(50.0),
    )
    ok = RunResult(
        graph_key="m__m", model="m", component="m", backend="coreml-mlpackage", fmt=".mlpackage",
        mode_id="all", mode_label="ANE + GPU + CPU", precision="fp16", status=STATUS_OK,
        model_path="/x.mlpackage", timing=_timing(2.5), accuracy=_acc(60.0),
    )
    fail = RunResult(
        graph_key="m__m", model="m", component="m", backend="coreai", fmt=".aimodel",
        mode_id="all", mode_label="Auto (all units)", precision="auto", status=STATUS_FAILED,
        model_path="/x.aimodel", note="RuntimeError: MPSGraph boom",
    )
    return [base, ok, fail]


def test_markdown_renders_all_sections():
    md = markdown.render(_results())
    for heading in ["## Summary", "## Key findings", "## Aggregate performance",
                    "## Latency breakdown", "## Latency matrix", "## Accuracy matrix",
                    "## Failures & notes"]:
        assert heading in md
    # professional latency columns present
    for col in ["load (ms)", "1st inference (ms)", "cold start (ms)", "CV %"]:
        assert col in md
    assert "m__m" in md
    assert "20.0× faster" not in md.split("## Key findings")[0]  # speedup ratio only in findings
    # baseline accuracy cell is "ref"; failed run shows "fail"; fastest finding present
    assert "20.0× faster" in md  # 50ms / 2.5ms
    assert "MPSGraph boom" in md


def test_collect_and_roundtrip(tmp_path):
    runs = tmp_path / "runs" / "m__m"
    for r in _results():
        r.save(runs / f"{r.backend}__{r.mode_id}.json")
    loaded = _collect.collect_results(tmp_path / "runs")
    assert len(loaded) == 3
    cols = _collect.column_spec(loaded)
    # canonical order: baseline (onnxruntime) first
    assert cols[0].is_baseline
    assert {c.backend for c in cols} == {"onnxruntime", "coreml-mlpackage", "coreai"}


def test_latency_only_accuracy_renders_na():
    """A run with status=ok but accuracy=None (latency-only) must show 'n/a', not 'skip'."""
    base = RunResult(
        graph_key="m__m", model="m", component="m", backend="onnxruntime", fmt=".onnx",
        mode_id="cpu", mode_label="CPU", precision="fp32", status=STATUS_OK,
        model_path="/x.onnx", is_baseline=True, timing=_timing(50.0),
    )
    # coreml-mlpackage/all is a known backend×mode so column_spec will include it
    lat_only = RunResult(
        graph_key="m__m", model="m", component="m", backend="coreml-mlpackage", fmt=".mlpackage",
        mode_id="all", mode_label="ANE + GPU + CPU", precision="fp16", status=STATUS_OK,
        model_path="/x.mlpackage", timing=_timing(10.0),
        # accuracy intentionally None: latency-only run
    )
    out = markdown.render([base, lat_only])
    # The accuracy matrix cell for the latency-only run must be "n/a"
    assert "n/a" in out


def test_latency_only_aggregate_no_fake_inf():
    """Aggregate PSNR for a backend with ok runs but no accuracy must render '—', not '∞'."""
    base = RunResult(
        graph_key="m__m", model="m", component="m", backend="onnxruntime", fmt=".onnx",
        mode_id="cpu", mode_label="CPU", precision="fp32", status=STATUS_OK,
        model_path="/x.onnx", is_baseline=True, timing=_timing(50.0),
    )
    lat_only = RunResult(
        graph_key="m__m", model="m", component="m", backend="coreml-mlpackage", fmt=".mlpackage",
        mode_id="all", mode_label="ANE + GPU + CPU", precision="fp16", status=STATUS_OK,
        model_path="/x.mlpackage", timing=_timing(10.0),
    )
    out = markdown.render([base, lat_only])
    agg_section = out.split("## Aggregate performance")[1].split("##")[0]
    # latency-only backend: aggregate PSNR must be "—", not "∞"
    assert "∞" not in agg_section


def test_tool_version_in_env_table():
    """If env contains tool_version, it must appear in the Environment table."""
    env = {
        "chip": "Apple M1", "cpu_cores": 8, "performance_cores": 4, "efficiency_cores": 4,
        "memory_gb": 16, "os": "macOS", "os_version": "14.0", "machine": "arm64",
        "python_version": "3.11.0", "tool_version": "0.5.0-test",
        "runtime_versions": {"onnxruntime": "1.18", "coremltools": "7.0", "coreai": None, "numpy": "1.26"},
    }
    out = markdown.render(_results(), env=env)
    assert "0.5.0-test" in out


def test_pe_cores_graceful_when_none():
    """When performance_cores/efficiency_cores are None, the CPU row must not show 'NoneP'."""
    env = {
        "chip": "x86", "cpu_cores": 4, "performance_cores": None, "efficiency_cores": None,
        "memory_gb": 8, "os": "Linux", "os_version": "6.0", "machine": "x86_64",
        "python_version": "3.11.0",
        "runtime_versions": {"onnxruntime": "1.18", "coremltools": None, "coreai": None, "numpy": "1.26"},
    }
    out = markdown.render(_results(), env=env)
    assert "NoneP" not in out
    assert "NoneE" not in out


def _cfg(**kw):
    from pathlib import Path

    from modelbenchx.config import BenchmarkConfig
    return BenchmarkConfig(test_model_dir=Path("/x"), results_dir=Path("/y"), **kw)


def test_methodology_reflects_input_samples():
    out = markdown.render(_results(), config=_cfg(seed=2, input_samples=3))
    assert "3 distinct seeded inputs" in out
    assert "seeds 2…4" in out
    assert "accuracy uses sample 0" in out
    assert "A single **seeded" not in out


def test_methodology_single_input_by_default():
    out = markdown.render(_results(), config=_cfg(seed=0))
    assert "A single **seeded, deterministic input**" in out
    assert "distinct seeded inputs" not in out


def test_env_throttle_caveat_rendered_when_throttled():
    env = {
        "chip": "Apple M1", "cpu_cores": 8, "performance_cores": 4, "efficiency_cores": 4,
        "memory_gb": 16, "os": "macOS", "os_version": "14.0", "machine": "arm64",
        "python_version": "3.11.0", "tool_version": "0.5.0",
        "runtime_versions": {"onnxruntime": "1.18", "coremltools": "7.0", "coreai": None, "numpy": "1.26"},
        "power_source": "Battery Power", "low_power_mode": True, "cpu_speed_limit": 70,
    }
    out = markdown.render(_results(), env=env)
    assert "CPU speed limit" in out and "70%" in out and "throttled" in out
    assert "Latency caveat" in out and "Battery Power" in out


def test_env_no_power_rows_when_absent():
    """An env dict without power keys (off-Darwin) must add no power rows/caveat."""
    env = {
        "chip": "x86", "cpu_cores": 4, "performance_cores": None, "efficiency_cores": None,
        "memory_gb": 8, "os": "Linux", "os_version": "6.0", "machine": "x86_64",
        "python_version": "3.11.0",
        "runtime_versions": {"onnxruntime": "1.18", "coremltools": None, "coreai": None, "numpy": "1.26"},
    }
    out = markdown.render(_results(), env=env)
    assert "CPU speed limit" not in out and "Latency caveat" not in out


def test_csv_and_json_write(tmp_path):
    res = _results()
    csv_report.write(tmp_path / "r.csv", res)
    json_report.write(tmp_path / "r.json", res)
    csv_text = (tmp_path / "r.csv").read_text()
    assert "graph_key" in csv_text and "coreml-mlpackage" in csv_text
    import json
    doc = json.loads((tmp_path / "r.json").read_text())
    assert len(doc["results"]) == 3


def test_aggregate_hard_failure_not_rendered_as_bit_exact():
    """A STATUS_OK run whose accuracy is a hard failure (min_psnr_db = -inf:
    incomparable / missing output) must not be aggregated as bit-exact (∞). It
    must render as −∞ in the aggregate, never as ∞."""
    base = RunResult(
        graph_key="m__m", model="m", component="m", backend="onnxruntime", fmt=".onnx",
        mode_id="cpu", mode_label="CPU", precision="fp32", status=STATUS_OK,
        model_path="/x.onnx", is_baseline=True, timing=_timing(50.0),
    )
    hard_fail = RunResult(
        graph_key="m__m", model="m", component="m", backend="coreml-mlpackage", fmt=".mlpackage",
        mode_id="all", mode_label="ANE + GPU + CPU", precision="fp16", status=STATUS_OK,
        model_path="/x.mlpackage", timing=_timing(10.0), accuracy=_acc(float("-inf")),
    )
    out = markdown.render([base, hard_fail])
    agg_section = out.split("## Aggregate performance")[1].split("##")[0]
    assert "| −∞ |" in agg_section  # failure preserved as −∞
    assert "| ∞ |" not in agg_section  # never silently upgraded to bit-exact


def test_aggregate_hard_failure_not_picked_as_most_accurate_fp16():
    """The 'Most accurate FP16 mode' finding must not rank a hard failure
    (-inf PSNR) as the most accurate configuration."""
    base = RunResult(
        graph_key="m__m", model="m", component="m", backend="onnxruntime", fmt=".onnx",
        mode_id="cpu", mode_label="CPU", precision="fp32", status=STATUS_OK,
        model_path="/x.onnx", is_baseline=True, timing=_timing(50.0),
    )
    good = RunResult(
        graph_key="g__g", model="g", component="g", backend="coreml-mlpackage", fmt=".mlpackage",
        mode_id="cpu_only", mode_label="CPU only", precision="fp16", status=STATUS_OK,
        model_path="/g.mlpackage", timing=_timing(10.0), accuracy=_acc(45.0),
    )
    bad = RunResult(
        graph_key="m__m", model="m", component="m", backend="coreml-mlmodel", fmt=".mlmodel",
        mode_id="all", mode_label="ANE + GPU + CPU", precision="fp16", status=STATUS_OK,
        model_path="/x.mlmodel", timing=_timing(10.0), accuracy=_acc(float("-inf")),
    )
    out = markdown.render([base, good, bad])
    findings = out.split("## Key findings")[1].split("## ")[0]
    assert "Most accurate FP16 mode" in findings
    # The accurate (45 dB) config must win, not the -inf hard failure.
    assert "45.0 dB" in findings
    assert "−∞ dB" not in findings


def test_bit_exact_aggregate_renders_inf():
    """A backend whose measured accuracy is all-∞ (bit-exact) must still render ∞ in the aggregate."""
    base = RunResult(
        graph_key="m__m", model="m", component="m", backend="onnxruntime", fmt=".onnx",
        mode_id="cpu", mode_label="CPU", precision="fp32", status=STATUS_OK,
        model_path="/x.onnx", is_baseline=True, timing=_timing(50.0),
    )
    exact = RunResult(
        graph_key="m__m", model="m", component="m", backend="coreml-mlpackage", fmt=".mlpackage",
        mode_id="all", mode_label="ANE + GPU + CPU", precision="fp16", status=STATUS_OK,
        model_path="/x.mlpackage", timing=_timing(10.0), accuracy=_acc(float("inf")),
    )
    out = markdown.render([base, exact])
    agg_section = out.split("## Aggregate performance")[1].split("##")[0]
    assert "∞" in agg_section  # measured bit-exact accuracy preserved (not suppressed to "—")


def _coreml(graph, psnr, *, backend="coreml-mlpackage", fmt=".mlpackage", mode_id="all",
            label="ANE + GPU + CPU"):
    return RunResult(
        graph_key=graph, model=graph.split("__")[0], component=graph.split("__")[-1],
        backend=backend, fmt=fmt, mode_id=mode_id, mode_label=label, precision="fp16",
        status=STATUS_OK, model_path=f"/{graph}{fmt}", timing=_timing(10.0), accuracy=_acc(psnr),
    )


def _ort_base(graph="a__a"):
    return RunResult(
        graph_key=graph, model=graph.split("__")[0], component=graph.split("__")[-1],
        backend="onnxruntime", fmt=".onnx", mode_id="cpu", mode_label="CPU", precision="fp32",
        status=STATUS_OK, model_path=f"/{graph}.onnx", is_baseline=True, timing=_timing(50.0),
    )


def _reject_bare_nonfinite(c):
    raise AssertionError(f"bare non-finite literal {c!r} in JSON (invalid per spec)")


def test_report_json_is_strict_valid_with_nonfinite(tmp_path):
    """report.json must be strict-valid JSON even when accuracy is −∞ (hard
    failure) or +∞ (bit-exact): no bare Infinity/NaN tokens that strict parsers
    (JS/Go/Rust) reject."""
    import json
    runs = [_ort_base(),
            _coreml("a__a", float("-inf")),
            _coreml("b__b", float("inf"), backend="coreml-mlmodel", fmt=".mlmodel")]
    json_report.write(tmp_path / "r.json", runs)
    text = (tmp_path / "r.json").read_text()
    doc = json.loads(text, parse_constant=_reject_bare_nonfinite)
    assert len(doc["results"]) == 3


def test_run_result_json_strict_valid_and_roundtrips(tmp_path):
    """A per-run result JSON with a non-finite field must be strict-valid and
    round-trip back to the same float (−∞ stays −∞)."""
    import json
    rr = _coreml("a__a", float("-inf"))
    p = tmp_path / "run.json"
    rr.save(p)
    json.loads(p.read_text(), parse_constant=_reject_bare_nonfinite)
    back = RunResult.load(p)
    assert back.accuracy.min_psnr_db == float("-inf")
    assert back.accuracy.per_output[0].psnr_db == float("-inf")


def test_aggregate_mixed_failure_not_hidden_by_median():
    """A backend×mode column that passes on some graphs but hard-fails (−∞) on
    another must not display a clean finite median that hides the failure. The
    median over passing graphs is fine, but the failure must be surfaced."""
    runs = [_ort_base(), _coreml("a__a", 40.0), _coreml("b__b", 44.0),
            _coreml("c__c", float("-inf"))]
    out = markdown.render(runs)
    agg = out.split("## Aggregate performance")[1].split("\n##")[0]
    assert "42.0 (1✗)" in agg     # median(40,44)=42 over passing graphs, failure surfaced
    assert "| 42.0 |" not in agg  # never a bare clean median that buries the −∞ graph


def test_column_spec_has_structured_framework_and_mode():
    """The legend reads framework/mode from structured Column fields, not by
    string-splitting Column.full (which breaks on labels containing ' — ')."""
    runs = [_ort_base(), _coreml("a__a", 40.0)]
    cols = _collect.column_spec(runs)
    ml = next(c for c in cols if c.backend == "coreml-mlpackage")
    assert ml.framework == "Core ML (ML Program / .mlpackage)"
    assert ml.mode_label == "ANE + GPU + CPU"
    legend = markdown.render(runs).split("## Backends & modes")[1].split("\n## ")[0]
    assert "ANE + GPU + CPU" in legend and "Core ML (ML Program / .mlpackage)" in legend


def test_most_accurate_fp16_excludes_columns_with_hidden_failures():
    """'Most accurate FP16 mode' must not crown a column that hides a hard
    failure. A clean lower-PSNR column must beat a higher-median column that
    only looks good because its −∞ graph was dropped from the median."""
    runs = [
        _ort_base(),
        _coreml("a__a", 40.0),                       # .mlpackage/all: looks like 40 ...
        _coreml("b__b", float("-inf")),              # ... but hard-fails here (hidden)
        _coreml("a__a", 38.0, backend="coreml-mlmodel", fmt=".mlmodel"),  # clean 38/39
        _coreml("b__b", 39.0, backend="coreml-mlmodel", fmt=".mlmodel"),
    ]
    out = markdown.render(runs)
    findings = out.split("## Key findings")[1].split("\n## ")[0]
    assert "Most accurate FP16 mode" in findings
    line = next(ln for ln in findings.splitlines() if "Most accurate FP16" in ln)
    assert "Neural Network" in line  # the clean .mlmodel column (38.5 dB) wins
    assert "ML Program" not in line  # not the .mlpackage column that hides a −∞

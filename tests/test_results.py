import json

from modelbenchx.results import (
    STATUS_OK,
    AccuracyStats,
    OutputAccuracy,
    RunResult,
    TimingStats,
    dumps,
    finitize,
)


def _sample() -> RunResult:
    return RunResult(
        graph_key="resnet50__resnet50",
        model="resnet50",
        component="resnet50",
        backend="coreml-mlpackage",
        fmt=".mlpackage",
        mode_id="all",
        mode_label="ALL (ANE+GPU+CPU)",
        precision="fp16",
        status=STATUS_OK,
        model_path="/x/resnet50__resnet50.mlpackage",
        iters_requested=50,
        warmup_requested=5,
        timing=TimingStats(
            iters=50, mean_ms=1.2, median_ms=1.1, std_ms=0.1, min_ms=1.0,
            max_ms=2.0, p90_ms=1.4, p95_ms=1.5, p99_ms=1.9, throughput_ips=833.3,
            load_ms=40.0, first_call_ms=5.0, raw_ms=[1.0, 1.1, 1.2],
        ),
        accuracy=AccuracyStats(
            per_output=[OutputAccuracy(
                name="class_logits", shape=[1, 1000], dtype="float32",
                psnr_db=55.0, max_abs_err=0.01, max_rel_err=0.02, rmse=0.001,
                mae=0.0005, cosine=0.9999,
            )],
            min_psnr_db=55.0, max_abs_err=0.01, max_rel_err=0.02,
            mean_cosine=0.9999, all_finite_match=True,
        ),
        realized_device="ANE",
    )


def test_runresult_json_roundtrip(tmp_path):
    r = _sample()
    p = tmp_path / "r.json"
    r.save(p)
    back = RunResult.load(p)
    assert back == r
    assert back.timing.p99_ms == 1.9
    assert back.accuracy.per_output[0].name == "class_logits"


def test_string_field_named_like_a_sentinel_is_not_revived(tmp_path):
    """A tensor whose name is literally 'NaN'/'Infinity' (valid per the ONNX
    spec) must survive the JSON round-trip as that string; only numeric fields
    are revived from their non-finite tokens."""
    r = _sample()
    r.accuracy.per_output[0].name = "NaN"
    r.accuracy.per_output[0].psnr_db = float("-inf")
    r.accuracy.min_psnr_db = float("-inf")
    r.note = "Infinity"
    p = tmp_path / "r.json"
    r.save(p)
    back = RunResult.load(p)
    assert back.accuracy.per_output[0].name == "NaN"          # string, not float nan
    assert back.note == "Infinity"                            # string, not float inf
    assert back.accuracy.per_output[0].psnr_db == float("-inf")  # numeric field revived
    assert back.accuracy.min_psnr_db == float("-inf")


def test_finitize_handles_numpy_float_nonfinite():
    """finitize must convert numpy non-finite floats too (np.float32 is not a
    Python float subclass), so dumps never hits json's allow_nan=False guard."""
    import numpy as np
    out = finitize({"a": np.float32("inf"), "b": np.float64("-inf"), "c": np.float32(1.5)})
    assert out["a"] == "Infinity" and out["b"] == "-Infinity"
    json.loads(dumps({"v": np.float32("inf")}),
               parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(c)))


def test_save_is_atomic_no_tmp_left(tmp_path):
    r = _sample()
    p = tmp_path / "r.json"
    r.save(p)
    assert p.exists()
    assert not (tmp_path / "r.json.tmp").exists()


def test_raw_ms_nonfinite_roundtrip(tmp_path):
    """raw_ms travels through the same non-finite token machinery as scalar float
    fields; a non-finite element must revive to a float on load, not remain a
    string token (which would silently re-enter timing.summarize as a ghost inf)."""
    r = _sample()
    r.timing.raw_ms = [1.0, float("inf"), 3.0]
    p = tmp_path / "r.json"
    r.save(p)
    back = RunResult.load(p)
    assert back.timing.raw_ms == [1.0, float("inf"), 3.0]
    assert all(isinstance(v, float) for v in back.timing.raw_ms)

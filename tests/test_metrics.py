import math

import numpy as np
import pytest

from modelbenchx.metrics import accuracy, timing
from modelbenchx.results import TimingStats


def _t(**kw):
    base = dict(iters=10, mean_ms=10.0, median_ms=10.0, std_ms=2.0, min_ms=8.0,
                max_ms=12.0, p90_ms=11.0, p95_ms=11.5, p99_ms=12.0, throughput_ips=100.0)
    base.update(kw)
    return TimingStats(**base)


def test_cv_pct():
    assert timing.cv_pct(_t(mean_ms=10.0, std_ms=2.0)) == 20.0
    assert timing.cv_pct(_t(mean_ms=0.0)) is None  # guard divide-by-zero
    assert timing.cv_pct(None) is None
    # A multi-sample run with genuinely zero variance is steady (0%), not unknown.
    assert timing.cv_pct(_t(iters=10, mean_ms=10.0, std_ms=0.0)) == 0.0
    # A single sample has undefined variance: report None (not a fake 0% steady)
    # so it cannot pull the aggregate median CV toward zero.
    assert timing.cv_pct(_t(iters=1, mean_ms=10.0, std_ms=0.0)) is None


def test_cold_start_ms():
    assert timing.cold_start_ms(_t(load_ms=40.0, first_call_ms=5.0)) == 45.0
    assert timing.cold_start_ms(_t(load_ms=40.0, first_call_ms=None)) == 40.0  # load only
    assert timing.cold_start_ms(_t(load_ms=None)) is None
    assert timing.cold_start_ms(None) is None


# ---- timing ----------------------------------------------------------------

def test_timing_basic_stats():
    t = timing.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert t.iters == 5
    assert t.mean_ms == 3.0
    assert t.median_ms == 3.0
    assert t.min_ms == 1.0 and t.max_ms == 5.0
    assert math.isclose(t.throughput_ips, 1000.0 / 3.0)
    assert t.raw_ms == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_timing_single_sample_zero_std():
    t = timing.summarize([2.5], load_ms=10.0)
    assert t.std_ms == 0.0
    assert t.p99_ms == 2.5
    assert t.load_ms == 10.0
    assert timing.cv_pct(t) is None  # single-sample CV is undefined, not 0%


def test_summarize_requires_at_least_one_sample():
    with pytest.raises(ValueError):
        timing.summarize([])


def test_timing_drop_raw():
    t = timing.summarize([1.0, 2.0], keep_raw=False)
    assert t.raw_ms is None


# ---- accuracy --------------------------------------------------------------

def test_identical_is_perfect():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    o = accuracy.compare_tensor("x", a, a.copy())
    assert o.max_abs_err == 0.0
    assert math.isinf(o.psnr_db) and o.psnr_db > 0
    assert math.isclose(o.cosine, 1.0, rel_tol=1e-9)


def test_known_psnr():
    e = np.ones(4, dtype=np.float64)
    g = e.copy()
    g[0] = 1.1
    o = accuracy.compare_tensor("x", e, g)
    # rmse = sqrt(0.1^2 / 4) = 0.05; peak = 1 -> psnr = 20*log10(20) ~ 26.02 dB
    assert math.isclose(o.max_abs_err, 0.1, rel_tol=1e-9)
    assert math.isclose(o.rmse, 0.05, rel_tol=1e-9)
    assert math.isclose(o.psnr_db, 20 * math.log10(20), rel_tol=1e-9)


def test_nonfinite_masking_match():
    e = np.array([1.0, np.nan, 3.0])
    g = np.array([1.0, np.nan, 3.0])
    o = accuracy.compare_tensor("x", e, g)
    assert o.expected_nonfinite == 1
    assert o.nonfinite_match is True
    assert o.max_abs_err == 0.0


def test_nonfinite_mismatch_flagged():
    e = np.array([1.0, np.nan, 3.0])
    g = np.array([1.0, 2.0, 3.0])  # finite where ref is NaN -> divergence
    o = accuracy.compare_tensor("x", e, g)
    assert o.nonfinite_match is False


def test_nan_where_baseline_finite_is_hard_failure():
    """A candidate that emits NaN where the baseline is finite is a real
    divergence. The finite-overlap masking must not let it be reported as
    bit-exact (∞); it is a hard failure (−∞)."""
    e = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([1.0, 2.0, np.nan, 4.0])  # NaN where baseline is finite
    o = accuracy.compare_tensor("x", e, g)
    assert o.nonfinite_match is False
    assert math.isinf(o.psnr_db) and o.psnr_db < 0  # −∞, not +∞


def test_nan_divergence_makes_run_hard_failure():
    """Aggregated to the run level, a NaN-where-finite divergence yields
    min_psnr_db = −∞ and all_finite_match False (surfaced everywhere −∞ is)."""
    ref = {"y": np.array([1.0, 2.0, 3.0, 4.0])}
    got = {"y": np.array([1.0, 2.0, np.nan, 4.0])}
    stats = accuracy.compare_outputs(ref, got)
    assert stats.min_psnr_db == float("-inf")
    assert stats.all_finite_match is False


def test_reshape_when_size_matches():
    e = np.arange(4, dtype=np.float64).reshape(1, 4)
    g = np.arange(4, dtype=np.float64)  # (4,) vs (1,4)
    o = accuracy.compare_tensor("x", e, g)
    assert o.max_abs_err == 0.0


def test_size_mismatch_is_hard_failure():
    e = np.zeros(4)
    g = np.zeros(3)
    o = accuracy.compare_tensor("x", e, g)
    assert o.nonfinite_match is False
    assert math.isinf(o.psnr_db) and o.psnr_db < 0


def test_classify_psnr():
    assert accuracy.classify_psnr(float("inf")) == accuracy.PSNR_EXACT
    assert accuracy.classify_psnr(float("-inf")) == accuracy.PSNR_FAIL
    assert accuracy.classify_psnr(float("nan")) == accuracy.PSNR_FAIL
    assert accuracy.classify_psnr(40.0) == accuracy.PSNR_OK
    assert accuracy.classify_psnr(0.0) == accuracy.PSNR_OK


def test_compare_outputs_name_matching_and_aggregate():
    ref = {"a": np.ones(3), "b.c": np.ones(3)}
    got = {"a": np.ones(3), "b_c": np.ones(3) * 1.0}  # sanitized name match
    stats = accuracy.compare_outputs(ref, got)
    assert len(stats.per_output) == 2
    assert stats.all_finite_match is True
    assert math.isinf(stats.min_psnr_db)  # both perfect


def test_compare_outputs_missing_output():
    ref = {"a": np.ones(3), "b": np.ones(3)}
    got = {"a": np.ones(3)}
    stats = accuracy.compare_outputs(ref, got)
    assert "missing" in stats.note
    assert stats.min_psnr_db == float("-inf")
    assert stats.all_finite_match is False


def test_zero_baseline_rel_err_is_inf_not_zero():
    """A candidate that deviates from a zero baseline has undefined (infinite)
    relative error; reporting 0.0 there would read as 'relatively perfect' in the
    CSV/JSON. PSNR is a hard failure (-inf) since the signal's dynamic range is 0."""
    e = np.zeros(4, dtype=np.float32)
    g = np.full(4, 1e-3, dtype=np.float32)
    o = accuracy.compare_tensor("x", e, g)
    assert math.isinf(o.max_rel_err) and o.max_rel_err > 0
    assert math.isinf(o.psnr_db) and o.psnr_db < 0


def test_zero_baseline_zero_candidate_is_perfect():
    """Both sides zero: relative error is 0 (not inf) and the match is bit-exact."""
    z = np.zeros(4, dtype=np.float32)
    o = accuracy.compare_tensor("x", z, z.copy())
    assert o.max_rel_err == 0.0
    assert math.isinf(o.psnr_db) and o.psnr_db > 0


def test_partial_zero_baseline_rel_err_uses_defined_region():
    """Relative error is reported only over nonzero-baseline positions, so a single
    zero-baseline element does not inflate max_rel_err to inf; the deviation there
    is still captured by absolute error / PSNR (which stay finite)."""
    e = np.array([0.0, 2.0], dtype=np.float64)
    g = np.array([0.01, 2.2], dtype=np.float64)  # 0.01 dev at zero baseline; 10% at nonzero
    o = accuracy.compare_tensor("x", e, g)
    assert math.isclose(o.max_rel_err, 0.1, rel_tol=1e-9)  # |2.2-2|/2 only
    assert math.isclose(o.max_abs_err, 0.2)                # max abs over all positions
    assert math.isfinite(o.psnr_db)


def test_integer_dtype_outputs_compare_finite():
    """Integer outputs (e.g. argmax/class ids) promote to float64 and compare
    cleanly: no spurious non-finite count, finite PSNR when they differ."""
    e = np.array([1, 2, 3, 4], dtype=np.int32)
    g = np.array([1, 2, 3, 5], dtype=np.int32)
    o = accuracy.compare_tensor("ids", e, g)
    assert o.expected_nonfinite == 0
    assert o.nonfinite_match is True
    assert math.isfinite(o.psnr_db)
    assert math.isclose(o.max_abs_err, 1.0)

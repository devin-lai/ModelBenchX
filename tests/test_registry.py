from pathlib import Path

import pytest

from modelbenchx import registry

TEST_MODEL_DIR = Path(__file__).resolve().parent.parent / "test_model"
pytestmark = pytest.mark.skipif(
    not TEST_MODEL_DIR.exists(), reason="test_model/ data not present"
)


def test_discover_union_semantics():
    """discover() returns every graph present in any format (union, not intersection)."""
    reg = registry.discover(TEST_MODEL_DIR)
    # Union discovery never skips anything.
    assert reg.skipped == []
    # Every benchmarkable graph has at least one source.
    assert all(len(r.sources) >= 1 for r in reg.benchmarkable)
    # Sources reference only registered formats.
    assert all(set(r.sources) <= set(reg.per_format_counts) for r in reg.benchmarkable)


def test_per_format_counts_non_negative():
    """per_format_counts covers all registered formats with non-negative counts."""
    reg = registry.discover(TEST_MODEL_DIR)
    assert reg.per_format_counts
    assert all(v >= 0 for v in reg.per_format_counts.values())
    # Total benchmarkable graphs cannot exceed the sum across all format counts.
    assert len(reg.benchmarkable) <= sum(reg.per_format_counts.values())


def test_multi_component_model_resolves():
    reg = registry.discover(TEST_MODEL_DIR)
    by_key = {r.key: r for r in reg.benchmarkable}
    # sam2 splits into encoder/decoder; encoder is present in >=1 format.
    if "sam2__encoder" in by_key:
        rec = by_key["sam2__encoder"]
        assert rec.model == "sam2" and rec.component == "encoder"
        if "onnx" in rec.sources:
            onnx_src = rec.source("onnx")
            assert onnx_src.member and onnx_src.member.endswith("encoder.onnx")
            assert onnx_src.path.endswith("sam2-onnx-float.zip")

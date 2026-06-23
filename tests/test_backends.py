import pytest

from modelbenchx.backends import base


def test_select_backends_unknown_name_raises_with_choices():
    """A typo'd --backends value must raise a clean, actionable ValueError listing
    the choices, not a bare KeyError traceback out of Orchestrator.__init__."""
    with pytest.raises(ValueError) as exc:
        base.select_backends(("coreml", "bogus"), system="Darwin")
    msg = str(exc.value)
    assert "coreml" in msg and "bogus" in msg
    assert "onnxruntime" in msg  # lists valid choices


def test_every_backend_has_a_format_spec():
    for b in base.all_backends():
        assert b.discovery.suffix.startswith(".")
        assert callable(b.discovery.key_fn)


def test_apple_backends_are_darwin_gated():
    by = {b.name: b for b in base.all_backends()}
    assert by["coreml-mlmodel"].platforms == ("Darwin",)
    assert by["coreml-mlpackage"].platforms == ("Darwin",)
    assert by["coreai"].platforms == ("Darwin",)
    assert by["onnxruntime"].platforms is None  # cross-platform


def test_select_backends_filters_by_platform():
    names = [b.name for b in base.select_backends(None, system="Linux")]
    assert "onnxruntime" in names
    assert "coreml-mlmodel" not in names and "coreai" not in names and "coreml-mlpackage" not in names


def test_select_backends_keeps_baseline_first():
    chosen = base.select_backends(None, system="Darwin")
    assert chosen[0].is_baseline

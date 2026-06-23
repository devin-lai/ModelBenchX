from modelbenchx.registry import discover


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_union_discovery_partial_presence(tmp_path):
    # resnet in onnx(zip) + mlmodel; squeeze only in mlmodel.
    import zipfile
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir(parents=True)
    with zipfile.ZipFile(onnx_dir / "resnet-onnx-float.zip", "w") as z:
        z.writestr("resnet.onnx", b"x")
    _touch(tmp_path / "mlmodel" / "resnet__resnet.mlmodel")
    _touch(tmp_path / "mlmodel" / "squeeze__squeeze.mlmodel")

    reg = discover(tmp_path)
    keys = {r.key for r in reg.benchmarkable}
    assert keys == {"resnet__resnet", "squeeze__squeeze"}     # union, not intersection
    resnet = reg.get("resnet__resnet")
    assert set(resnet.sources) == {"onnx", "mlmodel"}
    assert set(reg.get("squeeze__squeeze").sources) == {"mlmodel"}


def test_bad_zip_is_skipped_not_fatal(tmp_path):
    # A corrupt archive must be skipped, never crash discovery.
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir(parents=True)
    (onnx_dir / "broken-onnx-float.zip").write_bytes(b"not a zip")
    reg = discover(tmp_path)  # must not raise
    assert reg.benchmarkable == []  # corrupt archive contributes nothing


def test_synth_backend_discovered_under_flag(tmp_path, monkeypatch):
    import importlib

    from modelbenchx import backends, registry

    monkeypatch.setenv("MODELBENCHX_SYNTH", "1")
    importlib.reload(backends.base)
    importlib.reload(registry)
    try:
        (tmp_path / "synth").mkdir()
        (tmp_path / "synth" / "m__m.npmodel").write_bytes(b"x")
        reg = registry.discover(tmp_path)
        assert "m__m" in {r.key for r in reg.benchmarkable}
    finally:
        monkeypatch.delenv("MODELBENCHX_SYNTH", raising=False)
        importlib.reload(backends.base)
        importlib.reload(registry)

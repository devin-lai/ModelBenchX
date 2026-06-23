"""Positional npz transport + 32-bit narrowing (workers/_io).

This is the data plane between processes: feeds and outputs travel as positional
npz (keys ``arr_0`` …) because ONNX tensor names are not valid ``np.savez``
keyword keys, and the ordered name list re-keys them on the far side. A break
here silently misaligns accuracy comparisons, so it is worth pinning down.
"""

import numpy as np

from modelbenchx.workers import _io as npio


def test_save_load_roundtrip_preserves_names_values_and_order(tmp_path):
    # Names that are invalid as np.savez kwargs (the reason for positional keys).
    names = ["/enc/Add_output_0", "123_logits", "x"]
    by_name = {
        "/enc/Add_output_0": np.arange(6, dtype=np.float32).reshape(2, 3),
        "123_logits": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "x": np.array([True, False, True]),
    }
    path = tmp_path / "f.npz"
    npio.save_named(path, names, by_name)
    out = npio.load_named(path, names)

    assert list(out) == names  # order preserved
    for n in names:
        assert np.array_equal(out[n], by_name[n])
        assert out[n].dtype == by_name[n].dtype


def test_load_uses_given_order_not_storage_order(tmp_path):
    # Saving in one order then loading with the same name list must map 1:1;
    # a different but length-matched name list re-keys by position.
    names = ["a", "b"]
    npio.save_named(tmp_path / "g.npz", names, {"a": np.array([1.0]), "b": np.array([2.0])})
    relabel = npio.load_named(tmp_path / "g.npz", ["first", "second"])
    assert relabel["first"][0] == 1.0 and relabel["second"][0] == 2.0


def test_save_load_samples_roundtrip(tmp_path):
    names = ["/enc/x", "y"]
    samples = [
        {"/enc/x": np.arange(3, dtype=np.float32), "y": np.ones(2, dtype=np.int32)},
        {"/enc/x": np.arange(3, dtype=np.float32) + 5, "y": np.full(2, 9, dtype=np.int32)},
    ]
    path = tmp_path / "s.npz"
    npio.save_samples(path, names, samples)
    back = npio.load_samples(path, names, 2)
    assert len(back) == 2
    assert list(back[0]) == names  # order preserved per sample
    assert np.array_equal(back[0]["/enc/x"], np.arange(3))
    assert np.array_equal(back[1]["/enc/x"], np.arange(3) + 5)
    assert np.array_equal(back[1]["y"], np.full(2, 9))


def test_narrow_array_narrows_only_wide_dtypes():
    assert npio.narrow_array(np.array([1], dtype=np.int64)).dtype == np.int32
    assert npio.narrow_array(np.array([1], dtype=np.uint64)).dtype == np.uint32
    assert npio.narrow_array(np.array([1.0], dtype=np.float64)).dtype == np.float32
    # Already-narrow dtypes are passed through unchanged (no copy of semantics).
    for dt in (np.float32, np.float16, np.int32, np.int8, np.bool_):
        assert npio.narrow_array(np.array([1], dtype=dt)).dtype == dt


def test_narrow_array_preserves_values_and_shape():
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    n = npio.narrow_array(a)
    assert n.shape == a.shape
    assert np.array_equal(n, a.astype(np.int32))

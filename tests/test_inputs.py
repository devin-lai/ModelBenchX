"""Seeded input generation (workers/_inputs).

This module decides the actual numbers every latency and accuracy measurement is
taken on, so its contract matters: a stable dtype mapping, deterministic feed for
a given seed (the premise of reusing one feed across all backends), correct
dynamic-dim resolution, and range-honoring values for image-like inputs.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from modelbenchx.workers import _inputs
from modelbenchx.workers import _inputs as _inp


def _model(inputs, initializers=()):
    """Build a minimal ModelProto from ``(name, elem_type, shape)`` input specs.

    ``shape`` dims may be ints (static) or strings (symbolic/dynamic). Only the
    graph inputs + initializers are read by ``generate_inputs``; no nodes needed.
    """
    vis = [helper.make_tensor_value_info(n, et, shape) for n, et, shape in inputs]
    inits = [
        helper.make_tensor(n, TensorProto.FLOAT, arr.shape, arr.flatten().tolist())
        for n, arr in initializers
    ]
    graph = helper.make_graph([], "g", vis, [], initializer=inits)
    return helper.make_model(graph)


def test_same_seed_gives_identical_feed():
    # Determinism is the whole point: every backend must see the same inputs.
    m = _model([("x", TensorProto.FLOAT, [2, 3])])
    a = _inputs.generate_inputs(m, {}, seed=0)
    b = _inputs.generate_inputs(m, {}, seed=0)
    assert set(a) == {"x"}
    assert a["x"].dtype == np.float32 and a["x"].shape == (2, 3)
    assert np.array_equal(a["x"], b["x"])


def test_different_seed_changes_feed():
    m = _model([("x", TensorProto.FLOAT, [4, 4])])
    a = _inputs.generate_inputs(m, {}, seed=0)
    b = _inputs.generate_inputs(m, {}, seed=1)
    assert not np.array_equal(a["x"], b["x"])


def test_dtype_mapping():
    m = _model([
        ("f", TensorProto.FLOAT, [2]),
        ("h", TensorProto.FLOAT16, [2]),
        ("i", TensorProto.INT64, [2]),
        ("b", TensorProto.BOOL, [2]),
    ])
    feed = _inputs.generate_inputs(m, {}, seed=0)
    assert feed["f"].dtype == np.float32
    assert feed["h"].dtype == np.float16
    assert feed["i"].dtype == np.int64
    assert feed["b"].dtype == np.bool_


def test_symbolic_dims_resolve_to_dynamic_dim_size():
    m = _model([("x", TensorProto.FLOAT, ["N", 3, "H", "W"])])
    feed = _inputs.generate_inputs(m, {}, seed=0, dynamic_dim_size=2)
    assert feed["x"].shape == (2, 3, 2, 2)  # symbolic -> dynamic_dim_size; static kept


def test_metadata_shape_overrides_symbolic_dims():
    m = _model([("x", TensorProto.FLOAT, ["N", 3, "H", "W"])])
    meta = {"x": {"shape": [1, 3, 224, 224]}}
    feed = _inputs.generate_inputs(m, meta, seed=0, dynamic_dim_size=1)
    assert feed["x"].shape == (1, 3, 224, 224)


def test_static_dims_win_over_dynamic_dim_size():
    m = _model([("x", TensorProto.FLOAT, [1, 10])])
    feed = _inputs.generate_inputs(m, {}, seed=0, dynamic_dim_size=99)
    assert feed["x"].shape == (1, 10)


def test_float_value_range_is_respected():
    m = _model([("x", TensorProto.FLOAT, [1000])])
    meta = {"x": {"value_range": [10.0, 11.0]}}
    x = _inputs.generate_inputs(m, meta, seed=0)["x"]
    assert x.min() >= 10.0 and x.max() <= 11.0  # uniform in range, not N(0,1)


def test_image_io_type_lands_in_unit_range():
    m = _model([("x", TensorProto.FLOAT, [3, 8, 8])])
    meta = {"x": {"io_type": "image"}}
    x = _inputs.generate_inputs(m, meta, seed=0)["x"]
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_initializer_shadowed_inputs_are_skipped():
    # An input that is also an initializer (a weight) is supplied by the graph,
    # so it must not receive a random feed.
    w = np.ones((2, 2), dtype=np.float32)
    m = _model(
        [("x", TensorProto.FLOAT, [2, 2]), ("W", TensorProto.FLOAT, [2, 2])],
        initializers=[("W", w)],
    )
    feed = _inputs.generate_inputs(m, {}, seed=0)
    assert set(feed) == {"x"}


def test_integer_inputs_are_small_and_nonnegative():
    m = _model([("idx", TensorProto.INT32, [100])])
    x = _inputs.generate_inputs(m, {}, seed=0)["idx"]
    assert x.dtype == np.int32
    assert x.min() >= 0 and x.max() <= 9  # indices: small, non-negative


def test_integer_value_range_caps_upper_bound():
    m = _model([("idx", TensorProto.INT32, [100])])
    meta = {"idx": {"value_range": [0, 3]}}
    x = _inputs.generate_inputs(m, meta, seed=0)["idx"]
    assert x.min() >= 0 and x.max() <= 3


def test_unsupported_dtype_raises():
    # STRING has no numpy mapping -> an explicit, early ValueError (not garbage).
    m = _model([("s", TensorProto.STRING, [2])])
    with pytest.raises(ValueError):
        _inputs.generate_inputs(m, {}, seed=0)


def test_generated_model_is_structurally_valid():
    # Guard the test helper itself: the synthetic model must be well-formed
    # enough that we are exercising generate_inputs, not a malformed proto.
    m = _model([("x", TensorProto.FLOAT, [1, 2])])
    assert isinstance(m, onnx.ModelProto)
    assert [i.name for i in m.graph.input] == ["x"]


def test_spec_from_onnx_resolves_and_maps():
    m = _model([("x", TensorProto.FLOAT, ["N", 3, "H", "W"])])
    specs = _inp.spec_from_onnx(m, {"x": {"shape": [1, 3, 8, 8], "io_type": "image"}})
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "x" and s.shape == (1, 3, 8, 8)
    assert s.dtype == np.dtype(np.float32) and s.io_type == "image"

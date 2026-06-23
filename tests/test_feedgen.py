import numpy as np

from modelbenchx.workers._feedgen import InputSpec, generate_from_spec, generate_samples


def test_same_seed_identical_feed():
    specs = [InputSpec("x", (2, 3), np.dtype(np.float32))]
    a = generate_from_spec(specs, seed=0)
    b = generate_from_spec(specs, seed=0)
    assert set(a) == {"x"} and a["x"].shape == (2, 3) and a["x"].dtype == np.float32
    assert np.array_equal(a["x"], b["x"])


def test_value_range_respected():
    specs = [InputSpec("x", (1000,), np.dtype(np.float32), value_range=(10.0, 11.0))]
    x = generate_from_spec(specs, seed=0)["x"]
    assert x.min() >= 10.0 and x.max() <= 11.0


def test_image_io_type_unit_range():
    specs = [InputSpec("x", (3, 8, 8), np.dtype(np.float32), io_type="image")]
    x = generate_from_spec(specs, seed=0)["x"]
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_integer_small_nonnegative():
    x = generate_from_spec([InputSpec("i", (100,), np.dtype(np.int32))], seed=0)["i"]
    assert x.dtype == np.int32 and x.min() >= 0 and x.max() <= 9


def test_integer_value_range_nonzero_lower_bound():
    # The lower bound must be honored, not only the upper: integer inputs whose
    # valid range starts above 0 (e.g. token ids over a vocab that starts at 1)
    # must never receive out-of-range indices.
    specs = [InputSpec("ids", (1000,), np.dtype(np.int64), value_range=(5.0, 50.0))]
    x = generate_from_spec(specs, seed=0)["ids"]
    assert x.min() >= 5 and x.max() <= 50


def test_integer_value_range_clamped_to_dtype_bounds():
    # A value_range wider than the integer dtype must be clamped, never wrapped by
    # the astype() cast: int8 holds [-128, 127], so a [0, 200] hint reaching
    # astype() would emit negatives (200 -> -56), corrupting the feed and the
    # reference baseline computed from it.
    x = generate_from_spec([InputSpec("q", (1000,), np.dtype(np.int8), value_range=(0.0, 200.0))], seed=0)["q"]
    assert x.dtype == np.int8 and x.min() >= 0 and x.max() <= 127
    # uint8 [0, 300] clamps to [0, 255] (no modular truncation past 255).
    u = generate_from_spec([InputSpec("u", (1000,), np.dtype(np.uint8), value_range=(0.0, 300.0))], seed=1)["u"]
    assert u.dtype == np.uint8 and u.min() >= 0 and u.max() <= 255


def test_bool_dtype():
    x = generate_from_spec([InputSpec("b", (5,), np.dtype(np.bool_))], seed=0)["b"]
    assert x.dtype == np.bool_


def test_generate_samples_count_determinism_and_distinctness():
    specs = [InputSpec("x", (2, 3), np.dtype(np.float32))]
    s = generate_samples(specs, seed=0, n=3)
    assert len(s) == 3
    # deterministic for the same (seed, n)
    again = generate_samples(specs, seed=0, n=3)
    for a, b in zip(s, again, strict=True):
        assert np.array_equal(a["x"], b["x"])
    # distinct samples (consecutive seeds), so latency reflects >1 input
    assert not np.array_equal(s[0]["x"], s[1]["x"])
    # sample 0 is exactly the single-feed generation at `seed` (K=1 compatibility)
    assert np.array_equal(s[0]["x"], generate_from_spec(specs, seed=0)["x"])


def test_generate_samples_n1_matches_single():
    specs = [InputSpec("x", (4,), np.dtype(np.float32))]
    s = generate_samples(specs, seed=7, n=1)
    assert len(s) == 1
    assert np.array_equal(s[0]["x"], generate_from_spec(specs, seed=7)["x"])

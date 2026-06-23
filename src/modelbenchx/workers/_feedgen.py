"""Seeded input generation from a framework-agnostic spec. numpy only.

No ``onnx``/``coremltools`` import, so the worker harness (used by every backend,
including the Apple ones) can build feeds without pulling a conflicting runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputSpec:
    """One model input: a concrete shape + dtype, with optional range hints."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    value_range: tuple[float, float] | None = None
    io_type: str | None = None


def generate_from_spec(specs: list[InputSpec], *, seed: int = 0) -> dict[str, np.ndarray]:
    """Build a deterministic feed for the given specs (one RNG, seeded once)."""
    rng = np.random.default_rng(seed)
    feed: dict[str, np.ndarray] = {}
    for s in specs:
        if s.dtype.kind == "f":
            if s.value_range is not None:
                lo, hi = float(s.value_range[0]), float(s.value_range[1])
                arr = rng.uniform(lo, hi, size=s.shape)
            elif s.io_type == "image":
                arr = rng.uniform(0.0, 1.0, size=s.shape)
            else:
                arr = rng.standard_normal(s.shape)
            feed[s.name] = arr.astype(s.dtype)
        elif s.dtype.kind == "b":
            feed[s.name] = rng.integers(0, 2, size=s.shape).astype(s.dtype)
        else:
            lo, hi = 0, 9
            if s.value_range is not None:
                lo = int(s.value_range[0])
                hi = max(lo, int(s.value_range[1]))
            feed[s.name] = rng.integers(lo, hi + 1, size=s.shape).astype(s.dtype)
    return feed


def generate_samples(specs: list[InputSpec], *, seed: int = 0, n: int = 1) -> list[dict[str, np.ndarray]]:
    """``n`` deterministic feeds using consecutive seeds (``seed`` … ``seed+n-1``).

    Sample 0 is exactly ``generate_from_spec(specs, seed=seed)``, so ``n == 1``
    reproduces single-feed behavior. Distinct inputs let latency reflect more
    than one point, useful for data-dependent graphs whose path varies by input.
    """
    return [generate_from_spec(specs, seed=seed + s) for s in range(n)]

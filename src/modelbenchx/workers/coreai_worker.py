"""Core AI worker. Runs a ``.aimodel`` via the native Core AI runtime.

One compute mode per invocation (``cpu_only`` fp32 / ``gpu`` / ``ane`` fp16 /
``all`` auto). Imports coreai + numpy only. The native runtime can ``abort()``
the process on a mode it cannot execute; that is expected and the parent turns
the resulting signal death into a recorded failure (this worker cannot catch it).

Run as: ``python -m modelbenchx.workers.coreai_worker <jobdir>``

Note on load_ms vs first_call_ms: the async executable context manager is entered
lazily on the first ``infer`` call (because ``load`` is sync). This means the
compile/specialize cost lands in ``first_call_ms`` (warmup) rather than
``load_ms``. ``cold_start_ms = load_ms + first_call_ms`` is unchanged, so the
reported cold start is preserved. This is intentional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import _io as npio
from ._harness import AsyncWorker, arun_worker


def _specialization_options(mode: str):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    if mode == "all":
        return None
    if mode == "cpu_only":
        return SpecializationOptions.cpu_only()
    kinds = {
        "cpu": ComputeUnitKind.cpu,
        "gpu": ComputeUnitKind.gpu,
        "ane": ComputeUnitKind.neural_engine,
    }
    if mode not in kinds:
        raise ValueError(f"unknown coreai mode {mode!r}")
    return SpecializationOptions.from_preferred_compute_unit_kind(kinds[mode]())


class CoreAIWorker(AsyncWorker):
    def load(self, meta):
        from coreai.authoring import AIModelAsset

        self._opts = _specialization_options(meta["mode"])
        self._asset = AIModelAsset.load(Path(meta["model_path"]))
        self._cm = self._asset.executable(self._opts)
        self._ai = None
        self._out_names = meta["output_names"]
        self._mode = meta["mode"]
        self._entrypoint = meta["entrypoint"]

    def build_feed(self, shared, meta):
        from coreai.runtime import NDArray

        names = meta["input_names"]
        return {n: NDArray(npio.narrow_array(shared[n])) for n in names}

    async def infer(self, feed):
        if self._ai is None:
            self._ai = await self._cm.__aenter__()
            self._fn = self._ai.load_function(self._entrypoint)
        return await self._fn(feed)

    def materialize(self, out):
        # Copy device outputs to host inside the timed region so Core AI latency
        # includes the device->host transfer, matching Core ML's predict() which
        # already returns host arrays. Previously .numpy() ran in extract_outputs
        # (outside timing), understating latency on large outputs.
        return {n: np.asarray(out[n].numpy()) for n in self._out_names}

    def output_names(self):
        return self._out_names

    def realized_device(self):
        return self._mode

    def extract_outputs(self, last):
        return last  # already host-materialized in materialize()

    async def aclose(self):
        if self._ai is not None:
            await self._cm.__aexit__(None, None, None)
            self._ai = None


if __name__ == "__main__":
    sys.exit(arun_worker(sys.argv[1], CoreAIWorker()))

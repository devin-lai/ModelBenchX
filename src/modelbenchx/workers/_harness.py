"""Shared worker harness: timing + the parent<->worker file protocol in one place.

A backend author implements ``Worker`` (or ``AsyncWorker``); the harness handles
load timing, feed resolution (generate-and-share or consume-shared), the steady-
state loop, output capture, and crash -> error.json. numpy + stdlib only. Never
imports onnx/coremltools/coreai.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from . import _bench
from . import _io as npio
from . import _protocol as P
from ._feedgen import InputSpec, generate_samples


class Worker:
    def load(self, meta: dict) -> None: ...
    def build_feed(self, shared: dict[str, Any] | None, meta: dict) -> Any:
        return shared
    def infer(self, feed: Any) -> Any: raise NotImplementedError
    def materialize(self, out: Any) -> Any:
        """Force host realization of ``infer``'s result. Called inside the timed
        region so device->host materialization is part of the measured latency.

        Default is a no-op: backends whose ``infer`` already returns host arrays
        (ONNX Runtime ``session.run``, Core ML ``predict``, TFLite ``get_tensor``)
        need nothing. Backends that return lazy device handles (Core AI) override
        this to copy to host, so they are timed on equal footing rather than
        deferring the copy to ``extract_outputs`` (outside timing).
        """
        return out
    def output_names(self) -> list[str]: raise NotImplementedError
    def extract_outputs(self, last: Any) -> dict[str, Any]:
        return dict(zip(self.output_names(), last, strict=True))
    def input_spec(self) -> list[InputSpec] | None:
        return None
    def realized_device(self) -> str | None:
        return None


class AsyncWorker(Worker):
    async def infer(self, feed: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError

    async def aclose(self) -> None:
        ...


def _resolve_feed(jobdir: Path, meta: dict, worker: Worker) -> list[dict[str, Any]] | None:
    """Resolve the input feed(s) as a list of samples. ``input_samples`` (default
    1) controls how many; sample 0 is always the canonical feed used for accuracy.
    With 1 sample the on-disk npz uses the original single-feed contract, so
    existing caches and behavior are unchanged."""
    n = int(meta.get("input_samples", 1))
    if meta.get("generate_feed"):
        specs = worker.input_spec()
        if specs is None:
            raise RuntimeError(
                "backend was asked to generate the feed but has no input_spec()"
            )
        samples = generate_samples(specs, seed=meta.get("seed", 0), n=n)
        names = list(samples[0])
        if n > 1:
            npio.save_samples(meta["shared_inputs_npz"], names, samples)
        else:
            npio.save_named(meta["shared_inputs_npz"], names, samples[0])
        npio.write_text_atomic(meta["shared_input_names_json"], json.dumps(names))
        return samples
    if meta.get("inputs_npz"):
        if n > 1:
            return npio.load_samples(meta["inputs_npz"], meta["input_names"], n)
        return [npio.load_named(meta["inputs_npz"], meta["input_names"])]
    return None


def _rotating_call(worker: Worker, feeds: list[Any]):
    """A zero-arg call that rotates through ``feeds`` each invocation, applying the
    timed ``materialize`` step. With one feed it always uses that feed (current
    behavior); with several it spreads latency across representative inputs."""
    counter = itertools.count()
    def _call() -> Any:
        return worker.materialize(worker.infer(feeds[next(counter) % len(feeds)]))
    return _call


def _finish(jobdir: Path, worker: Worker, load_ms: float, raw, first_ms, last) -> None:
    outputs = worker.extract_outputs(last)
    names = list(outputs)
    npio.save_named(jobdir / P.OUTPUTS, names, outputs)
    P.write_result(jobdir, {
        "raw_ms": raw, "load_ms": load_ms, "first_call_ms": first_ms,
        "realized_device": worker.realized_device(),
        "produced_output_names": names, "note": "",
    })


def run_worker(jobdir: str | Path, worker: Worker) -> int:
    jobdir = Path(jobdir)
    try:
        meta = P.read_meta(jobdir)
        t0 = perf_counter_ns()
        worker.load(meta)
        load_ms = (perf_counter_ns() - t0) / 1e6
        samples = _resolve_feed(jobdir, meta, worker)
        feeds = ([worker.build_feed(s, meta) for s in samples] if samples is not None
                 else [worker.build_feed(None, meta)])
        raw, first_ms, last = _bench.run_timed(
            _rotating_call(worker, feeds), warmup=meta["warmup"],
            min_iters=meta["min_iters"], max_iters=meta["max_iters"],
            time_budget_s=meta["time_budget_s"])
        if len(feeds) > 1:  # accuracy uses sample 0, independent of where timing stopped
            last = worker.materialize(worker.infer(feeds[0]))
        _finish(jobdir, worker, load_ms, raw, first_ms, last)
        return 0
    except Exception as exc:  # handled -> error.json, exit 1
        P.write_error(jobdir, exc)
        return 1


def arun_worker(jobdir: str | Path, worker: AsyncWorker) -> int:
    async def _run() -> None:
        jd = Path(jobdir)
        meta = P.read_meta(jd)
        t0 = perf_counter_ns()
        worker.load(meta)
        load_ms = (perf_counter_ns() - t0) / 1e6
        try:
            samples = _resolve_feed(jd, meta, worker)
            feeds = ([worker.build_feed(s, meta) for s in samples] if samples is not None
                     else [worker.build_feed(None, meta)])
            counter = itertools.count()

            async def _call():
                return worker.materialize(await worker.infer(feeds[next(counter) % len(feeds)]))

            raw, first_ms, last = await _bench.arun_timed(
                _call, warmup=meta["warmup"], min_iters=meta["min_iters"],
                max_iters=meta["max_iters"], time_budget_s=meta["time_budget_s"])
            if len(feeds) > 1:  # accuracy uses sample 0
                last = worker.materialize(await worker.infer(feeds[0]))
            _finish(jd, worker, load_ms, raw, first_ms, last)
        finally:
            # cleanup must never mask the original failure/abort
            with contextlib.suppress(Exception):
                await worker.aclose()

    try:
        asyncio.run(_run())
        return 0
    except Exception as exc:
        P.write_error(Path(jobdir), exc)
        return 1

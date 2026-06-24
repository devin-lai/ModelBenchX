"""Serial benchmark driver.

For each graph it runs only the backends whose format is present in that graph's
``sources``. Timing is always measured; accuracy is decoupled from it: when the
configured reference backend (default ``onnxruntime``) is among a graph's
sources it runs first, generating the shared feed and the reference outputs.
Every other backend's accuracy is computed against those outputs. When the
reference is absent, a backend that can generate inputs (``provides_feed``)
produces the feed instead and the graph's runs are latency-only. If neither a
reference nor a feed-capable backend is available the graph is skipped.

Runs are serial so they never contend for the CPU/GPU/ANE, and every
``(graph, backend, mode)`` result is cached on disk so a run is fully resumable.
Pure numpy and stdlib; never imports the conflicting runtimes.
"""

from __future__ import annotations

import contextlib
import json
import platform
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .backends.base import Backend, get_backend, select_backends
from .config import BenchmarkConfig
from .metrics import accuracy as acc
from .metrics import timing as tmetrics
from .registry import GraphRecord, Registry, discover
from .results import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    RunResult,
)
from .workers import _io as npio
from .workers import _protocol as P

ProgressFn = Callable[[str], None]

# Name of the per-graph file recording which feed-affecting parameters produced
# the cached inputs + reference outputs (see ``_feed_fingerprint``).
FEED_FINGERPRINT = "feed_fingerprint.json"

# How to read a cache written before fingerprinting existed: it was, by
# definition, produced with the original ``BenchmarkConfig`` defaults. Treating a
# missing fingerprint as these defaults means an unchanged default run keeps its
# cache, while any non-default feed parameter correctly invalidates it.
_LEGACY_FINGERPRINT = {
    "seed": 0,
    "dynamic_dim_size": 1,
    "ort_disable_optimizations": True,
    "reference_backend": "onnxruntime",
    "input_samples": 1,
}


def _noop(_: str) -> None:
    pass


@dataclass(frozen=True)
class _FeedState:
    """Outcome of ensuring a graph's shared feed (and, if a reference ran, its
    reference outputs) exist.

    ``input_names`` is ``None`` and ``ok`` is ``False`` when no feed could be
    produced: either the feed worker failed, or there was no reference and no
    feed-capable backend (downstream runs are then skipped). ``output_names`` and
    ``outputs`` are ``None`` whenever no reference ran (latency-only). ``feed_by``
    is the backend that produced the feed (it has already run and must not be
    re-run as a consumer), or ``None`` when the graph is skipped. ``feed_changed``
    is ``True`` when a stale cache was regenerated because the feed parameters
    changed, so this graph's cached backend results must be re-run to stay
    comparable.
    """

    input_names: list[str] | None
    output_names: list[str] | None
    outputs: dict | None
    ok: bool
    result: RunResult | None
    feed_by: Backend | None
    feed_changed: bool = False
    skip_note: str = "no feed available"


class Orchestrator:
    def __init__(self, config: BenchmarkConfig, progress: ProgressFn | None = None):
        self.cfg = config
        self.backends = select_backends(config.backends, system=platform.system())
        self.progress = progress or _noop
        registered = {b.name for b in self.backends}
        self.reference: Backend | None = (
            get_backend(config.reference_backend)
            if config.reference_backend in registered
            else None
        )

    # ---- public --------------------------------------------------------
    def discover(self) -> Registry:
        return discover(self.cfg.test_model_dir)

    def select_graphs(self, reg: Registry) -> list[GraphRecord]:
        graphs = [r for r in reg.benchmarkable if self.cfg.selects_model(r.key)]
        if self.cfg.smoke > 0:
            graphs.sort(key=lambda r: self._onnx_size(r))
            graphs = graphs[: self.cfg.smoke]
        return graphs

    def _warn_unmatched_filters(self, reg: Registry, graphs: list[GraphRecord]) -> None:
        """A typo in --models/--modes otherwise yields a silent, green, empty run
        that is easily mistaken for "nothing to benchmark". Surface it explicitly."""
        if self.cfg.models is not None and not graphs:
            self.progress(
                f"warning: --models {list(self.cfg.models)} matched 0 of "
                f"{len(reg.benchmarkable)} benchmarkable graph(s)"
            )
        if self.cfg.modes is not None:
            known = {m.id for b in self.backends for m in b.modes}
            unknown = [s for s in self.cfg.modes if s not in known]
            if unknown:
                self.progress(
                    f"warning: --modes {unknown} match no mode of any selected backend "
                    f"(known modes: {sorted(known)})"
                )

    def run(self) -> list[RunResult]:
        reg = self.discover()
        graphs = self.select_graphs(reg)
        self._warn_unmatched_filters(reg, graphs)
        planned = self._planned_modes()
        total = sum(self._planned_count(record) for record in graphs)
        self.progress(
            f"{len(graphs)} graph(s); {total} run(s) total; results -> {self.cfg.results_dir}"
        )

        results: list[RunResult] = []
        done = 0
        for gi, record in enumerate(graphs, 1):
            available = [b for b in self.backends if b.fmt in record.sources]
            ref = self.reference if (self.reference and self.reference.fmt in record.sources) else None
            state = self._ensure_feed(record, available, ref)

            # The feed worker (reference or feed-source) has already run; surface
            # its result and never re-run it as a consumer below.
            if state.result is not None:
                done += 1
                results.append(state.result)
                if state.feed_by is not None:
                    self.progress(
                        self._fmt(f"[{done}/{total}] {gi}/{len(graphs)}", record,
                                  state.feed_by, state.feed_by.modes[0], state.result)
                    )

            # A regenerated feed invalidates this graph's cached backend results.
            force_runs = self.cfg.force or state.feed_changed
            # Feed worker already ran feed_by.modes[0]; skip only that mode so a
            # multi-mode feed source still benchmarks its remaining modes.
            feed_by_name = state.feed_by.name if state.feed_by is not None else None
            feed_mode_id = state.feed_by.modes[0].id if state.feed_by is not None else None
            for backend in available:
                for mode_id in planned[backend.name]:
                    if backend.name == feed_by_name and mode_id == feed_mode_id:
                        continue
                    done += 1
                    rr = self._run_one(
                        record, backend, backend.mode(mode_id), state,
                        prefix=f"[{done}/{total}] {gi}/{len(graphs)}",
                        force=force_runs,
                    )
                    results.append(rr)
        return results

    # ---- internals -----------------------------------------------------
    def _planned_modes(self) -> dict[str, list[str]]:
        sel = self.cfg.modes
        out: dict[str, list[str]] = {}
        for b in self.backends:
            ids = [m.id for m in b.modes if sel is None or m.id in sel]
            out[b.name] = ids
        return out

    def _planned_count(self, record: GraphRecord) -> int:
        """How many runs this graph will record (the feed worker runs once; the
        remaining available backends run their planned modes)."""
        planned = self._planned_modes()
        available = [b for b in self.backends if b.fmt in record.sources]
        ref = self.reference if (self.reference and self.reference.fmt in record.sources) else None
        feed_by = self._feed_backend(available, ref)
        feed_by_name = feed_by.name if feed_by is not None else None
        feed_mode_id = feed_by.modes[0].id if feed_by is not None else None
        # Feed worker runs feed_by.modes[0] once (n=1); every other planned mode,
        # including the feed backend's remaining modes, is a separate consumer run.
        n = 0 if feed_by is None else 1
        for b in available:
            for mode_id in planned[b.name]:
                if b.name == feed_by_name and mode_id == feed_mode_id:
                    continue
                n += 1
        return n

    @staticmethod
    def _feed_backend(available: list[Backend], ref: Backend | None) -> Backend | None:
        if ref is not None:
            return ref
        return next((b for b in available if b.provides_feed), None)

    def _onnx_size(self, record: GraphRecord) -> int:
        try:
            return Path(record.source("onnx").path).stat().st_size
        except (OSError, KeyError):
            return 0

    def _cache_dir(self, key: str) -> Path:
        return self.cfg.cache_dir / key

    def _result_path(self, key: str, backend: str, mode_id: str) -> Path:
        return self.cfg.runs_dir / key / f"{backend}__{mode_id}.json"

    def _common_meta(self) -> dict:
        return dict(
            warmup=self.cfg.warmup,
            min_iters=self.cfg.min_iters,
            max_iters=self.cfg.max_iters,
            time_budget_s=self.cfg.time_budget_s,
            input_samples=self.cfg.input_samples,
        )

    def _feed_fingerprint(self) -> dict:
        """The config parameters that determine the generated feed + reference
        outputs.

        When any of these change, a cached feed is stale: regenerating only the
        feed would leave it incomparable to backend runs cached against the old
        feed, so a change forces a full re-run of the affected graph.
        """
        return {
            "seed": self.cfg.seed,
            "dynamic_dim_size": self.cfg.dynamic_dim_size,
            "ort_disable_optimizations": self.cfg.ort_disable_optimizations,
            "reference_backend": self.cfg.reference_backend,
            "input_samples": self.cfg.input_samples,
        }

    @staticmethod
    def _path_signature(path: str | Path) -> str:
        """Cheap content-identity of a model file or bundle directory: total
        byte size + latest mtime. Detects a re-exported model of the same
        canonical name without hashing potentially-gigabyte weights. ``.mlpackage``
        and ``.aimodel`` are directories, so their contents are walked (the dir's
        own mtime does not change when a nested weight file is rewritten)."""
        p = Path(path)
        try:
            if p.is_dir():
                total = 0
                latest = 0
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        st = f.stat()
                        total += st.st_size
                        latest = max(latest, st.st_mtime_ns)
                return f"dir:{total}:{latest}"
            st = p.stat()
            return f"file:{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return f"missing:{p}"  # path-qualified so distinct missing models differ

    def _graph_fingerprint(self, feed_source_path: str | Path) -> dict:
        """Feed fingerprint plus the identity of the model that generates the
        feed/baseline, so re-exporting that model invalidates the cached feed."""
        return {**self._feed_fingerprint(), "feed_model": self._path_signature(feed_source_path)}

    @staticmethod
    def _fingerprint_stale(read: dict, current: dict) -> bool:
        """Whether a cached feed fingerprint is stale vs the current one. Only the
        keys the cached fingerprint actually recorded are compared: a key added in
        a later version (e.g. ``input_samples``, ``feed_model``) is not, by its
        mere absence, a reason to re-run an existing cache, which would be a
        surprise mass re-run on upgrade. A genuine change to a recorded key is
        still detected."""
        return any(read[k] != current.get(k) for k in read)

    @staticmethod
    def _cached_is_fresh(rr: RunResult, current_sig: str) -> bool:
        """Whether a cached consumer run may be reused: stale if its model file
        changed since. A legacy result without a recorded signature is treated as
        fresh (no surprise mass re-run on upgrade)."""
        return rr.model_sig is None or rr.model_sig == current_sig

    @staticmethod
    def _load_result(result_path: Path) -> RunResult | None:
        """Load a cached run result, or None if it is missing or unreadable.

        The run file is written atomically (``RunResult.save`` = tmp + replace),
        but a hand-edit, a full disk, or a kill outside that window can still
        leave a corrupt JSON, or one whose shape no longer matches the dataclasses
        (a truncated object, or a field added by a newer version -> ``TypeError``).
        Treat any such file as a cache miss (regenerate) rather than letting the
        load abort the whole resumable sweep — the same policy ``report/_collect``
        already applies when reading results for a report."""
        try:
            return RunResult.load(result_path)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @classmethod
    def _feed_committed(cls, result_path: Path) -> bool:
        """Whether the cached feed/baseline for a graph is safe to reuse.

        The atomically-written run result is the commit marker: a feed worker
        killed mid-write leaves its shared ``inputs.npz``/``baseline_outputs.npz``
        possibly truncated or mixed-generation, but its run result is then absent
        or ``STATUS_FAILED``. Reusing the cache only when the result loaded and is
        ``STATUS_OK`` prevents pairing fresh inputs with a stale baseline (silently
        wrong accuracy) or loading a half-written npz."""
        rr = cls._load_result(result_path)
        return rr is not None and rr.status == STATUS_OK

    @staticmethod
    def _remove_partial_feed(*paths: Path) -> None:
        """Defense-in-depth: drop a feed/baseline cache whose worker failed, so a
        partial/mixed-generation set can never be resurrected by later code."""
        for p in paths:
            with contextlib.suppress(OSError):
                p.unlink()

    def _load_reference_cache(self, meta_path: Path, outputs_npz: Path, result_path: Path):
        """Load a committed reference feed cache (names + baseline outputs + run
        result); return None if any file is unreadable so the graph regenerates."""
        try:
            in_names, out_names = self._load_names(meta_path)
            outputs = npio.load_named(outputs_npz, out_names)
        except (OSError, ValueError, KeyError):
            return None
        rr = self._load_result(result_path)
        if rr is None:
            return None
        return in_names, out_names, outputs, rr

    @staticmethod
    def _await_thermal_recovery(read_speed, sleep, *, min_speed, max_wait_s, poll_s) -> float:
        """Block until the CPU speed limit recovers to ``min_speed`` or
        ``max_wait_s`` elapses; return seconds waited. ``read_speed``/``sleep`` are
        injected so the bounded loop is unit-tested without real hardware. An
        unknown speed (None: off-Darwin / pmset missing) never blocks."""
        waited = 0.0
        step = poll_s if poll_s > 0 else max_wait_s  # non-positive poll can't loop forever
        while waited < max_wait_s:
            speed = read_speed()
            if speed is None or speed >= min_speed:
                return waited
            sleep(poll_s)
            waited += step
        return waited

    @staticmethod
    def _read_cpu_speed():
        from . import environment
        return environment._capture_power_thermal()[2]

    def _worker_qos(self) -> str | None:
        """The QoS class to launch workers under (Darwin only); None elsewhere
        so a misconfigured ``worker_qos`` cannot reach ``taskpolicy`` off-macOS."""
        return self.cfg.worker_qos if platform.system() == "Darwin" else None

    def _maybe_cooldown(self) -> None:
        """Pause before a run until the SoC is unthrottled, if thermal gating is
        enabled. No-op by default."""
        if not self.cfg.thermal_gate:
            return
        waited = self._await_thermal_recovery(
            self._read_cpu_speed, time.sleep,
            min_speed=self.cfg.thermal_min_speed,
            max_wait_s=self.cfg.thermal_max_wait_s,
            poll_s=self.cfg.thermal_poll_s,
        )
        if waited > 0:
            self.progress(f"  thermal: waited {waited:.0f}s for CPU speed ≥ {self.cfg.thermal_min_speed}%")

    @staticmethod
    def _read_fingerprint(path: Path) -> dict:
        if not path.exists():
            return dict(_LEGACY_FINGERPRINT)
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return dict(_LEGACY_FINGERPRINT)

    def _ensure_feed(
        self, record: GraphRecord, available: list[Backend], ref: Backend | None
    ) -> _FeedState:
        """Guarantee the shared feed (and reference outputs when a reference is
        available) exist; run the feed worker if needed.

        With a reference: it runs in ``generate_feed`` mode, writing
        ``inputs.npz`` + ``baseline_outputs.npz`` + ``baseline_meta.json``.
        Without one: a ``provides_feed`` backend runs in ``generate_feed`` mode,
        writing ``inputs.npz`` + ``input_names.json`` only (latency-only). If
        neither exists the graph is skipped. A cache produced with different feed
        parameters is treated as stale and regenerated (``feed_changed=True``).
        """
        feed_by = self._feed_backend(available, ref)
        if feed_by is None:
            return _FeedState(
                None, None, None, False, None, None,
                skip_note="no feed source: no reference and no backend can generate inputs",
            )

        cache = self._cache_dir(record.key)
        inputs_npz = cache / "inputs.npz"
        fp_path = cache / FEED_FINGERPRINT
        feed_mode = feed_by.modes[0]
        result_path = self._result_path(record.key, feed_by.name, feed_mode.id)

        # The feed + reference outputs depend on the feed-generating model's bytes,
        # so a re-exported model of the same canonical name must invalidate them.
        current_fp = self._graph_fingerprint(record.source(feed_by.fmt).path)

        if ref is not None:
            meta_path = cache / "baseline_meta.json"
            outputs_npz = cache / "baseline_outputs.npz"
            have_cache = (
                meta_path.exists() and inputs_npz.exists()
                and outputs_npz.exists() and self._feed_committed(result_path)
            )
            feed_stale = have_cache and self._fingerprint_stale(self._read_fingerprint(fp_path), current_fp)
            if have_cache and not self.cfg.force and not feed_stale:
                loaded = self._load_reference_cache(meta_path, outputs_npz, result_path)
                if loaded is not None:
                    in_names, out_names, outputs, rr = loaded
                    return _FeedState(in_names, out_names, outputs, True, rr, feed_by)
            return self._run_reference(record, ref, feed_mode, cache, result_path,
                                       fp_path, current_fp, feed_stale)

        # No reference: a feed-capable backend generates inputs (latency-only).
        names_json = cache / "input_names.json"
        have_cache = names_json.exists() and inputs_npz.exists() and self._feed_committed(result_path)
        feed_stale = have_cache and self._fingerprint_stale(self._read_fingerprint(fp_path), current_fp)
        if have_cache and not self.cfg.force and not feed_stale:
            rr = self._load_result(result_path)
            try:
                in_names = json.loads(names_json.read_text())
            except (OSError, ValueError):
                in_names = None
            if rr is not None and in_names is not None:
                return _FeedState(in_names, None, None, True, rr, feed_by)
        return self._run_feed_source(record, feed_by, feed_mode, cache, names_json,
                                     result_path, fp_path, current_fp, feed_stale)

    def _run_reference(
        self, record, ref, mode, cache, result_path, fp_path, current_fp, feed_stale
    ) -> _FeedState:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mbx_ref_") as jd:
            meta = dict(
                generate_feed=True,
                shared_inputs_npz=str(cache / "inputs.npz"),
                shared_input_names_json=str(cache / "input_names.json"),
                seed=self.cfg.seed,
                model_path=record.source("onnx").path,
                onnx_member=record.source("onnx").member,
                cache_dir=str(cache),
                onnx_extract_dir=str(self.cfg.cache_dir / "_onnx_src" / record.model),
                dynamic_dim_size=self.cfg.dynamic_dim_size,
                ort_disable_optimizations=self.cfg.ort_disable_optimizations,
                **self._common_meta(),
            )
            P.write_meta(jd, meta)
            self._maybe_cooldown()
            t0 = time.time()
            outcome = P.execute(
                self.cfg.python_executable, ref.worker_module, jd, self.cfg.worker_timeout_s,
                qos=self._worker_qos(),
            )
            rr = self._result_from_outcome(
                record, ref, mode, outcome,
                duration_s=time.time() - t0, baseline_outputs=None, jobdir=Path(jd),
            )
        rr.save(result_path)
        if not outcome.ok:
            self._remove_partial_feed(
                cache / "inputs.npz", cache / "baseline_outputs.npz",
                cache / "baseline_meta.json", cache / "input_names.json", fp_path,
            )
            return _FeedState(None, None, None, False, rr, ref, feed_changed=feed_stale)
        npio.write_text_atomic(fp_path, json.dumps(current_fp, sort_keys=True))
        in_names, out_names = self._load_names(cache / "baseline_meta.json")
        outputs = npio.load_named(cache / "baseline_outputs.npz", out_names)
        return _FeedState(in_names, out_names, outputs, True, rr, ref, feed_changed=feed_stale)

    def _run_feed_source(
        self, record, feed_by, mode, cache, names_json, result_path, fp_path, current_fp, feed_stale
    ) -> _FeedState:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mbx_feed_") as jd:
            meta = dict(
                generate_feed=True,
                shared_inputs_npz=str(cache / "inputs.npz"),
                shared_input_names_json=str(names_json),
                seed=self.cfg.seed,
                model_path=record.source(feed_by.fmt).path,
                mode=mode.id,
                **self._common_meta(),
            )
            P.write_meta(jd, meta)
            self._maybe_cooldown()
            t0 = time.time()
            outcome = P.execute(
                self.cfg.python_executable, feed_by.worker_module, jd, self.cfg.worker_timeout_s,
                qos=self._worker_qos(),
            )
            rr = self._result_from_outcome(
                record, feed_by, mode, outcome,
                duration_s=time.time() - t0, baseline_outputs=None, jobdir=Path(jd),
            )
        rr.save(result_path)
        if not outcome.ok:
            self._remove_partial_feed(cache / "inputs.npz", names_json, fp_path)
            return _FeedState(None, None, None, False, rr, feed_by, feed_changed=feed_stale)
        npio.write_text_atomic(fp_path, json.dumps(current_fp, sort_keys=True))
        in_names = json.loads(names_json.read_text())
        return _FeedState(in_names, None, None, True, rr, feed_by, feed_changed=feed_stale)

    def _run_one(
        self,
        record: GraphRecord,
        backend: Backend,
        mode,
        state: _FeedState,
        prefix: str,
        force: bool = False,
    ) -> RunResult:
        result_path = self._result_path(record.key, backend.name, mode.id)
        model_sig = self._path_signature(record.source(backend.fmt).path)

        # Resumable: reuse a cached result unless forced (globally via --force or
        # because this graph's feed was regenerated), or the backend's own model
        # file changed since the cached run (a re-export of the same canonical
        # name). Otherwise we would report stale numbers for a different model.
        if result_path.exists() and not force:
            rr = self._load_result(result_path)
            # A cached SKIPPED result is never reused: a skip records only that no
            # feed was available at the time (e.g. this graph had no reference
            # backend yet), a precondition external to this (graph, backend, mode)
            # that the cache key does not capture. Re-attempt it on every resume so
            # adding the reference later actually benchmarks the backend instead of
            # leaving it permanently skipped until --force.
            if rr is not None and rr.status != STATUS_SKIPPED and self._cached_is_fresh(rr, model_sig):
                self.progress(f"{prefix} {record.key} {backend.name}/{mode.id}: cached ({rr.status})")
                return rr

        if not state.ok or state.input_names is None:
            rr = self._skipped(record, backend, mode, state.skip_note)
            rr.save(result_path)
            self.progress(self._fmt(prefix, record, backend, mode, rr))
            return rr

        with tempfile.TemporaryDirectory(prefix="mbx_job_") as jd:
            meta = dict(
                inputs_npz=str(self._cache_dir(record.key) / "inputs.npz"),
                input_names=state.input_names,
                output_names=state.output_names,
                model_path=record.source(backend.fmt).path,
                mode=mode.id,
                entrypoint=backend.entrypoint,
                **self._common_meta(),
            )
            P.write_meta(jd, meta)
            self._maybe_cooldown()
            t0 = time.time()
            outcome = P.execute(
                self.cfg.python_executable, backend.worker_module, jd, self.cfg.worker_timeout_s,
                qos=self._worker_qos(),
            )
            rr = self._result_from_outcome(
                record, backend, mode, outcome,
                duration_s=time.time() - t0, baseline_outputs=state.outputs,
                jobdir=Path(jd),
            )
        rr.save(result_path)
        self.progress(self._fmt(prefix, record, backend, mode, rr))
        return rr

    def _result_from_outcome(
        self, record, backend, mode, outcome, *, duration_s, baseline_outputs, jobdir
    ) -> RunResult:
        rr = RunResult(
            graph_key=record.key,
            model=record.model,
            component=record.component,
            backend=backend.name,
            fmt=f".{backend.fmt}" if backend.fmt != "onnx" else ".onnx",
            mode_id=mode.id,
            mode_label=mode.label,
            precision=mode.precision,
            status=STATUS_OK,
            model_path=record.source(backend.fmt).path,
            is_baseline=backend.is_baseline,
            iters_requested=self.cfg.max_iters,
            warmup_requested=self.cfg.warmup,
            timestamp=time.time(),
            duration_s=round(duration_s, 3),
            worker_returncode=outcome.returncode,
            model_sig=self._path_signature(record.source(backend.fmt).path),
        )
        if not outcome.ok:
            rr.status = STATUS_FAILED
            rr.note = outcome.error or "unknown failure"
            return rr

        res = outcome.result or {}
        rr.realized_device = res.get("realized_device")
        rr.note = res.get("note", "")
        rr.timing = tmetrics.summarize(
            res["raw_ms"],
            load_ms=res.get("load_ms"),
            first_call_ms=res.get("first_call_ms"),
            keep_raw=self.cfg.keep_raw_timings,
        )
        # Accuracy is attached only when a reference produced outputs and this
        # backend is not the reference itself; otherwise the run is latency-only.
        is_reference = self.reference is not None and backend.name == self.reference.name
        if baseline_outputs is not None and not is_reference:
            produced = res["produced_output_names"]
            got = npio.load_named(jobdir / P.OUTPUTS, produced)
            rr.accuracy = acc.compare_outputs(baseline_outputs, got)
        return rr

    def _skipped(self, record, backend, mode, note) -> RunResult:
        return RunResult(
            graph_key=record.key, model=record.model, component=record.component,
            backend=backend.name, fmt=f".{backend.fmt}" if backend.fmt != "onnx" else ".onnx",
            mode_id=mode.id, mode_label=mode.label, precision=mode.precision,
            status=STATUS_SKIPPED, model_path=record.source(backend.fmt).path,
            is_baseline=backend.is_baseline, note=note, timestamp=time.time(),
        )

    @staticmethod
    def _load_names(meta_path: Path) -> tuple[list[str], list[str]]:
        m = json.loads(meta_path.read_text())
        return m["input_names"], m["output_names"]

    @staticmethod
    def _fmt(prefix, record, backend, mode, rr: RunResult) -> str:
        if rr.status == STATUS_OK and rr.timing is not None:
            extra = f"mean={rr.timing.mean_ms:.3f}ms"
            if rr.accuracy is not None:
                extra += f" PSNR={rr.accuracy.min_psnr_db:.1f}dB"
            else:
                extra += " (latency-only)"
            return f"{prefix} {record.key} {backend.name}/{mode.id}: ok ({extra})"
        return f"{prefix} {record.key} {backend.name}/{mode.id}: {rr.status} ({rr.note[:60]})"

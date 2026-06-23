# ModelBenchX Design

A universal, extensible benchmarking framework for on-device inference runtimes.
It measures latency and numerical accuracy of the same model exported to
several formats, and emits formal reports that label each framework's running
mode (compute unit + precision) explicitly.

This document records the design and the decisions behind it. It is the source
of truth for why the code is shaped the way it is.

---

## 1. Goals

1. Benchmark four targets initially: ONNX Runtime (CPU FP32), Core ML
   `.mlpackage` (ML Program), Core ML `.mlmodel` (NeuralNetwork), and
   Core AI `.aimodel` (Apple Core AI runtime, via the `axon`/`coreai` stack).
2. Be extensible: adding a framework is one `Backend` subclass plus
   one worker module.
3. Compute accuracy of every backend against the ONNX Runtime FP32
   baseline.
4. Collect trustworthy performance data (warmup, steady-state sampling,
   percentiles, isolated runs).
5. Produce formal reports (Markdown + JSON + CSV) that label each run's
   mode and precision.

Non-goals: training, quantization, model conversion/repair. ModelBenchX never
modifies the models it measures.

---

## 2. Environment facts that shape the design

Discovered empirically on the target machine (macOS 27, Apple Silicon,
Python 3.11):

- `coremltools` and `onnx` cannot be imported in the same process. Doing so
  aborts the process with `SIGABRT` (a protobuf C++ symbol clash). Verified.
- The Core AI runtime can `abort()` (not raise) when it cannot execute a
  program on the selected compute unit, an abort no `try/except` can catch.
- Therefore each backend runs in its own subprocess, importing only its own
  runtime. The orchestrator imports neither `onnx` nor `coremltools`; it is
  pure `numpy` + stdlib. This sidesteps the clash and contains crashes: one
  model that aborts cannot take down the run.

Other facts:

- ONNX models are distributed as `…-onnx-float.zip`, each extracting to a folder
  containing one or more `.onnx` graphs (+ external `.data` weights,
  `metadata.json`, optional `labels.txt`/`.npy`). A zip may hold multiple graphs
  (e.g. `sam2` → `encoder.onnx`, `decoder.onnx`).
- `metadata.json` carries input/output names, shapes, dtypes, `value_range`, and
  `io_type`, used to generate valid, deterministic inputs.
- Core ML and Core AI generally preserve ONNX I/O names; Core ML may sanitize
  them, so name matching uses sanitized-name + positional fallback.

---

## 3. The canonical unit: a "graph"

A benchmarkable unit is a single `(model, component)` pair, keyed
`"<model>__<component>"`:

| Format | On-disk name | Canonical key |
|---|---|---|
| ONNX | `sam2-onnx-float.zip` → `encoder.onnx` | `sam2__encoder` |
| Core ML mlpackage | `sam2__encoder.mlpackage` | `sam2__encoder` |
| Core ML mlmodel | `sam2__encoder.mlmodel` | `sam2__encoder` |
| Core AI | `sam2-onnx-float__encoder.aimodel` | `sam2__encoder` |
| TFLite | `sam2__encoder.tflite` | `sam2__encoder` |

Discovery is a union: a graph is benchmarkable if it is present in at least one
registered format. The registry records which formats each graph has (its
`sources`); for a given graph only the backends whose format is present run, and
the report shows per-format coverage. Whether a graph yields accuracy numbers
depends on a reference (or feed-capable) backend being present for it (see the
accuracy model and worker-protocol sections).

Naming standardization is logical, not physical: the registry maps each
canonical key to its file in each format and uses that key everywhere. Source
files are never renamed (they are outputs of other tools; renaming could break
them).

---

## 4. Backend / mode matrix

Each backend declares the modes it supports. A "mode" is a compute-unit choice;
the report labels it with its precision.

| Backend | Format | Precision | Modes |
|---|---|---|---|
| `onnxruntime` | `.onnx` | FP32 | `cpu` *(accuracy baseline)* |
| `coreml-mlpackage` | `.mlpackage` (ML Program) | FP16 | `cpu_only`, `cpu_and_gpu`, `all` (ANE+GPU+CPU) |
| `coreml-mlmodel` | `.mlmodel` (NeuralNetwork) | FP16 | `cpu_only`, `cpu_and_gpu`, `all` (ANE+GPU+CPU) |
| `coreai` | `.aimodel` | FP32 (cpu) / FP16 (gpu, ane) | `cpu_only`, `gpu`, `ane`, `all` (auto) |
| `tflite` | `.tflite` | FP32 | `cpu`, `xnnpack` |

ONNX Runtime and TFLite are cross-platform; Core ML and Core AI are macOS-only
and are skipped on other systems. A graph present in every format runs up to 13
modes (1 + 3 + 3 + 4 + 2); a graph in fewer formats runs proportionally fewer.
Configurable per backend/mode; runs are resumable (each result is cached on disk).

Caveat recorded in the report: requesting `all`/`ane` selects a preferred
unit; the OS may still place ops elsewhere. ModelBenchX records the requested
mode. Where the framework exposes the realized placement, it is captured too.

---

## 5. Measurement methodology

- **Isolation & fairness:** runs execute serially, one subprocess at a time,
  so they do not contend for CPU/GPU/ANE.
- **Load vs steady-state:** model load/compile time is measured separately from
  inference. The first calls (compilation, lazy specialization) are warmup
  and discarded.
- **Sampling:** after `warmup` iterations, time `iters` iterations with
  `time.perf_counter_ns`. Default is adaptive: stop at ~3 s of measurement or
  `iters` (default 50), whichever comes first, with a floor of 10 iterations.
- **Stats:** mean, median, std, min, max, p90, p95, p99, throughput (1/mean).
- **Inputs:** one deterministic seeded input set per graph (honoring dtype,
  metadata shape, and `value_range`) is reused across all backends so accuracy
  is comparable and timing uses a fixed, representative feed. `--input-samples N`
  (default 1) rotates N distinct seeded inputs through the timed loop so latency
  reflects more than one point for data-dependent graphs; accuracy always uses
  sample 0, so the reference comparison is unchanged.
- **Environment captured:** chip, core counts, macOS build, and the version of
  every runtime, plus the live power/thermal state (power source, Low Power
  Mode, and the `pmset` CPU speed limit). A throttled or battery-powered host is
  flagged with a latency caveat in the report. `--thermal-gate` optionally pauses
  before each run until the SoC's CPU speed recovers, so a long serial sweep does
  not record throttled latencies; `--worker-qos <class>` is a best-effort
  `taskpolicy` hint for scheduling consistency (macOS QoS, not a hard CPU pin).

### ONNX Runtime configuration

ORT plays a dual role (perf sample and accuracy baseline). It runs in a
single, clearly-labeled mode: CPU, FP32, graph optimizations disabled +
KleidiAI disabled. Rationale: a benchmark must call "baseline" the same
outputs it reports, and `coreai-onnx` documents that ORT's graph optimizations
produce wrong outputs on macOS arm64 for several of these exact models
(grouped Conv via FusedConv, KleidiAI SGEMM on pruned weights, BASIC constant
folding on `rf_detr`). Correctness of the ground truth outweighs ORT's realistic
speed. The optimization level is configurable for users who want the realistic
ORT number instead.

### Per-call dispatch overhead (why we report min-latency and a floor)

Latencies are end-to-end times of one inference through each framework's Python
entry point (`session.run`, `MLModel.predict`, `await fn(feed)`). These entry
points carry different fixed per-call dispatch overheads, which a controlled
micro-benchmark on a tiny model (squeezenet) isolates: Core ML `predict` floors
at ~0.18 ms, while Core AI's async `await fn(feed)` floors at ~1.0 ms. That is a
fixed cost of the binding's command submit/synchronize round-trip, not asyncio
scheduling (~0.012 ms) or input boxing. This floor
dominates sub-millisecond models, so a 3-5× latency ratio there reflects binding
overhead, not engine speed; a residual ~2-2.5× remains for compute-heavy models
(Core AI runtime/compiler maturity vs Apple's production Core ML; Core AI's
`.aimodel` is fused and natively `optimize()`-d at conversion, runs on-ANE, and
matches ONNX accuracy, so this is not a conversion defect). The report therefore
surfaces min-latency and a best-case floor alongside the mean, and the
methodology section states this caveat. Core AI's Swift `axon` zero-allocation
IOBinding path (not exposed to Python) would remove the floor but is not
measurable from this harness.

Output materialization is timed for every backend. Core ML's `predict` and
ONNX Runtime's `session.run` already return host (numpy) arrays, so their
device→host copy is inside the timed call. Core AI returns lazy device handles,
so the worker's `materialize()` hook (a `.numpy()` host copy) runs inside the
timed region too; otherwise the copy would be deferred to output capture
(outside timing) and understate Core AI latency on large outputs. The hook is a
no-op for backends that already return host arrays, so this only equalizes the
async path.

---

## 6. Accuracy

For each non-baseline run, compare its outputs to the ORT FP32 baseline on the
identical input. Per output tensor, then aggregated to the worst case per run:

- PSNR (dB): primary, matches the existing conversion report.
- Max absolute error, max relative error, RMSE, MAE, cosine similarity.
- NaN/Inf handled by masking: non-finite positions are compared by identity;
  error metrics describe the finite-overlap region only (so one NaN does not
  poison the whole tensor). The count of reference non-finite values is reported.
  A non-finite mismatch, though (e.g. the candidate emits `NaN` where the
  baseline is finite) is a real divergence, not benign noise, so it is scored as
  a hard failure (PSNR −∞) rather than allowed to read as bit-exact from the
  matching finite overlap.
- A hard failure (−∞) is never hidden in the aggregate: the per-column median is
  taken over the comparable graphs, but any dropped −∞ graphs are counted and
  shown next to the median (`… (N✗)`), and a column with any hard failure is not
  eligible to be reported as the "most accurate" mode.
- Non-finite metric values (±∞ sentinels, NaN) are serialized in `report.json`
  and the per-run JSON as the JSON-safe string tokens `"Infinity"`/`"-Infinity"`/
  `"NaN"` (revived on load), so the output is valid JSON for strict parsers while
  preserving the bit-exact-vs-failure distinction.
- Zero dynamic-range baseline: PSNR is defined only when the baseline output has
  a nonzero peak (max |value| over its finite region), and relative error is only
  defined where the baseline is nonzero — so max relative error is taken over those
  positions (a near-zero element does not inflate it to ∞). When a baseline output
  is entirely zero, a candidate that matches it is still bit-exact (+∞), but a
  candidate that deviates is a hard failure (PSNR −∞, max relative error ∞): the
  deviation is real yet unquantifiable as a ratio against a zero signal. The
  absolute metrics (max abs error, RMSE, MAE) stay meaningful. Consequence: a
  graph whose reference output is identically zero reads as a hard failure against
  any non-bit-exact candidate. This is intentional — a zero reference cannot anchor
  a relative-accuracy claim — but if such outputs are expected, read them through
  the absolute-error columns rather than PSNR.

Outputs are matched to baseline by sanitized name, with positional fallback.

---

## 7. Architecture

```
modelbenchx/
  cli.py            # `modelbenchx discover|run|report`
  config.py         # BenchmarkConfig, run params, mode selection
  naming.py         # canonical-key parsing/normalization
  registry.py       # discover formats, normalize keys, build the benchmarkable union
  environment.py    # capture host + runtime versions (subprocess/stdlib only)
  results.py        # result dataclasses <-> JSON (numpy-free)
  orchestrator.py   # serial driver; baseline-first; resumable cache; accuracy
  metrics/
    timing.py       # latency stats (numpy)
    accuracy.py     # PSNR / errors / cosine vs baseline (numpy)
  backends/
    base.py         # Backend dataclass + the declarative BACKENDS registry
                    # (onnxruntime, coreml-mlpackage, coreml-mlmodel, coreai, tflite)
  workers/          # each runs in its own process, only its deps
    _protocol.py    # job/result file contract + crash/abort interpretation (pure)
    _harness.py     # worker harness: load/timing loop + parent<->worker protocol
    _bench.py       # steady-state sampling policy (sync + async)
    _io.py          # npz helpers, 32-bit narrowing (numpy only)
    _feedgen.py     # seeded sample generation from input specs (numpy only)
    _inputs.py      # ONNX input-spec extraction (imports onnx; worker-side only)
    onnx_worker.py  # extract zip, gen inputs, ORT baseline + timing
    coreml_worker.py
    coreai_worker.py
    tflite_worker.py  # TFLite (cross-platform); template for a new backend
    synth_worker.py   # synthetic numpy backend (tests only)
  report/
    _collect.py     # load cached run JSONs; column spec for the matrices
    markdown.py     # formal report
    json_report.py
    csv_report.py
```

### Worker protocol (uniform across backends)

The orchestrator creates a job directory and runs
`python -m modelbenchx.workers.<x> <jobdir>`:

| File | Direction | Contents |
|---|---|---|
| `meta.json` | parent → | model path, mode, entrypoint, I/O names, warmup/iters |
| `inputs.npz` | parent → | feed arrays (positional; names in meta) |
| `result.json` | → parent | timings, mode, precision, realized device, status |
| `outputs.npz` | → parent | output arrays (positional; names in meta), for accuracy |
| `error.json` | → parent | a handled exception `{type, message}` |

The worker writes `result.json` (+`outputs.npz`) XOR `error.json`. Any other
outcome (non-zero exit, death by signal, missing files) is interpreted by the
parent as a native crash/abort and recorded as a failed run with a note.
This interpreter is pure and unit-tested without the native runtimes.

The onnx worker runs first per graph: it extracts the zip, generates the
shared `inputs.npz`, runs the ORT baseline, and saves baseline `outputs.npz`.
Core ML / Core AI workers then consume the shared `inputs.npz`. This keeps
`onnx` and `coremltools` in separate processes and gives every backend the
identical feed.

---

## 8. Resumability & failure handling

- Each `(graph, backend, mode)` result is written to
  `results/runs/<graph>/<backend>__<mode>.json`. Re-running skips completed runs
  unless `--force`. A crash loses only the in-flight run.
- Model-file identity is tracked (a cheap size+mtime signature, walked for
  `.mlpackage`/`.aimodel` bundles). The shared feed/baseline is invalidated when
  the feed-generating model changes, and an individual cached run is re-measured
  when its model file is re-exported under the same canonical name, so a
  resumed run never silently reports numbers for a model that has since changed.
  (Caches written before this tracking existed are treated as fresh, so upgrading
  does not trigger a surprise mass re-run.)
- Per-graph shared artifacts (inputs, baseline outputs, extracted onnx) are
  cached under `results/cache/<graph>/`.
- Inference failures (load error, runtime exception, native abort, accuracy
  comparison failure) are recorded as `status=failed` with a note. Models are
  never modified or "repaired."
- The final report is regenerated purely from the on-disk result JSONs, so it
  can be produced at any time, including after a partial run.

---

## 9. Extensibility: adding a framework

See [`docs/adding-a-framework.md`](adding-a-framework.md) for the full
contributor guide.  The key contracts are summarised below.

### Backend declaration (`backends/base.py`)

Add a `Backend` entry to the `BACKENDS` tuple:

- `name`, `fmt`, `kind`, `worker_module`, `modes`, `label`: identity fields.
- `discovery: FormatSpec(suffix, key_fn, archive_member_suffix=None)` tells
  the registry how to find model files on disk.  Set `archive_member_suffix`
  only for zip-archived formats (ONNX uses `.zip`/`.onnx`).
- `platforms: tuple[str, ...] | None`: `None` = all OSes; `("Darwin",)` =
  macOS only.  `select_backends` automatically skips unsupported platforms
  before any run, so Apple-only backends are silent no-ops on Linux/x86.
- `provides_feed: bool`: `True` when the worker implements `input_spec()` and
  can generate the shared feed without an ONNX reference.  Required for
  latency-only benchmarking of a standalone model.

### Worker module (`workers/<fw>_worker.py`)

Subclass `Worker` (sync) or `AsyncWorker` (async runtimes):

| Method | Required | Notes |
|---|---|---|
| `load(meta)` | yes | Load/compile the model; import the runtime lazily here. |
| `infer(feed)` | yes | One forward pass; returns raw outputs. |
| `output_names()` | yes | Names in the same order as `infer()` output. |
| `build_feed(shared, meta)` | no | Adapt the shared canonical feed: rename, recast, or transpose (e.g. NCHW→NHWC). Default passes `shared` through unchanged. |
| `input_spec()` | no* | Required when `provides_feed=True`. Returns `list[InputSpec]`. |
| `extract_outputs(last)` | no | Default zips `output_names()` with the infer result. |
| `realized_device()` | no | Actual compute unit at runtime, if the framework exposes it. |

End the module with `sys.exit(run_worker(sys.argv[1], <FW>Worker()))` (or
`arun_worker` for async).  The runtime must be imported lazily (inside
`load` or a helper) so the module is importable without the runtime installed.

### What the harness provides

Union discovery: a graph enters the matrix as soon as its file is present
in any registered format; no longer a strict N-way intersection.

Serial + platform-gated scheduling: runs never contend for CPU/GPU/ANE;
backends outside the current OS's `platforms` tuple are silently skipped.

Steady-state timing: adaptive warmup convergence followed by a timed
window with nanosecond resolution, producing mean/median/std/min/max/p90-p99.

Accuracy: computed per graph only when the configurable `reference_backend`
(default `onnxruntime`) is present for that graph.  When absent, a
`provides_feed`-capable backend generates the shared input and the graph's runs
are latency-only (accuracy shows `n/a`).  If neither exists the graph is skipped.

Resumability: each `(graph, backend, mode)` result is cached on disk;
re-runs skip completed entries unless `--force`.

Reports: Markdown, JSON, and CSV regenerated at any time from the cache.

### Known limitation: `coreai` without an ONNX twin

The `coreai` worker derives its expected output names from the reference
backend's metadata.  A `.aimodel` graph with no ONNX twin (so no reference) is
currently recorded as a failure because output names are unavailable.  In
practice this is non-blocking: `coreai` models are converted from ONNX and
should ship with their ONNX twin.  See the contributor guide for details.

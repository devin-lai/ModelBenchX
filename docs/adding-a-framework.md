# Adding a Framework in 3 Steps

ModelBenchX is designed so adding a new inference runtime costs one `Backend`
declaration, one worker module, and one `pyproject.toml` line.  Everything
else is inherited automatically: union discovery, serial + platform-gated
scheduling, steady-state timing, accuracy comparison, resumable on-disk caching,
and Markdown/JSON/CSV reports.

The TFLite backend (`src/modelbenchx/workers/tflite_worker.py`) is the canonical
worked example and is referenced throughout this guide.

---

## Step 1: Declare a `Backend` in `backends/base.py`

Open `src/modelbenchx/backends/base.py` and add an entry to the `BACKENDS`
tuple.  Every field is mandatory except those with defaults.  `Backend`,
`FormatSpec`, `Mode`, and `FP32` are already in scope inside that file, so no
imports are needed.

```python
Backend(
    name="tflite",           # unique backend id used in CLI, result filenames, reports
    fmt="tflite",            # registry source key — matches what GraphRecord.source() returns
    kind="tflite",           # worker family (used for internal grouping)
    worker_module="modelbenchx.workers.tflite_worker",
    modes=(Mode("cpu", "CPU", FP32), Mode("xnnpack", "XNNPACK", FP32)),
    label="TFLite",          # human label in reports
    discovery=FormatSpec(
        suffix=".tflite",
        key_fn=lambda n: n[:-7] if n.endswith(".tflite") else n,
        # archive_member_suffix omitted: each graph is its own file, not zip-archived.
        # For zip-archived formats (like ONNX) pass e.g. archive_member_suffix=".onnx"
    ),
    platforms=None,          # None = all OSes; ("Darwin",) = macOS-only, auto-skipped elsewhere
    provides_feed=True,      # True when the worker implements input_spec() and can generate
                             # the shared feed — this lets a standalone .tflite file be
                             # benchmarked latency-only without an ONNX twin.
)
```

### Field reference

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Unique across all backends. Used in `--backends`, result filenames, reports. |
| `fmt` | `str` | Registry source key the orchestrator uses to look up model files. |
| `kind` | `str` | Worker family (`onnx`, `coreml`, `coreai`, `tflite`, …). |
| `worker_module` | `str` | Importable module path; executed as a subprocess. |
| `modes` | `tuple[Mode, ...]` | Compute-unit choices. Each `Mode(id, label, precision)`. |
| `label` | `str` | Human-readable framework name for reports. |
| `discovery` | `FormatSpec` | Tells the registry how to find model files on disk (see below). |
| `platforms` | `tuple[str, ...] \| None` | `None` = all OSes. `("Darwin",)` = macOS only; backends not supported on the current OS are silently skipped before any run. |
| `provides_feed` | `bool` | `True` when the worker implements `input_spec()`. Required for latency-only benchmarking with no ONNX reference. |
| `is_baseline` | `bool` | Default `False`. Only the `onnxruntime` backend sets this. |

### `FormatSpec`

```python
@dataclass(frozen=True)
class FormatSpec:
    suffix: str                               # file extension, e.g. ".tflite"
    key_fn: Callable[[str], str]              # filename -> canonical graph key
    archive_member_suffix: str | None = None  # set to ".onnx" for zip-archived formats
```

`archive_member_suffix` is only needed when a single archive file (`.zip`)
contains multiple graph files that are individually benchmarked (the ONNX
format uses this).  For most runtimes (one file per graph), omit it.

---

## Step 2: Write `workers/<fw>_worker.py`

Create `src/modelbenchx/workers/tflite_worker.py` (or your framework's
equivalent).  The module must:

1. Import only `numpy` and the shared harness at the top level.
2. Import the runtime lazily (inside `load` or a helper) so the module can
   be imported even when the runtime is not installed.
3. Subclass `Worker` (sync) or `AsyncWorker` (async runtimes like Core AI).
4. End with `sys.exit(run_worker(sys.argv[1], <FW>Worker()))`.

```python
from __future__ import annotations
import sys
import numpy as np
from ._feedgen import InputSpec
from ._harness import Worker, run_worker   # AsyncWorker + arun_worker for async


def _load_runtime(model_path: str, mode: str):
    """Lazy import — keeps the module importable without the runtime installed."""
    try:
        from ai_edge_litert.interpreter import Interpreter   # preferred
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    return Interpreter(model_path=model_path, num_threads=1)


class TFLiteWorker(Worker):

    # ------------------------------------------------------------------
    # load  (required)
    # Called once. meta["model_path"] and meta["mode"] are always present.
    # ------------------------------------------------------------------
    def load(self, meta: dict) -> None:
        self._itp = _load_runtime(meta["model_path"], meta["mode"])
        self._itp.allocate_tensors()
        self._in = self._itp.get_input_details()
        self._out = self._itp.get_output_details()

    # ------------------------------------------------------------------
    # input_spec  (optional — required only when provides_feed=True)
    # Returns a list of InputSpec describing this model's inputs so the
    # harness can generate a concrete feed without an ONNX twin.
    # ------------------------------------------------------------------
    def input_spec(self) -> list[InputSpec]:
        specs = []
        for d in self._in:
            shape = tuple(1 if x == -1 else int(x) for x in d["shape"])  # TFLite dynamic dim == -1; other runtimes may use 0 — adjust
            specs.append(InputSpec(d["name"], shape, np.dtype(d["dtype"])))
        return specs

    # ------------------------------------------------------------------
    # build_feed  (optional — default passes shared dict through unchanged)
    # Adapt the canonical shared feed to whatever infer() expects.
    # This is the canonical place to handle:
    #   - name mapping (shared names may differ from model's input names)
    #   - dtype casting
    #   - layout transposition (e.g. NCHW → NHWC for TFLite)
    # ------------------------------------------------------------------
    def build_feed(self, shared: dict | None, meta: dict) -> dict:
        """Map shared canonical feed to model inputs.

        Inputs are matched positionally (shared[i] → model_input[i]).
        dtype is cast to the dtype TFLite reports for each tensor.
        Layout: the shared feed may be NCHW (from an ONNX reference worker)
        while TFLite expects NHWC.  For a 4-D input whose shape matches only
        after an NCHW→NHWC transpose, np.transpose(arr, (0, 2, 3, 1)) is
        applied.  When TFLite itself is the feed source the arrays already
        match — no transpose is performed.
        """
        shared_names = list(shared or {})
        feed: dict[int, np.ndarray] = {}
        for det, name in zip(self._in, shared_names, strict=False):
            arr = np.asarray(shared[name]).astype(det["dtype"])
            expected_shape = tuple(int(x) if x > 0 else 1 for x in det["shape"])
            if arr.ndim == 4 and arr.shape != expected_shape:
                transposed = arr.transpose(0, 2, 3, 1)   # NCHW → NHWC
                if transposed.shape == expected_shape:
                    arr = transposed
            feed[det["index"]] = arr
        return feed

    # ------------------------------------------------------------------
    # infer  (required)
    # Receives whatever build_feed returned.  Returns any object that
    # extract_outputs can consume (default: zip with output_names()).
    # ------------------------------------------------------------------
    def infer(self, feed: dict) -> list[np.ndarray]:
        for index, arr in feed.items():
            self._itp.set_tensor(index, arr)
        self._itp.invoke()
        return [self._itp.get_tensor(o["index"]) for o in self._out]

    # ------------------------------------------------------------------
    # output_names  (required)
    # Must return names in the same order as the list returned by infer().
    # ------------------------------------------------------------------
    def output_names(self) -> list[str]:
        return [o["name"] for o in self._out]


if __name__ == "__main__":
    sys.exit(run_worker(sys.argv[1], TFLiteWorker()))
```

### Full `Worker` / `AsyncWorker` contract

These are the methods the harness calls and the ones you may override:

| Method | Required | Signature | Notes |
|---|---|---|---|
| `load` | yes | `(self, meta: dict) -> None` | Load/compile the model. `meta["model_path"]`, `meta["mode"]` always present. |
| `build_feed` | no | `(self, shared: dict \| None, meta: dict) -> Any` | Default returns `shared` unchanged. Override to rename, recast, or transpose. |
| `infer` | yes | `(self, feed: Any) -> Any` | Run one inference; return raw outputs. |
| `output_names` | yes | `(self) -> list[str]` | Names in the same order as `infer()` output. |
| `extract_outputs` | no | `(self, last: Any) -> dict[str, Any]` | Default zips `output_names()` with the list returned by `infer()`. Override when `infer()` returns a dict or needs post-processing. |
| `input_spec` | no* | `(self) -> list[InputSpec] \| None` | Required when `provides_feed=True`. Return `None` (default) to opt out. |
| `realized_device` | no | `(self) -> str \| None` | Return the actual compute unit chosen at runtime, if the framework exposes it. |

For async runtimes subclass `AsyncWorker` instead:

- `infer` becomes `async def infer(self, feed: Any) -> Any`.
- Add `async def aclose(self) -> None` for teardown (called in a `finally`
  block by the harness).
- Use `arun_worker` instead of `run_worker`:

  ```python
  if __name__ == "__main__":
      sys.exit(arun_worker(sys.argv[1], MyAsyncWorker()))
  ```

### About `InputSpec`

```python
@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    value_range: tuple[float, float] | None = None   # optional; used by generate_from_spec
    io_type: str | None = None                       # "image" → uniform [0,1] instead of N(0,1)
```

`generate_from_spec` uses `value_range` and `io_type` to produce sensible
random data.  If your model's metadata exposes these, pass them through; if not,
leave them `None` and the generator falls back to standard normal (floats) or
small integers.

---

## Step 3: Add the runtime as an optional extra in `pyproject.toml`

The runtime is imported only inside the worker subprocess, so it belongs in
`[project.optional-dependencies]`, not in `dependencies`:

```toml
[project.optional-dependencies]
tflite = ["ai-edge-litert"]   # alternates: tflite-runtime or tensorflow
```

Install with:

```sh
pip install -e ".[tflite]"
```

The orchestrator and all other workers never import this package.

---

## What you get for free

Once Steps 1-3 are done, the following require no additional work:

- Union discovery: the registry finds your format's files alongside ONNX,
  Core ML, etc., and creates a `GraphRecord` for each canonical key.  A graph
  appears in the run matrix as soon as its file is present; it is no longer a
  strict 4-way intersection.
- Serial + platform-gated scheduling: `select_backends` filters out backends
  whose `platforms` tuple does not include the current OS (via `platform.system()`).
  Your Apple-only backend is silently skipped on Linux/x86 without code changes.
- Steady-state timing harness: adaptive warmup convergence, then
  `min_iters`/`max_iters` timing window with nanosecond resolution, producing
  mean/median/std/min/max/p90/p95/p99/throughput.
- Accuracy comparison: when the configured `reference_backend` (default
  `onnxruntime`) is present for a graph, its outputs are compared to yours
  per-tensor (PSNR, max abs/rel error, RMSE, MAE, cosine similarity).  When no
  reference is present the run is latency-only and accuracy shows `n/a`.
- Resumable on-disk caching: each `(graph, backend, mode)` result is written
  to `results/runs/<graph>/<backend>__<mode>.json`.  Re-running skips completed
  entries unless `--force`.
- Markdown / JSON / CSV reports: regenerated at any time from cached results.

---

## Accuracy model

Accuracy is computed per graph only when the configured `reference_backend`
(default `onnxruntime`) is available for that graph.  When the reference is
absent, any backend that has `provides_feed=True` generates the shared input
feed and the graph's runs are latency-only: timing is measured but no
accuracy numbers are produced.

If neither a reference nor a feed-capable backend is available for a graph, the
graph is skipped entirely.

---

## Cross-platform note

The benchmark core (orchestrator, registry, reporting, `onnxruntime` and TFLite
workers) runs on Linux, macOS, and x86.  Apple-only backends (`coreml-mlpackage`,
`coreml-mlmodel`, `coreai`) declare `platforms=("Darwin",)` and are
auto-skipped off-macOS, with no code changes needed in the orchestrator.

---

## Known limitation: `coreai` without an ONNX twin

The `coreai` worker is a consumer: it derives its expected output names from
the reference backend's metadata.  If a graph has a `.aimodel` file but no ONNX
twin (so no reference ran), the `coreai` run is recorded as a failure because
the output names are unavailable.

In practice this is non-blocking: `coreai` models are converted from ONNX and
should ship with their ONNX twin.  Any runtime that implements `output_names()`
from its own model metadata (as TFLite does) is not affected.

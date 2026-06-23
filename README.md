# ModelBenchX

A universal, extensible benchmarking framework for on-device inference runtimes.
It measures latency and numerical accuracy of the same model exported to
several formats and emits formal reports that label each framework's running mode
(compute unit + precision) explicitly.

Backends:

| Backend | Format | Precision | Modes |
|---|---|---|---|
| ONNX Runtime | `.onnx` | FP32 | `cpu` *(accuracy baseline)* |
| Core ML | `.mlpackage` (ML Program) | FP16 | `cpu_only`, `cpu_and_gpu`, `all` (ANE+GPU+CPU) |
| Core ML | `.mlmodel` (Neural Network) | FP16 | `cpu_only`, `cpu_and_gpu`, `all` (ANE+GPU+CPU) |
| Core AI | `.aimodel` | FP32 (cpu) / FP16 (gpu, ane) | `cpu_only`, `gpu`, `ane`, `all` (auto) |
| TFLite | `.tflite` | FP32 | `cpu`, `xnnpack` |

ONNX Runtime and TFLite are cross-platform; Core ML and Core AI are macOS-only.
Accuracy is computed against the ONNX Runtime FP32 baseline. TFLite also serves
as the worked example in the contributor guide.

## Why subprocess workers

Two hard environment facts shape the architecture:

1. Importing `coremltools` and `onnx` into the same process aborts it
   (`SIGABRT`, a protobuf symbol clash).
2. The Core AI runtime can `abort()` the process (not raise) on a compute unit
   it cannot execute.

Each backend runs in its own subprocess importing only its own runtime,
and the orchestrator stays pure `numpy` + stdlib. A model that aborts is recorded
as a failed run; it cannot take down the benchmark. See `docs/design.md`.

## Install

```sh
pip install -e .            # core (numpy only)
# the runtimes (onnxruntime / coremltools / coreai) are expected in the env;
# each is imported only inside its own worker.
```

## Usage

```sh
# List the graphs that will be benchmarked, with per-format coverage.
modelbenchx discover --test-model test_model

# Quick end-to-end pipeline check on the 3 smallest graphs.
modelbenchx run --test-model test_model --smoke 3

# Full matrix (resumable; safe to re-run — completed runs are reused).
modelbenchx run --test-model test_model

# Useful filters.
modelbenchx run --backends coreai,coreml-mlpackage --modes cpu_only,all
modelbenchx run --models resnet50,sam2 --force

# Fidelity options (all optional; defaults preserve current behavior).
modelbenchx run --input-samples 4   # rotate 4 inputs through timing (accuracy uses sample 0)
modelbenchx run --thermal-gate      # pause before each run until the SoC is unthrottled

# Regenerate reports from cached results without re-running.
modelbenchx report --results results
```

Outputs land in `results/`:

```
results/
  runs/<graph>/<backend>__<mode>.json   # one cached result per run (resumable)
  cache/<graph>/                         # shared seeded feed + ONNX baseline outputs
  cache/_onnx_src/<model>/               # extracted ONNX (shared across components)
  reports/report.md | report.json | report.csv
```

`report.md` is the formal human report (environment, methodology, mode legend,
aggregate, latency matrix, accuracy matrix, failures, and skipped runs). `report.json`
and `report.csv` carry the full per-run / per-output data for further analysis.

## Methodology (accuracy of the numbers)

- Runs are serial and isolated, so they never contend for CPU/GPU/ANE.
- Model load/compile is timed separately; warmup calls are discarded; then a
  steady-state window is timed with a nanosecond clock and reduced to
  mean/median/std/min/max/p90/p95/p99/throughput.
- One seeded deterministic input per graph is reused by every backend, so
  latency and accuracy use an identical feed.
- Accuracy = worst-case PSNR (dB) across a graph's outputs vs the ONNX FP32
  baseline, plus max abs/rel error, RMSE, MAE and cosine similarity (NaN/Inf-safe).
  A NaN-where-baseline-is-finite divergence scores as a hard failure (−∞), and
  hard failures are never hidden inside an aggregate median.
- Output host-materialization is timed for every backend (Core AI's `.numpy()`
  copy runs inside the timed window, like Core ML's `predict`), so latencies are
  like-for-like.
- The report captures host power/thermal state (power source, Low Power Mode,
  CPU speed limit) and flags throttled/battery runs, since those depress latency.
- Mode labels report the requested compute unit; ANE/auto is a preference, not
  a guarantee of placement.

## Extending: add a framework

Backends are pluggable; the benchmark core is cross-platform (Apple-specific
backends auto-skip off-macOS).  See [`docs/adding-a-framework.md`](docs/adding-a-framework.md)
for the step-by-step contributor guide (TFLite is the worked example).

In brief:

1. Add a `Backend` entry in `src/modelbenchx/backends/base.py`: declare
   `name`, `fmt`, `kind`, `worker_module`, `modes`, `label`, a `FormatSpec`
   for discovery, `platforms` (`None` = all OSes; `("Darwin",)` = macOS only),
   and `provides_feed=True` if the worker can generate the shared input feed.
2. Add `src/modelbenchx/workers/<fw>_worker.py` subclassing `Worker` (or
   `AsyncWorker`); implement `load`, `build_feed` (for layout/dtype
   adaptation), `infer`, and `output_names`; import the runtime lazily.
3. Add the runtime as an optional extra in `pyproject.toml`.

Discovery, scheduling, steady-state timing, accuracy comparison (when an ONNX
reference is present), resumability, and reporting are all inherited.

## Tests

```sh
python -m pytest
```

The model files are never modified. Inference failures (load error, runtime
exception, native abort) are recorded as notes; nothing is repaired.

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).

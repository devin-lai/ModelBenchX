"""TFLite worker (template for adding a new inference framework).

This module imports only numpy + the shared harness at the top level; the tflite
runtime is loaded lazily so the module can be imported even when no tflite runtime
is installed.

Run as: ``python -m modelbenchx.workers.tflite_worker <jobdir>``
"""

from __future__ import annotations

import sys

import numpy as np

from ._feedgen import InputSpec
from ._harness import Worker, run_worker


def _interpreter(model_path: str, mode: str):
    """Return an Interpreter instance, trying runtimes in preference order.

    Lazy import: keeps the module importable when no tflite runtime is installed.
    Preferred: ai_edge_litert (LiteRT successor to tflite_runtime).
    Fallback 1: tflite_runtime (standalone wheel, lighter than full TF).
    Fallback 2: tensorflow.lite (ships with a full TensorFlow install).
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # preferred
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[no-redef]
    # `mode` is currently CPU-only; reserved for future delegate wiring
    # (e.g. XNNPACK via experimental_delegates). Kept in the signature so the
    # backend's modes map cleanly onto worker behavior later.
    return Interpreter(model_path=model_path, num_threads=1)


class TFLiteWorker(Worker):
    """TFLite inference worker. Canonical template for adding a new backend.

    Key design points (mirrored in the contributor guide):
    - Lazy runtime import in ``_interpreter()`` above.
    - ``input_spec`` resolves dynamic dims (-1 or 0) to 1 so the worker can act
      as a feed source (``provides_feed=True`` in the backend registration).
    - ``build_feed`` is the canonical example of per-worker feed adaptation:
      it maps shared canonical tensors to this model's inputs by position and
      transposes NCHW→NHWC for 4-D inputs when the shared feed came from an
      ONNX source (which uses NCHW layout).
    """

    def load(self, meta: dict) -> None:
        self._itp = _interpreter(meta["model_path"], meta["mode"])
        self._itp.allocate_tensors()
        self._in = self._itp.get_input_details()
        self._out = self._itp.get_output_details()

    def input_spec(self) -> list[InputSpec]:
        """Derive InputSpec from TFLite's input details.

        Dynamic dims (reported as -1 or 0 by TFLite) are resolved to 1 so
        generate_from_spec produces concrete-shaped arrays and this worker can
        serve as the feed source for a latency-only benchmark.
        """
        specs = []
        for d in self._in:
            shape = tuple(1 if x in (-1, 0) else int(x) for x in d["shape"])  # dynamic dim: -1 or 0
            specs.append(InputSpec(d["name"], shape, np.dtype(d["dtype"])))
        return specs

    def build_feed(self, shared: dict | None, meta: dict) -> dict:
        """Map the shared canonical feed to this model's inputs.

        This is the canonical example of per-worker feed adaptation:
        - Inputs are matched positionally (shared name[i] → model input[i]).
        - dtype is cast to whatever TFLite reports for each input tensor.
        - Layout: the shared feed may be NCHW (produced by an ONNX reference
          worker) while TFLite expects NHWC. For a 4-D input whose shared shape
          (N,C,H,W) does not match the expected shape but matches after an
          NCHW→NHWC transpose, we apply np.transpose(arr, (0, 2, 3, 1)).
          When TFLite is itself the feed source the feed already matches;
          no transpose is needed.
        """
        shared_names = list(shared or {})
        # Inputs are mapped positionally; fewer shared arrays than model inputs
        # would silently leave the trailing inputs at their allocate_tensors()
        # default (zeros) and report meaningless numbers as a successful run.
        # Fail loud instead — recorded as one failed run, never silently wrong.
        if len(shared_names) < len(self._in):
            raise ValueError(
                f"shared feed has {len(shared_names)} array(s) but the TFLite model "
                f"needs {len(self._in)} input(s); cannot map positionally"
            )
        feed: dict[int, np.ndarray] = {}
        for det, name in zip(self._in, shared_names, strict=False):
            arr = np.asarray(shared[name]).astype(det["dtype"])  # type: ignore[index]
            expected_shape = tuple(int(x) if x > 0 else 1 for x in det["shape"])
            if arr.ndim == 4 and arr.shape != expected_shape:
                # Transpose NCHW → NHWC if that fixes the shape mismatch.
                # This is the canonical example of per-worker layout adaptation.
                transposed = arr.transpose(0, 2, 3, 1)
                if transposed.shape == expected_shape:
                    arr = transposed
                    # NOTE: this fires when a 4-D input matches the model's expected shape only
                    # after an NCHW->NHWC transpose (the usual ONNX-reference→TFLite case). For a
                    # square tensor (C==H==W) the two layouts are indistinguishable by shape; if
                    # your models need that disambiguated, pass an explicit layout hint via meta.
            feed[det["index"]] = arr
        return feed

    def infer(self, feed: dict) -> list[np.ndarray]:
        for index, arr in feed.items():
            self._itp.set_tensor(index, arr)
        self._itp.invoke()
        return [self._itp.get_tensor(o["index"]) for o in self._out]

    def output_names(self) -> list[str]:
        return [o["name"] for o in self._out]


if __name__ == "__main__":
    sys.exit(run_worker(sys.argv[1], TFLiteWorker()))

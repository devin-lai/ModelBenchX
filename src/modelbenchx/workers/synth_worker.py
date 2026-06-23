"""Synthetic numpy 'framework' worker (testing the extension contract dep-free)."""
import sys

import numpy as np

from ._feedgen import InputSpec
from ._harness import Worker, run_worker


class SynthWorker(Worker):
    def load(self, meta):
        self._w = np.load(meta["model_path"])["w"]

    def input_spec(self):
        return [InputSpec("x", (1, self._w.shape[0]), np.dtype(np.float32))]

    def build_feed(self, shared, meta):
        return {"x": np.asarray(shared["x"], dtype=np.float32)}

    def infer(self, feed):
        return [feed["x"] @ self._w]

    def output_names(self):
        return ["y"]


if __name__ == "__main__":
    sys.exit(run_worker(sys.argv[1], SynthWorker()))

"""ModelBenchX: universal benchmarking for on-device inference runtimes.

The public surface stays import-light: nothing here pulls in ``onnx``,
``coremltools`` or ``coreai``. Those runtimes are imported only inside their
own subprocess workers (see ``modelbenchx.workers``), because importing
``onnx`` and ``coremltools`` into the same process aborts it.
"""

__version__ = "0.1.0"

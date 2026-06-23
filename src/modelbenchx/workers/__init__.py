"""Subprocess inference workers.

Each worker module is executed as ``python -m modelbenchx.workers.<x> <jobdir>``
in its own process and imports only its own runtime (plus numpy). They must not
be imported by the orchestrator.
"""

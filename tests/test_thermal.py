"""Thermal gating: the cooldown loop is pure and injectable (no real hardware).

The orchestrator can optionally pause before a run until the SoC's CPU speed
limit recovers, so a long serial sweep does not measure throttled latencies.
"""

from modelbenchx.orchestrator import Orchestrator


def test_await_thermal_recovery_waits_until_recovered():
    speeds = iter([80, 90, 100, 100])
    slept: list[float] = []
    waited = Orchestrator._await_thermal_recovery(
        lambda: next(speeds), slept.append,
        min_speed=100, max_wait_s=60.0, poll_s=5.0)
    assert waited == 10.0       # blocked through 80, 90; 100 releases it
    assert slept == [5.0, 5.0]


def test_await_thermal_recovery_gives_up_at_max_wait():
    slept: list[float] = []
    waited = Orchestrator._await_thermal_recovery(
        lambda: 50, slept.append,
        min_speed=100, max_wait_s=12.0, poll_s=5.0)
    assert waited >= 12.0       # bounded: never loops forever
    assert sum(slept) >= 10.0


def test_await_thermal_recovery_noops_when_speed_unavailable():
    waited = Orchestrator._await_thermal_recovery(
        lambda: None, lambda s: None,     # None = off-Darwin / pmset missing
        min_speed=100, max_wait_s=60.0, poll_s=5.0)
    assert waited == 0.0


def test_await_thermal_recovery_poll_zero_terminates():
    # A misconfigured poll interval of 0 must not loop forever.
    slept: list[float] = []
    waited = Orchestrator._await_thermal_recovery(
        lambda: 50, slept.append, min_speed=100, max_wait_s=10.0, poll_s=0.0)
    assert waited >= 10.0

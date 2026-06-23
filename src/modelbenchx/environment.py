"""Capture host and runtime-version info for the report header.

Versions are read via ``importlib.metadata`` (which reads package metadata
without importing the package), so this stays safe to call from the
orchestrator: it never imports ``onnx``/``coremltools``/``coreai`` itself.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _pkg_version(*candidates: str) -> str | None:
    for name in candidates:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _pmset(*args: str) -> str | None:
    try:
        out = subprocess.run(["pmset", *args], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _parse_cpu_speed_limit(therm_text: str) -> int | None:
    """The CPU speed clamp (%) from ``pmset -g therm``; < 100 means the SoC is
    being throttled (thermal or power), which directly depresses latency."""
    for line in therm_text.splitlines():
        if "CPU_Speed_Limit" in line and "=" in line:
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _parse_power_source(batt_text: str) -> str | None:
    """The active power source (e.g. ``AC Power`` / ``Battery Power``) from
    ``pmset -g batt``. Battery power can downclock Apple Silicon."""
    for line in batt_text.splitlines():
        if "drawing from" in line:
            parts = line.split("'")
            if len(parts) >= 2:
                return parts[1]
    return None


def _parse_low_power_mode(pmset_text: str) -> bool | None:
    """Low Power Mode state from ``pmset -g`` (it caps clocks)."""
    for line in pmset_text.splitlines():
        if "lowpowermode" in line:
            tok = line.split()
            if tok and tok[-1] in ("0", "1"):
                return tok[-1] == "1"
    return None


def _capture_power_thermal() -> tuple[str | None, bool | None, int | None]:
    """(power_source, low_power_mode, cpu_speed_limit) on Darwin; Nones elsewhere
    or when ``pmset`` is unavailable. Captured because a benchmark on a throttled
    or battery-powered laptop reports latencies that do not reflect the SoC's
    sustained capability."""
    batt = _pmset("-g", "batt")
    pm = _pmset("-g")
    therm = _pmset("-g", "therm")
    return (
        _parse_power_source(batt) if batt else None,
        _parse_low_power_mode(pm) if pm else None,
        _parse_cpu_speed_limit(therm) if therm else None,
    )


@dataclass
class Environment:
    os: str
    os_version: str
    machine: str
    chip: str | None
    cpu_cores: str | None
    performance_cores: str | None
    efficiency_cores: str | None
    memory_gb: float | None
    python_version: str
    tool_version: str | None
    runtime_versions: dict[str, str | None] = field(default_factory=dict)
    # Power/thermal state at capture time (Darwin; None elsewhere). Latency is
    # only trustworthy when the SoC is unthrottled (cpu_speed_limit == 100) and
    # not in Low Power Mode; battery power can also downclock.
    power_source: str | None = None
    low_power_mode: bool | None = None
    cpu_speed_limit: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _runtime_versions() -> dict[str, str | None]:
    return {
        "onnxruntime": _pkg_version("onnxruntime", "onnxruntime-silicon"),
        "onnx": _pkg_version("onnx"),
        "coremltools": _pkg_version("coremltools"),
        "coreai": _pkg_version("coreai", "coreai-torch"),
        "tflite": _pkg_version("tflite-runtime", "tensorflow-lite", "ai-edge-litert"),
        "tensorflow": _pkg_version("tensorflow", "tensorflow-cpu", "tensorflow-macos"),
        "numpy": _pkg_version("numpy"),
    }


def _capture_darwin() -> tuple[str | None, str | None, str | None, str | None, float | None]:
    """Return (chip, cpu_cores, performance_cores, efficiency_cores, memory_gb) via sysctl."""
    mem = _sysctl("hw.memsize")
    mem_gb = round(int(mem) / (1024**3), 1) if mem and mem.isdigit() else None
    return (
        _sysctl("machdep.cpu.brand_string"),
        _sysctl("hw.ncpu"),
        _sysctl("hw.perflevel0.logicalcpu"),
        _sysctl("hw.perflevel1.logicalcpu"),
        mem_gb,
    )


def _capture_linux() -> tuple[str | None, str | None, str | None, str | None, float | None]:
    """Return (chip, cpu_cores, performance_cores, efficiency_cores, memory_gb) from /proc."""
    chip: str | None = None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    chip = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    mem_gb: float | None = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    mem_gb = round(kb / (1024**2), 1)
                    break
    except OSError:
        pass

    cpu_count = os.cpu_count()
    cpu_cores = str(cpu_count) if cpu_count else None

    return chip, cpu_cores, None, None, mem_gb


def capture() -> Environment:
    system = platform.system()
    if system == "Darwin":
        chip, cpu_cores, p_cores, e_cores, mem_gb = _capture_darwin()
        os_version = platform.mac_ver()[0] or platform.release()
    elif system == "Linux":
        chip, cpu_cores, p_cores, e_cores, mem_gb = _capture_linux()
        os_version = platform.release()
    else:
        chip = None
        cpu_cores = str(os.cpu_count()) if os.cpu_count() else None
        p_cores = None
        e_cores = None
        mem_gb = None
        os_version = platform.release()

    power_source, low_power_mode, cpu_speed_limit = (
        _capture_power_thermal() if system == "Darwin" else (None, None, None)
    )
    return Environment(
        os=system,
        os_version=os_version,
        machine=platform.machine(),
        chip=chip,
        cpu_cores=cpu_cores,
        performance_cores=p_cores,
        efficiency_cores=e_cores,
        memory_gb=mem_gb,
        python_version=platform.python_version(),
        tool_version=_pkg_version("modelbenchx"),
        runtime_versions=_runtime_versions(),
        power_source=power_source,
        low_power_mode=low_power_mode,
        cpu_speed_limit=cpu_speed_limit,
    )

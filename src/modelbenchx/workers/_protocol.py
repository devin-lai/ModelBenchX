"""The parent<->worker file contract and crash interpretation.

A worker, run as ``python -m modelbenchx.workers.<x> <jobdir>``, reads
``meta.json`` (+ shared ``inputs.npz``) and writes either ``result.json`` +
``outputs.npz`` (exit 0) or ``error.json`` (exit 1, a handled exception). Any
other outcome is interpreted by the parent as a crash: death by signal (native
``abort()``), non-zero exit, missing files, or timeout. The crash interpreter
is pure and unit-tested without the native runtimes.
"""

from __future__ import annotations

import json
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

META = "meta.json"
INPUTS = "inputs.npz"
OUTPUTS = "outputs.npz"
RESULT = "result.json"
ERROR = "error.json"


@dataclass
class WorkerOutcome:
    ok: bool
    crashed: bool
    returncode: int
    result: dict | None = None
    error: str | None = None


# ---- worker side -----------------------------------------------------------

def read_meta(jobdir: str | Path) -> dict:
    return json.loads((Path(jobdir) / META).read_text())


def write_result(jobdir: str | Path, result: dict) -> None:
    (Path(jobdir) / RESULT).write_text(json.dumps(result))


def write_error(jobdir: str | Path, exc: BaseException) -> None:
    (Path(jobdir) / ERROR).write_text(
        json.dumps({"type": type(exc).__name__, "message": str(exc)})
    )


# ---- parent side -----------------------------------------------------------

def write_meta(jobdir: str | Path, meta: dict) -> None:
    Path(jobdir).mkdir(parents=True, exist_ok=True)
    (Path(jobdir) / META).write_text(json.dumps(meta))


def crash_message(returncode: int, stderr: str) -> str:
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    detail = f" ({lines[-1].strip()})" if lines else ""
    if returncode < 0:
        try:
            sig = signal.Signals(-returncode).name
        except ValueError:
            sig = f"signal {-returncode}"
        cause = f"native runtime aborted (killed by {sig})"
    else:
        cause = f"worker exited abnormally (exit code {returncode})"
    return f"{cause}{detail}"


def _load_json(path: Path) -> dict | None:
    """Read a protocol JSON file, returning None if it is missing or truncated.

    write_result/write_error are non-atomic, so a worker killed by a signal or a
    full disk mid-write can leave a partial file. A corrupt protocol file must be
    treated as a crash (one contained failed run), never raised — otherwise it
    would abort the whole sweep and defeat the subprocess-isolation design."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def interpret(returncode: int, stderr: str, jobdir: str | Path) -> WorkerOutcome:
    """Map a finished worker process to a structured outcome (pure)."""
    jobdir = Path(jobdir)
    if returncode == 0 and (jobdir / RESULT).exists():
        result = _load_json(jobdir / RESULT)
        if result is not None:
            return WorkerOutcome(ok=True, crashed=False, returncode=returncode, result=result)
    if (jobdir / ERROR).exists():
        info = _load_json(jobdir / ERROR)
        if info is not None:
            msg = f"{info.get('type', 'Error')}: {info.get('message', '')}".strip()
            return WorkerOutcome(ok=False, crashed=False, returncode=returncode, error=msg)
    return WorkerOutcome(
        ok=False, crashed=True, returncode=returncode,
        error=crash_message(returncode, stderr),
    )


def worker_command(
    python_executable: str, worker_module: str, jobdir: str | Path, qos: str | None = None
) -> list[str]:
    """The argv to launch a worker. When ``qos`` is set (Darwin only), wrap it in
    ``taskpolicy -c <qos>`` so the worker runs at a defined scheduling QoS for
    reproducibility (a hint toward consistent P/E-core placement, not a hard pin)."""
    cmd = [python_executable, "-m", worker_module, str(jobdir)]
    if qos:
        return ["taskpolicy", "-c", qos, *cmd]
    return cmd


def execute(
    python_executable: str,
    worker_module: str,
    jobdir: str | Path,
    timeout_s: float,
    *,
    qos: str | None = None,
) -> WorkerOutcome:
    """Run a worker subprocess to completion and interpret the result."""
    try:
        # The protocol is file-based; only stderr is consumed (for crash_message),
        # so worker stdout is discarded rather than buffered unbounded into the
        # parent. stderr stays piped so a native abort's last line is still shown.
        proc = subprocess.run(
            worker_command(python_executable, worker_module, jobdir, qos),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return WorkerOutcome(
            ok=False, crashed=True, returncode=-signal.SIGKILL,
            error=f"worker timed out after {timeout_s:.0f}s",
        )
    except OSError as exc:
        # The worker never launched (e.g. the interpreter or ``taskpolicy`` is
        # missing from PATH). Contain it as one failed run rather than letting it
        # abort the whole sweep — the same policy applied to a native crash.
        return WorkerOutcome(
            ok=False, crashed=True, returncode=-1,
            error=f"failed to launch worker ({worker_module}): {exc}",
        )
    return interpret(proc.returncode, proc.stderr, jobdir)

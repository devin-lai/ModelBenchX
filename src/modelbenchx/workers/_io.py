"""npz feed/result helpers and 32-bit narrowing. numpy only.

Arrays travel through ``.npz`` files **positionally** (keys ``arr_0``, ``arr_1``,
…) because ONNX tensor names (e.g. ``/enc/Add_output_0``) are not valid
``np.savez`` keyword keys. The ordered name list lives in ``meta.json`` and
re-keys them on the far side.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Core AI executes in a 32-bit world. Narrow wide host dtypes so the feed
# matches the asset's narrowed inputs (mirrors coreai-onnx's policy so the two
# never drift).
_NARROW = {
    np.dtype(np.int64): np.dtype(np.int32),
    np.dtype(np.uint64): np.dtype(np.uint32),
    np.dtype(np.float64): np.dtype(np.float32),
}


def narrow_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    target = _NARROW.get(a.dtype)
    return a.astype(target) if target is not None else a


def _atomic_savez(path: str | Path, arrays: Sequence[np.ndarray]) -> None:
    """Write a positional ``.npz`` atomically (temp file + ``os.replace``).

    These files land in the persistent cache and are read back on resume, so a
    worker killed mid-write (timeout SIGKILL, native ``abort()``) must never
    leave a truncated archive that a later run reads as valid. Writing to a
    sibling temp file and renaming makes the final path appear all-or-nothing.
    ``np.savez`` writes the zip straight to the file object, so (unlike the
    string-path form) it does not append a second ``.npz`` suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp.npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, *arrays)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def save_named(path: str | Path, names: Sequence[str], by_name: dict[str, np.ndarray]) -> None:
    """Save arrays positionally in the order given by ``names``."""
    _atomic_savez(path, [np.asarray(by_name[n]) for n in names])


def load_named(path: str | Path, names: Sequence[str]) -> dict[str, np.ndarray]:
    """Load a positional npz back into ``{name: array}`` using ``names`` order."""
    with np.load(str(path), allow_pickle=False) as data:
        return {n: np.array(data[f"arr_{i}"]) for i, n in enumerate(names)}


def save_samples(path: str | Path, names: Sequence[str], samples: Sequence[dict[str, np.ndarray]]) -> None:
    """Save K input samples positionally: sample ``s`` occupies ``arr_{s*len+i}``."""
    _atomic_savez(path, [np.asarray(sample[n]) for sample in samples for n in names])


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write a small text file (feed/baseline metadata, fingerprints) atomically,
    for the same reason as :func:`_atomic_savez`: a partial JSON left in the cache
    by an interrupted writer must not later be read as complete. Uses a unique
    temp file and cleans it up on error so a kill leaves no stray ``.tmp``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_samples(path: str | Path, names: Sequence[str], n_samples: int) -> list[dict[str, np.ndarray]]:
    """Inverse of :func:`save_samples`: load K samples in ``names`` order."""
    k = len(names)
    with np.load(str(path), allow_pickle=False) as data:
        return [
            {n: np.array(data[f"arr_{s * k + i}"]) for i, n in enumerate(names)}
            for s in range(n_samples)
        ]

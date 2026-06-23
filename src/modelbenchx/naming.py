"""Canonical naming for benchmarkable graphs.

A *graph* is one ``(model, component)`` pair. Every format names it slightly
differently on disk; this module normalizes all of them to a single canonical
key ``"<model>__<component>"`` so the four folders can be matched.

    ONNX zip          sam2-onnx-float.zip -> encoder.onnx   -> sam2__encoder
    Core ML mlpackage sam2__encoder.mlpackage               -> sam2__encoder
    Core ML mlmodel   sam2__encoder.mlmodel                 -> sam2__encoder
    Core AI           sam2-onnx-float__encoder.aimodel      -> sam2__encoder

This module is pure (stdlib only) and has no I/O, so it is cheaply unit-tested.
"""

from __future__ import annotations

import re

SEP = "__"
_ONNX_SUFFIX = "-onnx-float"


def canonical_key(model: str, component: str) -> str:
    """Join a model and component into the canonical key."""
    return f"{model}{SEP}{component}"


def split_key(key: str) -> tuple[str, str]:
    """Split a canonical key back into ``(model, component)``.

    The first ``__`` is the separator; a component may itself contain ``__``.
    """
    model, sep, component = key.partition(SEP)
    if not sep:
        raise ValueError(f"not a canonical key (missing '{SEP}'): {key!r}")
    return model, component


def model_from_onnx_zip(zip_name: str) -> str:
    """``squeezenet1_1-onnx-float.zip`` / ``…-onnx-float`` -> ``squeezenet1_1``."""
    name = zip_name
    if name.endswith(".zip"):
        name = name[:-4]
    if name.endswith(_ONNX_SUFFIX):
        name = name[: -len(_ONNX_SUFFIX)]
    return name


def onnx_member_key(model: str, onnx_member: str) -> str:
    """Key for a ``.onnx`` file inside a model's zip.

    ``onnx_member`` may be a bare name or a path inside the archive; only the
    file stem is used as the component (``encoder.onnx`` -> ``encoder``).
    """
    stem = onnx_member.rsplit("/", 1)[-1]
    if stem.endswith(".onnx"):
        stem = stem[: -len(".onnx")]
    return canonical_key(model, stem)


def key_from_coreml_filename(name: str) -> str:
    """``sam2__encoder.mlpackage`` / ``…​.mlmodel`` -> ``sam2__encoder``."""
    for ext in (".mlpackage", ".mlmodelc", ".mlmodel"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def key_from_aimodel_dirname(name: str) -> str:
    """``sam2-onnx-float__encoder.aimodel`` -> ``sam2__encoder``."""
    base = name[: -len(".aimodel")] if name.endswith(".aimodel") else name
    left, sep, right = base.partition(SEP)
    if not sep:
        return base
    if left.endswith(_ONNX_SUFFIX):
        left = left[: -len(_ONNX_SUFFIX)]
    return canonical_key(left, right)


_SANITIZE_RE = re.compile(r"[^0-9a-zA-Z_]")


def sanitize_feature_name(name: str) -> str:
    """Approximate the name munging frameworks apply to I/O feature names.

    Non-alphanumeric/underscore characters (e.g. ``/`` and ``.`` common in ONNX
    output names like ``/head/Add_output_0``) become ``_``; a leading digit is
    prefixed with ``_``. Used to match framework output names back to ONNX
    output names when they are not preserved verbatim.
    """
    s = _SANITIZE_RE.sub("_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s


def match_output_names(
    reference_names: list[str], actual_names: list[str]
) -> dict[str, str]:
    """Map each reference (ONNX) output name to an actual framework output name.

    Tries exact match, then sanitized-name equality, then positional fallback
    when the counts are equal. Returns ``{reference_name: actual_name}`` for the
    names that could be matched.
    """
    actual_set = set(actual_names)
    mapping: dict[str, str] = {}
    remaining_actual = list(actual_names)

    # Pass 1: exact.
    for ref in reference_names:
        if ref in actual_set:
            mapping[ref] = ref
            if ref in remaining_actual:
                remaining_actual.remove(ref)

    # Pass 2: sanitized equality among the not-yet-matched.
    san_to_actual: dict[str, str] = {}
    for a in remaining_actual:
        san_to_actual.setdefault(sanitize_feature_name(a), a)
    for ref in reference_names:
        if ref in mapping:
            continue
        cand = san_to_actual.get(sanitize_feature_name(ref))
        if cand is not None:
            mapping[ref] = cand
            remaining_actual.remove(cand)
            san_to_actual = {
                k: v for k, v in san_to_actual.items() if v != cand
            }

    # Pass 3: positional fallback when counts line up. Pair still-unmatched refs
    # with still-unused actuals; never re-grab an actual a prior pass claimed
    # (that would map a reference to the wrong tensor and orphan the right one).
    if len(mapping) < len(reference_names) and len(reference_names) == len(actual_names):
        used = set(mapping.values())
        leftover_actual = [a for a in actual_names if a not in used]
        unmatched_ref = [r for r in reference_names if r not in mapping]
        for ref, act in zip(unmatched_ref, leftover_actual, strict=False):
            mapping[ref] = act

    return mapping

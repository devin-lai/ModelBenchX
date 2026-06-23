"""Accuracy of a candidate output set against the ONNX Runtime baseline.

The element-level comparison mirrors the rules in ``coreai-onnx``:
promote to float64; mask non-finite positions so a single NaN does not poison a
whole tensor; report PSNR (dB) as the headline metric alongside absolute,
relative, RMSE/MAE and cosine similarity.
"""

from __future__ import annotations

import numpy as np

from .. import naming
from ..results import AccuracyStats, OutputAccuracy

# A PSNR value carries one of three meanings via float sentinels. Every reducer
# (aggregation, "most accurate", cell rendering) must agree on which is which,
# so the meaning is defined in exactly one place here rather than re-deriving
# `== inf` / `== -inf` checks at each call site (the source of repeated bugs).
PSNR_OK = "ok"        # a finite, comparable PSNR
PSNR_EXACT = "exact"  # +inf: bit-exact match
PSNR_FAIL = "fail"    # -inf (or NaN): a hard, incomparable failure


def classify_psnr(v: float) -> str:
    """Classify a PSNR value as PSNR_OK / PSNR_EXACT / PSNR_FAIL."""
    if v == float("inf"):
        return PSNR_EXACT
    if v == float("-inf") or v != v:  # -inf or NaN
        return PSNR_FAIL
    return PSNR_OK


def compare_tensor(name: str, expected: np.ndarray, got: np.ndarray) -> OutputAccuracy:
    """Compare one output tensor against its baseline reference."""
    e = np.asarray(expected)
    g = np.asarray(got)

    # Reconcile shape: accept a differently-shaped result only if the element
    # count matches (frameworks sometimes squeeze/keep singleton dims). A true
    # size mismatch is an incomparable output, flagged as a hard failure.
    if e.shape != g.shape:
        if e.size == g.size:
            g = g.reshape(e.shape)
        else:
            return OutputAccuracy(
                name=name,
                shape=list(e.shape),
                dtype=str(e.dtype),
                psnr_db=float("-inf"),
                max_abs_err=float("inf"),
                max_rel_err=float("inf"),
                rmse=float("inf"),
                mae=float("inf"),
                cosine=0.0,
                expected_nonfinite=0,
                nonfinite_match=False,
            )

    ef = e.astype(np.float64)
    gf = g.astype(np.float64)
    finite = np.isfinite(ef) & np.isfinite(gf)
    expected_nonfinite = int(ef.size - np.count_nonzero(np.isfinite(ef)))

    # Non-finite positions must coincide and agree (sign-aware for +/-inf, with
    # NaN==NaN). Otherwise it is a real divergence, not benign noise.
    same_finite_mask = np.array_equal(np.isfinite(ef), np.isfinite(gf))
    nonfinite_match = bool(
        same_finite_mask
        and np.array_equal(ef[~finite], gf[~finite], equal_nan=True)
    )

    abs_err = np.abs(np.where(finite, gf - ef, 0.0))
    n_finite = int(np.count_nonzero(finite))
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    denom = np.abs(np.where(finite, ef, 0.0))
    # Relative error is only defined where the baseline is nonzero, so the reported
    # max is taken over those positions — a near-zero baseline element (common in
    # ReLU-style outputs) must not inflate it to inf. When the baseline is entirely
    # zero yet the candidate still deviates, relative error is undefined everywhere:
    # surface that as inf rather than a misleading 0.0 that reads as "relatively
    # perfect" (PSNR is independently a hard failure in that case).
    nonzero = denom > 0
    if np.any(nonzero):
        max_rel = float((abs_err[nonzero] / denom[nonzero]).max())
    elif max_abs > 0:
        max_rel = float("inf")
    else:
        max_rel = 0.0
    rmse = float(np.sqrt(np.sum(abs_err**2) / n_finite)) if n_finite else 0.0
    mae = float(np.sum(abs_err) / n_finite) if n_finite else 0.0

    if rmse == 0.0:
        psnr = float("inf")
    else:
        peak = float(denom.max()) if denom.size else 0.0
        psnr = float(20.0 * np.log10(peak / rmse)) if peak > 0 else float("-inf")

    # A non-finite mismatch (e.g. the candidate emits NaN where the baseline is
    # finite) is a real divergence, not benign noise. The finite-overlap masking
    # above would otherwise let a bit-exact overlap report +inf and hide it, so
    # force a hard failure that surfaces wherever -inf does (matrix, aggregate).
    if not nonfinite_match:
        psnr = float("-inf")

    # Cosine similarity over the finite-overlap region.
    ev = ef[finite].ravel()
    gv = gf[finite].ravel()
    en = float(np.linalg.norm(ev))
    gn = float(np.linalg.norm(gv))
    if en == 0.0 and gn == 0.0:
        cosine = 1.0
    elif en == 0.0 or gn == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(ev, gv) / (en * gn))

    return OutputAccuracy(
        name=name,
        shape=list(e.shape),
        dtype=str(e.dtype),
        psnr_db=psnr,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        rmse=rmse,
        mae=mae,
        cosine=cosine,
        expected_nonfinite=expected_nonfinite,
        nonfinite_match=nonfinite_match,
    )


def compare_outputs(
    reference: dict[str, np.ndarray],
    got: dict[str, np.ndarray],
) -> AccuracyStats:
    """Compare a full output set, matching names then aggregating worst-case."""
    ref_names = list(reference)
    mapping = naming.match_output_names(ref_names, list(got))

    per_output: list[OutputAccuracy] = []
    notes: list[str] = []
    for ref_name in ref_names:
        actual = mapping.get(ref_name)
        if actual is None or actual not in got:
            notes.append(f"output '{ref_name}' missing from candidate")
            per_output.append(
                OutputAccuracy(
                    name=ref_name, shape=list(np.asarray(reference[ref_name]).shape),
                    dtype=str(np.asarray(reference[ref_name]).dtype),
                    psnr_db=float("-inf"), max_abs_err=float("inf"),
                    max_rel_err=float("inf"), rmse=float("inf"), mae=float("inf"),
                    cosine=0.0, nonfinite_match=False,
                )
            )
            continue
        per_output.append(compare_tensor(ref_name, reference[ref_name], got[actual]))

    # Aggregate to run-level worst case. Filter NaNs defensively.
    psnrs = [o.psnr_db for o in per_output]
    min_psnr = min(psnrs) if psnrs else float("-inf")
    max_abs = max((o.max_abs_err for o in per_output), default=float("inf"))
    max_rel = max((o.max_rel_err for o in per_output), default=float("inf"))
    cosines = [o.cosine for o in per_output]
    mean_cos = float(np.mean(cosines)) if cosines else 0.0
    all_finite_match = all(o.nonfinite_match for o in per_output)

    return AccuracyStats(
        per_output=per_output,
        min_psnr_db=min_psnr,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        mean_cosine=mean_cos,
        all_finite_match=all_finite_match,
        note="; ".join(notes),
    )

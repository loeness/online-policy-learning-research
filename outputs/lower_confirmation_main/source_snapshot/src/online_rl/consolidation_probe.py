"""Pure NumPy helpers for probing shared-parameter consolidation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


_MATRIX_SHAPE = (3, 3)


def _real_finite_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Return a float64 view/copy after checking shape and finite real values."""
    array = np.asarray(value)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_effective(effective: np.ndarray) -> np.ndarray:
    values = _real_finite_array(effective, "effective")
    if values.ndim != 4 or values.shape[1:] != (3, 3, 3):
        raise ValueError(f"effective must have shape (n, 3, 3, 3), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("effective must contain at least one source")
    return values


def _weighted_candidate(
    effective: np.ndarray,
    anchor_psi: Sequence[np.ndarray] | None,
    anchor_bands: Sequence[np.ndarray] | None,
    ridge_relative: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_count = effective.shape[0]
    if anchor_psi is None or anchor_bands is None:
        raise ValueError("weighted extraction requires anchor_psi and anchor_bands")
    if len(anchor_psi) != source_count or len(anchor_bands) != source_count:
        raise ValueError("anchor_psi and anchor_bands must each have one entry per source")
    if not np.isfinite(ridge_relative) or ridge_relative < 0:
        raise ValueError("ridge_relative must be finite and nonnegative")

    gram = np.zeros(_MATRIX_SHAPE, dtype=np.float64)
    numerator = np.zeros(_MATRIX_SHAPE, dtype=np.float64)
    row_counts: list[int] = []
    for source_index, (psi_raw, bands_raw) in enumerate(zip(anchor_psi, anchor_bands)):
        psi = _real_finite_array(psi_raw, f"anchor_psi[{source_index}]")
        bands_array = np.asarray(bands_raw)
        if psi.ndim != 2 or psi.shape[1] != 3:
            raise ValueError(f"anchor_psi[{source_index}] must have shape (k, 3), got {psi.shape}")
        if psi.shape[0] == 0:
            raise ValueError(f"anchor_psi[{source_index}] must contain at least one row")
        if bands_array.shape != (psi.shape[0],):
            raise ValueError(
                f"anchor_bands[{source_index}] must have shape ({psi.shape[0]},), "
                f"got {bands_array.shape}"
            )
        if bands_array.dtype.kind not in "iu":
            raise ValueError(f"anchor_bands[{source_index}] must contain integer band indices")
        bands = bands_array.astype(np.int64, copy=False)
        if np.any((bands < 0) | (bands > 2)):
            raise ValueError(f"anchor_bands[{source_index}] values must be in 0..2")

        row_counts.append(int(psi.shape[0]))
        source_gram = np.einsum("ri,rj->ij", psi, psi) / psi.shape[0]
        gram += source_gram
        effective_for_rows = effective[source_index, bands]
        source_numerator = np.einsum("rab,rb,rj->aj", effective_for_rows, psi, psi)
        numerator += source_numerator / psi.shape[0]

    prior = effective.mean(axis=(0, 1))
    ridge_lambda = float(ridge_relative * np.trace(gram) / 3.0 + 1e-8)
    regularized_gram = gram + ridge_lambda * np.eye(3, dtype=np.float64)
    regularized_numerator = numerator + ridge_lambda * prior
    # Solve C @ regularized_gram = regularized_numerator without forming an inverse.
    candidate = np.linalg.solve(regularized_gram.T, regularized_numerator.T).T
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[0] <= 0:
        gram_condition = float("inf")
    else:
        gram_condition = float(eigenvalues[-1] / eigenvalues[0])

    diagnostics: dict[str, Any] = {
        "source_count": int(source_count),
        "row_counts": tuple(row_counts),
        "row_count": int(sum(row_counts)),
        "ridge_lambda": ridge_lambda,
        "gram_eigenvalues": tuple(float(value) for value in eigenvalues),
        "gram_condition": gram_condition,
        "coefficient_norm": float(np.linalg.norm(candidate)),
    }
    return candidate, diagnostics


def extract_common(
    method: str,
    effective: np.ndarray,
    *,
    anchor_psi: Sequence[np.ndarray] | None = None,
    anchor_bands: Sequence[np.ndarray] | None = None,
    seed: int = 0,
    ridge_relative: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a shared 3 by 3 coefficient matrix from effective parameters.

    ``effective`` is indexed by source, band, and the final two matrix axes.
    Weighted extraction uses one normalized least-squares contribution per
    source, so contexts with different row counts have equal weight.
    """
    values = _validate_effective(effective)
    source_count = values.shape[0]
    prior = values.mean(axis=(0, 1))

    if method == "local_only":
        result = np.zeros(_MATRIX_SHAPE, dtype=np.float32)
        return result, {
            "method": method,
            "source_count": int(source_count),
            "coefficient_norm": 0.0,
        }
    if method == "mean":
        result = prior.astype(np.float32)
        return result, {
            "method": method,
            "source_count": int(source_count),
            "coefficient_norm": float(np.linalg.norm(result)),
        }
    if method == "weighted":
        candidate, diagnostics = _weighted_candidate(
            values, anchor_psi, anchor_bands, ridge_relative
        )
        diagnostics["method"] = method
        return candidate.astype(np.float32), diagnostics
    if method == "random":
        candidate, weighted_diagnostics = _weighted_candidate(
            values, anchor_psi, anchor_bands, ridge_relative
        )
        reference_norm = float(np.linalg.norm(candidate.astype(np.float32)))
        random_generator = np.random.default_rng(seed)
        direction = random_generator.standard_normal(_MATRIX_SHAPE)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0 or reference_norm == 0.0:
            result = np.zeros(_MATRIX_SHAPE, dtype=np.float32)
        else:
            result = (direction * (reference_norm / direction_norm)).astype(np.float32)
        diagnostics = dict(weighted_diagnostics)
        diagnostics.update(
            {
                "method": method,
                "seed": int(seed),
                "reference_norm": reference_norm,
                "coefficient_norm": float(np.linalg.norm(result)),
            }
        )
        return result, diagnostics
    raise ValueError(f"unknown extraction method: {method!r}")


def recompose_shared(
    local: np.ndarray,
    shared: np.ndarray,
    groups: np.ndarray,
    *,
    source_slots: Sequence[int],
    group: int,
    common: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Assign slots to a shared group while preserving their effective weights."""
    local_array = np.asarray(local)
    shared_array = np.asarray(shared)
    groups_array = np.asarray(groups)
    if local_array.ndim != 2 or local_array.shape[1] != 27:
        raise ValueError(f"local must have shape (S, 27), got {local_array.shape}")
    if shared_array.ndim != 2 or shared_array.shape[1] != 9 or shared_array.shape[0] == 0:
        raise ValueError(f"shared must have shape (G, 9) with G > 0, got {shared_array.shape}")
    if groups_array.shape != (local_array.shape[0],):
        raise ValueError(f"groups must have shape ({local_array.shape[0]},), got {groups_array.shape}")
    if local_array.dtype.kind != "f" or shared_array.dtype.kind != "f":
        raise ValueError("local and shared must contain floating point values")
    if groups_array.dtype.kind not in "iu":
        raise ValueError("groups must contain integer group indices")
    local_values = _real_finite_array(local_array, "local")
    shared_values = _real_finite_array(shared_array, "shared")
    if np.any((groups_array < -1) | (groups_array >= shared_array.shape[0])):
        raise ValueError("groups contains an invalid group index")

    if isinstance(group, (bool, np.bool_)) or not isinstance(group, (int, np.integer)):
        raise ValueError("group must be an integer group index")
    destination = int(group)
    if destination < 0 or destination >= shared_array.shape[0]:
        raise ValueError("group is outside the shared parameter array")

    if isinstance(source_slots, (str, bytes)):
        raise ValueError("source_slots must be a nonempty sequence of unique integer slots")
    try:
        slots_raw = list(source_slots)
    except TypeError as exc:
        raise ValueError("source_slots must be a nonempty sequence of unique integer slots") from exc
    if not slots_raw:
        raise ValueError("source_slots must not be empty")
    if any(
        isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer))
        for slot in slots_raw
    ):
        raise ValueError("source_slots must contain integer slot indices")
    slots = [int(slot) for slot in slots_raw]
    if len(set(slots)) != len(slots):
        raise ValueError("source_slots must be unique")
    if any(slot < 0 or slot >= local_array.shape[0] for slot in slots):
        raise ValueError("source_slots contains an out-of-range slot")

    common_values = _real_finite_array(common, "common", _MATRIX_SHAPE)
    old_effective = local_values.reshape(-1, 3, 3, 3).copy()
    assigned_mask = groups_array >= 0
    old_effective[assigned_mask] += shared_values[groups_array[assigned_mask]].reshape(
        -1, 1, 3, 3
    )
    affected_mask = groups_array == destination
    affected_mask[np.asarray(slots, dtype=np.int64)] = True
    affected_slots = np.flatnonzero(affected_mask)

    new_local = local_array.copy()
    new_shared = shared_array.copy()
    new_groups = groups_array.copy()
    new_shared[destination] = common_values.reshape(9).astype(new_shared.dtype, copy=False)
    new_groups[np.asarray(slots, dtype=np.int64)] = destination
    compensated = old_effective[affected_slots] - common_values
    new_local.reshape(-1, 3, 3, 3)[affected_slots] = compensated.astype(
        new_local.dtype, copy=False
    )

    new_effective = new_local.astype(np.float64).reshape(-1, 3, 3, 3).copy()
    new_assigned_mask = new_groups >= 0
    new_effective[new_assigned_mask] += new_shared.astype(np.float64)[
        new_groups[new_assigned_mask]
    ].reshape(-1, 1, 3, 3)
    preservation_error = float(
        np.max(np.abs(new_effective[affected_slots] - old_effective[affected_slots]))
    )
    return new_local, new_shared, new_groups, {
        "affected_slots": tuple(int(slot) for slot in affected_slots),
        "max_effective_preservation_error": preservation_error,
    }

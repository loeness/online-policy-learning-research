"""Behavioral distillation of two resident policies into one shared parent.

This module uses only observations, frozen actor features, source ownership, and
the two residents' effective coefficient matrices. It does not use rewards,
future observations, or a task simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


_OWNER_COUNT = 2
_BAND_COUNT = 3
_ACTION_COUNT = 4
_COEFFICIENT_SHAPE = (3, 3)
_MAX_ROWS_PER_OWNER = 128
_TRAIN_FRACTION = 0.75
_RIDGE = 1.0e-4
_MAX_ITERATIONS = 64
_MAX_EVALUATIONS = 256
_LBFGS_HISTORY_SIZE = 10


@dataclass(frozen=True)
class FrozenPolicyFeatures:
    """Array-only frozen actor inputs used by the behavioral fit."""

    base_logits: np.ndarray
    source_psi: np.ndarray
    contrast_basis: np.ndarray
    temperature: float
    epsilon: float


class _EvaluationBudgetExhausted(RuntimeError):
    """Private signal used to enforce the hard L-BFGS closure-call budget."""


def _as_float_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_features(features: FrozenPolicyFeatures, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not isinstance(features, FrozenPolicyFeatures):
        raise TypeError("features must be FrozenPolicyFeatures")
    base_logits = _as_float_array(features.base_logits, "base_logits")
    source_psi = _as_float_array(features.source_psi, "source_psi")
    contrast_basis = _as_float_array(features.contrast_basis, "contrast_basis", (4, 3))
    if base_logits.shape != (count, _ACTION_COUNT):
        raise ValueError(f"base_logits must have shape ({count}, 4)")
    if source_psi.shape != (count, 3):
        raise ValueError(f"source_psi must have shape ({count}, 3)")
    temperature = float(features.temperature)
    epsilon = float(features.epsilon)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be finite and in [0, 1)")
    return base_logits, source_psi, contrast_basis, temperature, epsilon


def extract_frozen_policy_features(core: Any, observations: Any) -> FrozenPolicyFeatures:
    """Extract frozen policy features without mutating the core or its RNGs.

    ``observations`` must be the already observed 13-dimensional policy input,
    including the supplied route identifier in its final coordinate.
    """
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != 13:
        raise ValueError("observations must have shape N by 13")
    if not np.isfinite(obs).all():
        raise ValueError("observations must contain only finite values")
    source_basis = getattr(core, "source_basis", None)
    if not isinstance(source_basis, torch.Tensor):
        raise TypeError("core must expose a tensor source_basis")
    if not hasattr(core, "mlp_extractor") or not hasattr(core.mlp_extractor, "forward_actor"):
        raise TypeError("core must expose mlp_extractor.forward_actor")
    if not hasattr(core, "extract_features") or not hasattr(core, "action_net"):
        raise TypeError("core must expose extract_features and action_net")
    if not hasattr(core, "_contrast_basis"):
        raise TypeError("core must expose _contrast_basis")

    device = source_basis.device
    dtype = source_basis.dtype
    obs_tensor = torch.as_tensor(obs, dtype=dtype, device=device)
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs_tensor))
        phi = torch.cat((latent, torch.ones_like(latent[:, :1])), dim=1)
        source_psi = F.linear(phi, source_basis)
        base_logits = core.action_net(latent)
        contrast_basis = core._contrast_basis.to(device=device, dtype=dtype)

    return FrozenPolicyFeatures(
        base_logits=base_logits.detach().cpu().numpy().astype(np.float64, copy=True),
        source_psi=source_psi.detach().cpu().numpy().astype(np.float64, copy=True),
        contrast_basis=contrast_basis.detach().cpu().numpy().astype(np.float64, copy=True),
        temperature=float(core.temperature),
        epsilon=float(core.epsilon),
    )


def _log_mixed_probabilities(logits: np.ndarray, temperature: float, epsilon: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    maximum = np.max(scaled, axis=1, keepdims=True)
    log_softmax = scaled - maximum - np.log(np.exp(scaled - maximum).sum(axis=1, keepdims=True))
    if epsilon == 0.0:
        return log_softmax
    return np.logaddexp(math.log1p(-epsilon) + log_softmax, math.log(epsilon / _ACTION_COUNT))


def candidate_probabilities(features: FrozenPolicyFeatures, coefficients: Any) -> np.ndarray:
    """Return the policy probabilities for a shared 3 by 3 parent."""
    base_logits, source_psi, contrast_basis, temperature, epsilon = _validate_features(
        features, len(np.asarray(features.base_logits))
    )
    candidate = _as_float_array(coefficients, "coefficients", _COEFFICIENT_SHAPE)
    contrast = source_psi @ candidate.T
    logits = base_logits + contrast @ contrast_basis.T
    return np.exp(_log_mixed_probabilities(logits, temperature, epsilon))


def _validate_probabilities(value: Any, count: int) -> np.ndarray:
    probabilities = _as_float_array(value, "teacher_probabilities")
    if probabilities.shape != (count, _ACTION_COUNT):
        raise ValueError(f"teacher_probabilities must have shape ({count}, 4)")
    if np.any(probabilities < 0.0):
        raise ValueError("teacher_probabilities must be nonnegative")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("teacher_probabilities rows must sum to one")
    return probabilities


def behavioral_objective_and_gradient(
    coefficients: Any,
    features: FrozenPolicyFeatures,
    teacher_probabilities: Any,
    row_weights: Any,
    common_mean: Any,
) -> tuple[float, np.ndarray]:
    """Return weighted forward KL plus ridge and its exact coefficient gradient.

    ``row_weights`` should give each source an equal total weight. The gradient
    differentiates through the temperature-scaled softmax and uniform epsilon
    mixture, including the epsilon mixture's exact derivative.
    """
    candidate = _as_float_array(coefficients, "coefficients", _COEFFICIENT_SHAPE)
    common = _as_float_array(common_mean, "common_mean", _COEFFICIENT_SHAPE)
    base_logits, source_psi, contrast_basis, temperature, epsilon = _validate_features(
        features, len(np.asarray(features.base_logits))
    )
    count = len(base_logits)
    teacher = _validate_probabilities(teacher_probabilities, count)
    weights = _as_float_array(row_weights, "row_weights", (count,))
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("row_weights must be nonnegative and sum to one")

    contrast = source_psi @ candidate.T
    logits = base_logits + contrast @ contrast_basis.T
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    softmax = np.exp(shifted)
    softmax /= softmax.sum(axis=1, keepdims=True)
    student = (1.0 - epsilon) * softmax + epsilon / _ACTION_COUNT
    teacher_log = np.zeros_like(teacher)
    teacher_positive = teacher > 0.0
    teacher_log[teacher_positive] = np.log(teacher[teacher_positive])
    student_log = np.log(student)
    row_kl = np.sum(np.where(teacher_positive, teacher * (teacher_log - student_log), 0.0), axis=1)
    ridge_delta = candidate - common
    ridge_penalty = _RIDGE * float(np.mean(ridge_delta * ridge_delta))
    objective = float(weights @ row_kl + ridge_penalty)

    ratio = teacher / student
    softmax_weighted_ratio = np.sum(softmax * ratio, axis=1, keepdims=True)
    logit_gradient = ((1.0 - epsilon) / temperature) * softmax * (softmax_weighted_ratio - ratio)
    coefficient_gradient = (logit_gradient @ contrast_basis).T @ (source_psi * weights[:, None])
    coefficient_gradient += (2.0 * _RIDGE / candidate.size) * ridge_delta
    return objective, np.asarray(coefficient_gradient, dtype=np.float64)


def _bands_from_observations(observations: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(observations[:, 0], 0.0, 1.0) * 3.0).astype(np.int64).clip(0, 2)


def _teacher_probabilities(
    features: FrozenPolicyFeatures,
    effective_matrices: np.ndarray,
    owner_indices: np.ndarray,
    bands: np.ndarray,
) -> np.ndarray:
    count = len(owner_indices)
    base_logits, source_psi, contrast_basis, temperature, epsilon = _validate_features(features, count)
    row_coefficients = effective_matrices[owner_indices, bands]
    contrast = np.einsum("nij,nj->ni", row_coefficients, source_psi)
    logits = base_logits + contrast @ contrast_basis.T
    return np.exp(_log_mixed_probabilities(logits, temperature, epsilon))


def _stable_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _equal_source_weights(owner_indices: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(owner_indices), dtype=np.float64)
    counts = np.bincount(owner_indices, minlength=_OWNER_COUNT)
    for owner in range(_OWNER_COUNT):
        if counts[owner]:
            weights[owner_indices == owner] = 0.5 / int(counts[owner])
    return weights


def fit_behavioral_candidate(
    observations: Any,
    owner_indices: Any,
    effective_matrices: Any,
    features: FrozenPolicyFeatures,
    *,
    min_rows_per_owner: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one shared parent from observed rows and two frozen resident policies.

    Rows are grouped by the explicit owner index, preserving their input order.
    The latest 128 rows from each owner are retained; their first 75 percent
    train the parent and the final 25 percent are diagnostic holdout rows only.
    The optimizer sees no outcome signal, future observations, or simulator.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    obs_raw = np.asarray(observations, dtype=np.float32)
    if obs_raw.ndim != 2 or obs_raw.shape[1] != 13:
        raise ValueError("observations must have shape N by 13")
    if not np.isfinite(obs_raw).all():
        raise ValueError("observations must contain only finite values")
    count = len(obs_raw)
    owner_raw = np.asarray(owner_indices)
    if owner_raw.shape != (count,) or owner_raw.dtype.kind not in "iu":
        raise ValueError("owner_indices must be an integer vector matching observations")
    owners = owner_raw.astype(np.int64, copy=False)
    if np.any((owners < 0) | (owners >= _OWNER_COUNT)):
        raise ValueError("owner_indices must contain only owner zero or one")
    effective = _as_float_array(effective_matrices, "effective_matrices", (2, 3, 3, 3))
    base_logits, source_psi, contrast_basis, temperature, epsilon = _validate_features(features, count)
    if isinstance(min_rows_per_owner, bool) or int(min_rows_per_owner) != min_rows_per_owner or int(min_rows_per_owner) < 1:
        raise ValueError("min_rows_per_owner must be a positive integer")
    minimum = int(min_rows_per_owner)

    common_mean64 = effective.mean(axis=(0, 1), dtype=np.float64)
    initial = common_mean64.astype(np.float32).astype(np.float64)
    selected_rows: list[np.ndarray] = []
    original_counts = np.bincount(owners, minlength=_OWNER_COUNT)
    truncated_counts: list[int] = []
    for owner in range(_OWNER_COUNT):
        rows = np.flatnonzero(owners == owner)
        truncated_counts.append(max(0, len(rows) - _MAX_ROWS_PER_OWNER))
        if len(rows) > _MAX_ROWS_PER_OWNER:
            rows = rows[-_MAX_ROWS_PER_OWNER:]
        selected_rows.append(rows)
    selected_counts = [len(rows) for rows in selected_rows]

    bands_all = _bands_from_observations(obs_raw)
    train_rows: list[np.ndarray] = []
    holdout_rows: list[np.ndarray] = []
    for rows in selected_rows:
        if len(rows) == 0:
            split_at = 0
        elif len(rows) == 1:
            split_at = 1
        else:
            split_at = int(math.floor(_TRAIN_FRACTION * len(rows)))
            split_at = min(max(split_at, 1), len(rows) - 1)
        train_rows.append(rows[:split_at])
        holdout_rows.append(rows[split_at:])

    train_indices = np.concatenate(train_rows) if any(len(rows) for rows in train_rows) else np.empty(0, dtype=np.int64)
    holdout_indices = np.concatenate(holdout_rows) if any(len(rows) for rows in holdout_rows) else np.empty(0, dtype=np.int64)

    def sliced_features(indices: np.ndarray) -> FrozenPolicyFeatures:
        return FrozenPolicyFeatures(
            base_logits=base_logits[indices],
            source_psi=source_psi[indices],
            contrast_basis=contrast_basis,
            temperature=temperature,
            epsilon=epsilon,
        )

    train_features = sliced_features(train_indices)
    train_owners = owners[train_indices]
    train_teacher = _teacher_probabilities(train_features, effective, train_owners, bands_all[train_indices])
    train_weights = _equal_source_weights(train_owners)
    train_counts = [len(rows) for rows in train_rows]
    holdout_features = sliced_features(holdout_indices)
    holdout_owners = owners[holdout_indices]
    holdout_teacher = _teacher_probabilities(
        holdout_features, effective, holdout_owners, bands_all[holdout_indices]
    )
    holdout_counts = [len(rows) for rows in holdout_rows]

    def split_metrics(
        split_features: FrozenPolicyFeatures,
        teacher: np.ndarray,
        split_owner_indices: np.ndarray,
    ) -> tuple[list[float | None], float | None]:
        if len(split_owner_indices) == 0:
            return [None, None], None
        log_candidate = _log_mixed_probabilities(
            split_features.base_logits + (split_features.source_psi @ initial.T) @ split_features.contrast_basis.T,
            split_features.temperature,
            split_features.epsilon,
        )
        log_teacher = np.zeros_like(teacher)
        positive = teacher > 0.0
        log_teacher[positive] = np.log(teacher[positive])
        kl = np.sum(np.where(positive, teacher * (log_teacher - log_candidate), 0.0), axis=1)
        by_owner: list[float | None] = []
        for owner in range(_OWNER_COUNT):
            mask = split_owner_indices == owner
            by_owner.append(float(np.mean(kl[mask])) if np.any(mask) else None)
        if any(value is None for value in by_owner):
            mean_value = None
        else:
            mean_value = float(sum(float(value) for value in by_owner) / _OWNER_COUNT)
        return by_owner, mean_value

    train_kl_by_owner_init, train_kl_mean_init = split_metrics(train_features, train_teacher, train_owners)
    holdout_kl_by_owner_init, holdout_kl_mean_init = split_metrics(
        holdout_features, holdout_teacher, holdout_owners
    )

    finite_inputs = True
    feature_rank = int(np.linalg.matrix_rank(source_psi[train_indices])) if len(train_indices) else 0
    input_arrays = (obs_raw, owner_raw, effective_matrices, features.base_logits, features.source_psi, features.contrast_basis)
    input_bytes = sum(int(np.asarray(array).nbytes) for array in input_arrays)
    selected_count = sum(selected_counts)
    selected_input_bytes = int(
        obs_raw.dtype.itemsize * 13 * selected_count
        + owners.dtype.itemsize * selected_count
        + np.asarray(features.base_logits).dtype.itemsize * 4 * selected_count
        + np.asarray(features.source_psi).dtype.itemsize * 3 * selected_count
        + effective.dtype.itemsize * effective.size
        + contrast_basis.dtype.itemsize * contrast_basis.size
    )
    # Includes the arrays retained by the objective/diagnostic path and a
    # conservative estimate for the 10-entry L-BFGS history, gradient and step.
    temporary_array_bytes_estimate = int(
        selected_count * (_ACTION_COUNT + 3 + _ACTION_COUNT + 1) * 8
        + len(train_indices) * (8 + 8)
        + len(holdout_indices) * 8
        + (_LBFGS_HISTORY_SIZE * 2 + 6) * initial.size * 8
    )
    optimizer_state_bytes_estimate = int(
        (_LBFGS_HISTORY_SIZE * 2 + 6) * initial.size * np.dtype(np.float64).itemsize
    )
    support_sufficient = all(value >= minimum for value in selected_counts)
    status = "insufficient_support"
    candidate = initial.copy()
    iterations = 0
    closure_evaluations = 0
    optimizer_state: dict[str, Any] = {}
    objective_init: float | None = None
    objective_fit: float | None = None
    ridge_penalty_fit = 0.0
    optimizer_status = "not_run"

    if support_sufficient:
        objective_init, _ = behavioral_objective_and_gradient(
            initial, train_features, train_teacher, train_weights, common_mean64
        )
        best_candidate = initial.copy()
        best_objective = float(objective_init)
        initial_train_loss = float(train_kl_mean_init) if train_kl_mean_init is not None else float(objective_init)
        parameter = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float64).clone())
        optimizer = torch.optim.LBFGS(
            [parameter],
            lr=1.0,
            max_iter=_MAX_ITERATIONS,
            max_eval=_MAX_EVALUATIONS,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
            history_size=_LBFGS_HISTORY_SIZE,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal closure_evaluations, best_candidate, best_objective
            if closure_evaluations >= _MAX_EVALUATIONS:
                raise _EvaluationBudgetExhausted("L-BFGS closure evaluation budget exhausted")
            closure_evaluations += 1
            current = parameter.detach().cpu().numpy().astype(np.float64, copy=True)
            objective, gradient = behavioral_objective_and_gradient(
                current, train_features, train_teacher, train_weights, common_mean64
            )
            train_kl_by_owner, train_kl_mean = _split_kl_for_candidate(
                train_features, train_teacher, train_owners, current
            )
            if (
                math.isfinite(objective)
                and math.isfinite(train_kl_mean)
                and train_kl_mean <= initial_train_loss + 1.0e-12
                and objective <= best_objective
            ):
                best_candidate = current
                best_objective = float(objective)
            parameter.grad = torch.as_tensor(gradient, dtype=parameter.dtype, device=parameter.device)
            return torch.as_tensor(objective, dtype=parameter.dtype, device=parameter.device)

        try:
            optimizer.step(closure)
            optimizer_status = "completed"
            status = "fitted"
        except _EvaluationBudgetExhausted:
            optimizer_status = "budget_exhausted"
            status = "budget_exhausted"
        optimizer_state = optimizer.state.get(parameter, {})
        iterations = int(optimizer_state.get("n_iter", 0))
        candidate = best_candidate

    candidate32 = np.asarray(candidate, dtype=np.float32).reshape(3, 3).copy()
    finite_candidate = bool(np.isfinite(candidate32).all())
    if not finite_candidate:
        raise FloatingPointError("behavioral candidate is not finite")
    candidate_for_metrics = candidate32.astype(np.float64)
    train_kl_by_owner_fit, train_kl_mean_fit = _split_kl_for_candidate(
        train_features, train_teacher, train_owners, candidate_for_metrics
    )
    # Float32 is the persistent representation. Revert to the float32 mean if
    # rounding the selected iterate would violate the no-worse-than-start rule.
    if (
        support_sufficient
        and train_kl_mean_init is not None
        and train_kl_mean_fit is not None
        and train_kl_mean_fit > train_kl_mean_init + 1.0e-12
    ):
        candidate32 = initial.astype(np.float32)
        candidate_for_metrics = candidate32.astype(np.float64)
        train_kl_by_owner_fit, train_kl_mean_fit = _split_kl_for_candidate(
            train_features, train_teacher, train_owners, candidate_for_metrics
        )
        status = "mean_retained_after_float32_rounding"
    holdout_kl_by_owner_fit, holdout_kl_mean_fit = _split_kl_for_candidate(
        holdout_features, holdout_teacher, holdout_owners, candidate_for_metrics
    )
    if support_sufficient:
        objective_fit, _ = behavioral_objective_and_gradient(
            candidate_for_metrics, train_features, train_teacher, train_weights, common_mean64
        )
        ridge_penalty_fit = _RIDGE * float(np.mean((candidate_for_metrics - common_mean64) ** 2))
    else:
        objective_fit = train_kl_mean_fit
        objective_init = train_kl_mean_init
    cpu_elapsed = max(0.0, time.process_time() - cpu_start)
    wall_elapsed = max(0.0, time.perf_counter() - wall_start)
    diagnostics: dict[str, Any] = {
        "format": "retention_behavioral_candidate_v1",
        "status": status,
        "optimizer": "torch_lbfgs_strong_wolfe",
        "optimizer_status": optimizer_status,
        "optimizer_max_iterations": _MAX_ITERATIONS,
        "optimizer_max_evaluations": _MAX_EVALUATIONS,
        "optimizer_function_evaluations": int(closure_evaluations),
        "total_objective_evaluations": int(closure_evaluations + (2 if support_sufficient else 0)),
        "iterations": int(iterations),
        "min_rows_per_owner": minimum,
        "max_rows_per_owner": _MAX_ROWS_PER_OWNER,
        "input_rows_by_owner": [int(value) for value in original_counts.tolist()],
        "rows_used_by_owner": [int(value) for value in selected_counts],
        "truncated_rows_by_owner": [int(value) for value in truncated_counts],
        "train_rows_by_owner": [int(value) for value in train_counts],
        "holdout_rows_by_owner": [int(value) for value in holdout_counts],
        "ridge": _RIDGE,
        "objective_init": None if objective_init is None else float(objective_init),
        "objective_fit": None if objective_fit is None else float(objective_fit),
        "ridge_penalty_fit": float(ridge_penalty_fit),
        "train_kl_mean_init": train_kl_mean_init,
        "train_kl_mean_fit": train_kl_mean_fit,
        "train_kl_by_owner_init": train_kl_by_owner_init,
        "train_kl_by_owner_fit": train_kl_by_owner_fit,
        "holdout_kl_mean_init": holdout_kl_mean_init,
        "holdout_kl_mean_fit": holdout_kl_mean_fit,
        "holdout_kl_by_owner_init": holdout_kl_by_owner_init,
        "holdout_kl_by_owner_fit": holdout_kl_by_owner_fit,
        "source_psi_rank": feature_rank,
        "finite_inputs": bool(finite_inputs),
        "finite_candidate": finite_candidate,
        "cpu_time_seconds": float(cpu_elapsed),
        "wall_time_seconds": float(wall_elapsed),
        "input_array_bytes": int(input_bytes),
        "selected_input_array_bytes": int(selected_input_bytes),
        "temporary_array_bytes_estimate": int(temporary_array_bytes_estimate),
        "optimizer_workspace_bytes_estimate": int(optimizer_state_bytes_estimate),
        "persistent_candidate_bytes": int(candidate32.nbytes),
        "parameter_hashes": {
            "common_mean_sha256": _stable_hash(initial),
            "candidate_sha256": _stable_hash(candidate32),
            "effective_matrices_sha256": _stable_hash(effective),
        },
    }
    return candidate32, diagnostics


def _split_kl_for_candidate(
    features: FrozenPolicyFeatures,
    teacher: np.ndarray,
    owner_indices: np.ndarray,
    candidate: np.ndarray,
) -> tuple[list[float | None], float | None]:
    if len(owner_indices) == 0:
        return [None, None], None
    base_logits, source_psi, contrast_basis, temperature, epsilon = _validate_features(features, len(owner_indices))
    logits = base_logits + (source_psi @ candidate.T) @ contrast_basis.T
    log_candidate = _log_mixed_probabilities(logits, temperature, epsilon)
    teacher_log = np.zeros_like(teacher)
    positive = teacher > 0.0
    teacher_log[positive] = np.log(teacher[positive])
    kl = np.sum(np.where(positive, teacher * (teacher_log - log_candidate), 0.0), axis=1)
    by_owner: list[float | None] = []
    for owner in range(_OWNER_COUNT):
        mask = owner_indices == owner
        by_owner.append(float(np.mean(kl[mask])) if np.any(mask) else None)
    present = [float(value) for value in by_owner if value is not None]
    mean_value = float(sum(present) / _OWNER_COUNT) if len(present) == _OWNER_COUNT else None
    return by_owner, mean_value


__all__ = [
    "FrozenPolicyFeatures",
    "behavioral_objective_and_gradient",
    "candidate_probabilities",
    "extract_frozen_policy_features",
    "fit_behavioral_candidate",
]

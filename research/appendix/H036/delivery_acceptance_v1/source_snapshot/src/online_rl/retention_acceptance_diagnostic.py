"""Training-only H036 acceptance probes for the frozen diagnostic phase.

This module deliberately separates selection from scoring.  Selection sees
only copied training rows.  A later score call requires a replay-issued cohort
permit, so an individual locked probe cannot be scored out of sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import numbers
import time
from typing import Any, Mapping

import numpy as np
import torch

from . import retention_behavioral_candidate as behavioral
from . import retention_protected_candidate as protected


_TRACE_FORMAT = "h036_selection_trace_v1"
_LOCK_FORMAT = "h036_acceptance_locked_v1"
_FIT_FORMAT = "h036_fit_v1"
_LAMBDA = 1
_RIDGE = protected._RIDGE
_MAX_ITERATIONS = 64
_MAX_EVALUATIONS = 256
_HISTORY_SIZE = 10
_TRAIN_ROWS = 96
_KL_TOLERANCE = 1.0e-12
_COEFFICIENT_SHAPE = (3, 3)
_LABEL_SELECTORS = {"V": "F64", "UG": "U64", "UR": "U64"}


@dataclass(frozen=True)
class TrainingView:
    """A copied, immutable view containing training rows and provenance only."""

    seed: int
    prepared_id: str
    input_identity_json: str
    training_hash: str
    new_features: behavioral.FrozenPolicyFeatures
    new_teacher: np.ndarray
    new_owners: np.ndarray
    new_weights: np.ndarray
    old_features: behavioral.FrozenPolicyFeatures
    old_teacher: np.ndarray
    old_weights: np.ndarray
    new_source_indices: np.ndarray
    old_source_indices: np.ndarray
    mean64: np.ndarray
    c0: np.ndarray


class _EvaluationBudgetExhausted(RuntimeError):
    """Raised before a closure beyond the frozen evaluation budget."""


def _readonly(value: Any, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _readonly_features(
    features: behavioral.FrozenPolicyFeatures, indices: np.ndarray,
) -> behavioral.FrozenPolicyFeatures:
    return behavioral.FrozenPolicyFeatures(
        base_logits=_readonly(features.base_logits[indices], np.float64),
        source_psi=_readonly(features.source_psi[indices], np.float64),
        contrast_basis=_readonly(features.contrast_basis, np.float64),
        temperature=float(features.temperature),
        epsilon=float(features.epsilon),
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _normalized_identity(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("prepared.input_identity must be a plain dict")
    try:
        return json.loads(_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared.input_identity must be finite JSON data") from exc


def _update_array_hash(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(repr(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _training_hash(
    seed: int,
    new_features: behavioral.FrozenPolicyFeatures,
    new_teacher: np.ndarray,
    new_owners: np.ndarray,
    new_weights: np.ndarray,
    old_features: behavioral.FrozenPolicyFeatures,
    old_teacher: np.ndarray,
    old_weights: np.ndarray,
    new_source_indices: np.ndarray,
    old_source_indices: np.ndarray,
    mean64: np.ndarray,
    c0: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"h036_training_view_v1\0")
    digest.update(str(int(seed)).encode("ascii") + b"\0")
    for prefix, features in (("new", new_features), ("old", old_features)):
        for name in ("base_logits", "source_psi", "contrast_basis"):
            _update_array_hash(digest, f"{prefix}.{name}", getattr(features, name))
        digest.update(f"{prefix}.temperature={features.temperature!r}\0".encode("ascii"))
        digest.update(f"{prefix}.epsilon={features.epsilon!r}\0".encode("ascii"))
    for name, array in (
        ("new_teacher", new_teacher), ("new_owners", new_owners),
        ("new_weights", new_weights), ("old_teacher", old_teacher),
        ("old_weights", old_weights), ("new_source_indices", new_source_indices),
        ("old_source_indices", old_source_indices), ("mean64", mean64), ("c0", c0),
    ):
        _update_array_hash(digest, name, array)
    return digest.hexdigest()


def _require_array(value: Any, name: str, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f"{name} must be a NumPy array with dtype {np.dtype(dtype)} and shape {shape}")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} must contain only finite values")
    return value


def _validate_training_view(view: Any) -> TrainingView:
    if not isinstance(view, TrainingView):
        raise TypeError("training_view must be a TrainingView")
    if isinstance(view.seed, bool) or not isinstance(view.seed, numbers.Integral):
        raise ValueError("training_view.seed must be an integer")
    if not isinstance(view.prepared_id, str) or not view.prepared_id:
        raise ValueError("training_view.prepared_id must be nonempty")
    try:
        json.loads(view.input_identity_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("training_view.input_identity_json must be finite JSON") from exc
    if not isinstance(view.new_features, behavioral.FrozenPolicyFeatures):
        raise TypeError("training_view.new_features must be FrozenPolicyFeatures")
    if not isinstance(view.old_features, behavioral.FrozenPolicyFeatures):
        raise TypeError("training_view.old_features must be FrozenPolicyFeatures")
    for name, features, count in (("new_features", view.new_features, 192), ("old_features", view.old_features, 96)):
        behavioral._validate_features(features, count)
        if float(features.temperature) != 2.0 or float(features.epsilon) != 0.05:
            raise ValueError(f"training_view.{name} must use temperature 2.0 and epsilon 0.05")
    _require_array(view.new_teacher, "new_teacher", np.float64, (192, 4))
    _require_array(view.new_owners, "new_owners", np.int64, (192,))
    if not np.array_equal(view.new_owners, np.repeat(np.arange(2, dtype=np.int64), 96)):
        raise ValueError("training_view.new_owners must have 96 rows for each owner")
    _require_array(view.new_weights, "new_weights", np.float64, (192,))
    _require_array(view.old_teacher, "old_teacher", np.float64, (96, 4))
    _require_array(view.old_weights, "old_weights", np.float64, (96,))
    if not np.array_equal(view.new_weights, np.full(192, 0.5 / 96, dtype=np.float64)):
        raise ValueError("training_view.new_weights must give each owner total weight one half")
    if not np.array_equal(view.old_weights, np.full(96, 1.0 / 96, dtype=np.float64)):
        raise ValueError("training_view.old_weights must be uniform")
    _require_array(view.new_source_indices, "new_source_indices", np.int64, (192,))
    _require_array(view.old_source_indices, "old_source_indices", np.int64, (96,))
    expected_new_sources = np.concatenate((
        np.arange(96, dtype=np.int64), np.arange(128, 224, dtype=np.int64),
    ))
    expected_old_sources = np.arange(96, dtype=np.int64)
    if not np.array_equal(view.new_source_indices, expected_new_sources):
        raise ValueError("training_view.new_source_indices differ from the frozen train split")
    if not np.array_equal(view.old_source_indices, expected_old_sources):
        raise ValueError("training_view.old_source_indices differ from the frozen train split")
    _require_array(view.mean64, "mean64", np.float64, _COEFFICIENT_SHAPE)
    _require_array(view.c0, "c0", np.float64, _COEFFICIENT_SHAPE)
    expected_c0 = view.mean64.astype(np.float32).astype(np.float64)
    if not np.array_equal(view.c0, expected_c0):
        raise ValueError("training_view.c0 must equal float64(float32(mean64))")
    expected_hash = _training_hash(
        int(view.seed), view.new_features, view.new_teacher, view.new_owners, view.new_weights,
        view.old_features, view.old_teacher, view.old_weights, view.new_source_indices,
        view.old_source_indices, view.mean64, view.c0,
    )
    if view.training_hash != expected_hash:
        raise ValueError("training_view training hash does not match its arrays")
    return view


def make_training_view(prepared: dict[str, Any]) -> TrainingView:
    """Copy only the frozen 192 new and 96 old training rows from prepared.

    The function intentionally never reads holdout arrays, split labels beyond
    the fixed training indices, observations, checkpoint state, or rewards.
    """
    if type(prepared) is not dict:
        raise TypeError("prepared must be a plain dict")
    if prepared.get("format") != "h036_prepared_v1":
        raise ValueError("prepared.format must be h036_prepared_v1")
    seed = prepared.get("seed")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError("prepared.seed must be an integer")
    prepared_id = prepared.get("prepared_id")
    if not isinstance(prepared_id, str) or not prepared_id:
        raise ValueError("prepared.prepared_id must be nonempty")
    identity = _normalized_identity(prepared.get("input_identity"))

    # Read only the original training index arrays.  Holdout indices are never
    # inspected by this projection.
    new_indices = prepared.get("new_train_indices")
    old_indices = prepared.get("old_train_indices")
    expected_new_indices = np.concatenate((np.arange(96, dtype=np.int64), np.arange(128, 224, dtype=np.int64)))
    expected_old_indices = np.arange(96, dtype=np.int64)
    if not isinstance(new_indices, np.ndarray) or new_indices.dtype != np.int64 or new_indices.shape != (192,):
        raise ValueError("prepared.new_train_indices must be int64 with shape (192,)")
    if not isinstance(old_indices, np.ndarray) or old_indices.dtype != np.int64 or old_indices.shape != (96,):
        raise ValueError("prepared.old_train_indices must be int64 with shape (96,)")
    if not np.array_equal(new_indices, expected_new_indices) or not np.array_equal(old_indices, expected_old_indices):
        raise ValueError("prepared training indices differ from the frozen H036 train split")
    new_indices = np.array(new_indices, dtype=np.int64, copy=True)
    old_indices = np.array(old_indices, dtype=np.int64, copy=True)
    new_all = prepared.get("new_features")
    old_all = prepared.get("old_features")
    if not isinstance(new_all, behavioral.FrozenPolicyFeatures) or not isinstance(old_all, behavioral.FrozenPolicyFeatures):
        raise TypeError("prepared must contain frozen new_features and old_features")
    new_features = _readonly_features(new_all, new_indices)
    old_features = _readonly_features(old_all, old_indices)
    new_teacher_all = prepared.get("new_teacher")
    old_teacher_all = prepared.get("old_teacher")
    if not isinstance(new_teacher_all, np.ndarray) or new_teacher_all.dtype != np.float64 or new_teacher_all.shape != (256, 4):
        raise ValueError("prepared.new_teacher must be float64 with shape (256, 4)")
    if not isinstance(old_teacher_all, np.ndarray) or old_teacher_all.dtype != np.float64 or old_teacher_all.shape != (128, 4):
        raise ValueError("prepared.old_teacher must be float64 with shape (128, 4)")
    owners_all = prepared.get("new_owners")
    if not isinstance(owners_all, np.ndarray) or owners_all.dtype != np.int64 or owners_all.shape != (256,):
        raise ValueError("prepared.new_owners must be int64 with shape (256,)")
    if not np.array_equal(owners_all, np.repeat(np.arange(2, dtype=np.int64), 128)):
        raise ValueError("prepared.new_owners must have the frozen owner row order")
    new_teacher = _readonly(new_teacher_all[new_indices], np.float64)
    old_teacher = _readonly(old_teacher_all[old_indices], np.float64)
    new_owners = _readonly(owners_all[new_indices], np.int64)
    new_weights = _readonly(np.full(192, 0.5 / 96, dtype=np.float64))
    old_weights = _readonly(np.full(96, 1.0 / 96, dtype=np.float64))
    mean_source = prepared.get("mean64")
    _require_array(mean_source, "prepared.mean64", np.float64, _COEFFICIENT_SHAPE)
    mean64 = _readonly(mean_source, np.float64)
    c0 = _readonly(mean64.astype(np.float32).astype(np.float64), np.float64)
    new_source_indices = _readonly(new_indices, np.int64)
    old_source_indices = _readonly(old_indices, np.int64)
    training_hash = _training_hash(
        int(seed), new_features, new_teacher, new_owners, new_weights, old_features, old_teacher,
        old_weights, new_source_indices, old_source_indices, mean64, c0,
    )
    result = TrainingView(
        seed=int(seed), prepared_id=prepared_id,
        input_identity_json=_json_bytes(identity).decode("utf-8"),
        training_hash=training_hash, new_features=new_features, new_teacher=new_teacher,
        new_owners=new_owners, new_weights=new_weights, old_features=old_features,
        old_teacher=old_teacher, old_weights=old_weights,
        new_source_indices=new_source_indices, old_source_indices=old_source_indices,
        mean64=mean64, c0=c0,
    )
    return _validate_training_view(result)


def _candidate_hash(value: np.ndarray, dtype: Any) -> str:
    return protected._candidate_hash(value, np.dtype(dtype))


def _array_raw_sha256(value: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _trace_path_sha256(diagnostics: Mapping[str, Any]) -> str:
    """Return the same training-path digest as the standard-library replayer."""
    from . import retention_acceptance_replay as replay

    return replay.replay_diagnostics(diagnostics)["path_sha256"]


def _trace_digest(trace: Mapping[str, Any]) -> str:
    candidates = trace.get("candidates")
    diagnostics = trace.get("diagnostics")
    if type(candidates) is not dict or set(candidates) != {"C0", "F64", "U64"}:
        raise ValueError("trace candidates must contain exactly C0, F64, and U64")
    if type(diagnostics) is not dict or type(diagnostics.get("closure_records")) is not list:
        raise ValueError("trace diagnostics are incomplete")
    identity = {
        "format": trace.get("format"), "seed": trace.get("seed"),
        "lambda_value": trace.get("lambda_value"), "training_hash": trace.get("training_hash"),
        "path_sha256": trace.get("path_sha256"), "origin_indices": trace.get("origin_indices"),
        "closure_records": diagnostics["closure_records"],
        "initial_fullL": diagnostics.get("initial_fullL"),
        "initial_candidate_sha256": diagnostics.get("initial_candidate_sha256"),
        "best_float64_candidate_sha256": diagnostics.get("best_float64_candidate_sha256"),
        "u_best_float64_candidate_sha256": diagnostics.get("u_best_float64_candidate_sha256"),
        "best_float64_fullL": diagnostics.get("best_float64_fullL"),
        "u_best_float64_fullL": diagnostics.get("u_best_float64_fullL"),
        "selected_float32_candidate_sha256": diagnostics.get("selected_float32_candidate_sha256"),
        "candidate_sha256": diagnostics.get("candidate_sha256"),
        "fallback_applied": diagnostics.get("fallback_applied"),
        "fallback_reason": diagnostics.get("fallback_reason"),
        "termination_reason": diagnostics.get("termination_reason"),
        "iterations": diagnostics.get("iterations"),
    }
    digest = hashlib.sha256(b"h036_selection_trace_v1\0")
    digest.update(_json_bytes(identity))
    for label in ("C0", "F64", "U64"):
        array = np.asarray(candidates[label])
        if array.dtype != np.dtype(np.float64) or array.shape != _COEFFICIENT_SHAPE or not np.isfinite(array).all():
            raise ValueError(f"trace candidate {label} must be finite float64 3 by 3")
        digest.update(label.encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def fit_selection_trace(training_view: TrainingView) -> dict[str, Any]:
    """Run the original lambda-one L-BFGS path with a noninterfering U64 tally."""
    view = _validate_training_view(training_view)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    memory_before = protected.process_memory_snapshot()

    c0 = np.asarray(view.c0, dtype=np.float64).copy()
    c0_float32 = c0.astype(np.float32)
    mean64 = np.asarray(view.mean64, dtype=np.float64)
    initial_full_l, initial_gradient = protected.protected_objective_and_gradient(
        c0, view.new_features, view.new_teacher, view.new_weights,
        view.old_features, view.old_teacher, view.old_weights, mean64, _LAMBDA,
    )
    objective_evaluations = protected._objective_call_count(_LAMBDA)
    metric_evaluations = 0
    if not math.isfinite(initial_full_l) or not np.isfinite(initial_gradient).all():
        raise FloatingPointError("initial objective or gradient is not finite")
    initial_ridge = protected._ridge_value(c0, mean64)
    initial_new_train_by_owner, initial_new_train_kl = protected._owner_kl_diagnostics(
        view.new_features, view.new_teacher, view.new_owners, c0,
    )
    initial_old_train_kl = protected._weighted_forward_kl(
        view.old_features, view.old_teacher, view.old_weights, c0,
    )
    metric_evaluations += 2
    if not all(math.isfinite(value) for value in (initial_ridge, initial_new_train_kl, initial_old_train_kl)):
        raise FloatingPointError("initial training KL or ridge is not finite")

    f_candidate = c0.copy()
    f_full_l = float(initial_full_l)
    f_new_kl = float(initial_new_train_kl)
    f_old_kl = float(initial_old_train_kl)
    u_candidate = c0.copy()
    u_full_l = float(initial_full_l)
    u_new_kl = float(initial_new_train_kl)
    u_old_kl = float(initial_old_train_kl)
    f_origin = 0
    u_origin = 0
    closure_records: list[dict[str, Any]] = []
    closure_evaluations = 0
    parameter = torch.nn.Parameter(torch.as_tensor(c0, dtype=torch.float64).clone())
    optimizer = torch.optim.LBFGS(
        [parameter], lr=1.0, max_iter=_MAX_ITERATIONS, max_eval=_MAX_EVALUATIONS,
        tolerance_grad=1.0e-9, tolerance_change=1.0e-12,
        history_size=_HISTORY_SIZE, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations, objective_evaluations, metric_evaluations
        nonlocal f_candidate, f_full_l, f_new_kl, f_old_kl, f_origin
        nonlocal u_candidate, u_full_l, u_new_kl, u_old_kl, u_origin
        if closure_evaluations >= _MAX_EVALUATIONS:
            raise _EvaluationBudgetExhausted("L-BFGS closure evaluation budget exhausted")
        closure_evaluations += 1
        current = parameter.detach().cpu().numpy().astype(np.float64, copy=True)
        try:
            full_l, gradient = protected.protected_objective_and_gradient(
                current, view.new_features, view.new_teacher, view.new_weights,
                view.old_features, view.old_teacher, view.old_weights, mean64, _LAMBDA,
            )
            objective_evaluations += protected._objective_call_count(_LAMBDA)
            new_kl = protected._owner_kl_diagnostics(
                view.new_features, view.new_teacher, view.new_owners, current,
            )[1]
            old_kl = protected._weighted_forward_kl(
                view.old_features, view.old_teacher, view.old_weights, current,
            )
            metric_evaluations += 2
            ridge = protected._ridge_value(current, mean64)
            if not all(math.isfinite(value) for value in (full_l, new_kl, old_kl, ridge)) or not np.isfinite(gradient).all():
                raise FloatingPointError("closure objective, gradient, KL, or ridge is not finite")
        except FloatingPointError as exc:
            partial = {
                "closure_records": list(closure_records),
                "attempted_closure_count": int(closure_evaluations),
                "reason": str(exc),
                "attempted_candidate_hash": _candidate_hash(current, np.dtype(np.float64)),
            }
            raise protected._ClosureEvaluationError(str(exc), partial) from exc

        f_accepted = False
        if new_kl > initial_new_train_kl + _KL_TOLERANCE:
            reject_reason = "new_train_kl_gate"
        elif full_l <= f_full_l:
            f_accepted = True
            reject_reason = None
            f_candidate = current.copy()
            f_full_l = float(full_l)
            f_new_kl = float(new_kl)
            f_old_kl = float(old_kl)
            f_origin = int(closure_evaluations)
        else:
            reject_reason = "full_objective_above_incumbent"

        u_updated = full_l <= u_full_l
        if u_updated:
            u_candidate = current.copy()
            u_full_l = float(full_l)
            u_new_kl = float(new_kl)
            u_old_kl = float(old_kl)
            u_origin = int(closure_evaluations)

        closure_records.append({
            "index": int(closure_evaluations), "fullL": float(full_l),
            "newKL": float(new_kl), "oldKL": float(old_kl), "ridge": float(ridge),
            "accepted": bool(f_accepted), "reject_reason": reject_reason,
            "candidate_hash": _candidate_hash(current, np.dtype(np.float64)),
            "u_updated": bool(u_updated),
        })
        parameter.grad = torch.as_tensor(gradient, dtype=parameter.dtype, device=parameter.device)
        return torch.as_tensor(full_l, dtype=parameter.dtype, device=parameter.device)

    termination_reason = "optimizer_completed"
    try:
        optimizer.step(closure)
    except _EvaluationBudgetExhausted:
        termination_reason = "evaluation_budget_exhausted"
    optimizer_state = optimizer.state.get(parameter, {})
    iterations = int(optimizer_state.get("n_iter", 0))
    optimizer_state_tensor_bytes = protected._tensor_bytes(optimizer_state)

    selected_float32 = f_candidate.astype(np.float32)
    selected_float32_new_train_kl = protected._owner_kl_diagnostics(
        view.new_features, view.new_teacher, view.new_owners,
        selected_float32.astype(np.float64),
    )[1]
    metric_evaluations += 1
    if not math.isfinite(selected_float32_new_train_kl):
        raise FloatingPointError("float32 new-train KL is not finite")
    fallback_applied = selected_float32_new_train_kl > initial_new_train_kl + _KL_TOLERANCE
    fallback_reason = "float32_new_train_kl_gate" if fallback_applied else None
    f32_candidate = c0_float32.copy() if fallback_applied else selected_float32.copy()
    returned_is_c0 = bool(np.array_equal(f32_candidate, c0_float32))

    final64 = f32_candidate.astype(np.float64)
    final_full_l, final_gradient = protected.protected_objective_and_gradient(
        final64, view.new_features, view.new_teacher, view.new_weights,
        view.old_features, view.old_teacher, view.old_weights, mean64, _LAMBDA,
    )
    objective_evaluations += protected._objective_call_count(_LAMBDA)
    final_new_train_by_owner, final_new_train_kl = protected._owner_kl_diagnostics(
        view.new_features, view.new_teacher, view.new_owners, final64,
    )
    final_old_train_kl = protected._weighted_forward_kl(
        view.old_features, view.old_teacher, view.old_weights, final64,
    )
    metric_evaluations += 2
    final_ridge = protected._ridge_value(final64, mean64)
    if (
        not math.isfinite(final_full_l) or not np.isfinite(final_gradient).all()
        or not all(math.isfinite(value) for value in (final_new_train_kl, final_old_train_kl, final_ridge))
    ):
        raise FloatingPointError("final objective, gradient, training KL, or ridge is not finite")

    candidates = {
        "C0": _readonly(c0, "<f8"),
        "F64": _readonly(f_candidate, "<f8"),
        "U64": _readonly(u_candidate, "<f8"),
    }
    origin_indices = {"C0": 0, "F64": int(f_origin), "U64": int(u_origin)}
    initial_metrics = {
        "new_train_kl_by_owner": [float(value) for value in initial_new_train_by_owner],
        "new_train_kl": float(initial_new_train_kl), "old_train_kl": float(initial_old_train_kl),
        "ridge": float(initial_ridge),
    }
    diagnostics: dict[str, Any] = {
        "optimizer": "torch_lbfgs_strong_wolfe",
        "optimizer_max_iterations": _MAX_ITERATIONS,
        "optimizer_max_evaluations": _MAX_EVALUATIONS,
        "optimizer_history_size": _HISTORY_SIZE,
        "optimizer_learning_rate": 1.0,
        "optimizer_tolerance_grad": 1.0e-9,
        "optimizer_tolerance_change": 1.0e-12,
        "closure_count": int(closure_evaluations), "iterations": int(iterations),
        "termination_reason": termination_reason, "closure_records": closure_records,
        "initial_fullL": float(initial_full_l),
        "initial_candidate_sha256": _candidate_hash(c0, np.dtype(np.float64)),
        "best_float64_candidate_sha256": _candidate_hash(f_candidate, np.dtype(np.float64)),
        "best_float64_fullL": float(f_full_l), "best_float64_new_train_kl": float(f_new_kl),
        "best_float64_old_train_kl": float(f_old_kl), "best_float64_origin_index": int(f_origin),
        "u_best_float64_candidate_sha256": _candidate_hash(u_candidate, np.dtype(np.float64)),
        "u_best_float64_fullL": float(u_full_l), "u_best_float64_new_train_kl": float(u_new_kl),
        "u_best_float64_old_train_kl": float(u_old_kl), "u_best_float64_origin_index": int(u_origin),
        "selected_float32_candidate_sha256": _candidate_hash(selected_float32, np.dtype(np.float32)),
        "selected_float32_new_train_kl": float(selected_float32_new_train_kl),
        "final_float32_fullL": float(final_full_l), "final_float32_ridge": float(final_ridge),
        "candidate_sha256": _candidate_hash(f32_candidate, np.dtype(np.float32)),
        "returned_is_C0": returned_is_c0, "fallback_applied": bool(fallback_applied),
        "fallback_reason": fallback_reason, "initial_metrics": initial_metrics,
        "final_metrics": {
            "new_train_kl_by_owner": [float(value) for value in final_new_train_by_owner],
            "new_train_kl": float(final_new_train_kl), "old_train_kl": float(final_old_train_kl),
        },
        "ridge": float(_RIDGE),
        "computation_counts": {
            "initialization_objective_evaluations": int(protected._objective_call_count(_LAMBDA)),
            "closure_objective_evaluations": int(closure_evaluations * protected._objective_call_count(_LAMBDA)),
            "final_objective_evaluations": int(protected._objective_call_count(_LAMBDA)),
            "total_objective_evaluations": int(objective_evaluations),
            "initialization_kl_metric_evaluations": 2,
            "closure_kl_metric_evaluations": int(2 * closure_evaluations),
            "float32_gate_kl_metric_evaluations": 1,
            "final_kl_metric_evaluations": 2,
            "total_kl_metric_evaluations": int(metric_evaluations),
            "total_objective_and_kl_evaluations": int(objective_evaluations + metric_evaluations),
        },
        "training_rows": {"new": 192, "old": 96},
        "prepared_id": view.prepared_id, "seed": int(view.seed),
        "teacher_array_bytes": int(view.new_teacher.nbytes + view.old_teacher.nbytes),
        "training_view_array_bytes": int(sum(
            array.nbytes
            for features in (view.new_features, view.old_features)
            for array in (features.base_logits, features.source_psi, features.contrast_basis)
        ) + sum(array.nbytes for array in (
            view.new_teacher, view.new_owners, view.new_weights, view.old_teacher,
            view.old_weights, view.new_source_indices, view.old_source_indices,
            view.mean64, view.c0,
        ))),
        "candidate_array_bytes": int(sum(array.nbytes for array in candidates.values())),
        "scalar_trace_bytes": 0,
        "optimizer_state_tensor_bytes": int(optimizer_state_tensor_bytes),
        "memory_before": memory_before,
        "memory_after": protected.process_memory_snapshot(),
        "memory_measurement_scope": "process-lifetime RSS snapshots; not fit-isolated peak",
        "cpu_time_seconds": max(0.0, time.process_time() - cpu_start),
        "wall_time_seconds": max(0.0, time.perf_counter() - wall_start),
        "torch_num_threads": int(torch.get_num_threads()),
    }
    diagnostics["scalar_trace_bytes"] = int(len(_json_bytes({
        key: value for key, value in diagnostics.items() if key != "scalar_trace_bytes"
    })))
    path_sha256 = _trace_path_sha256(diagnostics)
    trace: dict[str, Any] = {
        "format": _TRACE_FORMAT, "seed": int(view.seed), "prepared_id": view.prepared_id,
        "input_identity": json.loads(view.input_identity_json),
        "lambda_value": _LAMBDA, "training_hash": view.training_hash,
        "origin_indices": origin_indices, "candidates": candidates,
        "path_sha256": path_sha256, "diagnostics": diagnostics,
    }
    trace["trace_sha256"] = _trace_digest(trace)
    return trace


def _validate_trace(view: TrainingView, trace: Mapping[str, Any]) -> None:
    if type(trace) is not dict or trace.get("format") != _TRACE_FORMAT:
        raise TypeError("trace must be an h036_selection_trace_v1 dict")
    if trace.get("seed") != view.seed or trace.get("prepared_id") != view.prepared_id:
        raise ValueError("trace does not belong to this training view")
    if trace.get("training_hash") != view.training_hash:
        raise ValueError("trace training hash differs from this training view")
    if trace.get("lambda_value") != _LAMBDA:
        raise ValueError("trace lambda must be fixed at one")
    diagnostics = trace.get("diagnostics")
    if type(diagnostics) is not dict:
        raise ValueError("trace diagnostics are missing")
    records = diagnostics.get("closure_records")
    if type(records) is not list or len(records) != diagnostics.get("closure_count"):
        raise ValueError("trace closure record count is inconsistent")
    if len(records) > _MAX_EVALUATIONS:
        raise ValueError("trace exceeds the frozen closure evaluation budget")
    if diagnostics.get("optimizer_max_iterations") != _MAX_ITERATIONS:
        raise ValueError("trace optimizer iteration budget differs from the frozen value")
    if diagnostics.get("optimizer_max_evaluations") != _MAX_EVALUATIONS:
        raise ValueError("trace optimizer evaluation budget differs from the frozen value")
    if diagnostics.get("optimizer_history_size") != _HISTORY_SIZE:
        raise ValueError("trace optimizer history size differs from the frozen value")
    if diagnostics.get("optimizer_learning_rate") != 1.0:
        raise ValueError("trace optimizer learning rate differs from the frozen value")
    if diagnostics.get("optimizer_tolerance_grad") != 1.0e-9:
        raise ValueError("trace optimizer gradient tolerance differs from the frozen value")
    if diagnostics.get("optimizer_tolerance_change") != 1.0e-12:
        raise ValueError("trace optimizer change tolerance differs from the frozen value")
    iterations = diagnostics.get("iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, numbers.Integral) or not 0 <= iterations <= _MAX_ITERATIONS:
        raise ValueError("trace optimizer iteration count is invalid")
    if diagnostics.get("termination_reason") not in {"optimizer_completed", "evaluation_budget_exhausted"}:
        raise ValueError("trace termination reason is invalid")
    if trace.get("path_sha256") != _trace_path_sha256(diagnostics):
        raise ValueError("trace path digest does not match closure records")
    actual = _trace_digest(trace)
    if trace.get("trace_sha256") != actual:
        raise ValueError("trace SHA-256 does not match its candidates and path")
    origins = trace.get("origin_indices")
    if type(origins) is not dict or set(origins) != {"C0", "F64", "U64"}:
        raise ValueError("trace origin_indices are incomplete")
    for key in origins:
        value = origins[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
            raise ValueError("trace origin indices must be nonnegative integers")
    if not np.array_equal(np.asarray(trace["candidates"]["C0"]), view.c0):
        raise ValueError("trace C0 differs from the original float32(M) initialization")
    c0_hash = _candidate_hash(view.c0, np.dtype(np.float64))
    if diagnostics.get("initial_candidate_sha256") != c0_hash:
        raise ValueError("trace initial candidate identity differs from C0")
    if records:
        if records[0].get("candidate_hash") != c0_hash:
            raise ValueError("trace first closure is not the original C0 candidate")
    elif diagnostics.get("best_float64_candidate_sha256") != c0_hash:
        raise ValueError("zero-closure trace does not retain the C0 initial candidate")
    initial = diagnostics.get("initial_metrics")
    if type(initial) is not dict or not all(
        math.isfinite(float(initial.get(name, float("nan"))))
        for name in ("new_train_kl", "old_train_kl", "ridge")
    ):
        raise ValueError("trace initial training metrics are incomplete or nonfinite")
    if _candidate_hash(trace["candidates"]["C0"], np.dtype(np.float64)) != c0_hash:
        raise ValueError("trace C0 helper hash does not match the training view")

    # Reconstruct both exact incumbent paths from the finite scalar records.
    # This checks the <= tie rule and proves that U did not skip any visited
    # full-objective point before a locked probe can be formed.
    f_best_l = float(diagnostics["initial_fullL"])
    u_best_l = float(diagnostics["initial_fullL"])
    f_origin = u_origin = 0
    f_hash = u_hash = c0_hash
    baseline_kl = float(initial["new_train_kl"])
    for expected_index, record in enumerate(records, start=1):
        if type(record) is not dict or record.get("index") != expected_index:
            raise ValueError("trace closure indices must be contiguous and ordered")
        for key in ("fullL", "newKL", "oldKL", "ridge"):
            if not isinstance(record.get(key), (int, float)) or not math.isfinite(float(record[key])):
                raise FloatingPointError(f"trace closure {expected_index} {key} is nonfinite")
        candidate_hash = record.get("candidate_hash")
        if not isinstance(candidate_hash, str) or len(candidate_hash) != 64:
            raise ValueError(f"trace closure {expected_index} candidate hash is invalid")
        full_l = float(record["fullL"])
        new_kl = float(record["newKL"])
        f_should_update = new_kl <= baseline_kl + _KL_TOLERANCE and full_l <= f_best_l
        u_should_update = full_l <= u_best_l
        if record.get("accepted") is not f_should_update:
            raise ValueError(f"trace closure {expected_index} F64 update flag is inconsistent")
        if record.get("u_updated") is not u_should_update:
            raise ValueError(f"trace closure {expected_index} U64 update flag is inconsistent")
        expected_reject = (
            None if f_should_update else
            "new_train_kl_gate" if new_kl > baseline_kl + _KL_TOLERANCE else
            "full_objective_above_incumbent"
        )
        if record.get("reject_reason") != expected_reject:
            raise ValueError(f"trace closure {expected_index} F64 reject reason is inconsistent")
        if f_should_update:
            f_best_l, f_origin, f_hash = full_l, expected_index, candidate_hash
        if u_should_update:
            u_best_l, u_origin, u_hash = full_l, expected_index, candidate_hash
    if int(origins["F64"]) != f_origin or int(origins["U64"]) != u_origin:
        raise ValueError("trace F64/U64 source indices do not match the closure path")
    if _candidate_hash(trace["candidates"]["F64"], np.dtype(np.float64)) != f_hash:
        raise ValueError("trace F64 coefficients do not match their closure identity")
    if _candidate_hash(trace["candidates"]["U64"], np.dtype(np.float64)) != u_hash:
        raise ValueError("trace U64 coefficients do not match their closure identity")
    if diagnostics.get("best_float64_candidate_sha256") != f_hash:
        raise ValueError("trace F64 helper hash differs from the selected closure")
    if diagnostics.get("u_best_float64_candidate_sha256") != u_hash:
        raise ValueError("trace U64 helper hash differs from the selected closure")
    if float(diagnostics.get("best_float64_fullL", float("nan"))) != f_best_l:
        raise ValueError("trace F64 full objective differs from the replayed incumbent")
    if float(diagnostics.get("u_best_float64_fullL", float("nan"))) != u_best_l:
        raise ValueError("trace U64 full objective differs from the replayed incumbent")
    if diagnostics.get("best_float64_origin_index") != f_origin:
        raise ValueError("trace F64 diagnostic origin index differs from replay")
    if diagnostics.get("u_best_float64_origin_index") != u_origin:
        raise ValueError("trace U64 diagnostic origin index differs from replay")


def _locked_digest(locked: Mapping[str, Any]) -> str:
    probes = locked.get("probes")
    if type(probes) is not dict or set(probes) != {"V", "UG", "UR"}:
        raise ValueError("locked probes must contain exactly V, UG, and UR")
    metadata = {key: value for key, value in locked.items() if key not in {"probes", "lock_sha256"}}
    digest = hashlib.sha256(b"h036_acceptance_locked_v1\0")
    digest.update(_json_bytes(metadata))
    for label in ("V", "UG", "UR"):
        entry = probes[label]
        if type(entry) is not dict:
            raise TypeError(f"probe {label} must be a plain dict")
        entry_metadata = {key: value for key, value in entry.items() if key not in {"candidate", "raw_cast_candidate"}}
        digest.update(label.encode("ascii") + b"\0")
        digest.update(_json_bytes(entry_metadata))
        for key in ("raw_cast_candidate", "candidate"):
            array = entry.get(key)
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float32) or array.shape != _COEFFICIENT_SHAPE:
                raise ValueError(f"probe {label}.{key} must be float32 3 by 3")
            if not np.isfinite(array).all():
                raise FloatingPointError(f"probe {label}.{key} must be finite")
            digest.update(key.encode("ascii") + b"\0")
            digest.update(np.ascontiguousarray(array, dtype="<f4").tobytes())
    return digest.hexdigest()


def _validate_locked(locked: Any) -> dict[str, Any]:
    if type(locked) is not dict or locked.get("format") != _LOCK_FORMAT:
        raise TypeError("locked must be an h036_acceptance_locked_v1 dict")
    if locked.get("diagnostic_only") is not True:
        raise ValueError("locked probes must be diagnostic_only")
    if locked.get("lock_sha256") != _locked_digest(locked):
        raise ValueError("locked probe SHA-256 does not match its contents")
    for label, selector in _LABEL_SELECTORS.items():
        entry = locked["probes"][label]
        if entry.get("selector") != selector:
            raise ValueError(f"probe {label} selector is invalid")
        candidate = entry["candidate"]
        raw = entry["raw_cast_candidate"]
        if entry.get("candidate_sha256") != _candidate_hash(candidate, np.dtype(np.float32)):
            raise ValueError(f"probe {label} candidate helper hash mismatch")
        if entry.get("raw_sha256") != _array_raw_sha256(candidate, "<f4"):
            raise ValueError(f"probe {label} candidate raw-byte hash mismatch")
        if entry.get("raw_cast_candidate_sha256") != _candidate_hash(raw, np.dtype(np.float32)):
            raise ValueError(f"probe {label} raw-cast helper hash mismatch")
        if entry.get("raw_cast_sha256") != _array_raw_sha256(raw, "<f4"):
            raise ValueError(f"probe {label} raw-cast byte hash mismatch")
        gate_enabled = label != "UR"
        if entry.get("post_cast_gate_enabled") is not gate_enabled:
            raise ValueError(f"probe {label} post-cast gate setting is invalid")
        if not isinstance(entry.get("post_cast_gate_pass"), bool) or not isinstance(entry.get("fallback_applied"), bool):
            raise ValueError(f"probe {label} gate metadata must be boolean")
        fallback = gate_enabled and not entry["post_cast_gate_pass"]
        if entry["fallback_applied"] is not fallback:
            raise ValueError(f"probe {label} fallback does not match the post-cast gate")
        expected_origin = 0 if fallback else entry.get("origin_index")
        if entry.get("candidate_origin_index") != expected_origin:
            raise ValueError(f"probe {label} candidate origin does not match its fallback")
        source = entry.get("source")
        if type(source) is not dict:
            raise ValueError(f"probe {label} source must be a plain dict")
        for field in (
            "selector", "origin_index", "float64_candidate_sha256", "candidate_origin_index",
            "raw_cast_candidate_sha256", "raw_cast_sha256", "post_cast_gate_enabled",
            "post_cast_gate_pass", "fallback_applied",
        ):
            if source.get(field) != entry.get(field):
                raise ValueError(f"probe {label} source.{field} differs from its locked metadata")
        if not fallback and (
            entry["candidate_sha256"] != entry["raw_cast_candidate_sha256"]
            or entry["raw_sha256"] != entry["raw_cast_sha256"]
        ):
            raise ValueError(f"probe {label} candidate differs from its raw float32 cast")
        if fallback and (
            entry["candidate_sha256"] != locked.get("c0_candidate_sha256")
            or entry["raw_sha256"] != locked.get("c0_raw_sha256")
        ):
            raise ValueError(f"probe {label} fallback candidate differs from locked C0")
    for label, entry in locked["probes"].items():
        expected_aliases = sorted(
            other for other, other_entry in locked["probes"].items()
            if entry["raw_sha256"] == other_entry["raw_sha256"]
        )
        if entry.get("aliases") != expected_aliases:
            raise ValueError(f"probe {label} alias labels do not match candidate bytes")
    return locked


def lock_acceptance_probes(training_view: TrainingView, trace: dict[str, Any]) -> dict[str, Any]:
    """Lock V, UG, and UR using only the training view and trace candidates."""
    view = _validate_training_view(training_view)
    _validate_trace(view, trace)
    diagnostics = trace["diagnostics"]
    c0_32 = np.asarray(trace["candidates"]["C0"], dtype=np.float64).astype(np.float32)
    base_new_kl = float(diagnostics["initial_metrics"]["new_train_kl"])
    entries: dict[str, dict[str, Any]] = {}
    for label, selector in _LABEL_SELECTORS.items():
        selected64 = np.asarray(trace["candidates"][selector], dtype=np.float64)
        raw_cast = selected64.astype(np.float32)
        raw_cast_kl = protected._owner_kl_diagnostics(
            view.new_features, view.new_teacher, view.new_owners, raw_cast.astype(np.float64),
        )[1]
        gate_pass = bool(raw_cast_kl <= base_new_kl + _KL_TOLERANCE)
        gate_enabled = label in {"V", "UG"}
        fallback = bool(gate_enabled and not gate_pass)
        final_candidate = c0_32.copy() if fallback else raw_cast.copy()
        final_candidate = np.ascontiguousarray(final_candidate, dtype=np.float32)
        final_candidate.setflags(write=False)
        raw_cast = np.ascontiguousarray(raw_cast, dtype=np.float32)
        raw_cast.setflags(write=False)
        entries[label] = {
            "candidate": final_candidate,
            "candidate_sha256": _candidate_hash(final_candidate, np.dtype(np.float32)),
            "raw_sha256": _array_raw_sha256(final_candidate, "<f4"),
            "raw_cast_candidate": raw_cast,
            "raw_cast_candidate_sha256": _candidate_hash(raw_cast, np.dtype(np.float32)),
            "raw_cast_sha256": _array_raw_sha256(raw_cast, "<f4"),
            "origin_index": int(trace["origin_indices"][selector]),
            "candidate_origin_index": 0 if fallback else int(trace["origin_indices"][selector]),
            "selector": selector,
            "float64_candidate_sha256": diagnostics[
                "best_float64_candidate_sha256" if selector == "F64" else "u_best_float64_candidate_sha256"
            ],
            "raw_cast_new_train_kl": float(raw_cast_kl),
            "baseline_new_train_kl": base_new_kl,
            "gate_tolerance": _KL_TOLERANCE,
            "post_cast_gate_enabled": bool(gate_enabled),
            "post_cast_gate_pass": gate_pass,
            "fallback_applied": fallback,
            "diagnostic_only": True,
            "source": {
                "selector": selector,
                "origin_index": int(trace["origin_indices"][selector]),
                "float64_candidate_sha256": diagnostics[
                    "best_float64_candidate_sha256" if selector == "F64" else "u_best_float64_candidate_sha256"
                ],
                "candidate_origin_index": 0 if fallback else int(trace["origin_indices"][selector]),
                "raw_cast_candidate_sha256": _candidate_hash(raw_cast, np.dtype(np.float32)),
                "raw_cast_sha256": _array_raw_sha256(raw_cast, "<f4"),
                "post_cast_gate_enabled": bool(gate_enabled),
                "post_cast_gate_pass": gate_pass,
                "fallback_applied": fallback,
            },
        }
    for label, entry in entries.items():
        entry["aliases"] = sorted(
            other for other, other_entry in entries.items()
            if entry["raw_sha256"] == other_entry["raw_sha256"]
        )
    locked: dict[str, Any] = {
        "format": _LOCK_FORMAT, "seed": int(view.seed), "prepared_id": view.prepared_id,
        "path_sha256": trace["path_sha256"], "trace_sha256": trace["trace_sha256"],
        "training_hash": view.training_hash, "diagnostic_only": True,
        "logical_case": json.loads(view.input_identity_json).get("logical_case"),
        "binding": "wildcard",
        "c0_candidate_sha256": _candidate_hash(c0_32, np.dtype(np.float32)),
        "c0_raw_sha256": _array_raw_sha256(c0_32, "<f4"),
        "labels": ["V", "UG", "UR"], "probes": entries,
    }
    locked["lock_sha256"] = _locked_digest(locked)
    return _validate_locked(locked)


def locked_probe_manifest(locked_probe: dict[str, Any]) -> dict[str, Any]:
    """Export the verified lock identity used by replay's cohort builder.

    Candidate arrays are deliberately omitted; both the helper identity and
    exact float32 byte identity remain bound in the per-probe metadata.
    """
    locked = _validate_locked(locked_probe)
    probes = {
        label: {
            "source": copy.deepcopy(entry["source"]),
            "candidate_sha256": entry["candidate_sha256"],
            "raw_sha256": entry["raw_sha256"],
        }
        for label, entry in locked["probes"].items()
    }
    result = {
        "seed": int(locked["seed"]),
        "prepared_id": locked["prepared_id"],
        "path_sha256": locked["path_sha256"],
        "lock_sha256": locked["lock_sha256"],
        "logical_case": locked["logical_case"],
        "binding": locked["binding"],
        "probes": probes,
    }
    _json_bytes(result)
    return result


def encode_and_score_locked_probe(
    prepared: dict[str, Any], locked_probe: dict[str, Any], label: str, *,
    stage_permit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the original complete scorer with a verified diagnostic adapter."""
    if type(prepared) is not dict:
        raise TypeError("prepared must be a plain dict")
    if prepared.get("format") != "h036_prepared_v1":
        raise ValueError("prepared.format must be h036_prepared_v1")
    if not isinstance(label, str) or label not in {"V", "UG", "UR"}:
        raise ValueError("label must be V, UG, or UR")
    locked = _validate_locked(locked_probe)
    if locked.get("prepared_id") != prepared.get("prepared_id") or locked.get("seed") != prepared.get("seed"):
        raise ValueError("locked probes do not belong to these prepared inputs")
    if stage_permit is None:
        raise PermissionError("scoring requires a replay-issued stage_permit")
    from . import retention_acceptance_replay as replay

    cohort = stage_permit.get("cohort") if type(stage_permit) is dict else None
    cases = cohort.get("cases") if type(cohort) is dict else None
    expected_case = cases.get(str(prepared["seed"])) if type(cases) is dict else None
    if type(expected_case) is not dict:
        raise ValueError("stage_permit does not embed this seed's cohort case")
    if expected_case.get("reuse") is not False:
        raise PermissionError("locked scoring is only authorized for the five recovered prefixes")
    if expected_case.get("logical_case") != locked.get("logical_case"):
        raise ValueError("locked probes do not match the cohort logical case")
    if expected_case.get("binding") != locked.get("binding"):
        raise ValueError("locked probes do not match the cohort binding")
    actual_lock_identity = locked_probe_manifest(locked)

    # Bind source-file and foundation identity to this exact prepared object.
    input_identity = prepared.get("input_identity")
    if (type(input_identity) is not dict
            or input_identity.get("logical_case") != locked.get("logical_case")):
        raise ValueError("prepared logical case differs from the locked identity")
    expected_inputs = expected_case.get("inputs")
    if (type(input_identity.get("files")) is not dict
            or type(expected_inputs) is not dict or set(expected_inputs) != set(input_identity["files"])):
        raise ValueError("prepared input files differ from the cohort case")
    for name, expected_ref in expected_inputs.items():
        actual_ref = input_identity["files"].get(name)
        actual_sha = actual_ref.get("sha256") if type(actual_ref) is dict else actual_ref
        if actual_sha != expected_ref.get("sha256"):
            raise ValueError(f"prepared input {name} SHA differs from the cohort case")
        if type(actual_ref) is not dict or actual_ref.get("bytes") != expected_ref.get("bytes"):
            raise ValueError(f"prepared input {name} byte count differs from the cohort case")
    foundation = expected_case.get("foundation")
    if type(foundation) is not dict:
        raise ValueError("cohort case is missing its foundation identity")
    if prepared.get("foundation_sha256") != foundation.get("sha256"):
        raise ValueError("prepared foundation SHA differs from the cohort case")
    if prepared.get("foundation_bytes") != foundation.get("bytes"):
        raise ValueError("prepared foundation size differs from the cohort case")

    replay.validate_stage_permit(stage_permit, actual_lock_identity, label)
    stage = stage_permit["stage"]
    entry = locked["probes"][label]
    candidate = np.array(entry["candidate"], dtype=np.float32, order="C", copy=True)
    fit_result = {
        "format": _FIT_FORMAT,
        "prepared_id": prepared["prepared_id"],
        "lambda_value": _LAMBDA,
        "candidate": candidate,
        "diagnostics": {"candidate_sha256": entry["candidate_sha256"]},
    }
    from . import retention_protected_inputs as inputs

    result = inputs.encode_and_score_stage0(prepared, fit_result, "wildcard")
    result["diagnostic_only"] = True
    result["selection_source"] = {
        "label": label, "selector": entry["selector"],
        "origin_index": entry["origin_index"],
        "candidate_origin_index": entry["candidate_origin_index"],
        "post_cast_gate_enabled": entry["post_cast_gate_enabled"],
        "post_cast_gate_pass": entry["post_cast_gate_pass"],
        "fallback_applied": entry["fallback_applied"],
        "candidate_sha256": entry["candidate_sha256"],
        "raw_sha256": entry["raw_sha256"],
        "trace_sha256": locked["trace_sha256"], "lock_sha256": locked["lock_sha256"],
        "cohort_lock_sha256": stage_permit["cohort"]["cohort_lock_sha256"],
        "stage": stage,
    }
    return result


__all__ = [
    "TrainingView", "make_training_view", "fit_selection_trace",
    "lock_acceptance_probes", "locked_probe_manifest", "encode_and_score_locked_probe",
]

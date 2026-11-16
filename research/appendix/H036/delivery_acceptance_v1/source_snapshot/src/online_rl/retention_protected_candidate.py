"""Fit one shared parent while optionally protecting an old keep prior.

The module consumes only a prepared, frozen input bundle.  It does not inspect
checkpoints, install a candidate, update an actor, use rewards, or interact with
an environment.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import numbers
import platform
import time
from typing import Any

import numpy as np
import torch

try:
    import resource as _resource
except ImportError:  # Windows
    _resource = None

from . import retention_behavioral_candidate as behavioral


_FORMAT = "h036_prepared_v1"
_FIT_FORMAT = "h036_fit_v1"
_RIDGE = behavioral._RIDGE
_MAX_ITERATIONS = 64
_MAX_EVALUATIONS = 256
_HISTORY_SIZE = 10
_TRAIN_ROWS = 96
_HOLDOUT_ROWS = 32
_ACTION_COUNT = 4
_COEFFICIENT_SHAPE = (3, 3)
_KL_TOLERANCE = 1.0e-12


class _EvaluationBudgetExhausted(RuntimeError):
    """Internal signal raised before a closure beyond the fixed budget."""


class _ClosureEvaluationError(FloatingPointError):
    """Floating-point closure failure carrying completed optimization evidence."""

    def __init__(self, message: str, partial_diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.partial_diagnostics = partial_diagnostics


def _exact_array(
    value: Any,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_features(features: Any, count: int, name: str) -> behavioral.FrozenPolicyFeatures:
    if not isinstance(features, behavioral.FrozenPolicyFeatures):
        raise TypeError(f"{name} must be FrozenPolicyFeatures")
    for field, expected_shape in (
        ("base_logits", (count, 4)),
        ("source_psi", (count, 3)),
        ("contrast_basis", (4, 3)),
    ):
        value = getattr(features, field)
        if not isinstance(value, np.ndarray) or value.dtype.kind != "f":
            raise TypeError(f"{name}.{field} must be a floating-point NumPy array")
        if value.shape != expected_shape:
            raise ValueError(f"{name}.{field} must have shape {expected_shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name}.{field} must contain only finite values")
    temperature = float(features.temperature)
    epsilon = float(features.epsilon)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name}.temperature must be finite and positive")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon < 1.0:
        raise ValueError(f"{name}.epsilon must be finite and in [0, 1)")
    return features


def _validate_teacher(value: Any, name: str, count: int) -> np.ndarray:
    teacher = _exact_array(value, name, np.dtype(np.float64), (count, _ACTION_COUNT))
    if np.any(teacher < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    sums = teacher.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{name} rows must sum to one")
    return teacher


def _validate_lambda(lambda_value: Any) -> int:
    if isinstance(lambda_value, (bool, np.bool_)) or not isinstance(lambda_value, numbers.Integral):
        raise ValueError("lambda_value must be integer zero or one, not bool")
    value = int(lambda_value)
    if value not in (0, 1):
        raise ValueError("lambda_value must be zero or one")
    return value


def _feature_payload(features: behavioral.FrozenPolicyFeatures) -> tuple[Any, ...]:
    return (
        features.base_logits,
        features.source_psi,
        features.contrast_basis,
        float(features.temperature),
        float(features.epsilon),
    )


def _update_array_hash(digest: Any, name: str, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
    digest.update(repr(contiguous.shape).encode("ascii") + b"\0")
    digest.update(contiguous.tobytes())


def _prepared_hash(prepared: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_FORMAT.encode("ascii") + b"\0")
    digest.update(str(prepared["prepared_id"]).encode("utf-8") + b"\0")
    digest.update(str(int(prepared["seed"])).encode("ascii") + b"\0")
    source_identity = prepared["input_identity"]
    if isinstance(source_identity.get("files"), dict):
        # Placement is provenance. The same bytes remain the same scientific
        # input when the validation owner relocates them to the Linux worker.
        source_identity = {
            "logical_case": source_identity.get("logical_case"),
            "files": {
                name: details["sha256"] if isinstance(details, dict) else details
                for name, details in source_identity["files"].items()
            },
        }
    else:
        source_identity = {
            key: value for key, value in source_identity.items()
            if key not in {"prefix_path", "path", "absolute_path"}
        }
    identity = json.dumps(
        source_identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest.update(identity.encode("utf-8") + b"\0")
    for key in (
        "new_observations",
        "new_owners",
        "new_effective",
        "new_teacher",
        "old_observations",
        "old_teacher",
        "new_train_indices",
        "new_holdout_indices",
        "old_train_indices",
        "old_holdout_indices",
        "mean64",
    ):
        _update_array_hash(digest, key, prepared[key])
    for key in ("new_features", "old_features"):
        for field, value in zip(
            ("base_logits", "source_psi", "contrast_basis", "temperature", "epsilon"),
            _feature_payload(prepared[key]),
        ):
            if isinstance(value, np.ndarray):
                _update_array_hash(digest, f"{key}.{field}", value)
            else:
                digest.update(f"{key}.{field}={value!r}".encode("ascii") + b"\0")
    return digest.hexdigest()


def _validate_prepared(prepared: Any) -> dict[str, Any]:
    if type(prepared) is not dict:
        raise TypeError("prepared must be a plain dict")
    required = {
        "format",
        "seed",
        "new_observations",
        "new_owners",
        "new_effective",
        "new_features",
        "new_teacher",
        "old_observations",
        "old_features",
        "old_teacher",
        "new_train_indices",
        "new_holdout_indices",
        "old_train_indices",
        "old_holdout_indices",
        "mean64",
        "prepared_id",
        "input_identity",
    }
    missing = required.difference(prepared)
    if missing:
        raise ValueError(f"prepared is missing required fields: {sorted(missing)}")
    if prepared["format"] != _FORMAT:
        raise ValueError(f"prepared.format must be {_FORMAT}")
    seed = prepared["seed"]
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError("prepared.seed must be an integer")
    if not isinstance(prepared["prepared_id"], str) or not prepared["prepared_id"]:
        raise ValueError("prepared.prepared_id must be a nonempty string")
    if type(prepared["input_identity"]) is not dict:
        raise TypeError("prepared.input_identity must be a dict")
    try:
        json.dumps(prepared["input_identity"], sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared.input_identity must be JSON serializable") from exc

    new_obs = _exact_array(prepared["new_observations"], "new_observations", np.float32, (256, 13))
    new_owners = _exact_array(prepared["new_owners"], "new_owners", np.int64, (256,))
    if not np.array_equal(new_owners, np.repeat(np.arange(2, dtype=np.int64), 128)):
        raise ValueError("new_owners must contain 128 owner-zero rows followed by 128 owner-one rows")
    _exact_array(prepared["new_effective"], "new_effective", np.float32, (2, 3, 3, 3))
    new_features = _validate_features(prepared["new_features"], 256, "new_features")
    if float(new_features.temperature) != 2.0 or float(new_features.epsilon) != 0.05:
        raise ValueError("new_features must use temperature 2.0 and epsilon 0.05")
    _validate_teacher(prepared["new_teacher"], "new_teacher", 256)

    _exact_array(prepared["old_observations"], "old_observations", np.float32, (128, 13))
    old_features = _validate_features(prepared["old_features"], 128, "old_features")
    if float(old_features.temperature) != 2.0 or float(old_features.epsilon) != 0.05:
        raise ValueError("old_features must use temperature 2.0 and epsilon 0.05")
    if not np.array_equal(new_features.contrast_basis, old_features.contrast_basis):
        raise ValueError("new_features and old_features must share the same contrast_basis")
    _validate_teacher(prepared["old_teacher"], "old_teacher", 128)

    new_train = _exact_array(prepared["new_train_indices"], "new_train_indices", np.int64, (192,))
    expected_new_train = np.concatenate((np.arange(96, dtype=np.int64), np.arange(128, 224, dtype=np.int64)))
    if not np.array_equal(new_train, expected_new_train):
        raise ValueError("new_train_indices must select the first 96 rows of each owner")
    new_holdout = _exact_array(prepared["new_holdout_indices"], "new_holdout_indices", np.int64, (64,))
    expected_new_holdout = np.concatenate((np.arange(96, 128, dtype=np.int64), np.arange(224, 256, dtype=np.int64)))
    if not np.array_equal(new_holdout, expected_new_holdout):
        raise ValueError("new_holdout_indices must select the final 32 rows of each owner")
    old_train = _exact_array(prepared["old_train_indices"], "old_train_indices", np.int64, (96,))
    if not np.array_equal(old_train, np.arange(96, dtype=np.int64)):
        raise ValueError("old_train_indices must be np.arange(96)")
    old_holdout = _exact_array(prepared["old_holdout_indices"], "old_holdout_indices", np.int64, (32,))
    if not np.array_equal(old_holdout, np.arange(96, 128, dtype=np.int64)):
        raise ValueError("old_holdout_indices must select rows 96 through 127")
    mean64 = _exact_array(prepared["mean64"], "mean64", np.float64, (3, 3))
    expected_mean = prepared["new_effective"].mean(axis=(0, 1), dtype=np.float64)
    if not np.array_equal(mean64, expected_mean):
        raise ValueError("mean64 must equal the float64 mean of new_effective")

    # Force validation of both data-dependent feature bundles here, before any
    # optimization state is created.  The observations are intentionally not
    # used to reconstruct features or teachers in this fitting-only module.
    del new_obs, new_features, old_features
    return prepared


def _slice_features(
    features: behavioral.FrozenPolicyFeatures,
    indices: np.ndarray,
) -> behavioral.FrozenPolicyFeatures:
    return behavioral.FrozenPolicyFeatures(
        base_logits=features.base_logits[indices],
        source_psi=features.source_psi[indices],
        contrast_basis=features.contrast_basis,
        temperature=float(features.temperature),
        epsilon=float(features.epsilon),
    )


def _log_forward_kl_rows(
    features: behavioral.FrozenPolicyFeatures,
    teacher: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    base = np.asarray(features.base_logits, dtype=np.float64)
    source = np.asarray(features.source_psi, dtype=np.float64)
    basis = np.asarray(features.contrast_basis, dtype=np.float64)
    contrast = source @ coefficients.T
    logits = base + contrast @ basis.T
    log_student = behavioral._log_mixed_probabilities(logits, float(features.temperature), float(features.epsilon))
    log_teacher = np.zeros_like(teacher)
    positive = teacher > 0.0
    log_teacher[positive] = np.log(teacher[positive])
    return np.sum(np.where(positive, teacher * (log_teacher - log_student), 0.0), axis=1)


def _weighted_forward_kl(
    features: behavioral.FrozenPolicyFeatures,
    teacher: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    rows = _log_forward_kl_rows(features, teacher, coefficients)
    value = float(weights @ rows)
    if not math.isfinite(value):
        raise FloatingPointError("forward KL metric is not finite")
    return value


def _owner_kl_diagnostics(
    features: behavioral.FrozenPolicyFeatures,
    teacher: np.ndarray,
    owners: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[list[float], float]:
    rows = _log_forward_kl_rows(features, teacher, coefficients)
    by_owner = [float(np.mean(rows[owners == owner])) for owner in (0, 1)]
    return by_owner, float((by_owner[0] + by_owner[1]) / 2.0)


def _ridge_value(coefficients: np.ndarray, mean64: np.ndarray) -> float:
    delta = coefficients - mean64
    return float(_RIDGE * np.mean(delta * delta))


def _objective_call_count(lambda_value: int) -> int:
    return 1 if lambda_value == 0 else 2


def protected_objective_and_gradient(
    C: Any,
    new_features: behavioral.FrozenPolicyFeatures,
    new_teacher: Any,
    new_weights: Any,
    old_features: behavioral.FrozenPolicyFeatures,
    old_teacher: Any,
    old_weights: Any,
    mean64: Any,
    lambda_value: Any,
) -> tuple[float, np.ndarray]:
    """Return ``L_new + lambda * (L_old - ridge)`` and its exact gradient.

    Each underlying behavioural objective includes one ridge term.  At lambda
    zero the original new-only objective and gradient are returned directly.
    """
    lam = _validate_lambda(lambda_value)
    candidate = behavioral._as_float_array(C, "C", _COEFFICIENT_SHAPE)
    common = behavioral._as_float_array(mean64, "mean64", _COEFFICIENT_SHAPE)
    if not isinstance(new_features, behavioral.FrozenPolicyFeatures):
        raise TypeError("new_features must be FrozenPolicyFeatures")
    if not isinstance(old_features, behavioral.FrozenPolicyFeatures):
        raise TypeError("old_features must be FrozenPolicyFeatures")
    new_count = len(np.asarray(new_features.base_logits))
    old_count = len(np.asarray(old_features.base_logits))
    behavioral._validate_features(new_features, new_count)
    behavioral._validate_features(old_features, old_count)
    if new_count == 0 or old_count == 0:
        raise ValueError("new and old objective feature sets must be nonempty")
    new_teacher_valid = behavioral._validate_probabilities(new_teacher, new_count)
    old_teacher_valid = behavioral._validate_probabilities(old_teacher, old_count)
    new_weights_valid = behavioral._as_float_array(new_weights, "new_weights", (new_count,))
    old_weights_valid = behavioral._as_float_array(old_weights, "old_weights", (old_count,))
    for name, weights in (("new_weights", new_weights_valid), ("old_weights", old_weights_valid)):
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(f"{name} must be nonnegative and sum to one")

    new_loss, new_gradient = behavioral.behavioral_objective_and_gradient(
        candidate, new_features, new_teacher_valid, new_weights_valid, common,
    )
    if lam == 0:
        return new_loss, new_gradient

    old_loss, old_gradient = behavioral.behavioral_objective_and_gradient(
        candidate, old_features, old_teacher_valid, old_weights_valid, common,
    )
    ridge = _ridge_value(candidate, common)
    ridge_gradient = (2.0 * _RIDGE / candidate.size) * (candidate - common)
    objective = float(new_loss + old_loss - ridge)
    gradient = np.asarray(new_gradient + old_gradient - ridge_gradient, dtype=np.float64)
    if not math.isfinite(objective) or not np.isfinite(gradient).all():
        raise FloatingPointError("protected objective or gradient is not finite")
    return objective, gradient


def process_memory_snapshot() -> dict[str, int | None]:
    """Return this Python process's OS-reported memory counters, when available."""
    if platform.system() == "Windows":
        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_process = kernel32.GetCurrentProcess
            get_process.argtypes = []
            get_process.restype = ctypes.c_void_p
            get_memory = psapi.GetProcessMemoryInfo
            get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
            get_memory.restype = ctypes.c_int
            ok = get_memory(
                get_process(),
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                return {"working_set_bytes": None, "ownprocess_peak_memory_bytes": None, "private_bytes": None}
            return {
                "working_set_bytes": int(counters.WorkingSetSize),
                "ownprocess_peak_memory_bytes": int(counters.PeakWorkingSetSize),
                "private_bytes": int(counters.PagefileUsage),
            }
        except (AttributeError, OSError):
            return {"working_set_bytes": None, "ownprocess_peak_memory_bytes": None, "private_bytes": None}

    try:
        if _resource is None:
            return {"working_set_bytes": None, "ownprocess_peak_memory_bytes": None, "private_bytes": None}
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        peak = int(usage.ru_maxrss)
        if platform.system() != "Darwin":
            peak *= 1024
        return {
            "working_set_bytes": None,
            "ownprocess_peak_memory_bytes": peak,
            "private_bytes": None,
        }
    except (AttributeError, OSError, ValueError):
        return {"working_set_bytes": None, "ownprocess_peak_memory_bytes": None, "private_bytes": None}


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _candidate_hash(value: np.ndarray, dtype: np.dtype[Any]) -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(repr(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _temporary_array_estimate_bytes(
    new_train_count: int,
    old_train_count: int,
    new_holdout_count: int,
    old_holdout_count: int,
) -> int:
    # This estimates NumPy closure workspace and sliced split arrays only; it
    # excludes the process high-water mark and the measured optimizer tensors.
    row_count = max(new_train_count, old_train_count)
    closure_workspace = row_count * (3 + 10 * _ACTION_COUNT + 8) * np.dtype(np.float64).itemsize
    split_storage = (
        (new_train_count + old_train_count + new_holdout_count + old_holdout_count)
        * (4 + 3 + _ACTION_COUNT + 1)
        * np.dtype(np.float64).itemsize
    )
    optimizer_vectors = (_HISTORY_SIZE * 2 + 6) * 9 * np.dtype(np.float64).itemsize
    return int(closure_workspace + split_storage + optimizer_vectors)


def fit_protected_candidate(prepared: dict[str, Any], lambda_value: Any) -> dict[str, Any]:
    """Fit a float32 shared 3 by 3 candidate from fixed H036 prepared inputs."""
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    memory_before = process_memory_snapshot()
    lam = _validate_lambda(lambda_value)
    _validate_prepared(prepared)
    input_hash_before = _prepared_hash(prepared)

    new_features_all = prepared["new_features"]
    old_features_all = prepared["old_features"]
    new_teacher_all = prepared["new_teacher"]
    old_teacher_all = prepared["old_teacher"]
    new_train_indices = prepared["new_train_indices"]
    new_holdout_indices = prepared["new_holdout_indices"]
    old_train_indices = prepared["old_train_indices"]
    old_holdout_indices = prepared["old_holdout_indices"]
    mean64 = prepared["mean64"]

    new_train_features = _slice_features(new_features_all, new_train_indices)
    new_holdout_features = _slice_features(new_features_all, new_holdout_indices)
    new_train_teacher = new_teacher_all[new_train_indices]
    new_holdout_teacher = new_teacher_all[new_holdout_indices]
    new_train_owners = prepared["new_owners"][new_train_indices]
    new_holdout_owners = prepared["new_owners"][new_holdout_indices]
    new_weights = np.full(_TRAIN_ROWS * 2, 0.5 / _TRAIN_ROWS, dtype=np.float64)
    old_train_features = _slice_features(old_features_all, old_train_indices)
    old_holdout_features = _slice_features(old_features_all, old_holdout_indices)
    old_train_teacher = old_teacher_all[old_train_indices]
    old_holdout_teacher = old_teacher_all[old_holdout_indices]
    old_train_weights = np.full(_TRAIN_ROWS, 1.0 / _TRAIN_ROWS, dtype=np.float64)
    old_holdout_weights = np.full(_HOLDOUT_ROWS, 1.0 / _HOLDOUT_ROWS, dtype=np.float64)
    new_holdout_weights = np.full(2 * _HOLDOUT_ROWS, 0.5 / _HOLDOUT_ROWS, dtype=np.float64)

    c0 = mean64.astype(np.float32).astype(np.float64)
    c0_float32 = c0.astype(np.float32)
    if not np.isfinite(c0).all() or not np.isfinite(c0_float32).all():
        raise FloatingPointError("C0 is not finite")

    objective_evaluations = 0
    metric_evaluations = 0

    # Initial full objective establishes the incumbent before any closure.
    initial_full_l, initial_gradient = protected_objective_and_gradient(
        c0,
        new_train_features,
        new_train_teacher,
        new_weights,
        old_train_features,
        old_train_teacher,
        old_train_weights,
        mean64,
        lam,
    )
    objective_evaluations += _objective_call_count(lam)
    if not math.isfinite(initial_full_l) or not np.isfinite(initial_gradient).all():
        raise FloatingPointError("initial objective or gradient is not finite")
    initial_ridge = _ridge_value(c0, mean64)
    initial_new_train_by_owner, initial_new_train_kl = _owner_kl_diagnostics(
        new_train_features, new_train_teacher, new_train_owners, c0,
    )
    initial_old_train_kl = _weighted_forward_kl(
        old_train_features, old_train_teacher, old_train_weights, c0,
    )
    initial_new_holdout_by_owner, initial_new_holdout_kl = _owner_kl_diagnostics(
        new_holdout_features, new_holdout_teacher, new_holdout_owners, c0,
    )
    initial_old_holdout_kl = _weighted_forward_kl(
        old_holdout_features, old_holdout_teacher, old_holdout_weights, c0,
    )
    metric_evaluations += 4
    if not all(
        math.isfinite(value)
        for value in (
            initial_ridge,
            initial_new_train_kl,
            initial_old_train_kl,
            initial_new_holdout_kl,
            initial_old_holdout_kl,
        )
    ):
        raise FloatingPointError("initial KL or ridge metric is not finite")

    best_candidate = c0.copy()
    best_full_l = float(initial_full_l)
    best_new_train_kl = float(initial_new_train_kl)
    best_old_train_kl = float(initial_old_train_kl)
    closure_records: list[dict[str, Any]] = []
    closure_evaluations = 0
    parameter = torch.nn.Parameter(torch.as_tensor(c0, dtype=torch.float64).clone())
    optimizer = torch.optim.LBFGS(
        [parameter],
        lr=1.0,
        max_iter=_MAX_ITERATIONS,
        max_eval=_MAX_EVALUATIONS,
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-12,
        history_size=_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations, objective_evaluations, metric_evaluations
        nonlocal best_candidate, best_full_l, best_new_train_kl, best_old_train_kl
        if closure_evaluations >= _MAX_EVALUATIONS:
            raise _EvaluationBudgetExhausted("L-BFGS closure evaluation budget exhausted")
        closure_evaluations += 1
        current = parameter.detach().cpu().numpy().astype(np.float64, copy=True)
        try:
            full_l, gradient = protected_objective_and_gradient(
                current,
                new_train_features,
                new_train_teacher,
                new_weights,
                old_train_features,
                old_train_teacher,
                old_train_weights,
                mean64,
                lam,
            )
            objective_evaluations += _objective_call_count(lam)
            new_kl = _owner_kl_diagnostics(
                new_train_features, new_train_teacher, new_train_owners, current,
            )[1]
            old_kl = _weighted_forward_kl(old_train_features, old_train_teacher, old_train_weights, current)
            metric_evaluations += 2
            ridge = _ridge_value(current, mean64)
            if (
                not math.isfinite(full_l)
                or not np.isfinite(gradient).all()
                or not math.isfinite(new_kl)
                or not math.isfinite(old_kl)
                or not math.isfinite(ridge)
            ):
                raise FloatingPointError("closure objective, gradient, KL, or ridge is not finite")
        except FloatingPointError as exc:
            partial = {
                "closure_records": list(closure_records),
                "attempted_closure_count": int(closure_evaluations),
                "reason": str(exc),
                "attempted_candidate_hash": _candidate_hash(current, np.dtype(np.float64)),
            }
            raise _ClosureEvaluationError(str(exc), partial) from exc

        accepted = False
        if new_kl > initial_new_train_kl + _KL_TOLERANCE:
            reject_reason = "new_train_kl_gate"
        elif full_l <= best_full_l:
            accepted = True
            reject_reason = None
            best_candidate = current.copy()
            best_full_l = float(full_l)
            best_new_train_kl = float(new_kl)
            best_old_train_kl = float(old_kl)
        else:
            reject_reason = "full_objective_above_incumbent"
        closure_records.append(
            {
                "index": int(closure_evaluations),
                "fullL": float(full_l),
                "newKL": float(new_kl),
                "oldKL": float(old_kl),
                "ridge": float(ridge),
                "accepted": bool(accepted),
                "reject_reason": reject_reason,
                "candidate_hash": _candidate_hash(current, np.dtype(np.float64)),
            }
        )
        parameter.grad = torch.as_tensor(gradient, dtype=parameter.dtype, device=parameter.device)
        return torch.as_tensor(full_l, dtype=parameter.dtype, device=parameter.device)

    termination_reason = "optimizer_completed"
    try:
        optimizer.step(closure)
    except _EvaluationBudgetExhausted:
        termination_reason = "evaluation_budget_exhausted"
    optimizer_state = optimizer.state.get(parameter, {})
    iterations = int(optimizer_state.get("n_iter", 0))
    optimizer_state_tensor_bytes = _tensor_bytes(optimizer_state)

    # The original new-train KL rule is the only post-cast fallback gate.
    selected_float32 = best_candidate.astype(np.float32)
    selected_float32_new_train_kl = _owner_kl_diagnostics(
        new_train_features,
        new_train_teacher,
        new_train_owners,
        selected_float32.astype(np.float64),
    )[1]
    metric_evaluations += 1
    if not math.isfinite(selected_float32_new_train_kl):
        raise FloatingPointError("float32 new-train KL is not finite")
    fallback_applied = selected_float32_new_train_kl > initial_new_train_kl + _KL_TOLERANCE
    fallback_reason = "float32_new_train_kl_gate" if fallback_applied else None
    candidate32 = c0_float32.copy() if fallback_applied else selected_float32.copy()
    returned_is_c0 = bool(np.array_equal(candidate32, c0_float32))

    final64 = candidate32.astype(np.float64)
    final_full_l, final_gradient = protected_objective_and_gradient(
        final64,
        new_train_features,
        new_train_teacher,
        new_weights,
        old_train_features,
        old_train_teacher,
        old_train_weights,
        mean64,
        lam,
    )
    objective_evaluations += _objective_call_count(lam)
    final_new_train_by_owner, final_new_train_kl = _owner_kl_diagnostics(
        new_train_features, new_train_teacher, new_train_owners, final64,
    )
    final_old_train_kl = _weighted_forward_kl(
        old_train_features, old_train_teacher, old_train_weights, final64,
    )
    final_new_holdout_by_owner, final_new_holdout_kl = _owner_kl_diagnostics(
        new_holdout_features, new_holdout_teacher, new_holdout_owners, final64,
    )
    final_old_holdout_kl = _weighted_forward_kl(
        old_holdout_features, old_holdout_teacher, old_holdout_weights, final64,
    )
    metric_evaluations += 4
    final_ridge = _ridge_value(final64, mean64)
    if (
        not math.isfinite(final_full_l)
        or not np.isfinite(final_gradient).all()
        or not all(
            math.isfinite(value)
            for value in (final_new_train_kl, final_old_train_kl, final_new_holdout_kl, final_old_holdout_kl, final_ridge)
        )
    ):
        raise FloatingPointError("final objective, gradient, KL, or ridge is not finite")

    input_hash_after = _prepared_hash(prepared)
    input_hash_unchanged = input_hash_before == input_hash_after
    if not input_hash_unchanged:
        raise RuntimeError("prepared input changed during fitting")

    teacher_array_bytes = int(new_teacher_all.nbytes + old_teacher_all.nbytes)
    temporary_estimate = _temporary_array_estimate_bytes(
        len(new_train_indices), len(old_train_indices), len(new_holdout_indices), len(old_holdout_indices),
    )
    memory_after = process_memory_snapshot()
    cpu_elapsed = max(0.0, time.process_time() - cpu_start)
    wall_elapsed = max(0.0, time.perf_counter() - wall_start)
    closures_term = int(closure_evaluations * _objective_call_count(lam))
    total_objective_evaluations = int(objective_evaluations)
    total_kl_metric_evaluations = int(metric_evaluations)
    diagnostics: dict[str, Any] = {
        "optimizer": "torch_lbfgs_strong_wolfe",
        "optimizer_max_iterations": _MAX_ITERATIONS,
        "optimizer_max_evaluations": _MAX_EVALUATIONS,
        "optimizer_history_size": _HISTORY_SIZE,
        "optimizer_learning_rate": 1.0,
        "optimizer_tolerance_grad": 1.0e-9,
        "optimizer_tolerance_change": 1.0e-12,
        "closure_count": int(closure_evaluations),
        "iterations": int(iterations),
        "termination_reason": termination_reason,
        "closure_records": closure_records,
        "initial_fullL": float(initial_full_l),
        "best_float64_candidate": best_candidate.tolist(),
        "best_float64_candidate_sha256": _candidate_hash(best_candidate, np.dtype(np.float64)),
        "best_float64_fullL": float(best_full_l),
        "best_float64_new_train_kl": float(best_new_train_kl),
        "best_float64_old_train_kl": float(best_old_train_kl),
        "selected_float32_candidate_sha256": _candidate_hash(selected_float32, np.dtype(np.float32)),
        "selected_float32_new_train_kl": float(selected_float32_new_train_kl),
        "final_float32_fullL": float(final_full_l),
        "final_float32_ridge": float(final_ridge),
        "candidate_sha256": _candidate_hash(candidate32, np.dtype(np.float32)),
        "returned_is_C0": returned_is_c0,
        "fallback_applied": bool(fallback_applied),
        "fallback_reason": fallback_reason,
        "initial_metrics": {
            "new_train_kl_by_owner": initial_new_train_by_owner,
            "new_train_kl": float(initial_new_train_kl),
            "old_train_kl": float(initial_old_train_kl),
            "new_holdout_kl_by_owner": initial_new_holdout_by_owner,
            "new_holdout_kl": float(initial_new_holdout_kl),
            "old_holdout_kl": float(initial_old_holdout_kl),
            "ridge": float(initial_ridge),
        },
        "final_metrics": {
            "new_train_kl_by_owner": final_new_train_by_owner,
            "new_train_kl": float(final_new_train_kl),
            "old_train_kl": float(final_old_train_kl),
            "new_holdout_kl_by_owner": final_new_holdout_by_owner,
            "new_holdout_kl": float(final_new_holdout_kl),
            "old_holdout_kl": float(final_old_holdout_kl),
        },
        "computation_counts": {
            "initialization_objective_evaluations": int(_objective_call_count(lam)),
            "closure_objective_evaluations": int(closures_term),
            "final_objective_evaluations": int(_objective_call_count(lam)),
            "total_objective_evaluations": total_objective_evaluations,
            "initialization_kl_metric_evaluations": 4,
            "closure_kl_metric_evaluations": int(2 * closure_evaluations),
            "float32_gate_kl_metric_evaluations": 1,
            "final_kl_metric_evaluations": 4,
            "total_kl_metric_evaluations": total_kl_metric_evaluations,
            "total_objective_and_kl_evaluations": int(total_objective_evaluations + total_kl_metric_evaluations),
        },
        "ridge": float(_RIDGE),
        "teacher_array_bytes": teacher_array_bytes,
        "optimizer_state_tensor_bytes": int(optimizer_state_tensor_bytes),
        "temporary_array_bytes_estimate": temporary_estimate,
        "temporary_array_estimate_scope": "NumPy closure workspace and split arrays; excludes measured optimizer tensors and process peak",
        "memory_before": memory_before,
        "memory_after": memory_after,
        "ownprocess_peak_memory_bytes": memory_after["ownprocess_peak_memory_bytes"],
        "ownprocess_peak_memory_scope": "OS process high-water mark for this process only; not an isolated fit peak",
        "cpu_time_seconds": float(cpu_elapsed),
        "wall_time_seconds": float(wall_elapsed),
        "input_hash_before": input_hash_before,
        "input_hash_after": input_hash_after,
        "input_hash_unchanged": input_hash_unchanged,
        "prepared_id": prepared["prepared_id"],
        "seed": int(prepared["seed"]),
        "new_train_rows": int(len(new_train_indices)),
        "new_holdout_rows": int(len(new_holdout_indices)),
        "old_train_rows": int(len(old_train_indices)),
        "old_holdout_rows": int(len(old_holdout_indices)),
        "persistent_candidate_bytes": int(candidate32.nbytes),
    }
    return {
        "format": _FIT_FORMAT,
        "lambda_value": lam,
        "candidate": candidate32,
        "prepared_id": prepared["prepared_id"],
        "diagnostics": diagnostics,
    }


__all__ = [
    "fit_protected_candidate",
    "process_memory_snapshot",
    "protected_objective_and_gradient",
]

"""Conservative H037 same-group parent handoff candidate.

The function in this module builds and scores an encoded actor candidate. It is
deliberately side-effect free: it does not install the result into ``core`` or
modify the caller's arrays, evidence, or random-number-generator states.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import math
import random
from numbers import Integral, Real
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn import functional as F

from .retention_behavioral_candidate import (
    FrozenPolicyFeatures,
    extract_frozen_policy_features,
)
from .write_decision_policy import decode_actor, encode_actor


_SLOT_COUNT = 2
_BAND_COUNT = 3
_ACTION_COUNT = 4
_LOCAL_WIDTH = 27
_SHARED_WIDTH = 9
_OBSERVATION_WIDTH = 13
_REQUIRED_CONFIG = frozenset(
    {
        "lambda_old",
        "ridge",
        "max_iter",
        "max_eval",
        "delta_v",
        "epsilon_old",
        "epsilon_res",
        "compensation_atol",
        "codec",
        "actor_budget_bytes",
    }
)
_LBFGS_HISTORY_SIZE = 10


class _ClosureBudgetExceeded(RuntimeError):
    """Raised if torch LBFGS requests more closures than the caller allowed."""


def _failure(reason: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "reject",
        "accepted": False,
        "reason": reason,
        "actor_bytes": None,
        "metrics": {} if metrics is None else metrics,
        "error": None,
    }


def _execution_error(exc: Exception, metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a structured execution failure, never an ordinary gate refusal."""
    return {
        "status": "error",
        "accepted": False,
        "reason": f"{type(exc).__name__}: {exc}",
        "actor_bytes": None,
        "metrics": metrics,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _float32_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if shape is not None and raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float32).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be representable as finite float32 values")
    return result


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _real(value: Any, name: str, *, minimum: float, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= minimum if strict else result < minimum:
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("config must be a mapping with all required H037 settings")
    provided = set(value.keys())
    missing = sorted(_REQUIRED_CONFIG - provided)
    unexpected = sorted(provided - _REQUIRED_CONFIG)
    if missing:
        raise ValueError(f"config is missing required keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"config contains unsupported keys: {', '.join(map(str, unexpected))}")

    result: dict[str, Any] = {
        "lambda_old": _real(value["lambda_old"], "lambda_old", minimum=0.0),
        "ridge": _real(value["ridge"], "ridge", minimum=0.0),
        "max_iter": _integer(value["max_iter"], "max_iter", minimum=1),
        "max_eval": _integer(value["max_eval"], "max_eval", minimum=1),
        "delta_v": _real(value["delta_v"], "delta_v", minimum=0.0, strict=True),
        "epsilon_old": _real(value["epsilon_old"], "epsilon_old", minimum=0.0),
        "epsilon_res": _real(value["epsilon_res"], "epsilon_res", minimum=0.0),
        "compensation_atol": _real(
            value["compensation_atol"], "compensation_atol", minimum=0.0, strict=True
        ),
        "actor_budget_bytes": _integer(value["actor_budget_bytes"], "actor_budget_bytes", minimum=0),
    }
    codec = value["codec"]
    if not isinstance(codec, str) or codec not in {"float32", "int8"}:
        raise ValueError("codec must be 'float32' or 'int8'")
    result["codec"] = codec
    return result


def _validate_observations(value: Any, name: str) -> np.ndarray:
    observations = _float32_array(value, name)
    if observations.ndim != 2 or observations.shape[1] != _OBSERVATION_WIDTH:
        raise ValueError(f"{name} must have shape (N, {_OBSERVATION_WIDTH})")
    if observations.shape[0] == 0:
        raise ValueError(f"{name} must be nonempty")
    return observations


def _routed_observations(observations: np.ndarray, slot: int) -> np.ndarray:
    """Return a copy whose route is a valid, one-based local slot identifier."""
    routed = observations.copy()
    routed[:, -1] = np.float32(slot + 1)
    return routed


def _bands(observations: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(observations[:, 0], 0.0, 1.0) * _BAND_COUNT).astype(np.int64).clip(0, 2)


def _feature_arrays(
    features: FrozenPolicyFeatures,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not isinstance(features, FrozenPolicyFeatures):
        raise TypeError("frozen feature extraction returned an unexpected value")
    base = np.asarray(features.base_logits, dtype=np.float64)
    psi = np.asarray(features.source_psi, dtype=np.float64)
    basis = np.asarray(features.contrast_basis, dtype=np.float64)
    if base.shape != (count, _ACTION_COUNT):
        raise ValueError(f"base_logits must have shape ({count}, {_ACTION_COUNT})")
    if psi.shape != (count, 3):
        raise ValueError(f"source_psi must have shape ({count}, 3)")
    if basis.shape != (_ACTION_COUNT, 3):
        raise ValueError("contrast_basis must have shape (4, 3)")
    if not all(np.isfinite(array).all() for array in (base, psi, basis)):
        raise ValueError("frozen policy features must be finite")
    temperature = float(features.temperature)
    epsilon = float(features.epsilon)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("core temperature must be finite and positive")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon < 1.0:
        raise ValueError("core epsilon must be finite and in [0, 1)")
    return base, psi, basis, temperature, epsilon


def _log_mixed_probabilities(logits: np.ndarray, temperature: float, epsilon: float) -> np.ndarray:
    scaled = logits / temperature
    maximum = np.max(scaled, axis=1, keepdims=True)
    shifted = scaled - maximum
    log_softmax = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    if epsilon == 0.0:
        return log_softmax
    return np.logaddexp(
        math.log1p(-epsilon) + log_softmax,
        math.log(epsilon / _ACTION_COUNT),
    )


def _log_probabilities_for_coefficients(
    features: FrozenPolicyFeatures,
    coefficients_by_row: np.ndarray,
) -> np.ndarray:
    count = len(coefficients_by_row)
    base, psi, basis, temperature, epsilon = _feature_arrays(features, count)
    coefficients = np.asarray(coefficients_by_row, dtype=np.float64)
    if coefficients.shape != (count, 3, 3) or not np.isfinite(coefficients).all():
        raise ValueError("row coefficients must be finite with shape (N, 3, 3)")
    contrast = np.einsum("nij,nj->ni", coefficients, psi)
    logits = base + contrast @ basis.T
    if not np.isfinite(logits).all():
        raise FloatingPointError("policy logits are non-finite")
    return _log_mixed_probabilities(logits, temperature, epsilon)


def _probabilities_for_coefficients(
    features: FrozenPolicyFeatures,
    coefficients_by_row: np.ndarray,
) -> np.ndarray:
    probabilities = np.exp(_log_probabilities_for_coefficients(features, coefficients_by_row))
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("policy probabilities are non-finite")
    return probabilities


def _effective_rows(local_matrix: np.ndarray, parent: np.ndarray, observations: np.ndarray) -> np.ndarray:
    band_rows = _bands(observations)
    local_by_band = np.asarray(local_matrix, dtype=np.float64).reshape(3, 3, 3)
    return local_by_band[band_rows] + np.asarray(parent, dtype=np.float64).reshape(1, 3, 3)


def _teacher_probabilities(
    features: FrozenPolicyFeatures,
    local_matrix: np.ndarray,
    parent: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    return _probabilities_for_coefficients(
        features,
        _effective_rows(local_matrix, parent, observations),
    )


def _mean_forward_kl(teacher: np.ndarray, student_log_probabilities: np.ndarray) -> float:
    teacher = np.asarray(teacher, dtype=np.float64)
    student_log = np.asarray(student_log_probabilities, dtype=np.float64)
    if teacher.ndim != 2 or teacher.shape[1] != _ACTION_COUNT or student_log.shape != teacher.shape:
        raise ValueError("teacher and student probabilities must have matching (N, 4) shapes")
    if len(teacher) == 0 or not np.isfinite(teacher).all() or not np.isfinite(student_log).all():
        raise ValueError("KL inputs must be nonempty and finite")
    if np.any(teacher < 0.0) or not np.allclose(teacher.sum(axis=1), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("teacher rows must be valid action distributions")
    positive = teacher > 0.0
    teacher_log = np.zeros_like(teacher)
    np.log(teacher, out=teacher_log, where=positive)
    terms = np.zeros_like(teacher)
    terms[positive] = teacher[positive] * (teacher_log[positive] - student_log[positive])
    value = float(np.mean(np.sum(terms, axis=1)))
    if not math.isfinite(value):
        raise FloatingPointError("mean forward KL is non-finite")
    return value


def _torch_features(
    features: FrozenPolicyFeatures,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    count = len(features.base_logits)
    base, psi, basis, temperature, epsilon = _feature_arrays(features, count)
    return (
        torch.as_tensor(base, dtype=torch.float64),
        torch.as_tensor(psi, dtype=torch.float64),
        torch.as_tensor(basis, dtype=torch.float64),
        temperature,
        epsilon,
    )


def _torch_log_probabilities(
    coefficients: torch.Tensor,
    features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float],
) -> torch.Tensor:
    base, psi, basis, temperature, epsilon = features
    contrast = psi @ coefficients.transpose(0, 1)
    logits = base + contrast @ basis.transpose(0, 1)
    log_softmax = F.log_softmax(logits / temperature, dim=1)
    if epsilon == 0.0:
        return log_softmax
    uniform_log = torch.full_like(log_softmax, math.log(epsilon / _ACTION_COUNT))
    return torch.logaddexp(log_softmax + math.log1p(-epsilon), uniform_log)


def _torch_mean_forward_kl(teacher: torch.Tensor, student_log: torch.Tensor) -> torch.Tensor:
    positive = teacher > 0.0
    safe_teacher_log = torch.where(positive, torch.log(teacher.clamp_min(torch.finfo(teacher.dtype).tiny)), 0.0)
    terms = torch.where(positive, teacher * (safe_teacher_log - student_log), 0.0)
    return terms.sum(dim=1).mean()


def _fit_parent(
    initial_parent: np.ndarray,
    victim_features: FrozenPolicyFeatures,
    victim_teacher: np.ndarray,
    old_features: FrozenPolicyFeatures,
    old_teacher: np.ndarray,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> np.ndarray:
    victim_inputs = _torch_features(victim_features)
    old_inputs = _torch_features(old_features)
    victim_target = torch.as_tensor(victim_teacher, dtype=torch.float64)
    old_target = torch.as_tensor(old_teacher, dtype=torch.float64)
    c0 = np.asarray(initial_parent, dtype=np.float64).reshape(3, 3)
    parameter = torch.nn.Parameter(torch.as_tensor(c0.copy(), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [parameter],
        lr=1.0,
        max_iter=config["max_iter"],
        max_eval=config["max_eval"],
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-12,
        history_size=_LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations
        if closure_evaluations >= config["max_eval"]:
            raise _ClosureBudgetExceeded("L-BFGS closure evaluation budget exhausted")
        closure_evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        victim_log = _torch_log_probabilities(parameter, victim_inputs)
        old_log = _torch_log_probabilities(parameter, old_inputs)
        victim_kl = _torch_mean_forward_kl(victim_target, victim_log)
        old_kl = _torch_mean_forward_kl(old_target, old_log)
        ridge = config["ridge"] * torch.mean((parameter - torch.as_tensor(c0, dtype=parameter.dtype)) ** 2)
        objective = victim_kl + config["lambda_old"] * old_kl + ridge
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError("L-BFGS objective became non-finite")
        objective.backward()
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
            raise FloatingPointError("L-BFGS gradient became non-finite")
        return objective

    try:
        optimizer.step(closure)
    except Exception:
        state = optimizer.state.get(parameter, {})
        metrics["iterations"] = int(state.get("n_iter", 0))
        metrics["closure_evaluations"] = int(closure_evaluations)
        raise
    state = optimizer.state.get(parameter, {})
    metrics["iterations"] = int(state.get("n_iter", 0))
    metrics["closure_evaluations"] = int(closure_evaluations)
    result = parameter.detach().cpu().numpy().astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise FloatingPointError("fitted parent is non-finite after float32 conversion")
    return result.reshape(3, 3)


@contextmanager
def _preserve_global_rng(core: Any) -> Iterator[None]:
    """Restore global RNG state even if a core forward uses a stochastic layer."""
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_cpu_state = torch.random.get_rng_state()
    device = getattr(getattr(core, "source_basis", None), "device", None)
    torch_cuda_state = None
    if device is not None and getattr(device, "type", None) == "cuda":
        torch_cuda_state = torch.cuda.get_rng_state(device)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_cpu_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state(torch_cuda_state, device)


def _max_resident_drift(
    resident_features: dict[int, FrozenPolicyFeatures],
    resident_observations: dict[int, np.ndarray],
    resident_teachers: dict[int, np.ndarray],
    decoded_local: np.ndarray,
    decoded_shared: np.ndarray,
) -> float:
    maximum = 0.0
    for slot in sorted(resident_features):
        teacher = resident_teachers[slot]
        effective = _effective_rows(
            decoded_local[slot],
            decoded_shared[0],
            resident_observations[slot],
        )
        student = _probabilities_for_coefficients(resident_features[slot], effective)
        if not np.isfinite(student).all():
            raise FloatingPointError(f"resident slot {slot} probabilities are non-finite")
        drift = float(np.max(np.abs(student - teacher)))
        if not math.isfinite(drift):
            raise FloatingPointError(f"resident slot {slot} probability drift is non-finite")
        maximum = max(maximum, drift)
    return maximum


def _propose_impl(
    core: Any,
    *,
    victim_slot: Any,
    local: Any,
    shared: Any,
    groups: Any,
    evidence: Any,
    config: Any,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    settings = _validate_config(config)
    slot = _integer(victim_slot, "victim_slot", minimum=0)
    if slot >= _SLOT_COUNT:
        raise ValueError("victim_slot must be 0 or 1")

    local32 = _float32_array(local, "local", (_SLOT_COUNT, _LOCAL_WIDTH))
    shared32 = _float32_array(shared, "shared", (1, _SHARED_WIDTH))
    raw_groups = np.asarray(groups)
    if raw_groups.shape != (_SLOT_COUNT,) or raw_groups.dtype.kind not in "iu":
        raise ValueError("groups must be an integer array with shape (2,)")
    group_values = [int(value) for value in raw_groups.tolist()]
    if any(value not in (-1, 0) for value in group_values):
        raise ValueError("groups values must be -1 or 0")
    groups8 = np.asarray(group_values, dtype=np.int64)
    if groups8[slot] != 0:
        raise ValueError("victim slot must be bound to group 0")
    resident_slots = [index for index in range(_SLOT_COUNT) if index != slot and groups8[index] == 0]
    if not resident_slots:
        raise ValueError("H037 requires at least one bound non-victim resident")

    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be a mapping")
    required_evidence = {"victim_train", "victim_holdout", "old_train", "old_holdout", "resident_holdout"}
    missing_evidence = sorted(required_evidence - set(evidence.keys()))
    if missing_evidence:
        raise ValueError(f"evidence is missing required keys: {', '.join(missing_evidence)}")
    observations = {
        name: _validate_observations(evidence[name], name)
        for name in ("victim_train", "victim_holdout", "old_train", "old_holdout")
    }
    resident_raw = evidence["resident_holdout"]
    if not isinstance(resident_raw, Mapping) or len(resident_raw) == 0:
        raise ValueError("resident_holdout must be a nonempty mapping")
    resident_keys: set[int] = set()
    for key in resident_raw.keys():
        parsed = _integer(key, "resident_holdout slot", minimum=0)
        if parsed >= _SLOT_COUNT:
            raise ValueError("resident_holdout contains a slot outside [0, 1]")
        resident_keys.add(parsed)
    if resident_keys != set(resident_slots):
        raise ValueError("resident_holdout must cover exactly every bound non-victim slot")
    resident_observations = {
        int(key): _validate_observations(value, f"resident_holdout[{int(key)}]")
        for key, value in resident_raw.items()
    }

    parent0 = shared32[0].reshape(3, 3).copy()
    local_matrices = local32.reshape(_SLOT_COUNT, _BAND_COUNT, 3, 3).copy()

    # Historical route fields describe their sampling-time owner. Re-route only
    # local observation copies to a currently valid slot before frozen scoring.
    victim_route = slot
    routed_observations = {
        name: (_routed_observations(value, victim_route) if name.startswith("victim_") else value.copy())
        for name, value in observations.items()
    }
    # Old-prior anchors are explicitly sourced from current resident slots.
    # Keep their one-based owner routes rather than silently relabeling them as
    # the victim; current W0 ignores route in its feature net, but future
    # implementations must not rely on that incidental property.
    for name in ("old_train", "old_holdout"):
        routes = routed_observations[name][:, -1]
        allowed_routes = np.asarray([index + 1 for index in resident_slots], dtype=np.float32)
        if not np.isin(routes, allowed_routes).all():
            raise ValueError(f"{name} must carry valid resident routes")
    routed_resident_observations = {
        resident_slot: _routed_observations(value, resident_slot)
        for resident_slot, value in resident_observations.items()
    }
    features = {
        name: extract_frozen_policy_features(core, value)
        for name, value in routed_observations.items()
    }
    resident_features = {
        resident_slot: extract_frozen_policy_features(core, value)
        for resident_slot, value in routed_resident_observations.items()
    }

    victim_teacher_train = _teacher_probabilities(
        features["victim_train"], local_matrices[slot], parent0, routed_observations["victim_train"]
    )
    old_teacher_train = _teacher_probabilities(
        features["old_train"], np.zeros((3, 3, 3), dtype=np.float64), parent0,
        routed_observations["old_train"],
    )
    victim_teacher_holdout = _teacher_probabilities(
        features["victim_holdout"], local_matrices[slot], parent0, routed_observations["victim_holdout"]
    )
    old_teacher_holdout = _teacher_probabilities(
        features["old_holdout"], np.zeros((3, 3, 3), dtype=np.float64), parent0,
        routed_observations["old_holdout"],
    )
    resident_teachers = {
        resident_slot: _teacher_probabilities(
            resident_features[resident_slot], local_matrices[resident_slot], parent0,
            routed_resident_observations[resident_slot],
        )
        for resident_slot in resident_slots
    }

    # Explicit no-write baseline: clear the victim, then use the same real
    # actor encoder and decoder as the fitted candidate.
    no_write_local = local32.copy()
    no_write_local[slot] = np.float32(0.0)
    baseline_actor = encode_actor(
        no_write_local,
        shared32.copy(),
        groups8.copy(),
        codec=settings["codec"],
        budget_bytes=settings["actor_budget_bytes"],
    ).data
    baseline_local, baseline_shared, baseline_groups = decode_actor(baseline_actor)
    if not np.array_equal(baseline_groups.astype(np.int64), groups8):
        raise ValueError("baseline codec changed group bindings")
    metrics["baseline_actor_nbytes"] = len(baseline_actor)
    baseline_victim_kl = _mean_forward_kl(
        victim_teacher_holdout,
        _log_probabilities_for_coefficients(
            features["victim_holdout"],
            _effective_rows(baseline_local[slot], baseline_shared[0], routed_observations["victim_holdout"]),
        ),
    )
    baseline_old_kl = _mean_forward_kl(
        old_teacher_holdout,
        _log_probabilities_for_coefficients(
            features["old_holdout"],
            np.broadcast_to(baseline_shared[0].reshape(1, 3, 3), (len(routed_observations["old_holdout"]), 3, 3)),
        ),
    )
    baseline_resident_drift = _max_resident_drift(
        resident_features,
        routed_resident_observations,
        resident_teachers,
        baseline_local,
        baseline_shared,
    )

    candidate_parent = _fit_parent(
        parent0,
        features["victim_train"],
        victim_teacher_train,
        features["old_train"],
        old_teacher_train,
        settings,
        metrics,
    )

    candidate_local = local32.copy()
    candidate_local[slot] = np.float32(0.0)
    candidate_shared = shared32.copy()
    candidate_shared[0] = candidate_parent.reshape(_SHARED_WIDTH)
    compensation_error = 0.0
    for resident_slot in resident_slots:
        original_effective = (
            local32[resident_slot].reshape(3, 3, 3).astype(np.float64)
            + parent0.astype(np.float64)[None, :, :]
        )
        compensated = (
            local32[resident_slot].reshape(3, 3, 3)
            + shared32[0].reshape(1, 3, 3)
            - candidate_parent.reshape(1, 3, 3)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            candidate_local[resident_slot] = compensated.reshape(_LOCAL_WIDTH)
        if not np.isfinite(candidate_local[resident_slot]).all():
            raise FloatingPointError(f"resident slot {resident_slot} compensation is non-finite")
        compensated_effective = (
            candidate_local[resident_slot].reshape(3, 3, 3).astype(np.float64)
            + candidate_parent.astype(np.float64)[None, :, :]
        )
        compensation_error = max(
            compensation_error,
            float(np.max(np.abs(compensated_effective - original_effective))),
        )
    metrics["pre_codec_compensation_max_abs_error"] = compensation_error
    if compensation_error > settings["compensation_atol"]:
        return _failure("pre-codec resident compensation error exceeds compensation_atol", metrics)

    candidate_actor = encode_actor(
        candidate_local,
        candidate_shared,
        groups8.copy(),
        codec=settings["codec"],
        budget_bytes=settings["actor_budget_bytes"],
    ).data
    decoded_local, decoded_shared, decoded_groups = decode_actor(candidate_actor)
    if not np.array_equal(decoded_groups.astype(np.int64), groups8):
        raise ValueError("candidate codec changed group bindings")

    candidate_victim_kl = _mean_forward_kl(
        victim_teacher_holdout,
        _log_probabilities_for_coefficients(
            features["victim_holdout"],
            _effective_rows(decoded_local[slot], decoded_shared[0], routed_observations["victim_holdout"]),
        ),
    )
    candidate_old_kl = _mean_forward_kl(
        old_teacher_holdout,
        _log_probabilities_for_coefficients(
            features["old_holdout"],
            np.broadcast_to(decoded_shared[0].reshape(1, 3, 3), (len(routed_observations["old_holdout"]), 3, 3)),
        ),
    )
    candidate_resident_drift = _max_resident_drift(
        resident_features,
        routed_resident_observations,
        resident_teachers,
        decoded_local,
        decoded_shared,
    )
    victim_improvement = baseline_victim_kl - candidate_victim_kl
    metrics.update(
        {
            "iterations": int(metrics.get("iterations", 0)),
            "closure_evaluations": int(metrics.get("closure_evaluations", 0)),
            "baseline_victim_holdout_kl": baseline_victim_kl,
            "candidate_victim_holdout_kl": candidate_victim_kl,
            "victim_holdout_kl_improvement": victim_improvement,
            "baseline_old_holdout_kl": baseline_old_kl,
            "candidate_old_holdout_kl": candidate_old_kl,
            "candidate_old_holdout_kl_delta_vs_no_write": candidate_old_kl - baseline_old_kl,
            "baseline_resident_max_probability_drift": baseline_resident_drift,
            "candidate_resident_max_probability_drift": candidate_resident_drift,
            "candidate_resident_drift_delta_vs_no_write": candidate_resident_drift - baseline_resident_drift,
            "candidate_actor_nbytes": len(candidate_actor),
            "acceptance_gate_domain": "encoded_actor_roundtrip",
        }
    )

    failed_gates: list[str] = []
    if victim_improvement < settings["delta_v"]:
        failed_gates.append("victim holdout KL improvement below delta_v")
    if candidate_old_kl > settings["epsilon_old"]:
        failed_gates.append("old holdout KL exceeds epsilon_old")
    if candidate_resident_drift > settings["epsilon_res"]:
        failed_gates.append("resident probability drift exceeds epsilon_res")
    if failed_gates:
        return _failure("; ".join(failed_gates), metrics)

    return {
        "status": "accept",
        "accepted": True,
        "reason": "accepted",
        "actor_bytes": candidate_actor,
        "candidate_sha256": hashlib.sha256(candidate_actor).hexdigest(),
        "metrics": metrics,
        "error": None,
    }


def propose_h037(
    core: Any,
    *,
    victim_slot: int,
    local: Any,
    shared: Any,
    groups: Any,
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit and validate one H037 parent candidate without installing it.

    The caller supplies already time-filtered observations and confirms the
    victim binding. This function independently validates the binding shapes,
    uses copied valid routes, scores the current teachers, and returns either
    the complete encoded actor payload or an explicit conservative rejection.
    """
    metrics: dict[str, Any] = {"iterations": 0, "closure_evaluations": 0}
    try:
        with _preserve_global_rng(core):
            return _propose_impl(
                core,
                victim_slot=victim_slot,
                local=local,
                shared=shared,
                groups=groups,
                evidence=evidence,
                config=config,
                metrics=metrics,
            )
    except Exception as exc:
        return _execution_error(exc, metrics)


__all__ = ["propose_h037"]

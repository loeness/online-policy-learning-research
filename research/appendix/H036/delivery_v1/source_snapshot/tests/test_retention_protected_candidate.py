"""Synthetic contract checks for the H036 protected candidate fitter."""
from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl import retention_behavioral_candidate as behavioral  # noqa: E402
from online_rl import retention_protected_candidate as protected  # noqa: E402


def _basis() -> np.ndarray:
    basis = np.zeros((4, 3), dtype=np.float64)
    for column in range(3):
        scale = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


def _features(count: int, seed: int) -> behavioral.FrozenPolicyFeatures:
    rng = np.random.default_rng(seed)
    return behavioral.FrozenPolicyFeatures(
        base_logits=rng.normal(0.0, 0.12, size=(count, 4)).astype(np.float64),
        source_psi=rng.normal(0.0, 0.7, size=(count, 3)).astype(np.float64),
        contrast_basis=_basis(),
        temperature=2.0,
        epsilon=0.05,
    )


def _probabilities(features: behavioral.FrozenPolicyFeatures, coefficients: np.ndarray) -> np.ndarray:
    return behavioral.candidate_probabilities(features, coefficients).astype(np.float64)


def _prepared(seed: int = 761) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    new_obs = np.zeros((256, 13), dtype=np.float32)
    new_obs[:, 0] = np.tile(np.choose(np.arange(128) % 3, [0.1, 0.45, 0.85]), 2)
    new_obs[:128, -1] = 1.0
    new_obs[128:, -1] = 2.0
    owners = np.repeat(np.arange(2, dtype=np.int64), 128)
    effective = rng.normal(0.0, 0.12, size=(2, 3, 3, 3)).astype(np.float32)
    new_features = _features(256, seed + 1)
    new_teacher = behavioral._teacher_probabilities(
        new_features,
        effective,
        owners,
        behavioral._bands_from_observations(new_obs),
    ).astype(np.float64)

    old_obs = np.zeros((128, 13), dtype=np.float32)
    old_obs[:, 0] = np.choose(np.arange(128) % 3, [0.1, 0.45, 0.85])
    old_features = _features(128, seed + 2)
    old_teacher = _probabilities(
        old_features,
        np.asarray([[0.13, -0.04, 0.02], [0.08, 0.03, -0.07], [-0.05, 0.06, 0.01]]),
    )
    return {
        "format": "h036_prepared_v1",
        "seed": 26201,
        "new_observations": new_obs,
        "new_owners": owners,
        "new_effective": effective,
        "new_features": new_features,
        "new_teacher": new_teacher,
        "old_observations": old_obs,
        "old_features": old_features,
        "old_teacher": old_teacher,
        "new_train_indices": np.concatenate((np.arange(96), np.arange(128, 224))).astype(np.int64),
        "new_holdout_indices": np.concatenate((np.arange(96, 128), np.arange(224, 256))).astype(np.int64),
        "old_train_indices": np.arange(96, dtype=np.int64),
        "old_holdout_indices": np.arange(96, 128, dtype=np.int64),
        "mean64": effective.mean(axis=(0, 1), dtype=np.float64),
        "prepared_id": f"synthetic-{seed}",
        "input_identity": {"source": "synthetic_test", "seed": seed},
    }


def _new_train(prepared: dict[str, object]):
    features = prepared["new_features"]
    indices = prepared["new_train_indices"]
    return (
        protected._slice_features(features, indices),
        prepared["new_teacher"][indices],
        np.full(192, 1.0 / 192.0, dtype=np.float64),
    )


def _old_train(prepared: dict[str, object]):
    features = protected._slice_features(prepared["old_features"], prepared["old_train_indices"])
    teacher = prepared["old_teacher"][prepared["old_train_indices"]]
    return features, teacher, np.full(96, 1.0 / 96.0, dtype=np.float64)


def _numpy_rng_state_equal(left, right) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_lambda_zero_returns_existing_objective_and_gradient_exactly() -> None:
    prepared = _prepared()
    new_features, new_teacher, new_weights = _new_train(prepared)
    old_features, old_teacher, old_weights = _old_train(prepared)
    point = np.asarray([[0.1, 0.03, -0.04], [-0.02, 0.07, 0.05], [0.08, -0.01, 0.06]])

    old_pair = behavioral.behavioral_objective_and_gradient(
        point, new_features, new_teacher, new_weights, prepared["mean64"],
    )
    new_pair = protected.protected_objective_and_gradient(
        point,
        new_features,
        new_teacher,
        new_weights,
        old_features,
        old_teacher,
        old_weights,
        prepared["mean64"],
        0,
    )

    assert new_pair[0] == old_pair[0]
    np.testing.assert_array_equal(new_pair[1], old_pair[1])


def test_lambda_one_gradient_finite_difference_and_ridge_is_added_once() -> None:
    prepared = _prepared(762)
    new_features, new_teacher, new_weights = _new_train(prepared)
    old_features, old_teacher, old_weights = _old_train(prepared)
    point = np.asarray([[0.1, 0.03, -0.04], [-0.02, 0.07, 0.05], [0.08, -0.01, 0.06]])
    common = prepared["mean64"]
    new_value, new_gradient = behavioral.behavioral_objective_and_gradient(
        point, new_features, new_teacher, new_weights, common,
    )
    old_value, old_gradient = behavioral.behavioral_objective_and_gradient(
        point, old_features, old_teacher, old_weights, common,
    )
    ridge = behavioral._RIDGE * float(np.mean((point - common) ** 2))
    expected_value = new_value + old_value - ridge
    expected_gradient = (
        new_gradient
        + old_gradient
        - (2.0 * behavioral._RIDGE / point.size) * (point - common)
    )
    objective, gradient = protected.protected_objective_and_gradient(
        point,
        new_features,
        new_teacher,
        new_weights,
        old_features,
        old_teacher,
        old_weights,
        common,
        1,
    )
    assert objective == expected_value
    np.testing.assert_array_equal(gradient, expected_gradient)

    numerical = np.zeros_like(point)
    delta = 1.0e-6
    for index in np.ndindex(point.shape):
        plus = point.copy()
        minus = point.copy()
        plus[index] += delta
        minus[index] -= delta
        loss_plus = protected.protected_objective_and_gradient(
            plus, new_features, new_teacher, new_weights,
            old_features, old_teacher, old_weights, common, 1,
        )[0]
        loss_minus = protected.protected_objective_and_gradient(
            minus, new_features, new_teacher, new_weights,
            old_features, old_teacher, old_weights, common, 1,
        )[0]
        numerical[index] = (loss_plus - loss_minus) / (2.0 * delta)
    np.testing.assert_allclose(gradient, numerical, rtol=2.0e-6, atol=3.0e-8)


def test_lambda_zero_fit_matches_original_fitter_on_same_single_thread_cpu() -> None:
    prepared = _prepared(763)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected, _ = behavioral.fit_behavioral_candidate(
            prepared["new_observations"],
            prepared["new_owners"],
            prepared["new_effective"],
            prepared["new_features"],
        )
        actual = protected.fit_protected_candidate(prepared, 0)
    finally:
        torch.set_num_threads(previous_threads)

    np.testing.assert_array_equal(actual["candidate"], expected)
    assert actual["format"] == "h036_fit_v1"
    assert actual["lambda_value"] == 0
    assert actual["prepared_id"] == prepared["prepared_id"]
    assert actual["diagnostics"]["input_hash_unchanged"] is True
    assert actual["diagnostics"]["closure_count"] <= 256
    json.dumps(actual["diagnostics"], allow_nan=False)


def test_holdout_edits_and_old_teacher_edits_do_not_change_lambda_zero_candidate() -> None:
    prepared = _prepared(764)
    base = protected.fit_protected_candidate(prepared, 0)["candidate"]

    changed_holdout = copy.deepcopy(prepared)
    changed_holdout["new_features"].base_logits[96:128] += 0.25
    changed_holdout["new_features"].base_logits[224:256] -= 0.2
    changed_holdout["new_teacher"][96:128] = np.roll(changed_holdout["new_teacher"][96:128], 1, axis=1)
    changed_holdout["new_teacher"][224:256] = np.roll(changed_holdout["new_teacher"][224:256], 2, axis=1)
    holdout_result = protected.fit_protected_candidate(changed_holdout, 0)["candidate"]

    changed_old_teacher = copy.deepcopy(prepared)
    changed_old_teacher["old_teacher"] = np.roll(changed_old_teacher["old_teacher"], 1, axis=1)
    old_result = protected.fit_protected_candidate(changed_old_teacher, 0)["candidate"]

    np.testing.assert_array_equal(holdout_result, base)
    np.testing.assert_array_equal(old_result, base)


def test_invalid_input_and_boolean_lambda_raise_instead_of_fallback() -> None:
    prepared = _prepared(765)
    bad_split = copy.deepcopy(prepared)
    bad_split["old_train_indices"][-1] = 94
    with pytest.raises(ValueError, match="old_train_indices"):
        protected.fit_protected_candidate(bad_split, 0)
    with pytest.raises(ValueError, match="not bool"):
        protected.fit_protected_candidate(prepared, True)

    bad_teacher = copy.deepcopy(prepared)
    bad_teacher["new_teacher"][0, 0] += 0.1
    with pytest.raises(ValueError, match="sum to one"):
        protected.fit_protected_candidate(bad_teacher, 0)


def test_prepared_features_require_frozen_actor_configuration_and_shared_basis() -> None:
    prepared = _prepared(769)
    prepared["new_features"] = behavioral.FrozenPolicyFeatures(
        base_logits=prepared["new_features"].base_logits,
        source_psi=prepared["new_features"].source_psi,
        contrast_basis=prepared["new_features"].contrast_basis,
        temperature=1.99,
        epsilon=0.05,
    )
    with pytest.raises(ValueError, match="temperature 2.0"):
        protected.fit_protected_candidate(prepared, 0)

    prepared = _prepared(770)
    different_basis = prepared["old_features"].contrast_basis.copy()
    different_basis[0, 0] += 0.01
    prepared["old_features"] = behavioral.FrozenPolicyFeatures(
        base_logits=prepared["old_features"].base_logits,
        source_psi=prepared["old_features"].source_psi,
        contrast_basis=different_basis,
        temperature=2.0,
        epsilon=0.05,
    )
    with pytest.raises(ValueError, match="share the same contrast_basis"):
        protected.fit_protected_candidate(prepared, 0)


def test_budget_exhaustion_returns_incumbent_and_never_exceeds_256_closures(monkeypatch) -> None:
    prepared = _prepared(766)

    def exhaust_budget(self, closure):
        for _ in range(257):
            closure()

    monkeypatch.setattr(torch.optim.LBFGS, "step", exhaust_budget)
    result = protected.fit_protected_candidate(prepared, 1)

    assert result["diagnostics"]["termination_reason"] == "evaluation_budget_exhausted"
    assert result["diagnostics"]["closure_count"] == 256
    assert result["diagnostics"]["returned_is_C0"] is True
    assert result["diagnostics"]["fallback_applied"] is False
    assert len(result["diagnostics"]["closure_records"]) == 256


def test_float32_gate_can_fallback_to_c0_and_records_reason(monkeypatch) -> None:
    prepared = _prepared(767)
    original_metric = protected._owner_kl_diagnostics
    c0_64 = prepared["mean64"].astype(np.float32).astype(np.float64)
    new_features, new_teacher, new_weights = _new_train(prepared)
    state = {
        "after_step": False,
        "gate_used": False,
        "initial_new_train": original_metric(
            new_features,
            new_teacher,
            prepared["new_owners"][prepared["new_train_indices"]],
            c0_64,
        )[1],
    }

    def controlled_step(self, closure):
        closure()
        state["after_step"] = True

    def controlled_metric(features, teacher, owners, coefficients):
        actual = original_metric(features, teacher, owners, coefficients)
        if state["after_step"] and not state["gate_used"]:
            state["gate_used"] = True
            return actual[0], float(state["initial_new_train"] + 1.0e-6)
        return actual

    monkeypatch.setattr(torch.optim.LBFGS, "step", controlled_step)
    monkeypatch.setattr(protected, "_owner_kl_diagnostics", controlled_metric)
    result = protected.fit_protected_candidate(prepared, 0)
    c0 = prepared["mean64"].astype(np.float32)

    np.testing.assert_array_equal(result["candidate"], c0)
    assert result["diagnostics"]["termination_reason"] == "optimizer_completed"
    assert result["diagnostics"]["fallback_applied"] is True
    assert result["diagnostics"]["fallback_reason"] == "float32_new_train_kl_gate"
    assert result["diagnostics"]["returned_is_C0"] is True


def test_nonfinite_closure_error_carries_partial_attempt_diagnostics(monkeypatch) -> None:
    prepared = _prepared(771)
    original_objective = protected.protected_objective_and_gradient
    calls = {"count": 0}

    def nonfinite_on_closure(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            return original_objective(*args, **kwargs)
        return float("nan"), np.full((3, 3), np.nan, dtype=np.float64)

    def one_valid_then_bad(self, closure):
        closure()
        closure()

    monkeypatch.setattr(protected, "protected_objective_and_gradient", nonfinite_on_closure)
    monkeypatch.setattr(torch.optim.LBFGS, "step", one_valid_then_bad)
    with pytest.raises(FloatingPointError, match="not finite") as raised:
        protected.fit_protected_candidate(prepared, 0)

    partial = raised.value.partial_diagnostics
    assert len(partial["closure_records"]) == 1
    assert partial["closure_records"][0]["index"] == 1
    assert partial["attempted_closure_count"] == 2
    assert "not finite" in partial["reason"]
    assert len(partial["attempted_candidate_hash"]) == 64


def test_fit_preserves_input_arrays_and_python_numpy_torch_rng_states() -> None:
    prepared = _prepared(768)
    arrays_before = {
        key: value.copy()
        for key, value in prepared.items()
        if isinstance(value, np.ndarray)
    }
    features_before = {
        key: (value.base_logits.copy(), value.source_psi.copy(), value.contrast_basis.copy())
        for key, value in prepared.items()
        if isinstance(value, behavioral.FrozenPolicyFeatures)
    }
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    result = protected.fit_protected_candidate(prepared, 0)

    for key, value in arrays_before.items():
        np.testing.assert_array_equal(prepared[key], value)
    for key, values in features_before.items():
        current = prepared[key]
        np.testing.assert_array_equal(current.base_logits, values[0])
        np.testing.assert_array_equal(current.source_psi, values[1])
        np.testing.assert_array_equal(current.contrast_basis, values[2])
    assert random.getstate() == python_state
    assert _numpy_rng_state_equal(numpy_state, np.random.get_state())
    assert torch.equal(torch_state, torch.get_rng_state())
    assert result["diagnostics"]["input_hash_before"] == result["diagnostics"]["input_hash_after"]


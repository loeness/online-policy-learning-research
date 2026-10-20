"""Focused tests for behavioral distillation of resident retention policies."""
from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.retention_behavioral_candidate import (  # noqa: E402
    FrozenPolicyFeatures,
    behavioral_objective_and_gradient,
    candidate_probabilities,
    extract_frozen_policy_features,
    fit_behavioral_candidate,
)
from scripts import run_storage_decision as storage_runner  # noqa: E402


def _contrast_basis() -> np.ndarray:
    basis = np.zeros((4, 3), dtype=np.float64)
    for column in range(3):
        scale = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


def _features(count: int, *, seed: int = 4, epsilon: float = 0.17, temperature: float = 1.3) -> FrozenPolicyFeatures:
    rng = np.random.default_rng(seed)
    return FrozenPolicyFeatures(
        base_logits=rng.normal(0.0, 0.12, size=(count, 4)),
        source_psi=rng.normal(0.0, 0.7, size=(count, 3)),
        contrast_basis=_contrast_basis(),
        temperature=temperature,
        epsilon=epsilon,
    )


def _observations(owner_counts: tuple[int, int], *, bands: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    owners = np.concatenate([
        np.full(owner_counts[0], 0, dtype=np.int64),
        np.full(owner_counts[1], 1, dtype=np.int64),
    ])
    observations = np.zeros((len(owners), 13), dtype=np.float32)
    if bands is None:
        band_values = np.arange(len(owners), dtype=np.int64) % 3
    else:
        band_values = np.asarray(bands, dtype=np.int64)
        assert band_values.shape == (len(owners),)
    observations[:, 0] = np.choose(band_values, [0.1, 0.45, 0.85])
    observations[:, -1] = owners + 1
    return observations, owners


def _kl_rows(teacher: np.ndarray, student: np.ndarray) -> np.ndarray:
    log_teacher = np.zeros_like(teacher)
    positive = teacher > 0.0
    log_teacher[positive] = np.log(teacher[positive])
    return np.sum(np.where(positive, teacher * (log_teacher - np.log(student)), 0.0), axis=1)


def test_representable_identical_teacher_is_recovered_by_initial_common_mean():
    observations, owners = _observations((32, 32))
    features = _features(len(owners), seed=8)
    shared = np.asarray([[0.12, -0.04, 0.02], [0.08, 0.03, -0.07], [-0.05, 0.06, 0.01]], dtype=np.float32)
    effective = np.broadcast_to(shared, (2, 3, 3, 3)).copy()

    candidate, diagnostics = fit_behavioral_candidate(observations, owners, effective, features)

    np.testing.assert_allclose(candidate, shared, rtol=0.0, atol=1e-7)
    assert diagnostics["status"] == "fitted"
    assert diagnostics["train_kl_mean_fit"] <= diagnostics["train_kl_mean_init"] + 1e-12
    assert diagnostics["holdout_kl_mean_fit"] <= 1e-12
    assert diagnostics["source_psi_rank"] == 3
    assert diagnostics["finite_inputs"] and diagnostics["finite_candidate"]
    assert diagnostics["persistent_candidate_bytes"] == 9 * np.dtype(np.float32).itemsize


def test_nonuniform_band_behavioral_fit_beats_full_matrix_parameter_mean():
    counts = (40, 40)
    one_owner_bands = [0] * 24 + [1] * 3 + [2] * 3 + [0] * 8 + [1, 2]
    observations, owners = _observations(counts, bands=one_owner_bands * 2)
    features = FrozenPolicyFeatures(
        base_logits=np.zeros((80, 4), dtype=np.float64),
        source_psi=np.tile(np.asarray([[1.0, 0.0, 0.0]]), (80, 1)),
        contrast_basis=_contrast_basis(),
        temperature=1.7,
        epsilon=0.04,
    )
    effective = np.zeros((2, 3, 3, 3), dtype=np.float32)
    effective[:, 0, 0, 0] = 0.8
    effective[:, 1, 0, 0] = -0.3
    effective[:, 2, 0, 0] = -0.3

    candidate, diagnostics = fit_behavioral_candidate(observations, owners, effective, features)

    assert candidate[0, 0] > effective.mean(axis=(0, 1))[0, 0]
    assert diagnostics["train_kl_mean_fit"] < diagnostics["train_kl_mean_init"] - 1e-4
    assert diagnostics["holdout_kl_mean_fit"] < diagnostics["holdout_kl_mean_init"]
    assert diagnostics["train_rows_by_owner"] == [30, 30]
    assert diagnostics["holdout_rows_by_owner"] == [10, 10]


def test_owner_indices_select_the_matching_teacher_matrix():
    observations, owners = _observations((32, 32), bands=[0] * 64)
    features = FrozenPolicyFeatures(
        base_logits=np.zeros((64, 4), dtype=np.float64),
        source_psi=np.vstack((np.tile([[1.0, 0.0, 0.0]], (32, 1)), np.tile([[2.0, 0.0, 0.0]], (32, 1)))),
        contrast_basis=_contrast_basis(),
        temperature=1.2,
        epsilon=0.1,
    )
    effective = np.zeros((2, 3, 3, 3), dtype=np.float32)
    effective[0, :, 0, 0] = 0.9
    effective[1, :, 0, 0] = -0.2
    candidate, diagnostics = fit_behavioral_candidate(observations, owners, effective, features)
    common = effective.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)

    teacher0 = candidate_probabilities(
        FrozenPolicyFeatures(features.base_logits[:32], features.source_psi[:32], features.contrast_basis,
                             features.temperature, features.epsilon),
        effective[0, 0],
    )
    student0 = candidate_probabilities(
        FrozenPolicyFeatures(features.base_logits[:32], features.source_psi[:32], features.contrast_basis,
                             features.temperature, features.epsilon),
        common,
    )
    expected_owner0 = float(np.mean(_kl_rows(teacher0, student0)))
    assert diagnostics["train_kl_by_owner_init"][0] == pytest.approx(expected_owner0, abs=1e-12)

    swapped_candidate, swapped_diagnostics = fit_behavioral_candidate(
        observations, 1 - owners, effective, features
    )
    assert abs(float(candidate[0, 0] - swapped_candidate[0, 0])) > 0.02
    assert diagnostics["train_kl_by_owner_init"] != pytest.approx(
        swapped_diagnostics["train_kl_by_owner_init"], abs=1e-5
    )


def test_exact_gradient_matches_finite_difference_with_epsilon_mixture():
    features = _features(7, seed=22, epsilon=0.23, temperature=0.8)
    truth = np.asarray([[0.3, -0.2, 0.1], [-0.1, 0.25, 0.2], [0.08, 0.04, -0.12]])
    teacher = candidate_probabilities(features, truth)
    weights = np.full(7, 1.0 / 7.0)
    common = np.zeros((3, 3), dtype=np.float64)
    point = np.asarray([[0.1, 0.03, -0.04], [-0.02, 0.07, 0.05], [0.08, -0.01, 0.06]])

    objective, gradient = behavioral_objective_and_gradient(point, features, teacher, weights, common)
    numerical = np.zeros_like(point)
    step = 1.0e-6
    for index in np.ndindex(point.shape):
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        loss_plus = behavioral_objective_and_gradient(plus, features, teacher, weights, common)[0]
        loss_minus = behavioral_objective_and_gradient(minus, features, teacher, weights, common)[0]
        numerical[index] = (loss_plus - loss_minus) / (2.0 * step)

    assert math.isfinite(objective)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)


class _ActorExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(nn.Linear(4, 7), nn.Tanh(), nn.Linear(7, 5), nn.Tanh())

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[:, :4])


class _FrozenCore(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            self.mlp_extractor = _ActorExtractor()
            self.action_net = nn.Linear(5, 4)
            self.source_basis = nn.Parameter(torch.randn(3, 6) * 0.08)
        self.register_buffer("_contrast_basis", torch.as_tensor(_contrast_basis(), dtype=torch.float32))
        self.temperature = 0.73
        self.epsilon = 0.19
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-3)

    def extract_features(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


def _numpy_rng_state_equal(left, right) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_core_feature_helper_matches_existing_probs_and_mutates_no_core_optimizer_or_rng():
    core = _FrozenCore()
    core.train()
    observations, _ = _observations((18, 18))
    observations[:, 1:4] = np.linspace(-0.8, 0.8, len(observations), dtype=np.float32)[:, None]
    candidate = np.asarray([[0.3, -0.1, 0.04], [0.07, 0.12, -0.06], [-0.04, 0.02, 0.08]], dtype=np.float32)
    effective = np.broadcast_to(candidate, (3, 3, 3)).copy()
    parameters_before = [parameter.detach().clone() for parameter in core.parameters()]
    optimizer_before = copy.deepcopy(core.optimizer.state_dict())
    torch_rng_before = torch.get_rng_state().clone()
    numpy_rng_before = np.random.get_state()
    python_rng_before = random.getstate()
    mode_before = core.training

    features = extract_frozen_policy_features(core, observations)
    actual = candidate_probabilities(features, candidate)
    expected = storage_runner._coefficient_probs(core, observations, effective)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert core.training is mode_before
    assert all(torch.equal(before, after) for before, after in zip(parameters_before, core.parameters()))
    assert core.optimizer.state_dict() == optimizer_before
    assert torch.equal(torch_rng_before, torch.get_rng_state())
    assert _numpy_rng_state_equal(numpy_rng_before, np.random.get_state())
    assert python_rng_before == random.getstate()


def test_insufficient_support_returns_mean_and_nonfinite_inputs_are_rejected():
    observations, owners = _observations((12, 15))
    features = _features(len(owners), seed=9)
    rng = np.random.default_rng(91)
    effective = rng.normal(0.0, 0.1, size=(2, 3, 3, 3)).astype(np.float32)

    candidate, diagnostics = fit_behavioral_candidate(observations, owners, effective, features)

    common = effective.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(candidate, common)
    assert diagnostics["status"] == "insufficient_support"
    assert diagnostics["optimizer_function_evaluations"] == 0
    assert diagnostics["finite_candidate"]
    assert diagnostics["persistent_candidate_bytes"] == candidate.nbytes
    bad_effective = effective.copy()
    bad_effective[1, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_behavioral_candidate(observations, owners, bad_effective, features)


def test_latest_128_rows_per_owner_are_used_and_optimizer_budget_is_bounded():
    observations, owners = _observations((160, 160))
    features = _features(len(owners), seed=31)
    effective = np.zeros((2, 3, 3, 3), dtype=np.float32)
    effective[:, 0, 0, 0] = 0.4
    effective[:, 1, 0, 0] = -0.1
    effective[:, 2, 0, 0] = -0.1

    _, diagnostics = fit_behavioral_candidate(observations, owners, effective, features)

    assert diagnostics["rows_used_by_owner"] == [128, 128]
    assert diagnostics["truncated_rows_by_owner"] == [32, 32]
    assert diagnostics["optimizer_function_evaluations"] <= 256
    assert diagnostics["iterations"] <= 64
    assert diagnostics["persistent_candidate_bytes"] == 36


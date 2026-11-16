"""Synthetic H037 evidence, checkpoint, and fail-closed contract checks.

These checks use fixed arrays and a narrow frozen-feature spy. They do not
construct or interact with a task environment.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl import h037_candidate
from online_rl.h037_evidence import H037Evidence
from online_rl.h037_integration import H037Controller


def _observations(markers: list[float]) -> np.ndarray:
    rows = np.zeros((len(markers), 13), dtype=np.float32)
    rows[:, 0] = np.float32(0.25)
    rows[:, 1] = np.asarray(markers, dtype=np.float32)
    # Deliberately use the invalid / inactive route so callers must repair a
    # copy to the live one-based slot before frozen feature extraction.
    rows[:, -1] = np.float32(0.0)
    return rows


def _candidate_config() -> dict[str, object]:
    return {
        "lambda_old": 1.0,
        "ridge": 1.0e-4,
        "max_iter": 1,
        "max_eval": 4,
        "delta_v": 1.0e-3,
        "epsilon_old": 1.0,
        "epsilon_res": 1.0,
        "compensation_atol": 1.0e-6,
        "codec": "float32",
        "actor_budget_bytes": 512,
    }


def _controller_config() -> dict[str, object]:
    return {
        "format": "h037_controller_v1",
        **_candidate_config(),
        "holdout_fraction": 0.25,
        "min_victim_train_rows": 1,
        "min_victim_holdout_rows": 1,
        "min_old_train_rows": 1,
        "min_old_holdout_rows": 1,
        "min_resident_holdout_rows": 1,
    }


def _candidate_inputs(config: dict[str, object] | None = None):
    old_train = _observations([4.0, 5.0])
    old_holdout = _observations([6.0])
    old_train[:, -1] = 2.0
    old_holdout[:, -1] = 2.0
    evidence = {
        "victim_train": _observations([1.0, 2.0]),
        "victim_holdout": _observations([3.0]),
        "old_train": old_train,
        "old_holdout": old_holdout,
        "resident_holdout": {1: _observations([7.0])},
    }
    return {
        "core": object(),
        "victim_slot": 0,
        "local": np.zeros((2, 27), dtype=np.float32),
        "shared": np.zeros((1, 9), dtype=np.float32),
        "groups": np.asarray([0, 0], dtype=np.int64),
        "evidence": evidence,
        "config": _candidate_config() if config is None else config,
    }


def test_h037_evidence_isolates_identity_and_preserves_ring_order_across_checkpoint():
    evidence = H037Evidence(capacity_per_slot=3, slots=2)
    evidence.record(0, 4, "victim-a", 0, 2, _observations([10.0, 11.0]))
    evidence.record(1, 8, "resident", 0, 2, _observations([80.0]))
    evidence.record(0, 4, "victim-a", 0, 3, _observations([12.0, 13.0, 14.0]))

    wrapped = evidence.get(0, 4, "victim-a", 0)
    assert wrapped["observations"][:, 1].tolist() == [11.0, 12.0, 13.0, 14.0][-3:]
    assert wrapped["parent_versions"].tolist() == [3, 3, 3]

    state = evidence.state_dict()
    restored = H037Evidence.from_state_dict(state)
    restored_batch = restored.get(0, 4, "victim-a", 0)
    np.testing.assert_array_equal(restored_batch["observations"], wrapped["observations"])
    np.testing.assert_array_equal(restored_batch["parent_versions"], wrapped["parent_versions"])

    # A fresh generation replaces only slot 0. The prior identity becomes
    # unavailable while the resident slot and its order remain intact.
    restored.record(0, 5, "victim-b", 0, 4, _observations([15.0]))
    assert restored.get(0, 4, "victim-a", 0)["observations"].shape == (0, 13)
    assert restored.get(0, 5, "victim-b", 0)["observations"][:, 1].tolist() == [15.0]
    assert restored.get(1, 8, "resident", 0)["observations"][:, 1].tolist() == [80.0]


def test_controller_checkpoint_preserves_rollout_order_parent_versions_and_routes():
    class CoreStub:
        def __init__(self):
            self.shared = np.zeros((1, 9), dtype=np.float32)

        def read_shared_matrix(self):
            return self.shared.copy()

    core = CoreStub()
    agent = SimpleNamespace(parent_group=0, core=core)
    controller = H037Controller(_controller_config(), H037Evidence(4, slots=2))

    def row(marker: float):
        observation = _observations([marker])[0]
        return SimpleNamespace(
            obs=observation,
            slot=1,
            generation=7,
            task_name="resident-task",
            group_id=0,
        )

    first = row(21.0)
    second = row(22.0)
    controller.record_rollout(
        agent, [first, second], slot=1, generation=7, task="resident-task", group=0
    )
    # A changed parent is recorded with a new version; existing row
    # provenance remains intact and input route fields are left untouched.
    core.shared[0, 0] = np.float32(0.5)
    controller.record_rollout(
        agent, [row(23.0)], slot=1, generation=7, task="resident-task", group=0
    )

    batch = controller.evidence.get(1, 7, "resident-task", 0)
    assert batch["observations"][:, 1].tolist() == [21.0, 22.0, 23.0]
    assert batch["parent_versions"].tolist() == [0, 0, 1]
    assert np.all(batch["observations"][:, -1] == 2.0)
    assert first.obs[-1] == 0.0 and second.obs[-1] == 0.0

    stream = io.BytesIO()
    torch.save({"h037_controller": controller.state_dict()}, stream)
    stream.seek(0)
    payload = torch.load(stream, map_location="cpu", weights_only=False)
    restored = H037Controller.from_state_dict(payload["h037_controller"])
    assert H037Controller._state_equal(controller.state_dict(), restored.state_dict())
    restored_batch = restored.evidence.get(1, 7, "resident-task", 0)
    np.testing.assert_array_equal(restored_batch["observations"], batch["observations"])
    np.testing.assert_array_equal(restored_batch["parent_versions"], batch["parent_versions"])


def test_candidate_evidence_uses_time_order_and_excludes_stale_parent_rows():
    evidence = H037Evidence(capacity_per_slot=8, slots=2)
    for marker, version in zip([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [0, 0, 0, 1, 0, 0, 0]):
        evidence.record(0, 4, "victim", 0, version, _observations([marker]))
    for marker, version in zip([20.0, 21.0, 22.0, 23.0], [0, 0, 1, 0]):
        evidence.record(1, 9, "resident", 0, version, _observations([marker]))

    class CoreStub:
        @staticmethod
        def group_for_slot(_slot: int) -> int:
            return 0

    agent = SimpleNamespace(
        slot_generation=np.asarray([4, 9], dtype=np.int64),
        slot_group_tags=np.asarray([0, 0], dtype=np.int8),
        slots=2,
        core=CoreStub(),
        cache=SimpleNamespace(slot_to_task=["victim", "resident"]),
    )
    controller = H037Controller(_controller_config(), evidence)

    selected, stats, reason, resident_slots = controller._build_candidate_evidence(
        agent,
        {"evicted_task": "victim"},
        victim_slot=0,
        parent_group=0,
    )

    assert reason is None
    assert selected is not None
    assert resident_slots == (1,)
    # Parent version 1 is excluded before the ordered 4/2 train/holdout split.
    assert stats["victim"] == {
        "total_rows": 7,
        "current_parent_rows": 6,
        "excluded_parent_rows": 1,
    }
    np.testing.assert_array_equal(selected["victim_train"][:, 1], [10.0, 11.0, 12.0, 14.0])
    np.testing.assert_array_equal(selected["victim_holdout"][:, 1], [15.0, 16.0])
    assert np.all(selected["victim_train"][:, -1] == 1.0)
    assert np.all(selected["victim_holdout"][:, -1] == 1.0)
    assert stats["residents"]["1"] == {
        "total_rows": 4,
        "current_parent_rows": 3,
        "excluded_parent_rows": 1,
    }
    np.testing.assert_array_equal(selected["old_train"][:, 1], [20.0, 21.0])
    np.testing.assert_array_equal(selected["old_holdout"][:, 1], [23.0])
    np.testing.assert_array_equal(selected["resident_holdout"][1][:, 1], [23.0])


@pytest.mark.parametrize("missing_key", ["delta_v", "compensation_atol"])
def test_candidate_rejects_missing_acceptance_thresholds(missing_key: str):
    inputs = _candidate_inputs()
    config = dict(inputs["config"])
    config.pop(missing_key)
    inputs["config"] = config

    result = h037_candidate.propose_h037(**inputs)

    assert result["accepted"] is False
    assert result["actor_bytes"] is None
    assert "missing required keys" in result["reason"]
    assert missing_key in result["reason"]


def test_candidate_rejects_empty_required_evidence_before_feature_extraction():
    inputs = _candidate_inputs()
    inputs["evidence"]["victim_train"] = np.empty((0, 13), dtype=np.float32)

    result = h037_candidate.propose_h037(**inputs)

    assert result["accepted"] is False
    assert result["actor_bytes"] is None
    assert "victim_train must be nonempty" in result["reason"]


def test_candidate_repairs_route_zero_to_valid_one_based_routes(monkeypatch: pytest.MonkeyPatch):
    seen_routes: list[np.ndarray] = []

    def feature_spy(_core, observations: np.ndarray):
        seen_routes.append(observations[:, -1].copy())
        count = len(observations)
        return h037_candidate.FrozenPolicyFeatures(
            base_logits=np.zeros((count, 4), dtype=np.float64),
            source_psi=np.zeros((count, 3), dtype=np.float64),
            contrast_basis=np.zeros((4, 3), dtype=np.float64),
            temperature=1.0,
            epsilon=0.05,
        )

    monkeypatch.setattr(h037_candidate, "extract_frozen_policy_features", feature_spy)
    inputs = _candidate_inputs()
    original_inputs = {
        name: value.copy()
        for name, value in inputs["evidence"].items()
        if isinstance(value, np.ndarray)
    }
    original_resident = inputs["evidence"]["resident_holdout"][1].copy()

    result = h037_candidate.propose_h037(**inputs)

    # Victim feature batches use route 1, while old-prior and resident batches
    # retain resident route 2. The zero-improvement case is expected to fail
    # the explicit delta_v gate after traversing its real fit/codec path.
    assert len(seen_routes) == 5
    for routes in seen_routes[:2]:
        np.testing.assert_array_equal(routes, np.ones_like(routes))
    for routes in seen_routes[2:]:
        np.testing.assert_array_equal(routes, np.full_like(routes, 2.0))
    assert result["accepted"] is False
    assert result["status"] == "reject"
    assert result["error"] is None
    assert "victim holdout KL improvement below delta_v" in result["reason"]
    for name, original in original_inputs.items():
        np.testing.assert_array_equal(inputs["evidence"][name], original)
    np.testing.assert_array_equal(inputs["evidence"]["resident_holdout"][1], original_resident)


def test_candidate_execution_exception_is_not_a_scientific_rejection(monkeypatch: pytest.MonkeyPatch):
    def fail_feature_extraction(_core, _observations: np.ndarray):
        raise RuntimeError("synthetic frozen feature failure")

    monkeypatch.setattr(h037_candidate, "extract_frozen_policy_features", fail_feature_extraction)

    result = h037_candidate.propose_h037(**_candidate_inputs())

    assert result["accepted"] is False
    assert result["status"] == "error"
    assert result["reason"] == "RuntimeError: synthetic frozen feature failure"
    assert result["error"] == {
        "type": "RuntimeError",
        "message": "synthetic frozen feature failure",
    }
    assert result["actor_bytes"] is None


def test_int8_candidate_and_no_write_gates_use_their_decoded_actor_packages(
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = _candidate_inputs()
    inputs["local"][0, 0] = np.float32(1.0)
    config = dict(inputs["config"])
    config.update(
        {
            "codec": "int8",
            "actor_budget_bytes": 89,
            "lambda_old": 0.05,
            "ridge": 1.0e-6,
            "max_iter": 10,
            "max_eval": 30,
            "delta_v": 1.0e-3,
            "epsilon_old": 10.0,
            "epsilon_res": 10.0,
        }
    )
    inputs["config"] = config

    extracted_features: list[h037_candidate.FrozenPolicyFeatures] = []

    def feature_spy(_core, observations: np.ndarray):
        count = len(observations)
        base_logits = np.zeros((count, 4), dtype=np.float64)
        source_psi = np.zeros((count, 3), dtype=np.float64)
        source_psi[:, 0] = 1.0
        contrast_basis = np.zeros((4, 3), dtype=np.float64)
        contrast_basis[:, 0] = np.asarray([1.0, -1.0, 0.0, 0.0])
        features = h037_candidate.FrozenPolicyFeatures(
            base_logits=base_logits,
            source_psi=source_psi,
            contrast_basis=contrast_basis,
            temperature=1.0,
            epsilon=0.05,
        )
        extracted_features.append(features)
        return features

    real_encode_actor = h037_candidate.encode_actor
    encoded_calls: list[tuple[str, Any]] = []

    def encode_spy(local, shared, groups, *, codec: str, budget_bytes: int):
        encoded = real_encode_actor(
            local,
            shared,
            groups,
            codec=codec,
            budget_bytes=budget_bytes,
        )
        encoded_calls.append((codec, encoded))
        return encoded

    monkeypatch.setattr(h037_candidate, "extract_frozen_policy_features", feature_spy)
    monkeypatch.setattr(h037_candidate, "encode_actor", encode_spy)

    result = h037_candidate.propose_h037(**inputs)

    assert result["accepted"] is True
    assert result["status"] == "accept"
    assert len(encoded_calls) == 2
    assert [codec for codec, _encoded in encoded_calls] == ["int8", "int8"]
    no_write_package = encoded_calls[0][1].data
    candidate_package = encoded_calls[1][1].data
    assert len(no_write_package) == len(candidate_package) == 89
    assert candidate_package != no_write_package
    assert result["actor_bytes"] == candidate_package

    no_write_local, no_write_shared, no_write_groups = h037_candidate.decode_actor(no_write_package)
    candidate_local, candidate_shared, candidate_groups = h037_candidate.decode_actor(candidate_package)
    np.testing.assert_array_equal(no_write_groups, inputs["groups"])
    np.testing.assert_array_equal(candidate_groups, inputs["groups"])
    assert result["metrics"]["acceptance_gate_domain"] == "encoded_actor_roundtrip"
    assert result["metrics"]["baseline_actor_nbytes"] == len(no_write_package)
    assert result["metrics"]["candidate_actor_nbytes"] == len(candidate_package)

    victim_observations = inputs["evidence"]["victim_holdout"].copy()
    victim_observations[:, -1] = np.float32(1.0)
    old_observations = inputs["evidence"]["old_holdout"]
    victim_features = extracted_features[1]
    old_features = extracted_features[3]
    resident_observations = inputs["evidence"]["resident_holdout"][1]
    resident_features = extracted_features[4]
    original_parent = inputs["shared"][0].reshape(3, 3)
    victim_teacher = h037_candidate._teacher_probabilities(
        victim_features,
        inputs["local"][0],
        original_parent,
        victim_observations,
    )
    old_teacher = h037_candidate._teacher_probabilities(
        old_features,
        np.zeros((3, 3, 3), dtype=np.float64),
        original_parent,
        old_observations,
    )
    resident_teacher = h037_candidate._teacher_probabilities(
        resident_features,
        inputs["local"][1],
        original_parent,
        resident_observations,
    )

    def victim_kl(local: np.ndarray, shared: np.ndarray) -> float:
        student_log = h037_candidate._log_probabilities_for_coefficients(
            victim_features,
            h037_candidate._effective_rows(local[0], shared[0], victim_observations),
        )
        return h037_candidate._mean_forward_kl(victim_teacher, student_log)

    def old_kl(shared: np.ndarray) -> float:
        student_coefficients = np.broadcast_to(
            shared[0].reshape(1, 3, 3), (len(old_observations), 3, 3)
        )
        student_log = h037_candidate._log_probabilities_for_coefficients(
            old_features, student_coefficients
        )
        return h037_candidate._mean_forward_kl(old_teacher, student_log)

    def resident_drift(local: np.ndarray, shared: np.ndarray) -> float:
        student = h037_candidate._probabilities_for_coefficients(
            resident_features,
            h037_candidate._effective_rows(
                local[1], shared[0], resident_observations
            ),
        )
        return float(np.max(np.abs(student - resident_teacher)))

    baseline_victim = victim_kl(no_write_local, no_write_shared)
    candidate_victim = victim_kl(candidate_local, candidate_shared)
    baseline_old = old_kl(no_write_shared)
    candidate_old = old_kl(candidate_shared)
    baseline_resident = resident_drift(no_write_local, no_write_shared)
    candidate_resident = resident_drift(candidate_local, candidate_shared)
    metrics = result["metrics"]
    assert metrics["baseline_victim_holdout_kl"] == pytest.approx(baseline_victim)
    assert metrics["candidate_victim_holdout_kl"] == pytest.approx(candidate_victim)
    assert metrics["baseline_old_holdout_kl"] == pytest.approx(baseline_old)
    assert metrics["candidate_old_holdout_kl"] == pytest.approx(candidate_old)
    assert metrics["baseline_resident_max_probability_drift"] == pytest.approx(
        baseline_resident
    )
    assert metrics["candidate_resident_max_probability_drift"] == pytest.approx(
        candidate_resident
    )
    assert metrics["victim_holdout_kl_improvement"] == pytest.approx(
        baseline_victim - candidate_victim
    )
    assert metrics["candidate_old_holdout_kl_delta_vs_no_write"] == pytest.approx(
        candidate_old - baseline_old
    )
    assert metrics["candidate_resident_drift_delta_vs_no_write"] == pytest.approx(
        candidate_resident - baseline_resident
    )
    assert metrics["victim_holdout_kl_improvement"] >= config["delta_v"]
    assert candidate_old <= config["epsilon_old"]
    assert candidate_resident <= config["epsilon_res"]


"""Focused tests for independent retention-candidate integrity checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import audit_retention_candidate as audit_module
from online_rl.storage_decision import semantic_hash
from online_rl import retention_behavioral_candidate as behavioral


def test_candidate_module_hash_is_a_distinct_independently_reproducible_scheme() -> None:
    value = np.arange(9, dtype=np.float32).reshape(3, 3)
    digest = hashlib.sha256(b"float32\0" + repr(value.shape).encode("ascii") + value.tobytes()).hexdigest()

    assert audit_module._candidate_module_hash(value) == digest
    assert audit_module._candidate_module_hash(value) != semantic_hash(value)


def test_independent_feature_objective_and_analytic_gradient_match_finite_difference() -> None:
    rng = np.random.default_rng(761)
    count = 12
    features = behavioral.FrozenPolicyFeatures(
        base_logits=rng.normal(size=(count, 4)).astype(np.float32),
        source_psi=rng.normal(size=(count, 3)).astype(np.float32),
        contrast_basis=rng.normal(size=(4, 3)).astype(np.float32),
        temperature=0.83,
        epsilon=0.025,
    )
    teacher_logits = rng.normal(size=(count, 4))
    teacher = np.exp(teacher_logits - teacher_logits.max(axis=1, keepdims=True))
    teacher /= teacher.sum(axis=1, keepdims=True)
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    common = rng.normal(scale=0.1, size=(3, 3)).astype(np.float64)
    candidate = rng.normal(scale=0.2, size=(3, 3)).astype(np.float64)
    objective, gradient = behavioral.behavioral_objective_and_gradient(
        candidate, features, teacher, weights, common,
    )
    independent = audit_module._feature_objective(
        features, candidate, teacher, weights, common, behavioral._RIDGE,
    )
    assert abs(objective - independent) < 1e-12

    delta = 1e-6
    finite_difference = np.empty_like(candidate)
    for row in range(3):
        for column in range(3):
            plus = candidate.copy()
            minus = candidate.copy()
            plus[row, column] += delta
            minus[row, column] -= delta
            finite_difference[row, column] = (
                audit_module._feature_objective(features, plus, teacher, weights, common, behavioral._RIDGE)
                - audit_module._feature_objective(features, minus, teacher, weights, common, behavioral._RIDGE)
            ) / (2 * delta)
    assert np.max(np.abs(gradient - finite_difference)) < 2e-5


def _raw_step(global_step: int, bout: int, slot: int, generation: int, physical_obs: np.ndarray) -> dict[str, str]:
    return {
        "global_step": str(global_step),
        "bout": str(bout),
        "slot": str(slot),
        "generation": str(generation),
        "obs": json.dumps(np.asarray(physical_obs, dtype=np.float32).tolist()),
    }


def test_evidence_selection_rebuilds_latest_rows_and_rejects_provenance_mutation() -> None:
    from scripts.run_knowledge_loop import augment_observation

    rows: list[dict[str, str]] = []
    rollouts: list[dict[str, str]] = []
    source_records = []
    evidence_obs: list[np.ndarray] = []
    evidence_owners: list[int] = []
    evidence_slots: list[int] = []
    evidence_generations: list[int] = []
    evidence_bouts: list[int] = []
    evidence_steps: list[int] = []
    for owner, (slot, generation, bout) in enumerate(((0, 4, 7), (1, 5, 8))):
        start_step = owner * 512
        for local_step in range(1, 513):
            global_step = start_step + local_step
            physical_obs = np.asarray([local_step / 600.0, owner, 0.1, -0.2], dtype=np.float32)
            rows.append(_raw_step(global_step, bout, slot, generation, physical_obs))
            if local_step > 384:
                evidence_obs.append(np.asarray(augment_observation(physical_obs, slot + 1), dtype=np.float32))
                evidence_owners.append(owner)
                evidence_slots.append(slot)
                evidence_generations.append(generation)
                evidence_bouts.append(bout)
                evidence_steps.append(global_step)
        rollouts.extend([
            {"bout": str(bout), "rollout_index": "0", "rows": "256"},
            {"bout": str(bout), "rollout_index": "1", "rows": "256"},
        ])
        source_records.append({"slot": slot, "generation": generation, "source_bout": bout,
                              "latest_source_global_step": start_step + 512})
    evidence = {
        "observations": np.stack(evidence_obs).astype(np.float32),
        "owner_indices": np.asarray(evidence_owners, dtype=np.int8),
        "source_slots": np.asarray(evidence_slots, dtype=np.int8),
        "generations": np.asarray(evidence_generations, dtype=np.int64),
        "bouts": np.asarray(evidence_bouts, dtype=np.int64),
        "global_steps": np.asarray(evidence_steps, dtype=np.int64),
    }
    latest = [512, 1024]
    good = audit_module.Audit(Path("."))
    audit_module._audit_evidence_selection(good, evidence, source_records, rows, rollouts,
                                           "synthetic", latest)
    assert good.errors == []

    changed = {key: value.copy() for key, value in evidence.items()}
    changed["global_steps"][0] += 1
    bad = audit_module.Audit(Path("."))
    audit_module._audit_evidence_selection(bad, changed, source_records, rows, rollouts,
                                           "synthetic", latest)
    assert any(item["code"] == "evidence_provenance" for item in bad.errors)


def test_episode_state_check_detects_a_discontinuous_transition() -> None:
    start_observation = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    next_observation = np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    start = {"driver": {"episode_ordinal": 9, "episode_reset_seed": 111,
                        "episode_length": 4, "obs": start_observation.tolist()}}
    end = {"driver": {"episode_ordinal": 9, "episode_reset_seed": 111,
                      "episode_length": 5, "obs": next_observation.tolist()}}
    rows = [{
        "episode_id": "9", "reset_seed": "111", "obs": json.dumps(start_observation.tolist()),
        "next_obs": json.dumps(next_observation.tolist()), "environment_done": "0",
    }]
    good = audit_module.Audit(Path("."))
    audit_module._audit_episode_state(good, rows, start, end, 5, "synthetic")
    assert good.errors == []

    rows[0]["obs"] = json.dumps([9.0, 9.0, 9.0, 9.0])
    bad = audit_module.Audit(Path("."))
    audit_module._audit_episode_state(bad, rows, start, end, 5, "synthetic")
    assert any(item["code"] == "episode_state_observation" for item in bad.errors)


def test_package_cap_counts_the_persistent_evidence_sidecar(tmp_path: Path) -> None:
    folder = tmp_path / "package"
    folder.mkdir()
    (folder / "checkpoint.pt").write_bytes(b"c" * 20)
    (folder / "actor.bin").write_bytes(b"a" * 5)
    (folder / "evidence.npz").write_bytes(b"e" * 30)
    foundation_bytes = 50
    actual = audit_module._package_components(folder, foundation_bytes)
    assert actual["full_persistent_bytes"] == 105
    record = {
        **actual,
        "package_budget_bytes": 100,
        "deployment_state_bytes": foundation_bytes + 20 + 5,
        "package_bytes": actual["full_persistent_bytes"],
        "actor_container_bytes": 5,
    }
    audit = audit_module.Audit(tmp_path)
    audit_module._audit_package(audit, folder, record, foundation_bytes,
                                {"package_budget_bytes": 100}, "synthetic")
    assert any(item["code"] == "package_cap" for item in audit.errors)


"""Focused tests for the single-decision controlled storage runner."""
from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl.storage_decision import (  # noqa: E402
    BoundedGroupEvidence,
    apply_action,
    decide_from_past,
    effective_matrices,
    pair_mapping,
    semantic_hash,
)
from online_rl.write_decision_policy import encode_actor  # noqa: E402
from scripts import run_storage_decision as runner  # noqa: E402
from scripts import run_write_stream as ws  # noqa: E402


class _NullEnv:
    def close(self):
        return None


def _reference(seed: int):
    torch.manual_seed(seed)
    observation_space = spaces.Box(
        low=np.asarray([-2.0] * (OBS_DIM - 1) + [0.0], dtype=np.float32),
        high=np.asarray([2.0] * (OBS_DIM - 1) + [8.0], dtype=np.float32),
        dtype=np.float32,
    )
    return SimpleNamespace(
        policy=TreePolicy(observation_space, spaces.Discrete(4), lr_schedule=lambda _: 1e-2),
        env=_NullEnv(),
    )


def _evidence_row(index: int):
    obs = np.zeros(13, dtype=np.float32)
    obs[0] = (index % 3 + 0.5) / 3.0
    probs = np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    return SimpleNamespace(
        obs=obs,
        action=index % 4,
        probabilities=probs,
        log_prob=float(np.log(probs[index % 4])),
        advantage=float(index) / 10.0,
        returns=float(index) / 20.0,
        episode_id=index // 7,
    )


def test_compensated_actions_preserve_resident_effective_policy_and_actor_size():
    rng = np.random.default_rng(72)
    local = rng.normal(0.0, 0.02, size=(2, 27)).astype(np.float32)
    shared = rng.normal(0.0, 0.02, size=(1, 9)).astype(np.float32)
    physical_groups = np.asarray([0, -1], dtype=np.int8)
    logical_tags = np.asarray([0, 1], dtype=np.int8)
    old = rng.normal(0.0, 0.02, size=(3, 3)).astype(np.float32)
    new = rng.normal(0.0, 0.02, size=(3, 3)).astype(np.float32)
    before = effective_matrices(local, shared, physical_groups)

    for action in ("keep", "replace", "mean", "none"):
        changed_local, changed_shared, changed_groups, diagnostic = apply_action(
            local, shared, physical_groups, logical_tags, old, new, action,
        )
        after = effective_matrices(changed_local, changed_shared, changed_groups)
        np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-7)
        assert diagnostic["max_effective_preservation_error"] <= 1e-7
        encoded = encode_actor(changed_local, changed_shared, changed_groups, codec="int8", budget_bytes=89)
        assert len(encoded.data) == 89


def test_bounded_group_evidence_keeps_latest_rows_and_roundtrips():
    evidence = BoundedGroupEvidence(256)
    evidence.add_rollout([_evidence_row(i) for i in range(300)], group=0, task="A0+", bout=4)
    evidence.add_rollout([_evidence_row(i) for i in range(11)], group=1, task="B1-", bout=5)
    state = evidence.state_dict()
    restored = BoundedGroupEvidence.from_state_dict(state)
    old = restored.arrays(0)
    new = restored.arrays(1)
    assert len(old["action"]) == 256
    assert old["episode_id"][-1] == 299 // 7
    assert old["task"][0] == "A0+"
    assert len(new["action"]) == 11
    np.testing.assert_array_equal(new["behavior_probs"][0], np.full(4, 0.25, dtype=np.float32))


def test_past_surrogate_requires_supported_scores_for_both_groups():
    evidence = BoundedGroupEvidence(256)
    evidence.add_rollout([_evidence_row(i) for i in range(8)], group=0, task="A0+", bout=1)
    history = [0, 0, 0, 0, 0, 0, 1, 1]

    def one_group_supported(rows, candidate, baseline):
        if len(rows["action"]):
            return {"gain": 5.0, "lower": 2.0, "supported": True}
        return {"gain": None, "lower": None, "supported": False}

    result = decide_from_past(
        history, evidence, np.ones((3, 3), dtype=np.float32),
        -np.ones((3, 3), dtype=np.float32), one_group_supported,
    )
    assert result["past_only_selectors"]["frequency"] == "keep"
    assert result["past_only_selectors"]["recency"] == "replace"
    assert result["past_only_selectors"]["surrogate_gain"] == "keep"
    assert result["past_only_selectors"]["surrogate_lower"] == "keep"
    assert result["surrogate"]["frequency_weighted_gain"] is None
    assert result["surrogate"]["frequency_weighted_lower"] is None


def test_known_task_cache_evicts_the_least_recent_resident():
    cache = ws.BoundedTaskCache(2, "lru", seed=17)
    first, second, third = ws.TASKS[:3]
    cache.resolve(first)
    cache.resolve(second)
    cache.resolve(first)
    allocation = cache.resolve(third)
    assert allocation["allocated"] is True
    assert allocation["evicted_task"] == second
    assert cache.slot_to_task[allocation["slot"]] == third


def test_semantic_checkpoint_roundtrip_restores_optimizer_rng_cache_and_evidence(monkeypatch, tmp_path):
    foundation = tmp_path / "foundation.pt"
    dataset = tmp_path / "dataset.bin"
    foundation.write_bytes(b"test W0 source bytes")
    dataset.write_bytes(b"test data fingerprint")
    base = {
        "foundation": str(foundation), "dataset": str(dataset),
        "exploration_epsilon": 0.05, "temperature": 2.0,
    }
    monkeypatch.setattr(ws, "_foundation", lambda _base, seed: _reference(seed))
    config = {
        "local_learning_rate": 0.003, "shared_learning_rate": 0.0,
        "critic_learning_rate": 0.003, "package_budget_bytes": 262144,
        "evidence_rows_per_group": 256, "torch_threads": 1,
    }
    agent = runner._new_agent(2047, base, config)
    agent.cache.resolve(ws.TASKS[0])
    agent.recent_groups = [0, 1, 0, 1, 0, 1, 0, 1]
    agent.slot_group_tags[:] = np.asarray([0, 1], dtype=np.int8)
    agent.evidence.add_rollout([_evidence_row(i) for i in range(16)], group=0, task="A0+", bout=2)
    agent.evidence.add_rollout([_evidence_row(i) for i in range(12)], group=1, task="B1-", bout=3)

    critic_group = next(group for group in agent.optimizer.param_groups if group.get("name") == "critic")
    for parameter in critic_group["params"]:
        parameter.grad = torch.full_like(parameter, 1e-3)
    agent.optimizer.step()
    agent.optimizer.zero_grad(set_to_none=True)
    assert agent.optimizer.state

    foundation_sha = runner._sha(foundation)
    original = runner._capture_learner(agent, foundation_sha)
    original_hash = semantic_hash(original)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(original, checkpoint)
    loaded = runner._capture_learner_from_saved(checkpoint)
    restored = runner._restore_learner(loaded, base, config)
    restored_payload = runner._capture_learner(restored, foundation_sha)
    assert semantic_hash(restored_payload) == original_hash
    assert restored.optimizer.step_count == agent.optimizer.step_count
    assert len(restored.evidence.arrays(0)["action"]) == 16
    assert restored.cache.state_dict()["task_to_slot"] == agent.cache.state_dict()["task_to_slot"]


def test_failed_case_interaction_count_comes_from_persisted_step_rows(tmp_path):
    case = tmp_path / "cases" / "s1_dynamics_old_major"
    prefix = case / "prefix"
    branch_a = case / "branches" / "keep__old_major"
    branch_b = case / "branches" / "replace__new_major"
    prefix.mkdir(parents=True)
    branch_a.mkdir(parents=True)
    branch_b.mkdir(parents=True)
    fields = ("phase", "global_step", "obs")
    for path, count in ((prefix / "steps.csv", 3), (branch_a / "steps.csv", 2), (branch_b / "steps.csv", 1)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(count):
                writer.writerow({"phase": "test", "global_step": index, "obs": "[]"})
    counts = runner._persisted_step_counts(case)
    assert counts == {
        "prefix_environment_steps": 3,
        "branch_environment_steps": 3,
        "environment_steps_total": 6,
    }


def test_pair_mapping_is_deterministic_and_controlled():
    left = pair_mapping(20101, "reward_conflict")
    right = pair_mapping(20101, "reward_conflict")
    assert left == right
    assert left["families"]["old"] == left["families"]["new"]
    assert left["preferences"]["old"] == -left["preferences"]["new"]
    assert left["groups"] == {"old": 0, "new": 1}

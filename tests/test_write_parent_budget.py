"""Checks bounded parent reuse without task labels or trained utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl.write_decision_evidence import EvidenceWindow, RewardAwareGroups  # noqa: E402
from online_rl.write_decision_policy import FlexibleSharedPolicy  # noqa: E402
from online_rl.write_parent_budget import ParentBudgetManager, try_recycle  # noqa: E402


torch.set_num_threads(1)


def _profile(direction: int, *, reward: float = 0.5):
    window = EvidenceWindow(capacity=24)
    xs = {
        0: (1, 4, 7),
        1: (0, 3, 6),
        2: (1, 4, 7),
        3: (2, 5, 7),
    }[direction]
    for band, x in enumerate(xs):
        for action in range(4):
            for _ in range(2):
                y = 4
                next_x, next_y = x, y
                if direction == 0:
                    next_y -= 1
                elif direction == 1:
                    next_x += 1
                elif direction == 2:
                    next_y += 1
                else:
                    next_x -= 1
                obs = np.asarray([x, y, 7, 7], dtype=np.float32) / 8.0
                next_obs = np.asarray([next_x, next_y, 7, 7], dtype=np.float32) / 8.0
                window.observe(obs, action, reward, next_obs, False)
    return window.profile()


def _make_core():
    torch.manual_seed(701)
    reference = TreePolicy(
        observation_space=spaces.Box(
            low=np.asarray([-2.0] * 12 + [0.0], dtype=np.float32),
            high=np.asarray([2.0] * 12 + [8.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )
    core = FlexibleSharedPolicy.from_reference(
        reference, local_slots=6, shared_groups=2, train_critic=False
    )
    with torch.no_grad():
        core.source_basis.zero_()
        core.source_basis[0, -1] = 1.0
    return core


def _profile_store():
    profiles = {
        "movement_counts": np.zeros((6, 3, 4, 4), dtype=np.uint16),
        "reward_xtx": np.zeros((6, 3, 3), dtype=np.float32),
        "reward_xty": np.zeros((6, 3), dtype=np.float32),
        "reward_n": np.zeros(6, dtype=np.float32),
        "seen": np.zeros(6, dtype=np.uint8),
        "shadow_group": np.full(6, -1, dtype=np.int8),
    }
    profile_by_slot = {
        0: _profile(1),
        1: _profile(3),
        2: _profile(0),
        3: _profile(0),
    }
    for slot, profile in profile_by_slot.items():
        profiles["movement_counts"][slot] = profile.movement_counts
        profiles["reward_xtx"][slot] = profile.reward_xtx
        profiles["reward_xty"][slot] = profile.reward_xty
        profiles["reward_n"][slot] = profile.reward_n
        profiles["seen"][slot] = 1
    profiles["shadow_group"][:2] = (0, 1)
    return profiles


def _full_grouper(profiles):
    groups = RewardAwareGroups(max_groups=2, min_support=8, reward_min_samples=16, mode="reward")
    for slot in (0, 1):
        assignment = groups.assign({
            "movement_counts": profiles["movement_counts"][slot],
            "reward_xtx": profiles["reward_xtx"][slot],
            "reward_xty": profiles["reward_xty"][slot],
            "reward_n": profiles["reward_n"][slot],
        })
        assert assignment.group == slot
    return groups


def _configure_actor(core, *, group0_member=True):
    if group0_member:
        core.set_slot_group(0, 0, preserve_effective=False)
    core.set_slot_group(1, 1, preserve_effective=False)
    local = np.zeros((6, 27), dtype=np.float32)
    local[0] = 0.125
    local[1] = 0.0625
    local[2] = 0.25
    local[3] = -0.125
    shared = np.zeros((2, 9), dtype=np.float32)
    shared[0] = 0.25
    shared[1] = -0.125
    core.install_local_matrix(local)
    core.install_shared_matrix(shared)


def _observations():
    obs = torch.zeros((6, OBS_DIM), dtype=torch.float32)
    obs[:, 0] = torch.tensor([0.1, 0.3, 0.5, 0.8, 0.2, 0.9])
    obs[:, -1] = torch.arange(1, 7, dtype=torch.float32)
    return obs


def _probabilities(core):
    with torch.no_grad():
        return core.get_distribution(_observations()).distribution.probs.detach().clone()


def test_recycle_reuses_lru_parent_and_preserves_all_resident_logits():
    profiles = _profile_store()
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)
    generations = np.asarray([1, 1, 2, 2, -1, -1], dtype=np.int64)
    manager = ParentBudgetManager()
    manager.touch(0, 100)
    manager.touch(1, 200)
    expected_probs = _probabilities(core)
    expected_group1_counts = grouper.group_counts[1].copy()
    expected_group_bytes = grouper.nbytes
    expected_profile_shapes = {name: value.shape for name, value in profiles.items()}

    result = try_recycle(
        manager,
        core,
        grouper,
        profiles,
        current_slot=2,
        slot_generations=generations,
        step=300,
    )

    assert result.recycled is True
    assert result.reason == "recycled_lru_parent"
    assert result.group == 0 and result.group_version == 1
    assert result.previous_group_version == 0
    assert result.removed_slots == (0,)
    assert result.added_slots == (2, 3)
    assert result.affected_generations == ((0, 1), (2, 2), (3, 2))
    assert result.committed_profile_slots == (2, 3)
    assert result.seen_updates == ((0, True), (2, True), (3, True))
    assert result.grouper.nbytes == expected_group_bytes
    assert result.grouper.group_counts[0].sum() == 2 * int(profiles["movement_counts"][2].sum())
    assert np.count_nonzero(result.grouper.group_counts[0, :, :, 1]) == 0
    np.testing.assert_array_equal(result.grouper.group_counts[1], expected_group1_counts)
    np.testing.assert_array_equal(_probabilities(core), expected_probs)
    np.testing.assert_array_equal(core.read_local_matrix()[0], np.full(27, 0.375, dtype=np.float32))
    np.testing.assert_array_equal(core.read_shared_matrix()[0], np.zeros(9, dtype=np.float32))
    assert core.group_for_slot(0) == -1
    assert core.group_for_slot(1) == 1
    assert core.group_for_slot(2) == core.group_for_slot(3) == 0
    assert profiles["shadow_group"].tolist() == [-1, 1, 0, 0, -1, -1]
    assert {name: value.shape for name, value in profiles.items()} == expected_profile_shapes
    assert manager.nbytes == 48
    assert manager.touch_count.tolist() == [2, 1]


def test_empty_resident_parent_has_priority_without_partner():
    profiles = _profile_store()
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core, group0_member=False)
    manager = ParentBudgetManager()
    manager.touch(0, 200)
    manager.touch(1, 100)

    result = try_recycle(
        manager,
        core,
        grouper,
        profiles,
        current_slot=2,
        slot_generations=np.ones(6, dtype=np.int64),
        step=250,
    )

    assert result.recycled is True
    assert result.reason == "recycled_empty_parent"
    assert result.group == 0
    assert result.removed_slots == ()
    assert result.added_slots == (2,)
    assert result.committed_profile_slots == (2,)
    assert core.group_for_slot(2) == 0
    assert core.group_for_slot(1) == 1


def test_one_current_profile_cannot_be_its_own_compatibility_partner():
    profiles = _profile_store()
    profiles["seen"][3:] = 0
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)
    manager = ParentBudgetManager()
    before_groups = grouper.state_dict()
    before_probs = _probabilities(core)
    before_version = manager.version.copy()

    result = try_recycle(
        manager,
        core,
        grouper,
        profiles,
        current_slot=2,
        slot_generations=np.ones(6, dtype=np.int64),
        step=30,
    )

    assert result.recycled is False
    assert result.reason == "no_compatible_unpaired_profile"
    assert result.grouper is grouper
    np.testing.assert_array_equal(grouper.group_counts, before_groups["movement_counts"])
    np.testing.assert_array_equal(_probabilities(core), before_probs)
    np.testing.assert_array_equal(manager.version, before_version)


def test_insufficient_current_profile_does_not_recycle_or_mutate():
    profiles = _profile_store()
    profiles["movement_counts"][2].fill(0)
    profiles["reward_xtx"][2].fill(0)
    profiles["reward_xty"][2].fill(0)
    profiles["reward_n"][2] = 0
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)
    manager = ParentBudgetManager()
    before_probs = _probabilities(core)

    result = try_recycle(
        manager,
        core,
        grouper,
        profiles,
        current_slot=2,
        slot_generations=np.ones(6, dtype=np.int64),
        step=30,
    )

    assert result.recycled is False
    assert result.reason == "not_capacity_full"
    assert result.assignment.reason == "insufficient_profile_support"
    assert result.grouper is grouper
    np.testing.assert_array_equal(_probabilities(core), before_probs)
    np.testing.assert_array_equal(manager.version, np.zeros(2, dtype=np.uint64))


def test_seen_but_insufficient_partner_profile_does_not_recycle():
    profiles = _profile_store()
    profiles["seen"][3] = 1
    profiles["movement_counts"][3].fill(0)
    profiles["reward_xtx"][3].fill(0)
    profiles["reward_xty"][3].fill(0)
    profiles["reward_n"][3] = 0
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)
    manager = ParentBudgetManager()
    before_probs = _probabilities(core)

    result = try_recycle(
        manager,
        core,
        grouper,
        profiles,
        current_slot=2,
        slot_generations=np.ones(6, dtype=np.int64),
        step=30,
    )

    assert result.recycled is False
    assert result.reason == "no_compatible_unpaired_profile"
    np.testing.assert_array_equal(_probabilities(core), before_probs)
    np.testing.assert_array_equal(manager.version, np.zeros(2, dtype=np.uint64))


def test_vacant_generation_with_stale_seen_profile_cannot_be_partner():
    profiles = _profile_store()
    profiles["seen"][4] = 1
    profiles["movement_counts"][4] = profiles["movement_counts"][3]
    profiles["reward_xtx"][4] = profiles["reward_xtx"][3]
    profiles["reward_xty"][4] = profiles["reward_xty"][3]
    profiles["reward_n"][4] = profiles["reward_n"][3]
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)

    result = try_recycle(
        ParentBudgetManager(),
        core,
        grouper,
        profiles,
        current_slot=2,
        slot_generations=np.asarray([1, 1, 2, -1, -1, -1], dtype=np.int64),
        step=30,
    )

    assert result.recycled is False
    assert result.reason == "no_compatible_unpaired_profile"


def test_partner_must_match_reward_as_well_as_dynamics():
    profiles = _profile_store()
    incompatible = _profile(0, reward=-0.5)
    profiles["movement_counts"][3] = incompatible.movement_counts
    profiles["reward_xtx"][3] = incompatible.reward_xtx
    profiles["reward_xty"][3] = incompatible.reward_xty
    profiles["reward_n"][3] = incompatible.reward_n
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)

    result = try_recycle(
        ParentBudgetManager(),
        core,
        grouper,
        profiles,
        current_slot=2,
        slot_generations=np.ones(6, dtype=np.int64),
        step=30,
    )

    assert result.recycled is False
    assert result.reason == "no_compatible_unpaired_profile"


def test_matching_existing_parent_never_triggers_recycle():
    profiles = _profile_store()
    matching = _profile(1)
    profiles["movement_counts"][2] = matching.movement_counts
    profiles["reward_xtx"][2] = matching.reward_xtx
    profiles["reward_xty"][2] = matching.reward_xty
    profiles["reward_n"][2] = matching.reward_n
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)
    manager = ParentBudgetManager()
    before_probs = _probabilities(core)

    result = try_recycle(
        manager,
        core,
        grouper,
        profiles,
        current_slot=2,
        slot_generations=np.ones(6, dtype=np.int64),
        step=30,
    )

    assert result.recycled is False
    assert result.reason == "not_capacity_full"
    assert result.assignment.group == 0
    np.testing.assert_array_equal(_probabilities(core), before_probs)
    np.testing.assert_array_equal(manager.version, np.zeros(2, dtype=np.uint64))


def test_parent_budget_state_round_trip_is_fixed_size_and_validated():
    manager = ParentBudgetManager()
    manager.touch(1, 10)
    manager.touch(1, 15)
    state = manager.state_dict()
    restored = ParentBudgetManager.from_state_dict(state)

    assert manager.nbytes == restored.nbytes == 48
    np.testing.assert_array_equal(restored.last_used, manager.last_used)
    np.testing.assert_array_equal(restored.touch_count, manager.touch_count)
    np.testing.assert_array_equal(restored.version, manager.version)
    bad = dict(state)
    bad["last_used"] = np.zeros(3, dtype=np.uint64)
    with pytest.raises(ValueError, match="length two"):
        ParentBudgetManager.from_state_dict(bad)

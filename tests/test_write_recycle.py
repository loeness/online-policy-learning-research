"""Focused tests for the separate parent-recycling stream wrapper."""
from __future__ import annotations

import json
import math
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
from online_rl.write_decision_evidence import EvidenceWindow, RewardAwareGroups  # noqa: E402
from online_rl.write_decision_gate import AnchorBank  # noqa: E402
from online_rl.write_decision_policy import FlexibleSharedPolicy  # noqa: E402
from online_rl.write_cumulative_evidence import CumulativeEvidence  # noqa: E402
from online_rl.write_parent_budget import ParentBudgetManager  # noqa: E402
from scripts import run_write_recycle as recycle  # noqa: E402
from scripts import run_write_stream as stream  # noqa: E402


torch.set_num_threads(1)
# Importing the executable module installs hooks in every process. Keep normal
# stream tests isolated; the fixture below installs them only for these tests.
recycle.uninstall_hooks()


@pytest.fixture(autouse=True)
def _hooks():
    recycle.install_hooks()
    yield
    recycle.uninstall_hooks()


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


def _make_core(shared_groups: int = 2) -> FlexibleSharedPolicy:
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
        reference, local_slots=6, shared_groups=shared_groups, train_critic=False
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
    by_slot = {0: _profile(1), 1: _profile(3), 2: _profile(0), 3: _profile(0)}
    for slot, profile in by_slot.items():
        profiles["movement_counts"][slot] = profile.movement_counts
        profiles["reward_xtx"][slot] = profile.reward_xtx
        profiles["reward_xty"][slot] = profile.reward_xty
        profiles["reward_n"][slot] = profile.reward_n
        profiles["seen"][slot] = 1
    profiles["shadow_group"][:2] = (0, 1)
    return profiles, by_slot


def _full_grouper(profiles):
    grouper = RewardAwareGroups(
        max_groups=2, min_support=8, reward_min_samples=16, mode="reward"
    )
    for slot in (0, 1):
        result = grouper.assign({
            "movement_counts": profiles["movement_counts"][slot],
            "reward_xtx": profiles["reward_xtx"][slot],
            "reward_xty": profiles["reward_xty"][slot],
            "reward_n": profiles["reward_n"][slot],
        })
        assert result.group == slot
    return grouper


def _configure_actor(core):
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


def _obs_batch():
    obs = torch.zeros((6, OBS_DIM), dtype=torch.float32)
    obs[:, 0] = torch.tensor([0.1, 0.3, 0.5, 0.8, 0.2, 0.9])
    obs[:, -1] = torch.arange(1, 7, dtype=torch.float32)
    return obs


def _probs(core):
    with torch.no_grad():
        return core.get_distribution(_obs_batch()).distribution.probs.detach().clone()


def _agent(core, grouper, profiles, *, parent_recycle=True, base_path=None):
    slots = int(core.local_slots)
    counts = {
        "raw_steps": 30, "environment_steps": 0,
        "group_retries": 0, "group_abstentions": 0, "group_max_abstentions": 0,
        "effective_preserving_bindings": 0, "cold_prior_bindings": 0,
        "parent_recycles": 0, "parent_detached_slots": 0, "parent_joined_slots": 0,
    }
    task_names = stream.TASKS
    return SimpleNamespace(
        slots=slots, core=core, grouper=grouper, profiles=profiles,
        profile_reasons=["not_profiled"] * slots,
        grouped_generation=np.full(slots, -1, dtype=np.int64),
        group_attempt_support=np.zeros(slots, dtype=np.uint64),
        slot_generation=np.asarray([1, 1, 2, 2, -1, -1][:slots], dtype=np.int64),
        fitted_steps=np.asarray([10, 10, 3, 3, 0, 0][:slots], dtype=np.uint64),
        anchors=AnchorBank(slots=slots, per_slot=4, seed=43),
        parent_budget_manager=ParentBudgetManager(), counts=counts,
        cycle_state=None, cycle_step=0, rollout=[],
        run_config={
            "phase": "known", "group_retry_interval": 16,
            "actor_codec": "float32", "actor_budget_bytes": 1024,
            "package_budget_bytes": 262144,
            "parent_recycle": parent_recycle, "cumulative_profiles": True,
        },
        condition="reward_shared", external_condition="reward_shared", seed=7,
        optimizer=torch.optim.Adam(core.parameters()), action_rng=np.random.default_rng(9),
        base_config={"foundation": str(base_path) if base_path else ""},
        router=None, cache=None, windows=[EvidenceWindow(24) for _ in range(slots)],
        reward_by_task={name: 0.0 for name in task_names},
        steps_by_task={name: 0 for name in task_names},
        observed_reward_by_task={name: 0.0 for name in task_names},
        observed_steps_by_task={name: 0 for name in task_names},
        goal_episodes_by_task={name: 0 for name in task_names},
        complete_episodes_by_task={name: 0 for name in task_names},
        positive_complete_episodes_by_task={name: 0 for name in task_names},
        next_episode_id=0, current_known_slot=None, decision_cpu_s=0.0,
        evaluation_cpu_s=0.0,
    )


class _Rows:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(dict(row))


def test_recycle_preserves_resident_logits_before_roundtrip_and_records_parent_versions(monkeypatch):
    profiles, by_slot = _profile_store()
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)
    agent = _agent(core, grouper, profiles)
    agent.parent_budget_manager.touch(0, 10)
    agent.parent_budget_manager.touch(1, 20)
    for slot in (0, 2, 3):
        generation = int(agent.slot_generation[slot])
        agent.anchors.add(slot, generation, [{
            "obs": _obs_batch()[slot].numpy(), "action": 0,
            "log_prob": math.log(0.25), "probabilities": np.full(4, 0.25),
            "advantage": 1.0, "returns": 1.0, "episode_id": slot,
            "generation": generation,
        }])
    before = _probs(core)
    basis_before = core.source_basis.detach().clone()
    captured = {}
    original_roundtrip = stream.step1.roundtrip_actor

    def inspect_before_quantize(current_agent, config):
        captured["logits"] = _probs(current_agent.core)
        result = original_roundtrip(current_agent, config)
        captured["roundtrip"] = result
        return result

    monkeypatch.setattr(stream.step1, "roundtrip_actor", inspect_before_quantize)
    rows = _Rows()
    result = recycle._wrapped_maybe_assign_group(
        agent, 2, by_slot[2], rows, bout=5
    )

    assert result["committed"] is True
    assert result["group"] == 0
    assert result["parent_recycle"]["previous_group_version"] == 0
    assert result["parent_recycle"]["group_version"] == 1
    assert agent.counts["parent_recycles"] == 1
    assert agent.counts["parent_detached_slots"] == 1
    assert agent.counts["parent_joined_slots"] == 2
    assert agent.core.group_for_slot(0) == -1
    assert agent.core.group_for_slot(1) == 1
    assert agent.core.group_for_slot(2) == agent.core.group_for_slot(3) == 0
    torch.testing.assert_close(captured["logits"], before, rtol=0, atol=0)
    torch.testing.assert_close(_probs(core), before, rtol=0, atol=0)
    torch.testing.assert_close(core.source_basis, basis_before, rtol=0, atol=0)
    assert all(agent.anchors.arrays(slot, int(agent.slot_generation[slot]))["obs"].shape[0] == 0
               for slot in (0, 2, 3))
    assert len(rows.rows) == 1
    scores = json.loads(rows.rows[0]["scores"])
    assert scores["parent_recycle"]["parent_versions"] == [1, 0]
    assert scores["parent_recycle"]["added_slots"] == [2, 3]


def test_parent_recycle_disabled_retains_fixed_capacity_and_manager_is_untouched():
    profiles, by_slot = _profile_store()
    grouper = _full_grouper(profiles)
    core = _make_core()
    _configure_actor(core)
    agent = _agent(core, grouper, profiles, parent_recycle=False)
    manager_before = agent.parent_budget_manager.state_dict()
    before = _probs(core)
    rows = _Rows()

    result = recycle._wrapped_maybe_assign_group(agent, 2, by_slot[2], rows, bout=0)

    assert result["committed"] is False
    assert result["reason"] == "max_groups_reached"
    assert agent.parent_budget_manager.version.tolist() == manager_before["parent_version"].tolist()
    assert agent.parent_budget_manager.touch_count.tolist() == manager_before["touch_count"].tolist()
    torch.testing.assert_close(_probs(core), before, rtol=0, atol=0)
    assert agent.counts["parent_recycles"] == 0
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(recycle._ORIGINALS, "route_before_action", lambda *_args: {"slot": 1})
    try:
        recycle._wrapped_route_before_action(agent, np.zeros(4, dtype=np.float32))
    finally:
        monkeypatch.undo()
    assert agent.parent_budget_manager.touch_count.tolist() == [0, 0]


def test_bootstrap_route_does_not_change_parent_lru_recency():
    profiles, _ = _profile_store()
    core = _make_core()
    _configure_actor(core)
    agent = _agent(core, _full_grouper(profiles), profiles)
    agent.current_known_slot = 0
    agent.cache = SimpleNamespace(slot_to_task=["resident"] * 6)
    obs = np.asarray([0.25, 0.5, 0.875, 0.875], dtype=np.float32)

    recycle._wrapped_record_visible_event(
        agent, obs, 0, 0.5, obs, False, task_name="not-a-routing-feature", probe=False
    )
    route = recycle._wrapped_route_before_action(agent, obs)

    assert route["slot"] == 0
    assert agent.parent_budget_manager.touch_count.tolist() == [0, 0]
    assert agent._parent_bootstrap_route_pending is False

    agent.counts["raw_steps"] += 1
    recycle._wrapped_route_before_action(agent, obs)
    assert agent.parent_budget_manager.touch_count.tolist() == [1, 0]


def test_group_zero_condition_skips_recycling_without_changing_actor():
    profiles, by_slot = _profile_store()
    core = _make_core(shared_groups=0)
    local_before = core.read_local_matrix()
    agent = _agent(core, RewardAwareGroups(max_groups=1), profiles)
    before = _probs(core)

    result = recycle._wrapped_maybe_assign_group(agent, 2, by_slot[2], None, bout=0)

    assert result["reason"] == "grouping_disabled"
    np.testing.assert_array_equal(core.read_local_matrix(), local_before)
    torch.testing.assert_close(_probs(core), before, rtol=0, atol=0)
    assert agent.parent_budget_manager.version.tolist() == [0, 0]


def test_checkpoint_persists_manager_and_recomputes_full_package_bytes(tmp_path, monkeypatch):
    foundation = tmp_path / "foundation.pt"
    foundation.write_bytes(b"W0 fixture" * 17)
    profiles, _ = _profile_store()
    core = _make_core()
    grouper = RewardAwareGroups(max_groups=2, mode="reward")
    agent = _agent(core, grouper, profiles, base_path=foundation)
    agent.cumulative_evidence = CumulativeEvidence(6)
    sample_obs = np.asarray([0.125, 0.25, 0.5, 0.75], dtype=np.float32)
    agent.cumulative_evidence.observe(2, sample_obs, 0, 1.0, sample_obs, False)
    agent.optimizer = stream.step1.LeafAdam([{"params": list(core.parameters()), "lr": 1e-3}])
    agent.run_config.update({"actor_codec": "int8", "actor_budget_bytes": 236,
                             "independent7_actor_budget_bytes": 236, "phase": "known"})
    agent.parent_budget_manager.touch(1, 19)
    agent.parent_budget_manager.version[1] = np.uint64(4)
    path = tmp_path / "checkpoint.pt"
    capacity_rows = []

    result = stream._save_stream_checkpoint(agent, path, "foundation-hash", capacity_rows)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["parent_budget_manager"]["parent_version"].tolist() == [0, 4]
    assert payload["cumulative_evidence"]["nbytes"] == agent.cumulative_evidence.nbytes
    assert result["parent_state_bytes"] == 48
    assert result["cumulative_state_bytes"] == agent.cumulative_evidence.nbytes
    assert result["checkpoint_bytes"] == path.stat().st_size
    assert result["foundation_bytes"] == foundation.stat().st_size
    assert result["full_persistent_bytes"] == foundation.stat().st_size + path.stat().st_size
    assert result["full_persistent_bytes"] <= 262144
    assert capacity_rows[-1]["checkpoint_bytes"] == path.stat().st_size

    monkeypatch.setitem(
        recycle._ORIGINALS, "restore_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(counts={}),
    )
    restored = stream.restore_stream_checkpoint(path)
    assert isinstance(restored.parent_budget_manager, ParentBudgetManager)
    assert restored.parent_budget_manager.version.tolist() == [0, 4]
    assert restored.parent_budget_manager.last_used.tolist() == [0, 19]
    assert restored.cumulative_evidence.profile_for_slot(2).reward_n == 1

    old_payload = dict(payload)
    old_payload.pop("parent_budget_manager")
    old_payload.pop("cumulative_evidence")
    old_path = tmp_path / "older_checkpoint.pt"
    torch.save(old_payload, old_path)
    monkeypatch.setitem(
        recycle._ORIGINALS, "restore_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            counts={}, slots=6,
            run_config={"phase": "known", "cumulative_profiles": True},
        ),
    )
    older = stream.restore_stream_checkpoint(old_path)
    assert older.parent_budget_manager.version.tolist() == [0, 0]
    assert older.parent_budget_manager.touch_count.tolist() == [0, 0]
    assert older.cumulative_evidence.profile_for_slot(2).reward_n == 0


def test_known_event_builds_cumulative_profile_used_by_assignment(monkeypatch):
    profiles, _ = _profile_store()
    core = _make_core(shared_groups=0)
    agent = _agent(core, RewardAwareGroups(max_groups=1), profiles)
    agent.current_known_slot = 2
    agent.cumulative_evidence = CumulativeEvidence(6)
    obs = np.asarray([0.25, 0.5, 0.875, 0.875], dtype=np.float32)
    next_obs = np.asarray([0.25, 0.375, 0.875, 0.875], dtype=np.float32)

    event = recycle._wrapped_record_visible_event(
        agent, obs, 0, -0.5, next_obs, False, task_name="observer-only", probe=True
    )
    cumulative_profile = agent.cumulative_evidence.profile_for_slot(2)
    result = recycle._wrapped_maybe_assign_group(agent, 2, _profile(3), None, bout=0)

    assert event["active_slot"] == 2
    assert result["reason"] == "grouping_disabled"
    np.testing.assert_array_equal(
        agent.profiles["movement_counts"][2], cumulative_profile.movement_counts
    )
    np.testing.assert_array_equal(agent.profiles["reward_xtx"][2], cumulative_profile.reward_xtx)
    assert agent.profiles["reward_n"][2] == cumulative_profile.reward_n == 1


def test_cumulative_evidence_is_cleared_on_slot_generation_reset(monkeypatch):
    profiles, _ = _profile_store()
    agent = _agent(_make_core(), _full_grouper(profiles), profiles)
    agent.cumulative_evidence = CumulativeEvidence(6)
    obs = np.asarray([0.25, 0.5, 0.875, 0.875], dtype=np.float32)
    agent.cumulative_evidence.observe(3, obs, 0, 0.75, obs, False)
    monkeypatch.setitem(recycle._ORIGINALS, "reset_allocated_slot", lambda *_args, **_kwargs: 9)

    generation = recycle._wrapped_reset_allocated_slot(agent, 3, reason="test_reset")

    assert generation == 9
    assert agent.cumulative_evidence.profile_for_slot(3).reward_n == 0
    assert agent.cumulative_evidence.nbytes > 0


def test_loader_freezes_comparison_flags():
    config = stream._load_config()
    assert config["parent_recycle"] is True
    assert config["cumulative_profiles"] is True
    recycle._FORCE_NO_RECYCLE = True
    try:
        no_recycle = stream._load_config()
    finally:
        recycle._FORCE_NO_RECYCLE = False
    assert no_recycle["parent_recycle"] is False
    assert no_recycle["cumulative_profiles"] is True


def test_new_agent_initializes_fixed_cumulative_and_parent_state(monkeypatch):
    monkeypatch.setitem(
        recycle._ORIGINALS, "new_agent",
        lambda _condition, _seed, config, _base: SimpleNamespace(
            slots=7, run_config=dict(config), counts={}
        ),
    )

    agent = recycle._wrapped_new_agent(
        "independent7_lru", 11,
        {"phase": "known", "cumulative_profiles": True, "parent_recycle": False}, {},
    )

    assert agent.run_config["parent_recycle"] is False
    assert agent.run_config["cumulative_profiles"] is True
    assert isinstance(agent.cumulative_evidence, CumulativeEvidence)
    assert agent.cumulative_evidence.slots == 7
    assert agent.parent_budget_manager.nbytes == 48
    assert agent.counts["parent_recycles"] == 0

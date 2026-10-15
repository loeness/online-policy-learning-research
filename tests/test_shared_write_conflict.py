"""Focused contracts for the same-dynamics reward-conflict diagnostic."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.shared_write_conflict_env import SharedWriteConflictEnv
from online_rl.shared_write_grouping import DynamicsGrouper
from scripts.run_shared_write_conflict import (
    _assert_probe_pair_equal,
    build_probe_profile,
    collect_uniform_probe,
    evaluate_stage,
    make_agent,
    restore_run,
    save_checkpoint,
)


def test_reward_sign_reverses_without_changing_transition_or_termination() -> None:
    plus = SharedWriteConflictEnv("plus")
    minus = SharedWriteConflictEnv("minus")
    plus_obs, _ = plus.reset(seed=410)
    minus_obs, _ = minus.reset(seed=410)
    np.testing.assert_array_equal(plus_obs, minus_obs)

    for action in [0, 1, 2, 3] * 8:
        plus_next, plus_reward, plus_term, plus_trunc, plus_info = plus.step(action)
        minus_next, minus_reward, minus_term, minus_trunc, minus_info = minus.step(action)
        np.testing.assert_array_equal(plus_next, minus_next)
        assert minus_reward == pytest.approx(-plus_reward)
        assert (plus_term, plus_trunc, plus_info) == (minus_term, minus_trunc, minus_info)
        if plus_term or plus_trunc:
            plus_obs, _ = plus.reset()
            minus_obs, _ = minus.reset()
            np.testing.assert_array_equal(plus_obs, minus_obs)
        else:
            plus_obs, minus_obs = plus_next, minus_next
    assert plus.observation_space.shape == minus.observation_space.shape == (4,)


def test_profile_builder_ignores_reward_and_mode_and_groups_equal_profiles() -> None:
    rows = []
    examples = [
        (np.array([0.0, 0.5, 1.0, 0.5], dtype=np.float32), 0,
         np.array([0.0, 0.375, 1.0, 0.5], dtype=np.float32)),
        (np.array([0.375, 0.5, 0.0, 0.5], dtype=np.float32), 1,
         np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)),
        (np.array([0.75, 0.5, 0.0, 0.5], dtype=np.float32), 2,
         np.array([0.75, 0.625, 0.0, 0.5], dtype=np.float32)),
        (np.array([0.25, 0.5, 0.0, 0.5], dtype=np.float32), 3,
         np.array([0.125, 0.5, 0.0, 0.5], dtype=np.float32)),
    ]
    for obs, action, nxt in examples:
        for _ in range(2):
            rows.append({
                "mode": "plus", "obs": obs, "action": action,
                "reward": 1.25, "next_obs": nxt,
            })
            rows.append({
                "mode": "minus", "obs": obs, "action": action,
                "reward": -1.25, "next_obs": nxt,
            })

    plus_profile = build_probe_profile([row for row in rows if row["mode"] == "plus"])
    minus_profile = build_probe_profile([row for row in rows if row["mode"] == "minus"])
    np.testing.assert_array_equal(plus_profile, minus_profile)

    grouper = DynamicsGrouper()
    first = grouper.assign(plus_profile)
    second = grouper.assign(minus_profile)
    assert first.group == second.group == 0
    assert first.reason == "first_group"
    assert second.reason == "matched"


def test_uniform_probe_reuses_exact_state_and_action_streams_across_reward_modes() -> None:
    plus_rows, plus_profile = collect_uniform_probe("plus", seed=1123, steps=64)
    minus_rows, minus_profile = collect_uniform_probe("minus", seed=1123, steps=64)
    repeated_rows, repeated_profile = collect_uniform_probe("plus", seed=1123, steps=64)
    _assert_probe_pair_equal(plus_rows, minus_rows)
    for first, repeated in zip(plus_rows, repeated_rows):
        assert first["action"] == repeated["action"]
        np.testing.assert_array_equal(first["obs"], repeated["obs"])
        np.testing.assert_array_equal(first["next_obs"], repeated["next_obs"])
    np.testing.assert_array_equal(plus_profile, minus_profile)
    np.testing.assert_array_equal(plus_profile, repeated_profile)


def test_factories_keep_two_shared_groups_but_use_only_group_zero() -> None:
    config = json.loads((ROOT / "configs/shared_write_conflict.json").read_text())
    independent, _ = make_agent("independent_conflict", 853, config)
    shared, _ = make_agent("shared_conflict", 853, config)
    frozen, _ = make_agent("frozen_conflict", 853, config)

    assert not tuple(independent.core.shared_parameters())
    for agent in (shared, frozen):
        assert len(tuple(agent.core.shared_parameters())) == 2
        assert [agent.core.group_for_slot(slot) for slot in range(6)] == [0, 0, -1, -1, -1, -1]
        np.testing.assert_array_equal(agent.core.read_shared_matrix()[1], np.zeros(9))
    assert [group["name"] for group in shared.optimizer.param_groups] == ["local", "critic", "shared"]
    assert [group["name"] for group in independent.optimizer.param_groups] == ["local", "critic"]


def test_checkpoint_round_trip_restores_actor_grouping_and_probe_state(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/shared_write_conflict.json").read_text())
    config.update(condition="shared_conflict", seed=853)
    agent, base = make_agent("shared_conflict", 853, config)
    agent.profiles = {
        "plus": np.zeros((3, 4, 4), dtype=np.uint16),
        "minus": np.zeros((3, 4, 4), dtype=np.uint16),
    }
    agent.counts["minibatch_updates"] = 17
    agent.optimizer.step_count = 29
    with torch.no_grad():
        for slot in range(6):
            agent.core._nodes[str(slot + 1)].A.fill_(0.001 * (slot + 1))
        agent.core._shared_coefficients[0].C.fill_(0.02)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_checkpoint(agent, tmp_path, "initial", config, base)

    expected_python_rng = random.getstate()
    expected_numpy_rng = np.random.get_state()
    expected_torch_rng = torch.get_rng_state().clone()
    restored, _ = restore_run(tmp_path, "initial", restore_global_rng=True)
    observation = torch.tensor([[0.2, 0.3, 0.8, 0.5, *([0.0] * 8), 1.0]])
    expected = agent.core.get_distribution(observation).distribution.probs.detach()
    actual = restored.core.get_distribution(observation).distribution.probs.detach()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.counts["minibatch_updates"] == 17
    assert restored.optimizer.step_count == 29
    assert restored.grouper.state_dict()["config"] == agent.grouper.state_dict()["config"]
    assert restored.core.group_for_slot(0) == restored.core.group_for_slot(1) == 0
    assert restored.core.group_for_slot(2) == -1
    assert restored.profiles.keys() == agent.profiles.keys()
    assert np.array_equal(restored.slot_generation, agent.slot_generation)
    assert (tmp_path / "actor_initial.bin").stat().st_size <= config["actor_budget_bytes"]
    assert restored.restore_metadata["whole_package_bytes"] <= config["whole_package_cap_bytes"]
    assert random.getstate() == expected_python_rng
    restored_numpy_rng = np.random.get_state()
    assert expected_numpy_rng[0] == restored_numpy_rng[0]
    np.testing.assert_array_equal(expected_numpy_rng[1], restored_numpy_rng[1])
    assert expected_numpy_rng[2:] == restored_numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)


def test_evaluation_repeats_paired_plans_without_mutating_training_rng() -> None:
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/shared_write_conflict.json").read_text())
    agent, _ = make_agent("shared_conflict", 853, config)
    before_action_rng = json.dumps(agent.action_rng.bit_generator.state, sort_keys=True)
    before_torch_rng = torch.get_rng_state().clone()
    before_python_rng = random.getstate()
    before_numpy_rng = np.random.get_state()
    before_state = {name: value.detach().clone() for name, value in agent.core.state_dict().items()}

    first, first_steps = evaluate_stage(agent, "initial", episodes=2, seed=93853)
    second, second_steps = evaluate_stage(agent, "initial", episodes=2, seed=93853)
    assert first == second
    assert first_steps == second_steps
    assert set(first) == {"plus", "minus"}
    assert first["minus"]["goal_success_rate"] is None
    assert first["plus"]["episodes"][0]["reset_seed"] == first["minus"]["episodes"][0]["reset_seed"]
    assert first["plus"]["episodes"][0]["action_seed"] == first["minus"]["episodes"][0]["action_seed"]
    for mode in ("plus", "minus"):
        for row in first[mode]["episodes"]:
            assert len(row["actions"]) == row["steps"]
            assert row["task_success"] == (row["goal_reached"] if mode == "plus" else None)
    assert json.dumps(agent.action_rng.bit_generator.state, sort_keys=True) == before_action_rng
    assert torch.equal(torch.get_rng_state(), before_torch_rng)
    assert random.getstate() == before_python_rng
    after_numpy_rng = np.random.get_state()
    assert before_numpy_rng[0] == after_numpy_rng[0]
    np.testing.assert_array_equal(before_numpy_rng[1], after_numpy_rng[1])
    assert before_numpy_rng[2:] == after_numpy_rng[2:]
    for name, value in agent.core.state_dict().items():
        assert torch.equal(value, before_state[name]), name

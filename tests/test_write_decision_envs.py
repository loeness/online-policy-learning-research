from collections import deque
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.c1_envs import C1Context, C1NavigationEnv, mapping_for_state  # noqa: E402
from online_rl.write_decision_envs import (  # noqa: E402
    DOMAINS,
    DetourWriteDecisionEnv,
    TaskSpec,
    WriteDecisionNavigationEnv,
    make_env,
    task_from_name,
    task_name,
    task_specs,
)


def _cell(obs: np.ndarray) -> tuple[int, int]:
    return tuple(int(round(float(value) * 8)) for value in obs[:2])  # type: ignore[return-value]


def _goal(obs: np.ndarray) -> tuple[int, int]:
    return tuple(int(round(float(value) * 8)) for value in obs[2:])  # type: ignore[return-value]


def test_registry_has_12_distinct_round_trippable_preference_specs() -> None:
    registry = task_specs()
    assert len(registry) == len(set(registry)) == 12
    assert list(registry) == [
        f"{family}{variant}{sign}"
        for family in ("A", "B")
        for variant in range(3)
        for sign in ("+", "-")
    ]
    assert {(spec.family, spec.variant, spec.preference) for spec in registry.values()} == {
        (family, variant, preference)
        for family in ("A", "B")
        for variant in range(3)
        for preference in (1, -1)
    }
    assert all(task_from_name(name) == spec for name, spec in registry.items())
    assert all(task_name(spec) == name for name, spec in registry.items())

    with pytest.raises(ValueError):
        task_from_name("A3+")
    with pytest.raises(ValueError):
        TaskSpec("A", 0, 0)
    with pytest.raises(ValueError):
        TaskSpec("B", True, 1)


@pytest.mark.parametrize("domain", DOMAINS)
def test_preference_pair_has_same_trajectory_and_exactly_opposite_rewards(domain: str) -> None:
    positive = make_env(TaskSpec("B", 1, 1), domain=domain)
    negative = make_env(TaskSpec("B", 1, -1), domain=domain)
    obs_pos, info_pos = positive.reset(seed=31415)
    obs_neg, info_neg = negative.reset(seed=31415)
    np.testing.assert_array_equal(obs_pos, obs_neg)
    assert info_pos == info_neg

    for action in (0, 1, 2, 3) * 16:
        obs_pos, reward_pos, term_pos, trunc_pos, info_pos = positive.step(action)
        obs_neg, reward_neg, term_neg, trunc_neg, info_neg = negative.step(action)
        np.testing.assert_array_equal(obs_pos, obs_neg)
        assert reward_neg == pytest.approx(-reward_pos)
        assert (term_pos, trunc_pos, info_pos) == (term_neg, trunc_neg, info_neg)
        if term_pos or trunc_pos:
            break


@pytest.mark.parametrize("domain", DOMAINS)
def test_set_task_mid_episode_changes_no_state_and_emits_no_task_marker(domain: str) -> None:
    env = make_env(TaskSpec("A", 0, 1), domain=domain)
    obs, _ = env.reset(seed=27)
    env.step(1)
    snapshot = (env._position, env._goal, env._episode_length, env.observer_counters())

    env.set_task(TaskSpec("B", 2, -1))

    assert (env._position, env._goal, env._episode_length, env.observer_counters()) == snapshot
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info) == {"goal_reached", "episode_length", "distance"}
    assert not any(token in key for key in info for token in ("family", "variant", "preference", "task", "context"))


def test_detour_free_cells_are_connected_and_reset_endpoints_are_reachable() -> None:
    env = make_env(TaskSpec("A", 0, 1), domain="detour")
    assert isinstance(env, DetourWriteDecisionEnv)
    assert len(env.WALL_CELLS) == 8
    assert (4, 4) not in env.WALL_CELLS

    free_cells = {(x, y) for x in range(9) for y in range(9)} - env.WALL_CELLS
    start = next(iter(free_cells))
    seen = {start}
    pending = deque([start])
    while pending:
        x, y = pending.popleft()
        for dx, dy in env._DELTAS:
            neighbor = (x + dx, y + dy)
            if neighbor in free_cells and neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    assert seen == free_cells

    for seed in range(20):
        obs, info = env.reset(seed=seed)
        x, y = _cell(obs)
        goal_x, goal_y = _goal(obs)
        assert (x, y) not in env.WALL_CELLS
        assert (goal_x, goal_y) not in env.WALL_CELLS
        assert abs(goal_x - x) + abs(goal_y - y) >= 2
        assert info["distance"] >= 2


def test_detour_censors_wall_movement_and_allows_crossing_at_opening() -> None:
    env = make_env(TaskSpec("A", 0, 1), domain="detour")
    env.reset(seed=18)
    # Outside A0's local band, action 0 maps to physical right.
    assert mapping_for_state(C1Context("A", 0), 3)[0] == 1

    env._position = (3, 0)
    env._goal = (8, 0)
    before_distance = env._distance()
    obs, reward, terminated, truncated, info = env.step(0)
    assert _cell(obs) == (3, 0)
    assert reward == pytest.approx(-0.03)
    assert not terminated and not truncated
    assert info["distance"] == before_distance

    env._position = (3, 4)
    env._goal = (8, 4)
    obs, _, _, _, _ = env.step(0)
    assert _cell(obs) == (4, 4)


@pytest.mark.parametrize("domain", DOMAINS)
def test_environments_are_seeded_gym_envs_with_goal_metadata_separate_from_reward(domain: str) -> None:
    first = make_env(TaskSpec("A", 0, -1), domain=domain)
    second = make_env(TaskSpec("A", 0, -1), domain=domain)
    assert isinstance(first, gym.Env)
    assert isinstance(first, C1NavigationEnv)
    assert isinstance(first, WriteDecisionNavigationEnv)
    assert first.grid_size == 9
    assert first.max_steps == 64
    assert first.observation_space.shape == (4,)
    assert first.action_space.n == 4

    first_obs, first_info = first.reset(seed=729)
    second_obs, second_info = second.reset(seed=729)
    np.testing.assert_array_equal(first_obs, second_obs)
    assert first_info == second_info
    assert first.observation_space.contains(first_obs)

    # Force a one-step goal so negative preference produces negative reward
    # while goal metadata and Gym termination still report the goal correctly.
    first._position = (0, 4)
    first._goal = (1, 4)
    first._episode_length = 0
    first.set_task(TaskSpec("A", 0, -1))
    _, reward, terminated, truncated, info = first.step(0)
    assert reward < 0.0
    assert terminated is True
    assert truncated is False
    assert info["goal_reached"] is True

    check_env(second, skip_render_check=True)


def test_navigation_domain_keeps_original_c1_reward_formula() -> None:
    env = make_env(TaskSpec("A", 0, 1), domain="navigation")
    assert isinstance(env, WriteDecisionNavigationEnv)
    assert not isinstance(env, DetourWriteDecisionEnv)
    env.reset(seed=7)
    env._position = (0, 0)
    env._goal = (1, 0)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == pytest.approx(0.25 * (1 - 0) - 0.03 + 1.0)
    assert terminated and not truncated
    assert info["goal_reached"] is True


def test_factory_rejects_unknown_domain_and_string_task_names() -> None:
    with pytest.raises(ValueError):
        make_env(TaskSpec("A", 0, 1), domain="other")
    with pytest.raises(TypeError):
        make_env("A0+")  # type: ignore[arg-type]

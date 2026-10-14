import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.c1_envs import (  # noqa: E402
    C1Context,
    C1NavigationEnv,
    C1_CONTEXTS,
    c1_contexts,
    context_from_index,
    context_index,
    context_name,
    mapping_for_state,
)
from online_rl.envs import NavigationEnv  # noqa: E402


def _cell(obs: np.ndarray) -> tuple[int, int]:
    return tuple(int(round(float(value) * 8)) for value in obs[:2])  # type: ignore[return-value]


def _adjacency(context: C1Context, reverse: bool = False) -> dict[tuple[int, int], set[tuple[int, int]]]:
    deltas = NavigationEnv._DELTAS
    graph = {(x, y): set() for x in range(9) for y in range(9)}
    for x, y in graph:
        for direction in mapping_for_state(context, x):
            dx, dy = deltas[direction]
            destination = (
                min(max(x + dx, 0), 8),
                min(max(y + dy, 0), 8),
            )
            if reverse:
                graph[destination].add((x, y))
            else:
                graph[(x, y)].add(destination)
    return graph


def _reachable(graph: dict[tuple[int, int], set[tuple[int, int]]], start: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {start}
    pending = deque([start])
    while pending:
        state = pending.popleft()
        for neighbor in graph[state] - seen:
            seen.add(neighbor)
            pending.append(neighbor)
    return seen


def _seed_for_start_side(side: int) -> int:
    probe = C1NavigationEnv()
    for seed in range(100):
        obs, _ = probe.reset(seed=seed)
        if _cell(obs)[0] == (0 if side == 0 else 8):
            return seed
    raise AssertionError(f"could not find seed for start side {side}")


def _shortest_mapped_rollout(
    env: C1NavigationEnv, spec: C1Context, obs: np.ndarray
) -> list[tuple[int, int]]:
    visited = [_cell(obs)]
    for _ in range(env.max_steps):
        x, y = _cell(obs)
        goal_x = int(round(float(obs[2]) * 8))
        goal_y = int(round(float(obs[3]) * 8))
        if x != goal_x:
            physical_direction = 1 if goal_x > x else 3
        elif y != goal_y:
            physical_direction = 2 if goal_y > y else 0
        else:
            break

        action = mapping_for_state(spec, x).index(physical_direction)
        previous_distance = abs(goal_x - x) + abs(goal_y - y)
        obs, reward, terminated, truncated, info = env.step(action)
        expected_reward = 0.25 * (previous_distance - info["distance"]) - 0.03
        if terminated:
            expected_reward += 1.0
        assert reward == pytest.approx(expected_reward)
        visited.append(_cell(obs))
        if terminated:
            assert info["success"] is True
            assert truncated is False
            return visited
        assert truncated is False
    raise AssertionError("mapped shortest-path rollout did not reach the goal")


def test_c1_env_is_a_gymnasium_navigation_env_and_passes_checker() -> None:
    env = C1NavigationEnv()
    assert isinstance(env, NavigationEnv)
    assert isinstance(env, gym.Env)
    assert env.grid_size == 9
    assert env.max_steps == 64
    assert env.observation_space.shape == (4,)
    assert env.action_space.n == 4
    check_env(env, skip_render_check=True)


def test_context_registry_has_six_stable_keys_and_helpers() -> None:
    registry = c1_contexts()
    assert tuple(registry) == ("A0", "A1", "A2", "B0", "B1", "B2")
    assert tuple(C1_CONTEXTS) == tuple(registry)
    for index, key in enumerate(registry):
        spec = registry[key]
        assert context_name(spec) == key
        assert context_name(index) == key
        assert context_index(spec) == index
        assert context_index(key) == index
        assert context_from_index(index) == spec

    with pytest.raises(ValueError):
        C1Context("C", 0)
    with pytest.raises(ValueError):
        C1Context("A", 3)
    with pytest.raises(ValueError):
        C1Context("A", True)


def test_context_labels_never_enter_observations_or_info() -> None:
    same_seed = 314
    runs = []
    for spec in C1_CONTEXTS.values():
        env = C1NavigationEnv(spec)
        obs, info = env.reset(seed=same_seed)
        assert env.observation_space.contains(obs)
        assert set(info) == {"success", "episode_length", "distance"}
        assert info == {"success": False, "episode_length": 0, "distance": info["distance"]}
        runs.append((obs, info, env.step(0)[4]))

    reference_obs, reference_info, _ = runs[0]
    for obs, info, step_info in runs[1:]:
        np.testing.assert_array_equal(obs, reference_obs)
        assert info == reference_info
        assert set(step_info) == {"success", "episode_length", "distance"}
        assert all("family" not in key and "variant" not in key and "context" not in key for key in step_info)


def test_family_swap_then_local_swap_produces_expected_mapping() -> None:
    for variant in range(3):
        local_x = variant * 3
        outside_x = (local_x + 3) % 9
        assert mapping_for_state(C1Context("A", variant), local_x) == (1, 0, 3, 2)
        assert mapping_for_state(C1Context("A", variant), outside_x) == (1, 0, 2, 3)
        assert mapping_for_state(C1Context("B", variant), local_x) == (3, 2, 1, 0)
        assert mapping_for_state(C1Context("B", variant), outside_x) == (3, 1, 2, 0)

    for spec in C1_CONTEXTS.values():
        for x in range(9):
            mapping = mapping_for_state(spec, x)
            assert sorted(mapping) == [0, 1, 2, 3]

    with pytest.raises(ValueError):
        mapping_for_state(C1Context("A", 0), -1)


def test_step_uses_mapping_at_the_current_state_before_movement() -> None:
    env = C1NavigationEnv(C1Context("B", 1))
    env.reset(seed=18)

    # At x = 2, action 1 is physical right and enters variant 1's band. If the
    # environment used the destination x, it would instead move down.
    env._position = (2, 4)
    env._goal = (8, 4)
    _, _, _, _, _ = env.step(1)
    assert env._position == (3, 4)

    # At x = 3, the local right/down swap applies to the same action.
    env._position = (3, 4)
    env._goal = (8, 4)
    _, _, _, _, _ = env.step(1)
    assert env._position == (3, 5)


def test_each_context_grid_has_strongly_connected_directed_moves() -> None:
    cells = {(x, y) for x in range(9) for y in range(9)}
    for spec in C1_CONTEXTS.values():
        forward = _adjacency(spec)
        reverse = _adjacency(spec, reverse=True)
        assert _reachable(forward, (0, 0)) == cells
        assert _reachable(reverse, (0, 0)) == cells


def test_reset_is_seeded_and_always_uses_opposite_edges() -> None:
    first = C1NavigationEnv(C1Context("B", 2))
    second = C1NavigationEnv(C1Context("B", 2))
    obs_a, info_a = first.reset(seed=729)
    obs_b, info_b = second.reset(seed=729)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a == info_b

    for seed in range(20):
        obs, _ = first.reset(seed=seed)
        x, _ = _cell(obs)
        goal_x = int(round(float(obs[2]) * 8))
        assert x in (0, 8)
        assert goal_x == 8 - x


def test_success_requires_visiting_every_x_band_for_each_context_and_side() -> None:
    side_seeds = (_seed_for_start_side(0), _seed_for_start_side(8))
    expected_bands = {0, 1, 2}
    for spec in C1_CONTEXTS.values():
        for seed in side_seeds:
            env = C1NavigationEnv(spec)
            obs, _ = env.reset(seed=seed)
            visited = _shortest_mapped_rollout(env, spec, obs)
            assert {x // 3 for x, _ in visited} == expected_bands
            counters = env.observer_counters()
            assert set(counters) == {"band_0_visits", "band_1_visits", "band_2_visits"}
            assert all(counters[f"band_{band}_visits"] > 0 for band in range(3))
            total_before = sum(counters.values())
            env.reset(seed=seed + 1000)
            assert sum(env.observer_counters().values()) == total_before + 1

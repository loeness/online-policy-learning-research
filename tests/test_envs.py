import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.envs import (  # noqa: E402
    ContextSpec,
    NavigationEnv,
    contexts,
    expert_action,
    expert_distribution,
)


def _seed_with_interior_start() -> int:
    env = NavigationEnv()
    for seed in range(500):
        obs, _ = env.reset(seed=seed)
        if 0.0 < obs[0] < 1.0 and 0.0 < obs[1] < 1.0:
            return seed
    raise AssertionError("could not sample an interior start cell")


def test_context_rotation_maps_source_action_indices_clockwise() -> None:
    seed = _seed_with_interior_start()
    expected_delta = {
        "normal": (0, -1),
        "cw": (1, 0),
        "ccw": (-1, 0),
    }

    for name, delta in expected_delta.items():
        env = NavigationEnv()
        env.set_context(contexts()[name])
        obs, _ = env.reset(seed=seed)
        next_obs, _, _, _, _ = env.step(0)
        moved = (next_obs[:2] - obs[:2]) * (env.grid_size - 1)
        np.testing.assert_allclose(moved, delta)


def test_agent_visible_interface_does_not_include_context_metadata() -> None:
    same_seed = 41
    source = NavigationEnv(context=contexts()["normal"])
    shifted = NavigationEnv(context=contexts()["cw"])

    source_obs, source_info = source.reset(seed=same_seed)
    shifted_obs, shifted_info = shifted.reset(seed=same_seed)
    np.testing.assert_array_equal(source_obs, shifted_obs)
    assert source_info == shifted_info
    assert set(source_info) == {"success", "episode_length", "distance"}
    assert source_info["success"] is False
    assert source_info["episode_length"] == 0
    assert source_info["distance"] > 0

    _, _, _, _, info = shifted.step(0)
    assert set(info) == {"success", "episode_length", "distance"}
    assert all("context" not in key.lower() for key in info)


def test_seed_reproduces_slip_transitions() -> None:
    spec = contexts()["cw_slip"]
    first = NavigationEnv(context=spec)
    second = NavigationEnv(context=spec)
    obs_a, info_a = first.reset(seed=912)
    obs_b, info_b = second.reset(seed=912)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a == info_b

    for action in (0, 1, 1, 2, 3, 0, 2, 2, 1, 3):
        result_a = first.step(action)
        result_b = second.step(action)
        np.testing.assert_array_equal(result_a[0], result_b[0])
        assert result_a[1:] == result_b[1:]
        if result_a[2] or result_a[3]:
            break


def test_source_expert_reaches_goal_and_step_limit_truncates() -> None:
    env = NavigationEnv(max_steps=64)
    obs, _ = env.reset(seed=21)
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(expert_action(obs))
        if terminated or truncated:
            break
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["distance"] == 0

    capped = NavigationEnv(max_steps=1)
    for seed in range(500):
        obs, info = capped.reset(seed=seed)
        if info["distance"] > 1:
            break
    else:
        raise AssertionError("could not sample a goal more than one step away")
    _, _, terminated, truncated, info = capped.step(expert_action(obs))
    assert terminated is False
    assert truncated is True
    assert info["success"] is False
    assert info["episode_length"] == 1


def test_expert_distribution_is_uniform_over_shortest_path_actions() -> None:
    obs = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    distribution = expert_distribution(obs)
    np.testing.assert_array_equal(distribution, [0.0, 0.5, 0.5, 0.0])
    assert expert_action(obs) == 2

    at_goal = expert_distribution([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(at_goal, np.zeros(4, dtype=np.float32))


def test_context_spec_rejects_invalid_rotation_and_slip() -> None:
    with pytest.raises(ValueError):
        ContextSpec("half-turn", 2)
    with pytest.raises(ValueError):
        ContextSpec("fractional-turn", 0.5)
    with pytest.raises(ValueError):
        ContextSpec("invalid-slip", 0, 1.1)

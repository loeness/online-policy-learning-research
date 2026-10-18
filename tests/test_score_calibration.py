import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.score_calibration import (  # noqa: E402
    grid_observations,
    paired_rollouts,
    probability_index,
    restore_navigation,
    roundtrip_common,
    snapshot_navigation,
)
from online_rl.write_decision_envs import TaskSpec, make_env  # noqa: E402
from online_rl.write_decision_policy import decode_actor, encode_actor  # noqa: E402


def test_navigation_snapshot_restores_observation_rng_time_and_counters_without_stepping() -> None:
    env = make_env(TaskSpec("B", 2, -1), domain="navigation")
    env.reset(seed=20261016)
    env._position = (2, 5)
    env._goal = (6, 1)
    env._episode_length = 11
    env._observer_region_visits = [7, 13, 4]
    # Advance only the local RNG to check that restoration keeps its exact state.
    env.np_random.random()

    snapshot = snapshot_navigation(env)
    json.dumps(snapshot)
    restored = restore_navigation(snapshot)
    try:
        assert snapshot_navigation(restored) == snapshot
        np.testing.assert_array_equal(restored._observation(), env._observation())
        assert restored._episode_length == 11
        assert restored._observer_region_visits == [7, 13, 4]
        assert restored._has_reset is True
        assert restored.context.family == "B"
        assert restored.context.variant == 2
        assert restored._preference == -1
        assert restored.np_random.bit_generator.state == env.np_random.bit_generator.state
    finally:
        env.close()
        restored.close()


def test_grid_observation_order_matches_probability_index_for_every_row() -> None:
    observations = grid_observations()
    assert observations.shape == (6561, 13)
    assert observations.dtype == np.float32
    np.testing.assert_array_equal(observations[:, 4:12], np.zeros((6561, 8), dtype=np.float32))
    np.testing.assert_array_equal(observations[:, 12], np.ones(6561, dtype=np.float32))
    assert [probability_index(obs[:4]) for obs in observations] == list(range(6561))

    def expected_index(x: int, y: int, goal_x: int, goal_y: int) -> int:
        return (((x * 9 + y) * 9 + goal_x) * 9) + goal_y

    for coordinates in ((0, 0, 0, 0), (2, 5, 6, 1), (8, 8, 8, 8)):
        row = expected_index(*coordinates)
        np.testing.assert_array_equal(observations[row, :4], np.asarray(coordinates) / 8)
        assert probability_index(observations[row, :4]) == row

    with pytest.raises(ValueError, match="four"):
        probability_index([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="grid"):
        probability_index([0.1, 0.0, 0.0, 0.0])


class _FakeNavigationEnv:
    """Tiny deterministic rollout double; it never wraps a real env.step."""

    def __init__(self, snapshot: dict[str, list[float]], *, truncate_after: int = 2) -> None:
        self._obs = np.asarray(snapshot["observation"], dtype=np.float32)
        self.truncate_after = truncate_after
        self._episode_length = 0
        self.np_random = np.random.default_rng(917)
        self.actions: list[int] = []
        self.reset_seeds: list[int] = []
        self.step_calls = 0
        self.closed = False

    def _observation(self) -> np.ndarray:
        return self._obs.copy()

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, bool]]:
        assert seed is not None
        self.reset_seeds.append(seed)
        self.np_random = np.random.default_rng(seed)
        self._episode_length = 0
        return self._observation(), {"goal_reached": False}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, bool]]:
        self.actions.append(action)
        self.step_calls += 1
        self._episode_length += 1
        truncated = self._episode_length >= self.truncate_after
        return self._observation(), action / 10.0, False, truncated, {"goal_reached": False}

    def close(self) -> None:
        self.closed = True


def test_paired_rollouts_share_actions_and_reset_seeds_and_use_exact_step_budget() -> None:
    created: list[_FakeNavigationEnv] = []

    def factory(snapshot):
        env = _FakeNavigationEnv(snapshot)
        created.append(env)
        return env

    snapshot = {"observation": [0.0, 0.0, 1.0, 1.0]}
    same_policy = np.full((6561, 4), 0.25, dtype=np.float64)
    result = paired_rollouts(
        snapshot,
        same_policy,
        same_policy.copy(),
        horizon=5,
        replicates=3,
        seed=48017,
        env_factory=factory,
    )

    assert result["step_calls"] == 2 * 5 * 3
    assert len(created) == 6
    assert all(env.step_calls == 5 for env in created)
    assert all(env.closed for env in created)
    for index in range(0, len(created), 2):
        candidate, baseline = created[index:index + 2]
        assert candidate.actions == baseline.actions
        assert candidate.reset_seeds == baseline.reset_seeds
        assert len(candidate.reset_seeds) == len(baseline.reset_seeds) == 2

    for replicate in result["per_replicate"]:
        assert replicate["candidate"]["raw_steps"] == 5
        assert replicate["baseline"]["raw_steps"] == 5
        assert replicate["candidate"]["post_termination_reset_count"] == 2
        assert replicate["baseline"]["post_termination_reset_count"] == 2
        assert replicate["candidate"]["post_termination_reset_seeds"] == replicate["baseline"]["post_termination_reset_seeds"]
        assert replicate["audit"]["first_observations_equal"] is True
        assert replicate["audit"]["initial_rng_state_equal"] is True
        assert replicate["audit"]["paired_action_uniform_per_raw_step"] is True
        assert replicate["audit"]["corresponding_reset_seeds_equal"] is True
        assert replicate["differences"]["undiscounted_return"] == 0.0
        assert replicate["differences"]["discounted_first64_return"] == 0.0
        assert replicate["differences"]["goals"] == 0
    assert "p_value" not in result


def test_reset_seed_audit_compares_shared_ordinal_prefix_with_unequal_resets() -> None:
    created: list[_FakeNavigationEnv] = []

    def factory(snapshot):
        # Candidate truncates each step; baseline truncates every third step.
        truncate_after = 1 if len(created) == 0 else 3
        env = _FakeNavigationEnv(snapshot, truncate_after=truncate_after)
        created.append(env)
        return env

    candidate = np.zeros((6561, 4), dtype=np.float64)
    candidate[:, 0] = 1.0
    baseline = np.zeros((6561, 4), dtype=np.float64)
    baseline[:, 1] = 1.0
    result = paired_rollouts(
        {"observation": [0.0, 0.0, 1.0, 1.0]},
        candidate,
        baseline,
        horizon=5,
        replicates=1,
        seed=48017,
        env_factory=factory,
    )

    assert result["step_calls"] == 2 * 5
    assert created[0].reset_seeds[:1] == created[1].reset_seeds
    assert len(created[0].reset_seeds) == 4
    assert len(created[1].reset_seeds) == 1
    audit = result["per_replicate"][0]["audit"]
    assert audit["first_observations_equal"] is True
    assert audit["initial_rng_state_equal"] is True
    assert audit["paired_action_uniform_per_raw_step"] is True
    assert audit["corresponding_reset_seeds_equal"] is True
    assert audit["full_reset_sequences_equal"] is False


def test_common_matrix_roundtrip_uses_shared_int8_row_and_reports_error() -> None:
    matrix = np.asarray(
        [[0.2, -0.7, 0.05], [1.1, 0.0, -0.3], [-0.15, 0.45, 0.9]],
        dtype=np.float32,
    )
    roundtripped, diagnostics = roundtrip_common(matrix)

    expected_payload = encode_actor(
        np.zeros((1, 27), dtype=np.float32),
        np.stack((matrix.reshape(9), np.zeros(9, dtype=np.float32))),
        np.asarray([0], dtype=np.int8),
        codec="int8",
    )
    _, decoded_shared, groups = decode_actor(expected_payload.data)
    expected = decoded_shared[0].reshape(3, 3)
    assert roundtripped.dtype == np.float32
    np.testing.assert_array_equal(roundtripped, expected)
    np.testing.assert_array_equal(groups, [0])
    assert diagnostics["max_abs_error"] == pytest.approx(np.max(np.abs(roundtripped - matrix)))
    assert diagnostics["rmse"] == pytest.approx(np.sqrt(np.mean(np.square(roundtripped - matrix))))

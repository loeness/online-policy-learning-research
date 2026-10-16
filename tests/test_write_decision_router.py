import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.write_decision_envs import TaskSpec, make_env  # noqa: E402
from online_rl.write_decision_evidence import RewardAwareGroups  # noqa: E402
from online_rl.write_decision_router import WriteDecisionRouter  # noqa: E402


DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _obs(x: int, y: int = 4, goal_x: int = 8, goal_y: int = 4) -> np.ndarray:
    return np.asarray([x / 8, y / 8, goal_x / 8, goal_y / 8], dtype=np.float32)


def _next(obs: np.ndarray, direction: int) -> np.ndarray:
    x, y = int(round(float(obs[0]) * 8)), int(round(float(obs[1]) * 8))
    dx, dy = DELTAS[direction]
    result = obs.copy()
    result[0] = min(max(x + dx, 0), 8) / 8
    result[1] = min(max(y + dy, 0), 8) / 8
    return result


def _feed(router: WriteDecisionRouter, obs: np.ndarray, action: int, direction: int, reward: float = 0.0):
    return router.observe(obs, action, reward, _next(obs, direction), False)


def _teach_reward_model(router: WriteDecisionRouter, sign: float, cycles: int = 10) -> None:
    """Provide supported moving and blocked examples for a 2-feature reward fit."""
    for _ in range(cycles):
        _feed(router, _obs(1), 0, 1, 0.2 * sign)
        _feed(router, _obs(3), 0, 1, 0.2 * sign)
        blocked = _obs(0)
        router.observe(blocked, 1, 0.0, blocked, False)


def _feed_real_episode(env, router: WriteDecisionRouter, seed: int) -> list[tuple[np.ndarray, int, float, np.ndarray, bool]]:
    obs, _ = env.reset(seed=seed)
    transitions = []
    # For C1 A0, action 0 advances right in each band. This provides repeated
    # visible support without giving the router the context or the seed.
    for _ in range(8):
        before = obs.copy()
        obs, reward, terminated, truncated, _ = env.step(0)
        assert not truncated
        router.observe(before, 0, reward, obs, terminated)
        transitions.append((before, 0, reward, obs.copy(), terminated))
        if terminated:
            break
    return transitions


def _seed_starting_left(env) -> int:
    for seed in range(100):
        obs, _ = env.reset(seed=seed)
        if int(round(float(obs[0]) * 8)) == 0:
            return seed
    raise AssertionError("could not find a seed with a left-edge start")


def _teach_pattern(router: WriteDecisionRouter, code: tuple[int, int], previous: tuple[int, int] | None) -> None:
    """Teach a two-cell deterministic profile using only visible transitions."""
    cells = (_obs(1), _obs(3))
    if previous is None:
        schedule = [0] * 4 + [1] * 4
    else:
        first = next(index for index in range(2) if code[index] != previous[index])
        other = 1 - first
        schedule = [first] * 4 + [other] * 4
    for cell_index in schedule:
        obs = cells[cell_index]
        _feed(router, obs, cell_index, code[cell_index])


def _teach_group_compatible_pattern(router: WriteDecisionRouter) -> None:
    cells = ((_obs(1), 0), (_obs(1), 1), (_obs(3), 0), (_obs(6), 0))
    for _ in range(2):
        for obs, action in cells:
            _feed(router, obs, action, 1, 0.0)


def _feed_candidate_after_conflict(
    router: WriteDecisionRouter,
    code: tuple[int, int],
    previous: tuple[int, int],
) -> dict[str, int | float | str | bool | None]:
    cells = (_obs(1), _obs(3))
    first = next(index for index in range(2) if code[index] != previous[index])
    other = 1 - first
    results = []
    for cell_index in [first] * 4 + [other] * 4:
        obs = cells[cell_index]
        results.append(_feed(router, obs, cell_index, code[cell_index]))
    return results[-1]


def test_route_stays_unknown_until_supported_window_is_ready() -> None:
    router = WriteDecisionRouter(window=32)
    assert router.route(_obs(1))["slot"] == -1
    for _ in range(3):
        _feed(router, _obs(1), 1, 1, 0.22)
    assert router.route(_obs(1))["slot"] == -1
    for _ in range(5):
        _feed(router, _obs(3), 1, 1, 0.22)
    assert router.route(_obs(3))["slot"] == 0
    assert router.active_slots == (0,)


def test_unsupported_action_response_is_learned_without_false_conflict() -> None:
    router = WriteDecisionRouter(window=32)
    for _ in range(8):
        _feed(router, _obs(1), 0, 1, 0.2)
        _feed(router, _obs(3), 0, 1, 0.2)
    assert router.active_slot == 0

    # Action 1 has no support in this cell, so a rightward result is new
    # evidence for slot 0 rather than a contradiction or a new slot.
    result = _feed(router, _obs(1), 1, 1, 0.2)
    assert result["active_slot"] == 0
    assert result["allocated_slot"] is None
    assert result["switched"] is False
    assert result["reason"] == "active_profile"


def test_profile_for_slot_is_numeric_copy_accepted_by_shared_groups() -> None:
    router = WriteDecisionRouter(window=32)
    _teach_group_compatible_pattern(router)
    profile = router.profile_for_slot(0)

    assert set(profile) == {"movement_counts", "reward_xtx", "reward_xty", "reward_n"}
    assert profile["movement_counts"].dtype == np.uint16
    assert profile["reward_xtx"].dtype == np.float32
    assert profile["reward_xty"].dtype == np.float32
    assert isinstance(profile["reward_n"], np.float32)

    groups = RewardAwareGroups(min_support=8, reward_min_samples=8)
    assignment = groups.assign(profile)
    assert assignment.group == 0
    assert groups.active_groups == (0,)

    profile["movement_counts"].fill(0)
    profile["reward_xtx"].fill(0.0)
    profile["reward_xty"].fill(0.0)
    assert int(router.profile_for_slot(0)["movement_counts"].sum()) == 8


def test_clear_recent_drops_route_and_history_but_preserves_profiles() -> None:
    router = WriteDecisionRouter(window=32)
    _teach_pattern(router, (1, 2), None)
    assert router.active_slot == 0
    assert router.history_size == 0

    # A fresh event starts a bounded diagnostic history while the known route
    # continues to collect its persistent profile.
    _feed(router, _obs(1), 0, 1, 0.0)
    assert router.active_slot == 0
    assert router.history_size == 1
    before = router.profile_for_slot(0)

    router.clear_recent()
    assert router.active_slot == -1
    assert router.history_size == 0
    assert router.route(_obs(1))["slot"] == -1
    assert router.active_slots == (0,)

    after = router.profile_for_slot(0)
    for name in ("movement_counts", "reward_xtx", "reward_xty"):
        np.testing.assert_array_equal(after[name], before[name])
    assert after["reward_n"] == before["reward_n"]


def test_real_navigation_stream_detects_reward_flip_and_reuses_original_profile() -> None:
    env = make_env(TaskSpec("A", 0, 1), domain="navigation")
    router = WriteDecisionRouter(window=64)
    seed = _seed_starting_left(env)

    _feed_real_episode(env, router, seed)
    _feed_real_episode(env, router, seed)
    assert router.route(env.reset(seed=seed)[0])["slot"] == 0

    env.set_task(TaskSpec("A", 0, -1))
    flipped = _feed_real_episode(env, router, seed)
    assert flipped
    assert router.active_slot == 1
    assert router.route(flipped[-1][3])["slot"] == 1
    assert router.active_slots == (0, 1)

    # The opposite slot needs 16 reward observations before its single-step
    # residual detector is allowed to declare another reward boundary.
    _feed_real_episode(env, router, seed)
    env.set_task(TaskSpec("A", 0, 1))
    reused = _feed_real_episode(env, router, seed)
    assert reused
    assert router.active_slot == 0
    assert router.active_slots == (0, 1)
    assert router._allocations == 2


def test_support_aware_first_terminal_reward_is_learned_without_context_switch() -> None:
    router = WriteDecisionRouter(window=64, reward_support_aware=True)
    env = make_env(TaskSpec("A", 0, 1), domain="navigation")
    seed = _seed_starting_left(env)
    obs, _ = env.reset(seed=seed)

    # Keep the real agent stream away from the goal while identifying both
    # movement and intercept reward features.
    before = obs.copy()
    obs, reward, terminated, truncated, _ = env.step(0)  # x = 0 -> 1
    assert not terminated and not truncated
    router.observe(before, 0, reward, obs, terminated)
    for _ in range(10):
        for action in (0, 2):  # x = 1 <-> 2 under the visible C1 mapping
            before = obs.copy()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
            router.observe(before, action, reward, obs, terminated)

    assert router.active_slot == 0
    assert router._reward_n[0] >= 16
    assert router._reward_xtx[0, 2, 2] == 0.0
    before_slots = router.active_slots

    # Continue along the row to x = 7, align y, then let the real environment
    # produce the first goal termination after the model has stabilized.
    while int(round(float(obs[0]) * 8)) < 7:
        before = obs.copy()
        obs, reward, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
        router.observe(before, 0, reward, obs, terminated)
    goal_y = int(round(float(obs[3]) * 8))
    while int(round(float(obs[1]) * 8)) != goal_y:
        y = int(round(float(obs[1]) * 8))
        action = 1 if y > goal_y else 2
        before = obs.copy()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
        router.observe(before, action, reward, obs, terminated)
    before = obs.copy()
    obs, reward, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated
    terminal = router.observe(before, 0, reward, obs, terminated)

    assert terminal["reason"] == "active_profile"
    assert terminal["active_slot"] == 0
    assert terminal["allocated_slot"] is None
    assert router.active_slots == before_slots == (0,)
    assert router._reward_xtx[0, 2, 2] == 1.0


def test_support_aware_router_detects_reversal_in_identifiable_reward_subspace() -> None:
    router = WriteDecisionRouter(window=64, reward_support_aware=True)
    _teach_reward_model(router, sign=1.0)
    assert router.active_slot == 0

    first_reversed = _feed(router, _obs(1), 0, 1, -0.2)
    assert first_reversed["reason"] == "reward_residual"
    assert first_reversed["active_slot"] == -1

    allocation = None
    # The conflict event starts the new numeric window; continue the same
    # visible reward/movement stream until it has enough evidence to compare.
    for _ in range(4):
        for obs in (_obs(1), _obs(3)):
            allocation = _feed(router, obs, 0, 1, -0.2)
            if allocation["allocated_slot"] is not None:
                break
        if allocation is not None and allocation["allocated_slot"] is not None:
            break
        blocked = _obs(0)
        allocation = router.observe(blocked, 1, 0.0, blocked, False)
        if allocation["allocated_slot"] is not None:
            break

    assert allocation is not None
    assert allocation["reason"] == "opposite_reward_profile"
    assert allocation["allocated_slot"] == 1
    assert router.active_slot == 1
    assert router.active_slots == (0, 1)


def test_support_aware_router_abstains_without_movement_support() -> None:
    router = WriteDecisionRouter(window=32, reward_support_aware=True)
    blocked = _obs(0)
    for _ in range(24):
        result = router.observe(blocked, 1, 0.0, blocked, False)

    assert result["reason"] == "insufficient_support"
    assert result["active_slot"] == -1
    assert router.active_slots == ()


def test_reward_support_aware_config_round_trips_and_old_state_defaults_false() -> None:
    aware = WriteDecisionRouter(window=32, reward_support_aware=True)
    restored = WriteDecisionRouter.from_state_dict(aware.state_dict())
    assert restored.reward_support_aware is True
    assert restored.nbytes == aware.nbytes

    legacy_state = aware.state_dict()
    del legacy_state["config"]["reward_support_aware"]
    legacy_restored = WriteDecisionRouter.from_state_dict(legacy_state)
    assert legacy_restored.reward_support_aware is False
    assert legacy_restored.nbytes == aware.nbytes

    assert WriteDecisionRouter().reward_support_aware is False


@pytest.mark.parametrize(
    ("eviction", "expected_victim"),
    (("lru", 0), ("lfu", 1), ("random", None)),
)
def test_slot_overflow_evicts_only_bounded_slot_metadata_and_restores_rng(
    eviction: str, expected_victim: int | None
) -> None:
    router = WriteDecisionRouter(max_slots=6, window=16, eviction=eviction, seed=912)
    patterns = [(index // 4, index % 4) for index in range(7)]
    _teach_pattern(router, patterns[0], None)
    if eviction == "lru" or eviction == "lfu":
        # Slot 0 is the most frequent slot, but remains the least recent.
        if eviction == "lfu":
            for _ in range(8):
                _feed(router, _obs(1), 0, patterns[0][0])
                _feed(router, _obs(3), 1, patterns[0][1])
    for index in range(1, 6):
        _feed_candidate_after_conflict(router, patterns[index], patterns[index - 1])
    assert router.active_slots == (0, 1, 2, 3, 4, 5)
    old_slot_zero = router._counts[0].copy()

    restored = WriteDecisionRouter.from_state_dict(router.state_dict())
    result_a = _feed_candidate_after_conflict(router, patterns[6], patterns[5])
    result_b = _feed_candidate_after_conflict(restored, patterns[6], patterns[5])
    assert result_a == result_b
    assert result_a["allocated_slot"] is not None
    assert result_a["evicted_slot"] is not None
    if expected_victim is not None:
        assert result_a["evicted_slot"] == expected_victim
    assert router.active_slots == tuple(range(6))
    assert router._evictions == restored._evictions == 1
    assert not np.array_equal(router._counts[0], old_slot_zero) or result_a["evicted_slot"] != 0
    assert router._last_used.shape == router._frequency.shape == (router.max_slots,)


def test_snapshot_restores_partial_unknown_window_and_numeric_storage() -> None:
    env = make_env(TaskSpec("A", 0, 1), domain="navigation")
    router = WriteDecisionRouter(window=64)
    seed = _seed_starting_left(env)
    _feed_real_episode(env, router, seed)
    _feed_real_episode(env, router, seed)
    env.set_task(TaskSpec("A", 0, -1))
    obs, _ = env.reset(seed=seed)
    for _ in range(3):
        before = obs.copy()
        obs, reward, terminated, truncated, _ = env.step(0)
        assert not truncated
        router.observe(before, 0, reward, obs, terminated)
    assert router.active_slot == -1
    assert router.history_size == 3

    snapshot = router.state_dict()
    restored = WriteDecisionRouter.from_state_dict(snapshot)
    assert restored.history_size == router.history_size
    assert restored.route(obs) == router.route(obs)
    assert restored.nbytes == router.nbytes == snapshot["nbytes"]
    assert snapshot["nbytes"] == sum(value.nbytes for name, value in snapshot.items() if isinstance(value, np.ndarray))
    assert all("task" not in key and "label" not in key and "context" not in key for key in snapshot)

    for _ in range(5):
        before = obs.copy()
        obs, reward, terminated, truncated, _ = env.step(0)
        assert not truncated
        result_a = router.observe(before, 0, reward, obs, terminated)
        result_b = restored.observe(before, 0, reward, obs, terminated)
        assert result_a == result_b
    assert restored.state_dict()["active_slot"] == router.state_dict()["active_slot"] == 1


def test_invalid_visible_events_and_unbounded_slot_count_are_rejected() -> None:
    with pytest.raises(ValueError, match="6 or 7"):
        WriteDecisionRouter(max_slots=8)
    with pytest.raises(ValueError, match="window"):
        WriteDecisionRouter(window=0)
    router = WriteDecisionRouter()
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        router.route(np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError, match="one cardinal"):
        router.observe(_obs(1), 0, 0.0, _obs(4), False)
    with pytest.raises(ValueError, match="terminated"):
        router.observe(_obs(1), 0, 0.0, _next(_obs(1), 1), 0)

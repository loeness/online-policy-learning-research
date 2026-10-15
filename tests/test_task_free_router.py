"""Causal discovery, censoring, persistence, and storage checks for C1 routing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import online_rl.c1_envs as c1_envs  # noqa: E402
from online_rl.task_free_router import TaskFreeRouter  # noqa: E402


DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _obs(x: int, y: int = 4) -> np.ndarray:
    return np.asarray([x / 8, y / 8, 1.0, 1.0], dtype=np.float32)


def _next(obs: np.ndarray, direction: int) -> np.ndarray:
    x, y = int(round(float(obs[0]) * 8)), int(round(float(obs[1]) * 8))
    dx, dy = DELTAS[direction]
    return _obs(min(max(x + dx, 0), 8), min(max(y + dy, 0), 8))


def _teach_seed(router: TaskFreeRouter) -> None:
    for action in range(4):
        obs = _obs(1)
        router.observe(obs, action, 0.0, _next(obs, action))
    assert router.active_slots == (0,)
    for _ in range(2):
        for action in range(4):
            obs = _obs(1)
            router.observe(obs, action, 0.0, _next(obs, action))


def test_router_reads_only_observable_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("router consulted a hidden C1 mapping helper")

    monkeypatch.setattr(c1_envs, "mapping_for_state", forbidden)
    router = TaskFreeRouter()
    _teach_seed(router)
    assert router.route(_obs(1))["slot"] == 0


def test_stationary_wall_response_is_censored_and_does_not_seed_a_slot() -> None:
    router = TaskFreeRouter(max_contexts=2, window=8)
    wall = _obs(0)
    info = router.observe(wall, 3, 0.0, wall.copy())
    assert info["censored"] is True
    assert info["event"] is None
    assert router.active_slots == ()
    assert router.route(wall)["reason"] == "insufficient_support"
    assert router._censored == 1


def test_unsupported_band_action_is_not_conflict_or_novelty_in_robust_mode() -> None:
    router = TaskFreeRouter(max_contexts=3, window=8)
    _teach_seed(router)
    # The profile has not encountered this band/action cell. A rightward
    # response is learned as new support instead of splitting a context.
    obs = _obs(4)
    info = router.observe(obs, 0, 0.0, _next(obs, 1))
    assert info["allocated_slot"] is None
    assert info["reason"] == "insufficient_support"
    assert router.active_slots == (0,)
    assert router.route(_obs(4))["slot"] == 0


def test_conflicting_supported_cells_confirm_a_new_slot() -> None:
    router = TaskFreeRouter(max_contexts=3, window=8, confirm_events=3)
    _teach_seed(router)

    # The original profile predicts up for action 0 and right for action 1.
    contradictions = ((0, 1), (1, 2), (0, 1))
    allocations = []
    for action, direction in contradictions:
        obs = _obs(1)
        info = router.observe(obs, action, 0.0, _next(obs, direction))
        allocations.append(info["allocated_slot"])
    assert allocations[-1] == 1
    assert router.active_slots == (0, 1)
    assert router.route(_obs(1))["slot"] == 1


def test_recent_candidate_scores_include_unsupported_cells_as_uniform() -> None:
    router = TaskFreeRouter(max_contexts=2, window=8)
    router._active = [True, True]
    events = [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
    router._recent.extend(events)
    router._counts[0, 0, 0, 0] = 3  # One supported cell, two unknown cells.
    router._counts[1, 0, 0, 0] = 3
    router._counts[1, 0, 1, 1] = 3
    router._counts[1, 0, 2, 2] = 3
    scores, supports = router._route_scores()
    assert scores[1] > scores[0]
    assert supports.tolist() == [1, 3]


def test_isolated_conflict_recovers_after_same_cell_repeats() -> None:
    router = TaskFreeRouter(window=8)
    _teach_seed(router)
    conflict_obs = _obs(1)
    router.observe(conflict_obs, 0, 0.0, _next(conflict_obs, 1))
    assert router.route(conflict_obs)["slot"] == -1

    # A matching response in the same known cell twice clears provisional
    # conflict evidence without discarding the learned profile.
    router.observe(conflict_obs, 0, 0.0, _next(conflict_obs, 0))
    router.observe(conflict_obs, 0, 0.0, _next(conflict_obs, 0))
    assert router._route_unresolved is False
    obs = _obs(1)
    router.observe(obs, 1, 0.0, _next(obs, 1))
    assert router.route(obs)["slot"] == 0


def test_stale_conflict_expires_after_recent_window() -> None:
    router = TaskFreeRouter(window=3)
    _teach_seed(router)
    known = _obs(1)
    router.observe(known, 0, 0.0, _next(known, 1))
    assert router.route(known)["slot"] == -1

    # Ordinary observations in a previously unsupported cell are not
    # contradictory evidence. Once the conflict leaves the bounded time
    # window, the old profile can resolve again from the recent evidence.
    unknown = _obs(4)
    for _ in range(4):
        router.observe(unknown, 3, 0.0, _next(unknown, 3))
    assert router._route_unresolved is False
    assert router.route(unknown)["slot"] == 0


def test_full_bank_replaces_the_least_recently_used_slot() -> None:
    router = TaskFreeRouter(max_contexts=2, window=8, confirm_events=3)
    _teach_seed(router)
    for action, direction in ((0, 1), (1, 2), (0, 1)):
        obs = _obs(1)
        router.observe(obs, action, 0.0, _next(obs, direction))
    assert router.active_slots == (0, 1)
    # Grow support in the current slot, then present a third profile with
    # supported contradictions. Slot 0 has been least recently used.
    for action, direction in ((0, 1), (1, 2), (0, 1)):
        obs = _obs(1)
        router.observe(obs, action, 0.0, _next(obs, direction))
    for action, direction in ((0, 3), (1, 3), (0, 3)):
        obs = _obs(1)
        info = router.observe(obs, action, 0.0, _next(obs, direction))
    assert info["allocated_slot"] == 0
    assert info["evicted_slot"] == 0
    assert router.evictions == 1
    assert router.active_slots == (0, 1)


def test_reset_history_keeps_profiles_and_round_trip_preserves_predictions() -> None:
    router = TaskFreeRouter(window=8)
    _teach_seed(router)
    before_counts = router._counts.copy()
    state = router.state_dict()
    restored = TaskFreeRouter.from_state_dict(state)
    np.testing.assert_array_equal(restored._counts, router._counts)
    np.testing.assert_array_equal(
        restored.predict_direction(_obs(1), 0), router.predict_direction(_obs(1), 0)
    )

    restored.reset_history()
    np.testing.assert_array_equal(restored._counts, before_counts)
    assert restored.active_slots == (0,)
    assert len(restored._recent) == 0
    assert restored.route(_obs(1))["slot"] == -1

    for action in (0, 1):
        obs = _obs(1)
        restored.observe(obs, action, 0.0, _next(obs, action))
    assert restored.route(_obs(1))["slot"] == 0


def test_profile_counts_are_capped_and_memory_breakdown_is_bounded() -> None:
    router = TaskFreeRouter(window=3)
    _teach_seed(router)
    for _ in range(20):
        obs = _obs(1)
        router.observe(obs, 0, 0.0, _next(obs, 0))
    assert int(router._counts[0, 0, 0].sum()) <= 3
    sizes = router.nbytes()
    assert sizes["total"] == sum(value for key, value in sizes.items() if key != "total")
    assert sizes["profile_counts"] == router._counts.nbytes
    assert sizes["recent_window"] > 0


def test_invalid_transition_and_snapshot_capacity_are_rejected() -> None:
    router = TaskFreeRouter()
    with pytest.raises(ValueError, match="move one cell"):
        router.observe(_obs(2), 0, 0.0, _obs(4))
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        router.route(np.zeros(5, dtype=np.float32))

    state = router.state_dict()
    state["recent"] = [[0, 0, 0]] * (router.window + 1)
    with pytest.raises(ValueError, match="exceeds configured window"):
        TaskFreeRouter.from_state_dict(state)

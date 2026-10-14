"""Checks for the calibrated, bounded C1 history routing diagnostic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import online_rl.c1_envs as c1_envs  # noqa: E402
from online_rl.c1_envs import C1NavigationEnv, C1_CONTEXTS, context_index  # noqa: E402
from online_rl.c1_routing import (  # noqa: E402
    C1HistoryRouter,
    OUTCOME_BLOCKED,
    ROUTE_PARENT_A,
    ROUTE_PARENT_B,
    ROUTE_ROOT,
)


def _transition(spec, x: int, y: int, action: int):
    """Take a real environment step from a selected visible grid cell."""

    env = C1NavigationEnv(spec)
    env.reset(seed=17)
    env._position = (x, y)
    env._goal = (8 - x, 8 - y)
    obs = env._observation()
    next_obs, _, _, _, _ = env.step(action)
    return obs, action, next_obs


def _calibrate_all_profiles(router: C1HistoryRouter, repeats: int = 5) -> None:
    for key, spec in C1_CONTEXTS.items():
        for band in range(3):
            x = band * 3 + 1
            for action in range(4):
                for _ in range(repeats):
                    obs, action_value, next_obs = _transition(spec, x, 4, action)
                    router.observe_calibration(key, obs, action_value, next_obs)


def _calibrate_left_boundary_action_zero(router: C1HistoryRouter, repeats: int = 12) -> None:
    for key, spec in C1_CONTEXTS.items():
        for _ in range(repeats):
            obs, action, next_obs = _transition(spec, 0, 4, 0)
            router.observe_calibration(key, obs, action, next_obs)


def test_calibration_records_five_outcomes_and_boundary_zero_moves() -> None:
    router = C1HistoryRouter()
    _calibrate_left_boundary_action_zero(router)

    assert router.outcome_counts.shape == (6, 3, 4, 5)
    assert router.direction_counts.shape == (6, 3, 4, 4)
    assert router.outcome_counts[context_index("B0"), 0, 0, OUTCOME_BLOCKED] == 12
    assert router.outcome_counts[context_index("A0"), 0, 0, 1] == 12
    # The blocked B response is retained as left-direction evidence at x = 0.
    assert router.direction_counts[context_index("B0"), 0, 0, 3] == pytest.approx(12)


def test_numeric_endpoint_routes_calibrate_a0_and_b2_through_snapshot() -> None:
    router = C1HistoryRouter()
    a0_transition = _transition(C1_CONTEXTS["A0"], 0, 4, 0)
    b2_transition = _transition(C1_CONTEXTS["B2"], 8, 4, 1)

    router.observe_calibration(1, *a0_transition)
    router.observe_calibration(6, *b2_transition)
    assert router.outcome_counts[context_index("A0"), 0, 0, 1] == 1
    assert router.outcome_counts[context_index("B2"), 2, 1, 2] == 1
    assert router.outcome_counts[context_index("A1")].sum() == 0
    assert router.outcome_counts[context_index("B1")].sum() == 0

    restored = C1HistoryRouter.from_snapshot(router.snapshot())
    assert restored.outcome_counts[context_index("A0"), 0, 0, 1] == 1
    assert restored.outcome_counts[context_index("B2"), 2, 1, 2] == 1
    assert restored.outcome_counts.sum() == 2


def test_blocked_inference_uses_all_directions_that_clip_at_visible_wall() -> None:
    router = C1HistoryRouter()
    _calibrate_left_boundary_action_zero(router)
    spec = C1_CONTEXTS["B0"]

    obs, action, blocked = _transition(spec, 0, 4, 0)
    assert np.array_equal(obs[:2], blocked[:2])
    router.observe(obs, action, blocked)
    assert router.route(obs) == ROUTE_ROOT  # One informative event is insufficient.
    router.observe(obs, action, blocked)

    assert router.family_posterior[1] > router.family_posterior[0]
    assert router.family_posterior[1] >= 0.8
    assert router.route(obs) == ROUTE_PARENT_B


def test_opposite_recent_transitions_overturn_old_profile_match() -> None:
    router = C1HistoryRouter(window=6, evidence_decay=0.9)
    _calibrate_left_boundary_action_zero(router)
    b_spec = C1_CONTEXTS["B0"]
    a_spec = C1_CONTEXTS["A0"]

    blocked_event = _transition(b_spec, 0, 4, 0)
    router.observe(*blocked_event)
    router.observe(*blocked_event)
    assert router.route(blocked_event[0]) == ROUTE_PARENT_B

    # Four contradictory right moves arrive while both older blocked events
    # remain in the six-event window. Their decayed evidence no longer freezes
    # recognition on family B.
    for _ in range(4):
        router.observe(*_transition(a_spec, 0, 4, 0))
    assert len(router.events) == 6
    assert router.route() == ROUTE_PARENT_A


def test_profile_route_uses_cached_unlabeled_inference() -> None:
    router = C1HistoryRouter(window=32, confidence_threshold=0.8)
    _calibrate_all_profiles(router)
    spec = C1_CONTEXTS["A2"]

    for _ in range(4):
        router.observe(*_transition(spec, 7, 4, 2))
    assert router.route(_transition(spec, 7, 4, 2)[0]) == 3  # A2 is route ID 3.
    previous_posterior = router.posterior.copy()
    route_calls = router.route_calls
    assert router.route() == 3
    np.testing.assert_array_equal(router.posterior, previous_posterior)
    assert router.route_calls == route_calls + 1
    assert router.diagnostics()["inference_calls"] == 4


def test_router_never_calls_hidden_mapping_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calibration_rows = []
    for key, spec in C1_CONTEXTS.items():
        calibration_rows.append((key, *_transition(spec, 0, 4, 0)))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("router accessed hidden C1 mapping helper")

    monkeypatch.setattr(c1_envs, "mapping_for_state", forbidden)
    router = C1HistoryRouter()
    for key, obs, action, next_obs in calibration_rows:
        router.observe_calibration(key, obs, action, next_obs)
    for _, obs, action, next_obs in calibration_rows:
        router.observe(obs, action, next_obs)
        router.route(obs)


def test_snapshot_restore_and_clear_keep_calibrated_bank() -> None:
    router = C1HistoryRouter(window=8)
    _calibrate_all_profiles(router, repeats=2)
    for _ in range(3):
        router.observe(*_transition(C1_CONTEXTS["A2"], 7, 4, 2))
    saved = router.snapshot()
    restored = C1HistoryRouter.from_snapshot(saved)

    np.testing.assert_array_equal(restored.outcome_counts, router.outcome_counts)
    np.testing.assert_array_equal(restored.direction_counts, router.direction_counts)
    assert len(restored.events) == len(router.events)
    assert restored.route() == router.route()
    np.testing.assert_array_equal(restored.features(), router.features())

    outcome_bank = restored.outcome_counts.copy()
    direction_bank = restored.direction_counts.copy()
    restored.clear_history()
    assert len(restored.events) == 0
    assert restored.route() == ROUTE_ROOT
    np.testing.assert_array_equal(restored.outcome_counts, outcome_bank)
    np.testing.assert_array_equal(restored.direction_counts, direction_bank)


def test_bounded_history_identical_profiles_and_diagnostic_features() -> None:
    router = C1HistoryRouter(window=32)
    router.direction_counts[:] = np.asarray([0.0, 0.0, 0.0, 8.0])
    router.outcome_counts[:] = 1
    for _ in range(100):
        router.observe(*_transition(C1_CONTEXTS["A0"], 0, 4, 0))

    assert len(router.events) == 32
    assert router.route() == ROUTE_ROOT
    assert router.last_confidence == pytest.approx(1 / 6)
    assert router.last_family_confidence == pytest.approx(0.5)
    features = router.features()
    assert features.shape == (8,)
    assert np.all(np.isfinite(features))
    assert np.all((features >= 0.0) & (features <= 1.0))
    diagnostics = router.diagnostics()
    assert diagnostics["route_calls"] == 1
    assert diagnostics["unknown_routes"] == 1
    assert diagnostics["inference_cost"] > 0


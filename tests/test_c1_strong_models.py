"""Causality and adaptation checks for unlabeled C1 transition models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import online_rl.c1_envs as c1_envs  # noqa: E402
from online_rl.c1_strong_models import C1ModelController, load_source_prior  # noqa: E402


IDENTITY = (0, 1, 2, 3)
SWAP_UP_RIGHT_DOWN_LEFT = (1, 0, 3, 2)


def _obs(cell: tuple[int, int]) -> np.ndarray:
    x, y = cell
    return np.asarray([x / 8, y / 8, 1.0, 1.0], dtype=np.float32)


def _transition(cell: tuple[int, int], action: int, mapping: tuple[int, ...]):
    deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))
    dx, dy = deltas[mapping[action]]
    destination = (min(max(cell[0] + dx, 0), 8), min(max(cell[1] + dy, 0), 8))
    return _obs(cell), action, _obs(destination)


def _teach(controller: C1ModelController, mapping: tuple[int, ...], band: int = 0) -> None:
    x = band * 3 + 1
    for action in range(4):
        controller.observe(*_transition((x, 4), action, mapping))


@pytest.fixture
def identity_source() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def test_source_prior_comes_from_recorded_moving_responses() -> None:
    prior = load_source_prior()
    assert prior.shape == (4, 4)
    assert np.all(prior.sum(axis=1) > 0)
    assert np.all(np.diag(prior) > prior.sum(axis=1) - np.diag(prior))


def test_controller_is_unlabeled_and_ignores_reward(identity_source, monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("controller accessed hidden mapping metadata")

    monkeypatch.setattr(c1_envs, "mapping_for_state", forbidden)
    first = C1ModelController("window_assignment", source_prior=identity_source)
    second = C1ModelController("window_assignment", source_prior=identity_source)
    event = _transition((1, 4), 0, SWAP_UP_RIGHT_DOWN_LEFT)
    first.observe(*event, reward=-4.0)
    second.observe(*event, reward=1e6)
    probs_a = first.action_probabilities(event[0], [0.1, 0.7, 0.1, 0.1])
    probs_b = second.action_probabilities(event[0], [0.1, 0.7, 0.1, 0.1])
    np.testing.assert_array_equal(probs_a, probs_b)
    assert first.parameter_update_count == 0
    assert first.statistical_update_count == 1
    assert not any("context" in name or "switch" in name for name in vars(first))


def test_source_initialization_preserves_source_policy_and_unknown_map_stays_uncertain(identity_source) -> None:
    base = np.asarray([0.05, 0.80, 0.10, 0.05])
    counts = C1ModelController("window_counts", source_prior=identity_source)
    assigned = C1ModelController("window_assignment", source_prior=identity_source)
    obs = _obs((4, 4))
    np.testing.assert_allclose(counts.action_probabilities(obs, base), base, atol=1e-12)
    initial = assigned.action_probabilities(obs, base)
    assert initial[1] > initial[0]
    assert initial[1] < 1.0
    assert np.allclose(initial.sum(), 1.0)


def test_window_assignment_learns_a_new_permutation_from_transitions(identity_source) -> None:
    controller = C1ModelController("window_assignment", window=16, source_prior=identity_source)
    _teach(controller, SWAP_UP_RIGHT_DOWN_LEFT)
    base = np.asarray([1.0, 0.0, 0.0, 0.0])
    action_probs = controller.action_probabilities(_obs((1, 4)), base)
    assert int(np.argmax(action_probs)) == SWAP_UP_RIGHT_DOWN_LEFT.index(0)
    assert action_probs.max() > 0.99
    assert controller.confidence > 0.99


def test_window_counts_learn_and_forget_after_a_switch(identity_source) -> None:
    controller = C1ModelController(
        "window_counts", window=4, source_prior=identity_source, source_prior_strength=0.1
    )
    _teach(controller, SWAP_UP_RIGHT_DOWN_LEFT)
    base = np.asarray([1.0, 0.0, 0.0, 0.0])
    before = controller.action_probabilities(_obs((1, 4)), base)
    assert int(np.argmax(before)) == 1

    _teach(controller, IDENTITY)
    after = controller.action_probabilities(_obs((1, 4)), base)
    assert int(np.argmax(after)) == 0
    assert len(controller.events) == 4


def test_adaptive_assignment_detects_and_learns_a_mapping_change(identity_source) -> None:
    controller = C1ModelController("adaptive_assignment", window=16, source_prior=identity_source)
    _teach(controller, SWAP_UP_RIGHT_DOWN_LEFT)
    assert controller.action_probabilities(_obs((1, 4)), [1, 0, 0, 0]).argmax() == 1

    controller.observe(*_transition((1, 4), 0, IDENTITY))
    assert controller.change_detections == 1
    for action in range(4):
        controller.observe(*_transition((1, 4), action, IDENTITY))
    probabilities = controller.action_probabilities(_obs((1, 4)), [1, 0, 0, 0])
    assert int(np.argmax(probabilities)) == 0
    assert probabilities.max() > 0.99


def test_blocked_moves_are_censored_for_counts_but_inform_assignment(identity_source) -> None:
    counts = C1ModelController("window_counts", source_prior=identity_source)
    assignment = C1ModelController("window_assignment", source_prior=identity_source)
    counts_before = counts._counts.copy()
    confidence_before = assignment.confidence
    blocked = _transition((0, 4), 3, IDENTITY)  # Source left command clips at x = 0.

    counts.observe(*blocked)
    assignment.observe(*blocked)
    np.testing.assert_array_equal(counts._counts, counts_before)
    assert len(counts.events) == 1
    assert assignment.confidence > confidence_before


@pytest.mark.parametrize("kind", ("window_counts", "window_assignment", "adaptive_assignment"))
def test_bounded_memory_snapshot_restore_and_clear(identity_source, kind: str) -> None:
    controller = C1ModelController(kind, window=5, source_prior=identity_source)
    for _ in range(12):
        controller.observe(*_transition((1, 4), 0, SWAP_UP_RIGHT_DOWN_LEFT))
    assert len(controller.events) == 5
    summary = controller.memory_summary()
    assert summary["retained_events"] == summary["max_events"] == 5
    assert summary["estimated_controller_bytes"] >= summary["state_array_bytes"]

    restored = C1ModelController.from_snapshot(controller.snapshot())
    assert restored.events == controller.events
    np.testing.assert_allclose(
        restored.action_probabilities(_obs((1, 4)), [0.7, 0.1, 0.1, 0.1]),
        controller.action_probabilities(_obs((1, 4)), [0.7, 0.1, 0.1, 0.1]),
    )
    assert restored.statistical_update_count == controller.statistical_update_count
    assert restored.parameter_update_count == 0

    restored.clear_history()
    assert restored.events == ()
    assert restored.statistical_update_count == 0
    cleared = restored.action_probabilities(_obs((1, 4)), [1, 0, 0, 0])
    assert int(np.argmax(cleared)) == 0
    assert cleared[0] > 0.90


def test_adaptive_information_gain_explores_only_while_mapping_is_ambiguous(identity_source) -> None:
    controller = C1ModelController("adaptive_assignment", source_prior=identity_source)
    base = np.asarray([1.0, 0.0, 0.0, 0.0])
    initial = controller.action_probabilities(_obs((4, 4)), base)
    assert initial[0] < 1.0
    _teach(controller, IDENTITY)
    learned = controller.action_probabilities(_obs((1, 4)), base)
    assert learned[0] > 0.99
    assert np.all(np.isfinite(controller.features()))
    assert controller.features().shape == (8,)


"""Tests for bounded cumulative evidence by resident slot."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.shared_write_grouping import empty_profile, observe_profile  # noqa: E402
from online_rl.write_decision_evidence import EvidenceWindow, _reward_features  # noqa: E402
import online_rl.write_cumulative_evidence as cumulative_module  # noqa: E402
from online_rl.write_cumulative_evidence import CumulativeEvidence  # noqa: E402


_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _obs(position: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    return np.asarray((*position, *goal), dtype=np.float32) / np.float32(8.0)


def _synthetic_transitions(count: int, seed: int = 8402):
    """Generate legal clipped-grid transitions with reset episodes."""

    rng = np.random.default_rng(seed)
    rows = []
    position = (0, 0)
    goal = (8, 8)
    episode_steps = 0
    for _ in range(count):
        if episode_steps == 0:
            position = (int(rng.integers(9)), int(rng.integers(9)))
            goal = (int(rng.integers(9)), int(rng.integers(9)))
            while goal == position:
                goal = (int(rng.integers(9)), int(rng.integers(9)))

        action = int(rng.integers(4))
        dx, dy = _DELTAS[action]
        next_position = (
            min(8, max(0, position[0] + dx)),
            min(8, max(0, position[1] + dy)),
        )
        terminated = next_position == goal
        distance_before = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
        distance_after = abs(next_position[0] - goal[0]) + abs(next_position[1] - goal[1])
        reward = 0.2 * (distance_before - distance_after) - 0.03 + float(terminated)
        rows.append(
            (
                _obs(position, goal),
                action,
                reward,
                _obs(next_position, goal),
                terminated,
            )
        )
        if terminated or episode_steps >= 63:
            episode_steps = 0
        else:
            episode_steps += 1
            position = next_position
    return rows


def _assert_profile_equal(expected, actual):
    assert np.array_equal(expected.movement_counts, actual.movement_counts)
    np.testing.assert_allclose(expected.reward_xtx, actual.reward_xtx, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(expected.reward_xty, actual.reward_xty, rtol=1e-6, atol=1e-5)
    assert actual.reward_n == expected.reward_n


def _single_event_counts(row):
    counts = empty_profile()
    observe_profile(counts, row[0], row[1], row[3])
    return counts


def test_one_thousand_synthetic_transitions_match_cumulative_window() -> None:
    rows = _synthetic_transitions(1_000)
    window = EvidenceWindow(capacity=1_000)
    cumulative = CumulativeEvidence(slots=1)

    for row in rows:
        window.observe(*row)
        cumulative.observe(0, *row)

    _assert_profile_equal(window.profile(), cumulative.profile_for_slot(0))
    assert cumulative.reward_n.tolist() == [1_000]


def test_clear_is_slot_local_and_bounds_storage_by_slot_count() -> None:
    rows_a = _synthetic_transitions(80, seed=1)
    rows_b = _synthetic_transitions(35, seed=2)
    cumulative = CumulativeEvidence(slots=2)
    expected_b = EvidenceWindow(capacity=35)
    for row in rows_a:
        cumulative.observe(0, *row)
    for row in rows_b:
        cumulative.observe(1, *row)
        expected_b.observe(*row)

    before_b = cumulative.profile_for_slot(1)
    cumulative.clear(0)
    empty_a = cumulative.profile_for_slot(0)
    _assert_profile_equal(EvidenceWindow(capacity=1).profile(), empty_a)
    _assert_profile_equal(expected_b.profile(), cumulative.profile_for_slot(1))
    _assert_profile_equal(before_b, cumulative.profile_for_slot(1))

    for slots in range(1, 9):
        instance = CumulativeEvidence(slots)
        assert instance.nbytes == slots * 200
        assert instance.nbytes <= 8 * 200
        assert instance.state_dict()["nbytes"] == instance.nbytes


def test_rescale_halves_all_statistics_before_adding_next_sample() -> None:
    row = _synthetic_transitions(1, seed=17)[0]
    cumulative = CumulativeEvidence(slots=1)
    state = cumulative.state_dict()
    counts = _single_event_counts(row).astype(np.uint32) * np.uint32(60_000)
    state["movement_counts"][0] = counts.astype(np.uint16)
    state["reward_n"][0] = 60_000
    features = _reward_features(row[0], row[3], row[4])
    state["reward_xtx"][0] = np.outer(features, features) * 60_000.0
    state["reward_xty"][0] = features * float(row[2]) * 60_000.0
    state["nbytes"] = cumulative.nbytes
    cumulative = CumulativeEvidence.from_state_dict(state)

    old_counts = cumulative._movement_counts[0].astype(np.uint32).copy()
    old_xtx = cumulative._reward_xtx[0].copy()
    old_xty = cumulative._reward_xty[0].copy()
    cumulative.observe(0, *row)

    expected_counts = (old_counts // 2 + _single_event_counts(row).astype(np.uint32)).astype(
        np.uint16
    )
    np.testing.assert_array_equal(cumulative._movement_counts[0], expected_counts)
    np.testing.assert_allclose(
        cumulative._reward_xtx[0], old_xtx * 0.5 + np.outer(features, features)
    )
    np.testing.assert_allclose(
        cumulative._reward_xty[0], old_xty * 0.5 + features * float(row[2])
    )
    assert cumulative.reward_n.tolist() == [30_001]
    assert cumulative.rescale_count.tolist() == [1]


def test_round_trip_and_invalid_input_validation_preserve_finite_bounded_state() -> None:
    rows = _synthetic_transitions(30)
    original = CumulativeEvidence(slots=3)
    for index, row in enumerate(rows):
        original.observe(index % 3, *row)

    state = original.state_dict()
    restored = CumulativeEvidence.from_state_dict(state)
    assert restored.nbytes == original.nbytes
    assert restored.reward_n.tolist() == original.reward_n.tolist()
    assert restored.rescale_count.tolist() == original.rescale_count.tolist()
    for slot in range(3):
        _assert_profile_equal(original.profile_for_slot(slot), restored.profile_for_slot(slot))

    before = restored.state_dict()
    invalid = (rows[0][0], 4, rows[0][2], rows[0][3], rows[0][4])
    with pytest.raises(ValueError, match="action"):
        restored.observe(0, *invalid)
    after = restored.state_dict()
    for name in ("movement_counts", "reward_xtx", "reward_xty", "reward_n", "rescale_count"):
        np.testing.assert_array_equal(before[name], after[name])

    malformed = dict(state)
    malformed["reward_xtx"] = malformed["reward_xtx"].astype(np.float32)
    with pytest.raises(ValueError, match="reward_xtx"):
        CumulativeEvidence.from_state_dict(malformed)


def test_profile_does_not_replay_events_and_invalid_rescale_input_is_atomic(monkeypatch) -> None:
    original_observe_profile = cumulative_module.observe_profile
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_observe_profile(*args, **kwargs)

    monkeypatch.setattr(cumulative_module, "observe_profile", counted)
    cumulative = CumulativeEvidence(slots=1)
    rows = _synthetic_transitions(100)
    for row in rows:
        before = calls
        cumulative.observe(0, *row)
        assert calls == before + 1
    before_profile = calls
    for _ in range(50):
        cumulative.profile_for_slot(0)
    assert calls == before_profile

    # Set valid near-threshold statistics, then make an invalid action. Since
    # event validation precedes rescaling, the invalid call has no side effect.
    state = cumulative.state_dict()
    state["reward_n"][0] = 60_000
    state["movement_counts"][0].fill(0)
    state["movement_counts"][0, 0, 0, 0] = 60_000
    cumulative = CumulativeEvidence.from_state_dict(state)
    before = cumulative.state_dict()
    bad = (rows[0][0], 7, rows[0][2], rows[0][3], rows[0][4])
    with pytest.raises(ValueError, match="action"):
        cumulative.observe(0, *bad)
    after = cumulative.state_dict()
    for name in ("movement_counts", "reward_xtx", "reward_xty", "reward_n", "rescale_count"):
        np.testing.assert_array_equal(before[name], after[name])


@pytest.mark.parametrize("slots", [0, 9, True, 1.5])
def test_slot_capacity_must_be_fixed_between_one_and_eight(slots) -> None:
    with pytest.raises(ValueError, match="slots"):
        CumulativeEvidence(slots)

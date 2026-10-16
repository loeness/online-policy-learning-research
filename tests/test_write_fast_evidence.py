"""Correctness checks for incremental transition-evidence summaries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.envs import ContextSpec, NavigationEnv  # noqa: E402
from online_rl.write_decision_evidence import EvidenceWindow  # noqa: E402
import online_rl.write_fast_evidence as fast_evidence  # noqa: E402
from online_rl.write_fast_evidence import IncrementalEvidenceWindow  # noqa: E402


def _legal_transitions(count: int, seed: int = 4901):
    """Generate real transitions from the ordinary 9 by 9 navigation env."""

    env = NavigationEnv(ContextSpec("normal", 0))
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    episode = 0
    for _ in range(count):
        action = int(rng.integers(4))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rows.append((obs.copy(), action, reward, next_obs.copy(), terminated))
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
        else:
            obs = next_obs
    env.close()
    return rows


def _assert_profiles_equal(expected, actual) -> None:
    assert np.array_equal(actual.movement_counts, expected.movement_counts)
    np.testing.assert_allclose(actual.reward_xtx, expected.reward_xtx, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(actual.reward_xty, expected.reward_xty, rtol=1e-6, atol=1e-5)
    assert actual.reward_n == expected.reward_n


def test_incremental_profile_matches_recomputed_window_on_legal_transitions() -> None:
    rows = _legal_transitions(1_000)
    expected = EvidenceWindow(capacity=128)
    actual = IncrementalEvidenceWindow(capacity=128)
    saw_terminal = False

    for step, row in enumerate(rows):
        expected.observe(*row)
        actual.observe(*row)
        saw_terminal |= bool(row[-1])
        if step % 19 == 0 or step == len(rows) - 1:
            _assert_profiles_equal(expected.profile(), actual.profile())

    assert saw_terminal, "seeded legal navigation stream should include goal transitions"
    assert actual.nbytes == expected.nbytes + actual.cache_nbytes
    assert actual.cache_nbytes == (3 * 4 * 4 * 4) + (3 * 3 * 8) + (3 * 8)


@pytest.mark.parametrize("capacity", [1, 4, 128])
def test_overwrite_zero_motion_and_terminal_features_match(capacity: int) -> None:
    blocked = np.asarray((0, 0, 8, 8), dtype=np.float32) / np.float32(8)
    near_goal = np.asarray((1, 0, 0, 0), dtype=np.float32) / np.float32(8)
    goal = np.asarray((0, 0, 0, 0), dtype=np.float32) / np.float32(8)
    rows = [
        # A blocked boundary action is a valid zero-movement transition.
        (blocked, 0, -0.03, blocked, False),
        (near_goal, 3, 0.97, goal, True),
        *_legal_transitions(20, seed=91),
    ]
    expected = EvidenceWindow(capacity=capacity)
    actual = IncrementalEvidenceWindow(capacity=capacity)

    for row in rows:
        expected.observe(*row)
        actual.observe(*row)
        _assert_profiles_equal(expected.profile(), actual.profile())


def test_round_trip_rebuilds_cache_and_continues_for_base_or_incremental_state() -> None:
    rows = _legal_transitions(41, seed=713)
    base = EvidenceWindow(capacity=17)
    incremental = IncrementalEvidenceWindow(capacity=17)
    for row in rows[:11]:
        base.observe(*row)
        incremental.observe(*row)

    base_state = base.state_dict()
    fast_state = incremental.state_dict()
    assert not {"movement_counts_cache", "reward_xtx_cache", "reward_xty_cache"}.intersection(
        fast_state
    )
    assert fast_state["nbytes"] == base.nbytes
    restored_from_base = IncrementalEvidenceWindow.from_state_dict(base_state)
    restored = IncrementalEvidenceWindow.from_state_dict(fast_state)
    assert isinstance(restored, IncrementalEvidenceWindow)
    assert restored.nbytes == restored_from_base.nbytes
    _assert_profiles_equal(base.profile(), restored.profile())
    _assert_profiles_equal(base.profile(), restored_from_base.profile())

    for row in rows[11:]:
        base.observe(*row)
        restored.observe(*row)
        _assert_profiles_equal(base.profile(), restored.profile())


def test_profile_does_not_scan_history(monkeypatch) -> None:
    original = fast_evidence.observe_profile
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fast_evidence, "observe_profile", counted)
    window = IncrementalEvidenceWindow(capacity=8)
    rows = _legal_transitions(32, seed=117)
    for row in rows:
        before = calls
        was_full = window.size == window.capacity
        window.observe(*row)
        expected_updates = 2 if was_full else 1
        assert calls - before == expected_updates

    before_profile = calls
    for _ in range(100):
        window.profile()
    assert calls == before_profile

    restored = IncrementalEvidenceWindow.from_state_dict(window.state_dict())
    assert calls - before_profile == window.size
    assert restored.size == window.size


def test_invalid_event_does_not_change_incremental_window_or_cache() -> None:
    window = IncrementalEvidenceWindow(capacity=2)
    row = _legal_transitions(1)[0]
    window.observe(*row)
    before = window.profile()

    invalid = (row[0], 9, row[2], row[3], row[4])
    with pytest.raises(ValueError, match="action"):
        window.observe(*invalid)

    _assert_profiles_equal(before, window.profile())

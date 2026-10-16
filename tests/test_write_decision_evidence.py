"""Bounded, label-free transition and reward evidence tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.write_decision_evidence import (  # noqa: E402
    EvidenceWindow,
    RewardAwareGroups,
)


_DIRECTIONS = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
}


def _observation(x: int, y: int, goal_x: int, goal_y: int) -> np.ndarray:
    return np.asarray((x, y, goal_x, goal_y), dtype=np.float32) / np.float32(8.0)


def _cell_event(
    cell: tuple[int, int, int], *, terminated: bool
) -> tuple[np.ndarray, int, np.ndarray, bool]:
    band, action, direction = cell
    dx, dy = _DIRECTIONS[direction]
    x, y = band * 3 + 1, 4
    next_x, next_y = x + dx, y + dy
    if terminated:
        goal_x, goal_y = next_x, next_y
    else:
        goal_x, goal_y = x - dx, y - dy
    return (
        _observation(x, y, goal_x, goal_y),
        action,
        _observation(next_x, next_y, goal_x, goal_y),
        terminated,
    )


def _profile_for_cells(
    cells: list[tuple[int, int, int]],
    *,
    repeats: int = 2,
    reward_fn=None,
) -> tuple[object, list[float]]:
    count = len(cells) * repeats
    window = EvidenceWindow(max(count, 1))
    rewards: list[float] = []
    for cell in cells:
        for _ in range(repeats):
            obs, action, next_obs, terminated = _cell_event(cell, terminated=True)
            features = np.asarray((8.0, 1.0, 1.0), dtype=np.float64)
            reward = float(reward_fn(features) if reward_fn else 1.0)
            rewards.append(reward)
            window.observe(obs, action, reward, next_obs, terminated)
    return window.profile(), rewards


def _conditional_reward_profiles():
    cells = [(0, action, action) for action in range(4)]
    window_x = EvidenceWindow(16)
    window_constant = EvidenceWindow(16)
    observed_rewards_x: list[float] = []
    observed_rewards_constant: list[float] = []

    # Eight goal transitions give Δd = 1; eight blocked transitions give
    # Δd = 0. Both reward streams have mean 0.5, but different conditional fits.
    for cell in cells:
        for _ in range(2):
            obs, action, next_obs, terminated = _cell_event(cell, terminated=True)
            feature = 8.0
            reward_x = 0.125 * feature
            reward_constant = 0.5
            observed_rewards_x.append(reward_x)
            observed_rewards_constant.append(reward_constant)
            window_x.observe(obs, action, reward_x, next_obs, terminated)
            window_constant.observe(obs, action, reward_constant, next_obs, terminated)

    blocked_obs = _observation(8, 4, 0, 0)
    for _ in range(8):
        window_x.observe(blocked_obs, 1, 0.0, blocked_obs, False)
        window_constant.observe(blocked_obs, 1, 0.5, blocked_obs, False)
        observed_rewards_x.append(0.0)
        observed_rewards_constant.append(0.5)

    assert np.mean(observed_rewards_x) == pytest.approx(np.mean(observed_rewards_constant))
    return window_x.profile(), window_constant.profile()


def _profile_with_reward_model(
    *,
    sign: float = 1.0,
    terminal_coef: float = 0.0,
    observe_terminal: bool = True,
    repeats: int = 4,
):
    """Build real transition samples with balanced movement and termination."""

    window = EvidenceWindow(4 * repeats)
    for action, (dx, dy) in _DIRECTIONS.items():
        x, y = 4, 4
        next_x, next_y = x + dx, y + dy
        for repeat in range(repeats):
            pattern = repeat % 4
            toward = pattern in (0, 2)
            goal_x, goal_y = (next_x, next_y) if toward else (x, y)
            obs = _observation(x, y, goal_x, goal_y)
            next_obs = _observation(next_x, next_y, goal_x, goal_y)
            current_distance = abs(x - goal_x) + abs(y - goal_y)
            next_distance = abs(next_x - goal_x) + abs(next_y - goal_y)
            delta_distance = float((current_distance - next_distance) * 8)
            terminated = bool(observe_terminal and pattern in (0, 1))
            reward = sign * (0.1 * delta_distance + 2.0 + terminal_coef * float(terminated))
            window.observe(obs, action, reward, next_obs, terminated)
    return window.profile()


def _rank_deficient_reward_profile():
    window = EvidenceWindow(16)
    for action, (dx, dy) in _DIRECTIONS.items():
        x, y = 4, 4
        next_x, next_y = x + dx, y + dy
        for _ in range(4):
            # Every row has the same reward movement feature (+8), making it
            # collinear with the intercept despite nonzero diagonal support.
            obs = _observation(x, y, next_x, next_y)
            next_obs = _observation(next_x, next_y, next_x, next_y)
            window.observe(obs, action, 2.8, next_obs, False)
    return window.profile()


def test_same_transition_map_with_opposite_reward_is_split_by_reward_evidence() -> None:
    cells = [(band, action, action) for band in range(2) for action in range(2)]
    positive, _ = _profile_for_cells(cells, repeats=4, reward_fn=lambda features: 0.2 * features[0] + 0.5)
    negative, _ = _profile_for_cells(cells, repeats=4, reward_fn=lambda features: -(0.2 * features[0] + 0.5))
    groups = RewardAwareGroups()

    first = groups.assign(positive)
    second = groups.assign(negative)

    assert first.group == 0 and first.reason == "first_group"
    assert second.group == 1 and second.reason == "opposite_reward_group"
    assert second.scores[0]["overlap"] == 4
    assert second.scores[0]["agreement"] == 1.0
    assert second.scores[0]["reward_cosine"] <= -0.99


def test_conditional_reward_fit_separates_equal_mean_streams() -> None:
    reward_x, reward_constant = _conditional_reward_profiles()
    groups = RewardAwareGroups()
    assert groups.assign(reward_x).group == 0

    result = groups.assign(reward_constant, commit=False)
    assert result.group == -1
    assert result.reason == "reward_mismatch"
    assert result.scores[0]["agreement"] == 1.0
    assert result.scores[0]["reward_cosine"] < 0.5


def test_support_aware_reward_similarity_drops_unobserved_terminal_feature() -> None:
    full = _profile_with_reward_model(terminal_coef=0.0, observe_terminal=True)
    no_terminal = _profile_with_reward_model(observe_terminal=False)
    groups = RewardAwareGroups(reward_support_aware=True)

    assert groups.assign(full).group == 0
    restored = RewardAwareGroups.from_state_dict(groups.state_dict())
    assert restored.config == groups.config
    assert restored.reward_support_aware is True
    result = groups.assign(no_terminal, commit=False)
    restored_result = restored.assign(no_terminal, commit=False)

    assert result.group == 0 and result.reason == "matched"
    assert restored_result == result
    score = result.scores[0]
    assert score["compared_feature_count"] == 2
    assert score["compared_feature_mask"] == 0b011
    assert score["dropped_unsupported_terminal"] == 1
    assert score["reward_comparable"] == 1
    assert score["reward_cosine"] >= 0.5


def test_support_aware_opposite_reward_needs_observed_terminal_support() -> None:
    positive = _profile_with_reward_model(sign=1.0, terminal_coef=0.2)
    negative = _profile_with_reward_model(sign=-1.0, terminal_coef=0.2)
    groups = RewardAwareGroups(reward_support_aware=True)

    assert groups.assign(positive).group == 0
    opposite = groups.assign(negative)
    assert opposite.group == 1 and opposite.reason == "opposite_reward_group"
    assert opposite.scores[0]["compared_feature_count"] == 3
    assert opposite.scores[0]["compared_feature_mask"] == 0b111
    assert opposite.scores[0]["dropped_unsupported_terminal"] == 0
    assert opposite.scores[0]["reward_cosine"] <= -0.99


def test_support_aware_no_terminal_profiles_compare_only_observed_features() -> None:
    first = _profile_with_reward_model(terminal_coef=0.0, observe_terminal=False)
    second = _profile_with_reward_model(terminal_coef=0.0, observe_terminal=False)
    groups = RewardAwareGroups(reward_support_aware=True)

    assert groups.assign(first).group == 0
    result = groups.assign(second, commit=False)
    assert result.group == 0 and result.reason == "matched"
    assert result.scores[0]["compared_feature_count"] == 2
    assert result.scores[0]["dropped_unsupported_terminal"] == 1


def test_support_aware_low_reward_support_abstains_without_new_group() -> None:
    full = _profile_with_reward_model(sign=1.0, repeats=4)
    low_support_opposite = _profile_with_reward_model(sign=-1.0, repeats=2)
    groups = RewardAwareGroups(reward_support_aware=True)

    assert groups.assign(full).group == 0
    result = groups.assign(low_support_opposite)
    assert result.group == -1 and result.reason == "insufficient_reward_support"
    assert groups.active_groups == (0,)


def test_support_aware_rank_deficient_required_features_abstain() -> None:
    profile = _rank_deficient_reward_profile()
    groups = RewardAwareGroups(reward_support_aware=True)
    assert groups.assign(profile).group == 0

    result = groups.assign(profile, commit=False)
    assert result.group == -1
    assert result.reason == "insufficient_reward_identifiability"
    assert result.scores[0]["compared_feature_count"] == 2
    assert result.scores[0]["compared_feature_mask"] == 0b011
    assert result.scores[0]["rank_identifiable"] == 0
    assert result.scores[0]["reward_comparable"] == 0


def test_support_aware_terminal_mismatch_without_observation_is_uncertain() -> None:
    # This sample has no terminal feature observations, so a hypothetical
    # terminal coefficient cannot be inferred from its zero XTy entry.
    terminal_observed = _profile_with_reward_model(terminal_coef=0.1, observe_terminal=True)
    terminal_unobserved = _profile_with_reward_model(
        terminal_coef=-0.2, observe_terminal=False
    )
    groups = RewardAwareGroups(reward_support_aware=True)

    assert groups.assign(terminal_observed).group == 0
    result = groups.assign(terminal_unobserved, commit=False)
    assert result.group == 0 and result.reason == "matched"
    score = result.scores[0]
    assert score["compared_feature_mask"] == 0b011
    assert score["dropped_unsupported_terminal"] == 1
    assert score["compared_feature_count"] == 2


def test_insufficient_support_ambiguous_match_and_full_capacity_abstain() -> None:
    unsupported, _ = _profile_for_cells([(0, action, 0) for action in range(3)])
    groups = RewardAwareGroups()
    result = groups.assign(unsupported)
    assert result.group == -1 and result.reason == "insufficient_profile_support"

    low_reward_support, _ = _profile_for_cells(
        [(0, action, action) for action in range(4)], repeats=2
    )
    result = RewardAwareGroups().assign(low_reward_support)
    assert result.group == -1 and result.reason == "insufficient_reward_support"

    region_a = [(0, action, 0) for action in range(4)]
    region_b = [(1, action, 2) for action in range(4)]
    shared_zero = [(2, action, 0) for action in range(4)]
    shared_one = [(2, action, 1) for action in range(4)]
    profile_zero, _ = _profile_for_cells(region_a + shared_zero)
    profile_one, _ = _profile_for_cells(region_b + shared_one)
    query, _ = _profile_for_cells(
        region_a + region_b + shared_zero[:2] + shared_one[2:]
    )

    ambiguous = RewardAwareGroups()
    assert ambiguous.assign(profile_zero).group == 0
    assert ambiguous.assign(profile_one).group == 1
    tied = ambiguous.assign(query)
    assert tied.group == -1 and tied.reason == "ambiguous_match"
    assert tied.scores[0]["agreement"] == pytest.approx(0.75)
    assert tied.scores[1]["agreement"] == pytest.approx(0.75)

    map_zero = [(0, action, 0) for action in range(4)]
    map_one = [(0, action, 1) for action in range(4)]
    map_two = [(0, action, 2) for action in range(4)]
    full = RewardAwareGroups()
    assert full.assign(_profile_for_cells(map_zero, repeats=4)[0]).group == 0
    assert full.assign(_profile_for_cells(map_one, repeats=4)[0]).group == 1
    overflow = full.assign(_profile_for_cells(map_two, repeats=4)[0])
    assert overflow.group == -1 and overflow.reason == "max_groups_reached"
    assert full.active_groups == (0, 1)


def test_window_and_group_numeric_state_round_trip_and_validate() -> None:
    cells = [(0, action, action) for action in range(4)]
    profile, _ = _profile_for_cells(cells, repeats=4)
    window = EvidenceWindow(20)
    for cell in cells:
        for _ in range(4):
            obs, action, next_obs, terminated = _cell_event(cell, terminated=True)
            window.observe(obs, action, 1.0, next_obs, terminated)

    restored_window = EvidenceWindow.from_state_dict(window.state_dict())
    restored_profile = restored_window.profile()
    assert restored_window.nbytes == window.nbytes
    assert restored_profile.nbytes == profile.nbytes
    assert np.array_equal(restored_profile.movement_counts, profile.movement_counts)
    assert np.allclose(restored_profile.reward_xtx, profile.reward_xtx)
    assert np.allclose(restored_profile.reward_xty, profile.reward_xty)
    assert np.isfinite(restored_profile.coefficients()).all()

    groups = RewardAwareGroups()
    groups.assign(restored_profile)
    state = groups.state_dict()
    restored_groups = RewardAwareGroups.from_state_dict(state)
    assert restored_groups.config == groups.config
    assert restored_groups.active_groups == groups.active_groups
    assert restored_groups.nbytes == groups.nbytes == state["nbytes"]
    assert np.array_equal(restored_groups.group_counts, groups.group_counts)
    assert np.isfinite(restored_groups.state_dict()["reward_xtx"]).all()

    bad = dict(state)
    bad["reward_xty"] = bad["reward_xty"].astype(np.float64)
    with pytest.raises(ValueError, match="reward_xty"):
        RewardAwareGroups.from_state_dict(bad)


def test_default_group_mode_state_restore_preserves_legacy_assignment_output() -> None:
    first = _profile_with_reward_model(sign=1.0, terminal_coef=0.0)
    opposite = _profile_with_reward_model(sign=-1.0, terminal_coef=0.0)
    groups = RewardAwareGroups()
    assert groups.reward_support_aware is False
    groups.assign(first)
    state = groups.state_dict()

    restored = RewardAwareGroups.from_state_dict(state)
    legacy_state = groups.state_dict()
    legacy_state["config"].pop("reward_support_aware")
    restored_legacy = RewardAwareGroups.from_state_dict(legacy_state)
    expected = groups.assign(opposite, commit=False)

    for candidate in (restored, restored_legacy):
        actual = candidate.assign(opposite, commit=False)
        assert actual.group == expected.group
        assert actual.reason == expected.reason
        assert actual.scores == expected.scores
        assert candidate.reward_support_aware is False


def test_window_ring_is_fixed_capacity_and_rejects_invalid_state_counts() -> None:
    window = EvidenceWindow(capacity=2)
    blocked = _observation(8, 4, 0, 0)
    window.observe(blocked, 1, 0.0, blocked, False)
    window.observe(blocked, 1, 1.0, blocked, False)
    window.observe(blocked, 1, 2.0, blocked, False)
    assert window.size == 2
    assert float(window.profile().reward_n) == 2.0

    state = window.state_dict()
    state["size"] = 3
    with pytest.raises(ValueError, match="size"):
        EvidenceWindow.from_state_dict(state)

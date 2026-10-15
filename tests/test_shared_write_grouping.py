"""Label-free transition profiles and bounded dynamics grouping tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.shared_write_grouping import (  # noqa: E402
    DynamicsGrouper,
    empty_profile,
    observe_profile,
    supported_directions,
)


def profile_for(cells: list[tuple[int, int, int]]) -> np.ndarray:
    """Create anonymous, pure synthetic cells from (band, action, direction)."""

    profile = empty_profile()
    for band, action, direction in cells:
        profile[band, action, direction] = 4
        profile[band, action, (direction + 1) % 4] = 1
    return profile


def test_observe_profile_bins_actual_movement_and_ignores_blocked_actions() -> None:
    profile = empty_profile()
    assert profile.shape == (3, 4, 4)
    assert profile.dtype == np.uint16

    moved = observe_profile(
        profile,
        obs=(2 / 8, 3 / 8, 7 / 8, 6 / 8),
        action=1,
        next_obs=(3 / 8, 3 / 8, 7 / 8, 6 / 8),
    )
    assert moved is True
    assert profile[0, 1, 1] == 1  # band 0, action 1, physical right

    before = profile.copy()
    moved = observe_profile(
        profile,
        obs=(1.0, 4 / 8, 0.0, 4 / 8),
        action=1,
        next_obs=(1.0, 4 / 8, 0.0, 4 / 8),
    )
    assert moved is False
    assert np.array_equal(profile, before)

    # The right edge belongs to the final x band.
    assert observe_profile(
        profile,
        obs=(7 / 8, 4 / 8, 0.0, 0.0),
        action=3,
        next_obs=(1.0, 4 / 8, 0.0, 0.0),
    )
    assert profile[2, 3, 1] == 1


def test_supported_directions_require_support_and_eighty_percent_purity() -> None:
    profile = empty_profile()
    profile[0, 0, 2] = 2
    profile[0, 1, 3] = 4
    profile[0, 1, 0] = 1
    profile[0, 2, 1] = 3
    profile[0, 2, 2] = 1

    directions = supported_directions(profile)
    assert directions.shape == (3, 4)
    assert directions.dtype == np.int8
    assert directions[0, 0] == 2
    assert directions[0, 1] == 3  # Exactly 80 percent purity.
    assert directions[0, 2] == -1
    assert directions[0, 3] == -1


def test_matching_novelty_and_two_group_capacity_use_anonymous_profiles() -> None:
    cells = [(band, action, 0) for band in range(3) for action in (0, 1)]
    first = profile_for(cells)
    same = profile_for(cells)
    distinct = profile_for([(band, action, 1) for band, action, _ in cells])
    third = profile_for([(band, action, 2) for band, action, _ in cells])
    grouper = DynamicsGrouper()

    allocated = grouper.assign(first)
    assert allocated.group == 0 and allocated.reason == "first_group"
    matched = grouper.assign(same)
    assert matched.group == 0 and matched.reason == "matched"
    assert matched.scores[0] == {"overlap": 6, "agreement": 1.0}
    novel = grouper.assign(distinct)
    assert novel.group == 1 and novel.reason == "novel_group"
    assert novel.scores[0] == {"overlap": 6, "agreement": 0.0}

    overflow = grouper.assign(third)
    assert overflow.group == -1 and overflow.reason == "max_groups_reached"
    assert len(grouper.active_groups) == 2
    assert grouper.group_counts.shape == (2, 3, 4, 4)


def test_partial_overlap_and_tied_match_remain_unresolved() -> None:
    grouper = DynamicsGrouper()
    small = profile_for([(0, action, 0) for action in range(3)])
    assert grouper.assign(small).group == -1

    base_cells = [(0, action, 0) for action in range(4)]
    grouper.assign(profile_for(base_cells))
    partial = profile_for([(0, action, 0) for action in range(3)])
    result = grouper.assign(partial, commit=False)
    assert result.group == -1
    assert result.reason == "insufficient_overlap"
    assert result.scores[0] == {"overlap": 3, "agreement": 1.0}

    # Two groups overlap on four cells but disagree there. The query matches
    # each group on a separate four-cell region and half of the shared region.
    region_a = [(0, action) for action in range(4)]
    region_b = [(1, action) for action in range(4)]
    shared = [(2, action) for action in range(4)]
    group_zero = profile_for([(band, action, 0) for band, action in region_a + shared])
    group_one = profile_for(
        [(band, action, 2) for band, action in region_b]
        + [(band, action, 1) for band, action in shared]
    )
    tied_query = profile_for(
        [(band, action, 0) for band, action in region_a]
        + [(band, action, 2) for band, action in region_b]
        + [(band, action, 0) for band, action in shared[:2]]
        + [(band, action, 1) for band, action in shared[2:]]
    )
    tied = DynamicsGrouper()
    assert tied.assign(group_zero).group == 0
    assert tied.assign(group_one).group == 1
    before = tied.group_counts
    result = tied.assign(tied_query)
    assert result.group == -1 and result.reason == "ambiguous_match"
    assert result.scores[0] == {"overlap": 8, "agreement": 0.75}
    assert result.scores[1] == {"overlap": 8, "agreement": 0.75}
    assert np.array_equal(tied.group_counts, before)


def test_counts_saturate_and_state_round_trips_as_plain_numeric_data() -> None:
    grouper = DynamicsGrouper(max_groups=2)
    profile = empty_profile()
    profile[0, 0, 1] = np.iinfo(np.uint16).max
    assert observe_profile(
        profile,
        obs=(0.0, 0.0, 1.0, 1.0),
        action=0,
        next_obs=(1 / 8, 0.0, 1.0, 1.0),
    )
    assert profile[0, 0, 1] == np.iinfo(np.uint16).max

    seed_profile = profile_for([(band, action, 3) for band in range(3) for action in (0, 1)])
    grouper.assign(seed_profile)
    state = grouper.state_dict()
    assert isinstance(state["group_counts"], np.ndarray)
    assert state["group_counts"].dtype == np.uint16
    assert state["nbytes"] == 2 * 3 * 4 * 4 * np.dtype(np.uint16).itemsize
    assert state["nbytes"] == grouper.nbytes == grouper.memory_nbytes
    assert all(isinstance(value, (int, float)) for value in state["config"].values())

    restored = DynamicsGrouper.from_state_dict(state)
    assert restored.config == grouper.config
    assert restored.active_groups == grouper.active_groups
    assert restored.nbytes == grouper.nbytes
    assert np.array_equal(restored.group_counts, grouper.group_counts)

    state["group_counts"][0, 0, 0, 0] = 99
    assert restored.group_counts[0, 0, 0, 0] != 99


@pytest.mark.parametrize(
    "obs,action,next_obs,match",
    [
        ((0, 0), 0, (0, 1), "shape"),
        ((0, 0, 0, 0), 4, (0, 1 / 8, 0, 0), "action"),
        ((0, 0, 0, 0), 0, (0.1, 0, 0, 0), "grid"),
        ((0, 0, 0, 0), 0, (1 / 8, 1 / 8, 0, 0), "cardinal"),
    ],
)
def test_observer_rejects_malformed_events(obs, action, next_obs, match) -> None:
    with pytest.raises(ValueError, match=match):
        observe_profile(empty_profile(), obs, action, next_obs)


"""Bounded grouping from agent-visible transition profiles.

This is a deterministic candidate for 9 by 9 navigation observations. It uses
only normalized positions, commanded actions, and observed movement; it has no
context labels or transition-map lookup. It should not be read as a general
purpose dynamics discovery algorithm.

Profiles count observed physical directions in a ``(3, 4, 4)`` uint16 tensor
indexed by x band, commanded action, and movement direction. Cumulative
profiles should be committed at most once: callers must not repeatedly commit
the same cumulative profile, or its evidence will be counted again.
"""

from __future__ import annotations

from collections.abc import Mapping
from operator import index as as_index
from typing import Any, NamedTuple

import numpy as np


PROFILE_SHAPE = (3, 4, 4)
_GRID_MAX = 8
_MAX_COUNT = int(np.iinfo(np.uint16).max)
_MIN_PURITY = 0.8
_FIRST_GROUP_SUPPORT = 4
_STATE_VERSION = 1


class GroupAssignment(NamedTuple):
    """Result of an assignment attempt, including per-group comparison scores."""

    group: int
    reason: str
    scores: dict[int, dict[str, int | float]]


def empty_profile() -> np.ndarray:
    """Return an empty uint16 profile for a 9 by 9 navigation stream."""

    return np.zeros(PROFILE_SHAPE, dtype=np.uint16)


def _profile_array(profile: object) -> np.ndarray:
    counts = np.asarray(profile)
    if counts.shape != PROFILE_SHAPE:
        raise ValueError(f"profile must have shape {PROFILE_SHAPE}, got {counts.shape}")
    if counts.dtype != np.uint16:
        raise ValueError("profile must use uint16 counts")
    return counts


def _grid_position(observation: object, name: str) -> tuple[int, int]:
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must contain four finite normalized coordinates in [0, 1]")
    cells = np.rint(values * _GRID_MAX).astype(np.int64)
    if not np.allclose(values * _GRID_MAX, cells, rtol=0.0, atol=1e-4):
        raise ValueError(f"{name} coordinates must lie on the 9 by 9 grid")
    return int(cells[0]), int(cells[1])


def observe_profile(
    profile: np.ndarray,
    obs: object,
    action: int,
    next_obs: object,
) -> bool:
    """Add one agent-visible movement to ``profile``.

    Returns ``True`` when a one-cell cardinal movement was counted. A blocked
    action with no position change returns ``False`` and adds no evidence.
    Positions are the first two values of normalized four-coordinate
    observations. Direction IDs follow commanded-action convention: up, right,
    down, left. Malformed non-grid or multi-cell transitions raise ``ValueError``.
    Counts saturate at uint16 maximum.
    """

    counts = _profile_array(profile)
    if isinstance(action, bool):
        raise ValueError("action must be an integer in {0, 1, 2, 3}")
    try:
        action_index = as_index(action)
    except TypeError as exc:
        raise ValueError("action must be an integer in {0, 1, 2, 3}") from exc
    if not 0 <= action_index < 4:
        raise ValueError("action must be an integer in {0, 1, 2, 3}")

    x, y = _grid_position(obs, "obs")
    next_x, next_y = _grid_position(next_obs, "next_obs")
    delta_x, delta_y = next_x - x, next_y - y
    if delta_x == 0 and delta_y == 0:
        return False
    if abs(delta_x) + abs(delta_y) != 1:
        raise ValueError("moving transitions must change position by one cardinal grid cell")

    if delta_y == -1:
        direction = 0  # up
    elif delta_x == 1:
        direction = 1  # right
    elif delta_y == 1:
        direction = 2  # down
    else:
        direction = 3  # left

    band = min(x // 3, 2)
    current = int(counts[band, action_index, direction])
    if current < _MAX_COUNT:
        counts[band, action_index, direction] = current + 1
    return True


def supported_directions(
    profile: object,
    *,
    min_cell_support: int = 2,
) -> np.ndarray:
    """Return supported direction IDs, using ``-1`` for unknown cells.

    A cell is supported when its top direction count reaches
    ``min_cell_support`` and comprises at least 80 percent of that cell's
    observations. The returned array has shape ``(3, 4)`` and dtype int8.
    """

    counts = _profile_array(profile)
    support = _positive_integer(min_cell_support, "min_cell_support", maximum=_MAX_COUNT)
    totals = counts.sum(axis=-1, dtype=np.uint64)
    top_counts = counts.max(axis=-1)
    result = np.full((3, 4), -1, dtype=np.int8)
    observed = totals > 0
    top_directions = counts.argmax(axis=-1)
    purity = np.zeros((3, 4), dtype=np.float64)
    purity[observed] = top_counts[observed] / totals[observed]
    known = (top_counts >= support) & (purity >= _MIN_PURITY)
    result[known] = top_directions[known].astype(np.int8)
    return result


def _positive_integer(value: object, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = as_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1 or (maximum is not None and result > maximum):
        suffix = f" no greater than {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return result


def _probability(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1]") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


class DynamicsGrouper:
    """Assign unlabeled transition profiles to at most two bounded groups."""

    def __init__(
        self,
        max_groups: int = 2,
        min_overlap: int = 4,
        min_agreement: float = 0.6,
        min_cell_support: int = 2,
        novelty_max_agreement: float = 0.3,
    ) -> None:
        if isinstance(max_groups, bool):
            raise ValueError("max_groups must be one or two")
        try:
            group_limit = as_index(max_groups)
        except TypeError as exc:
            raise ValueError("max_groups must be one or two") from exc
        if group_limit not in (1, 2):
            raise ValueError("max_groups must be one or two")
        overlap = _positive_integer(min_overlap, "min_overlap", maximum=12)
        cell_support = _positive_integer(min_cell_support, "min_cell_support", maximum=_MAX_COUNT)
        agreement = _probability(min_agreement, "min_agreement")
        novelty = _probability(novelty_max_agreement, "novelty_max_agreement")
        if novelty >= agreement:
            raise ValueError("novelty_max_agreement must be less than min_agreement")

        self.max_groups = group_limit
        self.min_overlap = overlap
        self.min_agreement = agreement
        self.min_cell_support = cell_support
        self.novelty_max_agreement = novelty
        self._group_counts = np.zeros((group_limit, *PROFILE_SHAPE), dtype=np.uint16)

    @property
    def config(self) -> dict[str, int | float]:
        """Return a plain numeric copy of grouping thresholds."""

        return {
            "max_groups": self.max_groups,
            "min_overlap": self.min_overlap,
            "min_agreement": self.min_agreement,
            "min_cell_support": self.min_cell_support,
            "novelty_max_agreement": self.novelty_max_agreement,
        }

    @property
    def group_counts(self) -> np.ndarray:
        """Return a copy of the fixed-capacity uint16 aggregate tensor."""

        return self._group_counts.copy()

    @property
    def active_groups(self) -> tuple[int, ...]:
        """Return occupied group indices in stable numeric order."""

        return tuple(int(index) for index in np.flatnonzero(self._group_counts.any(axis=(1, 2, 3))))

    @property
    def nbytes(self) -> int:
        """Return aggregate tensor storage bytes, excluding Python metadata."""

        return int(self._group_counts.nbytes)

    @property
    def memory_nbytes(self) -> int:
        """Alias for the numeric aggregate-memory byte count."""

        return self.nbytes

    def _score(self, profile: np.ndarray, group: int) -> tuple[int, float]:
        incoming = supported_directions(profile, min_cell_support=self.min_cell_support)
        aggregate = supported_directions(
            self._group_counts[group], min_cell_support=self.min_cell_support
        )
        overlap_mask = (incoming >= 0) & (aggregate >= 0)
        overlap = int(np.count_nonzero(overlap_mask))
        if overlap == 0:
            return 0, 0.0
        agreements = int(np.count_nonzero(incoming[overlap_mask] == aggregate[overlap_mask]))
        return overlap, agreements / overlap

    def _add_profile(self, group: int, profile: np.ndarray) -> None:
        summed = self._group_counts[group].astype(np.uint32) + profile.astype(np.uint32)
        self._group_counts[group] = np.minimum(summed, _MAX_COUNT).astype(np.uint16)

    def assign(self, profile: object, *, commit: bool = True) -> GroupAssignment:
        """Assign a cumulative profile and optionally add its counts once.

        Groups are chosen by direction agreement only where both profiles have
        supported cells. A unique highest score meeting overlap and agreement
        thresholds matches a group. With free capacity, a new group is created
        only after every active group has enough overlap and low agreement.
        ``commit=True`` adds the supplied counts to the selected aggregate;
        callers must not recommit the same cumulative profile.
        """

        counts = _profile_array(profile)
        if not isinstance(commit, (bool, np.bool_)):
            raise ValueError("commit must be a boolean")
        incoming_directions = supported_directions(
            counts, min_cell_support=self.min_cell_support
        )
        supported_count = int(np.count_nonzero(incoming_directions >= 0))
        active = self.active_groups
        score_rows: dict[int, dict[str, int | float]] = {}
        for group in active:
            overlap, agreement = self._score(counts, group)
            score_rows[group] = {"overlap": overlap, "agreement": agreement}

        if not active:
            if supported_count < _FIRST_GROUP_SUPPORT:
                return GroupAssignment(-1, "insufficient_profile_support", score_rows)
            group = 0
            if commit:
                self._add_profile(group, counts)
            return GroupAssignment(group, "first_group", score_rows)

        qualified = [
            group
            for group, score in score_rows.items()
            if score["overlap"] >= self.min_overlap
            and score["agreement"] >= self.min_agreement
        ]
        if qualified:
            best_agreement = max(float(score_rows[group]["agreement"]) for group in qualified)
            winners = [
                group
                for group in qualified
                if np.isclose(float(score_rows[group]["agreement"]), best_agreement, rtol=0.0, atol=1e-12)
            ]
            if len(winners) != 1:
                return GroupAssignment(-1, "ambiguous_match", score_rows)
            group = winners[0]
            if commit:
                self._add_profile(group, counts)
            return GroupAssignment(group, "matched", score_rows)

        if len(active) < self.max_groups:
            sufficiently_distinct = all(
                score["overlap"] >= self.min_overlap
                and score["agreement"] <= self.novelty_max_agreement
                for score in score_rows.values()
            )
            if sufficiently_distinct:
                group = next(index for index in range(self.max_groups) if index not in active)
                if commit:
                    self._add_profile(group, counts)
                return GroupAssignment(group, "novel_group", score_rows)

            if any(score["overlap"] < self.min_overlap for score in score_rows.values()):
                reason = "insufficient_overlap"
            else:
                reason = "between_match_and_novelty"
            return GroupAssignment(-1, reason, score_rows)

        return GroupAssignment(-1, "max_groups_reached", score_rows)

    def state_dict(self) -> dict[str, Any]:
        """Return plain config, a copied numeric array, and aggregate byte count."""

        return {
            "version": _STATE_VERSION,
            "config": self.config,
            "group_counts": self._group_counts.copy(),
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DynamicsGrouper":
        """Restore a validated grouper from :meth:`state_dict` output."""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported DynamicsGrouper state version")
        config = state.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("state config must be a mapping")
        grouper = cls(**dict(config))
        counts = np.asarray(state.get("group_counts"))
        expected_shape = (grouper.max_groups, *PROFILE_SHAPE)
        if counts.shape != expected_shape or counts.dtype != np.uint16:
            raise ValueError(f"group_counts must be uint16 with shape {expected_shape}")
        if "nbytes" in state and state["nbytes"] != int(counts.nbytes):
            raise ValueError("state nbytes does not match group_counts storage")
        grouper._group_counts[...] = counts
        return grouper


__all__ = [
    "DynamicsGrouper",
    "GroupAssignment",
    "empty_profile",
    "observe_profile",
    "supported_directions",
]

"""Bounded, label-free evidence for shared-write decisions.

The component consumes only agent-visible transitions and rewards. It keeps a
fixed ring of recent events and summarizes reward with online linear-regression
sufficient statistics. Group profiles are separate persistent aggregates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from operator import index as as_index
from typing import Any, NamedTuple

import numpy as np

from .shared_write_grouping import (
    PROFILE_SHAPE,
    empty_profile,
    observe_profile,
    supported_directions,
)


_STATE_VERSION = 1
_FEATURES = 3
_MAX_COUNT = int(np.iinfo(np.uint16).max)
_RIDGE = 1e-4
_MIN_DIRECTION_SUPPORT = 2
_FIRST_GROUP_CELLS = 4
_MIN_OVERLAP = 4
_MIN_AGREEMENT = 0.6
_NOVELTY_MAX_AGREEMENT = 0.3
_MIN_REWARD_COSINE = 0.5
_OPPOSITE_REWARD_COSINE = -0.5


@dataclass(frozen=True)
class EvidenceProfile:
    """Numeric summary returned by :meth:`EvidenceWindow.profile`.

    ``reward_n`` counts all window events, including blocked transitions. The
    regression features are ``[delta_manhattan * 8, 1, terminated]``.
    """

    movement_counts: np.ndarray
    reward_xtx: np.ndarray
    reward_xty: np.ndarray
    reward_n: np.float32

    @property
    def nbytes(self) -> int:
        """Bytes in the numeric NumPy fields, excluding Python metadata."""

        return int(
            self.movement_counts.nbytes
            + self.reward_xtx.nbytes
            + self.reward_xty.nbytes
            + np.asarray(self.reward_n).nbytes
        )

    def coefficients(self) -> np.ndarray:
        """Fit the ridge-regularized reward model from sufficient statistics."""

        return _fit_coefficients(self.reward_xtx, self.reward_xty)


class GroupAssignment(NamedTuple):
    """Assignment result with numeric evidence scores for each active group."""

    group: int
    reason: str
    scores: dict[int, dict[str, int | float]]


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


def _grid_observation(observation: object, name: str) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must contain four finite normalized coordinates in [0, 1]")
    scaled = values * 8.0
    cells = np.rint(scaled).astype(np.int64)
    if not np.allclose(scaled, cells, rtol=0.0, atol=1e-4):
        raise ValueError(f"{name} coordinates must lie on the 9 by 9 grid")
    return cells


def _validate_event(
    obs: object,
    action: object,
    reward: object,
    next_obs: object,
    terminated: object,
) -> tuple[np.ndarray, int, np.float32, np.ndarray, np.uint8]:
    current = _grid_observation(obs, "obs")
    following = _grid_observation(next_obs, "next_obs")
    if isinstance(action, (bool, np.bool_)):
        raise ValueError("action must be an integer in {0, 1, 2, 3}")
    try:
        action_index = as_index(action)
    except TypeError as exc:
        raise ValueError("action must be an integer in {0, 1, 2, 3}") from exc
    if not 0 <= action_index < 4:
        raise ValueError("action must be an integer in {0, 1, 2, 3}")
    try:
        reward_value = float(reward)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reward must be finite and representable as float32") from exc
    if not np.isfinite(reward_value):
        raise ValueError("reward must be finite and representable as float32")
    with np.errstate(over="ignore", invalid="ignore"):
        reward32 = np.float32(reward_value)
    if not np.isfinite(reward32):
        raise ValueError("reward must be finite and representable as float32")
    if not isinstance(terminated, (bool, np.bool_)):
        raise ValueError("terminated must be a boolean")
    return (
        current.astype(np.float32) / np.float32(8.0),
        int(action_index),
        reward32,
        following.astype(np.float32) / np.float32(8.0),
        np.uint8(terminated),
    )


def _reward_features(obs: object, next_obs: object, terminated: bool | np.bool_) -> np.ndarray:
    current = _grid_observation(obs, "obs")
    following = _grid_observation(next_obs, "next_obs")
    current_distance = abs(int(current[2] - current[0])) + abs(int(current[3] - current[1]))
    next_distance = abs(int(following[2] - following[0])) + abs(int(following[3] - following[1]))
    delta_distance = (current_distance - next_distance) * 8.0
    return np.asarray((delta_distance, 1.0, float(terminated)), dtype=np.float64)


def _fit_coefficients(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    matrix = np.asarray(xtx, dtype=np.float64)
    vector = np.asarray(xty, dtype=np.float64)
    coefficients = np.linalg.solve(matrix + _RIDGE * np.eye(_FEATURES), vector)
    return coefficients.astype(np.float32)


def _profile_fields(profile: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    if isinstance(profile, EvidenceProfile):
        movement_counts = profile.movement_counts
        reward_xtx = profile.reward_xtx
        reward_xty = profile.reward_xty
        reward_n_value = profile.reward_n
    elif isinstance(profile, Mapping):
        try:
            movement_counts = profile["movement_counts"]
            reward_xtx = profile["reward_xtx"]
            reward_xty = profile["reward_xty"]
            reward_n_value = profile["reward_n"]
        except KeyError as exc:
            raise ValueError("profile is missing a numeric evidence field") from exc
    else:
        raise TypeError("profile must be an EvidenceProfile or numeric mapping")

    counts = np.asarray(movement_counts)
    xtx = np.asarray(reward_xtx)
    xty = np.asarray(reward_xty)
    n_array = np.asarray(reward_n_value)
    if counts.shape != PROFILE_SHAPE or counts.dtype != np.uint16:
        raise ValueError(f"movement_counts must be uint16 with shape {PROFILE_SHAPE}")
    if xtx.shape != (_FEATURES, _FEATURES) or xtx.dtype != np.float32:
        raise ValueError("reward_xtx must be float32 with shape (3, 3)")
    if xty.shape != (_FEATURES,) or xty.dtype != np.float32:
        raise ValueError("reward_xty must be float32 with shape (3,)")
    if n_array.shape != () or n_array.dtype != np.float32:
        raise ValueError("reward_n must be a float32 scalar")
    reward_n = np.float32(n_array)
    if not np.isfinite(xtx).all() or not np.isfinite(xty).all() or not np.isfinite(reward_n):
        raise ValueError("profile regression statistics must be finite")
    if not 0.0 <= float(reward_n) <= _MAX_COUNT:
        raise ValueError("profile reward_n must be in [0, 65535]")
    if not np.allclose(xtx, xtx.T, rtol=0.0, atol=1e-4):
        raise ValueError("profile reward_xtx must be symmetric")
    return counts, xtx, xty, reward_n


class EvidenceWindow:
    """A fixed-capacity ring buffer of visible transition and reward events."""

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = _positive_integer(capacity, "capacity", maximum=_MAX_COUNT)
        self._obs = np.zeros((self.capacity, 4), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.uint8)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, 4), dtype=np.float32)
        self._terminated = np.zeros(self.capacity, dtype=np.uint8)
        self._size = 0
        self._next_index = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        """Bytes held by the fixed numeric ring arrays, excluding Python."""

        return int(
            self._obs.nbytes
            + self._actions.nbytes
            + self._rewards.nbytes
            + self._next_obs.nbytes
            + self._terminated.nbytes
        )

    def observe(
        self,
        obs: object,
        action: int,
        reward: float,
        next_obs: object,
        terminated: bool,
    ) -> None:
        """Append one validated event, overwriting the oldest when full."""

        current, action_index, reward32, following, done = _validate_event(
            obs, action, reward, next_obs, terminated
        )
        index = self._next_index
        self._obs[index] = current
        self._actions[index] = action_index
        self._rewards[index] = reward32
        self._next_obs[index] = following
        self._terminated[index] = done
        self._next_index = (index + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def profile(self) -> EvidenceProfile:
        """Recompute bounded movement and reward statistics from the ring."""

        movement_counts = empty_profile()
        xtx64 = np.zeros((_FEATURES, _FEATURES), dtype=np.float64)
        xty64 = np.zeros(_FEATURES, dtype=np.float64)
        start = (self._next_index - self._size) % self.capacity
        for offset in range(self._size):
            index = (start + offset) % self.capacity
            obs = self._obs[index]
            next_obs = self._next_obs[index]
            observe_profile(movement_counts, obs, int(self._actions[index]), next_obs)
            features = _reward_features(obs, next_obs, bool(self._terminated[index]))
            reward = float(self._rewards[index])
            xtx64 += np.outer(features, features)
            xty64 += features * reward

        with np.errstate(over="ignore", invalid="ignore"):
            reward_xtx = xtx64.astype(np.float32)
            reward_xty = xty64.astype(np.float32)
        if not np.isfinite(reward_xtx).all() or not np.isfinite(reward_xty).all():
            raise ValueError("window reward statistics exceed finite float32 range")
        return EvidenceProfile(
            movement_counts=movement_counts,
            reward_xtx=reward_xtx,
            reward_xty=reward_xty,
            reward_n=np.float32(self._size),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a copyable, torch.save-compatible representation."""

        return {
            "version": _STATE_VERSION,
            "capacity": self.capacity,
            "size": self._size,
            "next_index": self._next_index,
            "obs": self._obs.copy(),
            "actions": self._actions.copy(),
            "rewards": self._rewards.copy(),
            "next_obs": self._next_obs.copy(),
            "terminated": self._terminated.copy(),
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EvidenceWindow":
        """Restore a window after validating numeric shapes, dtypes, and values."""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported EvidenceWindow state version")
        window = cls(state.get("capacity"))
        size, next_index = state.get("size"), state.get("next_index")
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise ValueError("state size must be an integer")
        if isinstance(next_index, bool) or not isinstance(next_index, (int, np.integer)):
            raise ValueError("state next_index must be an integer")
        if not 0 <= int(size) <= window.capacity:
            raise ValueError("state size is outside the window capacity")
        if not 0 <= int(next_index) < window.capacity:
            raise ValueError("state next_index is outside the window capacity")
        if int(size) < window.capacity and int(next_index) != int(size):
            raise ValueError("partially filled state next_index must equal size")

        specifications = (
            ("obs", (window.capacity, 4), np.dtype(np.float32)),
            ("actions", (window.capacity,), np.dtype(np.uint8)),
            ("rewards", (window.capacity,), np.dtype(np.float32)),
            ("next_obs", (window.capacity, 4), np.dtype(np.float32)),
            ("terminated", (window.capacity,), np.dtype(np.uint8)),
        )
        restored: dict[str, np.ndarray] = {}
        for name, shape, dtype in specifications:
            array = np.asarray(state.get(name))
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"state {name} must have dtype {dtype} and shape {shape}")
            restored[name] = array
        if not np.isfinite(restored["obs"]).all() or not np.isfinite(restored["next_obs"]).all():
            raise ValueError("state observations must be finite")
        if not np.isfinite(restored["rewards"]).all():
            raise ValueError("state rewards must be finite")
        if np.any(restored["actions"] > 3) or np.any(restored["terminated"] > 1):
            raise ValueError("state action or terminated values are out of range")
        for index in range(int(size)):
            _grid_observation(restored["obs"][index], "state obs")
            _grid_observation(restored["next_obs"][index], "state next_obs")
        expected_nbytes = sum(array.nbytes for array in restored.values())
        if "nbytes" in state and state["nbytes"] != expected_nbytes:
            raise ValueError("state nbytes does not match ring storage")

        window._obs[...] = restored["obs"]
        window._actions[...] = restored["actions"]
        window._rewards[...] = restored["rewards"]
        window._next_obs[...] = restored["next_obs"]
        window._terminated[...] = restored["terminated"]
        window._size = int(size)
        window._next_index = int(next_index)
        return window


def _reward_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first64) * np.linalg.norm(second64))
    if denominator <= 1e-12:
        return 0.0
    value = float(np.dot(first64, second64) / denominator)
    return float(np.clip(value, -1.0, 1.0))


class RewardAwareGroups:
    """Assign evidence to at most two groups using dynamics and reward fit."""

    def __init__(
        self,
        max_groups: int = 2,
        min_support: int = 8,
        reward_min_samples: int = 16,
        mode: str = "reward",
    ) -> None:
        if isinstance(max_groups, bool):
            raise ValueError("max_groups must be one or two")
        try:
            group_limit = as_index(max_groups)
        except TypeError as exc:
            raise ValueError("max_groups must be one or two") from exc
        if group_limit not in (1, 2):
            raise ValueError("max_groups must be one or two")
        self.min_support = _positive_integer(min_support, "min_support", maximum=_MAX_COUNT)
        self.reward_min_samples = _positive_integer(
            reward_min_samples, "reward_min_samples", maximum=_MAX_COUNT
        )
        if mode not in ("reward", "dynamics"):
            raise ValueError("mode must be 'reward' or 'dynamics'")
        self.max_groups = int(group_limit)
        self.mode = mode
        self._movement_counts = np.zeros((self.max_groups, *PROFILE_SHAPE), dtype=np.uint16)
        self._reward_xtx = np.zeros((self.max_groups, _FEATURES, _FEATURES), dtype=np.float32)
        self._reward_xty = np.zeros((self.max_groups, _FEATURES), dtype=np.float32)
        self._reward_n = np.zeros(self.max_groups, dtype=np.float32)
        self._active = np.zeros(self.max_groups, dtype=np.uint8)

    @property
    def config(self) -> dict[str, int | str]:
        return {
            "max_groups": self.max_groups,
            "min_support": self.min_support,
            "reward_min_samples": self.reward_min_samples,
            "mode": self.mode,
        }

    @property
    def active_groups(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self._active))

    @property
    def group_counts(self) -> np.ndarray:
        return self._movement_counts.copy()

    @property
    def nbytes(self) -> int:
        """Bytes held by aggregate numeric arrays, excluding Python metadata."""

        return int(
            self._movement_counts.nbytes
            + self._reward_xtx.nbytes
            + self._reward_xty.nbytes
            + self._reward_n.nbytes
            + self._active.nbytes
        )

    def _score_dynamics(self, counts: np.ndarray, group: int) -> tuple[int, float]:
        incoming = supported_directions(counts, min_cell_support=_MIN_DIRECTION_SUPPORT)
        aggregate = supported_directions(
            self._movement_counts[group], min_cell_support=_MIN_DIRECTION_SUPPORT
        )
        overlap_mask = (incoming >= 0) & (aggregate >= 0)
        overlap = int(np.count_nonzero(overlap_mask))
        if overlap == 0:
            return 0, 0.0
        agreement = int(np.count_nonzero(incoming[overlap_mask] == aggregate[overlap_mask]))
        return overlap, agreement / overlap

    def _add_profile(
        self,
        group: int,
        counts: np.ndarray,
        xtx: np.ndarray,
        xty: np.ndarray,
        reward_n: np.float32,
    ) -> None:
        total_counts = self._movement_counts[group].astype(np.uint32) + counts.astype(np.uint32)
        self._movement_counts[group] = np.minimum(total_counts, _MAX_COUNT).astype(np.uint16)

        old_xtx = self._reward_xtx[group].astype(np.float64)
        old_xty = self._reward_xty[group].astype(np.float64)
        incoming_xtx = xtx.astype(np.float64, copy=True)
        incoming_xty = xty.astype(np.float64, copy=True)
        old_n = float(self._reward_n[group])
        incoming_n = float(reward_n)
        while old_n + incoming_n > _MAX_COUNT:
            old_xtx *= 0.5
            old_xty *= 0.5
            incoming_xtx *= 0.5
            incoming_xty *= 0.5
            old_n *= 0.5
            incoming_n *= 0.5

        with np.errstate(over="ignore", invalid="ignore"):
            self._reward_xtx[group] = (old_xtx + incoming_xtx).astype(np.float32)
            self._reward_xty[group] = (old_xty + incoming_xty).astype(np.float32)
        self._reward_n[group] = np.float32(old_n + incoming_n)
        if (
            not np.isfinite(self._reward_xtx[group]).all()
            or not np.isfinite(self._reward_xty[group]).all()
            or not np.isfinite(self._reward_n[group])
        ):
            raise ValueError("aggregate reward statistics exceed finite float32 range")
        self._active[group] = 1

    def assign(self, profile: object, *, commit: bool = True) -> GroupAssignment:
        """Assign or abstain based on observed movement and conditional reward.

        A successful commit adds the supplied profile once. Recommitting a
        cumulative profile repeats its evidence; callers should commit at
        explicit window boundaries, with resets controlled by the runner.
        """

        if not isinstance(commit, (bool, np.bool_)):
            raise ValueError("commit must be a boolean")
        counts, xtx, xty, reward_n = _profile_fields(profile)
        incoming_directions = supported_directions(
            counts, min_cell_support=_MIN_DIRECTION_SUPPORT
        )
        supported_count = int(np.count_nonzero(incoming_directions >= 0))
        movement_n = int(counts.sum(dtype=np.uint64))
        reward_supported = float(reward_n) >= self.reward_min_samples
        score_rows: dict[int, dict[str, int | float]] = {}
        incoming_coef: np.ndarray | None = None
        if reward_supported:
            incoming_coef = _fit_coefficients(xtx, xty)

        for group in self.active_groups:
            overlap, agreement = self._score_dynamics(counts, group)
            group_reward_supported = float(self._reward_n[group]) >= self.reward_min_samples
            cosine = 0.0
            if reward_supported and group_reward_supported and incoming_coef is not None:
                group_coef = _fit_coefficients(self._reward_xtx[group], self._reward_xty[group])
                cosine = _reward_cosine(incoming_coef, group_coef)
            score_rows[group] = {
                "overlap": overlap,
                "agreement": agreement,
                "reward_cosine": cosine,
                "reward_supported": int(reward_supported and group_reward_supported),
            }

        if movement_n < self.min_support or supported_count < _FIRST_GROUP_CELLS:
            return GroupAssignment(-1, "insufficient_profile_support", score_rows)
        if self.mode == "reward" and not reward_supported:
            return GroupAssignment(-1, "insufficient_reward_support", score_rows)

        active = self.active_groups
        if not active:
            if commit:
                self._add_profile(0, counts, xtx, xty, reward_n)
            return GroupAssignment(0, "first_group", score_rows)

        qualified: list[int] = []
        for group, score in score_rows.items():
            dynamics_match = (
                score["overlap"] >= _MIN_OVERLAP
                and score["agreement"] >= _MIN_AGREEMENT
            )
            reward_match = (
                self.mode == "dynamics"
                or (
                    score["reward_supported"]
                    and score["reward_cosine"] >= _MIN_REWARD_COSINE
                )
            )
            if dynamics_match and reward_match:
                qualified.append(group)

        if qualified:
            best_key = max(
                (
                    float(score_rows[group]["agreement"]),
                    float(score_rows[group]["reward_cosine"])
                    if self.mode == "reward"
                    else 0.0,
                )
                for group in qualified
            )
            winners = [
                group
                for group in qualified
                if np.isclose(
                    float(score_rows[group]["agreement"]), best_key[0], rtol=0.0, atol=1e-12
                )
                and np.isclose(
                    float(score_rows[group]["reward_cosine"])
                    if self.mode == "reward"
                    else 0.0,
                    best_key[1],
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
            if len(winners) != 1:
                return GroupAssignment(-1, "ambiguous_match", score_rows)
            group = winners[0]
            if commit:
                self._add_profile(group, counts, xtx, xty, reward_n)
            return GroupAssignment(group, "matched", score_rows)

        distinct_from_all = True
        insufficient_overlap = False
        between_thresholds = False
        reward_mismatch = False
        for score in score_rows.values():
            overlap = int(score["overlap"])
            agreement = float(score["agreement"])
            if overlap < _MIN_OVERLAP:
                insufficient_overlap = True
                distinct_from_all = False
                continue
            dynamics_novel = agreement <= _NOVELTY_MAX_AGREEMENT
            reward_opposite = (
                self.mode == "reward"
                and bool(score["reward_supported"])
                and agreement >= _MIN_AGREEMENT
                and float(score["reward_cosine"]) <= _OPPOSITE_REWARD_COSINE
            )
            if not (dynamics_novel or reward_opposite):
                distinct_from_all = False
                if self.mode == "reward" and agreement >= _MIN_AGREEMENT:
                    reward_mismatch = True
                elif agreement > _NOVELTY_MAX_AGREEMENT:
                    between_thresholds = True

        if distinct_from_all:
            if len(active) < self.max_groups:
                group = next(index for index in range(self.max_groups) if index not in active)
                if commit:
                    self._add_profile(group, counts, xtx, xty, reward_n)
                reason = "opposite_reward_group" if any(
                    self.mode == "reward"
                    and score["overlap"] >= _MIN_OVERLAP
                    and score["agreement"] >= _MIN_AGREEMENT
                    and score["reward_supported"]
                    and score["reward_cosine"] <= _OPPOSITE_REWARD_COSINE
                    for score in score_rows.values()
                ) else "novel_group"
                return GroupAssignment(group, reason, score_rows)
            return GroupAssignment(-1, "max_groups_reached", score_rows)

        if insufficient_overlap:
            return GroupAssignment(-1, "insufficient_overlap", score_rows)
        if reward_mismatch:
            return GroupAssignment(-1, "reward_mismatch", score_rows)
        if between_thresholds:
            return GroupAssignment(-1, "between_match_and_novelty", score_rows)
        return GroupAssignment(-1, "ambiguous_evidence", score_rows)

    def state_dict(self) -> dict[str, Any]:
        """Return validated numeric aggregates and configuration."""

        return {
            "version": _STATE_VERSION,
            "config": self.config,
            "movement_counts": self._movement_counts.copy(),
            "reward_xtx": self._reward_xtx.copy(),
            "reward_xty": self._reward_xty.copy(),
            "reward_n": self._reward_n.copy(),
            "active": self._active.copy(),
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RewardAwareGroups":
        """Restore aggregate arrays after strict shape and value validation."""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported RewardAwareGroups state version")
        config = state.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("state config must be a mapping")
        groups = cls(**dict(config))
        specifications = (
            ("movement_counts", (groups.max_groups, *PROFILE_SHAPE), np.dtype(np.uint16)),
            ("reward_xtx", (groups.max_groups, _FEATURES, _FEATURES), np.dtype(np.float32)),
            ("reward_xty", (groups.max_groups, _FEATURES), np.dtype(np.float32)),
            ("reward_n", (groups.max_groups,), np.dtype(np.float32)),
            ("active", (groups.max_groups,), np.dtype(np.uint8)),
        )
        restored: dict[str, np.ndarray] = {}
        for name, shape, dtype in specifications:
            array = np.asarray(state.get(name))
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"state {name} must have dtype {dtype} and shape {shape}")
            restored[name] = array
        if not np.isfinite(restored["reward_xtx"]).all():
            raise ValueError("state reward_xtx must be finite")
        if not np.isfinite(restored["reward_xty"]).all():
            raise ValueError("state reward_xty must be finite")
        if not np.isfinite(restored["reward_n"]).all():
            raise ValueError("state reward_n must be finite")
        if np.any(restored["reward_n"] < 0.0) or np.any(restored["reward_n"] > _MAX_COUNT):
            raise ValueError("state reward_n must be in [0, 65535]")
        if np.any(restored["active"] > 1):
            raise ValueError("state active values must be zero or one")
        if not np.allclose(restored["reward_xtx"], restored["reward_xtx"].transpose(0, 2, 1), rtol=0.0, atol=1e-4):
            raise ValueError("state reward_xtx arrays must be symmetric")
        for group in range(groups.max_groups):
            has_evidence = bool(
                restored["movement_counts"][group].any() or restored["reward_n"][group] > 0
            )
            if bool(restored["active"][group]) != has_evidence:
                raise ValueError("state active flags do not match aggregate evidence")

        expected_nbytes = sum(array.nbytes for array in restored.values())
        if "nbytes" in state and state["nbytes"] != expected_nbytes:
            raise ValueError("state nbytes does not match aggregate storage")
        groups._movement_counts[...] = restored["movement_counts"]
        groups._reward_xtx[...] = restored["reward_xtx"]
        groups._reward_xty[...] = restored["reward_xty"]
        groups._reward_n[...] = restored["reward_n"]
        groups._active[...] = restored["active"]
        return groups


__all__ = ["EvidenceProfile", "EvidenceWindow", "GroupAssignment", "RewardAwareGroups"]

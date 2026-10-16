"""Bounded, label-free routing for reward-aware write-decision streams.

The router consumes only observations, actions, rewards, next observations,
and termination flags. It is a conservative diagnostic for the known 9 by 9
C1 navigation interface, not a general POMDP solution. Its reward model uses
Manhattan-distance change; the detour generator's wall-aware shaping distance
is therefore misspecified and may produce residual alarms.
"""

from __future__ import annotations

from collections.abc import Mapping
from operator import index as as_index
from typing import Any, NamedTuple

import numpy as np

from .shared_write_grouping import empty_profile, observe_profile, supported_directions


_STATE_VERSION = 1
_FEATURES = 3
_RIDGE = 1e-4
_MIN_CELL_SUPPORT = 2
_MIN_OVERLAP = 2
_MIN_MOVING_EVENTS = 4
_MIN_SUPPORTED_CELLS = 2
_MIN_WINDOW_REWARDS = 8
_MIN_STABLE_REWARDS = 16
_MIN_REWARD_COSINE = 0.5
_RELIABLE_OPPOSITE_COSINE = -0.5
_REWARD_RESIDUAL = 0.15
_MAX_PROFILE_COUNT = int(np.iinfo(np.uint16).max)
_MAX_REWARD_COUNT = int(np.iinfo(np.uint16).max)
_DIRECTION_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_UINT64_MASK = (1 << 64) - 1


class _WindowProfile(NamedTuple):
    counts: np.ndarray
    reward_xtx: np.ndarray
    reward_xty: np.ndarray
    reward_n: int
    moving_events: int
    supported_cells: int


def _positive_integer(value: object, name: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = as_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or (maximum is not None and result > maximum):
        suffix = f" in [{minimum}, {maximum}]" if maximum is not None else f" at least {minimum}"
        raise ValueError(f"{name} must be an integer {suffix}")
    return int(result)


def _grid_observation(value: object, name: str) -> np.ndarray:
    observation = np.asarray(value, dtype=np.float64)
    if observation.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    if not np.isfinite(observation).all() or np.any(observation < 0.0) or np.any(observation > 1.0):
        raise ValueError(f"{name} must contain four finite normalized coordinates in [0, 1]")
    scaled = observation * 8.0
    cells = np.rint(scaled).astype(np.int64)
    if not np.allclose(scaled, cells, rtol=0.0, atol=1e-4):
        raise ValueError(f"{name} coordinates must lie on the 9 by 9 grid")
    return cells


def _event(
    obs: object,
    action: object,
    reward: object,
    next_obs: object,
    terminated: object,
) -> tuple[np.ndarray, int, np.float32, np.ndarray, np.uint8]:
    current_cells = _grid_observation(obs, "obs")
    next_cells = _grid_observation(next_obs, "next_obs")
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
    with np.errstate(over="ignore", invalid="ignore"):
        reward32 = np.float32(reward_value)
    if not np.isfinite(reward_value) or not np.isfinite(reward32):
        raise ValueError("reward must be finite and representable as float32")
    if not isinstance(terminated, (bool, np.bool_)):
        raise ValueError("terminated must be a boolean")
    return (
        (current_cells.astype(np.float32) / np.float32(8.0)),
        int(action_index),
        reward32,
        (next_cells.astype(np.float32) / np.float32(8.0)),
        np.uint8(terminated),
    )


def _reward_features(obs: object, next_obs: object, terminated: bool | np.bool_) -> np.ndarray:
    current = _grid_observation(obs, "obs")
    following = _grid_observation(next_obs, "next_obs")
    before_distance = abs(int(current[2] - current[0])) + abs(int(current[3] - current[1]))
    after_distance = abs(int(following[2] - following[0])) + abs(int(following[3] - following[1]))
    return np.asarray((before_distance - after_distance, 1.0, float(terminated)), dtype=np.float64)


def _fit_coefficients(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    return np.linalg.solve(xtx + _RIDGE * np.eye(_FEATURES), xty)


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _rng_words(generator: np.random.Generator) -> np.ndarray:
    state = generator.bit_generator.state
    pcg = state["state"]
    words = np.zeros(6, dtype=np.uint64)
    for offset, key in ((0, "state"), (2, "inc")):
        value = int(pcg[key])
        words[offset] = np.uint64(value & _UINT64_MASK)
        words[offset + 1] = np.uint64((value >> 64) & _UINT64_MASK)
    words[4] = np.uint64(state.get("has_uint32", 0))
    words[5] = np.uint64(state.get("uinteger", 0))
    return words


def _restore_rng(words: np.ndarray) -> np.random.Generator:
    generator = np.random.default_rng(0)
    state = generator.bit_generator.state
    state["state"]["state"] = int(words[0]) | (int(words[1]) << 64)
    state["state"]["inc"] = int(words[2]) | (int(words[3]) << 64)
    state["has_uint32"] = int(words[4])
    state["uinteger"] = int(words[5])
    generator.bit_generator.state = state
    return generator


class WriteDecisionRouter:
    """Maintain at most six or seven anonymous profiles for a visible stream.

    A route remains unknown until a bounded transition window contains at
    least four movements, two distinct supported band/action cells, and eight
    reward samples. Existing profiles match only with at least two supported
    overlapping cells, no direction conflicts, and reward coefficient cosine
    at least 0.5. Reward residual alarms are enabled only after 16 samples.
    """

    def __init__(
        self,
        max_slots: int = 6,
        window: int = 128,
        eviction: str = "lru",
        seed: int = 0,
    ) -> None:
        if isinstance(max_slots, (bool, np.bool_)):
            raise ValueError("max_slots must be 6 or 7")
        try:
            slot_limit = as_index(max_slots)
        except TypeError as exc:
            raise ValueError("max_slots must be 6 or 7") from exc
        if slot_limit not in (6, 7):
            raise ValueError("max_slots must be 6 or 7")
        self.max_slots = int(slot_limit)
        self.window = _positive_integer(window, "window", maximum=_MAX_PROFILE_COUNT)
        if eviction not in ("lru", "random", "lfu"):
            raise ValueError("eviction must be 'lru', 'random', or 'lfu'")
        if isinstance(seed, (bool, np.bool_)):
            raise ValueError("seed must be a nonnegative integer")
        try:
            seed_value = as_index(seed)
        except TypeError as exc:
            raise ValueError("seed must be a nonnegative integer") from exc
        if seed_value < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.eviction = eviction

        self._counts = np.zeros((self.max_slots, 3, 4, 4), dtype=np.uint16)
        self._reward_xtx = np.zeros((self.max_slots, _FEATURES, _FEATURES), dtype=np.float64)
        self._reward_xty = np.zeros((self.max_slots, _FEATURES), dtype=np.float64)
        self._reward_n = np.zeros(self.max_slots, dtype=np.uint32)
        self._active = np.zeros(self.max_slots, dtype=np.uint8)
        self._last_used = np.zeros(self.max_slots, dtype=np.int64)
        self._frequency = np.zeros(self.max_slots, dtype=np.uint64)

        self._history_obs = np.zeros((self.window, 4), dtype=np.float32)
        self._history_actions = np.zeros(self.window, dtype=np.uint8)
        self._history_rewards = np.zeros(self.window, dtype=np.float32)
        self._history_next_obs = np.zeros((self.window, 4), dtype=np.float32)
        self._history_terminated = np.zeros(self.window, dtype=np.uint8)
        self._history_size = 0
        self._history_next = 0

        self._rng = np.random.default_rng(int(seed_value))
        self._rng_state = _rng_words(self._rng)
        self._clock = 0
        self._active_slot = -1
        self._allocations = 0
        self._evictions = 0
        self._switches = 0
        self._last_reason = "insufficient_support"

    @property
    def active_slots(self) -> tuple[int, ...]:
        """Return occupied numeric slots, without any external labels."""

        return tuple(int(slot) for slot in np.flatnonzero(self._active))

    @property
    def active_slot(self) -> int:
        """Return the current route, or ``-1`` while evidence is unresolved."""

        return int(self._active_slot)

    @property
    def history_size(self) -> int:
        return int(self._history_size)

    @property
    def nbytes(self) -> int:
        """Actual NumPy storage for profiles, reward stats, history, and RNG."""

        arrays = (
            self._counts,
            self._reward_xtx,
            self._reward_xty,
            self._reward_n,
            self._active,
            self._last_used,
            self._frequency,
            self._history_obs,
            self._history_actions,
            self._history_rewards,
            self._history_next_obs,
            self._history_terminated,
            self._rng_state,
        )
        return int(sum(array.nbytes for array in arrays))

    def route(self, obs: object) -> dict[str, int | float | str]:
        """Choose the current profile before acting, abstaining while unknown."""

        cells = _grid_observation(obs, "obs")
        del cells  # Validation only: routing never reads a task marker or map.
        if self._active_slot < 0:
            return {"slot": -1, "reason": self._last_reason, "confidence": 0.0}
        slot = self._active_slot
        known_cells = int(np.count_nonzero(supported_directions(self._counts[slot]) >= 0))
        movement_confidence = min(1.0, known_cells / float(_MIN_SUPPORTED_CELLS))
        reward_confidence = min(1.0, float(self._reward_n[slot]) / _MIN_STABLE_REWARDS)
        confidence = float(movement_confidence * reward_confidence)
        return {"slot": slot, "reason": self._last_reason, "confidence": confidence}

    def profile_for_slot(self, slot: object) -> dict[str, np.ndarray | np.float32]:
        """Return a copy of one occupied slot's numeric evidence summary.

        The mapping uses ``RewardAwareGroups`` field names and contains no
        task, context, or observer labels. Returned arrays are independent
        copies, so a grouping consumer cannot mutate the router's profile.
        """

        slot_index = self._slot_index(slot)
        if not self._active[slot_index]:
            raise ValueError("slot has no profile")
        xtx = self._reward_xtx[slot_index]
        xty = self._reward_xty[slot_index]
        scale = max(
            1.0,
            float(np.max(np.abs(xtx))) / np.finfo(np.float32).max,
            float(np.max(np.abs(xty))) / np.finfo(np.float32).max,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            xtx32 = (xtx / scale).astype(np.float32)
            xty32 = (xty / scale).astype(np.float32)
        if not np.isfinite(xtx32).all() or not np.isfinite(xty32).all():
            raise ValueError("slot reward statistics cannot be represented as float32")
        return {
            "movement_counts": self._counts[slot_index].copy(),
            "reward_xtx": xtx32,
            "reward_xty": xty32,
            "reward_n": np.float32(self._reward_n[slot_index]),
        }

    def clear_recent(self) -> None:
        """Clear the bounded event history and current route, keeping profiles."""

        self._clear_history()
        self._active_slot = -1
        self._last_reason = "insufficient_support"

    def observe(
        self,
        obs: object,
        action: object,
        reward: object,
        next_obs: object,
        terminated: object,
    ) -> dict[str, int | float | str | bool | None]:
        """Learn one visible event and report assignment or routing changes."""

        current, action_index, reward_value, following, done = _event(
            obs, action, reward, next_obs, terminated
        )
        self._clock += 1
        previous_slot = self._active_slot
        allocated_slot: int | None = None
        evicted_slot: int | None = None

        conflict = self._conflict_reason(
            self._active_slot, current, action_index, reward_value, following, bool(done)
        ) if self._active_slot >= 0 else None

        if conflict is not None:
            self._active_slot = -1
            self._clear_history()
            self._append_history(current, action_index, reward_value, following, done)
            self._last_reason = conflict
        elif self._active_slot >= 0:
            slot = self._active_slot
            self._add_event_to_slot(slot, current, action_index, reward_value, following, bool(done))
            self._append_history(current, action_index, reward_value, following, done)
            self._last_used[slot] = self._clock
            self._frequency[slot] += np.uint64(1)
            self._last_reason = "active_profile"
        else:
            self._append_history(current, action_index, reward_value, following, done)
            profile = self._history_profile()
            if self._window_ready(profile):
                decision = self._assign_window(profile)
                self._last_reason = decision["reason"]
                if decision["slot"] >= 0:
                    slot = int(decision["slot"])
                    allocated_slot = decision["allocated_slot"]
                    evicted_slot = decision["evicted_slot"]
                    self._commit_profile(slot, profile)
                    self._active_slot = slot
                    self._last_used[slot] = self._clock
                    self._frequency[slot] += np.uint64(profile.reward_n)
                    self._clear_history()
            else:
                self._last_reason = "insufficient_support"

        switched = self._active_slot != previous_slot
        if switched:
            self._switches += 1
        if allocated_slot is not None:
            self._allocations += 1
        if evicted_slot is not None:
            self._evictions += 1
        return {
            "allocated_slot": allocated_slot,
            "evicted_slot": evicted_slot,
            "active_slot": self._active_slot,
            "switched": switched,
            "reason": self._last_reason,
            "confidence": float(self.route(current)["confidence"]),
        }

    def _conflict_reason(
        self,
        slot: int,
        obs: np.ndarray,
        action: int,
        reward: np.float32,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> str | None:
        direction = self._observed_direction(obs, next_obs)
        if direction is not None:
            x = int(round(float(obs[0]) * 8.0))
            prediction = supported_directions(self._counts[slot], min_cell_support=_MIN_CELL_SUPPORT)
            expected = int(prediction[min(x // 3, 2), action])
            if expected >= 0 and expected != direction:
                return "dynamics_conflict"

        if int(self._reward_n[slot]) >= _MIN_STABLE_REWARDS:
            features = _reward_features(obs, next_obs, terminated)
            coefficients = _fit_coefficients(self._reward_xtx[slot], self._reward_xty[slot])
            residual = float(reward) - float(np.dot(features, coefficients))
            if abs(residual) > _REWARD_RESIDUAL:
                return "reward_residual"
        return None

    @staticmethod
    def _observed_direction(obs: np.ndarray, next_obs: np.ndarray) -> int | None:
        current = _grid_observation(obs, "obs")
        following = _grid_observation(next_obs, "next_obs")
        dx, dy = int(following[0] - current[0]), int(following[1] - current[1])
        if dx == 0 and dy == 0:
            return None
        if abs(dx) + abs(dy) != 1:
            raise ValueError("moving transitions must change position by one cardinal grid cell")
        if dy == -1:
            return 0
        if dx == 1:
            return 1
        if dy == 1:
            return 2
        return 3

    def _slot_index(self, value: object) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("slot must be an integer within configured capacity")
        try:
            slot = as_index(value)
        except TypeError as exc:
            raise ValueError("slot must be an integer within configured capacity") from exc
        if not 0 <= slot < self.max_slots:
            raise ValueError("slot must be an integer within configured capacity")
        return int(slot)

    def _append_history(
        self,
        obs: np.ndarray,
        action: int,
        reward: np.float32,
        next_obs: np.ndarray,
        terminated: np.uint8,
    ) -> None:
        index = self._history_next
        self._history_obs[index] = obs
        self._history_actions[index] = action
        self._history_rewards[index] = reward
        self._history_next_obs[index] = next_obs
        self._history_terminated[index] = terminated
        self._history_next = (index + 1) % self.window
        self._history_size = min(self.window, self._history_size + 1)

    def _clear_history(self) -> None:
        self._history_size = 0
        self._history_next = 0
        self._history_obs.fill(0.0)
        self._history_actions.fill(0)
        self._history_rewards.fill(0.0)
        self._history_next_obs.fill(0.0)
        self._history_terminated.fill(0)

    def _history_indices(self) -> range | list[int]:
        start = (self._history_next - self._history_size) % self.window
        return [(start + offset) % self.window for offset in range(self._history_size)]

    def _history_profile(self) -> _WindowProfile:
        counts = empty_profile()
        reward_xtx = np.zeros((_FEATURES, _FEATURES), dtype=np.float64)
        reward_xty = np.zeros(_FEATURES, dtype=np.float64)
        moving_events = 0
        for index in self._history_indices():
            obs = self._history_obs[index]
            next_obs = self._history_next_obs[index]
            moved = observe_profile(counts, obs, int(self._history_actions[index]), next_obs)
            moving_events += int(moved)
            features = _reward_features(obs, next_obs, bool(self._history_terminated[index]))
            reward = float(self._history_rewards[index])
            reward_xtx += np.outer(features, features)
            reward_xty += features * reward
        supported = supported_directions(counts, min_cell_support=_MIN_CELL_SUPPORT)
        return _WindowProfile(
            counts,
            reward_xtx,
            reward_xty,
            self._history_size,
            moving_events,
            int(np.count_nonzero(supported >= 0)),
        )

    @staticmethod
    def _window_ready(profile: _WindowProfile) -> bool:
        return (
            profile.moving_events >= _MIN_MOVING_EVENTS
            and profile.supported_cells >= _MIN_SUPPORTED_CELLS
            and profile.reward_n >= _MIN_WINDOW_REWARDS
        )

    def _reward_cosine(self, first: _WindowProfile, slot: int) -> float | None:
        if first.reward_n < _MIN_WINDOW_REWARDS or int(self._reward_n[slot]) < _MIN_WINDOW_REWARDS:
            return None
        incoming = _fit_coefficients(first.reward_xtx, first.reward_xty)
        existing = _fit_coefficients(self._reward_xtx[slot], self._reward_xty[slot])
        return _cosine(incoming, existing)

    def _assign_window(self, profile: _WindowProfile) -> dict[str, Any]:
        active = self.active_slots
        if not active:
            slot, evicted = self._allocate_slot()
            return {"slot": slot, "allocated_slot": slot, "evicted_slot": evicted, "reason": "first_profile"}

        incoming = supported_directions(profile.counts, min_cell_support=_MIN_CELL_SUPPORT)
        scores: dict[int, tuple[int, float]] = {}
        distinctions: dict[int, bool] = {}
        for slot in active:
            existing = supported_directions(self._counts[slot], min_cell_support=_MIN_CELL_SUPPORT)
            overlap_mask = (incoming >= 0) & (existing >= 0)
            overlap = int(np.count_nonzero(overlap_mask))
            conflicts = int(np.count_nonzero(incoming[overlap_mask] != existing[overlap_mask]))
            cosine = self._reward_cosine(profile, slot)
            reward_match = cosine is not None and cosine >= _MIN_REWARD_COSINE
            if overlap >= _MIN_OVERLAP and conflicts == 0 and reward_match:
                scores[slot] = (overlap, float(cosine))
            reward_opposite = (
                cosine is not None and cosine <= _RELIABLE_OPPOSITE_COSINE
            )
            movement_conflict = overlap >= _MIN_OVERLAP and conflicts >= 1
            distinctions[slot] = movement_conflict or reward_opposite

        if scores:
            best = max(scores.values())
            winners = [
                slot
                for slot, score in scores.items()
                if np.isclose(score[0], best[0], rtol=0.0, atol=1e-12)
                and np.isclose(score[1], best[1], rtol=0.0, atol=1e-12)
            ]
            if len(winners) == 1:
                return {
                    "slot": winners[0],
                    "allocated_slot": None,
                    "evicted_slot": None,
                    "reason": "known_profile_reused",
                }
            return {"slot": -1, "allocated_slot": None, "evicted_slot": None, "reason": "ambiguous_match"}

        if all(distinctions.values()):
            opposite_reward = any(
                cosine is not None and cosine <= _RELIABLE_OPPOSITE_COSINE
                for slot in active
                if (cosine := self._reward_cosine(profile, slot)) is not None
            )
            slot, evicted = self._allocate_slot()
            reason = "opposite_reward_profile" if opposite_reward else "novel_profile"
            return {"slot": slot, "allocated_slot": slot, "evicted_slot": evicted, "reason": reason}

        if any(not distinct for distinct in distinctions.values()):
            reason = "insufficient_distinction"
        else:
            reason = "ambiguous_profile"
        return {"slot": -1, "allocated_slot": None, "evicted_slot": None, "reason": reason}

    def _allocate_slot(self) -> tuple[int, int | None]:
        empty = np.flatnonzero(self._active == 0)
        if len(empty):
            slot = int(empty[0])
            return slot, None
        active = np.flatnonzero(self._active)
        if self.eviction == "lru":
            slot = int(min(active, key=lambda item: (int(self._last_used[item]), int(item))))
        elif self.eviction == "lfu":
            slot = int(min(active, key=lambda item: (int(self._frequency[item]), int(self._last_used[item]), int(item))))
        else:
            slot = int(self._rng.choice(active))
            self._rng_state = _rng_words(self._rng)
        evicted = slot
        self._clear_slot(slot)
        return slot, evicted

    def _clear_slot(self, slot: int) -> None:
        self._counts[slot].fill(0)
        self._reward_xtx[slot].fill(0.0)
        self._reward_xty[slot].fill(0.0)
        self._reward_n[slot] = 0
        self._active[slot] = 0
        self._last_used[slot] = 0
        self._frequency[slot] = 0

    def _commit_profile(self, slot: int, profile: _WindowProfile) -> None:
        summed_counts = self._counts[slot].astype(np.uint32) + profile.counts.astype(np.uint32)
        self._counts[slot] = np.minimum(summed_counts, _MAX_PROFILE_COUNT).astype(np.uint16)
        incoming_xtx = profile.reward_xtx.copy()
        incoming_xty = profile.reward_xty.copy()
        old_xtx = self._reward_xtx[slot].copy()
        old_xty = self._reward_xty[slot].copy()
        old_n = int(self._reward_n[slot])
        incoming_n = int(profile.reward_n)
        while old_n + incoming_n > _MAX_REWARD_COUNT:
            old_xtx *= 0.5
            old_xty *= 0.5
            incoming_xtx *= 0.5
            incoming_xty *= 0.5
            old_n //= 2
            incoming_n //= 2
        self._reward_xtx[slot] = old_xtx + incoming_xtx
        self._reward_xty[slot] = old_xty + incoming_xty
        self._reward_n[slot] = np.uint32(old_n + incoming_n)
        self._active[slot] = 1

    def _add_event_to_slot(
        self,
        slot: int,
        obs: np.ndarray,
        action: int,
        reward: np.float32,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        observe_profile(self._counts[slot], obs, action, next_obs)
        if int(self._reward_n[slot]) >= _MAX_REWARD_COUNT:
            self._reward_xtx[slot] *= 0.5
            self._reward_xty[slot] *= 0.5
            self._reward_n[slot] //= 2
        features = _reward_features(obs, next_obs, terminated)
        self._reward_xtx[slot] += np.outer(features, features)
        self._reward_xty[slot] += features * float(reward)
        self._reward_n[slot] += np.uint32(1)

    def _all_arrays(self) -> dict[str, np.ndarray]:
        return {
            "counts": self._counts,
            "reward_xtx": self._reward_xtx,
            "reward_xty": self._reward_xty,
            "reward_n": self._reward_n,
            "active": self._active,
            "last_used": self._last_used,
            "frequency": self._frequency,
            "history_obs": self._history_obs,
            "history_actions": self._history_actions,
            "history_rewards": self._history_rewards,
            "history_next_obs": self._history_next_obs,
            "history_terminated": self._history_terminated,
            "rng_state": self._rng_state,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return fixed-size numeric profiles, history, metadata, and RNG state."""

        arrays = {name: value.copy() for name, value in self._all_arrays().items()}
        return {
            "version": _STATE_VERSION,
            "config": {
                "max_slots": self.max_slots,
                "window": self.window,
                "eviction": self.eviction,
            },
            **arrays,
            "history_size": self._history_size,
            "history_next": self._history_next,
            "clock": self._clock,
            "active_slot": self._active_slot,
            "allocations": self._allocations,
            "evictions": self._evictions,
            "switches": self._switches,
            "last_reason": self._last_reason,
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "WriteDecisionRouter":
        """Restore a snapshot after validating all bounded array shapes/dtypes."""

        if not isinstance(state, Mapping) or state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported WriteDecisionRouter state")
        config = state.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("state config must be a mapping")
        router = cls(**dict(config))
        specifications = {
            "counts": ((router.max_slots, 3, 4, 4), np.dtype(np.uint16)),
            "reward_xtx": ((router.max_slots, _FEATURES, _FEATURES), np.dtype(np.float64)),
            "reward_xty": ((router.max_slots, _FEATURES), np.dtype(np.float64)),
            "reward_n": ((router.max_slots,), np.dtype(np.uint32)),
            "active": ((router.max_slots,), np.dtype(np.uint8)),
            "last_used": ((router.max_slots,), np.dtype(np.int64)),
            "frequency": ((router.max_slots,), np.dtype(np.uint64)),
            "history_obs": ((router.window, 4), np.dtype(np.float32)),
            "history_actions": ((router.window,), np.dtype(np.uint8)),
            "history_rewards": ((router.window,), np.dtype(np.float32)),
            "history_next_obs": ((router.window, 4), np.dtype(np.float32)),
            "history_terminated": ((router.window,), np.dtype(np.uint8)),
            "rng_state": ((6,), np.dtype(np.uint64)),
        }
        restored: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in specifications.items():
            array = np.asarray(state.get(name))
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"state {name} must have dtype {dtype} and shape {shape}")
            restored[name] = array
        if not np.isfinite(restored["reward_xtx"]).all() or not np.isfinite(restored["reward_xty"]).all():
            raise ValueError("reward statistics must be finite")
        if not np.isfinite(restored["history_obs"]).all() or not np.isfinite(restored["history_next_obs"]).all():
            raise ValueError("history observations must be finite")
        if not np.isfinite(restored["history_rewards"]).all():
            raise ValueError("history rewards must be finite")
        if np.any(restored["active"] > 1) or np.any(restored["history_actions"] > 3) or np.any(restored["history_terminated"] > 1):
            raise ValueError("state contains out-of-range flags or actions")
        if not np.allclose(restored["reward_xtx"], restored["reward_xtx"].transpose(0, 2, 1), rtol=0.0, atol=1e-8):
            raise ValueError("state reward_xtx must be symmetric")

        history_size = _positive_integer(state.get("history_size", 0), "history_size", minimum=0, maximum=router.window)
        history_next = _positive_integer(state.get("history_next", 0), "history_next", minimum=0, maximum=router.window - 1)
        if history_size < router.window and history_next != history_size:
            raise ValueError("partial history next index must equal its size")
        for index in router._validated_history_indices(history_size, history_next):
            _grid_observation(restored["history_obs"][index], "state history_obs")
            _grid_observation(restored["history_next_obs"][index], "state history_next_obs")
        for slot in range(router.max_slots):
            has_evidence = bool(restored["counts"][slot].any() or restored["reward_n"][slot] > 0)
            if bool(restored["active"][slot]) != has_evidence:
                raise ValueError("state active flags do not match slot evidence")
            if not np.allclose(restored["reward_xtx"][slot], restored["reward_xtx"][slot].T, rtol=0.0, atol=1e-8):
                raise ValueError("state reward_xtx must be symmetric")

        active_slot = int(state.get("active_slot", -1))
        if active_slot < -1 or active_slot >= router.max_slots:
            raise ValueError("active_slot is outside configured capacity")
        if active_slot >= 0 and not restored["active"][active_slot]:
            raise ValueError("active_slot has no profile")
        if "nbytes" in state and state["nbytes"] != sum(array.nbytes for array in restored.values()):
            raise ValueError("state nbytes does not match numeric array storage")

        for name, array in restored.items():
            getattr(router, f"_{name}")[...] = array
        router._history_size = history_size
        router._history_next = history_next
        router._clock = int(state.get("clock", 0))
        router._active_slot = active_slot
        router._allocations = int(state.get("allocations", 0))
        router._evictions = int(state.get("evictions", 0))
        router._switches = int(state.get("switches", 0))
        router._last_reason = str(state.get("last_reason", "insufficient_support"))
        router._rng = _restore_rng(router._rng_state)
        return router

    def _validated_history_indices(self, size: int, next_index: int) -> list[int]:
        start = (next_index - size) % self.window
        return [(start + offset) % self.window for offset in range(size)]


__all__ = ["WriteDecisionRouter"]

"""Bounded, evidence-gated routing for recurring C1 transition contexts.

The router consumes normalized ``(x, y, goal_x, goal_y)`` observations, an
action, and the resulting position. It uses only the visible x band, action,
and observed movement. A bounded coherent suffix is held provisionally and
compared with every persistent profile before any profile is updated.
"""

from __future__ import annotations

from collections import Counter, deque
from operator import index as as_index
import sys
from typing import Any, Mapping

import numpy as np


_ACTION_COUNT = 4
_BAND_COUNT = 3
_DIRECTION_COUNT = 4
_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_SNAPSHOT_VERSION = 1
_HARD_CONTEXT_CAP = 6


class EvidenceRouter:
    """Discover recurring deterministic transition profiles from observations.

    Parameters match the useful subset of :class:`TaskFreeRouter`'s interface.
    New profiles require at least ``min_seed_events`` moving events and
    ``min_seed_cells`` distinct band/action cells, each supported by at least
    ``min_cell_support`` observations. Reuse compares all locally supported
    cells with every profile. A profile with no supported disagreement can be
    reused from partial evidence; cells absent from that profile remain
    unknown. A new profile is created only after every stored profile has at
    least ``min_conflict_cells`` supported conflicts.

    This is an exact compatibility model for mostly deterministic transition
    laws. It does not estimate noise tolerance.
    """

    _DIRECTION_NAMES = ("up", "right", "down", "left")

    def __init__(
        self,
        max_contexts: int = _HARD_CONTEXT_CAP,
        window: int = 32,
        *,
        alpha: float = 0.5,
        min_seed_events: int = 4,
        min_seed_cells: int = 2,
        min_cell_support: int = 2,
        min_conflict_cells: int = 2,
        max_support_per_cell: int | None = None,
    ) -> None:
        self.max_contexts = self._positive_integer(max_contexts, "max_contexts")
        if self.max_contexts > _HARD_CONTEXT_CAP:
            raise ValueError(f"max_contexts must not exceed {_HARD_CONTEXT_CAP}")
        self.window = self._positive_integer(window, "window")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)
        self.min_seed_events = self._positive_integer(min_seed_events, "min_seed_events")
        self.min_seed_cells = self._positive_integer(min_seed_cells, "min_seed_cells")
        self.min_cell_support = self._positive_integer(min_cell_support, "min_cell_support")
        self.min_conflict_cells = self._positive_integer(min_conflict_cells, "min_conflict_cells")
        self.max_support_per_cell = (
            min(self.window, 65535)
            if max_support_per_cell is None
            else self._positive_integer(max_support_per_cell, "max_support_per_cell")
        )
        if self.max_support_per_cell > 65535:
            raise ValueError("max_support_per_cell must fit in uint16")

        self._counts = np.zeros(
            (self.max_contexts, _BAND_COUNT, _ACTION_COUNT, _DIRECTION_COUNT),
            dtype=np.uint16,
        )
        self._active: list[bool] = [False] * self.max_contexts
        self._created_at: list[int | None] = [None] * self.max_contexts
        self._last_used: list[int] = [0] * self.max_contexts
        self._clock = 0
        self._next_event_id = 0
        self._active_slot = -1
        self._suffix_provisional = False
        self._recent: deque[tuple[int, int, int, int]] = deque(maxlen=self.window)
        self._committed_ids: list[set[int]] = [set() for _ in range(self.max_contexts)]
        self._allocations = 0
        self._evictions = 0
        self._switches = 0
        self._censored = 0
        self._last_reason = "insufficient_support"

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            result = as_index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if result < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(result)

    @staticmethod
    def _observation_cell(observation: Any, name: str) -> tuple[int, int]:
        try:
            values = np.asarray(observation, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite normalized C1 observation of shape (4,)") from exc
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite normalized C1 observation of shape (4,)")
        if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
            raise ValueError(f"{name} coordinates must lie in [0, 1]")
        position = values[:2] * 8.0
        rounded = np.rint(position)
        if np.any(np.abs(position - rounded) > 1e-4):
            raise ValueError(f"{name} position must lie on the normalized 9 by 9 grid")
        return int(rounded[0]), int(rounded[1])

    @staticmethod
    def _action_index(action: Any) -> int:
        if isinstance(action, bool):
            raise ValueError("action must be an integer in {0, 1, 2, 3}")
        try:
            value = as_index(action)
        except TypeError as exc:
            raise ValueError("action must be an integer in {0, 1, 2, 3}") from exc
        if value not in range(_ACTION_COUNT):
            raise ValueError("action must be an integer in {0, 1, 2, 3}")
        return int(value)

    @staticmethod
    def _direction(before: tuple[int, int], after: tuple[int, int]) -> int | None:
        delta = (after[0] - before[0], after[1] - before[1])
        if delta == (0, 0):
            return None
        try:
            return _DELTAS.index(delta)
        except ValueError as exc:
            raise ValueError("C1 transitions must move one cell or remain blocked") from exc

    def _cell_total(self, slot: int, band: int, action: int) -> int:
        return int(self._counts[slot, band, action].sum())

    def _profile_direction(self, slot: int, band: int, action: int) -> int | None:
        row = self._counts[slot, band, action]
        if int(row.sum()) < self.min_cell_support:
            return None
        return int(np.argmax(row))

    def _suffix_summary(self) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], list[int]], int]:
        per_cell: dict[tuple[int, int], Counter[int]] = {}
        ids_by_cell: dict[tuple[int, int], list[int]] = {}
        for event_id, band, action, direction in self._recent:
            key = (band, action)
            per_cell.setdefault(key, Counter())[direction] += 1
            ids_by_cell.setdefault(key, []).append(event_id)
        supported: dict[tuple[int, int], tuple[int, int]] = {}
        supported_ids: dict[tuple[int, int], list[int]] = {}
        supported_event_count = 0
        for key, directions in per_cell.items():
            direction, count = max(directions.items(), key=lambda item: item[1])
            if count < self.min_cell_support:
                continue
            # A cell with competing directions is unknown until a coherent
            # suffix supplies a single supported direction.
            if sum(value for other, value in directions.items() if other != direction):
                continue
            supported[key] = (direction, count)
            ids = ids_by_cell[key]
            supported_ids[key] = ids[-count:]
            supported_event_count += count
        return supported, supported_ids, supported_event_count

    def _ready(self, supported: Mapping[tuple[int, int], tuple[int, int]]) -> bool:
        return (
            len(self._recent) >= self.min_seed_events
            and len(supported) >= self.min_seed_cells
        )

    def _candidate_analysis(
        self,
        supported: Mapping[tuple[int, int], tuple[int, int]],
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]], bool]:
        """Return candidates (slot, overlap cells, overlap support), conflicts, ready."""
        candidates: list[tuple[int, int, int]] = []
        conflict_counts: list[tuple[int, int]] = []
        for slot, active in enumerate(self._active):
            if not active:
                continue
            overlap = 0
            overlap_support = 0
            conflicts = 0
            for (band, action), (direction, local_count) in supported.items():
                profile_direction = self._profile_direction(slot, band, action)
                if profile_direction is None:
                    continue
                if profile_direction == direction:
                    overlap += 1
                    overlap_support += local_count
                else:
                    conflicts += 1
            conflict_counts.append((slot, conflicts))
            if conflicts == 0 and overlap:
                candidates.append((slot, overlap, overlap_support))
        return candidates, conflict_counts, bool(conflict_counts) and all(
            conflicts >= self.min_conflict_cells for _, conflicts in conflict_counts
        )

    def _route_result(self) -> dict[str, Any]:
        supported, _, supported_event_count = self._suffix_summary()
        recent_count = len(self._recent)
        support = float(supported_event_count / recent_count) if recent_count else 0.0
        common = {
            "confidence": 0.0,
            "support": support,
            "recent_support": int(supported_event_count),
        }
        if not self._ready(supported):
            reason = "insufficient_support"
            if len(self._recent) >= self.min_seed_events and supported:
                reason = "insufficient_supported_cells"
            self._last_reason = reason
            return {"slot": -1, **common, "status": "unresolved", "reason": reason}

        if not any(self._active):
            self._last_reason = "initial_profile_pending"
            return {
                "slot": -1,
                **common,
                "status": "unresolved",
                "reason": self._last_reason,
            }

        candidates, conflicts, all_contradicted = self._candidate_analysis(supported)
        if candidates:
            candidates.sort(key=lambda item: (-item[1], -item[2], item[0]))
            best = candidates[0]
            tied = len(candidates) > 1 and candidates[1][1:] == best[1:]
            if tied:
                self._last_reason = "ambiguous_profile_match"
                return {
                    "slot": -1,
                    **common,
                    "status": "ambiguous",
                    "reason": self._last_reason,
                }
            slot, overlap, _ = best
            confidence = float(np.clip(support * overlap / (overlap + 1.0), 0.0, 1.0))
            all_local_cells = len(supported)
            reason = "supported" if overlap == all_local_cells else "partial_profile_match"
            self._last_reason = reason
            return {
                "slot": slot,
                "confidence": confidence,
                "support": support,
                "recent_support": int(supported_event_count),
                "status": "resolved",
                "reason": reason,
            }

        if all_contradicted:
            self._last_reason = "new_profile_confirmed"
        elif any(conflicts for _, conflicts in conflicts):
            self._last_reason = "insufficient_contradiction"
        else:
            self._last_reason = "unseen_cells_only"
        return {
            "slot": -1,
            **common,
            "status": "unresolved",
            "reason": self._last_reason,
        }

    def route(self, observation: Any) -> dict[str, Any]:
        """Return the supported profile before the next action."""
        self._observation_cell(observation, "observation")
        result = self._route_result()
        slot = int(result["slot"])
        if slot >= 0:
            self._last_used[slot] = self._clock
        return result

    def _remove_recent_commit_ids(self, event_id: int) -> None:
        for committed in self._committed_ids:
            committed.discard(event_id)

    def _clear_suffix(self, *, provisional: bool) -> None:
        self._recent.clear()
        for committed in self._committed_ids:
            committed.clear()
        self._suffix_provisional = provisional

    def _current_event_contradicts_profile(self, event: tuple[int, int, int]) -> bool:
        if self._active_slot < 0 or not self._active[self._active_slot]:
            return False
        band, action, direction = event
        learned = self._profile_direction(self._active_slot, band, action)
        return learned is not None and learned != direction

    def _current_suffix_contradicts(self, event: tuple[int, int, int]) -> bool:
        band, action, direction = event
        for _, old_band, old_action, old_direction in self._recent:
            if old_band == band and old_action == action and old_direction != direction:
                return True
        return False

    def _append(self, event: tuple[int, int, int]) -> None:
        if len(self._recent) == self.window:
            expired_id = self._recent[0][0]
            self._remove_recent_commit_ids(expired_id)
        event_id = self._next_event_id
        self._next_event_id += 1
        band, action, direction = event
        self._recent.append((event_id, band, action, direction))

    def _increment(self, slot: int, event_id: int, band: int, action: int, direction: int) -> None:
        if event_id in self._committed_ids[slot]:
            return
        current = int(self._counts[slot, band, action, direction])
        if current < self.max_support_per_cell:
            self._counts[slot, band, action, direction] = current + 1
        self._committed_ids[slot].add(event_id)
        self._last_used[slot] = self._clock

    def _commit_supported(self, slot: int) -> None:
        supported, supported_ids, _ = self._suffix_summary()
        for (band, action), (direction, _) in supported.items():
            for event_id in supported_ids[(band, action)]:
                self._increment(slot, event_id, band, action, direction)

    def _allocate(self) -> tuple[int, int | None]:
        evicted: int | None = None
        try:
            slot = self._active.index(False)
        except ValueError:
            slot = min(range(self.max_contexts), key=lambda index: (self._last_used[index], index))
            evicted = slot
            self._evictions += 1
            self._counts[slot].fill(0)
            self._committed_ids[slot].clear()
        self._active[slot] = True
        self._created_at[slot] = self._clock
        self._last_used[slot] = self._clock
        self._commit_supported(slot)
        self._active_slot = slot
        self._suffix_provisional = False
        self._allocations += 1
        self._last_reason = "new_profile_confirmed"
        return slot, evicted

    def observe(self, obs: Any, action: Any, reward: float, next_obs: Any) -> dict[str, Any]:
        """Learn a movement event and report profile creation or reuse."""
        del reward
        before = self._observation_cell(obs, "obs")
        after = self._observation_cell(next_obs, "next_obs")
        command = self._action_index(action)
        self._clock += 1
        direction = self._direction(before, after)
        if direction is None:
            self._censored += 1
            return {
                "event": None,
                "censored": True,
                "active_slot": self._active_slot,
                "allocated_slot": None,
                "evicted_slot": None,
                "switched": False,
                "reason": "censored_wall_no_motion",
            }

        event = (before[0] // 3, command, direction)
        contradicted_profile = (
            not self._suffix_provisional
            and self._current_event_contradicts_profile(event)
        )
        contradicted_suffix = self._current_suffix_contradicts(event)
        if contradicted_profile:
            self._clear_suffix(provisional=True)
        elif contradicted_suffix:
            self._clear_suffix(provisional=True)

        self._append(event)
        result = self._route_result()
        allocated_slot: int | None = None
        evicted_slot: int | None = None
        switched = False
        slot = int(result["slot"])

        if slot >= 0:
            previous = self._active_slot
            if previous >= 0 and previous != slot:
                self._switches += 1
                switched = True
            self._active_slot = slot
            self._suffix_provisional = False
            self._commit_supported(slot)
            self._last_used[slot] = self._clock
        elif self._ready(self._suffix_summary()[0]):
            supported = self._suffix_summary()[0]
            if not any(self._active):
                allocated_slot, evicted_slot = self._allocate()
            else:
                _, _, all_contradicted = self._candidate_analysis(supported)
                if all_contradicted:
                    allocated_slot, evicted_slot = self._allocate()
            if allocated_slot is not None:
                slot = allocated_slot
                result = self._route_result()
                result["slot"] = slot
                result["confidence"] = float(
                    np.clip(result["support"] * len(supported) / (len(supported) + 1.0), 0.0, 1.0)
                )
                result["status"] = "resolved"
                result["reason"] = "new_profile_confirmed"

        if allocated_slot is not None:
            self._last_reason = "new_profile_confirmed"
        else:
            self._last_reason = str(result["reason"])
        return {
            "event": {"band": event[0], "action": command, "direction": direction},
            "censored": False,
            "active_slot": self._active_slot,
            "allocated_slot": allocated_slot,
            "evicted_slot": evicted_slot,
            "switched": switched,
            "reason": self._last_reason,
        }

    def predict_direction(self, observation: Any, action: Any) -> np.ndarray:
        """Return a smoothed physical-direction prediction before the action."""
        x, _ = self._observation_cell(observation, "observation")
        command = self._action_index(action)
        routed = self.route(observation)
        if routed["slot"] < 0:
            return np.full(_DIRECTION_COUNT, 1.0 / _DIRECTION_COUNT, dtype=np.float64)
        row = self._counts[int(routed["slot"]), x // 3, command].astype(np.float64)
        probabilities = row + self.alpha
        return probabilities / probabilities.sum()

    def reset_history(self) -> None:
        """Clear provisional evidence while keeping persistent profiles."""
        self._clear_suffix(provisional=True)
        self._active_slot = -1
        self._last_reason = "insufficient_support"

    def state_dict(self) -> dict[str, Any]:
        """Return a self-contained, JSON-compatible snapshot."""
        return {
            "version": _SNAPSHOT_VERSION,
            "config": {
                "max_contexts": self.max_contexts,
                "window": self.window,
                "alpha": self.alpha,
                "min_seed_events": self.min_seed_events,
                "min_seed_cells": self.min_seed_cells,
                "min_cell_support": self.min_cell_support,
                "min_conflict_cells": self.min_conflict_cells,
                "max_support_per_cell": self.max_support_per_cell,
            },
            "counts": self._counts.tolist(),
            "active": list(self._active),
            "created_at": list(self._created_at),
            "last_used": list(self._last_used),
            "clock": self._clock,
            "next_event_id": self._next_event_id,
            "active_slot": self._active_slot,
            "suffix_provisional": self._suffix_provisional,
            "recent": [list(event) for event in self._recent],
            "committed_ids": [sorted(ids) for ids in self._committed_ids],
            "allocations": self._allocations,
            "evictions": self._evictions,
            "switches": self._switches,
            "censored": self._censored,
            "last_reason": self._last_reason,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EvidenceRouter":
        """Restore a snapshot produced by :meth:`state_dict`."""
        if not isinstance(state, Mapping) or state.get("version") != _SNAPSHOT_VERSION:
            raise ValueError("unsupported EvidenceRouter state")
        config = state.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("EvidenceRouter state is missing config")
        try:
            router = cls(**dict(config))
            counts = np.asarray(state["counts"], dtype=np.uint16)
            if counts.shape != router._counts.shape or np.any(counts > router.max_support_per_cell):
                raise ValueError("profile count array does not match config")
            router._counts[:] = counts
            active = list(state["active"])
            created_at = list(state["created_at"])
            last_used = list(state["last_used"])
            if any(len(values) != router.max_contexts for values in (active, created_at, last_used)):
                raise ValueError("slot metadata length does not match config")
            router._active = [bool(value) for value in active]
            router._created_at = [None if value is None else int(value) for value in created_at]
            router._last_used = [int(value) for value in last_used]
            router._clock = int(state["clock"])
            router._next_event_id = int(state["next_event_id"])
            router._active_slot = int(state["active_slot"])
            if router._active_slot < -1 or router._active_slot >= router.max_contexts:
                raise ValueError("active slot is outside configured capacity")
            router._suffix_provisional = bool(state["suffix_provisional"])
            router._recent.extend(router._validated_events(state["recent"], router.window))
            committed = state["committed_ids"]
            if len(committed) != router.max_contexts:
                raise ValueError("committed evidence length does not match config")
            recent_ids = {item[0] for item in router._recent}
            for slot, values in enumerate(committed):
                ids = {int(value) for value in values}
                if not ids.issubset(recent_ids):
                    raise ValueError("committed evidence refers to an event outside the suffix")
                router._committed_ids[slot] = ids
            router._allocations = int(state.get("allocations", 0))
            router._evictions = int(state.get("evictions", 0))
            router._switches = int(state.get("switches", 0))
            router._censored = int(state.get("censored", 0))
            router._last_reason = str(state.get("last_reason", "insufficient_support"))
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("EvidenceRouter state is malformed") from exc
        return router

    @staticmethod
    def _validated_events(events: Any, limit: int) -> list[tuple[int, int, int, int]]:
        if not isinstance(events, (list, tuple)) or len(events) > limit:
            raise ValueError("event list exceeds configured window")
        result: list[tuple[int, int, int, int]] = []
        for event in events:
            if not isinstance(event, (list, tuple)) or len(event) != 4:
                raise ValueError("recent events must contain id, band, action, and direction")
            event_id, band, action, direction = (int(value) for value in event)
            if event_id < 0 or band not in range(_BAND_COUNT) or action not in range(_ACTION_COUNT) or direction not in range(_DIRECTION_COUNT):
                raise ValueError("event value is outside the C1 range")
            result.append((event_id, band, action, direction))
        return result

    def nbytes(self) -> dict[str, int]:
        """Estimate bytes owned by profiles and bounded routing state."""
        profile_bytes = int(self._counts.nbytes)
        recent_bytes = sys.getsizeof(self._recent) + sum(
            sys.getsizeof(event) + sum(sys.getsizeof(value) for value in event)
            for event in self._recent
        )
        committed_bytes = sys.getsizeof(self._committed_ids)
        for values in self._committed_ids:
            committed_bytes += sys.getsizeof(values) + sum(sys.getsizeof(value) for value in values)
        metadata_bytes = (
            sys.getsizeof(self._active) + sys.getsizeof(self._created_at)
            + sys.getsizeof(self._last_used) + sys.getsizeof(self._active_slot)
            + sys.getsizeof(self._clock) + sys.getsizeof(self._next_event_id)
            + sys.getsizeof(self._suffix_provisional)
        )
        breakdown = {
            "profile_counts": profile_bytes,
            "recent_window": int(recent_bytes),
            "committed_evidence": int(committed_bytes),
            "slot_metadata": int(metadata_bytes),
        }
        breakdown["total"] = int(sum(breakdown.values()))
        return breakdown

    @property
    def active_slots(self) -> tuple[int, ...]:
        """Slots that currently hold a learned profile."""
        return tuple(slot for slot, active in enumerate(self._active) if active)

    @property
    def allocations(self) -> int:
        return self._allocations

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def switches(self) -> int:
        return self._switches


__all__ = ["EvidenceRouter"]

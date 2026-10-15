"""From-zero discovery of recurring C1 transition contexts.

The router consumes only normalized observations, commanded actions, and the
resulting observable movement.  It keeps bounded per-slot action-to-direction
counts and a bounded recent window.  Censored no-motion responses do not
update the model.
"""

from __future__ import annotations

from collections import defaultdict, deque
from operator import index as as_index
import sys
from typing import Any, Mapping

import numpy as np


_ACTION_COUNT = 4
_BAND_COUNT = 3
_DIRECTION_COUNT = 4
_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_SNAPSHOT_VERSION = 1


class TaskFreeRouter:
    """Discover recurring contexts without labels or a pretrained context bank.

    Parameters
    ----------
    max_contexts:
        Number of persistent, zero-based context slots.
    window:
        Maximum number of recent observable transitions retained for routing
        diagnostics and provisional discovery.  Per-cell slot counts also cap
        at this value.
    mode:
        ``"robust"`` requires supported contradictions before discovery and
        treats an unobserved band/action cell as unknown. ``"naive"`` also
        treats such cells as novelty evidence; this is included as a deliberately
        weak comparison.
    alpha:
        Symmetric Dirichlet smoothing for predictive transition probabilities.
    confirm_events:
        Number of contradictory moving events needed to confirm a new profile
        while a bank already exists (subject to the bounded window). Reusing a
        known profile intentionally requires ``min(2, confirm_events)``
        supported matches across two distinct band/action cells.
    min_seed_cells:
        Distinct band/action cells required before allocating the first slot.
    max_support_per_cell:
        Per-direction count cap for each slot and band/action cell. Defaults to
        ``window``.

    ``obs`` and ``next_obs`` must be normalized C1 observations of shape
    ``(4,)``: ``(x, y, goal_x, goal_y)``. Routing uses only the visible
    position and its band; it never reads goal coordinates. The first slot
    requires ``min_seed_cells`` distinct moving band/action cells. Slot numbers
    are stable until evicted. When full, allocation replaces the
    least-recently-used slot.
    """

    _DIRECTION_NAMES = ("up", "right", "down", "left")

    def __init__(
        self,
        max_contexts: int = 6,
        window: int = 32,
        *,
        mode: str = "robust",
        alpha: float = 0.5,
        confirm_events: int = 3,
        min_seed_cells: int = 4,
        max_support_per_cell: int | None = None,
    ) -> None:
        self.max_contexts = self._positive_integer(max_contexts, "max_contexts")
        self.window = self._positive_integer(window, "window")
        if mode not in ("robust", "naive"):
            raise ValueError("mode must be 'robust' or 'naive'")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.mode = mode
        self.alpha = float(alpha)
        self.confirm_events = self._positive_integer(confirm_events, "confirm_events")
        self.min_seed_cells = self._positive_integer(min_seed_cells, "min_seed_cells")
        self.max_support_per_cell = (
            self.window
            if max_support_per_cell is None
            else self._positive_integer(max_support_per_cell, "max_support_per_cell")
        )

        self._counts = np.zeros(
            (self.max_contexts, _BAND_COUNT, _ACTION_COUNT, _DIRECTION_COUNT),
            dtype=np.uint32,
        )
        self._active: list[bool] = [False] * self.max_contexts
        self._created_at: list[int | None] = [None] * self.max_contexts
        self._last_used: list[int] = [0] * self.max_contexts
        self._clock = 0
        self._active_slot = -1
        self._route_unresolved = False
        self._recent: deque[tuple[int, int, int]] = deque(maxlen=self.window)
        self._novel_pending: deque[tuple[int, int, int]] = deque(maxlen=self.window)
        self._candidate_pending: dict[int, deque[tuple[int, int, int]]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self._candidate_last_clock: dict[int, int] = {}
        self._recovery_hits: dict[tuple[int, int], int] = {}
        self._novel_last_clock: int | None = None
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

    def _probability(self, slot: int, event: tuple[int, int, int]) -> float:
        band, action, direction = event
        row = self._counts[slot, band, action]
        total = int(row.sum())
        return float((int(row[direction]) + self.alpha) / (total + _DIRECTION_COUNT * self.alpha))

    def _event_support(self, slot: int, event: tuple[int, int, int]) -> int:
        band, action, _ = event
        return self._cell_total(slot, band, action)

    def _matching_slots(self, event: tuple[int, int, int]) -> list[tuple[int, float]]:
        matches: list[tuple[int, float]] = []
        for slot in range(self.max_contexts):
            if not self._active[slot] or self._event_support(slot, event) < 2:
                continue
            probability = self._probability(slot, event)
            if probability >= 0.60:
                matches.append((slot, probability))
        return matches

    @staticmethod
    def _distinct_cells(events: Any) -> set[tuple[int, int]]:
        return {(int(event[0]), int(event[1])) for event in events}

    def _add(self, slot: int, event: tuple[int, int, int]) -> None:
        band, action, direction = event
        current = int(self._counts[slot, band, action, direction])
        if current < self.max_support_per_cell:
            self._counts[slot, band, action, direction] = current + 1
        self._last_used[slot] = self._clock

    def _allocate(self, seed_events: list[tuple[int, int, int]]) -> tuple[int, int | None]:
        evicted: int | None = None
        try:
            slot = self._active.index(False)
        except ValueError:
            slot = min(range(self.max_contexts), key=lambda index: (self._last_used[index], index))
            evicted = slot
            self._evictions += 1
            self._counts[slot].fill(0)
            if self._active_slot == slot:
                self._active_slot = -1
        self._active[slot] = True
        self._created_at[slot] = self._clock
        self._last_used[slot] = self._clock
        for event in seed_events:
            self._add(slot, event)
        self._active_slot = slot
        self._route_unresolved = False
        self._recent.clear()
        self._recent.extend(seed_events[-self.window :])
        self._novel_pending.clear()
        self._candidate_pending.clear()
        self._candidate_last_clock.clear()
        self._recovery_hits.clear()
        self._novel_last_clock = None
        self._allocations += 1
        self._last_reason = "new_context_confirmed"
        return slot, evicted

    def _switch(self, slot: int, evidence: list[tuple[int, int, int]]) -> None:
        self._active_slot = slot
        self._switches += 1
        self._route_unresolved = False
        self._recent.clear()
        self._recent.extend(evidence[-self.window :])
        self._novel_pending.clear()
        self._candidate_pending.clear()
        self._candidate_last_clock.clear()
        self._recovery_hits.clear()
        self._novel_last_clock = None
        for event in evidence:
            self._add(slot, event)
        self._last_reason = "known_context_reused"

    def _update_candidate_evidence(self, matches: list[tuple[int, float]], event: tuple[int, int, int]) -> int | None:
        for slot in list(self._candidate_pending):
            if self._clock - self._candidate_last_clock.get(slot, 0) > self.window:
                del self._candidate_pending[slot]
                self._candidate_last_clock.pop(slot, None)
        matched_slots = {slot for slot, _ in matches if slot != self._active_slot}
        if matched_slots:
            for slot in list(self._candidate_pending):
                if slot not in matched_slots:
                    del self._candidate_pending[slot]
                    self._candidate_last_clock.pop(slot, None)
        for slot in matched_slots:
            evidence = self._candidate_pending[slot]
            evidence.append(event)
            self._candidate_last_clock[slot] = self._clock
            if (
                len(evidence) >= min(2, self.confirm_events)
                and len(self._distinct_cells(evidence)) >= 2
            ):
                buffered = list(evidence)
                self._switch(slot, buffered)
                return slot
        return None

    def _update_novelty(self, event: tuple[int, int, int], *, contradiction: bool) -> tuple[int | None, int | None]:
        if (
            contradiction
            or self.mode == "naive"
            or (self._active_slot < 0 and not any(self._active))
        ):
            self._novel_pending.append(event)
            self._novel_last_clock = self._clock
        if self._active_slot < 0:
            threshold = (
                min(self.min_seed_cells, self.window)
                if not any(self._active)
                else min(self.confirm_events, self.window)
            )
            unique_cells = len(self._distinct_cells(self._novel_pending))
            if unique_cells >= threshold:
                new_slot, evicted = self._allocate(list(self._novel_pending))
                return new_slot, evicted
        elif (
            len(self._novel_pending) >= min(self.confirm_events, self.window)
            and len(self._distinct_cells(self._novel_pending)) >= min(2, self.confirm_events, self.window)
        ):
            new_slot, evicted = self._allocate(list(self._novel_pending))
            return new_slot, evicted
        return None, None

    def observe(self, obs: Any, action: Any, reward: float, next_obs: Any) -> dict[str, Any]:
        """Learn one transition and report any slot allocation or switch.

        Reward is accepted for runner compatibility and is intentionally
        ignored. Stationary responses are censored because they reveal only
        that some physical direction was clipped by a wall.
        """

        del reward
        before = self._observation_cell(obs, "obs")
        after = self._observation_cell(next_obs, "next_obs")
        command = self._action_index(action)
        self._clock += 1
        self._expire_stale_provisional_evidence()
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
        self._recent.append(event)
        self._last_reason = "insufficient_support"
        allocated_slot: int | None = None
        evicted_slot: int | None = None
        switched = False
        predictive_probability: float | None = None
        surprise: float | None = None

        if self._active_slot >= 0:
            active = self._active_slot
            support = self._event_support(active, event)
            probability = self._probability(active, event)
            predictive_probability = probability if support else None
            surprise = (1.0 - probability) if support else None
            contradiction = support >= 1 and probability < 0.30
            unknown = support == 0

            if not contradiction and not (self.mode == "naive" and unknown):
                if unknown and self.mode == "robust":
                    matches = self._matching_slots(event)
                    if self._update_candidate_evidence(matches, event) is not None:
                        switched = True
                    else:
                        self._add(active, event)
                else:
                    self._add(active, event)
                if contradiction is False and support and probability >= 0.30:
                    self._recover_from_consistent_evidence(event)
                if not self._route_unresolved:
                    self._last_reason = "supported" if support >= 2 else "insufficient_support"
            else:
                self._route_unresolved = True
                matches = self._matching_slots(event)
                if self._update_candidate_evidence(matches, event) is not None:
                    switched = True
                    self._last_reason = "known_context_reused"
                else:
                    self._last_reason = "conflict" if contradiction else "unseen_cell"
                    allocated_slot, evicted_slot = self._update_novelty(event, contradiction=contradiction)
                    if allocated_slot is not None:
                        self._last_reason = "new_context_confirmed"
        else:
            matches = self._matching_slots(event)
            if matches:
                best = max(matches, key=lambda item: (item[1], self._last_used[item[0]]))[0]
                self._route_unresolved = True
                self._candidate_last_clock[best] = self._clock
                self._candidate_pending[best].append(event)
                evidence = list(self._candidate_pending[best])
                if (
                    len(evidence) >= min(2, self.confirm_events)
                    and len(self._distinct_cells(evidence)) >= 2
                ):
                    self._switch(best, evidence)
                    switched = True
                    self._last_reason = "known_context_reused"
                else:
                    self._last_reason = "insufficient_support"
            else:
                self._route_unresolved = True
                self._last_reason = "conflict" if any(self._active) else "insufficient_support"
                allocated_slot, evicted_slot = self._update_novelty(
                    event,
                    contradiction=any(self._active) and self._contradicts_any(event),
                )

        if self._active_slot >= 0 and not self._route_unresolved:
            self._last_used[self._active_slot] = self._clock
        return {
            "event": {"band": event[0], "action": command, "direction": direction},
            "censored": False,
            "active_slot": self._active_slot,
            "allocated_slot": allocated_slot,
            "evicted_slot": evicted_slot,
            "switched": switched,
            "reason": self._last_reason,
            "predictive_probability": predictive_probability,
            "surprise": surprise,
        }

    def _recover_from_consistent_evidence(self, event: tuple[int, int, int]) -> None:
        """Clear isolated provisional conflicts after repeat evidence at that cell."""

        if not self._route_unresolved or not self._novel_pending:
            return
        band, action, direction = event
        key = (band, action)
        if not any(item[:2] == key for item in self._novel_pending):
            return
        self._recovery_hits[key] = self._recovery_hits.get(key, 0) + 1
        if self._recovery_hits[key] < min(2, self.confirm_events):
            return
        remaining = [item for item in self._novel_pending if item[:2] != key]
        self._novel_pending.clear()
        self._novel_pending.extend(remaining)
        self._recovery_hits.pop(key, None)
        if not self._novel_pending:
            self._candidate_pending.clear()
            self._candidate_last_clock.clear()
            self._route_unresolved = False
            self._novel_last_clock = None
            self._recent.clear()
            self._recent.extend((event,))
            self._last_reason = "recovered_after_isolated_conflict"

    def _expire_stale_provisional_evidence(self) -> None:
        if self._novel_last_clock is not None and self._clock - self._novel_last_clock > self.window:
            self._novel_pending.clear()
            self._recovery_hits.clear()
            self._novel_last_clock = None
            if self._active_slot >= 0 and not self._candidate_pending:
                self._route_unresolved = False
                self._last_reason = "recent_conflict_expired"
        for slot in list(self._candidate_pending):
            if self._clock - self._candidate_last_clock.get(slot, 0) > self.window:
                del self._candidate_pending[slot]
                self._candidate_last_clock.pop(slot, None)
        if self._active_slot >= 0 and self._route_unresolved and not self._novel_pending and not self._candidate_pending:
            self._route_unresolved = False
            self._last_reason = "recent_conflict_expired"

    def _contradicts_any(self, event: tuple[int, int, int]) -> bool:
        return any(
            self._event_support(slot, event) >= 1 and self._probability(slot, event) < 0.30
            for slot in range(self.max_contexts)
            if self._active[slot]
        )

    def _route_scores(self) -> tuple[np.ndarray, np.ndarray]:
        log_scores = np.full(self.max_contexts, -np.inf, dtype=np.float64)
        supports = np.zeros(self.max_contexts, dtype=np.int64)
        for slot in range(self.max_contexts):
            if not self._active[slot]:
                continue
            score = 0.0
            for event in self._recent:
                support = self._event_support(slot, event)
                if support >= 2:
                    supports[slot] += 1
                # All candidates score the same recent events. An unsupported
                # cell therefore contributes the smoothed uniform likelihood,
                # instead of disappearing from the candidate's denominator.
                score += np.log(max(self._probability(slot, event), 1e-12))
            if self._recent:
                log_scores[slot] = score
        return log_scores, supports

    def route(self, observation: Any) -> dict[str, Any]:
        """Return the current context slot before the next action is taken."""

        self._observation_cell(observation, "observation")
        scores, supports = self._route_scores()
        available = np.flatnonzero(np.isfinite(scores))
        if self._route_unresolved or self._active_slot < 0:
            reason = self._last_reason
            if not len(available):
                reason = "insufficient_support" if not any(self._active) else reason
            return {
                "slot": -1,
                "confidence": 0.0,
                "support": 0.0,
                "recent_support": int(supports.max(initial=0)),
                "status": "unresolved",
                "reason": reason,
            }

        slot = self._active_slot
        support_events = int(supports[slot])
        if len(available):
            maximum = float(np.max(scores[available]))
            weights = np.exp(scores[available] - maximum)
            probabilities = weights / weights.sum()
            posterior = float(probabilities[np.flatnonzero(available == slot)[0]]) if slot in available else 0.0
        else:
            posterior = 0.0
        evidence_factor = support_events / (support_events + 2.0)
        confidence = float(np.clip(posterior * evidence_factor, 0.0, 1.0))
        recent_count = len(self._recent)
        support = float(support_events / recent_count) if recent_count else 0.0
        if support_events < 2:
            reason = "insufficient_support"
        elif posterior < 0.70:
            reason = "ambiguity"
        else:
            reason = "supported"
        self._last_used[slot] = self._clock
        resolved = posterior >= 0.70 and support_events >= 2
        return {
            "slot": slot if resolved else -1,
            "confidence": confidence,
            "support": support,
            "recent_support": support_events,
            "status": "resolved" if resolved else "uncertain",
            "reason": reason,
        }

    def predict_direction(self, observation: Any, action: Any) -> np.ndarray:
        """Return a four-way physical-direction prediction before the action."""

        x, _ = self._observation_cell(observation, "observation")
        command = self._action_index(action)
        routed = self.route(observation)
        if routed["slot"] < 0:
            return np.full(_DIRECTION_COUNT, 1.0 / _DIRECTION_COUNT, dtype=np.float64)
        row = self._counts[routed["slot"], x // 3, command].astype(np.float64)
        probabilities = row + self.alpha
        return probabilities / probabilities.sum()

    def explore_action(self, observation: Any, rng: np.random.Generator) -> int:
        """Choose among poorly supported commands as an optional active probe."""

        x, _ = self._observation_cell(observation, "observation")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        route = self.route(observation)
        if route["slot"] < 0:
            return int(rng.integers(_ACTION_COUNT))
        support = self._counts[route["slot"], x // 3].sum(axis=1)
        candidates = np.flatnonzero(support == support.min())
        return int(rng.choice(candidates))

    def reset_history(self) -> None:
        """Clear recent evidence and unresolved routing while keeping slot profiles."""

        self._recent.clear()
        self._novel_pending.clear()
        self._candidate_pending.clear()
        self._candidate_last_clock.clear()
        self._active_slot = -1
        self._route_unresolved = True
        self._recovery_hits.clear()
        self._novel_last_clock = None
        self._last_reason = "insufficient_support"

    def state_dict(self) -> dict[str, Any]:
        """Return a self-contained, JSON-compatible snapshot."""

        return {
            "version": _SNAPSHOT_VERSION,
            "config": {
                "max_contexts": self.max_contexts,
                "window": self.window,
                "mode": self.mode,
                "alpha": self.alpha,
                "confirm_events": self.confirm_events,
                "min_seed_cells": self.min_seed_cells,
                "max_support_per_cell": self.max_support_per_cell,
            },
            "counts": self._counts.tolist(),
            "active": list(self._active),
            "created_at": list(self._created_at),
            "last_used": list(self._last_used),
            "clock": self._clock,
            "active_slot": self._active_slot,
            "route_unresolved": self._route_unresolved,
            "recent": [list(event) for event in self._recent],
            "novel_pending": [list(event) for event in self._novel_pending],
            "candidate_pending": {str(slot): [list(event) for event in events]
                                  for slot, events in self._candidate_pending.items()},
            "candidate_last_clock": {str(slot): clock for slot, clock in self._candidate_last_clock.items()},
            "recovery_hits": {f"{band}:{action}": count for (band, action), count in self._recovery_hits.items()},
            "novel_last_clock": self._novel_last_clock,
            "allocations": self._allocations,
            "evictions": self._evictions,
            "switches": self._switches,
            "censored": self._censored,
            "last_reason": self._last_reason,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TaskFreeRouter":
        """Restore state produced by :meth:`state_dict`."""

        if not isinstance(state, Mapping) or state.get("version") != _SNAPSHOT_VERSION:
            raise ValueError("unsupported TaskFreeRouter state")
        config = state.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("TaskFreeRouter state is missing config")
        try:
            router = cls(**dict(config))
            counts = np.asarray(state["counts"], dtype=np.uint32)
            if counts.shape != router._counts.shape:
                raise ValueError("slot count array shape does not match config")
            router._counts[:] = counts
            active = list(state["active"])
            created_at = list(state["created_at"])
            last_used = list(state["last_used"])
            if len(active) != router.max_contexts or len(created_at) != router.max_contexts or len(last_used) != router.max_contexts:
                raise ValueError("slot metadata length does not match config")
            router._active = [bool(value) for value in active]
            router._created_at = [None if value is None else int(value) for value in created_at]
            router._last_used = [int(value) for value in last_used]
            router._clock = int(state["clock"])
            router._active_slot = int(state["active_slot"])
            if router._active_slot < -1 or router._active_slot >= router.max_contexts:
                raise ValueError("active slot is outside configured capacity")
            router._route_unresolved = bool(state["route_unresolved"])
            router._recent.extend(router._validated_events(state["recent"], router.window))
            router._novel_pending.extend(router._validated_events(state["novel_pending"], router.window))
            for slot_text, events in state.get("candidate_pending", {}).items():
                slot = int(slot_text)
                if slot not in range(router.max_contexts):
                    raise ValueError("candidate slot is outside configured capacity")
                router._candidate_pending[slot].extend(router._validated_events(events, router.window))
            for slot_text, clock in state.get("candidate_last_clock", {}).items():
                slot = int(slot_text)
                if slot not in range(router.max_contexts):
                    raise ValueError("candidate clock slot is outside configured capacity")
                router._candidate_last_clock[slot] = int(clock)
            for cell_text, count in state.get("recovery_hits", {}).items():
                band_text, action_text = cell_text.split(":", 1)
                cell = (int(band_text), int(action_text))
                if cell[0] not in range(_BAND_COUNT) or cell[1] not in range(_ACTION_COUNT):
                    raise ValueError("recovery cell is outside the C1 range")
                router._recovery_hits[cell] = int(count)
            novel_last_clock = state.get("novel_last_clock")
            router._novel_last_clock = None if novel_last_clock is None else int(novel_last_clock)
            router._allocations = int(state.get("allocations", 0))
            router._evictions = int(state.get("evictions", 0))
            router._switches = int(state.get("switches", 0))
            router._censored = int(state.get("censored", 0))
            router._last_reason = str(state.get("last_reason", "insufficient_support"))
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("TaskFreeRouter state is malformed") from exc
        return router

    @staticmethod
    def _validated_events(events: Any, limit: int) -> list[tuple[int, int, int]]:
        if not isinstance(events, (list, tuple)) or len(events) > limit:
            raise ValueError("event list exceeds configured window")
        result: list[tuple[int, int, int]] = []
        for event in events:
            if not isinstance(event, (list, tuple)) or len(event) != 3:
                raise ValueError("events must contain band, action, and direction")
            values = tuple(int(value) for value in event)
            if any(value < 0 for value in values) or values[0] >= _BAND_COUNT or values[1] >= _ACTION_COUNT or values[2] >= _DIRECTION_COUNT:
                raise ValueError("event value is outside the C1 range")
            result.append(values)
        return result

    def nbytes(self) -> dict[str, int]:
        """Estimate bytes owned by persistent profiles and bounded Python state."""

        profile_bytes = int(self._counts.nbytes)
        recent_bytes = sys.getsizeof(self._recent) + sum(
            sys.getsizeof(event) + sum(sys.getsizeof(value) for value in event) for event in self._recent
        )
        pending_bytes = sys.getsizeof(self._novel_pending) + sum(
            sys.getsizeof(event) + sum(sys.getsizeof(value) for value in event) for event in self._novel_pending
        )
        candidate_bytes = sys.getsizeof(self._candidate_pending)
        for slot, events in self._candidate_pending.items():
            candidate_bytes += sys.getsizeof(slot) + sys.getsizeof(events)
            candidate_bytes += sum(
                sys.getsizeof(event) + sum(sys.getsizeof(value) for value in event) for event in events
            )
        candidate_bytes += sys.getsizeof(self._candidate_last_clock)
        candidate_bytes += sum(sys.getsizeof(slot) + sys.getsizeof(clock) for slot, clock in self._candidate_last_clock.items())
        metadata_bytes = (
            sys.getsizeof(self._active) + sys.getsizeof(self._created_at) + sys.getsizeof(self._last_used)
            + sys.getsizeof(self._active_slot) + sys.getsizeof(self._clock) + sys.getsizeof(self._novel_last_clock)
        )
        breakdown = {
            "profile_counts": profile_bytes,
            "recent_window": int(recent_bytes),
            "novel_pending": int(pending_bytes),
            "candidate_pending": int(candidate_bytes),
            "recovery_hits": int(sys.getsizeof(self._recovery_hits) + sum(
                sys.getsizeof(key) + sum(sys.getsizeof(value) for value in key) + sys.getsizeof(value)
                for key, value in self._recovery_hits.items()
            )),
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


__all__ = ["TaskFreeRouter"]

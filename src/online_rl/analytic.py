"""A bounded, history-only action-rotation controller for navigation.

The controller compares observed position changes with the learned normal
response for each *executed* action. It does not inspect environment state,
task labels, or hidden context metadata.
"""

from __future__ import annotations

from collections import deque
from operator import index as as_index
from typing import Any, Mapping

import numpy as np


class AnalyticController:
    """Estimate a global quarter-turn from recent observable transitions.

    ``source_responses`` is the learned normal mapping, with one ``(x, y)``
    displacement per action in up, right, down, left order. ``observe`` takes
    only the pre-action observation, the action sent to the environment, and
    the resulting observation. Zero-displacement events are treated as wall
    collisions and do not enter the estimate.
    """

    _SNAPSHOT_VERSION = 1

    def __init__(
        self,
        source_responses: Any,
        window: int = 16,
        minimum_events: int = 5,
    ) -> None:
        self._window = self._positive_integer(window, "window")
        self._minimum_events = self._positive_integer(minimum_events, "minimum_events")
        if self._minimum_events > self._window:
            raise ValueError("minimum_events must be no greater than window")

        try:
            responses = np.asarray(source_responses, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_responses must be a finite array with shape (4, 2)") from exc
        if responses.shape != (4, 2) or not np.all(np.isfinite(responses)):
            raise ValueError("source_responses must be a finite array with shape (4, 2)")
        norms = np.linalg.norm(responses, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("each source response must have nonzero movement")

        self._source_directions = responses / norms[:, None]
        self._history: deque[tuple[float, float]] = deque(maxlen=self._window)

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
        return result

    @staticmethod
    def _action_index(action: Any) -> int:
        if isinstance(action, bool):
            raise ValueError("executed_action must be an integer in {0, 1, 2, 3}")
        try:
            result = as_index(action)
        except TypeError as exc:
            raise ValueError("executed_action must be an integer in {0, 1, 2, 3}") from exc
        if result not in (0, 1, 2, 3):
            raise ValueError("executed_action must be an integer in {0, 1, 2, 3}")
        return result

    @staticmethod
    def _position(observation: Any, name: str) -> np.ndarray:
        try:
            values = np.asarray(observation, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite observation with shape (4,)") from exc
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite observation with shape (4,)")
        return values[:2]

    def observe(self, before: Any, executed_action: int, after: Any) -> None:
        """Record one observable transition, ignoring a zero movement."""

        action = self._action_index(executed_action)
        before_position = self._position(before, "before")
        after_position = self._position(after, "after")
        movement = after_position - before_position
        movement_norm = float(np.linalg.norm(movement))
        if movement_norm <= 1e-12:
            return

        observed_direction = movement / movement_norm
        reference = self._source_directions[action]
        dot = float(np.dot(reference, observed_direction))
        # Screen coordinates have positive y downward, so this cross-product
        # sign is positive for a clockwise quarter-turn.
        cross = float(reference[0] * observed_direction[1] - reference[1] * observed_direction[0])
        self._history.append((dot, cross))

    @property
    def estimated_rotation(self) -> int:
        """Return the nearest observed rotation in ``{-1, 0, 1}`` quarter-turns."""

        if len(self._history) < self._minimum_events:
            return 0
        mean_dot = sum(event[0] for event in self._history) / len(self._history)
        mean_cross = sum(event[1] for event in self._history) / len(self._history)
        signatures = ((0, 1.0, 0.0), (1, 0.0, 1.0), (-1, 0.0, -1.0))
        return min(
            signatures,
            key=lambda signature: (mean_dot - signature[1]) ** 2 + (mean_cross - signature[2]) ** 2,
        )[0]

    def map_action(self, intended_action: int) -> int:
        """Map a fixed-foundation action back through the estimated rotation."""

        intended = self._action_index(intended_action)
        return (intended - self.estimated_rotation) % 4

    def clear_working_memory(self) -> None:
        """Forget recent transition evidence and return to the conservative estimate."""

        self._history.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return bounded, JSON-serializable state sufficient for restoration."""

        return {
            "version": self._SNAPSHOT_VERSION,
            "source_responses": (self._source_directions).tolist(),
            "window": self._window,
            "minimum_events": self._minimum_events,
            "events": [[dot, cross] for dot, cross in self._history],
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "AnalyticController":
        """Restore a controller from :meth:`snapshot` output."""

        if not isinstance(snapshot, Mapping):
            raise ValueError("snapshot must be a mapping")
        if snapshot.get("version") != cls._SNAPSHOT_VERSION:
            raise ValueError("unsupported analytic-controller snapshot version")
        try:
            controller = cls(
                snapshot["source_responses"],
                window=snapshot["window"],
                minimum_events=snapshot["minimum_events"],
            )
            events = snapshot["events"]
        except (KeyError, TypeError) as exc:
            raise ValueError("snapshot is missing controller state") from exc
        if not isinstance(events, (list, tuple)) or len(events) > controller._window:
            raise ValueError("snapshot events must fit within the configured window")
        for event in events:
            if not isinstance(event, (list, tuple)) or len(event) != 2:
                raise ValueError("each snapshot event must contain dot and cross values")
            try:
                dot, cross = float(event[0]), float(event[1])
            except (TypeError, ValueError) as exc:
                raise ValueError("snapshot events must be finite numeric pairs") from exc
            if not np.isfinite(dot) or not np.isfinite(cross):
                raise ValueError("snapshot events must be finite numeric pairs")
            controller._history.append((dot, cross))
        return controller

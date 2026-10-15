"""Unlabeled, state-conditioned transition models for the C1 task.

These controllers leave a supplied source policy in physical-action space and
use recent observed transitions to turn its desired physical-action
distribution into command-action probabilities. They never receive a context
label, a switch clock, reward, or an environment mapping helper.
"""

from __future__ import annotations

from collections import deque
from itertools import permutations
from operator import index as as_index
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


_ACTION_COUNT = 4
_BAND_COUNT = 3
_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_DELTAS_ARRAY = np.asarray(_DELTAS, dtype=np.int8)
_PERMUTATIONS = np.asarray(tuple(permutations(range(_ACTION_COUNT))), dtype=np.int8)
_REPO_SOURCE_DATA = Path(__file__).resolve().parents[2] / "data" / "pretrain_source.npz"


def load_source_prior(path: str | Path | None = None) -> np.ndarray:
    """Count observed source-environment responses by command and direction.

    The default reads only the source pretraining archive. Stationary wall
    responses are omitted because their physical direction is censored.
    Returned rows are source commands and columns are observed physical
    directions, both in up, right, down, left order.
    """

    source_path = _REPO_SOURCE_DATA if path is None else Path(path)
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            observations = np.asarray(archive["observations"], dtype=np.float64)
            actions = np.asarray(archive["actions"], dtype=np.int64)
            next_observations = np.asarray(archive["next_observations"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"could not load source transition data from {source_path}") from exc
    if (
        observations.ndim != 2
        or observations.shape[1] < 2
        or next_observations.shape != observations.shape
        or actions.shape != (observations.shape[0],)
        or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(next_observations))
        or np.any((actions < 0) | (actions >= _ACTION_COUNT))
    ):
        raise ValueError("source archive has malformed observations or actions")

    counts = np.zeros((_ACTION_COUNT, _ACTION_COUNT), dtype=np.float64)
    before = np.rint(observations[:, :2] * 8.0).astype(np.int64)
    after = np.rint(next_observations[:, :2] * 8.0).astype(np.int64)
    movement = after - before
    direction_by_delta = {delta: direction for direction, delta in enumerate(_DELTAS)}
    for command, (dx, dy) in zip(actions.tolist(), movement.tolist()):
        direction = direction_by_delta.get((int(dx), int(dy)))
        if direction is not None:
            counts[int(command), direction] += 1.0
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("source archive must contain a moving response for every command")
    return counts


class C1ModelController:
    """Adapt a fixed source actor using one of three bounded online models.

    ``base_probs`` in :meth:`action_probabilities` is the frozen source actor's
    distribution over desired physical directions. The returned vector is a
    distribution over commands to send to C1. ``source_prior`` is a 4 by 4
    matrix of source command-to-observed-direction counts; by default it is
    learned from ``data/pretrain_source.npz``.

    ``window_counts`` keeps independent recent direction counts for each
    x-band and command. ``window_assignment`` keeps a Bayesian posterior over
    all 24 command-to-direction permutations per band. Its prior favors the
    source-learned mapping while leaving every bijection possible.
    ``adaptive_assignment`` uses the same posterior with recency weighting,
    detects surprising changes from current evidence, and adds small
    information-gain exploration while assignments remain uncertain.
    """

    _SNAPSHOT_VERSION = 1
    _KINDS = ("window_counts", "window_assignment", "adaptive_assignment")

    def __init__(
        self,
        kind: str = "window_counts",
        window: int = 32,
        source_prior: Any | None = None,
        *,
        source_prior_strength: float = 1.0,
        assignment_source_prior: float = 0.999,
        observation_noise: float = 1e-3,
        adaptive_decay: float = 0.92,
        change_threshold: float = 0.05,
        change_confidence: float = 0.80,
        exploration_rate: float = 0.08,
    ) -> None:
        if kind not in self._KINDS:
            raise ValueError(f"kind must be one of {self._KINDS}")
        self.kind = kind
        self.window = self._positive_integer(window, "window")
        if not np.isfinite(source_prior_strength) or source_prior_strength <= 0:
            raise ValueError("source_prior_strength must be finite and positive")
        if not 0 < assignment_source_prior < 1:
            raise ValueError("assignment_source_prior must be in (0, 1)")
        if not 0 < observation_noise < 1:
            raise ValueError("observation_noise must be in (0, 1)")
        if not 0 < adaptive_decay <= 1:
            raise ValueError("adaptive_decay must be in (0, 1]")
        if not 0 < change_threshold < 1:
            raise ValueError("change_threshold must be in (0, 1)")
        if not 0 < change_confidence <= 1:
            raise ValueError("change_confidence must be in (0, 1]")
        if not 0 <= exploration_rate <= 1:
            raise ValueError("exploration_rate must be in [0, 1]")

        prior = load_source_prior() if source_prior is None else source_prior
        try:
            prior_array = np.asarray(prior, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_prior must be a finite 4 by 4 count matrix") from exc
        if (
            prior_array.shape != (_ACTION_COUNT, _ACTION_COUNT)
            or not np.all(np.isfinite(prior_array))
            or np.any(prior_array < 0)
            or np.any(prior_array.sum(axis=1) <= 0)
        ):
            raise ValueError("source_prior must be a finite non-negative 4 by 4 matrix with nonempty rows")

        self.source_prior_strength = float(source_prior_strength)
        self.assignment_source_prior = float(assignment_source_prior)
        self.observation_noise = float(observation_noise)
        self.adaptive_decay = float(adaptive_decay)
        self.change_threshold = float(change_threshold)
        self.change_confidence = float(change_confidence)
        self.exploration_rate = float(exploration_rate)
        self._source_probabilities = prior_array / prior_array.sum(axis=1, keepdims=True)
        self._source_counts = self._source_probabilities * self.source_prior_strength

        self._source_permutation = self._best_source_permutation(self._source_probabilities)
        self._prior_weights = self._make_assignment_prior()
        self._events: deque[tuple[int, int, int, int, int]] = deque(maxlen=self.window)
        self._counts = np.zeros((_BAND_COUNT, _ACTION_COUNT, _ACTION_COUNT), dtype=np.float64)
        self._posterior = np.tile(self._prior_weights, (_BAND_COUNT, 1))
        self._last_band = 0
        self._statistical_updates = 0
        self.change_detections = 0
        self.last_surprise = 0.0
        self._rebuild_counts()
        self._rebuild_posteriors()

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
            raise ValueError(f"{name} must be a finite normalized C1 observation with shape (4,)") from exc
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite normalized C1 observation with shape (4,)")
        if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
            raise ValueError(f"{name} coordinates must lie in [0, 1]")
        scaled = values[:2] * 8.0
        rounded = np.rint(scaled)
        if np.any(np.abs(scaled - rounded) > 1e-4):
            raise ValueError(f"{name} position must lie on the normalized 9 by 9 grid")
        return int(rounded[0]), int(rounded[1])

    @staticmethod
    def _action_index(action: Any) -> int:
        if isinstance(action, bool):
            raise ValueError("action must be an integer in {0, 1, 2, 3}")
        try:
            result = as_index(action)
        except TypeError as exc:
            raise ValueError("action must be an integer in {0, 1, 2, 3}") from exc
        if result not in range(_ACTION_COUNT):
            raise ValueError("action must be an integer in {0, 1, 2, 3}")
        return int(result)

    @staticmethod
    def _transition_direction(before: tuple[int, int], after: tuple[int, int]) -> int | None:
        delta = (after[0] - before[0], after[1] - before[1])
        for direction, candidate in enumerate(_DELTAS):
            if delta == candidate:
                return direction
        if delta == (0, 0):
            # A blocked observation is censored; count models get no direction
            # evidence, while assignment models still evaluate wall clipping.
            return None
        raise ValueError("C1 transitions must move one cell or remain blocked")

    @staticmethod
    def _clip(cell: tuple[int, int], direction: int) -> tuple[int, int]:
        dx, dy = _DELTAS[direction]
        return min(max(cell[0] + dx, 0), 8), min(max(cell[1] + dy, 0), 8)

    @staticmethod
    def _best_source_permutation(probabilities: np.ndarray) -> np.ndarray:
        # The source archive is normally a deterministic bijection. Exhaustive
        # scoring also gives a stable valid prior if a row is noisy.
        scores = np.log(
            np.maximum(probabilities[np.arange(_ACTION_COUNT)[None, :], _PERMUTATIONS], 1e-12)
        ).sum(axis=1)
        return _PERMUTATIONS[int(np.argmax(scores))].copy()

    def _make_assignment_prior(self) -> np.ndarray:
        prior = np.full(len(_PERMUTATIONS), (1.0 - self.assignment_source_prior) / (len(_PERMUTATIONS) - 1))
        source_matches = np.all(_PERMUTATIONS == self._source_permutation[None, :], axis=1)
        prior[source_matches] = self.assignment_source_prior
        return prior / prior.sum()

    def _event_direction(self, event: tuple[int, int, int, int, int]) -> int | None:
        x, y, action, next_x, next_y = event
        return self._transition_direction((x, y), (next_x, next_y))

    def _rebuild_counts(self) -> None:
        self._counts[:] = self._source_counts[None, :, :]
        for event in self._events:
            direction = self._event_direction(event)
            if direction is not None:
                x, _, action, _, _ = event
                self._counts[x // 3, action, direction] += 1.0

    def _event_likelihood(self, event: tuple[int, int, int, int, int]) -> np.ndarray:
        x, y, action, next_x, next_y = event
        predicted = np.clip(np.asarray([x, y]) + _DELTAS_ARRAY[_PERMUTATIONS[:, action]], 0, 8)
        matches = (predicted[:, 0] == next_x) & (predicted[:, 1] == next_y)
        return np.where(matches, 1.0 - self.observation_noise, self.observation_noise / 4.0)

    def _normalized_posterior(self, log_weights: np.ndarray) -> np.ndarray:
        maximum = float(np.max(log_weights))
        weights = np.exp(log_weights - maximum)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            return self._prior_weights.copy()
        return weights / total

    def _rebuild_posteriors(self) -> None:
        log_prior = np.log(np.maximum(self._prior_weights, 1e-300))
        self._posterior[:] = self._prior_weights[None, :]
        if not self._events or self.kind == "window_counts":
            return
        events = np.asarray(self._events, dtype=np.int8)
        directions = _PERMUTATIONS[:, events[:, 2]].T
        predicted = np.clip(events[:, None, :2] + _DELTAS_ARRAY[directions], 0, 8)
        matches = np.all(predicted == events[:, None, 3:5], axis=-1)
        log_likelihoods = np.log(np.where(matches, 1.0 - self.observation_noise, self.observation_noise / 4.0))
        for band in range(_BAND_COUNT):
            selected = log_likelihoods[events[:, 0] // 3 == band]
            count = len(selected)
            weights = self.adaptive_decay ** np.arange(count - 1, -1, -1) if self.kind == "adaptive_assignment" else np.ones(count)
            log_weights = log_prior + (selected * weights[:, None]).sum(axis=0)
            self._posterior[band] = self._normalized_posterior(log_weights)

    def _assignment_predictive_probability(self, event: tuple[int, int, int, int, int]) -> float:
        x = event[0]
        band = x // 3
        likelihood = self._event_likelihood(event)
        return float(np.dot(self._posterior[band], likelihood))

    def observe(self, obs: Any, action: Any, next_obs: Any, reward: float | None = None) -> None:
        """Update from a pre-action observation, executed command, and response.

        ``reward`` is accepted for runner compatibility but is intentionally
        ignored. No label, hidden state, or switch clock is used.
        """

        del reward
        before = self._observation_cell(obs, "obs")
        after = self._observation_cell(next_obs, "next_obs")
        command = self._action_index(action)
        self._transition_direction(before, after)
        event = (before[0], before[1], command, after[0], after[1])
        band = before[0] // 3
        self._last_band = band

        if self.kind == "adaptive_assignment":
            predictive = self._assignment_predictive_probability(event)
            posterior_confidence = float(np.max(self._posterior[band]))
            self.last_surprise = 1.0 - predictive
            band_evidence = sum(item[0] // 3 == band for item in self._events)
            if (
                band_evidence > 0
                and predictive < self.change_threshold
                and posterior_confidence >= self.change_confidence
            ):
                # Keep other bands' evidence; a change in one state region
                # should not erase observations from the other regions.
                self._events = deque(
                    (item for item in self._events if item[0] // 3 != band), maxlen=self.window
                )
                self.change_detections += 1

        self._events.append(event)
        self._statistical_updates += 1
        self._rebuild_counts()
        self._rebuild_posteriors()

    def _direction_matrix(self, band: int) -> np.ndarray:
        if self.kind == "window_counts":
            rows = self._counts[band]
            return rows / rows.sum(axis=1, keepdims=True)
        posterior = self._posterior[band]
        matrix = np.zeros((_ACTION_COUNT, _ACTION_COUNT), dtype=np.float64)
        for permutation, weight in zip(_PERMUTATIONS, posterior):
            matrix[np.arange(_ACTION_COUNT), permutation] += weight
        return matrix

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        values = probabilities[probabilities > 0]
        return float(-np.sum(values * np.log(values)))

    def _information_gain(self, cell: tuple[int, int], band: int) -> np.ndarray:
        posterior = self._posterior[band]
        gains = np.zeros(_ACTION_COUNT, dtype=np.float64)
        for action in range(_ACTION_COUNT):
            probabilities: dict[tuple[int, int], float] = {}
            for permutation, weight in zip(_PERMUTATIONS, posterior):
                destination = self._clip(cell, int(permutation[action]))
                probabilities[destination] = probabilities.get(destination, 0.0) + float(weight)
            gains[action] = self._entropy(np.fromiter(probabilities.values(), dtype=np.float64))
        return gains

    def action_probabilities(self, obs: Any, base_probs: Any) -> np.ndarray:
        """Map a source actor's desired directions to command probabilities."""

        cell = self._observation_cell(obs, "obs")
        try:
            base = np.asarray(base_probs, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("base_probs must be a finite non-negative vector of length 4") from exc
        if base.shape != (_ACTION_COUNT,) or not np.all(np.isfinite(base)) or np.any(base < 0):
            raise ValueError("base_probs must be a finite non-negative vector of length 4")
        base_total = float(base.sum())
        if not np.isfinite(base_total) or base_total <= 0:
            raise ValueError("base_probs must have positive total probability")
        base = base / base_total

        band = cell[0] // 3
        self._last_band = band
        direction_model = self._direction_matrix(band)
        command_probs = direction_model @ base
        command_total = float(command_probs.sum())
        command_probs = command_probs / command_total if command_total > 0 else np.full(4, 0.25)

        if self.kind == "adaptive_assignment" and self.exploration_rate > 0:
            gains = self._information_gain(cell, band)
            gain_total = float(gains.sum())
            prior_entropy = self._entropy(self._prior_weights)
            current_entropy = self._entropy(self._posterior[band])
            ambiguity = min(1.0, current_entropy / max(prior_entropy, 1e-12))
            if gain_total > 1e-12 and ambiguity > 1e-6:
                explore = gains / gain_total
                strength = self.exploration_rate * ambiguity
                command_probs = (1.0 - strength) * command_probs + strength * explore

        command_probs = np.maximum(command_probs, 0.0)
        return (command_probs / command_probs.sum()).astype(np.float64)

    @property
    def confidence(self) -> float:
        """Current mapping certainty in the last queried or observed band."""

        if self.kind == "window_counts":
            return float(np.mean(np.max(self._direction_matrix(self._last_band), axis=1)))
        return float(np.max(self._posterior[self._last_band]))

    @property
    def parameter_update_count(self) -> int:
        """Gradient/parameter updates; this statistical controller performs none."""

        return 0

    @property
    def statistical_update_count(self) -> int:
        """Number of observed transitions incorporated into working memory."""

        return self._statistical_updates

    @property
    def events(self) -> tuple[tuple[int, int, int, int, int], ...]:
        """A read-only view of the bounded recent transition records."""

        return tuple(self._events)

    def features(self) -> np.ndarray:
        """Return eight bounded diagnostics for the most recently used band.

        The first four values are per-command mapping confidence; the last
        four are normalized entropy for each command's direction posterior.
        """

        matrix = self._direction_matrix(self._last_band)
        confidence = np.max(matrix, axis=1)
        entropy = np.asarray([self._entropy(row) / np.log(_ACTION_COUNT) for row in matrix])
        return np.concatenate([confidence, entropy]).astype(np.float32)

    def clear_history(self) -> None:
        """Discard online evidence and restore source-learned prior beliefs."""

        self._events.clear()
        self._last_band = 0
        self._statistical_updates = 0
        self.change_detections = 0
        self.last_surprise = 0.0
        self._rebuild_counts()
        self._rebuild_posteriors()

    def memory_summary(self) -> dict[str, int | str]:
        """Report controller-owned arrays and bounded Python event storage."""

        arrays = [self._source_probabilities, self._source_counts, self._prior_weights,
                  self._counts, self._posterior, _PERMUTATIONS, _DELTAS_ARRAY]
        array_bytes = sum(array.nbytes for array in arrays)
        history_bytes = sys.getsizeof(self._events)
        for event in self._events:
            history_bytes += sys.getsizeof(event) + sum(sys.getsizeof(value) for value in event)
        return {
            "kind": self.kind,
            "window": self.window,
            "retained_events": len(self._events),
            "max_events": self.window,
            "state_array_bytes": int(array_bytes),
            "history_python_bytes": int(history_bytes),
            "estimated_controller_bytes": int(array_bytes + history_bytes),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return bounded JSON-serializable state for exact restoration."""

        return {
            "version": self._SNAPSHOT_VERSION,
            "kind": self.kind,
            "window": self.window,
            "source_prior": self._source_probabilities.tolist(),
            "source_prior_strength": self.source_prior_strength,
            "assignment_source_prior": self.assignment_source_prior,
            "observation_noise": self.observation_noise,
            "adaptive_decay": self.adaptive_decay,
            "change_threshold": self.change_threshold,
            "change_confidence": self.change_confidence,
            "exploration_rate": self.exploration_rate,
            "events": [list(event) for event in self._events],
            "statistical_updates": self._statistical_updates,
            "change_detections": self.change_detections,
            "last_surprise": self.last_surprise,
            "last_band": self._last_band,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "C1ModelController":
        """Restore a controller produced by :meth:`snapshot`."""

        if not isinstance(snapshot, Mapping) or snapshot.get("version") != cls._SNAPSHOT_VERSION:
            raise ValueError("unsupported C1 model-controller snapshot")
        try:
            controller = cls(
                kind=snapshot["kind"],
                window=snapshot["window"],
                source_prior=snapshot["source_prior"],
                source_prior_strength=snapshot["source_prior_strength"],
                assignment_source_prior=snapshot["assignment_source_prior"],
                observation_noise=snapshot["observation_noise"],
                adaptive_decay=snapshot["adaptive_decay"],
                change_threshold=snapshot["change_threshold"],
                change_confidence=snapshot["change_confidence"],
                exploration_rate=snapshot["exploration_rate"],
            )
            events = snapshot["events"]
        except (KeyError, TypeError) as exc:
            raise ValueError("snapshot is missing C1 model-controller state") from exc
        if not isinstance(events, (list, tuple)) or len(events) > controller.window:
            raise ValueError("snapshot events must fit within the configured window")
        for event in events:
            if not isinstance(event, (list, tuple)) or len(event) != 5:
                raise ValueError("each event must contain five grid-cell/action values")
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in event):
                raise ValueError("snapshot event values must be integers")
            x, y, action, next_x, next_y = (int(value) for value in event)
            if not (0 <= x <= 8 and 0 <= y <= 8 and 0 <= next_x <= 8 and 0 <= next_y <= 8):
                raise ValueError("snapshot event positions must lie on the 9 by 9 grid")
            controller._action_index(action)
            controller._transition_direction((x, y), (next_x, next_y))
            controller._events.append((x, y, action, next_x, next_y))
        for field in ("statistical_updates", "change_detections", "last_band"):
            value = snapshot.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"snapshot {field} must be a non-negative integer")
            if field == "last_band" and value >= _BAND_COUNT:
                raise ValueError("snapshot last_band must be between 0 and 2")
            target = {
                "statistical_updates": "_statistical_updates",
                "change_detections": "change_detections",
                "last_band": "_last_band",
            }[field]
            setattr(controller, target, value)
        surprise = snapshot.get("last_surprise", 0.0)
        if isinstance(surprise, bool) or not isinstance(surprise, (int, float)) or not np.isfinite(surprise):
            raise ValueError("snapshot last_surprise must be finite")
        controller.last_surprise = float(surprise)
        controller._rebuild_counts()
        controller._rebuild_posteriors()
        return controller


__all__ = ["C1ModelController", "load_source_prior"]

"""Infer C1 routes from a bounded history of observed action responses.

This is an infer-only diagnostic for replay after oracle-labeled calibration.
The six profile banks are fit from real transitions supplied through
``observe_calibration``; inference itself receives only observations, actions,
and next observations. The calibration labels deliberately carry provenance:
this is not blind training or a method for discovering the C1 tree.

Reported posterior confidence measures agreement with the calibrated
transition models. It is separate from policy-improvement reliability and has
no statistical guarantee.
"""
from __future__ import annotations

from collections import deque
from operator import index as as_index
from typing import Any

import numpy as np

from .c1_envs import context_index


PROFILE_COUNT = 6
BAND_COUNT = 3
ACTION_COUNT = 4
OUTCOME_COUNT = 5
OUTCOME_BLOCKED = 4
ROUTE_ROOT = 0
ROUTE_PARENT_A = 7
ROUTE_PARENT_B = 8

# Endpoint routes are kept separate from root 0; parents occupy 7 and 8.
# A0 through B2 therefore map to route IDs 1 through 6.
PROFILE_ROUTE_IDS = tuple(range(1, PROFILE_COUNT + 1))


class C1HistoryRouter:
    """Bounded empirical model bank for the six C1 transition profiles.

    Call :meth:`observe_calibration` only while the endpoint ID is available
    during oracle training. After calibration, use :meth:`observe` and
    :meth:`route`; those methods have no label or switch-hint input.

    Calibration records both the required five-outcome row counts and a
    Dirichlet distribution over physical directions. A blocked calibration
    outcome is partial direction evidence: its count is split over all
    directions that clip at the visible boundary. At inference, a blocked
    outcome has likelihood equal to the sum of those direction probabilities.
    """

    def __init__(
        self,
        window: int = 64,
        *,
        confidence_threshold: float | None = None,
        dirichlet_alpha: float = 1.0,
        evidence_decay: float = 0.94,
        minimum_informative_observations: int = 2,
        minimum_effective_evidence: float = 1.0,
        profile_threshold: float = 0.80,
        family_threshold: float = 0.80,
        confidence_margin: float = 0.15,
        informative_log_span: float = 0.10,
    ) -> None:
        if isinstance(window, bool):
            raise ValueError("window must be an integer between 1 and 64")
        try:
            window = as_index(window)
        except TypeError as exc:
            raise ValueError("window must be an integer between 1 and 64") from exc
        if not 1 <= window <= 64:
            raise ValueError("window must be an integer between 1 and 64")
        if not np.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if not np.isfinite(evidence_decay) or not 0 < evidence_decay <= 1:
            raise ValueError("evidence_decay must be in (0, 1]")
        if isinstance(minimum_informative_observations, bool) or minimum_informative_observations < 1:
            raise ValueError("minimum_informative_observations must be a positive integer")
        if not np.isfinite(minimum_effective_evidence) or minimum_effective_evidence < 0:
            raise ValueError("minimum_effective_evidence must be finite and non-negative")
        if confidence_threshold is not None:
            # Runner-facing shorthand used by the C1 replay harness.
            profile_threshold = confidence_threshold
            family_threshold = confidence_threshold
        for name, value in (
            ("profile_threshold", profile_threshold),
            ("family_threshold", family_threshold),
            ("confidence_margin", confidence_margin),
            ("informative_log_span", informative_log_span),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")

        self.window = int(window)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.evidence_decay = float(evidence_decay)
        self.minimum_informative_observations = int(minimum_informative_observations)
        self.minimum_effective_evidence = float(minimum_effective_evidence)
        self.profile_threshold = float(profile_threshold)
        self.family_threshold = float(family_threshold)
        self.confidence_margin = float(confidence_margin)
        self.informative_log_span = float(informative_log_span)

        # Required empirical outcome counts: [profile, band, action, outcome].
        self.outcome_counts = np.zeros(
            (PROFILE_COUNT, BAND_COUNT, ACTION_COUNT, OUTCOME_COUNT), dtype=np.int64
        )
        # Direction counts retain censored wall evidence as fractional counts.
        self.direction_counts = np.zeros(
            (PROFILE_COUNT, BAND_COUNT, ACTION_COUNT, ACTION_COUNT), dtype=np.float64
        )
        # Each entry is the actual (obs, action, next_obs) transition, with no label.
        self.events: deque[tuple[np.ndarray, int, np.ndarray]] = deque(maxlen=self.window)

        self.posterior = np.full(PROFILE_COUNT, 1.0 / PROFILE_COUNT, dtype=np.float64)
        self.family_posterior = np.asarray([0.5, 0.5], dtype=np.float64)
        self.informative_observations = 0
        self.effective_informative_evidence = 0.0
        self.last_confidence = 1.0 / PROFILE_COUNT
        self.last_family_confidence = 0.5
        self.last_margin = 0.0
        self._cached_route = ROUTE_ROOT

        # Diagnostics distinguish posterior computation from route requests.
        self.inference_calls = 0
        self.inference_cost = 0
        self.last_inference_cost = 0
        self.route_calls = 0
        self.unknown_routes = 0

    @property
    def confidence(self) -> float:
        """Current maximum profile posterior (recognition confidence only)."""

        return float(self.last_confidence)

    @staticmethod
    def _cell_observation(obs: Any) -> tuple[np.ndarray, tuple[int, int]]:
        values = np.asarray(obs, dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("C1 observations must be finite arrays of shape (4,)")
        if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
            raise ValueError("C1 observation coordinates must lie in [0, 1]")
        cell_values = np.rint(values[:2] * 8.0)
        if np.any(np.abs(values[:2] * 8.0 - cell_values) > 1e-4):
            raise ValueError("position coordinates must be normalized 9 by 9 grid cells")
        cell = (int(cell_values[0]), int(cell_values[1]))
        return values.astype(np.float32), cell

    @staticmethod
    def _action(action: Any) -> int:
        if isinstance(action, bool):
            raise ValueError("action must be an integer in {0, 1, 2, 3}")
        try:
            value = as_index(action)
        except TypeError as exc:
            raise ValueError("action must be an integer in {0, 1, 2, 3}") from exc
        if value not in range(ACTION_COUNT):
            raise ValueError("action must be an integer in {0, 1, 2, 3}")
        return int(value)

    @staticmethod
    def _transition_outcome(
        before_cell: tuple[int, int], after_cell: tuple[int, int]
    ) -> tuple[int, tuple[int, ...]]:
        x, y = before_cell
        dx = after_cell[0] - x
        dy = after_cell[1] - y
        if (dx, dy) == (0, -1):
            return 0, ()
        if (dx, dy) == (1, 0):
            return 1, ()
        if (dx, dy) == (0, 1):
            return 2, ()
        if (dx, dy) == (-1, 0):
            return 3, ()
        if (dx, dy) != (0, 0):
            raise ValueError("C1 transitions must move one cell or remain blocked")

        # Action order and physical direction order are both up, right, down, left.
        clipping_directions: list[int] = []
        if y == 0:
            clipping_directions.append(0)
        if x == 8:
            clipping_directions.append(1)
        if y == 8:
            clipping_directions.append(2)
        if x == 0:
            clipping_directions.append(3)
        return OUTCOME_BLOCKED, tuple(clipping_directions)

    def observe_calibration(self, endpoint: Any, obs: Any, action: Any, next_obs: Any) -> None:
        """Fit one real transition into the profile named during oracle learning.

        Endpoint provenance is deliberate: this method supports labeled
        calibration only. Inference paths never receive the endpoint ID.
        """

        # The runner's endpoint routes are 1..6 (root is 0), so numeric IDs
        # must not be interpreted as zero-based profile indices. Named keys
        # and C1Context values remain useful for calibration diagnostics.
        if isinstance(endpoint, bool):
            raise ValueError("endpoint route must be an integer in [1, 6] or a C1 profile key")
        try:
            endpoint_route = as_index(endpoint)
        except TypeError:
            profile = context_index(endpoint)
        else:
            if endpoint_route not in PROFILE_ROUTE_IDS:
                raise ValueError("endpoint route must be an integer in [1, 6]")
            profile = int(endpoint_route) - 1
        before, before_cell = self._cell_observation(obs)
        after, after_cell = self._cell_observation(next_obs)
        action_index = self._action(action)
        outcome, clipping_directions = self._transition_outcome(before_cell, after_cell)
        x, _ = before_cell
        band = x // 3
        self.outcome_counts[profile, band, action_index, outcome] += 1
        if outcome == OUTCOME_BLOCKED:
            if clipping_directions:
                share = 1.0 / len(clipping_directions)
                self.direction_counts[profile, band, action_index, list(clipping_directions)] += share
        else:
            self.direction_counts[profile, band, action_index, outcome] += 1.0

    def observe(self, obs: Any, action: Any, next_obs: Any) -> None:
        """Append an unlabeled transition and refresh cached route evidence."""

        before, before_cell = self._cell_observation(obs)
        after, after_cell = self._cell_observation(next_obs)
        action_index = self._action(action)
        # Validate transitions at the boundary of the inference API too.
        self._transition_outcome(before_cell, after_cell)
        self.events.append((before.copy(), action_index, after.copy()))
        self._infer()

    def _direction_probabilities(self, profile: int, band: int, action: int) -> np.ndarray:
        counts = self.direction_counts[profile, band, action]
        alpha = self.dirichlet_alpha
        probabilities = counts + alpha
        return probabilities / probabilities.sum()

    def _event_likelihoods(
        self, before_cell: tuple[int, int], action: int, after_cell: tuple[int, int]
    ) -> np.ndarray:
        outcome, clipping_directions = self._transition_outcome(before_cell, after_cell)
        band = before_cell[0] // 3
        likelihoods = np.empty(PROFILE_COUNT, dtype=np.float64)
        for profile in range(PROFILE_COUNT):
            directions = self._direction_probabilities(profile, band, action)
            if outcome == OUTCOME_BLOCKED:
                # A blocked move censors direction. Every physical direction
                # that clips at this visible boundary remains compatible.
                likelihoods[profile] = (
                    float(directions[list(clipping_directions)].sum())
                    if clipping_directions
                    else 0.25
                )
            else:
                likelihoods[profile] = float(directions[outcome])
        return likelihoods

    def _infer(self, *, count_cost: bool = True) -> None:
        log_scores = np.zeros(PROFILE_COUNT, dtype=np.float64)
        informative = 0
        effective_informative = 0.0
        epsilon = np.finfo(np.float64).tiny
        cost = 0
        event_list = list(self.events)
        for index, (before, action, after) in enumerate(event_list):
            _, before_cell = self._cell_observation(before)
            _, after_cell = self._cell_observation(after)
            likelihoods = self._event_likelihoods(before_cell, action, after_cell)
            log_likelihoods = np.log(np.maximum(likelihoods, epsilon))
            age = len(event_list) - index - 1
            weight = self.evidence_decay ** age
            log_scores += weight * log_likelihoods
            span = float(log_likelihoods.max() - log_likelihoods.min())
            if span >= self.informative_log_span:
                informative += 1
                effective_informative += weight
            cost += PROFILE_COUNT

        shifted = log_scores - float(log_scores.max())
        probabilities = np.exp(shifted)
        self.posterior = probabilities / probabilities.sum()
        self.family_posterior = np.asarray(
            [self.posterior[:3].sum(), self.posterior[3:].sum()], dtype=np.float64
        )
        self.informative_observations = informative
        self.effective_informative_evidence = effective_informative
        order = np.argsort(self.posterior)[::-1]
        self.last_confidence = float(self.posterior[int(order[0])])
        self.last_margin = (
            float(self.posterior[int(order[0])] - self.posterior[int(order[1])])
            if len(order) > 1
            else 1.0
        )
        self.last_family_confidence = float(self.family_posterior.max())

        sufficient_evidence = (
            informative >= self.minimum_informative_observations
            and effective_informative >= self.minimum_effective_evidence
        )
        self._cached_route = ROUTE_ROOT
        if sufficient_evidence and self.last_confidence >= self.profile_threshold and self.last_margin >= self.confidence_margin:
            self._cached_route = PROFILE_ROUTE_IDS[int(order[0])]
        else:
            family_order = np.argsort(self.family_posterior)[::-1]
            family = int(family_order[0])
            family_margin = float(self.family_posterior[family_order[0]] - self.family_posterior[family_order[1]])
            if (
                sufficient_evidence
                and self.last_family_confidence >= self.family_threshold
                and family_margin >= self.confidence_margin
            ):
                self._cached_route = ROUTE_PARENT_A if family == 0 else ROUTE_PARENT_B

        self.last_inference_cost = cost
        if count_cost:
            self.inference_calls += 1
            self.inference_cost += cost

    def route(self, obs: Any = None) -> int:
        """Return a cached endpoint, family parent, or unknown/root route.

        ``obs`` is validated for a stable runner API but does not add evidence;
        only :meth:`observe` can update inference. Endpoint IDs use 1 through
        6, family parents use 7 and 8, and 0 means unknown/root.
        """

        if obs is not None:
            self._cell_observation(obs)
        self.route_calls += 1
        route_id = int(self._cached_route)
        if route_id == ROUTE_ROOT:
            self.unknown_routes += 1
        return route_id

    def features(self) -> np.ndarray:
        """Return eight normalized classification features for policy input."""

        entropy = -float(np.sum(self.posterior * np.log(np.maximum(self.posterior, 1e-12))))
        normalized_entropy = entropy / np.log(PROFILE_COUNT)
        calibrated = np.any(self.outcome_counts.sum(axis=(1, 2, 3)) > 0)
        coverage = float(calibrated.sum()) / PROFILE_COUNT
        return np.asarray(
            [
                self.last_confidence,
                self.last_family_confidence,
                len(self.events) / self.window,
                min(1.0, self.effective_informative_evidence / self.window),
                float(np.clip(normalized_entropy, 0.0, 1.0)),
                coverage,
                float(1 <= self._cached_route <= PROFILE_COUNT),
                float(self._cached_route in (ROUTE_PARENT_A, ROUTE_PARENT_B)),
            ],
            dtype=np.float32,
        )

    def clear_history(self) -> None:
        """Clear recognition history and cache while retaining calibration."""

        self.events.clear()
        self.posterior = np.full(PROFILE_COUNT, 1.0 / PROFILE_COUNT, dtype=np.float64)
        self.family_posterior = np.asarray([0.5, 0.5], dtype=np.float64)
        self.informative_observations = 0
        self.effective_informative_evidence = 0.0
        self.last_confidence = 1.0 / PROFILE_COUNT
        self.last_family_confidence = 0.5
        self.last_margin = 0.0
        self._cached_route = ROUTE_ROOT
        self.last_inference_cost = 0

    def clear_working_memory(self) -> None:
        """Compatibility alias for clearing only recent recognition history."""

        self.clear_history()

    def diagnostics(self) -> dict[str, int | float]:
        """Return route, inference-cost, and current-confidence counters."""

        return {
            "inference_calls": int(self.inference_calls),
            "inference_cost": int(self.inference_cost),
            "last_inference_cost": int(self.last_inference_cost),
            "route_calls": int(self.route_calls),
            "unknown_routes": int(self.unknown_routes),
            "history_length": len(self.events),
            "informative_observations": int(self.informative_observations),
            "effective_informative_evidence": float(self.effective_informative_evidence),
            "profile_confidence": float(self.last_confidence),
            "family_confidence": float(self.last_family_confidence),
            "cached_route": int(self._cached_route),
        }

    def snapshot(self) -> dict[str, Any]:
        """Serialize all empirical arrays, bounded events, settings, and counters."""

        return {
            "version": 1,
            "window": self.window,
            "dirichlet_alpha": self.dirichlet_alpha,
            "evidence_decay": self.evidence_decay,
            "minimum_informative_observations": self.minimum_informative_observations,
            "minimum_effective_evidence": self.minimum_effective_evidence,
            "profile_threshold": self.profile_threshold,
            "family_threshold": self.family_threshold,
            "confidence_margin": self.confidence_margin,
            "informative_log_span": self.informative_log_span,
            "outcome_counts": self.outcome_counts.tolist(),
            "direction_counts": self.direction_counts.tolist(),
            "events": [
                {"obs": obs.tolist(), "action": action, "next_obs": next_obs.tolist()}
                for obs, action, next_obs in self.events
            ],
            "inference_calls": self.inference_calls,
            "inference_cost": self.inference_cost,
            "last_inference_cost": self.last_inference_cost,
            "route_calls": self.route_calls,
            "unknown_routes": self.unknown_routes,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "C1HistoryRouter":
        """Restore the calibration arrays and bounded unlabeled event history."""

        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("unsupported C1 router snapshot")
        router = cls(
            window=snapshot["window"],
            dirichlet_alpha=snapshot["dirichlet_alpha"],
            evidence_decay=snapshot["evidence_decay"],
            minimum_informative_observations=snapshot["minimum_informative_observations"],
            minimum_effective_evidence=snapshot["minimum_effective_evidence"],
            profile_threshold=snapshot["profile_threshold"],
            family_threshold=snapshot["family_threshold"],
            confidence_margin=snapshot["confidence_margin"],
            informative_log_span=snapshot["informative_log_span"],
        )
        outcome_counts = np.asarray(snapshot["outcome_counts"], dtype=np.int64)
        direction_counts = np.asarray(snapshot["direction_counts"], dtype=np.float64)
        if outcome_counts.shape != router.outcome_counts.shape:
            raise ValueError("snapshot outcome_counts have the wrong shape")
        if direction_counts.shape != router.direction_counts.shape:
            raise ValueError("snapshot direction_counts have the wrong shape")
        if np.any(outcome_counts < 0) or np.any(direction_counts < 0) or not np.all(np.isfinite(direction_counts)):
            raise ValueError("snapshot contains invalid calibration counts")
        router.outcome_counts[:] = outcome_counts
        router.direction_counts[:] = direction_counts

        for event in snapshot["events"]:
            before, _ = router._cell_observation(event["obs"])
            after, _ = router._cell_observation(event["next_obs"])
            action = router._action(event["action"])
            router._transition_outcome(
                router._cell_observation(before)[1], router._cell_observation(after)[1]
            )
            router.events.append((before.copy(), action, after.copy()))
        router._infer(count_cost=False)
        for field in (
            "inference_calls",
            "inference_cost",
            "last_inference_cost",
            "route_calls",
            "unknown_routes",
        ):
            value = snapshot.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"snapshot {field} must be a non-negative integer")
            setattr(router, field, value)
        return router


def restore_c1_router(snapshot: dict[str, Any]) -> C1HistoryRouter:
    """Rebuild a :class:`C1HistoryRouter` from :meth:`snapshot` output."""

    return C1HistoryRouter.from_snapshot(snapshot)


__all__ = [
    "C1HistoryRouter",
    "OUTCOME_BLOCKED",
    "PROFILE_ROUTE_IDS",
    "ROUTE_PARENT_A",
    "ROUTE_PARENT_B",
    "ROUTE_ROOT",
    "restore_c1_router",
]

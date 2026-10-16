"""Controlled navigation environments for C1 context experiments.

The context changes the action-to-movement mapping while keeping the agent's
observation and ``info`` interfaces identical across contexts.  Context labels
and observer-only region counters are available through environment helpers,
not through agent-visible data.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as as_index
from typing import Any

import gymnasium as gym
import numpy as np

from online_rl.envs import ContextSpec, NavigationEnv


@dataclass(frozen=True)
class C1Context:
    """One hidden C1 family and local-band variant."""

    family: str
    variant: int

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or self.family not in ("A", "B"):
            raise ValueError("family must be 'A' or 'B'")
        if isinstance(self.variant, bool):
            raise ValueError("variant must be an integer in {0, 1, 2}")
        try:
            variant = as_index(self.variant)
        except TypeError as exc:
            raise ValueError("variant must be an integer in {0, 1, 2}") from exc
        if variant not in (0, 1, 2):
            raise ValueError("variant must be an integer in {0, 1, 2}")
        object.__setattr__(self, "variant", variant)


_CONTEXT_KEYS = ("A0", "A1", "A2", "B0", "B1", "B2")
C1_CONTEXTS: dict[str, C1Context] = {
    key: C1Context(key[0], int(key[1])) for key in _CONTEXT_KEYS
}


def c1_contexts() -> dict[str, C1Context]:
    """Return a fresh registry of the six externally named C1 contexts."""

    return dict(C1_CONTEXTS)


def context_name(context: C1Context | str | int) -> str:
    """Return the canonical context key, accepting a spec, key, or index."""

    if isinstance(context, C1Context):
        key = f"{context.family}{context.variant}"
    elif isinstance(context, str):
        key = context
    else:
        if isinstance(context, bool):
            raise ValueError("context index must be an integer in [0, 5]")
        try:
            context_index_value = as_index(context)
        except TypeError as exc:
            raise ValueError("context index must be an integer in [0, 5]") from exc
        if not 0 <= context_index_value < len(_CONTEXT_KEYS):
            raise ValueError("context index must be an integer in [0, 5]")
        return _CONTEXT_KEYS[context_index_value]

    if key not in C1_CONTEXTS:
        raise ValueError(f"unknown C1 context: {key!r}")
    return key


def context_index(context: C1Context | str) -> int:
    """Return the stable zero-based index for a C1 context."""

    return _CONTEXT_KEYS.index(context_name(context))


def context_from_index(index: int) -> C1Context:
    """Return the C1 context at a stable zero-based index."""

    return C1_CONTEXTS[context_name(index)]


def mapping_for_state(spec: C1Context, x: int) -> tuple[int, int, int, int]:
    """Return the four physical directions selected by actions at grid x.

    Entries are indexed by commanded action (up, right, down, left), and each
    entry is a physical direction in the same order.  The family permutation
    is applied first; the selected local swap is then applied to its result.
    This helper is intended for diagnostics and tests, not agent routing.
    """

    if not isinstance(spec, C1Context):
        raise TypeError("spec must be a C1Context")
    if isinstance(x, bool):
        raise ValueError("x must be an integer in [0, 8]")
    try:
        x_value = as_index(x)
    except TypeError as exc:
        raise ValueError("x must be an integer in [0, 8]") from exc
    if not 0 <= x_value <= 8:
        raise ValueError("x must be an integer in [0, 8]")

    # Each permutation maps a commanded direction to a physical direction.
    family_permutation = list(range(4))
    if spec.family == "A":
        family_permutation[0], family_permutation[1] = 1, 0
        local_pair = (2, 3)
    else:
        family_permutation[0], family_permutation[3] = 3, 0
        local_pair = (1, 2)

    band_start = spec.variant * 3
    if band_start <= x_value < band_start + 3:
        local_permutation = list(range(4))
        left, right = local_pair
        local_permutation[left], local_permutation[right] = right, left
        return tuple(local_permutation[direction] for direction in family_permutation)  # type: ignore[return-value]

    return tuple(family_permutation)  # type: ignore[return-value]


class C1NavigationEnv(NavigationEnv):
    """A 9 by 9 controlled navigation task with six hidden contexts.

    Reset starts at x = 0 or x = 8 with equal probability and samples the
    start and goal y coordinates independently.  A successful episode must
    therefore traverse all three full-height x bands.
    """

    def __init__(self, context: C1Context | None = None) -> None:
        # Keep the common spaces and reward/info helpers from NavigationEnv;
        # C1 owns the transition map and uses fixed task dimensions.
        super().__init__(context=ContextSpec("normal", 0), grid_size=9, max_steps=64)
        self.context = C1Context("A", 0) if context is None else context
        if not isinstance(self.context, C1Context):
            raise TypeError("context must be a C1Context or None")
        self._observer_region_visits = [0, 0, 0]

    def set_context(self, spec: C1Context) -> None:
        """Set the hidden transition context for subsequent steps."""

        if not isinstance(spec, C1Context):
            raise TypeError("spec must be a C1Context")
        self.context = spec

    def observer_counters(self) -> dict[str, int]:
        """Return cumulative observer-only state visits for the three x bands."""

        return {
            f"band_{band}_visits": visits
            for band, visits in enumerate(self._observer_region_visits)
        }

    def _record_region_visit(self) -> None:
        x, _ = self._position
        self._observer_region_visits[x // 3] += 1

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, bool | int]]:
        """Reset to a random edge and its opposite, using Gymnasium's RNG."""

        if options:
            raise ValueError("reset options are not supported")
        gym.Env.reset(self, seed=seed)

        start_x = 0 if int(self.np_random.integers(2)) == 0 else 8
        start_y = int(self.np_random.integers(9))
        goal_y = int(self.np_random.integers(9))
        self._position = (start_x, start_y)
        self._goal = (8 - start_x, goal_y)
        self._episode_length = 0
        self._has_reset = True
        self._record_region_visit()
        return self._observation(), self._info(success=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        """Apply one context-dependent action and return Gymnasium's result."""

        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        if isinstance(action, bool):
            raise ValueError("action must be an integer in {0, 1, 2, 3}")
        try:
            action_index = as_index(action)
        except TypeError as exc:
            raise ValueError("action must be an integer in {0, 1, 2, 3}") from exc
        if not self.action_space.contains(action_index):
            raise ValueError("action must be an integer in {0, 1, 2, 3}")

        previous_distance = self._distance()
        x, y = self._position
        physical_direction = mapping_for_state(self.context, x)[action_index]
        dx, dy = self._DELTAS[physical_direction]
        self._position = (
            min(max(x + dx, 0), self.grid_size - 1),
            min(max(y + dy, 0), self.grid_size - 1),
        )
        self._episode_length += 1
        self._record_region_visit()

        distance = self._distance()
        success = distance == 0
        reward = 0.25 * (previous_distance - distance) - 0.03
        if success:
            reward += 1.0
        terminated = success
        truncated = self._episode_length >= self.max_steps and not terminated
        return self._observation(), float(reward), terminated, truncated, self._info(success)


__all__ = [
    "C1Context",
    "C1NavigationEnv",
    "C1_CONTEXTS",
    "c1_contexts",
    "context_from_index",
    "context_index",
    "context_name",
    "mapping_for_state",
]

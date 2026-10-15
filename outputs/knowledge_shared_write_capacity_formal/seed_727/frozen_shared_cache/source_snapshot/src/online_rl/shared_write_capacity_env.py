"""Deterministic 9 by 9 contexts for the bounded shared-write cache study.

The visible observation and episode interface are inherited from C1.  The
only added variable changes which adjacent physical directions are exchanged
inside one three-column band; it is never exposed to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as as_index
from typing import Any

import numpy as np

from .c1_envs import C1Context, C1NavigationEnv


@dataclass(frozen=True)
class CapacityContext(C1Context):
    """A C1 family/band context with one controlled local-map variant."""

    local_mode: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.local_mode, bool):
            raise ValueError("local_mode must be an integer in {0, 1}")
        try:
            mode = as_index(self.local_mode)
        except TypeError as exc:
            raise ValueError("local_mode must be an integer in {0, 1}") from exc
        if mode not in (0, 1):
            raise ValueError("local_mode must be an integer in {0, 1}")
        object.__setattr__(self, "local_mode", mode)


CAPACITY_CONTEXTS: dict[str, CapacityContext] = {
    f"{family}_v{variant}_m{mode}": CapacityContext(family, variant, mode)
    for family in ("A", "B")
    for variant in range(3)
    for mode in range(2)
}


def capacity_context_name(context: CapacityContext | str) -> str:
    """Return the canonical capacity-context key."""

    if isinstance(context, CapacityContext):
        key = f"{context.family}_v{context.variant}_m{context.local_mode}"
    elif isinstance(context, str):
        key = context
    else:
        raise TypeError("context must be a CapacityContext or string key")
    if key not in CAPACITY_CONTEXTS:
        raise ValueError(f"unknown capacity context: {key!r}")
    return key


def mapping_for_capacity_state(
    spec: CapacityContext, x: int
) -> tuple[int, int, int, int]:
    """Return action-to-physical-direction map, applying family then local swap."""

    if not isinstance(spec, CapacityContext):
        raise TypeError("spec must be a CapacityContext")
    if isinstance(x, bool):
        raise ValueError("x must be an integer in [0, 8]")
    try:
        x_value = as_index(x)
    except TypeError as exc:
        raise ValueError("x must be an integer in [0, 8]") from exc
    if not 0 <= x_value <= 8:
        raise ValueError("x must be an integer in [0, 8]")

    family_permutation = list(range(4))
    if spec.family == "A":
        family_permutation[0], family_permutation[1] = 1, 0
        local_pair = (2, 3) if spec.local_mode == 0 else (1, 2)
    else:
        family_permutation[0], family_permutation[3] = 3, 0
        local_pair = (1, 2) if spec.local_mode == 0 else (2, 3)

    band_start = spec.variant * 3
    if band_start <= x_value < band_start + 3:
        local_permutation = list(range(4))
        left, right = local_pair
        local_permutation[left], local_permutation[right] = right, left
        return tuple(local_permutation[direction] for direction in family_permutation)  # type: ignore[return-value]
    return tuple(family_permutation)  # type: ignore[return-value]


class SharedWriteCapacityEnv(C1NavigationEnv):
    """C1-compatible environment using one of the 12 capacity contexts."""

    def __init__(self, context: CapacityContext) -> None:
        if not isinstance(context, CapacityContext):
            raise TypeError("context must be a CapacityContext")
        super().__init__(context)

    def set_context(self, spec: CapacityContext) -> None:
        if not isinstance(spec, CapacityContext):
            raise TypeError("spec must be a CapacityContext")
        self.context = spec

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        """Apply the capacity context while preserving the C1 step contract."""

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
        physical_direction = mapping_for_capacity_state(self.context, x)[action_index]
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
    "CapacityContext",
    "CAPACITY_CONTEXTS",
    "SharedWriteCapacityEnv",
    "capacity_context_name",
    "mapping_for_capacity_state",
]

"""Reward-aware C1 environments for shared-write diagnostics.

These environments keep the C1 observation and action interfaces while
changing the hidden transition context and reward preference. Task metadata is
held by the runner; it is never returned in observations or ``info``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from operator import index as as_index
import numpy as np

from .c1_envs import C1Context, C1NavigationEnv, mapping_for_state


@dataclass(frozen=True)
class TaskSpec:
    """A hidden C1 context and the sign of its reward preference."""

    family: str
    variant: int
    preference: int

    def __post_init__(self) -> None:
        # Delegate family and variant validation to the canonical C1 type.
        context = C1Context(self.family, self.variant)
        if isinstance(self.preference, bool):
            raise ValueError("preference must be +1 or -1")
        try:
            preference = as_index(self.preference)
        except TypeError as exc:
            raise ValueError("preference must be +1 or -1") from exc
        if preference not in (-1, 1):
            raise ValueError("preference must be +1 or -1")
        object.__setattr__(self, "family", context.family)
        object.__setattr__(self, "variant", context.variant)
        object.__setattr__(self, "preference", preference)


def _validate_spec(spec: TaskSpec) -> TaskSpec:
    if not isinstance(spec, TaskSpec):
        raise TypeError("spec must be a TaskSpec")
    return spec


def task_name(spec: TaskSpec) -> str:
    """Return the runner-facing canonical name, such as ``A0+`` or ``B2-``."""

    spec = _validate_spec(spec)
    sign = "+" if spec.preference == 1 else "-"
    return f"{spec.family}{spec.variant}{sign}"


def task_from_name(name: str) -> TaskSpec:
    """Parse a canonical runner schedule name into its hidden task spec."""

    if not isinstance(name, str):
        raise TypeError("task name must be a string")
    if len(name) != 3 or name[0] not in ("A", "B") or name[1] not in "012" or name[2] not in "+-":
        raise ValueError("task name must look like 'A0+' or 'B2-'")
    return TaskSpec(name[0], int(name[1]), 1 if name[2] == "+" else -1)


_TASKS: dict[str, TaskSpec] = {
    task_name(TaskSpec(family, variant, preference)): TaskSpec(family, variant, preference)
    for family in ("A", "B")
    for variant in range(3)
    for preference in (1, -1)
}


def task_specs() -> dict[str, TaskSpec]:
    """Return the 12 externally named A/B, variant, and preference tasks."""

    return dict(_TASKS)


TASK_SPECS = _TASKS
DOMAINS = ("navigation", "detour")


class WriteDecisionNavigationEnv(C1NavigationEnv):
    """C1 navigation dynamics with a hidden signed reward preference."""

    domain = "navigation"

    def __init__(self, spec: TaskSpec) -> None:
        spec = _validate_spec(spec)
        self._preference = spec.preference
        super().__init__(C1Context(spec.family, spec.variant))

    def set_task(self, spec: TaskSpec) -> None:
        """Change hidden context and reward sign without resetting the episode."""

        spec = _validate_spec(spec)
        self.set_context(C1Context(spec.family, spec.variant))
        self._preference = spec.preference

    def _info(self, success: bool) -> dict[str, bool | int]:
        # Goal status is evaluator metadata, not a reward-derived success label.
        return {
            "goal_reached": bool(success),
            "episode_length": self._episode_length,
            "distance": self._distance(),
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        observation, base_reward, terminated, truncated, info = super().step(action)
        return observation, float(self._preference * base_reward), terminated, truncated, info


class DetourWriteDecisionEnv(WriteDecisionNavigationEnv):
    """C1 action mapping on a 9 by 9 grid with one fixed wall and one opening.

    The observation remains ``[x, y, goal_x, goal_y]``. Movement into a wall
    cell is censored, and detour reward uses shortest-path distance around the
    fixed geometry.
    """

    domain = "detour"
    WALL_CELLS = frozenset((4, y) for y in range(9) if y != 4)
    STEP_COST = 0.03
    POTENTIAL_SCALE = 0.10

    def _distance(self, position: tuple[int, int] | None = None) -> int:
        start = self._position if position is None else position
        if start in self.WALL_CELLS or self._goal in self.WALL_CELLS:
            raise RuntimeError("agent and goal must remain outside wall cells")
        pending: deque[tuple[int, int, int]] = deque([(start[0], start[1], 0)])
        seen = {start}
        while pending:
            x, y, distance = pending.popleft()
            if (x, y) == self._goal:
                return distance
            for dx, dy in self._DELTAS:
                neighbor = (x + dx, y + dy)
                if (
                    0 <= neighbor[0] < self.grid_size
                    and 0 <= neighbor[1] < self.grid_size
                    and neighbor not in self.WALL_CELLS
                    and neighbor not in seen
                ):
                    seen.add(neighbor)
                    pending.append((neighbor[0], neighbor[1], distance + 1))
        raise RuntimeError("detour wall geometry unexpectedly disconnects the grid")

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
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
        destination = (
            min(max(x + dx, 0), self.grid_size - 1),
            min(max(y + dy, 0), self.grid_size - 1),
        )
        if destination not in self.WALL_CELLS:
            self._position = destination
        self._episode_length += 1
        self._record_region_visit()

        distance = self._distance()
        goal_reached = distance == 0
        base_reward = -self.STEP_COST + self.POTENTIAL_SCALE * (previous_distance - distance)
        if goal_reached:
            base_reward += 1.0
        terminated = goal_reached
        truncated = self._episode_length >= self.max_steps and not terminated
        info = self._info(goal_reached)
        return (
            self._observation(),
            float(self._preference * base_reward),
            terminated,
            truncated,
            info,
        )


def make_env(spec: TaskSpec, domain: str = "navigation") -> WriteDecisionNavigationEnv:
    """Build one reward-aware environment for a runner-supplied task spec."""

    spec = _validate_spec(spec)
    if domain == "navigation":
        return WriteDecisionNavigationEnv(spec)
    if domain == "detour":
        return DetourWriteDecisionEnv(spec)
    raise ValueError(f"domain must be one of {DOMAINS}")


__all__ = [
    "DOMAINS",
    "TASK_SPECS",
    "TaskSpec",
    "WriteDecisionNavigationEnv",
    "DetourWriteDecisionEnv",
    "make_env",
    "task_from_name",
    "task_name",
    "task_specs",
]

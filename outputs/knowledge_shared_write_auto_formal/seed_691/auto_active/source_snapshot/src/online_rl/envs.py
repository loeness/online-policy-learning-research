"""Small navigation environments for source pretraining and context shifts.

The observation and action spaces are identical in every context. A context
changes only how a chosen action is translated into movement, and its identity
is never included in observations or ``info``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from operator import index as as_index
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ContextSpec:
    """Hidden action-mapping parameters for one deployment context.

    ``rotation`` is measured in clockwise quarter turns: 0 is the source
    mapping, 1 turns each action clockwise, and -1 turns it counterclockwise.
    With probability ``slip``, an action is additionally turned one quarter
    turn clockwise or counterclockwise with equal probability.
    """

    name: str
    rotation: int
    slip: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("context name must not be empty")
        if isinstance(self.rotation, bool):
            raise ValueError("rotation must be one of -1, 0, or 1 quarter turns")
        try:
            rotation = as_index(self.rotation)
        except TypeError as exc:
            raise ValueError("rotation must be one of -1, 0, or 1 quarter turns") from exc
        if rotation not in (-1, 0, 1):
            raise ValueError("rotation must be one of -1, 0, or 1 quarter turns")
        if not math.isfinite(self.slip) or not 0.0 <= self.slip <= 1.0:
            raise ValueError("slip must be a finite probability in [0, 1]")


def contexts() -> dict[str, ContextSpec]:
    """Return the source and four shifted action-mapping contexts."""

    return {
        "normal": ContextSpec("normal", 0),
        "cw": ContextSpec("cw", 1),
        "cw_slip": ContextSpec("cw_slip", 1, 0.12),
        "ccw": ContextSpec("ccw", -1),
        "ccw_slip": ContextSpec("ccw_slip", -1, 0.12),
    }


class NavigationEnv(gym.Env[np.ndarray, int]):
    """Navigate an agent to a randomly chosen goal on a square grid.

    Observations are ``[x, y, goal_x, goal_y]`` normalized to ``[0, 1]``.
    Actions retain the source convention in all contexts: up, right, down,
    and left. Context changes are applied only between episodes by the caller
    through :meth:`set_context`.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    # Action index order: up, right, down, left. A positive index rotation is
    # clockwise in screen/grid coordinates where y increases downwards.
    _DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def __init__(
        self,
        context: ContextSpec | None = None,
        grid_size: int = 9,
        max_steps: int = 64,
    ) -> None:
        super().__init__()
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
            raise ValueError("grid_size must be an integer greater than or equal to 2")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.context = context if context is not None else ContextSpec("normal", 0)
        if not isinstance(self.context, ContextSpec):
            raise TypeError("context must be a ContextSpec or None")

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        self._position = (0, 0)
        self._goal = (1, 1)
        self._episode_length = 0
        self._has_reset = False

    def set_context(self, spec: ContextSpec) -> None:
        """Set the hidden transition context for subsequent steps.

        The runner should call this between episodes. It does not reset the
        environment and does not add context information to agent-visible data.
        """

        if not isinstance(spec, ContextSpec):
            raise TypeError("spec must be a ContextSpec")
        self.context = spec

    def _observation(self) -> np.ndarray:
        denominator = float(self.grid_size - 1)
        x, y = self._position
        goal_x, goal_y = self._goal
        return np.asarray(
            [x / denominator, y / denominator, goal_x / denominator, goal_y / denominator],
            dtype=np.float32,
        )

    def _distance(self, position: tuple[int, int] | None = None) -> int:
        x, y = self._position if position is None else position
        goal_x, goal_y = self._goal
        return abs(goal_x - x) + abs(goal_y - y)

    def _info(self, success: bool) -> dict[str, bool | int]:
        return {
            "success": success,
            "episode_length": self._episode_length,
            "distance": self._distance(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, bool | int]]:
        """Sample distinct start and goal cells using Gymnasium's seeded RNG."""

        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported")
        self._position = (
            int(self.np_random.integers(self.grid_size)),
            int(self.np_random.integers(self.grid_size)),
        )
        goal_index = int(self.np_random.integers(self.grid_size * self.grid_size - 1))
        start_index = self._position[1] * self.grid_size + self._position[0]
        if goal_index >= start_index:
            goal_index += 1
        self._goal = (goal_index % self.grid_size, goal_index // self.grid_size)
        self._episode_length = 0
        self._has_reset = True
        return self._observation(), self._info(success=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        """Apply one action and return Gymnasium's five-value step result."""

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
        effective_action = (action_index + self.context.rotation) % 4
        if self.context.slip > 0.0 and self.np_random.random() < self.context.slip:
            # A local slip changes the commanded direction by one adjacent
            # direction, preserving some alignment with the chosen action.
            effective_action = (effective_action + int(self.np_random.choice((-1, 1)))) % 4

        dx, dy = self._DELTAS[effective_action]
        x, y = self._position
        self._position = (
            min(max(x + dx, 0), self.grid_size - 1),
            min(max(y + dy, 0), self.grid_size - 1),
        )
        self._episode_length += 1

        distance = self._distance()
        success = distance == 0
        reward = 0.25 * (previous_distance - distance) - 0.03
        if success:
            reward += 1.0
        terminated = success
        truncated = self._episode_length >= self.max_steps and not terminated
        return self._observation(), float(reward), terminated, truncated, self._info(success)


def expert_action(obs: np.ndarray | list[float] | tuple[float, ...]) -> int:
    """Choose a deterministic shortest-path action for the source mapping."""

    state = np.asarray(obs)
    if state.shape != (4,):
        raise ValueError("obs must have shape (4,)")
    dx = float(state[2] - state[0])
    dy = float(state[3] - state[1])
    if abs(dx) > abs(dy):
        return 1 if dx > 0 else 3
    return 2 if dy > 0 else 0


def expert_distribution(obs: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Return a uniform distribution over all shortest-path source actions.

    At the goal there is no shortest-path action to take, so this returns an
    all-zero vector; callers should omit goal observations from behavior
    cloning loss.
    """

    state = np.asarray(obs)
    if state.shape != (4,):
        raise ValueError("obs must have shape (4,)")
    dx = float(state[2] - state[0])
    dy = float(state[3] - state[1])
    actions: list[int] = []
    if dx > 0:
        actions.append(1)
    elif dx < 0:
        actions.append(3)
    if dy > 0:
        actions.append(2)
    elif dy < 0:
        actions.append(0)
    distribution = np.zeros(4, dtype=np.float32)
    if actions:
        distribution[actions] = 1.0 / len(actions)
    return distribution


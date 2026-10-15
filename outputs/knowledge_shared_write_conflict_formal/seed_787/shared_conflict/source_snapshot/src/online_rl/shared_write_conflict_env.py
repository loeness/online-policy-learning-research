"""C1 action dynamics with an unobserved reward-sign conflict.

Both modes use the same A0 transition map and observations. Minus reverses
the reward returned by C1 while preserving its termination and info fields.
"""

from __future__ import annotations

import numpy as np

from .c1_envs import C1NavigationEnv, C1_CONTEXTS


MODES = ("plus", "minus")


class SharedWriteConflictEnv(C1NavigationEnv):
    """C1 A0 with either its original reward or its negation."""

    def __init__(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.reward_sign = 1.0 if mode == "plus" else -1.0
        super().__init__(C1_CONTEXTS["A0"])

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, float(self.reward_sign * reward), terminated, truncated, info


__all__ = ["MODES", "SharedWriteConflictEnv"]

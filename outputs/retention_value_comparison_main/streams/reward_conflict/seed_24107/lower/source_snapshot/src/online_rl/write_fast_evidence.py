"""Incremental implementation of the bounded write-evidence window."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .shared_write_grouping import PROFILE_SHAPE, empty_profile, observe_profile
from .write_decision_evidence import (
    EvidenceProfile,
    EvidenceWindow,
    _FEATURES,
    _reward_features,
    _validate_event,
)


class IncrementalEvidenceWindow(EvidenceWindow):
    """A fixed ring window whose numeric profile is maintained incrementally.

    The five ring arrays remain the persistent state. The additional arrays
    are a runtime cache derived from those events and are rebuilt on restore.
    """

    def __init__(self, capacity: int = 128) -> None:
        super().__init__(capacity)
        self._movement_counts_cache = np.zeros(PROFILE_SHAPE, dtype=np.int32)
        self._reward_xtx_cache = np.zeros((_FEATURES, _FEATURES), dtype=np.float64)
        self._reward_xty_cache = np.zeros(_FEATURES, dtype=np.float64)

    @property
    def nbytes(self) -> int:
        """Runtime bytes in ring and derived cache arrays, excluding Python."""

        return int(
            super().nbytes
            + self._movement_counts_cache.nbytes
            + self._reward_xtx_cache.nbytes
            + self._reward_xty_cache.nbytes
        )

    @property
    def cache_nbytes(self) -> int:
        """Bytes held by derived profile caches, excluding Python metadata."""

        return int(
            self._movement_counts_cache.nbytes
            + self._reward_xtx_cache.nbytes
            + self._reward_xty_cache.nbytes
        )

    def _apply_event(self, index: int, sign: int) -> None:
        """Add or subtract one stored ring event from the fixed-size caches."""

        counts = empty_profile()
        obs = self._obs[index]
        next_obs = self._next_obs[index]
        observe_profile(counts, obs, int(self._actions[index]), next_obs)
        self._movement_counts_cache += sign * counts.astype(np.int32)

        features = _reward_features(obs, next_obs, bool(self._terminated[index]))
        reward = float(self._rewards[index])
        self._reward_xtx_cache += sign * np.outer(features, features)
        self._reward_xty_cache += sign * features * reward

    def observe(
        self,
        obs: object,
        action: int,
        reward: float,
        next_obs: object,
        terminated: bool,
    ) -> None:
        """Append one validated event and update profile caches in O(1)."""

        # Prevalidate so malformed input cannot subtract a live event before
        # the parent performs its normal validation and ring write.
        _validate_event(obs, action, reward, next_obs, terminated)
        index = self._next_index
        if self._size == self.capacity:
            self._apply_event(index, -1)

        super().observe(obs, action, reward, next_obs, terminated)
        self._apply_event(index, 1)

    def profile(self) -> EvidenceProfile:
        """Return fixed-size copies of the maintained window statistics."""

        with np.errstate(over="ignore", invalid="ignore"):
            movement_counts = np.clip(
                self._movement_counts_cache, 0, np.iinfo(np.uint16).max
            ).astype(np.uint16)
            reward_xtx = self._reward_xtx_cache.astype(np.float32)
            reward_xty = self._reward_xty_cache.astype(np.float32)
        if not np.isfinite(reward_xtx).all() or not np.isfinite(reward_xty).all():
            raise ValueError("window reward statistics exceed finite float32 range")
        return EvidenceProfile(
            movement_counts=movement_counts,
            reward_xtx=reward_xtx,
            reward_xty=reward_xty,
            reward_n=np.float32(self._size),
        )

    def state_dict(self) -> dict[str, Any]:
        """Save only ring state; cached values are recoverable from its events."""

        state = super().state_dict()
        # The parent serializer observes ``self.nbytes`` dynamically, which
        # includes this subclass's runtime caches. Persistent state contains
        # ring arrays only, so store the corresponding ring byte count.
        state["nbytes"] = int(
            self._obs.nbytes
            + self._actions.nbytes
            + self._rewards.nbytes
            + self._next_obs.nbytes
            + self._terminated.nbytes
        )
        return state

    def _rebuild_cache(self) -> None:
        self._movement_counts_cache.fill(0)
        self._reward_xtx_cache.fill(0.0)
        self._reward_xty_cache.fill(0.0)
        start = (self._next_index - self._size) % self.capacity
        for offset in range(self._size):
            self._apply_event((start + offset) % self.capacity, 1)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "IncrementalEvidenceWindow":
        """Restore ring state and rebuild its derived caches with one scan."""

        window = super().from_state_dict(state)
        window._rebuild_cache()
        return window


__all__ = ["IncrementalEvidenceWindow"]

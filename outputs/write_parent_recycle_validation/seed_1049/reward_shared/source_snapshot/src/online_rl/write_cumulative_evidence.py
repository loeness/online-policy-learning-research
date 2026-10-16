"""Bounded cumulative transition and reward statistics per resident slot."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index as as_index
from typing import Any

import numpy as np

from .shared_write_grouping import PROFILE_SHAPE, empty_profile, observe_profile
from .write_decision_evidence import (
    EvidenceProfile,
    _FEATURES,
    _reward_features,
    _validate_event,
)


_STATE_VERSION = 1
_MAX_SLOTS = 8
_MAX_SUPPORT = 60_000


def _validate_slots(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("slots must be an integer in [1, 8]")
    try:
        slots = as_index(value)
    except TypeError as exc:
        raise ValueError("slots must be an integer in [1, 8]") from exc
    if not 1 <= slots <= _MAX_SLOTS:
        raise ValueError("slots must be an integer in [1, 8]")
    return int(slots)


def _validate_slot(value: object, slots: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("slot must be an integer in the configured range")
    try:
        slot = as_index(value)
    except TypeError as exc:
        raise ValueError("slot must be an integer in the configured range") from exc
    if not 0 <= slot < slots:
        raise ValueError("slot must be an integer in the configured range")
    return int(slot)


class CumulativeEvidence:
    """Fixed-size sufficient statistics for up to eight resident slots.

    The object stores no transition history and consumes no task or context
    markers. Its state consists only of bounded movement counts, reward
    regression sufficient statistics, per-slot support, and rescale counts.
    """

    def __init__(self, slots: int) -> None:
        self.slots = _validate_slots(slots)
        self._movement_counts = np.zeros(
            (self.slots, *PROFILE_SHAPE), dtype=np.uint16
        )
        self._reward_xtx = np.zeros(
            (self.slots, _FEATURES, _FEATURES), dtype=np.float64
        )
        self._reward_xty = np.zeros((self.slots, _FEATURES), dtype=np.float64)
        self._reward_n = np.zeros(self.slots, dtype=np.uint32)
        self._rescale_count = np.zeros(self.slots, dtype=np.uint32)

    @property
    def nbytes(self) -> int:
        """Bytes in the resident numeric arrays, excluding Python metadata."""

        return int(
            self._movement_counts.nbytes
            + self._reward_xtx.nbytes
            + self._reward_xty.nbytes
            + self._reward_n.nbytes
            + self._rescale_count.nbytes
        )

    @property
    def reward_n(self) -> np.ndarray:
        """Return a copy of the bounded sample support for every slot."""

        return self._reward_n.copy()

    @property
    def rescale_count(self) -> np.ndarray:
        """Return a copy of the number of support rescalings per slot."""

        return self._rescale_count.copy()

    def clear(self, slot: int) -> None:
        """Clear all sufficient statistics for one resident slot."""

        slot_index = _validate_slot(slot, self.slots)
        self._movement_counts[slot_index].fill(0)
        self._reward_xtx[slot_index].fill(0.0)
        self._reward_xty[slot_index].fill(0.0)
        self._reward_n[slot_index] = 0
        self._rescale_count[slot_index] = 0

    def observe(
        self,
        slot: int,
        obs: object,
        action: int,
        reward: float,
        next_obs: object,
        terminated: bool,
    ) -> None:
        """Accumulate one visible transition in O(1) space and time."""

        slot_index = _validate_slot(slot, self.slots)
        current, action_index, reward32, following, done = _validate_event(
            obs, action, reward, next_obs, terminated
        )

        if self._reward_n[slot_index] >= _MAX_SUPPORT:
            self._movement_counts[slot_index] //= np.uint16(2)
            self._reward_xtx[slot_index] *= 0.5
            self._reward_xty[slot_index] *= 0.5
            self._reward_n[slot_index] //= np.uint32(2)
            self._rescale_count[slot_index] = min(
                int(self._rescale_count[slot_index]) + 1,
                int(np.iinfo(np.uint32).max),
            )

        one_event = empty_profile()
        observe_profile(one_event, current, action_index, following)
        # Support was reduced before insertion, so adding one event cannot
        # overflow the uint16 per-cell counters.
        self._movement_counts[slot_index] += one_event

        features = _reward_features(current, following, bool(done))
        self._reward_xtx[slot_index] += np.outer(features, features)
        self._reward_xty[slot_index] += features * float(reward32)
        self._reward_n[slot_index] += np.uint32(1)

    def profile_for_slot(self, slot: int) -> EvidenceProfile:
        """Return one slot's profile using the public EvidenceProfile dtypes."""

        slot_index = _validate_slot(slot, self.slots)
        with np.errstate(over="ignore", invalid="ignore"):
            reward_xtx = self._reward_xtx[slot_index].astype(np.float32)
            reward_xty = self._reward_xty[slot_index].astype(np.float32)
        if not np.isfinite(reward_xtx).all() or not np.isfinite(reward_xty).all():
            raise ValueError("cumulative reward statistics exceed finite float32 range")
        return EvidenceProfile(
            movement_counts=self._movement_counts[slot_index].copy(),
            reward_xtx=reward_xtx,
            reward_xty=reward_xty,
            reward_n=np.float32(self._reward_n[slot_index]),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return copied sufficient-statistic arrays suitable for torch.save."""

        return {
            "version": _STATE_VERSION,
            "slots": self.slots,
            "movement_counts": self._movement_counts.copy(),
            "reward_xtx": self._reward_xtx.copy(),
            "reward_xty": self._reward_xty.copy(),
            "reward_n": self._reward_n.copy(),
            "rescale_count": self._rescale_count.copy(),
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CumulativeEvidence":
        """Restore validated fixed-size sufficient statistics."""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported CumulativeEvidence state version")
        slots = _validate_slots(state.get("slots"))
        result = cls(slots)
        specifications = (
            ("movement_counts", (slots, *PROFILE_SHAPE), np.dtype(np.uint16)),
            ("reward_xtx", (slots, _FEATURES, _FEATURES), np.dtype(np.float64)),
            ("reward_xty", (slots, _FEATURES), np.dtype(np.float64)),
            ("reward_n", (slots,), np.dtype(np.uint32)),
            ("rescale_count", (slots,), np.dtype(np.uint32)),
        )
        restored: dict[str, np.ndarray] = {}
        for name, shape, dtype in specifications:
            array = np.asarray(state.get(name))
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"state {name} must have dtype {dtype} and shape {shape}")
            restored[name] = array

        if not np.isfinite(restored["reward_xtx"]).all() or not np.isfinite(
            restored["reward_xty"]
        ).all():
            raise ValueError("state reward statistics must be finite")
        if np.any(restored["reward_n"] > _MAX_SUPPORT):
            raise ValueError("state reward_n exceeds the bounded support")
        if np.any(restored["movement_counts"].astype(np.uint32) > restored["reward_n"][:, None, None, None]):
            raise ValueError("state movement counts exceed reward_n support")

        expected_nbytes = sum(array.nbytes for array in restored.values())
        if "nbytes" in state and state["nbytes"] != expected_nbytes:
            raise ValueError("state nbytes does not match numeric storage")

        result._movement_counts[...] = restored["movement_counts"]
        result._reward_xtx[...] = restored["reward_xtx"]
        result._reward_xty[...] = restored["reward_xty"]
        result._reward_n[...] = restored["reward_n"]
        result._rescale_count[...] = restored["rescale_count"]
        return result


__all__ = ["CumulativeEvidence"]

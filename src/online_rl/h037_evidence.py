"""Fixed-capacity, slot-scoped observation evidence for H037.

Callers decide when observations are causally available and record them at that
boundary. This cache preserves only the supplied rows; it does not infer task
identity, inspect future data, or run any fitting or experiment code.
"""
from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any

import numpy as np


_STATE_VERSION = 1
_OBSERVATION_WIDTH = 13
_TASK_BYTES = 64
_GROUPS = (0, 1)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Return an integer-index value, rejecting bools and lossy coercions."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or result > _INT64_MAX:
        raise ValueError(f"{name} must be in [{minimum}, {_INT64_MAX}]")
    return int(result)


def _task_bytes(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("task must be a nonempty string")
    if "\x00" in value:
        raise ValueError("task must not contain a NUL character")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("task must be valid UTF-8") from exc
    if len(encoded) > _TASK_BYTES:
        raise ValueError(f"UTF-8 task identity must fit in {_TASK_BYTES} bytes")
    return encoded


class H037Evidence:
    """Keep the most recent finite 13-D observations independently per slot.

    A slot's local identity is its ``(generation, task, group)`` tuple. A new
    tuple clears that slot before recording. Parent versions are stored for
    every row, so a parent update never rewrites the provenance of earlier
    observations. Task IDs are UTF-8 encoded into a fixed 64-byte field.

    ``nbytes`` reports the exact size of this object's NumPy backing arrays,
    including fixed-width identity encoding and ring metadata. It excludes
    Python object overhead and checkpoint/container serialization overhead.
    """

    def __init__(self, capacity_per_slot: int, slots: int = 2) -> None:
        self.capacity_per_slot = _integer(capacity_per_slot, "capacity_per_slot", minimum=1)
        self.slots = _integer(slots, "slots", minimum=1)

        self._observations = np.zeros(
            (self.slots, self.capacity_per_slot, _OBSERVATION_WIDTH), dtype=np.float32,
        )
        self._parent_versions = np.full(
            (self.slots, self.capacity_per_slot), -1, dtype=np.int64,
        )
        self._generations = np.full(self.slots, -1, dtype=np.int64)
        self._groups = np.full(self.slots, -1, dtype=np.int8)
        self._tasks = np.zeros(self.slots, dtype=f"S{_TASK_BYTES}")
        self._counts = np.zeros(self.slots, dtype=np.int64)
        self._next_indices = np.zeros(self.slots, dtype=np.int64)

    @property
    def nbytes(self) -> int:
        """Exact bytes allocated by the cache's NumPy arrays."""
        return sum(array.nbytes for array in self._arrays().values())

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "observations": self._observations,
            "parent_versions": self._parent_versions,
            "generations": self._generations,
            "groups": self._groups,
            "tasks": self._tasks,
            "counts": self._counts,
            "next_indices": self._next_indices,
        }

    def _slot(self, value: object) -> int:
        slot = _integer(value, "slot")
        if slot >= self.slots:
            raise ValueError("slot is outside configured capacity")
        return slot

    @staticmethod
    def _group(value: object) -> int:
        group = _integer(value, "group")
        if group not in _GROUPS:
            raise ValueError("group must be zero or one")
        return group

    def _clear_slot(self, slot: int) -> None:
        self._observations[slot].fill(0.0)
        self._parent_versions[slot].fill(-1)
        self._generations[slot] = -1
        self._groups[slot] = -1
        self._tasks[slot] = b""
        self._counts[slot] = 0
        self._next_indices[slot] = 0

    def record(
        self,
        slot: int,
        generation: int,
        task: str,
        group: int,
        parent_version: int,
        observations: np.ndarray,
    ) -> None:
        """Append rows, evicting oldest rows when the per-slot bound is reached."""
        slot_index = self._slot(slot)
        generation_value = _integer(generation, "generation")
        task_value = _task_bytes(task)
        group_value = self._group(group)
        parent_value = _integer(parent_version, "parent_version")

        if not isinstance(observations, np.ndarray):
            raise ValueError("observations must be a NumPy array")
        incoming = observations
        if incoming.ndim != 2 or incoming.shape[1] != _OBSERVATION_WIDTH:
            raise ValueError("observations must have shape (N, 13)")
        if not np.issubdtype(incoming.dtype, np.number) or np.issubdtype(incoming.dtype, np.complexfloating):
            raise ValueError("observations must have a real numeric dtype")
        if not np.isfinite(incoming).all():
            raise ValueError("observations must be finite")

        start = max(0, len(incoming) - self.capacity_per_slot)
        for row_index in range(start, len(incoming)):
            with np.errstate(over="ignore", invalid="ignore"):
                row = np.asarray(incoming[row_index], dtype=np.float32)
            if not np.isfinite(row).all():
                raise ValueError("observations must be representable as finite float32 values")

        if int(self._generations[slot_index]) != generation_value or \
                int(self._groups[slot_index]) != group_value or \
                self._task_value(slot_index) != task_value.decode("utf-8"):
            self._clear_slot(slot_index)
            self._generations[slot_index] = generation_value
            self._groups[slot_index] = group_value
            self._tasks[slot_index] = task_value

        count = int(self._counts[slot_index])
        next_index = int(self._next_indices[slot_index])
        for row_index in range(start, len(incoming)):
            with np.errstate(over="ignore", invalid="ignore"):
                row = np.asarray(incoming[row_index], dtype=np.float32)
            if not np.isfinite(row).all():
                raise ValueError("observations must be representable as finite float32 values")
            self._observations[slot_index, next_index] = row
            self._parent_versions[slot_index, next_index] = parent_value
            next_index = (next_index + 1) % self.capacity_per_slot
            count = min(count + 1, self.capacity_per_slot)

        self._counts[slot_index] = count
        self._next_indices[slot_index] = next_index

    def _task_value(self, slot: int) -> str | None:
        if int(self._generations[slot]) < 0:
            return None
        try:
            return self._tasks[slot].tobytes().rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError as exc:  # Defensive against damaged in-memory state.
            raise ValueError("stored task identity is not valid UTF-8") from exc

    def get(self, slot: int, generation: int, task: str, group: int) -> dict[str, Any]:
        """Return an ordered copy only when the complete slot identity matches."""
        slot_index = self._slot(slot)
        generation_value = _integer(generation, "generation")
        task_value = _task_bytes(task).decode("utf-8")
        group_value = self._group(group)

        matches = (
            int(self._generations[slot_index]) == generation_value
            and int(self._groups[slot_index]) == group_value
            and self._task_value(slot_index) == task_value
        )
        if not matches:
            return {
                "slot": slot_index,
                "identity": None,
                "observations": np.empty((0, _OBSERVATION_WIDTH), dtype=np.float32),
                "parent_versions": np.empty((0,), dtype=np.int64),
            }

        count = int(self._counts[slot_index])
        next_index = int(self._next_indices[slot_index])
        if count < self.capacity_per_slot:
            indices = slice(0, count)
            observations = self._observations[slot_index, indices].copy()
            parent_versions = self._parent_versions[slot_index, indices].copy()
        else:
            observations = np.concatenate((
                self._observations[slot_index, next_index:],
                self._observations[slot_index, :next_index],
            ), axis=0)
            parent_versions = np.concatenate((
                self._parent_versions[slot_index, next_index:],
                self._parent_versions[slot_index, :next_index],
            ), axis=0)
        return {
            "slot": slot_index,
            "identity": {
                "generation": generation_value,
                "task": task_value,
                "group": group_value,
            },
            "observations": observations,
            "parent_versions": parent_versions,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return detached arrays and the format/configuration needed to restore."""
        return {
            "version": _STATE_VERSION,
            "config": {
                "capacity_per_slot": self.capacity_per_slot,
                "slots": self.slots,
                "observation_width": _OBSERVATION_WIDTH,
                "task_bytes": _TASK_BYTES,
            },
            **{name: array.copy() for name, array in self._arrays().items()},
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "H037Evidence":
        """Restore a snapshot after validating identity, ring, and array invariants."""
        if not isinstance(state, Mapping) or state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported H037Evidence state")
        config = state.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("state config must be a mapping")
        if config.get("observation_width") != _OBSERVATION_WIDTH or config.get("task_bytes") != _TASK_BYTES:
            raise ValueError("state uses an incompatible H037Evidence encoding")
        result = cls(config.get("capacity_per_slot"), config.get("slots"))

        specifications = {
            "observations": ((result.slots, result.capacity_per_slot, _OBSERVATION_WIDTH), np.dtype(np.float32)),
            "parent_versions": ((result.slots, result.capacity_per_slot), np.dtype(np.int64)),
            "generations": ((result.slots,), np.dtype(np.int64)),
            "groups": ((result.slots,), np.dtype(np.int8)),
            "tasks": ((result.slots,), np.dtype(f"S{_TASK_BYTES}")),
            "counts": ((result.slots,), np.dtype(np.int64)),
            "next_indices": ((result.slots,), np.dtype(np.int64)),
        }
        restored: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in specifications.items():
            array = np.asarray(state.get(name))
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"state {name} must have dtype {dtype} and shape {shape}")
            restored[name] = array.copy()
        if not np.isfinite(restored["observations"]).all():
            raise ValueError("state observations must be finite")

        for slot in range(result.slots):
            count = int(restored["counts"][slot])
            next_index = int(restored["next_indices"][slot])
            generation = int(restored["generations"][slot])
            group = int(restored["groups"][slot])
            if count < 0 or count > result.capacity_per_slot:
                raise ValueError("state evidence exceeds per-slot capacity")
            if next_index < 0 or next_index >= result.capacity_per_slot:
                raise ValueError("state ring index is outside per-slot capacity")
            if count < result.capacity_per_slot and next_index != count:
                raise ValueError("partial state ring index must equal its row count")
            if count == 0 and next_index != 0:
                raise ValueError("empty state ring index must be zero")

            encoded_task = restored["tasks"][slot].tobytes()
            try:
                task = encoded_task.rstrip(b"\x00").decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("state task identity must be valid UTF-8") from exc
            if b"\x00" in encoded_task.rstrip(b"\x00"):
                raise ValueError("state task identity must not contain NUL")

            if generation == -1:
                if count != 0 or group != -1 or task:
                    raise ValueError("uninitialized slots cannot contain identity or rows")
            elif generation < 0 or group not in _GROUPS or not task:
                raise ValueError("occupied slot identity is invalid")

            occupied = np.zeros(result.capacity_per_slot, dtype=bool)
            if count < result.capacity_per_slot:
                occupied[:count] = True
            else:
                occupied[:] = True
            if np.any(restored["parent_versions"][slot, occupied] < 0):
                raise ValueError("stored parent versions must be nonnegative")
            if np.any(restored["parent_versions"][slot, ~occupied] != -1):
                raise ValueError("unused parent version entries must use the empty sentinel")
            if np.any(restored["observations"][slot, ~occupied] != 0.0):
                raise ValueError("unused observation entries must be zero")

        if "nbytes" in state and state["nbytes"] != sum(array.nbytes for array in restored.values()):
            raise ValueError("state nbytes does not match NumPy array storage")
        for name, array in restored.items():
            getattr(result, f"_{name}")[...] = array
        return result

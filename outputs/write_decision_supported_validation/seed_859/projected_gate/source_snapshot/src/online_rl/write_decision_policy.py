"""Flexible local/shared write policy and compact actor codec.

Route zero keeps the copied W0 actor unchanged. Routes one through
``local_slots`` select independent band-local matrices and may additionally
select one shared matrix broadcast over all three bands.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index as as_index
import struct
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .c1_policy import _LOCAL_PHI, _mark_residual
from .c1_tensor_controls import _BandCoefficients
from .policy import TreePolicy
from .shared_write_policy import SharedWritePolicy


_HEADER = struct.Struct("<4sBBBBI")
_MAGIC = b"WDK1"
_VERSION = 1
_LOCAL_WIDTH = 27
_SHARED_WIDTH = 9
_MAX_LOCAL_SLOTS = 8
_MAX_SHARED_GROUPS = 128
_CODECS = {"float32": 0, "int8": 1}
_CODEC_NAMES = {value: key for key, value in _CODECS.items()}


class FlexibleSharedPolicy(SharedWritePolicy):
    """Frozen W0 source basis with a configurable number of local routes."""

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__(temperature=temperature)
        self.local_slots = 6

    @classmethod
    def from_reference(
        cls,
        reference: TreePolicy,
        *,
        local_slots: int = 6,
        shared_groups: int = 2,
        exploration_epsilon: float = 0.05,
        temperature: float = 2.0,
        train_critic: bool = True,
    ) -> "FlexibleSharedPolicy":
        """Copy a reference policy, then resize its six-route residual layout."""
        slot_count = cls._validate_local_slot_count(local_slots)
        group_count = cls._validate_flexible_group_count(shared_groups)
        result = super().from_reference(
            reference,
            shared_groups=group_count,
            exploration_epsilon=exploration_epsilon,
            temperature=temperature,
            train_critic=train_critic,
        )
        result._resize_local_slots(slot_count)
        result._set_trainable_flags(train_critic=train_critic)
        return result

    @staticmethod
    def _validate_local_slot_count(local_slots: int) -> int:
        if isinstance(local_slots, bool):
            raise ValueError("local_slots must be an integer from one through eight")
        try:
            value = int(local_slots)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("local_slots must be an integer from one through eight") from exc
        if value != local_slots or not 1 <= value <= _MAX_LOCAL_SLOTS:
            raise ValueError("local_slots must be an integer from one through eight")
        return value

    @staticmethod
    def _validate_flexible_group_count(shared_groups: int) -> int:
        value = SharedWritePolicy._validate_group_count(shared_groups)
        if value not in (0, 2):
            raise ValueError("shared_groups must be zero or two")
        return value

    def _resize_local_slots(self, count: int) -> None:
        """Keep existing local tensors and create any extra matrices at zero."""
        current = set(self._nodes.keys())
        desired = {str(slot) for slot in range(1, count + 1)}
        for key in current - desired:
            del self._nodes[key]
            self._node_shapes.pop(int(key), None)
        for route in range(1, count + 1):
            key = str(route)
            if key not in self._nodes:
                self._nodes[key] = _BandCoefficients(like=self.source_basis)
                self._node_shapes[route] = (3, _LOCAL_PHI)
        self.local_slots = count
        self.slot_groups = torch.full(
            (count,), -1, dtype=torch.long, device=self.source_basis.device
        )

    def _validate_slot(self, slot: int) -> int:
        """Validate a zero-based local slot against this policy's size."""
        if isinstance(slot, bool):
            raise ValueError("slot must be an integer within the local slot range")
        try:
            value = int(slot)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("slot must be an integer within the local slot range") from exc
        if value != slot or value < 0:
            raise ValueError("slot must be an integer within the local slot range")
        # SharedWritePolicy calls this validator through self, so a per-instance
        # range is required once the layout can contain other than six slots.
        if value >= getattr(self, "local_slots", 6):
            raise ValueError("slot must be an integer within the local slot range")
        return value

    def _correction(self, obs: torch.Tensor, latent_pi: torch.Tensor) -> torch.Tensor:
        """Apply a correction only for route IDs represented by local slots."""
        obs = self._as_batch(obs)
        ids = obs[:, -1].round().long().clamp(0, 8)
        base_phi, _ = self._feature_banks(obs, latent_pi)
        bands = torch.floor(obs[:, 0].clamp(0.0, 1.0) * 3.0).long().clamp(0, 2)
        result = latent_pi.new_zeros((latent_pi.shape[0], 4))
        contrast_basis = self._contrast_basis.to(device=latent_pi.device, dtype=latent_pi.dtype)

        for route_value in torch.unique(ids, sorted=True).tolist():
            route = int(route_value)
            if route < 1 or route > self.local_slots:
                continue
            rows = torch.nonzero(ids == route, as_tuple=False).reshape(-1)
            phi = base_phi.index_select(0, rows)
            active_bands = bands.index_select(0, rows)
            psi = F.linear(phi, self.source_basis)
            coefficients = self._nodes[str(route)].A.index_select(0, active_bands)
            group = int(self.slot_groups[route - 1].item())
            if group >= 0:
                coefficients = coefficients + self._shared_coefficients[group].C.unsqueeze(0)
            contrast = torch.bmm(coefficients, psi.unsqueeze(-1)).squeeze(-1)
            result = result.index_add(0, rows, contrast @ contrast_basis.T)
        return result

    def read_local_matrix(self) -> np.ndarray:
        """Return local coefficients as a copied ``(local_slots, 27)`` array."""
        if not self.local_slots:
            return np.empty((0, 27), dtype=self.source_basis.detach().cpu().numpy().dtype)
        return np.stack(
            [self._nodes[str(slot + 1)].A.detach().cpu().numpy().reshape(27)
             for slot in range(self.local_slots)]
        ).copy()

    def install_local_matrix(self, matrix: np.ndarray) -> None:
        """Install a ``(local_slots, 27)`` local coefficient array."""
        values = np.asarray(matrix)
        expected = (self.local_slots, 27)
        if values.shape != expected:
            raise ValueError(f"local matrix must have shape {expected}")
        with torch.no_grad():
            for slot in range(self.local_slots):
                target = self._nodes[str(slot + 1)].A
                target.copy_(torch.as_tensor(
                    values[slot], device=target.device, dtype=target.dtype
                ).reshape(3, 3, 3))


@dataclass(frozen=True)
class EncodedActor:
    """Actor payload with its exact serialized size and decoder."""

    data: bytes

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def decode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return decode_actor(self.data)

    @classmethod
    def frombytes(cls, data: bytes | bytearray | memoryview) -> "EncodedActor":
        raw = bytes(data)
        decode_actor(raw)
        return cls(raw)


def _matrix32(value: object, shape: tuple[int, int], name: str) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric matrix with shape {shape}") from exc
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix, dtype="<f4")


def _groups_array(value: object, slot_count: int, group_count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (slot_count,):
        raise ValueError(f"groups must have shape ({slot_count},), got {raw.shape}")
    if raw.dtype.kind not in "iu":
        raise ValueError("groups must contain integers")
    values = [int(item) for item in raw.tolist()]
    if any(item < -1 or item >= group_count for item in values):
        raise ValueError(f"groups values must be in [-1, {group_count - 1}]")
    return np.ascontiguousarray(values, dtype=np.int8)


def _quantize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] == 0:
        return np.empty((0,), dtype="<f4"), np.empty(matrix.shape, dtype=np.int8)
    scales = np.max(np.abs(matrix), axis=1).astype(np.float32) / np.float32(127.0)
    scales[scales == 0.0] = 1.0
    quantized = np.rint(matrix / scales[:, None]).clip(-127, 127).astype(np.int8)
    return np.ascontiguousarray(scales, dtype="<f4"), np.ascontiguousarray(quantized)


def _encode_matrix(matrix: np.ndarray, codec: str) -> bytes:
    if codec == "float32":
        return matrix.tobytes(order="C")
    scales, quantized = _quantize_rows(matrix)
    return scales.tobytes(order="C") + quantized.tobytes(order="C")


def encode_actor(
    local: object,
    shared: object,
    groups: object,
    *,
    codec: Literal["float32", "int8"] = "float32",
    budget_bytes: int = 256 * 1024,
) -> EncodedActor:
    """Encode flexible actor matrices under an exact payload byte budget.

    The payload uses a 12 byte WDK1 header, then local matrices, shared
    matrices, and signed slot-to-group assignments. Int8 matrices store one
    little-endian float32 scale per row followed by row-major int8 values.
    """
    if codec not in _CODECS:
        raise ValueError("codec must be 'float32' or 'int8'")
    if isinstance(budget_bytes, bool):
        raise ValueError("budget_bytes must be a non-negative integer")
    try:
        budget = as_index(budget_bytes)
    except TypeError as exc:
        raise ValueError("budget_bytes must be a non-negative integer") from exc
    if budget < 0:
        raise ValueError("budget_bytes must be a non-negative integer")

    raw_local = np.asarray(local)
    if raw_local.ndim != 2 or raw_local.shape[1] != _LOCAL_WIDTH:
        raise ValueError(f"local must have shape (N, {_LOCAL_WIDTH})")
    slot_count = int(raw_local.shape[0])
    if not 1 <= slot_count <= _MAX_LOCAL_SLOTS:
        raise ValueError(f"local slot count must be in [1, {_MAX_LOCAL_SLOTS}]")
    local_matrix = _matrix32(local, (slot_count, _LOCAL_WIDTH), "local")

    raw_shared = np.asarray(shared)
    if raw_shared.ndim != 2 or raw_shared.shape[1] != _SHARED_WIDTH:
        raise ValueError(f"shared must have shape (G, {_SHARED_WIDTH})")
    group_count = int(raw_shared.shape[0])
    if not 0 <= group_count <= _MAX_SHARED_GROUPS:
        raise ValueError(f"shared group count must be in [0, {_MAX_SHARED_GROUPS}]")
    shared_matrix = _matrix32(shared, (group_count, _SHARED_WIDTH), "shared")
    group_array = _groups_array(groups, slot_count, group_count)

    header = _HEADER.pack(
        _MAGIC, _VERSION, _CODECS[codec], slot_count, group_count, _LOCAL_WIDTH
    )
    data = (
        header
        + _encode_matrix(local_matrix, codec)
        + _encode_matrix(shared_matrix, codec)
        + group_array.tobytes(order="C")
    )
    if len(data) > budget:
        raise ValueError(f"encoded actor is {len(data)} bytes, exceeding budget of {budget}")
    return EncodedActor(data)


def _decode_matrix(
    data: memoryview,
    offset: int,
    rows: int,
    width: int,
    codec: str,
) -> tuple[np.ndarray, int]:
    if codec == "float32":
        size = rows * width * 4
        matrix = np.frombuffer(data[offset:offset + size], dtype="<f4").copy().reshape(rows, width)
        if not np.isfinite(matrix).all():
            raise ValueError("serialized matrices must contain only finite values")
        return matrix, offset + size

    scales = np.frombuffer(data[offset:offset + rows * 4], dtype="<f4").copy()
    offset += rows * 4
    values = np.frombuffer(data[offset:offset + rows * width], dtype=np.int8).copy().reshape(rows, width)
    offset += rows * width
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("serialized int8 row scales must be finite and positive")
    if np.any(values == -128):
        raise ValueError("serialized int8 values must be in [-127, 127]")
    matrix = values.astype(np.float32) * scales[:, None]
    if not np.isfinite(matrix).all():
        raise ValueError("serialized matrices must contain only finite values")
    return matrix, offset


def decode_actor(data: bytes | bytearray | memoryview) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a WDK1 actor payload and reject malformed or trailing bytes."""
    try:
        raw = memoryview(data).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError("data must be a bytes-like object") from exc
    if len(raw) < _HEADER.size:
        raise ValueError("truncated actor header")
    magic, version, codec_id, slot_count, group_count, width = _HEADER.unpack(raw[:_HEADER.size])
    if magic != _MAGIC:
        raise ValueError("invalid actor magic")
    if version != _VERSION:
        raise ValueError(f"unsupported actor version: {version}")
    if codec_id not in _CODEC_NAMES:
        raise ValueError(f"unsupported actor codec: {codec_id}")
    if not 1 <= slot_count <= _MAX_LOCAL_SLOTS:
        raise ValueError(f"invalid local slot count: {slot_count}")
    if not 0 <= group_count <= _MAX_SHARED_GROUPS:
        raise ValueError(f"invalid shared group count: {group_count}")
    if width != _LOCAL_WIDTH:
        raise ValueError(f"invalid local matrix width: {width}")

    codec = _CODEC_NAMES[codec_id]
    expected_size = _HEADER.size + slot_count
    if codec == "float32":
        expected_size += (slot_count * _LOCAL_WIDTH + group_count * _SHARED_WIDTH) * 4
    else:
        expected_size += slot_count * 4 + slot_count * _LOCAL_WIDTH
        expected_size += group_count * 4 + group_count * _SHARED_WIDTH
    if len(raw) != expected_size:
        raise ValueError(f"invalid actor payload length: expected {expected_size} bytes, got {len(raw)}")

    offset = _HEADER.size
    local, offset = _decode_matrix(raw, offset, slot_count, _LOCAL_WIDTH, codec)
    shared, offset = _decode_matrix(raw, offset, group_count, _SHARED_WIDTH, codec)
    assignments = np.frombuffer(raw[offset:offset + slot_count], dtype=np.int8).copy()
    if np.any(assignments < -1) or np.any(assignments >= group_count):
        raise ValueError(f"serialized groups must be in [-1, {group_count - 1}]")
    return local, shared, assignments


__all__ = ["EncodedActor", "FlexibleSharedPolicy", "decode_actor", "encode_actor"]

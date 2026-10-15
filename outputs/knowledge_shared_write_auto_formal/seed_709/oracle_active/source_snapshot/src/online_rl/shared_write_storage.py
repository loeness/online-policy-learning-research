"""Compact, exact serialization for shared-write actor knowledge.

The binary format stores a ``(6, 27)`` local matrix, a ``(G, 9)`` shared
matrix, and six signed slot-to-group assignments. It uses a fixed 12-byte
header and contains no pickle or NumPy container metadata. ``float32`` keeps
the matrices exact after conversion to float32; ``int8`` uses a symmetric
per-row scale for bounded-error quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as as_index
import struct
from typing import Literal

import numpy as np


_HEADER = struct.Struct("<4sBBBBI")
_MAGIC = b"SWK1"
_VERSION = 1
_N_SLOTS = 6
_LOCAL_WIDTH = 27
_GROUP_WIDTH = 9
_MAX_GROUPS = 6
_CODECS = {"float32": 0, "int8": 1}
_CODEC_NAMES = {value: key for key, value in _CODECS.items()}


@dataclass(frozen=True)
class EncodedSharedKnowledge:
    """One serialized payload, with its exact on-disk size and decoder."""

    data: bytes

    @property
    def nbytes(self) -> int:
        """Return the exact serialized byte count, including the header."""

        return len(self.data)

    def decode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode this payload into local, shared, and assignment arrays."""

        return decode_shared_knowledge(self.data)

    @classmethod
    def frombytes(cls, data: bytes | bytearray | memoryview) -> "EncodedSharedKnowledge":
        """Wrap and validate an existing serialized payload."""

        raw = bytes(data)
        decode_shared_knowledge(raw)
        return cls(raw)


def _matrix32(value: object, shape: tuple[int, int], name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric matrix with shape {shape}") from exc
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix, dtype="<f4")


def _assignments(value: object, group_count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (_N_SLOTS,):
        raise ValueError(f"slot_groups must have shape ({_N_SLOTS},), got {raw.shape}")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("slot_groups must contain integers")
    if np.any(raw < -1) or np.any(raw >= group_count):
        raise ValueError(f"slot_groups values must be in [-1, {group_count - 1}]")
    return np.ascontiguousarray(raw, dtype=np.int8)


def _quantize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.max(np.abs(matrix), axis=1).astype(np.float32) / np.float32(127.0)
    scales[scales == 0.0] = 1.0
    quantized = np.rint(matrix / scales[:, None]).clip(-127, 127).astype(np.int8)
    return np.ascontiguousarray(scales, dtype="<f4"), np.ascontiguousarray(quantized)


def _encode_matrix(matrix: np.ndarray, codec: str) -> bytes:
    if codec == "float32":
        return matrix.tobytes(order="C")
    scales, quantized = _quantize_rows(matrix)
    return scales.tobytes(order="C") + quantized.tobytes(order="C")


def encode_shared_knowledge(
    local: object,
    shared: object,
    slot_groups: object,
    *,
    codec: Literal["float32", "int8"] = "float32",
    budget_bytes: int = 1024,
) -> EncodedSharedKnowledge:
    """Encode shared-write actor matrices with an exact byte budget.

    The wire order is header, local matrix, shared matrix, and assignments.
    For int8 matrices, the little-endian float32 row-scale vector precedes
    the row-major int8 values. The symmetric scale is
    ``max(abs(row)) / 127`` (or 1 for an all-zero row); values are rounded to
    the nearest integer and clipped to ``[-127, 127]``.
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

    local_matrix = _matrix32(local, (_N_SLOTS, _LOCAL_WIDTH), "local")
    shared_array = np.asarray(shared)
    if shared_array.ndim != 2 or shared_array.shape[1:] != (_GROUP_WIDTH,):
        raise ValueError(f"shared must have shape (G, {_GROUP_WIDTH})")
    group_count = int(shared_array.shape[0])
    if not 0 <= group_count <= _MAX_GROUPS:
        raise ValueError(f"shared group count must be in [0, {_MAX_GROUPS}]")
    shared_matrix = _matrix32(shared, (group_count, _GROUP_WIDTH), "shared")
    groups = _assignments(slot_groups, group_count)

    header = _HEADER.pack(
        _MAGIC,
        _VERSION,
        _CODECS[codec],
        _N_SLOTS,
        group_count,
        _LOCAL_WIDTH,
    )
    data = header + _encode_matrix(local_matrix, codec) + _encode_matrix(shared_matrix, codec) + groups.tobytes()
    if len(data) > budget:
        raise ValueError(f"encoded shared knowledge is {len(data)} bytes, exceeding budget of {budget}")
    return EncodedSharedKnowledge(data)


def _decode_matrix(
    data: memoryview,
    offset: int,
    rows: int,
    width: int,
    codec: str,
) -> tuple[np.ndarray, int]:
    if codec == "float32":
        size = rows * width * 4
        matrix = np.frombuffer(data[offset : offset + size], dtype="<f4").copy().reshape(rows, width)
        if not np.isfinite(matrix).all():
            raise ValueError("serialized matrices must contain only finite values")
        return matrix, offset + size

    scale_size = rows * 4
    scales = np.frombuffer(data[offset : offset + scale_size], dtype="<f4").copy()
    offset += scale_size
    value_size = rows * width
    values = np.frombuffer(data[offset : offset + value_size], dtype=np.int8).copy().reshape(rows, width)
    offset += value_size
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("serialized int8 row scales must be finite and positive")
    if np.any(values == -128):
        raise ValueError("serialized int8 matrix values must be in [-127, 127]")
    return values.astype(np.float32) * scales[:, None], offset


def decode_shared_knowledge(
    data: bytes | bytearray | memoryview,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a shared-write payload, rejecting malformed or extra bytes."""

    try:
        raw = memoryview(data).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError("data must be a bytes-like object") from exc
    if len(raw) < _HEADER.size:
        raise ValueError("truncated shared-knowledge header")
    magic, version, codec_id, slot_count, group_count, width = _HEADER.unpack(raw[: _HEADER.size])
    if magic != _MAGIC:
        raise ValueError("invalid shared-knowledge magic")
    if version != _VERSION:
        raise ValueError(f"unsupported shared-knowledge version: {version}")
    if codec_id not in _CODEC_NAMES:
        raise ValueError(f"unsupported shared-knowledge codec: {codec_id}")
    if slot_count != _N_SLOTS:
        raise ValueError(f"invalid slot count: {slot_count}")
    if not 0 <= group_count <= _MAX_GROUPS:
        raise ValueError(f"invalid group count: {group_count}")
    if width != _LOCAL_WIDTH:
        raise ValueError(f"invalid local matrix width: {width}")

    codec = _CODEC_NAMES[codec_id]
    expected_size = _HEADER.size + _N_SLOTS
    if codec == "float32":
        expected_size += (_N_SLOTS * _LOCAL_WIDTH + group_count * _GROUP_WIDTH) * 4
    else:
        expected_size += _N_SLOTS * 4 + _N_SLOTS * _LOCAL_WIDTH
        expected_size += group_count * 4 + group_count * _GROUP_WIDTH
    if len(raw) != expected_size:
        raise ValueError(f"invalid payload length: expected {expected_size} bytes, got {len(raw)}")

    offset = _HEADER.size
    local, offset = _decode_matrix(raw, offset, _N_SLOTS, _LOCAL_WIDTH, codec)
    shared, offset = _decode_matrix(raw, offset, group_count, _GROUP_WIDTH, codec)
    groups = np.frombuffer(raw[offset : offset + _N_SLOTS], dtype=np.int8).copy()
    if np.any(groups < -1) or np.any(groups >= group_count):
        raise ValueError(f"serialized slot_groups must be in [-1, {group_count - 1}]")
    return local, shared, groups


__all__ = ["EncodedSharedKnowledge", "decode_shared_knowledge", "encode_shared_knowledge"]

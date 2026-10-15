"""Budgeted, portable codecs for static C1 coefficient knowledge.

The encoded object owns only its numeric representation. It never keeps the
input matrix, and ``nbytes`` counts the contiguous representation arrays plus
the fixed 16 byte method/shape/type header. Container overhead is reported
separately by ``save`` callers through the resulting file size.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np


CODEC_VERSION = 1
HEADER_BYTES = 16
WIDTH = 27

_METHOD_IDS = {
    "dense_f32": 1,
    "sparse_topk": 2,
    "fp16": 3,
    "int8": 4,
    "global_mean_sparse": 5,
    "centroid2_sparse": 6,
    "truncated_svd": 7,
    "global_topk": 8,
}
METHODS = tuple(_METHOD_IDS)
_METHODS_BY_ID = {value: key for key, value in _METHOD_IDS.items()}
_BINARY_HEADER = struct.Struct("<4sHHII")


def _index_dtype(cell_count: int) -> np.dtype:
    # Match the declared format boundary: use uint16 only below 65,536 cells.
    return np.dtype(np.uint16 if cell_count < 65536 else np.uint32)


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != WIDTH:
        raise ValueError(f"matrix must have shape (N, {WIDTH})")
    if value.shape[0] > np.iinfo(np.uint32).max:
        raise ValueError("matrix has too many rows for the compact header")
    if not np.isfinite(value).all():
        raise ValueError("matrix must contain only finite values")
    return np.ascontiguousarray(value)


def _top_indices(residual: np.ndarray, count: int) -> np.ndarray:
    flat = np.asarray(residual, dtype=np.float32).reshape(-1)
    if count <= 0 or flat.size == 0:
        return np.empty((0,), dtype=np.int64)
    # The secondary key makes ties deterministic and independent of sort kind.
    order = np.lexsort((np.arange(flat.size, dtype=np.int64), -np.abs(flat)))
    positive = int(np.count_nonzero(flat))
    return order[: min(int(count), positive)]


def _sparse_arrays(residual: np.ndarray, count: int) -> dict[str, np.ndarray]:
    indices = _top_indices(residual, count)
    flat = np.asarray(residual, dtype=np.float32).reshape(-1)
    return {
        "indices": np.ascontiguousarray(indices.astype(_index_dtype(flat.size))),
        "values": np.ascontiguousarray(flat[indices].astype(np.float32)),
    }


def _independent_sparse_arrays(matrix: np.ndarray, count_per_row: int) -> dict[str, np.ndarray]:
    """Keep the same top-K count within each row, with globally addressed cells."""
    n, width = matrix.shape
    if n == 0 or count_per_row <= 0:
        index_dtype = _index_dtype(matrix.size)
        return {
            "indices": np.empty((0,), dtype=index_dtype),
            "values": np.empty((0,), dtype=np.float32),
        }
    indices = []
    values = []
    for row_index, row in enumerate(matrix):
        # Include zero-valued slots only when needed to keep the per-row K
        # uniform. They do not change reconstruction, and K is selected by
        # coefficient error (so trailing all-zero slots are excluded below).
        order = np.lexsort((np.arange(width, dtype=np.int64), -np.abs(row)))
        selected = order[:count_per_row]
        indices.append(selected + row_index * width)
        values.append(row[selected])
    return {
        "indices": np.ascontiguousarray(np.concatenate(indices).astype(_index_dtype(matrix.size))),
        "values": np.ascontiguousarray(np.concatenate(values).astype(np.float32)),
    }


def _kmeans_two(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic two-means over rows, with no context/family labels."""
    n = rows.shape[0]
    if n == 0:
        return np.zeros((2, WIDTH), dtype=np.float32), np.empty((0,), dtype=np.uint8)
    squared = np.sum((rows - rows[0]) ** 2, axis=1, dtype=np.float64)
    first = 0
    second = int(np.argmax(squared))
    centers = np.stack((rows[first], rows[second])).astype(np.float32, copy=True)
    assignments = np.zeros(n, dtype=np.uint8)
    for _ in range(64):
        distances = np.sum((rows[:, None, :] - centers[None, :, :]) ** 2,
                           axis=2, dtype=np.float64)
        updated_assignments = np.argmin(distances, axis=1).astype(np.uint8)
        updated_centers = centers.copy()
        for cluster in range(2):
            selected = rows[updated_assignments == cluster]
            if selected.size:
                updated_centers[cluster] = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        if np.array_equal(updated_assignments, assignments) and np.array_equal(updated_centers, centers):
            assignments = updated_assignments
            centers = updated_centers
            break
        assignments, centers = updated_assignments, updated_centers
    return centers, assignments


@dataclass(frozen=True)
class CompactKnowledge:
    """One compact coefficient matrix and the arrays needed to reconstruct it."""

    method: str
    shape: tuple[int, int]
    _arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.method not in _METHOD_IDS:
            raise ValueError(f"unknown compact knowledge method {self.method!r}")
        if len(self.shape) != 2 or self.shape[1] != WIDTH or self.shape[0] < 0:
            raise ValueError(f"shape must be (N, {WIDTH})")
        normalized: dict[str, np.ndarray] = {}
        for name, value in self._arrays.items():
            array = np.ascontiguousarray(value)
            array.setflags(write=False)
            normalized[str(name)] = array
        object.__setattr__(self, "_arrays", normalized)

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        """Read-only views of the stored representation arrays."""
        return self._arrays

    @property
    def nbytes(self) -> int:
        """Representation bytes: stored ndarray data plus the fixed header."""
        return HEADER_BYTES + sum(int(array.nbytes) for array in self._arrays.values())

    @property
    def parameter_summary(self) -> dict[str, int | str]:
        result: dict[str, int | str] = {"method": self.method}
        if self.method == "sparse_topk":
            result["topk_per_row"] = int(self._arrays["indices"].size // max(1, self.shape[0]))
            result["sparse_entries"] = int(self._arrays["indices"].size)
        if self.method == "global_topk":
            result["global_topk_entries"] = int(self._arrays["indices"].size)
        if self.method in ("global_mean_sparse", "centroid2_sparse"):
            result["sparse_entries"] = int(self._arrays["indices"].size)
        if self.method == "truncated_svd":
            result["rank"] = int(self._arrays["left"].shape[1])
        if self.method == "int8":
            result["scale_mode"] = "per_row_symmetric"
        if self.method == "centroid2_sparse":
            result["centroids"] = int(self._arrays["centroids"].shape[0])
        return result

    def decode(self) -> np.ndarray:
        """Reconstruct the matrix as a new float32 array."""
        n, width = self.shape
        if n == 0:
            return np.empty((0, width), dtype=np.float32)
        arrays = self._arrays
        if self.method == "dense_f32":
            result = arrays["values"].astype(np.float32, copy=True)
        elif self.method in ("sparse_topk", "global_topk"):
            result = np.zeros(self.shape, dtype=np.float32)
            result.reshape(-1)[arrays["indices"].astype(np.int64)] = arrays["values"].astype(np.float32)
        elif self.method == "fp16":
            result = arrays["values"].astype(np.float32)
        elif self.method == "int8":
            result = arrays["values"].astype(np.float32) * arrays["scales"].astype(np.float32)[:, None]
        elif self.method == "global_mean_sparse":
            result = np.broadcast_to(arrays["mean"].astype(np.float32), self.shape).copy()
            result.reshape(-1)[arrays["indices"].astype(np.int64)] += arrays["values"].astype(np.float32)
        elif self.method == "centroid2_sparse":
            assignments = arrays["assignments"].astype(np.int64)
            result = arrays["centroids"].astype(np.float32)[assignments].copy()
            result.reshape(-1)[arrays["indices"].astype(np.int64)] += arrays["values"].astype(np.float32)
        elif self.method == "truncated_svd":
            result = arrays["left"].astype(np.float32) @ arrays["right"].astype(np.float32)
        else:  # pragma: no cover - guarded by __post_init__
            raise ValueError(f"unsupported compact knowledge method {self.method!r}")
        return np.ascontiguousarray(result, dtype=np.float32)

    def to_payload(self) -> dict[str, Any]:
        """Return a portable payload containing only codec state."""
        return {
            "format_version": CODEC_VERSION,
            "method": self.method,
            "shape": list(self.shape),
            "arrays": {name: array.copy() for name, array in self._arrays.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CompactKnowledge":
        if int(payload.get("format_version", -1)) != CODEC_VERSION:
            raise ValueError("unsupported compact knowledge payload version")
        method = str(payload.get("method", ""))
        if method not in _METHOD_IDS:
            raise ValueError(f"unknown compact knowledge method {method!r}")
        raw_shape = payload.get("shape")
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
            raise ValueError("compact knowledge payload shape must have two dimensions")
        shape = (int(raw_shape[0]), int(raw_shape[1]))
        if shape[0] < 0 or shape[1] != WIDTH:
            raise ValueError(f"compact knowledge payload shape must be (N, {WIDTH})")
        raw_arrays = payload.get("arrays")
        if not isinstance(raw_arrays, Mapping):
            raise ValueError("compact knowledge payload must include an arrays mapping")
        arrays = {str(name): np.asarray(value) for name, value in raw_arrays.items()}
        result = cls(method, shape, arrays)
        # Decode validates method-specific required arrays and compatible shapes.
        result.decode()
        return result

    def save(self, path: str | Path) -> Path:
        """Write a portable NPZ; its container size is intentionally separate."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps({
            "format_version": CODEC_VERSION,
            "method": self.method,
            "shape": list(self.shape),
        }, separators=(",", ":")).encode("utf-8")
        np.savez_compressed(
            path,
            __metadata__=np.frombuffer(metadata, dtype=np.uint8),
            **{name: array for name, array in self._arrays.items()},
        )
        # NumPy appends .npz when the caller omitted it. Return the actual path.
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    def save_binary(self, path: str | Path) -> Path:
        """Write the exact header-plus-arrays binary representation.

        Unlike the NPZ helper, the file length equals ``nbytes`` exactly and
        can therefore be used as a materialized budget artifact.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = _BINARY_HEADER.pack(
            b"CKN1", CODEC_VERSION, _METHOD_IDS[self.method], self.shape[0], self.shape[1]
        )
        order = _binary_array_order(self.method)
        body = b"".join(np.ascontiguousarray(self._arrays[name]).tobytes() for name in order)
        raw = header + body
        if len(raw) != self.nbytes:
            raise ValueError(f"binary byte count {len(raw)} does not match accounted nbytes {self.nbytes}")
        path.write_bytes(raw)
        return path

    @classmethod
    def load_binary(cls, path: str | Path) -> "CompactKnowledge":
        raw = Path(path).read_bytes()
        if len(raw) < HEADER_BYTES:
            raise ValueError("compact binary is shorter than its fixed header")
        magic, version, method_id, rows, width = _BINARY_HEADER.unpack(raw[:HEADER_BYTES])
        if magic != b"CKN1":
            raise ValueError("invalid compact knowledge binary magic")
        if int(version) != CODEC_VERSION:
            raise ValueError("unsupported compact knowledge binary version")
        if method_id not in _METHODS_BY_ID:
            raise ValueError("unknown compact knowledge binary method ID")
        shape = (int(rows), int(width))
        method = _METHODS_BY_ID[int(method_id)]
        if shape[1] != WIDTH:
            raise ValueError(f"compact binary must have {WIDTH} columns")
        if shape[0] == 0:
            if len(raw) != HEADER_BYTES:
                raise ValueError("empty compact binary must contain only its header")
            return cls(method, shape, {})

        body = memoryview(raw)[HEADER_BYTES:]
        offset = 0
        n, width = shape
        cells = n * width
        index_dtype = _index_dtype(cells)

        def take(name: str, dtype: np.dtype, count: int, array_shape: tuple[int, ...]) -> np.ndarray:
            nonlocal offset
            size = int(count) * dtype.itemsize
            end = offset + size
            if end > len(body):
                raise ValueError(f"truncated compact binary array {name!r}")
            values = np.frombuffer(body[offset:end], dtype=dtype, count=int(count)).copy()
            offset = end
            return values.reshape(array_shape)

        arrays: dict[str, np.ndarray]
        if method == "dense_f32":
            arrays = {"values": take("values", np.dtype(np.float32), cells, shape)}
        elif method in ("sparse_topk", "global_topk"):
            entry_size = index_dtype.itemsize + np.dtype(np.float32).itemsize
            remainder = len(body)
            if remainder % entry_size:
                raise ValueError("sparse binary payload length is not entry-aligned")
            entries = remainder // entry_size
            arrays = {
                "indices": take("indices", index_dtype, entries, (entries,)),
                "values": take("values", np.dtype(np.float32), entries, (entries,)),
            }
        elif method == "fp16":
            arrays = {"values": take("values", np.dtype(np.float16), cells, shape)}
        elif method == "int8":
            arrays = {
                "values": take("values", np.dtype(np.int8), cells, shape),
                "scales": take("scales", np.dtype(np.float32), n, (n,)),
            }
        elif method == "global_mean_sparse":
            mean = take("mean", np.dtype(np.float32), width, (width,))
            entry_size = index_dtype.itemsize + np.dtype(np.float32).itemsize
            remainder = len(body) - offset
            if remainder % entry_size:
                raise ValueError("global-mean sparse payload length is not entry-aligned")
            entries = remainder // entry_size
            arrays = {
                "mean": mean,
                "indices": take("indices", index_dtype, entries, (entries,)),
                "values": take("values", np.dtype(np.float32), entries, (entries,)),
            }
        elif method == "centroid2_sparse":
            centroids = take("centroids", np.dtype(np.float32), 2 * width, (2, width))
            assignments = take("assignments", np.dtype(np.uint8), n, (n,))
            entry_size = index_dtype.itemsize + np.dtype(np.float32).itemsize
            remainder = len(body) - offset
            if remainder % entry_size:
                raise ValueError("two-centroid sparse payload length is not entry-aligned")
            entries = remainder // entry_size
            arrays = {
                "centroids": centroids,
                "assignments": assignments,
                "indices": take("indices", index_dtype, entries, (entries,)),
                "values": take("values", np.dtype(np.float32), entries, (entries,)),
            }
        else:  # truncated_svd
            factor_bytes_per_rank = np.dtype(np.float32).itemsize * (n + width)
            if len(body) % factor_bytes_per_rank:
                raise ValueError("SVD binary payload length is not rank-aligned")
            rank = len(body) // factor_bytes_per_rank
            if not 1 <= rank <= min(n, width):
                raise ValueError("SVD binary rank is outside the shape bounds")
            arrays = {
                "left": take("left", np.dtype(np.float32), n * rank, (n, rank)),
                "right": take("right", np.dtype(np.float32), rank * width, (rank, width)),
            }
        if offset != len(body):
            raise ValueError("compact binary has trailing bytes")
        result = cls(method, shape, arrays)
        if result.nbytes != len(raw):
            raise ValueError("compact binary physical and accounted byte counts differ")
        result.decode()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "CompactKnowledge":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(data["__metadata__"].tobytes().decode("utf-8"))
            arrays = {name: data[name].copy() for name in data.files if name != "__metadata__"}
        return cls.from_payload({**metadata, "arrays": arrays})


def _require_budget(budget_bytes: int) -> int:
    if isinstance(budget_bytes, bool) or int(budget_bytes) != budget_bytes:
        raise ValueError("budget_bytes must be an integer")
    budget = int(budget_bytes)
    if budget < HEADER_BYTES:
        raise ValueError(f"budget_bytes must cover the {HEADER_BYTES}-byte codec header")
    return budget


def _binary_array_order(method: str) -> tuple[str, ...]:
    return {
        "dense_f32": ("values",),
        "sparse_topk": ("indices", "values"),
        "global_topk": ("indices", "values"),
        "fp16": ("values",),
        "int8": ("values", "scales"),
        "global_mean_sparse": ("mean", "indices", "values"),
        "centroid2_sparse": ("centroids", "assignments", "indices", "values"),
        "truncated_svd": ("left", "right"),
    }[method]


def minimum_nbytes(matrix: np.ndarray, method: str) -> int:
    """Return the method's minimum serialized numeric size, including header."""
    matrix = _validate_matrix(matrix)
    if method not in _METHOD_IDS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    n, width = matrix.shape
    if n == 0:
        return HEADER_BYTES
    cells = n * width
    if method == "dense_f32":
        payload_bytes = 4 * cells
    elif method in ("sparse_topk", "global_topk"):
        payload_bytes = 0
    elif method == "fp16":
        payload_bytes = 2 * cells
    elif method == "int8":
        payload_bytes = cells + 4 * n
    elif method == "global_mean_sparse":
        payload_bytes = 4 * width
    elif method == "centroid2_sparse":
        payload_bytes = 4 * 2 * width + n
    else:  # truncated_svd
        payload_bytes = 4 * (n + width)
    return HEADER_BYTES + payload_bytes


def encode(matrix: np.ndarray, method: str, budget_bytes: int) -> CompactKnowledge:
    """Encode a finite ``(N, 27)`` matrix within a charged representation budget.

    Sparse counts and SVD rank are selected from available representation
    bytes and coefficient reconstruction error alone; no evaluation result is
    consulted. Arrays and their data types are the complete stored state.
    """
    matrix = _validate_matrix(matrix)
    method = str(method)
    if method not in _METHOD_IDS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    budget = _require_budget(budget_bytes)
    n, width = matrix.shape
    cell_count = n * width
    index_bytes = _index_dtype(cell_count).itemsize

    # No row data is needed for an empty matrix. The shape is enough to decode.
    if n == 0:
        return CompactKnowledge(method, matrix.shape, {})

    arrays: dict[str, np.ndarray]
    if method == "dense_f32":
        arrays = {"values": matrix.astype(np.float32, copy=True)}
    elif method == "sparse_topk":
        bytes_per_uniform_k = n * (index_bytes + 4)
        max_k = min(width, max(0, (budget - HEADER_BYTES) // bytes_per_uniform_k))
        useful_k = max((int(np.count_nonzero(row)) for row in matrix), default=0)
        count_per_row = min(max_k, useful_k)
        arrays = _independent_sparse_arrays(matrix, count_per_row)
    elif method == "global_topk":
        count = min(cell_count, max(0, (budget - HEADER_BYTES) // (index_bytes + 4)))
        arrays = _sparse_arrays(matrix, count)
    elif method == "fp16":
        arrays = {"values": matrix.astype(np.float16)}
    elif method == "int8":
        scales = np.max(np.abs(matrix), axis=1).astype(np.float32) / np.float32(127.0)
        scales[scales == 0] = np.float32(1.0)
        quantized = np.rint(matrix / scales[:, None]).clip(-127, 127).astype(np.int8)
        arrays = {"values": quantized, "scales": scales}
    elif method == "global_mean_sparse":
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        base_bytes = HEADER_BYTES + mean.nbytes
        if budget < base_bytes:
            raise ValueError(f"global_mean_sparse needs at least {base_bytes} bytes for its mean vector")
        residual = matrix - mean[None, :]
        count = min(cell_count, (budget - base_bytes) // (index_bytes + 4))
        arrays = {"mean": mean, **_sparse_arrays(residual, count)}
    elif method == "centroid2_sparse":
        centroids, assignments = _kmeans_two(matrix)
        base_bytes = HEADER_BYTES + centroids.nbytes + assignments.nbytes
        if budget < base_bytes:
            raise ValueError(f"centroid2_sparse needs at least {base_bytes} bytes for two centroids and assignments")
        residual = matrix - centroids[assignments.astype(np.int64)]
        count = min(cell_count, (budget - base_bytes) // (index_bytes + 4))
        arrays = {"centroids": centroids, "assignments": assignments, **_sparse_arrays(residual, count)}
    elif method == "truncated_svd":
        left, singular, right_t = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
        rank_limit = min(left.shape[1], max(0, (budget - HEADER_BYTES) // (4 * (n + width))))
        if rank_limit < 1:
            minimum = HEADER_BYTES + 4 * (n + width)
            raise ValueError(f"truncated_svd needs at least {minimum} bytes for rank one")
        # Use the greatest affordable rank that reduces coefficient error;
        # exact zero singular tails do not consume bytes.
        positive_rank = int(np.count_nonzero(singular[:rank_limit] > 0.0))
        rank = max(1, positive_rank)
        root = np.sqrt(singular[:rank])
        left_factor = (left[:, :rank] * root[None, :]).astype(np.float32)
        right_factor = (root[:, None] * right_t[:rank, :]).astype(np.float32)
        # Canonicalize signs without changing the product.
        for column in range(rank):
            pivot = int(np.argmax(np.abs(left_factor[:, column])))
            if left_factor[pivot, column] < 0:
                left_factor[:, column] *= -1
                right_factor[column, :] *= -1
        arrays = {"left": left_factor, "right": right_factor}
    else:  # pragma: no cover - guarded above
        raise ValueError(f"unsupported compact knowledge method {method!r}")

    result = CompactKnowledge(method, matrix.shape, arrays)
    if result.nbytes > budget:
        raise ValueError(f"{method} needs at least {result.nbytes} bytes for its representation")
    return result


__all__ = [
    "CODEC_VERSION", "HEADER_BYTES", "METHODS", "CompactKnowledge", "encode",
    "minimum_nbytes",
]

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.compact_knowledge import (  # noqa: E402
    HEADER_BYTES,
    METHODS,
    CompactKnowledge,
    encode,
    minimum_nbytes,
)


def _matrix(rows=6):
    values = np.arange(rows * 27, dtype=np.float32).reshape(rows, 27)
    return (values - 50.0) / 31.0


@pytest.mark.parametrize("method", METHODS)
def test_codec_roundtrip_is_portable_and_charged_to_budget(method, tmp_path):
    matrix = _matrix()
    # Large enough for every representation, including an exact dense copy.
    encoded = encode(matrix, method, 4096)
    assert encoded.nbytes <= 4096
    assert encoded.nbytes == HEADER_BYTES + sum(array.nbytes for array in encoded.arrays.values())

    restored = CompactKnowledge.from_payload(encoded.to_payload())
    np.testing.assert_allclose(restored.decode(), encoded.decode(), rtol=0.0, atol=0.0)
    assert restored.method == method
    assert restored.shape == matrix.shape
    assert not hasattr(encoded, "original")
    assert not hasattr(encoded, "source_matrix")

    path = encoded.save(tmp_path / f"{method}.npz")
    from_file = CompactKnowledge.load(path)
    np.testing.assert_allclose(from_file.decode(), encoded.decode(), rtol=0.0, atol=0.0)
    assert path.stat().st_size > 0

    binary_path = encoded.save_binary(tmp_path / f"{method}.ckb")
    assert binary_path.stat().st_size == encoded.nbytes
    from_binary = CompactKnowledge.load_binary(binary_path)
    np.testing.assert_allclose(from_binary.decode(), encoded.decode(), rtol=0.0, atol=0.0)
    assert from_binary.nbytes == binary_path.stat().st_size


def test_dense_f32_exact_and_sparse_budget_is_respected():
    matrix = _matrix()
    dense = encode(matrix, "dense_f32", 664)
    np.testing.assert_array_equal(dense.decode(), matrix)
    assert dense.nbytes == 16 + matrix.nbytes == 664

    sparse = encode(matrix, "sparse_topk", 192)
    assert sparse.nbytes <= 192
    assert sparse.arrays["indices"].dtype == np.uint16
    assert sparse.arrays["indices"].size == sparse.arrays["values"].size
    # Each target row gets the same independent top-K count.
    k = sparse.arrays["indices"].size // len(matrix)
    for row_index, row in enumerate(matrix):
        selected = sparse.arrays["indices"][row_index * k:(row_index + 1) * k].astype(np.int64)
        local_indices = selected - row_index * matrix.shape[1]
        expected = np.sort(np.abs(row))[-k:]
        actual = np.sort(np.abs(row[local_indices]))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "method,budget",
    [("dense_f32", 648), ("fp16", 320), ("int8", 192),
     ("global_mean_sparse", 123), ("centroid2_sparse", 237),
     ("truncated_svd", 147)],
)
def test_methods_fail_clearly_when_fixed_representation_cannot_fit(method, budget):
    with pytest.raises(ValueError, match="needs at least|truncated_svd needs at least|budget"):
        encode(_matrix(), method, budget)


def test_empty_and_single_row_shapes_decode_without_hidden_source():
    empty = np.empty((0, 27), dtype=np.float32)
    for method in METHODS:
        packed = encode(empty, method, HEADER_BYTES)
        assert packed.nbytes == HEADER_BYTES
        assert packed.decode().shape == (0, 27)

    row = (np.arange(27, dtype=np.float32) - 13.0)[None, :]
    for method in METHODS:
        # Methods with fixed metadata need more space than the header.
        packed = encode(row, method, 4096)
        assert packed.decode().shape == (1, 27)
        assert packed.nbytes <= 4096
        assert np.isfinite(packed.decode()).all()


def test_centroid_codec_has_no_family_or_context_ids_and_handles_identical_rows():
    matrix = np.repeat(_matrix(1), 6, axis=0)
    packed = encode(matrix, "centroid2_sparse", 238)
    np.testing.assert_allclose(packed.decode(), matrix, rtol=0.0, atol=0.0)
    assert packed.arrays["assignments"].dtype == np.uint8
    assert packed.arrays["assignments"].shape == (6,)
    assert packed.parameter_summary["centroids"] == 2
    assert not any("family" in name or "context" in name for name in packed.arrays)


@pytest.mark.parametrize("matrix", [np.zeros((2, 27), np.float32), np.full((2, 27), -1.0, np.float32)])
def test_quantizers_handle_zero_scale_and_negative_sign(matrix):
    int8 = encode(matrix, "int8", 256)
    decoded = int8.decode()
    np.testing.assert_allclose(decoded, matrix, rtol=0.0, atol=1e-7)
    assert int8.arrays["values"].dtype == np.int8
    assert np.max(np.abs(int8.arrays["values"].astype(np.int16))) <= 127
    assert np.all(int8.arrays["scales"] > 0)


def test_invalid_matrix_and_unknown_method_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        encode(np.ones((2, 26), np.float32), "dense_f32", 1024)
    with pytest.raises(ValueError, match="finite"):
        encode(np.full((1, 27), np.nan, np.float32), "dense_f32", 1024)
    with pytest.raises(ValueError, match="method"):
        encode(np.zeros((1, 27), np.float32), "oracle", 1024)


def test_global_topk_allocates_entries_across_all_rows():
    matrix = np.zeros((6, 27), dtype=np.float32)
    matrix[0, 0] = 100.0
    matrix[1:, 0] = 1.0
    matrix[5, 26] = 2.0
    packed = encode(matrix, "global_topk", 40)
    assert packed.nbytes <= 40
    selected = packed.arrays["indices"].astype(np.int64)
    np.testing.assert_array_equal(selected, np.array([0, 161, 27, 54], dtype=np.int64))
    decoded = packed.decode()
    assert decoded[0, 0] == 100.0
    assert decoded[1, 0] == 1.0
    assert decoded[5, 26] == 2.0


def test_minimum_sizes_include_header_and_all_numeric_metadata():
    matrix = _matrix()
    expected = {
        "dense_f32": 664,
        "sparse_topk": 16,
        "global_topk": 16,
        "fp16": 340,
        "int8": 202,
        "global_mean_sparse": 124,
        "centroid2_sparse": 238,
        "truncated_svd": 148,
    }
    assert {method: minimum_nbytes(matrix, method) for method in METHODS} == expected

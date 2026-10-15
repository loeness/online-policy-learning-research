"""Size, fidelity, and malformed-input checks for shared-write storage."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.shared_write_storage import (  # noqa: E402
    EncodedSharedKnowledge,
    decode_shared_knowledge,
    encode_shared_knowledge,
)


def matrices(group_count: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.linspace(-2.0, 3.0, 6 * 27, dtype=np.float32).reshape(6, 27)
    shared = np.linspace(-0.7, 0.9, group_count * 9, dtype=np.float32).reshape(group_count, 9)
    if group_count == 0:
        groups = np.full(6, -1, dtype=np.int8)
    else:
        groups = np.asarray([0, -1, group_count - 1, 0, -1, group_count - 1], dtype=np.int8)
    return local, shared, groups


@pytest.mark.parametrize("group_count,expected_bytes", [(2, 738), (0, 666)])
def test_float32_has_exact_size_and_round_trips(tmp_path, group_count, expected_bytes) -> None:
    local, shared, groups = matrices(group_count)
    payload = encode_shared_knowledge(
        local, shared, groups, codec="float32", budget_bytes=1024
    )
    decoded_local, decoded_shared, decoded_groups = payload.decode()

    assert isinstance(payload, EncodedSharedKnowledge)
    assert payload.nbytes == expected_bytes == len(payload.data)
    assert np.array_equal(decoded_local, local)
    assert np.array_equal(decoded_shared, shared)
    assert np.array_equal(decoded_groups, groups)
    output = tmp_path / "shared-knowledge.bin"
    output.write_bytes(payload.data)
    assert output.stat().st_size == payload.nbytes
    assert decode_shared_knowledge(output.read_bytes())[1].shape == (group_count, 9)


@pytest.mark.parametrize("group_count,expected_bytes", [(2, 230), (0, 204)])
def test_int8_uses_per_row_scales_and_bounded_error(group_count, expected_bytes) -> None:
    local, shared, groups = matrices(group_count)
    payload = encode_shared_knowledge(local, shared, groups, codec="int8")
    restored_local, restored_shared, restored_groups = payload.decode()

    assert payload.nbytes == expected_bytes
    assert np.array_equal(restored_groups, groups)
    for original, restored in ((local, restored_local), (shared, restored_shared)):
        max_error = np.max(np.abs(original - restored), axis=1)
        row_scale = np.max(np.abs(original), axis=1) / 127.0
        assert np.all(max_error <= row_scale / 2.0 + 2e-7)


def test_budget_checks_actual_encoded_bytes() -> None:
    local, shared, groups = matrices(2)

    with pytest.raises(ValueError, match="738 bytes"):
        encode_shared_knowledge(local, shared, groups, budget_bytes=737)
    encoded = encode_shared_knowledge(local, shared, groups, budget_bytes=738)
    assert encoded.nbytes == 738


@pytest.mark.parametrize(
    "local,shared,groups,kwargs,match",
    [
        (np.zeros((5, 27)), np.zeros((2, 9)), [0] * 6, {}, "local must have shape"),
        (np.full((6, 27), np.nan), np.zeros((2, 9)), [0] * 6, {}, "finite values"),
        (np.zeros((6, 27)), np.zeros((7, 9)), [-1] * 6, {}, "group count"),
        (np.zeros((6, 27)), np.zeros((2, 9)), [2] * 6, {}, "slot_groups values"),
        (np.zeros((6, 27)), np.zeros((0, 9)), [0] * 6, {}, "slot_groups values"),
        (np.zeros((6, 27)), np.zeros((2, 9)), [0.0] * 6, {}, "slot_groups must contain integers"),
        (np.zeros((6, 27)), np.zeros((2, 9)), [0] * 6, {"codec": "float16"}, "codec must"),
        (np.zeros((6, 27)), np.zeros((2, 9)), [0] * 6, {"budget_bytes": -1}, "budget_bytes"),
        (np.zeros((6, 27)), np.zeros((2, 9)), [0] * 6, {"budget_bytes": True}, "budget_bytes"),
    ],
)
def test_encoder_rejects_invalid_inputs(local, shared, groups, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        encode_shared_knowledge(local, shared, groups, **kwargs)


def test_decoder_rejects_bad_headers_lengths_nonfinite_values_and_assignments() -> None:
    local, shared, groups = matrices(2)
    data = bytearray(encode_shared_knowledge(local, shared, groups).data)

    with pytest.raises(ValueError, match="truncated"):
        decode_shared_knowledge(data[:8])
    for offset, value, message in (
        (0, ord("X"), "magic"),
        (4, 2, "version"),
        (5, 9, "codec"),
        (6, 5, "slot count"),
        (7, 7, "group count"),
    ):
        corrupt = bytearray(data)
        corrupt[offset] = value
        with pytest.raises(ValueError, match=message):
            decode_shared_knowledge(corrupt)

    with pytest.raises(ValueError, match="payload length"):
        decode_shared_knowledge(data[:-1])
    with pytest.raises(ValueError, match="payload length"):
        decode_shared_knowledge(data + b"\x00")

    nonfinite = bytearray(data)
    nonfinite[12:16] = struct.pack("<f", float("inf"))
    with pytest.raises(ValueError, match="finite"):
        decode_shared_knowledge(nonfinite)

    invalid_assignment = bytearray(data)
    invalid_assignment[-1] = 2
    with pytest.raises(ValueError, match="slot_groups"):
        decode_shared_knowledge(invalid_assignment)


def test_int8_decoder_rejects_nonpositive_or_nonfinite_row_scales() -> None:
    local, shared, groups = matrices(2)
    data = bytearray(encode_shared_knowledge(local, shared, groups, codec="int8").data)

    data[12:16] = struct.pack("<f", float("nan"))
    with pytest.raises(ValueError, match="scales"):
        decode_shared_knowledge(data)

    data = bytearray(encode_shared_knowledge(local, shared, groups, codec="int8").data)
    data[12 + 6 * 4] = 128  # -128 is reserved and never emitted by the encoder.
    with pytest.raises(ValueError, match=r"\[-127, 127\]"):
        decode_shared_knowledge(data)


def test_frombytes_validates_and_wraps_payload() -> None:
    local, shared, groups = matrices(0)
    payload = encode_shared_knowledge(local, shared, groups, codec="int8")

    wrapped = EncodedSharedKnowledge.frombytes(bytearray(payload.data))
    assert wrapped.nbytes == payload.nbytes
    assert np.array_equal(wrapped.decode()[2], groups)
    with pytest.raises(ValueError, match="magic"):
        EncodedSharedKnowledge.frombytes(b"not a valid payload")


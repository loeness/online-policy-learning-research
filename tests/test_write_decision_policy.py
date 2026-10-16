"""Checks for flexible shared write routes and the WDK1 actor codec."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl.write_decision_policy import (  # noqa: E402
    FlexibleSharedPolicy,
    decode_actor,
    encode_actor,
)


torch.set_num_threads(1)


@pytest.fixture
def reference() -> TreePolicy:
    torch.manual_seed(621)
    return TreePolicy(
        observation_space=spaces.Box(
            low=np.asarray([-2.0] * 12 + [0.0], dtype=np.float32),
            high=np.asarray([2.0] * 12 + [8.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )


def observations(routes: list[int], x: list[float] | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(72)
    result = torch.rand((len(routes), OBS_DIM), generator=generator)
    if x is not None:
        result[:, 0] = torch.tensor(x, dtype=torch.float32)
    result[:, -1] = torch.tensor(routes, dtype=torch.float32)
    return result


def set_bias_only_basis(policy: FlexibleSharedPolicy) -> None:
    with torch.no_grad():
        policy.source_basis.zero_()
        policy.source_basis[0, -1] = 1.0


def correction(policy: FlexibleSharedPolicy, obs: torch.Tensor) -> torch.Tensor:
    latent = policy.mlp_extractor.forward_actor(policy.extract_features(obs))
    return policy._correction(obs, latent)


def test_factory_resizes_routes_and_preserves_w0_actor_and_base_route(reference):
    policy = FlexibleSharedPolicy.from_reference(
        reference, local_slots=7, shared_groups=0, train_critic=False
    )
    assert policy.local_slots == 7
    assert tuple(policy._nodes.keys()) == tuple(str(route) for route in range(1, 8))
    assert policy.slot_groups.shape == (7,)
    assert policy.read_local_matrix().shape == (7, 27)
    assert policy.read_shared_matrix().shape == (0, 9)
    assert torch.count_nonzero(policy._nodes["7"].A) == 0
    assert torch.equal(policy.source_basis, policy._contrast_basis.T @ torch.cat(
        [reference.action_net.weight, reference.action_net.bias[:, None]], dim=1
    ))
    local = np.arange(7 * 27, dtype=np.float32).reshape(7, 27) / 100.0
    policy.install_local_matrix(local)
    np.testing.assert_array_equal(policy.read_local_matrix(), local)
    for name in ("weight", "bias"):
        torch.testing.assert_close(
            getattr(policy.action_net, name), getattr(reference.action_net, name), rtol=0, atol=0
        )

    obs = observations([0, 7])
    with torch.no_grad():
        latent = policy.mlp_extractor.forward_actor(policy.extract_features(obs))
        base = policy.action_net(latent)
        torch.testing.assert_close(policy._correction(obs, latent)[0], torch.zeros(4), rtol=0, atol=0)
        torch.testing.assert_close(policy.action_net(latent)[0], base[0], rtol=0, atol=0)


def test_route_seven_gradients_and_shared_group_across_multiple_slots(reference):
    policy = FlexibleSharedPolicy.from_reference(
        reference, local_slots=7, shared_groups=2, train_critic=False
    )
    set_bias_only_basis(policy)
    policy.set_slot_group(0, 0, preserve_effective=False)
    policy.set_slot_group(6, 0, preserve_effective=False)
    shared = np.zeros((2, 9), dtype=np.float32)
    shared[0, 0] = 0.4
    policy.install_shared_matrix(shared)

    obs = observations([1, 7], [0.1, 0.8])
    output = correction(policy, obs)
    assert torch.count_nonzero(output[0]) > 0
    assert torch.count_nonzero(output[1]) > 0
    output[:, 0].sum().backward()

    shared_parameters = tuple(policy.shared_parameters())
    assert shared_parameters[0].grad is not None
    assert torch.count_nonzero(shared_parameters[0].grad) > 0
    assert shared_parameters[1].grad is None
    assert policy._nodes["1"].A.grad is not None
    assert policy._nodes["7"].A.grad is not None
    assert policy._nodes["2"].A.grad is None
    assert policy.source_basis.grad is None


def test_reassignment_compensates_local_matrix_and_dynamic_reset(reference):
    policy = FlexibleSharedPolicy.from_reference(
        reference, local_slots=7, shared_groups=2, train_critic=False
    )
    with torch.no_grad():
        policy._nodes["7"].A.normal_(mean=0.0, std=0.2)
    policy.install_shared_matrix(np.arange(18, dtype=np.float32).reshape(2, 9) / 13)
    policy.set_slot_group(6, 0, preserve_effective=False)
    before = policy._nodes["7"].A.detach().clone() + policy._shared_coefficients[0].C.detach()

    policy.set_slot_group(6, 1, preserve_effective=True)
    after = policy._nodes["7"].A.detach().clone() + policy._shared_coefficients[1].C.detach()
    torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-7)
    assert policy.group_for_slot(6) == 1
    policy.reset_slot(6)
    assert policy.group_for_slot(6) == -1
    assert torch.count_nonzero(policy._nodes["7"].A) == 0
    with pytest.raises(ValueError, match="local slot range"):
        policy.group_for_slot(7)


@pytest.mark.parametrize("codec", ["float32", "int8"])
def test_actor_codec_roundtrip_and_actual_payload_sizes(codec):
    rng = np.random.default_rng(173)
    local = rng.normal(size=(6, 27)).astype(np.float32)
    shared = rng.normal(size=(2, 9)).astype(np.float32)
    groups = np.asarray([0, 1, -1, 0, -1, 1], dtype=np.int8)
    encoded = encode_actor(local, shared, groups, codec=codec)
    decoded_local, decoded_shared, decoded_groups = encoded.decode()
    assert encoded.data[:4] == b"WDK1"
    assert encoded.nbytes == len(encoded.data)
    np.testing.assert_array_equal(decoded_groups, groups)
    if codec == "float32":
        np.testing.assert_array_equal(decoded_local, local)
        np.testing.assert_array_equal(decoded_shared, shared)
        assert encoded.nbytes == 12 + 6 + 108 * 6 + 36 * 2
    else:
        np.testing.assert_allclose(decoded_local, local, rtol=0.01, atol=0.01)
        np.testing.assert_allclose(decoded_shared, shared, rtol=0.01, atol=0.01)
        assert encoded.nbytes == 230


def test_seven_independent_slots_fit_exactly_236_int8_bytes():
    encoded = encode_actor(
        np.zeros((7, 27), dtype=np.float32),
        np.empty((0, 9), dtype=np.float32),
        np.full((7,), -1, dtype=np.int8),
        codec="int8",
        budget_bytes=236,
    )
    assert encoded.nbytes == 236
    decoded_local, decoded_shared, decoded_groups = decode_actor(encoded.data)
    assert decoded_local.shape == (7, 27)
    assert decoded_shared.shape == (0, 9)
    np.testing.assert_array_equal(decoded_groups, np.full(7, -1, dtype=np.int8))
    with pytest.raises(ValueError, match="exceeding budget"):
        encode_actor(
            np.zeros((7, 27), dtype=np.float32),
            np.empty((0, 9), dtype=np.float32),
            np.full((7,), -1, dtype=np.int8),
            codec="int8",
            budget_bytes=235,
        )


def test_codec_rejects_invalid_inputs_truncation_and_int8_overflow():
    local = np.zeros((6, 27), dtype=np.float32)
    shared = np.zeros((2, 9), dtype=np.float32)
    groups = np.full((6,), -1, dtype=np.int8)
    with pytest.raises(ValueError, match="finite"):
        encode_actor(np.full((6, 27), np.nan), shared, groups)
    with pytest.raises(ValueError, match="shape"):
        encode_actor(np.zeros((6, 26)), shared, groups)
    with pytest.raises(ValueError, match="groups values"):
        encode_actor(local, shared, np.full(6, 2, dtype=np.int8))
    with pytest.raises(ValueError, match="truncated"):
        decode_actor(b"WDK1")

    encoded = encode_actor(local, shared, groups, codec="int8")
    damaged = bytearray(encoded.data)
    damaged[12 + 4 * 6] = 0x80
    with pytest.raises(ValueError, match=r"\[-127, 127\]"):
        decode_actor(damaged)

    invalid_groups = bytearray(encoded.data)
    invalid_groups[-1] = 2
    with pytest.raises(ValueError, match="serialized groups"):
        decode_actor(invalid_groups)

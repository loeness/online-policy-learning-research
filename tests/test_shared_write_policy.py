"""Behavioral checks for the active shared correction policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl.shared_write_policy import SharedWritePolicy  # noqa: E402


torch.set_num_threads(1)


@pytest.fixture
def reference() -> TreePolicy:
    """Use the same seeded W0 reference fixture as test_c1_tensor_controls."""
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


def set_bias_only_basis(policy: SharedWritePolicy) -> None:
    """Make shared-output and gradient assertions independent of random W0."""
    with torch.no_grad():
        policy.source_basis.zero_()
        policy.source_basis[0, -1] = 1.0


def correction(policy: SharedWritePolicy, obs: torch.Tensor) -> torch.Tensor:
    latent = policy.mlp_extractor.forward_actor(policy.extract_features(obs))
    return policy._correction(obs, latent)


def test_zero_coefficients_preserve_w0_logits_and_route_zero_is_foundation(reference):
    policy = SharedWritePolicy.from_reference(reference, train_critic=False)
    obs = observations([0, 1, 2, 3, 4, 5, 6, 7, 8])
    with torch.no_grad():
        latent = policy.mlp_extractor.forward_actor(policy.extract_features(obs))
        base_logits = policy.action_net(latent)
        actual_logits = base_logits + policy._correction(obs, latent)
        head = torch.cat([policy.action_net.weight, policy.action_net.bias[:, None]], dim=1)
        expected_basis = policy._contrast_basis.T @ head

    torch.testing.assert_close(policy.source_basis, expected_basis, rtol=0, atol=0)
    torch.testing.assert_close(actual_logits, base_logits, rtol=0, atol=0)
    torch.testing.assert_close(policy._correction(obs, latent), torch.zeros_like(base_logits), rtol=0, atol=0)
    assert policy.source_basis.requires_grad is False
    assert all(torch.count_nonzero(node.A) == 0 for node in policy._nodes.values())
    assert all(torch.count_nonzero(parameter) == 0 for parameter in policy.shared_parameters())
    assert policy.group_for_slot(0) == -1


def test_shared_update_changes_only_assigned_slots(reference):
    policy = SharedWritePolicy.from_reference(reference, shared_groups=2, train_critic=False)
    set_bias_only_basis(policy)
    policy.set_slot_group(0, 0, preserve_effective=False)
    shared = np.zeros((2, 9), dtype=np.float32)
    shared[0, 0] = 0.75
    policy.install_shared_matrix(shared)

    obs = observations([1, 2, 0], [0.1, 0.6, 0.9])
    output = correction(policy, obs)
    assert torch.count_nonzero(output[0]) > 0
    torch.testing.assert_close(output[1:], torch.zeros_like(output[1:]), rtol=0, atol=0)


def test_two_assigned_slots_accumulate_gradient_in_shared_parameter(reference):
    policy = SharedWritePolicy.from_reference(reference, shared_groups=2, train_critic=False)
    set_bias_only_basis(policy)
    policy.set_slot_group(0, 0, preserve_effective=False)
    policy.set_slot_group(1, 0, preserve_effective=False)
    optimizer = policy.make_optimizer(learning_rate=0.01, transfer=False)
    obs = observations([1, 2], [0.2, 0.8])
    weights = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)

    optimizer.zero_grad(set_to_none=True)
    loss = (correction(policy, obs) * weights).sum()
    loss.backward()
    shared = tuple(policy.shared_parameters())
    assert shared[0].grad is not None
    assert torch.count_nonzero(shared[0].grad) > 0
    assert shared[1].grad is None
    joint_gradient = shared[0].grad.detach().clone()

    separate_gradients = []
    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        (correction(policy, obs[index:index + 1]) * weights[index:index + 1]).sum().backward()
        separate_gradients.append(shared[0].grad.detach().clone())
    torch.testing.assert_close(joint_gradient, separate_gradients[0] + separate_gradients[1])

    optimizer.zero_grad(set_to_none=True)
    (correction(policy, obs) * weights).sum().backward()
    previous = shared[0].detach().clone()
    optimizer.step()
    assert not torch.equal(shared[0], previous)
    assert policy.source_basis.grad is None


def test_reassignment_preserves_effective_matrix_and_policy(reference):
    policy = SharedWritePolicy.from_reference(reference, shared_groups=2, train_critic=False)
    with torch.no_grad():
        policy.source_basis.normal_(mean=0.0, std=0.1)
        policy._nodes["1"].A.normal_(mean=0.0, std=0.2)
    shared = np.arange(18, dtype=np.float32).reshape(2, 9) / 13.0
    policy.install_shared_matrix(shared)
    policy.set_slot_group(0, 0, preserve_effective=False)

    local_before = policy._nodes["1"].A.detach().clone()
    old_shared = policy._shared_coefficients[0].C.detach().clone()
    new_shared = policy._shared_coefficients[1].C.detach().clone()
    effective_before = local_before + old_shared.unsqueeze(0)
    obs = observations([1, 1, 1], [0.1, 0.5, 0.9])
    output_before = correction(policy, obs).detach().clone()

    policy.set_slot_group(0, 1, preserve_effective=True)
    effective_after = policy._nodes["1"].A.detach() + new_shared.unsqueeze(0)
    output_after = correction(policy, obs).detach().clone()

    torch.testing.assert_close(effective_after, effective_before, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(output_after, output_before, rtol=1e-6, atol=1e-7)
    assert policy.group_for_slot(0) == 1


def test_matrix_install_reset_zero_group_count_and_state_dict_roundtrip(reference):
    policy = SharedWritePolicy.from_reference(reference, shared_groups=2, train_critic=False)
    local = np.arange(162, dtype=np.float32).reshape(6, 27) / 100.0
    shared = np.arange(18, dtype=np.float32).reshape(2, 9) / 20.0
    policy.install_local_matrix(local)
    policy.install_shared_matrix(shared)
    np.testing.assert_array_equal(policy.read_local_matrix(), local)
    np.testing.assert_array_equal(policy.read_shared_matrix(), shared)
    policy.set_slot_group(0, 1, preserve_effective=False)
    policy.reset_slot(0)
    assert policy.group_for_slot(0) == -1
    assert np.count_nonzero(policy.read_local_matrix()[0]) == 0

    restored = SharedWritePolicy.from_reference(reference, shared_groups=2, train_critic=False)
    restored.load_state_dict(policy.state_dict(), strict=True)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    obs = observations([0, 1, 4, 6], [0.1, 0.4, 0.8, 0.9])
    torch.testing.assert_close(correction(restored, obs), correction(policy, obs), rtol=0, atol=0)
    assert restored.slot_groups.tolist() == policy.slot_groups.tolist()
    assert restored.source_basis.requires_grad is False

    independent = SharedWritePolicy.from_reference(reference, shared_groups=0, train_critic=False)
    assert independent.read_shared_matrix().shape == (0, 9)
    independent.set_slot_group(0, -1)
    with pytest.raises(ValueError, match="shared group"):
        independent.set_slot_group(0, 0)


def test_freeze_shared_survives_optimizer_rebuild(reference):
    policy = SharedWritePolicy.from_reference(reference, train_critic=False)
    policy.freeze_shared()
    optimizer = policy.make_optimizer(learning_rate=0.01, transfer=False)
    assert policy.source_basis.requires_grad is False
    assert all(not parameter.requires_grad for parameter in policy.shared_parameters())
    assert all(parameter.requires_grad for parameter in (node.A for node in policy._nodes.values()))
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert all(id(parameter) not in optimized for parameter in policy.shared_parameters())


"""Checks for packed and active-only residual storage."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.compression import PackedResidualPolicy, export_payload  # noqa: E402
from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402


torch.set_num_threads(1)


@pytest.fixture
def policy() -> TreePolicy:
    torch.manual_seed(811)
    result = TreePolicy(
        observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )
    with torch.no_grad():
        result.residuals[1].copy_(torch.randn_like(result.residuals[1]) + 0.01)
        result.residuals[2].copy_(torch.randn_like(result.residuals[2]) - 0.02)
    result.set_path(1, [])
    result.set_path(2, [])
    return result


def _observations(routes: list[int]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(159)
    observations = torch.rand((len(routes), OBS_DIM), generator=generator)
    observations[:, -1] = torch.as_tensor(routes, dtype=torch.float32)
    return observations


def _tensor_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in state.values())


def _seed_dense_optimizer(policy: TreePolicy) -> torch.optim.Optimizer:
    policy.configure_learning("history", learning_rate=0.013)
    optimizer = policy.optimizer
    optimizer.zero_grad(set_to_none=True)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    sum(parameter.square().sum() for parameter in trainable).backward()
    optimizer.step()
    return optimizer


def test_full_packing_matches_dense_policy_for_mixed_routes(policy: TreePolicy) -> None:
    observations = _observations([0, 1, 2, 3, 2, 1, 0, 3])
    packed = PackedResidualPolicy.from_dense(policy, [1, 2])
    dense_control = PackedResidualPolicy.from_dense(policy, [1, 2], storage="dense")

    with torch.no_grad():
        expected = policy.get_distribution(observations).distribution.probs
        packed_probs = packed.get_distribution(observations).distribution.probs
        dense_probs = dense_control.get_distribution(observations).distribution.probs
        expected_values = policy.predict_values(observations)
        packed_values = packed.predict_values(observations)

    torch.testing.assert_close(packed_probs, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(dense_probs, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(packed_values, expected_values, rtol=1e-6, atol=1e-7)
    assert not any(key.startswith("residuals.") for key in packed.state_dict())
    assert "path_masks" not in packed.state_dict()
    assert packed.state_dict()["_nodes.1.indices"].dtype == torch.uint16
    assert packed.state_dict()["_nodes.2.indices"].dtype == torch.uint16
    assert "indices" not in str(packed.to_spec())


def test_active_only_dense_is_fair_control_and_full_index_overhead_is_reported(
    policy: TreePolicy,
) -> None:
    active_dense = PackedResidualPolicy.from_dense(policy, [1, 2], storage="dense")
    full_packed = PackedResidualPolicy.from_dense(policy, [1, 2], retain_fraction=1.0)
    original_bytes = _tensor_bytes(policy.state_dict())
    dense_bytes = active_dense.persistent_tensor_bytes()
    packed_bytes = full_packed.persistent_tensor_bytes()

    assert dense_bytes < original_bytes  # six unused residual slots and path mask removed
    assert packed_bytes < original_bytes
    assert packed_bytes > dense_bytes  # two uint16 coordinate arrays cost more at full retention
    packed_report = full_packed.storage_report()
    dense_report = active_dense.storage_report()
    assert packed_report["coordinate_index_bytes"] == 2 * 4 * 65 * 2
    assert dense_report["coordinate_index_bytes"] == 0
    assert packed_report["persistent_policy_tensor_bytes"] == packed_bytes


def test_dense_adam_state_transfers_by_parameter_name_and_coordinates(policy: TreePolicy) -> None:
    dense_optimizer = _seed_dense_optimizer(policy)
    packed = PackedResidualPolicy.from_dense(policy, [1, 2], retain_fraction=0.25)
    optimizer = packed.make_optimizer(dense_optimizer=dense_optimizer)

    node = packed._nodes["1"]
    dense_state = dense_optimizer.state[policy.residuals[1]]
    packed_state = optimizer.state[node.values]
    coordinates = node.indices.to(torch.long)
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            packed_state[key], dense_state[key].reshape(-1).index_select(0, coordinates)
        )
    torch.testing.assert_close(packed_state["step"], dense_state["step"])
    assert optimizer.step_count == dense_optimizer.step_count
    assert optimizer.param_groups[0]["lr"] == dense_optimizer.param_groups[0]["lr"]
    assert optimizer.param_groups[0]["eps"] == dense_optimizer.param_groups[0]["eps"]

    critic_name = "mlp_extractor.value_net.0.weight"
    packed_named = dict(packed.named_parameters())
    dense_named = dict(policy.named_parameters())
    assert critic_name in dense_named and critic_name in packed_named
    torch.testing.assert_close(
        optimizer.state[packed_named[critic_name]]["exp_avg"],
        dense_optimizer.state[dense_named[critic_name]]["exp_avg"],
    )


def test_idle_packed_node_does_not_advance_adam_state(policy: TreePolicy) -> None:
    dense_optimizer = _seed_dense_optimizer(policy)
    packed = PackedResidualPolicy.from_dense(policy, [1, 2])
    optimizer = packed.make_optimizer(dense_optimizer=dense_optimizer)
    node2 = packed._nodes["2"].values
    state_before = {key: value.clone() for key, value in optimizer.state[node2].items()}
    value_before = node2.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    observations = _observations([1, 1, 1, 1])
    actions = torch.zeros(4, dtype=torch.long)
    loss = -packed.get_distribution(observations).log_prob(actions).mean()
    loss.backward()
    assert packed._nodes["1"].values.grad is not None
    assert node2.grad is None
    node2.grad = torch.zeros_like(node2)
    optimizer.step()

    torch.testing.assert_close(node2, value_before, rtol=0, atol=0)
    for key, value in state_before.items():
        torch.testing.assert_close(optimizer.state[node2][key], value, rtol=0, atol=0)


def test_payload_roundtrip_recreates_policy_and_named_optimizer(policy: TreePolicy) -> None:
    dense_optimizer = _seed_dense_optimizer(policy)
    packed = PackedResidualPolicy.from_dense(policy, [1, 2], retain_fraction=0.75)
    optimizer = packed.make_optimizer(dense_optimizer=dense_optimizer)
    router = {"active": 2, "prototypes": [[1.0, 0.0], [0.0, 1.0]]}
    rng = {"torch": torch.tensor([5, 7], dtype=torch.uint8)}
    payload = export_payload(
        packed,
        optimizer,
        router,
        metadata={"checkpoint": "B_history_keep_s67"},
        rng=rng,
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    restored = PackedResidualPolicy.from_payload(loaded)

    restored_policy = restored["policy"]
    restored_optimizer = restored["optimizer"]
    for key, value in packed.state_dict().items():
        torch.testing.assert_close(restored_policy.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(
        restored_policy.get_distribution(_observations([0, 1, 2, 3])).distribution.probs,
        packed.get_distribution(_observations([0, 1, 2, 3])).distribution.probs,
    )
    assert restored["router"] == router
    assert restored["metadata"] == {"checkpoint": "B_history_keep_s67"}
    torch.testing.assert_close(restored["rng"]["torch"], rng["torch"])
    assert restored_optimizer.step_count == optimizer.step_count
    original_manifest = packed.optimizer_manifest(optimizer)
    restored_manifest = restored_policy.optimizer_manifest(restored_optimizer)
    assert [g["param_names"] for g in restored_manifest["param_groups"]] == [
        g["param_names"] for g in original_manifest["param_groups"]
    ]
    for name in ("_nodes.1.values", "mlp_extractor.value_net.0.weight"):
        for key, value in original_manifest["state"][name].items():
            actual = restored_manifest["state"][name][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value


def test_quarter_retention_has_finite_loss_and_gradients(policy: TreePolicy) -> None:
    packed = PackedResidualPolicy.from_dense(policy, [1, 2], retain_fraction=0.25)
    assert packed._nodes["1"].values.numel() == 65
    assert packed._nodes["2"].values.numel() == 65

    observations = _observations([1, 2, 1, 2, 0, 3])
    actions = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    values, log_prob, entropy = packed.evaluate_actions(observations, actions)
    loss = values.mean() + log_prob.mean() + entropy.mean()
    assert torch.isfinite(loss)
    loss.backward()
    for node in ("1", "2"):
        gradient = packed._nodes[node].values.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    assert packed.persistent_tensor_bytes() < PackedResidualPolicy.from_dense(
        policy, [1, 2], storage="dense"
    ).persistent_tensor_bytes()

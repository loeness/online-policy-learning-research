"""Behavioral checks for alternative active-leaf residual layouts."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.alternative_compression import compress_policy, restore_payload  # noqa: E402
from online_rl.compression import PackedResidualPolicy, export_payload  # noqa: E402
from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402


torch.set_num_threads(1)
METHODS = (
    "dense",
    "topk50",
    "contrast",
    "svd2",
    "columns32",
    "bf16",
    "center_topk50",
    "activation_topk50",
)


@pytest.fixture
def reference() -> PackedResidualPolicy:
    torch.manual_seed(913)
    source = TreePolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )
    with torch.no_grad():
        source.residuals[1].copy_(torch.randn_like(source.residuals[1]) + 0.2)
        source.residuals[2].copy_(torch.randn_like(source.residuals[2]) - 0.1)
    source.set_path(1, [])
    source.set_path(2, [])
    return PackedResidualPolicy.from_dense(source, [1, 2], storage="dense")


def observations(routes: list[int]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(451)
    obs = torch.rand((len(routes), OBS_DIM), generator=generator)
    obs[:, -1] = torch.tensor(routes, dtype=torch.float32)
    return obs


def _probs(policy, obs):
    return policy.get_distribution(obs).distribution.probs


def _one_update(policy, optimizer):
    obs = observations([1, 2, 1, 2, 1, 2])
    actions = torch.tensor([0, 2, 1, 3, 0, 1], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    values, log_prob, _ = policy.evaluate_actions(obs, actions)
    loss = -log_prob.mean() + 0.03 * values.square().mean()
    loss.backward()
    optimizer.step()
    return obs


def test_contrast_removes_only_the_softmax_gauge(reference: PackedResidualPolicy) -> None:
    compressed, _ = compress_policy(reference, "contrast")
    obs = observations([0, 1, 2, 3, 2, 1])
    torch.testing.assert_close(_probs(compressed, obs), _probs(reference, obs), rtol=2e-6, atol=2e-7)
    assert compressed._nodes["1"].values.shape == (3, 65)
    assert compressed._nodes["2"].values.shape == (3, 65)


@pytest.mark.parametrize("method", METHODS)
def test_layout_and_optimizer_roundtrip_after_update(reference: PackedResidualPolicy, method: str) -> None:
    calibration = observations([1, 1, 1, 2, 2, 2])
    policy, _ = compress_policy(reference, method, calibration_obs=calibration)
    optimizer = policy.make_optimizer(learning_rate=0.003, transfer=False)
    obs = _one_update(policy, optimizer)

    payload = export_payload(policy, optimizer, {"active": 2}, metadata={"method": method})
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    restored = restore_payload(loaded)

    restored_policy = restored["policy"]
    restored_optimizer = restored["optimizer"]
    assert restored_policy._storage == method
    for name, value in policy.state_dict().items():
        restored_value = restored_policy.state_dict()[name]
        assert restored_value.dtype == value.dtype
        torch.testing.assert_close(restored_value, value, rtol=0, atol=0)
    torch.testing.assert_close(_probs(restored_policy, obs), _probs(policy, obs), rtol=2e-6, atol=2e-7)
    assert restored_optimizer.step_count == optimizer.step_count == 1
    assert restored["router"] == {"active": 2}
    assert restored["metadata"] == {"method": method}

    before_manifest = policy.optimizer_manifest(optimizer)
    after_manifest = restored_policy.optimizer_manifest(restored_optimizer)
    assert [group["param_names"] for group in after_manifest["param_groups"]] == [
        group["param_names"] for group in before_manifest["param_groups"]
    ]
    assert before_manifest["state"].keys() == after_manifest["state"].keys()
    for name, before_state in before_manifest["state"].items():
        for key, value in before_state.items():
            actual = after_manifest["state"][name][key]
            if torch.is_tensor(value):
                assert actual.dtype == value.dtype
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value


def test_bf16_uses_native_bfloat16_moments_without_fp32_master(reference: PackedResidualPolicy) -> None:
    policy, _ = compress_policy(reference, "bf16")
    optimizer = policy.make_optimizer(learning_rate=0.003)
    _one_update(policy, optimizer)

    for node in policy._nodes.values():
        assert node.values.dtype == torch.bfloat16
        state = optimizer.state[node.values]
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16
        assert state["step"].dtype == torch.float32
        assert not any(
            torch.is_tensor(value)
            and value.dtype == torch.float32
            and value.shape == node.values.shape
            for value in state.values()
        )


def test_svd_factors_columns_and_frozen_actor_have_expected_shapes(reference: PackedResidualPolicy) -> None:
    svd, _ = compress_policy(reference, "svd2")
    columns, _ = compress_policy(reference, "columns32", calibration_obs=observations([1, 2, 1, 2]))

    assert svd._nodes["1"].left.shape == (3, 2)
    assert svd._nodes["1"].right.shape == (2, 65)
    assert svd.storage_report()["learned_coordinate_count"] == 2 * (6 + 130)
    assert columns._nodes["1"].values.shape == (4, 32)
    assert columns._nodes["1"].indices.dtype == torch.uint16
    assert columns._nodes["1"].indices.numel() == 32
    assert 64 in columns._nodes["1"].indices.to(torch.long).tolist()
    assert columns.storage_report()["learned_coordinate_count"] == 2 * 4 * 32
    assert columns.storage_report()["index_bytes"] == 2 * 32 * 2

    for name, parameter in columns.named_parameters():
        if name.startswith("mlp_extractor.policy_net") or name.startswith("action_net"):
            assert not parameter.requires_grad
    for name, parameter in columns.named_parameters():
        if name.startswith("_nodes."):
            assert parameter.requires_grad
            assert getattr(parameter, "tree_residual", False)

    optimizer = svd.make_optimizer()
    optimizer.zero_grad(set_to_none=True)
    obs = observations([1, 1, 1])
    actions = torch.tensor([0, 2, 3], dtype=torch.long)
    loss = -svd.get_distribution(obs).log_prob(actions).mean()
    loss.backward()
    for factor in (svd._nodes["1"].left, svd._nodes["1"].right):
        assert factor.grad is not None
        assert torch.isfinite(factor.grad).all()


def test_columns_and_activation_selection_have_distinct_calibration_ranking(
    reference: PackedResidualPolicy,
) -> None:
    calibration = observations([1, 1, 1, 1, 2, 2, 2, 2])
    centered, _ = compress_policy(reference, "center_topk50", calibration_obs=calibration)
    activation, _ = compress_policy(reference, "activation_topk50", calibration_obs=calibration)
    for node_id in ("1", "2"):
        centered_node = centered._nodes[node_id]
        activation_node = activation._nodes[node_id]
        assert centered_node.indices.numel() == activation_node.indices.numel() == 130
        assert centered_node.indices.dtype == activation_node.indices.dtype == torch.uint16
        assert centered_node.values.shape == activation_node.values.shape == (130,)


def test_leafadam_leaves_unused_route_gradient_absent(reference: PackedResidualPolicy) -> None:
    policy, _ = compress_policy(reference, "svd2")
    optimizer = policy.make_optimizer()
    optimizer.zero_grad(set_to_none=True)
    actions = torch.tensor([0, 1, 2], dtype=torch.long)
    loss = -policy.get_distribution(observations([1, 1, 1])).log_prob(actions).mean()
    loss.backward()
    assert policy._nodes["1"].left.grad is not None
    assert policy._nodes["2"].left.grad is None
    optimizer.step()
    assert policy._nodes["2"].left not in optimizer.state


def test_restore_payload_dispatches_legacy_packed_layout(reference: PackedResidualPolicy) -> None:
    optimizer = reference.make_optimizer(learning_rate=0.003, transfer=False)
    payload = export_payload(reference, optimizer, router_snapshot=None)
    restored = restore_payload(payload)
    assert type(restored["policy"]) is PackedResidualPolicy
    assert restored["policy"]._storage == "dense"


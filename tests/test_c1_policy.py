"""Behavioral checks for factorized residual policies in C1."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.c1_policy import C1ResidualPolicy  # noqa: E402
from online_rl.compression import export_payload  # noqa: E402
from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402


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


def test_zero_residual_keeps_foundation_logits_and_applies_epsilon_mixture(reference: TreePolicy) -> None:
    core = C1ResidualPolicy.from_reference(reference)
    obs = observations([0, 1, 2, 3, 4, 5, 6], [0.05, 0.18, 0.38, 0.52, 0.70, 0.85, 1.0])
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        base_logits = core.action_net(latent)
        correction = core._correction(obs, latent)
        probabilities = core.get_distribution(obs).distribution.probs
        expected = (1.0 - core.epsilon) * base_logits.softmax(-1) + core.epsilon / 4.0

    torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0, atol=0)
    torch.testing.assert_close(probabilities, expected, rtol=1e-6, atol=1e-7)
    source_probs = reference.get_distribution(obs).distribution.probs
    assert torch.equal(probabilities.argmax(-1), source_probs.argmax(-1))
    for name, parameter in core.named_parameters():
        if name.startswith("mlp_extractor.policy_net") or name.startswith("action_net"):
            assert not parameter.requires_grad
    assert any(parameter.requires_grad for parameter in core.mlp_extractor.value_net.parameters())
    assert any(getattr(parameter, "tree_residual", False) for parameter in core.parameters())


def test_local_feature_bank_is_one_hot_by_observed_x_band(reference: TreePolicy) -> None:
    core = C1ResidualPolicy.from_reference(reference)
    obs = observations([1, 2, 3], [0.1, 0.5, 0.9])
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        base_phi, local_phi = core._feature_banks(obs, latent)

    assert base_phi.shape == (3, 65)
    assert local_phi.shape == (3, 195)
    local = local_phi.reshape(3, 3, 65)
    torch.testing.assert_close(local[0, 0], base_phi[0])
    torch.testing.assert_close(local[1, 1], base_phi[1])
    torch.testing.assert_close(local[2, 2], base_phi[2])
    assert torch.count_nonzero(local[0, 1:]) == 0
    assert torch.count_nonzero(local[1, [0, 2]]) == 0
    assert torch.count_nonzero(local[2, :2]) == 0


def test_temperature_scales_logits_before_epsilon_mixture(reference: TreePolicy) -> None:
    core = C1ResidualPolicy.from_reference(reference, temperature=2.0, exploration_epsilon=0.05)
    obs = observations([0, 1, 3, 6], [0.1, 0.3, 0.6, 0.9])
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        logits = core.action_net(latent) + core._correction(obs, latent)
        expected = (1.0 - core.epsilon) * torch.softmax(logits / 2.0, dim=-1) + core.epsilon / 4.0
        actual = core.get_distribution(obs).distribution.probs
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="temperature"):
        C1ResidualPolicy.from_reference(reference, temperature=0.0)


def test_tree_endpoint_adds_parent_and_local_leaf_paths(reference: TreePolicy) -> None:
    core = C1ResidualPolicy.from_reference(reference, method="tree", leaf_rank=1)
    obs = observations([1], [0.5])
    with torch.no_grad():
        parent = core._parents["7"]
        leaf = core._nodes["1"]
        parent.U[0, 0] = 0.7
        parent.V[0, -1] = 0.4
        leaf.U[1, 0] = -0.3
        leaf.V[0, 65 + 64] = 0.6
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        base_phi, local_phi = core._feature_banks(obs, latent)
        expected = torch.nn.functional.linear(base_phi, core._parent_matrix(7))
        expected = expected + torch.nn.functional.linear(local_phi, core._effective_matrix(1))
        actual = core._correction(obs, latent)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert core._paths[1] == (7, 1)
    assert core._node_shapes[7] == (3, 65)
    assert core._node_shapes[1] == (3, 195)


def test_flat_parent_fallback_averages_first_two_siblings_without_parameter_gradients(
    reference: TreePolicy,
) -> None:
    core = C1ResidualPolicy.from_reference(reference, method="dense")
    with torch.no_grad():
        core._nodes["1"].values.fill_(0.2)
        core._nodes["2"].values.fill_(0.6)
    obs = observations([7], [0.5])
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        _, local_phi = core._feature_banks(obs, latent)
        expected_matrix = (core._effective_matrix(1) + core._effective_matrix(2)) * 0.5
        expected = torch.nn.functional.linear(local_phi, expected_matrix)
        actual = core._correction(obs, latent)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert not actual.requires_grad
    assert all(parameter.grad is None for parameter in core._residual_parameters())


def test_leafadam_keeps_idle_tree_leaf_and_parent_gradients_absent(reference: TreePolicy) -> None:
    core = C1ResidualPolicy.from_reference(reference, method="tree", leaf_rank=1)
    optimizer = core.make_optimizer(learning_rate=0.01, transfer=False)
    obs = observations([1, 1, 1], [0.1, 0.2, 0.3])
    actions = torch.tensor([0, 2, 3])
    optimizer.zero_grad(set_to_none=True)
    loss = -core.get_distribution(obs).log_prob(actions).mean()
    loss.backward()

    assert core._nodes["1"].U.grad is not None
    assert core._nodes["2"].U.grad is None
    assert core._parents["7"].U.grad is not None
    assert core._parents["8"].U.grad is None
    optimizer.step()
    assert core._parents["8"].U not in optimizer.state


@pytest.mark.parametrize("method", ["flat", "dictionary", "dense", "tree"])
def test_initialize_heldout_uses_sibling_mean_or_parent_transfer(reference: TreePolicy, method: str) -> None:
    core = C1ResidualPolicy.from_reference(
        reference,
        method=method,
        leaf_rank=2,
        dictionary_rank=6,
    )
    with torch.no_grad():
        if method == "flat":
            for node_id, scale in (("1", 1.0), ("2", 3.0)):
                node = core._nodes[node_id]
                node.U.zero_()
                node.V.zero_()
                node.U[:2, :2].copy_(torch.eye(2))
                node.V[:2, :8].fill_(scale)
            expected = (core._nodes["1"].U @ core._nodes["1"].V + core._nodes["2"].U @ core._nodes["2"].V) / 2
            core.initialize_heldout(3)
            torch.testing.assert_close(core._nodes["3"].U @ core._nodes["3"].V, expected, rtol=2e-5, atol=1e-6)
        elif method == "dictionary":
            core._nodes["1"].A.fill_(1.0)
            core._nodes["2"].A.fill_(3.0)
            core.initialize_heldout(3)
            torch.testing.assert_close(core._nodes["3"].A, torch.full_like(core._nodes["3"].A, 2.0))
        elif method == "dense":
            core._nodes["1"].values.fill_(0.25)
            core._nodes["2"].values.fill_(0.75)
            core.initialize_heldout(3)
            torch.testing.assert_close(core._nodes["3"].values, torch.full_like(core._nodes["3"].values, 0.5))
        else:
            core._nodes["3"].U.fill_(2.0)
            core._nodes["3"].V.fill_(2.0)
            previous_u = core._nodes["3"].U.detach().clone()
            core.initialize_heldout(3)
            torch.testing.assert_close(core._nodes["3"].U, previous_u)
            assert torch.count_nonzero(core._nodes["3"].V) == 0


def test_tree_heldout_leaf_starts_at_zero_and_can_learn_after_one_step(reference: TreePolicy) -> None:
    core = C1ResidualPolicy.from_reference(reference, method="tree", leaf_rank=1)
    node = core._nodes["3"]
    with torch.no_grad():
        node.U.zero_()
        node.V.fill_(3.0)
    core.initialize_heldout(3)
    assert torch.count_nonzero(node.U) > 0
    assert torch.count_nonzero(node.V) == 0

    obs = observations([3], [0.5])
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        assert torch.count_nonzero(core._correction(obs, latent)) == 0

    optimizer = core.make_optimizer(learning_rate=0.01, transfer=False)
    optimizer.zero_grad(set_to_none=True)
    loss = -core.get_distribution(obs).log_prob(torch.tensor([1])).mean()
    loss.backward()
    assert node.V.grad is not None
    assert torch.isfinite(node.V.grad).all()
    assert torch.count_nonzero(node.V.grad) > 0
    optimizer.step()
    assert torch.count_nonzero(node.V) > 0


def test_payload_roundtrip_restores_policy_predictions_and_leafadam_manifest(reference: TreePolicy) -> None:
    core = C1ResidualPolicy.from_reference(
        reference, method="dense", exploration_epsilon=0.05, temperature=2.0
    )
    optimizer = core.make_optimizer(learning_rate=0.003, transfer=False)
    obs = observations([1, 2, 1, 2, 7, 8], [0.1, 0.4, 0.2, 0.6, 0.5, 0.8])
    actions = torch.tensor([0, 2, 1, 3, 0, 1])
    optimizer.zero_grad(set_to_none=True)
    values, log_prob, _ = core.evaluate_actions(obs, actions)
    (values.mean() - log_prob.mean()).backward()
    optimizer.step()
    payload = export_payload(core, optimizer, {"route": 2}, metadata={"layout": "dense"})
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    restored = C1ResidualPolicy.from_payload(loaded)

    restored_core = restored["policy"]
    restored_optimizer = restored["optimizer"]
    assert restored_core._storage == "dense"
    assert restored_core.epsilon == pytest.approx(0.05)
    assert restored_core.temperature == pytest.approx(2.0)
    assert restored["router"] == {"route": 2}
    assert restored["metadata"] == {"layout": "dense"}
    assert restored_optimizer.step_count == optimizer.step_count == 1
    for name, value in core.state_dict().items():
        torch.testing.assert_close(restored_core.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(
        restored_core.get_distribution(obs).distribution.probs,
        core.get_distribution(obs).distribution.probs,
        rtol=0,
        atol=0,
    )
    before = core.optimizer_manifest(optimizer)
    after = restored_core.optimizer_manifest(restored_optimizer)
    assert [group["param_names"] for group in before["param_groups"]] == [
        group["param_names"] for group in after["param_groups"]
    ]
    assert before["state"].keys() == after["state"].keys()
    for name, state in before["state"].items():
        for key, value in state.items():
            actual = after["state"][name][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value
    report = core.storage_report()
    assert report["persistent_policy_tensor_bytes"] == core.persistent_tensor_bytes()
    assert report["residual_value_bytes"] == sum(
        parameter.numel() * parameter.element_size() for parameter in core._residual_parameters()
    )
    assert report["coordinate_index_bytes"] == 0
    assert not any(name.startswith("residuals.") or name == "path_masks" for name in core.state_dict())

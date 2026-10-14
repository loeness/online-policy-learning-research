"""Focused behavioral checks for the C1 flat grouped controls."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.c1_modular_controls import (  # noqa: E402
    FlatGroupedAdditivePolicy,
    GroupedDictionaryPolicy,
)
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


def _same_state(left: torch.nn.Module, right: torch.nn.Module) -> None:
    assert list(left.state_dict()) == list(right.state_dict())
    for name, value in left.state_dict().items():
        torch.testing.assert_close(right.state_dict()[name], value, rtol=0, atol=0)


def _same_manifest(left: dict, right: dict) -> None:
    assert [g["param_names"] for g in left["param_groups"]] == [
        g["param_names"] for g in right["param_groups"]
    ]
    assert left["state"].keys() == right["state"].keys()
    assert left["step_count"] == right["step_count"]
    for name, state in left["state"].items():
        for key, value in state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(right["state"][name][key], value, rtol=0, atol=0)
            else:
                assert right["state"][name][key] == value


def test_flat_grouped_additive_is_tree_equivalent_and_manifest_transferable(reference: TreePolicy) -> None:
    torch.manual_seed(91)
    tree = C1ResidualPolicy.from_reference(
        reference, method="tree", leaf_rank=2, exploration_epsilon=0.05, temperature=2.0
    )
    torch.manual_seed(91)
    flat = FlatGroupedAdditivePolicy.from_reference(
        reference, leaf_rank=2, exploration_epsilon=0.05, temperature=2.0
    )

    _same_state(tree, flat)
    assert sum(parameter.numel() for parameter in tree.parameters()) == sum(
        parameter.numel() for parameter in flat.parameters()
    )
    assert tree.persistent_tensor_bytes() == flat.persistent_tensor_bytes()
    assert flat._paths == {}
    assert flat.to_spec()["group_membership"] == {"7": [1, 2, 3], "8": [4, 5, 6]}
    assert flat.to_spec()["paths"] == {}

    # Make the low-rank adapters nonzero so both routes and gradient paths are exercised.
    generator = torch.Generator().manual_seed(22)
    with torch.no_grad():
        for parameter in tree._residual_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.08)
    flat.load_state_dict(tree.state_dict(), strict=True)

    obs = observations([1, 2, 3, 4, 5, 6, 7, 8], [0.1, 0.4, 0.9, 0.2, 0.55, 0.8, 0.35, 0.7])
    actions = torch.tensor([0, 2, 1, 3, 0, 1, 2, 3])
    with torch.no_grad():
        latent_tree = tree.mlp_extractor.forward_actor(tree.extract_features(obs))
        latent_flat = flat.mlp_extractor.forward_actor(flat.extract_features(obs))
        torch.testing.assert_close(latent_flat, latent_tree, rtol=0, atol=0)
        correction_tree = tree._correction(obs, latent_tree)
        correction_flat = flat._correction(obs, latent_flat)
        torch.testing.assert_close(correction_flat, correction_tree, rtol=0, atol=0)
        probabilities_tree = tree.get_distribution(obs).distribution.probs
        probabilities_flat = flat.get_distribution(obs).distribution.probs
        torch.testing.assert_close(probabilities_flat, probabilities_tree, rtol=0, atol=0)

    opt_tree = tree.make_optimizer(learning_rate=0.003, transfer=False)
    opt_flat = flat.make_optimizer(learning_rate=0.003, transfer=False)
    opt_tree.zero_grad(set_to_none=True)
    opt_flat.zero_grad(set_to_none=True)
    value_tree, logp_tree, entropy_tree = tree.evaluate_actions(obs, actions)
    value_flat, logp_flat, entropy_flat = flat.evaluate_actions(obs, actions)
    loss_tree = value_tree.mean() - logp_tree.mean() + 0.01 * entropy_tree.mean()
    loss_flat = value_flat.mean() - logp_flat.mean() + 0.01 * entropy_flat.mean()
    loss_tree.backward()
    loss_flat.backward()
    for (tree_name, tree_parameter), (flat_name, flat_parameter) in zip(
        tree.named_parameters(), flat.named_parameters(), strict=True
    ):
        assert tree_name == flat_name
        if tree_parameter.grad is None:
            assert flat_parameter.grad is None
        else:
            torch.testing.assert_close(flat_parameter.grad, tree_parameter.grad, rtol=0, atol=0)
    opt_tree.step()
    opt_flat.step()
    _same_state(tree, flat)
    _same_manifest(tree.optimizer_manifest(opt_tree), flat.optimizer_manifest(opt_flat))

    copied = FlatGroupedAdditivePolicy.from_tree(tree)
    _same_state(tree, copied)
    transferred = copied._optimizer_from_manifest(tree.optimizer_manifest(opt_tree))
    _same_manifest(tree.optimizer_manifest(opt_tree), copied.optimizer_manifest(transferred))


def test_flat_grouped_additive_payload_roundtrip(reference: TreePolicy) -> None:
    tree = C1ResidualPolicy.from_reference(reference, method="tree", leaf_rank=1)
    flat = FlatGroupedAdditivePolicy.from_tree(tree)
    optimizer = flat.make_optimizer(learning_rate=0.004, transfer=False)
    obs = observations([1, 2, 7], [0.2, 0.65, 0.4])
    loss = -flat.get_distribution(obs).log_prob(torch.tensor([0, 3, 1])).mean()
    loss.backward()
    optimizer.step()

    restored = FlatGroupedAdditivePolicy.from_payload(
        export_payload(flat, optimizer, {"route": 2}, metadata={"control": "grouped_additive"})
    )
    restored_policy = restored["policy"]
    assert restored_policy._paths == {}
    assert restored["router"] == {"route": 2}
    assert restored["metadata"] == {"control": "grouped_additive"}
    _same_state(flat, restored_policy)
    _same_manifest(
        flat.optimizer_manifest(optimizer),
        restored_policy.optimizer_manifest(restored["optimizer"]),
    )
    torch.testing.assert_close(
        restored_policy.get_distribution(obs).distribution.probs,
        flat.get_distribution(obs).distribution.probs,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("rank", "warm_bytes", "limit"),
    [(3, 14_688, 15_952), (6, 29_376, 30_700)],
)
def test_grouped_dictionary_shapes_and_warmed_residual_budget(
    reference: TreePolicy, rank: int, warm_bytes: int, limit: int
) -> None:
    core = GroupedDictionaryPolicy.from_reference(reference, dictionary_rank=rank, train_critic=False)
    report = core.storage_report()

    assert core.shared_V.shape == (2, rank, 195)
    assert core.group_membership == {7: (1, 2, 3), 8: (4, 5, 6)}
    assert core._paths == {}
    assert report["shared_basis_parameters"] == 2 * rank * 195
    assert report["leaf_coefficient_parameters"] == 6 * 3 * rank
    assert report["residual_value_bytes"] == (2 * rank * 195 + 6 * 3 * rank) * 4
    assert report["fully_warmed_residual_and_adam_bytes"] == warm_bytes
    assert warm_bytes <= limit
    assert report["coordinate_index_bytes"] == 0


def test_grouped_dictionary_local_gate_and_idle_family_adam(reference: TreePolicy) -> None:
    core = GroupedDictionaryPolicy.from_reference(reference, dictionary_rank=3, train_critic=False)
    optimizer = core.make_optimizer(learning_rate=0.01, transfer=False)
    obs_a = observations([1], [0.8])
    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(obs_a).log_prob(torch.tensor([0])).mean()).backward()

    v_a = core._group_bases["7"].values
    v_b = core._group_bases["8"].values
    assert v_a.grad is not None
    assert v_b.grad is None
    assert core._nodes["1"].A.grad is not None
    assert core._nodes["2"].A.grad is None
    optimizer.step()  # The zero-start coefficient learns before its basis can move.

    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(obs_a).log_prob(torch.tensor([0])).mean()).backward()
    assert v_a.grad is not None
    assert torch.count_nonzero(v_a.grad[:, : 2 * 65]) == 0
    assert torch.count_nonzero(v_a.grad[:, 2 * 65 :]) > 0
    optimizer.step()
    before_idle_family = v_a.detach().clone()
    state_before_idle = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[v_a].items()
    }

    obs_b = observations([4], [0.1])
    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(obs_b).log_prob(torch.tensor([3])).mean()).backward()
    assert v_a.grad is None
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(obs_b).log_prob(torch.tensor([3])).mean()).backward()
    assert v_a.grad is None
    assert v_b.grad is not None
    optimizer.step()
    torch.testing.assert_close(v_a, before_idle_family, rtol=0, atol=0)
    for key, value in state_before_idle.items():
        actual = optimizer.state[v_a][key]
        if torch.is_tensor(value):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        else:
            assert actual == value


def test_grouped_dictionary_seen_leaf_fallback_and_heldout_initialization(reference: TreePolicy) -> None:
    core = GroupedDictionaryPolicy.from_reference(reference, dictionary_rank=3, train_critic=False)
    v = core._group_bases["7"].values
    with torch.no_grad():
        v.zero_()
        v[0, 0] = 1.0
        core._nodes["1"].A[0, 0] = 1.0
        core._nodes["2"].A[0, 0] = 3.0
        core._nodes["3"].A[0, 0] = 9.0
    assert core.seen_mask.tolist() == [True, True, False, True, True, False]

    obs = observations([7], [0.1])
    latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
    _, local_phi = core._feature_banks(obs, latent)
    expected_first_two = torch.stack([core._effective_matrix(1), core._effective_matrix(2)]).mean(0)
    torch.testing.assert_close(core._fallback_correction(7, local_phi), torch.nn.functional.linear(local_phi, expected_first_two))
    assert not core._fallback_correction(7, local_phi).requires_grad

    core.initialize_heldout(3)
    torch.testing.assert_close(core._nodes["3"].A, (core._nodes["1"].A + core._nodes["2"].A) / 2)
    with torch.no_grad():
        core._nodes["3"].A[0, 0] = 9.0  # represent later adaptation of this leaf
    core.mark_seen(3)
    expected_all = torch.stack([core._effective_matrix(endpoint) for endpoint in (1, 2, 3)]).mean(0)
    torch.testing.assert_close(core._fallback_correction(7, local_phi), torch.nn.functional.linear(local_phi, expected_all))


def test_grouped_dictionary_payload_restores_groups_seen_mask_and_optimizer(reference: TreePolicy) -> None:
    core = GroupedDictionaryPolicy.from_reference(reference, dictionary_rank=3, train_critic=False)
    optimizer = core.make_optimizer(learning_rate=0.005, transfer=False)
    obs = observations([1, 4], [0.2, 0.7])
    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(obs).log_prob(torch.tensor([1, 2])).mean()).backward()
    optimizer.step()
    core.mark_seen(3)

    restored = GroupedDictionaryPolicy.from_payload(
        export_payload(core, optimizer, None, metadata={"control": "grouped_dictionary"})
    )
    clone = restored["policy"]
    assert clone.group_membership == core.group_membership
    assert clone._paths == {}
    assert clone.seen_mask.tolist() == core.seen_mask.tolist()
    assert clone.to_spec()["c1_residual"]["shared_basis_shape"] == [2, 3, 195]
    _same_state(core, clone)
    _same_manifest(core.optimizer_manifest(optimizer), clone.optimizer_manifest(restored["optimizer"]))
    torch.testing.assert_close(
        clone.get_distribution(obs).distribution.probs,
        core.get_distribution(obs).distribution.probs,
        rtol=0,
        atol=0,
    )

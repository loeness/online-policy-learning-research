"""Behavioral and capacity checks for the C1 source-basis tensor controls."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.c1_envs import C1Context, mapping_for_state  # noqa: E402
from online_rl.c1_policy import C1ResidualPolicy  # noqa: E402
from online_rl.c1_tensor_controls import TensorSourceBasisPolicy  # noqa: E402
from online_rl.compression import export_payload  # noqa: E402
from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl import c1_supplement_experiment  # noqa: E402
from online_rl.c1_tensor_experiment import _installed_factories  # noqa: E402


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


def materialized_correction(core: TensorSourceBasisPolicy, obs: torch.Tensor,
                            latent: torch.Tensor) -> torch.Tensor:
    """Reference path using the declared 3 by 195 effective matrices."""
    obs = core._as_batch(obs)
    ids = obs[:, -1].round().long().clamp(0, 8)
    base_phi, local_phi = core._feature_banks(obs, latent)
    result = latent.new_zeros((latent.shape[0], 4))
    for route in torch.unique(ids, sorted=True).tolist():
        route = int(route)
        rows = torch.nonzero(ids == route, as_tuple=False).reshape(-1)
        local = local_phi.index_select(0, rows)
        if route in range(1, 7):
            correction = torch.nn.functional.linear(local, core._effective_matrix(route))
        elif route in (7, 8):
            if core._storage == "hierarchical_output_mixer":
                reduced = core._families[str(route)].C @ core.source_basis
                family = core._contrast_basis @ reduced
                matrix = torch.cat([family, family, family], dim=1)
                correction = torch.nn.functional.linear(local, matrix)
            else:
                members = (1, 2, 3) if route == 7 else (4, 5, 6)
                seen = [member for member in members if bool(core.seen_mask[member - 1])]
                with torch.no_grad():
                    matrix = torch.stack([core._effective_matrix(i) for i in seen]).mean(0)
                correction = torch.nn.functional.linear(local, matrix)
        else:
            continue
        result = result.index_add(0, rows, correction)
    return result


@pytest.mark.parametrize("method", ["tensor_learned", "output_mixer", "hierarchical_output_mixer"])
def test_zero_coefficients_preserve_w0_logits_and_basis_comes_from_centered_head(reference, method):
    core = TensorSourceBasisPolicy.from_reference(
        reference, method=method, dictionary_rank=3, exploration_epsilon=0.05, temperature=2.0
    )
    obs = observations([1, 2, 3, 4, 5, 6, 7, 8], [0.08, 0.36, 0.69, 0.95, 0.21, 0.51, 0.81, 0.44])
    with torch.no_grad():
        head = torch.cat([core.action_net.weight, core.action_net.bias[:, None]], dim=1)
        expected_basis = core._contrast_basis.T @ head
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        base_logits = core.action_net(latent)
        correction = core._correction(obs, latent)

    torch.testing.assert_close(core.source_basis, expected_basis, rtol=0, atol=0)
    torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0, atol=0)
    assert core.source_basis.requires_grad is (method == "tensor_learned")
    assert core._paths == {}
    assert core.to_spec()["c1_residual"]["source_basis_initialization"] == "centered_W0_action_head"
    assert torch.isfinite(base_logits).all()


def test_test_only_permutation_is_exactly_representable_by_each_band_mixer(reference):
    """Use an environment mapping only as a mathematical representability test."""
    core = TensorSourceBasisPolicy.from_reference(reference, method="output_mixer", train_critic=False)
    context = C1Context("A", 1)
    permutation = mapping_for_state(context, 4)
    P = torch.zeros((4, 4))
    for action, physical_direction in enumerate(permutation):
        P[action, physical_direction] = 1.0
    contrast = core._contrast_basis
    reduced_permutation = contrast.T @ P @ contrast
    endpoint, band = 2, 1
    with torch.no_grad():
        core._nodes[str(endpoint)].A[band].copy_(reduced_permutation - torch.eye(3))

    obs = observations([endpoint], [0.5])
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        actual_logits = core.action_net(latent) + core._correction(obs, latent)
        base_logits = core.action_net(latent)
        expected_logits = base_logits @ P.T
        actual_probs = torch.softmax(actual_logits, dim=-1)
        expected_probs = torch.softmax(expected_logits, dim=-1)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("method", ["tensor_learned", "output_mixer", "hierarchical_output_mixer"])
def test_band_gating_coefficients_and_idle_leaf_adam(reference, method):
    core = TensorSourceBasisPolicy.from_reference(reference, method=method, train_critic=False)
    optimizer = core.make_optimizer(learning_rate=0.01, transfer=False)
    route_a, route_b = observations([1], [0.8]), observations([4], [0.1])

    def update(obs, action):
        optimizer.zero_grad(set_to_none=True)
        (-core.get_distribution(obs).log_prob(torch.tensor([action])).mean()).backward()

    update(route_a, 0)
    leaf_a = core._nodes["1"].A
    leaf_b = core._nodes["4"].A
    assert leaf_a.grad is not None
    assert leaf_b.grad is None
    assert torch.count_nonzero(leaf_a.grad[0:2]) == 0
    assert torch.count_nonzero(leaf_a.grad[2]) > 0
    if method == "tensor_learned":
        assert core.source_basis.grad is not None
    else:
        assert core.source_basis.grad is None
    if method == "hierarchical_output_mixer":
        assert core._families["7"].C.grad is not None
        assert core._families["8"].C.grad is None
    optimizer.step()
    before_value = leaf_a.detach().clone()
    before_state = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[leaf_a].items()
    }

    update(route_b, 3)
    assert leaf_a.grad is None
    assert leaf_b.grad is not None
    optimizer.step()
    torch.testing.assert_close(leaf_a, before_value, rtol=0, atol=0)
    for key, value in before_state.items():
        current = optimizer.state[leaf_a][key]
        if torch.is_tensor(value):
            torch.testing.assert_close(current, value, rtol=0, atol=0)
        else:
            assert current == value


def test_seen_fallback_expands_after_heldout_initialization(reference):
    core = TensorSourceBasisPolicy.from_reference(reference, method="output_mixer", train_critic=False)
    with torch.no_grad():
        core.source_basis.zero_()
        core.source_basis[0, 0] = 1.0
        core._nodes["1"].A[0, 0, 0] = 1.0
        core._nodes["2"].A[0, 0, 0] = 3.0
        core._nodes["3"].A[0, 0, 0] = 9.0
    assert core.seen_mask.tolist() == [True, True, False, True, True, False]
    core.initialize_heldout(3)
    torch.testing.assert_close(core._nodes["3"].A, (core._nodes["1"].A + core._nodes["2"].A) / 2)
    assert core.seen_mask.tolist() == [True, True, True, True, True, False]

    obs = observations([7], [0.1])
    latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
    _, local = core._feature_banks(obs, latent)
    all_seen = torch.stack([core._effective_matrix(i) for i in (1, 2, 3)]).mean(0)
    expected = torch.nn.functional.linear(local, all_seen)
    torch.testing.assert_close(core._fallback_correction(7, local), expected)
    assert not core._fallback_correction(7, local).requires_grad


@pytest.mark.parametrize(
    ("method", "adaptive_bytes"),
    [("tensor_learned", 4_312), ("output_mixer", 2_748), ("hierarchical_output_mixer", 2_972)],
)
def test_payload_restores_named_adam_state_freeze_flag_and_capacity(reference, method, adaptive_bytes, tmp_path):
    core = TensorSourceBasisPolicy.from_reference(reference, method=method, train_critic=False)
    optimizer = core.make_optimizer(learning_rate=0.003, transfer=False)
    obs = observations([1, 4], [0.8, 0.1])
    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(obs).log_prob(torch.tensor([0, 3])).mean()).backward()
    optimizer.step()

    report = core.storage_report()
    assert report["source_basis_bytes"] == 780
    assert report["band_coefficient_parameters"] == 162
    assert report["fully_warmed_residual_and_adam_bytes"] == adaptive_bytes
    assert report["coordinate_index_bytes"] == 6
    assert report["fully_warmed_adaptive_tensor_bytes"] == adaptive_bytes + 6
    limit = 15_952
    assert report["residual_value_bytes"] + report["coordinate_index_bytes"] + sum(
        value.numel() * value.element_size()
        for parameter, state in optimizer.state.items()
        if getattr(parameter, "tree_residual", False)
        for value in state.values()
        if torch.is_tensor(value)
    ) <= limit

    restored = TensorSourceBasisPolicy.from_payload(
        export_payload(core, optimizer, {"route": 2}, metadata={"method": method})
    )
    copy = restored["policy"]
    assert copy.basis_trainable is core.basis_trainable
    assert copy.source_basis.requires_grad is core.source_basis.requires_grad
    optimized_names = {
        name
        for group in core.optimizer_manifest(optimizer)["param_groups"]
        for name in group["param_names"]
    }
    assert ("source_basis" in optimized_names) is (method == "tensor_learned")
    assert restored["router"] == {"route": 2}
    assert restored["metadata"] == {"method": method}
    assert set(core.optimizer_manifest(optimizer)["state"]) == set(
        copy.optimizer_manifest(restored["optimizer"])["state"]
    )
    for name, value in core.state_dict().items():
        torch.testing.assert_close(copy.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(
        copy.get_distribution(obs).distribution.probs,
        core.get_distribution(obs).distribution.probs,
        rtol=0,
        atol=0,
    )

    payload_path = Path(tmp_path) / "tensor_policy_test_payload.pt"
    torch.save(export_payload(core, optimizer, {"route": 2}, metadata={"method": method}), payload_path)
    with _installed_factories():
        dispatched = c1_supplement_experiment._restore_payload(payload_path)
    assert isinstance(dispatched["policy"], TensorSourceBasisPolicy)
    assert dispatched["continual_state"] is None


@pytest.mark.parametrize(
    ("method", "residual_adam_bytes", "adaptive_bytes"),
    [
        ("tensor_learned", 4_312, 4_318),
        ("output_mixer", 2_748, 2_754),
        ("hierarchical_output_mixer", 2_972, 2_978),
    ],
)
def test_exact_all_route_warmed_adaptive_bytes(reference, method, residual_adam_bytes, adaptive_bytes):
    torch.manual_seed(914)
    core = TensorSourceBasisPolicy.from_reference(reference, method=method, train_critic=False)
    with torch.no_grad():
        for node in core._nodes.values():
            node.A.normal_(mean=0.0, std=0.05)
        for family in core._families.values():
            family.C.normal_(mean=0.0, std=0.05)
    optimizer = core.make_optimizer(learning_rate=0.003, transfer=False)
    obs = observations(
        [1, 2, 3, 4, 5, 6], [0.1, 0.4, 0.9, 0.1, 0.4, 0.9]
    )
    optimizer.zero_grad(set_to_none=True)
    loss = -core.get_distribution(obs).log_prob(torch.tensor([0, 1, 2, 3, 1, 0])).sum()
    loss.backward()
    optimizer.step()

    trainable_residuals = [parameter for parameter in core._residual_parameters() if parameter.requires_grad]
    assert all(parameter in optimizer.state for parameter in trainable_residuals)
    actual_optimizer_bytes = sum(
        value.numel() * value.element_size()
        for parameter in trainable_residuals
        for value in optimizer.state[parameter].values()
        if torch.is_tensor(value)
    )
    report = core.storage_report()
    assert report["fully_warmed_residual_and_adam_bytes"] == residual_adam_bytes
    assert report["fully_warmed_adaptive_tensor_bytes"] == adaptive_bytes
    assert report["residual_value_bytes"] + actual_optimizer_bytes == residual_adam_bytes
    assert report["residual_value_bytes"] + report["coordinate_index_bytes"] + actual_optimizer_bytes == adaptive_bytes
    assert report["coordinate_index_bytes"] == 6


def test_tensor_learned_basis_is_trainable_while_output_mixer_basis_is_fixed(reference):
    learned = TensorSourceBasisPolicy.from_reference(reference, method="tensor_learned", train_critic=False)
    frozen = TensorSourceBasisPolicy.from_reference(reference, method="output_mixer", train_critic=False)
    hierarchical = TensorSourceBasisPolicy.from_reference(
        reference, method="hierarchical_output_mixer", train_critic=False
    )
    obs = observations([1, 4], [0.8, 0.1])
    for core in (learned, frozen, hierarchical):
        opt = core.make_optimizer(learning_rate=0.01, transfer=False)
        # The zero-start coefficients must move first; only then can the
        # shared basis receive a nonzero gradient.
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            (-core.get_distribution(obs).log_prob(torch.tensor([0, 3])).mean()).backward()
            opt.step()
    assert learned.source_basis.requires_grad
    assert not frozen.source_basis.requires_grad
    assert not hierarchical.source_basis.requires_grad
    assert not torch.equal(learned.source_basis, frozen.source_basis)
    assert frozen.source_basis.grad is None
    assert hierarchical.source_basis.grad is None


@pytest.mark.parametrize("method", ["tensor_learned", "output_mixer", "hierarchical_output_mixer"])
def test_factorized_correction_matches_materialized_outputs_and_gradients(reference, method):
    torch.manual_seed(882)
    factorized = TensorSourceBasisPolicy.from_reference(
        reference, method=method, train_critic=False
    )
    with torch.no_grad():
        for endpoint in range(1, 7):
            factorized._nodes[str(endpoint)].A.normal_(mean=0.0, std=0.04)
        for family in factorized._families.values():
            family.C.normal_(mean=0.0, std=0.04)
    materialized = TensorSourceBasisPolicy.from_reference(
        reference, method=method, train_critic=False
    )
    materialized.load_state_dict(factorized.state_dict(), strict=True)

    obs = observations(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [0.08, 0.43, 0.92, 0.19, 0.56, 0.79, 0.31, 0.69],
    )
    latent_fast = factorized.mlp_extractor.forward_actor(factorized.extract_features(obs))
    latent_reference = materialized.mlp_extractor.forward_actor(materialized.extract_features(obs))
    fast_output = factorized._correction(obs, latent_fast)
    reference_output = materialized_correction(materialized, obs, latent_reference)
    torch.testing.assert_close(fast_output, reference_output, rtol=2e-5, atol=1e-6)

    weights = torch.arange(1, fast_output.numel() + 1, dtype=fast_output.dtype).reshape_as(fast_output)
    (fast_output * weights).sum().backward()
    (reference_output * weights).sum().backward()
    ref_parameters = dict(materialized.named_parameters())
    for name, parameter in factorized.named_parameters():
        expected = ref_parameters[name].grad
        if parameter.grad is None or expected is None:
            assert parameter.grad is None and expected is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=1e-6, msg=name)


def test_hierarchical_mixer_uses_family_prior_and_zero_local_heldout_leaf(reference):
    core = TensorSourceBasisPolicy.from_reference(
        reference, method="hierarchical_output_mixer", train_critic=False
    )
    with torch.no_grad():
        core._families["7"].C[0, 0] = 0.7
        core._families["8"].C[0, 0] = -0.3
        core._nodes["1"].A.fill_(0.4)
        core._nodes["2"].A.fill_(0.8)
        core._nodes["3"].A.fill_(9.0)
    core.initialize_heldout(3)
    assert torch.count_nonzero(core._nodes["3"].A) == 0
    assert core.seen_mask.tolist() == [True, True, True, True, True, False]

    obs = observations([7], [0.1])
    latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
    _, local = core._feature_banks(obs, latent)
    family_reduced = core._families["7"].C @ core.source_basis
    family_matrix = core._contrast_basis @ family_reduced
    expected = torch.nn.functional.linear(local, torch.cat([family_matrix] * 3, dim=1))
    actual = core._fallback_correction(7, local)
    torch.testing.assert_close(actual, expected)
    assert actual.requires_grad
    actual.sum().backward()
    assert core._families["7"].C.grad is not None
    assert core._nodes["1"].A.grad is None
    assert core._nodes["2"].A.grad is None

    active = observations([1], [0.1])
    optimizer = core.make_optimizer(learning_rate=0.01, transfer=False)
    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(active).log_prob(torch.tensor([0])).mean()).backward()
    family_a, family_b = core._families["7"].C, core._families["8"].C
    assert family_a.grad is not None
    assert family_b.grad is None
    optimizer.step()
    state_before = {key: value.detach().clone() for key, value in optimizer.state[family_a].items()}
    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(observations([4], [0.1])).log_prob(torch.tensor([3])).mean()).backward()
    assert family_a.grad is None
    assert family_b.grad is not None
    optimizer.step()
    for key, value in state_before.items():
        torch.testing.assert_close(optimizer.state[family_a][key], value, rtol=0, atol=0)


def test_hierarchical_parent_route_trains_family_coefficient_only(reference):
    core = TensorSourceBasisPolicy.from_reference(
        reference, method="hierarchical_output_mixer", train_critic=False
    )
    obs = observations([7, 8], [0.2, 0.8])
    optimizer = core.make_optimizer(learning_rate=0.01, transfer=False)
    optimizer.zero_grad(set_to_none=True)
    (-core.get_distribution(obs).log_prob(torch.tensor([0, 3])).sum()).backward()
    assert core._families["7"].C.grad is not None
    assert torch.count_nonzero(core._families["7"].C.grad) > 0
    assert core._families["8"].C.grad is not None
    assert torch.count_nonzero(core._families["8"].C.grad) > 0
    assert all(node.A.grad is None for node in core._nodes.values())
    optimizer.step()

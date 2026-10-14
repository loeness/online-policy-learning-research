"""Contract tests for the compact-head full shared C1 control."""
from __future__ import annotations

import io
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.c1_context_head_controls import C1ContextHeadFullPolicy  # noqa: E402
from online_rl import c1_context_head_experiment as context_experiment  # noqa: E402
from online_rl import c1_supplement_experiment as neural_experiment  # noqa: E402
from online_rl.c1_continual_controls import C1OnlineEWC, export_payload  # noqa: E402
from online_rl.c1_policy import C1ResidualPolicy  # noqa: E402
from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402


torch.set_num_threads(1)


@pytest.fixture
def reference() -> TreePolicy:
    torch.manual_seed(3001)
    return TreePolicy(
        observation_space=spaces.Box(
            low=np.asarray([-2.0] * 12 + [0.0], dtype=np.float32),
            high=np.asarray([2.0] * 12 + [8.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )


def make_observations(routes: list[int], x: list[float] | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(913)
    result = torch.rand((len(routes), OBS_DIM), generator=generator)
    if x is not None:
        result[:, 0] = torch.as_tensor(x, dtype=torch.float32)
    result[:, -1] = torch.as_tensor(routes, dtype=torch.float32)
    return result


def test_zero_route_band_biases_and_context_heads_preserve_w0(reference: TreePolicy) -> None:
    policy = C1ContextHeadFullPolicy.from_reference(reference, epsilon=0.05, temperature=2.0)
    obs = make_observations(list(range(9)), [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    with torch.no_grad():
        source_features = reference.extract_features(obs)
        source_logits = reference.action_net(reference.mlp_extractor.forward_actor(source_features))
        policy_logits = policy.actor_logits(obs)
        expected = (1.0 - policy.epsilon) * torch.softmax(source_logits / policy.temperature, dim=-1)
        expected = expected + policy.epsilon / 4.0
        actual = policy.get_distribution(obs).distribution.probs

    torch.testing.assert_close(policy_logits, source_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert policy.context_heads.__len__() == 6
    assert all(tuple(head.shape) == (3, 195) for head in policy.context_heads)
    assert sum(int(head.numel() * head.element_size()) for head in policy.context_heads) == 14_040
    assert all(torch.count_nonzero(head) == 0 for head in policy.context_heads)
    assert torch.count_nonzero(policy.route_embedding) == 0
    assert torch.count_nonzero(policy.band_embedding) == 0
    assert policy.parameter_update_scope == "full_shared_actor_critic_route_band_and_six_context_heads"


def test_context_heads_represent_oracle_residual_policy_logits(reference: TreePolicy) -> None:
    """This is a function-space test only; no environment mapping enters training."""
    torch.manual_seed(1818)
    teacher = C1ResidualPolicy.from_reference(
        reference, method="dense", exploration_epsilon=0.05, temperature=2.0,
    )
    policy = C1ContextHeadFullPolicy.from_reference(reference, epsilon=0.05, temperature=2.0)
    with torch.no_grad():
        for endpoint in range(1, 7):
            teacher._nodes[str(endpoint)].values.normal_(mean=0.0, std=0.015)
            policy.context_heads[endpoint - 1].copy_(teacher._nodes[str(endpoint)].values)
    obs = make_observations([1, 1, 2, 3, 4, 5, 6, 6], [0.0, 0.375, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
    with torch.no_grad():
        teacher_probs = teacher.get_distribution(obs).distribution.probs
        policy_probs = policy.get_distribution(obs).distribution.probs
    torch.testing.assert_close(policy_probs, teacher_probs, rtol=1e-6, atol=1e-7)


def test_heldout_head_initializes_from_seen_siblings_and_parent_averages_all(reference: TreePolicy) -> None:
    policy = C1ContextHeadFullPolicy.from_reference(reference)
    with torch.no_grad():
        policy.context_heads[0].fill_(2.0)
        policy.context_heads[1].fill_(6.0)
        policy.context_heads[3].fill_(-1.0)
        policy.context_heads[4].fill_(1.0)
        policy.route_embedding[0].fill_(2.0)
        policy.route_embedding[1].fill_(6.0)
        policy.route_embedding[3].fill_(-1.0)
        policy.route_embedding[4].fill_(1.0)
    policy.initialize_heldout(3)
    policy.initialize_heldout(6)
    torch.testing.assert_close(policy.context_heads[2], torch.full((3, 195), 4.0))
    torch.testing.assert_close(policy.context_heads[5], torch.zeros((3, 195)))
    torch.testing.assert_close(policy.route_embedding[2], torch.full((64,), 4.0))
    assert policy.known_endpoint_mask.tolist() == [True] * 6

    # Route 7 averages endpoints 1, 2, and 3 after initialization.
    obs = make_observations([7], [0.5])
    with torch.no_grad():
        features = policy.extract_features(obs)
        first = policy.mlp_extractor.policy_net[0]
        hidden = torch.nn.functional.linear(features[:, :4], first.weight, first.bias)
        hidden += policy._route_shift(obs) + policy.band_embedding[(obs[:, 0] * 8).round().long() // 3]
        latent = policy.mlp_extractor.policy_net[1:](hidden)
        base_phi = torch.cat((latent, torch.ones_like(latent[:, :1])), dim=1)
        band = torch.floor(obs[:, 0] * 3).long()
        local = torch.nn.functional.one_hot(band, 3).to(latent.dtype).unsqueeze(-1) * base_phi.unsqueeze(1)
        local = local.reshape(1, 195)
        mean_head = torch.stack([policy.context_heads[i] for i in range(3)]).mean(0)
        contrast = torch.nn.functional.linear(local, mean_head)
        expected_correction = torch.nn.functional.linear(contrast, policy._context_contrast_basis)
        actual_correction = policy._context_head_logits(obs, latent)
    torch.testing.assert_close(actual_correction, expected_correction, rtol=0, atol=0)


def test_inactive_context_head_has_no_gradient_or_adam_state(reference: TreePolicy) -> None:
    policy = C1ContextHeadFullPolicy.from_reference(reference)
    optimizer = policy.make_optimizer(learning_rate=0.002)
    observations = make_observations([2, 2, 2, 2], [0.1, 0.2, 0.3, 0.4])
    values, log_probability, entropy = policy.evaluate_actions(observations, torch.tensor([0, 1, 2, 3]))
    (values.square().mean() - log_probability.mean() - 0.01 * entropy.mean()).backward()

    assert policy.context_heads[1].grad is not None
    assert torch.count_nonzero(policy.context_heads[1].grad) > 0
    for index, head in enumerate(policy.context_heads):
        if index != 1:
            assert head.grad is None
    backbone_before = [parameter.detach().clone() for parameter in policy.mlp_extractor.policy_net.parameters()]
    optimizer.step()
    assert policy.context_heads[1] in optimizer.state
    for index, head in enumerate(policy.context_heads):
        if index != 1:
            assert head not in optimizer.state

    # Shared backbone and context head both participate in online adaptation.
    assert any(torch.count_nonzero(parameter.grad) for parameter in policy.mlp_extractor.policy_net.parameters())
    assert any(not torch.equal(before, after) for before, after in zip(
        backbone_before, policy.mlp_extractor.policy_net.parameters()
    ))
    assert policy.storage_report(optimizer)["shared_actor_copies"] == 1


def test_ewc_accounts_for_every_context_head(reference: TreePolicy) -> None:
    policy = C1ContextHeadFullPolicy.from_reference(reference)
    source = make_observations([0] * 64)[:, :4]
    actions = torch.arange(64) % 4
    ewc = C1OnlineEWC(policy)
    ewc.initialize_source_prior(policy, source, actions, sample_size=64, seed=32)
    assert "context_heads.0" in policy.actor_named_parameters()
    assert "context_heads.5" in ewc.fisher
    assert ewc.fisher["context_heads.0"].shape == (3, 195)
    actor_bytes = sum(parameter.numel() * parameter.element_size()
                      for parameter in policy.actor_named_parameters().values())
    assert ewc.storage_report()["reference_actor_bytes"] == actor_bytes
    assert ewc.storage_report()["fisher_tensor_bytes"] == actor_bytes


def test_named_adam_checkpoint_restores_heads_route_mask_and_distribution(reference: TreePolicy) -> None:
    policy = C1ContextHeadFullPolicy.from_reference(reference)
    policy.initialize_heldout(3)
    optimizer = policy.make_optimizer(learning_rate=0.003)
    obs = make_observations([1, 3, 7, 8], [0.1, 0.8, 0.5, 0.9])
    values, log_probability, _ = policy.evaluate_actions(obs, torch.tensor([0, 2, 1, 3]))
    (values.mean() - log_probability.mean()).backward()
    optimizer.step()
    payload = export_payload(policy, optimizer, {"route": 3}, {"phase": "revisit"})
    data = io.BytesIO()
    torch.save(payload, data)
    data.seek(0)
    restored = C1ContextHeadFullPolicy.from_payload(torch.load(data, map_location="cpu", weights_only=False))
    loaded, loaded_optimizer = restored["policy"], restored["optimizer"]

    assert loaded.known_endpoint_mask.tolist() == policy.known_endpoint_mask.tolist()
    for index in range(6):
        torch.testing.assert_close(loaded.context_heads[index], policy.context_heads[index], rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.get_distribution(obs).distribution.probs,
        policy.get_distribution(obs).distribution.probs,
        rtol=0, atol=0,
    )
    expected = policy.optimizer_manifest(optimizer)
    actual = loaded.optimizer_manifest(loaded_optimizer)
    assert expected["state"].keys() == actual["state"].keys()
    assert {"context_heads.0", "context_heads.2"}.issubset(expected["state"])

    report = loaded.storage_report(loaded_optimizer)
    assert report["context_head_parameter_bytes"] == 14_040
    assert report["adaptive_actor_tensor_bytes"] == (
        report["trainable_actor_parameter_bytes"] + report["actor_optimizer_tensor_bytes"]
    )


def test_shared_runner_hooks_dispatch_context_head_and_restore_checkpoint(reference: TreePolicy, tmp_path, monkeypatch) -> None:
    class _FakeEnv:
        def close(self):
            pass

    monkeypatch.setattr(
        neural_experiment,
        "_foundation",
        lambda _config, _seed: SimpleNamespace(policy=reference, env=_FakeEnv()),
    )
    original_factory = neural_experiment._make_core
    original_loader = neural_experiment._restore_payload
    original_budget_dispatch = neural_experiment._is_method_full
    config = {"exploration_epsilon": 0.05, "temperature": 2.0}
    params = {"learning_rate": 0.002, "ent_coef": 0.06}

    with context_experiment._context_head_hooks():
        assert neural_experiment._is_method_full(context_experiment.METHOD)
        core, optimizer = neural_experiment._make_core(
            config, 923, context_experiment.METHOD, {"leaf_rank": 2, "dictionary_rank": 12}, params,
        )
        payload = export_payload(core, optimizer, {"routes": 8}, {"phase": "test"})
        path = tmp_path / "context_head.pt"
        torch.save(payload, path)
        restored = neural_experiment._restore_payload(path)
        assert isinstance(restored["policy"], C1ContextHeadFullPolicy)
        assert restored["router"] == {"routes": 8}
        assert restored["continual_state"] is None

    assert neural_experiment._make_core is original_factory
    assert neural_experiment._restore_payload is original_loader
    assert neural_experiment._is_method_full is original_budget_dispatch


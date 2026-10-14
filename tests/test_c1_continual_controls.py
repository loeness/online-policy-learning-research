"""Checks for the C1 full fine-tuning and bounded replay controls."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.c1_continual_controls import (  # noqa: E402
    C1BoundedBehaviorReservoir,
    C1ContinualPPO,
    C1FullFineTunePolicy,
    C1OnlineEWC,
    behavior_cloning_kl,
    export_payload,
)
from online_rl.c1_policy import C1ResidualPolicy  # noqa: E402
from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl.compression_experiment import PackedSB3Adapter  # noqa: E402
from online_rl.experiment import BlankEnv  # noqa: E402
from online_rl.envs import NavigationEnv  # noqa: E402


torch.set_num_threads(1)


@pytest.fixture
def reference() -> TreePolicy:
    torch.manual_seed(144)
    return TreePolicy(
        observation_space=spaces.Box(
            low=np.asarray([-2.0] * 12 + [0.0], dtype=np.float32),
            high=np.asarray([2.0] * 12 + [8.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )


def observations(routes: list[int]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(312)
    result = torch.rand((len(routes), OBS_DIM), generator=generator)
    result[:, -1] = torch.tensor(routes, dtype=torch.float32)
    return result


def test_zero_route_embedding_preserves_source_actor_logits_for_every_route(reference: TreePolicy) -> None:
    policy = C1FullFineTunePolicy.from_reference(reference)
    obs = observations([0, 1, 2, 3, 4, 5, 6, 7, 8])
    with torch.no_grad():
        source_features = reference.extract_features(obs)
        source_logits = reference.action_net(reference.mlp_extractor.forward_actor(source_features))
        actual_logits = policy.actor_logits(obs)
        probs = policy.get_distribution(obs).distribution.probs
        expected = (1.0 - policy.epsilon) * torch.softmax(source_logits / policy.temperature, dim=-1)
        expected = expected + policy.epsilon / 4.0

    torch.testing.assert_close(actual_logits, source_logits, rtol=0, atol=0)
    torch.testing.assert_close(probs, expected, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(policy.route_embedding) == 0
    assert policy.parameter_update_scope == "full_actor_critic"


def test_hidden_route_biases_condition_endpoints_and_average_known_siblings(reference: TreePolicy) -> None:
    policy = C1FullFineTunePolicy.from_reference(reference)
    with torch.no_grad():
        policy.route_embedding.copy_(torch.arange(384, dtype=torch.float32).reshape(6, 64))
    policy.set_known_endpoints([1, 3, 4])
    obs = observations([1, 7, 8, 0])

    shift = policy._route_shift(obs)
    torch.testing.assert_close(shift[0], policy.route_embedding[0])
    torch.testing.assert_close(shift[1], (policy.route_embedding[0] + policy.route_embedding[2]) / 2)
    torch.testing.assert_close(shift[2], policy.route_embedding[3])
    torch.testing.assert_close(shift[3], torch.zeros(64))

    plain = C1FullFineTunePolicy.from_reference(reference, route_conditioned=False)
    assert plain.route_embedding is None
    assert plain.parameter_update_scope == "full_actor_critic_plain"
    with pytest.raises(ValueError, match="no route embeddings"):
        plain.set_known_endpoints([1])


def test_initialize_heldout_copies_mean_of_seen_hidden_biases(reference: TreePolicy) -> None:
    policy = C1FullFineTunePolicy.from_reference(reference)
    assert policy.known_endpoint_mask.tolist() == [True, True, False, True, True, False]
    with torch.no_grad():
        policy.route_embedding[0].fill_(2.0)
        policy.route_embedding[1].fill_(6.0)
        policy.route_embedding[3].fill_(-1.0)
        policy.route_embedding[4].fill_(1.0)

    policy.initialize_heldout(3)
    policy.initialize_heldout(6)
    torch.testing.assert_close(policy.route_embedding[2], torch.full((64,), 4.0))
    torch.testing.assert_close(policy.route_embedding[5], torch.zeros(64))
    assert policy.known_endpoint_mask.tolist() == [True, True, True, True, True, True]
    obs = observations([7, 8])
    torch.testing.assert_close(policy._route_shift(obs)[0], policy.route_embedding[:3].mean(0))
    torch.testing.assert_close(policy._route_shift(obs)[1], policy.route_embedding[3:].mean(0))


def test_full_actor_critic_and_route_embedding_receive_online_loss_gradients(reference: TreePolicy) -> None:
    policy = C1FullFineTunePolicy.from_reference(reference)
    optimizer = policy.make_optimizer(learning_rate=0.01)
    obs = observations([1, 2, 1, 2])
    actions = torch.tensor([0, 2, 1, 3])
    optimizer.zero_grad()
    values, log_prob, entropy = policy.evaluate_actions(obs, actions)
    loss = values.square().mean() - log_prob.mean() - 0.01 * entropy.mean()
    loss.backward()

    assert policy.route_embedding.grad is not None
    assert torch.isfinite(policy.route_embedding.grad).all()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.mlp_extractor.policy_net.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.action_net.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.mlp_extractor.value_net.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.value_net.parameters())

    actor_before = [p.detach().clone() for p in policy.actor_named_parameters().values()]
    optimizer.step()
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, policy.actor_named_parameters().values()))
    report = policy.storage_report(optimizer)
    assert report["shared_actor_copies"] == 1
    assert report["adaptive_actor_tensor_bytes"] == (
        report["trainable_actor_parameter_bytes"] + report["actor_optimizer_tensor_bytes"]
    )


def test_full_policy_payload_roundtrip_restores_probabilities_and_named_adam_state(reference: TreePolicy) -> None:
    policy = C1FullFineTunePolicy.from_reference(reference)
    optimizer = policy.make_optimizer(learning_rate=0.003)
    obs = observations([1, 2, 7, 8])
    optimizer.zero_grad()
    values, log_prob, _ = policy.evaluate_actions(obs, torch.tensor([0, 2, 1, 3]))
    (values.mean() - log_prob.mean()).backward()
    optimizer.step()
    payload = export_payload(policy, optimizer, {"route": 2}, {"seed": 4})
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    restored = C1FullFineTunePolicy.from_payload(torch.load(buffer, map_location="cpu", weights_only=False))

    restored_policy = restored["policy"]
    restored_optimizer = restored["optimizer"]
    assert restored["router"] == {"route": 2}
    assert restored["metadata"] == {"seed": 4}
    assert isinstance(restored_optimizer, torch.optim.Adam)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(restored_policy.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(
        restored_policy.get_distribution(obs).distribution.probs,
        policy.get_distribution(obs).distribution.probs,
        rtol=0,
        atol=0,
    )
    before = policy.optimizer_manifest(optimizer)
    after = restored_policy.optimizer_manifest(restored_optimizer)
    assert before["class"] == after["class"] == "Adam"
    assert before["state"].keys() == after["state"].keys()
    for name, state in before["state"].items():
        for key, value in state.items():
            actual = after["state"][name][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value


def test_bounded_reservoir_is_fixed_size_and_teacher_kl_is_finite(reference: TreePolicy) -> None:
    policy = C1FullFineTunePolicy.from_reference(reference)
    reservoir = C1BoundedBehaviorReservoir(capacity=8, seed=5)
    inputs = observations(list(range(1, 9)))
    for row in inputs:
        assert reservoir.observe(policy, row)
    # Once full, each subsequent observation may or may not replace a sampled row.
    for row in observations([1, 2, 3, 4, 5, 6]):
        reservoir.observe(policy, row)

    assert reservoir.size == 8
    assert reservoir.seen == 14
    obs, teacher = reservoir.sample(6)
    assert obs.shape == (6, OBS_DIM)
    assert teacher.shape == (6, 4)
    assert np.allclose(teacher.sum(axis=1), 1.0)
    loss = behavior_cloning_kl(policy, obs, teacher)
    assert torch.isfinite(loss)
    report = reservoir.storage_report()
    assert report["array_storage_bytes"] == 8 * 68
    assert report["bytes_per_item"] == 68
    restored = C1BoundedBehaviorReservoir(capacity=8, seed=99)
    restored.load_state_dict(reservoir.state_dict())
    np.testing.assert_array_equal(restored.observations, reservoir.observations)
    np.testing.assert_array_equal(restored.teacher_probabilities, reservoir.teacher_probabilities)
    np.testing.assert_array_equal(restored.sample(5)[0], reservoir.sample(5)[0])


def test_diagonal_fisher_is_mean_of_per_example_score_squares(reference: TreePolicy) -> None:
    policy = C1FullFineTunePolicy.from_reference(reference)
    obs = observations([1, 2, 1])
    actions = torch.tensor([0, 2, 1])
    actual = C1OnlineEWC.estimate_diagonal_fisher(policy, obs, actions)
    expected = {name: torch.zeros_like(parameter) for name, parameter in policy.actor_named_parameters().items()}
    named = policy.actor_named_parameters()
    for row in range(len(obs)):
        log_probability = policy.get_distribution(obs[row]).log_prob(actions[row].reshape(1)).sum()
        gradients = torch.autograd.grad(log_probability, tuple(named.values()), allow_unused=True)
        for (name, _), gradient in zip(named.items(), gradients):
            if gradient is not None:
                expected[name].add_(gradient.detach().square() / len(obs))
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-8)

    ewc = C1OnlineEWC(policy)
    ewc.initialize_source_prior(policy, obs[:, :4], actions, sample_size=2)
    assert ewc.enabled
    assert ewc.source_prior_observations == 2
    assert ewc.storage_report()["ewc_state_tensor_bytes"] == (
        ewc.storage_report()["fisher_tensor_bytes"] + ewc.storage_report()["reference_actor_bytes"]
    )


def test_ewc_gradient_hook_adds_lambda_fisher_times_distance(reference: TreePolicy) -> None:
    policy = C1FullFineTunePolicy.from_reference(reference)
    obs = observations([1, 2, 1])
    actions = torch.tensor([0, 2, 1])
    ewc = C1OnlineEWC(policy)
    ewc.initialize_source_prior(policy, obs, actions, sample_size=3)
    name = "mlp_extractor.policy_net.0.weight"
    parameter = dict(policy.named_parameters())[name]
    with torch.no_grad():
        parameter.add_(0.05)
    reference_value = ewc.reference[name].clone()
    fisher_value = ewc.fisher[name].clone()

    policy.zero_grad(set_to_none=True)
    policy.get_distribution(obs).distribution.logits[:, 0].sum().backward()
    base_gradient = parameter.grad.detach().clone()
    policy.zero_grad(set_to_none=True)
    handles = ewc.register_gradient_hooks(policy, coefficient=2.0)
    try:
        policy.get_distribution(obs).distribution.logits[:, 0].sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    expected = base_gradient + 2.0 * fisher_value * (parameter.detach() - reference_value)
    torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-7)


def test_reservoir_rejects_malformed_teacher_data() -> None:
    reservoir = C1BoundedBehaviorReservoir(capacity=2)
    with pytest.raises(ValueError, match="sum to one"):
        reservoir.add(np.zeros(OBS_DIM), np.asarray([0.25, 0.25, 0.25, 0.0]))
    with pytest.raises(ValueError, match="13 value"):
        reservoir.add(np.zeros(12), np.full(4, 0.25))


def test_continual_ppo_runs_normal_ppo_then_one_bounded_replay_step() -> None:
    env = DummyVecEnv([lambda: BlankEnv(NavigationEnv())])
    model = C1ContinualPPO(
        TreePolicy,
        env,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        learning_rate=0.001,
        replay_capacity=4,
        bc_coefficient=0.1,
        bc_batch_size=2,
        replay_seed=7,
        ewc_interval=16,
        ewc_batch_size=4,
        device="cpu",
        verbose=0,
    )
    core = C1FullFineTunePolicy.from_reference(model.policy)
    optimizer = core.make_optimizer(learning_rate=0.001)
    model.policy = PackedSB3Adapter(core, optimizer)
    model.observe_online(np.asarray([0.1, 0.2, 0.9, 0.8, *([0.0] * 8), 1.0], dtype=np.float32))
    model.learn(total_timesteps=16, progress_bar=False)

    assert model.behavior_reservoir.seen == 1
    assert model.continual_bc_updates == 1
    assert model.ewc_consolidation_count == 1
    assert model.online_ewc is not None
    assert model.online_ewc.observations_used == 4
    state = model.continual_state_dict()
    model.load_continual_state_dict(state)
    assert model.online_ewc.consolidations == 1
    env.close()


def test_replay_only_ppo_accepts_frozen_base_residual_core() -> None:
    env = DummyVecEnv([lambda: BlankEnv(NavigationEnv())])
    model = C1ContinualPPO(
        TreePolicy,
        env,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        learning_rate=0.001,
        replay_capacity=128,
        bc_coefficient=0.1,
        bc_batch_size=2,
        ewc_coefficient=0.0,
        device="cpu",
        verbose=0,
    )
    core = C1ResidualPolicy.from_reference(model.policy, method="flat")
    optimizer = core.make_optimizer(learning_rate=0.001, transfer=False)
    model.policy = PackedSB3Adapter(core, optimizer)
    model.observe_online(np.asarray([0.1, 0.2, 0.9, 0.8, *([0.0] * 8), 1.0], dtype=np.float32))
    model.learn(total_timesteps=16, progress_bar=False)

    assert model.behavior_reservoir.capacity == 128
    assert model.continual_bc_updates == 1
    assert model.online_ewc is None
    env.close()

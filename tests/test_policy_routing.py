"""Behavioral checks for the tree policy and causal history router."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl.routing import HistoryRouter  # noqa: E402


torch.set_num_threads(1)


@pytest.fixture
def policy() -> TreePolicy:
    torch.manual_seed(17)
    return TreePolicy(
        observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )


def _observation(route: int, *, history: np.ndarray | None = None) -> np.ndarray:
    raw = np.asarray([0.2, 0.4, 0.8, 0.6], dtype=np.float32)
    causal = np.zeros(8, dtype=np.float32) if history is None else history
    return np.concatenate([raw, causal, np.asarray([route], dtype=np.float32)])


def _distribution(policy: TreePolicy, observations: np.ndarray | torch.Tensor):
    tensor = torch.as_tensor(observations, dtype=torch.float32)
    return policy.get_distribution(tensor).distribution


def test_zero_residuals_reproduce_base_actor_distribution_for_each_route(
    policy: TreePolicy,
) -> None:
    varied_history = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    observations = torch.as_tensor(
        np.stack(
            [
                _observation(0),
                _observation(1, history=varied_history),
                _observation(5, history=-varied_history),
            ]
        )
    )

    with torch.no_grad():
        latent = policy.mlp_extractor.forward_actor(policy.extract_features(observations))
        base_probabilities = policy.action_net(latent[:1]).softmax(dim=-1)
        routed_probabilities = _distribution(policy, observations).probs

    torch.testing.assert_close(
        routed_probabilities,
        base_probabilities.expand_as(routed_probabilities),
    )


def test_causal_history_does_not_change_actor_with_a_fixed_nonzero_route(
    policy: TreePolicy,
) -> None:
    first_history = np.zeros(8, dtype=np.float32)
    second_history = np.asarray([0.8, -0.4, 0.2, 0.6, 0, 0, 0, 0], dtype=np.float32)
    first = torch.as_tensor(_observation(2, history=first_history)[None, :])
    second = torch.as_tensor(_observation(2, history=second_history)[None, :])

    with torch.no_grad():
        policy.residuals[2][:, -1] = torch.tensor([1.0, -0.5, 0.25, -0.75])
        same_route = _distribution(policy, torch.cat([first, second])).probs
        base_route = _distribution(
            policy, torch.as_tensor(_observation(0, history=first_history)[None, :])
        ).probs

    torch.testing.assert_close(same_route[0], same_route[1])
    assert not torch.allclose(same_route[0], base_route[0])


def test_critic_uses_causal_history_and_ignores_logged_route_id(policy: TreePolicy) -> None:
    first = torch.as_tensor(_observation(0)[None, :])
    changed_history = np.zeros(8, dtype=np.float32)
    changed_history[0] = 0.7
    second = torch.as_tensor(_observation(0, history=changed_history)[None, :])
    same_critic_input_new_route = torch.as_tensor(
        _observation(6, history=changed_history)[None, :]
    )

    # Isolate one history coordinate so the behavioral dependency is observable.
    with torch.no_grad():
        layers = policy.mlp_extractor.value_net
        hidden_in, hidden_out = layers[0], layers[2]
        hidden_in.weight.zero_()
        hidden_in.bias.zero_()
        hidden_out.weight.zero_()
        hidden_out.bias.zero_()
        hidden_in.weight[0, 4] = 1.0
        hidden_out.weight[0, 0] = 1.0
        policy.value_net.weight.zero_()
        policy.value_net.bias.zero_()
        policy.value_net.weight[0, 0] = 1.0

        values = policy.predict_values(torch.cat([first, second, same_critic_input_new_route]))

    assert values[0].item() != pytest.approx(values[1].item())
    assert values[1].item() == pytest.approx(values[2].item())


def test_logged_routes_select_leaf_gradients_and_idle_adam_leaf_keeps_momentum(
    policy: TreePolicy,
) -> None:
    policy.configure_learning("history", learning_rate=0.03)
    optimizer = policy.optimizer

    def loss_for(routes: list[int], actions: list[int]) -> torch.Tensor:
        observations = torch.as_tensor(np.stack([_observation(route) for route in routes]))
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        return -policy.get_distribution(observations).log_prob(action_tensor).mean()

    # Give leaf 3 Adam momentum, then train using a logged batch for leaves 1 and 2.
    optimizer.zero_grad(set_to_none=True)
    loss_for([3, 3], [0, 0]).backward()
    optimizer.step()
    leaf3 = policy.residuals[3]
    assert leaf3 in optimizer.state
    leaf3_state = optimizer.state[leaf3]
    step_before = leaf3_state["step"].clone()
    exp_avg_before = leaf3_state["exp_avg"].clone()
    exp_avg_sq_before = leaf3_state["exp_avg_sq"].clone()
    leaf3_before = leaf3.detach().clone()

    leaf1_before = policy.residuals[1].detach().clone()
    leaf2_before = policy.residuals[2].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss_for([2, 1], [0, 1]).backward()

    for selected in (1, 2):
        gradient = policy.residuals[selected].grad
        assert gradient is not None
        assert torch.count_nonzero(gradient).item() > 0
    idle_gradient = leaf3.grad
    assert idle_gradient is not None
    assert torch.count_nonzero(idle_gradient).item() == 0

    optimizer.step()

    assert not torch.equal(policy.residuals[1], leaf1_before)
    assert not torch.equal(policy.residuals[2], leaf2_before)
    torch.testing.assert_close(leaf3, leaf3_before, rtol=0, atol=0)
    torch.testing.assert_close(leaf3_state["step"], step_before, rtol=0, atol=0)
    torch.testing.assert_close(leaf3_state["exp_avg"], exp_avg_before, rtol=0, atol=0)
    torch.testing.assert_close(leaf3_state["exp_avg_sq"], exp_avg_sq_before, rtol=0, atol=0)


def test_path_residuals_add_to_the_same_logits_as_their_path_sum(policy: TreePolicy) -> None:
    observation = torch.as_tensor(_observation(3)[None, :])

    with torch.no_grad():
        for node, scale in ((1, 0.001), (2, -0.002), (3, 0.003)):
            values = torch.arange(policy.residuals[node].numel(), dtype=torch.float32)
            policy.residuals[node].copy_(values.reshape_as(policy.residuals[node]) * scale)
        policy.set_path(3, [1, 2])

        features = policy.extract_features(observation)
        latent = policy.mlp_extractor.forward_actor(features)
        phi = torch.cat([latent, torch.ones_like(latent[:, :1])], dim=1)
        expected_residual = policy.residuals[1] + policy.residuals[2] + policy.residuals[3]
        expected_logits = policy.action_net(latent) + torch.einsum(
            "ac,bc->ba", expected_residual, phi
        )
        actual_probabilities = _distribution(policy, observation).probs

    torch.testing.assert_close(actual_probabilities, expected_logits.softmax(dim=-1))


def _source_router() -> HistoryRouter:
    # Source action responses are up, right, down, and left.
    source_responses = np.asarray(
        [[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.float32
    )
    return HistoryRouter(source_responses, window=16, minimum_events=5)


def _observe_direction(router: HistoryRouter, direction: tuple[float, float], count: int) -> None:
    before = np.asarray([0.5, 0.5], dtype=np.float32)
    after = before + np.asarray(direction, dtype=np.float32) * 0.125
    for _ in range(count):
        router.observe(before, action=0, after=after)


def test_router_distinguishes_source_cw_and_ccw_from_observed_action_results() -> None:
    router = _source_router()

    _observe_direction(router, (0, -1), 5)
    assert router.route() == 0
    assert len(router.prototypes) == 1

    _observe_direction(router, (1, 0), 20)
    assert router.route() == 1
    assert len(router.prototypes) == 2
    np.testing.assert_allclose(router.prototypes[1], [0, 1], atol=1e-6)

    _observe_direction(router, (-1, 0), 20)
    assert router.route() == 2
    assert len(router.prototypes) == 3
    np.testing.assert_allclose(router.prototypes[2], [0, -1], atol=1e-6)


def test_router_reuses_existing_prototype_when_context_returns_after_a_shift() -> None:
    router = _source_router()
    _observe_direction(router, (0, -1), 5)
    _observe_direction(router, (1, 0), 20)
    cw_prototype = router.prototypes[1].copy()
    _observe_direction(router, (-1, 0), 20)
    assert router.route() == 2
    assert len(router.prototypes) == 3

    _observe_direction(router, (1, 0), 20)

    assert len(router.prototypes) == 3
    assert router.active == 1
    assert router.route() == 1
    np.testing.assert_allclose(router.prototypes[1], cw_prototype, atol=0.02)


def test_working_memory_reset_keeps_learned_router_prototypes() -> None:
    router = _source_router()
    _observe_direction(router, (0, -1), 5)
    _observe_direction(router, (1, 0), 20)
    _observe_direction(router, (-1, 0), 20)
    prototypes_before = [prototype.copy() for prototype in router.prototypes]
    counts_before = list(router.counts)
    assert router.events

    router.clear_working_memory()

    assert not router.events
    assert router.active == 0
    assert router.counts == counts_before
    assert len(router.prototypes) == len(prototypes_before)
    for actual, expected in zip(router.prototypes, prototypes_before):
        np.testing.assert_array_equal(actual, expected)

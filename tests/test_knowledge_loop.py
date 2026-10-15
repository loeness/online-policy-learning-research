import math

import numpy as np
import pytest

from scripts.run_knowledge_loop import (
    CONSOLIDATE_EVERY,
    GAMMA,
    GAE_LAMBDA,
    ROLLOUT_STEPS,
    Transition,
    _finalize_gae,
    _sample_action,
    actor_valid_mask,
    bootstrap_observation,
    bootstrap_value,
    make_hidden_schedule,
    rollout_number_at_step,
)


class _FixedActionRng:
    def __init__(self, action):
        self.action = action
        self.seen_probabilities = None

    def choice(self, count, p):
        assert count == 4
        self.seen_probabilities = np.asarray(p, dtype=np.float64).copy()
        return self.action


def _row(*, reward, value, next_value, done, terminated):
    return Transition(
        obs=np.zeros(13, dtype=np.float32),
        action=0,
        reward=reward,
        value=value,
        next_value=next_value,
        log_prob=0.0,
        probabilities=np.full(4, 0.25, dtype=np.float32),
        done=done,
        terminated=terminated,
        slot=-1,
        generation=-1,
        step=1,
    )


def test_sampled_action_log_probability_uses_selected_action():
    rng = _FixedActionRng(2)
    action, log_prob = _sample_action(np.array([0.1, 0.2, 0.3, 0.4]), rng)

    assert action == 2
    assert rng.seen_probabilities == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert log_prob == pytest.approx(math.log(0.3))


def test_gae_bootstraps_truncation_but_not_true_termination():
    truncated = _row(reward=2.0, value=0.5, next_value=10.0, done=True, terminated=False)
    _finalize_gae([truncated])
    assert truncated.advantage == pytest.approx(2.0 + GAMMA * 10.0 - 0.5)
    assert truncated.returns == pytest.approx(truncated.advantage + truncated.value)

    terminated = _row(reward=2.0, value=0.5, next_value=10.0, done=True, terminated=True)
    _finalize_gae([terminated])
    assert terminated.advantage == pytest.approx(2.0 - 0.5)


def test_route_aware_bootstrap_uses_next_action_route_id():
    class _Agent:
        shared_head = False
        controller = None

        def __init__(self):
            self.calls = 0

        def route_before_action(self, obs):
            self.calls += 1
            return {"slot": 3, "confidence": 0.8}

    agent = _Agent()
    visible = np.array([0.25, 0.75, 1.0, 0.0], dtype=np.float32)
    policy_obs = bootstrap_observation(agent, visible)

    assert agent.calls == 1
    assert policy_obs.shape == (13,)
    assert policy_obs[:4] == pytest.approx(visible)
    assert policy_obs[-1] == 4.0  # slot 3 maps to one-based endpoint 4

    class _RouteDependentCritic:
        @staticmethod
        def predict_values(tensor):
            return tensor[:, -1:] * 2.0

    agent.core = _RouteDependentCritic()
    assert bootstrap_value(agent, visible) == pytest.approx(8.0)


def test_shared_head_and_unresolved_bootstrap_routes_are_explicit():
    class _Agent:
        controller = None

        def __init__(self, shared, slot):
            self.shared_head = shared
            self.slot = slot

        def route_before_action(self, obs):
            return {"slot": self.slot}

    obs = np.zeros(4, dtype=np.float32)
    assert bootstrap_observation(_Agent(True, 4), obs)[-1] == 1.0
    assert bootstrap_observation(_Agent(False, -1), obs)[-1] == 0.0


def test_rollout_schedule_is_three_fit_rollouts_then_one_holdout():
    expected = [1, 2, 3, 4, 1, 2, 3, 4]
    steps = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
    assert [rollout_number_at_step(step) for step in steps] == expected
    assert CONSOLIDATE_EVERY == 4 * ROLLOUT_STEPS


def test_hidden_schedule_is_deterministic_bounded_and_complete():
    schedule = make_hidden_schedule(32768, 123)
    assert schedule == make_hidden_schedule(32768, 123)
    assert sum(row["length"] for row in schedule) == 32768
    assert all(1024 <= row["length"] <= 2048 for row in schedule)
    assert schedule[0]["start_step"] == 0
    assert schedule[-1]["end_step"] == 32768
    assert all(left["end_step"] == right["start_step"] for left, right in zip(schedule, schedule[1:]))


def test_generation_mask_drops_replaced_context_actor_samples():
    mask = actor_valid_mask(
        np.array([0, 1, -1]),
        np.array([2, 3, -1]),
        [2, 4, 0, 0, 0, 0],
        arm="int8",
    )
    assert mask.tolist() == [True, False, False]
    assert actor_valid_mask(
        np.array([-1, 1]), np.array([-1, 999]), [0, 7], arm="single_head_online"
    ).tolist() == [True, True]

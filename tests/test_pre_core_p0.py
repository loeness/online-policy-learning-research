"""Focused checks for P0's constant route and paired episode evaluator."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_pre_core_p0 import (  # noqa: E402
    CONTEXTS,
    FixedContextRouteOneEnv,
    evaluate_episodes,
    make_eval_plan,
)


class FixedProbabilityPolicy(nn.Module):
    def __init__(self, probabilities: list[float]) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(probabilities, dtype=torch.float32).log())
        self.seen_routes: list[int] = []

    def get_distribution(self, observations: torch.Tensor):
        self.seen_routes.extend(observations[:, -1].round().long().tolist())
        probabilities = torch.softmax(self.logits, dim=0).expand(observations.shape[0], -1)
        return SimpleNamespace(distribution=SimpleNamespace(probs=probabilities))


def test_fixed_context_wrapper_keeps_policy_route_constant() -> None:
    env = FixedContextRouteOneEnv("B1")
    try:
        observation, _ = env.reset(seed=319)
        assert observation.shape == (13,)
        assert observation.dtype == np.float32
        assert observation[-1] == 1.0
        assert env.context == "B1"
        assert env.env.context.family == "B"
        assert env.env.context.variant == 1

        next_observation, _, terminated, truncated, _ = env.step(0)
        assert next_observation.shape == (13,)
        assert next_observation[-1] == 1.0
        assert not (terminated or truncated)
    finally:
        env.close()


def test_episode_evaluator_pairs_starts_and_uniform_actions_without_updates() -> None:
    plan = make_eval_plan(271, "A0", 3)
    baseline = FixedProbabilityPolicy([0.25, 0.25, 0.25, 0.25])
    candidate = FixedProbabilityPolicy([0.70, 0.10, 0.10, 0.10])
    baseline_state = {name: value.detach().clone() for name, value in baseline.state_dict().items()}
    candidate_state = {name: value.detach().clone() for name, value in candidate.state_dict().items()}

    baseline_rows = evaluate_episodes(
        baseline,
        target="A0",
        plan=plan,
        source="W0_zero_correction",
    )
    candidate_rows = evaluate_episodes(
        candidate,
        target="A0",
        plan=plan,
        source="A0",
    )

    assert len(baseline_rows) == len(candidate_rows) == 3
    assert [row["reset_seed"] for row in baseline_rows] == plan["reset_seeds"]
    assert [row["reset_seed"] for row in candidate_rows] == plan["reset_seeds"]
    assert [row["initial_observation"] for row in baseline_rows] == [
        row["initial_observation"] for row in candidate_rows
    ]
    assert all(row["completed"] and 1 <= row["length"] <= 64 for row in baseline_rows + candidate_rows)
    assert baseline.seen_routes and set(baseline.seen_routes) == {1}
    assert candidate.seen_routes and set(candidate.seen_routes) == {1}
    assert all(torch.equal(baseline.state_dict()[key], value) for key, value in baseline_state.items())
    assert all(torch.equal(candidate.state_dict()[key], value) for key, value in candidate_state.items())


def test_eval_plan_is_reproducible_and_has_one_uniform_per_possible_step() -> None:
    left = make_eval_plan(283, "A1", 2)
    right = make_eval_plan(283, "A1", 2)
    assert left["reset_seeds"] == right["reset_seeds"]
    assert left["action_uniform_seed"] == right["action_uniform_seed"]
    np.testing.assert_array_equal(left["action_uniforms"], right["action_uniforms"])
    assert left["action_uniforms"].shape == (2, 64)
    assert np.all((left["action_uniforms"] >= 0.0) & (left["action_uniforms"] < 1.0))
    other_target = make_eval_plan(283, "B1", 2)
    assert set(left["reset_seeds"]).isdisjoint(other_target["reset_seeds"])
    assert left["action_uniform_seed"] != other_target["action_uniform_seed"]
    assert left["reset_seeds"] != other_target["reset_seeds"]


def test_frozen_deployment_seed_target_episode_seeds_are_all_distinct() -> None:
    reset_seeds = [
        reset_seed
        for seed in (271, 283, 307)
        for target in CONTEXTS
        for reset_seed in make_eval_plan(seed, target, 32)["reset_seeds"]
    ]
    assert len(reset_seeds) == 3 * 6 * 32
    assert len(set(reset_seeds)) == len(reset_seeds)


@pytest.mark.parametrize("episodes", [0, -1])
def test_eval_plan_rejects_nonpositive_episode_counts(episodes: int) -> None:
    with pytest.raises(ValueError, match="episodes must be positive"):
        make_eval_plan(271, "A0", episodes)

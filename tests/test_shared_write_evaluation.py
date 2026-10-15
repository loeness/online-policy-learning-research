"""Read-only and paired-plan contracts for the C1 evaluation helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from online_rl.c1_policy import C1ResidualPolicy  # noqa: E402
from online_rl.policy import TreePolicy  # noqa: E402
from scripts.shared_write_evaluation import evaluate_context  # noqa: E402


@pytest.fixture
def core() -> C1ResidualPolicy:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8031)
        reference = TreePolicy(
            observation_space=spaces.Box(
                low=np.asarray([-2.0] * 12 + [0.0], dtype=np.float32),
                high=np.asarray([2.0] * 12 + [8.0], dtype=np.float32),
                dtype=np.float32,
            ),
            action_space=spaces.Discrete(4),
            lr_schedule=lambda _: 1e-2,
        )
        return C1ResidualPolicy.from_reference(
            reference,
            method="dense",
            exploration_epsilon=0.05,
            temperature=1.3,
        )


def _numpy_global_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_paired_evaluation_replays_full_episode_plans_without_mutation(core) -> None:
    core.train()
    # Preserve and verify a mixed child mode as well as the top-level mode.
    core.mlp_extractor.policy_net.eval()
    module_modes = [(module, module.training) for module in core.modules()]
    state_before = {name: value.detach().clone() for name, value in core.state_dict().items()}
    grad_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in core.named_parameters()
    }
    py_rng_before = random.getstate()
    np_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()

    first = evaluate_context(core, "A0", 1, seed=47031, episodes=3)
    second = evaluate_context(core, "A0", 1, seed=47031, episodes=3)

    assert first["episodes"] == second["episodes"]
    assert first["total_steps"] == sum(row["steps"] for row in first["episodes"])
    assert first["mean_return"] == pytest.approx(
        np.mean([row["return"] for row in first["episodes"]])
    )
    assert first["mean_success"] == pytest.approx(
        np.mean([float(row["success"]) for row in first["episodes"]])
    )
    for row in first["episodes"]:
        assert len(row["actions"]) == row["steps"]
        assert row["steps"] <= 64
        # The environment records the reset state and every post-action state.
        assert sum(row["observer_region_visits"].values()) == row["steps"] + 1

    for name, value in core.state_dict().items():
        assert torch.equal(value, state_before[name]), name
    for name, parameter in core.named_parameters():
        old_grad = grad_before[name]
        assert (parameter.grad is None) == (old_grad is None), name
        if old_grad is not None:
            assert torch.equal(parameter.grad, old_grad), name
    assert [(module, module.training) for module, _ in module_modes] == module_modes
    assert random.getstate() == py_rng_before
    assert _numpy_global_state_equal(np.random.get_state(), np_rng_before)
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)


def test_episodes_receive_distinct_reset_and_action_streams(core) -> None:
    result = evaluate_context(core, "B2", 6, seed=519, episodes=2)
    rows = result["episodes"]

    assert len({row["reset_seed"] for row in rows}) == 2
    assert len({row["action_seed"] for row in rows}) == 2
    assert all(row["reset_seed"] != row["action_seed"] for row in rows)
    assert len({row["reset_seed"] for row in rows} | {row["action_seed"] for row in rows}) == 4
    assert all(row["context"] == "B2" and row["route_id"] == 6 for row in rows)


def test_custom_context_spec_can_be_evaluated(core) -> None:
    from online_rl.c1_envs import C1Context

    result = evaluate_context(
        core,
        "A0",
        1,
        seed=9,
        episodes=1,
        context=C1Context("B", 2),
    )

    assert result["context"] == "A0"
    assert result["context_spec"] == {"family": "B", "variant": 2}
    assert result["episodes"][0]["steps"] <= 64


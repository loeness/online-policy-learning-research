"""Focused round-trip checks for reusable A/B+ deployment bundles."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from online_rl.controls import (  # noqa: E402
    _new_policy_model,
    clear_router_working_memory,
    history_router_from_snapshot,
    load_deployment_bundle,
    save_deployment_bundle,
)
from online_rl.routing import HistoryRouter  # noqa: E402


torch.set_num_threads(1)


def _router_with_learned_prototype() -> HistoryRouter:
    responses = np.asarray([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    router = HistoryRouter(responses, window=16, minimum_events=5)
    before = np.asarray([0.5, 0.5], dtype=np.float32)
    after = before + np.asarray([0.125, 0.0], dtype=np.float32)
    for _ in range(12):
        router.observe(before, 0, after)
    assert len(router.prototypes) > 1
    router.pending_reset = router.active
    assert router.pending_reset is not None
    return router


def _trainable_states_by_name(model):
    names = {id(parameter): name for name, parameter in model.policy.named_parameters()}
    return {
        names[id(parameter)]: state
        for parameter, state in model.policy.optimizer.state.items()
    }


def test_bundle_round_trip_keeps_nonzero_residual_adam_router_logits_and_rng(tmp_path: Path) -> None:
    config = {"seed": 9, "learning_rate": 0.01, "ent_coef": 0.0}
    model = _new_policy_model(config)
    model.policy.configure_learning("history_keep", config["learning_rate"])
    observation = torch.as_tensor(
        [[0.2, 0.4, 0.8, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32
    )
    action = torch.as_tensor([2], dtype=torch.long)
    model.policy.optimizer.zero_grad(set_to_none=True)
    loss = -model.policy.get_distribution(observation).log_prob(action).mean()
    loss.backward()
    model.policy.optimizer.step()
    model.policy.optimizer.step_count = 23
    assert torch.count_nonzero(model.policy.residuals[1]).item() > 0
    assert model.policy.residuals[1] in model.policy.optimizer.state

    router = _router_with_learned_prototype()
    with torch.no_grad():
        expected_logits = model.policy.get_distribution(observation).distribution.logits.detach().clone()
    expected_router = router.snapshot()

    random.seed(4107)
    np.random.seed(5307)
    torch.manual_seed(6707)
    bundle = tmp_path / "deployment.pt"
    metadata = save_deployment_bundle(bundle, model, router, {"test_marker": "roundtrip"})
    expected_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert metadata["historic_complete_rng_state_available"] is False
    assert bundle.is_file()

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    restored, restored_router, restored_metadata = load_deployment_bundle(bundle, config)
    actual_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual_rng == pytest.approx(expected_rng, abs=0.0)
    assert restored_metadata["optimizer_step_count"] == 23
    assert restored_metadata["test_marker"] == "roundtrip"
    assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state)
    with torch.no_grad():
        actual_logits = restored.policy.get_distribution(observation).distribution.logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    assert json.dumps(restored_router.snapshot(), sort_keys=True) == json.dumps(expected_router, sort_keys=True)

    original_optimizer = _trainable_states_by_name(model)
    restored_optimizer = _trainable_states_by_name(restored)
    assert original_optimizer.keys() == restored_optimizer.keys()
    for name in original_optimizer:
        assert original_optimizer[name].keys() == restored_optimizer[name].keys()
        for state_name, expected in original_optimizer[name].items():
            actual = restored_optimizer[name][state_name]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert actual == expected
    model.env.close()
    restored.env.close()


def test_cold_router_clear_removes_pending_state_but_keeps_prototypes() -> None:
    router = _router_with_learned_prototype()
    expected_prototypes = [prototype.copy() for prototype in router.prototypes]
    expected_counts = list(router.counts)
    expected_recurs = list(router.recurs)

    clear_router_working_memory(router)

    assert not router.events
    assert router.active == 0
    assert router.confidence == 0.0
    assert router.novel_streak == 0
    assert router.last_entered == 0
    assert router.pending_reset is None
    assert router.counts == expected_counts
    assert router.recurs == expected_recurs
    assert len(router.prototypes) == len(expected_prototypes)
    for actual, expected in zip(router.prototypes, expected_prototypes):
        np.testing.assert_array_equal(actual, expected)


def test_router_snapshot_rebuild_preserves_all_state_fields() -> None:
    router = _router_with_learned_prototype()
    restored = history_router_from_snapshot(router.snapshot())
    assert json.dumps(restored.snapshot(), sort_keys=True) == json.dumps(router.snapshot(), sort_keys=True)


def test_analytic_stage_b_budget_is_independent_of_optimizer_continuation(tmp_path):
    from online_rl.controls import _run_analytic_variant
    source = np.load("data/pretrain_source.npz")["source_responses"]
    config = {"stage_b_steps": 64, "continuation_steps": 8192, "bout_steps": 16,
              "early_window_steps": 16, "agent_budget_bytes": 262144}
    result = _run_analytic_variant(tmp_path, config, "B", 41, source,
                                  Path("data/foundation.pt"), "analytic")
    assert result["steps"] == 64
    assert len(result["context_schedule"]) == 8
    assert result["policy_updates"] == 0

"""Small synthetic integration checks for the retention stream runner."""
from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.write_decision_policy import encode_actor  # noqa: E402
from scripts import run_retention_stream as runner  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return json.loads(runner.CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def make_agent(config):
    def create(method="retention_gate", budget="low", seed=731):
        return runner.make_agent(config, method, budget, seed)

    return create


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def _synthetic_row(step: int, *, episode_id: int = 4) -> SimpleNamespace:
    obs = np.zeros(13, dtype=np.float32)
    obs[0] = 0.25
    obs[-1] = 1.0
    return SimpleNamespace(
        obs=obs,
        action=0,
        log_prob=float(np.log(0.25)),
        probabilities=np.full(4, 0.25, dtype=np.float32),
        advantage=0.5,
        returns=1.0,
        episode_id=episode_id,
        step=step,
        done=False,
    )


def test_schedule_visits_six_contexts_then_replays_the_same_paired_set(config):
    first = runner.task_schedule(19001)
    repeated = runner.task_schedule(19001)

    assert first == repeated
    assert len(first) == 12
    initial, revisit = first[:6], first[6:]
    assert len(set(initial)) == 6
    assert set(revisit) == set(initial)
    assert len(set(revisit)) == 6
    # The first four contexts form two within-family pairs; the last two add
    # each family's third variant before recurrence begins.
    first_family = initial[0][0]
    second_family = initial[2][0]
    assert first_family != second_family
    assert [task[0] for task in initial[:2]] == [first_family, first_family]
    assert [task[0] for task in initial[2:4]] == [second_family, second_family]
    assert {task[0] for task in initial[4:]} == {first_family, second_family}


@pytest.mark.parametrize(
    ("method", "budget", "slots", "groups", "configured_cap"),
    [
        ("retention_gate", "low", 3, 2, 140),
        ("independent", "low", 4, 0, 140),
        ("retention_gate", "wide", 6, 2, 236),
        ("independent", "wide", 7, 0, 236),
    ],
)
def test_configured_layouts_fit_their_int8_actor_caps(
    config, method, budget, slots, groups, configured_cap
):
    layout = runner.agent_config(config, method, budget)
    local = np.zeros((slots, 27), dtype=np.float32)
    shared = np.zeros((groups, 9), dtype=np.float32)
    assignments = np.full(slots, -1, dtype=np.int8)

    encoded = encode_actor(
        local,
        shared,
        assignments,
        codec="int8",
        budget_bytes=layout["actor_budget_bytes"],
    )

    expected_bytes = 12 + 32 * slots + 13 * groups
    assert layout["local_slots"] == slots
    assert layout["actor_budget_bytes"] == configured_cap
    assert encoded.nbytes == expected_bytes
    assert encoded.nbytes <= configured_cap
    with pytest.raises(ValueError, match="exceeding budget"):
        encode_actor(
            local,
            shared,
            assignments,
            codec="int8",
            budget_bytes=expected_bytes - 1,
        )


def test_coefficient_probabilities_match_w0_and_random_route_group_distributions(make_agent):
    agent = make_agent(seed=739)
    core = agent.core
    rng = np.random.default_rng(1107)
    observations = rng.normal(size=(24, 13)).astype(np.float32)
    observations[:, 0] = rng.uniform(0.0, 1.0, size=len(observations))
    observations[:, -1] = np.tile([1.0, 2.0, 3.0], 8)

    # The factory actor starts at W0, with no local/shared residual.
    w0_actual = core.get_distribution(torch.as_tensor(observations)).distribution.probs.detach().cpu().numpy()
    w0_coefficients = runner.coefficient_probs(core, observations, np.zeros((3, 3, 3)))
    np.testing.assert_allclose(w0_coefficients, w0_actual, rtol=1e-6, atol=1e-7)

    local = rng.normal(0.0, 0.08, size=(agent.slots, 27)).astype(np.float32)
    shared = rng.normal(0.0, 0.06, size=(core.shared_groups, 9)).astype(np.float32)
    groups = np.asarray([0, 1, -1], dtype=np.int64)
    core.install_local_matrix(local)
    core.install_shared_matrix(shared)
    with torch.no_grad():
        core.slot_groups.copy_(torch.as_tensor(groups))
    actual = core.get_distribution(torch.as_tensor(observations)).distribution.probs.detach().cpu().numpy()

    for slot in range(agent.slots):
        selected = observations[:, -1] == slot + 1
        predicted = runner.coefficient_probs(core, observations[selected], runner.effective(core, slot))
        np.testing.assert_allclose(predicted, actual[selected], rtol=1e-6, atol=1e-7)

    assert core.temperature > 0
    assert 0.0 <= core.epsilon < 1.0


def test_checkpoint_roundtrip_restores_bounded_state_and_counts_actor_sidecar(make_agent, tmp_path):
    old_torch_rng = torch.get_rng_state().clone()
    try:
        agent = make_agent(seed=743)
        config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
        agent.counts.update({"raw_steps": 37, "environment_steps": 37, "fit_steps": 12})
        agent.slot_generation[:] = [2, 1, 0]
        agent.fitted_steps[:] = [256, 512, 0]
        local_rng = np.random.default_rng(443)
        agent.core.install_local_matrix(local_rng.normal(0, 0.08, (agent.slots, 27)).astype(np.float32))
        agent.core.install_shared_matrix(local_rng.normal(0, 0.05, (agent.core.shared_groups, 9)).astype(np.float32))
        with torch.no_grad():
            agent.core.slot_groups.copy_(torch.as_tensor([0, 1, 0], dtype=torch.long))

        allocation = agent.cache.resolve("A0+")
        assert allocation["slot"] == 0
        agent.windows[0].observe(
            [0.125, 0.25, 0.375, 0.5], 1, 0.5,
            [0.25, 0.25, 0.375, 0.5], False,
        )
        row = _synthetic_row(37)
        profile_result = agent.archive.observe(agent.windows[0].profile(), [row], step=37)
        prototype = int(profile_result["prototype"])
        agent.anchors.add(0, 2, [{
            "obs": row.obs,
            "action": row.action,
            "behavior_log_prob": row.log_prob,
            "behavior_probs": row.probabilities,
            "advantage": row.advantage,
            "return": row.returns,
            "episode_id": row.episode_id,
        }])
        agent.profiles["seen"][0] = 1
        agent.profiles["reward_n"][0] = 1.0
        agent.slot_prototype[0] = prototype
        agent.slot_prototype_version[0] = 1
        agent.prototype_group[prototype] = 0
        agent.prototype_version[prototype] = 1
        agent.group_versions[:] = [3, 2]
        agent.prior_injected[0] = 1
        agent.pending = {
            "group": 0,
            "parent_version": 3,
            "common": np.full((3, 3), 0.125, dtype=np.float32),
            "created_step": 37,
            "sources": [{"slot": 0, "generation": 2, "prototype": prototype}],
        }
        agent.action_rng.random()
        agent.behavior_rng.random()
        codec = runner.roundtrip_actor(agent)
        assert codec["actor_bytes"] == 134
        assert codec["codec_max_coefficient_error"] > 0

        expected_torch_rng = torch.get_rng_state().clone()
        expected_policy_hash = runner.policy_hash(agent.core)
        expected_fields = {
            "counts": copy.deepcopy(agent.counts),
            "slot_generation": agent.slot_generation.copy(),
            "fitted_steps": agent.fitted_steps.copy(),
            "cache": agent.cache.state_dict(),
            "grouper": agent.grouper.state_dict(),
            "archive": agent.archive.state_dict(),
            "anchors": agent.anchors.state_dict(),
            "profiles": copy.deepcopy(agent.profiles),
            "windows": [window.state_dict() for window in agent.windows],
            "slot_prototype": agent.slot_prototype.copy(),
            "slot_prototype_version": agent.slot_prototype_version.copy(),
            "prototype_group": agent.prototype_group.copy(),
            "prototype_version": agent.prototype_version.copy(),
            "group_versions": agent.group_versions.copy(),
            "prior_injected": agent.prior_injected.copy(),
            "pending": copy.deepcopy(agent.pending),
            "action_rng": copy.deepcopy(agent.action_rng.bit_generator.state),
            "behavior_rng": copy.deepcopy(agent.behavior_rng.bit_generator.state),
            "optimizer": agent.optimizer.state_dict(),
        }

        checkpoint = tmp_path / "retention.pt"
        sizes = runner.save_checkpoint(agent, config, "retention_gate", "low", checkpoint)

        sidecar = checkpoint.with_suffix(".actor.bin")
        assert sidecar.is_file()
        assert sidecar.stat().st_size == sizes["actor_sidecar_bytes"] == sizes["actor_bytes"]
        assert sizes["full_physical_bytes"] == (
            sizes["checkpoint_bytes"] + sizes["actor_sidecar_bytes"] + sizes["foundation_bytes"]
        )
        assert sizes["full_physical_bytes"] <= config["package_budget_bytes"]
        restored = runner.restore_checkpoint(checkpoint)

        assert runner.policy_hash(restored.core) == expected_policy_hash
        assert restored.optimizer.step_count == getattr(agent.optimizer, "step_count", 0)
        for name, expected in expected_fields.items():
            if name == "counts":
                actual = restored.counts
            elif name == "slot_generation":
                actual = restored.slot_generation
            elif name == "fitted_steps":
                actual = restored.fitted_steps
            elif name == "cache":
                actual = restored.cache.state_dict()
            elif name == "grouper":
                actual = restored.grouper.state_dict()
            elif name == "archive":
                actual = restored.archive.state_dict()
            elif name == "anchors":
                actual = restored.anchors.state_dict()
            elif name == "profiles":
                actual = restored.profiles
            elif name == "windows":
                actual = [window.state_dict() for window in restored.windows]
            elif name == "pending":
                actual = restored.pending
            elif name == "action_rng":
                actual = restored.action_rng.bit_generator.state
            elif name == "behavior_rng":
                actual = restored.behavior_rng.bit_generator.state
            elif name == "optimizer":
                actual = restored.optimizer.state_dict()
            else:
                actual = getattr(restored, name)
            _assert_nested_equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_torch_rng)
    finally:
        torch.set_rng_state(old_torch_rng)


@pytest.mark.parametrize("raw_step_offset,row_step_offset", [(0, 0), (1, 0)])
def test_evaluate_pending_rejects_rows_from_the_proposal_batch(
    make_agent, raw_step_offset, row_step_offset
):
    agent = make_agent(seed=751 + raw_step_offset)
    created = 90
    agent.pending = {
        "group": 0,
        "parent_version": 0,
        "common": np.zeros((3, 3), dtype=np.float32),
        "created_step": created,
        "sources": [],
    }
    agent.counts["raw_steps"] = created + raw_step_offset

    with pytest.raises(RuntimeError, match="proposal batch"):
        runner.evaluate_pending(agent, json.loads(runner.CONFIG.read_text()), [_synthetic_row(created + row_step_offset)], 0)


def test_retention_gate_rejects_unknown_historical_profile_even_with_fresh_support(
    make_agent, monkeypatch
):
    agent = make_agent(seed=757)
    empty_profile = agent.windows[0].profile()
    for step in (1, 2):
        agent.archive.observe(empty_profile, [], step=step)
    prototypes = agent.archive.active_indices()
    assert len(prototypes) == 2
    agent.prototype_group[list(prototypes)] = 0
    agent.pending = {
        "group": 0,
        "parent_version": 0,
        "common": np.full((3, 3), 0.1, dtype=np.float32),
        "created_step": 9,
        "sources": [],
    }
    agent.counts["raw_steps"] = 10
    calls = {"history": 0}

    def synthetic_score(_agent, _config, evidence, _candidate, _baseline):
        if "source_slot" in evidence:
            calls["history"] += 1
            if calls["history"] == 1:
                return {"status": "supported", "supported": True, "lower": 0.2}
            return {"status": "unknown", "supported": False, "lower": None}
        return {"status": "supported", "supported": True, "lower": 0.2}

    monkeypatch.setattr(runner, "score_common", synthetic_score)
    result = runner.evaluate_pending(
        agent,
        json.loads(runner.CONFIG.read_text()),
        [_synthetic_row(10)],
        0,
    )

    assert result["accepted"] is False
    assert result["fresh_score"]["supported"] is True
    assert [score["supported"] for score in result["historical_scores"]] == [True, False]
    assert result["reason"] == "insufficient_supported_profiles"
    assert agent.pending is None


def test_compensated_install_preserves_every_resident_when_common_changes(make_agent):
    agent = make_agent(seed=761)
    core = agent.core
    rng = np.random.default_rng(1189)
    local = rng.normal(0.0, 0.1, size=(agent.slots, 27)).astype(np.float32)
    shared = rng.normal(0.0, 0.1, size=(core.shared_groups, 9)).astype(np.float32)
    groups = np.zeros(agent.slots, dtype=np.int64)
    core.install_local_matrix(local)
    core.install_shared_matrix(shared)
    with torch.no_grad():
        core.slot_groups.copy_(torch.as_tensor(groups))
    before = [runner.effective(core, slot).copy() for slot in range(agent.slots)]
    common = rng.normal(0.0, 0.1, size=(3, 3))

    new_local, new_shared, new_groups, diagnostics = runner.compensated_install(
        local, shared, groups, group=0, common=common
    )

    assert diagnostics["affected_slots"] == (0, 1, 2)
    assert diagnostics["max_effective_preservation_error"] < 1e-6
    assert np.all(new_groups == 0)
    assert not np.array_equal(new_shared[0].reshape(3, 3), shared[0].reshape(3, 3))
    np.testing.assert_array_equal(new_shared[1], shared[1])
    for slot, expected in enumerate(before):
        after = new_local[slot].reshape(3, 3, 3) + new_shared[0].reshape(1, 3, 3)
        np.testing.assert_allclose(after, expected, rtol=1e-6, atol=1e-6)


def test_abstaining_assignment_keeps_resident_actor_route_but_withholds_fresh_group(
    make_agent, monkeypatch
):
    agent = make_agent(seed=769)
    slot = 0
    agent.core.set_slot_group(slot, 0, preserve_effective=False)
    shared = np.zeros((agent.core.shared_groups, 9), dtype=np.float32)
    shared[0] = 0.25
    agent.core.install_shared_matrix(shared)
    agent.counts["raw_steps"] = 10
    agent.pending = {
        "group": 0,
        "parent_version": 0,
        "common": np.full((3, 3), 0.1, dtype=np.float32),
        "created_step": 1,
        "sources": [],
    }
    monkeypatch.setattr(agent.grouper, "assign", lambda *_args, **_kwargs: SimpleNamespace(
        group=-1, reason="abstain", scores={}
    ))
    monkeypatch.setattr(
        agent.archive,
        "observe",
        lambda _profile, _rows, *, step: {"prototype": 0, "version": 1},
    )
    monkeypatch.setattr(runner.loop, "_finalize_gae", lambda _rows: None)
    monkeypatch.setattr(
        runner.loop,
        "_ppo_update",
        lambda _agent, rows: {"updates": 0, "synthetic_rows": len(rows)},
    )
    score_calls = []

    def unexpected_same_group_score(*_args, **_kwargs):
        score_calls.append(True)
        pytest.fail("abstaining fresh evidence was scored as same-group evidence")

    monkeypatch.setattr(runner, "score_common", unexpected_same_group_score)
    evaluate_groups = []
    evaluate_pending = runner.evaluate_pending

    def capture_evaluate_group(current_agent, current_config, rows, group):
        evaluate_groups.append(group)
        return evaluate_pending(current_agent, current_config, rows, group)

    monkeypatch.setattr(runner, "evaluate_pending", capture_evaluate_group)

    observation = np.zeros(13, dtype=np.float32)
    observation[0] = 0.25
    observation[-1] = slot + 1
    probabilities = runner.wd._policy_probs(agent.core, observation[None])[0]
    row = SimpleNamespace(
        obs=observation,
        action=0,
        log_prob=float(np.log(probabilities[0])),
        probabilities=probabilities.astype(np.float32),
        value=0.75,
        advantage=0.5,
        returns=1.0,
        episode_id=4,
        step=10,
        done=False,
    )
    events, updates = io.StringIO(), io.StringIO()

    runner.process_rollout(agent, json.loads(runner.CONFIG.read_text()), [row], slot, events, updates)

    assert evaluate_groups == [-1]
    assert agent.core.group_for_slot(slot) == 0
    assert agent.prototype_group[0] == -1
    assert agent.slot_prototype[slot] == 0
    assert agent.pending is not None
    assert score_calls == []
    assert events.getvalue() == ""
    update_record = json.loads(updates.getvalue())
    assert update_record["group"] == 0
    assert update_record["fresh_assignment_group"] == -1
    assert update_record["group_reason"] == "abstain"

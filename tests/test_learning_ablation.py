"""No-environment checks for the no-gradient learning-interaction runner."""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_learning_ablation as ablation  # noqa: E402
from scripts import run_source_dose as source_dose  # noqa: E402
from scripts import run_storage_decision as base  # noqa: E402


def _rng_snapshot():
    return {
        "torch": torch.get_rng_state().clone(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "python": copy.deepcopy(random.getstate()),
    }


def _assert_rng_equal(left, right):
    assert torch.equal(left["torch"], right["torch"])
    assert left["numpy"][0] == right["numpy"][0]
    np.testing.assert_array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert left["python"] == right["python"]


def _populated_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.25, -0.5], dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return parameter, optimizer


def test_no_gradient_update_does_not_change_parameters_optimizer_or_rng():
    parameter, optimizer = _populated_optimizer()
    agent = SimpleNamespace(parameter=parameter, optimizer=optimizer)
    before_parameter = parameter.detach().clone()
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_rng = _rng_snapshot()

    update = ablation._no_gradient_update(agent, [object(), object()])

    assert update == {
        "updates": 0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
    }
    torch.testing.assert_close(parameter.detach(), before_parameter, rtol=0, atol=0)
    assert base.semantic_hash(optimizer.state_dict()) == base.semantic_hash(before_optimizer)
    _assert_rng_equal(before_rng, _rng_snapshot())


def test_postrollout_callback_corrects_training_labels_and_only_metadata():
    parameter, optimizer = _populated_optimizer()
    agent = SimpleNamespace(
        parameter=parameter,
        optimizer=optimizer,
        runtime_counts={"rollouts_trained": 7, "rollouts_collected": 2, "minibatch_updates": 19},
        counts={"fit_steps": 1792},
    )
    before_parameter = parameter.detach().clone()
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_rng = _rng_snapshot()

    ablation._correct_no_gradient_metadata(
        agent=agent,
        phase="future",
        bout_record={"bout": 1},
        slot=0,
        generation=1,
        bout_step=256,
        rollout_index=0,
        rollout_rows=256,
        update={"updates": 0},
    )

    assert agent.runtime_counts == {
        "rollouts_trained": 6,
        "rollouts_collected": 3,
        "minibatch_updates": 19,
    }
    assert agent.counts["fit_steps"] == 1536
    torch.testing.assert_close(parameter.detach(), before_parameter, rtol=0, atol=0)
    assert base.semantic_hash(optimizer.state_dict()) == base.semantic_hash(before_optimizer)
    _assert_rng_equal(before_rng, _rng_snapshot())
    with pytest.raises(RuntimeError, match="optimizer update"):
        ablation._correct_no_gradient_metadata(
            agent=agent,
            phase="future",
            bout_record={},
            slot=0,
            generation=1,
            bout_step=256,
            rollout_index=1,
            rollout_rows=256,
            update={"updates": 1},
        )


def test_no_gradient_scope_limits_actions_forces_fresh_branches_and_restores_patches():
    original_actions = base.ACTIONS
    original_matcher = source_dose._matching_reusable_branch
    original_update = base.loop._ppo_update

    with ablation._no_gradient_execution():
        assert tuple(base.ACTIONS) == ("replace", "none")
        assert source_dose._matching_reusable_branch(
            [{"action": "replace", "future_regime": "old_major", "action_state_hash": "same"}],
            action="replace",
            future_regime="old_major",
            action_state_hash="same",
        ) is None
        assert base.loop._ppo_update is ablation._no_gradient_update

    assert base.ACTIONS is original_actions
    assert source_dose._matching_reusable_branch is original_matcher
    assert base.loop._ppo_update is original_update


def test_snapshot_sources_copies_inputs_provenance_and_protocol_without_local_instructions(tmp_path):
    source_root = tmp_path / "source-dose"
    (source_root / "source_snapshot").mkdir(parents=True)
    output_root = tmp_path / "ablation"
    summary = {
        "source_manifest": {
            "foundation": {"path": "W0.bin", "sha256": "foundation-hash"},
            "dataset": {"path": "tasks.json", "sha256": "dataset-hash"},
        },
    }
    config = {"fixture": "no-environment-snapshot"}
    base._write_json(source_root / "summary.json", summary)
    base._write_json(source_root / "config.json", config)
    base._write_json(source_root / "source_snapshot" / "manifest.json", {"format": "fixture-manifest"})

    manifest = ablation._snapshot_sources(output_root, source_root)
    snapshot = output_root / "source_snapshot"

    assert manifest["foundation"] == summary["source_manifest"]["foundation"]
    assert manifest["dataset"] == summary["source_manifest"]["dataset"]
    assert manifest["source_dose_input"]["files"]["summary.json"]["sha256"] == ablation._sha(
        source_root / "summary.json"
    )
    assert manifest["source_dose_input"]["files"]["config.json"]["sha256"] == ablation._sha(
        source_root / "config.json"
    )
    assert manifest["source_dose_input"]["source_manifest"]["sha256"] == ablation._sha(
        source_root / "source_snapshot" / "manifest.json"
    )
    assert (snapshot / "scripts" / "run_learning_ablation.py").is_file()
    assert (snapshot / "docs" / "LEARNING_INTERACTION_PROTOCOL.md").is_file()
    assert manifest["excluded_local_instructions"] == ["Agent.md", "AGENTS.md"]
    assert not any(
        path.name.casefold() in {"agent.md", "agents.md"}
        for path in snapshot.rglob("*")
        if path.is_file()
    )


def _checkpoint_payload(*, changed_actor: bool = False, changed_critic: bool = False):
    critic_weight = torch.tensor([[2.0, -1.0]])
    if changed_critic:
        critic_weight[0, 0] += 0.25
    local = torch.arange(27, dtype=torch.float32)
    if changed_actor:
        local[0] += 0.5
    return {
        "core_state": {
            "source_basis": torch.ones((3, 3)),
            "slot_groups": torch.tensor([-1, 0], dtype=torch.long),
            "seen_mask": torch.tensor([True, False]),
            "_nodes.1.A": local,
            "_nodes.2.A": torch.zeros(27),
            "_shared_coefficients.0.C": torch.ones(9),
            "mlp_extractor.policy_net.0.weight": torch.ones((2, 2)),
            "action_net.weight": torch.ones((4, 2)),
            "mlp_extractor.value_net.0.weight": critic_weight,
            "value_net.weight": torch.ones((1, 2)),
        },
        "optimizer_state": {"state": {}, "param_groups": []},
        "optimizer_step_count": 23,
        "counts": {"fit_steps": 1000},
        "runtime_counts": {
            "environment_steps": 12,
            "rollouts_trained": 5,
            "rollouts_collected": 0,
            "minibatch_updates": 23,
            "slot_allocations": 0,
            "slot_evictions": 0,
            "prior_bindings": 0,
        },
        "fitted_steps": np.asarray([256, 512], dtype=np.uint64),
        "slot_generation": np.asarray([0, 1], dtype=np.int64),
    }


def test_branch_audit_accepts_only_storage_actor_changes_without_learning(tmp_path):
    branch = tmp_path / "replace__old_major"
    post_action = branch / "post_action"
    post_action.mkdir(parents=True)
    foundation = tmp_path / "W0.bin"
    foundation.write_bytes(b"frozen foundation")
    foundation_sha = base._sha(foundation)
    before = _checkpoint_payload()
    after = copy.deepcopy(before)
    after["core_state"]["_nodes.1.A"][0] += 0.5
    after["core_state"]["slot_groups"] = torch.tensor([0, 0], dtype=torch.long)
    after["runtime_counts"].update({
        "environment_steps": 524,
        "rollouts_collected": 2,
        "slot_allocations": 1,
        "slot_evictions": 1,
        "prior_bindings": 1,
    })
    after["fitted_steps"] = np.asarray([0, 512], dtype=np.uint64)
    after["slot_generation"] = np.asarray([1, 1], dtype=np.int64)
    torch.save(before, post_action / "checkpoint.pt")
    torch.save(after, branch / "checkpoint.pt")

    audit = ablation._audit_branch(
        branch,
        foundation=foundation,
        expected_foundation_sha=foundation_sha,
        expected_environment_steps=512,
    )

    assert audit["passed"]
    assert audit["critic_unchanged"]
    assert audit["w0_core_unchanged"]
    assert audit["w0_unchanged"]
    assert audit["optimizer_state_unchanged"]
    assert audit["minibatch_updates"] == 0
    assert audit["rollouts_trained"] == 0
    assert audit["rollouts_collected"] == 2
    assert audit["fit_steps_delta"] == 0
    assert audit["fitted_steps_start"] == [256, 512]
    assert audit["fitted_steps_end"] == [0, 512]
    assert not audit["fitted_steps_equal"]
    assert audit["fitted_steps_not_increased"]
    assert audit["actor_changes_restricted_to_lru_prior_codec_state"]


def test_branch_audit_rejects_critic_change(tmp_path):
    branch = tmp_path / "none__new_major"
    post_action = branch / "post_action"
    post_action.mkdir(parents=True)
    foundation = tmp_path / "W0.bin"
    foundation.write_bytes(b"frozen foundation")
    foundation_sha = base._sha(foundation)
    before = _checkpoint_payload()
    after = copy.deepcopy(before)
    after["core_state"]["mlp_extractor.value_net.0.weight"][0, 0] += 0.25
    after["runtime_counts"].update({"environment_steps": 524, "rollouts_collected": 2})
    torch.save(before, post_action / "checkpoint.pt")
    torch.save(after, branch / "checkpoint.pt")

    audit = ablation._audit_branch(
        branch,
        foundation=foundation,
        expected_foundation_sha=foundation_sha,
        expected_environment_steps=512,
    )

    assert not audit["passed"]
    assert not audit["critic_unchanged"]
    assert not audit["actor_changes_restricted_to_lru_prior_codec_state"]

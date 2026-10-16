"""Behavioral checks for the controlled write-decision runner helpers."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from scripts import run_write_decision as runner  # noqa: E402


torch.set_num_threads(1)


class _NullEnv:
    def close(self):
        pass


def _reference(seed: int):
    torch.manual_seed(seed)
    policy = TreePolicy(
        observation_space=spaces.Box(
            low=np.asarray([-2.0] * (OBS_DIM - 1) + [0.0], dtype=np.float32),
            high=np.asarray([2.0] * (OBS_DIM - 1) + [8.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )
    return SimpleNamespace(policy=policy, env=_NullEnv())


@pytest.fixture
def agent_factory(monkeypatch, tmp_path):
    foundation = tmp_path / "foundation.pt"
    dataset = tmp_path / "dataset.bin"
    foundation.write_bytes(b"small deterministic test W0")
    dataset.write_bytes(b"test dataset fingerprint")
    base = {
        "foundation": str(foundation),
        "dataset": str(dataset),
        "exploration_epsilon": 0.05,
        "temperature": 2.0,
    }
    monkeypatch.setattr(runner, "_foundation", lambda _base, seed: _reference(seed))
    config = runner._load_config()
    slots = {name: index for index, name in enumerate(runner.TASK_NAMES)}

    def create(condition="trial_gate", seed=913):
        agent = runner.make_agent(
            condition,
            seed,
            config,
            base_config=base,
            slot_map=slots,
        )
        return agent, base, config, slots

    return create


def _set_bias_only(core):
    with torch.no_grad():
        core.source_basis.zero_()
        core.source_basis[0, -1] = 1.0


def _route_observations() -> torch.Tensor:
    obs = torch.zeros((2, OBS_DIM), dtype=torch.float32)
    obs[:, 0] = torch.tensor([0.1, 0.8])
    obs[:, -1] = torch.tensor([1.0, 2.0])
    return obs


def _logits(core, obs):
    with torch.no_grad():
        return core.get_distribution(obs).distribution.logits.detach().clone()


def test_real_anchor_bank_old_route_reaches_gate_with_scalar_generation(agent_factory):
    """Exercise the live bank-to-gate path, rather than a hand-built evidence dict."""
    agent, _, config, _ = agent_factory()
    config = {**config, "gate_gain": -100.0, "gate_kl": 100.0, "min_ess": 1}
    for slot in (0, 1):
        agent.core.set_slot_group(slot, 0, preserve_effective=False)
        agent.slot_generation[slot] = 1

    def rows_for(slot):
        rows = []
        for index in range(24):
            obs = np.zeros(13, dtype=np.float32)
            obs[-1] = slot + 1
            rows.append(SimpleNamespace(
                obs=obs, action=index % 4, log_prob=math.log(0.25),
                probabilities=np.full(4, 0.25, dtype=np.float32),
                advantage=float(index % 4 == 0), returns=0.0,
                generation=1, episode_id=slot * 10 + index // 12,
            ))
        return rows

    agent.anchors.add(1, 1, rows_for(1))
    cycle = {
        "slot": 0, "task_name": "A0+", "a_start": np.zeros((3, 3, 3)),
        "c_start": np.zeros((3, 3)), "fit_band_counts": np.ones(3),
    }
    with torch.no_grad():
        agent.core._shared_coefficients[0].C.fill_(0.001)
    result = runner.finish_cycle(agent, cycle, rows_for(0), config)
    assert result["old_anchor_references"] == [{"slot": 1, "generation": 1, "rows": 24}]
    assert result["gate"]["old_slots"][0]["generation"] == 1


def test_int8_live_commit_equals_the_saved_deployment_actor(agent_factory, tmp_path):
    agent, base, config, _ = agent_factory(condition="reward_shared")
    config = {**config, "actor_codec": "int8", "actor_budget_bytes": 236}
    agent.run_config = config
    rng = np.random.default_rng(712)
    agent.core.install_local_matrix(rng.normal(size=(6, 27)).astype(np.float32))
    agent.core.install_shared_matrix(rng.normal(size=(2, 9)).astype(np.float32))
    cycle = {"slot": 0, "task_name": "A0+"}
    result = runner.finish_cycle(agent, cycle, [], config)
    assert result["actor_codec"] == "int8"
    assert result["serialization_max_coefficient_error"] > 0
    path = tmp_path / "deployment.pt"
    runner.save_checkpoint(agent, path, runner._sha(Path(base["foundation"])))
    payload = torch.load(path, weights_only=False)
    local, shared, groups = runner.decode_actor(payload["actor_binary"])
    live = runner._actor_arrays(agent.core)
    assert np.array_equal(local, live[0])
    assert np.array_equal(shared, live[1])
    assert np.array_equal(groups, live[2])


def test_compensated_candidate_and_rejection_keep_current_logits_and_other_group(agent_factory):
    agent, _, _, _ = agent_factory()
    core = agent.core
    _set_bias_only(core)
    core.set_slot_group(0, 0, preserve_effective=False)
    core.set_slot_group(1, 0, preserve_effective=False)
    local = np.zeros((6, 27), dtype=np.float32)
    local[0] = 0.125
    local[1] = 0.0625
    shared = np.zeros((2, 9), dtype=np.float32)
    shared[0] = 0.25
    shared[1] = 0.0625
    core.install_local_matrix(local)
    core.install_shared_matrix(shared)
    obs = _route_observations()[:1]
    before = _logits(core, obs)

    candidate = runner.propose_compensated_actor(
        core,
        slot=0,
        group=0,
        a_trial=np.full((3, 3, 3), 0.125, dtype=np.float32),
        c_start=np.zeros((3, 3), dtype=np.float32),
        c_trial=np.full((3, 3), 0.25, dtype=np.float32),
        c_proposal=np.full((3, 3), 0.5, dtype=np.float32),
    )
    runner._install_actor_arrays(core, *candidate)
    torch.testing.assert_close(_logits(core, obs), before, rtol=0, atol=0)
    np.testing.assert_array_equal(core.read_shared_matrix()[1], shared[1])
    np.testing.assert_array_equal(core.read_local_matrix()[1], local[1])

    core.install_local_matrix(local)
    core.install_shared_matrix(shared)
    runner.commit_candidate(
        core,
        slot=0,
        group=0,
        a_trial=np.full((3, 3, 3), 0.125, dtype=np.float32),
        c_start=np.zeros((3, 3), dtype=np.float32),
        c_trial=np.full((3, 3), 0.25, dtype=np.float32),
        c_proposal=np.full((3, 3), 0.5, dtype=np.float32),
        accepted=False,
    )
    torch.testing.assert_close(_logits(core, obs), before, rtol=0, atol=0)
    np.testing.assert_array_equal(core.read_shared_matrix()[0], 0.0)
    np.testing.assert_array_equal(core.read_shared_matrix()[1], shared[1])
    np.testing.assert_array_equal(core.read_local_matrix()[0], np.full(27, 0.375, np.float32))
    np.testing.assert_array_equal(core.read_local_matrix()[1], local[1])


def test_trial_rollback_counts_one_rejection_and_keeps_the_trained_local_effect(agent_factory):
    agent, _, config, _ = agent_factory(condition="trial_rollback")
    core = agent.core
    _set_bias_only(core)
    core.set_slot_group(0, 0, preserve_effective=False)
    core.set_slot_group(1, 0, preserve_effective=False)
    local = np.zeros((6, 27), dtype=np.float32)
    local[0] = 0.125
    local[1] = 0.0625
    shared = np.zeros((2, 9), dtype=np.float32)
    shared[0] = 0.25
    shared[1] = 0.0625
    core.install_local_matrix(local)
    core.install_shared_matrix(shared)
    expected_logits = _logits(core, _route_observations()[:1])
    cycle = {
        "slot": 0,
        "generation": 1,
        "group": 0,
        "task_name": runner.TASK_NAMES[0],
        "a_start": np.zeros((3, 3, 3), dtype=np.float32),
        "c_start": np.zeros((3, 3), dtype=np.float32),
        "fit_band_counts": np.asarray([10, 5, 2]),
    }
    agent.slot_generation[0] = 1

    record = runner.finish_cycle(agent, cycle, [], config)

    assert record["reason"] == "trial_rollback_control"
    assert record["accepted"] is False
    assert agent.counts["trial_rollback_count"] == 1
    assert agent.counts["rejected_count"] == 1
    assert agent.counts["decision_count"] == 1
    torch.testing.assert_close(_logits(core, _route_observations()[:1]), expected_logits, rtol=0, atol=0)
    np.testing.assert_array_equal(core.read_shared_matrix()[0], 0.0)
    np.testing.assert_array_equal(core.read_shared_matrix()[1], shared[1])
    np.testing.assert_array_equal(core.read_local_matrix()[0], np.full(27, 0.375, np.float32))


def test_projected_anchor_split_keeps_whole_episodes_on_one_side():
    item = {
        "slot": 2,
        "generation": 7,
        "obs": np.arange(5 * OBS_DIM, dtype=np.float32).reshape(5, OBS_DIM),
        "action": np.arange(5, dtype=np.int64) % 4,
        "behavior_log_prob": np.full(5, math.log(0.25), dtype=np.float32),
        "behavior_probs": np.full((5, 4), 0.25, dtype=np.float32),
        "advantage": np.arange(5, dtype=np.float32),
        "return": np.arange(5, dtype=np.float32),
        "generation_rows": np.full(5, 7, dtype=np.int64),
        "episode_id": np.asarray([8, 2, 8, 4, 2], dtype=np.int64),
    }

    reference, test = runner._partition_projected_evidence([item])

    ref_ids = set(np.unique(reference[0]["episode_id"]).tolist())
    test_ids = set(np.unique(test[0]["episode_id"]).tolist())
    assert ref_ids == {2, 8}
    assert test_ids == {4}
    assert ref_ids.isdisjoint(test_ids)
    assert len(reference[0]["action"]) + len(test[0]["action"]) == len(item["action"])


def test_projected_gate_marks_empty_test_partition_unsupported(agent_factory, monkeypatch):
    agent, _, config, _ = agent_factory(condition="projected_gate")
    row = SimpleNamespace(
        obs=np.zeros(OBS_DIM, dtype=np.float32),
        action=0,
        log_prob=math.log(0.25),
        probabilities=np.full(4, 0.25, dtype=np.float32),
        advantage=1.0,
        episode_id=1,
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_arrays",
        lambda *_args, **_kwargs: {
            "support_ok": True,
            "mean_delta": 0.1,
            "episode_se": 0.0,
            "ess": 1.0,
            "sample_count": 1,
            "episode_count": 1,
        },
    )
    result = runner._gate_candidate(
        agent,
        slot=0,
        group=0,
        holdout_rows=[row],
        candidate_local=agent.core.read_local_matrix(),
        candidate_shared=agent.core.read_shared_matrix(),
        c_start=agent.core.read_shared_matrix()[0].reshape(3, 3),
        old_evidence=[],
        config=config,
        require_old_evidence=True,
        evidence_partition={"reference": {"rows": 1}, "test": {"rows": 0}},
    )

    assert result["accepted"] is False
    assert result["reason"] == "projected_test_evidence_empty"
    assert result["old_test_support"]["support_ok"] is False
    assert result["old_evidence_partition"]["test"]["rows"] == 0


def test_checkpoint_round_trip_restores_actor_profiles_anchors_and_real_capacity(agent_factory, tmp_path):
    agent, base, config, _ = agent_factory(seed=919)
    core = agent.core
    core.set_slot_group(0, 0, preserve_effective=False)
    local = np.arange(6 * 27, dtype=np.float32).reshape(6, 27) / 1000.0
    shared = np.arange(2 * 9, dtype=np.float32).reshape(2, 9) / 100.0
    core.install_local_matrix(local)
    core.install_shared_matrix(shared)
    agent.slot_generation[0] = 1
    agent.profiles["seen"][0] = 1
    agent.profiles["shadow_group"][0] = 0
    agent.profile_reasons[0] = "test_assignment"
    agent.counts["environment_steps"] = 1024
    agent.anchors.add(0, 1, [{
        "obs": np.zeros(OBS_DIM, dtype=np.float32),
        "action": 2,
        "behavior_log_prob": math.log(0.25),
        "behavior_probs": np.full(4, 0.25, dtype=np.float32),
        "advantage": 0.75,
        "return": 1.25,
        "episode_id": 42,
    }])
    expected_probs = core.get_distribution(_route_observations()).distribution.probs.detach().clone()
    path = tmp_path / "checkpoint.pt"
    sizes = runner.save_checkpoint(agent, path, runner._sha(Path(base["foundation"])))

    assert sizes["actor_bytes"] == 738
    assert sizes["full_persistent_bytes"] == sizes["checkpoint_bytes"] + sizes["foundation_bytes"]
    assert sizes["foundation_bytes"] == Path(base["foundation"]).stat().st_size
    assert sizes["full_persistent_bytes"] <= config["package_budget_bytes"]
    restored = runner.restore_agent_checkpoint(path, base_config=base)

    np.testing.assert_array_equal(restored.core.read_local_matrix(), local)
    np.testing.assert_array_equal(restored.core.read_shared_matrix(), shared)
    assert restored.profile_reasons[0] == "test_assignment"
    assert restored.counts["environment_steps"] == 1024
    np.testing.assert_array_equal(restored.anchors.arrays(0, 1)["episode_id"], [42])
    torch.testing.assert_close(
        restored.core.get_distribution(_route_observations()).distribution.probs.detach(),
        expected_probs,
        rtol=0,
        atol=0,
    )


class _Recorder:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(dict(row))


class _ShortPairedEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self, *, seed=None):
        self.step_count = 0
        self.seed = seed
        return np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        next_obs = np.asarray([self.step_count / 8.0, 0.0, 1.0, 1.0], dtype=np.float32)
        reward = 1.0 if int(action) == 2 else -0.25
        terminated = self.step_count == 3
        return next_obs, reward, terminated, False, {"goal_reached": int(action) == 2}

    def close(self):
        pass


def _numpy_rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_evaluation_pairs_task_action_plans_without_consuming_training_rng(agent_factory, monkeypatch):
    agent, _, config, _ = agent_factory(condition="independent_fast", seed=929)
    monkeypatch.setattr(runner, "make_env", lambda _spec, _domain: _ShortPairedEnv())
    eval_config = {**config, "evaluation_episodes": 2}
    before_action_rng = copy.deepcopy(agent.action_rng.bit_generator.state)
    before_numpy = np.random.get_state()
    before_python = __import__("random").getstate()
    before_torch = torch.get_rng_state().clone()

    first_steps, first_evaluations = _Recorder(), _Recorder()
    first, first_cost = runner.evaluate_stage(
        agent,
        "initial",
        eval_config,
        steps_writer=first_steps,
        evaluations_writer=first_evaluations,
    )
    second_steps, second_evaluations = _Recorder(), _Recorder()
    second, second_cost = runner.evaluate_stage(
        agent,
        "final",
        eval_config,
        steps_writer=second_steps,
        evaluations_writer=second_evaluations,
    )

    first_plan = [(row["taskobserver"], row["bout"], row["action"]) for row in first_steps.rows]
    second_plan = [(row["taskobserver"], row["bout"], row["action"]) for row in second_steps.rows]
    assert first_plan == second_plan
    assert first_cost == second_cost == len(first_plan)
    for name in runner.TASK_NAMES:
        first_episodes = first[name]["episodes"]
        second_episodes = second[name]["episodes"]
        assert [
            (row["reset_seed"], row["action_seed"], row["return_value"], row["goal_reached"], row["length"])
            for row in first_episodes
        ] == [
            (row["reset_seed"], row["action_seed"], row["return_value"], row["goal_reached"], row["length"])
            for row in second_episodes
        ]
    assert agent.action_rng.bit_generator.state == before_action_rng
    assert _numpy_rng_state_equal(np.random.get_state(), before_numpy)
    assert __import__("random").getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert agent.counts["evaluation_steps"] == first_cost + second_cost


def _complete_validation_record(condition, seed, config):
    task_results = {}
    for task_index, task_name in enumerate(runner.TASK_NAMES):
        task_results[task_name] = {
            "episodes": [{
                "reset_seed": seed + 50000 + task_index * 1000 + episode * 17,
                "action_seed": seed + 70000 + task_index * 1000 + episode * 17,
                "complete": 1,
            } for episode in range(config["evaluation_episodes"])],
        }
    return {
        "condition": condition,
        "seed": seed,
        "sequence": list(config["sequence"]),
        "bout_steps": config["validation_bout_steps"],
        "counts": {
            "environment_steps": config["validation_bout_steps"] * len(config["sequence"]),
            "probe_steps": len(runner.TASK_NAMES) * config["probe_steps"],
            "evaluation_steps": 144,
        },
        "costs": {"evaluation_steps": 144},
        "evaluations": {
            stage: task_results for stage in ("initial", "after_source", "final")
        },
        "reward_per_step": float(runner.PROPOSED_CONDITIONS.index(condition) if condition in runner.PROPOSED_CONDITIONS else 0),
    }


def _complete_validation_summary(config):
    return {
        "phase": "validation",
        "conditions": list(config["validation_conditions"]),
        "seeds": list(config["validation_seeds"]),
        "training_steps": len(config["validation_conditions"])
        * len(config["validation_seeds"])
        * config["validation_bout_steps"]
        * len(config["sequence"]),
        "records": [
            _complete_validation_record(condition, seed, config)
            for seed in config["validation_seeds"]
            for condition in config["validation_conditions"]
        ],
    }


def test_selection_requires_full_configured_validation_and_rejects_duplicates(tmp_path):
    config = runner._load_config()
    summary_path = tmp_path / "summary.json"
    selection_path = tmp_path / "selection.json"
    summary = _complete_validation_summary(config)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    selection = runner.select_conditions(summary_path, selection_path)

    assert len(selection["selected_conditions"]) == 2
    assert selection["formal_conditions"][:3] == list(runner.SIMPLE_BASELINES)
    duplicate_summary = copy.deepcopy(summary)
    duplicate_summary["records"][-1] = copy.deepcopy(duplicate_summary["records"][0])
    summary_path.write_text(json.dumps(duplicate_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        runner.select_conditions(summary_path, tmp_path / "duplicate-selection.json")


def test_selection_rejects_formal_phase_and_formal_evaluation_override(tmp_path):
    config = runner._load_config()
    summary = _complete_validation_summary(config)
    summary["phase"] = "formal"
    summary_path = tmp_path / "formal.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="validation-phase"):
        runner.select_conditions(summary_path, tmp_path / "selection.json")
    with pytest.raises(ValueError, match="evaluation count"):
        runner.run_cohort(
            "formal",
            tmp_path / "formal-output",
            input_path=tmp_path / "selection.json",
            evaluation_episodes=1,
        )

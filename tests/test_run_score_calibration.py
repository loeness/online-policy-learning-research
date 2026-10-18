from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.score_calibration import snapshot_navigation  # noqa: E402
from online_rl.write_decision_envs import TaskSpec, make_env  # noqa: E402
from scripts import run_score_calibration as calibration  # noqa: E402
from scripts import run_retention_stream as stream  # noqa: E402


class _Core:
    def __init__(self) -> None:
        self.shared = np.zeros((2, 9), dtype=np.float32)
        self.slot_groups = np.asarray([0, -1, -1], dtype=np.int64)

    def state_dict(self):
        return {
            "shared": torch.as_tensor(self.shared.copy()),
            "slot_groups": torch.as_tensor(self.slot_groups.copy()),
        }

    def read_shared_matrix(self) -> np.ndarray:
        return self.shared.copy()

    def install_shared_matrix(self, value: np.ndarray) -> None:
        self.shared[...] = value

    def group_for_slot(self, slot: int) -> int:
        return int(self.slot_groups[slot])


def _evidence(source_slot: int = 0, source_generation: int = 1) -> dict[str, np.ndarray]:
    return {
        "action": np.asarray([0, 1], dtype=np.int8),
        "source_slot": np.asarray([source_slot, source_slot], dtype=np.int64),
        "source_generation": np.asarray([source_generation, source_generation], dtype=np.int64),
    }


def _agent(step: int = 12_288) -> SimpleNamespace:
    cache = stream.ws.BoundedTaskCache(capacity=1, seed=41)
    cache.resolve("A0+")
    core = _Core()
    core.shared[0] = np.linspace(-0.4, 0.4, 9, dtype=np.float32)
    return SimpleNamespace(
        core=core,
        cache=cache,
        slot_generation=np.asarray([1, 0, 0], dtype=np.int64),
        counts={"raw_steps": int(step), "write_accepts": 0},
        action_rng=np.random.default_rng(51),
        behavior_rng=np.random.default_rng(52),
        pending={
            "group": 0,
            "parent_version": 3,
            "common": np.asarray(
                [[0.2, 0.1, 0.0], [0.0, -0.1, 0.3], [0.2, 0.0, -0.2]],
                dtype=np.float32,
            ),
            "created_step": int(step) - 512,
            "sources": [{"slot": 0, "generation": 1, "prototype": 0}],
        },
        archive=SimpleNamespace(arrays=lambda _prototype: _evidence()),
    )


def _decision(agent: SimpleNamespace, *_args, accepted: bool = True) -> dict:
    pending = agent.pending
    step = int(agent.counts["raw_steps"])
    agent.pending = None
    return {
        "event": "write_test",
        "step": step,
        "accepted": accepted,
        "reason": "retention_supported" if accepted else "mean_lower_not_above_threshold",
        "proposal_step": int(pending["created_step"]),
        "first_test_row_step": int(pending["created_step"] + 1),
        "group": int(pending["group"]),
        "common": np.asarray(pending["common"]).tolist(),
        "sources": pending["sources"],
        "historical_scores": [
            {"prototype": 0, "lower": 0.025, "gain": 0.04, "stderr": 0.015}
        ],
        "fresh_score": {"lower": 0.01, "gain": 0.03, "stderr": 0.02},
    }


def _observer(bout_steps: int = 2048) -> calibration.CalibrationObserver:
    config = {
        "capture_initial_bouts": 6,
        "anchor_offsets": [0, 64, 128, 192],
        "replicates": 8,
        "mc_interval_level": 0.95,
        "benefit_margin": 0.0,
    }
    return calibration.CalibrationObserver(config, seed=19703, budget="low", bout_steps=bout_steps)


def test_deployed_cold_table_matches_real_policy_on_all_three_routes():
    config = json.loads(stream.CONFIG.read_text(encoding="utf-8"))
    agent = stream.make_agent(config, "retention_gate", "low", seed=829)
    core = agent.core
    assert agent.slots == 3

    rng = np.random.default_rng(311)
    raw_c0 = rng.normal(0.0, 0.2, size=(3, 3)).astype(np.float32)
    cold_c0_q, _ = calibration.roundtrip_common(raw_c0)
    assert np.any(raw_c0 != 0.0)
    assert np.any(cold_c0_q != 0.0)

    core.install_local_matrix(np.zeros((3, 27), dtype=np.float32))
    shared = np.zeros((core.shared_groups, 9), dtype=np.float32)
    shared[0] = cold_c0_q.reshape(9)
    core.install_shared_matrix(shared)
    for slot in range(agent.slots):
        core.set_slot_group(slot, 0, preserve_effective=False)
        assert core.group_for_slot(slot) == 0
    np.testing.assert_array_equal(core.read_local_matrix(), 0.0)
    np.testing.assert_array_equal(core.read_shared_matrix()[0].reshape(3, 3), cold_c0_q)

    observations = np.zeros((13, 13), dtype=np.float32)
    observations[:, :4] = rng.integers(0, 9, size=(13, 4)).astype(np.float32) / 8.0
    observations[:, 4:12] = rng.uniform(-1.0, 1.0, size=(13, 8)).astype(np.float32)
    grid = calibration.grid_observations()
    lookup = stream.coefficient_probs(core, grid, cold_c0_q)
    indices = [calibration.probability_index(row[:4]) for row in observations]
    looked_up = lookup[indices]

    per_route_actual = []
    for route in (1, 2, 3):
        routed = observations.copy()
        routed[:, 12] = route
        actual = core.get_distribution(torch.as_tensor(routed)).distribution.probs.detach().cpu().numpy()
        direct = stream.coefficient_probs(core, routed, cold_c0_q)
        np.testing.assert_allclose(actual, direct, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual, looked_up, rtol=1e-6, atol=1e-7)
        per_route_actual.append(actual)

    for actual in per_route_actual[1:]:
        np.testing.assert_allclose(actual, per_route_actual[0], rtol=0.0, atol=0.0)


def test_evaluate_freezes_old_parent_before_write_and_leaves_learner_state_unchanged(monkeypatch):
    observer = _observer()
    agent = _agent()
    observer.update_lineage(agent)
    old_parent = agent.core.shared[0].reshape(3, 3).copy()
    candidate = np.asarray(agent.pending["common"]).copy()
    before = calibration.learner_signature(agent)

    monkeypatch.setattr(
        calibration.stream,
        "score_common",
        lambda *_args, **_kwargs: {
            "supported": True,
            "lower": 0.02,
            "gain": 0.04,
            "stderr": 0.02,
        },
    )
    result = observer.evaluate(
        _decision,
        agent,
        {},
        [SimpleNamespace(step=11_777)],
        0,
    )

    assert result["event"] == "write_test"
    assert calibration.learner_signature(agent) == before
    np.testing.assert_array_equal(observer.predictions[0]["old_parent_raw"], old_parent)
    np.testing.assert_array_equal(observer.predictions[0]["candidate_raw"], candidate)
    assert observer.predictions[0]["prediction_frozen_before_outcome"] is True

    # A later accepted install must not mutate the saved baseline snapshot.
    updated_shared = agent.core.shared.copy()
    updated_shared[0] = candidate.reshape(9)
    agent.core.install_shared_matrix(updated_shared)
    np.testing.assert_array_equal(observer.predictions[0]["old_parent_raw"], old_parent)


def test_evaluate_records_accepted_and_rejected_prefix_tests_but_not_later(monkeypatch):
    observer = _observer()
    monkeypatch.setattr(
        calibration.stream,
        "score_common",
        lambda *_args, **_kwargs: {
            "supported": True,
            "lower": 0.02,
            "gain": 0.04,
            "stderr": 0.02,
        },
    )
    seen_decisions = []
    for step, accepted in ((12_287, True), (12_288, False), (12_289, True)):
        agent = _agent(step)
        observer.update_lineage(agent)
        result = observer.evaluate(
            lambda *args, accepted=accepted: _decision(args[0], accepted=accepted),
            agent,
            {},
            [SimpleNamespace(step=step - 255)],
            0,
        )
        seen_decisions.append(result)

    assert [event["accepted"] for event in seen_decisions] == [True, False, True]
    assert [prediction["step"] for prediction in observer.predictions] == [12_287, 12_288]
    assert [prediction["accepted"] for prediction in observer.predictions] == [True, False]
    assert all(type(prediction["step"]) is int for prediction in observer.predictions)


def test_source_generation_lineage_survives_cache_slot_reuse(monkeypatch):
    observer = _observer()
    agent = _agent(step=12_288)
    observer.update_lineage(agent)
    assert observer.lineage[(0, 1)] == "A0+"

    # Reuse slot zero for another task and generation; the historical key stays fixed.
    agent.cache.resolve("B0+")
    agent.slot_generation[0] = 2
    observer.update_lineage(agent)
    assert observer.lineage[(0, 1)] == "A0+"
    assert observer.lineage[(0, 2)] == "B0+"

    agent.archive.arrays = lambda _prototype: _evidence(source_generation=1)
    monkeypatch.setattr(
        calibration.stream,
        "score_common",
        lambda *_args, **_kwargs: {
            "supported": True,
            "lower": 0.02,
            "gain": 0.04,
            "stderr": 0.02,
        },
    )
    observer.evaluate(
        _decision,
        agent,
        {},
        [SimpleNamespace(step=12_033)],
        0,
    )
    prediction = observer.predictions[0]
    assert prediction["contexts"] == ["A0+"]
    assert prediction["proposal_sources"][0]["task_observer"] == "A0+"
    assert prediction["historical_provenance"][0]["context_counts"] == {"A0+": 2}
    assert prediction["missing_source_lineage"] == []


def test_natural_anchor_is_pre_step_and_snapshot_does_not_mutate_live_environment(monkeypatch):
    observer = _observer()
    observer.learner_steps = observer.cutoff
    observer.seen_tasks.add("A0+")
    env = make_env(TaskSpec("A", 0, 1), domain="navigation")
    env.reset(seed=8102)
    env._position = (1, 2)
    env._goal = (6, 5)
    env._episode_length = 9
    wrapped = calibration.ObservedEnvironment(env, observer)
    try:
        wrapped.set_task(TaskSpec("A", 0, 1))
        before = snapshot_navigation(env)
        monkeypatch.setattr(env, "step", lambda _action: pytest.fail("anchor capture stepped the environment"))

        observer.before_step(env)

        after = snapshot_navigation(env)
        assert after == before
        assert observer.learner_steps == observer.cutoff
        assert observer.observer_steps == 0
        anchor = observer.anchors["A0+"][0]
        assert anchor["offset"] == 0
        assert type(anchor["raw_step"]) is int
        assert anchor["raw_step"] == observer.cutoff + 1
        assert anchor["raw_step"] > observer.cutoff
        assert anchor["snapshot"] == before
        assert observer.snapshot_audits == [True]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("replicate_gains", "expected_sign"),
    [
        ([0.0] * 8, "ambiguous"),
        ([1.0] * 8, "beneficial"),
        ([-1.0] * 8, "harmful"),
    ],
)
def test_outcome_summary_uses_equal_replicates_and_ci_sign_labels(replicate_gains, expected_sign):
    result = calibration.outcome_summary(replicate_gains, [0.3] * 8, _observer().config)

    assert result["sign"] == expected_sign
    assert result["stderr"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(result["mean_gain"])
    assert result["ci_high"] == pytest.approx(result["mean_gain"])
    assert result["replicate_gains"] == replicate_gains
    assert result["prototype_weighted_replicate_gains"] == [0.3] * 8
    assert result["interval_scope"].startswith("Pointwise paired Monte Carlo interval")


def test_outcome_summary_rejects_wrong_replicate_count():
    with pytest.raises(RuntimeError, match="invalid paired outcome replicates"):
        calibration.outcome_summary([0.1, 0.2], [0.0, 0.0], _observer().config)

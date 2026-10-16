import csv
import json
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online_rl.write_decision_router import WriteDecisionRouter  # noqa: E402
from online_rl.policy import OBS_DIM, TreePolicy  # noqa: E402
from online_rl.write_decision_policy import FlexibleSharedPolicy  # noqa: E402
from online_rl.write_decision_gate import AnchorBank  # noqa: E402
from scripts import run_knowledge_loop as loop  # noqa: E402
from scripts import run_write_stream as stream  # noqa: E402


def _obs(x: int = 2, y: int = 4, gx: int = 8, gy: int = 4) -> np.ndarray:
    return np.asarray([x / 8, y / 8, gx / 8, gy / 8], dtype=np.float32)


def _next(obs: np.ndarray, direction: int = 1) -> np.ndarray:
    deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))
    dx, dy = deltas[direction]
    result = obs.copy()
    x, y = int(round(float(obs[0]) * 8)), int(round(float(obs[1]) * 8))
    result[0] = min(max(x + dx, 0), 8) / 8
    result[1] = min(max(y + dy, 0), 8) / 8
    return result


def test_fixed_schedule_is_twelve_tasks_then_reverse() -> None:
    assert stream.stream_task_order() == stream.TASKS + tuple(reversed(stream.TASKS))
    assert stream.stream_task_order()[:12] == (
        "A0+", "A0-", "B0+", "B0-", "A1+", "A1-",
        "B1+", "B1-", "A2+", "A2-", "B2+", "B2-",
    )
    assert len(stream.stream_task_order()) == 24
    config = stream._load_config()
    assert config["base_sequence"] == list(stream.TASKS)
    assert config["sequence"] == list(stream.STREAM_SEQUENCE)


def test_stream_actor_uses_int8_deployment_capacity() -> None:
    config = stream._load_config()
    assert config["actor_codec"] == "int8"
    assert config["actor_budget_bytes"] == 236
    assert config["independent7_actor_budget_bytes"] == 236
    stream._validate_local_slot_conditions(["independent7_lru"], 7)
    with pytest.raises(ValueError, match="independent G0"):
        stream._validate_local_slot_conditions(["reward_shared"], 7)


def test_foundation_override_is_resolved_and_keeps_base_dataset(tmp_path: Path) -> None:
    foundation = tmp_path / "other_w0.pt"
    foundation.write_bytes(b"synthetic-checkpoint")
    base = {"foundation": "default-w0.pt", "dataset": "fixed-dataset.npz"}
    resolved = stream._apply_foundation_override(base, {"foundation": str(foundation)})
    assert resolved["foundation"] == str(foundation.resolve())
    assert resolved["dataset"] == "fixed-dataset.npz"
    with pytest.raises(FileNotFoundError, match="foundation checkpoint"):
        stream._apply_foundation_override(base, {"foundation": str(tmp_path / "missing.pt")})


def test_selection_resolution_includes_required_baselines_and_controls(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({
        "format": "write_decision_selection_v1",
        "selected_conditions": ["trial_gate", "projected_gate"],
    }), encoding="utf-8")
    selection = stream._load_selection(selection_path)

    assert stream.resolve_conditions(selection) == [
        "trial_gate", "projected_gate", "reward_shared", "independent_fast",
        "independent6", "independent7_lru", "independent7_random", "independent7_lfu",
    ]
    assert stream.resolve_conditions(selection, ["selected", "independent6"]) == [
        "trial_gate", "projected_gate", "independent6",
    ]
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps({
        "format": "write_decision_selection_v1",
        "selected_conditions": ["trial_gate"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        stream._load_selection(invalid_path)


def test_known_cache_has_bounded_lru_state_and_serializes_eviction_metadata() -> None:
    cache = stream.BoundedTaskCache(capacity=6, eviction="lru", seed=4)
    for task in stream.TASKS[:6]:
        cache.resolve(task)
    cache.resolve(stream.TASKS[0])
    inserted = cache.resolve(stream.TASKS[6])

    assert inserted["allocated"] is True
    assert inserted["evicted_slot"] == 1
    assert inserted["evicted_task"] == stream.TASKS[1]
    assert len(cache.task_to_slot) == len(cache.slot_to_task) == 6
    restored = stream.BoundedTaskCache.from_state_dict(cache.state_dict())
    assert restored.task_to_slot == cache.task_to_slot
    assert restored.slot_to_task == cache.slot_to_task
    assert restored.nbytes == cache.nbytes


@pytest.mark.parametrize("eviction", ["random", "lfu"])
def test_independent7_cache_eviction_is_seeded_and_bounded(eviction: str) -> None:
    first = stream.BoundedTaskCache(capacity=7, eviction=eviction, seed=912)
    second = stream.BoundedTaskCache(capacity=7, eviction=eviction, seed=912)
    for task in stream.TASKS[:7]:
        first.resolve(task)
        second.resolve(task)
    assert first.resolve(stream.TASKS[7]) == second.resolve(stream.TASKS[7])
    assert len(first.task_to_slot) == 7


def test_unknown_behavior_log_probability_matches_actual_uniform_mixture() -> None:
    base = np.asarray([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
    behavior = stream.unknown_behavior_probs(base, 0.5)
    np.testing.assert_allclose(behavior, [0.475, 0.175, 0.175, 0.175], atol=1e-7)
    assert np.isclose(float(behavior.sum()), 1.0)
    with pytest.raises(ValueError, match="sum to one"):
        stream.unknown_behavior_probs([1, 1, 1, 1], 0.5)


def test_unknown_router_events_do_not_depend_on_observer_task_label() -> None:
    first = WriteDecisionRouter(window=32, seed=71)
    second = WriteDecisionRouter(window=32, seed=71)
    agent_a = SimpleNamespace(run_config={"phase": "unknown"}, router=first)
    agent_b = SimpleNamespace(run_config={"phase": "unknown"}, router=second)
    obs = _obs(2)
    for index in range(16):
        current = _obs(1 if index % 2 == 0 else 3)
        nxt = _next(current, 1)
        event_a = stream._record_visible_event(agent_a, current, 0, 0.2, nxt, False,
                                              task_name="A0+", probe=False)
        event_b = stream._record_visible_event(agent_b, current, 0, 0.2, nxt, False,
                                              task_name="B2-", probe=False)
        assert event_a == event_b
    left, right = first.state_dict(), second.state_dict()
    for name, value in left.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, right[name])
        elif name != "rng_state":
            assert value == right[name]
    del obs


def test_unknown_router_rows_are_masked_from_actor_ppo() -> None:
    mask = loop.actor_valid_mask(
        np.asarray([-1, 0], dtype=np.int64),
        np.asarray([-1, 2], dtype=np.int64),
        [0, 0, 1, 0, 0, 0],
        arm="write_stream",
    )
    np.testing.assert_array_equal(mask, [False, False])
    mask = loop.actor_valid_mask(
        np.asarray([-1, 0], dtype=np.int64),
        np.asarray([-1, 1], dtype=np.int64),
        [0], arm="single_head_online",
    )
    np.testing.assert_array_equal(mask, [True, True])


def test_package_capacity_counts_foundation_plus_checkpoint() -> None:
    assert stream.full_persistent_bytes(55_477, 100_000) == 155_477
    assert stream.full_persistent_bytes(55_477, 206_667) == 262_144
    with pytest.raises(ValueError, match="nonnegative"):
        stream.full_persistent_bytes(55_477, -1)


def test_group_summary_reports_unassigned_after_shared_capacity_fills() -> None:
    summary = stream._summarize_group_coverage({
        "A0+": {"steps": 10, "grouped_steps": 6, "unassigned_steps": 4,
                "group_full_steps": 5, "unassigned_when_full_steps": 3,
                "resident_slots": [0], "groups_used": [0]},
        "B0+": {"steps": 8, "grouped_steps": 4, "unassigned_steps": 4,
                "group_full_steps": 7, "unassigned_when_full_steps": 2,
                "resident_slots": [1], "groups_used": [1]},
        "B0-": {"steps": 5, "grouped_steps": 5, "unassigned_steps": 0,
                "group_full_steps": 5, "unassigned_when_full_steps": 0,
                "resident_slots": [2], "groups_used": [0]},
    })
    assert summary["unassigned_steps_total"] == 8
    assert summary["unassigned_step_fraction"] == pytest.approx(8 / 23)
    assert summary["group_full_steps_total"] == 17
    assert summary["unassigned_when_full_steps_total"] == 5
    assert summary["unassigned_when_full_step_fraction"] == pytest.approx(5 / 17)
    assert summary["unassigned_when_full_task_coverage"] == ["A0+", "B0+"]


def _group_binding_fixture(*, fitted_steps: int):
    torch.manual_seed(827)
    reference = TreePolicy(
        observation_space=spaces.Box(
            low=np.asarray([-2.0] * (OBS_DIM - 1) + [0.0], dtype=np.float32),
            high=np.asarray([2.0] * (OBS_DIM - 1) + [8.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1e-2,
    )
    core = FlexibleSharedPolicy.from_reference(reference, local_slots=1, shared_groups=2, train_critic=False)
    with torch.no_grad():
        core.source_basis.normal_(mean=0.0, std=0.1)
        core._nodes["1"].A.normal_(mean=0.0, std=0.2)
    core.install_shared_matrix(np.full((2, 9), 0.15, dtype=np.float32))

    class Grouper:
        min_support = 8

        @staticmethod
        def assign(_profile, *, commit):
            return SimpleNamespace(group=0, reason="supported_match", scores={"group_0": 0.9})

    movement = np.zeros((3, 4, 4), dtype=np.uint16)
    movement[0, 0, 1] = 4
    movement[1, 0, 1] = 4
    profile = {
        "movement_counts": movement,
        "reward_xtx": np.eye(3, dtype=np.float32),
        "reward_xty": np.ones(3, dtype=np.float32),
        "reward_n": np.float32(16),
    }
    agent = SimpleNamespace(
        core=core,
        slots=1,
        slot_generation=np.asarray([3], dtype=np.int64),
        fitted_steps=np.asarray([fitted_steps], dtype=np.uint64),
        grouped_generation=np.asarray([-1], dtype=np.int64),
        group_attempt_support=np.zeros(1, dtype=np.uint64),
        profiles={
            "movement_counts": np.zeros((1, 3, 4, 4), dtype=np.uint16),
            "reward_xtx": np.zeros((1, 3, 3), dtype=np.float32),
            "reward_xty": np.zeros((1, 3), dtype=np.float32),
            "reward_n": np.zeros(1, dtype=np.float32),
            "seen": np.zeros(1, dtype=np.uint8),
            "shadow_group": np.full(1, -1, dtype=np.int8),
        },
        profile_reasons=["not_profiled"],
        grouper=Grouper(),
        counts={
            "group_retries": 0, "group_abstentions": 0, "group_max_abstentions": 0,
            "cold_prior_bindings": 0, "effective_preserving_bindings": 0,
        },
        run_config={"group_retry_interval": 16},
        cycle_state=None,
        rollout=[],
        cycle_step=0,
    )
    return agent, profile


def _real_actor_logits(core, route: int = 1) -> torch.Tensor:
    obs = loop.augment_observation(_obs(2, 3, 7, 5), route)
    tensor = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(tensor))
        return core.action_net(latent) + core._correction(tensor, latent)


def test_fitted_slot_group_binding_preserves_real_actor_logits() -> None:
    agent, profile = _group_binding_fixture(fitted_steps=16)
    before = _real_actor_logits(agent.core)

    event = stream._maybe_assign_group(agent, 0, profile)

    after = _real_actor_logits(agent.core)
    assert event == {"group": 0, "reason": "supported_match", "binding": "preserve_effective", "committed": True}
    torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-7)
    assert agent.counts["effective_preserving_bindings"] == 1
    assert agent.counts["cold_prior_bindings"] == 0


def test_cold_slot_binding_is_marked_as_shared_prior_heuristic() -> None:
    agent, profile = _group_binding_fixture(fitted_steps=0)
    before = _real_actor_logits(agent.core)

    event = stream._maybe_assign_group(agent, 0, profile)

    after = _real_actor_logits(agent.core)
    assert event["binding"] == "cold_prior_binding"
    assert agent.counts["cold_prior_bindings"] == 1
    assert not torch.equal(after, before)


def test_generation_reset_clears_fitted_step_evidence() -> None:
    local = torch.nn.Parameter(torch.ones((3, 3, 3), dtype=torch.float32))

    class Core:
        shared_groups = 2
        _nodes = {"1": SimpleNamespace(A=local)}

        @staticmethod
        def group_for_slot(_slot: int) -> int:
            return -1

    anchors = AnchorBank(slots=1, per_slot=2, seed=7)
    anchors.add(0, 2, [{
        "obs": np.zeros(13), "action": 0, "log_prob": float(np.log(.25)),
        "probabilities": np.full(4, .25), "advantage": 1.0,
        "return": 1.0, "generation": 2, "episode_id": 1,
    }])

    agent = SimpleNamespace(
        core=Core(), optimizer=SimpleNamespace(state={}), slots=1,
        anchors=anchors, slot_generation=np.asarray([2], dtype=np.int64),
        fitted_steps=np.asarray([384], dtype=np.uint64),
        profiles={
            "movement_counts": np.ones((1, 3, 4, 4), dtype=np.uint16),
            "reward_xtx": np.ones((1, 3, 3), dtype=np.float32),
            "reward_xty": np.ones((1, 3), dtype=np.float32),
            "reward_n": np.ones(1, dtype=np.float32),
            "seen": np.ones(1, dtype=np.uint8),
            "shadow_group": np.zeros(1, dtype=np.int8),
        },
        grouped_generation=np.asarray([2], dtype=np.int64),
        group_attempt_support=np.asarray([64], dtype=np.uint64),
        profile_reasons=["assigned"], windows=[object()],
        counts={"slot_allocations": 0}, run_config={"probe_steps": 128},
    )

    generation = stream._reset_allocated_slot(agent, 0, reason="evicted")

    assert generation == 3
    assert agent.fitted_steps[0] == 0
    assert agent.counts["slot_allocations"] == 1
    assert len(agent.anchors.arrays(0, 2)["action"]) == 0
    assert agent.anchors.state_dict()["active_generations"][0] == 3


def test_mixed_task_episode_is_reported_and_excluded_from_positive_denominator() -> None:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=("interaction_kind", "taskobserver", "episode_start_taskobserver", "episode_task_observers",
                    "mixed_task_episode", "bout", "episode_id", "episode", "return_value", "goal_reached",
                    "positive_success", "length", "complete"),
    )
    writer.writeheader()
    agent = SimpleNamespace(
        counts={"mixed_task_episodes": 0, "completed_episodes": 0, "all_goal_episodes": 0,
                "positive_completed_episodes": 0, "positive_goal_episodes": 0},
        goal_episodes_by_task={name: 0 for name in stream.TASKS},
        complete_episodes_by_task={name: 0 for name in stream.TASKS},
        positive_complete_episodes_by_task={name: 0 for name in stream.TASKS},
    )

    stream._complete_episode(
        writer, agent, kind="training", task_name="A0-", start_task_name="A0+", bout=1,
        episode_id=0, episode_index=0, episode_return=-1.0, episode_length=5,
        goal_reached=True, preference=-1, task_observers={"A0+", "A0-"},
    )

    row = __import__("csv").DictReader(io.StringIO(output.getvalue()))
    recorded = next(row)
    assert recorded["mixed_task_episode"] == "1"
    assert recorded["positive_success"] == ""
    assert recorded["episode_task_observers"] == '["A0+","A0-"]'
    assert agent.counts["mixed_task_episodes"] == 1
    assert agent.counts["positive_completed_episodes"] == 0
    assert agent.positive_complete_episodes_by_task["A0+"] == 0
    assert agent.positive_complete_episodes_by_task["A0-"] == 0

    stream._complete_episode(
        writer, agent, kind="training", task_name="A0+", start_task_name="A0+", bout=2,
        episode_id=1, episode_index=1, episode_return=1.0, episode_length=5,
        goal_reached=True, preference=1, task_observers={"A0+"},
    )
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert rows[0]["positive_success"] == ""
    assert rows[1]["positive_success"] == "1"
    assert agent.counts["positive_completed_episodes"] == 1
    assert agent.positive_complete_episodes_by_task["A0+"] == 1
    assert agent.goal_episodes_by_task["A0+"] == 1


def test_partial_cycle_without_holdout_skips_gate_and_keeps_effective_update(monkeypatch, tmp_path: Path) -> None:
    shared = torch.nn.Parameter(torch.ones((3, 3), dtype=torch.float32))
    local = torch.nn.Parameter(torch.zeros((3, 3, 3), dtype=torch.float32))

    class Core:
        shared_groups = 2
        _nodes = {"1": SimpleNamespace(A=local)}
        _shared_coefficients = [SimpleNamespace(C=shared)]

        @staticmethod
        def group_for_slot(slot: int) -> int:
            assert slot == 0
            return 0

        @staticmethod
        def shared_parameters():
            return (shared,)

    agent = SimpleNamespace(
        rollout=[], rollout_phase=None, condition="trial_gate", external_condition="trial_gate",
        core=Core(), optimizer=SimpleNamespace(state={}), slots=1,
        cycle_state={"slot": 0, "generation": 3, "group": 0, "fit_rows": 64,
                     "task_name": "unknown", "c_start": np.zeros((3, 3), dtype=np.float32)},
        holdout_rows=[], cycle_step=64,
        counts={"environment_steps": 64}, run_config={},
        current_route=(0, 3),
    )
    monkeypatch.setattr(stream, "_save_stream_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(stream.step1, "roundtrip_actor", lambda *args, **kwargs: {"actor_codec": "int8"})
    monkeypatch.setattr(stream.step1, "_finish_cycle", lambda *args, **kwargs: pytest.fail("empty holdout must not enter the gate"))

    stream._finish_homogeneous(
        agent, decisions_stream=io.StringIO(), capacity_rows=[], folder=tmp_path,
        foundation_sha="unused", reason="route_change",
    )

    torch.testing.assert_close(shared, torch.zeros_like(shared))
    torch.testing.assert_close(local, torch.ones_like(local))
    assert agent.cycle_state is None
    assert agent.current_route is None


def test_phase_workers_domains_and_capacity_are_strict() -> None:
    assert stream._validate_phase("known") == "known"
    assert stream._validate_phase("unknown") == "unknown"
    with pytest.raises(ValueError, match="phase"):
        stream._validate_phase("formal")
    with pytest.raises(ValueError, match="one or two"):
        stream.run_cohort("known", "unused", selection_path="missing.json", workers=3)


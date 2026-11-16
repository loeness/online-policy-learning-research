"""Synthetic checks for H037 candidate commits and paired restore state.

These checks use small in-memory actors and monkeypatched candidate results.
They do not instantiate or step a task environment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from online_rl import h037_candidate, h037_integration
from online_rl.h037_evidence import H037Evidence
from online_rl.h037_integration import H037Controller
from online_rl.policy import OBS_DIM, TreePolicy
from online_rl.write_decision_policy import FlexibleSharedPolicy, encode_actor
from scripts import run_h037_initial as h037_runner


def _controller_config() -> dict[str, Any]:
    return {
        "format": "h037_controller_v1",
        "lambda_old": 1.0,
        "ridge": 1.0e-4,
        "max_iter": 1,
        "max_eval": 4,
        "delta_v": 1.0e-3,
        "epsilon_old": 1.0,
        "epsilon_res": 1.0,
        "compensation_atol": 1.0e-6,
        "codec": "float32",
        "actor_budget_bytes": 512,
        "holdout_fraction": 0.25,
        "min_victim_train_rows": 1,
        "min_victim_holdout_rows": 1,
        "min_old_train_rows": 1,
        "min_old_holdout_rows": 1,
        "min_resident_holdout_rows": 1,
    }


class _Slot(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.full((3, 3, 3), value, dtype=torch.float32))


class _Shared(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.C = nn.Parameter(torch.full((3, 3), value, dtype=torch.float32))


class _Core(nn.Module):
    """Small torch module exposing the actor methods H037Controller uses."""

    def __init__(self) -> None:
        super().__init__()
        self._nodes = nn.ModuleDict({"1": _Slot(0.2), "2": _Slot(-0.3)})
        self._shared_coefficients = nn.ModuleList([_Shared(0.05)])
        self.register_buffer("slot_groups", torch.tensor([0, 0], dtype=torch.long))
        self.shared_groups = 1
        self._shared_frozen = False

    def group_for_slot(self, slot: int) -> int:
        return int(self.slot_groups[slot].item())

    def read_local_matrix(self) -> np.ndarray:
        return np.stack(
            [self._nodes[str(slot + 1)].A.detach().cpu().numpy().reshape(27) for slot in range(2)]
        ).copy()

    def install_local_matrix(self, matrix: np.ndarray) -> None:
        values = np.asarray(matrix)
        if values.shape != (2, 27):
            raise ValueError("local matrix shape mismatch")
        with torch.no_grad():
            for slot in range(2):
                self._nodes[str(slot + 1)].A.copy_(
                    torch.as_tensor(values[slot], dtype=torch.float32).reshape(3, 3, 3)
                )

    def read_shared_matrix(self) -> np.ndarray:
        return np.stack(
            [group.C.detach().cpu().numpy().reshape(9) for group in self._shared_coefficients]
        ).copy()

    def install_shared_matrix(self, matrix: np.ndarray) -> None:
        values = np.asarray(matrix)
        if values.shape != (1, 9):
            raise ValueError("shared matrix shape mismatch")
        with torch.no_grad():
            self._shared_coefficients[0].C.copy_(torch.as_tensor(values[0], dtype=torch.float32).reshape(3, 3))

    def shared_parameters(self):
        return tuple(group.C for group in self._shared_coefficients)

    def freeze_shared(self, freeze: bool = True) -> None:
        self._shared_frozen = bool(freeze)
        for parameter in self.shared_parameters():
            parameter.requires_grad_(not self._shared_frozen)


def _optimizer_with_state(core: _Core) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(core.parameters(), lr=1.0e-3)
    for parameter in core.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.full_like(parameter, 2.0),
        }
    optimizer.step_count = 7
    return optimizer


def _observations(slot: int, base: float) -> np.ndarray:
    rows = np.zeros((4, 13), dtype=np.float32)
    rows[:, 0] = np.asarray([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
    rows[:, 1] = base + np.arange(4, dtype=np.float32)
    rows[:, -1] = np.float32(slot + 1)
    return rows


class _Cache:
    """Small stateful cache stub with the controller's resident-slot API."""

    def __init__(self) -> None:
        self.slot_to_task = ["victim", "resident"]
        self.task_to_slot = {"victim": 0, "resident": 1}
        self.clock = 12
        self.rng = np.random.default_rng(72)

    def state_dict(self) -> dict[str, Any]:
        return {
            "slot_to_task": list(self.slot_to_task),
            "task_to_slot": dict(self.task_to_slot),
            "clock": self.clock,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "_Cache":
        cache = cls()
        cache.slot_to_task = list(state["slot_to_task"])
        cache.task_to_slot = dict(state["task_to_slot"])
        cache.clock = int(state["clock"])
        cache.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        return cache


def _agent_and_controller() -> tuple[Any, H037Controller]:
    core = _Core()
    agent = SimpleNamespace(
        core=core,
        optimizer=_optimizer_with_state(core),
        slots=2,
        parent_group=0,
        slot_group_tags=np.asarray([0, 0], dtype=np.int8),
        slot_generation=np.asarray([4, 9], dtype=np.int64),
        cache=_Cache(),
        action_rng=np.random.default_rng(314),
        behavior_rng=np.random.default_rng(271),
    )
    evidence = H037Evidence(capacity_per_slot=8, slots=2)
    evidence.record(0, 4, "victim", 0, 0, _observations(0, 10.0))
    evidence.record(1, 9, "resident", 0, 0, _observations(1, 20.0))
    controller = H037Controller(_controller_config(), evidence)
    # Establish the known parent identity before capturing rollback baselines.
    controller._sync_parent_identity(agent)
    return agent, controller


def _allocation() -> dict[str, Any]:
    return {"allocated": True, "slot": 0, "evicted_slot": 0, "evicted_task": "victim"}


def _encoded_candidate(agent: Any) -> bytes:
    local = agent.core.read_local_matrix()
    shared = agent.core.read_shared_matrix()
    groups = agent.core.slot_groups.detach().cpu().numpy().astype(np.int8, copy=True)
    new_shared = shared.copy()
    new_shared[0] += np.float32(0.15)
    new_local = local.copy()
    new_local[0] = 0.0
    new_local[1] = (new_local[1].reshape(3, 3, 3) - np.float32(0.15)).reshape(27)
    return encode_actor(new_local, new_shared, groups, codec="float32", budget_bytes=512).data


def _accepted_candidate(data: bytes) -> dict[str, Any]:
    return {
        "status": "accept",
        "accepted": True,
        "error": None,
        "reason": "synthetic_accept",
        "actor_bytes": data,
        "candidate_sha256": hashlib.sha256(data).hexdigest(),
        "metrics": {"synthetic": True},
    }


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def _agent_snapshot(agent: Any, controller: H037Controller) -> dict[str, Any]:
    return {
        "core": copy.deepcopy(agent.core.state_dict()),
        "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
        "optimizer_step_count": agent.optimizer.step_count,
        "cache": copy.deepcopy(agent.cache.state_dict()),
        "local_rngs": {
            "action": copy.deepcopy(agent.action_rng.bit_generator.state),
            "behavior": copy.deepcopy(agent.behavior_rng.bit_generator.state),
        },
        "evidence": copy.deepcopy(controller.evidence.state_dict()),
        "controller": copy.deepcopy(controller.state_dict()),
    }


def _rng_snapshot() -> tuple[Any, Any, torch.Tensor]:
    return copy.deepcopy(np.random.get_state()), copy.deepcopy(random.getstate()), torch.get_rng_state().clone()


def _assert_rollback_state(before: dict[str, Any], after: dict[str, Any]) -> None:
    assert _tree_equal(before["core"], after["core"])
    assert _tree_equal(before["optimizer"], after["optimizer"])
    assert before["optimizer_step_count"] == after["optimizer_step_count"]
    assert _tree_equal(before["cache"], after["cache"])
    assert _tree_equal(before["local_rngs"], after["local_rngs"])
    assert _tree_equal(before["evidence"], after["evidence"])


def test_pre_reset_commits_candidate_and_clears_affected_optimizer_state(monkeypatch: pytest.MonkeyPatch) -> None:
    agent, controller = _agent_and_controller()
    before_local = agent.core.read_local_matrix()
    before_shared = agent.core.read_shared_matrix()
    victim_parameter = agent.core._nodes["1"].A
    resident_parameter = agent.core._nodes["2"].A
    shared_parameter = agent.core.shared_parameters()[0]
    candidate_bytes = _encoded_candidate(agent)
    monkeypatch.setattr(h037_integration, "propose_h037", lambda _core, **_kwargs: _accepted_candidate(candidate_bytes))
    receipts: list[dict[str, Any]] = []

    def capacity_check(_agent: Any, **kwargs: Any) -> dict[str, Any]:
        receipts.append(kwargs)
        assert kwargs["controller_state"]["parent_version"] == 1
        assert kwargs["proposed_parent_version"] == 1
        return {
            "fits": True,
            "package_bytes": kwargs["actor_bytes"] + 100,
            "budget_bytes": 1000,
            "controller_state_included": True,
            "parent_version": kwargs["proposed_parent_version"],
        }

    event = controller.pre_reset(agent, _allocation(), capacity_check=capacity_check)

    assert event["status"] == "accept"
    assert event["parent_version_before"] == 0
    assert event["parent_version_after"] == 1
    assert len(receipts) == 1
    assert not np.array_equal(agent.core.read_shared_matrix(), before_shared)
    assert np.all(agent.core.read_local_matrix()[0] == 0.0)
    assert np.array_equal(agent.core.slot_groups.cpu().numpy(), np.asarray([0, 0]))
    assert not np.array_equal(agent.core.read_local_matrix()[1], before_local[1])
    assert victim_parameter in agent.optimizer.state
    assert resident_parameter not in agent.optimizer.state
    assert shared_parameter not in agent.optimizer.state
    assert controller.counters["accepted"] == 1


def test_pre_reset_pure_candidate_rejection_returns_reject_without_state_change(monkeypatch: pytest.MonkeyPatch) -> None:
    agent, controller = _agent_and_controller()
    before = _agent_snapshot(agent, controller)
    monkeypatch.setattr(
        h037_integration,
        "propose_h037",
        lambda _core, **_kwargs: {
            "status": "reject",
            "accepted": False,
            "error": None,
            "reason": "synthetic_gate_reject",
            "metrics": {},
        },
    )
    global_rng_before = _rng_snapshot()

    event = controller.pre_reset(
        agent,
        _allocation(),
        capacity_check=lambda *_args, **_kwargs: pytest.fail("capacity must not be checked after candidate rejection"),
    )

    after = _agent_snapshot(agent, controller)
    assert event["status"] == "reject"
    assert event["reason"] == "synthetic_gate_reject"
    _assert_rollback_state(before, after)
    assert _tree_equal(global_rng_before, _rng_snapshot())
    assert controller.counters["rejected"] == 1
    assert controller.eligible_attempt_used is True


def test_pre_reset_candidate_execution_error_is_not_downgraded_to_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, controller = _agent_and_controller()
    before = _agent_snapshot(agent, controller)
    monkeypatch.setattr(
        h037_integration,
        "propose_h037",
        lambda _core, **_kwargs: {
            "status": "error",
            "accepted": False,
            "error": {"type": "RuntimeError", "message": "synthetic learner failure"},
            "reason": "RuntimeError: synthetic learner failure",
            "metrics": {},
        },
    )

    event = controller.pre_reset(
        agent,
        _allocation(),
        capacity_check=lambda *_args, **_kwargs: pytest.fail(
            "capacity must not run after candidate execution error"
        ),
    )

    after = _agent_snapshot(agent, controller)
    assert event["status"] == "error"
    assert event["reason"] == "RuntimeError: synthetic learner failure"
    assert event["error"] == {
        "type": "RuntimeError",
        "message": "synthetic learner failure",
    }
    _assert_rollback_state(before, after)
    assert controller.counters["errors"] == 1
    assert controller.counters["rejected"] == 0


def test_pre_reset_capacity_rejection_rolls_back_candidate_and_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    agent, controller = _agent_and_controller()
    before = _agent_snapshot(agent, controller)
    before_shared = agent.core.read_shared_matrix()
    candidate_bytes = _encoded_candidate(agent)
    monkeypatch.setattr(h037_integration, "propose_h037", lambda _core, **_kwargs: _accepted_candidate(candidate_bytes))
    global_rng_before = _rng_snapshot()

    def capacity_check(_agent: Any, **kwargs: Any) -> dict[str, Any]:
        assert not np.array_equal(agent.core.read_shared_matrix(), before_shared)
        return {
            "fits": False,
            "package_bytes": 1200,
            "budget_bytes": 1000,
            "controller_state_included": True,
            "parent_version": kwargs["proposed_parent_version"],
        }

    event = controller.pre_reset(agent, _allocation(), capacity_check=capacity_check)

    after = _agent_snapshot(agent, controller)
    assert event["status"] == "skip"
    assert event["reason"] == "complete_package_exceeds_budget"
    _assert_rollback_state(before, after)
    assert before["controller"]["parent_version"] == after["controller"]["parent_version"] == 0
    assert _tree_equal(global_rng_before, _rng_snapshot())
    assert controller.counters["skipped"] == 1


@pytest.mark.parametrize("side_effect_stage", ["candidate", "capacity_check"])
def test_candidate_or_capacity_side_effect_is_error_and_fully_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
    side_effect_stage: str,
) -> None:
    agent, controller = _agent_and_controller()
    before = _agent_snapshot(agent, controller)
    candidate_bytes = _encoded_candidate(agent)
    global_rng_before = _rng_snapshot()

    def mutate_transactional_state() -> None:
        agent.core._nodes["1"].A.data.add_(0.5)
        agent.cache.clock += 1
        agent.cache.rng.random()
        agent.action_rng.random()
        agent.behavior_rng.random()
        agent.optimizer.step_count += 1
        np.random.random()
        random.random()
        torch.rand(1)

    if side_effect_stage == "candidate":
        def candidate_with_side_effect(_core: Any, **_kwargs: Any) -> dict[str, Any]:
            mutate_transactional_state()
            return {
                "status": "reject",
                "accepted": False,
                "error": None,
                "reason": "synthetic_gate_reject",
                "metrics": {},
            }

        monkeypatch.setattr(h037_integration, "propose_h037", candidate_with_side_effect)
        capacity_check = lambda *_args, **_kwargs: pytest.fail("capacity must not run after a candidate side effect")
    else:
        monkeypatch.setattr(
            h037_integration,
            "propose_h037",
            lambda _core, **_kwargs: _accepted_candidate(candidate_bytes),
        )

        def capacity_check(_agent: Any, **kwargs: Any) -> dict[str, Any]:
            mutate_transactional_state()
            return {
                "fits": True,
                "package_bytes": kwargs["actor_bytes"],
                "budget_bytes": 1000,
                "controller_state_included": True,
                "parent_version": kwargs["proposed_parent_version"],
            }

    event = controller.pre_reset(agent, _allocation(), capacity_check=capacity_check)

    after = _agent_snapshot(agent, controller)
    assert event["status"] == "error"
    assert "transactional state" in event["reason"]
    _assert_rollback_state(before, after)
    assert _tree_equal(global_rng_before, _rng_snapshot())
    assert controller.counters["errors"] == 1


def test_pre_reset_exception_restores_model_optimizer_evidence_and_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    agent, controller = _agent_and_controller()
    before = _agent_snapshot(agent, controller)
    evidence_before = copy.deepcopy(controller.evidence.state_dict())
    candidate_bytes = _encoded_candidate(agent)
    monkeypatch.setattr(h037_integration, "propose_h037", lambda _core, **_kwargs: _accepted_candidate(candidate_bytes))
    rng_before = _rng_snapshot()

    def capacity_check(_agent: Any, **_kwargs: Any) -> dict[str, Any]:
        agent.cache.clock += 1
        agent.cache.rng.random()
        agent.action_rng.random()
        agent.behavior_rng.random()
        np.random.random()
        random.random()
        torch.rand(3)
        raise RuntimeError("synthetic capacity callback failure")

    event = controller.pre_reset(agent, _allocation(), capacity_check=capacity_check)

    after = _agent_snapshot(agent, controller)
    assert event["status"] == "error"
    assert "synthetic capacity callback failure" in event["reason"]
    _assert_rollback_state(before, after)
    assert _tree_equal(evidence_before, controller.evidence.state_dict())
    assert controller.parent_version == 0
    rng_after = _rng_snapshot()
    assert _tree_equal(rng_before, rng_after)


def _prepare_task_spy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    event: dict[str, Any],
    event_path: Path,
) -> tuple[Any, list[tuple[Any, ...]], object]:
    calls: list[tuple[Any, ...]] = []
    capacity_check = object()
    allocation = {
        "allocated": True,
        "slot": 0,
        "evicted_slot": 0,
        "evicted_task": "victim",
    }

    class CacheSpy:
        def resolve(self, task: str) -> dict[str, Any]:
            calls.append(("resolve", task))
            return dict(allocation)

    class ControllerSpy:
        def pre_reset(
            self,
            _agent: Any,
            resolved: dict[str, Any],
            *,
            capacity_check: Any,
        ) -> dict[str, Any]:
            calls.append(("pre_reset", copy.deepcopy(resolved), capacity_check))
            return dict(event)

    class CoreSpy:
        def __init__(self) -> None:
            self.slot_groups = [0, 0]

        def group_for_slot(self, slot: int) -> int:
            return self.slot_groups[slot]

        def set_slot_group(self, slot: int, group: int, *, preserve_effective: bool) -> None:
            calls.append(("set_slot_group", slot, group, preserve_effective))
            self.slot_groups[slot] = group

    agent = SimpleNamespace(
        cache=CacheSpy(),
        h037_enabled=True,
        h037_controller=ControllerSpy(),
        h037_capacity_check=capacity_check,
        h037_event_path=event_path,
        core=CoreSpy(),
        slots=2,
        parent_group=0,
        slot_group_tags=np.asarray([0, 0], dtype=np.int8),
        runtime_counts={"slot_allocations": 2, "slot_evictions": 3, "prior_bindings": 4},
        counts={"slot_evictions": 5},
    )

    def reset_allocated_slot(_agent: Any, slot: int, *, reason: str) -> None:
        calls.append(("reset", slot, reason))

    monkeypatch.setattr(h037_runner.ws, "_reset_allocated_slot", reset_allocated_slot)

    return agent, calls, capacity_check


def test_prepare_task_ordinary_gate_rejection_keeps_the_single_lru_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "reject.jsonl"
    event = {
        "format": "h037_event_v1",
        "status": "reject",
        "reason": "synthetic_gate_reject",
    }
    agent, calls, capacity_check = _prepare_task_spy(
        monkeypatch,
        event=event,
        event_path=event_path,
    )

    slot, returned_allocation = h037_runner._prepare_task(agent, "new-task", 0)

    assert slot == 0
    assert [entry[0] for entry in calls] == ["resolve", "pre_reset", "reset", "set_slot_group"]
    assert calls[0] == ("resolve", "new-task")
    assert calls[1][1] == {
        "allocated": True,
        "slot": 0,
        "evicted_slot": 0,
        "evicted_task": "victim",
    }
    assert calls[1][2] is capacity_check
    assert calls[2] == ("reset", 0, "known_task_lru_allocation")
    assert sum(entry[0] == "reset" for entry in calls) == 1
    assert json.loads(event_path.read_text(encoding="utf-8")) == event
    assert returned_allocation["inherited_parent"] is True
    assert returned_allocation["physical_group_at_start"] == 0
    assert returned_allocation["parent_group_at_start"] == 0
    assert agent.runtime_counts == {"slot_allocations": 3, "slot_evictions": 4, "prior_bindings": 5}
    assert agent.counts == {"slot_evictions": 6}
    assert agent.current_known_slot == 0
    assert agent.core.slot_groups == [0, 0]


@pytest.mark.parametrize(
    ("event", "expected_reason"),
    [
        (
            {
                "format": "h037_event_v1",
                "status": "error",
                "reason": "RuntimeError: synthetic execution failure",
            },
            "RuntimeError: synthetic execution failure",
        ),
        (
            {
                "format": "h037_event_v1",
                "status": "skip",
                "reason": "complete_package_exceeds_budget",
            },
            "complete_package_exceeds_budget",
        ),
    ],
    ids=["candidate-error", "package-capacity"],
)
def test_prepare_task_persists_stopping_event_and_stops_before_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    event: dict[str, Any],
    expected_reason: str,
) -> None:
    event_path = tmp_path / f"{event['status']}.jsonl"
    agent, calls, _capacity_check = _prepare_task_spy(
        monkeypatch,
        event=event,
        event_path=event_path,
    )
    runtime_counts_before = copy.deepcopy(agent.runtime_counts)
    counts_before = copy.deepcopy(agent.counts)
    fsync_calls: list[int] = []
    real_fsync = h037_runner.os.fsync

    def fsync_spy(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(h037_runner.os, "fsync", fsync_spy)
    interactions: list[str] = []

    def prepare_then_interact() -> None:
        h037_runner._prepare_task(agent, "new-task", 0)
        interactions.append("next_environment_step")

    with pytest.raises(h037_runner.H037StopError, match=expected_reason):
        prepare_then_interact()

    lines = event_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == event
    assert len(fsync_calls) == 1
    assert [entry[0] for entry in calls] == ["resolve", "pre_reset"]
    assert sum(entry[0] == "reset" for entry in calls) == 0
    assert not any(entry[0] == "set_slot_group" for entry in calls)
    assert interactions == []
    assert agent.runtime_counts == runtime_counts_before
    assert agent.counts == counts_before
    assert sum(entry[0] == "resolve" for entry in calls) == 1
    assert sum(entry[0] == "pre_reset" for entry in calls) == 1


class _Stateful:
    def __init__(self, state: Any) -> None:
        self.state = copy.deepcopy(state)

    def state_dict(self) -> Any:
        return copy.deepcopy(self.state)


class _StateFactory:
    @classmethod
    def from_state_dict(cls, state: Any) -> _Stateful:
        return _Stateful(state)


class _CheckpointOptimizer:
    def __init__(self) -> None:
        self._state = {"moment": torch.tensor([2.0, 3.0])}
        self.step_count = 5

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._state = copy.deepcopy(state)


class _PrefixAgent:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.core = _Core()
        self.optimizer = _CheckpointOptimizer()
        cache_rng = np.random.default_rng(seed + 300)
        self.cache = _Stateful({
            "slot_to_task": ["task-a", "task-b"],
            "task_to_slot": {"task-a": 0, "task-b": 1},
            "clock": 8,
            "rng_state": copy.deepcopy(cache_rng.bit_generator.state),
        })
        self.evidence = _Stateful({"rows": [[1, 2], [3, 4]]})
        self.counts = {"updates": 6}
        self.runtime_counts = {"steps": 31}
        self.slot_generation = np.asarray([2, 7], dtype=np.int64)
        self.fitted_steps = np.asarray([11, 13], dtype=np.uint64)
        self.recent_groups = [0, 1]
        self.parent_group = 0
        self.slot_group_tags = np.asarray([0, 1], dtype=np.int8)
        self.action_rng = np.random.default_rng(seed + 100)
        self.behavior_rng = np.random.default_rng(seed + 200)
        self.h037_controller = H037Controller(_controller_config(), H037Evidence(2, slots=2))
        self.slots = 2
        self.profiles = {"main": np.asarray([0.25, 0.75], dtype=np.float32)}
        self.profile_reasons = ["prefix-profile"]
        self.grouped_generation = np.asarray([2, 7], dtype=np.int64)
        self.group_attempt_support = np.asarray([4, 5], dtype=np.int64)
        self.windows = [_Stateful({"slot": 0, "rows": [1]}), _Stateful({"slot": 1, "rows": [2]})]
        self.grouper = _Stateful({"groups": [0, 1], "rng": [17, 19]})
        self.anchors = _Stateful({"anchors": ["a", "b"]})
        self.current_known_slot = 1
        self.reward_by_task = {"task-a": 1.5}
        self.steps_by_task = {"task-a": 32}
        self.observed_reward_by_task = {"task-a": 1.25}
        self.observed_steps_by_task = {"task-a": 30}
        self.goal_episodes_by_task = {"task-a": 2}
        self.complete_episodes_by_task = {"task-a": 3}
        self.positive_complete_episodes_by_task = {"task-a": 2}
        self.next_episode_id = 9
        self.decision_cpu_s = 0.125
        self.evaluation_cpu_s = 0.25


class _FakeSimulator:
    def __init__(self, state: dict[str, Any]) -> None:
        self.episode = int(state["episode"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


def test_runner_restores_both_branches_from_common_prefix_with_matching_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h037_runner.sd, "_new_agent", lambda seed, _base, _config: _PrefixAgent(seed))
    monkeypatch.setattr(h037_runner.ws, "BoundedTaskCache", _StateFactory)
    monkeypatch.setattr(h037_runner.sd, "BoundedGroupEvidence", _StateFactory)
    monkeypatch.setattr(h037_runner.ws, "EvidenceWindow", _StateFactory)
    monkeypatch.setattr(h037_runner.ws, "RewardAwareGroups", _StateFactory)
    monkeypatch.setattr(h037_runner.ws, "AnchorBank", _StateFactory)

    random.seed(843)
    np.random.seed(843)
    torch.manual_seed(843)
    prefix_agent = _PrefixAgent(23)
    for slot, generation, task, group, markers, versions in (
        (0, 2, "task-a", 0, [31.0, 32.0, 33.0], [0, 1, 0]),
        (1, 7, "task-b", 1, [71.0, 72.0, 73.0], [1, 1, 0]),
    ):
        for marker, version in zip(markers, versions):
            prefix_agent.h037_controller.evidence.record(
                slot,
                generation,
                task,
                group,
                version,
                _observations(slot, marker)[:1],
            )
    expected_evidence = {
        (0, 2, "task-a", 0): prefix_agent.h037_controller.evidence.get(0, 2, "task-a", 0),
        (1, 7, "task-b", 1): prefix_agent.h037_controller.evidence.get(1, 7, "task-b", 1),
    }
    expected_markers = {
        (0, 2, "task-a", 0): [32.0, 33.0],
        (1, 7, "task-b", 1): [72.0, 73.0],
    }
    expected_versions = {
        (0, 2, "task-a", 0): [1, 0],
        (1, 7, "task-b", 1): [1, 0],
    }
    for identity, batch in expected_evidence.items():
        expected_identity = {
            "generation": identity[1],
            "task": identity[2],
            "group": identity[3],
        }
        assert batch["identity"] == expected_identity
        assert len(batch["observations"]) == 2
        np.testing.assert_array_equal(batch["observations"][:, 1], expected_markers[identity])
        np.testing.assert_array_equal(batch["parent_versions"], expected_versions[identity])
    foundation_sha = "f" * 64
    prefix_payload = h037_runner._bundle(prefix_agent, foundation_sha)

    prefix_rng = np.random.default_rng(404)
    prefix_simulator = {
        "environment": {"episode": 3, "rng": copy.deepcopy(prefix_rng.bit_generator.state)},
        "driver": {"obs": [0.1, 0.2, 0.3], "phase": "prefix", "step": 17},
    }

    def capture_simulator(env: _FakeSimulator, runtime: dict[str, Any]) -> dict[str, Any]:
        return {
            "environment": {"episode": env.episode, "rng": copy.deepcopy(env.rng.bit_generator.state)},
            "driver": copy.deepcopy(runtime),
        }

    def restore_simulator(snapshot: dict[str, Any]) -> tuple[_FakeSimulator, dict[str, Any]]:
        return _FakeSimulator(snapshot["environment"]), copy.deepcopy(snapshot["driver"])

    monkeypatch.setattr(h037_runner.sd, "_capture_simulator", capture_simulator)
    monkeypatch.setattr(h037_runner.sd, "_restore_simulator", restore_simulator)
    draws_by_branch = []

    for _branch in ("lru", "handoff"):
        restored = h037_runner.sd._restore_learner(prefix_payload, {}, {})
        restored.h037_controller = H037Controller.from_state_dict(prefix_payload["h037_controller"])
        h037_runner._restore_stream_aux(restored, prefix_payload)
        assert h037_runner._same_state(h037_runner._bundle(restored, foundation_sha), prefix_payload)
        for identity, expected_batch in expected_evidence.items():
            restored_batch = restored.h037_controller.evidence.get(*identity)
            assert restored_batch["identity"] == expected_batch["identity"]
            assert len(restored_batch["observations"]) == 2
            np.testing.assert_array_equal(
                restored_batch["observations"], expected_batch["observations"]
            )
            np.testing.assert_array_equal(
                restored_batch["observations"][:, 1], expected_markers[identity]
            )
            np.testing.assert_array_equal(
                restored_batch["parent_versions"], expected_batch["parent_versions"]
            )
            np.testing.assert_array_equal(
                restored_batch["parent_versions"], expected_versions[identity]
            )

        branch_env, branch_runtime = h037_runner.sd._restore_simulator(prefix_simulator)
        assert h037_runner._same_state(
            h037_runner.sd._capture_simulator(branch_env, branch_runtime), prefix_simulator
        )
        draws_by_branch.append(
            (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
                float(restored.action_rng.random()),
                float(restored.behavior_rng.random()),
                float(branch_env.rng.random()),
            )
        )

    assert h037_runner._same_state(draws_by_branch[0], draws_by_branch[1])


def test_h037_teacher_matches_real_flexible_policy_across_bands_and_routes() -> None:
    torch_rng_before = torch.get_rng_state().clone()
    torch.manual_seed(621)
    reference = TreePolicy(
        observation_space=spaces.Box(
            low=np.asarray([-2.0] * 12 + [0.0], dtype=np.float32),
            high=np.asarray([2.0] * 12 + [8.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(4),
        lr_schedule=lambda _: 1.0e-2,
    )
    policy = FlexibleSharedPolicy.from_reference(
        reference,
        local_slots=2,
        shared_groups=1,
        exploration_epsilon=0.05,
        temperature=2.0,
        train_critic=False,
    )
    torch.set_rng_state(torch_rng_before)
    policy.eval()

    with torch.no_grad():
        policy.source_basis.zero_()
        policy.source_basis[0, -1] = 1.0

    local = np.zeros((2, 27), dtype=np.float32)
    local_by_band = np.asarray(
        [
            [[0.30, 0.02, 0.00], [-0.12, 0.01, 0.00], [0.08, 0.00, 0.02]],
            [[-0.25, 0.00, 0.02], [0.18, 0.01, 0.00], [-0.06, 0.02, 0.00]],
            [[0.45, 0.01, 0.00], [-0.20, 0.00, 0.01], [0.11, 0.02, 0.00]],
        ],
        dtype=np.float32,
    )
    local[0] = local_by_band.reshape(27)
    local[1] = (local_by_band * np.float32(-0.4) + np.float32(0.03)).reshape(27)
    shared = np.asarray(
        [[0.17, -0.08, 0.04, -0.13, 0.11, 0.02, 0.07, -0.05, 0.16]],
        dtype=np.float32,
    )
    policy.install_local_matrix(local)
    policy.install_shared_matrix(shared)
    policy.set_slot_group(0, 0, preserve_effective=False)
    policy.set_slot_group(1, -1, preserve_effective=False)
    assert policy.group_for_slot(0) == 0
    assert policy.temperature == 2.0
    assert policy.epsilon == 0.05

    rng = np.random.default_rng(72)
    observations = rng.uniform(-0.75, 0.75, size=(6, OBS_DIM)).astype(np.float32)
    observations[:, 0] = np.asarray([0.1, 0.2, 0.4, 0.55, 0.7, 0.9], dtype=np.float32)
    observations[:, -1] = np.float32(1.0)
    bands = np.floor(np.clip(observations[:, 0], 0.0, 1.0) * 3.0).astype(np.int64).clip(0, 2)
    np.testing.assert_array_equal(bands, np.asarray([0, 0, 1, 1, 2, 2]))

    features = h037_candidate.extract_frozen_policy_features(policy, observations)
    teacher = h037_candidate._teacher_probabilities(
        features,
        local[0],
        shared[0],
        observations,
    )
    with torch.no_grad():
        policy_probs = policy.get_distribution(torch.as_tensor(observations)).distribution.probs
    policy_probs_np = policy_probs.detach().cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(teacher, policy_probs_np, rtol=2.0e-5, atol=2.0e-6)

    # The retained local=0 prior still uses its bound shared parent.
    local_zero = np.zeros(27, dtype=np.float32)
    policy.install_local_matrix(np.stack((local_zero, local[1])))
    assert policy.group_for_slot(0) == 0
    prior_features = h037_candidate.extract_frozen_policy_features(policy, observations)
    prior_teacher = h037_candidate._teacher_probabilities(
        prior_features,
        local_zero,
        shared[0],
        observations,
    )
    with torch.no_grad():
        bound_prior_probs = policy.get_distribution(torch.as_tensor(observations)).distribution.probs
    bound_prior_probs_np = bound_prior_probs.detach().cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(prior_teacher, bound_prior_probs_np, rtol=2.0e-5, atol=2.0e-6)

    policy.set_slot_group(0, -1, preserve_effective=False)
    with torch.no_grad():
        unbound_prior_probs = policy.get_distribution(torch.as_tensor(observations)).distribution.probs
    unbound_prior_probs_np = unbound_prior_probs.detach().cpu().numpy().astype(np.float64)
    assert float(np.max(np.abs(bound_prior_probs_np - unbound_prior_probs_np))) > 1.0e-5
    policy.set_slot_group(0, 0, preserve_effective=False)

    route_zero_observations = observations.copy()
    route_zero_observations[:, -1] = np.float32(0.0)
    route_zero_features = h037_candidate.extract_frozen_policy_features(policy, route_zero_observations)
    route_zero_teacher = h037_candidate._teacher_probabilities(
        route_zero_features,
        local_zero,
        np.zeros(9, dtype=np.float32),
        route_zero_observations,
    )
    with torch.no_grad():
        route_zero_probs = policy.get_distribution(torch.as_tensor(route_zero_observations)).distribution.probs
    route_zero_probs_np = route_zero_probs.detach().cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(route_zero_teacher, route_zero_probs_np, rtol=2.0e-5, atol=2.0e-6)
    assert float(np.max(np.abs(bound_prior_probs_np - route_zero_probs_np))) > 1.0e-5

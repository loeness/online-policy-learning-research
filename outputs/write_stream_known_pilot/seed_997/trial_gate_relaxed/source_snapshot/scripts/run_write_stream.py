"""Run known-cache or task-agnostic unknown-boundary write streams.

Task names are maintained by the outer schedule for environment changes and
observer ledgers. The unknown-boundary learner receives only visible
observations, actions, rewards, next observations, and termination flags.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import torch

from online_rl.c1_supplement_experiment import _foundation, seed_everything
from online_rl.policy import LeafAdam
from online_rl.write_decision_envs import make_env, task_from_name, task_specs
from online_rl.write_decision_evidence import EvidenceWindow, RewardAwareGroups
from online_rl.write_decision_gate import AnchorBank
from online_rl.write_decision_policy import FlexibleSharedPolicy, decode_actor, encode_actor
from online_rl.write_decision_router import WriteDecisionRouter
from scripts import run_knowledge_loop as loop
from scripts import run_write_decision as step1


CONFIG_PATH = ROOT / "configs" / "write_stream.json"
PROTOCOL_PATH = ROOT / "docs" / "WRITE_STREAM_PROTOCOL.md"
TASKS = ("A0+", "A0-", "B0+", "B0-", "A1+", "A1-", "B1+", "B1-", "A2+", "A2-", "B2+", "B2-")
STREAM_SEQUENCE = TASKS + tuple(reversed(TASKS))
PROPOSED = ("trial_gate", "trial_gate_relaxed", "projected_gate", "local_promote")
BASE_CONDITIONS = ("reward_shared", "independent_fast")
EXTRA_CONDITIONS = ("independent6", "independent7_lru", "independent7_random", "independent7_lfu")
CONDITIONS = set((*PROPOSED, *BASE_CONDITIONS, *EXTRA_CONDITIONS))
INDEPENDENT_CONDITIONS = {"independent_fast", "independent6", "independent7_lru", "independent7_random", "independent7_lfu"}
ROLLOUT_STEPS = loop.ROLLOUT_STEPS
CYCLE_STEPS = 4 * ROLLOUT_STEPS
PARTIAL_MIN = 32
SOURCE_FILES = (
    Path("scripts/run_write_stream.py"),
    Path("scripts/run_write_decision.py"),
    Path("scripts/run_knowledge_loop.py"),
    Path("scripts/run_shared_write.py"),
    Path("src/online_rl/write_decision_envs.py"),
    Path("src/online_rl/write_decision_evidence.py"),
    Path("src/online_rl/write_decision_router.py"),
    Path("src/online_rl/write_decision_policy.py"),
    Path("src/online_rl/write_decision_gate.py"),
    Path("src/online_rl/shared_write_grouping.py"),
    Path("src/online_rl/shared_write_policy.py"),
    Path("src/online_rl/shared_write_storage.py"),
    Path("src/online_rl/c1_supplement_experiment.py"),
    Path("src/online_rl/c1_tensor_controls.py"),
    Path("src/online_rl/c1_policy.py"),
    Path("src/online_rl/compression.py"),
    Path("src/online_rl/c1_envs.py"),
    Path("src/online_rl/experiment.py"),
    Path("src/online_rl/policy.py"),
    Path("configs/write_stream.json"),
    Path("configs/write_decision.json"),
    Path("configs/c1.json"),
    Path("docs/WRITE_STREAM_PROTOCOL.md"),
    Path("docs/WRITE_DECISION_PROTOCOL.md"),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _write_jsonl(stream, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, default=_json_default) + "\n")
    stream.flush()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_persistent_bytes(foundation_bytes: int, checkpoint_bytes: int) -> int:
    """Count the W0 source together with its complete serialized learner state."""
    foundation_size, checkpoint_size = int(foundation_bytes), int(checkpoint_bytes)
    if foundation_size < 0 or checkpoint_size < 0:
        raise ValueError("persistent byte counts must be nonnegative")
    return foundation_size + checkpoint_size


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if tuple(config.get("base_sequence", ())) != TASKS or tuple(config.get("sequence", ())) != STREAM_SEQUENCE:
        raise ValueError("write-stream config must declare the 12-task base order and full 24-bout sequence")
    return config


def _load_selection(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != "write_decision_selection_v1":
        raise ValueError("selection must use write_decision_selection_v1")
    selected = payload.get("selected_conditions")
    if not isinstance(selected, list) or len(selected) != 2 or any(item not in PROPOSED for item in selected):
        raise ValueError("selection must contain exactly two Step 1 proposed conditions")
    return {"path": source, "payload": payload, "selected": list(selected)}


def _apply_foundation_override(base_config: Mapping[str, Any], run_config: Mapping[str, Any]) -> dict[str, Any]:
    """Use an explicitly requested W0 path while retaining the base dataset."""
    base = dict(base_config)
    override = run_config.get("foundation")
    if override is None:
        return base
    foundation_path = Path(override).resolve()
    if not foundation_path.is_file():
        raise FileNotFoundError(f"foundation checkpoint does not exist: {foundation_path}")
    base["foundation"] = str(foundation_path)
    return base


def resolve_conditions(selection: Mapping[str, Any], requested: Sequence[str] | None = None) -> list[str]:
    if requested:
        result: list[str] = []
        for name in requested:
            if name == "selected":
                result.extend(selection["selected"])
            elif name in CONDITIONS:
                result.append(str(name))
            else:
                raise ValueError(f"unknown stream condition: {name}")
        if not result:
            raise ValueError("conditions must not be empty")
        return list(dict.fromkeys(result))
    return list(dict.fromkeys([*selection["selected"], *BASE_CONDITIONS, *EXTRA_CONDITIONS]))


def _validate_phase(phase: str) -> str:
    if phase not in ("known", "unknown"):
        raise ValueError("phase must be 'known' or 'unknown'")
    return phase


def _validate_local_slot_conditions(conditions: Sequence[str], slots: int | None) -> None:
    if slots == 7:
        incompatible = [name for name in conditions if name not in INDEPENDENT_CONDITIONS]
        if incompatible:
            raise ValueError(
                "seven local slots fit the 236-byte actor cap only for independent G0 conditions; "
                f"incompatible conditions: {incompatible}"
            )


def stream_task_order() -> tuple[str, ...]:
    return STREAM_SEQUENCE


class BoundedTaskCache:
    """Bounded outer-harness task-to-slot cache with measurable eviction."""

    def __init__(self, capacity: int = 6, eviction: str = "lru", seed: int = 0) -> None:
        if capacity not in (6, 7):
            raise ValueError("cache capacity must be six or seven")
        if eviction not in ("lru", "lfu", "random"):
            raise ValueError("eviction must be lru, lfu, or random")
        self.capacity = int(capacity)
        self.eviction = eviction
        self.task_to_slot: dict[str, int] = {}
        self.slot_to_task: list[str | None] = [None] * capacity
        self.last_used = np.zeros(capacity, dtype=np.int64)
        self.frequency = np.zeros(capacity, dtype=np.uint64)
        self.clock = 0
        self.rng = np.random.default_rng(int(seed))

    @property
    def nbytes(self) -> int:
        return int(self.last_used.nbytes + self.frequency.nbytes)

    def resolve(self, task_name: str) -> dict[str, Any]:
        if task_name not in TASKS:
            raise ValueError(f"unknown task name: {task_name}")
        self.clock += 1
        if task_name in self.task_to_slot:
            slot = self.task_to_slot[task_name]
            self.last_used[slot] = self.clock
            self.frequency[slot] += np.uint64(1)
            return {"slot": slot, "allocated": False, "evicted_slot": None, "evicted_task": None}

        available = [slot for slot, resident in enumerate(self.slot_to_task) if resident is None]
        evicted_slot = None
        evicted_task = None
        if available:
            slot = available[0]
        elif self.eviction == "lru":
            slot = min(range(self.capacity), key=lambda item: (int(self.last_used[item]), item))
        elif self.eviction == "lfu":
            slot = min(range(self.capacity), key=lambda item: (int(self.frequency[item]), int(self.last_used[item]), item))
        else:
            slot = int(self.rng.integers(self.capacity))
        if self.slot_to_task[slot] is not None:
            evicted_slot, evicted_task = slot, self.slot_to_task[slot]
            self.task_to_slot.pop(str(evicted_task), None)
        self.slot_to_task[slot] = task_name
        self.task_to_slot[task_name] = slot
        self.last_used[slot] = self.clock
        self.frequency[slot] = np.uint64(1)
        return {"slot": slot, "allocated": True, "evicted_slot": evicted_slot, "evicted_task": evicted_task}

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "eviction": self.eviction,
            "task_to_slot": dict(self.task_to_slot),
            "slot_to_task": list(self.slot_to_task),
            "last_used": self.last_used.copy(),
            "frequency": self.frequency.copy(),
            "clock": self.clock,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BoundedTaskCache":
        cache = cls(int(state["capacity"]), str(state["eviction"]))
        last_used, frequency = np.asarray(state["last_used"]), np.asarray(state["frequency"])
        if last_used.shape != (cache.capacity,) or last_used.dtype != np.int64:
            raise ValueError("cache last_used has invalid shape or dtype")
        if frequency.shape != (cache.capacity,) or frequency.dtype != np.uint64:
            raise ValueError("cache frequency has invalid shape or dtype")
        slots = list(state["slot_to_task"])
        mapping = dict(state["task_to_slot"])
        if len(slots) != cache.capacity or len(mapping) > cache.capacity:
            raise ValueError("cache mapping exceeds configured capacity")
        for slot, task in enumerate(slots):
            if task is not None and mapping.get(task) != slot:
                raise ValueError("cache forward and reverse mappings disagree")
        cache.last_used[...] = last_used
        cache.frequency[...] = frequency
        cache.slot_to_task = slots
        cache.task_to_slot = mapping
        cache.clock = int(state["clock"])
        cache.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        if "nbytes" in state and int(state["nbytes"]) != cache.nbytes:
            raise ValueError("cache nbytes does not match array storage")
        return cache


def unknown_behavior_probs(base_probabilities: object, uniform_mix: float = 0.5) -> np.ndarray:
    """Mix the base policy with the configured uniform exploration mass."""
    base = np.asarray(base_probabilities, dtype=np.float64)
    if base.shape != (4,) or not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError("base probabilities must be a finite four-action distribution")
    if not np.isclose(float(base.sum()), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("base probabilities must sum to one")
    if not 0.0 <= float(uniform_mix) <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1]")
    mixed = (1.0 - float(uniform_mix)) * base + float(uniform_mix) * 0.25
    return mixed.astype(np.float32)


def _new_agent(
    condition: str,
    seed: int,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
):
    """Build a Step 1-compatible policy with six or seven local routes."""
    torch.set_num_threads(1)
    condition = str(condition)
    if condition.startswith("independent7_"):
        eviction = condition.removeprefix("independent7_")
        slots = int(config.get("local_slots") or 7)
        base_condition = "independent_fast"
    elif condition == "independent6":
        eviction = "lru"
        slots = int(config.get("local_slots") or 6)
        base_condition = "independent_fast"
    else:
        eviction = "lru"
        slots = int(config.get("local_slots") or 6)
        base_condition = condition
    if slots not in (6, 7):
        raise ValueError("local_slots override must be six or seven")
    if condition in INDEPENDENT_CONDITIONS:
        groups = 0
        local_lr = float(config["fast_local_learning_rate"])
    else:
        groups = 2
        local_lr = float(config["local_learning_rate"])
    if slots == 7 and groups == 2:
        raise ValueError("seven local slots with two shared groups exceed the 236-byte actor cap")

    seed_everything(int(seed))
    reference = _foundation(base_config, int(seed))
    try:
        core = FlexibleSharedPolicy.from_reference(
            reference.policy,
            local_slots=slots,
            shared_groups=groups,
            exploration_epsilon=float(base_config.get("exploration_epsilon", 0.05)),
            temperature=float(base_config.get("temperature", 2.0)),
            train_critic=True,
        )
    finally:
        reference.env.close()
    core._set_trainable_flags(train_critic=True)
    local_parameters = [core._nodes[str(slot + 1)].A for slot in range(slots)]
    shared_parameters = list(core.shared_parameters())
    actor_ids = {id(parameter) for parameter in (*local_parameters, *shared_parameters)}
    critic_parameters = [
        parameter for parameter in core.parameters()
        if parameter.requires_grad and id(parameter) not in actor_ids
    ]
    optimizer_groups: list[dict[str, Any]] = [{"params": local_parameters, "lr": local_lr, "name": "local"}]
    if shared_parameters:
        optimizer_groups.append({"params": shared_parameters, "lr": float(config["shared_learning_rate"]), "name": "shared"})
    optimizer_groups.append({"params": critic_parameters, "lr": float(config["critic_learning_rate"]), "name": "critic"})
    optimizer = LeafAdam(optimizer_groups, eps=1e-5)
    mode = "dynamics" if base_condition == "dynamics_shared" else "reward"
    anchor_seed = int(seed) + 779
    agent = SimpleNamespace(
        condition=base_condition,
        external_condition=condition,
        seed=int(seed),
        slots=slots,
        eviction=eviction,
        core=core,
        optimizer=optimizer,
        arm="write_stream",
        action_rng=np.random.default_rng(int(seed) + 1100),
        slot_generation=np.zeros(slots, dtype=np.int64),
        fitted_steps=np.zeros(slots, dtype=np.uint64),
        anchors=AnchorBank(slots=slots, per_slot=int(config["anchors_per_slot"]), seed=anchor_seed),
        grouper=RewardAwareGroups(
            max_groups=2, mode=mode,
            reward_min_samples=int(config.get("group_reward_min_samples", 16)),
            reward_support_aware=bool(config.get("reward_support_aware", True)),
        ),
        profiles={
            "movement_counts": np.zeros((slots, 3, 4, 4), dtype=np.uint16),
            "reward_xtx": np.zeros((slots, 3, 3), dtype=np.float32),
            "reward_xty": np.zeros((slots, 3), dtype=np.float32),
            "reward_n": np.zeros(slots, dtype=np.float32),
            "seen": np.zeros(slots, dtype=np.uint8),
            "shadow_group": np.full(slots, -1, dtype=np.int8),
        },
        profile_reasons=["not_profiled"] * slots,
        grouped_generation=np.full(slots, -1, dtype=np.int64),
        group_attempt_support=np.zeros(slots, dtype=np.uint64),
        windows=[EvidenceWindow(int(config["probe_steps"])) for _ in range(slots)],
        counts={
            "environment_steps": 0, "raw_steps": 0, "probe_steps": 0, "evaluation_steps": 0,
            "fit_steps": 0, "holdout_steps": 0, "rollouts_trained": 0, "minibatch_updates": 0,
            "stale_actor_rows": 0, "unknown_actor_rows": 0, "boundary_rows_excluded": 0,
            "partial_rollout_dropped": 0, "accepted_count": 0, "rejected_count": 0,
            "trial_rollback_count": 0, "provisional_accept_count": 0, "decision_count": 0,
            "direct_commit_count": 0, "completed_episodes": 0, "positive_completed_episodes": 0,
            "positive_goal_episodes": 0, "all_goal_episodes": 0, "slot_allocations": 0,
            "slot_evictions": 0, "route_switches": 0, "group_retries": 0,
            "group_abstentions": 0, "group_max_abstentions": 0,
            "mixed_task_episodes": 0, "cold_prior_bindings": 0,
            "effective_preserving_bindings": 0,
        },
        reward_by_task={name: 0.0 for name in TASKS},
        steps_by_task={name: 0 for name in TASKS},
        observed_reward_by_task={name: 0.0 for name in TASKS},
        observed_steps_by_task={name: 0 for name in TASKS},
        goal_episodes_by_task={name: 0 for name in TASKS},
        complete_episodes_by_task={name: 0 for name in TASKS},
        positive_complete_episodes_by_task={name: 0 for name in TASKS},
        next_episode_id=0,
        decision_cpu_s=0.0,
        evaluation_cpu_s=0.0,
        base_config=dict(base_config),
        run_config=dict(config),
        cache=None,
        router=None,
        current_route=None,
        rollout=[],
        rollout_phase=None,
        cycle_state=None,
        cycle_step=0,
        holdout_rows=[],
        event_buffer=[],
    )
    return agent


def _make_cache(agent, config: Mapping[str, Any]) -> BoundedTaskCache:
    return BoundedTaskCache(agent.slots, agent.eviction, int(agent.seed) + 8811)


def _actor_arrays(core: FlexibleSharedPolicy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        core.read_local_matrix(),
        core.read_shared_matrix(),
        core.slot_groups.detach().cpu().numpy().astype(np.int8, copy=True),
    )


def _clear_optimizer_state(agent, slot: int | None = None) -> None:
    parameters = []
    if slot is None:
        parameters.extend(agent.core._nodes[str(index + 1)].A for index in range(agent.slots))
        parameters.extend(agent.core.shared_parameters())
    else:
        parameters.append(agent.core._nodes[str(slot + 1)].A)
    for parameter in parameters:
        agent.optimizer.state.pop(parameter, None)


def _reset_allocated_slot(agent, slot: int, *, reason: str) -> int:
    """Reset a local generation only after its old homogeneous data is flushed."""
    with torch.no_grad():
        agent.core._nodes[str(slot + 1)].A.zero_()
    if agent.core.group_for_slot(slot) >= 0:
        agent.core.set_slot_group(slot, -1, preserve_effective=False)
    _clear_optimizer_state(agent, slot)
    agent.anchors.clear_slot(slot)
    agent.slot_generation[slot] += 1
    agent.fitted_steps[slot] = 0
    agent.profiles["movement_counts"][slot].fill(0)
    agent.profiles["reward_xtx"][slot].fill(0.0)
    agent.profiles["reward_xty"][slot].fill(0.0)
    agent.profiles["reward_n"][slot] = 0.0
    agent.profiles["seen"][slot] = 0
    agent.profiles["shadow_group"][slot] = -1
    agent.grouped_generation[slot] = -1
    agent.group_attempt_support[slot] = 0
    agent.profile_reasons[slot] = reason
    agent.windows[slot] = EvidenceWindow(int(agent.run_config["probe_steps"]))
    agent.counts["slot_allocations"] += 1
    return int(agent.slot_generation[slot])


def _store_profile(agent, slot: int, profile: object) -> None:
    if isinstance(profile, Mapping):
        movement = np.asarray(profile["movement_counts"], dtype=np.uint16)
        xtx = np.asarray(profile["reward_xtx"], dtype=np.float32)
        xty = np.asarray(profile["reward_xty"], dtype=np.float32)
        reward_n = np.float32(profile["reward_n"])
    else:
        movement, xtx, xty, reward_n = profile.movement_counts, profile.reward_xtx, profile.reward_xty, profile.reward_n
    agent.profiles["movement_counts"][slot] = movement
    agent.profiles["reward_xtx"][slot] = xtx
    agent.profiles["reward_xty"][slot] = xty
    agent.profiles["reward_n"][slot] = reward_n
    agent.profiles["seen"][slot] = 1


def _maybe_assign_group(agent, slot: int, profile: object, assignments_writer=None, *, bout: int = -1) -> dict[str, Any] | None:
    generation = int(agent.slot_generation[slot])
    if agent.grouped_generation[slot] == generation:
        return None
    _store_profile(agent, slot, profile)
    if agent.core.shared_groups == 0:
        agent.profile_reasons[slot] = "grouping_disabled"
        return {"group": -1, "reason": "grouping_disabled", "committed": False}
    cycle = agent.cycle_state
    if cycle is not None and (
        int(cycle.get("fit_rows", 0)) > 0 or agent.rollout or int(agent.cycle_step) > 0
    ):
        return {"group": _current_group(agent, slot), "reason": "deferred_until_cycle_boundary", "committed": False}
    movement = np.asarray(profile["movement_counts"] if isinstance(profile, Mapping) else profile.movement_counts)
    support = int(movement.sum(dtype=np.uint64))
    if support < int(agent.grouper.min_support):
        return None
    reward_n = int(profile["reward_n"] if isinstance(profile, Mapping) else profile.reward_n)
    evidence_size = support + reward_n
    if evidence_size < int(agent.group_attempt_support[slot]) + int(agent.run_config.get("group_retry_interval", 16)):
        return None
    agent.group_attempt_support[slot] = np.uint64(evidence_size)
    agent.counts["group_retries"] += 1
    assignment = agent.grouper.assign(profile, commit=False)
    if assignment.group < 0:
        agent.counts["group_abstentions"] += 1
        if assignment.reason == "max_groups_reached":
            agent.counts["group_max_abstentions"] += 1
        agent.profile_reasons[slot] = str(assignment.reason)
        if assignments_writer is not None:
            assignments_writer.writerow({
                "phase": agent.run_config["phase"], "bout": bout, "slot": slot,
                "generation": generation, "group": -1, "reason": assignment.reason,
                "profile_rows": float(profile["reward_n"] if isinstance(profile, Mapping) else profile.reward_n),
                "scores": json.dumps(assignment.scores, ensure_ascii=False), "committed": 0,
            })
        return {"group": -1, "reason": str(assignment.reason), "committed": False}
    committed = agent.grouper.assign(profile, commit=True)
    preserve_effective = bool(agent.fitted_steps[slot] > 0)
    binding = "preserve_effective" if preserve_effective else "cold_prior_binding"
    agent.core.set_slot_group(slot, int(committed.group), preserve_effective=preserve_effective)
    if preserve_effective:
        agent.counts["effective_preserving_bindings"] += 1
    else:
        agent.counts["cold_prior_bindings"] += 1
    agent.grouped_generation[slot] = generation
    agent.profiles["shadow_group"][slot] = int(committed.group)
    agent.profile_reasons[slot] = str(committed.reason)
    if assignments_writer is not None:
        assignments_writer.writerow({
            "phase": agent.run_config["phase"], "bout": bout, "slot": slot,
            "generation": generation, "group": int(committed.group), "reason": committed.reason,
            "binding": binding,
            "profile_rows": float(profile["reward_n"] if isinstance(profile, Mapping) else profile.reward_n),
            "scores": json.dumps(committed.scores, ensure_ascii=False), "committed": 1,
        })
    return {"group": int(committed.group), "reason": str(committed.reason), "binding": binding, "committed": True}


def _append_ledger(writer, *, kind: str, global_step: int, bout: int, task_name: str,
                   slot: int, generation: int, route_after: int, action: int, reward: float,
                   probabilities: np.ndarray, log_prob: float, obs: np.ndarray, next_obs: np.ndarray,
                   terminated: bool, truncated: bool, phase: str, excluded: bool = False) -> None:
    writer.writerow({
        "interaction_kind": kind, "raw_step": global_step, "bout": bout,
        "taskobserver": task_name, "slot_before": slot, "generation_before": generation,
        "active_after": route_after, "action": int(action), "reward": float(reward),
        "behavior_probs": json.dumps(np.asarray(probabilities).tolist(), separators=(",", ":")),
        "behavior_logp": float(log_prob), "obs": json.dumps(np.asarray(obs).tolist(), separators=(",", ":")),
        "next_obs": json.dumps(np.asarray(next_obs).tolist(), separators=(",", ":")),
        "terminated": int(terminated), "truncated": int(truncated), "phase": phase,
        "boundary_excluded": int(excluded),
    })


def _policy_probs(core: FlexibleSharedPolicy, observation: np.ndarray) -> tuple[np.ndarray, float]:
    tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32), dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        probabilities = core.get_distribution(tensor).distribution.probs.reshape(-1).cpu().numpy().astype(np.float32)
        value = float(core.predict_values(tensor).reshape(-1)[0].cpu().item())
    return probabilities, value


def _sample(probs: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
    action = int(rng.choice(4, p=np.asarray(probs, dtype=np.float64)))
    return action, math.log(max(float(probs[action]), 1e-12))


def _route_before_action(agent, obs: np.ndarray) -> dict[str, Any]:
    if agent.run_config["phase"] == "unknown":
        return agent.router.route(obs)
    task = agent.cache.slot_to_task[agent.current_known_slot] if agent.current_known_slot is not None else None
    del task
    if agent.current_known_slot is None:
        return {"slot": -1, "reason": "no_resident_task", "confidence": 0.0}
    return {"slot": int(agent.current_known_slot), "reason": "known_cache", "confidence": 1.0}


def _cycle_start(agent, slot: int, generation: int, task_observer: str) -> None:
    if slot >= 0:
        group = int(agent.core.group_for_slot(slot))
        c_start = agent.core._shared_coefficients[group].C.detach().cpu().numpy().copy() if group >= 0 else np.zeros((3, 3), dtype=np.float32)
        a_start = agent.core._nodes[str(slot + 1)].A.detach().cpu().numpy().copy()
    else:
        group = -1
        c_start = np.zeros((3, 3), dtype=np.float32)
        a_start = np.zeros((3, 3, 3), dtype=np.float32)
    agent.cycle_state = {
        "slot": slot, "generation": generation, "group": group,
        "task_name": task_observer if agent.run_config["phase"] == "known" else "unknown",
        "a_start": a_start, "c_start": c_start, "fit_band_counts": np.zeros(3, dtype=np.int64),
        "fit_rows": 0,
    }
    agent.cycle_step = 0
    agent.rollout = []
    agent.rollout_phase = None
    agent.holdout_rows = []


def _process_rollout(agent, *, decisions_stream, capacity_rows, folder: Path, foundation_sha: str,
                     assignments_writer=None, bout: int = -1, force: bool = False) -> None:
    rows = agent.rollout
    if not rows:
        return
    phase = agent.rollout_phase or ("fit" if agent.cycle_step <= 3 * ROLLOUT_STEPS else "holdout")
    agent.rollout = []
    agent.rollout_phase = None
    if len(rows) < PARTIAL_MIN and force:
        agent.counts["partial_rollout_dropped"] += len(rows)
        return
    if phase == "fit":
        update = loop._ppo_update(agent, rows)
        agent.counts["fit_steps"] += len(rows)
        if agent.cycle_state is not None:
            agent.cycle_state["fit_rows"] += len(rows)
        if agent.cycle_state is not None and int(agent.cycle_state["slot"]) >= 0:
            slot = int(agent.cycle_state["slot"])
            if int(update.get("updates", 0)) > 0:
                generation = int(agent.cycle_state["generation"])
                actor_rows = sum(
                    int(row.slot == slot and row.generation == generation)
                    for row in rows
                )
                agent.fitted_steps[slot] += np.uint64(actor_rows)
            for row in rows:
                band = min(int(np.floor(np.clip(float(row.obs[0]), 0.0, 1.0) * 3.0)), 2)
                agent.cycle_state["fit_band_counts"][band] += 1
    else:
        loop._finalize_gae(rows)
        agent.holdout_rows = list(rows)
        agent.counts["holdout_steps"] += len(rows)


def _save_stream_checkpoint(agent, path: Path, foundation_sha: str, capacity_rows: list[dict[str, Any]]) -> dict[str, int]:
    local, shared, groups = _actor_arrays(agent.core)
    actor_budget = (
        int(agent.run_config["independent7_actor_budget_bytes"])
        if agent.slots == 7 and agent.core.shared_groups == 0
        else int(agent.run_config["actor_budget_bytes"])
    )
    actor = encode_actor(local, shared, groups, codec="int8", budget_bytes=actor_budget)
    payload = {
        "format": "write_stream_agent_v1", "phase": agent.run_config["phase"],
        "condition": agent.external_condition, "seed": agent.seed,
        "run_config": dict(agent.run_config),
        "foundation_sha256": foundation_sha, "actor_binary": actor.data,
        "critic": loop._critic_state(agent.core), "optimizer": agent.optimizer.state_dict(),
        "router": agent.router.state_dict() if agent.router is not None else None,
        "cache": agent.cache.state_dict() if agent.cache is not None else None,
        "profiles": {name: array.copy() for name, array in agent.profiles.items()},
        "profile_reasons": list(agent.profile_reasons),
        "grouped_generation": agent.grouped_generation.copy(),
        "group_attempt_support": agent.group_attempt_support.copy(),
        "windows": [window.state_dict() for window in agent.windows] if agent.run_config["phase"] == "known" else None,
        "grouper": agent.grouper.state_dict(), "anchors": agent.anchors.state_dict(),
        "slot_generation": agent.slot_generation.copy(), "fitted_steps": agent.fitted_steps.copy(),
        "counts": dict(agent.counts),
        "reward_by_task": dict(agent.reward_by_task), "steps_by_task": dict(agent.steps_by_task),
        "observed_reward_by_task": dict(agent.observed_reward_by_task),
        "observed_steps_by_task": dict(agent.observed_steps_by_task),
        "goal_episodes_by_task": dict(agent.goal_episodes_by_task),
        "complete_episodes_by_task": dict(agent.complete_episodes_by_task),
        "positive_complete_episodes_by_task": dict(agent.positive_complete_episodes_by_task),
        "next_episode_id": int(agent.next_episode_id),
        "current_known_slot": agent.current_known_slot if hasattr(agent, "current_known_slot") else None,
        "decision_cpu_s": float(agent.decision_cpu_s), "evaluation_cpu_s": float(agent.evaluation_cpu_s),
        "optimizer_step_count": int(getattr(agent.optimizer, "step_count", 0)),
        "rng": {
            "python": random.getstate(), "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "action_generator": copy.deepcopy(agent.action_rng.bit_generator.state),
        },
        "shared_frozen": bool(agent.core._shared_frozen), "train_critic": bool(agent.core.train_critic),
        "recovery_scope": "Learner state only; the live environment trajectory is not restored.",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    foundation_bytes = Path(agent.base_config["foundation"]).stat().st_size
    cap = int(agent.run_config["package_budget_bytes"])
    full_bytes = full_persistent_bytes(foundation_bytes, len(data))
    if full_bytes > cap:
        raise ValueError(f"foundation plus checkpoint is {full_bytes} bytes, exceeding package cap {cap}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    result = {
        "step": int(agent.counts["environment_steps"]), "actor_bytes": actor.nbytes,
        "actor_budget_bytes": actor_budget, "checkpoint_bytes": len(data),
        "foundation_bytes": int(foundation_bytes), "full_persistent_bytes": full_bytes,
        "package_budget_bytes": cap, "profile_array_bytes": int(sum(array.nbytes for array in agent.profiles.values())),
        "fitted_steps_bytes": int(agent.fitted_steps.nbytes),
        "router_array_bytes": int(agent.router.nbytes) if agent.router is not None else 0,
        "cache_array_bytes": int(agent.cache.nbytes) if agent.cache is not None else 0,
        "grouper_array_bytes": int(agent.grouper.nbytes), "anchor_array_bytes": int(agent.anchors.nbytes),
        "optimizer_tensor_bytes": step1._optimizer_tensor_bytes(agent.optimizer.state_dict()),
        "process_rss_bytes": int(psutil.Process().memory_info().rss),
    }
    capacity_rows.append(result)
    return result


def restore_stream_checkpoint(path: str | Path, base_config: Mapping[str, Any] | None = None):
    """Restore complete learner state; the in-flight environment is not saved."""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "write_stream_agent_v1":
        raise ValueError("unsupported write-stream checkpoint format")
    if base_config is None:
        base = _apply_foundation_override(loop.load_base_config(), payload.get("run_config", {}))
    else:
        base = dict(base_config)
    foundation = Path(base["foundation"])
    if payload.get("foundation_sha256") != _sha(foundation):
        raise ValueError("checkpoint W0 foundation hash does not match")
    run_config = dict(payload.get("run_config", {}))
    if not run_config:
        raise ValueError("checkpoint is missing run configuration")
    agent = _new_agent(str(payload["condition"]), int(payload["seed"]), run_config, base)
    local, shared, groups = decode_actor(payload["actor_binary"])
    agent.core.install_local_matrix(local)
    agent.core.install_shared_matrix(shared)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(groups.astype(np.int64), dtype=torch.long))
    loop._load_critic_state(agent.core, payload["critic"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    agent.optimizer.step_count = int(payload.get("optimizer_step_count", 0))
    agent.router = WriteDecisionRouter.from_state_dict(payload["router"]) if payload.get("router") is not None else None
    agent.cache = BoundedTaskCache.from_state_dict(payload["cache"]) if payload.get("cache") is not None else None
    agent.grouper = RewardAwareGroups.from_state_dict(payload["grouper"])
    agent.anchors = AnchorBank.from_state_dict(payload["anchors"])
    for name, target in agent.profiles.items():
        value = np.asarray(payload["profiles"][name])
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"checkpoint profile {name} has an invalid shape or dtype")
        target[...] = value
    agent.profile_reasons = list(payload["profile_reasons"])
    agent.grouped_generation[...] = np.asarray(payload["grouped_generation"], dtype=np.int64)
    agent.group_attempt_support[...] = np.asarray(payload["group_attempt_support"], dtype=np.uint64)
    if payload.get("windows") is not None:
        restored_windows = [EvidenceWindow.from_state_dict(row) for row in payload["windows"]]
        if len(restored_windows) != agent.slots:
            raise ValueError("checkpoint evidence window count does not match slot capacity")
        agent.windows = restored_windows
    agent.slot_generation[...] = np.asarray(payload["slot_generation"], dtype=np.int64)
    fitted_steps = np.asarray(payload.get("fitted_steps", np.zeros(agent.slots, dtype=np.uint64)))
    if fitted_steps.shape != (agent.slots,) or fitted_steps.dtype != np.uint64:
        raise ValueError("checkpoint fitted_steps has an invalid shape or dtype")
    agent.fitted_steps[...] = fitted_steps
    agent.counts.update(payload["counts"])
    agent.reward_by_task.update(payload["reward_by_task"])
    agent.steps_by_task.update(payload["steps_by_task"])
    agent.observed_reward_by_task.update(payload.get("observed_reward_by_task", {}))
    agent.observed_steps_by_task.update(payload.get("observed_steps_by_task", {}))
    agent.goal_episodes_by_task.update(payload["goal_episodes_by_task"])
    agent.complete_episodes_by_task.update(payload["complete_episodes_by_task"])
    agent.positive_complete_episodes_by_task.update(payload.get("positive_complete_episodes_by_task", {}))
    agent.next_episode_id = int(payload["next_episode_id"])
    agent.current_known_slot = payload.get("current_known_slot")
    agent.decision_cpu_s = float(payload.get("decision_cpu_s", 0.0))
    agent.evaluation_cpu_s = float(payload.get("evaluation_cpu_s", 0.0))
    agent.action_rng.bit_generator.state = copy.deepcopy(payload["rng"]["action_generator"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy_global"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    agent.core._set_trainable_flags(train_critic=bool(payload["train_critic"]))
    agent.core.freeze_shared(bool(payload["shared_frozen"]))
    agent.current_route = None
    return agent


def _finish_homogeneous(agent, *, decisions_stream, capacity_rows, folder, foundation_sha, assignments_writer=None, bout=-1, reason="route_change") -> None:
    _process_rollout(
        agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
        folder=folder, foundation_sha=foundation_sha, assignments_writer=assignments_writer,
        bout=bout, force=True,
    )
    cycle = agent.cycle_state
    if cycle is not None and int(cycle["slot"]) >= 0 and int(cycle["fit_rows"]) > 0:
        direct_conditions = {"reward_shared", "dynamics_shared", "independent_fast"}
        if agent.condition in direct_conditions or agent.holdout_rows:
            decision = step1._finish_cycle(agent, cycle, agent.holdout_rows, agent.run_config)
        else:
            # A route boundary can interrupt a cycle before its fresh holdout.
            # Keep the effective policy as a local correction, but do not use
            # an empty/stale holdout to promote shared knowledge.
            slot = int(cycle["slot"])
            group = int(agent.core.group_for_slot(slot))
            if group >= 0:
                c_start = torch.as_tensor(
                    np.asarray(cycle["c_start"], dtype=np.float32).reshape(3, 3),
                    dtype=agent.core._shared_coefficients[group].C.dtype,
                    device=agent.core._shared_coefficients[group].C.device,
                )
                shared = agent.core._shared_coefficients[group].C
                with torch.no_grad():
                    delta = shared.detach().clone() - c_start
                    agent.core._nodes[str(slot + 1)].A.add_(delta.unsqueeze(0))
                    shared.copy_(c_start)
            decision = {
                "step": int(agent.counts["environment_steps"]),
                "condition": agent.condition, "slot": slot,
                "generation": int(cycle["generation"]), "group": group,
                "taskobserver": cycle["task_name"], "accepted": False,
                "reason": "partial_cycle_no_fresh_holdout",
                "current_policy_preserved": True,
                "gate": None,
            }
            decision.update(step1.roundtrip_actor(agent, agent.run_config))
        _clear_optimizer_state(agent)
        decision.update({"stream_reason": reason, "external_condition": agent.external_condition})
        _write_jsonl(decisions_stream, decision)
        _save_stream_checkpoint(
            agent,
            folder / "checkpoints" / f"step_{agent.counts['environment_steps']:08d}_slot_{cycle['slot']}_gen_{cycle['generation']}.pt",
            foundation_sha,
            capacity_rows,
        )
    elif cycle is not None and int(cycle["slot"]) < 0:
        _write_jsonl(decisions_stream, {
            "step": int(agent.counts["environment_steps"]), "condition": agent.external_condition,
            "slot": -1, "generation": -1, "reason": "unknown_route_has_no_actor_commit", "stream_reason": reason,
        })
    agent.cycle_state = None
    agent.cycle_step = 0
    agent.rollout = []
    agent.rollout_phase = None
    agent.holdout_rows = []
    agent.current_route = None


def _source_snapshot(folder: Path, base_config: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = folder / "source_snapshot"
    source_rows = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing source snapshot input: {relative}")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if _sha(source) != _sha(destination):
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        source_rows[relative.as_posix()] = {"sha256": _sha(destination), "bytes": destination.stat().st_size}
    selection_source = Path(selection["path"])
    shutil.copy2(selection_source, snapshot / "selection.json")
    foundation = Path(base_config["foundation"])
    dataset = Path(base_config["dataset"])
    manifest = {
        "sources": source_rows,
        "selection": {"sha256": _sha(snapshot / "selection.json"), "bytes": (snapshot / "selection.json").stat().st_size},
        "foundation": {"path": str(foundation), "sha256": _sha(foundation), "bytes": foundation.stat().st_size},
        "dataset": {"path": str(dataset), "sha256": _sha(dataset), "bytes": dataset.stat().st_size},
    }
    _write_json(snapshot / "manifest.json", manifest)
    return manifest


def _complete_episode(episode_writer, agent, *, kind, task_name, bout, episode_id,
                      episode_index, episode_return, episode_length, goal_reached,
                      preference, complete=True, start_task_name=None,
                      task_observers: Sequence[str] | None = None):
    observers = tuple(sorted(set(task_observers or (task_name,))))
    mixed_task_episode = len(observers) > 1
    positive_success = "" if mixed_task_episode else int(bool(goal_reached and preference > 0))
    episode_writer.writerow({
        "interaction_kind": kind, "taskobserver": task_name,
        "episode_start_taskobserver": start_task_name if start_task_name is not None else task_name,
        "episode_task_observers": json.dumps(observers, separators=(",", ":")),
        "mixed_task_episode": int(mixed_task_episode),
        "bout": bout,
        "episode_id": episode_id, "episode": episode_index, "return_value": float(episode_return),
        "goal_reached": int(goal_reached), "positive_success": positive_success,
        "length": int(episode_length), "complete": int(complete),
    })
    if complete and mixed_task_episode:
        agent.counts["mixed_task_episodes"] += 1
    if kind == "training":
        agent.counts["completed_episodes"] += int(complete)
        agent.counts["all_goal_episodes"] += int(bool(goal_reached) and bool(complete))
        if complete and not mixed_task_episode:
            agent.complete_episodes_by_task[task_name] += 1
        if preference > 0 and complete and not mixed_task_episode:
            agent.counts["positive_completed_episodes"] += 1
            agent.counts["positive_goal_episodes"] += int(bool(goal_reached))
            agent.goal_episodes_by_task[task_name] += int(bool(goal_reached))
            agent.positive_complete_episodes_by_task[task_name] += 1


def _probabilities_for(agent, slot: int, obs: np.ndarray, *, unknown: bool) -> tuple[np.ndarray, float]:
    policy_slot = slot if slot >= 0 else -1
    route_id = policy_slot + 1 if policy_slot >= 0 else 0
    policy_obs = loop.augment_observation(obs, route_id)
    base, value = _policy_probs(agent.core, policy_obs)
    if unknown and slot < 0:
        base = unknown_behavior_probs(base, float(agent.run_config["unknown_uniform_mix"]))
    return base, value


def _record_visible_event(agent, obs, action, reward, next_obs, terminated, *, task_name, probe):
    if agent.run_config["phase"] == "unknown":
        return agent.router.observe(obs, action, reward, next_obs, bool(terminated))
    slot = int(agent.current_known_slot)
    agent.windows[slot].observe(obs, action, reward, next_obs, bool(terminated))
    return {"allocated_slot": None, "evicted_slot": None, "active_slot": slot, "switched": False, "reason": "known_cache"}


def _current_group(agent, slot: int) -> int:
    return int(agent.core.group_for_slot(slot)) if 0 <= slot < agent.slots else -1


def _run_training_stream(agent, env, *, steps_writer, episodes_writer, route_writer,
                         assignments_writer, decisions_stream, capacity_rows, folder,
                         foundation_sha, bout_steps: int, task_sequence: Sequence[str]) -> dict[str, dict[str, Any]]:
    obs, _ = env.reset(seed=int(agent.seed) + 4100)
    episode_id = int(agent.next_episode_id)
    episode_index = 0
    episode_return, episode_length, episode_goal = 0.0, 0, False
    episode_has_training = False
    episode_start_task = task_sequence[0]
    episode_task_observers = {task_sequence[0]}
    active_spec = task_from_name(task_sequence[0])
    known_probe_left = 0
    known_probe_slot: int | None = None
    group_coverage = {
        name: {"steps": 0, "grouped_steps": 0, "unassigned_steps": 0,
               "group_full_steps": 0, "unassigned_when_full_steps": 0,
               "slots": set(), "groups": set()}
        for name in TASKS
    }

    for bout, task_name in enumerate(task_sequence):
        spec = task_from_name(task_name)
        active_spec = spec
        env.set_task(spec)
        bout_return = 0.0
        if agent.run_config["phase"] == "known":
            allocation = agent.cache.resolve(task_name)
            slot = int(allocation["slot"])
            if allocation["allocated"]:
                _finish_homogeneous(agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
                                    folder=folder, foundation_sha=foundation_sha,
                                    assignments_writer=assignments_writer, bout=bout, reason="known_cache_allocation")
                if allocation["evicted_slot"] is not None:
                    agent.counts["slot_evictions"] += 1
                _reset_allocated_slot(agent, slot, reason="known_cache_allocation")
                agent.current_known_slot = slot
                known_probe_slot = slot
                known_probe_left = int(agent.run_config["probe_steps"])
            else:
                if agent.current_known_slot is not None and int(agent.current_known_slot) != slot:
                    _finish_homogeneous(agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
                                        folder=folder, foundation_sha=foundation_sha,
                                        assignments_writer=assignments_writer, bout=bout, reason="known_cache_route_change")
                agent.current_known_slot = slot
                known_probe_slot = None
                known_probe_left = 0
        if agent.run_config["phase"] == "unknown":
            agent.current_known_slot = None

        training_steps_this_bout = 0
        raw_steps_this_bout = 0
        while training_steps_this_bout < bout_steps:
            raw_steps_this_bout += 1
            if raw_steps_this_bout > bout_steps + int(agent.run_config["probe_steps"]) + 1:
                raise RuntimeError("bout exceeded its training and paid-probe interaction budget")
            before = np.asarray(obs, dtype=np.float32).copy()
            route = _route_before_action(agent, before)
            route_slot = int(route["slot"])
            generation = int(agent.slot_generation[route_slot]) if route_slot >= 0 else -1
            route_key = (route_slot, generation)
            is_probe = known_probe_left > 0 and agent.run_config["phase"] == "known"
            observer_task = task_name if agent.run_config["phase"] == "known" else "unknown"
            if agent.current_route is not None and agent.current_route != route_key:
                _finish_homogeneous(agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
                                    folder=folder, foundation_sha=foundation_sha,
                                    assignments_writer=assignments_writer, bout=bout, reason="pre_action_route_change")
            if not is_probe and route_slot >= 0:
                profile = (
                    agent.windows[route_slot].profile()
                    if agent.run_config["phase"] == "known"
                    else agent.router.profile_for_slot(route_slot)
                )
                _maybe_assign_group(agent, route_slot, profile, assignments_writer, bout=bout)
            if not is_probe and agent.cycle_state is None:
                _cycle_start(agent, route_slot, generation, observer_task)
            if not is_probe:
                agent.current_route = route_key

            if is_probe:
                behavior_probs = np.full(4, 0.25, dtype=np.float32)
                action = int(agent.action_rng.integers(4))
                log_prob = math.log(0.25)
                _, value = _probabilities_for(agent, route_slot, before, unknown=False)
                phase_name = "uniform_probe"
            else:
                unknown = agent.run_config["phase"] == "unknown"
                behavior_probs, value = _probabilities_for(agent, route_slot, before, unknown=unknown)
                action, log_prob = _sample(behavior_probs, agent.action_rng)
                phase_name = "unknown_explore" if unknown and route_slot < 0 else "train"

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            event = _record_visible_event(agent, before, action, reward, next_obs, terminated,
                                          task_name=task_name, probe=is_probe)
            agent.counts["raw_steps"] += 1
            agent.observed_reward_by_task[task_name] += float(reward)
            agent.observed_steps_by_task[task_name] += 1
            episode_task_observers.add(task_name)
            if is_probe:
                agent.counts["probe_steps"] += 1
                known_probe_left -= 1
            else:
                agent.counts["environment_steps"] += 1
                agent.reward_by_task[task_name] += float(reward)
                agent.steps_by_task[task_name] += 1
                bout_return += float(reward)
                training_steps_this_bout += 1
                episode_has_training = True
            episode_return += float(reward)
            episode_length += 1
            episode_goal = episode_goal or bool(info.get("goal_reached", False))

            active_after = int(event.get("active_slot", -1))
            changed = bool(event.get("switched", False) or event.get("allocated_slot") is not None)
            if agent.run_config["phase"] == "known" and is_probe:
                _store_profile(agent, int(known_probe_slot), agent.windows[int(known_probe_slot)].profile())
                support = int(agent.profiles["movement_counts"][int(known_probe_slot)].sum(dtype=np.uint64))
                if known_probe_left == 0:
                    _maybe_assign_group(agent, int(known_probe_slot), agent.windows[int(known_probe_slot)].profile(),
                                        assignments_writer, bout=bout)
                elif support >= agent.grouper.min_support:
                    _maybe_assign_group(agent, int(known_probe_slot), agent.windows[int(known_probe_slot)].profile(),
                                        assignments_writer, bout=bout)
            elif agent.run_config["phase"] == "known" and not is_probe:
                _store_profile(agent, route_slot, agent.windows[route_slot].profile())
                _maybe_assign_group(agent, route_slot, agent.windows[route_slot].profile(), assignments_writer, bout=bout)
            elif (agent.run_config["phase"] == "unknown" and active_after >= 0
                  and event.get("allocated_slot") is None):
                profile = agent.router.profile_for_slot(active_after)
                _store_profile(agent, active_after, profile)
                _maybe_assign_group(agent, active_after, profile, assignments_writer, bout=bout)

            if event.get("allocated_slot") is not None:
                _finish_homogeneous(agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
                                    folder=folder, foundation_sha=foundation_sha,
                                    assignments_writer=assignments_writer, bout=bout, reason="unknown_router_allocation")
                allocated_slot = int(event["allocated_slot"])
                if event.get("evicted_slot") is not None:
                    agent.counts["slot_evictions"] += 1
                _reset_allocated_slot(agent, allocated_slot, reason="unknown_router_allocation")
                # The router profile persists across local-head reset; it has no task labels.
                profile = agent.router.profile_for_slot(allocated_slot)
                _store_profile(agent, allocated_slot, profile)
                _maybe_assign_group(agent, allocated_slot, profile, assignments_writer, bout=bout)
            elif changed:
                _finish_homogeneous(agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
                                    folder=folder, foundation_sha=foundation_sha,
                                    assignments_writer=assignments_writer, bout=bout, reason=str(event.get("reason", "route_change")))

            switched = changed
            if switched:
                agent.counts["route_switches"] += 1
                agent.counts["boundary_rows_excluded"] += int(not is_probe)

            if not is_probe:
                metric_slot = route_slot if not switched else -1
                metric_group = _current_group(agent, metric_slot)
                coverage = group_coverage[task_name]
                coverage["steps"] += 1
                if metric_slot >= 0:
                    coverage["slots"].add(metric_slot)
                if metric_group >= 0:
                    coverage["grouped_steps"] += 1
                    coverage["groups"].add(metric_group)
                else:
                    coverage["unassigned_steps"] += 1
                group_capacity = int(agent.core.shared_groups)
                groups_full = group_capacity > 0 and len(agent.grouper.active_groups) >= group_capacity
                if groups_full:
                    coverage["group_full_steps"] += 1
                    if metric_group < 0:
                        coverage["unassigned_when_full_steps"] += 1

            if not is_probe and not switched:
                next_route = _route_before_action(agent, np.asarray(next_obs, dtype=np.float32))
                next_slot = int(next_route["slot"])
                next_route_id = next_slot + 1 if next_slot >= 0 else 0
                next_policy_obs = loop.augment_observation(next_obs, next_route_id)
                with torch.no_grad():
                    next_value = float(agent.core.predict_values(torch.as_tensor(next_policy_obs, dtype=torch.float32).reshape(1, -1)).reshape(-1)[0].cpu().item())
                if done and terminated:
                    next_value = 0.0
                if agent.cycle_state is None:
                    _cycle_start(agent, route_slot, generation, observer_task)
                rollout_position = int(agent.cycle_step)
                rollout_phase = "fit" if rollout_position < 3 * ROLLOUT_STEPS else "holdout"
                if agent.rollout_phase is None:
                    agent.rollout_phase = rollout_phase
                transition = loop.Transition(
                    loop.augment_observation(before, route_slot + 1 if route_slot >= 0 else 0),
                    int(action), float(reward), float(value), float(next_value), float(log_prob),
                    np.asarray(behavior_probs, dtype=np.float32).copy(), done, bool(terminated),
                    route_slot, generation, int(agent.counts["environment_steps"]),
                )
                transition.episode_id = episode_id
                agent.rollout.append(transition)
                if route_slot < 0:
                    agent.counts["unknown_actor_rows"] += 1
                agent.cycle_step += 1
                if len(agent.rollout) >= ROLLOUT_STEPS:
                    _process_rollout(agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
                                     folder=folder, foundation_sha=foundation_sha,
                                     assignments_writer=assignments_writer, bout=bout)
                if agent.cycle_step >= CYCLE_STEPS:
                    cycle = agent.cycle_state
                    if cycle is not None and int(cycle["slot"]) >= 0 and int(cycle["fit_rows"]) > 0:
                        _process_rollout(agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
                                         folder=folder, foundation_sha=foundation_sha,
                                         assignments_writer=assignments_writer, bout=bout, force=True)
                        decision = step1._finish_cycle(agent, cycle, agent.holdout_rows, agent.run_config)
                        _clear_optimizer_state(agent)
                        decision.update({"stream_reason": "complete_1024_cycle", "external_condition": agent.external_condition})
                        _write_jsonl(decisions_stream, decision)
                        _save_stream_checkpoint(agent, folder / "checkpoints" / f"step_{agent.counts['environment_steps']:08d}.pt",
                                                foundation_sha, capacity_rows)
                    agent.cycle_state = None
                    agent.cycle_step = 0
                    agent.rollout = []
                    agent.rollout_phase = None
                    agent.holdout_rows = []
                    agent.current_route = None

            _append_ledger(
                steps_writer, kind="probe" if is_probe else "training",
                global_step=int(agent.counts["raw_steps"]), bout=bout, task_name=task_name,
                slot=route_slot, generation=generation, route_after=active_after,
                action=action, reward=reward, probabilities=behavior_probs, log_prob=log_prob,
                obs=before, next_obs=next_obs, terminated=bool(terminated), truncated=bool(truncated),
                phase=phase_name if not is_probe else "uniform_probe", excluded=switched and not is_probe,
            )
            if route_writer is not None and agent.run_config["phase"] == "unknown":
                route_writer.writerow({
                    "raw_step": agent.counts["raw_steps"], "bout": bout, "taskobserver": task_name,
                    "slot_before": route_slot, "active_after": active_after,
                    "allocated_slot": event.get("allocated_slot"), "evicted_slot": event.get("evicted_slot"),
                    "switched": int(switched), "reason": event.get("reason", route.get("reason")),
                    "confidence": event.get("confidence", route.get("confidence", 0.0)),
                })

            if done:
                episode_kind = "training" if episode_has_training else "probe"
                _complete_episode(episodes_writer, agent, kind=episode_kind,
                                  task_name=task_name, start_task_name=episode_start_task,
                                  bout=bout, episode_id=episode_id,
                                  episode_index=episode_index, episode_return=episode_return,
                                  episode_length=episode_length, goal_reached=episode_goal,
                                  preference=spec.preference,
                                  task_observers=episode_task_observers)
                episode_id += 1
                episode_index += 1
                episode_return, episode_length, episode_goal = 0.0, 0, False
                episode_has_training = False
                episode_start_task = task_name
                episode_task_observers = {task_name}
                obs, _ = env.reset()
            else:
                obs = next_obs
            if is_probe and known_probe_left == 0:
                profile = agent.windows[int(known_probe_slot)].profile()
                _maybe_assign_group(agent, int(known_probe_slot), profile, assignments_writer, bout=bout)

    if episode_length:
        _complete_episode(episodes_writer, agent, kind="training" if episode_has_training else "probe",
                          task_name=task_name, start_task_name=episode_start_task,
                          bout=len(task_sequence) - 1, episode_id=episode_id, episode_index=episode_index,
                          episode_return=episode_return, episode_length=episode_length,
                          goal_reached=episode_goal, preference=active_spec.preference, complete=False,
                          task_observers=episode_task_observers)
    agent.next_episode_id = episode_id
    _finish_homogeneous(agent, decisions_stream=decisions_stream, capacity_rows=capacity_rows,
                        folder=folder, foundation_sha=foundation_sha,
                        assignments_writer=assignments_writer, bout=len(task_sequence) - 1,
                        reason="stream_end")
    return {
        name: {
            "training_steps": int(row["steps"]),
            "grouped_steps": int(row["grouped_steps"]),
            "unassigned_steps": int(row["unassigned_steps"]),
            "group_full_steps": int(row["group_full_steps"]),
            "unassigned_when_full_steps": int(row["unassigned_when_full_steps"]),
            "resident_slots": sorted(int(slot) for slot in row["slots"]),
            "groups_used": sorted(int(group) for group in row["groups"]),
        }
        for name, row in group_coverage.items()
    }


def _summarize_group_coverage(group_coverage: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize post-hoc grouped/unassigned frequencies by observer task."""
    unassigned_tasks = sorted(name for name, row in group_coverage.items() if int(row["unassigned_steps"]) > 0)
    full_unassigned_tasks = sorted(
        name for name, row in group_coverage.items()
        if int(row["unassigned_when_full_steps"]) > 0
    )
    unassigned_steps = sum(int(row["unassigned_steps"]) for row in group_coverage.values())
    grouped_steps = sum(int(row["grouped_steps"]) for row in group_coverage.values())
    group_full_steps = sum(int(row["group_full_steps"]) for row in group_coverage.values())
    unassigned_when_full = sum(int(row["unassigned_when_full_steps"]) for row in group_coverage.values())
    return {
        "unassigned_task_coverage": unassigned_tasks,
        "tasks_with_unassigned_steps": len(unassigned_tasks),
        "unassigned_steps_total": unassigned_steps,
        "unassigned_step_fraction": unassigned_steps / max(1, grouped_steps + unassigned_steps),
        "group_full_steps_total": group_full_steps,
        "unassigned_when_full_steps_total": unassigned_when_full,
        "unassigned_when_full_step_fraction": unassigned_when_full / max(1, group_full_steps),
        "unassigned_when_full_task_coverage": full_unassigned_tasks,
        "tasks_with_unassigned_steps_after_group_full": len(full_unassigned_tasks),
        "per_task": dict(group_coverage),
    }


def _clone_evaluate(agent, *, tasks: Sequence[str], episodes: int, steps_writer, eval_writer) -> dict[str, Any]:
    """Evaluate a cloned endpoint; no live policy, router, or optimizer is changed."""
    started = time.process_time()
    core = copy.deepcopy(agent.core)
    core.eval()
    cache = BoundedTaskCache.from_state_dict(agent.cache.state_dict()) if agent.cache is not None else None
    router = WriteDecisionRouter.from_state_dict(agent.router.state_dict()) if agent.router is not None else None
    if router is not None:
        router.clear_recent()
    results: dict[str, Any] = {}
    eval_steps = 0
    for task_index, name in enumerate(tasks):
        spec = task_from_name(name)
        env = make_env(spec, str(agent.run_config["domain"]))
        if cache is not None:
            route = cache.resolve(name)
            slot = int(route["slot"])
            if route["allocated"]:
                with torch.no_grad():
                    core._nodes[str(slot + 1)].A.zero_()
                if core.group_for_slot(slot) >= 0:
                    core.set_slot_group(slot, -1, preserve_effective=False)
        returns, goals, positives, lengths = [], [], [], []
        for episode in range(episodes):
            obs, _ = env.reset(seed=int(agent.seed) + 81000 + task_index * 1000 + episode * 17)
            episode_return, length, goal, done = 0.0, 0, False, False
            while not done:
                route = router.route(obs) if router is not None else {"slot": slot}
                route_slot = int(route["slot"])
                route_id = route_slot + 1 if route_slot >= 0 else 0
                policy_obs = loop.augment_observation(obs, route_id)
                probs, _ = _policy_probs(core, policy_obs)
                unknown = router is not None and route_slot < 0
                behavior = unknown_behavior_probs(probs, float(agent.run_config["unknown_uniform_mix"])) if unknown else probs
                action, log_prob = _sample(behavior, np.random.default_rng(int(agent.seed) + 90000 + task_index * 10000 + episode * 100 + length))
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                if router is not None:
                    event = router.observe(obs, action, reward, next_obs, bool(terminated))
                    allocated = event.get("allocated_slot")
                    if allocated is not None:
                        with torch.no_grad():
                            core._nodes[str(int(allocated) + 1)].A.zero_()
                        if core.group_for_slot(int(allocated)) >= 0:
                            core.set_slot_group(int(allocated), -1, preserve_effective=False)
                eval_steps += 1
                length += 1
                episode_return += float(reward)
                goal = goal or bool(info.get("goal_reached", False))
                _append_ledger(steps_writer, kind="evaluation",
                               global_step=int(agent.counts["raw_steps"]) + eval_steps, bout=-1,
                               task_name=name, slot=route_slot, generation=-1, route_after=(event.get("active_slot", route_slot) if router is not None else route_slot),
                               action=action, reward=reward, probabilities=behavior, log_prob=log_prob,
                               obs=obs, next_obs=next_obs, terminated=bool(terminated), truncated=bool(truncated),
                               phase="final_clone", excluded=False)
                obs = next_obs
            positive_success = bool(goal and spec.preference > 0)
            row = {"phase": "final_clone", "taskobserver": name, "episode": episode,
                   "return_value": episode_return, "goal_reached": int(goal),
                   "positive_success": int(positive_success), "length": length, "complete": 1}
            eval_writer.writerow(row)
            returns.append(episode_return)
            goals.append(int(goal))
            positives.append(int(positive_success))
            lengths.append(length)
        env.close()
        results[name] = {
            "mean_signed_return": float(np.mean(returns)),
            "mean_goal_rate": float(np.mean(goals)),
            "positive_success_rate": float(np.mean(positives)),
            "episodes": episodes,
            "mean_length": float(np.mean(lengths)),
        }
    return {"tasks": results, "evaluation_steps": eval_steps, "cpu_s": time.process_time() - started}


def _run_one(config: Mapping[str, Any], selection: Mapping[str, Any], output: Path) -> dict[str, Any]:
    phase = _validate_phase(str(config["phase"]))
    condition = str(config["condition"])
    seed = int(config["seed"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preserve existing stream output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base = _apply_foundation_override(loop.load_base_config(), config)
    agent = _new_agent(condition, seed, config, base)
    foundation_sha = _sha(Path(base["foundation"]))
    manifest = _source_snapshot(output, base, selection)
    if phase == "known":
        agent.cache = _make_cache(agent, config)
        agent.current_known_slot = None
    else:
        agent.router = WriteDecisionRouter(
            max_slots=agent.slots, window=int(config["unknown_router_window"]),
            eviction=agent.eviction, seed=seed + 991,
            reward_support_aware=bool(config.get("reward_support_aware", True)),
        )
        agent.current_known_slot = None
    run_config = {**dict(config), "condition": condition, "phase": phase, "seed": seed}
    _write_json(output / "config.json", {**run_config, "selection": str(selection["path"]), "foundation_sha256": foundation_sha})
    stream_files = []
    capacity_rows: list[dict[str, Any]] = []
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    ledger_fields = ("interaction_kind", "raw_step", "bout", "taskobserver", "slot_before", "generation_before", "active_after", "action", "reward", "behavior_probs", "behavior_logp", "obs", "next_obs", "terminated", "truncated", "phase", "boundary_excluded")
    episode_fields = ("interaction_kind", "taskobserver", "episode_start_taskobserver", "episode_task_observers", "mixed_task_episode", "bout", "episode_id", "episode", "return_value", "goal_reached", "positive_success", "length", "complete")
    route_fields = ("raw_step", "bout", "taskobserver", "slot_before", "active_after", "allocated_slot", "evicted_slot", "switched", "reason", "confidence")
    assignment_fields = ("phase", "bout", "slot", "generation", "group", "reason", "binding", "profile_rows", "scores", "committed")
    try:
        with (output / "steps.csv").open("w", newline="", encoding="utf-8") as steps_stream, \
             (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episodes_stream, \
             (output / "routing.csv").open("w", newline="", encoding="utf-8") as route_stream, \
             (output / "profile_assignments.csv").open("w", newline="", encoding="utf-8") as assignments_stream, \
             (output / "decisions.jsonl").open("w", encoding="utf-8") as decisions_stream, \
             (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as eval_stream:
            stream_files = [steps_stream, episodes_stream, route_stream, assignments_stream, decisions_stream, eval_stream]
            steps_writer = csv.DictWriter(steps_stream, fieldnames=ledger_fields)
            episodes_writer = csv.DictWriter(episodes_stream, fieldnames=episode_fields)
            route_writer = csv.DictWriter(route_stream, fieldnames=route_fields)
            assignments_writer = csv.DictWriter(assignments_stream, fieldnames=assignment_fields)
            eval_writer = csv.DictWriter(eval_stream, fieldnames=("phase", "taskobserver", "episode", "return_value", "goal_reached", "positive_success", "length", "complete"))
            for writer in (steps_writer, episodes_writer, route_writer, assignments_writer, eval_writer):
                writer.writeheader()
            first = task_from_name(STREAM_SEQUENCE[0])
            env = make_env(first, str(config["domain"]))
            group_coverage = _run_training_stream(
                agent, env, steps_writer=steps_writer, episodes_writer=episodes_writer,
                route_writer=route_writer if phase == "unknown" else None,
                assignments_writer=assignments_writer, decisions_stream=decisions_stream,
                capacity_rows=capacity_rows, folder=output, foundation_sha=foundation_sha,
                bout_steps=int(config["bout_steps"]), task_sequence=STREAM_SEQUENCE,
            )
            env.close()
            coverage_fields = ("taskobserver", "training_steps", "grouped_steps", "unassigned_steps", "group_full_steps", "unassigned_when_full_steps", "resident_slots", "groups_used")
            with (output / "observer_group_coverage.csv").open("w", newline="", encoding="utf-8") as coverage_stream:
                coverage_writer = csv.DictWriter(coverage_stream, fieldnames=coverage_fields)
                coverage_writer.writeheader()
                for task, row in group_coverage.items():
                    coverage_writer.writerow({"taskobserver": task, **row,
                                              "resident_slots": json.dumps(row["resident_slots"]),
                                              "groups_used": json.dumps(row["groups_used"])})
            eval_targets = TASKS if int(config["evaluation_targets"]) == 12 else TASKS[:6]
            final_eval = _clone_evaluate(agent, tasks=eval_targets, episodes=int(config["evaluation_episodes"]),
                                         steps_writer=steps_writer, eval_writer=eval_writer)
            agent.counts["evaluation_steps"] = int(final_eval["evaluation_steps"])
            _save_stream_checkpoint(agent, output / "checkpoint_final.pt", foundation_sha, capacity_rows)
    finally:
        for stream in stream_files:
            if not stream.closed:
                stream.flush()
                stream.close()
    _write_json(output / "capacity.json", capacity_rows)
    training_reward = float(sum(agent.reward_by_task.values()))
    training_steps = int(agent.counts["environment_steps"])
    active_slots = agent.cache.slot_to_task if agent.cache is not None else (agent.router.active_slots if agent.router is not None else ())
    active_profile_slots = [slot for slot, active in enumerate(agent.profiles["seen"]) if bool(active)]
    unassigned_profile_slots = [slot for slot in active_profile_slots if agent.core.group_for_slot(slot) < 0]
    coverage_summary = _summarize_group_coverage(group_coverage)
    group_capacity = int(agent.core.shared_groups)
    group_reason_counts: dict[str, int] = {}
    for reason in agent.profile_reasons:
        group_reason_counts[str(reason)] = group_reason_counts.get(str(reason), 0) + 1
    result = {
        "phase": phase, "condition": condition, "seed": seed, "domain": config["domain"],
        "sequence": list(STREAM_SEQUENCE), "bout_steps": int(config["bout_steps"]),
        "counts": dict(agent.counts), "training_reward": training_reward,
        "reward_per_step": training_reward / max(1, training_steps),
        "positive_goal_rate": agent.counts["positive_goal_episodes"] / max(1, agent.counts["positive_completed_episodes"]),
        "per_task_training": {
            name: {"reward": agent.reward_by_task[name], "steps": agent.steps_by_task[name],
                   "reward_per_step": agent.reward_by_task[name] / max(1, agent.steps_by_task[name]),
                   "complete_episodes": agent.complete_episodes_by_task[name],
                   "positive_complete_episodes": agent.positive_complete_episodes_by_task[name],
                   "positive_goal_episodes": agent.goal_episodes_by_task[name]}
            for name in TASKS
        },
        "per_task_observed_stream": {
            name: {"reward": agent.observed_reward_by_task[name],
                   "steps": agent.observed_steps_by_task[name],
                   "reward_per_step": agent.observed_reward_by_task[name] / max(1, agent.observed_steps_by_task[name])}
            for name in TASKS
        },
        "evaluations": final_eval["tasks"], "evaluation_steps": int(final_eval["evaluation_steps"]),
        "costs": {"training_steps": training_steps, "probe_steps": int(agent.counts["probe_steps"]),
                  "evaluation_steps": int(final_eval["evaluation_steps"]),
                  "total_interactions": int(agent.counts["raw_steps"] + final_eval["evaluation_steps"]),
                  "wall_s": time.perf_counter() - start_wall, "cpu_s": time.process_time() - start_cpu,
                  "decision_cpu_s": float(agent.decision_cpu_s), "evaluation_cpu_s": float(final_eval["cpu_s"]),
                  "process_rss_bytes": int(psutil.Process().memory_info().rss)},
        "resident_cache": agent.cache.state_dict() if agent.cache is not None else None,
        "router_state_bytes": int(agent.router.nbytes) if agent.router is not None else 0,
        "profile_state_bytes": int(sum(array.nbytes for array in agent.profiles.values())),
        "fitted_steps_bytes": int(agent.fitted_steps.nbytes),
        "group_state_bytes": int(agent.grouper.nbytes), "anchor_state_bytes": int(agent.anchors.nbytes),
        "group_saturation": {
            "active_groups": list(agent.grouper.active_groups),
            "group_capacity": group_capacity,
            "grouping_enabled": group_capacity > 0,
            "groups_full": bool(group_capacity > 0 and len(agent.grouper.active_groups) >= group_capacity),
            "active_slots": len(active_profile_slots),
            "unassigned_active_slots": len(unassigned_profile_slots),
            "unassigned_slot_ids": unassigned_profile_slots,
            "group_abstentions": int(agent.counts["group_abstentions"]),
            "max_groups_abstentions": int(agent.counts["group_max_abstentions"]),
            "cold_prior_bindings": int(agent.counts["cold_prior_bindings"]),
            "effective_preserving_bindings": int(agent.counts["effective_preserving_bindings"]),
            "profile_reason_frequencies": group_reason_counts,
            **coverage_summary,
        },
        "source_manifest": manifest,
        "limitations": [
            "The known-cache phase uses observer task names only for bounded resident-slot lookup and ledgers.",
            "The unknown-boundary router sees visible transitions only; the fixed linear reward model can be misspecified on detour shaping.",
            "Group assignment abstains when learned numeric profiles do not distinguish a group or both groups are occupied.",
            "A never-fitted local slot can inherit a learned group prior as a cold_prior_binding heuristic; that transfer is not evidence of benefit. Fitted slots preserve their effective logits when bound.",
            "Episodes may contain multiple observer tasks because bout changes preserve the live episode. Mixed-task episodes are reported but excluded from positive-goal episode denominators; reward and step totals remain attributed per interaction.",
            "Checkpoints restore learner state but not the continuous environment position or in-flight episode.",
        ],
    }
    _write_json(output / "summary.json", result)
    return result


def run_cohort(phase: str, output: str | Path, *, selection_path: str | Path,
               workers: int = 1, conditions: Sequence[str] | None = None,
               seeds: Sequence[int] | None = None, bout_steps: int | None = None,
               domain: str | None = None, local_slots: int | None = None,
               evaluation_targets: int | None = None,
               foundation: str | Path | None = None) -> dict[str, Any]:
    phase = _validate_phase(phase)
    if not 1 <= int(workers) <= 2:
        raise ValueError("workers must be one or two")
    selection = _load_selection(selection_path)
    config = _load_config()
    run_conditions = resolve_conditions(selection, conditions)
    run_seeds = [int(value) for value in (config["seeds"] if seeds is None else seeds)]
    if not run_seeds:
        raise ValueError("seeds must not be empty")
    steps = int(config["bout_steps"] if bout_steps is None else bout_steps)
    if steps < 1:
        raise ValueError("bout_steps must be positive")
    selected_domain = str(domain or config["domain"])
    if selected_domain not in ("navigation", "detour"):
        raise ValueError("domain must be navigation or detour")
    slots = local_slots if local_slots is not None else config.get("local_slots")
    if slots is not None and int(slots) not in (6, 7):
        raise ValueError("local_slots override must be six or seven")
    _validate_local_slot_conditions(run_conditions, None if slots is None else int(slots))
    eval_count = int(config["evaluation_targets"] if evaluation_targets is None else evaluation_targets)
    if eval_count not in (6, 12):
        raise ValueError("evaluation_targets must be six or twelve")
    run_config = {
        **config, "phase": phase, "bout_steps": steps, "domain": selected_domain,
        "local_slots": slots, "evaluation_targets": eval_count,
    }
    if foundation is not None:
        foundation_path = Path(foundation).resolve()
        if not foundation_path.is_file():
            raise FileNotFoundError(f"foundation checkpoint does not exist: {foundation_path}")
        run_config["foundation"] = str(foundation_path)
    target = Path(output).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"Preserve existing stream cohort output: {target}")
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "frozen_config.json", {
        **run_config, "conditions": run_conditions, "seeds": run_seeds,
        "selection_path": str(selection["path"]), "selection_sha256": _sha(Path(selection["path"])),
        "protocol_sha256": _sha(PROTOCOL_PATH),
    })
    tasks = []
    for seed in run_seeds:
        for condition in run_conditions:
            folder = target / f"seed_{seed}" / condition
            tasks.append((
                {**run_config, "seed": seed, "condition": condition},
                selection, folder,
            ))
    if workers == 1:
        records = [_run_one(config, selection, folder) for config, selection, folder in tasks]
    else:
        records = []
        with ProcessPoolExecutor(max_workers=min(2, int(workers))) as executor:
            futures = [executor.submit(_run_one, config, selection, folder) for config, selection, folder in tasks]
            for future in as_completed(futures):
                records.append(future.result())
    records.sort(key=lambda row: (int(row["seed"]), str(row["condition"])))
    summary = {
        "phase": phase, "base_sequence": list(TASKS), "sequence": list(STREAM_SEQUENCE),
        "conditions": run_conditions, "seeds": run_seeds,
        "records": records, "protocol_sha256": _sha(PROTOCOL_PATH),
        "cohort_cost": {key: int(sum(row["costs"][key] for row in records))
                        for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")},
    }
    _write_json(target / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("known", "unknown"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bout-steps", type=int)
    parser.add_argument("--domain", choices=("navigation", "detour"))
    parser.add_argument("--local-slots", type=int, choices=(6, 7))
    parser.add_argument("--evaluation-targets", type=int, choices=(6, 12))
    parser.add_argument("--foundation", type=Path, help="override the W0 foundation checkpoint")
    args = parser.parse_args()
    result = run_cohort(
        args.phase, args.output, selection_path=args.selection, workers=args.workers,
        conditions=args.conditions, seeds=args.seeds, bout_steps=args.bout_steps,
        domain=args.domain, local_slots=args.local_slots, evaluation_targets=args.evaluation_targets,
        foundation=args.foundation,
    )
    print(json.dumps({"phase": result["phase"], "conditions": result["conditions"],
                      "seeds": result["seeds"], "cohort_cost": result["cohort_cost"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

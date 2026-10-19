"""Run bounded causal online policy updates for the C1 navigation task.

The hidden environment schedule lives in the outer experiment harness. The
router and policy receive only observations, actions, rewards, and transitions.
No experiment runs when this module is imported.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import torch

from online_rl.c1_envs import C1NavigationEnv, C1_CONTEXTS, context_index
from online_rl.c1_strong_models import C1ModelController
from online_rl.c1_tensor_controls import TensorSourceBasisPolicy
from online_rl.compact_knowledge import CompactKnowledge, encode
from online_rl.task_free_router import TaskFreeRouter


ACTOR_BUDGET_BYTES = 320
CONSOLIDATE_EVERY = 1024
ROLLOUT_STEPS = 256
BATCH_SIZE = 64
PPO_EPOCHS = 4
LEARNING_RATE = 0.003
ENTROPY_COEF = 0.06
GAMMA = 0.98
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
TARGET_KL = 0.06
MAX_GRAD_NORM = 0.5
ANCHORS_PER_SLOT = 64
MAX_CONTEXTS = 6
NEURAL_ARMS = ("single_head_online", "sparse", "int8", "svd", "evidence_gate")
ALL_ARMS = ("frozen", *NEURAL_ARMS, "adaptive_assignment")
CODEC_FOR_ARM = {
    "single_head_online": "int8",
    "sparse": "sparse_topk",
    "int8": "int8",
    "svd": "truncated_svd",
}
GATE_CODECS = ("global_mean_sparse", "centroid2_sparse", "truncated_svd", "int8")
SOURCE_SNAPSHOT_FILES = (
    Path("scripts/run_knowledge_loop.py"),
    Path("docs/KNOWLEDGE_ORGANIZATION_PROTOCOL.md"),
    Path("configs/c1.json"),
    Path("src/online_rl/compact_knowledge.py"),
    Path("src/online_rl/task_free_router.py"),
    Path("src/online_rl/c1_envs.py"),
    Path("src/online_rl/c1_strong_models.py"),
    Path("src/online_rl/c1_tensor_controls.py"),
    Path("src/online_rl/c1_tensor_experiment.py"),
    Path("src/online_rl/c1_policy.py"),
    Path("src/online_rl/c1_supplement_experiment.py"),
    Path("src/online_rl/policy.py"),
    Path("src/online_rl/experiment.py"),
    Path("src/online_rl/envs.py"),
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _snapshot_sources(output: Path, base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the exact source files used by a run and record all input hashes."""
    snapshot_root = output / "source_snapshot"
    copied: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_SNAPSHOT_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_digest, snapshot_digest = _hash(source), _hash(destination)
        if source_digest != snapshot_digest:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        copied[str(relative)] = {"sha256": snapshot_digest, "bytes": destination.stat().st_size}
    foundation = Path(base_config["foundation"])
    dataset = Path(base_config["dataset"])
    manifest = {
        "sources": copied,
        "foundation": {"path": str(foundation), "sha256": _hash(foundation), "bytes": foundation.stat().st_size},
        "dataset": {"path": str(dataset), "sha256": _hash(dataset), "bytes": dataset.stat().st_size},
    }
    _write_json(snapshot_root / "manifest.json", manifest)
    return manifest


def make_hidden_schedule(total_steps: int, seed: int) -> list[dict[str, Any]]:
    """Create a randomized external schedule whose segments sum exactly."""
    if total_steps < 1024:
        raise ValueError("total_steps must be at least 1024")
    rng = np.random.default_rng(int(seed))
    segment_count = max(1, int(math.ceil(total_steps / 1536.0)))
    while segment_count * 1024 > total_steps:
        segment_count -= 1
    while segment_count * 2048 < total_steps:
        segment_count += 1

    lengths: list[int] = []
    remaining = int(total_steps)
    for index in range(segment_count):
        later = segment_count - index - 1
        low = max(1024, remaining - 2048 * later)
        high = min(2048, remaining - 1024 * later)
        length = int(rng.integers(low, high + 1)) if low < high else low
        lengths.append(length)
        remaining -= length
    if remaining != 0 or sum(lengths) != total_steps:
        raise AssertionError("schedule partition failed to cover requested steps")

    contexts = tuple(C1_CONTEXTS)
    names: list[str] = []
    while len(names) < segment_count:
        names.extend(str(item) for item in rng.permutation(contexts))
    names = names[:segment_count]
    schedule: list[dict[str, Any]] = []
    start = 0
    for segment_id, (name, length) in enumerate(zip(names, lengths)):
        schedule.append({
            "segment_id": segment_id,
            "start_step": start,
            "end_step": start + length,
            "length": length,
            "context": name,
        })
        start += length
    return schedule


def hidden_context_at(schedule: list[Mapping[str, Any]], step: int) -> Mapping[str, Any]:
    """Outer-harness lookup; this function is never passed to the learner."""
    for segment in schedule:
        if int(segment["start_step"]) <= step < int(segment["end_step"]):
            return segment
    raise IndexError(f"step {step} is outside the hidden schedule")


def augment_observation(obs: np.ndarray, route_id: int) -> np.ndarray:
    """Match the C1 policy representation: 4 visible values, 8 zero history, route."""
    value = np.asarray(obs, dtype=np.float32)
    if value.shape != (4,):
        raise ValueError("C1 policy input must be a four-value observation")
    return np.concatenate([value, np.zeros(8, dtype=np.float32), np.asarray([route_id], dtype=np.float32)])


def rollout_number_at_step(step: int) -> int:
    """Return the one-based rollout index within a consolidation cycle."""
    if step < 1:
        raise ValueError("step must be a positive one-based environment step")
    return ((int(step) - 1) % CONSOLIDATE_EVERY) // ROLLOUT_STEPS + 1


def route_id_for(agent: "OnlineArm", route: Mapping[str, Any]) -> int:
    """Map an action-before-step router result to the policy's route feature."""
    if agent.shared_head:
        return 1
    if agent.controller is not None:
        return 0
    slot = int(route.get("slot", -1))
    return slot + 1 if 0 <= slot < MAX_CONTEXTS else 0


def bootstrap_observation(agent: "OnlineArm", obs: np.ndarray) -> np.ndarray:
    """Prepare a causal next-state value input with the next action's route."""
    if agent.controller is not None or agent.shared_head:
        route = {"slot": -1}
    else:
        # The prior transition has already been observed. This is the same
        # action-before-action router decision the next iteration will use.
        route = agent.route_before_action(obs)
    return augment_observation(obs, route_id_for(agent, route))


def bootstrap_value(agent: "OnlineArm", obs: np.ndarray) -> float:
    policy_obs = torch.as_tensor(bootstrap_observation(agent, obs), dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        return float(agent.core.predict_values(policy_obs).reshape(-1)[0].cpu().item())


def actor_valid_mask(
    slots: np.ndarray,
    generations: np.ndarray,
    current_generations: Iterable[int],
    *,
    arm: str,
) -> np.ndarray:
    """Mask actor samples whose discovered slot was reset after collection."""
    slots = np.asarray(slots, dtype=np.int64)
    generations = np.asarray(generations, dtype=np.int64)
    current = np.asarray(tuple(current_generations), dtype=np.int64)
    if slots.shape != generations.shape:
        raise ValueError("slot and generation arrays must have matching shapes")
    if arm == "single_head_online":
        # The one shared correction is never reset when a discovered router
        # slot is evicted, so all collected actor rows remain current.
        return np.ones(slots.shape, dtype=bool)
    valid = np.zeros(slots.shape, dtype=bool)
    for index, (slot, generation) in enumerate(zip(slots.flat, generations.flat)):
        if slot < 0:
            valid.flat[index] = arm == "single_head_online"
        elif slot < len(current):
            valid.flat[index] = generation == current[slot]
    return valid


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    value: float
    next_value: float
    log_prob: float
    probabilities: np.ndarray
    done: bool
    terminated: bool
    slot: int
    generation: int
    step: int
    advantage: float = 0.0
    returns: float = 0.0


class AnchorReplay:
    """Fixed per-slot replay of observed inputs and committed behavior policy."""

    def __init__(self, slots: int = MAX_CONTEXTS, capacity: int = ANCHORS_PER_SLOT) -> None:
        self.capacity = int(capacity)
        self.items_by_slot: list[deque[dict[str, Any]]] = [
            deque(maxlen=self.capacity) for _ in range(slots)
        ]

    def add(self, slot: int, generation: int, obs: np.ndarray, probabilities: np.ndarray) -> None:
        if slot < 0 or slot >= len(self.items_by_slot):
            return
        self.items_by_slot[slot].append({
            "generation": int(generation),
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "probabilities": np.asarray(probabilities, dtype=np.float32).copy(),
        })

    def clear_slot(self, slot: int) -> None:
        self.items_by_slot[int(slot)].clear()

    def active_items(self, generations: Iterable[int], *, slots: Iterable[int] | None = None) -> list[dict[str, Any]]:
        current = tuple(int(value) for value in generations)
        allowed = set(range(len(self.items_by_slot))) if slots is None else {int(v) for v in slots}
        result: list[dict[str, Any]] = []
        for slot, items in enumerate(self.items_by_slot):
            if slot not in allowed:
                continue
            for item in items:
                if item["generation"] == current[slot]:
                    result.append({**item, "slot": slot})
        return result

    @property
    def payload_nbytes(self) -> int:
        return int(sum(
            item["obs"].nbytes + item["probabilities"].nbytes + 8
            for rows in self.items_by_slot for item in rows
        ))

    @property
    def nbytes(self) -> int:
        # Include bounded Python containers and ndarray object overhead as
        # well as their numeric payloads.
        size = sum(sys.getsizeof(rows) for rows in self.items_by_slot)
        for rows in self.items_by_slot:
            for item in rows:
                size += sys.getsizeof(item)
                size += sys.getsizeof(item["generation"])
                size += sys.getsizeof(item["obs"])
                size += sys.getsizeof(item["probabilities"])
        return int(size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "slots": [[{
                "generation": item["generation"],
                "obs": item["obs"].copy(),
                "probabilities": item["probabilities"].copy(),
            } for item in rows] for rows in self.items_by_slot],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "AnchorReplay":
        replay = cls(slots=len(state["slots"]), capacity=int(state["capacity"]))
        for slot, rows in enumerate(state["slots"]):
            if len(rows) > replay.capacity:
                raise ValueError("anchor replay state exceeds capacity")
            for item in rows:
                obs = np.asarray(item["obs"], dtype=np.float32)
                probs = np.asarray(item["probabilities"], dtype=np.float32)
                if obs.shape != (13,) or probs.shape != (4,):
                    raise ValueError("anchor replay record has an invalid shape")
                replay.items_by_slot[slot].append({
                    "generation": int(item["generation"]),
                    "obs": obs.copy(),
                    "probabilities": probs.copy(),
                })
        return replay


class OnlineArm:
    """One learner/controller with compact committed actor knowledge."""

    def __init__(self, arm: str, seed: int, base_config: Mapping[str, Any]) -> None:
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.seed = int(seed)
        self.base_config = dict(base_config)
        torch.set_num_threads(1)
        self.core, self.optimizer = _make_policy_and_optimizer(seed, base_config, train=arm in NEURAL_ARMS)
        self.router = TaskFreeRouter(max_contexts=MAX_CONTEXTS, window=32)
        self.controller = (
            C1ModelController(kind="adaptive_assignment", window=32)
            if arm == "adaptive_assignment" else None
        )
        self.action_rng = np.random.default_rng(seed + 1100)
        self.slot_generation = np.zeros(MAX_CONTEXTS, dtype=np.int64)
        self.anchors = AnchorReplay()
        self.counts: dict[str, Any] = {
            "environment_steps": 0,
            "rollouts_trained": 0,
            "minibatch_updates": 0,
            "stale_actor_rows": 0,
            "accepted": 0,
            "rejected": 0,
            "codec_commits": 0,
            "optimizer_reset_parameters": 0,
            "optimizer_reset_events": 0,
            "slot_allocations": 0,
            "slot_evictions": 0,
            "local_trial_steps": 0,
            "validation_steps": 0,
        }
        self.compact: CompactKnowledge | None = None
        self.committed_matrix = np.zeros((6, 27), dtype=np.float32)
        self.committed_method = "int8"
        if arm in NEURAL_ARMS:
            shape = (1, 27) if arm == "single_head_online" else (6, 27)
            method = CODEC_FOR_ARM.get(arm, "int8")
            self.compact = encode(np.zeros(shape, np.float32), method, ACTOR_BUDGET_BYTES)
            self.committed_method = method
            self.committed_matrix = self.compact.decode()
            _install_matrix(self.core, self.committed_matrix, shared=arm == "single_head_online")

    @property
    def shared_head(self) -> bool:
        return self.arm == "single_head_online"

    def route_before_action(self, obs: np.ndarray) -> dict[str, Any]:
        return self.router.route(obs)

    def action(
        self, obs: np.ndarray, route: Mapping[str, Any]
    ) -> tuple[int, np.ndarray, int, np.ndarray, float, float]:
        slot = int(route.get("slot", -1)) if self.controller is None else -1
        route_id = route_id_for(self, route)
        policy_obs = augment_observation(obs, route_id)
        tensor_obs = torch.as_tensor(policy_obs, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            base_distribution = self.core.get_distribution(tensor_obs).distribution
            base_probs = base_distribution.probs.reshape(-1).cpu().numpy().astype(np.float64)
            if self.controller is not None:
                probs = self.controller.action_probabilities(obs, base_probs)
            else:
                probs = base_probs / base_probs.sum()
            value = float(self.core.predict_values(tensor_obs).reshape(-1)[0].cpu().item())
        action, log_prob = _sample_action(probs, self.action_rng)
        probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
        probs /= probs.sum()
        return action, policy_obs, slot, probs.astype(np.float32), value, log_prob

    def observe_transition(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray) -> dict[str, Any] | None:
        if self.controller is not None:
            self.controller.observe(obs, action, next_obs, reward=reward)
            return None
        event = self.router.observe(obs, action, reward, next_obs)
        allocated = event.get("allocated_slot")
        if allocated is not None:
            self._reset_allocated_slot(int(allocated), evicted=event.get("evicted_slot") is not None)
        return event

    def _reset_allocated_slot(self, slot: int, *, evicted: bool) -> None:
        if slot not in range(MAX_CONTEXTS):
            raise ValueError("router allocated a slot outside configured capacity")
        self.slot_generation[slot] += 1
        self.counts["slot_allocations"] += 1
        self.counts["slot_evictions"] += int(evicted)
        self.anchors.clear_slot(slot)
        if self.arm in NEURAL_ARMS and not self.shared_head:
            self.committed_matrix[slot].fill(0.0)
            trial = _read_matrix(self.core)
            trial[slot].fill(0.0)
            _install_matrix(self.core, trial, shared=False)
            self._reencode_committed(self.committed_matrix, preferred=self.committed_method)
            if self.optimizer is not None:
                parameter = self.core._nodes[str(slot + 1)].A
                had_state = parameter in self.optimizer.state
                self.optimizer.state.pop(parameter, None)
                self.counts["optimizer_reset_events"] += 1
                self.counts["optimizer_reset_parameters"] += int(had_state)

    def _reencode_committed(self, matrix: np.ndarray, *, preferred: str) -> None:
        if self.arm not in NEURAL_ARMS:
            return
        try:
            compact = encode(matrix, preferred, ACTOR_BUDGET_BYTES)
        except ValueError:
            compact = encode(matrix, "int8", ACTOR_BUDGET_BYTES)
        self.compact = compact
        self.committed_method = compact.method
        self.committed_matrix = compact.decode()
        _install_matrix(self.core, self.committed_matrix, shared=self.shared_head)


def _make_policy_and_optimizer(
    seed: int, base_config: Mapping[str, Any], *, train: bool
) -> tuple[TensorSourceBasisPolicy, torch.optim.Optimizer | None]:
    from online_rl.c1_tensor_experiment import _tensor_make_core

    core, optimizer = _tensor_make_core(
        base_config,
        seed,
        "output_mixer",
        {"leaf_rank": 1},
        {"learning_rate": LEARNING_RATE, "ent_coef": ENTROPY_COEF},
    )
    if not train:
        for parameter in core.parameters():
            parameter.requires_grad_(False)
        return core, None
    return core, optimizer


def _read_matrix(core: TensorSourceBasisPolicy) -> np.ndarray:
    return np.stack([
        core._nodes[str(endpoint)].A.detach().cpu().numpy().reshape(27).copy()
        for endpoint in range(1, 7)
    ]).astype(np.float32)


def _install_matrix(core: TensorSourceBasisPolicy, matrix: np.ndarray, *, shared: bool = False) -> None:
    value = np.asarray(matrix, dtype=np.float32)
    expected_rows = 1 if shared else 6
    if value.shape != (expected_rows, 27) or not np.isfinite(value).all():
        raise ValueError(f"actor matrix must be finite with shape ({expected_rows}, 27)")
    with torch.no_grad():
        for endpoint in range(1, 7):
            node = core._nodes[str(endpoint)].A
            if shared:
                node.zero_()
                if endpoint == 1:
                    node.copy_(torch.as_tensor(value[0].reshape(3, 3, 3), dtype=node.dtype, device=node.device))
            else:
                node.copy_(torch.as_tensor(value[endpoint - 1].reshape(3, 3, 3), dtype=node.dtype, device=node.device))


def _finalize_gae(rows: list[Transition]) -> None:
    advantage = 0.0
    for row in reversed(rows):
        bootstrap = 0.0 if row.terminated else 1.0
        continuation = 0.0 if row.done else 1.0
        delta = row.reward + GAMMA * row.next_value * bootstrap - row.value
        advantage = delta + GAMMA * GAE_LAMBDA * continuation * advantage
        row.advantage = float(advantage)
        row.returns = float(advantage + row.value)


def _sample_action(probabilities: np.ndarray, rng: Any) -> tuple[int, float]:
    """Sample a discrete action and return its behavior-policy log probability."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (4,) or not np.isfinite(probs).all() or np.any(probs < 0.0):
        raise ValueError("probabilities must be a finite nonnegative vector of length four")
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum()
    action = int(rng.choice(4, p=probs))
    return action, float(math.log(float(probs[action])))


def _ppo_update(agent: OnlineArm, rows: list[Transition]) -> dict[str, float]:
    if agent.optimizer is None or not rows:
        return {"updates": 0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    _finalize_gae(rows)
    observations = torch.as_tensor(np.stack([row.obs for row in rows]), dtype=torch.float32)
    actions = torch.as_tensor([row.action for row in rows], dtype=torch.long)
    old_log_prob = torch.as_tensor([row.log_prob for row in rows], dtype=torch.float32)
    advantages = torch.as_tensor([row.advantage for row in rows], dtype=torch.float32)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = torch.as_tensor([row.returns for row in rows], dtype=torch.float32)
    slots = np.asarray([row.slot for row in rows], dtype=np.int64)
    versions = np.asarray([row.generation for row in rows], dtype=np.int64)
    valid_actor = actor_valid_mask(slots, versions, agent.slot_generation, arm=agent.arm)
    agent.counts["stale_actor_rows"] += int((~valid_actor).sum())
    valid_actor_tensor = torch.as_tensor(valid_actor, dtype=torch.bool)

    losses: list[tuple[float, float, float, float]] = []
    for _epoch in range(PPO_EPOCHS):
        order = agent.action_rng.permutation(len(rows))
        early_stop = False
        for start in range(0, len(rows), BATCH_SIZE):
            indices = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            values, log_prob, entropy = agent.core.evaluate_actions(
                observations.index_select(0, indices), actions.index_select(0, indices)
            )
            values = values.reshape(-1)
            log_prob = log_prob.reshape(-1)
            entropy = entropy.reshape(-1)
            actor_mask = valid_actor_tensor.index_select(0, indices)
            batch_advantage = advantages.index_select(0, indices)
            if torch.any(actor_mask):
                ratio = torch.exp(log_prob[actor_mask] - old_log_prob.index_select(0, indices)[actor_mask])
                approx_kl = torch.mean((ratio - 1.0) - torch.log(ratio.clamp_min(1e-12)))
                if float(approx_kl.detach()) > 1.5 * TARGET_KL:
                    early_stop = True
                    break
                unclipped = ratio * batch_advantage[actor_mask]
                clipped = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * batch_advantage[actor_mask]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                entropy_loss = entropy[actor_mask].mean()
            else:
                policy_loss = values.new_zeros(())
                entropy_loss = values.new_zeros(())
                approx_kl = values.new_zeros(())
            value_loss = torch.mean((values - returns.index_select(0, indices)) ** 2)
            loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy_loss
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in agent.core.parameters() if parameter.requires_grad],
                MAX_GRAD_NORM,
            )
            agent.optimizer.step()
            agent.counts["minibatch_updates"] += 1
            losses.append((
                float(policy_loss.detach()),
                float(value_loss.detach()),
                float(entropy_loss.detach()),
                float(approx_kl.detach()),
            ))
        if early_stop:
            break
    agent.counts["rollouts_trained"] += 1
    mean = np.asarray(losses, dtype=np.float64).mean(axis=0) if losses else np.zeros(4)
    return {
        "updates": len(losses),
        "policy_loss": float(mean[0]),
        "value_loss": float(mean[1]),
        "entropy": float(mean[2]),
        "approx_kl": float(mean[3]),
    }


def _policy_probs(core: TensorSourceBasisPolicy, observations: np.ndarray) -> np.ndarray:
    if len(observations) == 0:
        return np.empty((0, 4), dtype=np.float32)
    tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        probs = core.get_distribution(tensor).distribution.probs
    return probs.detach().cpu().numpy().astype(np.float32)


def _kl_rows(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = np.clip(np.asarray(reference, dtype=np.float64), 1e-12, 1.0)
    cand = np.clip(np.asarray(candidate, dtype=np.float64), 1e-12, 1.0)
    return np.sum(ref * (np.log(ref) - np.log(cand)), axis=1)


def _matrix_probs(
    agent: OnlineArm,
    matrix: np.ndarray,
    obs: np.ndarray,
    *,
    restore: np.ndarray,
) -> np.ndarray:
    _install_matrix(agent.core, matrix, shared=agent.shared_head)
    result = _policy_probs(agent.core, obs)
    _install_matrix(agent.core, restore, shared=agent.shared_head)
    return result


def _local_surrogate(
    agent: OnlineArm, matrix: np.ndarray, validation: list[Transition]
) -> tuple[float, int]:
    selected = [
        row for row in validation
        if row.slot >= 0 and row.slot < MAX_CONTEXTS
        and row.generation == agent.slot_generation[row.slot]
    ]
    if agent.shared_head:
        selected = list(validation)
    if not selected:
        return float("nan"), 0
    observations = np.stack([row.obs for row in selected])
    actions = np.asarray([row.action for row in selected], dtype=np.int64)
    advantages = np.asarray([row.advantage for row in selected], dtype=np.float64)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    original = _read_matrix(agent.core)
    baseline = agent.committed_matrix.copy()
    old_probs = _matrix_probs(agent, baseline, observations, restore=original)
    new_probs = _matrix_probs(agent, matrix, observations, restore=original)
    old_action = np.clip(old_probs[np.arange(len(actions)), actions], 1e-12, 1.0)
    new_action = np.clip(new_probs[np.arange(len(actions)), actions], 1e-12, 1.0)
    ratio = new_action / old_action
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantages
    return float(np.minimum(unclipped, clipped).mean() - advantages.mean()), len(selected)


def _anchor_records(agent: OnlineArm) -> list[dict[str, Any]]:
    return agent.anchors.active_items(agent.slot_generation)


def _old_context_kl(
    agent: OnlineArm,
    matrix: np.ndarray,
    touched_slots: set[int],
) -> tuple[float, int, dict[int, float]]:
    records = agent.anchors.active_items(
        agent.slot_generation,
        slots=[slot for slot in range(MAX_CONTEXTS) if slot not in touched_slots],
    )
    if not records:
        return 0.0, 0, {}
    obs = np.stack([item["obs"] for item in records])
    slots = np.asarray([item["slot"] for item in records], dtype=np.int64)
    original = _read_matrix(agent.core)
    old_probs = _matrix_probs(agent, agent.committed_matrix, obs, restore=original)
    candidate_probs = _matrix_probs(agent, matrix, obs, restore=original)
    values = _kl_rows(old_probs, candidate_probs)
    by_slot = {int(slot): float(values[slots == slot].mean()) for slot in np.unique(slots)}
    return float(max(by_slot.values(), default=0.0)), len(records), by_slot


def _candidate_anchor_kl(agent: OnlineArm, candidate_matrix: np.ndarray) -> float:
    """Score codecs against prior anchors only, never the current holdout."""
    anchors = _anchor_records(agent)
    if not anchors:
        # Deterministic coefficient-space tie-break before any anchor exists.
        trial = _read_matrix(agent.core)
        return float(np.mean((candidate_matrix - trial) ** 2))
    matrix_obs = np.stack([item["obs"] for item in anchors])
    # Score the current dense trial directly, avoiding an extra matrix copy.
    trial_probs = _policy_probs(agent.core, matrix_obs)
    original = _read_matrix(agent.core)
    candidate_probs = _matrix_probs(agent, candidate_matrix, matrix_obs, restore=original)
    return float(_kl_rows(trial_probs, candidate_probs).mean())


def _encode_trial(agent: OnlineArm, method: str) -> CompactKnowledge:
    matrix = _read_matrix(agent.core)
    if agent.shared_head:
        matrix = matrix[:1]
    return encode(matrix, method, ACTOR_BUDGET_BYTES)


def _gate_metrics(
    agent: OnlineArm,
    compact: CompactKnowledge,
    validation: list[Transition],
    touched_slots: set[int],
) -> dict[str, Any]:
    decoded = compact.decode()
    surrogate, local_count = _local_surrogate(agent, decoded, validation)
    old_kl, old_count, by_slot = _old_context_kl(agent, decoded, touched_slots)
    accepted = (
        local_count > 0
        and math.isfinite(surrogate)
        and surrogate >= 0.0
        and math.isfinite(old_kl)
        and old_kl <= 0.02
    )
    return {
        "accepted": bool(accepted),
        "local_surrogate_improvement": surrogate,
        "local_validation_samples": local_count,
        "old_context_anchor_kl": old_kl,
        "old_context_anchor_samples": old_count,
        "old_context_anchor_kl_by_slot": by_slot,
    }


def _reset_actor_optimizer(agent: OnlineArm) -> int:
    if agent.optimizer is None:
        return 0
    reset = 0
    for parameter in agent.core._residual_parameters():
        if parameter in agent.optimizer.state:
            agent.optimizer.state.pop(parameter, None)
            reset += 1
    agent.counts["optimizer_reset_events"] += 1
    agent.counts["optimizer_reset_parameters"] += reset
    return reset


def _consolidate(
    agent: OnlineArm,
    validation: list[Transition],
    *,
    cycle: int,
) -> dict[str, Any]:
    if agent.arm not in NEURAL_ARMS:
        return {
            "step": cycle * CONSOLIDATE_EVERY,
            "arm": agent.arm,
            "kind": "controller_checkpoint",
            "selected_codec": None,
            "actor_bytes": 0,
            "accepted": None,
        }
    trial = _read_matrix(agent.core)
    touched = {
        int(row.slot) for row in validation
        if row.slot >= 0 and row.generation == agent.slot_generation[row.slot]
    }
    record: dict[str, Any] = {
        "step": cycle * CONSOLIDATE_EVERY,
        "arm": agent.arm,
        "kind": "consolidation",
        "touched_slots": sorted(touched),
        "validation_samples": len(validation),
        "actor_bytes_before": 0 if agent.compact is None else agent.compact.nbytes,
    }

    selected: CompactKnowledge | None = None
    decision = "accepted"
    if agent.arm == "evidence_gate":
        candidates: list[tuple[float, str]] = []
        has_prior_anchors = bool(_anchor_records(agent))
        for method in GATE_CODECS:
            try:
                compact = encode(trial, method, ACTOR_BUDGET_BYTES)
            except ValueError as exc:
                record.setdefault("infeasible_candidates", []).append({"method": method, "reason": str(exc)})
                continue
            trial_matrix = compact.decode()
            score = _candidate_anchor_kl(agent, trial_matrix)
            candidates.append((score, method))
            record.setdefault("candidates", []).append({
                "method": method,
                "actor_bytes": compact.nbytes,
                "selection_score": score,
                "selection_metric": "prior_anchor_kl" if has_prior_anchors else "coefficient_mse",
            })
        candidates.sort(key=lambda item: (item[0], GATE_CODECS.index(item[1])))
        selected_method = None
        if candidates:
            _, selected_method = candidates[0]
            candidate = encode(trial, selected_method, ACTOR_BUDGET_BYTES)
            gate = _gate_metrics(agent, candidate, validation, touched)
            record["selected_candidate"] = selected_method
            record["selected_gate"] = gate
            if gate["accepted"]:
                selected = candidate
            else:
                record["fallback"] = "int8"
                int8_candidate = (
                    encode(trial, "int8", ACTOR_BUDGET_BYTES)
                    if any(item[1] == "int8" for item in candidates) else None
                )
                if int8_candidate is not None and selected_method != "int8":
                    fallback_gate = _gate_metrics(agent, int8_candidate, validation, touched)
                    record["fallback_gate"] = fallback_gate
                    if fallback_gate["accepted"]:
                        selected = int8_candidate
                        record["selected_candidate"] = "int8"
                if selected is None:
                    decision = "rejected"
        else:
            decision = "rejected"
    else:
        codec = CODEC_FOR_ARM[agent.arm]
        selected = encode(trial[:1] if agent.shared_head else trial, codec, ACTOR_BUDGET_BYTES)
        record["selected_candidate"] = codec

    if selected is None:
        agent.counts["rejected"] += 1
        agent.committed_matrix = agent.compact.decode() if agent.compact is not None else np.zeros_like(trial)
        if agent.compact is not None:
            _install_matrix(agent.core, agent.committed_matrix, shared=agent.shared_head)
        record["accepted"] = False
        record["decision"] = "rejected_keep_committed"
        record["selected_codec"] = agent.committed_method
    else:
        if selected.nbytes > ACTOR_BUDGET_BYTES:
            raise RuntimeError("encoded actor knowledge exceeded the fixed budget")
        agent.compact = selected
        agent.committed_method = selected.method
        agent.committed_matrix = selected.decode()
        _install_matrix(agent.core, agent.committed_matrix, shared=agent.shared_head)
        agent.counts["accepted"] += 1
        agent.counts["codec_commits"] += 1
        record["accepted"] = True
        record["decision"] = decision
        record["selected_codec"] = selected.method

    _finalize_gae(validation)
    if agent.arm in NEURAL_ARMS:
        # The held-out rollout becomes next cycle's bounded replay only after
        # the decision, so it cannot validate and fit the same proposal.
        obs_stack = np.stack([row.obs for row in validation]) if validation else np.empty((0, 13), np.float32)
        committed_probs = _policy_probs(agent.core, obs_stack)
        for row, probabilities in zip(validation, committed_probs):
            if row.slot >= 0 and row.generation == agent.slot_generation[row.slot]:
                agent.anchors.add(row.slot, row.generation, row.obs, probabilities)
    reset_parameters = _reset_actor_optimizer(agent)
    record["residual_optimizer_parameters_reset"] = reset_parameters
    record["actor_bytes_after"] = 0 if agent.compact is None else agent.compact.nbytes
    record["actor_budget_bytes"] = ACTOR_BUDGET_BYTES
    record["anchor_bytes_after"] = agent.anchors.nbytes
    record["accepted_count"] = agent.counts["accepted"]
    record["rejected_count"] = agent.counts["rejected"]
    return record


def _critic_state(core: TensorSourceBasisPolicy) -> dict[str, torch.Tensor]:
    state = core.state_dict()
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if name.startswith("mlp_extractor.value_net.") or name.startswith("value_net.")
    }


def _load_critic_state(core: TensorSourceBasisPolicy, state: Mapping[str, torch.Tensor]) -> None:
    live = core.state_dict()
    for name, value in state.items():
        if name not in live:
            raise ValueError(f"checkpoint critic tensor {name!r} is absent from the policy")
        live[name].copy_(value)
    core.load_state_dict(live, strict=True)


def _rng_state(agent: OnlineArm) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "action_generator": copy.deepcopy(agent.action_rng.bit_generator.state),
    }


def _checkpoint_payload(agent: OnlineArm) -> dict[str, Any]:
    # This payload intentionally excludes the dense six-head trial workspace.
    return {
        "format_version": 1,
        "arm": agent.arm,
        "seed": agent.seed,
        "foundation_sha256": _hash(Path(agent.base_config["foundation"])),
        "compact_actor": None if agent.compact is None else agent.compact.to_payload(),
        "critic": _critic_state(agent.core),
        "optimizer": None if agent.optimizer is None else agent.optimizer.state_dict(),
        "router": None if agent.controller is not None else agent.router.state_dict(),
        "controller": None if agent.controller is None else agent.controller.snapshot(),
        "anchors": agent.anchors.state_dict(),
        "slot_generation": agent.slot_generation.copy(),
        "counts": copy.deepcopy(agent.counts),
        "committed_method": agent.committed_method,
        "rng": _rng_state(agent),
    }


def save_checkpoint(agent: OnlineArm, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(_checkpoint_payload(agent), temporary)
    os.replace(temporary, path)
    return path.stat().st_size


def restore_agent_checkpoint(path: str | Path, base_config: Mapping[str, Any]) -> OnlineArm:
    """Restore policy/router/optimizer/RNG state for evaluation or revisit.

    This is not a mid-run training-resume checkpoint: environment state, hidden
    schedule position, and an in-flight rollout are intentionally not stored.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("unsupported knowledge-loop checkpoint")
    expected_foundation = _hash(Path(base_config["foundation"]))
    if payload.get("foundation_sha256") != expected_foundation:
        raise ValueError("checkpoint foundation hash does not match the configured W0 source")
    arm = str(payload["arm"])
    agent = OnlineArm(arm, int(payload["seed"]), base_config)
    compact_payload = payload.get("compact_actor")
    if compact_payload is not None:
        agent.compact = CompactKnowledge.from_payload(compact_payload)
        agent.committed_matrix = agent.compact.decode()
        agent.committed_method = str(payload["committed_method"])
        _install_matrix(agent.core, agent.committed_matrix, shared=agent.shared_head)
    _load_critic_state(agent.core, payload["critic"])
    if agent.optimizer is not None and payload.get("optimizer") is not None:
        agent.optimizer.load_state_dict(payload["optimizer"])
    if payload.get("router") is not None:
        agent.router = TaskFreeRouter.from_state_dict(payload["router"])
    if payload.get("controller") is not None:
        agent.controller = C1ModelController.from_snapshot(payload["controller"])
    agent.anchors = AnchorReplay.from_state_dict(payload["anchors"])
    agent.slot_generation = np.asarray(payload["slot_generation"], dtype=np.int64).copy()
    agent.counts = copy.deepcopy(payload["counts"])
    agent.action_rng.bit_generator.state = copy.deepcopy(payload["rng"]["action_generator"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy_global"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    return agent


def _router_bytes(agent: OnlineArm) -> dict[str, int]:
    if agent.controller is not None:
        summary = agent.controller.memory_summary()
        return {
            "persistent": int(summary["state_array_bytes"]),
            "working_history": int(summary["history_python_bytes"]),
            "total": int(summary["estimated_controller_bytes"]),
        }
    return agent.router.nbytes()


def _rss_bytes() -> int:
    """Return process resident set size at a recorded experiment boundary."""
    return int(psutil.Process().memory_info().rss)


def _optimizer_bytes(optimizer: torch.optim.Optimizer | None) -> tuple[int, int]:
    if optimizer is None:
        return 0, 0
    residual_ids = {
        id(parameter)
        for group in optimizer.param_groups for parameter in group["params"]
        if getattr(parameter, "tree_residual", False)
    }
    actor = 0
    critic = 0
    for parameter, state in optimizer.state.items():
        size = sum(int(value.numel() * value.element_size()) for value in state.values() if torch.is_tensor(value))
        if id(parameter) in residual_ids:
            actor += size
        else:
            critic += size
    return actor, critic


def _capacity_row(agent: OnlineArm, step: int, rollout_rows: int) -> dict[str, Any]:
    actor_bytes = 0 if agent.compact is None else int(agent.compact.nbytes)
    actor_workspace = int(6 * 27 * 4) if agent.arm in NEURAL_ARMS else 0
    committed_cache = int(agent.committed_matrix.nbytes) if agent.compact is not None else 0
    trial_snapshot = int(6 * 27 * 4) if agent.arm in NEURAL_ARMS else 0
    gate_candidate_workspace = int(2 * 6 * 27 * 4) if agent.arm == "evidence_gate" else 0
    candidate_codec_workspace = ACTOR_BUDGET_BYTES if agent.arm == "evidence_gate" else 0
    actor_adam, critic_adam = _optimizer_bytes(agent.optimizer)
    critic_weights = sum(
        int(tensor.numel() * tensor.element_size())
        for name, tensor in _critic_state(agent.core).items()
    )
    router = _router_bytes(agent)
    router_numeric = int(router.get("persistent", router.get("profile_counts", 0)))
    if agent.controller is not None:
        anchor_numeric = agent.anchors.payload_nbytes
        anchor_python = agent.anchors.nbytes - anchor_numeric
    else:
        anchor_numeric = agent.anchors.payload_nbytes
        anchor_python = agent.anchors.nbytes - anchor_numeric
    # Numeric fields only; Python object/container overhead is not included.
    rollout_bytes = int(rollout_rows * (13 * 4 + 4 + 4 * 8 + 4 * 4 + 1 + 1 + 3 * 8 + 2 * 8))
    return {
        "step": step,
        "arm": agent.arm,
        "actor_compact_bytes": actor_bytes,
        "actor_budget_bytes": ACTOR_BUDGET_BYTES,
        "dense_trial_workspace_bytes": actor_workspace,
        "decoded_committed_cache_bytes": committed_cache,
        "consolidation_trial_snapshot_bytes": trial_snapshot,
        "gate_candidate_decode_workspace_bytes": gate_candidate_workspace,
        "candidate_codec_workspace_bytes": candidate_codec_workspace,
        "actor_optimizer_state_bytes": actor_adam,
        "critic_weights_bytes": critic_weights,
        "critic_optimizer_state_bytes": critic_adam,
        "router_persistent_bytes": router.get("persistent", router.get("profile_counts", 0)),
        "router_working_bytes": router.get("working_history", router.get("total", 0) - router.get("profile_counts", 0)),
        "router_total_bytes": router.get("total", 0),
        "anchor_replay_bytes": agent.anchors.nbytes,
        "anchor_numeric_payload_bytes": anchor_numeric,
        "anchor_python_container_estimate_bytes": anchor_python,
        "ppo_temporary_rollout_estimate_bytes": rollout_bytes,
        "peak_neural_working_estimate_bytes": (
            actor_workspace + committed_cache + trial_snapshot + gate_candidate_workspace
            + candidate_codec_workspace + rollout_bytes
        ),
        "numeric_persistent_state_estimate_bytes": (
            actor_bytes + actor_workspace + committed_cache + actor_adam + critic_weights
            + critic_adam + router_numeric + anchor_numeric
        ),
        "python_container_estimate_bytes": (
            max(0, int(router.get("total", 0)) - router_numeric) + anchor_python
        ),
        "transient_numeric_workspace_estimate_bytes": (
            trial_snapshot + gate_candidate_workspace + candidate_codec_workspace + rollout_bytes
        ),
        "process_rss_bytes": _rss_bytes(),
        "compact_total_persistent_estimate_bytes": (
            actor_bytes + actor_workspace + committed_cache + actor_adam + critic_weights + critic_adam
            + router.get("total", 0) + agent.anchors.nbytes
        ),
    }


def _save_anchor_codec(agent: OnlineArm, output: Path, step: int) -> dict[str, int]:
    if agent.compact is None:
        return {"binary_bytes": 0, "npz_bytes": 0}
    binary_path = output / f"actor_knowledge_step_{step:08d}.bin"
    npz_path = output / f"actor_knowledge_step_{step:08d}.npz"
    agent.compact.save_binary(binary_path)
    agent.compact.save(npz_path)
    if binary_path.stat().st_size != agent.compact.nbytes:
        raise RuntimeError("binary compact actor file does not match its exact encoded byte count")
    return {"binary_bytes": binary_path.stat().st_size, "npz_bytes": npz_path.stat().st_size}


def _metrics_for_controller(agent: OnlineArm) -> dict[str, Any]:
    if agent.controller is None:
        return {}
    return {
        "confidence": float(agent.controller.confidence),
        "statistical_updates": int(agent.controller.statistical_update_count),
        "change_detections": int(agent.controller.change_detections),
        "surprise": float(agent.controller.last_surprise),
    }


def _episode_row(step: int, episode: int, reward: float, length: int, info: Mapping[str, Any], contexts: set[str]) -> dict[str, Any]:
    return {
        "step": step,
        "episode": episode,
        "return": reward,
        "success": int(bool(info.get("success", False))),
        "length": length,
        "mixed_context": int(len(contexts) > 1),
        "hidden_contexts": "|".join(sorted(contexts)),
    }


def run_arm(
    arm: str,
    *,
    seed: int,
    total_steps: int,
    schedule: list[dict[str, Any]],
    base_config: Mapping[str, Any],
    output: Path,
    phase: str,
    run_revisit: bool,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    agent = OnlineArm(arm, seed, base_config)
    env = C1NavigationEnv()
    obs, _ = env.reset(seed=seed + 2200)
    step_file = (output / "steps.csv").open("w", newline="", encoding="utf-8")
    episode_file = (output / "episodes.csv").open("w", newline="", encoding="utf-8")
    commit_file = (output / "commits.csv").open("w", newline="", encoding="utf-8")
    capacity_file = (output / "capacity.csv").open("w", newline="", encoding="utf-8")
    step_fields = [
        "step", "arm", "hidden_context", "segment_id", "route_slot", "route_id",
        "route_confidence", "route_support", "route_status", "route_reason",
        "action", "reward", "terminated", "truncated", "episode", "allocation_slot",
        "evicted_slot", "slot_generation", "policy_update_count",
    ]
    episode_fields = ["step", "episode", "return", "success", "length", "mixed_context", "hidden_contexts"]
    commit_fields = [
        "step", "arm", "kind", "selected_codec", "actor_bytes_before", "actor_bytes_after",
        "actor_budget_bytes", "validation_samples", "accepted", "decision", "touched_slots",
        "selected_candidate", "selected_gate", "candidates", "fallback", "fallback_gate",
        "infeasible_candidates", "residual_optimizer_parameters_reset", "checkpoint_file_bytes",
        "actor_binary_file_bytes", "actor_npz_file_bytes",
    ]
    capacity_fields = [
        "step", "arm", "actor_compact_bytes", "actor_budget_bytes", "dense_trial_workspace_bytes",
        "decoded_committed_cache_bytes", "consolidation_trial_snapshot_bytes",
        "gate_candidate_decode_workspace_bytes", "candidate_codec_workspace_bytes",
        "actor_optimizer_state_bytes", "critic_weights_bytes", "critic_optimizer_state_bytes",
        "router_persistent_bytes", "router_working_bytes", "router_total_bytes", "anchor_replay_bytes",
        "anchor_numeric_payload_bytes", "anchor_python_container_estimate_bytes",
        "ppo_temporary_rollout_estimate_bytes", "peak_neural_working_estimate_bytes",
        "numeric_persistent_state_estimate_bytes", "python_container_estimate_bytes",
        "transient_numeric_workspace_estimate_bytes", "process_rss_bytes",
        "compact_total_persistent_estimate_bytes",
    ]
    step_writer, episode_writer = csv.DictWriter(step_file, fieldnames=step_fields), csv.DictWriter(episode_file, fieldnames=episode_fields)
    commit_writer, capacity_writer = csv.DictWriter(commit_file, fieldnames=commit_fields), csv.DictWriter(capacity_file, fieldnames=capacity_fields)
    step_writer.writeheader(); episode_writer.writeheader(); commit_writer.writeheader(); capacity_writer.writeheader()

    current_rollout: list[Transition] = []
    validation_rollout: list[Transition] = []
    last_completed_rollout_rows = 0
    episode_return, episode_steps, episode = 0.0, 0, 0
    episode_contexts: set[str] = set()
    episode_successes = 0
    reward_total = 0.0
    completed_episodes = 0
    start_time = time.perf_counter()
    cpu_start = time.process_time()
    rss_after_initialization = _rss_bytes()
    last_capacity_rows = 0

    for zero_step in range(total_steps):
        segment = hidden_context_at(schedule, zero_step)
        hidden_name = str(segment["context"])
        env.set_context(C1_CONTEXTS[hidden_name])
        route = agent.route_before_action(obs) if agent.controller is None else {
            "slot": -1, "confidence": float(agent.controller.confidence), "support": 0.0,
            "status": "model_controller", "reason": "adaptive_assignment",
        }
        action, policy_obs, slot, probabilities, value, log_prob = agent.action(obs, route)
        generation = int(agent.slot_generation[slot]) if 0 <= slot < MAX_CONTEXTS else -1
        next_obs, reward, terminated, truncated, info = env.step(action)
        allocation = agent.observe_transition(obs, action, float(reward), next_obs)
        done = bool(terminated or truncated)
        next_value = bootstrap_value(agent, next_obs)
        transition = Transition(
            obs=np.asarray(policy_obs, dtype=np.float32).copy(),
            action=action,
            reward=float(reward),
            value=value,
            next_value=next_value,
            log_prob=log_prob,
            probabilities=probabilities.copy(),
            done=done,
            terminated=bool(terminated),
            slot=slot,
            generation=generation,
            step=zero_step + 1,
        )
        if arm in NEURAL_ARMS:
            current_rollout.append(transition)
            if len(current_rollout) == ROLLOUT_STEPS:
                last_completed_rollout_rows = len(current_rollout)
                rollout_number = rollout_number_at_step(zero_step + 1)
                _finalize_gae(current_rollout)
                if rollout_number <= 3:
                    agent.counts["local_trial_steps"] += len(current_rollout)
                    _ppo_update(agent, current_rollout)
                else:
                    validation_rollout.extend(current_rollout)
                    agent.counts["validation_steps"] += len(current_rollout)
                current_rollout = []

        agent.counts["environment_steps"] += 1
        reward_total += float(reward)
        episode_return += float(reward)
        episode_steps += 1
        episode_contexts.add(hidden_name)
        steps_done = zero_step + 1
        step_writer.writerow({
            "step": steps_done,
            "arm": arm,
            "hidden_context": hidden_name,
            "segment_id": segment["segment_id"],
            "route_slot": slot,
            "route_id": (1 if agent.shared_head else slot + 1 if slot >= 0 else 0),
            "route_confidence": route.get("confidence", ""),
            "route_support": route.get("support", ""),
            "route_status": route.get("status", ""),
            "route_reason": route.get("reason", ""),
            "action": action,
            "reward": reward,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "episode": episode,
            "allocation_slot": "" if allocation is None or allocation.get("allocated_slot") is None else allocation.get("allocated_slot"),
            "evicted_slot": "" if allocation is None or allocation.get("evicted_slot") is None else allocation.get("evicted_slot"),
            "slot_generation": generation,
            "policy_update_count": agent.counts["minibatch_updates"],
        })
        if done:
            episode += 1
            completed_episodes += 1
            episode_successes += int(bool(info.get("success", False)))
            episode_writer.writerow(_episode_row(steps_done, episode, episode_return, episode_steps, info, episode_contexts))
            episode_return, episode_steps, episode_contexts = 0.0, 0, set()
            obs, _ = env.reset()
        else:
            obs = next_obs

        if arm in NEURAL_ARMS and steps_done % CONSOLIDATE_EVERY == 0:
            cycle = steps_done // CONSOLIDATE_EVERY
            commit = _consolidate(agent, validation_rollout, cycle=cycle)
            codec_file_bytes = _save_anchor_codec(agent, output, steps_done)
            checkpoint_size = save_checkpoint(agent, output / "checkpoint_latest.pt")
            commit["checkpoint_file_bytes"] = checkpoint_size
            commit["actor_binary_file_bytes"] = codec_file_bytes["binary_bytes"]
            commit["actor_npz_file_bytes"] = codec_file_bytes["npz_bytes"]
            commit_writer.writerow({key: _csv_value(commit.get(key, "")) for key in commit_fields})
            capacity_writer.writerow(_capacity_row(agent, steps_done, len(validation_rollout)))
            validation_rollout = []
            last_capacity_rows = 0
        elif steps_done % ROLLOUT_STEPS == 0:
            capacity_writer.writerow(_capacity_row(agent, steps_done, last_completed_rollout_rows))
            last_capacity_rows = len(current_rollout)
            last_completed_rollout_rows = 0

    # All sanctioned durations are exact multiples of the consolidation cycle.
    if current_rollout or validation_rollout:
        raise RuntimeError("run ended with an incomplete PPO/consolidation cycle")
    elapsed = time.perf_counter() - start_time
    cpu_seconds = time.process_time() - cpu_start
    router_metrics = (
        agent.controller.memory_summary() if agent.controller is not None
        else {**agent.router.nbytes(), "allocations": agent.router.allocations,
              "evictions": agent.router.evictions, "switches": agent.router.switches,
              "active_slots": list(agent.router.active_slots)}
    )
    checkpoint_size = save_checkpoint(agent, output / "checkpoint_latest.pt")
    compact_file_bytes = _save_anchor_codec(agent, output, total_steps)
    summary = {
        "phase": phase,
        "seed": seed,
        "arm": arm,
        "steps": total_steps,
        "total_reward": reward_total,
        "reward_per_step": reward_total / max(1, total_steps),
        "completed_episodes": completed_episodes,
        "success_rate": episode_successes / max(1, completed_episodes),
        "training_environment_steps": total_steps,
        "validation_environment_steps": int(agent.counts["validation_steps"]),
        "revisit_environment_steps": 0,
        "elapsed_seconds": elapsed,
        "process_cpu_seconds": cpu_seconds,
        "process_rss_after_initialization_bytes": rss_after_initialization,
        "process_rss_final_bytes": _rss_bytes(),
        "steps_per_cpu_second": total_steps / max(cpu_seconds, 1e-9),
        "actor_compact_numeric_bytes": 0 if agent.compact is None else agent.compact.nbytes,
        "actor_compact_binary_file_bytes": compact_file_bytes["binary_bytes"],
        "actor_compact_npz_file_bytes": compact_file_bytes["npz_bytes"],
        "actor_budget_bytes": ACTOR_BUDGET_BYTES,
        "critic_weight_bytes": sum(int(t.numel() * t.element_size()) for t in _critic_state(agent.core).values()),
        "critic_optimizer_bytes": _optimizer_bytes(agent.optimizer)[1],
        "anchor_replay_bytes": agent.anchors.nbytes,
        "router_memory": router_metrics,
        "counts": agent.counts,
        "slot_generation": agent.slot_generation.tolist(),
        "checkpoint_file_bytes": checkpoint_size,
        "source_snapshot_manifest": "source_snapshot/manifest.json",
        "hidden_schedule": schedule,
        "foundation_sha256": _hash(Path(base_config["foundation"])),
        "protocol_sha256": _hash(ROOT / "docs" / "KNOWLEDGE_ORGANIZATION_PROTOCOL.md"),
        "custom_ppo": {
            "rollout": ROLLOUT_STEPS, "batch": BATCH_SIZE, "epochs": PPO_EPOCHS,
            "learning_rate": LEARNING_RATE, "entropy": ENTROPY_COEF, "gamma": GAMMA,
            "gae": GAE_LAMBDA, "clip": CLIP_RANGE, "target_kl": TARGET_KL,
            "fit_rollouts_per_cycle": 3, "heldout_rollouts_per_cycle": 1,
            "manual_implementation_notes": [
                "Advantages are normalized once across each 256-step rollout.",
                "The value loss is MSE with coefficient 0.5; target-KL stopping is checked before the minibatch update.",
                "Actor samples from a replaced router generation are masked; the shared critic still uses them.",
                "No vectorized environments or SB3 rollout buffer are used in this single-stream harness.",
            ],
        },
        "checkpoint_scope": (
            "policy, router/controller, bounded anchors, optimizer and RNG restoration for revisit; "
            "not a mid-run resume because environment, schedule position, and rollout state are not saved"
        ),
        "source_prior_note": (
            "adaptive_assignment uses the existing source-prior transition controller; "
            "its online state is counted separately and it has no actor correction package."
            if arm == "adaptive_assignment" else ""
        ),
        "limitations": [
            "C1 is a controlled six-context navigation simulator, not a general deployment benchmark.",
            "The evidence gate is a finite empirical heuristic, not a certified safety guarantee.",
            "Output-policy KL is used to rank compact candidates and check preservation; no gradient policy distillation is performed.",
            "The adaptive_assignment baseline uses a source-transition prior and has a different state representation.",
            "The checkpoint restores policy, router/controller, anchors, optimizer, and RNG for revisit; it lacks environment, schedule-position, and rollout state for arbitrary mid-run resume.",
            "RSS samples are measured process resident-set sizes; component capacity fields are numeric payload or bounded Python-container estimates and exclude the shared W0 foundation allocation.",
        ],
    }
    _write_json(output / "summary.json", summary)
    step_file.close(); episode_file.close(); commit_file.close(); capacity_file.close()
    env.close()
    return summary


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=_json_default)
    return value


def _evaluation_action(
    core: TensorSourceBasisPolicy,
    obs: np.ndarray,
    router: TaskFreeRouter,
    arm: str,
    rng: np.random.Generator,
) -> tuple[int, dict[str, Any]]:
    route = router.route(obs)
    slot = int(route["slot"])
    route_id = 1 if arm == "single_head_online" else slot + 1 if slot >= 0 else 0
    policy_obs = torch.as_tensor(augment_observation(obs, route_id), dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        probs = core.get_distribution(policy_obs).distribution.probs.reshape(-1).cpu().numpy()
    action = int(rng.choice(4, p=probs / probs.sum()))
    return action, route


def run_paired_revisit(
    agent: OnlineArm,
    *,
    seed: int,
    output: Path,
    episodes_per_context: int = 8,
) -> dict[str, Any]:
    """Evaluate retained compact corrections against zero corrections."""
    if agent.controller is not None:
        return {
            "status": "not_applicable",
            "reason": "adaptive_assignment has no compact actor corrections or task-free router",
            "training_steps": 0,
        }
    rss_start = _rss_bytes()
    base_router_state = agent.router.state_dict()
    retained_matrix = agent.committed_matrix.copy()
    results: list[dict[str, Any]] = []
    total_steps = 0
    for context_position, name in enumerate(C1_CONTEXTS):
        for condition in ("retain_parameters", "zero_corrections"):
            router = TaskFreeRouter.from_state_dict(base_router_state)
            router.reset_history()
            if condition == "retain_parameters":
                matrix = retained_matrix
            else:
                matrix = np.zeros_like(retained_matrix)
            _install_matrix(agent.core, matrix, shared=agent.shared_head)
            env = C1NavigationEnv(C1_CONTEXTS[name])
            # Identical per-context RNG streams pair the two conditions.
            local_rng = np.random.default_rng(seed + 99000 + context_position * 200)
            returns: list[float] = []
            successes: list[int] = []
            interactions = 0
            for episode_idx in range(episodes_per_context):
                obs, _ = env.reset(seed=seed + 88000 + context_position * 100 + episode_idx)
                episode_return = 0.0
                for _ in range(env.max_steps):
                    action, route = _evaluation_action(agent.core, obs, router, agent.arm, local_rng)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    router.observe(obs, action, float(reward), next_obs)
                    episode_return += float(reward)
                    interactions += 1
                    obs = next_obs
                    if terminated or truncated:
                        returns.append(episode_return)
                        successes.append(int(bool(info["success"])))
                        break
            env.close()
            total_steps += interactions
            results.append({
                "context": name,
                "condition": condition,
                "episodes": episodes_per_context,
                "mean_return": float(np.mean(returns)),
                "success_rate": float(np.mean(successes)),
                "environment_steps": interactions,
                "router_allocations": router.allocations,
                "router_evictions": router.evictions,
            })
    _install_matrix(agent.core, retained_matrix, shared=agent.shared_head)
    _write_json(output / "revisit.json", {
        "status": "completed",
        "episodes_per_context_per_condition": episodes_per_context,
        "strategy_parameter_updates": 0,
        "route_learning_enabled": True,
        "total_environment_steps": total_steps,
        "process_rss_start_bytes": rss_start,
        "process_rss_end_bytes": _rss_bytes(),
        "paired_rows": results,
    })
    return {
        "status": "completed", "total_environment_steps": total_steps,
        "process_rss_start_bytes": rss_start, "process_rss_end_bytes": _rss_bytes(),
        "rows": results,
    }


def load_base_config() -> dict[str, Any]:
    config = json.loads((ROOT / "configs" / "c1.json").read_text(encoding="utf-8"))
    config["foundation"] = str((ROOT / config["foundation"]).resolve())
    config["dataset"] = str((ROOT / config["dataset"]).resolve())
    return config


def run_phase(
    phase: str,
    *,
    seeds: Iterable[int] | None = None,
    arms: Iterable[str] | None = None,
    steps: int | None = None,
    output_dir: str | Path | None = None,
    run_revisit: bool | None = None,
) -> dict[str, Any]:
    if phase not in ("pilot", "formal"):
        raise ValueError("phase must be pilot or formal")
    if phase == "formal":
        seeds = tuple(seeds if seeds is not None else (401, 419, 433))
        arms = tuple(arms if arms is not None else ALL_ARMS)
        steps = int(steps if steps is not None else 32768)
        run_revisit = True if run_revisit is None else bool(run_revisit)
        if tuple(seeds) != (401, 419, 433) or steps != 32768 or tuple(arms) != ALL_ARMS:
            raise ValueError("formal protocol is frozen to seeds 401, 419, 433; all seven arms; 32,768 steps")
        if not run_revisit:
            raise ValueError("formal runs require the paired revisit evaluation")
    else:
        seeds = tuple(seeds if seeds is not None else (997,))
        arms = tuple(arms if arms is not None else ("frozen", "int8", "evidence_gate", "adaptive_assignment"))
        steps = int(steps if steps is not None else 2048)
        run_revisit = False if run_revisit is None else bool(run_revisit)
        if tuple(seeds) != (997,) or steps != 2048:
            raise ValueError("pilot protocol is frozen to seed 997 and 2,048 steps")
    unknown = set(arms) - set(ALL_ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    if steps % CONSOLIDATE_EVERY:
        raise ValueError("steps must be an exact multiple of the 1,024-step consolidation interval")
    target = Path(output_dir) if output_dir is not None else ROOT / f"outputs/knowledge_loop_{phase}"
    base_config = load_base_config()
    all_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        schedule = make_hidden_schedule(steps, seed + 7000)
        for arm in arms:
            arm_output = target / f"seed_{seed}" / arm
            # Preserve an existing failed run rather than silently replacing it.
            if arm_output.exists() and any(arm_output.iterdir()):
                raise FileExistsError(f"output already exists: {arm_output}")
            source_manifest = _snapshot_sources(arm_output, base_config)
            config_record = {
                "phase": phase, "seed": seed, "arm": arm, "steps": steps,
                "hidden_schedule_seed": seed + 7000, "hidden_schedule": schedule,
                "arm_order": list(arms), "actor_budget_bytes": ACTOR_BUDGET_BYTES,
                "protocol_sha256": _hash(ROOT / "docs" / "KNOWLEDGE_ORGANIZATION_PROTOCOL.md"),
                "foundation_sha256": _hash(Path(base_config["foundation"])),
                "source_snapshot_manifest": "source_snapshot/manifest.json",
                "source_snapshot_file_count": len(source_manifest["sources"]),
            }
            _write_json(arm_output / "config.json", config_record)
            summary = run_arm(
                arm, seed=seed, total_steps=steps, schedule=schedule,
                base_config=base_config, output=arm_output, phase=phase,
                run_revisit=bool(run_revisit),
            )
            if run_revisit:
                # Re-create the learner from its compact checkpoint so the
                # paired test uses the saved canonical representation.
                restored = restore_agent_checkpoint(arm_output / "checkpoint_latest.pt", base_config)
                revisit = run_paired_revisit(restored, seed=seed, output=arm_output)
                summary["revisit"] = revisit
                summary["revisit_environment_steps"] = int(revisit.get("total_environment_steps", 0))
                _write_json(arm_output / "summary.json", summary)
            all_summaries.append(summary)
    aggregate = {
        "phase": phase,
        "seeds": list(seeds),
        "arms": list(arms),
        "steps_per_run": steps,
        "runs": all_summaries,
        "paired_by_seed": True,
        "interpretation": "mechanism study on C1; no SOTA or certified-safety claim",
    }
    _write_json(target / "summary.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-revisit", action="store_true")
    args = parser.parse_args()
    revisit = False if args.no_revisit else (None if args.phase == "pilot" else True)
    run_phase(args.phase, output_dir=args.output_dir, run_revisit=revisit)


if __name__ == "__main__":
    main()

"""Run the frozen Step 1 write-decision comparison.

This runner deliberately keeps task labels in its external schedule/router.
Only bounded transition and reward profiles are passed to group assignment.
"""
from __future__ import annotations

import argparse
import contextlib
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
from online_rl.write_decision_envs import make_env, task_from_name
from online_rl.write_decision_evidence import EvidenceWindow, RewardAwareGroups
from online_rl.write_decision_gate import AnchorBank, evaluate_change, project_delta
from online_rl.write_decision_policy import FlexibleSharedPolicy, decode_actor, encode_actor
from scripts import run_knowledge_loop as loop


CONFIG_PATH = ROOT / "configs" / "write_decision.json"
PROTOCOL_PATH = ROOT / "docs" / "WRITE_DECISION_PROTOCOL.md"
TASK_NAMES = ("A0+", "A0-", "A1+", "A1-", "A2+", "A2-")
SIMPLE_BASELINES = ("independent_fast", "dynamics_shared", "reward_shared")
PROPOSED_CONDITIONS = ("trial_gate", "trial_gate_relaxed", "projected_gate", "local_promote")
VALID_CONDITIONS = (
    "independent_fast",
    "dynamics_shared",
    "reward_shared",
    "trial_gate",
    "trial_gate_relaxed",
    "projected_gate",
    "local_promote",
    "trial_rollback",
)
SOURCE_FILES = (
    Path("scripts/run_write_decision.py"),
    Path("scripts/run_knowledge_loop.py"),
    Path("scripts/run_shared_write.py"),
    Path("src/online_rl/write_decision_policy.py"),
    Path("src/online_rl/write_decision_evidence.py"),
    Path("src/online_rl/write_decision_gate.py"),
    Path("src/online_rl/write_decision_envs.py"),
    Path("src/online_rl/shared_write_policy.py"),
    Path("src/online_rl/shared_write_storage.py"),
    Path("src/online_rl/shared_write_grouping.py"),
    Path("src/online_rl/c1_supplement_experiment.py"),
    Path("src/online_rl/c1_tensor_controls.py"),
    Path("src/online_rl/c1_policy.py"),
    Path("src/online_rl/compression.py"),
    Path("src/online_rl/c1_envs.py"),
    Path("src/online_rl/experiment.py"),
    Path("src/online_rl/policy.py"),
    Path("configs/write_decision.json"),
    Path("configs/c1.json"),
    Path("docs/WRITE_DECISION_PROTOCOL.md"),
)
ROLLOUT_STEPS = loop.ROLLOUT_STEPS
CYCLE_STEPS = 4 * ROLLOUT_STEPS


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _write_jsonl(stream, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, default=_json_default) + "\n")
    stream.flush()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_sources(folder: Path, base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy runner dependencies and record the exact W0/config inputs."""
    snapshot_root = folder / "source_snapshot"
    files: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash, copied_hash = _sha(source), _sha(destination)
        if source_hash != copied_hash:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        files[relative.as_posix()] = {
            "sha256": copied_hash,
            "bytes": destination.stat().st_size,
        }
    foundation = Path(base_config["foundation"]).resolve()
    dataset = Path(base_config["dataset"]).resolve()
    manifest = {
        "sources": files,
        "foundation": {
            "path": str(foundation),
            "sha256": _sha(foundation),
            "bytes": foundation.stat().st_size,
        },
        "dataset": {
            "path": str(dataset),
            "sha256": _sha(dataset),
            "bytes": dataset.stat().st_size,
        },
    }
    _write_json(snapshot_root / "manifest.json", manifest)
    return manifest


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _check_condition(condition: str) -> str:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {VALID_CONDITIONS}")
    return condition


def _slot_map(seed: int) -> dict[str, int]:
    order = np.random.default_rng(int(seed) + 8811).permutation(len(TASK_NAMES))
    return {name: int(order[index]) for index, name in enumerate(TASK_NAMES)}


def _make_agent(
    condition: str,
    seed: int,
    config: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any] | None = None,
    slot_map: Mapping[str, int] | None = None,
):
    """Create one frozen-W0 policy, bounded evidence stores, and LeafAdam."""
    condition = _check_condition(condition)
    seed = int(seed)
    seed_everything(seed)
    base = dict(base_config if base_config is not None else loop.load_base_config())
    reference = _foundation(base, seed)
    try:
        groups = 0 if condition == "independent_fast" else 2
        core = FlexibleSharedPolicy.from_reference(
            reference.policy,
            local_slots=6,
            shared_groups=groups,
            exploration_epsilon=float(base.get("exploration_epsilon", 0.05)),
            temperature=float(base.get("temperature", 2.0)),
            train_critic=True,
        )
    finally:
        reference.env.close()

    core._set_trainable_flags(train_critic=True)
    actual_slots = dict(slot_map or _slot_map(seed))
    local_parameters = [core._nodes[str(slot + 1)].A for slot in range(6)]
    shared_parameters = list(core.shared_parameters())
    actor_ids = {id(parameter) for parameter in (*local_parameters, *shared_parameters)}
    critic_parameters = [
        parameter
        for parameter in core.parameters()
        if parameter.requires_grad and id(parameter) not in actor_ids
    ]
    local_lr = float(
        config["fast_local_learning_rate"]
        if condition == "independent_fast"
        else config["local_learning_rate"]
    )
    parameter_groups: list[dict[str, Any]] = [
        {"params": local_parameters, "lr": local_lr, "name": "local"},
    ]
    if shared_parameters:
        parameter_groups.append({
            "params": shared_parameters,
            "lr": float(config["shared_learning_rate"]),
            "name": "shared",
        })
    parameter_groups.append({
        "params": critic_parameters,
        "lr": float(config["critic_learning_rate"]),
        "name": "critic",
    })
    optimizer = LeafAdam(parameter_groups, eps=1e-5)
    mode = "dynamics" if condition == "dynamics_shared" else "reward"
    profiles = {
        "movement_counts": np.zeros((6, 3, 4, 4), dtype=np.uint16),
        "reward_xtx": np.zeros((6, 3, 3), dtype=np.float32),
        "reward_xty": np.zeros((6, 3), dtype=np.float32),
        "reward_n": np.zeros(6, dtype=np.float32),
        "seen": np.zeros(6, dtype=np.uint8),
        "shadow_group": np.full(6, -1, dtype=np.int8),
    }
    counts = {
        "environment_steps": 0,
        "probe_steps": 0,
        "evaluation_steps": 0,
        "fit_steps": 0,
        "holdout_steps": 0,
        "rollouts_trained": 0,
        "minibatch_updates": 0,
        "stale_actor_rows": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "trial_rollback_count": 0,
        "provisional_accept_count": 0,
        "decision_count": 0,
        "direct_commit_count": 0,
        "completed_episodes": 0,
        "positive_completed_episodes": 0,
        "positive_goal_episodes": 0,
        "all_goal_episodes": 0,
    }
    agent = SimpleNamespace(
        arm=condition,
        condition=condition,
        seed=seed,
        core=core,
        optimizer=optimizer,
        action_rng=np.random.default_rng(seed + 1100),
        slot_map=actual_slots,
        slot_generation=np.zeros(6, dtype=np.int64),
        anchors=AnchorBank(
            slots=6,
            per_slot=int(config["anchors_per_slot"]),
            seed=seed + 779,
        ),
        grouper=RewardAwareGroups(max_groups=2, mode=mode),
        profiles=profiles,
        profile_reasons=["not_profiled"] * 6,
        counts=counts,
        next_episode_id=0,
        reward_by_task={name: 0.0 for name in TASK_NAMES},
        steps_by_task={name: 0 for name in TASK_NAMES},
        goal_episodes_by_task={name: 0 for name in TASK_NAMES},
        complete_episodes_by_task={name: 0 for name in TASK_NAMES},
        decision_cpu_s=0.0,
        evaluation_cpu_s=0.0,
        base_config=base,
        run_config=dict(config),
    )
    if sorted(actual_slots.values()) != list(range(6)) or set(actual_slots) != set(TASK_NAMES):
        raise ValueError("slot_map must map all six tasks to a permutation of slots zero through five")
    return agent


def make_agent(
    condition: str,
    seed: int,
    config: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any] | None = None,
    slot_map: Mapping[str, int] | None = None,
):
    """Public factory for the reusable write-decision learner state."""
    return _make_agent(
        condition, seed, config, base_config=base_config, slot_map=slot_map
    )


def _actor_arrays(core: FlexibleSharedPolicy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        core.read_local_matrix(),
        core.read_shared_matrix(),
        core.slot_groups.detach().cpu().numpy().astype(np.int8, copy=True),
    )


def _install_actor_arrays(
    core: FlexibleSharedPolicy,
    local: np.ndarray,
    shared: np.ndarray,
    groups: np.ndarray | None = None,
) -> None:
    core.install_local_matrix(local)
    core.install_shared_matrix(shared)
    if groups is not None:
        with torch.no_grad():
            core.slot_groups.copy_(torch.as_tensor(groups, dtype=torch.long, device=core.slot_groups.device))


@contextlib.contextmanager
def _temporary_actor(
    core: FlexibleSharedPolicy,
    local: np.ndarray,
    shared: np.ndarray,
):
    """Install temporary matrices and always restore the live policy."""
    saved_local, saved_shared, _ = _actor_arrays(core)
    try:
        _install_actor_arrays(core, local, shared)
        yield
    finally:
        _install_actor_arrays(core, saved_local, saved_shared)


def _policy_probs(core: FlexibleSharedPolicy, observations: np.ndarray) -> np.ndarray:
    if len(observations) == 0:
        return np.empty((0, 4), dtype=np.float32)
    values = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        result = core.get_distribution(values).distribution.probs
    return result.detach().cpu().numpy().astype(np.float32)


def _action_from_uniform(probabilities: np.ndarray, uniform: float) -> int:
    cumulative = np.cumsum(np.asarray(probabilities, dtype=np.float64))
    return min(int(np.searchsorted(cumulative, float(uniform), side="right")), 3)


def _write_step(
    writer,
    *,
    interaction_kind: str,
    interaction_step: int,
    training_step: int,
    bout: int,
    task_name: str,
    slot: int,
    group: int,
    generation: int,
    phase: str,
    obs: np.ndarray,
    next_obs: np.ndarray,
    action: int,
    reward: float,
    probabilities: np.ndarray,
    log_prob: float,
    terminated: bool,
    truncated: bool,
) -> None:
    writer.writerow({
        "interaction_kind": interaction_kind,
        "interaction_step": interaction_step,
        "training_step": training_step,
        "bout": bout,
        "taskobserver": task_name,
        "slot": slot,
        "group": group,
        "generation": generation,
        "phase": phase,
        "obs": json.dumps(np.asarray(obs).tolist(), separators=(",", ":")),
        "next_obs": json.dumps(np.asarray(next_obs).tolist(), separators=(",", ":")),
        "action": int(action),
        "reward": float(reward),
        "probabilities": json.dumps(np.asarray(probabilities).tolist(), separators=(",", ":")),
        "log_prob": float(log_prob),
        "terminated": int(terminated),
        "truncated": int(truncated),
    })


def _record_episode(writer, **row) -> None:
    writer.writerow(row)


def _profile_route(
    agent,
    task_name: str,
    *,
    config: Mapping[str, Any],
    bout: int,
    probes_writer,
    episodes_writer,
    assignments_writer,
) -> None:
    """Collect exactly one fixed uniform probe and commit its profile once."""
    slot = int(agent.slot_map[task_name])
    if agent.profiles["seen"][slot]:
        return
    route_index = TASK_NAMES.index(task_name)
    spec = task_from_name(task_name)
    env = make_env(spec, str(config.get("domain", "navigation")))
    reset_seed = agent.seed + 12000 + route_index * 1000
    action_seed = agent.seed + 31000 + route_index * 1000
    action_rng = np.random.default_rng(action_seed)
    obs, _ = env.reset(seed=reset_seed)
    window = EvidenceWindow(capacity=int(config["probe_steps"]))
    episode_id = agent.next_episode_id
    episode_return = 0.0
    episode_length = 0
    episode_number = 0
    goal = False
    for probe_step in range(int(config["probe_steps"])):
        action = int(action_rng.integers(0, 4))
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        window.observe(obs, action, reward, next_obs, bool(terminated))
        episode_return += float(reward)
        episode_length += 1
        goal = goal or bool(info.get("goal_reached", False))
        _write_step(
            probes_writer,
            interaction_kind="probe",
            interaction_step=probe_step + 1,
            training_step=agent.counts["environment_steps"],
            bout=bout,
            task_name=task_name,
            slot=slot,
            group=-1,
            generation=0,
            phase="uniform_probe",
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            probabilities=np.full(4, 0.25, dtype=np.float32),
            log_prob=math.log(0.25),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        agent.counts["probe_steps"] += 1
        if done:
            _record_episode(
                episodes_writer,
                interaction_kind="probe",
                taskobserver=task_name,
                slot=slot,
                generation=0,
                episode_id=episode_id,
                episode=episode_number,
                return_value=episode_return,
                goal_reached=int(goal),
                length=episode_length,
                complete=1,
            )
            episode_id = agent.next_episode_id + 1 + episode_number
            episode_number += 1
            episode_return, episode_length, goal = 0.0, 0, False
            obs, _ = env.reset()
        else:
            obs = next_obs
    if episode_length:
        _record_episode(
            episodes_writer,
            interaction_kind="probe",
            taskobserver=task_name,
            slot=slot,
            generation=0,
            episode_id=episode_id,
            episode=episode_number,
            return_value=episode_return,
            goal_reached=int(goal),
            length=episode_length,
            complete=0,
        )
    env.close()

    profile = window.profile()
    agent.profiles["movement_counts"][slot] = profile.movement_counts
    agent.profiles["reward_xtx"][slot] = profile.reward_xtx
    agent.profiles["reward_xty"][slot] = profile.reward_xty
    agent.profiles["reward_n"][slot] = profile.reward_n
    agent.profiles["seen"][slot] = 1
    assignment = agent.grouper.assign(profile, commit=True)
    agent.profiles["shadow_group"][slot] = assignment.group
    agent.profile_reasons[slot] = str(assignment.reason)
    deployed_group = assignment.group if agent.condition != "independent_fast" else -1
    agent.core.set_slot_group(slot, deployed_group, preserve_effective=False)
    agent.slot_generation[slot] = 1
    assignments_writer.writerow({
        "bout": bout,
        "taskobserver": task_name,
        "slot": slot,
        "generation": 1,
        "group": deployed_group,
        "shadow_group": assignment.group,
        "reason": assignment.reason,
        "profile_rows": int(profile.reward_n),
        "profile_bytes": int(profile.nbytes),
        "scores": json.dumps(assignment.scores, ensure_ascii=False),
    })


def _episode_id(row) -> int:
    return int(getattr(row, "episode_id"))


def _anchor_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [{
        "obs": np.asarray(row.obs, dtype=np.float32),
        "action": int(row.action),
        "behavior_log_prob": float(row.log_prob),
        "behavior_probs": np.asarray(row.probabilities, dtype=np.float32),
        "advantage": float(row.advantage),
        "return": float(row.returns),
        "generation": int(row.generation),
        "episode_id": _episode_id(row),
    } for row in rows]


def _old_group_evidence(agent, group: int, current_slot: int) -> list[dict[str, Any]]:
    evidence = []
    for slot in range(6):
        if slot == current_slot or agent.core.group_for_slot(slot) != group:
            continue
        generation = int(agent.slot_generation[slot])
        rows = agent.anchors.arrays(slot, generation)
        if len(rows["action"]):
            evidence.append({"slot": slot, "generation": generation, **rows})
    return evidence


def _partition_projected_evidence(
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole old episodes into deterministic gradient and gate partitions."""
    references: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for item in evidence:
        episode_ids = np.asarray(item["episode_id"], dtype=np.int64)
        ordered = np.sort(np.unique(episode_ids))
        reference_ids = set(int(value) for value in ordered[::2])
        test_ids = set(int(value) for value in ordered[1::2])
        for ids, destination in ((reference_ids, references), (test_ids, tests)):
            if not ids:
                continue
            mask = np.isin(episode_ids, np.fromiter(sorted(ids), dtype=np.int64))
            sliced: dict[str, Any] = {}
            for key, value in item.items():
                if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(mask):
                    sliced[key] = value[mask].copy()
                else:
                    sliced[key] = value
            destination.append(sliced)
    return references, tests


def _combine_evidence(evidence: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    fields = (
        "obs", "action", "behavior_log_prob", "behavior_probs", "advantage", "episode_id"
    )
    return {
        key: np.concatenate([np.asarray(row[key]) for row in evidence], axis=0)
        for key in fields
    }


def _evaluate_arrays(
    agent,
    incumbent_local: np.ndarray,
    incumbent_shared: np.ndarray,
    candidate_local: np.ndarray,
    candidate_shared: np.ndarray,
    rows: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    relaxed: bool,
) -> dict[str, Any]:
    obs = np.asarray(rows["obs"], dtype=np.float32)
    with _temporary_actor(agent.core, incumbent_local, incumbent_shared):
        incumbent_probs = _policy_probs(agent.core, obs)
    with _temporary_actor(agent.core, candidate_local, candidate_shared):
        candidate_probs = _policy_probs(agent.core, obs)
    return evaluate_change(
        incumbent_probs,
        candidate_probs,
        rows["action"],
        rows["behavior_log_prob"],
        rows["behavior_probs"],
        rows["advantage"],
        rows["episode_id"],
        min_ess=float(config["min_ess"]),
        kl_limit=0.15 if relaxed else float(config["gate_kl"]),
        min_episodes=int(config.get("gate_min_episodes", 2)),
    )


def _evidence_ess(
    agent,
    local: np.ndarray,
    shared: np.ndarray,
    rows: Mapping[str, np.ndarray],
) -> float:
    if not len(rows["action"]):
        return 0.0
    with _temporary_actor(agent.core, local, shared):
        probabilities = _policy_probs(agent.core, rows["obs"])
    actions = np.asarray(rows["action"], dtype=np.int64)
    logp = np.asarray(rows["behavior_log_prob"], dtype=np.float64)
    selected = np.clip(probabilities[np.arange(len(actions)), actions], 1e-12, 1.0)
    log_weights = np.log(selected.astype(np.float64)) - logp
    maximum = float(np.max(log_weights))
    weights = np.exp(log_weights - maximum)
    square_sum = float(np.dot(weights, weights))
    return float(weights.sum() ** 2 / square_sum) if square_sum > 0 else 0.0


def _partition_evidence_report(
    agent,
    local: np.ndarray,
    shared: np.ndarray,
    references: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def summarize(evidence):
        slots = []
        for item in evidence:
            episode_ids = sorted(int(value) for value in np.unique(item["episode_id"]))
            slots.append({
                "slot": int(item["slot"]),
                "generation": int(item["generation"]),
                "rows": int(len(item["action"])),
                "episode_ids": episode_ids,
                "episode_count": len(episode_ids),
                "ess": _evidence_ess(agent, local, shared, item),
            })
        if not evidence:
            return {"rows": 0, "episode_count": 0, "episode_ids": [], "ess": 0.0, "slots": []}
        combined = _combine_evidence(evidence)
        episode_ids = sorted(int(value) for value in np.unique(combined["episode_id"]))
        return {
            "rows": int(len(combined["action"])),
            "episode_count": len(episode_ids),
            "episode_ids": episode_ids,
            "ess": _evidence_ess(agent, local, shared, combined),
            "slots": slots,
        }

    return {
        "split": "sorted_episode_id_alternating_reference_test",
        "reference": summarize(references),
        "test": summarize(tests),
    }


def _projection_gradients(agent, group: int, c_start: np.ndarray, evidence):
    """Get ascent gradients for mean ratio-times-advantage on old anchors."""
    parameter = agent.core._shared_coefficients[group].C
    previous = parameter.detach().clone()
    gradients: list[np.ndarray] = []
    references: list[dict[str, int]] = []
    try:
        with torch.no_grad():
            parameter.copy_(torch.as_tensor(c_start, dtype=parameter.dtype, device=parameter.device))
        for item in evidence:
            obs = torch.as_tensor(item["obs"], dtype=torch.float32)
            actions = torch.as_tensor(item["action"], dtype=torch.long)
            behavior_logp = torch.as_tensor(item["behavior_log_prob"], dtype=torch.float32)
            advantages = torch.as_tensor(item["advantage"], dtype=torch.float32)
            with torch.enable_grad():
                probabilities = agent.core.get_distribution(obs).distribution.probs
                selected = probabilities[torch.arange(len(actions)), actions]
                ratios = selected / behavior_logp.exp().clamp_min(1e-12)
                objective = (ratios * advantages).mean()
                gradient = torch.autograd.grad(objective, parameter, allow_unused=True)[0]
            if gradient is not None:
                gradients.append(gradient.detach().cpu().numpy().reshape(9))
                references.append({
                    "slot": int(item["slot"]),
                    "generation": int(item["generation"]),
                    "rows": int(len(item["action"])),
                    "episode_ids": sorted(int(value) for value in np.unique(item["episode_id"])),
                })
    finally:
        with torch.no_grad():
            parameter.copy_(previous)
    matrix = np.stack(gradients) if gradients else np.empty((0, 9), dtype=np.float64)
    return matrix, references


def _gate_candidate(
    agent,
    *,
    slot: int,
    group: int,
    holdout_rows: Sequence[Any],
    candidate_local: np.ndarray,
    candidate_shared: np.ndarray,
    c_start: np.ndarray,
    old_evidence,
    config: Mapping[str, Any],
    require_old_evidence: bool = False,
    evidence_partition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    relaxed = agent.condition == "trial_gate_relaxed"
    current = {
        "obs": np.stack([row.obs for row in holdout_rows]).astype(np.float32),
        "action": np.asarray([row.action for row in holdout_rows], dtype=np.int64),
        "behavior_log_prob": np.asarray([row.log_prob for row in holdout_rows], dtype=np.float32),
        "behavior_probs": np.stack([row.probabilities for row in holdout_rows]).astype(np.float32),
        "advantage": np.asarray([row.advantage for row in holdout_rows], dtype=np.float32),
        "episode_id": np.asarray([_episode_id(row) for row in holdout_rows], dtype=np.int64),
    }
    baseline_shared = agent.core.read_shared_matrix()
    baseline_shared[group] = c_start.reshape(9)
    baseline_local = agent.core.read_local_matrix()
    baseline_local[slot] = 0.0
    cold_candidate_local = candidate_local.copy()
    cold_candidate_local[slot] = 0.0
    cold = _evaluate_arrays(
        agent,
        baseline_local,
        baseline_shared,
        cold_candidate_local,
        candidate_shared,
        current,
        config,
        relaxed=relaxed,
    )
    multiplier = 0.0 if relaxed else 1.0
    cold_lcb = float(cold["mean_delta"] - multiplier * cold["episode_se"])
    result: dict[str, Any] = {
        "accepted": False,
        "reason": "cold_support_failed",
        "current_cold": cold,
        "current_cold_lcb": cold_lcb,
        "old_slots": [],
        "old_aggregate": None,
        "skipped_unsupported_slots": [],
        "relaxed": relaxed,
        "old_evidence_partition": evidence_partition,
    }
    if not bool(cold["support_ok"]):
        return result
    if cold_lcb < float(config["gate_gain"]):
        result["reason"] = "cold_gain_below_threshold"
        return result

    if not old_evidence:
        if require_old_evidence:
            result.update(
                accepted=False,
                reason="projected_test_evidence_empty",
                old_test_support={
                    "support_ok": False,
                    "sample_count": 0,
                    "episode_count": 0,
                    "ess": 0.0,
                },
            )
            return result
        result.update(accepted=True, reason="provisional_current_evidence")
        return result

    actual_local = agent.core.read_local_matrix()
    slot_scores = []
    supported = []
    for item in old_evidence:
        score = _evaluate_arrays(
            agent,
            actual_local,
            baseline_shared,
            candidate_local,
            candidate_shared,
            item,
            config,
            relaxed=relaxed,
        )
        slot_scores.append({
            "slot": int(item["slot"]),
            "generation": int(item["generation"]),
            "rows": int(len(item["action"])),
            **score,
        })
        if score["support_ok"]:
            supported.append(item)
        else:
            result["skipped_unsupported_slots"].append(int(item["slot"]))
    result["old_slots"] = slot_scores
    if result["skipped_unsupported_slots"] and not relaxed:
        result["reason"] = "old_slot_unsupported"
        return result
    if not supported:
        if relaxed:
            result.update(accepted=True, reason="gate_accepted_old_unsupported_skipped")
        else:
            result["reason"] = "no_supported_old_slot"
        return result

    combined = _combine_evidence(supported)
    aggregate = _evaluate_arrays(
        agent,
        actual_local,
        baseline_shared,
        candidate_local,
        candidate_shared,
        combined,
        config,
        relaxed=relaxed,
    )
    aggregate_lcb = float(aggregate["mean_delta"] - multiplier * aggregate["episode_se"])
    loss_tolerance = 0.02 if relaxed else float(config["gate_loss_tolerance"])
    result["old_aggregate"] = {**aggregate, "lcb": aggregate_lcb}
    if aggregate_lcb < -loss_tolerance:
        result["reason"] = "old_group_loss_exceeds_tolerance"
        return result
    reason = "gate_accepted_old_unsupported_skipped" if result["skipped_unsupported_slots"] else "gate_accepted"
    result.update(accepted=True, reason=reason)
    return result


def _apply_trial_result(
    core: FlexibleSharedPolicy,
    *,
    slot: int,
    group: int,
    a_trial: np.ndarray,
    c_start: np.ndarray,
    c_trial: np.ndarray,
    c_proposal: np.ndarray,
    accepted: bool,
) -> None:
    """Commit or reject a shared delta while preserving the trial policy."""
    target = core._nodes[str(slot + 1)].A
    shared = core._shared_coefficients[group].C
    c_trial_tensor = torch.as_tensor(c_trial, dtype=target.dtype, device=target.device)
    c_start_tensor = torch.as_tensor(c_start, dtype=target.dtype, device=target.device)
    c_proposal_tensor = torch.as_tensor(c_proposal, dtype=target.dtype, device=target.device)
    a_trial_tensor = torch.as_tensor(a_trial, dtype=target.dtype, device=target.device)
    with torch.no_grad():
        if accepted:
            target.copy_(a_trial_tensor + c_trial_tensor.unsqueeze(0) - c_proposal_tensor.unsqueeze(0))
            shared.copy_(c_proposal_tensor)
        else:
            target.copy_(a_trial_tensor + c_trial_tensor.unsqueeze(0) - c_start_tensor.unsqueeze(0))
            shared.copy_(c_start_tensor)


def propose_compensated_actor(
    core: FlexibleSharedPolicy,
    *,
    slot: int,
    group: int,
    a_trial: np.ndarray,
    c_start: np.ndarray,
    c_trial: np.ndarray,
    c_proposal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a shared proposal while retaining the trial actor's current logits."""
    if group < 0 or group >= core.shared_groups:
        raise ValueError("group must identify an installed shared matrix")
    local, shared, groups = _actor_arrays(core)
    core._validate_slot(slot)
    trial_local = np.asarray(a_trial, dtype=np.float32).reshape(3, 3, 3)
    np.asarray(c_start, dtype=np.float32).reshape(3, 3)
    trial_shared = np.asarray(c_trial, dtype=np.float32).reshape(3, 3)
    proposed_shared = np.asarray(c_proposal, dtype=np.float32).reshape(3, 3)
    local[slot] = (
        trial_local + trial_shared[None, :, :] - proposed_shared[None, :, :]
    ).reshape(27)
    shared[group] = proposed_shared.reshape(9)
    return local, shared, groups


def commit_candidate(
    core: FlexibleSharedPolicy,
    *,
    slot: int,
    group: int,
    a_trial: np.ndarray,
    c_start: np.ndarray,
    c_trial: np.ndarray,
    c_proposal: np.ndarray,
    accepted: bool,
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """Commit a decoded actor candidate or reject it with local compensation."""
    if accepted:
        if candidate is None:
            raise ValueError("an accepted candidate requires decoded actor arrays")
        _install_actor_arrays(core, *candidate)
        return
    _apply_trial_result(
        core,
        slot=slot,
        group=group,
        a_trial=a_trial,
        c_start=c_start,
        c_trial=c_trial,
        c_proposal=c_proposal,
        accepted=False,
    )


def score_candidate(agent, **kwargs) -> dict[str, Any]:
    """Public access to the one-shot evidence gate for a decoded candidate."""
    return _gate_candidate(agent, **kwargs)


def _reset_actor_optimizer(agent) -> None:
    parameters = [agent.core._nodes[str(slot + 1)].A for slot in range(6)]
    parameters.extend(agent.core.shared_parameters())
    for parameter in parameters:
        agent.optimizer.state.pop(parameter, None)


def _finish_cycle(
    agent,
    cycle: Mapping[str, Any],
    holdout_rows: Sequence[Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply exactly one end-of-cycle commit, gate, promotion, or rollback."""
    started = time.process_time()
    slot = int(cycle["slot"])
    generation = int(agent.slot_generation[slot])
    if holdout_rows:
        agent.anchors.add(slot, generation, _anchor_rows(holdout_rows))
    group = int(agent.core.group_for_slot(slot))
    condition = agent.condition
    record: dict[str, Any] = {
        "step": int(agent.counts["environment_steps"]),
        "condition": condition,
        "slot": slot,
        "generation": generation,
        "group": group,
        "taskobserver": cycle["task_name"],
        "accepted": False,
        "reason": "uninitialized",
        "current_policy_preserved": None,
        "trial_rollback": condition == "trial_rollback",
        "projection": None,
        "candidate_actor_bytes": None,
        "candidate_actor_budget_bytes": int(config["actor_budget_bytes"]),
    }
    if condition in ("dynamics_shared", "reward_shared", "independent_fast"):
        record.update(
            accepted=True,
            reason="direct_joint_ppo_commit" if condition != "independent_fast" else "independent_local_commit",
            current_policy_preserved=True,
        )
        agent.counts["direct_commit_count"] += 1
        _reset_actor_optimizer(agent)
        record["decision_cpu_s"] = time.process_time() - started
        agent.decision_cpu_s += record["decision_cpu_s"]
        return record

    if condition == "local_promote":
        agent.core.freeze_shared(False)
    if group < 0:
        record["reason"] = "no_assigned_group"
        record["current_policy_preserved"] = True
        record["counters"] = _record_gate_count(agent, accepted=False)
        if condition == "trial_rollback":
            agent.counts["trial_rollback_count"] += 1
        _reset_actor_optimizer(agent)
        record["decision_cpu_s"] = time.process_time() - started
        agent.decision_cpu_s += record["decision_cpu_s"]
        return record

    a_trial = agent.core._nodes[str(slot + 1)].A.detach().cpu().numpy().copy()
    c_trial = agent.core._shared_coefficients[group].C.detach().cpu().numpy().copy()
    c_start = np.asarray(cycle["c_start"], dtype=np.float32).reshape(3, 3).copy()
    a_start = np.asarray(cycle["a_start"], dtype=np.float32).reshape(3, 3, 3).copy()
    raw_delta: np.ndarray
    if condition == "local_promote":
        local_delta = a_trial - a_start
        band_support = np.asarray(cycle["fit_band_counts"], dtype=np.float64)
        if band_support.sum() > 0:
            weighted_delta = np.average(local_delta, axis=0, weights=band_support)
        else:
            weighted_delta = np.zeros((3, 3), dtype=np.float64)
        raw_delta = float(config["promotion_scale"]) * weighted_delta
        c_trial = c_start.copy()
    else:
        raw_delta = c_trial - c_start

    norm_before = float(np.linalg.norm(raw_delta))
    proposal_norm = float(config["proposal_norm"])
    limited_delta = raw_delta.copy()
    if norm_before > proposal_norm and norm_before > 0.0:
        limited_delta *= proposal_norm / norm_before
    projection_record = None
    old_evidence = _old_group_evidence(agent, group, slot)
    reference_evidence: list[dict[str, Any]] = []
    gate_evidence = old_evidence
    if condition == "projected_gate":
        reference_evidence, gate_evidence = _partition_projected_evidence(old_evidence)
        gradients, references = _projection_gradients(agent, group, c_start, reference_evidence)
        projected = project_delta(
            limited_delta.reshape(9),
            gradients,
            tolerance=float(config["gate_loss_tolerance"]),
        )
        projected_delta = projected.delta.reshape(3, 3)
        projected_norm = float(np.linalg.norm(projected_delta))
        if projected_norm > proposal_norm and projected_norm > 0.0:
            projected_delta *= proposal_norm / projected_norm
        limited_delta = projected_delta
        projection_record = {
            "satisfied": bool(projected.satisfied),
            "max_violation": float(projected.max_violation),
            "passes_used": int(projected.passes_used),
            "reference_gradients": gradients,
            "source_references": references,
            "pre_projection_norm": norm_before,
            "projected_norm_before_cap": projected_norm,
            "projected_norm_after_cap": float(np.linalg.norm(limited_delta)),
        }
    c_proposal = np.asarray(c_start + limited_delta.astype(np.float32), dtype=np.float32)
    applied_delta = np.asarray(c_proposal - c_start, dtype=np.float32)

    # Compensation on all bands leaves the current effective matrix unchanged.
    candidate_local, candidate_shared, current_groups = propose_compensated_actor(
        agent.core,
        slot=slot,
        group=group,
        a_trial=a_trial,
        c_start=c_start,
        c_trial=c_trial,
        c_proposal=c_proposal,
    )
    encoded = encode_actor(
        candidate_local,
        candidate_shared,
        current_groups,
        codec="float32",
        budget_bytes=int(config["actor_budget_bytes"]),
    )
    decoded_local, decoded_shared, decoded_groups = decode_actor(encoded.data)
    evidence_partition = (
        _partition_evidence_report(
            agent, decoded_local, decoded_shared, reference_evidence, gate_evidence
        )
        if condition == "projected_gate" else None
    )
    record.update({
        "candidate_actor_bytes": encoded.nbytes,
        "proposal_norm_before_cap": norm_before,
        "proposal_norm_after_cap": float(np.linalg.norm(limited_delta)),
        "proposal_norm_installed": float(np.linalg.norm(applied_delta)),
        "delta_raw": raw_delta,
        "delta_proposed": applied_delta,
        "delta_proposed_requested": limited_delta,
        "c_start": c_start,
        "c_proposal": c_proposal,
        "projection": projection_record,
        "old_anchor_partition": evidence_partition,
        "old_anchor_references": [
            {"slot": item["slot"], "generation": item["generation"], "rows": len(item["action"])}
            for item in old_evidence
        ],
    })

    gate_result: dict[str, Any] | None = None
    accepted = False
    if condition == "trial_rollback":
        reason = "trial_rollback_control"
    elif condition == "projected_gate" and projection_record is not None and not projection_record["satisfied"]:
        reason = "projection_infeasible"
    else:
        gate_result = _gate_candidate(
            agent,
            slot=slot,
            group=group,
            holdout_rows=holdout_rows,
            candidate_local=decoded_local,
            candidate_shared=decoded_shared,
            c_start=c_start,
            old_evidence=gate_evidence,
            config=config,
            require_old_evidence=condition == "projected_gate" and bool(old_evidence),
            evidence_partition=evidence_partition,
        )
        accepted = bool(gate_result["accepted"])
        reason = str(gate_result["reason"])
    if accepted:
        commit_candidate(
            agent.core,
            slot=slot,
            group=group,
            a_trial=a_trial,
            c_start=c_start,
            c_trial=c_trial,
            c_proposal=c_proposal,
            accepted=True,
            candidate=(decoded_local, decoded_shared, decoded_groups),
        )
        _record_gate_count(agent, accepted=True)
        if reason == "provisional_current_evidence":
            agent.counts["provisional_accept_count"] += 1
    else:
        commit_candidate(
            agent.core,
            slot=slot,
            group=group,
            a_trial=a_trial,
            c_start=c_start,
            c_trial=c_trial,
            c_proposal=c_proposal,
            accepted=False,
        )
        _record_gate_count(agent, accepted=False)
    if condition == "trial_rollback":
        agent.counts["trial_rollback_count"] += 1
    record.update({
        "accepted": accepted,
        "reason": reason,
        "current_policy_preserved": True,
        "gate": gate_result,
        "counts": {
            "accepted": int(agent.counts["accepted_count"]),
            "rejected": int(agent.counts["rejected_count"]),
            "trial_rollback": int(agent.counts["trial_rollback_count"]),
        },
    })
    _reset_actor_optimizer(agent)
    record["decision_cpu_s"] = time.process_time() - started
    agent.decision_cpu_s += record["decision_cpu_s"]
    return record


def finish_cycle(agent, cycle, holdout_rows, config) -> dict[str, Any]:
    """Public one-cycle training/gating/commit helper for stream reuse."""
    return _finish_cycle(agent, cycle, holdout_rows, config)


def _record_gate_count(agent, *, accepted: bool) -> dict[str, int]:
    if accepted:
        agent.counts["accepted_count"] += 1
    else:
        agent.counts["rejected_count"] += 1
    agent.counts["decision_count"] += 1
    return {
        "accepted": int(agent.counts["accepted_count"]),
        "rejected": int(agent.counts["rejected_count"]),
        "decisions": int(agent.counts["decision_count"]),
    }


def _temporary_cold_score_policy(core: FlexibleSharedPolicy, slot: int, group: int, c_value: np.ndarray):
    local, shared, _ = _actor_arrays(core)
    local[slot] = 0.0
    shared[group] = np.asarray(c_value, dtype=np.float32).reshape(9)
    return local, shared


def _evaluate_stage(
    agent,
    stage: str,
    config: Mapping[str, Any],
    *,
    steps_writer,
    evaluations_writer,
) -> tuple[dict[str, Any], int]:
    """Run paired complete-episode evaluation without changing learner RNG."""
    started = time.process_time()
    training_step = int(agent.counts["environment_steps"])
    episodes = int(config["evaluation_episodes"])
    records: dict[str, Any] = {}
    total_steps = 0
    interaction_index = int(agent.counts["evaluation_steps"])
    training_mode = agent.core.training
    agent.core.eval()
    try:
        for task_index, name in enumerate(TASK_NAMES):
            spec = task_from_name(name)
            env = make_env(spec, str(config.get("domain", "navigation")))
            slot = int(agent.slot_map[name])
            task_rows = []
            for episode in range(episodes):
                reset_seed = agent.seed + 50000 + task_index * 1000 + episode * 17
                action_seed = agent.seed + 70000 + task_index * 1000 + episode * 17
                action_rng = np.random.default_rng(action_seed)
                obs, _ = env.reset(seed=reset_seed)
                episode_return = 0.0
                episode_length = 0
                success = False
                done = False
                while not done:
                    policy_obs = loop.augment_observation(obs, slot + 1)
                    probs = _policy_probs(agent.core, policy_obs.reshape(1, -1))[0]
                    action = _action_from_uniform(probs, action_rng.random())
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    selected = max(float(probs[action]), 1e-12)
                    log_prob = math.log(selected)
                    interaction_index += 1
                    total_steps += 1
                    episode_return += float(reward)
                    episode_length += 1
                    success = success or bool(info.get("goal_reached", False))
                    _write_step(
                        steps_writer,
                        interaction_kind="evaluation",
                        interaction_step=interaction_index,
                        training_step=training_step,
                        bout=-1,
                        task_name=name,
                        slot=slot,
                        group=agent.core.group_for_slot(slot),
                        generation=int(agent.slot_generation[slot]),
                        phase=stage,
                        obs=policy_obs,
                        next_obs=loop.augment_observation(next_obs, slot + 1),
                        action=action,
                        reward=reward,
                        probabilities=probs,
                        log_prob=log_prob,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                    obs = next_obs
                row = {
                    "stage": stage,
                    "taskobserver": name,
                    "family": spec.family,
                    "variant": spec.variant,
                    "preference": spec.preference,
                    "negative_goal": int(spec.preference < 0),
                    "slot": slot,
                    "episode": episode,
                    "reset_seed": reset_seed,
                    "action_seed": action_seed,
                    "return_value": episode_return,
                    "goal_reached": int(success),
                    "length": episode_length,
                    "complete": 1,
                }
                evaluations_writer.writerow(row)
                task_rows.append(row)
            env.close()
            records[name] = {
                "mean_return": float(np.mean([row["return_value"] for row in task_rows])),
                "mean_goal_rate": float(np.mean([row["goal_reached"] for row in task_rows])),
                "episodes": task_rows,
                "negative_goal": bool(task_from_name(name).preference < 0),
            }
    finally:
        agent.core.train(training_mode)
    agent.counts["evaluation_steps"] += total_steps
    agent.evaluation_cpu_s += time.process_time() - started
    return records, total_steps


def evaluate_stage(
    agent,
    stage: str,
    config: Mapping[str, Any],
    *,
    steps_writer,
    evaluations_writer,
) -> tuple[dict[str, Any], int]:
    """Public paired evaluator used by standalone and streaming runs."""
    return _evaluate_stage(
        agent,
        stage,
        config,
        steps_writer=steps_writer,
        evaluations_writer=evaluations_writer,
    )


def _optimizer_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_optimizer_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_optimizer_tensor_bytes(item) for item in value)
    return 0


def _checkpoint_payload(agent, foundation_sha: str) -> dict[str, Any]:
    local, shared, groups = _actor_arrays(agent.core)
    actor = encode_actor(
        local, shared, groups, codec="float32",
        budget_bytes=int(agent.run_config["actor_budget_bytes"]),
    )
    return {
        "format": "write_decision_agent_v1",
        "condition": agent.condition,
        "seed": int(agent.seed),
        "run_config": dict(agent.run_config),
        "foundation_sha256": foundation_sha,
        "actor_binary": actor.data,
        "critic": loop._critic_state(agent.core),
        "optimizer": agent.optimizer.state_dict(),
        "anchors": agent.anchors.state_dict(),
        "grouper": agent.grouper.state_dict(),
        "profiles": {name: value.copy() for name, value in agent.profiles.items()},
        "profile_reasons": list(agent.profile_reasons),
        "slot_map": dict(agent.slot_map),
        "slot_generation": agent.slot_generation.copy(),
        "profile_seen": agent.profiles["seen"].copy(),
        "counts": dict(agent.counts),
        "reward_by_task": dict(agent.reward_by_task),
        "steps_by_task": dict(agent.steps_by_task),
        "goal_episodes_by_task": dict(agent.goal_episodes_by_task),
        "complete_episodes_by_task": dict(agent.complete_episodes_by_task),
        "next_episode_id": int(agent.next_episode_id),
        "rng": {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "action_generator": copy.deepcopy(agent.action_rng.bit_generator.state),
            "anchor_generator": copy.deepcopy(agent.anchors._rng.bit_generator.state),
        },
        "shared_frozen": bool(agent.core._shared_frozen),
        "train_critic": bool(agent.core.train_critic),
        "recovery_scope": "Agent state recovery only; in-flight environment trajectory is not restored.",
    }


def _save_checkpoint(agent, path: Path, foundation_sha: str) -> dict[str, int]:
    payload = _checkpoint_payload(agent, foundation_sha)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    cap = int(agent.run_config["package_budget_bytes"])
    foundation_bytes = int(Path(agent.base_config["foundation"]).stat().st_size)
    full_persistent_bytes = len(data) + foundation_bytes
    if full_persistent_bytes > cap:
        raise ValueError(
            "W0 foundation plus serialized agent state is "
            f"{full_persistent_bytes} bytes, exceeding package cap {cap}"
        )
    actor = payload["actor_binary"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_suffix(".actor.bin").write_bytes(actor)
    return {
        "actor_bytes": len(actor),
        "actor_budget_bytes": int(agent.run_config["actor_budget_bytes"]),
        "checkpoint_bytes": len(data),
        "foundation_bytes": foundation_bytes,
        "full_persistent_bytes": full_persistent_bytes,
        "package_budget_bytes": cap,
        "anchor_array_bytes": int(agent.anchors.nbytes),
        "profile_array_bytes": int(sum(value.nbytes for value in agent.profiles.values())),
        "grouper_array_bytes": int(agent.grouper.nbytes),
        "optimizer_tensor_bytes": _optimizer_tensor_bytes(agent.optimizer.state_dict()),
        "process_rss_bytes": int(psutil.Process().memory_info().rss),
    }


def save_checkpoint(agent, path: str | Path, foundation_sha: str) -> dict[str, int]:
    """Public complete-state checkpoint writer with configured byte caps."""
    return _save_checkpoint(agent, Path(path), foundation_sha)


def _restore_checkpoint_payload(agent, payload: Mapping[str, Any], foundation_sha: str) -> None:
    if payload.get("format") != "write_decision_agent_v1":
        raise ValueError("unsupported write-decision checkpoint format")
    if payload.get("condition") != agent.condition or int(payload.get("seed", -1)) != agent.seed:
        raise ValueError("checkpoint condition or seed does not match the agent")
    if payload.get("foundation_sha256") != foundation_sha:
        raise ValueError("checkpoint W0 foundation hash does not match")
    local, shared, groups = decode_actor(payload["actor_binary"])
    _install_actor_arrays(agent.core, local, shared, groups)
    loop._load_critic_state(agent.core, payload["critic"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    agent.anchors = AnchorBank.from_state_dict(payload["anchors"])
    agent.grouper = RewardAwareGroups.from_state_dict(payload["grouper"])
    profiles = payload["profiles"]
    for name, target in agent.profiles.items():
        value = np.asarray(profiles[name])
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"checkpoint profile {name} has an invalid shape or dtype")
        target[...] = value
    agent.profile_reasons = list(payload["profile_reasons"])
    agent.slot_map = dict(payload["slot_map"])
    agent.slot_generation[...] = np.asarray(payload["slot_generation"], dtype=np.int64)
    agent.counts.update(payload["counts"])
    agent.reward_by_task.update(payload["reward_by_task"])
    agent.steps_by_task.update(payload["steps_by_task"])
    agent.goal_episodes_by_task.update(payload["goal_episodes_by_task"])
    agent.complete_episodes_by_task.update(payload["complete_episodes_by_task"])
    agent.next_episode_id = int(payload["next_episode_id"])
    agent.action_rng.bit_generator.state = copy.deepcopy(payload["rng"]["action_generator"])
    agent.anchors._rng.bit_generator.state = copy.deepcopy(payload["rng"]["anchor_generator"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy_global"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    agent.core.freeze_shared(bool(payload["shared_frozen"]))
    agent.core._set_trainable_flags(train_critic=bool(payload["train_critic"]))
    if bool(payload["shared_frozen"]):
        agent.core.freeze_shared(True)


def restore_agent_checkpoint(path: str | Path, base_config: Mapping[str, Any] | None = None):
    """Recreate and restore an agent checkpoint for state-inspection/recovery."""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(payload.get("run_config", {}))
    if not config:
        config = _load_config()
    base = dict(base_config if base_config is not None else loop.load_base_config())
    if base_config is None and config.get("foundation") is not None:
        base["foundation"] = str(Path(config["foundation"]).expanduser().resolve())
    foundation_sha = _sha(Path(base["foundation"]))
    agent = _make_agent(
        payload["condition"],
        int(payload["seed"]),
        config,
        base_config=base,
        slot_map=payload["slot_map"],
    )
    _restore_checkpoint_payload(agent, payload, foundation_sha)
    return agent


def _finish_and_save(
    agent,
    cycle: Mapping[str, Any],
    holdout_rows: Sequence[Any],
    config: Mapping[str, Any],
    folder: Path,
    foundation_sha: str,
    decisions_stream,
    capacity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = _finish_cycle(agent, cycle, holdout_rows, config)
    _write_jsonl(decisions_stream, decision)
    sizes = _save_checkpoint(
        agent,
        folder / "checkpoints" / f"checkpoint_step_{agent.counts['environment_steps']:08d}.pt",
        foundation_sha,
    )
    capacity_rows.append({"step": int(agent.counts["environment_steps"]), **sizes})
    return decision


def _train_bout(
    agent,
    task_name: str,
    *,
    bout: int,
    config: Mapping[str, Any],
    steps_writer,
    episodes_writer,
    decisions_stream,
    capacity_rows: list[dict[str, Any]],
    folder: Path,
    foundation_sha: str,
) -> None:
    slot = int(agent.slot_map[task_name])
    if not agent.profiles["seen"][slot]:
        _profile_route(
            agent,
            task_name,
            config=config,
            bout=bout,
            probes_writer=agent._probes_writer,
            episodes_writer=episodes_writer,
            assignments_writer=agent._assignments_writer,
        )
    spec = task_from_name(task_name)
    env = make_env(spec, str(config.get("domain", "navigation")))
    reset_seed = agent.seed + 2200 + bout * 10000
    obs, _ = env.reset(seed=reset_seed)
    generation = int(agent.slot_generation[slot])
    rollout: list[Any] = []
    episode_id = agent.next_episode_id
    episode_index = 0
    episode_return = 0.0
    episode_length = 0
    episode_goal = False
    cycle_state: dict[str, Any] | None = None
    holdout_rows: list[Any] = []
    bout_reward = 0.0
    for _offset in range(int(config["bout_steps"])):
        global_step = int(agent.counts["environment_steps"])
        cycle_position = global_step % CYCLE_STEPS
        rollout_position = cycle_position // ROLLOUT_STEPS
        if cycle_position == 0:
            if agent.condition == "local_promote":
                agent.core.freeze_shared(True)
            group = int(agent.core.group_for_slot(slot))
            c_start = (
                agent.core._shared_coefficients[group].C.detach().cpu().numpy().copy()
                if group >= 0 else np.zeros((3, 3), dtype=np.float32)
            )
            cycle_state = {
                "slot": slot,
                "generation": generation,
                "group": group,
                "task_name": task_name,
                "a_start": agent.core._nodes[str(slot + 1)].A.detach().cpu().numpy().copy(),
                "c_start": c_start,
                "fit_band_counts": np.zeros(3, dtype=np.int64),
            }
            holdout_rows = []

        policy_obs = loop.augment_observation(obs, slot + 1)
        observation_tensor = torch.as_tensor(policy_obs, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            probabilities = agent.core.get_distribution(observation_tensor).distribution.probs.numpy().reshape(4)
            value = float(agent.core.predict_values(observation_tensor).item())
        action, log_prob = loop._sample_action(probabilities, agent.action_rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_policy_obs = loop.augment_observation(next_obs, slot + 1)
        with torch.no_grad():
            next_value = float(agent.core.predict_values(
                torch.as_tensor(next_policy_obs, dtype=torch.float32).reshape(1, -1)
            ).item())
        transition = loop.Transition(
            policy_obs,
            action,
            float(reward),
            value,
            next_value,
            log_prob,
            np.asarray(probabilities, dtype=np.float32).copy(),
            done,
            bool(terminated),
            slot,
            generation,
            global_step + 1,
        )
        transition.episode_id = episode_id
        rollout.append(transition)
        agent.counts["environment_steps"] += 1
        step = int(agent.counts["environment_steps"])
        agent.reward_by_task[task_name] += float(reward)
        agent.steps_by_task[task_name] += 1
        bout_reward += float(reward)
        episode_return += float(reward)
        episode_length += 1
        episode_goal = episode_goal or bool(info.get("goal_reached", False))
        phase = "fit" if rollout_position < 3 else "holdout"
        _write_step(
            steps_writer,
            interaction_kind="training",
            interaction_step=step,
            training_step=step,
            bout=bout,
            task_name=task_name,
            slot=slot,
            group=agent.core.group_for_slot(slot),
            generation=generation,
            phase=phase,
            obs=policy_obs,
            next_obs=next_policy_obs,
            action=action,
            reward=reward,
            probabilities=probabilities,
            log_prob=log_prob,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

        if done:
            agent.counts["completed_episodes"] += 1
            agent.complete_episodes_by_task[task_name] += 1
            agent.counts["all_goal_episodes"] += int(episode_goal)
            if spec.preference > 0:
                agent.counts["positive_completed_episodes"] += 1
                agent.goal_episodes_by_task[task_name] += int(episode_goal)
                agent.counts["positive_goal_episodes"] += int(episode_goal)
            _record_episode(
                episodes_writer,
                interaction_kind="training",
                taskobserver=task_name,
                slot=slot,
                generation=generation,
                episode_id=episode_id,
                bout=bout,
                phase=phase,
                return_value=episode_return,
                goal_reached=int(episode_goal),
                length=episode_length,
                complete=1,
            )
            episode_id = agent.next_episode_id + episode_index + 1
            episode_index += 1
            episode_return, episode_length, episode_goal = 0.0, 0, False
            obs, _ = env.reset()
        else:
            obs = next_obs

        if len(rollout) == ROLLOUT_STEPS:
            if rollout_position < 3:
                assert cycle_state is not None
                for item in rollout:
                    band = min(int(np.floor(np.clip(float(item.obs[0]), 0.0, 1.0) * 3.0)), 2)
                    cycle_state["fit_band_counts"][band] += 1
                loop._ppo_update(agent, rollout)
                agent.counts["fit_steps"] += len(rollout)
            else:
                loop._finalize_gae(rollout)
                agent.counts["holdout_steps"] += len(rollout)
                holdout_rows = rollout
            rollout = []
            if step % CYCLE_STEPS == 0:
                assert cycle_state is not None
                _finish_and_save(
                    agent,
                    cycle_state,
                    holdout_rows,
                    config,
                    folder,
                    foundation_sha,
                    decisions_stream,
                    capacity_rows,
                )
                cycle_state = None
                holdout_rows = []

    if rollout:
        raise RuntimeError("bout ended with an unfinished PPO rollout")
    if cycle_state is not None:
        raise RuntimeError("bout steps must align to 1,024-step cycles")
    if episode_length:
        _record_episode(
            episodes_writer,
            interaction_kind="training",
            taskobserver=task_name,
            slot=slot,
            generation=generation,
            episode_id=episode_id,
            bout=bout,
            phase="bout_end",
            return_value=episode_return,
            goal_reached=int(episode_goal),
            length=episode_length,
            complete=0,
        )
        agent.next_episode_id = episode_id + 1
    else:
        agent.next_episode_id = episode_id
    env.close()
    agent.bout_rewards[task_name] = agent.bout_rewards.get(task_name, 0.0) + bout_reward


def _run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**_load_config(), **dict(config)}
    condition = _check_condition(str(merged.get("condition", "independent_fast")))
    merged["condition"] = condition
    merged["seed"] = int(merged.get("seed", 0))
    merged.setdefault("bout_steps", int(merged["validation_bout_steps"]))
    merged.setdefault("probe_steps", 256)
    merged.setdefault("evaluation_episodes", 8)
    if int(merged["probe_steps"]) != 256:
        raise ValueError("Step 1 route probes are fixed at 256 interactions")
    if int(merged["bout_steps"]) < CYCLE_STEPS or int(merged["bout_steps"]) % CYCLE_STEPS:
        raise ValueError("bout_steps must be a positive multiple of the 1,024-step PPO cycle")
    if int(merged["evaluation_episodes"]) < 1:
        raise ValueError("evaluation_episodes must be positive")
    if "output" not in merged and "output_root" not in merged:
        raise ValueError("run_one config must contain output or output_root")
    return merged


def run_one(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one condition/seed cell; intended as the reusable cohort primitive."""
    run_config = _run_config(config)
    condition, seed = run_config["condition"], run_config["seed"]
    root = Path(run_config.get("output_root", run_config.get("output"))).resolve()
    folder = root / f"seed_{seed}" / condition if "output_root" in run_config else root
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Preserve existing results: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    base_config = loop.load_base_config()
    if run_config.get("foundation") is not None:
        base_config["foundation"] = str(Path(run_config["foundation"]).expanduser().resolve())
    slot_map = _slot_map(seed)
    _write_json(folder / "config.json", {**run_config, "slot_map": slot_map})
    manifest = _snapshot_sources(folder, base_config)
    foundation_sha = manifest["foundation"]["sha256"]
    agent = _make_agent(condition, seed, run_config, base_config=base_config, slot_map=slot_map)
    agent._probes_writer = None
    agent._assignments_writer = None
    agent.bout_rewards = {name: 0.0 for name in TASK_NAMES}
    sequence = list(run_config.get("sequence", _load_config()["sequence"]))
    if not sequence:
        raise ValueError("sequence must contain at least one task")
    unknown_tasks = sorted(set(sequence) - set(TASK_NAMES))
    if unknown_tasks:
        raise ValueError(f"sequence contains unknown tasks: {unknown_tasks}")

    step_fields = [
        "interaction_kind", "interaction_step", "training_step", "bout", "taskobserver",
        "slot", "group", "generation", "phase", "obs", "next_obs", "action", "reward",
        "probabilities", "log_prob", "terminated", "truncated",
    ]
    probe_fields = step_fields
    episode_fields = [
        "interaction_kind", "taskobserver", "slot", "generation", "episode_id", "episode",
        "bout", "phase", "return_value", "goal_reached", "length", "complete",
    ]
    evaluation_fields = [
        "stage", "taskobserver", "family", "variant", "preference", "negative_goal",
        "slot", "episode", "reset_seed", "action_seed", "return_value", "goal_reached",
        "length", "complete",
    ]
    assignment_fields = [
        "bout", "taskobserver", "slot", "generation", "group", "shadow_group",
        "reason", "profile_rows", "profile_bytes", "scores",
    ]
    streams = []
    capacity_rows: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}
    total_eval_steps = 0
    total_probe_steps = 0
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        for name, fields in (
            ("steps.csv", step_fields),
            ("probes.csv", probe_fields),
            ("episodes.csv", episode_fields),
            ("evaluations.csv", evaluation_fields),
            ("profile_assignments.csv", assignment_fields),
        ):
            stream = (folder / name).open("w", newline="", encoding="utf-8")
            streams.append(stream)
        steps_writer, probes_writer, episodes_writer, eval_writer, assignments_writer = [
            csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            for stream, fields in zip(streams, (step_fields, probe_fields, episode_fields, evaluation_fields, assignment_fields))
        ]
        for writer in (steps_writer, probes_writer, episodes_writer, eval_writer, assignments_writer):
            writer.writeheader()
        decisions_stream = (folder / "decisions.jsonl").open("w", encoding="utf-8")
        streams.append(decisions_stream)
        agent._probes_writer = probes_writer
        agent._assignments_writer = assignments_writer

        evaluations["initial"], initial_eval_steps = _evaluate_stage(
            agent, "initial", run_config, steps_writer=steps_writer, evaluations_writer=eval_writer
        )
        total_eval_steps += initial_eval_steps
        for bout, task_name in enumerate(sequence):
            if not agent.profiles["seen"][agent.slot_map[task_name]]:
                before = int(agent.counts["probe_steps"])
                _profile_route(
                    agent,
                    task_name,
                    config=run_config,
                    bout=bout,
                    probes_writer=probes_writer,
                    episodes_writer=episodes_writer,
                    assignments_writer=assignments_writer,
                )
                total_probe_steps += int(agent.counts["probe_steps"]) - before
            _train_bout(
                agent,
                task_name,
                bout=bout,
                config=run_config,
                steps_writer=steps_writer,
                episodes_writer=episodes_writer,
                decisions_stream=decisions_stream,
                capacity_rows=capacity_rows,
                folder=folder,
                foundation_sha=foundation_sha,
            )
            if bout + 1 == 8:
                evaluations["after_source"], cost = _evaluate_stage(
                    agent, "after_source", run_config,
                    steps_writer=steps_writer, evaluations_writer=eval_writer,
                )
                total_eval_steps += cost
        evaluations["final"], final_eval_steps = _evaluate_stage(
            agent, "final", run_config, steps_writer=steps_writer, evaluations_writer=eval_writer
        )
        total_eval_steps += final_eval_steps
        decisions_stream.flush()
    finally:
        for stream in streams:
            stream.close()

    # CSV rows retain the raw task reward; compute run totals from training rows only.
    total_reward = 0.0
    with (folder / "steps.csv").open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["interaction_kind"] == "training":
                total_reward += float(row["reward"])
    total_reward_per_step = total_reward / max(1, int(agent.counts["environment_steps"]))
    capacity_path = folder / "capacity.json"
    _write_json(capacity_path, capacity_rows)
    result = {
        "condition": condition,
        "seed": seed,
        "sequence": sequence,
        "bout_steps": int(run_config["bout_steps"]),
        "counts": dict(agent.counts),
        "training_reward": float(total_reward),
        "reward_per_step": float(total_reward_per_step),
        "plus_goal_rate": float(
            agent.counts["positive_goal_episodes"]
            / max(1, agent.counts["positive_completed_episodes"])
        ),
        "per_task_training": {
            name: {
                "reward": float(agent.reward_by_task[name]),
                "steps": int(agent.steps_by_task[name]),
                "reward_per_step": float(agent.reward_by_task[name] / max(1, agent.steps_by_task[name])),
                "complete_episodes": int(agent.complete_episodes_by_task[name]),
                "positive_goal_episodes": int(agent.goal_episodes_by_task[name]),
            }
            for name in TASK_NAMES
        },
        "evaluations": evaluations,
        "costs": {
            "training_steps": int(agent.counts["environment_steps"]),
            "probe_steps": int(agent.counts["probe_steps"]),
            "evaluation_steps": int(agent.counts["evaluation_steps"]),
            "total_interactions": int(
                agent.counts["environment_steps"]
                + agent.counts["probe_steps"]
                + agent.counts["evaluation_steps"]
            ),
            "decision_cpu_s": float(agent.decision_cpu_s),
            "evaluation_cpu_s": float(agent.evaluation_cpu_s),
            "wall_s": float(time.perf_counter() - started),
            "cpu_s": float(time.process_time() - cpu_started),
            "process_rss_bytes": int(psutil.Process().memory_info().rss),
        },
        "slot_map": slot_map,
        "slot_groups": agent.core.slot_groups.detach().cpu().tolist(),
        "shadow_groups": agent.profiles["shadow_group"].tolist(),
        "profile_state_bytes": int(sum(array.nbytes for array in agent.profiles.values())),
        "anchor_state_bytes": int(agent.anchors.nbytes),
        "group_state_bytes": int(agent.grouper.nbytes),
        "capacity": capacity_rows,
        "foundation_sha256": foundation_sha,
        "limitations": [
            "Group assignments use bounded probe evidence and external route identity only selects a local slot.",
            "Gate scores are clipped off-policy advantage surrogates, not policy-value guarantees.",
            "Checkpoints restore learner state but not an in-flight environment trajectory.",
        ],
    }
    _write_json(folder / "summary.json", result)
    return result


def _validation_summary(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "summary.json"
    if not source.is_file():
        raise FileNotFoundError(f"validation summary not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("validation summary must contain a records list")
    return payload, source


def select_conditions(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Select two proposed rules by validation reward per training step."""
    validation, source = _validation_summary(input_path)
    config = _load_config()
    if validation.get("phase") != "validation":
        raise ValueError("selection requires a validation-phase cohort summary")
    if validation.get("conditions") != config["validation_conditions"]:
        raise ValueError("validation summary conditions do not match the configured eight conditions")
    if validation.get("seeds") != config["validation_seeds"]:
        raise ValueError("validation summary seeds do not match the configured validation seeds")

    expected_pairs = {
        (str(condition), int(seed))
        for condition in config["validation_conditions"]
        for seed in config["validation_seeds"]
    }
    records = validation["records"]
    if len(records) != len(expected_pairs):
        raise ValueError("validation summary must contain all 16 configured condition/seed records")
    observed_pairs: list[tuple[str, int]] = []
    expected_training_steps = int(config["validation_bout_steps"]) * len(config["sequence"])
    expected_probe_steps = len(TASK_NAMES) * int(config["probe_steps"])
    for row in records:
        pair = (str(row.get("condition", "")), int(row.get("seed", -1)))
        observed_pairs.append(pair)
        if int(row.get("bout_steps", -1)) != int(config["validation_bout_steps"]):
            raise ValueError(f"validation record {pair} has a nonconfigured bout length")
        if row.get("sequence") != config["sequence"]:
            raise ValueError(f"validation record {pair} has a nonconfigured task sequence")
        counts = row.get("counts", {})
        costs = row.get("costs", {})
        if int(counts.get("environment_steps", -1)) != expected_training_steps:
            raise ValueError(f"validation record {pair} has incomplete training steps")
        if int(counts.get("probe_steps", -1)) != expected_probe_steps:
            raise ValueError(f"validation record {pair} has incomplete 256-step route probes")
        if int(counts.get("evaluation_steps", -1)) != int(costs.get("evaluation_steps", -2)):
            raise ValueError(f"validation record {pair} has inconsistent evaluation-step counts")
        evaluations = row.get("evaluations", {})
        if set(evaluations) != {"initial", "after_source", "final"}:
            raise ValueError(f"validation record {pair} must include all three paired evaluation stages")
        for stage, task_results in evaluations.items():
            if set(task_results) != set(TASK_NAMES):
                raise ValueError(f"validation record {pair} has incomplete {stage} task evaluations")
            for task_index, task_name in enumerate(TASK_NAMES):
                episodes = task_results[task_name].get("episodes", [])
                if len(episodes) != int(config["evaluation_episodes"]):
                    raise ValueError(f"validation record {pair} has incomplete {stage}/{task_name} evaluation")
                for episode, item in enumerate(episodes):
                    expected_reset = int(pair[1]) + 50000 + task_index * 1000 + episode * 17
                    expected_action = int(pair[1]) + 70000 + task_index * 1000 + episode * 17
                    if (
                        int(item.get("reset_seed", -1)) != expected_reset
                        or int(item.get("action_seed", -1)) != expected_action
                        or int(item.get("complete", 0)) != 1
                    ):
                        raise ValueError(f"validation record {pair} has an invalid paired evaluation plan")
    if len(set(observed_pairs)) != len(observed_pairs) or set(observed_pairs) != expected_pairs:
        raise ValueError("validation summary must contain every configured pair exactly once")

    values: dict[str, list[float]] = {name: [] for name in PROPOSED_CONDITIONS}
    for row in records:
        condition = str(row.get("condition", ""))
        if condition in values:
            score = float(row["reward_per_step"])
            if not math.isfinite(score):
                raise ValueError(f"validation score for {condition} is not finite")
            values[condition].append(score)
    missing = [name for name, scores in values.items() if not scores]
    if len(missing) > len(PROPOSED_CONDITIONS) - 2:
        raise ValueError(f"validation lacks enough proposed candidates: {missing}")
    means = {
        name: float(np.mean(scores))
        for name, scores in values.items()
        if scores
    }
    ranked = sorted(means, key=lambda name: (-means[name], name))
    selected = ranked[:2]
    if len(selected) != 2:
        raise ValueError("validation results must include at least two proposed conditions")
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".json":
        output.mkdir(parents=True, exist_ok=True)
        output = output / "selection.json"
    elif output.parent.exists() and output.exists():
        raise FileExistsError(f"Preserve existing selection: {output}")
    selection = {
        "format": "write_decision_selection_v1",
        "rule": "Top two proposed conditions by mean validation online reward per training step.",
        "validation_summary": str(source),
        "selected_conditions": selected,
        "formal_conditions": [*SIMPLE_BASELINES, *selected],
        "candidate_mean_reward_per_step": means,
        "preserved_baselines": list(SIMPLE_BASELINES),
        "validation_training_steps": int(validation.get("training_steps", 0)),
    }
    _write_json(output, selection)
    return selection


def _formal_conditions(selection_path: str | Path) -> list[str]:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if payload.get("format") != "write_decision_selection_v1":
        raise ValueError("formal phase requires a write_decision_selection_v1 selection file")
    selected = payload.get("selected_conditions")
    if not isinstance(selected, list) or len(selected) != 2 or any(
        item not in PROPOSED_CONDITIONS for item in selected
    ):
        raise ValueError("selection file must contain exactly two proposed conditions")
    return list(dict.fromkeys([*SIMPLE_BASELINES, *selected]))


def run_cohort(
    phase: str,
    output: str | Path,
    *,
    workers: int = 1,
    input_path: str | Path | None = None,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    bout_steps: int | None = None,
    evaluation_episodes: int | None = None,
) -> dict[str, Any]:
    if phase == "select":
        if input_path is None:
            raise ValueError("select phase requires --input validation output")
        return select_conditions(input_path, output)
    if phase not in ("validation", "formal"):
        raise ValueError("phase must be validation, formal, or select")
    if not 1 <= int(workers) <= 2:
        raise ValueError("workers must be one or two")
    config = _load_config()
    if phase == "formal":
        if input_path is None:
            raise ValueError("formal phase requires --input selection.json")
        if (
            seeds is not None
            or conditions is not None
            or bout_steps is not None
            or evaluation_episodes is not None
        ):
            raise ValueError(
                "formal phase uses selected conditions, configured seeds, bout length, and evaluation count"
            )
        run_conditions = _formal_conditions(input_path)
        run_seeds = [int(value) for value in config["formal_seeds"]]
        run_bout_steps = int(config["formal_bout_steps"])
    else:
        run_conditions = list(conditions or config["validation_conditions"])
        run_seeds = [int(value) for value in (seeds or config["validation_seeds"])]
        run_bout_steps = int(bout_steps or config["validation_bout_steps"])
    unknown = sorted(set(run_conditions) - set(VALID_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    if not run_seeds:
        raise ValueError("seeds must not be empty")
    if run_bout_steps < CYCLE_STEPS or run_bout_steps % CYCLE_STEPS:
        raise ValueError("bout_steps must be a positive multiple of 1,024")
    run_config = {
        **config,
        "bout_steps": run_bout_steps,
        "evaluation_episodes": int(evaluation_episodes or config["evaluation_episodes"]),
        "phase": phase,
    }
    target = Path(output).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Preserve existing cohort output: {target}")
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "frozen_config.json", {
        **run_config,
        "conditions": run_conditions,
        "seeds": run_seeds,
        "selection_input": str(input_path) if input_path else None,
        "protocol_sha256": _sha(PROTOCOL_PATH),
    })
    tasks = [
        {**run_config, "condition": condition, "seed": seed, "output_root": str(target)}
        for seed in run_seeds
        for condition in run_conditions
    ]
    if int(workers) == 1:
        records = [run_one(item) for item in tasks]
    else:
        records = []
        with ProcessPoolExecutor(max_workers=min(int(workers), 2)) as executor:
            futures = [executor.submit(run_one, item) for item in tasks]
            for future in as_completed(futures):
                records.append(future.result())
    records.sort(key=lambda row: (int(row["seed"]), str(row["condition"])))
    cohort_cost = {
        key: int(sum(row["costs"][key] for row in records))
        for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")
    }
    summary = {
        "phase": phase,
        "conditions": run_conditions,
        "seeds": run_seeds,
        "records": records,
        "training_steps": int(cohort_cost["training_steps"]),
        "cohort_cost": cohort_cost,
        "protocol_sha256": _sha(PROTOCOL_PATH),
    }
    _write_json(target / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "formal", "select"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bout-steps", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    args = parser.parse_args()
    result = run_cohort(
        args.phase,
        args.output,
        workers=args.workers,
        input_path=args.input,
        conditions=args.conditions,
        seeds=args.seeds,
        bout_steps=args.bout_steps,
        evaluation_episodes=args.evaluation_episodes,
    )
    print(json.dumps({
        "phase": result.get("phase", "select"),
        "conditions": result.get("conditions", result.get("selected_conditions")),
        "seeds": result.get("seeds"),
        "cohort_cost": result.get("cohort_cost"),
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()

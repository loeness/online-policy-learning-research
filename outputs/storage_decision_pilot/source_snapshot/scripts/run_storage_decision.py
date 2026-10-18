"""Run the frozen single-shared-slot storage decision experiment.

Task IDs, groups, and bout boundaries are supplied uniformly by this controlled
mechanism harness; no unknown-context discovery claim is made. PPO still uses
only the observation, reward, action, and termination data.
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
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.retention_consolidation import surrogate_gain
from online_rl.score_calibration import restore_navigation, snapshot_navigation
from online_rl.storage_decision import (
    ACTIONS,
    BoundedGroupEvidence,
    apply_action,
    common_mean_effective,
    decide_from_past,
    pair_mapping,
    prefix_schedule,
    semantic_hash,
    task_name,
    template_schedule,
    validate_config,
)
from online_rl.write_decision_envs import make_env, task_from_name
from online_rl.write_decision_policy import decode_actor, encode_actor
from scripts import run_knowledge_loop as loop
from scripts import run_write_stream as ws


CONFIG_PATH = ROOT / "configs" / "storage_decision.json"
PROTOCOL_PATH = ROOT / "docs" / "STORAGE_DECISION_PROTOCOL.md"
SOURCE_FILES = (
    Path("scripts/run_storage_decision.py"),
    Path("scripts/run_knowledge_loop.py"),
    Path("scripts/run_write_stream.py"),
    Path("src/online_rl/storage_decision.py"),
    Path("src/online_rl/score_calibration.py"),
    Path("src/online_rl/retention_consolidation.py"),
    Path("src/online_rl/consolidation_probe.py"),
    Path("src/online_rl/write_decision_envs.py"),
    Path("src/online_rl/write_decision_policy.py"),
    Path("src/online_rl/shared_write_policy.py"),
    Path("src/online_rl/shared_write_storage.py"),
    Path("src/online_rl/write_decision_evidence.py"),
    Path("src/online_rl/write_fast_evidence.py"),
    Path("src/online_rl/c1_supplement_experiment.py"),
    Path("src/online_rl/c1_tensor_controls.py"),
    Path("src/online_rl/c1_policy.py"),
    Path("src/online_rl/c1_envs.py"),
    Path("src/online_rl/experiment.py"),
    Path("src/online_rl/policy.py"),
    Path("configs/storage_decision.json"),
    Path("docs/STORAGE_DECISION_PROTOCOL.md"),
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
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_sources(folder: Path, config_path: Path) -> dict[str, Any]:
    snapshot = folder / "source_snapshot"
    files: dict[str, dict[str, Any]] = {}
    paths = list(SOURCE_FILES)
    config_rel = Path(os.path.relpath(config_path.resolve(), ROOT.resolve()))
    if config_rel not in paths and config_path.is_relative_to(ROOT):
        paths.append(config_rel)
    for relative in paths:
        if any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts):
            raise ValueError("private or temporary material is excluded from source snapshots")
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_digest, copy_digest = _sha(source), _sha(target)
        if source_digest != copy_digest:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        files[relative.as_posix()] = {"sha256": copy_digest, "bytes": target.stat().st_size}
    base = loop.load_base_config()
    foundation, dataset = Path(base["foundation"]), Path(base["dataset"])
    manifest = {
        "sources": files,
        "foundation": {"path": str(foundation), "sha256": _sha(foundation), "bytes": foundation.stat().st_size},
        "dataset": {"path": str(dataset), "sha256": _sha(dataset), "bytes": dataset.stat().st_size},
        "excluded_local_instructions": ["Agent.md", "AGENTS.md"],
    }
    _write_json(snapshot / "manifest.json", manifest)
    return manifest


def _agent_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase": "known",
        "domain": "navigation",
        "local_slots": 2,
        "shared_groups": 1,
        "local_learning_rate": float(config["local_learning_rate"]),
        "fast_local_learning_rate": float(config["local_learning_rate"]),
        "shared_learning_rate": float(config["shared_learning_rate"]),
        "critic_learning_rate": float(config["critic_learning_rate"]),
        "anchors_per_slot": 24,
        "probe_steps": 256,
        "reward_support_aware": False,
        "group_reward_min_samples": 1,
        "actor_codec": "int8",
        "actor_budget_bytes": 89,
        "package_budget_bytes": int(config["package_budget_bytes"]),
    }


def _new_agent(seed: int, base_config: Mapping[str, Any], config: Mapping[str, Any]):
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    agent = ws._new_agent("reward_shared", int(seed), _agent_config(config), base_config)
    agent.core.freeze_shared(True)
    agent.cache = ws.BoundedTaskCache(2, "lru", int(seed) + 8811)
    agent.behavior_rng = np.random.default_rng(int(seed) + 1100)
    agent.evidence = BoundedGroupEvidence(int(config["evidence_rows_per_group"]))
    agent.recent_groups: list[int] = []
    agent.parent_group = -1
    agent.slot_group_tags = np.full(2, -1, dtype=np.int8)
    agent.runtime_counts = {
        "environment_steps": 0,
        "completed_episodes": 0,
        "slot_allocations": 0,
        "slot_evictions": 0,
        "rollouts_trained": 0,
        "minibatch_updates": 0,
    }
    return agent


def _capture_learner(agent, foundation_sha: str) -> dict[str, Any]:
    return {
        "format": "storage_decision_learner_v1",
        "seed": int(agent.seed),
        "foundation_sha256": str(foundation_sha),
        "core_state": {key: value.detach().cpu().clone() for key, value in agent.core.state_dict().items()},
        "optimizer_state": copy.deepcopy(agent.optimizer.state_dict()),
        "optimizer_step_count": int(getattr(agent.optimizer, "step_count", 0)),
        "shared_frozen": bool(agent.core._shared_frozen),
        "counts": copy.deepcopy(agent.counts),
        "runtime_counts": dict(agent.runtime_counts),
        "slot_generation": agent.slot_generation.copy(),
        "fitted_steps": agent.fitted_steps.copy(),
        "cache": agent.cache.state_dict(),
        "evidence": agent.evidence.state_dict(),
        "recent_groups": list(agent.recent_groups),
        "parent_group": int(agent.parent_group),
        "slot_group_tags": agent.slot_group_tags.copy(),
        "action_rng": copy.deepcopy(agent.action_rng.bit_generator.state),
        "behavior_rng": copy.deepcopy(agent.behavior_rng.bit_generator.state),
        "torch_rng": torch.get_rng_state().clone(),
        "numpy_rng": copy.deepcopy(np.random.get_state()),
        "python_rng": copy.deepcopy(random.getstate()),
    }


def _restore_learner(payload: Mapping[str, Any], base_config: Mapping[str, Any], config: Mapping[str, Any]):
    if payload.get("format") != "storage_decision_learner_v1":
        raise ValueError("unsupported learner checkpoint format")
    agent = _new_agent(int(payload["seed"]), base_config, config)
    agent.core.load_state_dict(payload["core_state"], strict=True)
    agent.optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    agent.optimizer.step_count = int(payload["optimizer_step_count"])
    agent.cache = ws.BoundedTaskCache.from_state_dict(payload["cache"])
    agent.evidence = BoundedGroupEvidence.from_state_dict(payload["evidence"])
    agent.counts = copy.deepcopy(payload["counts"])
    agent.runtime_counts = dict(payload["runtime_counts"])
    agent.slot_generation = np.asarray(payload["slot_generation"], dtype=np.int64).copy()
    agent.fitted_steps = np.asarray(payload["fitted_steps"], dtype=np.uint64).copy()
    agent.recent_groups = [int(value) for value in payload["recent_groups"]]
    agent.parent_group = int(payload["parent_group"])
    agent.slot_group_tags = np.asarray(payload["slot_group_tags"], dtype=np.int8).copy()
    agent.action_rng.bit_generator.state = copy.deepcopy(payload["action_rng"])
    agent.behavior_rng.bit_generator.state = copy.deepcopy(payload["behavior_rng"])
    if bool(payload["shared_frozen"]) != bool(agent.core._shared_frozen):
        agent.core.freeze_shared(bool(payload["shared_frozen"]))
    torch.set_rng_state(payload["torch_rng"].clone())
    np.random.set_state(copy.deepcopy(payload["numpy_rng"]))
    random.setstate(copy.deepcopy(payload["python_rng"]))
    return agent


def _capture_simulator(env, runtime: Mapping[str, Any]) -> dict[str, Any]:
    driver = copy.deepcopy(dict(runtime))
    driver["obs"] = np.asarray(driver["obs"], dtype=np.float32).tolist()
    return {"environment": snapshot_navigation(env), "driver": driver}


def _restore_simulator(snapshot: Mapping[str, Any]):
    return restore_navigation(snapshot["environment"]), copy.deepcopy(snapshot["driver"])


def _save_package(folder: Path, learner: Mapping[str, Any], actor_data: bytes, foundation: Path, config: Mapping[str, Any]) -> dict[str, int]:
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path, actor_path = folder / "checkpoint.pt", folder / "actor.bin"
    torch.save(dict(learner), checkpoint_path)
    actor_path.write_bytes(actor_data)
    foundation_bytes = foundation.stat().st_size
    checkpoint_bytes = checkpoint_path.stat().st_size
    actor_bytes = actor_path.stat().st_size
    package_bytes = foundation_bytes + checkpoint_bytes + actor_bytes
    if package_bytes > int(config["package_budget_bytes"]):
        raise ValueError(f"complete persistent package is {package_bytes} bytes, exceeding cap {config['package_budget_bytes']}")
    return {
        "foundation_bytes": int(foundation_bytes),
        "checkpoint_bytes": int(checkpoint_bytes),
        "actor_bytes": int(actor_bytes),
        "full_persistent_bytes": int(package_bytes),
        "package_budget_bytes": int(config["package_budget_bytes"]),
    }


def _encode_live_actor(agent) -> tuple[bytes, float, list[float]]:
    before_local, before_shared, before_groups = ws._actor_arrays(agent.core)
    before_effective = _effective_live(before_local, before_shared, before_groups)
    blob = encode_actor(before_local, before_shared, before_groups, codec="int8", budget_bytes=89)
    local, shared, groups = decode_actor(blob.data)
    agent.core.install_local_matrix(local)
    agent.core.install_shared_matrix(shared)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(groups.astype(np.int64), dtype=torch.long))
    after_effective = _effective_live(local, shared, groups)
    per_slot = np.max(np.abs(after_effective - before_effective), axis=(1, 2, 3)).astype(float).tolist()
    coefficient_error = max(
        float(np.max(np.abs(local - before_local), initial=0.0)),
        float(np.max(np.abs(shared - before_shared), initial=0.0)),
    )
    return blob.data, coefficient_error, per_slot


def _effective_live(local: np.ndarray, shared: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(local, dtype=np.float32).reshape(-1, 3, 3, 3).copy()
    for slot, group in enumerate(np.asarray(groups).tolist()):
        if int(group) == 0:
            result[slot] += np.asarray(shared, dtype=np.float32)[0].reshape(1, 3, 3)
    return result


def _policy_probabilities(core, policy_obs: np.ndarray) -> tuple[np.ndarray, float]:
    tensor = torch.as_tensor(np.asarray(policy_obs, dtype=np.float32), dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        probs = core.get_distribution(tensor).distribution.probs.reshape(-1).cpu().numpy().astype(np.float64)
        value = float(core.predict_values(tensor).reshape(-1)[0].cpu().item())
    probs /= probs.sum()
    return probs, value


def _resident_probabilities(core, observation: np.ndarray, slots: int = 2) -> np.ndarray:
    return np.stack([
        _policy_probabilities(core, loop.augment_observation(observation, slot + 1))[0]
        for slot in range(slots)
    ]).astype(np.float64)


def _probability_drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape or ref.ndim != 2 or ref.shape[1] != 4:
        raise ValueError("probability snapshots must have matching N by 4 shapes")
    absolute = np.abs(cand - ref)
    ref_safe = np.clip(ref, 1e-12, 1.0)
    cand_safe = np.clip(cand, 1e-12, 1.0)
    kl_rows = np.sum(ref_safe * (np.log(ref_safe) - np.log(cand_safe)), axis=1)
    return {
        "max_probability_absolute_drift": float(np.max(absolute, initial=0.0)),
        "per_resident_max_probability_absolute_drift": np.max(absolute, axis=1).astype(float).tolist(),
        "kl_reference_to_candidate": np.maximum(kl_rows, 0.0).astype(float).tolist(),
        "max_kl_reference_to_candidate": float(np.max(np.maximum(kl_rows, 0.0), initial=0.0)),
    }


def _coefficient_probs(core, observations: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    if obs.ndim != 2 or obs.shape[1] != 13:
        raise ValueError("counterfactual observations must be N by 13")
    if len(obs) == 0:
        return np.empty((0, 4), dtype=np.float32)
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        phi = torch.cat((latent, torch.ones_like(latent[:, :1])), dim=1)
        psi = torch.nn.functional.linear(phi, core.source_basis)
        coeff = torch.as_tensor(np.asarray(coefficients), dtype=psi.dtype)
        if coeff.shape == (3, 3):
            contrast = torch.nn.functional.linear(psi, coeff)
        elif coeff.shape == (3, 3, 3):
            bands = torch.floor(obs[:, 0].clamp(0, 1) * 3).long().clamp(0, 2)
            contrast = torch.bmm(coeff[bands], psi.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError("coefficients must have shape 3 by 3 or 3 by 3 by 3")
        logits = core.action_net(latent) + contrast @ core._contrast_basis.T
        probs = torch.softmax(logits / core.temperature, dim=-1)
        probs = (1.0 - core.epsilon) * probs + core.epsilon / 4.0
    return probs.cpu().numpy()


def _score_fn(core, config: Mapping[str, Any]):
    gate = config["gate"]
    fields = ("min_rows", "min_blocks", "min_ess_ratio", "max_mean_kl", "clip_ratio", "stderr_multiplier")
    options = {field: gate[field] for field in fields}

    def score(evidence: Mapping[str, Any], candidate: np.ndarray, baseline: np.ndarray):
        return surrogate_gain(
            dict(evidence),
            _coefficient_probs(core, evidence["obs"], candidate),
            _coefficient_probs(core, evidence["obs"], baseline),
            **options,
        )
    return score


def _reset_seed(seed: int, ordinal: int) -> int:
    return int(seed) * 1000003 + int(ordinal) * 7919 + 0x5E7


def _prepare_task(agent, task: str, group: int) -> tuple[int, dict[str, Any]]:
    allocation = agent.cache.resolve(task)
    slot = int(allocation["slot"])
    inherited_parent = False
    if allocation["allocated"]:
        ws._reset_allocated_slot(agent, slot, reason="known_task_lru_allocation")
        agent.runtime_counts["slot_allocations"] += 1
        evicted = int(allocation["evicted_slot"] is not None)
        agent.runtime_counts["slot_evictions"] += evicted
        agent.counts["slot_evictions"] += evicted
        agent.slot_group_tags[slot] = int(group)
        inherit = agent.parent_group == -2 or agent.parent_group == int(group)
        inherited_parent = bool(inherit)
        agent.core.set_slot_group(slot, 0 if inherit else -1, preserve_effective=False)
        if inherit:
            agent.runtime_counts["prior_bindings"] = agent.runtime_counts.get("prior_bindings", 0) + 1
    agent.current_known_slot = slot
    allocation["inherited_parent"] = inherited_parent
    allocation["physical_group_at_start"] = int(agent.core.group_for_slot(slot))
    allocation["parent_group_at_start"] = int(agent.parent_group)
    return slot, allocation


def _headers(folder: Path):
    step_fields = (
        "phase", "action", "future_regime", "global_step", "bout", "bout_step", "episode_id", "reset_seed",
        "taskobserver", "group", "slot", "generation", "obs", "next_obs", "action_id",
        "reward", "probabilities", "action_uniform", "behavior_log_prob", "terminated", "truncated",
        "environment_done", "learning_boundary", "goal_reached",
    )
    episode_fields = (
        "phase", "action", "future_regime", "episode_id", "reset_seed", "next_reset_seed",
        "task_at_completion", "group_at_completion",
        "return_value", "length", "goal_reached", "terminated", "truncated",
    )
    rollout_fields = (
        "phase", "action", "future_regime", "bout", "rollout_index", "rows", "updates",
        "policy_loss", "value_loss", "entropy", "approx_kl", "codec_max_coefficient_error",
        "codec_effective_errors", "actor_optimizer_state_cleared",
    )
    paths = [folder / name for name in ("steps.csv", "episodes.csv", "rollouts.csv")]
    streams = [path.open("a" if path.is_file() else "w", newline="", encoding="utf-8") for path in paths]
    writers = [csv.DictWriter(stream, fieldnames=fields) for stream, fields in zip(streams, (step_fields, episode_fields, rollout_fields))]
    for path, writer in zip(paths, writers):
        if path.stat().st_size == 0:
            writer.writeheader()
    return streams, writers


def _run_schedule(
    agent,
    env,
    runtime: dict[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    *,
    folder: Path,
    phase: str,
    action: str | None = None,
    future_regime: str | None = None,
) -> list[dict[str, Any]]:
    folder.mkdir(parents=True, exist_ok=True)
    streams, writers = _headers(folder)
    steps_writer, episodes_writer, rollouts_writer = writers
    bout_records: list[dict[str, Any]] = []
    try:
        for bout_record in schedule:
            bout = int(bout_record["bout"])
            task = str(bout_record["task"])
            group = int(bout_record["group"])
            steps = int(bout_record["steps"])
            slot, allocation = _prepare_task(agent, task, group)
            spec = task_from_name(task)
            env.set_task(spec)
            generation = int(agent.slot_generation[slot])
            bout_rewards: list[float] = []
            group_reward = 0.0
            rollout_index = 0
            for bout_step in range(1, steps + 1):
                obs = np.asarray(runtime["obs"], dtype=np.float32)
                policy_obs = loop.augment_observation(obs, slot + 1)
                probabilities, value = _policy_probabilities(agent.core, policy_obs)
                uniform = float(agent.behavior_rng.random())
                action_id = min(int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")), 3)
                log_prob = float(math.log(max(float(probabilities[action_id]), 1e-12)))
                next_obs, reward, terminated, truncated, info = env.step(action_id)
                next_policy_obs = loop.augment_observation(next_obs, slot + 1)
                _, next_value = _policy_probabilities(agent.core, next_policy_obs)
                if terminated:
                    next_value = 0.0
                environment_done = bool(terminated or truncated)
                learning_boundary = bout_step == steps
                done = bool(environment_done or learning_boundary)
                episode_id = int(runtime["episode_ordinal"])
                row = loop.Transition(
                    policy_obs.copy(), action_id, float(reward), value, next_value,
                    log_prob, probabilities.astype(np.float32).copy(), done, bool(terminated),
                    slot, generation, int(agent.runtime_counts["environment_steps"]) + 1,
                )
                row.episode_id = episode_id
                row.task_name = task
                row.group_id = group
                row.bout = bout
                agent.rollout.append(row)
                agent.runtime_counts["environment_steps"] += 1
                runtime["episode_return"] += float(reward)
                runtime["episode_length"] += 1
                runtime["episode_goal"] = bool(runtime["episode_goal"] or info.get("goal_reached", False))
                runtime["episode_last_task"] = task
                runtime["episode_last_group"] = group
                bout_rewards.append(float(reward))
                group_reward += float(reward)
                steps_writer.writerow({
                    "phase": phase, "action": action or "", "future_regime": future_regime or "",
                    "global_step": agent.runtime_counts["environment_steps"], "bout": bout,
                    "bout_step": bout_step, "episode_id": episode_id,
                    "reset_seed": runtime["episode_reset_seed"], "taskobserver": task,
                    "group": group, "slot": slot, "generation": generation,
                    "obs": json.dumps(np.asarray(obs).tolist(), separators=(",", ":")),
                    "next_obs": json.dumps(np.asarray(next_obs).tolist(), separators=(",", ":")),
                    "action_id": action_id, "reward": float(reward),
                    "probabilities": json.dumps(probabilities.tolist(), separators=(",", ":")),
                    "action_uniform": uniform, "behavior_log_prob": log_prob,
                    "terminated": int(bool(terminated)), "truncated": int(bool(truncated)),
                    "environment_done": int(environment_done), "learning_boundary": int(learning_boundary),
                    "goal_reached": int(bool(info.get("goal_reached", False))),
                })
                runtime["obs"] = np.asarray(next_obs, dtype=np.float32).copy()
                agent.counts["raw_steps"] += 1
                agent.counts["environment_steps"] += 1
                agent.counts["fit_steps"] += 1

                if len(agent.rollout) == 256:
                    rows = agent.rollout
                    agent.rollout = []
                    loop._finalize_gae(rows)
                    agent.evidence.add_rollout(rows, group=group, task=task, bout=bout)
                    update = loop._ppo_update(agent, rows)
                    agent.runtime_counts["rollouts_trained"] += 1
                    agent.runtime_counts["minibatch_updates"] += int(update["updates"])
                    if int(update["updates"]) > 0:
                        valid_rows = sum(int(item.slot == slot and item.generation == generation) for item in rows)
                        agent.fitted_steps[slot] += np.uint64(valid_rows)
                    actor_data, coefficient_error, effective_errors = _encode_live_actor(agent)
                    for row_index, old_row in enumerate(rows):
                        del row_index, old_row
                    rollouts_writer.writerow({
                        "phase": phase, "action": action or "", "future_regime": future_regime or "",
                        "bout": bout, "rollout_index": rollout_index, "rows": len(rows),
                        "updates": int(update["updates"]), "policy_loss": update["policy_loss"],
                        "value_loss": update["value_loss"], "entropy": update["entropy"],
                        "approx_kl": update["approx_kl"],
                        "codec_max_coefficient_error": coefficient_error,
                        "codec_effective_errors": json.dumps(effective_errors),
                        "actor_optimizer_state_cleared": 1,
                    })
                    ws._clear_optimizer_state(agent)
                    rollout_index += 1

                if environment_done:
                    episodes_writer.writerow({
                        "phase": phase, "action": action or "", "future_regime": future_regime or "",
                        "episode_id": episode_id, "reset_seed": runtime["episode_reset_seed"],
                        "next_reset_seed": _reset_seed(agent.seed, runtime["episode_ordinal"] + 1),
                        "task_at_completion": runtime["episode_last_task"],
                        "group_at_completion": runtime["episode_last_group"],
                        "return_value": float(runtime["episode_return"]),
                        "length": int(runtime["episode_length"]),
                        "goal_reached": int(bool(runtime["episode_goal"])),
                        "terminated": int(bool(terminated)), "truncated": int(bool(truncated)),
                    })
                    agent.runtime_counts["completed_episodes"] += 1
                    agent.counts["completed_episodes"] += 1
                    agent.counts["all_goal_episodes"] += int(bool(runtime["episode_goal"]))
                    agent.counts["positive_completed_episodes"] += int(bool(runtime["episode_goal"]))
                    runtime["episode_ordinal"] += 1
                    runtime["episode_return"] = 0.0
                    runtime["episode_length"] = 0
                    runtime["episode_goal"] = False
                    runtime["episode_reset_seed"] = _reset_seed(agent.seed, runtime["episode_ordinal"])
                    runtime["obs"], _ = env.reset(seed=runtime["episode_reset_seed"])
                    runtime["obs"] = np.asarray(runtime["obs"], dtype=np.float32)

            if agent.rollout:
                raise RuntimeError("bout ended with a partial rollout; schedules must align to rollout_steps")
            agent.recent_groups.append(group)
            agent.recent_groups = agent.recent_groups[-8:]
            full_return = float(sum(bout_rewards))
            bout_records.append({
                "phase": phase, "action": action, "future_regime": future_regime,
                "bout": bout, "task": task, "group": group, "slot": slot,
                "generation": generation, "allocated": bool(allocation["allocated"]),
                "physical_group_at_start": int(allocation["physical_group_at_start"]),
                "parent_group_at_start": int(allocation["parent_group_at_start"]),
                "prior_inherited": bool(allocation["inherited_parent"]),
                "evicted_task": allocation["evicted_task"], "steps": steps,
                "return_first_256": float(sum(bout_rewards[:256])),
                "return_first_512": float(sum(bout_rewards[:512])),
                "return_full": full_return,
            })
    finally:
        for stream in streams:
            stream.flush()
            stream.close()
    return bout_records


def _runtime_initial(seed: int, first_task: str) -> tuple[Any, dict[str, Any]]:
    env = make_env(task_from_name(first_task), "navigation")
    obs, _ = env.reset(seed=_reset_seed(seed, 0))
    return env, {
        "obs": np.asarray(obs, dtype=np.float32),
        "episode_ordinal": 0,
        "episode_reset_seed": _reset_seed(seed, 0),
        "episode_return": 0.0,
        "episode_length": 0,
        "episode_goal": False,
        "episode_last_task": first_task,
        "episode_last_group": 0,
    }


def _save_simulator(path: Path, env, runtime: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _capture_simulator(env, runtime)
    _write_json(path, snapshot)
    return snapshot


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_bytes(folder: Path, foundation: Path, actor_data: bytes, config: Mapping[str, Any]) -> dict[str, int]:
    learner = _capture_learner_from_saved(folder / "checkpoint.pt")
    return _save_package(folder, learner, actor_data, foundation, config)


def _capture_learner_from_saved(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _count_step_rows(path: Path) -> int | None:
    """Count persisted raw transitions, returning None for an unreadable ledger."""
    if not path.is_file():
        return 0
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            rows = sum(1 for _ in reader)
        return max(0, rows - 1)
    except (OSError, UnicodeError, csv.Error):
        return None


def _persisted_step_counts(case_folder: Path) -> dict[str, Any]:
    """Recover failed-case interaction counts from flushed prefix/branch CSVs."""
    prefix = _count_step_rows(case_folder / "prefix" / "steps.csv")
    branch_values = []
    branches_dir = case_folder / "branches"
    if branches_dir.is_dir():
        for branch in sorted(branches_dir.iterdir()):
            if branch.is_dir():
                branch_values.append(_count_step_rows(branch / "steps.csv"))
    branch_total = None if any(value is None for value in branch_values) else int(sum(branch_values))
    total = None if prefix is None or branch_total is None else int(prefix + branch_total)
    return {
        "prefix_environment_steps": prefix,
        "branch_environment_steps": branch_total,
        "environment_steps_total": total,
    }


def _case_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    seed, pair_kind, history_regime = int(payload["seed"]), str(payload["pair_kind"]), str(payload["history_regime"])
    config = validate_config(payload["config"])
    case_id = f"s{seed}_{pair_kind}_{history_regime}"
    case_folder = Path(payload["output"]) / "cases" / case_id
    base_config = loop.load_base_config()
    foundation = Path(base_config["foundation"])
    foundation_sha = _sha(foundation)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    mapping = pair_mapping(seed, pair_kind)
    schedule = prefix_schedule(mapping, config, history_regime)
    env, runtime = _runtime_initial(seed, schedule[0]["task"])
    agent = _new_agent(seed, base_config, config)
    prefix_start_cpu, prefix_start_wall = time.process_time(), time.perf_counter()
    prefix_folder = case_folder / "prefix"
    prefix_bouts = _run_schedule(agent, env, runtime, schedule[:2], folder=prefix_folder,
                                 phase="bootstrap")

    # Form the one retained old parent from the two live old locals.
    live_old_slots = [slot for slot, group in enumerate(agent.slot_group_tags) if int(group) == 0]
    if len(live_old_slots) != 2:
        raise RuntimeError(f"old parent formation requires two live old slots, found {live_old_slots}")
    local, shared, physical_groups = ws._actor_arrays(agent.core)
    old_candidate = common_mean_effective(local, shared, physical_groups, live_old_slots)
    # The pre-formation modules are unbound, so install the formed parent
    # directly with the requested old group and compensate both residents.
    from online_rl.storage_decision import compensated_single_parent_install
    old_local, old_shared, old_groups, formation = compensated_single_parent_install(
        local, shared, physical_groups, agent.slot_group_tags,
        parent=old_candidate, parent_group=0,
    )
    agent.core.install_local_matrix(old_local)
    agent.core.install_shared_matrix(old_shared)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(old_groups.astype(np.int64), dtype=torch.long))
    agent.parent_group = 0
    formation_actor, formation_codec_error, formation_effective_error = _encode_live_actor(agent)
    ws._clear_optimizer_state(agent)

    recent_schedule = schedule[2:]
    prefix_bouts.extend(_run_schedule(agent, env, runtime, recent_schedule,
                                      folder=prefix_folder, phase="prefix_history"))
    if len(agent.recent_groups) != 8 or agent.recent_groups[-2:] != [1, 1]:
        raise RuntimeError("prefix history must leave recent new-group variants zero and one")
    live_new_slots = [slot for slot, group in enumerate(agent.slot_group_tags) if int(group) == 1]
    if len(live_new_slots) != 2:
        raise RuntimeError(f"candidate formation requires two live new slots, found {live_new_slots}")
    local, shared, physical_groups = ws._actor_arrays(agent.core)
    old_parent = shared[0].reshape(3, 3).copy()
    new_parent = common_mean_effective(local, shared, physical_groups, live_new_slots)

    # The serialized prefix is the common fork point. Candidate arrays stay in
    # the observer decision record and are passed transiently to branch code.
    prefix_actor, prefix_codec_error, prefix_codec_effective_error = _encode_live_actor(agent)
    prefix_learner = _capture_learner(agent, foundation_sha)
    prefix_simulator = _capture_simulator(env, runtime)
    prefix_fork_hash = semantic_hash({"learner": prefix_learner, "simulator": prefix_simulator})
    prefix_learner_hash = semantic_hash(prefix_learner)
    prefix_capacity = _save_package(prefix_folder, prefix_learner, prefix_actor, foundation, config)
    _write_json(prefix_folder / "simulator.json", prefix_simulator)
    disk_prefix_learner = _capture_learner_from_saved(prefix_folder / "checkpoint.pt")
    disk_prefix_simulator = _load_json(prefix_folder / "simulator.json")
    if semantic_hash(disk_prefix_learner) != prefix_learner_hash:
        raise RuntimeError("serialized prefix checkpoint does not match its in-memory state")
    if semantic_hash({"learner": disk_prefix_learner, "simulator": disk_prefix_simulator}) != prefix_fork_hash:
        raise RuntimeError("serialized prefix simulator does not match its in-memory state")
    prefix_elapsed = {
        "cpu_s": float(time.process_time() - prefix_start_cpu),
        "wall_s": float(time.perf_counter() - prefix_start_wall),
    }
    prefix_summary = _summarize_bouts(prefix_bouts, agent, prefix_elapsed)
    prefix_summary.update({
        "seed": seed, "pair_kind": pair_kind, "history_regime": history_regime,
        "prefix_state_hash": prefix_fork_hash, "foundation_sha256": foundation_sha,
        "codec_max_coefficient_error": max(formation_codec_error, prefix_codec_error),
        "formation_effective_errors": formation["per_slot_preservation_error"],
        "formation_codec_effective_errors": formation_effective_error,
        "shared_frozen": bool(agent.core._shared_frozen),
        "capacity": prefix_capacity,
        "actor_payload_bytes": len(prefix_actor),
    })
    _write_json(prefix_folder / "summary.json", prefix_summary)

    score = _score_fn(agent.core, config)
    selectors = decide_from_past(agent.recent_groups, agent.evidence, old_parent, new_parent, score)
    decision = {
        "format": "storage_decision_case_v1",
        "case_id": case_id,
        "seed": seed,
        "pair_kind": pair_kind,
        "history_regime": history_regime,
        "mapping": mapping,
        "bootstrap_schedule": schedule[:2],
        "history_schedule": recent_schedule,
        "prefix_state_hash": prefix_fork_hash,
        "restored_state_hashes": {},
        "foundation_sha256": foundation_sha,
        "old_parent": old_parent.tolist(),
        "new_candidate": new_parent.tolist(),
        "temporary_candidate_workspace_bytes": 36,
        "prequantized_invariance_error": float(formation["max_effective_preservation_error"]),
        "formation_codec_coefficient_error": formation_codec_error,
        "formation_codec_effective_errors": formation_effective_error,
        "prefix_codec_coefficient_error": prefix_codec_error,
        "prefix_codec_effective_errors": prefix_codec_effective_error,
        "resident_slots_at_decision": [
            {"slot": int(slot), "task": agent.cache.slot_to_task[slot],
             "logical_group": int(agent.slot_group_tags[slot]),
             "physical_group": int(agent.core.group_for_slot(slot)),
             "generation": int(agent.slot_generation[slot])}
            for slot in range(agent.slots)
        ],
        "selectors": selectors["past_only_selectors"],
        "selector_diagnostics": selectors,
        "selector_contract": {
            "format": "storage_decision_past_selector_contract_v1",
            "inputs": {
                "recent_group_ids": "selector_diagnostics.recent_group_ids",
                "group_evidence": "prefix/checkpoint.pt:evidence",
                "old_parent": "old_parent",
                "new_candidate": "new_candidate",
                "surrogate_score_parameters": "config.gate",
            },
            "evidence_capacity_rows_per_group": int(config["evidence_rows_per_group"]),
            "evidence_support_rows": selectors["support_rows"],
            "outputs": "selector_diagnostics.past_only_selectors",
            "rules": {
                "frequency": "replace iff recent group 1 count exceeds group 0 count; ties keep",
                "recency": "replace iff the latest recent group 1 bout is after the latest group 0 bout",
                "surrogate_gain": "replace iff both groups have supported finite gain scores and the recent-frequency weighted score is positive; otherwise keep",
                "surrogate_lower": "replace iff both groups have supported finite lower scores and the recent-frequency weighted score is positive; otherwise keep",
                "fixed_actions": ["keep", "replace", "mean", "none"],
            },
        },
        "evidence_rows_per_group": selectors["support_rows"],
        "prefix_package": prefix_summary["capacity"],
        "wall_cpu": prefix_elapsed,
        "source_w0_unchanged_at_decision": _sha(foundation) == foundation_sha,
        "branches": [],
    }
    _write_json(case_folder / "decision.json", decision)

    branch_results: list[dict[str, Any]] = []
    active_branch_started: tuple[float, float] | None = None
    try:
        for future_regime in config["future_regimes"]:
            future = template_schedule(
                mapping, str(future_regime), kind="future", steps=int(config["bout_steps"]),
                start_bout=len(schedule),
            )
            for chosen_action in ACTIONS:
                restore_cpu, restore_wall = time.process_time(), time.perf_counter()
                active_branch_started = (restore_cpu, restore_wall)
                restored = _restore_learner(disk_prefix_learner, base_config, config)
                restored_env, restored_runtime = _restore_simulator(disk_prefix_simulator)
                restored_payload = _capture_learner(restored, foundation_sha)
                restored_snapshot = _capture_simulator(restored_env, restored_runtime)
                restored_hash = semantic_hash({"learner": restored_payload, "simulator": restored_snapshot})
                decision["restored_state_hashes"][chosen_action + "__" + str(future_regime)] = restored_hash
                if restored_hash != prefix_fork_hash:
                    raise RuntimeError("restored fork state does not match the common prefix semantic hash")

                branch_folder = case_folder / "branches" / f"{chosen_action}__{future_regime}"
                branch_folder.mkdir(parents=True, exist_ok=True)
                branch_local, branch_shared, branch_groups = ws._actor_arrays(restored.core)
                branch_start_counts = dict(restored.runtime_counts)
                fork_observation = np.asarray(restored_runtime["obs"], dtype=np.float32)
                fork_probabilities = _resident_probabilities(restored.core, fork_observation, restored.slots)
                before_effective = _effective_live(branch_local, branch_shared, branch_groups)
                post_local, post_shared, post_groups, intervention = apply_action(
                    branch_local, branch_shared, branch_groups, restored.slot_group_tags,
                    old_parent, new_parent, chosen_action,
                )
                restored.core.install_local_matrix(post_local)
                restored.core.install_shared_matrix(post_shared)
                with torch.no_grad():
                    restored.core.slot_groups.copy_(torch.as_tensor(post_groups.astype(np.int64), dtype=torch.long))
                restored.parent_group = {"keep": 0, "replace": 1, "mean": -2, "none": -1}[chosen_action]
                after_prequant = _effective_live(post_local, post_shared, post_groups)
                prequant_per_slot = np.max(np.abs(after_prequant - before_effective), axis=(1, 2, 3)).astype(float).tolist()
                prequant_probabilities = _resident_probabilities(restored.core, fork_observation, restored.slots)
                prequant_behavior_drift = _probability_drift(fork_probabilities, prequant_probabilities)
                action_actor, action_codec_error, action_codec_effective_error = _encode_live_actor(restored)
                postquant_probabilities = _resident_probabilities(restored.core, fork_observation, restored.slots)
                codec_behavior_drift = _probability_drift(fork_probabilities, postquant_probabilities)
                ws._clear_optimizer_state(restored)
                action_checkpoint = _capture_learner(restored, foundation_sha)
                action_hash = semantic_hash({"learner": action_checkpoint, "simulator": restored_snapshot})
                post_action_capacity = _save_package(branch_folder, action_checkpoint, action_actor, foundation, config)
                post_action_folder = branch_folder / "post_action"
                post_action_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(branch_folder / "checkpoint.pt", post_action_folder / "checkpoint.pt")
                shutil.copy2(branch_folder / "actor.bin", post_action_folder / "actor.bin")
                _write_json(branch_folder / "simulator_start.json", restored_snapshot)
                _write_json(post_action_folder / "simulator.json", restored_snapshot)
                _write_json(post_action_folder / "manifest.json", {
                    "role": "observer_audit_snapshot",
                    "included_in_deployment_capacity": False,
                    "action_state_hash": action_hash,
                })

                branch_start_cpu, branch_start_wall = time.process_time(), time.perf_counter()
                branch_bouts = _run_schedule(
                    restored, restored_env, restored_runtime, future,
                    folder=branch_folder, phase="future", action=chosen_action,
                    future_regime=str(future_regime),
                )
                final_actor, final_codec_error, final_effective_error = _encode_live_actor(restored)
                final_learner = _capture_learner(restored, foundation_sha)
                final_simulator = _capture_simulator(restored_env, restored_runtime)
                final_hash = semantic_hash({"learner": final_learner, "simulator": final_simulator})
                package = _save_package(branch_folder, final_learner, final_actor, foundation, config)
                _write_json(branch_folder / "simulator.json", final_simulator)
                if semantic_hash(prefix_learner) != prefix_learner_hash:
                    raise RuntimeError("a branch mutated the common in-memory prefix learner payload")
                if semantic_hash(disk_prefix_learner) != prefix_learner_hash:
                    raise RuntimeError("a branch mutated the common serialized prefix learner payload")
                elapsed = {"cpu_s": float(time.process_time() - restore_cpu),
                           "wall_s": float(time.perf_counter() - restore_wall),
                           "simulation_cpu_s": float(time.process_time() - branch_start_cpu),
                           "simulation_wall_s": float(time.perf_counter() - branch_start_wall),
                           "restore_cpu_s": float(time.process_time() - restore_cpu),
                           "restore_wall_s": float(time.perf_counter() - restore_wall)}
                summary = _summarize_bouts(branch_bouts, restored, elapsed, start_counts=branch_start_counts)
                summary.update({
                    "format": "storage_decision_branch_v1",
                    "case_id": case_id,
                    "seed": seed,
                    "pair_kind": pair_kind,
                    "history_regime": history_regime,
                    "action": chosen_action,
                    "future_regime": str(future_regime),
                    "primary_continuation": history_regime == str(future_regime),
                    "stress_continuation": history_regime != str(future_regime),
                    "prefix_state_hash": prefix_fork_hash,
                    "restored_state_hash": restored_hash,
                    "action_state_hash": action_hash,
                    "final_state_hash": final_hash,
                    "foundation_sha256": foundation_sha,
                    "foundation_unchanged": _sha(foundation) == foundation_sha,
                    "prequantized_effective_errors": prequant_per_slot,
                    "fork_observation": fork_observation.tolist(),
                    "prequantized_behavior_drift": prequant_behavior_drift,
                    "postquantized_behavior_drift": codec_behavior_drift,
                    "prequantized_resident_probabilities": prequant_probabilities.tolist(),
                    "postquantized_resident_probabilities": postquant_probabilities.tolist(),
                    "codec_max_coefficient_error": action_codec_error,
                    "codec_effective_errors": action_codec_effective_error,
                    "final_codec_max_coefficient_error": final_codec_error,
                    "final_codec_effective_errors": final_effective_error,
                    "preservation_diagnostic": intervention,
                    "capacity": package,
                    "capacity_after_action": post_action_capacity,
                    "post_action_checkpoint": "post_action/checkpoint.pt",
                    "post_action_simulator": "post_action/simulator.json",
                    "post_action_snapshot_included_in_deployment_capacity": False,
                    "actor_payload_bytes": len(final_actor),
                    "temporary_candidate_workspace_bytes": 36,
                    "future_schedule": future,
                    "prefix_bouts_included_in_reward": False,
                })
                _write_json(branch_folder / "summary.json", summary)
                branch_results.append(summary)
                active_branch_started = None
                decision["branches"].append({
                    "action": chosen_action,
                    "future_regime": str(future_regime),
                    "summary": f"branches/{chosen_action}__{future_regime}/summary.json",
                })
                _write_json(case_folder / "decision.json", decision)
                restored_env.close()
        decision["restored_state_hashes"] = dict(decision["restored_state_hashes"])
        _write_json(case_folder / "decision.json", decision)
        result = {
            "case_id": case_id, "seed": seed, "pair_kind": pair_kind,
            "history_regime": history_regime,
            "prefix": prefix_summary,
            "selectors": decision["selectors"],
            "selector_diagnostics": decision["selector_diagnostics"],
            "branches": branch_results,
            "costs": {
                "prefix_environment_steps": int(prefix_summary["environment_steps"]),
                "branch_environment_steps": int(sum(row["environment_steps"] for row in branch_results)),
                "environment_steps_total": int(prefix_summary["environment_steps"] + sum(row["environment_steps"] for row in branch_results)),
                "prefix_cpu_s": prefix_elapsed["cpu_s"],
                "branch_cpu_s": float(sum(row["costs"]["cpu_s"] for row in branch_results)),
                "prefix_wall_s": prefix_elapsed["wall_s"],
                "branch_wall_s": float(sum(row["costs"]["wall_s"] for row in branch_results)),
            },
            "errors": [],
        }
    except Exception:
        persisted = _persisted_step_counts(case_folder)
        error = {"case_id": case_id, "error_type": "branch_failure", "error": traceback.format_exc(),
                 "persisted_interactions": persisted, "count_source": "steps.csv rows"}
        _write_json(case_folder / "error.json", error)
        branch_cpu_partial = float(sum(row["costs"]["cpu_s"] for row in branch_results))
        branch_wall_partial = float(sum(row["costs"]["wall_s"] for row in branch_results))
        if active_branch_started is not None:
            branch_cpu_partial += max(0.0, time.process_time() - active_branch_started[0])
            branch_wall_partial += max(0.0, time.perf_counter() - active_branch_started[1])
        result = {
            "case_id": case_id, "seed": seed, "pair_kind": pair_kind,
            "history_regime": history_regime, "prefix": prefix_summary,
            "selectors": decision["selectors"], "selector_diagnostics": decision["selector_diagnostics"],
            "branches": branch_results, "costs": {
                "prefix_environment_steps": persisted["prefix_environment_steps"],
                "branch_environment_steps": persisted["branch_environment_steps"],
                "environment_steps_total": persisted["environment_steps_total"],
                "prefix_cpu_s": prefix_elapsed["cpu_s"], "branch_cpu_s": branch_cpu_partial,
                "prefix_wall_s": prefix_elapsed["wall_s"], "branch_wall_s": branch_wall_partial},
            "errors": [error],
        }
    finally:
        env.close()
    if _sha(foundation) != foundation_sha:
        result["errors"].append({"case_id": case_id, "error_type": "foundation_changed", "error": "W0 SHA changed during case"})
    return result


def _summarize_bouts(
    bouts: Sequence[Mapping[str, Any]], agent, elapsed: Mapping[str, float],
    *, start_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    total_reward = float(sum(float(row["return_full"]) for row in bouts))
    total_steps = int(sum(int(row["steps"]) for row in bouts))
    per_group = {}
    for group in (0, 1):
        rows = [row for row in bouts if int(row["group"]) == group]
        reward = float(sum(float(row["return_full"]) for row in rows))
        steps = int(sum(int(row["steps"]) for row in rows))
        per_group[str(group)] = {"reward_total": reward, "steps": steps, "reward_per_step": reward / max(1, steps)}
    baseline = dict(start_counts or {})
    count_names = ("environment_steps", "completed_episodes", "slot_allocations", "slot_evictions",
                   "rollouts_trained", "minibatch_updates")
    counts = {name: int(agent.runtime_counts.get(name, 0)) - int(baseline.get(name, 0)) for name in count_names}
    return {
        "environment_steps": total_steps,
        "reward_total": total_reward,
        "reward_per_step": total_reward / max(1, total_steps),
        "per_group": per_group,
        "per_bout": list(bouts),
        "counts": counts,
        "slot_allocations": counts["slot_allocations"],
        "slot_evictions": counts["slot_evictions"],
        "rollouts_trained": counts["rollouts_trained"],
        "minibatch_updates": counts["minibatch_updates"],
        "costs": {key: float(value) for key, value in elapsed.items()},
    }


def _case_payloads(config: Mapping[str, Any], output: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), "pair_kind": pair, "history_regime": history,
         "config": dict(config), "output": str(output)}
        for seed in seeds
        for pair in config["pair_kinds"]
        for history in config["history_regimes"]
    ]


def run(config: Mapping[str, Any] | str | Path, output: str | Path, *, seeds: Sequence[int] | None = None, pilot: bool = False) -> dict[str, Any]:
    config_path = Path(config).resolve() if isinstance(config, (str, Path)) else CONFIG_PATH
    raw_config = json.loads(config_path.read_text(encoding="utf-8")) if isinstance(config, (str, Path)) else dict(config)
    phase_config = copy.deepcopy(raw_config)
    phase = "pilot" if pilot else "main"
    if pilot:
        pilot_config = dict(phase_config["pilot"])
        for key in ("seeds", "pair_kinds", "history_regimes", "bootstrap_steps", "bout_steps"):
            phase_config[key] = pilot_config[key]
    normalized = validate_config(phase_config)
    run_seeds = tuple(int(value) for value in (seeds if seeds is not None else normalized["seeds"]))
    if not run_seeds:
        raise ValueError("at least one seed is required")
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    _write_json(output_path / "config.json", normalized)
    _write_json(output_path / "source_snapshot" / "config_original.json", raw_config)
    source_manifest = _snapshot_sources(output_path, config_path)
    jobs = _case_payloads(normalized, output_path, run_seeds)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    workers = max(1, min(int(normalized["max_workers"]), len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_case_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record = future.result()
                records.append(record)
                errors.extend(record.get("errors", []))
            except Exception:
                case_id = f"s{job['seed']}_{job['pair_kind']}_{job['history_regime']}"
                case_folder = output_path / "cases" / case_id
                persisted = _persisted_step_counts(case_folder)
                error = {"case_id": case_id, "error_type": "case_failure", "error": traceback.format_exc(),
                         "persisted_interactions": persisted, "count_source": "steps.csv rows"}
                _write_json(case_folder / "error.json", error)
                errors.append(error)
                records.append({"case_id": error["case_id"], "seed": job["seed"],
                                "pair_kind": job["pair_kind"], "history_regime": job["history_regime"],
                                "branches": [], "errors": [error],
                                "costs": {**persisted, "prefix_cpu_s": None,
                                          "branch_cpu_s": None, "prefix_wall_s": None, "branch_wall_s": None}})
    records.sort(key=lambda row: (int(row["seed"]), str(row["pair_kind"]), str(row["history_regime"])))
    branches = [branch for record in records for branch in record.get("branches", [])]
    def aggregate_cost(key: str) -> int | float | None:
        values = [record.get("costs", {}).get(key) for record in records]
        if any(value is None for value in values):
            return None
        return sum(values)

    costs = {
        "prefix_environment_steps": aggregate_cost("prefix_environment_steps"),
        "branch_environment_steps": aggregate_cost("branch_environment_steps"),
        "environment_steps_total": aggregate_cost("environment_steps_total"),
        "cpu_s": float(time.process_time() - start_cpu),
        "wall_s": float(time.perf_counter() - start_wall),
        "branch_cpu_s": aggregate_cost("branch_cpu_s"),
        "prefix_cpu_s": aggregate_cost("prefix_cpu_s"),
        "branch_wall_s": aggregate_cost("branch_wall_s"),
        "prefix_wall_s": aggregate_cost("prefix_wall_s"),
    }
    summary = {
        "format": "storage_decision_summary_v1", "phase": phase,
        "config": normalized, "source_manifest": source_manifest,
        "seeds": list(run_seeds), "case_count": len(records), "branch_count": len(branches),
        "cases": records, "branches": branches, "errors": errors, "costs": costs,
    }
    _write_json(output_path / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--pilot", action="store_true")
    arguments = parser.parse_args(argv)
    summary = run(arguments.config, arguments.output, seeds=arguments.seeds, pilot=arguments.pilot)
    print(json.dumps({"output": str(arguments.output), "phase": summary["phase"],
                      "case_count": summary["case_count"], "branch_count": summary["branch_count"],
                      "errors": len(summary["errors"]), "costs": summary["costs"]}, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run S1b with known routes and profile-only automatic shared grouping."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import torch

from online_rl.c1_envs import C1NavigationEnv, C1_CONTEXTS
from online_rl.c1_supplement_experiment import _foundation, seed_everything
from online_rl.policy import LeafAdam
from online_rl.shared_write_grouping import (
    DynamicsGrouper,
    empty_profile,
    observe_profile,
    supported_directions,
)
from online_rl.shared_write_policy import SharedWritePolicy
from online_rl.shared_write_storage import decode_shared_knowledge, encode_shared_knowledge
from scripts import run_knowledge_loop as loop
from scripts import run_shared_write as shared_base
from scripts.shared_write_evaluation import evaluate_context


CONFIG_PATH = ROOT / "configs/shared_write_auto.json"
CONDITIONS = ("independent", "oracle_active", "auto_active", "auto_frozen")
AUTOMATIC_CONDITIONS = frozenset(("auto_active", "auto_frozen"))
SHARED_CONDITIONS = frozenset(("oracle_active", *AUTOMATIC_CONDITIONS))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def resolve_phase_config(config: Mapping[str, Any], phase: str) -> tuple[dict[str, Any], list[int]]:
    """Apply frozen pilot/formal schedules without changing the source config."""
    if phase not in ("pilot", "formal"):
        raise ValueError("phase must be pilot or formal")
    resolved = dict(config)
    resolved["grouper"] = dict(config["grouper"])
    resolved["heldout_sequence"] = list(config["heldout_sequence"])
    if phase == "pilot":
        resolved["seen_sequence"] = list(config["pilot_seen_sequence"])
        resolved["bout_steps"] = int(config["pilot_bout_steps"])
        resolved["heldout_steps"] = int(config["pilot_heldout_steps"])
        resolved["cold_evaluation_episodes"] = int(config["pilot_cold_evaluation_episodes"])
        resolved["retention_evaluation_episodes"] = int(config["pilot_retention_evaluation_episodes"])
        seeds = [int(config["pilot_seed"])]
    else:
        resolved["seen_sequence"] = list(config["seen_sequence"])
        resolved["bout_steps"] = int(config["bout_steps"])
        resolved["heldout_steps"] = int(config["heldout_steps"])
        seeds = [int(value) for value in config["formal_seeds"]]
    return resolved, seeds


def make_agent(condition: str, seed: int, config: Mapping[str, Any]):
    """Create a fresh W0 policy, condition-specific optimizer, and grouper."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    seed_everything(seed)
    base = loop.load_base_config()
    reference = _foundation(base, seed)
    try:
        core = SharedWritePolicy.from_reference(
            reference.policy,
            shared_groups=2 if condition in SHARED_CONDITIONS else 0,
            exploration_epsilon=base["exploration_epsilon"],
            temperature=base["temperature"],
        )
    finally:
        reference.env.close()

    permutation = np.random.default_rng(seed + 8811).permutation(6)
    slots = {name: int(permutation[index]) for index, name in enumerate(C1_CONTEXTS)}
    if condition == "oracle_active":
        for name, slot in slots.items():
            core.set_slot_group(slot, 0 if C1_CONTEXTS[name].family == "A" else 1, preserve_effective=False)

    grouper = (
        DynamicsGrouper(**dict(config["grouper"]))
        if condition in AUTOMATIC_CONDITIONS
        else None
    )
    shared_parameters = list(core.shared_parameters())
    local_parameters = [core._nodes[str(endpoint)].A for endpoint in range(1, 7)]
    actor_ids = {id(parameter) for parameter in (*shared_parameters, *local_parameters)}
    critic_parameters = [
        parameter
        for parameter in core.parameters()
        if parameter.requires_grad and id(parameter) not in actor_ids
    ]
    groups = [
        {"params": local_parameters, "lr": float(config["local_learning_rate"]), "name": "local"},
        {"params": critic_parameters, "lr": float(config["critic_learning_rate"]), "name": "critic"},
    ]
    if shared_parameters:
        groups.append({"params": shared_parameters, "lr": float(config["shared_learning_rate"]), "name": "shared"})
    optimizer = LeafAdam(groups, eps=1e-5)
    agent = SimpleNamespace(
        core=core,
        optimizer=optimizer,
        arm="int8",
        condition=condition,
        seed=int(seed),
        action_rng=np.random.default_rng(seed + 1100),
        slot_generation=np.ones(6, dtype=np.int64),
        grouper=grouper,
        probe_profiles={},
        group_decisions=[],
        group_snapshots=[],
        counts={
            "stale_actor_rows": 0,
            "minibatch_updates": 0,
            "rollouts_trained": 0,
            "fit_steps": 0,
            "holdout_steps": 0,
            "environment_steps": 0,
            "probe_steps": 0,
        },
    )
    return agent, slots, base


def _probe_episode_seed(seed: int, context_index: int, stream_tag: int, episode: int) -> int:
    state = np.random.SeedSequence(
        [int(seed), int(context_index), int(stream_tag), int(episode)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def collect_probe(
    context_name: str,
    slot: int,
    seed: int,
    *,
    steps: int,
    writer: csv.DictWriter,
    counts: dict[str, int],
    seed_tag: int = 32017,
    env_factory=C1NavigationEnv,
) -> dict[str, Any]:
    """Collect a fixed uniform-action probe without invoking the policy or PPO."""
    if context_name not in C1_CONTEXTS:
        raise ValueError(f"unknown context: {context_name}")
    if isinstance(steps, bool) or int(steps) != steps or int(steps) <= 0:
        raise ValueError("steps must be a positive integer")
    context_index = tuple(C1_CONTEXTS).index(context_name)
    env = env_factory(C1_CONTEXTS[context_name])
    profile = empty_profile()
    episode = 1
    reset_seed = _probe_episode_seed(seed, context_index, seed_tag + 1, episode)
    action_seed = _probe_episode_seed(seed, context_index, seed_tag, episode)
    action_rng = np.random.default_rng(action_seed)
    episode_action_index = 0
    obs, _ = env.reset(seed=reset_seed)
    counted = 0
    try:
        for probe_index in range(int(steps)):
            action = int(action_rng.integers(0, 4))
            next_obs, reward, terminated, truncated, _info = env.step(action)
            profile_counted = observe_profile(profile, obs, action, next_obs)
            done = bool(terminated or truncated)
            counts["probe_steps"] += 1
            counted += 1
            writer.writerow(
                {
                    "probe_step": counts["probe_steps"],
                    "context": context_name,
                    "slot": int(slot),
                    "episode": episode,
                    "reset_seed": reset_seed,
                    "episode_action_seed": action_seed,
                    "episode_action_index": episode_action_index,
                    "obs": json.dumps(np.asarray(obs, dtype=np.float32).tolist(), separators=(",", ":")),
                    "action": action,
                    "reward": float(reward),
                    "next_obs": json.dumps(np.asarray(next_obs, dtype=np.float32).tolist(), separators=(",", ":")),
                    "terminated": int(bool(terminated)),
                    "truncated": int(bool(truncated)),
                    "profile_counted": int(profile_counted),
                    "interaction_cost": 1,
                }
            )
            if done and probe_index + 1 < int(steps):
                episode += 1
                reset_seed = _probe_episode_seed(seed, context_index, seed_tag + 1, episode)
                action_seed = _probe_episode_seed(seed, context_index, seed_tag, episode)
                action_rng = np.random.default_rng(action_seed)
                episode_action_index = 0
                obs, _ = env.reset(seed=reset_seed)
            else:
                obs = next_obs
                episode_action_index += 1
    finally:
        env.close()
    return {
        "context": context_name,
        "slot": int(slot),
        "steps": counted,
        "episodes": episode,
        "profile": profile,
        "supported_directions": supported_directions(profile),
        "profile_sha256": hashlib.sha256(profile.tobytes()).hexdigest(),
    }


def assign_probe_profile(agent, context_name: str, slot: int, profile: np.ndarray) -> dict[str, Any]:
    """Make one initial assignment; automatic grouping sees only ``profile``."""
    if agent.condition == "oracle_active":
        # This is the sole condition allowed to use known family membership.
        group = 0 if C1_CONTEXTS[context_name].family == "A" else 1
        reason, scores = "oracle_family", {}
    elif agent.condition in AUTOMATIC_CONDITIONS:
        assignment = agent.grouper.assign(profile)
        group = int(assignment.group)
        reason = str(assignment.reason)
        scores = assignment.scores
    elif agent.condition == "independent":
        group, reason, scores = -1, "no_shared_groups", {}
    else:
        raise ValueError(f"unknown condition: {agent.condition}")

    agent.core.set_slot_group(int(slot), group, preserve_effective=False)
    family_for_audit = str(C1_CONTEXTS[context_name].family)
    return {
        "condition": agent.condition,
        "seed": int(agent.seed),
        "context": context_name,
        "slot": int(slot),
        "assigned_group": group,
        "reason": reason,
        "scores": scores,
        "profile_sha256": hashlib.sha256(np.asarray(profile, dtype=np.uint16).tobytes()).hexdigest(),
        "supported_cells": int(np.count_nonzero(supported_directions(profile) >= 0)),
        "observer_family_audit_only": family_for_audit,
    }


def probe_context(
    agent,
    context_name: str,
    slot: int,
    *,
    probe_steps: int,
    probe_writer: csv.DictWriter,
    decision_writer: csv.DictWriter,
    assignment_writer: csv.DictWriter,
    seed_tag: int = 32017,
):
    """Collect, record, and commit one profile and its initial slot decision."""
    probe = collect_probe(
        context_name,
        slot,
        agent.seed,
        steps=probe_steps,
        writer=probe_writer,
        counts=agent.counts,
        seed_tag=seed_tag,
    )
    profile = probe["profile"]
    agent.probe_profiles[context_name] = profile.copy()
    decision = assign_probe_profile(agent, context_name, slot, profile)
    agent.group_decisions.append(decision)
    decision_writer.writerow(
        {
            **decision,
            "scores": json.dumps(decision["scores"], sort_keys=True, separators=(",", ":")),
            "supported_directions": json.dumps(probe["supported_directions"].tolist(), separators=(",", ":")),
        }
    )
    assignment_writer.writerow(
        {
            **decision,
            "scores": json.dumps(decision["scores"], sort_keys=True, separators=(",", ":")),
        }
    )
    if agent.grouper is not None:
        agent.group_snapshots.append(
            {
                "context": context_name,
                "slot": int(slot),
                "assigned_group": decision["assigned_group"],
                "group_counts": agent.grouper.group_counts,
                "nbytes": agent.grouper.nbytes,
            }
        )
    return probe, decision


def actor_payload(agent, config):
    """Reuse the S1a exact actor encoding helper for the same matrix layout."""
    return shared_base.actor_payload(agent, config)


def snapshot(folder: Path, config: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    """Copy every source that determines this run and record exact hashes."""
    paths = [
        *loop.SOURCE_SNAPSHOT_FILES,
        Path("scripts/run_shared_write.py"),
        Path("scripts/run_shared_write_auto.py"),
        Path("scripts/shared_write_evaluation.py"),
        Path("src/online_rl/shared_write_policy.py"),
        Path("src/online_rl/shared_write_storage.py"),
        Path("src/online_rl/shared_write_grouping.py"),
        Path("configs/shared_write.json"),
        Path("configs/shared_write_auto.json"),
        Path("docs/SHARED_WRITE_PROTOCOL.md"),
        Path("docs/SHARED_WRITE_AUTO_PROTOCOL.md"),
    ]
    manifest: dict[str, str] = {}
    for relative in paths:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        destination = folder / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = sha(destination)
        if digest != sha(source):
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        manifest[relative.as_posix()] = digest
    write_json(
        folder / "source_manifest.json",
        {
            "sources": manifest,
            "foundation_sha256": sha(Path(base["foundation"])),
            "dataset_sha256": sha(Path(base["dataset"])),
        },
    )
    write_json(folder / "config.json", dict(config))


def save(agent, folder: Path, name: str, config: Mapping[str, Any], slots: Mapping[str, int], base: Mapping[str, Any]):
    packed = actor_payload(agent, config)
    checkpoint = {
        "format": "active_shared_write_auto_v1",
        "condition": agent.condition,
        "seed": agent.seed,
        "actor_binary": packed.data,
        "critic": loop._critic_state(agent.core),
        "optimizer": agent.optimizer.state_dict(),
        "optimizer_step_count": int(getattr(agent.optimizer, "step_count", 0)),
        "slot_groups": agent.core.slot_groups.detach().cpu().clone(),
        "shared_frozen": agent.core._shared_frozen,
        "slot_map": dict(slots),
        "counts": agent.counts.copy(),
        "action_rng": agent.action_rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "grouper_state": None if agent.grouper is None else agent.grouper.state_dict(),
        "probe_profiles": {name: profile.copy() for name, profile in agent.probe_profiles.items()},
        "group_decisions": list(agent.group_decisions),
        "group_snapshots": list(agent.group_snapshots),
        "foundation_sha256": sha(Path(base["foundation"])),
        "scope": "Endpoint restoration only; no live environment or partial rollout.",
    }
    target = folder / f"checkpoint_{name}.pt"
    torch.save(checkpoint, target)
    (folder / f"actor_{name}.bin").write_bytes(packed.data)
    package_bytes = target.stat().st_size + Path(base["foundation"]).stat().st_size
    if package_bytes > int(config["whole_package_cap_bytes"]):
        raise ValueError(
            f"checkpoint plus W0 is {package_bytes} bytes, exceeding "
            f"{config['whole_package_cap_bytes']} bytes"
        )
    return {
        "actor_bytes": packed.nbytes,
        "checkpoint_bytes": target.stat().st_size,
        "whole_package_bytes": package_bytes,
    }


def restore_run(folder: str | Path, stage: str = "final"):
    """Restore through the S1b factory, including automatic-condition aliases."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    payload = torch.load(folder / f"checkpoint_{stage}.pt", map_location="cpu", weights_only=False)
    agent, slots, base = make_agent(config["condition"], int(config["seed"]), config)
    if payload["foundation_sha256"] != sha(Path(base["foundation"])):
        raise ValueError("checkpoint W0 foundation hash does not match the configured foundation")
    saved_slots = {str(name): int(slot) for name, slot in payload["slot_map"].items()}
    if saved_slots != slots:
        raise ValueError("checkpoint slot map does not match the seed's regenerated permutation")
    local, shared, groups = decode_shared_knowledge(payload["actor_binary"])
    agent.core.install_local_matrix(local)
    agent.core.install_shared_matrix(shared)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(groups, dtype=torch.long))
    agent.core.freeze_shared(bool(payload["shared_frozen"]))
    loop._load_critic_state(agent.core, payload["critic"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    agent.optimizer.step_count = int(payload.get("optimizer_step_count", 0))
    agent.counts.update(payload["counts"])
    agent.action_rng.bit_generator.state = payload["action_rng"]
    if payload.get("grouper_state") is not None:
        agent.grouper = DynamicsGrouper.from_state_dict(payload["grouper_state"])
    agent.probe_profiles = {
        name: np.asarray(profile, dtype=np.uint16).copy()
        for name, profile in payload.get("probe_profiles", {}).items()
    }
    agent.group_decisions = list(payload.get("group_decisions", []))
    agent.group_snapshots = list(payload.get("group_snapshots", []))
    if "torch_rng" in payload:
        torch.set_rng_state(payload["torch_rng"])
    return agent, dict(payload["slot_map"]), base


def _training_bout(agent, context_name, slot, duration, bout_index, seed, folder, slots, steps_writer, episode_writer, budget_rows, config, base):
    env = C1NavigationEnv(C1_CONTEXTS[context_name])
    obs, _ = env.reset(seed=seed + 2200 + bout_index * 10000)
    rollout = []
    episode_return, episode_length = 0.0, 0
    total_reward, successes, completed = 0.0, 0, 0
    try:
        for _offset in range(int(duration)):
            policy_obs = loop.augment_observation(obs, slot + 1)
            tensor = torch.as_tensor(policy_obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                probs = agent.core.get_distribution(tensor).distribution.probs.detach().cpu().numpy().reshape(4)
                value = float(agent.core.predict_values(tensor).item())
            action, log_prob = loop._sample_action(probs, agent.action_rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_policy_obs = loop.augment_observation(next_obs, slot + 1)
            with torch.no_grad():
                next_value = float(
                    agent.core.predict_values(torch.as_tensor(next_policy_obs, dtype=torch.float32).reshape(1, -1)).item()
                )
            step = agent.counts["environment_steps"] + 1
            rollout.append(
                loop.Transition(
                    policy_obs,
                    action,
                    float(reward),
                    value,
                    next_value,
                    log_prob,
                    probs,
                    done,
                    bool(terminated),
                    int(slot),
                    1,
                    step,
                )
            )
            agent.counts["environment_steps"] = step
            total_reward += float(reward)
            episode_return += float(reward)
            episode_length += 1
            if len(rollout) == loop.ROLLOUT_STEPS:
                if loop.rollout_number_at_step(step) <= 3:
                    loop._ppo_update(agent, rollout)
                    agent.counts["fit_steps"] += len(rollout)
                else:
                    agent.counts["holdout_steps"] += len(rollout)
                rollout = []
            steps_writer.writerow(
                {
                    "step": step,
                    "bout": bout_index,
                    "context": context_name,
                    "slot": slot,
                    "group": agent.core.group_for_slot(slot),
                    "action": action,
                    "reward": float(reward),
                    "terminated": int(bool(terminated)),
                    "truncated": int(bool(truncated)),
                    "log_prob": log_prob,
                    "policy_updates": agent.counts["minibatch_updates"],
                }
            )
            if done:
                success = bool(info.get("success", False))
                successes += int(success)
                completed += 1
                episode_writer.writerow(
                    {
                        "step": step,
                        "bout": bout_index,
                        "context": context_name,
                        "return": episode_return,
                        "success": int(success),
                        "length": episode_length,
                        "completed": True,
                    }
                )
                episode_return, episode_length = 0.0, 0
                obs, _ = env.reset()
            else:
                obs = next_obs
            if step % 1024 == 0:
                budget_rows.append({"step": step, **save(agent, folder, "latest", config, slots, base)})
        if rollout:
            raise AssertionError("training bout ended with a partial 256-step rollout")
        if episode_length:
            episode_writer.writerow(
                {
                    "step": agent.counts["environment_steps"],
                    "bout": bout_index,
                    "context": context_name,
                    "return": episode_return,
                    "success": 0,
                    "length": episode_length,
                    "completed": False,
                }
            )
    finally:
        env.close()
    return total_reward, successes, completed


def evaluate_stage(agent, slots, stage, config, evaluation_writer):
    result, steps = {}, 0
    for index, context in enumerate(C1_CONTEXTS):
        episodes = (
            config["cold_evaluation_episodes"]
            if context in config["heldout_sequence"]
            else config["retention_evaluation_episodes"]
        )
        value = evaluate_context(
            agent.core,
            context,
            slots[context] + 1,
            seed=int(agent.seed)
            + int(config["evaluation_seed_offset"])
            + index * int(config["evaluation_context_seed_stride"]),
            episodes=int(episodes),
        )
        for row in value["episodes"]:
            evaluation_writer.writerow(
                {
                    "stage": stage,
                    **row,
                    "length": row["steps"],
                    "observer_visits": json.dumps(row["observer_region_visits"], separators=(",", ":")),
                    "actions": json.dumps(row["actions"], separators=(",", ":")),
                }
            )
        steps += value["total_steps"]
        result[context] = {
            "mean_return": value["mean_return"],
            "success_rate": value["success_rate"],
            "environment_steps": value["total_steps"],
        }
    return result, steps


def _pairwise_assignment_accuracy(decisions: list[Mapping[str, Any]]) -> dict[str, int | float | None]:
    by_context = {str(row["context"]): row for row in decisions}
    names = [name for name in C1_CONTEXTS if name in by_context]
    correct = total = 0
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            left, right = by_context[first], by_context[second]
            predicted_same = (
                int(left["assigned_group"]) >= 0
                and int(right["assigned_group"]) >= 0
                and int(left["assigned_group"]) == int(right["assigned_group"])
            )
            actual_same = left["observer_family_audit_only"] == right["observer_family_audit_only"]
            correct += int(predicted_same == actual_same)
            total += 1
    return {
        "correct_pairs": correct,
        "total_pairs": total,
        "pairwise_accuracy": (correct / total) if total else None,
        "assigned_contexts": sum(int(row["assigned_group"]) >= 0 for row in decisions),
        "context_count": len(names),
    }


def run_one(condition: str, seed: int, config: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    folder = Path(output) / f"seed_{seed}" / condition
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Preserve existing results: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    config = {**dict(config), "condition": condition, "seed": int(seed)}
    agent, slots, base = make_agent(condition, seed, config)
    snapshot(folder, config, base)
    write_json(folder / "external_slot_map.json", slots)

    step_fields = ["step", "bout", "context", "slot", "group", "action", "reward", "terminated", "truncated", "log_prob", "policy_updates"]
    episode_fields = ["step", "bout", "context", "return", "success", "length", "completed"]
    evaluation_fields = ["stage", "context", "episode", "reset_seed", "action_seed", "return", "success", "length", "actions", "observer_visits"]
    probe_fields = ["probe_step", "context", "slot", "episode", "reset_seed", "episode_action_seed", "episode_action_index", "obs", "action", "reward", "next_obs", "terminated", "truncated", "profile_counted", "interaction_cost"]
    decision_fields = ["condition", "seed", "context", "slot", "assigned_group", "reason", "scores", "profile_sha256", "supported_cells", "supported_directions"]
    assignment_fields = ["condition", "seed", "context", "slot", "assigned_group", "reason", "scores", "profile_sha256", "supported_cells", "observer_family_audit_only"]

    filenames = ("steps.csv", "episodes.csv", "evaluations.csv", "probe_steps.csv", "group_decisions.csv", "group_assignments.csv")
    fields = (step_fields, episode_fields, evaluation_fields, probe_fields, decision_fields, assignment_fields)
    streams = [ (folder / name).open("w", newline="", encoding="utf-8") for name in filenames ]
    writers = [csv.DictWriter(stream, fieldnames=names, extrasaction="ignore") for stream, names in zip(streams, fields)]
    for writer in writers:
        writer.writeheader()
    steps_writer, episodes_writer, eval_writer, probe_writer, decision_writer, assignment_writer = writers

    evaluations: dict[str, Any] = {}
    budget_rows: list[dict[str, int]] = []
    total_evaluation_steps = 0
    started, cpu_started = time.perf_counter(), time.process_time()
    total_reward, successes, completed_episodes = 0.0, 0, 0
    seen_sequence = list(config["seen_sequence"])
    heldout_sequence = list(config["heldout_sequence"])
    sequence = [(name, int(config["bout_steps"]), "seen") for name in seen_sequence]
    sequence.extend((name, int(config["heldout_steps"]), "heldout") for name in heldout_sequence)
    freeze_after = int(config["freeze_after_seen_bouts"])
    try:
        for bout_index, (context_name, duration, phase) in enumerate(sequence):
            slot = slots[context_name]
            if context_name not in agent.probe_profiles:
                probe_context(
                    agent,
                    context_name,
                    slot,
                    probe_steps=int(config["probe_steps_per_context"]),
                    probe_writer=probe_writer,
                    decision_writer=decision_writer,
                    assignment_writer=assignment_writer,
                    seed_tag=int(config["probe_seed_tag"]),
                )

            reward, bout_successes, bout_completions = _training_bout(
                agent,
                context_name,
                slot,
                duration,
                bout_index,
                seed,
                folder,
                slots,
                steps_writer,
                episodes_writer,
                budget_rows,
                config,
                base,
            )
            total_reward += reward
            successes += bout_successes
            completed_episodes += bout_completions

            stage = None
            if bout_index + 1 == freeze_after:
                # Heldout probes occur after the two source bouts and before boot evaluation.
                for heldout in heldout_sequence:
                    if heldout not in agent.probe_profiles:
                        probe_context(
                            agent,
                            heldout,
                            slots[heldout],
                            probe_steps=int(config["probe_steps_per_context"]),
                            probe_writer=probe_writer,
                            decision_writer=decision_writer,
                            assignment_writer=assignment_writer,
                            seed_tag=int(config["probe_seed_tag"]),
                        )
                stage = "boot"
            if bout_index + 1 == len(seen_sequence):
                stage = "seen"
            if bout_index + 1 == len(sequence):
                stage = "adapted"
            if stage:
                evaluations[stage], cost = evaluate_stage(agent, slots, stage, config, eval_writer)
                total_evaluation_steps += cost
                save(agent, folder, stage, config, slots, base)
                if stage == "boot" and condition == "auto_frozen":
                    agent.core.freeze_shared()
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "condition": condition,
                        "bout": bout_index + 1,
                        "training_steps": agent.counts["environment_steps"],
                        "probe_steps": agent.counts["probe_steps"],
                        "stage": stage,
                    }
                ),
                flush=True,
            )
    finally:
        for stream in streams:
            stream.close()

    sizes = save(agent, folder, "final", config, slots, base)
    write_json(folder / "capacity.json", budget_rows)
    write_json(folder / "probe_profiles.json", agent.probe_profiles)
    write_json(folder / "group_snapshots.json", agent.group_snapshots)
    group_accuracy = _pairwise_assignment_accuracy(agent.group_decisions)
    result = {
        "seed": int(seed),
        "condition": condition,
        "counts": agent.counts,
        "total_reward": total_reward,
        "reward_per_step": total_reward / max(1, agent.counts["environment_steps"]),
        "online_success": successes / max(1, completed_episodes),
        "completed_episodes": completed_episodes,
        "training_steps": int(agent.counts["environment_steps"]),
        "probe_steps": int(agent.counts["probe_steps"]),
        "evaluation_steps": int(total_evaluation_steps),
        "total_interactions": int(
            agent.counts["environment_steps"] + agent.counts["probe_steps"] + total_evaluation_steps
        ),
        "evaluations": evaluations,
        "group_decisions": agent.group_decisions,
        "grouping_memory_bytes": 0 if agent.grouper is None else agent.grouper.nbytes,
        "group_accuracy": group_accuracy,
        "sizes": sizes,
        "wall_s": time.perf_counter() - started,
        "cpu_s": time.process_time() - cpu_started,
        "process_rss": psutil.Process().memory_info().rss,
        "scope": "Known route slots; family-free automatic grouping from one 512-step visible probe per context.",
    }
    write_json(folder / "summary.json", result)
    return result


def _read_training_prefix(path: Path, max_step: int) -> list[tuple[str, ...]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        return [
            tuple(str(row[field]) for field in ("step", "context", "slot", "group", "action", "reward", "terminated", "truncated", "log_prob"))
            for row in rows
            if int(row["step"]) <= max_step
        ]


def _read_probe_prefix(path: Path, contexts: set[str]) -> list[tuple[str, ...]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        fields = ("context", "slot", "episode", "reset_seed", "episode_action_seed", "episode_action_index", "obs", "action", "reward", "next_obs", "terminated", "truncated", "profile_counted", "interaction_cost")
        return [tuple(str(row[field]) for field in fields) for row in rows if row["context"] in contexts]


def audit_preboot_auto_pair(output: Path, seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    """Check active/frozen policies and behavior match through the boot checkpoint."""
    root = output / f"seed_{seed}"
    active_folder, frozen_folder = root / "auto_active", root / "auto_frozen"
    active = torch.load(active_folder / "checkpoint_boot.pt", map_location="cpu", weights_only=False)
    frozen = torch.load(frozen_folder / "checkpoint_boot.pt", map_location="cpu", weights_only=False)
    boot_steps = int(config["freeze_after_seen_bouts"]) * int(config["bout_steps"])
    actor_match = active["actor_binary"] == frozen["actor_binary"]
    critic_match = all(
        torch.equal(active["critic"][name], frozen["critic"][name])
        for name in active["critic"]
    ) and set(active["critic"]) == set(frozen["critic"])
    groups_match = torch.equal(active["slot_groups"], frozen["slot_groups"])
    grouping_match = active["group_decisions"] == frozen["group_decisions"]
    training_match = _read_training_prefix(active_folder / "steps.csv", boot_steps) == _read_training_prefix(
        frozen_folder / "steps.csv", boot_steps
    )
    preboot_probe_contexts = set(config["seen_sequence"][:2]) | set(config["heldout_sequence"])
    probes_match = _read_probe_prefix(active_folder / "probe_steps.csv", preboot_probe_contexts) == _read_probe_prefix(
        frozen_folder / "probe_steps.csv", preboot_probe_contexts
    )
    checks = {
        "actor_payload_match": actor_match,
        "critic_match": critic_match,
        "slot_groups_match": groups_match,
        "group_decisions_match": grouping_match,
        "training_action_reward_prefix_match": training_match,
        "source_and_heldout_probe_prefix_match": probes_match,
        "boot_shared_frozen_flag": bool(frozen["shared_frozen"]),
        "boot_active_shared_frozen_flag": bool(active["shared_frozen"]),
    }
    checks["passed"] = bool(
        actor_match
        and critic_match
        and groups_match
        and grouping_match
        and training_match
        and probes_match
        and not checks["boot_shared_frozen_flag"]
        and not checks["boot_active_shared_frozen_flag"]
    )
    if not checks["passed"]:
        raise AssertionError(f"auto_active/auto_frozen diverged before boot: {checks}")
    write_json(root / "preboot_auto_audit.json", checks)
    return {"seed": int(seed), **checks}


def run(phase: str, output: str | Path, workers: int = 1):
    config, seeds = resolve_phase_config(load_config(), phase)
    if isinstance(workers, bool) or workers not in (1, 2):
        raise ValueError("workers must be one or two")
    if workers > int(config["max_workers"]):
        raise ValueError("workers exceeds the configured maximum")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve completed or partial cohort: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_config.json", config)
    tasks = [(condition, seed, config, output) for seed in seeds for condition in CONDITIONS]
    if workers == 1:
        records = [run_one(*args) for args in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, int(config["max_workers"]))) as executor:
            pending = [executor.submit(run_one, *args) for args in tasks]
            records = [future.result() for future in as_completed(pending)]

    audits = [audit_preboot_auto_pair(output, seed, config) for seed in seeds]
    result = {
        "phase": phase,
        "conditions": list(CONDITIONS),
        "seeds": seeds,
        "records": records,
        "preboot_auto_audits": audits,
        "training_steps": sum(row["training_steps"] for row in records),
        "probe_steps": sum(row["probe_steps"] for row in records),
        "evaluation_steps": sum(row["evaluation_steps"] for row in records),
        "total_interactions": sum(row["total_interactions"] for row in records),
    }
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    arguments = parser.parse_args()
    run(arguments.phase, arguments.output, arguments.workers)

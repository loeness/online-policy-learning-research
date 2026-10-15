"""Controlled active shared-write study; all learning uses fresh on-policy data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import torch

from online_rl.c1_envs import C1NavigationEnv, C1_CONTEXTS
from online_rl.c1_supplement_experiment import _foundation, seed_everything
from online_rl.policy import LeafAdam
from online_rl.shared_write_policy import SharedWritePolicy
from online_rl.shared_write_storage import encode_shared_knowledge, decode_shared_knowledge
from scripts import run_knowledge_loop as loop
from scripts.shared_write_evaluation import evaluate_context


CONFIG_PATH = ROOT / "configs/shared_write.json"
CONDITIONS = ("independent", "independent_fast", "oracle_active", "oracle_frozen")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_agent(condition, seed, config):
    seed_everything(seed)
    base = loop.load_base_config()
    reference = _foundation(base, seed)
    try:
        core = SharedWritePolicy.from_reference(reference.policy,
            shared_groups=2 if condition.startswith("oracle") else 0,
            exploration_epsilon=base["exploration_epsilon"], temperature=base["temperature"])
    finally:
        reference.env.close()
    permutation = np.random.default_rng(seed + 8811).permutation(6)
    slots = {name: int(permutation[i]) for i, name in enumerate(C1_CONTEXTS)}
    if core.shared_groups:
        for name, slot in slots.items():
            core.set_slot_group(slot, 0 if name[0] == "A" else 1, preserve_effective=False)
    parents = list(core.shared_parameters())
    local = [core._nodes[str(i)].A for i in range(1, 7)]
    actor_ids = {id(p) for p in (*parents, *local)}
    critic = [p for p in core.parameters() if p.requires_grad and id(p) not in actor_ids]
    local_lr = config["fast_local_learning_rate"] if condition == "independent_fast" else config["local_learning_rate"]
    groups = [{"params": local, "lr": local_lr, "name": "local"},
              {"params": critic, "lr": config["critic_learning_rate"], "name": "critic"}]
    if parents:
        groups.append({"params": parents, "lr": config["shared_learning_rate"], "name": "shared"})
    optimizer = LeafAdam(groups, eps=1e-5)
    agent = SimpleNamespace(core=core, optimizer=optimizer, arm="int8", seed=seed,
        action_rng=np.random.default_rng(seed + 1100), slot_generation=np.ones(6, dtype=np.int64),
        counts={"stale_actor_rows": 0, "minibatch_updates": 0, "rollouts_trained": 0,
                "fit_steps": 0, "holdout_steps": 0, "environment_steps": 0})
    return agent, slots, base


def actor_payload(agent, config):
    return encode_shared_knowledge(agent.core.read_local_matrix(), agent.core.read_shared_matrix(),
        agent.core.slot_groups.detach().cpu().numpy(), codec=config["codec"],
        budget_bytes=config["actor_budget_bytes"])


def save(agent, folder, name, config, slots, base):
    packed = actor_payload(agent, config)
    checkpoint = {"format": "active_shared_write_v1", "condition": config["condition"],
        "seed": agent.seed, "actor_binary": packed.data, "critic": loop._critic_state(agent.core),
        "optimizer": agent.optimizer.state_dict(), "slot_groups": agent.core.slot_groups.cpu(),
        "shared_frozen": agent.core._shared_frozen, "slot_map": slots, "counts": agent.counts.copy(),
        "action_rng": agent.action_rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "foundation_sha256": sha(Path(base["foundation"])),
        "scope": "Stage endpoint restoration; no in-flight environment or rollout."}
    target = folder / f"checkpoint_{name}.pt"
    torch.save(checkpoint, target)
    (folder / f"actor_{name}.bin").write_bytes(packed.data)
    package = target.stat().st_size + Path(base["foundation"]).stat().st_size
    assert package <= config["whole_package_cap_bytes"], package
    return {"actor_bytes": packed.nbytes, "checkpoint_bytes": target.stat().st_size,
            "whole_package_bytes": package}


def snapshot(folder, config, base):
    files = list(loop.SOURCE_SNAPSHOT_FILES) + [Path(p) for p in (
        "scripts/run_shared_write.py", "scripts/shared_write_evaluation.py",
        "src/online_rl/shared_write_policy.py", "src/online_rl/shared_write_storage.py",
        "configs/shared_write.json", "docs/SHARED_WRITE_PROTOCOL.md")]
    manifest = {}
    for relative in files:
        path = ROOT / relative
        destination = folder / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[relative.as_posix()] = sha(destination)
    write_json(folder / "source_manifest.json", {"sources": manifest,
        "foundation_sha256": sha(Path(base["foundation"])), "dataset_sha256": sha(Path(base["dataset"]))})
    write_json(folder / "config.json", config)


def evaluate_stage(agent, slots, stage, config, evaluation_writer):
    result, steps = {}, 0
    for index, context in enumerate(C1_CONTEXTS):
        episodes = config["cold_evaluation_episodes"] if context in config["heldout_sequence"] else config["retention_evaluation_episodes"]
        value = evaluate_context(agent.core, context, slots[context] + 1,
                                 seed=agent.seed + 50000 + index * 1000, episodes=episodes)
        for row in value["episodes"]:
            evaluation_writer.writerow({"stage": stage, **row, "length": row["steps"],
                                         "observer_visits": row["observer_region_visits"]})
        steps += value["total_steps"]
        result[context] = {"mean_return": value["mean_return"], "success_rate": value["success_rate"],
                           "environment_steps": value["total_steps"]}
    return result, steps


def restore_run(folder, stage="final"):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    payload = torch.load(folder / f"checkpoint_{stage}.pt", map_location="cpu", weights_only=False)
    agent, slots, base = make_agent(config["condition"], config["seed"], config)
    assert payload["foundation_sha256"] == sha(Path(base["foundation"]))
    local, shared, groups = decode_shared_knowledge(payload["actor_binary"])
    agent.core.install_local_matrix(local)
    agent.core.install_shared_matrix(shared)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(groups, dtype=torch.long))
    agent.core.freeze_shared(payload["shared_frozen"])
    loop._load_critic_state(agent.core, payload["critic"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    agent.counts.update(payload["counts"])
    agent.action_rng.bit_generator.state = payload["action_rng"]
    return agent, slots, base


def run_one(condition, seed, config, output):
    torch.set_num_threads(1)
    folder = Path(output) / f"seed_{seed}" / condition
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Preserve existing results: {folder}")
    folder.mkdir(parents=True)
    config = {**config, "condition": condition, "seed": seed}
    agent, slots, base = make_agent(condition, seed, config)
    snapshot(folder, config, base)
    write_json(folder / "external_slot_map.json", slots)
    step_fields = ["step", "bout", "context", "slot", "group", "action", "reward", "terminated", "truncated", "log_prob", "policy_updates"]
    episode_fields = ["step", "bout", "context", "return", "success", "length", "completed"]
    eval_fields = ["stage", "context", "episode", "reset_seed", "action_seed", "return", "success", "length", "actions", "observer_visits"]
    evaluations, budget_rows, total_eval_steps = {}, [], 0
    started, cpu = time.perf_counter(), time.process_time()
    streams = [(folder / p).open("w", newline="", encoding="utf-8") for p in ("steps.csv", "episodes.csv", "evaluations.csv")]
    steps_out, episodes_out, eval_out = [csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        for f, fields in zip(streams, (step_fields, episode_fields, eval_fields))]
    for writer in (steps_out, episodes_out, eval_out):
        writer.writeheader()
    sequence = [(name, config["bout_steps"], "seen") for name in config["seen_sequence"]]
    sequence += [(name, config["heldout_steps"], "heldout") for name in config["heldout_sequence"]]
    total_reward, successes, complete_episodes = 0., 0, 0
    try:
        for bout, (context, duration, phase) in enumerate(sequence):
            env = C1NavigationEnv(C1_CONTEXTS[context])
            obs, _ = env.reset(seed=seed + 2200 + bout * 10000)
            slot = slots[context]
            rollout = []
            episode_return, episode_length = 0., 0
            parent_before = agent.core.read_shared_matrix().copy()
            for offset in range(duration):
                policy_obs = loop.augment_observation(obs, slot + 1)
                tensor = torch.as_tensor(policy_obs).reshape(1, -1)
                with torch.no_grad():
                    probs = agent.core.get_distribution(tensor).distribution.probs.numpy().reshape(4)
                    value = float(agent.core.predict_values(tensor).item())
                action, log_prob = loop._sample_action(probs, agent.action_rng)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                with torch.no_grad():
                    next_value = float(agent.core.predict_values(torch.as_tensor(
                        loop.augment_observation(next_obs, slot + 1)).reshape(1, -1)).item())
                step = agent.counts["environment_steps"] + 1
                rollout.append(loop.Transition(policy_obs, action, reward, value, next_value, log_prob,
                    probs, done, bool(terminated), slot, 1, step))
                if len(rollout) == loop.ROLLOUT_STEPS:
                    if loop.rollout_number_at_step(step) <= 3:
                        loop._ppo_update(agent, rollout)
                        agent.counts["fit_steps"] += len(rollout)
                    else:
                        agent.counts["holdout_steps"] += len(rollout)
                    rollout = []
                agent.counts["environment_steps"] = step
                total_reward += reward
                episode_return += reward
                episode_length += 1
                steps_out.writerow({"step": step, "bout": bout, "context": context, "slot": slot,
                    "group": agent.core.group_for_slot(slot), "action": action, "reward": reward,
                    "terminated": int(terminated), "truncated": int(truncated), "log_prob": log_prob,
                    "policy_updates": agent.counts["minibatch_updates"]})
                if done:
                    successes += int(info.get("success", False))
                    complete_episodes += 1
                    episodes_out.writerow({"step": step, "bout": bout, "context": context,
                        "return": episode_return, "success": int(info.get("success", False)),
                        "length": episode_length, "completed": True})
                    episode_return, episode_length = 0., 0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                if step % 1024 == 0:
                    budget_rows.append({"step": step, **save(agent, folder, "latest", config, slots, base)})
            assert not rollout
            if episode_length:
                episodes_out.writerow({"step": agent.counts["environment_steps"], "bout": bout,
                    "context": context, "return": episode_return, "success": 0,
                    "length": episode_length, "completed": False})
            env.close()
            if condition == "oracle_frozen" and bout + 1 > config["freeze_after_seen_bouts"]:
                assert np.array_equal(parent_before, agent.core.read_shared_matrix()), "frozen shared parameters changed"
            stage = None
            if bout + 1 == config["freeze_after_seen_bouts"]:
                stage = "boot"
            if bout + 1 == len(config["seen_sequence"]):
                stage = "seen"
            if bout + 1 == len(sequence):
                stage = "adapted"
            if stage:
                evaluations[stage], cost = evaluate_stage(agent, slots, stage, config, eval_out)
                total_eval_steps += cost
                save(agent, folder, stage, config, slots, base)
                if stage == "boot" and condition == "oracle_frozen":
                    agent.core.freeze_shared()
            print(json.dumps({"seed": seed, "condition": condition, "bout": bout + 1,
                              "steps": agent.counts["environment_steps"], "stage": stage}), flush=True)
    finally:
        for stream in streams:
            stream.close()
    sizes = save(agent, folder, "final", config, slots, base)
    write_json(folder / "capacity.json", budget_rows)
    result = {"seed": seed, "condition": condition, "counts": agent.counts,
        "total_reward": total_reward, "reward_per_step": total_reward / agent.counts["environment_steps"],
        "online_success": successes / max(1, complete_episodes), "completed_episodes": complete_episodes,
        "training_steps": agent.counts["environment_steps"], "evaluation_steps": total_eval_steps,
        "total_interactions": agent.counts["environment_steps"] + total_eval_steps,
        "evaluations": evaluations, "sizes": sizes, "wall_s": time.perf_counter() - started,
        "cpu_s": time.process_time() - cpu, "process_rss": psutil.Process().memory_info().rss,
        "scope": "Controlled context routes; family membership privileged only in oracle arms. One W0."}
    write_json(folder / "summary.json", result)
    return result


def run(phase, output, workers=1):
    config = json.loads(CONFIG_PATH.read_text())
    if phase == "pilot":
        config.update(seen_sequence=config["pilot_seen_sequence"], bout_steps=config["pilot_bout_steps"],
                      heldout_steps=config["pilot_bout_steps"], cold_evaluation_episodes=8,
                      retention_evaluation_episodes=4)
        seeds = [config["validation_seed"]]
    else:
        seeds = config["formal_seeds"]
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve completed or partial cohort: {output}")
    output.mkdir(parents=True)
    write_json(output / "frozen_config.json", config)
    tasks = [(condition, seed, config, str(output)) for seed in seeds for condition in config["conditions"]]
    records = []
    if workers == 1:
        records = [run_one(*args) for args in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, config["max_workers"])) as executor:
            pending = [executor.submit(run_one, *args) for args in tasks]
            for future in as_completed(pending):
                records.append(future.result())
    result = {"phase": phase, "records": records,
        "training_steps": sum(r["training_steps"] for r in records),
        "evaluation_steps": sum(r["evaluation_steps"] for r in records),
        "total_interactions": sum(r["total_interactions"] for r in records)}
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    run(args.phase, args.output, args.workers)

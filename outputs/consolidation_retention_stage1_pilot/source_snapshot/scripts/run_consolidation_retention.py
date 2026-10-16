"""Paid frozen-policy assay after transferring only a consolidated parent."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from scripts import run_write_decision as wd
from online_rl.consolidation_probe import extract_common, recompose_shared
from online_rl.write_decision_envs import make_env, task_from_name
from online_rl.write_decision_policy import decode_actor, encode_actor

CONFIG = ROOT / "configs/consolidation_retention_stage1.json"
PROTOCOL = ROOT / "docs/CONSOLIDATION_RETENTION_STAGE1_PROTOCOL.md"
SOURCES = (
    Path("scripts/run_consolidation_retention.py"),
    Path("src/online_rl/consolidation_probe.py"),
    Path("configs/consolidation_retention_stage1.json"),
    Path("docs/CONSOLIDATION_RETENTION_STAGE1_PROTOCOL.md"),
    *wd.SOURCE_FILES,
)


def write(path, obj):
    wd._write_json(Path(path), obj)


def sha(path):
    return wd._sha(Path(path))


def state_hash(core):
    digest = hashlib.sha256()
    for name, tensor in sorted(core.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze(core):
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)


def augmented_grid():
    coords = np.asarray([(x / 8, y / 8, gx / 8, gy / 8)
                         for x in range(9) for y in range(9)
                         for gx in range(9) for gy in range(9)
                         if (x, y) != (gx, gy)], dtype=np.float32)
    result = np.zeros((len(coords), 13), dtype=np.float32)
    result[:, :4] = coords
    return result


def logits_and_probs(core, observations):
    logits, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(observations), 1024):
            obs = torch.as_tensor(observations[start:start + 1024], dtype=torch.float32)
            latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
            logits.append((core.action_net(latent) + core._correction(obs, latent)).cpu().numpy())
            probabilities.append(core.get_distribution(obs).distribution.probs.cpu().numpy())
    return np.concatenate(logits), np.concatenate(probabilities)


def policy_grid(core):
    grid = augmented_grid()
    results = []
    for slot in range(core.local_slots):
        grid[:, -1] = slot + 1
        results.append(logits_and_probs(core, grid))
    return results


def compare_grid(before, after):
    max_logit = max(float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(before, after))
    max_probability = max(float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(before, after))
    kls = []
    for a, b in zip(before, after):
        p = np.clip(a[1].astype(np.float64), 1e-12, 1)
        q = np.clip(b[1].astype(np.float64), 1e-12, 1)
        kls.append(np.sum(p * (np.log(p) - np.log(q)), axis=1))
    return {"logit_max_error": max_logit, "probability_max_error": max_probability,
            "mean_KL": float(np.mean(np.concatenate(kls)))}


def source_inputs(agent, payload, config):
    local, shared, groups = wd._actor_arrays(agent.core)
    source_slots = [int(agent.slot_map[t]) for t in config["source_tasks"]]
    group = int(groups[source_slots[0]])
    if group < 0 or any(int(groups[s]) != group for s in source_slots):
        raise ValueError("sources do not have the same assigned parent")
    if np.any(shared):
        raise ValueError("source shared matrices must be zero")
    expected_steps = int(config["required_source_training_steps_each"])
    if any(int(payload["steps_by_task"][t]) != expected_steps for t in config["source_tasks"]):
        raise ValueError("unexpected source training budget")
    unseen_slot = int(agent.slot_map["A2+"])
    if int(payload["steps_by_task"]["A2+"]) != 0 or int(agent.profiles["seen"][unseen_slot]):
        raise ValueError("related target was already used during source training")
    effective, psis, bands, anchor_audit = [], [], [], []
    for task, slot in zip(config["source_tasks"], source_slots):
        generation = int(agent.slot_generation[slot])
        rows = agent.anchors.arrays(slot, generation)
        obs = rows["obs"]
        if not 0 < len(obs) <= int(agent.run_config["anchors_per_slot"]):
            raise ValueError("invalid bounded source anchor count")
        if not np.all(obs[:, -1] == slot + 1):
            raise ValueError("anchor route differs from source slot")
        tensor = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            latent = agent.core.mlp_extractor.forward_actor(agent.core.extract_features(tensor))
            phi = torch.cat((latent, torch.ones_like(latent[:, :1])), dim=1)
            psi = torch.nn.functional.linear(phi, agent.core.source_basis).cpu().numpy()
        psis.append(psi)
        bands.append(np.floor(np.clip(obs[:, 0], 0, 1) * 3).astype(np.int64).clip(0, 2))
        effective.append(local[slot].reshape(3, 3, 3) + shared[group].reshape(1, 3, 3))
        anchor_audit.append({"task": task, "slot": slot, "generation": generation,
                             "rows": len(obs), "episodes": len(np.unique(rows["episode_id"]))})
    return local, shared, groups, source_slots, group, np.stack(effective), psis, bands, anchor_audit


def make_deleted_capsule(source_agent, common, group, method, config, path):
    """Fresh offline W0 state plus only C; no source learner object is copied."""
    fields = ("local_learning_rate", "fast_local_learning_rate", "shared_learning_rate",
              "critic_learning_rate", "anchors_per_slot", "reward_support_aware")
    run_config = {key: source_agent.run_config[key] for key in fields}
    run_config.update({"phase": "consolidation_retention_probe", "condition": "trial_rollback",
                       "method": method, "actor_codec": config["actor_codec"],
                       "actor_budget_bytes": config["actor_budget_bytes"],
                       "package_budget_bytes": config["package_budget_bytes"],
                       "fixed_evaluation_slot": 0, "fixed_evaluation_group": group,
                       "target_training_enabled": False})
    fresh = wd.make_agent("trial_rollback", source_agent.seed, run_config,
                          base_config=source_agent.base_config,
                          slot_map={name: i for i, name in enumerate(wd.TASK_NAMES)})
    baseline_critic_hash = hashlib.sha256(b"".join(
        value.detach().cpu().numpy().tobytes()
        for _, value in sorted(wd.loop._critic_state(fresh.core).items())
    )).hexdigest()
    local = np.zeros((6, 27), dtype=np.float32)
    shared = np.zeros((2, 9), dtype=np.float32)
    shared[group] = common.reshape(9)
    groups = np.full(6, -1, dtype=np.int8)
    groups[0] = group
    wd._install_actor_arrays(fresh.core, local, shared, groups)
    sizes = wd.save_checkpoint(fresh, path, sha(fresh.base_config["foundation"]))
    del fresh
    restored = wd.restore_agent_checkpoint(path, base_config=source_agent.base_config)
    freeze(restored.core)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    a, c, g = decode_actor(saved["actor_binary"])
    other = np.delete(c, group, axis=0)
    actual_critic_hash = hashlib.sha256(b"".join(
        value.detach().cpu().numpy().tobytes()
        for _, value in sorted(saved["critic"].items())
    )).hexdigest()
    anchor = saved["anchors"]
    checks = {
        "every_local_zero": bool(np.count_nonzero(a) == 0),
        "other_parents_zero": bool(np.count_nonzero(other) == 0),
        "optimizer_empty": len(saved["optimizer"]["state"]) == 0,
        "all_anchor_buffers_empty": bool(not np.any(anchor["sizes"]) and
                                         not np.any(anchor["obs"]) and not np.any(anchor["seen"])),
        "movement_profiles_zero": bool(not np.any(saved["profiles"]["movement_counts"])),
        "reward_profiles_zero": bool(not np.any(saved["profiles"]["reward_xtx"]) and
                                     not np.any(saved["profiles"]["reward_xty"]) and
                                     not np.any(saved["profiles"]["reward_n"])),
        "all_profiles_unseen": bool(not np.any(saved["profiles"]["seen"])),
        "group_statistics_empty": bool(not np.any(saved["grouper"]["active"])),
        "critic_is_fresh_W0": actual_critic_hash == baseline_critic_hash,
        "online_counts_zero": saved["counts"]["environment_steps"] == 0 and
                              all(value == 0 for value in saved["steps_by_task"].values()),
        "no_online_rollout_or_history_payload": not any(key in saved for key in ("rollout", "history", "windows")),
        "fixed_fresh_route": int(g[0]) == group and np.all(g[1:] == -1),
        "all_actor_gradients_disabled": all(not p.requires_grad for p in restored.core.parameters()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"deletion audit failed: {checks}")
    return restored, sizes, {"passed": True, "checks": checks,
                             "boundary": "fresh W0 learner plus only the retained common C"}


def evaluate_frozen(agent, config, path):
    frozen_hash = state_hash(agent.core)
    targets, steps = {}, 0
    fields = ("seed", "method", "task", "episode", "step", "action", "reward", "uniform",
              "probabilities", "log_prob", "obs", "next_obs", "terminated", "truncated")
    with (path / "steps.csv").open("w", newline="", encoding="utf-8") as stream, \
         (path / "episodes.csv").open("w", newline="", encoding="utf-8") as ep_stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        episode_writer = csv.DictWriter(ep_stream, fieldnames=("seed", "method", "task", "episode", "reset_seed",
            "action_seed", "return", "length", "goal", "positive_success"))
        writer.writeheader()
        episode_writer.writeheader()
        for index, name in enumerate(config["targets"]):
            returns, lengths, goals, positives = [], [], [], []
            for episode in range(int(config["episodes"])):
                reset_seed = agent.seed + int(config["evaluation_reset_offset"]) + index * 10000 + episode * 31
                action_seed = agent.seed + int(config["evaluation_action_offset"]) + index * 10000 + episode * 17
                rng = np.random.default_rng(action_seed)
                env = make_env(task_from_name(name), config["domain"])
                obs, _ = env.reset(seed=reset_seed)
                total, length, goal, done = 0.0, 0, False, False
                while not done:
                    augmented = wd.loop.augment_observation(obs, 1)
                    probabilities = wd._policy_probs(agent.core, augmented[None])[0].astype(np.float64)
                    probabilities /= probabilities.sum()
                    uniform = float(rng.random())
                    action = wd._action_from_uniform(probabilities, uniform)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    length += 1
                    steps += 1
                    total += float(reward)
                    goal = goal or bool(info["goal_reached"])
                    writer.writerow({"seed": agent.seed, "method": agent.run_config["method"], "task": name,
                        "episode": episode, "step": length, "action": action, "reward": reward,
                        "uniform": uniform, "probabilities": json.dumps(probabilities.tolist()),
                        "log_prob": float(np.log(probabilities[action])), "obs": json.dumps(obs.tolist()),
                        "next_obs": json.dumps(next_obs.tolist()), "terminated": int(terminated), "truncated": int(truncated)})
                    obs = next_obs
                    done = bool(terminated or truncated)
                    if length > int(config["max_episode_steps"]):
                        raise RuntimeError("environment exceeded the frozen horizon")
                env.close()
                positive = bool(goal and task_from_name(name).preference > 0)
                episode_writer.writerow({"seed": agent.seed, "method": agent.run_config["method"], "task": name,
                    "episode": episode, "reset_seed": reset_seed, "action_seed": action_seed,
                    "return": total, "length": length, "goal": int(goal), "positive_success": int(positive)})
                returns.append(total)
                lengths.append(length)
                goals.append(int(goal))
                positives.append(int(positive))
            targets[name] = {"episode_returns": returns, "episode_lengths": lengths, "goal_flags": goals,
                "positive_success_flags": positives, "mean_signed_return": float(np.mean(returns)),
                "mean_goal_rate": float(np.mean(goals)), "positive_success_rate": float(np.mean(positives)),
                "evaluation_steps": sum(lengths)}
    unchanged = frozen_hash == state_hash(agent.core)
    if not unchanged:
        raise RuntimeError("target evaluation modified policy state")
    return targets, steps, unchanged


def run(config, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["max_torch_threads"]))
    started, cpu_started = time.perf_counter(), time.process_time()
    write(output / "frozen_config.json", config)
    source_snapshot = output / "source_snapshot"
    manifest = {}
    for relative in dict.fromkeys(SOURCES):
        # Whitelist only reproducibility inputs; never local instructions.
        if relative.as_posix().casefold() in {"agent.md", "agents.md"}:
            raise RuntimeError("private instruction path cannot enter a source snapshot")
        source = ROOT / relative
        target = source_snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = {"sha256": sha(target), "bytes": target.stat().st_size}
    write(source_snapshot / "manifest.json", manifest)
    records, audits, total_steps = [], [], 0
    try:
        for seed in config["seeds"]:
            source = ROOT / config["source_root"] / f"seed_{seed}" / config["source_condition"] / "checkpoints" / \
                     f"checkpoint_step_{int(config['source_checkpoint_step']):08d}.pt"
            source_sha = sha(source)
            payload = torch.load(source, map_location="cpu", weights_only=False)
            agent = wd.restore_agent_checkpoint(source)
            freeze(agent.core)
            local, shared, groups, slots, group, effective, psis, bands, anchor_audit = source_inputs(agent, payload, config)
            baseline_grid = policy_grid(agent.core)
            audit = {"seed": seed, "source": str(source.relative_to(ROOT)), "source_sha256": source_sha,
                "foundation_sha256": payload["foundation_sha256"], "steps_by_task": payload["steps_by_task"],
                "source_slots": slots, "source_group": group, "shared_max_abs": float(np.max(np.abs(shared))),
                "source_anchor_evidence": anchor_audit, "unseen_target_training_steps": 0}
            audits.append(audit)
            write(output / f"seed_{seed}" / "source_audit.json", audit)
            for method in config["methods"]:
                branch_start, branch_cpu = time.perf_counter(), time.process_time()
                folder = output / f"seed_{seed}" / method
                folder.mkdir(parents=True)
                common, proposal = extract_common(method, effective, anchor_psi=psis, anchor_bands=bands,
                    seed=seed + int(config["random_seed_offset"]), ridge_relative=float(config["ridge_relative"]))
                a, c, g, recomposition = recompose_shared(local, shared, groups,
                    source_slots=slots, group=group, common=common)
                wd._install_actor_arrays(agent.core, a, c, g)
                raw = compare_grid(baseline_grid, policy_grid(agent.core))
                if raw["logit_max_error"] > float(config["pre_codec_logit_tolerance"]):
                    raise RuntimeError(f"pre-codec source policy changed: {raw}")
                encoded = encode_actor(a, c, g, codec="int8", budget_bytes=int(config["actor_budget_bytes"]))
                qa, qc, qg = decode_actor(encoded.data)
                wd._install_actor_arrays(agent.core, qa, qc, qg)
                quantized = compare_grid(baseline_grid, policy_grid(agent.core))
                wd._install_actor_arrays(agent.core, local, shared, groups)
                clean, sizes, purge = make_deleted_capsule(agent, common, group, method, config, folder / "post_delete.pt")
                saved_sha = sha(folder / "post_delete.pt")
                targets, steps, frozen = evaluate_frozen(clean, config, folder)
                if saved_sha != sha(folder / "post_delete.pt"):
                    raise RuntimeError("target evaluation overwrote its deletion checkpoint")
                if source_sha != sha(source):
                    raise RuntimeError("input source checkpoint was modified")
                proposal.update({"common_matrix": common.tolist(), "recomposition": recomposition})
                result = {"seed": seed, "method": method, "path": str(folder.relative_to(output)),
                    "source_slots": slots, "target_group": group, "proposal": proposal,
                    "invariance": {"raw_logit_max_error": raw["logit_max_error"],
                        "raw_probability_max_error": raw["probability_max_error"],
                        "int8_logit_max_error": quantized["logit_max_error"], "int8_mean_KL": quantized["mean_KL"]},
                    "purge_audit": purge, "capacity": sizes, "targets": targets,
                    "new_training_steps": 0, "evaluation_steps": steps, "actor_frozen_during_eval": frozen,
                    "source_unchanged": True, "post_delete_checkpoint_sha256": saved_sha,
                    "cpu_s": time.process_time() - branch_cpu, "wall_s": time.perf_counter() - branch_start}
                write(folder / "summary.json", result)
                records.append(result)
                total_steps += steps
                print(f"completed seed={seed} method={method} eval_steps={steps}", flush=True)
        for relative, row in manifest.items():
            if sha(ROOT / relative) != row["sha256"]:
                raise RuntimeError(f"source changed during run: {relative}")
        summary = {"format": "consolidation_retention_stage1_v1", "status": "completed",
            "analysis_role": "exploratory", "seeds": config["seeds"], "methods": config["methods"],
            "targets": config["targets"], "records": records, "source_audits": audits,
            "cohort_cost": {"source_training_steps_reused": len(audits) * int(config["source_checkpoint_step"]),
                "new_training_steps": 0, "new_evaluation_steps": total_steps, "new_total_interactions": total_steps},
            "audit": {"passed": True, "failures": []}, "protocol_sha256": sha(PROTOCOL),
            "cpu_s": time.process_time() - cpu_started, "wall_s": time.perf_counter() - started,
            "limitations": config["limitations"]}
        write(output / "summary.json", summary)
        return summary
    except Exception as error:
        write(output / "FAILED_COHORT.json", {"status": "failed", "reason": repr(error),
            "completed_branches": len(records), "completed_evaluation_steps": total_steps,
            "partial_branch_steps_may_not_be_in_total": True, "count_is_lower_bound": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.seeds is not None:
        if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(config["seeds"]):
            raise ValueError("source seeds must be a unique nonempty subset of frozen seeds")
        config["seeds"] = args.seeds
    if args.episodes is not None:
        if args.episodes < 1:
            raise ValueError("episodes must be positive")
        config["episodes"] = args.episodes
        config["pilot_override"] = True
    summary = run(config, args.output)
    print(json.dumps(summary["cohort_cost"]), flush=True)


if __name__ == "__main__":
    main()

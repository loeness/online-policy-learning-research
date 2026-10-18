"""Known-index online streams with bounded, eviction-aware consolidation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.retention_consolidation import common_mean, compensated_install, surrogate_gain, decide_retention_write
from online_rl.retention_evidence import RetentionEvidenceArchive
from online_rl.write_decision_envs import make_env, task_from_name
from online_rl.write_decision_evidence import RewardAwareGroups
from online_rl.write_fast_evidence import IncrementalEvidenceWindow
from online_rl.write_decision_policy import encode_actor, decode_actor
from scripts import run_knowledge_loop as loop
from scripts import run_write_decision as wd
from scripts import run_write_stream as ws

CONFIG = ROOT / "configs/retention_stream_stage2.json"
PROTOCOL = ROOT / "docs/RETENTION_STREAM_STAGE2_PROTOCOL.md"
SOURCES = tuple(dict.fromkeys((
    Path("scripts/run_retention_stream.py"),
    Path("src/online_rl/retention_evidence.py"),
    Path("src/online_rl/retention_consolidation.py"),
    Path("src/online_rl/consolidation_probe.py"),
    Path("configs/retention_stream_stage2.json"),
    Path("docs/RETENTION_STREAM_STAGE2_PROTOCOL.md"), *ws.SOURCE_FILES)))


def write(path, value):
    ws._write_json(Path(path), value)


def sha(path):
    return ws._sha(Path(path))


def policy_hash(core):
    digest = hashlib.sha256()
    for key, tensor in sorted(core.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def task_schedule(seed):
    rng = np.random.default_rng(int(seed) + 33001)
    families = list(rng.permutation(["A", "B"]))
    variants = {name: list(rng.permutation(3)) for name in families}
    initial = [f"{f}{v}+" for f in families for v in variants[f][:2]]
    initial += [f"{f}{variants[f][2]}+" for f in families]
    return initial + list(rng.permutation(initial))


def agent_config(config, method, budget):
    layout = config["budgets"][budget]
    return {
        "phase": "known", "domain": config["domain"],
        "local_slots": layout["independent_slots"] if method == "independent" else layout["shared_slots"],
        "local_learning_rate": config["local_learning_rate"],
        "fast_local_learning_rate": config["fast_local_learning_rate"],
        "shared_learning_rate": config["shared_learning_rate"],
        "critic_learning_rate": config["critic_learning_rate"],
        "anchors_per_slot": config["archive_rows_per_prototype"],
        "probe_steps": config["profile_window"],
        "reward_support_aware": True, "group_reward_min_samples": 16,
        "actor_codec": "int8", "actor_budget_bytes": layout["actor_budget_bytes"],
        "package_budget_bytes": config["package_budget_bytes"],
    }


def make_agent(config, method, budget, seed):
    base = loop.load_base_config()
    condition = "independent_fast" if method == "independent" else "reward_shared"
    agent = ws._new_agent(condition, seed, agent_config(config, method, budget), base)
    agent.external_condition = method
    agent.core.freeze_shared(method != "direct_shared")
    agent.cache = ws.BoundedTaskCache(agent.slots, "lru", seed + 8811)
    agent.behavior_rng = np.random.default_rng(seed + 1100)
    agent.archive = RetentionEvidenceArchive(config["archive_prototypes"],
        config["archive_rows_per_prototype"], seed + 421)
    agent.slot_prototype = np.full(agent.slots, -1, dtype=np.int16)
    agent.slot_prototype_version = np.zeros(agent.slots, dtype=np.int64)
    agent.prototype_group = np.full(config["archive_prototypes"], -1, dtype=np.int8)
    agent.prototype_version = np.zeros(config["archive_prototypes"], dtype=np.int64)
    agent.group_versions = np.zeros(2, dtype=np.int64)
    agent.prior_injected = np.zeros(agent.slots, dtype=np.uint8)
    agent.pending = None
    agent.counts.update({"write_proposals": 0, "write_accepts": 0, "prior_bindings": 0,
                         "prior_rejects": 0, "pending_expired": 0})
    return agent


def coefficient_probs(core, observations, coefficients):
    """Actual temperature/epsilon distribution for a supplied effective matrix."""
    obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    if obs.ndim != 2 or obs.shape[1] != 13:
        raise ValueError("counterfactual observations must be N by 13")
    with torch.no_grad():
        latent = core.mlp_extractor.forward_actor(core.extract_features(obs))
        phi = torch.cat((latent, torch.ones_like(latent[:, :1])), dim=1)
        psi = torch.nn.functional.linear(phi, core.source_basis)
        c = torch.as_tensor(np.asarray(coefficients), dtype=psi.dtype)
        if c.shape == (3, 3):
            contrast = torch.nn.functional.linear(psi, c)
        elif c.shape == (3, 3, 3):
            bands = torch.floor(obs[:, 0].clamp(0, 1) * 3).long().clamp(0, 2)
            contrast = torch.bmm(c[bands], psi.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError("effective coefficients must be 3x3 or 3x3x3")
        logits = core.action_net(latent) + contrast @ core._contrast_basis.T
        probs = torch.softmax(logits / core.temperature, dim=-1)
        probs = (1.0 - core.epsilon) * probs + core.epsilon / 4.0
    return probs.cpu().numpy()


def effective(core, slot):
    value = core.read_local_matrix()[slot].reshape(3, 3, 3)
    group = core.group_for_slot(slot)
    if group >= 0:
        value = value + core.read_shared_matrix()[group].reshape(1, 3, 3)
    return value


def row_arrays(rows):
    return {
        "obs": np.stack([row.obs for row in rows]),
        "action": np.asarray([row.action for row in rows], dtype=np.int64),
        "behavior_probs": np.stack([row.probabilities for row in rows]),
        "behavior_log_prob": np.asarray([row.log_prob for row in rows]),
        "advantage": np.asarray([row.advantage for row in rows]),
        "return": np.asarray([row.returns for row in rows]),
        "episode_id": np.asarray([row.episode_id for row in rows], dtype=np.int64),
    }


def score_common(agent, config, evidence, candidate, baseline):
    fields = ("min_rows", "min_blocks", "min_ess_ratio", "max_mean_kl", "clip_ratio", "stderr_multiplier")
    return surrogate_gain(evidence, coefficient_probs(agent.core, evidence["obs"], candidate),
        coefficient_probs(agent.core, evidence["obs"], baseline),
        **{name: config["gate"][name] for name in fields})


def roundtrip_actor(agent):
    local, shared, groups = ws._actor_arrays(agent.core)
    blob = encode_actor(local, shared, groups, codec="int8",
                        budget_bytes=agent.run_config["actor_budget_bytes"])
    a, c, g = decode_actor(blob.data)
    agent.core.install_local_matrix(a)
    agent.core.install_shared_matrix(c)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(g.astype(np.int64)))
    error = max(float(np.max(np.abs(a - local))),
                float(np.max(np.abs(c - shared))) if len(c) else 0.0)
    return {"actor_bytes": blob.nbytes, "codec_max_coefficient_error": error}


def snapshot_sources(folder):
    manifest = {}
    for relative in SOURCES:
        if any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts):
            raise ValueError("private material cannot enter a source snapshot")
        source, target = ROOT / relative, folder / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = {"sha256": sha(target), "bytes": target.stat().st_size}
    write(folder / "source_snapshot/manifest.json", manifest)
    return manifest


def checkpoint_payload(agent, config, method, budget):
    a, c, g = ws._actor_arrays(agent.core)
    actor = encode_actor(a, c, g, codec="int8", budget_bytes=agent.run_config["actor_budget_bytes"])
    return {
        "format": "retention_stream_agent_v1", "method": method, "budget": budget,
        "seed": agent.seed, "config": config, "run_config": agent.run_config,
        "foundation_sha256": sha(agent.base_config["foundation"]), "actor_binary": actor.data,
        "critic": loop._critic_state(agent.core), "optimizer": agent.optimizer.state_dict(),
        "optimizer_step_count": int(getattr(agent.optimizer, "step_count", 0)), "shared_frozen": bool(agent.core._shared_frozen),
        "counts": dict(agent.counts), "slot_generation": agent.slot_generation.copy(),
        "fitted_steps": agent.fitted_steps.copy(), "cache": agent.cache.state_dict(),
        "grouper": agent.grouper.state_dict(), "archive": agent.archive.state_dict(),
        "anchors": agent.anchors.state_dict(), "profiles": agent.profiles,
        "windows": [window.state_dict() for window in agent.windows],
        "slot_prototype": agent.slot_prototype.copy(),
        "slot_prototype_version": agent.slot_prototype_version.copy(),
        "prototype_group": agent.prototype_group.copy(), "prototype_version": agent.prototype_version.copy(),
        "group_versions": agent.group_versions.copy(), "prior_injected": agent.prior_injected.copy(),
        "pending": copy.deepcopy(agent.pending), "action_rng": agent.action_rng.bit_generator.state,
        "behavior_rng": agent.behavior_rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "recovery_scope": "Bounded learner state; live simulator and observer trajectories are not restored.",
    }


def save_checkpoint(agent, config, method, budget, path):
    payload = checkpoint_payload(agent, config, method, budget)
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    actor = payload["actor_binary"]
    foundation_bytes = Path(agent.base_config["foundation"]).stat().st_size
    physical = len(data) + len(actor) + foundation_bytes
    if physical > config["package_budget_bytes"]:
        raise ValueError(f"complete checkpoint/actor/W0 {physical} exceeds package cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_suffix(".actor.bin").write_bytes(actor)
    return {"actor_bytes": len(actor), "checkpoint_bytes": len(data), "foundation_bytes": foundation_bytes,
        "actor_sidecar_bytes": len(actor), "full_physical_bytes": physical,
        "package_budget_bytes": config["package_budget_bytes"], "evidence_bytes": agent.archive.nbytes,
        "window_bytes": sum(window.nbytes for window in agent.windows),
        "optimizer_tensor_bytes": wd._optimizer_tensor_bytes(agent.optimizer.state_dict())}


def restore_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["format"] != "retention_stream_agent_v1":
        raise ValueError("unsupported retention checkpoint")
    agent = make_agent(value["config"], value["method"], value["budget"], value["seed"])
    if sha(agent.base_config["foundation"]) != value["foundation_sha256"]:
        raise ValueError("foundation hash mismatch")
    a, c, g = decode_actor(value["actor_binary"])
    agent.core.install_local_matrix(a)
    agent.core.install_shared_matrix(c)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(g.astype(np.int64)))
    loop._load_critic_state(agent.core, value["critic"])
    agent.optimizer.load_state_dict(value["optimizer"])
    agent.optimizer.step_count = value["optimizer_step_count"]
    agent.core.freeze_shared(value["shared_frozen"])
    agent.counts.update(value["counts"])
    agent.slot_generation[...] = value["slot_generation"]
    agent.fitted_steps[...] = value["fitted_steps"]
    agent.cache = ws.BoundedTaskCache.from_state_dict(value["cache"])
    agent.grouper = RewardAwareGroups.from_state_dict(value["grouper"])
    agent.archive = RetentionEvidenceArchive.from_state_dict(value["archive"])
    agent.anchors = wd.AnchorBank.from_state_dict(value["anchors"])
    agent.profiles = value["profiles"]
    agent.windows = [IncrementalEvidenceWindow.from_state_dict(row) for row in value["windows"]]
    for name in ("slot_prototype", "slot_prototype_version", "prototype_group", "prototype_version",
                 "group_versions", "prior_injected"):
        getattr(agent, name)[...] = value[name]
    agent.pending = value["pending"]
    agent.action_rng.bit_generator.state = value["action_rng"]
    agent.behavior_rng.bit_generator.state = value["behavior_rng"]
    torch.set_rng_state(value["torch_rng"])
    return agent


def evaluate_pending(agent, config, rows, group):
    pending = agent.pending
    if pending is None:
        return None
    raw = agent.counts["raw_steps"]
    if raw - pending["created_step"] > config["proposal_ttl_steps"] or \
            agent.group_versions[pending["group"]] != pending["parent_version"]:
        agent.pending = None
        agent.counts["pending_expired"] += 1
        return {"event": "write_cancel", "step": raw, "reason": "expired_or_parent_changed", "accepted": False}
    if group != pending["group"]:
        return None
    if raw <= pending["created_step"] or rows[0].step <= pending["created_step"]:
        raise RuntimeError("candidate cannot be validated on its proposal batch")
    candidate = np.asarray(pending["common"], dtype=np.float32)
    previous = agent.core.read_shared_matrix()[group].reshape(3, 3)
    scores = []
    for prototype in agent.archive.active_indices():
        if int(agent.prototype_group[prototype]) != group:
            continue
        score = score_common(agent, config, agent.archive.arrays(prototype), candidate, previous)
        score["prototype"] = prototype
        scores.append(score)
    fresh = score_common(agent, config, row_arrays(rows), candidate, previous)
    if agent.external_condition == "retention_gate":
        decision = decide_retention_write(scores,
            **{name: config["gate"][name] for name in ("min_supported_profiles", "min_mean_gain", "max_profile_loss")})
        accepted = bool(decision["accepted"] and all(score["supported"] for score in scores)
                        and fresh["supported"] and fresh["lower"] >= -config["gate"]["max_profile_loss"])
        reason = decision["reason"] if not decision["accepted"] else (
            "unknown_historical_or_fresh_evidence" if not accepted else "retention_supported")
    else:
        accepted, reason = True, "mean_unconditional_write"
    agent.pending = None
    return {"event": "write_test", "step": raw, "accepted": accepted, "reason": reason,
        "proposal_step": pending["created_step"], "first_test_row_step": rows[0].step,
        "group": group, "common": candidate.tolist(), "sources": pending["sources"],
        "historical_scores": scores, "fresh_score": fresh}


def process_rollout(agent, config, rows, slot, events, updates):
    """Test old candidate, learn current rows, then commit and propose."""
    loop._finalize_gae(rows)
    critic_diagnostics = {
        "critic_pre_update_return_mse": float(np.mean([(row.value - row.returns) ** 2 for row in rows])),
        "advantage_mean": float(np.mean([row.advantage for row in rows])),
        "advantage_std": float(np.std([row.advantage for row in rows])),
    }
    profile = agent.windows[slot].profile()
    assignment = agent.grouper.assign(profile, commit=True)
    assigned_group = assignment.group if agent.core.shared_groups else -1
    if assigned_group >= 0 and assigned_group != agent.core.group_for_slot(slot):
        agent.core.set_slot_group(slot, assigned_group, preserve_effective=True)
    # In this known-index control, an uninformative new window does not erase
    # an earlier assignment for the same still-resident context generation.
    group = agent.core.group_for_slot(slot) if agent.core.shared_groups else -1
    before_probs = wd._policy_probs(agent.core, np.stack([row.obs for row in rows]))
    behavior = np.stack([row.probabilities for row in rows])
    if float(np.max(np.abs(before_probs - behavior))) > 1e-5:
        raise RuntimeError("binding changed behavior before on-policy PPO")
    write_decision = evaluate_pending(agent, config, rows, group)
    prior = None
    if group >= 0 and not agent.prior_injected[slot] and \
            agent.fitted_steps[slot] < config["prior_max_fit_steps"]:
        common = agent.core.read_shared_matrix()[group].reshape(3, 3)
        if np.linalg.norm(common) > 1e-8:
            old = effective(agent.core, slot)
            score = score_common(agent, config, row_arrays(rows), old + common, old)
            gated = agent.external_condition in ("mean_reuse_gate", "retention_gate")
            accept = not gated or (score["supported"] and score["lower"] >= config["prior_gate_min_lower"])
            prior = {"common": common.copy(), "version": int(agent.group_versions[group]),
                     "accepted": bool(accept), "score": score}

    update = loop._ppo_update(agent, rows)
    agent.counts["fit_steps"] += len(rows)
    if update["updates"]:
        agent.fitted_steps[slot] += np.uint64(len(rows))
    archive_result = agent.archive.observe(profile, rows, step=agent.counts["raw_steps"])
    prototype, version = archive_result["prototype"], archive_result["version"]
    if int(agent.prototype_version[prototype]) != version:
        agent.prototype_group[prototype] = -1
        agent.prototype_version[prototype] = version
    # An abstaining fresh window must not inherit an old prototype's group tag.
    # The known resident may keep its acting group, but its evidence is withheld.
    agent.prototype_group[prototype] = assigned_group
    agent.slot_prototype[slot] = prototype
    agent.slot_prototype_version[slot] = version

    if write_decision is not None:
        if write_decision["accepted"]:
            local, shared, groups = ws._actor_arrays(agent.core)
            a, c, g, invariance = compensated_install(local, shared, groups, group=group,
                common=np.asarray(write_decision["common"], dtype=np.float32))
            if invariance["max_effective_preservation_error"] > 1e-5:
                raise RuntimeError("resident effective policy changed during consolidation")
            agent.core.install_local_matrix(a)
            agent.core.install_shared_matrix(c)
            with torch.no_grad():
                agent.core.slot_groups.copy_(torch.as_tensor(g.astype(np.int64)))
            agent.group_versions[group] += 1
            agent.counts["write_accepts"] += 1
            write_decision["invariance"] = invariance
        events.write(json.dumps(write_decision, default=ws._json_default) + "\n")

    if prior is not None:
        unchanged = group >= 0 and int(agent.group_versions[group]) == prior["version"]
        applied = bool(prior["accepted"] and unchanged)
        if applied:
            with torch.no_grad():
                agent.core._nodes[str(slot + 1)].A.add_(torch.as_tensor(prior["common"]).unsqueeze(0))
            agent.prior_injected[slot] = 1
            agent.counts["prior_bindings"] += 1
        else:
            agent.counts["prior_rejects"] += 1
        events.write(json.dumps({"event": "prior_test", "step": agent.counts["raw_steps"],
            "slot": slot, "generation": int(agent.slot_generation[slot]), "group": group,
            "applied": applied, "parent_version_unchanged": unchanged,
            "pre_local_update_score": prior["score"]}, default=ws._json_default) + "\n")

    codec_observations = np.stack([row.obs for row in rows])
    pre_codec_probs = wd._policy_probs(agent.core, codec_observations)
    codec = roundtrip_actor(agent)
    post_codec_probs = wd._policy_probs(agent.core, codec_observations)
    codec["codec_current_rollout_max_probability_error"] = float(np.max(np.abs(pre_codec_probs - post_codec_probs)))
    codec["codec_current_rollout_mean_kl"] = float(np.mean(np.sum(
        pre_codec_probs * np.log(np.maximum(pre_codec_probs, 1e-12) /
                                 np.maximum(post_codec_probs, 1e-12)), axis=1)))
    ws._clear_optimizer_state(agent)
    update_record = {"step": agent.counts["raw_steps"], "slot": slot,
        "generation": int(agent.slot_generation[slot]), "rows": len(rows), "group": group,
        "group_reason": assignment.reason, "fresh_assignment_group": assigned_group,
        "archive": archive_result, **update, **codec, **critic_diagnostics}
    updates.write(json.dumps(update_record, default=ws._json_default) + "\n")

    if agent.external_condition in ("mean", "mean_reuse_gate", "retention_gate") and agent.pending is None \
            and agent.counts["rollouts_trained"] % config["proposal_interval_rollouts"] == 0 and group >= 0:
        groups = agent.core.slot_groups.detach().cpu().numpy()
        sources = [int(index) for index in np.flatnonzero(groups == group)
                   if agent.fitted_steps[index] >= config["proposal_min_fit_steps"]]
        if len(sources) >= 2:
            local, shared, groups = ws._actor_arrays(agent.core)
            common = common_mean(local, shared, groups, sources)
            agent.pending = {"group": group, "parent_version": int(agent.group_versions[group]),
                "common": common, "created_step": int(agent.counts["raw_steps"]),
                "sources": [{"slot": s, "generation": int(agent.slot_generation[s]),
                    "prototype": int(agent.slot_prototype[s])} for s in sources]}
            agent.counts["write_proposals"] += 1
            events.write(json.dumps({"event": "write_proposal", **agent.pending}, default=ws._json_default) + "\n")


def run_one(config, method, budget, seed, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    agent = make_agent(config, method, budget, seed)
    foundation_hash = sha(agent.base_config["foundation"])
    schedule = task_schedule(seed)
    write(folder / "schedule.json", {"sequence": schedule, "bout_steps": config["bout_steps"],
        "raw_steps": len(schedule) * config["bout_steps"], "observer_only": True})
    write(folder / "run_config.json", {"method": method, "budget": budget, "seed": seed,
        "agent_config": agent.run_config, "analysis_role": "exploratory"})
    env = make_env(task_from_name(schedule[0]), config["domain"])
    obs, _ = env.reset(seed=seed + 22000)
    episode_id, total_reward, maximum = 0, 0.0, {}
    seen_tasks, evicted_tasks, recurrence, capacity_records = set(), set(), [], []
    steps_stream = (folder / "steps.csv").open("w", newline="", encoding="utf-8")
    steps_writer = csv.DictWriter(steps_stream, fieldnames=("raw_step", "bout", "task", "slot", "generation",
        "episode_id", "action", "uniform", "reward", "goal", "terminated", "truncated", "learning_boundary",
        "obs", "next_obs", "behavior_probs", "behavior_log_prob", "value", "next_value"))
    steps_writer.writeheader()
    events = (folder / "events.jsonl").open("w", encoding="utf-8")
    updates = (folder / "updates.jsonl").open("w", encoding="utf-8")
    try:
        for bout, task in enumerate(schedule):
            env.set_task(task_from_name(task))
            allocation = agent.cache.resolve(task)
            slot = allocation["slot"]
            if allocation["allocated"]:
                if allocation["evicted_task"] is not None:
                    evicted_tasks.add(allocation["evicted_task"])
                    agent.counts["slot_evictions"] += 1
                ws._reset_allocated_slot(agent, slot, reason="natural_lru_allocation")
                agent.slot_prototype[slot] = -1
                agent.slot_prototype_version[slot] = 0
                agent.prior_injected[slot] = 0
                events.write(json.dumps({"event": "allocation", "step": agent.counts["raw_steps"],
                    "bout": bout, "task_observer": task, **allocation}) + "\n")
            is_recurrence = task in seen_tasks
            was_evicted = is_recurrence and task in evicted_tasks
            seen_tasks.add(task)
            bout_rewards, bout_goals, rows = [], [], []
            first_gradient = None
            for within in range(config["bout_steps"]):
                raw_step = int(agent.counts["raw_steps"]) + 1
                augmented = loop.augment_observation(obs, slot + 1)
                with torch.no_grad():
                    tensor = torch.as_tensor(augmented[None], dtype=torch.float32)
                    probabilities = agent.core.get_distribution(tensor).distribution.probs[0].cpu().numpy().copy()
                    value = float(agent.core.predict_values(tensor).item())
                probabilities = probabilities.astype(np.float64)
                probabilities /= probabilities.sum()
                uniform = float(agent.behavior_rng.random())
                action = wd._action_from_uniform(probabilities, uniform)
                following, reward, terminated, truncated, info = env.step(action)
                augmented_next = loop.augment_observation(following, slot + 1)
                with torch.no_grad():
                    next_value = float(agent.core.predict_values(torch.as_tensor(augmented_next[None], dtype=torch.float32)).item())
                boundary = within + 1 == config["bout_steps"]
                row = loop.Transition(augmented, action, float(reward), value, next_value,
                    float(np.log(probabilities[action])), probabilities.astype(np.float32),
                    bool(terminated or truncated or boundary), bool(terminated or boundary),
                    slot, int(agent.slot_generation[slot]), raw_step)
                row.episode_id = episode_id
                rows.append(row)
                agent.windows[slot].observe(obs, action, reward, following, bool(terminated))
                agent.counts["raw_steps"] += 1
                agent.counts["environment_steps"] += 1
                total_reward += float(reward)
                bout_rewards.append(float(reward))
                bout_goals.append(bool(info["goal_reached"]))
                steps_writer.writerow({"raw_step": raw_step, "bout": bout, "task": task, "slot": slot,
                    "generation": int(agent.slot_generation[slot]), "episode_id": episode_id,
                    "action": action, "uniform": uniform, "reward": reward, "goal": int(info["goal_reached"]),
                    "terminated": int(terminated), "truncated": int(truncated), "learning_boundary": int(boundary),
                    "obs": json.dumps(obs.tolist()), "next_obs": json.dumps(following.tolist()),
                    "behavior_probs": json.dumps(probabilities.tolist()), "behavior_log_prob": row.log_prob,
                    "value": value, "next_value": next_value})
                obs = following
                if terminated or truncated:
                    episode_id += 1
                    obs, _ = env.reset(seed=seed + 22000 + episode_id * 31)
                if len(rows) == config["rollout_steps"] or boundary:
                    process_rollout(agent, config, rows, slot, events, updates)
                    if first_gradient is None:
                        first_gradient = within + 1
                    rows = []
            if is_recurrence:
                goal_positions = np.flatnonzero(bout_goals)
                recurrence.append({"task": task, "bout": bout, "previously_evicted": was_evicted,
                    "first256_reward_per_step": float(np.mean(bout_rewards[:256])),
                    "recovery512_reward_per_step": float(np.mean(bout_rewards[:512])),
                    "recovery1024_reward_per_step": float(np.mean(bout_rewards[:1024])),
                    "first_goal_step": int(goal_positions[0] + 1) if len(goal_positions) else None,
                    "first_gradient_step": first_gradient, "observed_steps": len(bout_rewards)})
            checkpoint = folder / "checkpoints" / f"step_{agent.counts['raw_steps']:08d}.pt"
            sizes = save_checkpoint(agent, config, method, budget, checkpoint)
            capacity_records.append({"step": agent.counts["raw_steps"], **sizes})
            for key, value in sizes.items():
                maximum[key] = max(maximum.get(key, 0), value)
            events.flush()
            updates.flush()
            steps_stream.flush()

        final_checkpoint = checkpoint
        restored = restore_checkpoint(final_checkpoint)
        if policy_hash(agent.core) != policy_hash(restored.core):
            raise RuntimeError("checkpoint actor/critic roundtrip changed policy state")
        if agent.archive.nbytes != restored.archive.nbytes or \
                agent.counts != restored.counts or agent.cache.state_dict()["task_to_slot"] != restored.cache.state_dict()["task_to_slot"]:
            raise RuntimeError("persistent learner state did not restore")
        expected_steps = len(schedule) * config["bout_steps"]
        if agent.counts["raw_steps"] != expected_steps or agent.counts["fit_steps"] != expected_steps:
            raise RuntimeError("raw time or all-data PPO submission accounting mismatch")
        result = {"format": "retention_stream_run_v1", "status": "completed", "analysis_role": "exploratory",
            "method": method, "budget": budget, "seed": seed, "raw_steps": expected_steps,
            "fit_steps": agent.counts["fit_steps"], "total_reward": total_reward,
            "reward_per_raw_step": total_reward / expected_steps,
            "slot_evictions": agent.counts["slot_evictions"],
            "rollouts_trained": agent.counts["rollouts_trained"], "minibatch_updates": agent.counts["minibatch_updates"],
            "write_proposals": agent.counts["write_proposals"], "write_accepts": agent.counts["write_accepts"],
            "prior_bindings": agent.counts["prior_bindings"], "prior_rejects": agent.counts["prior_rejects"],
            "capacity_max": maximum, "recurrences": recurrence,
            "cpu_s": time.process_time() - cpu_started, "wall_s": time.perf_counter() - started,
            "audit": {"passed": True, "source_unchanged": sha(agent.base_config["foundation"]) == foundation_hash,
                "persistent_roundtrip": True, "all_raw_rows_submitted_to_ppo": True},
            "limits": "Known local indices/boundaries; parameter caps plus package ceiling, not total-state optimal allocation. Proxy gates lack safety guarantees."}
        write(folder / "capacity.json", capacity_records)
        write(folder / "summary.json", result)
        return result
    except Exception as error:
        write(folder / "FAILED_RUN.json", {"status": "failed", "reason": repr(error),
            "completed_raw_steps": int(agent.counts["raw_steps"]), "method": method, "seed": seed,
            "budget": budget, "cpu_s": time.process_time() - cpu_started})
        raise
    finally:
        env.close()
        steps_stream.close()
        events.close()
        updates.close()


def run_cohort(config, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new output directory; preserve old experiments")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "frozen_config.json", config)
    manifest = snapshot_sources(output)
    started = time.perf_counter()
    jobs = [(method, budget, seed) for seed in config["seeds"] for budget in config["budgets"] for method in config["methods"]]
    records, errors = [], []
    with ProcessPoolExecutor(max_workers=config["max_workers"]) as executor:
        futures = {executor.submit(run_one, config, method, budget, seed,
            output / f"seed_{seed}" / budget / method): (method, budget, seed) for method, budget, seed in jobs}
        for future in as_completed(futures):
            method, budget, seed = futures[future]
            try:
                result = future.result()
                records.append(result)
                print(f"completed seed={seed} budget={budget} method={method} R/raw={result['reward_per_raw_step']:.6f}", flush=True)
            except Exception as error:
                errors.append({"method": method, "budget": budget, "seed": seed, "reason": repr(error)})
                print(f"failed seed={seed} budget={budget} method={method}: {error!r}", flush=True)
    changed = [relative for relative, entry in manifest.items() if sha(ROOT / relative) != entry["sha256"]]
    if changed:
        errors.append({"reason": "sources_changed_during_experiment", "paths": changed})
    total = sum(record["raw_steps"] for record in records)
    partial = 0
    for failure in output.rglob("FAILED_RUN.json"):
        partial += int(json.loads(failure.read_text(encoding="utf-8"))["completed_raw_steps"])
    summary = {"format": "retention_stream_cohort_v1", "status": "completed" if not errors else "failed",
        "analysis_role": "exploratory", "config": config, "records": records,
        "audit": {"passed": not errors, "failures": errors}, "failed_run_raw_steps": partial,
        "cohort_cost": {"new_raw_steps": total + partial, "new_evaluation_steps": 0,
            "new_total_interactions": total + partial},
        "cpu_s": sum(record["cpu_s"] for record in records), "wall_s": time.perf_counter() - started}
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "exploratory"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["mode"] = args.mode
    if args.mode == "pilot":
        config["seeds"] = [config["pilot_seed"]]
        config["bout_steps"] = config["pilot_bout_steps"]
        config["budgets"] = {"low": config["budgets"]["low"]}
    summary = run_cohort(config, args.output)
    if summary["status"] != "completed":
        raise SystemExit(1)
    print(json.dumps(summary["cohort_cost"]))


if __name__ == "__main__":
    main()

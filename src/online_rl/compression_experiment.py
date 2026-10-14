"""Measured Top-k deployment compression, with an active-only dense control."""
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

import numpy as np
import psutil
import torch
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from .compression import PackedResidualPolicy, export_payload
from .controls import (clear_router_working_memory, history_router_from_snapshot,
                       load_terminal_run, save_deployment_bundle)
from .envs import NavigationEnv, contexts
from .experiment import BudgetCallback, StreamEnv, FIELDS, make_model, seed_everything, write_json
from .policy import tensor_bytes


class PackedSB3Adapter(BasePolicy):
    """Use the compact module with the existing PPO loss and rollout code."""
    def __init__(self, core, optimizer):
        super().__init__(core.observation_space, core.action_space, normalize_images=False)
        self.core = core
        self.optimizer = optimizer

    def _predict(self, observation, deterministic=False):
        return self.core.get_distribution(observation).get_actions(deterministic=deterministic)

    def forward(self, obs, deterministic=False):
        return self.core(obs, deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        return self.core.evaluate_actions(obs, actions)

    def predict_values(self, obs):
        return self.core.predict_values(obs)

    def get_distribution(self, obs):
        return self.core.get_distribution(obs)


def actor_hash(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        if name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def storage_metrics(policy, optimizer):
    report = policy.storage_report()
    residual_optimizer = sum(tensor_bytes(state) for parameter, state in optimizer.state.items()
                             if getattr(parameter, "tree_residual", False))
    return {**report, "optimizer_tensor_bytes": tensor_bytes(optimizer.state_dict()),
            "residual_optimizer_tensor_bytes": residual_optimizer,
            "adaptive_tensor_bytes": report["residual_value_bytes"] +
                                     report["coordinate_index_bytes"] + residual_optimizer}


def probe_observations(nodes):
    rows = []
    for node in [0, *nodes]:
        for x in range(9):
            for y in range(9):
                for gx in range(9):
                    for gy in range(9):
                        if (x, y) != (gx, gy):
                            rows.append([x / 8, y / 8, gx / 8, gy / 8, *([0.] * 8), node])
    return torch.as_tensor(rows, dtype=torch.float32)


def probability_probe(reference, candidate, observations):
    maximum = 0.
    kl_sum = 0.
    with torch.no_grad():
        for batch in observations.split(512):
            p = reference.get_distribution(batch).distribution.probs
            q = candidate.get_distribution(batch).distribution.probs
            maximum = max(maximum, float((p - q).abs().max()))
            kl_sum += float((p * ((p + 1e-12).log() - (q + 1e-12).log())).sum())
    return {"max_probability_difference": maximum,
            "mean_grid_KL_reference_to_candidate": kl_sum / len(observations),
            "grid_observations": len(observations)}


def _load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = PackedResidualPolicy.from_payload(payload)
    del payload
    gc.collect()
    return restored


def evaluate_deployment(bundle_path, output, config, seed, context, cohort, method):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    restored = _load(bundle_path)
    policy, optimizer = restored["policy"], restored["optimizer"]
    router = history_router_from_snapshot(restored["router"])
    clear_router_working_memory(router)
    policy.eval()
    before_hash = actor_hash(policy)
    before_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    offset = config["validation_seed_offset"] if cohort == "validation" else config["test_seed_offset"]
    evaluation_seed = int(offset) + seed * 100 + config["contexts"].index(context)
    seed_everything(evaluation_seed)
    action_rng = np.random.default_rng(evaluation_seed + 1)
    env = NavigationEnv(contexts()[context])
    obs, _ = env.reset(seed=evaluation_seed)
    process = psutil.Process()
    rss = process.memory_info().rss
    peak_serialized_core = Path(bundle_path).stat().st_size
    total = early = 0.
    completed = successes = 0
    infer_time = router_time = 0.
    steps = int(config["deployment_steps"])
    early_steps = min(steps, int(config["early_window_steps"]))
    fields = ["source_seed", "cohort", "method", "context", "step", "episode",
              "x", "y", "gx", "gy", "action", "reward", "next_x", "next_y",
              "next_gx", "next_gy", "route", "confidence", "done", "success"]
    start = time.perf_counter()
    with (output / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for step in range(steps):
                rt = time.perf_counter()
                route, confidence = router.route(), router.confidence
                augmented = np.concatenate([obs, router.summary(), [route]]).astype(np.float32)
                router_time += time.perf_counter() - rt
                pt = time.perf_counter()
                probabilities = policy.get_distribution(torch.as_tensor(augmented[None])).distribution.probs[0]
                p = probabilities.numpy().astype(np.float64)
                p /= p.sum()
                action = int(action_rng.choice(4, p=p))
                infer_time += time.perf_counter() - pt
                next_obs, reward, terminated, truncated, info = env.step(action)
                rt = time.perf_counter()
                router.observe(obs, action, next_obs)
                router_time += time.perf_counter() - rt
                total += float(reward)
                if step < early_steps:
                    early += float(reward)
                done = terminated or truncated
                writer.writerow({"source_seed": seed, "cohort": cohort, "method": method,
                                 "context": context, "step": step + 1, "episode": completed + 1,
                                 "x": float(obs[0]), "y": float(obs[1]), "gx": float(obs[2]), "gy": float(obs[3]),
                                 "action": action, "reward": reward, "next_x": float(next_obs[0]),
                                 "next_y": float(next_obs[1]), "next_gx": float(next_obs[2]), "next_gy": float(next_obs[3]),
                                 "route": route, "confidence": confidence, "done": int(done),
                                 "success": int(info["success"]) if done else ""})
                obs = next_obs
                if done:
                    completed += 1
                    successes += int(info["success"])
                    obs, _ = env.reset()
                if (step + 1) % 256 == 0:
                    rss = max(rss, process.memory_info().rss)
                    live = export_payload(policy, optimizer, router.snapshot(),
                                          {"source_seed": seed, "last_observation": obs},
                                          {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                                           "action_generator": action_rng.bit_generator.state})
                    live_bytes = io.BytesIO()
                    torch.save(live, live_bytes)
                    peak_serialized_core = max(peak_serialized_core, live_bytes.tell())
                    if peak_serialized_core > int(config["agent_budget_bytes"]):
                        raise RuntimeError("Compressed deployment core exceeded its accounted budget")
                    del live, live_bytes
    elapsed = time.perf_counter() - start
    unchanged = all(torch.equal(before_state[k], v) for k, v in policy.state_dict().items())
    if not unchanged or before_hash != actor_hash(policy):
        raise RuntimeError("No-gradient compression deployment mutated policy parameters")
    env.close()
    result = {"source_seed": seed, "cohort": cohort, "method": method, "context": context,
              "evaluation_seed": evaluation_seed, "online_steps": steps,
              "online_cumulative_return": total, "early_window_cumulative_reward": early,
              "complete_episode_success_rate": successes / max(1, completed),
              "completed_episodes": completed, "successful_episodes": successes,
              "online_elapsed_s": elapsed, "policy_inference_elapsed_s": infer_time,
              "router_elapsed_s": router_time, "max_sampled_process_rss_bytes": rss,
              "policy_unchanged": unchanged, "gradient_updates": 0,
              "representation": storage_metrics(policy, optimizer),
              "starting_bundle_bytes": Path(bundle_path).stat().st_size,
              "max_sampled_serialized_core_bytes": peak_serialized_core,
              "training_buffer_bytes": 0,
              "observer_note": "parameter audit copies and raw research logs are not usable agent memory; RSS includes experiment observer overhead"}
    write_json(output / "summary.json", result)
    return result


class CompactBudgetCallback(BudgetCallback):
    def sample_capacity(self):
        adapter = self.model.policy
        meta = {"last_observation": self.model._last_obs,
                "last_episode_starts": self.model._last_episode_starts,
                "optimizer_step_count": adapter.optimizer.step_count}
        payload = export_payload(adapter.core, adapter.optimizer, self.stream.router.snapshot(), meta)
        storage = io.BytesIO()
        torch.save(payload, storage)
        rollout_bytes = sum(v.nbytes for v in vars(self.model.rollout_buffer).values() if isinstance(v, np.ndarray))
        core_bytes = storage.tell()
        values = {"policy_bytes": tensor_bytes(adapter.core.state_dict()),
                  "optimizer_bytes": tensor_bytes(adapter.optimizer.state_dict()),
                  "router_bytes": len(json.dumps(self.stream.router.snapshot()).encode()),
                  "rollout_bytes": rollout_bytes, "serialized_core_bytes": core_bytes,
                  "agent_state_bytes": core_bytes + rollout_bytes}
        self.stream.capacity_info = values
        self.peak = max(self.peak, values["agent_state_bytes"])
        self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
        self.writer.writerow({"global_steps": self.stream.total_steps, **values, "rss_bytes": self.rss_peak})
        if self.peak > self.budget:
            raise RuntimeError(f"Compact deployed training state exceeds budget: {self.peak}>{self.budget}")


def run_continuation(bundle_path, output, config, seed, method):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    restored = _load(bundle_path)
    core, optimizer = restored["policy"], restored["optimizer"]
    router = history_router_from_snapshot(restored["router"])
    clear_router_working_memory(router)
    before_actor = actor_hash(core)
    before_values = {name: node.values.detach().clone() for name, node in core._nodes.items()}
    initial_updates = optimizer.step_count
    evaluation_seed = int(config["test_seed_offset"]) + 10000 + seed
    seed_everything(evaluation_seed)
    run_config = {**config, "bout_steps": int(config["continuation_bout_steps"])}
    cap_fields = ["global_steps", "policy_bytes", "optimizer_bytes", "router_bytes", "rollout_bytes",
                  "serialized_core_bytes", "agent_state_bytes", "rss_bytes"]
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as ef, \
         (output / "capacity.csv").open("w", newline="", encoding="utf-8") as cf:
        ew = csv.DictWriter(ef, fieldnames=FIELDS)
        ew.writeheader()
        cw = csv.DictWriter(cf, fieldnames=cap_fields)
        cw.writeheader()
        stream = StreamEnv("compression", "history_keep", evaluation_seed, router, run_config, ew, ["cw", "ccw"])
        vec = DummyVecEnv([lambda: stream])
        model = make_model(vec, {**run_config, "seed": evaluation_seed})
        # Replacing the constructor's policy discards its dense residual slots.
        model.policy = PackedSB3Adapter(core, optimizer)
        stream.model = model
        callback = CompactBudgetCallback(stream, cw, int(config["agent_budget_bytes"]))
        start = time.perf_counter()
        model.learn(total_timesteps=int(config["continuation_steps"]), callback=callback)
        elapsed = time.perf_counter() - start
        updated = {name: int(torch.count_nonzero(node.values.detach() - before_values[name]))
                   for name, node in core._nodes.items()}
        if actor_hash(core) != before_actor:
            raise RuntimeError("Compact PPO continuation changed the foundation actor")
        after_payload = export_payload(core, optimizer, router.snapshot(),
                                       {"source_seed": seed, "method": method, "continuation_seed": evaluation_seed,
                                        "online_steps": stream.total_steps})
        torch.save(after_payload, output / "agent.pt")
        np.savez_compressed(output / "last_rollout_audit.npz", observations=model.rollout_buffer.observations,
                            actions=model.rollout_buffer.actions, log_probs=model.rollout_buffer.log_probs,
                            optimizer_versions=model.rollout_buffer.optimizer_versions,
                            reset_versions=model.rollout_buffer.reset_versions)
        result = {"source_seed": seed, "method": method, "online_steps": stream.total_steps,
                  "online_cumulative_return": stream.cumulative_return, "online_elapsed_s": elapsed,
                  "optimizer_steps_added": optimizer.step_count - initial_updates,
                  "updated_residual_coordinates": updated, "base_actor_unchanged": True,
                  "peak_accounted_state_bytes": callback.peak, "max_sampled_process_rss_bytes": callback.rss_peak,
                  "representation_after": storage_metrics(core, optimizer),
                  "scope": "same known context families; fixed sparse support, no new node allocation"}
        vec.close()
    write_json(output / "summary.json", result)
    return result


def select_candidate(records, representation, config):
    trials = []
    fractions = config["retain_fractions"]
    for fraction in fractions:
        method = f"packed_{fraction:g}"
        conditions = []
        for context in config["contexts"]:
            base = [r for r in records if r["method"] == "dense_compact" and r["context"] == context]
            candidate = [r for r in records if r["method"] == method and r["context"] == context]
            bs = statistics.mean(r["complete_episode_success_rate"] for r in base)
            cs = statistics.mean(r["complete_episode_success_rate"] for r in candidate)
            br = statistics.mean(r["online_cumulative_return"] for r in base)
            cr = statistics.mean(r["online_cumulative_return"] for r in candidate)
            conditions.append({"context": context, "success_drop": bs - cs,
                               "relative_return_drop": (br - cr) / br if br > 0 else None,
                               "success_ok": bs - cs <= config["maximum_success_drop"],
                               "return_ok": br <= 0 or (br - cr) / br <= config["maximum_relative_return_drop"]})
        refs = [r for r in representation if r["method"] == "dense_compact"]
        reps = [r for r in representation if r["method"] == method]
        savings = statistics.mean(1 - c["adaptive_tensor_bytes"] / b["adaptive_tensor_bytes"]
                                  for b, c in zip(refs, reps))
        total_smaller = all(c["bundle_bytes"] < b["bundle_bytes"] for b, c in zip(refs, reps))
        accepted = (savings >= config["minimum_adaptive_numeric_savings"] and total_smaller and
                    all(c["success_ok"] and c["return_ok"] for c in conditions))
        trials.append({"method": method, "fraction": fraction, "mean_adaptive_numeric_saving": savings,
                       "all_bundles_smaller": total_smaller, "per_context": conditions, "accepted": accepted})
    passing = [t for t in trials if t["accepted"]]
    return {"selected_method": min(passing, key=lambda t: t["fraction"])["method"] if passing else None,
            "selected_fraction": min(t["fraction"] for t in passing) if passing else None,
            "criteria": {k: config[k] for k in ["maximum_success_drop", "maximum_relative_return_drop",
                                                 "minimum_adaptive_numeric_savings"]}, "trials": trials}


def run_suite(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    representations = []
    bundles = {}
    original_references = []
    preflight = []
    # Build all bundles first, then release the uncompressed source models.
    for seed in config["source_seeds"]:
        model, router = load_terminal_run(Path(config["source_root"]) / f"B_history_keep_s{seed}", config)
        nodes = list(range(1, len(router.prototypes)))
        obs = probe_observations(nodes)
        origpath = output / "bundles" / f"original_s{seed}" / "agent.pt"
        origpath.parent.mkdir(parents=True, exist_ok=True)
        save_deployment_bundle(origpath, model, router, {"source_seed": seed})
        original_references.append({"source_seed": seed, "method": "original_eight_slots",
                                    "bundle_bytes": origpath.stat().st_size,
                                    "policy_tensor_bytes": tensor_bytes(model.policy.state_dict()),
                                    "optimizer_tensor_bytes": tensor_bytes(model.policy.optimizer.state_dict())})
        seed_everything(int(config["validation_seed_offset"]) + seed)
        for method, storage, fraction in [("dense_compact", "dense", 1.)] + [
                (f"packed_{f:g}", "packed", f) for f in config["retain_fractions"]]:
            policy = PackedResidualPolicy.from_dense(model.policy, nodes, storage=storage, retain_fraction=fraction)
            optimizer = policy.make_optimizer(model.policy.optimizer, learning_rate=config["learning_rate"])
            probe = probability_probe(model.policy, policy, obs)
            if fraction == 1. and probe["max_probability_difference"] > 2e-5:
                raise RuntimeError(f"Lossless representation changed policy: {probe}")
            path = output / "bundles" / f"{method}_s{seed}" / "agent.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = export_payload(policy, optimizer, router.snapshot(), {"source_seed": seed})
            torch.save(payload, path)
            rebuilt = _load(path)
            roundtrip = probability_probe(policy, rebuilt["policy"], obs)
            if roundtrip["max_probability_difference"] > 2e-5:
                raise RuntimeError("Compressed disk roundtrip changed the policy")
            representations.append({"source_seed": seed, "method": method, "fraction": fraction,
                                    **storage_metrics(policy, optimizer), "bundle_bytes": path.stat().st_size,
                                    "probability_probe": probe, "roundtrip_probe": roundtrip})
            bundles[(seed, method)] = path
            del rebuilt, payload, policy, optimizer
        model.env.close()
        del model, router, obs
        gc.collect()
    write_json(output / "representation.json", {"original_references": original_references,
                                                "representations": representations})
    records = []
    methods = ["dense_compact", *[f"packed_{f:g}" for f in config["retain_fractions"]]]
    for seed in config["source_seeds"]:
        for method in methods:
            for context in config["contexts"]:
                record = evaluate_deployment(bundles[(seed, method)],
                                             output / "validation" / f"{method}_s{seed}_{context}",
                                             config, seed, context, "validation", method)
                records.append(record)
            print(json.dumps({"phase": "validation", "source_seed": seed, "method": method}), flush=True)
    selection = select_candidate(records, representations, config)
    write_json(output / "selection.json", selection)
    selected = selection["selected_method"]
    print(json.dumps({"phase": "selection", "selected_method": selected}), flush=True)
    test_records = []
    continuations = []
    if selected is not None:
        for seed in config["source_seeds"]:
            for method in ["dense_compact", selected]:
                for context in config["contexts"]:
                    test_records.append(evaluate_deployment(bundles[(seed, method)],
                                                           output / "test" / f"{method}_s{seed}_{context}",
                                                           config, seed, context, "test", method))
                continuations.append(run_continuation(bundles[(seed, method)],
                                                      output / "continuation" / f"{method}_s{seed}",
                                                      config, seed, method))
                print(json.dumps({"phase": "test_and_continuation", "source_seed": seed, "method": method}), flush=True)
    result = {"scope": "packed Top-k representation pilot; no hierarchical consolidation",
              "config": config, "original_references": original_references,
              "representations": representations, "validation": records, "selection": selection,
              "test": test_records, "continuation": continuations,
              "total_interactions": sum(r["online_steps"] for r in records + test_records + continuations),
              "limitations": ["three existing learned checkpoints, one foundation seed",
                              "simple rotation family; no unseen sparse-support growth",
                              "empirical acceptance thresholds, not statistical or safety guarantees",
                              "sparse indices can increase full-retention storage",
                              "serialized bundle and numeric bytes differ from process RSS"]}
    write_json(output / "summary.json", result)
    return result

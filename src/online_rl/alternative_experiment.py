"""Fixed comparison of residual compression alternatives on learned A/B agents."""
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from .alternative_compression import compress_policy, restore_payload
from .compression import export_payload
from .compression_experiment import (CompactBudgetCallback, PackedSB3Adapter,
                                     actor_hash, probability_probe, probe_observations,
                                     storage_metrics)
from .controls import clear_router_working_memory, history_router_from_snapshot
from .envs import NavigationEnv, contexts
from .experiment import FIELDS, StreamEnv, make_model, seed_everything, write_json
from .policy import LeafAdam, tensor_bytes


def load_bundle(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = restore_payload(payload)
    del payload
    return result


def state_projection(core):
    residuals = [p for p in core.parameters() if getattr(p, "tree_residual", False)]
    # Native Adam allocates two zeros_like moments and one FP32 step per tensor.
    return core.storage_report()["coordinate_index_bytes"] + 3 * sum(
        p.numel() * p.element_size() for p in residuals) + 4 * len(residuals)


def replay_calibration(seed, count, source_router):
    """Recover recorded actor/critic inputs without any environment calls."""
    start = time.perf_counter()
    observations, hashes, replayed = [], {}, 0
    route_counts = {}
    for stream in ("cw", "ccw"):
        path = Path(f"outputs/ab_plus/no_gradient/s{seed}/{stream}/Pkeep_Hclear/steps.csv")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        router = history_router_from_snapshot(source_router)
        clear_router_working_memory(router)
        taken = 0
        with path.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                obs = np.array([float(record[k]) for k in ("obs_x", "obs_y", "obs_goal_x", "obs_goal_y")], np.float32)
                next_obs = np.array([float(record[k]) for k in ("next_x", "next_y", "next_goal_x", "next_goal_y")], np.float32)
                route = int(router.route())
                if route != int(record["route"]) or abs(router.confidence - float(record["route_confidence"])) > 2e-6:
                    raise RuntimeError(f"Calibration router replay mismatch at {path}:{record['step']}")
                if route != 0 and taken < count:
                    observations.append(np.concatenate([obs, router.summary(), [route]]).astype(np.float32))
                    route_counts[route] = route_counts.get(route, 0) + 1
                    taken += 1
                router.observe(obs, int(record["action"]), next_obs)
                replayed += 1
                if taken == count:
                    break
        if taken != count:
            raise RuntimeError("Not enough route-specific observed calibration states")
    if route_counts != {1: count, 2: count}:
        raise RuntimeError(f"Unexpected calibration routes {route_counts}; no unit-RMS fallback permitted")
    tensor = torch.as_tensor(np.stack(observations))
    return tensor, {"source_seed": seed, "states": len(tensor), "route_counts": route_counts,
                    "replayed_log_steps": replayed, "new_environment_interactions": 0,
                    "input_sha256": hashes, "elapsed_s": time.perf_counter() - start,
                    "tensor_bytes": tensor.numel() * tensor.element_size(),
                    "scope": "past online control trajectories explicitly reassigned to compression calibration; reconstructed actual13-D observations"}


def distill(reference, core, observations, config):
    core._set_trainable_flags(train_critic=False)
    residual_parameters = [p for p in core.parameters() if getattr(p, "tree_residual", False)]
    optimizer = LeafAdam(residual_parameters, lr=config["distillation_learning_rate"], eps=1e-5)
    before_actor = actor_hash(core)
    with torch.no_grad():
        teacher = reference.get_distribution(observations).distribution.probs.detach()
    losses = []
    peak_explicit = 0
    for _ in range(config["distillation_steps"]):
        student = core.get_distribution(observations).distribution.probs
        loss = (teacher * ((teacher + 1e-12).log() - (student + 1e-12).log())).sum(1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Distillation produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_parameters, 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        # Explicit unique tensor categories; framework/BLAS temporary buffers
        # are not claimed to be included in this working-set estimate.
        peak_explicit = max(peak_explicit, tensor_bytes(reference.state_dict()) +
                            tensor_bytes(core.state_dict()) + tensor_bytes(optimizer.state_dict()) +
                            tensor_bytes([observations, teacher, student]) +
                            tensor_bytes([p.grad for p in residual_parameters]))
    if actor_hash(core) != before_actor:
        raise RuntimeError("Distillation modified the foundation actor")
    core._set_trainable_flags(train_critic=True)
    return {"optimizer_steps": len(losses), "first_batch_teacher_KL": losses[0],
            "last_batch_teacher_KL": losses[-1], "peak_explicit_live_tensor_bytes": peak_explicit,
            "teacher_and_calibration_retained_in_bundle": False}


def construct(seed, method, reference, router, calibration, config, output):
    start = time.perf_counter()
    seed_everything(config["test_seed_offset"] + seed)
    layout = "svd2" if method == "svd2_distilled" else method
    core, metadata = compress_policy(reference, layout, calibration)
    for node in metadata["nodes"].values():
        if node["calibration_observations"] != config["calibration_states_per_route"]:
            raise RuntimeError("Missing own-route calibration would change the intended method")
    distillation = distill(reference, core, calibration, config) if method == "svd2_distilled" else None
    optimizer = core.make_optimizer(learning_rate=config["learning_rate"], transfer=False)
    if optimizer.state:
        raise RuntimeError("Online comparison must start with empty optimizer states")
    build_s = time.perf_counter() - start
    path = output / "bundles" / f"{method}_s{seed}" / "agent.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_payload(core, optimizer, router, {"source_seed": seed, "method": method,
                                                       "optimizer_initialization": "fresh"})
    torch.save(payload, path)
    restored = load_bundle(path)
    probes = probe_observations([1, 2])
    probe = probability_probe(reference, core, probes)
    roundtrip = probability_probe(core, restored["policy"], probes)
    if roundtrip["max_probability_difference"] > 2e-5:
        raise RuntimeError("Compression round trip changes the policy")
    if method in ("dense", "contrast") and probe["max_probability_difference"] > 2e-5:
        raise RuntimeError("Declared lossless layout changed action probabilities")
    explicit = tensor_bytes(reference.state_dict()) + tensor_bytes(core.state_dict()) + tensor_bytes(calibration)
    report = {"source_seed": seed, "method": method, "construction_elapsed_s": build_s,
              "compression_optimizer_steps": 0 if distillation is None else distillation["optimizer_steps"],
              "construction": metadata, "distillation": distillation,
              "construction_explicit_live_tensor_bytes_lower_bound": max(
                  explicit, 0 if distillation is None else distillation["peak_explicit_live_tensor_bytes"]),
              "workspace_scope": "explicit tensors; excludes unmeasured autograd/BLAS temporaries and filesystem serialization copies",
              "initial_representation": storage_metrics(core, optimizer), "bundle_bytes": path.stat().st_size,
              "predicted_warmed_adaptive_bytes": state_projection(core),
              "source_to_candidate_probability_probe": probe, "roundtrip_probe": roundtrip,
              "base_actor_unchanged": actor_hash(reference) == actor_hash(core),
              "optimizer_states_initially_empty": not bool(optimizer.state)}
    write_json(path.parent / "construction.json", report)
    del restored, payload, core, optimizer, probes
    gc.collect()
    return report


def evaluate(path, output, config, seed, method, context, phase):
    output.mkdir(parents=True, exist_ok=True)
    restored = load_bundle(path)
    core, optimizer = restored["policy"], restored["optimizer"]
    router = history_router_from_snapshot(restored["router"])
    clear_router_working_memory(router)
    core.eval()
    before = {name: value.detach().clone() for name, value in core.state_dict().items()}
    evaluation_seed = config["test_seed_offset"] + seed * 100 + config["contexts"].index(context)
    seed_everything(evaluation_seed)
    action_rng = np.random.default_rng(evaluation_seed + 1)
    env = NavigationEnv(contexts()[context])
    obs, _ = env.reset(seed=evaluation_seed)
    total = early = inference = router_time = 0.
    completed = successes = 0
    peak_core = path.stat().st_size
    process = psutil.Process()
    rss = process.memory_info().rss
    columns = ["source_seed", "phase", "method", "context", "step", "episode", "x", "y", "gx", "gy",
               "action", "reward", "next_x", "next_y", "next_gx", "next_gy", "route", "confidence", "done", "success"]
    start = time.perf_counter()
    with (output / "steps.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        with torch.no_grad():
            for step in range(config["deployment_steps"]):
                rt = time.perf_counter()
                route, confidence = int(router.route()), float(router.confidence)
                augmented = np.concatenate([obs, router.summary(), [route]]).astype(np.float32)
                router_time += time.perf_counter() - rt
                it = time.perf_counter()
                probabilities = core.get_distribution(torch.as_tensor(augmented[None])).distribution.probs[0]
                p = probabilities.numpy().astype(np.float64)
                if not np.all(np.isfinite(p)) or p.sum() <= 0:
                    raise RuntimeError("Non-finite deployment action probabilities")
                p /= p.sum()
                action = int(action_rng.choice(4, p=p))
                inference += time.perf_counter() - it
                after, reward, terminated, truncated, info = env.step(action)
                rt = time.perf_counter()
                router.observe(obs, action, after)
                router_time += time.perf_counter() - rt
                done = bool(terminated or truncated)
                total += reward
                early += reward if step < config["early_window_steps"] else 0
                writer.writerow({"source_seed": seed, "phase": phase, "method": method, "context": context,
                                 "step": step + 1, "episode": completed + 1,
                                 **dict(zip(("x", "y", "gx", "gy"), map(float, obs))),
                                 **dict(zip(("next_x", "next_y", "next_gx", "next_gy"), map(float, after))),
                                 "action": action, "reward": reward, "route": route, "confidence": confidence,
                                 "done": int(done), "success": int(info["success"]) if done else ""})
                obs = after
                if done:
                    completed += 1
                    successes += int(info["success"])
                    obs, _ = env.reset()
                if (step + 1) % 256 == 0:
                    rss = max(rss, process.memory_info().rss)
                    live = export_payload(core, optimizer, router.snapshot(), {"last_observation": obs},
                                          {"torch": torch.get_rng_state(), "action_generator": action_rng.bit_generator.state})
                    storage = io.BytesIO()
                    torch.save(live, storage)
                    peak_core = max(peak_core, storage.tell())
                    if peak_core > config["agent_budget_bytes"]:
                        raise RuntimeError("Evaluation exceeds declared serialized core budget")
                    del live, storage
    elapsed = time.perf_counter() - start
    if not all(torch.equal(before[name], value) for name, value in core.state_dict().items()):
        raise RuntimeError("No-gradient deployment changed policy state")
    env.close()
    report = {"source_seed": seed, "method": method, "context": context, "phase": phase,
              "evaluation_seed": evaluation_seed, "online_steps": config["deployment_steps"],
              "online_cumulative_return": total, "early_window_cumulative_reward": early,
              "complete_episode_success_rate": successes / max(1, completed),
              "completed_episodes": completed, "successful_episodes": successes,
              "online_elapsed_s": elapsed, "policy_inference_elapsed_s": inference,
              "router_elapsed_s": router_time, "representation": storage_metrics(core, optimizer),
              "starting_bundle_bytes": path.stat().st_size, "max_sampled_serialized_core_bytes": peak_core,
              "max_sampled_process_rss_bytes": rss, "policy_unchanged": True, "gradient_updates": 0,
              "observer_note": "RSS includes experiment observer audit copies; no claims of total instantaneous RAM bound"}
    write_json(output / "summary.json", report)
    return report


class MatchedStream(StreamEnv):
    def reset(self, **kwargs):
        if self.first_reset:
            self.first_reset = False
            kwargs["seed"] = self.seed
        return super().reset(**kwargs)


class CheckedBudget(CompactBudgetCallback):
    def sample_capacity(self):
        core = self.model.policy.core
        if not all(torch.isfinite(parameter).all() for parameter in core.parameters()):
            raise RuntimeError("Online repair produced non-finite model parameters")
        super().sample_capacity()


def continue_learning(path, output, config, seed, method):
    output.mkdir(parents=True, exist_ok=True)
    restored = load_bundle(path)
    core, optimizer = restored["policy"], restored["optimizer"]
    router = history_router_from_snapshot(restored["router"])
    clear_router_working_memory(router)
    before_actor = actor_hash(core)
    before_values = {name: p.detach().clone() for name, p in core.named_parameters() if getattr(p, "tree_residual", False)}
    continuation_seed = config["test_seed_offset"] + 10000 + seed
    seed_everything(continuation_seed)
    run_config = {**config, "bout_steps": config["continuation_bout_steps"]}
    columns = ["global_steps", "policy_bytes", "optimizer_bytes", "router_bytes", "rollout_bytes",
               "serialized_core_bytes", "agent_state_bytes", "rss_bytes"]
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as ef, \
         (output / "capacity.csv").open("w", encoding="utf-8", newline="") as cf:
        ew, cw = csv.DictWriter(ef, fieldnames=FIELDS), csv.DictWriter(cf, fieldnames=columns)
        ew.writeheader()
        cw.writeheader()
        stream = MatchedStream("compression", "history_keep", continuation_seed, router, run_config, ew, ["cw", "ccw"])
        vec = DummyVecEnv([lambda: stream])
        model = make_model(vec, {**run_config, "seed": continuation_seed})
        model.policy = PackedSB3Adapter(core, optimizer)
        stream.model = model
        if stream.method != "history_keep" or optimizer.step_count != 0 or optimizer.state:
            raise RuntimeError("Comparison requires history_keep and a fresh common optimizer")
        callback = CheckedBudget(stream, cw, config["agent_budget_bytes"])
        seed_everything(continuation_seed)  # Constructors must not shift action RNG.
        start = time.perf_counter()
        model.learn(total_timesteps=config["continuation_steps"], callback=callback)
        elapsed = time.perf_counter() - start
        if actor_hash(core) != before_actor:
            raise RuntimeError("Online repair altered the foundation actor")
        updated = {name: int(torch.count_nonzero(p.detach() - before_values[name]))
                   for name, p in core.named_parameters() if name in before_values}
        dtypes = {name: {key: str(value.dtype) for key, value in optimizer.state.get(p, {}).items() if torch.is_tensor(value)}
                  for name, p in core.named_parameters() if name in before_values}
        payload = export_payload(core, optimizer, router.snapshot(), {"source_seed": seed, "method": method,
                                                                     "continuation_seed": continuation_seed})
        torch.save(payload, output / "agent.pt")
        np.savez_compressed(output / "last_rollout_audit.npz", observations=model.rollout_buffer.observations,
                            actions=model.rollout_buffer.actions, log_probs=model.rollout_buffer.log_probs,
                            optimizer_versions=model.rollout_buffer.optimizer_versions,
                            reset_versions=model.rollout_buffer.reset_versions)
        report = {"source_seed": seed, "method": method, "online_steps": stream.total_steps,
                  "continuation_seed": continuation_seed, "online_cumulative_return": stream.cumulative_return,
                  "online_elapsed_s": elapsed, "optimizer_steps_added": optimizer.step_count,
                  "updated_residual_coordinates": updated, "residual_optimizer_dtypes": dtypes,
                  "base_actor_unchanged": True, "representation_after": storage_metrics(core, optimizer),
                  "peak_accounted_state_bytes": callback.peak, "max_sampled_process_rss_bytes": callback.rss_peak,
                  "bundle_bytes": (output / "agent.pt").stat().st_size,
                  "scope": "fixed-support known-family repair; cold Adam, no Part C"}
        vec.close()
    write_json(output / "summary.json", report)
    return report


def run_suite(config, output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing comparison has a different protocol; use a new directory")
    write_json(config_path, config)
    builds, calibrations, before, after, training, errors = [], [], [], [], [], []
    source_hashes = {}
    for seed in config["source_seeds"]:
        source = Path(config["source_root"]) / f"dense_compact_s{seed}" / "agent.pt"
        source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        restored = load_bundle(source)
        reference, router_snapshot = restored["policy"], restored["router"]
        del restored
        calibration, cal_report = replay_calibration(seed, config["calibration_states_per_route"], router_snapshot)
        calibrations.append(cal_report)
        np.savez_compressed(output / f"calibration_s{seed}.npz", observations=calibration.numpy())
        for method in config["methods"]:
            try:
                build_path = output / "bundles" / f"{method}_s{seed}" / "construction.json"
                build = (json.loads(build_path.read_text(encoding="utf-8")) if build_path.exists()
                         else construct(seed, method, reference, router_snapshot, calibration, config, output))
                builds.append(build)
                bundle = build_path.parent / "agent.pt"
                for context in config["contexts"]:
                    directory = output / "before" / f"{method}_s{seed}_{context}"
                    before.append(json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                                  if (directory / "summary.json").exists()
                                  else evaluate(bundle, directory, config, seed, method, context, "before"))
                directory = output / "continuation" / f"{method}_s{seed}"
                training.append(json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                                if (directory / "summary.json").exists()
                                else continue_learning(bundle, directory, config, seed, method))
                for context in config["contexts"]:
                    directory = output / "after" / f"{method}_s{seed}_{context}"
                    after.append(json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                                 if (directory / "summary.json").exists()
                                 else evaluate(output / "continuation" / f"{method}_s{seed}" / "agent.pt",
                                               directory, config, seed, method, context, "after"))
                print(json.dumps({"source_seed": seed, "method": method,
                                  "training_return": training[-1]["online_cumulative_return"],
                                  "adaptive_bytes_after": training[-1]["representation_after"]["adaptive_tensor_bytes"]}), flush=True)
            except Exception as error:
                record = {"source_seed": seed, "method": method, "error": repr(error)}
                errors.append(record)
                write_json(output / f"error_{method}_s{seed}.json", record)
                print(json.dumps(record), flush=True)
            result = {"scope": config["scope"], "config": config, "constructions": builds,
                      "calibrations": calibrations, "before": before, "training": training, "after": after,
                      "errors": errors, "source_sha256": source_hashes,
                      "total_interactions": sum(r["online_steps"] for r in before + training + after),
                      "complete": len(after) == len(config["methods"]) * len(config["source_seeds"]) * len(config["contexts"]) and not errors}
            write_json(output / "summary.json", result)
        del reference, calibration, router_snapshot
        gc.collect()
    return result

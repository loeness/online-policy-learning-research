"""Reproducible stage A/B pilot built on Stable-Baselines3 PPO."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from .envs import NavigationEnv, contexts, expert_distribution
from .policy import TreePolicy, OBS_DIM, MAX_NODES, capacity
from .routing import HistoryRouter

SEQUENCE = ["cw", "ccw", "cw", "cw_slip", "normal", "ccw_slip", "cw", "ccw"]
FIELDS = ["seed", "stage", "method", "episode", "global_steps", "context", "bout", "visit",
          "return", "success", "steps", "executed_node", "detected_node", "route_confidence",
          "update_count", "elapsed_s", "policy_bytes", "agent_state_bytes", "rollout_bytes",
          "is_eval", "mixed_context", "reset_count", "route_accuracy", "route_coverage",
          "cumulative_return", "reward_per_step"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def make_model(env, config):
    return PPO(TreePolicy, env, learning_rate=config.get("learning_rate", .003), n_steps=256,
               batch_size=64, n_epochs=4, gamma=.98, gae_lambda=.95,
               ent_coef=config.get("ent_coef", .04), clip_range=.2,
               target_kl=.06, policy_kwargs={"max_nodes": MAX_NODES},
               rollout_buffer_class=VersionedRolloutBuffer,
               seed=config.get("seed", 0), device="cpu", verbose=0)


class VersionedRolloutBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        self.optimizer_versions = np.zeros((self.buffer_size, self.n_envs), np.uint32)
        self.reset_versions = np.zeros((self.buffer_size, self.n_envs), np.uint32)


class BlankEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(-2., float(MAX_NODES), shape=(OBS_DIM,), dtype=np.float32)

    def augment(self, obs):
        return np.concatenate([obs, np.zeros(OBS_DIM - 4, np.float32)])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.augment(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self.augment(obs), reward, term, trunc, info


def generate_dataset(path, transitions=20000, seed=1001):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    env = NavigationEnv()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    ep = 0
    while len(rows) < transitions:
        targets = expert_distribution(obs)
        action = int(rng.choice(4, p=targets))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rows.append((obs.copy(), action, reward, next_obs.copy(), terminated or truncated, ep, targets))
        obs = next_obs
        if terminated or truncated:
            ep += 1
            obs, _ = env.reset()
    # Discounted returns on recorded trajectories, without querying counterfactual transitions.
    returns = np.zeros(len(rows), np.float32)
    total = 0.
    for i in range(len(rows) - 1, -1, -1):
        total = rows[i][2] + (.98 * total if not rows[i][4] else 0.)
        returns[i] = total
    states = np.asarray([r[0] for r in rows], np.float32)
    next_states = np.asarray([r[3] for r in rows], np.float32)
    actions = np.asarray([r[1] for r in rows], np.int64)
    responses = np.zeros((4, 2), np.float32)
    for a in range(4):
        movement = next_states[actions == a, :2] - states[actions == a, :2]
        valid = np.linalg.norm(movement, axis=1) > 1e-6
        responses[a] = movement[valid].mean(0)
    np.savez_compressed(path, observations=states, actions=actions,
                        next_observations=next_states, rewards=np.asarray([r[2] for r in rows], np.float32),
                        dones=np.asarray([r[4] for r in rows]), episodes=np.asarray([r[5] for r in rows]),
                        targets=np.asarray([r[6] for r in rows], np.float32), returns=returns,
                        source_responses=responses)
    metadata = {"seed": seed, "transitions": len(rows), "episodes": ep,
                "source": "recorded source-environment shortest-path expert interactions",
                "context": "normal only; deployment contexts excluded",
                "elapsed_s": time.perf_counter() - start,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(path.with_suffix(".json"), metadata)
    env.close()
    return metadata


def evaluate_fixed(model, context, node=0, seed=2001, episodes=64, deterministic=True):
    env = NavigationEnv(contexts()[context])
    rows = []
    steps = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.
        done = False
        while not done:
            augmented = np.concatenate([obs, np.zeros(8, np.float32), [node]]).astype(np.float32)
            action, _ = model.predict(augmented, deterministic=deterministic)
            obs, reward, term, trunc, info = env.step(int(action))
            total += reward
            steps += 1
            done = term or trunc
        rows.append((total, info["success"]))
    env.close()
    return {"mean_return": float(np.mean([x[0] for x in rows])),
            "success_rate": float(np.mean([x[1] for x in rows])), "episodes": episodes, "steps": steps}


def pretrain(dataset_path, checkpoint_path, config):
    start = time.perf_counter()
    seed_everything(0)
    data = np.load(dataset_path)
    observations = np.concatenate([data["observations"], np.zeros((len(data["observations"]), 9), np.float32)], axis=1)
    train_mask = data["episodes"] % 10 != 0
    train_ids = np.flatnonzero(train_mask)
    valid_ids = np.flatnonzero(~train_mask)
    model = make_model(BlankEnv(NavigationEnv()), {**config, "seed": 0})
    optimizer = torch.optim.Adam([p for n, p in model.policy.named_parameters() if not n.startswith("residuals")], lr=.002)
    rng = np.random.default_rng(7001)
    for step in range(config.get("bc_updates", 1500)):
        ids = rng.choice(train_ids, size=512, replace=True)
        obs = torch.as_tensor(observations[ids])
        targets = torch.as_tensor(data["targets"][ids])
        logits = model.policy.get_distribution(obs).distribution.logits
        actor_loss = -(targets * logits).sum(1).mean()
        value = model.policy.predict_values(obs).flatten()
        value_loss = (value - torch.as_tensor(data["returns"][ids])).square().mean()
        optimizer.zero_grad()
        (actor_loss + .25 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.)
        optimizer.step()
    with torch.no_grad():
        valid = torch.as_tensor(observations[valid_ids])
        logits = model.policy.get_distribution(valid).distribution.logits
        val_loss = float(-(torch.as_tensor(data["targets"][valid_ids]) * logits).sum(1).mean())
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.policy.state_dict(), checkpoint_path)
    metrics = {"source": evaluate_fixed(model, "normal", seed=2001),
               "unseen_cw_before_online": evaluate_fixed(model, "cw", seed=3001),
               "unseen_ccw_before_online": evaluate_fixed(model, "ccw", seed=4001),
               "validation_cross_entropy": val_loss, "elapsed_s": time.perf_counter() - start,
               "dataset_transitions": int(len(observations)), "train_transitions": int(len(train_ids)),
               "validation_transitions": int(len(valid_ids)), "bc_updates": config.get("bc_updates", 1500),
               "validation_split": "whole trajectory: episode ID mod 10",
               "weights_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()}
    write_json(Path(checkpoint_path).with_suffix(".json"), metrics)
    model.env.close()
    return metrics


class StreamEnv(gym.Wrapper):
    """World scheduling and logging are separated from agent-visible features."""
    def __init__(self, stage, method, seed, router, config, writer, sequence=None):
        super().__init__(NavigationEnv())
        self.observation_space = spaces.Box(-2., float(MAX_NODES), shape=(OBS_DIM,), dtype=np.float32)
        self.stage, self.method, self.seed = stage, method, seed
        self.router = router
        self.config = config
        self.sequence = sequence or (["cw"] if stage == "A" else SEQUENCE)
        self.bout_steps = config.get("bout_steps", 4096)
        self.writer = writer
        self.total_steps = 0
        self.cumulative_return = 0.
        self.episode = 0
        self.model = None
        self.before = None
        self.ep_return = 0.
        self.ep_steps = 0
        self.ep_contexts = []
        self.ep_nodes = []
        self.ep_confidences = []
        self.ep_route_correct = 0
        self.ep_route_covered = 0
        self.route_correct = 0
        self.route_covered = 0
        self.reset_count = 0
        self.capacity_info = {"policy_bytes": 0, "agent_state_bytes": 0, "rollout_bytes": 0}
        self.start_time = time.perf_counter()
        self.first_reset = True
        self.last_executed = 0

    def world_bout(self):
        return min(self.total_steps // self.bout_steps, len(self.sequence) - 1)

    def world_context(self):
        return self.sequence[self.world_bout()]

    def choose_route(self):
        if self.method in ("full", "frozen"):
            return 0
        if self.method == "global_residual":
            return 1
        if self.method == "oracle_keep":
            # Explicit diagnostic, never used by a formal history method.
            rotation = contexts()[self.world_context()].rotation
            return {0: 0, 1: 1, -1: 2}[rotation]
        node = self.router.route()
        if self.method == "history_reset" and self.router.pending_reset is not None:
            entered = int(self.router.pending_reset)
            if entered != 0 and self.model is not None:
                self.model.policy.zero_node(entered)
                self.reset_count += 1
            self.router.pending_reset = None
        return node

    def augment(self, obs):
        node = self.choose_route()
        self.last_executed = node
        return np.concatenate([obs, self.router.summary(), [node]]).astype(np.float32)

    def reset(self, **kwargs):
        if self.first_reset:
            kwargs["seed"] = self.seed * 10000 + 17
            self.first_reset = False
        self.env.set_context(contexts()[self.world_context()])
        obs, info = self.env.reset(**kwargs)
        self.before = obs
        self.ep_return, self.ep_steps = 0., 0
        self.ep_contexts, self.ep_nodes, self.ep_confidences = [], [], []
        self.ep_route_correct, self.ep_route_covered = 0, 0
        return self.augment(obs), info

    def step(self, action):
        context = self.world_context()
        bout = self.world_bout()
        self.env.set_context(contexts()[context])
        executed = self.last_executed
        self.last_action_optimizer_version = getattr(self.model.policy.optimizer, "step_count", 0) if self.model is not None else 0
        self.last_action_reset_version = self.reset_count
        confidence_before = self.router.confidence
        if self.method.startswith("history"):
            p = self.router.prototypes[executed]
            family = 0 if p[0] > .55 else (1 if p[1] > .55 else (-1 if p[1] < -.55 else 99))
            correct = family == contexts()[context].rotation and confidence_before >= .35
            self.route_correct += int(correct)
            self.route_covered += int(confidence_before >= .35)
            self.ep_route_correct += int(correct)
            self.ep_route_covered += int(confidence_before >= .35)
        obs, reward, term, trunc, info = self.env.step(int(action))
        self.router.observe(self.before, int(action), obs)
        self.before = obs
        self.total_steps += 1
        self.cumulative_return += reward
        self.ep_return += reward
        self.ep_steps += 1
        self.ep_contexts.append(context)
        self.ep_nodes.append(executed)
        self.ep_confidences.append(confidence_before)
        if term or trunc:
            majority_context = max(set(self.ep_contexts), key=self.ep_contexts.count)
            self.episode += 1
            row = {"seed": self.seed, "stage": self.stage, "method": self.method,
                   "episode": self.episode, "global_steps": self.total_steps, "context": majority_context,
                   "bout": bout, "visit": self.sequence[:bout + 1].count(majority_context),
                   "return": self.ep_return, "success": int(info["success"]), "steps": self.ep_steps,
                   "executed_node": max(set(self.ep_nodes), key=self.ep_nodes.count),
                   "detected_node": self.router.active,
                   "route_confidence": float(np.mean(self.ep_confidences)),
                   "update_count": getattr(self.model.policy.optimizer, "step_count", 0) if self.model is not None else 0,
                   "elapsed_s": time.perf_counter() - self.start_time,
                   "is_eval": False, "mixed_context": len(set(self.ep_contexts)) > 1,
                   "reset_count": self.reset_count,
                   "route_accuracy": self.ep_route_correct / max(1, self.ep_steps),
                   "route_coverage": self.ep_route_covered / max(1, self.ep_steps),
                   "cumulative_return": self.cumulative_return,
                   "reward_per_step": self.cumulative_return / max(1, self.total_steps),
                   **{k: self.capacity_info.get(k, 0) for k in ("policy_bytes", "agent_state_bytes", "rollout_bytes")}}
            self.writer.writerow(row)
        return self.augment(obs), reward, term, trunc, info


class BudgetCallback(BaseCallback):
    def __init__(self, stream, capacity_writer, budget):
        super().__init__()
        self.stream, self.writer, self.budget = stream, capacity_writer, budget
        self.peak = 0
        self.rss_peak = 0
        self.process = psutil.Process()

    def sample_capacity(self):
        values = capacity(self.model, self.stream.router)
        self.stream.capacity_info = values
        self.peak = max(self.peak, values["agent_state_bytes"])
        self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
        self.writer.writerow({"global_steps": self.stream.total_steps, **values,
                              "rss_bytes": self.rss_peak})
        if values["agent_state_bytes"] > self.budget:
            raise RuntimeError(f"Agent payload plus fixed rollout buffer exceeds budget: {values['agent_state_bytes']} > {self.budget}")

    def _on_step(self):
        position = self.model.rollout_buffer.pos
        self.model.rollout_buffer.optimizer_versions[position, 0] = self.stream.last_action_optimizer_version
        self.model.rollout_buffer.reset_versions[position, 0] = self.stream.last_action_reset_version
        return True

    def _on_training_start(self):
        self.sample_capacity()

    def _on_rollout_start(self):
        self.sample_capacity()

    def _on_training_end(self):
        self.sample_capacity()


def run_one(output_dir, stage, method, seed, dataset_path, checkpoint_path, config):
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset_path)
    router = HistoryRouter(data["source_responses"], window=config.get("route_window", 16),
                           novelty_threshold=config.get("novelty_threshold", .3))
    if not method.startswith("history"):
        router.max_nodes = 1
    sequence = ["cw"] if stage == "A" else SEQUENCE
    steps = config.get("a_steps", 16384) if stage == "A" else config.get("bout_steps", 4096) * len(sequence)
    if steps % 256:
        raise ValueError("Training interaction budget must be a multiple of the fixed 256-step rollout")
    run_config = {**config, "stage": stage, "method": method, "seed": seed, "sequence": sequence,
                  "steps": steps, "dataset": str(dataset_path), "checkpoint": str(checkpoint_path),
                  "oracle": method == "oracle_keep", "hidden_labels_agent_visible": False,
                  "phase_C_shared_consolidation": False,
                  "gain_gate": "new residual uses are experimental online trials; no certified improvement gate in AB",
                  "switch_rule": "fixed external interaction-step clock; changes may occur inside episodes"}
    write_json(output_dir / "config.json", run_config)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as episodes_file, \
         (output_dir / "capacity.csv").open("w", newline="", encoding="utf-8") as capacity_file:
        writer = csv.DictWriter(episodes_file, fieldnames=FIELDS)
        writer.writeheader()
        cap_fields = ["global_steps", "policy_bytes", "optimizer_bytes", "router_bytes", "rollout_bytes",
                      "serialized_core_bytes", "agent_state_bytes", "rss_bytes"]
        cap_writer = csv.DictWriter(capacity_file, fieldnames=cap_fields)
        cap_writer.writeheader()
        stream = StreamEnv(stage, method, seed, router, config, writer, sequence)
        vec = DummyVecEnv([lambda: stream])
        model = make_model(vec, {**config, "seed": seed})
        model.policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        model.policy.configure_learning(method, config.get("learning_rate", .003))
        stream.model = model
        callback = BudgetCallback(stream, cap_writer, config.get("agent_budget_bytes", 262144))
        callback.init_callback(model)
        start = time.perf_counter()
        if method == "frozen":
            callback.sample_capacity()
            obs = vec.reset()
            for t in range(steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, _, _, _ = vec.step(action)
                if (t + 1) % 256 == 0:
                    callback.sample_capacity()
        else:
            model.learn(total_timesteps=steps, callback=callback)
        elapsed = time.perf_counter() - start
        deployed_steps = stream.total_steps
        eval_start = time.perf_counter()
        eval_metrics = {}
        if stage == "A":
            node = 1 if method == "global_residual" else 0
            eval_metrics["shift_cw"] = evaluate_fixed(model, "cw", node=node, seed=seed * 10000 + 5001)
            eval_metrics["source_after_shift"] = evaluate_fixed(model, "normal", node=node, seed=seed * 10000 + 6001)
        model.save(output_dir / "agent.zip")
        if method != "frozen":
            np.savez_compressed(output_dir / "last_rollout_audit.npz",
                                observations=model.rollout_buffer.observations,
                                actions=model.rollout_buffer.actions,
                                log_probs=model.rollout_buffer.log_probs,
                                optimizer_versions=model.rollout_buffer.optimizer_versions,
                                reset_versions=model.rollout_buffer.reset_versions)
        write_json(output_dir / "router.json", router.snapshot())
        summary = {"stage": stage, "method": method, "seed": seed, "online_steps": deployed_steps,
                   "online_elapsed_s": elapsed, "evaluation_steps": sum(v["steps"] for v in eval_metrics.values()),
                   "evaluation_elapsed_s": time.perf_counter() - eval_start,
                   "online_cumulative_return": stream.cumulative_return,
                   "online_reward_per_step": stream.cumulative_return / max(1, deployed_steps),
                   "evaluation": eval_metrics, "peak_agent_state_bytes": callback.peak,
                   "peak_process_rss_bytes": callback.rss_peak, "budget_bytes": callback.budget,
                   "updates": getattr(model.policy.optimizer, "step_count", 0),
                   "ppo_epochs": model._n_updates, "created_nodes": router.creation_count,
                   "node_count": len(router.prototypes), "policy_resets": stream.reset_count,
                   "route_accuracy": stream.route_correct / max(1, deployed_steps),
                   "route_coverage": stream.route_covered / max(1, deployed_steps),
                   "limitations": ["synthetic globally rotated actuators and slip",
                                   "response dot/cross summary assumes planar action-response structure",
                                   "no shared parent promotion/compression evaluated in AB",
                                   "no calibrated policy-improvement safety gate",
                                   "payload+rollout hard bound; process RSS reported separately",
                                   "oracle_keep is diagnostic only"]}
        write_json(output_dir / "summary.json", summary)
        vec.close()
    return summary


def read_episodes(path):
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def run_suite(root, dataset_path, checkpoint_path, config, seeds=(11, 23, 37)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for stage, methods in [("A", ["frozen", "full", "global_residual"]),
                           ("B", ["frozen", "global_residual", "history_keep", "history_reset", "oracle_keep"])]:
        for seed in seeds:
            for method in methods:
                print(f"RUN {stage} seed={seed} method={method}", flush=True)
                result = run_one(root / f"{stage}_{method}_s{seed}", stage, method, seed,
                                 dataset_path, checkpoint_path, config)
                results.append(result)
                print(f"DONE {stage} {method}: {result['online_steps']} steps, {result['online_elapsed_s']:.1f}s, "
                      f"nodes={result['node_count']}, eval={result['evaluation']}", flush=True)
                import gc
                gc.collect()
    pretrain_metrics = json.loads(Path(checkpoint_path).with_suffix(".json").read_text(encoding="utf-8"))
    a_global = [r["evaluation"]["shift_cw"]["success_rate"] for r in results if r["stage"] == "A" and r["method"] == "global_residual"]
    a_frozen = [r["evaluation"]["shift_cw"]["success_rate"] for r in results if r["stage"] == "A" and r["method"] == "frozen"]
    improvement = float(np.mean(a_global) - np.mean(a_frozen))
    gate = {"source_success_minimum": .9, "online_success_improvement_minimum": .2,
            "source_success": pretrain_metrics["source"]["success_rate"], "online_improvement": improvement,
            "passed": pretrain_metrics["source"]["success_rate"] >= .9 and improvement >= .2,
            "interpretation": "pilot gate, not a statistical significance claim"}
    summary = {"pretrain": pretrain_metrics, "stage_A_gate": gate, "runs": results,
               "seeds": list(seeds), "config": config, "scope": "stage A/B mechanism pilot"}
    write_json(root / "summary.json", summary)
    write_json(root / "config.json", config)
    return summary

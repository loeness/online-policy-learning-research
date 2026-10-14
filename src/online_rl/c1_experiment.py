"""C1 fixed-structure mechanism study; oracle and calibrated-history are distinct."""
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .c1_envs import C1NavigationEnv, c1_contexts, context_index
from .c1_policy import C1ResidualPolicy
from .c1_routing import C1HistoryRouter
from .compression import export_payload
from .compression_experiment import PackedSB3Adapter, actor_hash, storage_metrics
from .experiment import BlankEnv, make_model, seed_everything, write_json
from .envs import NavigationEnv
from .policy import tensor_bytes

NAMES = tuple(c1_contexts())
EP_FIELDS = ["step", "episode", "context", "mixed_context", "return", "success", "length"]
STEP_FIELDS = ["step", "context", "route", "confidence", "reward", "action", "x", "y",
               "gx", "gy", "next_x", "next_y", "done", "success"]


def augment(obs, route, features=None):
    return np.concatenate([obs, np.zeros(8) if features is None else features, [route]]).astype(np.float32)


def foundation(config):
    base = make_model(DummyVecEnv([lambda: BlankEnv(NavigationEnv())]), config)
    base.policy.load_state_dict(torch.load(config["foundation"], map_location="cpu", weights_only=True))
    return base


def fresh_core(config, seed, method, tier):
    seed_everything(seed)
    base = foundation({**config, "seed": seed})
    core = C1ResidualPolicy.from_reference(base.policy, method=method,
                                          leaf_rank=tier.get("leaf_rank", 2),
                                          dictionary_rank=tier.get("dictionary_rank", 12),
                                          exploration_epsilon=config["exploration_epsilon"],
                                          temperature=config.get("temperature", 1.))
    base.env.close()
    optimizer = core.make_optimizer(learning_rate=config["learning_rate"], transfer=False)
    return core, optimizer


def new_router(config):
    return C1HistoryRouter(window=config["route_window"], confidence_threshold=config["route_confidence"])


def save_bundle(path, core, optimizer, router, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_payload(core, optimizer, router.snapshot(), metadata), path)
    return path.stat().st_size


def load_bundle(path):
    restored = C1ResidualPolicy.from_payload(torch.load(path, map_location="cpu", weights_only=False))
    return restored["policy"], restored["optimizer"], C1HistoryRouter.from_snapshot(restored["router"])


class C1Stream(gym.Wrapper):
    """Observer labels do not enter history-router observations or updates."""
    def __init__(self, schedule, seed, router, mode="oracle", steps_writer=None, episodes_writer=None):
        super().__init__(C1NavigationEnv())
        self.observation_space = spaces.Box(-2., 9., shape=(13,), dtype=np.float32)
        self.schedule = schedule
        self.seed_value, self.router, self.mode = seed, router, mode
        self.steps_writer, self.episodes_writer = steps_writer, episodes_writer
        self.total_steps, self.cumulative_return, self.episode = 0, 0., 0
        self.first_reset, self.last_route = True, 0
        self.ep_reward, self.ep_length, self.ep_contexts = 0., 0, set()
        self.model = None
        self.inference_route_s = 0.
        self.exact_correct, self.family_correct, self.parent_count, self.root_count = 0, 0, 0, 0
        self.switch_delays = []
        self.previous_context, self.pending_switch = None, 0

    def world_context(self):
        elapsed = 0
        for name, steps in self.schedule:
            elapsed += steps
            if self.total_steps < elapsed:
                return name
        return self.schedule[-1][0]

    def _augmented(self, obs):
        start = time.perf_counter()
        route = context_index(self.world_context()) + 1 if self.mode == "oracle" else self.router.route(obs)
        self.last_route = int(route)
        features = np.zeros(8) if self.mode == "oracle" else self.router.features()
        self.inference_route_s += time.perf_counter() - start
        return augment(obs, route, features)

    def reset(self, **kwargs):
        if self.first_reset:
            kwargs["seed"] = self.seed_value
            self.first_reset = False
        self.env.set_context(c1_contexts()[self.world_context()])
        self.before, info = self.env.reset(**kwargs)
        self.ep_reward, self.ep_length, self.ep_contexts = 0., 0, set()
        return self._augmented(self.before), info

    def step(self, action):
        name = self.world_context()
        self.env.set_context(c1_contexts()[name])
        route = self.last_route
        confidence_before = float(self.router.confidence) if self.mode != "oracle" else 1.
        if name != self.previous_context:
            self.pending_switch = self.total_steps
            self.current_switch = {"context": name, "switch_step": self.total_steps,
                                   "startup": self.previous_context is None,
                                   "first_correct_delay": None, "observed_steps": 0}
            self.switch_delays.append(self.current_switch)
            self.previous_context = name
            self.waiting = True
        correct = route == context_index(name) + 1
        family = (route in (1, 2, 3, 7) and name.startswith("A")) or (route in (4, 5, 6, 8) and name.startswith("B"))
        self.exact_correct += int(correct)
        self.family_correct += int(family)
        self.parent_count += int(route in (7, 8))
        self.root_count += int(route == 0)
        if self.waiting and correct:
            self.current_switch["first_correct_delay"] = self.total_steps - self.pending_switch
            self.waiting = False
        self.current_switch["observed_steps"] += 1
        nxt, reward, term, trunc, info = self.env.step(int(action))
        start = time.perf_counter()
        if self.mode == "oracle":
            self.router.observe_calibration(context_index(name) + 1, self.before, int(action), nxt)
        else:
            self.router.observe(self.before, int(action), nxt)
        self.inference_route_s += time.perf_counter() - start
        self.total_steps += 1
        self.cumulative_return += float(reward)
        self.ep_reward += float(reward)
        self.ep_length += 1
        self.ep_contexts.add(name)
        if self.steps_writer is not None:
            self.steps_writer.writerow(dict(step=self.total_steps, context=name, route=route,
                confidence=confidence_before, reward=reward,
                action=int(action), x=float(self.before[0]), y=float(self.before[1]),
                gx=float(self.before[2]), gy=float(self.before[3]), next_x=float(nxt[0]), next_y=float(nxt[1]),
                done=int(term or trunc), success=int(info["success"]) if term or trunc else ""))
        self.before = nxt
        if term or trunc:
            self.episode += 1
            if self.episodes_writer is not None:
                self.episodes_writer.writerow({"step": self.total_steps, "episode": self.episode, "context": name,
                    "mixed_context": int(len(self.ep_contexts) > 1), "return": self.ep_reward,
                    "success": int(info["success"]), "length": self.ep_length})
        return self._augmented(nxt), reward, term, trunc, info


class CapacityCallback(BaseCallback):
    def __init__(self, stream, writer, budget, adaptive_budget=None):
        super().__init__()
        self.stream, self.writer, self.budget = stream, writer, budget
        self.adaptive_budget = adaptive_budget
        self.peak, self.peak_core, self.peak_rollout, self.rss_peak, self.peak_explicit = 0, 0, 0, 0, 0

    def measure(self):
        core, optimizer = self.model.policy.core, self.model.policy.optimizer
        adaptive = storage_metrics(core, optimizer)["adaptive_tensor_bytes"]
        if self.adaptive_budget and adaptive > self.adaptive_budget:
            raise RuntimeError(f"C1 adaptive budget exceeded: {adaptive}>{self.adaptive_budget}")
        payload = export_payload(core, optimizer, self.stream.router.snapshot(),
            {"last_obs": self.model._last_obs, "last_episode_starts": self.model._last_episode_starts})
        blob = io.BytesIO()
        torch.save(payload, blob)
        core_bytes = blob.tell()
        rollout = sum(x.nbytes for x in vars(self.model.rollout_buffer).values() if isinstance(x, np.ndarray))
        self.peak = max(self.peak, core_bytes + rollout)
        self.peak_core, self.peak_rollout = max(self.peak_core, core_bytes), max(self.peak_rollout, rollout)
        self.rss_peak = max(self.rss_peak, psutil.Process().memory_info().rss)
        self.peak_explicit = max(self.peak_explicit, tensor_bytes(core.state_dict()) +
            tensor_bytes(optimizer.state_dict()) + tensor_bytes([p.grad for p in core.parameters()]) + rollout)
        if self.writer:
            self.writer.writerow({"step": self.stream.total_steps, "core_bytes": core_bytes,
                "rollout_bytes": rollout, "agent_bytes": core_bytes + rollout, "rss_bytes": self.rss_peak})
        if self.budget and self.peak > self.budget:
            raise RuntimeError(f"C1 budget exceeded: {self.peak}>{self.budget}")

    def _on_step(self):
        return True

    def _on_training_start(self):
        self.measure()

    def _on_rollout_start(self):
        self.measure()

    def _on_training_end(self):
        self.measure()


def train(core, optimizer, router, config, schedule, seed, output, budget=None, mode="oracle"):
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    before_hash = actor_hash(core)
    initial_updates = optimizer.step_count
    bank_hash = hashlib.sha256(router.direction_counts.tobytes() + router.outcome_counts.tobytes()).hexdigest()
    calibration_start = int(router.outcome_counts.sum())
    with (output / "steps.csv").open("w", newline="", encoding="utf-8") as sf, \
         (output / "episodes.csv").open("w", newline="", encoding="utf-8") as ef, \
         (output / "capacity.csv").open("w", newline="", encoding="utf-8") as cf:
        sw, ew = csv.DictWriter(sf, fieldnames=STEP_FIELDS), csv.DictWriter(ef, fieldnames=EP_FIELDS)
        cw = csv.DictWriter(cf, fieldnames=["step", "core_bytes", "rollout_bytes", "agent_bytes", "rss_bytes"])
        sw.writeheader(); ew.writeheader(); cw.writeheader()
        stream = C1Stream(schedule, seed + 17, router, mode, sw, ew)
        vec = DummyVecEnv([lambda: stream])
        model = make_model(vec, {**config, "seed": seed})
        model.policy = PackedSB3Adapter(core, optimizer)
        stream.model = model
        callback = CapacityCallback(stream, cw, budget, config.get("adaptive_budget_bytes"))
        start = time.perf_counter()
        cpu_start = time.process_time()
        model.learn(sum(steps for _, steps in schedule), callback=callback)
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - cpu_start
        metrics = {"online_steps": stream.total_steps, "online_cumulative_return": stream.cumulative_return,
            "online_elapsed_s": elapsed, "route_elapsed_s": stream.inference_route_s,
            "online_process_cpu_s": cpu_elapsed,
            "optimizer_updates": optimizer.step_count - initial_updates, "mode": mode,
            "schedule": schedule, "base_actor_unchanged": actor_hash(core) == before_hash,
            "representation": storage_metrics(core, optimizer), "max_core_bytes": callback.peak_core,
            "rollout_bytes": callback.peak_rollout, "max_agent_bytes": callback.peak,
            "max_sampled_process_rss_bytes": callback.rss_peak,
            "explicit_live_tensor_bytes_lower_bound": callback.peak_explicit,
            "route_exact_accuracy": stream.exact_correct / stream.total_steps,
            "route_family_accuracy": stream.family_correct / stream.total_steps,
            "parent_fallback_fraction": stream.parent_count / stream.total_steps,
            "foundation_fallback_fraction": stream.root_count / stream.total_steps,
            "switch_first_correct": stream.switch_delays,
            "calibration_interactions_added": int(router.outcome_counts.sum()) - calibration_start,
            "unknown_switches_not_given_to_router": mode != "oracle"}
        if mode == "oracle" and metrics["calibration_interactions_added"] != stream.total_steps:
            raise RuntimeError("Oracle calibration must consume exactly the training interactions")
        if mode != "oracle" and hashlib.sha256(router.direction_counts.tobytes() + router.outcome_counts.tobytes()).hexdigest() != bank_hash:
            raise RuntimeError("Hidden-label inference altered the calibrated model bank")
        if not metrics["base_actor_unchanged"]:
            raise RuntimeError("Frozen C1 foundation changed")
        metrics["bundle_bytes"] = save_bundle(output / "agent.pt", core, optimizer, router,
                                                {"mode": mode, "seed": seed, "steps": stream.total_steps})
        write_json(output / "summary.json", metrics)
        vec.close()
        del model, callback
    gc.collect()
    return metrics


def evaluate(core, context, seed, steps, output, route=None):
    """No-gradient separate stream, same action RNG and reset seed across arms."""
    env = C1NavigationEnv(c1_contexts()[context])
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    route = context_index(context) + 1 if route is None else route
    total, successes, completed, episode_reward, episode_length = 0., 0, 0, 0., 0
    inference = 0.
    before_hash = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for v in core.state_dict().values())).hexdigest()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            it = time.perf_counter()
            p = core.get_distribution(torch.as_tensor(augment(obs, route)[None])).distribution.probs[0].numpy().astype(float)
            p /= p.sum()
            action = int(rng.choice(4, p=p))
            inference += time.perf_counter() - it
            obs, reward, term, trunc, info = env.step(action)
            total += float(reward)
            episode_reward += float(reward)
            episode_length += 1
            if term or trunc:
                completed += 1
                successes += int(info["success"])
                obs, _ = env.reset()
                episode_reward, episode_length = 0., 0
    after_hash = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for v in core.state_dict().values())).hexdigest()
    if before_hash != after_hash:
        raise RuntimeError("Evaluation changed parameters")
    result = {"context": context, "route": route, "seed": seed, "online_steps": steps,
              "online_cumulative_return": total, "completed_episodes": completed,
              "successful_episodes": successes, "success_rate": successes / max(1, completed),
              "unfinished_episode_steps": episode_length, "unfinished_episode_return": episode_reward,
              "inference_elapsed_s": inference, "elapsed_s": time.perf_counter() - start,
              "gradient_updates": 0, "parameter_hash_unchanged": before_hash == after_hash}
    result["routing_mode"] = "oracle_diagnostic"
    env.close()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    return result


def run_precheck(config, output):
    root = output / config.get("precheck_directory", "precheck")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    rows = []
    for name in NAMES:
        case = root / name
        if (case / "result.json").exists():
            rows.append(json.loads((case / "result.json").read_text(encoding="utf-8")))
            continue
        seed = config["validation_seed"]
        core, optimizer = fresh_core(config, seed, "dense", {})
        router = new_router(config)
        pre = evaluate(core, name, seed * 100 + context_index(name), config["eval_steps"], case / "before.json")
        learn = train(core, optimizer, router, config, [(name, config["precheck_steps"])], seed, case / "initial")
        after = evaluate(core, name, seed * 100 + context_index(name), config["eval_steps"], case / "after_initial.json")
        extension = None
        if after["success_rate"] < config["learnability_success_target"]:
            extension = train(core, optimizer, router, config, [(name, config["precheck_extension_steps"])], seed + 1, case / "extension")
            after = evaluate(core, name, seed * 100 + context_index(name), config["eval_steps"], case / "after_extension.json")
        row = {"context": name, "before": pre, "initial_training": learn, "extension_training": extension,
               "after": after, "passed": after["success_rate"] >= config["learnability_success_target"]}
        write_json(case / "result.json", row)
        rows.append(row)
        print(json.dumps({"precheck": name, "success": after["success_rate"], "passed": row["passed"]}), flush=True)
        del core, optimizer, router
        gc.collect()
    result = {"passed": all(r["passed"] for r in rows), "cases": rows,
              "total_interactions": sum(r["before"]["online_steps"] + r["after"]["online_steps"] +
                  r["initial_training"]["online_steps"] + (r["extension_training"]["online_steps"] if r["extension_training"] else 0) +
                  (config["eval_steps"] if r["extension_training"] else 0) for r in rows)}
    write_json(root / "summary.json", result)
    return {"passed": result["passed"], "total_interactions": result["total_interactions"]}


def projected_budgets(config, output):
    """Fully warmed layouts measured before official reward observations."""
    start = time.perf_counter()
    result = {}
    for label, tier in config["tiers"].items():
        arms = {}
        for method in ("flat", "tree", "dictionary"):
            core, opt = fresh_core(config, config["validation_seed"], method, tier)
            for p in opt.param_groups[0]["params"]:
                # Direct state allocation models all slots warm, without changing policy.
                opt.state[p] = {"step": torch.tensor(1.), "exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
            router = new_router(config)
            # Calibrated arrays and bounded history are replayed from actual validation.
            for name in NAMES:
                source = output / config.get("precheck_directory", "precheck") / name / "initial" / "steps.csv"
                with source.open(encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        obs = np.array([float(row[k]) for k in ("x", "y", "gx", "gy")], np.float32)
                        nxt = obs.copy(); nxt[:2] = [float(row["next_x"]), float(row["next_y"])]
                        router.observe_calibration(context_index(name) + 1, obs, int(row["action"]), nxt)
            validation = output / config.get("precheck_directory", "precheck") / "A0" / "initial" / "steps.csv"
            with validation.open(encoding="utf-8") as f:
                records = list(csv.DictReader(f))[-config["route_window"]:]
            for row in records:
                obs = np.array([float(row[k]) for k in ("x", "y", "gx", "gy")], np.float32)
                nxt = obs.copy(); nxt[:2] = [float(row["next_x"]), float(row["next_y"])]
                router.observe(obs, int(row["action"]), nxt)
            blob = io.BytesIO()
            torch.save(export_payload(core, opt, router.snapshot(), {"last_obs": np.zeros((1, 13), np.float32),
                       "last_episode_starts": np.zeros(1, bool)}), blob)
            # Matches all NumPy rollout arrays of VersionedRolloutBuffer, 256x1.
            base = make_model(DummyVecEnv([lambda: BlankEnv(NavigationEnv())]), config)
            rollout = sum(x.nbytes for x in vars(base.rollout_buffer).values() if isinstance(x, np.ndarray))
            base.env.close()
            arms[method] = {"fully_warmed_bundle_bytes": blob.tell(), "rollout_bytes": rollout,
                           "total": blob.tell() + rollout, "representation": storage_metrics(core, opt)}
            del core, opt, base
        quantum = config["budget_rounding_bytes"]
        cap = ((max(r["total"] for r in arms.values()) + config["budget_safety_bytes"] + quantum - 1) // quantum) * quantum
        result[label] = {"cap_bytes": cap,
                         "adaptive_cap_bytes": max(r["representation"]["adaptive_tensor_bytes"] for r in arms.values()),
                         "arms": arms, "derived_without_formal_rewards": True}
    write_json(output / "budget_preflight.json", result)
    write_json(output / "budget_preflight_cost.json", {"elapsed_s": time.perf_counter() - start,
                                                       "new_environment_interactions": 0,
                                                       "scope": "validation-log replay and storage measurement, discarded before formal runs"})
    return result


def eval_all(core, seed, config, directory, names=NAMES):
    return {name: evaluate(core, name, seed * 1000 + context_index(name), config["eval_steps"], directory / f"{name}.json") for name in names}


def formal_arm(config, output, seed, method, label, tier, cap):
    path = output / "formal" / label / method / f"s{seed}"
    path.mkdir(parents=True, exist_ok=True)
    if (path / "result.json").exists():
        return json.loads((path / "result.json").read_text(encoding="utf-8"))
    write_json(path / "config.json", {**config, "seed": seed, "method": method, "tier": label, "cap_bytes": cap})
    core, optimizer = fresh_core(config, seed, method, tier)
    router = new_router(config)
    seen = train(core, optimizer, router, config,
                 [(name, config["seen_bout_steps"]) for name in config["seen_sequence"]], seed, path / "seen", cap)
    seen_eval = eval_all(core, seed, config, path / "seen_eval", ("A0", "A1", "B0", "B1"))
    cold = {}
    for name in config["heldout_sequence"]:
        eval_seed = seed * 1000 + context_index(name)
        parent_id = 7 if name.startswith("A") else 8
        cold[name] = {"foundation": evaluate(core, name, eval_seed, config["cold_eval_steps"], path / "cold" / f"{name}_foundation.json", 0),
                      "shared_or_sibling_reuse": evaluate(core, name, eval_seed, config["cold_eval_steps"], path / "cold" / f"{name}_reuse.json", parent_id)}
    init_start = time.perf_counter()
    for name in config["heldout_sequence"]:
        core.initialize_heldout(context_index(name) + 1)
    initialization_s = time.perf_counter() - init_start
    # No optimizer moments pre-exist for held-out slots, so no stale SVD moments.
    heldout = train(core, optimizer, router, config,
                    [(name, config["heldout_bout_steps"]) for name in config["heldout_sequence"]], seed + 100, path / "heldout", cap)
    adapted_eval = eval_all(core, seed, config, path / "adapted_eval")
    revisit = train(core, optimizer, router, config,
                    [(name, config["revisit_bout_steps"]) for name in config["revisit_sequence"]], seed + 200, path / "revisit", cap)
    final_eval = eval_all(core, seed, config, path / "final_eval")
    # Real file restore plus recent-context reset before hidden-switch online continuation.
    del core, optimizer, router
    core, optimizer, router = load_bundle(path / "revisit" / "agent.pt")
    router.clear_history()
    history = train(core, optimizer, router, config,
                    [(name, config["history_bout_steps"]) for name in config["history_sequence"]], seed + 300, path / "history", cap, "calibrated_history")
    history_eval = eval_all(core, seed, config, path / "history_eval")
    result = {"seed": seed, "method": method, "tier": label, "cap_bytes": cap,
              "seen": seen, "seen_eval": seen_eval, "cold": cold,
              "heldout_initialization_s": initialization_s, "heldout": heldout,
              "adapted_eval": adapted_eval, "revisit": revisit, "final_eval": final_eval,
              "history": history, "history_eval": history_eval,
              "history_scope": "hidden-label continuation with oracle-calibrated empirical prototypes; fixed tree and groups",
              "train_interactions": seen["online_steps"] + heldout["online_steps"] + revisit["online_steps"] + history["online_steps"],
              "evaluation_interactions": config["eval_steps"] * (4 + 6 + 6 + 6) + config["cold_eval_steps"] * 4}
    write_json(path / "result.json", result)
    del core, optimizer, router
    gc.collect()
    return result


def run_formal(config, output):
    precheck = json.loads((output / config.get("precheck_directory", "precheck") / "summary.json").read_text(encoding="utf-8"))
    if not precheck["passed"]:
        raise RuntimeError("Learnability precheck failed; official ranking prohibited by protocol")
    write_json(output / "frozen_config.json", config)
    budgets = projected_budgets(config, output) if not (output / "budget_preflight.json").exists() else json.loads((output / "budget_preflight.json").read_text(encoding="utf-8"))
    rows = []
    for label, tier in config["tiers"].items():
        for seed in config["seeds"]:
            for method in ("flat", "tree", "dictionary"):
                row = formal_arm({**config, "adaptive_budget_bytes": budgets[label]["adaptive_cap_bytes"]},
                                 output, seed, method, label, tier, budgets[label]["cap_bytes"])
                rows.append(row)
                print(json.dumps({"tier": label, "seed": seed, "method": method,
                                  "cold_reward": {n: r["shared_or_sibling_reuse"]["online_cumulative_return"] for n, r in row["cold"].items()},
                                  "final_success": {n: r["success_rate"] for n, r in row["final_eval"].items()},
                                  "history_accuracy": row["history"]["route_exact_accuracy"]}), flush=True)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(config["foundation"]), Path(config["dataset"]), Path("configs/c1.json"), Path("c1_protocol.md"),
               *sorted(Path("src/online_rl").glob("c1_*.py"))]}
    result = {"complete": len(rows) == len(config["tiers"]) * len(config["seeds"]) * 3,
              "runs": rows, "budgets": budgets, "config": config,
              "total_training_interactions": sum(r["train_interactions"] for r in rows),
              "total_evaluation_interactions": sum(r["evaluation_interactions"] for r in rows),
              "selected_precheck_interactions": precheck["total_interactions"],
              "validation_interactions": json.loads((output / "validation_ledger.json").read_text(encoding="utf-8"))["total_interactions"],
              "input_sha256": hashes, "scope": "fixed structure, common frozen foundation, simulated interactions"}
    write_json(output / "summary.json", result)
    return {key: result[key] for key in ("complete", "total_training_interactions", "total_evaluation_interactions", "validation_interactions")}

"""Step 1 C1 model-controller supplements and matched hidden diagnostics."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import torch
from torch import nn

from .c1_envs import C1NavigationEnv, c1_contexts
from .c1_experiment import load_bundle
from .c1_routing import C1HistoryRouter
from .c1_strong_models import C1ModelController, load_source_prior
from .experiment import BlankEnv, make_model, seed_everything, write_json
from .envs import NavigationEnv
from stable_baselines3.common.vec_env import DummyVecEnv


ROOT = Path(__file__).resolve().parents[2]
STAGES = ("seen", "heldout", "revisit", "history")
STEP_FIELDS = [
    "step", "stage", "bout", "context", "action", "reward", "x", "y", "gx", "gy",
    "next_x", "next_y", "done", "success", "model_confidence", "route", "route_confidence",
]
EPISODE_FIELDS = [
    "step", "episode", "stage", "context", "mixed_context", "return", "success", "length",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def _actor_logits(policy: Any, obs: np.ndarray) -> torch.Tensor:
    """Extract raw frozen actor logits before temperature and epsilon."""

    tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None], dtype=torch.float32)
    features = policy.extract_features(tensor)
    latent = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent)[0]


class FrozenSourceActor(nn.Module):
    """The W0 actor weights needed for acting, without its PPO critic."""

    def __init__(self, source_policy: Any) -> None:
        super().__init__()
        self.policy_net = copy.deepcopy(source_policy.mlp_extractor.policy_net).cpu().eval()
        self.action_net = copy.deepcopy(source_policy.action_net).cpu().eval()

    @torch.no_grad()
    def logits(self, obs: np.ndarray) -> torch.Tensor:
        values = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None, :4], dtype=torch.float32)
        return self.action_net(self.policy_net(values))[0]


def _foundation_actor(config: dict[str, Any]) -> FrozenSourceActor:
    blank = BlankEnv(NavigationEnv())
    base = make_model(DummyVecEnv([lambda: blank]), config)
    state = torch.load(config["foundation"], map_location="cpu", weights_only=True)
    base.policy.load_state_dict(state)
    actor = FrozenSourceActor(base.policy)
    base.env.close()
    del base
    return actor


def _base_probabilities(logits: torch.Tensor, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    probs = torch.softmax(logits.detach().cpu().to(torch.float64) / float(temperature), dim=-1).numpy()
    return probs / probs.sum()


def _epsilon_mix(probabilities: Any, epsilon: float) -> np.ndarray:
    """Apply the common uniform action mixture exactly once."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("action probabilities must be a finite non-negative vector of length 4")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("action probabilities must have positive finite mass")
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0, 1)")
    normalized = values / total
    mixed = (1.0 - epsilon) * normalized + epsilon / 4.0
    return mixed / mixed.sum()


def _actor_only_state(actor: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in actor.state_dict().items()}


def _rng_state(action_rng: np.random.Generator, env: C1NavigationEnv) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "action_generator": copy.deepcopy(action_rng.bit_generator.state),
        "environment_generator": copy.deepcopy(env.np_random.bit_generator.state),
    }


def make_checkpoint(
    actor: nn.Module,
    controller: Any,
    action_rng: np.random.Generator,
    env: C1NavigationEnv,
    *,
    run_kind: str,
    seed: int,
    global_step: int,
    stage: str,
    observation: np.ndarray,
) -> dict[str, Any]:
    """Build a checkpoint containing only retained actor/controller state."""

    if controller is None:
        controller_state = None
    elif isinstance(controller, C1ModelController):
        controller_state = controller.snapshot()
    else:
        controller_state = controller.snapshot()
    return {
        "format_version": 1,
        "run_kind": run_kind,
        "seed": int(seed),
        "global_step": int(global_step),
        "stage": stage,
        "actor": _actor_only_state(actor),
        "controller": controller_state,
        "rng": _rng_state(action_rng, env),
        "runtime": {
            "observation": np.asarray(observation, dtype=np.float32).tolist(),
            "position": list(env._position),
            "goal": list(env._goal),
            "episode_length": int(env._episode_length),
            "has_reset": bool(env._has_reset),
        },
    }


def restore_action_generator(checkpoint: dict[str, Any]) -> np.random.Generator:
    """Restore the exact action sampler stream from a saved run checkpoint."""

    state = checkpoint.get("rng", {}).get("action_generator")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no action-generator state")
    bit_generator = np.random.PCG64()
    bit_generator.state = copy.deepcopy(state)
    return np.random.Generator(bit_generator)


def restore_controller(checkpoint: dict[str, Any]) -> Any:
    state = checkpoint.get("controller")
    if state is None:
        return None
    if state.get("version") == 1 and "outcome_counts" in state:
        return C1HistoryRouter.from_snapshot(state)
    if state.get("version") == C1ModelController._SNAPSHOT_VERSION:
        return C1ModelController.from_snapshot(state)
    raise ValueError("checkpoint has an unsupported controller snapshot")


def _schedule(base: dict[str, Any], stage_sizes: dict[str, int]) -> list[tuple[str, str, int]]:
    entries: list[tuple[str, str, int]] = []
    for stage in STAGES:
        seq_key = f"{stage}_sequence"
        if seq_key not in base:
            continue
        for bout, context in enumerate(base[seq_key]):
            entries.append((stage, str(context), int(stage_sizes[stage])))
    return entries


def _episode_stats() -> dict[str, Any]:
    return {"steps": 0, "return": 0.0, "episodes": 0, "successes": 0, "mixed_episodes": 0}


@torch.no_grad()
def _probabilities_for(
    actor: Any,
    controller: Any,
    router: C1HistoryRouter | None,
    obs: np.ndarray,
    temperature: float,
    epsilon: float,
) -> tuple[np.ndarray, int | None, float | None, float, float]:
    """Return sampled-action probabilities plus separated inference timings."""

    actor_start = time.perf_counter()
    if router is not None:
        route = router.route(obs)
        route_confidence = float(router.confidence)
        augmented = np.concatenate([obs, router.features(), [route]]).astype(np.float32)
        distribution = actor.get_distribution(torch.as_tensor(augmented[None], dtype=torch.float32))
        probabilities = distribution.distribution.probs[0].detach().cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        # The C1 residual policy already implements its predeclared epsilon.
        model_time = 0.0
    else:
        logits = actor.logits(obs)
        base = _base_probabilities(logits, temperature)
        actor_elapsed = time.perf_counter() - actor_start
        model_start = time.perf_counter()
        routed = base if controller is None else controller.action_probabilities(obs, base)
        model_time = time.perf_counter() - model_start
        probabilities = _epsilon_mix(routed, epsilon)
        return probabilities, None, None, actor_elapsed, model_time
    actor_elapsed = time.perf_counter() - actor_start
    return probabilities, route, route_confidence, actor_elapsed, model_time


def run_trajectory(
    *,
    actor: Any,
    controller: Any,
    router: C1HistoryRouter | None,
    schedule: list[tuple[str, str, int]],
    seed: int,
    output: Path,
    temperature: float = 2.0,
    epsilon: float = 0.05,
    run_kind: str,
    save_checkpoints: bool = True,
) -> dict[str, Any]:
    """Run one continuous hidden-context trajectory without label resets."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not schedule or any(length <= 0 for _, _, length in schedule):
        raise ValueError("schedule must contain positive-length context bouts")
    seed_everything(seed)
    action_rng = np.random.default_rng(seed + 1)
    env = C1NavigationEnv()
    obs, _ = env.reset(seed=seed + 17)
    process = psutil.Process()
    max_rss = process.memory_info().rss
    step_count = 0
    episode_count = 0
    cumulative_return = 0.0
    episode_return = 0.0
    episode_length = 0
    episode_contexts: set[str] = set()
    context_metrics: dict[str, dict[str, Any]] = {}
    stage_metrics: dict[str, dict[str, Any]] = {}
    switches: list[dict[str, Any]] = []
    current_switch: dict[str, Any] | None = None
    total_actor_inference = 0.0
    total_model_action = 0.0
    total_model_update = 0.0
    total_route = 0.0
    total_env_step = 0.0
    source_tree_hash = None
    serialized_peak = explicit_peak = 0
    if router is not None:
        source_tree_hash = hashlib.sha256(
            router.direction_counts.tobytes() + router.outcome_counts.tobytes()
        ).hexdigest()

    with (output / "steps.csv").open("w", newline="", encoding="utf-8") as step_file, \
         (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file:
        step_writer = csv.DictWriter(step_file, fieldnames=STEP_FIELDS)
        episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS)
        step_writer.writeheader()
        episode_writer.writeheader()
        for stage, context, bout_steps in schedule:
            prior_timers = (total_actor_inference, total_model_action, total_model_update, total_route, total_env_step)
            env.set_context(c1_contexts()[context])
            stage_row = stage_metrics.setdefault(stage, {
                "online_steps": 0, "online_cumulative_return": 0.0,
                "completed_episodes": 0, "successful_episodes": 0,
                "wall_elapsed_s": 0.0, "process_cpu_s": 0.0,
                "actor_inference_elapsed_s": 0.0, "model_action_elapsed_s": 0.0,
                "model_update_elapsed_s": 0.0, "router_elapsed_s": 0.0,
                "environment_step_elapsed_s": 0.0,
            })
            stage_wall = time.perf_counter()
            stage_cpu = time.process_time()
            for bout_step in range(bout_steps):
                # The schedule label is held by this observer harness and is
                # never passed into the model or neural router.
                if current_switch is None or current_switch["context"] != context or current_switch["stage"] != stage:
                    if current_switch is not None and current_switch["first_success_delay"] is None:
                        current_switch["censored_after_steps"] = step_count - current_switch["switch_step"]
                    current_switch = {
                        "switch_index": len(switches) + 1,
                        "stage": stage,
                        "context": context,
                        "switch_step": step_count,
                        "first_success_step": None,
                        "first_success_delay": None,
                        "censored_after_steps": None,
                    }
                    switches.append(current_switch)
                context_row = context_metrics.setdefault(context, _episode_stats())
                probs, route, route_conf, actor_elapsed, action_model_elapsed = _probabilities_for(
                    actor, controller, router, obs, temperature, epsilon
                )
                action = int(action_rng.choice(4, p=probs))
                env_start = time.perf_counter()
                next_obs, reward, terminated, truncated, info = env.step(action)
                env_elapsed = time.perf_counter() - env_start
                if controller is not None:
                    update_start = time.perf_counter()
                    # Only the observable transition crosses the controller API.
                    controller.observe(obs, action, next_obs)
                    update_elapsed = time.perf_counter() - update_start
                else:
                    update_elapsed = 0.0
                if router is not None:
                    route_start = time.perf_counter()
                    router.observe(obs, action, next_obs)
                    route_update_elapsed = time.perf_counter() - route_start
                else:
                    route_update_elapsed = 0.0

                step_count += 1
                episode_length += 1
                episode_return += float(reward)
                cumulative_return += float(reward)
                episode_contexts.add(context)
                context_row["steps"] += 1
                context_row["return"] += float(reward)
                stage_row["online_steps"] += 1
                stage_row["online_cumulative_return"] += float(reward)
                total_actor_inference += actor_elapsed
                total_model_action += action_model_elapsed
                total_model_update += update_elapsed
                total_route += route_update_elapsed
                total_env_step += env_elapsed
                done = bool(terminated or truncated)
                step_writer.writerow({
                    "step": step_count, "stage": stage, "bout": bout_step + 1,
                    "context": context, "action": action, "reward": float(reward),
                    "x": float(obs[0]), "y": float(obs[1]), "gx": float(obs[2]), "gy": float(obs[3]),
                    "next_x": float(next_obs[0]), "next_y": float(next_obs[1]),
                    "done": int(done), "success": int(info["success"]) if done else "",
                    "model_confidence": float(controller.confidence) if controller is not None else "",
                    "route": route if route is not None else "",
                    "route_confidence": route_conf if route_conf is not None else "",
                })
                obs = next_obs
                if done:
                    episode_count += 1
                    success = int(info["success"])
                    episode_writer.writerow({
                        "step": step_count, "episode": episode_count, "stage": stage,
                        "context": context, "mixed_context": int(len(episode_contexts) > 1),
                        "return": episode_return, "success": success, "length": episode_length,
                    })
                    context_row["episodes"] += 1
                    context_row["successes"] += success
                    context_row["mixed_episodes"] += int(len(episode_contexts) > 1)
                    stage_row["completed_episodes"] += 1
                    stage_row["successful_episodes"] += success
                    if success and current_switch is not None and current_switch["first_success_delay"] is None:
                        current_switch["first_success_step"] = step_count
                        current_switch["first_success_delay"] = step_count - current_switch["switch_step"]
                    episode_return = 0.0
                    episode_length = 0
                    episode_contexts.clear()
                    obs, _ = env.reset()
                if step_count % 256 == 0:
                    max_rss = max(max_rss, process.memory_info().rss)
            stage_row["wall_elapsed_s"] += time.perf_counter() - stage_wall
            stage_row["process_cpu_s"] += time.process_time() - stage_cpu
            for key, value in (
                ("actor_inference_elapsed_s", total_actor_inference),
                ("model_action_elapsed_s", total_model_action),
                ("model_update_elapsed_s", total_model_update),
                ("router_elapsed_s", total_route),
                ("environment_step_elapsed_s", total_env_step),
            ):
                # Stage totals are measured from cumulative timers at each
                # boundary, avoiding perturbing the sample path.
                offset = ("actor_inference_elapsed_s", "model_action_elapsed_s", "model_update_elapsed_s", "router_elapsed_s", "environment_step_elapsed_s").index(key)
                stage_row[key] = stage_row.get(key, 0.0) + value - prior_timers[offset]
            if save_checkpoints:
                payload = make_checkpoint(
                    actor, controller if controller is not None else router, action_rng, env,
                    run_kind=run_kind, seed=seed, global_step=step_count, stage=stage, observation=obs,
                )
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                serialized_peak = max(serialized_peak, buffer.tell())
                if serialized_peak > 188416:
                    raise RuntimeError("Model or frozen inference state exceeded the low complete-agent cap")
                owned = controller.memory_summary()["estimated_controller_bytes"] if controller is not None else len(json.dumps(router.snapshot()).encode()) if router is not None else 0
                explicit_peak = max(explicit_peak, sum(t.numel()*t.element_size() for t in actor.state_dict().values()) + owned)
                checkpoint_path = output / f"checkpoint_{stage}.pt"
                torch.save(payload, checkpoint_path)
                stage_row["checkpoint_bytes"] = checkpoint_path.stat().st_size
                stage_row["checkpoint_sha256"] = _sha256(checkpoint_path)
            print(json.dumps({
                "phase": "trajectory_stage", "run": run_kind, "seed": seed,
                "stage": stage, "steps": stage_row["online_steps"],
                "cumulative_return": round(stage_row["online_cumulative_return"], 2),
                "successes": stage_row["successful_episodes"],
                "controller_updates": int(getattr(controller, "statistical_update_count", 0)),
            }), flush=True)

    if current_switch is not None and current_switch["first_success_delay"] is None:
        current_switch["censored_after_steps"] = step_count - current_switch["switch_step"]
    env.close()
    for metrics in context_metrics.values():
        metrics["success_rate"] = metrics["successes"] / max(1, metrics["episodes"])
        metrics["mean_reward_per_step"] = metrics["return"] / max(1, metrics["steps"])
    for metrics in stage_metrics.values():
        metrics["success_rate"] = metrics["successful_episodes"] / max(1, metrics["completed_episodes"])
    router_hash_after = None
    if router is not None:
        router_hash_after = hashlib.sha256(
            router.direction_counts.tobytes() + router.outcome_counts.tobytes()
        ).hexdigest()
        if router_hash_after != source_tree_hash:
            raise RuntimeError("hidden deployment changed the calibrated transition bank")
    controller_summary = controller.memory_summary() if isinstance(controller, C1ModelController) else None
    result = {
        "status": "complete", "run_kind": run_kind, "seed": int(seed),
        "online_steps": step_count, "online_cumulative_return": cumulative_return,
        "completed_episodes": episode_count,
        "successful_episodes": sum(int(row["successes"]) for row in context_metrics.values()),
        "complete_episode_success_rate": sum(int(row["successes"]) for row in context_metrics.values()) / max(1, episode_count),
        "gradient_updates": 0, "statistical_updates": int(getattr(controller, "statistical_update_count", 0)),
        "neural_gradient_updates": 0,
        "max_serialized_agent_bytes": serialized_peak,
        "explicit_owned_memory_estimate_bytes": explicit_peak,
        "temporary_model_workspace_upper_bound_bytes": 32768 if controller is not None else 0,
        "controller_memory": controller_summary,
        "controller_confidence": float(controller.confidence) if isinstance(controller, C1ModelController) else None,
        "max_sampled_process_rss_bytes": max_rss,
        "actor_inference_elapsed_s": total_actor_inference,
        "model_action_elapsed_s": total_model_action,
        "model_update_elapsed_s": total_model_update,
        "router_elapsed_s": total_route,
        "environment_step_elapsed_s": total_env_step,
        "context_metrics": context_metrics,
        "stage_metrics": stage_metrics,
        "switch_delays_to_success": switches,
        "router_bank_unchanged": router_hash_after == source_tree_hash if router is not None else None,
        "last_observation": np.asarray(obs, dtype=np.float32).tolist(),
    }
    _write_json(output / "summary.json", result)
    return result


def _stage_sizes(schedule_config: dict[str, Any], base: dict[str, Any]) -> dict[str, int]:
    return {stage: int(schedule_config.get(f"{stage}_bout_steps", base[f"{stage}_bout_steps"])) for stage in STAGES}


def _attempt_dir(root: Path, suite: str, label: str) -> Path:
    parent = root / suite / label
    for attempt in range(1, 100):
        candidate = parent / f"attempt_{attempt}"
        result = candidate / "summary.json"
        if result.exists():
            try:
                if json.loads(result.read_text(encoding="utf-8")).get("status") == "complete":
                    return candidate
            except (OSError, json.JSONDecodeError):
                pass
            continue
        if (candidate / "failure.json").exists():
            continue
        return candidate
    raise RuntimeError(f"too many prior attempts for {suite}/{label}")


def _execute_model(
    base: dict[str, Any], supplement: dict[str, Any], output: Path,
    seed: int, kind: str, sizes: dict[str, int], suite: str,
) -> dict[str, Any]:
    run_dir = _attempt_dir(output, suite, f"{kind}_s{seed}")
    if (run_dir / "summary.json").exists():
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    actor = None
    controller = None
    try:
        actor = _foundation_actor(base)
        source_start = time.process_time()
        controller = None if kind == "W0_no_model" else C1ModelController(kind=kind, window=32, source_prior=load_source_prior(base["dataset"]))
        source_cpu = time.process_time() - source_start
        schedule = _schedule(base, sizes)
        result = run_trajectory(
            actor=actor, controller=controller, router=None, schedule=schedule, seed=seed,
            output=run_dir, temperature=float(supplement["temperature"]),
            epsilon=float(supplement["exploration_epsilon"]), run_kind=kind,
        )
        result["actor_parameter_bytes"] = sum(t.numel() * t.element_size() for t in actor.state_dict().values())
        result["controller_serialized_bytes"] = 0 if controller is None else len(json.dumps(controller.snapshot(), separators=(",", ":")).encode("utf-8"))
        result["source_prior_processing_cpu_s"] = source_cpu
        result["source_prior_transitions"] = 0 if controller is None else 20000
        _write_json(run_dir / "summary.json", result)
        return result
    except Exception as exc:
        _write_json(run_dir / "failure.json", {"status": "failed", "run_kind": kind, "seed": seed,
                    "error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps({"phase": "trajectory_failed", "run": kind, "seed": seed,
                          "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return {"status": "failed", "run_kind": kind, "seed": seed, "error": str(exc), "output": str(run_dir)}
    finally:
        del actor, controller


def _source_actor_from_checkpoint(path: Path) -> Any:
    core, _optimizer, router = load_bundle(path)
    core.eval()
    # Inference uses the policy trunk, action head and residuals. Drop both
    # value paths before deployment and before writing compact checkpoints.
    if hasattr(core, "value_net"):
        del core.value_net
    value_net = getattr(core.mlp_extractor, "value_net", None)
    if value_net is not None:
        del core.mlp_extractor.value_net
    return core, router


def _actor_only_c1_state(actor: Any) -> dict[str, torch.Tensor]:
    prefixes = (
        "mlp_extractor.policy_net.", "action_net.", "_nodes.", "_parents.", "_dictionary_V",
    )
    return {
        name: value.detach().cpu().clone()
        for name, value in actor.state_dict().items()
        if name.startswith(prefixes)
    }


def run_original_tree_diagnostic(
    base: dict[str, Any], supplement: dict[str, Any], output: Path,
) -> list[dict[str, Any]]:
    """Run matched hidden deployment from original after-revisit checkpoints."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    formal_seeds = [int(value) for value in supplement["formal_seeds"]]
    source_seeds = [int(value) for value in supplement["source_checkpoint_seeds"]]
    sizes = _stage_sizes(supplement["formal_schedule"], base)
    schedule = _schedule(base, sizes)
    history_schedule = [entry for entry in schedule if entry[0] == "history"]
    if sum(item[2] for item in history_schedule) != 6 * int(sizes["history"]):
        raise RuntimeError("tree deployment history schedule is not the configured six-context schedule")
    for tier in ("low", "high"):
        for source_seed, deployment_seed in zip(source_seeds, formal_seeds):
            label = f"{tier}_s{source_seed}_to_s{deployment_seed}"
            run_dir = _attempt_dir(output, "tree_diagnostic", label)
            if (run_dir / "summary.json").exists():
                records.append(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = ROOT / "outputs" / "c1" / "formal" / tier / "tree" / f"s{source_seed}" / "revisit" / "agent.pt"
            try:
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing original after-revisit bundle: {checkpoint}")
                core, router = _source_actor_from_checkpoint(checkpoint)
                bank_hash = hashlib.sha256(router.direction_counts.tobytes() + router.outcome_counts.tobytes()).hexdigest()
                router.clear_history()  # Keeps only the calibrated bank and its priors.
                hidden_seed_schedule = [("history", context, steps) for _, context, steps in history_schedule]
                record = run_trajectory(
                    actor=core, controller=None, router=router, schedule=hidden_seed_schedule,
                    seed=deployment_seed, output=run_dir, temperature=float(supplement["temperature"]),
                    epsilon=float(supplement["exploration_epsilon"]), run_kind=f"original_tree_{tier}_s{source_seed}",
                )
                record.update({
                    "tier": tier, "source_seed": source_seed, "deployment_seed": deployment_seed,
                    "original_checkpoint": str(checkpoint.relative_to(ROOT)),
                    "original_checkpoint_sha256": _sha256(checkpoint),
                    "starting_calibration_bank_sha256": bank_hash,
                    "actor_parameter_bytes": sum(t.numel() * t.element_size() for t in _actor_only_c1_state(core).values()),
                    "controller_memory": {
                        "kind": "C1HistoryRouter", "retained_event_count": len(router.events),
                        "bank_numeric_bytes": router.direction_counts.nbytes + router.outcome_counts.nbytes,
                        "calibration_profiles": int(np.count_nonzero(router.outcome_counts.sum(axis=(1, 2, 3)))),
                    },
                })
                torch.save({
                    "format_version": 1, "run_kind": record["run_kind"],
                    "seed": deployment_seed, "actor": _actor_only_c1_state(core),
                    "controller": router.snapshot(), "controller_bank_unchanged": record["router_bank_unchanged"],
                    "source_checkpoint_sha256": _sha256(checkpoint),
                }, run_dir / "actor_controller.pt")
                _write_json(run_dir / "summary.json", record)
                records.append(record)
                print(json.dumps({"phase": "tree_diagnostic_complete", "tier": tier,
                                  "source_seed": source_seed, "deployment_seed": deployment_seed,
                                  "steps": record["online_steps"],
                                  "reward": round(record["online_cumulative_return"], 2),
                                  "success_rate": round(record["complete_episode_success_rate"], 4)}), flush=True)
            except Exception as exc:
                failure = {"status": "failed", "tier": tier, "source_seed": source_seed,
                           "deployment_seed": deployment_seed, "error_type": type(exc).__name__, "error": str(exc)}
                _write_json(run_dir / "failure.json", failure)
                records.append(failure)
                print(json.dumps({"phase": "tree_diagnostic_failed", **failure}), flush=True)
    return records


def run_matched_model_diagnostic(base, supplement, output):
    """Clear current history after revisit, preserving source prior and actor.

    These controllers store all learned transition evidence in their bounded
    current window. Clearing it removes that evidence; unlike the tree, they
    do not retain a calibrated bank or learned actor residuals.
    """
    records = []
    sizes = _stage_sizes(supplement["formal_schedule"], base)
    schedule = [item for item in _schedule(base, sizes) if item[0] == "history"]
    for candidate in supplement["model_candidates"]:
        kind = candidate["kind"]
        for seed in supplement["formal_seeds"]:
            destination = _attempt_dir(output, "model_diagnostic", f"{kind}_s{seed}_clear_history")
            if (destination / "summary.json").exists():
                records.append(json.loads((destination / "summary.json").read_text()))
                continue
            checkpoint_path = output / "formal" / f"{kind}_s{seed}" / "attempt_1" / "checkpoint_revisit.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            controller = restore_controller(checkpoint)
            controller.clear_history()
            actor = _foundation_actor(base)
            actor.load_state_dict(checkpoint["actor"])
            record = run_trajectory(actor=actor, controller=controller, router=None, schedule=schedule,
                seed=seed, output=destination, temperature=supplement["temperature"],
                epsilon=supplement["exploration_epsilon"], run_kind=f"{kind}_clear_history")
            record["prior_interactions"] = checkpoint["global_step"]
            record["current_history_cleared"] = True
            record["persistent_online_knowledge_after_clear"] = False
            record["source_checkpoint_sha256"] = _sha256(checkpoint_path)
            _write_json(destination / "summary.json", record)
            records.append(record)
    return records


def run_step1(base: dict[str, Any], supplement: dict[str, Any], output: Path, *, mode: str = "all") -> dict[str, Any]:
    """Run validation, formal model/control trajectories and tree diagnostics."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "base_config.json", base)
    _write_json(output / "supplement_config.json", supplement)
    provenance_inputs = {
        "base_config": ROOT / supplement["base_config"],
        "supplement_config": ROOT / "configs" / "c1_supplement.json",
        "protocol": ROOT / "docs" / "C1_SUPPLEMENT_PROTOCOL.md",
        "foundation": ROOT / base["foundation"],
        "source_dataset": ROOT / base["dataset"],
        "controller_source": ROOT / "src" / "online_rl" / "c1_strong_models.py",
        "experiment_source": ROOT / "src" / "online_rl" / "c1_model_experiment.py",
    }
    _write_json(output / "provenance.json", {str(name): _sha256(path) for name, path in provenance_inputs.items()})

    validation_sizes = _stage_sizes(supplement["validation_schedule"], base)
    formal_sizes = _stage_sizes(supplement["formal_schedule"], base)
    result: dict[str, Any] = {"status": "running", "scope": "C1 Step 1; no labels or C2", "validation": [], "formal": [], "tree_diagnostic": [], "model_diagnostic": []}

    if mode in ("all", "validation"):
        validation: list[dict[str, Any]] = []
        for seed in map(int, supplement["validation_seeds"]):
            for candidate in supplement["model_candidates"]:
                validation.append(_execute_model(base, supplement, output, seed,
                                                  candidate["kind"], validation_sizes, "validation"))
            validation.append(_execute_model(base, supplement, output, seed,
                                              "W0_no_model", validation_sizes, "validation"))
        result["validation"] = validation
        _write_json(output / "validation_summary.json", {"runs": validation})

    if mode in ("all", "formal"):
        formal: list[dict[str, Any]] = []
        arms = [candidate["kind"] for candidate in supplement["model_candidates"]] + ["W0_no_model"]
        for seed in map(int, supplement["formal_seeds"]):
            for kind in arms:
                formal.append(_execute_model(base, supplement, output, seed, kind, formal_sizes, "formal"))
        result["formal"] = formal
        _write_json(output / "formal_summary.json", {"runs": formal})

    if mode in ("all", "tree-diagnostic"):
        result["tree_diagnostic"] = run_original_tree_diagnostic(base, supplement, output)
        _write_json(output / "tree_diagnostic_summary.json", {"runs": result["tree_diagnostic"]})
        result["model_diagnostic"] = run_matched_model_diagnostic(base, supplement, output)
        _write_json(output / "model_diagnostic_summary.json", {"runs": result["model_diagnostic"]})

    result["status"] = "complete" if all(
        run.get("status") == "complete" for run in result.get("validation", []) + result.get("formal", [])
    ) and all(run.get("status") == "complete" for run in result.get("tree_diagnostic", [])) else "partial"
    result["input_sha256"] = {str(name): _sha256(path) for name, path in provenance_inputs.items()}
    result["validation_interactions"] = sum(int(run.get("online_steps", 0)) for run in result["validation"])
    result["formal_interactions"] = sum(int(run.get("online_steps", 0)) for run in result["formal"])
    result["tree_diagnostic_interactions"] = sum(int(run.get("online_steps", 0)) for run in result["tree_diagnostic"])
    _write_json(output / "summary.json", result)
    return result


__all__ = [
    "FrozenSourceActor", "_base_probabilities", "_epsilon_mix", "make_checkpoint",
    "restore_action_generator", "restore_controller", "run_original_tree_diagnostic",
    "run_step1", "run_trajectory",
]

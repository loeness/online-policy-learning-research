"""A/B+ deployment controls and reusable terminal-state bundles.

The source pilot saved Stable-Baselines3 policy and optimizer tensors, but it
did not save the complete historical RNG state. ``load_terminal_run`` rebuilds
the trained policy, optimizer and router from those real files; the control
runner then declares new, shared evaluation seeds for each matched branch.
"""
from __future__ import annotations

import argparse
import copy
import csv
from collections import deque
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv

from .envs import NavigationEnv, contexts
from .experiment import BlankEnv, BudgetCallback, SEQUENCE, make_model, seed_everything, write_json
from .policy import TreePolicy
from .routing import HistoryRouter


DEFAULT_CONTEXTS = ["normal", "cw", "ccw", "cw_slip", "ccw_slip"]
ANALYTIC_A_SEQUENCE = ["cw"]
ANALYTIC_B_SEQUENCE = list(SEQUENCE)
CONTROL_BRANCHES = ("P+H+", "P+H0", "P0H0")
BRANCH_DIRECTORY = {"P+H+": "Pkeep_Hkeep", "P+H0": "Pkeep_Hclear", "P0H0": "Pzero_Hclear"}
OPTIMIZER_BRANCHES = ("retained_optimizer", "reset_optimizer")
def _walk_env(env: Any):
    """Yield wrapped gym environments without depending on wrapper depth."""
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)


def _capture_rng_state(model: Any | None = None, extra_envs: list[Any] | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "envs": [],
        "vec_env_seeds": None,
        "vec_env_options": None,
    }
    vec_env = getattr(model, "env", None) if model is not None else None
    if vec_env is not None:
        state["vec_env_seeds"] = copy.deepcopy(getattr(vec_env, "_seeds", None))
        state["vec_env_options"] = copy.deepcopy(getattr(vec_env, "_options", None))
    roots = [
        ("model", index, env) for index, env in enumerate(getattr(vec_env, "envs", []) or [])
    ]
    roots.extend(("external", index, env) for index, env in enumerate(extra_envs or []))
    for root_name, env_index, env in roots:
        for depth, wrapped in enumerate(_walk_env(env)):
            generator = getattr(wrapped, "np_random", None)
            if generator is not None:
                state["envs"].append({
                    "root": root_name,
                    "env_index": env_index,
                    "wrapper_depth": depth,
                    "kind": "env",
                    "bit_generator": copy.deepcopy(generator.bit_generator.state),
                    "np_random_seed": getattr(wrapped, "np_random_seed", None),
                })
            for space_name in ("action_space", "observation_space"):
                space = getattr(wrapped, space_name, None)
                space_rng = getattr(space, "np_random", None)
                if space_rng is not None:
                    state["envs"].append({
                        "root": root_name,
                        "env_index": env_index,
                        "wrapper_depth": depth,
                        "kind": space_name,
                        "bit_generator": copy.deepcopy(space_rng.bit_generator.state),
                    })
    return state


def _restore_rng_state(state: dict[str, Any], model: Any | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    vec_env = getattr(model, "env", None) if model is not None else None
    wrapped_envs = getattr(vec_env, "envs", []) or []
    if vec_env is not None:
        if state.get("vec_env_seeds") is not None:
            vec_env._seeds = copy.deepcopy(state["vec_env_seeds"])
        if state.get("vec_env_options") is not None:
            vec_env._options = copy.deepcopy(state["vec_env_options"])
    for saved in state.get("envs", []):
        if saved.get("root", "model") != "model":
            continue
        env_index = saved["env_index"]
        depth = saved["wrapper_depth"]
        if env_index >= len(wrapped_envs):
            continue
        wrappers = list(_walk_env(wrapped_envs[env_index]))
        if depth >= len(wrappers):
            continue
        wrapped = wrappers[depth]
        if saved["kind"] == "env":
            generator = getattr(wrapped, "np_random", None)
        else:
            generator = getattr(getattr(wrapped, saved["kind"], None), "np_random", None)
        if generator is not None:
            generator.bit_generator.state = copy.deepcopy(saved["bit_generator"])
            if saved.get("kind") == "env" and saved.get("np_random_seed") is not None:
                setattr(wrapped, "_np_random_seed", saved["np_random_seed"])


def _optimizer_parameter_names(policy: TreePolicy) -> list[list[str]]:
    by_id = {id(parameter): name for name, parameter in policy.named_parameters()}
    return [
        [by_id[id(parameter)] for parameter in group["params"]]
        for group in policy.optimizer.param_groups
    ]


def _trainable_parameter_names(policy: TreePolicy) -> list[str]:
    return [name for name, parameter in policy.named_parameters() if parameter.requires_grad]


def _assert_actor_base_frozen(policy: TreePolicy) -> None:
    trainable_actor_base = [
        name for name, parameter in policy.named_parameters()
        if (name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net."))
        and parameter.requires_grad
    ]
    if trainable_actor_base:
        raise RuntimeError(f"learned actor base must be frozen for history controls: {trainable_actor_base}")


def _actor_base_hash(policy: TreePolicy) -> str:
    digest = hashlib.sha256()
    for name, parameter in policy.named_parameters():
        if name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net."):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def history_router_from_snapshot(snapshot: dict[str, Any]) -> HistoryRouter:
    """Rebuild every HistoryRouter field, including its pending reset token."""
    required = {
        "source_responses", "window", "prototypes", "counts", "recurs", "active",
        "confidence", "novel_streak", "creation_count", "last_entered", "pending_reset",
        "boundary_events", "threshold", "minimum_events", "margin", "max_nodes", "events",
    }
    missing = required.difference(snapshot)
    if missing:
        raise ValueError(f"router snapshot is missing fields: {sorted(missing)}")
    source_responses = np.asarray(snapshot["source_responses"], dtype=np.float32)
    if source_responses.shape != (4, 2) or not np.all(np.isfinite(source_responses)):
        raise ValueError("router source responses must be finite with shape (4, 2)")
    router = HistoryRouter(
        source_responses,
        window=int(snapshot["window"]),
        novelty_threshold=float(snapshot["threshold"]),
        minimum_events=int(snapshot["minimum_events"]),
        confidence_margin=float(snapshot["margin"]),
        max_nodes=int(snapshot["max_nodes"]),
    )
    prototypes = [np.asarray(prototype, dtype=np.float32).copy() for prototype in snapshot["prototypes"]]
    counts = [int(value) for value in snapshot["counts"]]
    recurs = [int(value) for value in snapshot["recurs"]]
    if not prototypes or len(prototypes) != len(counts) or len(counts) != len(recurs):
        raise ValueError("router prototypes, counts and recurs must have matching nonzero lengths")
    if any(prototype.shape != (2,) or not np.all(np.isfinite(prototype)) for prototype in prototypes):
        raise ValueError("router prototypes must be finite vectors with shape (2,)")
    if any(value < 0 for value in counts + recurs):
        raise ValueError("router counts and recurs must be nonnegative")
    active = int(snapshot["active"])
    if active < 0 or active >= len(prototypes):
        raise ValueError("router active node is outside its prototypes")
    last_entered = int(snapshot["last_entered"])
    if last_entered < 0 or last_entered >= len(prototypes):
        raise ValueError("router last-entered node is outside its prototypes")
    confidence = float(snapshot["confidence"])
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("router confidence must be finite and in [0, 1]")
    if len(snapshot["events"]) > router.window:
        raise ValueError("router event history exceeds its configured window")
    events = deque(maxlen=router.window)
    for feature, action in snapshot["events"]:
        vector = np.asarray(feature, dtype=np.float32)
        action = int(action)
        if vector.shape != (2,) or not np.all(np.isfinite(vector)) or action not in (0, 1, 2, 3):
            raise ValueError("router events require finite (2,) features and an action in {0,1,2,3}")
        events.append((vector.copy(), action))
    router.events = events
    router.prototypes = prototypes
    router.counts = counts
    router.recurs = recurs
    router.active = active
    router.confidence = confidence
    router.novel_streak = int(snapshot["novel_streak"])
    router.creation_count = int(snapshot["creation_count"])
    router.last_entered = last_entered
    router.pending_reset = None if snapshot["pending_reset"] is None else int(snapshot["pending_reset"])
    router.boundary_events = int(snapshot["boundary_events"])
    if router.pending_reset is not None and not 0 <= router.pending_reset < len(prototypes):
        raise ValueError("router pending reset is outside its prototypes")
    return router


def clear_router_working_memory(router: HistoryRouter) -> None:
    """Clear transient history, including the router's deferred reset token."""
    router.clear_working_memory()
    router.pending_reset = None


def _new_policy_model(config: dict[str, Any]):
    env = DummyVecEnv([lambda: BlankEnv(NavigationEnv())])
    model = make_model(env, config)
    model._controls_config = copy.deepcopy(config)
    return model


def load_terminal_run(run_dir: str | Path, config: dict[str, Any]) -> tuple[Any, HistoryRouter]:
    """Reconstruct a trained history_keep model and router from a completed run.

    The source ``agent.zip`` was produced by this repository. It contains PPO's
    policy and Adam tensors, but not ``LeafAdam.step_count``; that count is
    recovered from the source summary. Full historical RNG state is absent and
    is therefore explicitly marked unavailable in metadata when bundled.
    """
    source = Path(run_dir)
    run_config_path = source / "config.json"
    model_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    for key, value in config.items():
        model_config.setdefault(key, value)
    model_config.setdefault("learning_rate", 0.003)
    model_config.setdefault("seed", 0)

    model = _new_policy_model(model_config)
    archive_data, params, _ = load_from_zip_file(source / "agent.zip", device="cpu")
    if "policy" not in params or "policy.optimizer" not in params:
        model.env.close()
        raise ValueError(f"{source / 'agent.zip'} must contain policy and optimizer states")
    model.policy.load_state_dict(params["policy"], strict=True)
    model.policy.configure_learning("history_keep", float(model_config["learning_rate"]))
    expected_optimizer_names = _optimizer_parameter_names(model.policy)
    source_groups = params["policy.optimizer"].get("param_groups", [])
    if len(source_groups) != len(expected_optimizer_names):
        model.env.close()
        raise ValueError("source optimizer parameter groups do not match reconstructed history_keep policy")
    if [len(group.get("params", [])) for group in source_groups] != [len(names) for names in expected_optimizer_names]:
        model.env.close()
        raise ValueError("source optimizer parameter order differs from reconstructed history_keep policy")
    model.policy.optimizer.load_state_dict(params["policy.optimizer"])

    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    model.policy.optimizer.step_count = int(summary.get("updates", 0))
    model.num_timesteps = int(summary.get("online_steps", archive_data.get("num_timesteps", 0)))
    model._n_updates = int(summary.get("ppo_epochs", 0))
    model._controls_trainable_parameter_names = _trainable_parameter_names(model.policy)
    model._controls_optimizer_parameter_names = expected_optimizer_names
    _assert_actor_base_frozen(model.policy)

    router_path = source / "router.json"
    router = history_router_from_snapshot(json.loads(router_path.read_text(encoding="utf-8")))
    model._controls_provenance = {
        "source_run": str(source),
        "source_policy_optimizer_reconstructed": True,
        "source_optimizer_step_count": int(model.policy.optimizer.step_count),
        "historic_complete_rng_state_available": False,
        "historic_rng_limitation": "The source run did not save complete Python, NumPy, Torch, environment, and action-space RNG states.",
        "evaluation_rng": "A new declared common environment/action seed is set per seed and context.",
    }
    return model, router


def save_deployment_bundle(
    path: str | Path,
    model: Any,
    router: HistoryRouter,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a reusable policy/optimizer/router/RNG checkpoint to a real file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_actor_base_frozen(model.policy)
    trainable_names = _trainable_parameter_names(model.policy)
    optimizer_names = _optimizer_parameter_names(model.policy)
    bundle_metadata = {
        "format": "online_rl.deployment_bundle",
        "format_version": 1,
        "source_provenance": copy.deepcopy(getattr(model, "_controls_provenance", {})),
        "historic_complete_rng_state_available": bool(
            getattr(model, "_controls_provenance", {}).get("historic_complete_rng_state_available", False)
        ),
        "captured_bundle_rng_state_exact": True,
        "optimizer_step_count": int(getattr(model.policy.optimizer, "step_count", 0)),
        "model_num_timesteps": int(getattr(model, "num_timesteps", 0)),
        "trainable_parameter_names": trainable_names,
        "optimizer_parameter_names": optimizer_names,
        "optimizer_parameter_mapping": optimizer_names,
        **copy.deepcopy(metadata or {}),
    }
    payload = {
        "metadata": bundle_metadata,
        "config": copy.deepcopy(getattr(model, "_controls_config", {})),
        "policy_state_dict": {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()},
        "optimizer_state_dict": copy.deepcopy(model.policy.optimizer.state_dict()),
        "router_snapshot": copy.deepcopy(router.snapshot()),
        "rng_state": _capture_rng_state(model),
    }
    temp = target.with_name(target.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(target)
    bundle_metadata["bundle_path"] = str(target)
    bundle_metadata["bundle_file_bytes"] = target.stat().st_size
    return bundle_metadata


def _deployment_core_bytes(model: Any, router: HistoryRouter, extra_envs: list[Any] | None = None) -> int:
    """Count the serialized inference/learning core without rollout memory."""
    import io

    payload = {
        "metadata": {
            "format": "online_rl.deployment_bundle",
            "format_version": 1,
            "source_provenance": copy.deepcopy(getattr(model, "_controls_provenance", {})),
            "historic_complete_rng_state_available": bool(
                getattr(model, "_controls_provenance", {}).get("historic_complete_rng_state_available", False)
            ),
            "captured_bundle_rng_state_exact": True,
            "optimizer_step_count": int(getattr(model.policy.optimizer, "step_count", 0)),
            "model_num_timesteps": int(getattr(model, "num_timesteps", 0)),
        },
        "config": copy.deepcopy(getattr(model, "_controls_config", {})),
        "policy_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": model.policy.optimizer.state_dict(),
        "router_snapshot": router.snapshot(),
        "rng_state": _capture_rng_state(model, extra_envs=extra_envs),
        "trainable_parameter_names": _trainable_parameter_names(model.policy),
        "optimizer_parameter_names": _optimizer_parameter_names(model.policy),
        "optimizer_step_count": int(getattr(model.policy.optimizer, "step_count", 0)),
    }
    packed = io.BytesIO()
    torch.save(payload, packed)
    return packed.tell()


def load_deployment_bundle(
    path: str | Path,
    config: dict[str, Any] | None,
) -> tuple[Any, HistoryRouter, dict[str, Any]]:
    """Restore one isolated deployment trial from a trusted local bundle."""
    # Bundles are generated and consumed inside this experiment workspace and
    # contain Python/NumPy RNG tuples; explicit weights_only=False is required.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("metadata", {}).get("format") != "online_rl.deployment_bundle" or payload.get("metadata", {}).get("format_version") != 1:
        raise ValueError("unsupported deployment bundle format")
    stored_config = payload.get("config", {})
    model_config = {**(config or {}), **stored_config}
    model_config.setdefault("learning_rate", 0.003)
    model_config.setdefault("seed", 0)
    model = _new_policy_model(model_config)
    model.policy.load_state_dict(payload["policy_state_dict"], strict=True)
    model.policy.configure_learning("history_keep", float(model_config["learning_rate"]))
    actual_trainable = _trainable_parameter_names(model.policy)
    actual_optimizer_names = _optimizer_parameter_names(model.policy)
    metadata = copy.deepcopy(payload["metadata"])
    if actual_trainable != metadata["trainable_parameter_names"]:
        model.env.close()
        raise ValueError("bundle trainable parameter mapping differs from reconstructed policy")
    if actual_optimizer_names != metadata["optimizer_parameter_names"]:
        model.env.close()
        raise ValueError("bundle optimizer parameter mapping differs from reconstructed optimizer")
    model.policy.optimizer.load_state_dict(payload["optimizer_state_dict"])
    model.policy.optimizer.step_count = int(metadata["optimizer_step_count"])
    model.num_timesteps = int(metadata.get("model_num_timesteps", 0))
    model._controls_config = copy.deepcopy(model_config)
    model._controls_trainable_parameter_names = actual_trainable
    model._controls_optimizer_parameter_names = actual_optimizer_names
    model._controls_provenance = copy.deepcopy(metadata.get("source_provenance", {}))
    _assert_actor_base_frozen(model.policy)
    router = history_router_from_snapshot(payload["router_snapshot"])
    _restore_rng_state(payload["rng_state"], model)
    metadata["restored_from_bundle"] = str(path)
    return model, router, metadata


def _route_recognizes(router: HistoryRouter, route: int, context: str) -> bool:
    spec = contexts()[context]
    signature = {
        0: np.asarray([1.0, 0.0], dtype=np.float32),
        1: np.asarray([0.0, 1.0], dtype=np.float32),
        -1: np.asarray([0.0, -1.0], dtype=np.float32),
    }[spec.rotation]
    signature = signature * (1.0 - spec.slip)
    return (
        router.confidence >= 0.35
        and 0 <= route < len(router.prototypes)
        and float(np.linalg.norm(router.prototypes[route] - signature)) <= 0.38
    )


def _branch_policy(model: Any, router: HistoryRouter, branch: str) -> None:
    if branch == "P+H+":
        return
    if branch in ("P+H0", "P0H0"):
        clear_router_working_memory(router)
    if branch == "P0H0":
        with torch.no_grad():
            for residual in model.policy.residuals:
                residual.zero_()
    if branch not in CONTROL_BRANCHES:
        raise ValueError(f"unknown deployment control branch: {branch}")


def _run_no_gradient_branch(
    branch_dir: Path,
    bundle_path: Path,
    config: dict[str, Any],
    seed: int,
    target_context: str,
    branch: str,
) -> dict[str, Any]:
    branch_dir.mkdir(parents=True, exist_ok=True)
    model, router, bundle_metadata = load_deployment_bundle(bundle_path, config)
    source_prototypes = [prototype.copy() for prototype in router.prototypes]
    _branch_policy(model, router, branch)
    if len(router.prototypes) != len(source_prototypes):
        raise RuntimeError("deployment branch changed learned router prototypes")
    for actual, expected in zip(router.prototypes, source_prototypes):
        np.testing.assert_array_equal(actual, expected)
    _assert_actor_base_frozen(model.policy)

    context_order = config.get("deployment_contexts", config.get("target_contexts", DEFAULT_CONTEXTS))
    evaluation_seed = int(config.get("evaluation_seed_offset", 400000)) + seed * 100 + list(context_order).index(target_context)
    seed_everything(evaluation_seed)
    env = NavigationEnv(contexts()[target_context])
    obs, _ = env.reset(seed=evaluation_seed)
    steps = int(config.get("deployment_steps", 2048))
    early_window = int(config.get("early_window_steps", 128))
    deterministic = bool(config.get("deterministic", False))
    budget = int(config.get("agent_budget_bytes", 262144))
    cumulative_return = 0.0
    early_return = 0.0
    completed = 0
    successes = 0
    recognized_steps = 0
    recognized_first_step: int | None = None
    policy_elapsed = 0.0
    router_elapsed = 0.0
    online_start = time.perf_counter()
    actor_before_hash = _actor_base_hash(model.policy)
    peak_rss = 0
    peak_accounted = 0
    process = psutil.Process()
    step_fields = [
        "seed", "context", "branch", "step", "episode", "obs_x", "obs_y", "obs_goal_x", "obs_goal_y",
        "action", "mapped_action", "next_x", "next_y", "next_goal_x", "next_goal_y", "reward", "route",
        "route_confidence", "episode_done", "episode_success",
    ]
    with (branch_dir / "steps.csv").open("w", newline="", encoding="utf-8") as step_file:
        writer = csv.DictWriter(step_file, fieldnames=step_fields)
        writer.writeheader()
        for index in range(steps):
            route_start = time.perf_counter()
            route = int(router.route())
            history = router.summary()
            confidence = float(router.confidence)
            router_elapsed += time.perf_counter() - route_start
            recognized = _route_recognizes(router, route, target_context)
            recognized_steps += int(recognized)
            if recognized and recognized_first_step is None:
                recognized_first_step = index + 1
            augmented = np.concatenate([obs, history, [route]]).astype(np.float32)
            policy_start = time.perf_counter()
            with torch.no_grad():
                action, _ = model.predict(augmented, deterministic=deterministic)
            policy_elapsed += time.perf_counter() - policy_start
            action = int(np.asarray(action).reshape(-1)[0])
            next_obs, reward, terminated, truncated, info = env.step(action)
            router_start = time.perf_counter()
            router.observe(obs, action, next_obs)
            router_elapsed += time.perf_counter() - router_start
            obs_before = obs
            obs = next_obs
            cumulative_return += float(reward)
            if index < early_window:
                early_return += float(reward)
            done = bool(terminated or truncated)
            writer.writerow({
                "seed": seed, "context": target_context, "branch": branch, "step": index + 1,
                "episode": completed + 1, "obs_x": float(obs_before[0]), "obs_y": float(obs_before[1]),
                "obs_goal_x": float(obs_before[2]), "obs_goal_y": float(obs_before[3]),
                "action": action, "mapped_action": action, "next_x": float(next_obs[0]), "next_y": float(next_obs[1]),
                "next_goal_x": float(next_obs[2]), "next_goal_y": float(next_obs[3]),
                "reward": float(reward), "route": route,
                "route_confidence": confidence, "episode_done": int(done),
                "episode_success": int(bool(info["success"])) if done else "",
            })
            if done:
                completed += 1
                successes += int(bool(info["success"]))
                obs, _ = env.reset()
            if (index + 1) % 256 == 0 or index + 1 == steps:
                state_bytes = _deployment_core_bytes(model, router, extra_envs=[env])
                peak_accounted = max(peak_accounted, state_bytes)
                peak_rss = max(peak_rss, int(process.memory_info().rss))
                if peak_accounted > budget:
                    raise RuntimeError(
                        f"deployment core exceeds accounted budget: {peak_accounted} > {budget} bytes"
                    )
    env.close()
    online_elapsed = time.perf_counter() - online_start
    actor_after_hash = _actor_base_hash(model.policy)
    if actor_after_hash != actor_before_hash:
        model.env.close()
        raise RuntimeError("actor base changed during no-gradient deployment")
    model.env.close()
    summary = {
        "seed": seed, "context": target_context, "branch": branch, "steps": steps,
        "evaluation_seed": evaluation_seed, "deterministic": deterministic,
        "online_elapsed_s": online_elapsed,
        "early_window_steps": min(steps, early_window),
        "early_window_cumulative_reward": early_return,
        "early_window_reward_per_step": early_return / max(1, min(steps, early_window)),
        "full_cumulative_reward": cumulative_return,
        "full_reward_per_step": cumulative_return / max(1, steps),
        "completed_episodes": completed, "successful_episodes": successes,
        "complete_episode_success_rate": successes / max(1, completed),
        "recognized_first_step": recognized_first_step,
        "recognized_coverage": recognized_steps / max(1, steps),
        "policy_inference_elapsed_s": policy_elapsed, "router_calls_elapsed_s": router_elapsed,
        "peak_accounted_agent_bytes": peak_accounted, "fixed_training_buffer_bytes": 0,
        "serialized_payload_scope": "policy state, Adam optimizer state, router snapshot, named parameter mapping, config, metadata and captured RNG; fixed training buffer is 0 bytes",
        "peak_process_rss_bytes_sampled": peak_rss, "agent_budget_bytes": budget,
        "source_optimizer_step_count": bundle_metadata["optimizer_step_count"],
        "gradient_updates": 0, "router_prototypes_preserved": len(source_prototypes),
        "actor_base_sha256_before": actor_before_hash, "actor_base_sha256_after": actor_after_hash,
        "raw_steps_csv": str(branch_dir / "steps.csv"),
    }
    write_json(branch_dir / "summary.json", summary)
    return summary


def evaluate_deployment_branch(
    bundle_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    seed: int,
    target_context: str,
    branch: str,
) -> dict[str, Any]:
    """Run one independently restored no-gradient deployment from a disk bundle."""
    return _run_no_gradient_branch(
        Path(output_dir), Path(bundle_path), config, int(seed), target_context, branch
    )


def _core_bytes(
    policy: TreePolicy,
    controller_snapshot: dict[str, Any] | None,
    rng_state: dict[str, Any],
    config: dict[str, Any],
) -> int:
    import io

    packed = io.BytesIO()
    torch.save({
        "policy": policy.state_dict(),
        "controller": controller_snapshot,
        "rng_state": rng_state,
        "config": config,
    }, packed)
    return packed.tell()


def _confidence_from_snapshot(snapshot: dict[str, Any]) -> float:
    events = snapshot.get("events", [])
    minimum = max(1, int(snapshot.get("minimum_events", 1)))
    window = max(1, int(snapshot.get("window", 1)))
    return min(1.0, len(events) / max(minimum, window))


def _run_analytic_variant(
    output_dir: Path,
    config: dict[str, Any],
    stage: str,
    seed: int,
    source_responses: np.ndarray,
    checkpoint_path: Path,
    variant: str,
) -> dict[str, Any]:
    # Keep this import lazy: ordinary terminal-bundle users need not import the
    # optional analytic controller module.
    from .analytic import AnalyticController

    model_config = json.loads(Path(config.get("foundation_config", "outputs/validated/config.json")).read_text(encoding="utf-8"))
    model = _new_policy_model(model_config)
    foundation = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.policy.load_state_dict(foundation, strict=True)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    model.policy.eval()
    # The analytic baselines only infer; remove PPO's constructor-created
    # optimizer so the fixed baseline has no optimizer state in RAM.
    model.policy.optimizer = None
    _assert_actor_base_frozen(model.policy)
    actor_before_hash = _actor_base_hash(model.policy)
    controller = None
    if variant == "analytic":
        controller = AnalyticController(
            source_responses,
            window=int(config.get("analytic_window", config.get("route_window", 16))),
            minimum_events=int(config.get("analytic_minimum_events", config.get("minimum_events", 5))),
        )
    elif variant != "frozen_foundation":
        model.env.close()
        raise ValueError(f"unknown analytic control variant: {variant}")

    sequence = ANALYTIC_A_SEQUENCE if stage == "A" else ANALYTIC_B_SEQUENCE
    bout_steps = int(config.get("bout_steps", 4096))
    total_steps = int(
        config.get("a_steps", config.get("stage_a_steps", 16384))
        if stage == "A"
        else config.get("stage_b_steps", bout_steps * len(ANALYTIC_B_SEQUENCE))
    )
    stage_seed_offset = 10000 if stage == "A" else 20000
    evaluation_seed = int(config.get("evaluation_seed_offset", 400000)) + stage_seed_offset + seed
    deterministic = bool(config.get("deterministic", False))
    budget = int(config.get("agent_budget_bytes", 262144))
    seed_everything(evaluation_seed)
    env = NavigationEnv()
    obs, _ = env.reset(seed=evaluation_seed)
    cumulative_return = 0.0
    early_steps = int(config.get("early_window_steps", 128))
    early_return = 0.0
    episode = 0
    successes = 0
    completed = 0
    policy_elapsed = 0.0
    routing_elapsed = 0.0
    peak_rss = 0
    peak_core_bytes = 0
    process = psutil.Process()
    start = time.perf_counter()
    step_path = output_dir / "steps.csv"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed", "stage", "variant", "step", "context", "episode", "obs_x", "obs_y", "obs_goal_x",
        "obs_goal_y", "action", "mapped_action", "next_x", "next_y", "next_goal_x", "next_goal_y",
        "reward", "route", "route_confidence", "episode_done", "episode_success",
    ]
    with step_path.open("w", newline="", encoding="utf-8") as step_file:
        writer = csv.DictWriter(step_file, fieldnames=fields)
        writer.writeheader()
        for index in range(total_steps):
            context_name = sequence[min(index // bout_steps, len(sequence) - 1)]
            env.set_context(contexts()[context_name])
            augmented = np.concatenate([obs, np.zeros(8, dtype=np.float32), [0.0]]).astype(np.float32)
            policy_start = time.perf_counter()
            with torch.no_grad():
                action, _ = model.predict(augmented, deterministic=deterministic)
            policy_elapsed += time.perf_counter() - policy_start
            intended = int(np.asarray(action).reshape(-1)[0])
            route = int(getattr(controller, "estimated_rotation", 0)) if controller is not None else 0
            route_confidence = _confidence_from_snapshot(controller.snapshot()) if controller is not None else 0.0
            routing_start = time.perf_counter()
            mapped = int(controller.map_action(intended)) if controller is not None else intended
            routing_elapsed += time.perf_counter() - routing_start
            next_obs, reward, terminated, truncated, info = env.step(mapped)
            if controller is not None:
                routing_start = time.perf_counter()
                # The controller compares the movement with the action actually
                # sent to NavigationEnv so inverse mapping remains observable.
                controller.observe(obs, mapped, next_obs)
                routing_elapsed += time.perf_counter() - routing_start
            cumulative_return += float(reward)
            if index < early_steps:
                early_return += float(reward)
            done = bool(terminated or truncated)
            writer.writerow({
                "seed": seed, "stage": stage, "variant": variant, "step": index + 1,
                "context": context_name, "episode": episode + 1,
                "obs_x": float(obs[0]), "obs_y": float(obs[1]), "obs_goal_x": float(obs[2]),
                "obs_goal_y": float(obs[3]), "action": intended, "mapped_action": mapped,
                "next_x": float(next_obs[0]), "next_y": float(next_obs[1]),
                "next_goal_x": float(next_obs[2]), "next_goal_y": float(next_obs[3]),
                "reward": float(reward), "route": route, "route_confidence": route_confidence,
                "episode_done": int(done), "episode_success": int(bool(info["success"])) if done else "",
            })
            obs = next_obs
            if done:
                completed += 1
                successes += int(bool(info["success"]))
                episode += 1
                obs, _ = env.reset()
            if (index + 1) % 256 == 0 or index + 1 == total_steps:
                snapshot = controller.snapshot() if controller is not None else None
                core_bytes = _core_bytes(
                    model.policy, snapshot, _capture_rng_state(model, extra_envs=[env]), model_config
                )
                peak_core_bytes = max(peak_core_bytes, core_bytes)
                peak_rss = max(peak_rss, int(process.memory_info().rss))
                if core_bytes > budget:
                    raise RuntimeError(f"analytic deployed core exceeds budget: {core_bytes} > {budget} bytes")
    elapsed = time.perf_counter() - start
    actor_after_hash = _actor_base_hash(model.policy)
    if actor_after_hash != actor_before_hash:
        env.close()
        model.env.close()
        raise RuntimeError("fixed foundation actor changed during analytic control deployment")
    env.close()
    model.env.close()
    snapshot = controller.snapshot() if controller is not None else None
    summary = {
        "seed": seed, "stage": stage, "variant": variant, "steps": total_steps,
        "context_schedule": sequence, "bout_steps": bout_steps, "evaluation_seed": evaluation_seed,
        "deterministic": deterministic, "policy_updates": 0,
        "early_window_steps": min(total_steps, early_steps),
        "early_window_cumulative_reward": early_return,
        "early_window_reward_per_step": early_return / max(1, min(total_steps, early_steps)),
        "full_cumulative_reward": cumulative_return,
        "full_reward_per_step": cumulative_return / max(1, total_steps),
        "completed_episodes": completed, "successful_episodes": successes,
        "complete_episode_success_rate": successes / max(1, completed),
        "policy_inference_elapsed_s": policy_elapsed, "routing_elapsed_s": routing_elapsed,
        "elapsed_s": elapsed, "peak_serialized_core_bytes": peak_core_bytes,
        "online_elapsed_s": elapsed,
        "actor_base_sha256_before": actor_before_hash,
        "actor_base_sha256_after": actor_after_hash,
        "serialized_payload_scope": "policy state, controller snapshot, captured RNG state and model config; optimizer and fixed training buffer are 0 bytes",
        "optimizer_bytes": 0, "fixed_training_buffer_bytes": 0,
        "peak_process_rss_bytes_sampled": peak_rss, "agent_budget_bytes": budget,
        "controller_final_state": snapshot,
        "raw_steps_csv": str(step_path),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _run_optimizer_branch(
    branch_dir: Path,
    bundle_path: Path,
    config: dict[str, Any],
    seed: int,
    branch: str,
) -> dict[str, Any]:
    from .experiment import StreamEnv

    branch_dir.mkdir(parents=True, exist_ok=True)
    model, router, _ = load_deployment_bundle(bundle_path, config)
    clear_router_working_memory(router)
    source_optimizer_steps = int(getattr(model.policy.optimizer, "step_count", 0))
    if branch == "reset_optimizer":
        model.policy.optimizer.state.clear()
        model.policy.optimizer.step_count = 0
    elif branch != "retained_optimizer":
        model.env.close()
        raise ValueError(f"unknown optimizer branch: {branch}")
    starting_optimizer_steps = int(getattr(model.policy.optimizer, "step_count", 0))

    sequence = list(config.get("continuation_sequence", ["cw", "ccw"]))
    bout_steps = int(config.get("continuation_bout_steps", 4096))
    interaction_steps = int(config.get("continuation_steps", bout_steps * len(sequence)))
    stream_config = {**config, "bout_steps": bout_steps}
    initial_policy_state = {
        name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()
    }
    episode_path = branch_dir / "episodes.csv"
    capacity_path = branch_dir / "capacity.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as episode_file, capacity_path.open(
        "w", newline="", encoding="utf-8"
    ) as capacity_file:
        episode_fields = [
            "seed", "stage", "method", "episode", "global_steps", "context", "bout", "visit", "return",
            "success", "steps", "executed_node", "detected_node", "route_confidence", "update_count",
            "elapsed_s", "policy_bytes", "agent_state_bytes", "rollout_bytes", "is_eval", "mixed_context",
            "reset_count", "route_accuracy", "route_coverage", "cumulative_return", "reward_per_step",
        ]
        episode_writer = csv.DictWriter(episode_file, fieldnames=episode_fields)
        episode_writer.writeheader()
        capacity_fields = [
            "global_steps", "policy_bytes", "optimizer_bytes", "router_bytes", "rollout_bytes",
            "serialized_core_bytes", "agent_state_bytes", "rss_bytes",
        ]
        capacity_writer = csv.DictWriter(capacity_file, fieldnames=capacity_fields)
        capacity_writer.writeheader()
        stream = StreamEnv("B+_optimizer", "history_keep", seed, router, stream_config, episode_writer, sequence)
        vec = DummyVecEnv([lambda: stream])
        model.set_env(vec, force_reset=True)
        stream.model = model
        callback = BudgetCallback(stream, capacity_writer, int(config.get("agent_budget_bytes", 262144)))
        callback.init_callback(model)
        start = time.perf_counter()
        model.learn(total_timesteps=interaction_steps, callback=callback, reset_num_timesteps=True)
        elapsed = time.perf_counter() - start
        optimizer_end = int(getattr(model.policy.optimizer, "step_count", 0))
        final_policy_state = model.policy.state_dict()
        squared_delta = 0.0
        residual_squared_delta = 0.0
        actor_base_max_abs_delta = 0.0
        for name, before in initial_policy_state.items():
            delta = final_policy_state[name].detach().cpu() - before
            squared_delta += float(delta.double().square().sum())
            if name.startswith("residuals."):
                residual_squared_delta += float(delta.double().square().sum())
            if name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net."):
                actor_base_max_abs_delta = max(actor_base_max_abs_delta, float(delta.abs().max()))
        if actor_base_max_abs_delta != 0.0:
            raise RuntimeError("actor base changed during history_keep optimizer continuation")
        checkpoint_path = branch_dir / "agent.zip"
        model.save(checkpoint_path)
        router_path = branch_dir / "router.json"
        write_json(router_path, router.snapshot())
        bundle_path_out = branch_dir / "deployment_bundle.pt"
        save_deployment_bundle(
            bundle_path_out,
            model,
            router,
            metadata={
                "continuation_branch": branch,
                "source_seed": seed,
                "optimizer_steps_added": optimizer_end - starting_optimizer_steps,
                "historic_complete_rng_state_available": False,
            },
        )
        summary = {
            "seed": seed, "branch": branch, "sequence": sequence, "bout_steps": bout_steps,
            "steps": stream.total_steps, "online_cumulative_return": stream.cumulative_return,
            "online_reward_per_step": stream.cumulative_return / max(1, stream.total_steps),
            "starting_optimizer_steps": starting_optimizer_steps,
            "source_optimizer_steps": source_optimizer_steps,
            "ending_optimizer_steps": optimizer_end,
            "optimizer_steps_added": optimizer_end - starting_optimizer_steps,
            "policy_delta_l2": squared_delta ** 0.5,
            "residual_delta_l2": residual_squared_delta ** 0.5,
            "actor_base_max_abs_delta": actor_base_max_abs_delta,
            "checkpoint_path": str(checkpoint_path),
            "router_path": str(router_path),
            "deployment_bundle_path": str(bundle_path_out),
            "peak_agent_state_bytes": callback.peak, "peak_process_rss_bytes_sampled": callback.rss_peak,
            "agent_budget_bytes": callback.budget, "elapsed_s": elapsed,
            "route_accuracy": stream.route_correct / max(1, stream.total_steps),
            "route_coverage": stream.route_covered / max(1, stream.total_steps),
            "episodes_csv": str(episode_path), "capacity_csv": str(capacity_path),
        }
        write_json(branch_dir / "summary.json", summary)
        vec.close()
    return summary


def run_suite(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Run no-gradient context controls, analytic controls and optimizer isolation."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in config.get("source_seeds", config.get("seeds", (41, 53, 67))))
    target_contexts = list(config.get("deployment_contexts", config.get("target_contexts", DEFAULT_CONTEXTS)))
    deployment_steps = int(config.get("deployment_steps", 2048))
    early_window_steps = int(config.get("early_window_steps", 128))
    source_root = Path(config.get("source_root", "outputs/validated"))
    dataset_path = Path(config.get("dataset", config.get("dataset_path", "data/pretrain_source.npz")))
    checkpoint_path = Path(config.get("foundation", config.get("foundation_checkpoint", "data/foundation.pt")))
    bundle_dir = root / "deployment_bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundles: dict[int, Path] = {}
    no_gradient_runs = []
    optimizer_runs = []
    analytic_runs = []
    effective_config = {
        **config,
        "seeds": list(seeds),
        "deployment_contexts": target_contexts,
        "deployment_steps": deployment_steps,
        "early_window_steps": early_window_steps,
        "agent_budget_bytes": int(config.get("agent_budget_bytes", 262144)),
    }
    write_json(root / "config.json", effective_config)

    # Reconstruct once per real source terminal checkpoint, then independently
    # reload the saved bundle before every branch/context deployment.
    for seed in seeds:
        source_dir = source_root / f"B_history_keep_s{seed}"
        model, router = load_terminal_run(source_dir, config)
        bundle_path = bundle_dir / f"B_history_keep_s{seed}.pt"
        provenance = copy.deepcopy(getattr(model, "_controls_provenance", {}))
        bundle_metadata = save_deployment_bundle(
            bundle_path,
            model,
            router,
            metadata={
                "source_seed": seed,
                "source_step_count": int(getattr(model, "num_timesteps", 0)),
                "source_optimizer_step_count": int(getattr(model.policy.optimizer, "step_count", 0)),
                "source_provenance": provenance,
                "historic_complete_rng_state_available": False,
                "declared_evaluation_rng_policy": "new common environment/action seed per source seed and target context",
            },
        )
        model.env.close()
        bundles[seed] = bundle_path
        write_json(bundle_dir / f"B_history_keep_s{seed}.json", bundle_metadata)

        for context_name in target_contexts:
            if context_name not in contexts():
                raise ValueError(f"unknown target context: {context_name}")
            for branch in CONTROL_BRANCHES:
                branch_dir = root / "no_gradient" / f"s{seed}" / context_name / BRANCH_DIRECTORY[branch]
                result = evaluate_deployment_branch(bundle_path, branch_dir, config, seed, context_name, branch)
                no_gradient_runs.append(result)
                print(
                    f"DONE no-gradient seed={seed} context={context_name} branch={branch}: "
                    f"reward={result['full_cumulative_reward']:.3f}, success={result['complete_episode_success_rate']:.3f}",
                    flush=True,
                )
                gc.collect()

        if bool(config.get("optimizer_isolation", True)):
            for branch in OPTIMIZER_BRANCHES:
                branch_dir = root / "optimizer_isolation" / f"s{seed}" / branch
                result = _run_optimizer_branch(branch_dir, bundle_path, config, seed, branch)
                optimizer_runs.append(result)
                print(
                    f"DONE optimizer seed={seed} branch={branch}: "
                    f"steps={result['steps']}, optimizer_steps_added={result['optimizer_steps_added']}",
                    flush=True,
                )
                gc.collect()

    analytic_stages = list(config.get("analytic_stages", ["A", "B"]))
    if analytic_stages:
        dataset = np.load(dataset_path)
        source_responses = np.asarray(dataset["source_responses"], dtype=np.float32)
        for stage in analytic_stages:
            if stage not in ("A", "B"):
                raise ValueError(f"analytic stage must be A or B, got {stage}")
            for seed in seeds:
                for variant in ("analytic", "frozen_foundation"):
                    run_dir = root / "analytic" / stage / f"s{seed}" / variant
                    result = _run_analytic_variant(
                        run_dir, config, stage, seed, source_responses, checkpoint_path, variant
                    )
                    analytic_runs.append(result)
                    print(
                        f"DONE analytic stage={stage} seed={seed} variant={variant}: "
                        f"reward={result['full_cumulative_reward']:.3f}, "
                        f"success={result['complete_episode_success_rate']:.3f}",
                        flush=True,
                    )
                    gc.collect()

    summary = {
        "config": effective_config,
        "no_gradient_controls": no_gradient_runs,
        "optimizer_isolation": optimizer_runs,
        "analytic_controls": analytic_runs,
        "source_checkpoint_reconstruction": "policy state, history_keep Adam state, summary step counter and full router snapshot restored; source historic RNG unavailable",
        "deployment_rng": "new declared common environment/action seeds are used for matched branches",
        "budget_accounting": "deployed core plus fixed training buffer; no-gradient buffer is 0 bytes; frozen analytic variants have no optimizer and 0 byte training buffer; optimizer continuation includes the fixed rollout buffer",
    }
    write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B+ deployment, analytic, and optimizer controls")
    parser.add_argument("--config", default="configs/ab_plus.json")
    parser.add_argument("--output", default="outputs/ab_plus")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = run_suite(config, args.output)
    print(json.dumps({
        "no_gradient_runs": len(summary["no_gradient_controls"]),
        "optimizer_runs": len(summary["optimizer_isolation"]),
        "analytic_runs": len(summary["analytic_controls"]),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

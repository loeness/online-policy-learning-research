"""Run the frozen pre-core local-update and transfer diagnostic.

This is an experimenter-isolated C1 diagnostic, not a task-free agent: each
training process is held in one known simulator context while its policy sees
only the ordinary observation and the constant route coordinate 1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import psutil
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.c1_envs import C1NavigationEnv, c1_contexts, context_index, context_name  # noqa: E402
from online_rl.c1_experiment import augment  # noqa: E402
from online_rl.c1_tensor_controls import TensorSourceBasisPolicy  # noqa: E402
from online_rl.c1_tensor_experiment import _tensor_make_core  # noqa: E402
from online_rl.compression import export_payload  # noqa: E402
from online_rl.compression_experiment import PackedSB3Adapter, actor_hash  # noqa: E402
from online_rl.experiment import make_model, seed_everything  # noqa: E402


CONTEXTS = tuple(c1_contexts())
ROLLOUT_STEPS = 256
BATCH_SIZE = 64
PPO_EPOCHS = 4
LEARNING_RATE = 0.003
ENTROPY_COEFFICIENT = 0.06
MAX_EPISODE_STEPS = 64
ROUTE_ID = 1
DEFAULT_SEEDS = (271, 283, 307)
TRAIN_EPISODE_FIELDS = (
    "seed", "source", "episode", "reset_seed", "initial_x", "initial_y",
    "goal_x", "goal_y", "episode_return", "success", "length", "completed",
)
EVAL_EPISODE_FIELDS = (
    "seed", "source", "target", "episode", "reset_seed", "action_uniform_seed",
    "initial_x", "initial_y", "goal_x", "goal_y", "episode_return", "success",
    "length", "completed", "baseline_return", "paired_gain",
)


class FixedContextRouteOneEnv(gym.Wrapper):
    """Expose C1's four visible values and a fixed route ID of one."""

    def __init__(self, context: str) -> None:
        key = context_name(context)
        super().__init__(C1NavigationEnv(c1_contexts()[key]))
        self.context = key
        self.observation_space = spaces.Box(-2.0, 9.0, shape=(13,), dtype=np.float32)
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_initial: np.ndarray | None = None
        self._episode_reset_seed: int | None = None
        self.episode_records: list[dict[str, Any]] = []

    def _policy_observation(self, observation: np.ndarray) -> np.ndarray:
        return augment(observation, ROUTE_ID)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_initial = np.asarray(observation, dtype=np.float32).copy()
        reset_seed = kwargs.get("seed")
        self._episode_reset_seed = None if reset_seed is None else int(reset_seed)
        return self._policy_observation(observation), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        self._episode_length += 1
        if terminated or truncated:
            initial = self._episode_initial
            if initial is not None:
                self.episode_records.append({
                    "source": self.context,
                    "reset_seed": self._episode_reset_seed,
                    "initial_x": float(initial[0]),
                    "initial_y": float(initial[1]),
                    "goal_x": float(initial[2]),
                    "goal_y": float(initial[3]),
                    "episode_return": float(self._episode_return),
                    "success": int(bool(info.get("success", False))),
                    "length": int(self._episode_length),
                    "completed": True,
                })
            self._episode_return = 0.0
            self._episode_length = 0
            self._episode_initial = None
            self._episode_reset_seed = None
        return self._policy_observation(observation), reward, terminated, truncated, info

    def finish_partial_episode(self) -> None:
        """Keep a budget-ending partial episode visible in the training trace."""
        if self._episode_length <= 0 or self._episode_initial is None:
            return
        initial = self._episode_initial
        self.episode_records.append({
            "source": self.context,
            "reset_seed": self._episode_reset_seed,
            "initial_x": float(initial[0]),
            "initial_y": float(initial[1]),
            "goal_x": float(initial[2]),
            "goal_y": float(initial[3]),
            "episode_return": float(self._episode_return),
            "success": 0,
            "length": int(self._episode_length),
            "completed": False,
        })
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_initial = None
        self._episode_reset_seed = None


def _rng_snapshot() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy_name": numpy_state[0],
        "numpy_keys": numpy_state[1].copy(),
        "numpy_position": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch_cpu": torch.get_rng_state().clone(),
    }


def _rng_snapshots_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["python"] == right["python"]
        and left["numpy_name"] == right["numpy_name"]
        and np.array_equal(left["numpy_keys"], right["numpy_keys"])
        and left["numpy_position"] == right["numpy_position"]
        and left["numpy_has_gauss"] == right["numpy_has_gauss"]
        and left["numpy_cached_gaussian"] == right["numpy_cached_gaussian"]
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
    )


def capture_first_rollout_actor_gradient(model, core: TensorSourceBasisPolicy) -> dict[str, Any]:
    """Differentiate the normalized first-rollout actor surrogate without .grad writes."""
    buffer = model.rollout_buffer
    parameter = core._nodes[str(ROUTE_ID)].A
    if parameter.grad is not None:
        raise RuntimeError("the first-rollout probe unexpectedly found a pre-existing A gradient")
    before = _rng_snapshot()
    observation_shape = tuple(core.observation_space.shape)
    observations = np.asarray(buffer.observations).reshape((-1, *observation_shape)).copy()
    actions = np.asarray(buffer.actions).copy().reshape(-1)
    advantages = np.asarray(buffer.advantages).copy().reshape(-1)
    device = parameter.device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    normalized_advantages = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
    _, log_probability, _ = model.policy.evaluate_actions(obs_tensor, action_tensor)
    surrogate_loss = -(log_probability * normalized_advantages).mean()
    gradient, = torch.autograd.grad(
        surrogate_loss,
        parameter,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    after = _rng_snapshot()
    rng_unchanged = _rng_snapshots_equal(before, after)
    if not rng_unchanged:
        raise RuntimeError("first-rollout gradient capture changed a global RNG state")
    if parameter.grad is not None:
        raise RuntimeError("torch.autograd.grad unexpectedly wrote into parameter.grad")
    gradient_vector = gradient.detach().cpu().contiguous().reshape(-1).tolist()
    return {
        "route": ROUTE_ID,
        "parameter_name": f"_nodes.{ROUTE_ID}.A",
        "parameter_shape": list(parameter.shape),
        "coordinate_count": int(parameter.numel()),
        "rollout_observations": int(observations.shape[0]),
        "normalization": "full first-rollout population; torch std (Bessel-corrected) plus 1e-8",
        "actor_surrogate_loss": float(surrogate_loss.detach().cpu()),
        "gradient_l2_norm": float(torch.linalg.vector_norm(gradient.detach()).cpu()),
        "gradient_vector": gradient_vector,
        "rng_state_unchanged": rng_unchanged,
        "parameter_grad_remained_none": parameter.grad is None,
    }


class TrainingDiagnosticsCallback(BaseCallback):
    def __init__(self, core: TensorSourceBasisPolicy) -> None:
        super().__init__()
        self.core = core
        self.first_gradient: dict[str, Any] | None = None
        self.rss_samples: list[dict[str, Any]] = []
        self.process = psutil.Process()
        self.started_at = 0.0

    def _sample_rss(self, phase: str) -> None:
        self.rss_samples.append({
            "phase": phase,
            "timesteps": int(self.num_timesteps),
            "elapsed_s": float(time.perf_counter() - self.started_at),
            "rss_bytes": int(self.process.memory_info().rss),
        })

    def _on_training_start(self) -> None:
        self.started_at = time.perf_counter()
        self._sample_rss("training_start")

    def _on_rollout_start(self) -> None:
        self._sample_rss("rollout_start")

    def _on_rollout_end(self) -> None:
        if self.first_gradient is None:
            self.first_gradient = capture_first_rollout_actor_gradient(self.model, self.core)
        self._sample_rss("rollout_end_before_ppo_update")

    def _on_training_end(self) -> None:
        self._sample_rss("training_end")

    def _on_step(self) -> bool:
        return True


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    from online_rl.c1_supplement_experiment import load_protocol

    supplement, base = load_protocol(ROOT / "configs" / "c1_supplement.json")
    config = dict(base)
    config["temperature"] = float(supplement.get("temperature", 2.0))
    return supplement, config


def _make_source_core(config: dict[str, Any], seed: int):
    return _tensor_make_core(
        config,
        seed,
        "output_mixer",
        {"leaf_rank": 1, "dictionary_rank": 3},
        {"learning_rate": LEARNING_RATE, "ent_coef": ENTROPY_COEFFICIENT},
    )


def _build_training_model(core, optimizer, context: str, seed: int, config: dict[str, Any]):
    env = DummyVecEnv([lambda: FixedContextRouteOneEnv(context)])
    model = make_model(env, {
        **config,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "ent_coef": ENTROPY_COEFFICIENT,
    })
    model.policy = PackedSB3Adapter(core, optimizer)
    return model, env


def _jsonable_state_hash(core: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(core.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_source(
    *,
    seed: int,
    source: str,
    train_steps: int,
    config: dict[str, Any],
    branch_dir: Path,
    smoke_test: bool,
) -> dict[str, Any]:
    seed_everything(seed)
    core, optimizer = _make_source_core(config, seed)
    initial_a = core._nodes[str(ROUTE_ID)].A.detach().cpu().contiguous().reshape(-1).clone()
    initial_source_basis = core.source_basis.detach().cpu().clone()
    initial_actor_hash = actor_hash(core)
    initial_state_hash = _jsonable_state_hash(core)
    model, env = _build_training_model(core, optimizer, source, seed, config)
    callback = TrainingDiagnosticsCallback(core)
    initial_optimizer_steps = int(getattr(optimizer, "step_count", 0))
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    try:
        model.learn(total_timesteps=train_steps, callback=callback, progress_bar=False)
        elapsed_wall = time.perf_counter() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        if model.num_timesteps != train_steps:
            raise RuntimeError(f"PPO collected {model.num_timesteps} steps, expected {train_steps}")
        if callback.first_gradient is None:
            raise RuntimeError("the first-rollout actor gradient was not captured")
        env.envs[0].finish_partial_episode()
        final_a = core._nodes[str(ROUTE_ID)].A.detach().cpu().contiguous().reshape(-1).clone()
        update_vector = (final_a - initial_a).tolist()
        inactive = {
            str(endpoint): float(torch.linalg.vector_norm(core._nodes[str(endpoint)].A.detach()).cpu())
            for endpoint in range(2, 7)
        }
        inactive_zero = all(value == 0.0 for value in inactive.values())
        if not inactive_zero:
            raise RuntimeError(f"inactive output_mixer slots changed: {inactive}")
        if actor_hash(core) != initial_actor_hash:
            raise RuntimeError("the frozen W0 actor changed during local training")
        if core.source_basis.requires_grad:
            raise RuntimeError("output_mixer source basis must stay frozen")
        if not torch.equal(initial_source_basis, core.source_basis.detach().cpu()):
            raise RuntimeError("the source basis changed during training")
        optimizer_steps = int(getattr(optimizer, "step_count", 0)) - initial_optimizer_steps
        maximum_optimizer_steps = (
            (train_steps // ROLLOUT_STEPS) * PPO_EPOCHS * (ROLLOUT_STEPS // BATCH_SIZE)
        )
        if not 0 < optimizer_steps <= maximum_optimizer_steps:
            raise RuntimeError(
                f"PPO optimizer steps {optimizer_steps} outside (0, {maximum_optimizer_steps}]"
            )

        branch_dir.mkdir(parents=True, exist_ok=True)
        payload = export_payload(
            core,
            optimizer,
            router_snapshot=None,
            metadata={
                "scope": "P0 isolated C1 local update",
                "seed": seed,
                "source_context": source,
                "route": ROUTE_ID,
                "train_steps": train_steps,
                "learning_rate": LEARNING_RATE,
                "ent_coef": ENTROPY_COEFFICIENT,
                "smoke_test": bool(smoke_test),
            },
        )
        checkpoint_path = branch_dir / "agent.pt"
        torch.save(payload, checkpoint_path)
        train_episode_path = branch_dir / "training_episodes.csv"
        with train_episode_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRAIN_EPISODE_FIELDS)
            writer.writeheader()
            for index, row in enumerate(env.envs[0].episode_records, start=1):
                writer.writerow({
                    "seed": seed,
                    "source": source,
                    "episode": index,
                    **row,
                })
        training_episodes = list(env.envs[0].episode_records)
        result = {
            "seed": seed,
            "source_context": source,
            "route": ROUTE_ID,
            "train_steps_requested": train_steps,
            "train_steps_actual": int(model.num_timesteps),
            "optimizer_steps_added": optimizer_steps,
            "optimizer_steps_maximum": maximum_optimizer_steps,
            "optimizer_epochs_stopped_early_by_target_kl": optimizer_steps < maximum_optimizer_steps,
            "ppo": {
                "n_steps": ROLLOUT_STEPS,
                "batch_size": BATCH_SIZE,
                "n_epochs": PPO_EPOCHS,
                "gamma": 0.98,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "target_kl": 0.06,
                "learning_rate": LEARNING_RATE,
                "ent_coef": ENTROPY_COEFFICIENT,
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
            },
            "first_rollout_actor_gradient": callback.first_gradient,
            "initial_A_vector": initial_a.tolist(),
            "final_A_vector": final_a.tolist(),
            "A_update_vector": update_vector,
            "A_update_l2_norm": float(torch.linalg.vector_norm(final_a - initial_a)),
            "inactive_A_l2_norms": inactive,
            "inactive_A_slots_unchanged": inactive_zero,
            "initial_actor_hash": initial_actor_hash,
            "final_actor_hash": actor_hash(core),
            "actor_hash_unchanged": actor_hash(core) == initial_actor_hash,
            "initial_full_policy_hash": initial_state_hash,
            "final_full_policy_hash": _jsonable_state_hash(core),
            "source_basis_trainable": bool(core.source_basis.requires_grad),
            "source_basis_unchanged": torch.equal(initial_source_basis, core.source_basis.detach().cpu()),
            "storage_report": core.storage_report(),
            "policy_tensor_bytes": sum(
                int(tensor.numel() * tensor.element_size())
                for tensor in core.state_dict().values()
            ),
            "optimizer_tensor_bytes": _tensor_bytes(optimizer.state_dict()),
            "allocated_output_mixer_slots": 6,
            "active_output_mixer_slots": [ROUTE_ID],
            "rss_samples": callback.rss_samples,
            "max_sampled_rss_bytes": max(
                (int(sample["rss_bytes"]) for sample in callback.rss_samples),
                default=0,
            ),
            "training_completed_episodes": sum(bool(row["completed"]) for row in training_episodes),
            "training_episode_records": len(training_episodes),
            "training_episode_file": str(train_episode_path.relative_to(ROOT)),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_bytes": int(checkpoint_path.stat().st_size),
            "elapsed_wall_s": elapsed_wall,
            "elapsed_process_cpu_s": elapsed_cpu,
        }
        (branch_dir / "branch.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    finally:
        env.close()
        del model, core, optimizer, callback


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_tensor_bytes(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(child) for child in value)
    return 0


def make_eval_plan(seed: int, target: str, episodes: int) -> dict[str, Any]:
    """Make one common reset/action-randomness plan per seed and target."""
    if seed < 0:
        raise ValueError("seeds must be nonnegative")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    target = context_name(target)
    target_index = context_index(target)
    # Fixed, collision-free seed ranges for the frozen deployment seeds and
    # six targets. The same plan is reused across all source candidates.
    seed_base = int(seed) * 1_000_000 + (target_index + 1) * 10_000
    reset_seeds = [seed_base + episode for episode in range(episodes)]
    action_seed = seed_base + 8_000
    action_rng = np.random.default_rng(action_seed)
    uniforms = action_rng.random((episodes, MAX_EPISODE_STEPS), dtype=np.float64)
    return {
        "target": target,
        "target_index": target_index,
        "reset_seeds": [int(value) for value in reset_seeds],
        "action_uniform_seed": action_seed,
        "action_uniforms": uniforms,
        "uniform_generator": "numpy.default_rng(PCG64), precomputed episodes x 64 float64 uniforms",
        "reset_seed_generator": "seed * 1,000,000 + (target_index + 1) * 10,000 + episode_index",
    }


def evaluate_episodes(
    core,
    *,
    target: str,
    plan: dict[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    """Evaluate complete episodes using common resets and inverse-CDF uniforms."""
    target = context_name(target)
    policy_hash_before = _jsonable_state_hash(core)
    env = C1NavigationEnv(c1_contexts()[target])
    results: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for episode_index, reset_seed in enumerate(plan["reset_seeds"]):
                observation, _ = env.reset(seed=int(reset_seed))
                initial = np.asarray(observation, dtype=np.float32).copy()
                episode_return = 0.0
                success = False
                length = 0
                for step_index in range(MAX_EPISODE_STEPS):
                    policy_observation = torch.as_tensor(
                        augment(observation, ROUTE_ID)[None],
                        dtype=torch.float32,
                    )
                    probabilities = core.get_distribution(policy_observation).distribution.probs[0]
                    probabilities = probabilities.detach().cpu().numpy().astype(np.float64, copy=False)
                    probabilities = probabilities / probabilities.sum()
                    uniform = float(plan["action_uniforms"][episode_index, step_index])
                    action = min(
                        int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")),
                        len(probabilities) - 1,
                    )
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    length += 1
                    if terminated or truncated:
                        success = bool(info.get("success", False))
                        break
                else:
                    raise RuntimeError("C1 environment failed to terminate or truncate by 64 steps")
                results.append({
                    "seed": None,
                    "source": source,
                    "target": target,
                    "episode": episode_index + 1,
                    "reset_seed": int(reset_seed),
                    "action_uniform_seed": int(plan["action_uniform_seed"]),
                    "initial_observation": [float(value) for value in initial],
                    "episode_return": float(episode_return),
                    "success": int(success),
                    "length": int(length),
                    "completed": True,
                })
    finally:
        env.close()
    if _jsonable_state_hash(core) != policy_hash_before:
        raise RuntimeError("no-gradient evaluation changed persistent policy state")
    return results


def _initial_observations_equal(left: Iterable[dict[str, Any]], right: Iterable[dict[str, Any]]) -> bool:
    left_rows = list(left)
    right_rows = list(right)
    return len(left_rows) == len(right_rows) and all(
        row_a["reset_seed"] == row_b["reset_seed"]
        and row_a["initial_observation"] == row_b["initial_observation"]
        for row_a, row_b in zip(left_rows, right_rows)
    )


def _evaluation_csv_row(row: dict[str, Any], *, seed: int, baseline_return: float | None) -> dict[str, Any]:
    initial = row["initial_observation"]
    gain = None if baseline_return is None else float(row["episode_return"] - baseline_return)
    return {
        "seed": seed,
        "source": row["source"],
        "target": row["target"],
        "episode": row["episode"],
        "reset_seed": row["reset_seed"],
        "action_uniform_seed": row["action_uniform_seed"],
        "initial_x": initial[0],
        "initial_y": initial[1],
        "goal_x": initial[2],
        "goal_y": initial[3],
        "episode_return": row["episode_return"],
        "success": row["success"],
        "length": row["length"],
        "completed": row["completed"],
        "baseline_return": baseline_return,
        "paired_gain": gain,
    }


def _aggregate_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([float(row["episode_return"]) for row in rows], dtype=np.float64)
    successes = np.asarray([int(row["success"]) for row in rows], dtype=np.float64)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.float64)
    result = {
        "episodes": int(len(rows)),
        "mean_episode_return": float(returns.mean()) if len(rows) else None,
        "success_rate": float(successes.mean()) if len(rows) else None,
        "mean_episode_length": float(lengths.mean()) if len(rows) else None,
    }
    gains = [row.get("paired_gain") for row in rows]
    gain_values = np.asarray([float(value) for value in gains if value is not None], dtype=np.float64)
    if len(gain_values):
        result.update({
            "mean_paired_return_gain": float(gain_values.mean()),
            "sd_paired_return_gain": float(gain_values.std(ddof=1)) if len(gain_values) > 1 else 0.0,
            "positive_gain_episodes": int(np.count_nonzero(gain_values > 0.0)),
            "negative_gain_episodes": int(np.count_nonzero(gain_values < 0.0)),
            "zero_gain_episodes": int(np.count_nonzero(gain_values == 0.0)),
        })
    return result


def _parse_int_list(value: str) -> list[int]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    return [int(part) for part in values]


def _parse_context_list(value: str) -> list[str]:
    values = [context_name(part.strip()) for part in value.split(",") if part.strip()]
    return values


def _git_value(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_steps <= 0 or args.train_steps % ROLLOUT_STEPS != 0:
        raise ValueError(f"--train-steps must be a positive multiple of {ROLLOUT_STEPS}")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must contain one or more unique integers")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be nonnegative")
    if not args.sources or len(args.sources) != len(set(args.sources)):
        raise ValueError("--sources must contain one or more unique C1 contexts")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    supplement, config = _load_configs()
    formal_default = (
        tuple(args.seeds) == DEFAULT_SEEDS
        and tuple(args.sources) == CONTEXTS
        and args.train_steps == 4096
        and args.episodes == 32
        and not args.smoke
    )
    source_files = [
        ROOT / "docs" / "PRE_CORE_VALIDATION_PROTOCOL.md",
        ROOT / "configs" / "c1.json",
        ROOT / "configs" / "c1_supplement.json",
        ROOT / "data" / "foundation.pt",
        ROOT / "requirements-lock.txt",
        ROOT / "src" / "online_rl" / "c1_envs.py",
        ROOT / "src" / "online_rl" / "c1_experiment.py",
        ROOT / "src" / "online_rl" / "c1_policy.py",
        ROOT / "src" / "online_rl" / "c1_tensor_controls.py",
        ROOT / "src" / "online_rl" / "c1_tensor_experiment.py",
        ROOT / "src" / "online_rl" / "compression.py",
        ROOT / "src" / "online_rl" / "compression_experiment.py",
        ROOT / "src" / "online_rl" / "experiment.py",
        Path(__file__).resolve(),
    ]
    provenance = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "experimenter-isolated C1 local update and cross-context transfer diagnostic",
        "smoke_test": bool(args.smoke),
        "formal_protocol_defaults": formal_default,
        "protocol_git_commit": _git_value(["log", "-1", "--format=%H", "--", "docs/PRE_CORE_VALIDATION_PROTOCOL.md"]),
        "git_head_at_launch": _git_value(["rev-parse", "HEAD"]),
        "protocol_sha256": _sha256(ROOT / "docs" / "PRE_CORE_VALIDATION_PROTOCOL.md"),
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in source_files
            if path.is_file()
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "stable_baselines3": _package_version("stable-baselines3"),
            "gymnasium": _package_version("gymnasium"),
            "numpy": np.__version__,
        },
        "diagnostic_boundaries": [
            "one existing pretrained W0 foundation",
            "context label selects only the fixed simulator during training and the target during evaluation",
            "policy observations always use route 1 and never contain context or family labels",
            "no context transition map, expert actions, or hidden context features enter policy updates",
            "all six output_mixer slots are allocated; only route 1 is active and inactive slots are recorded",
            "results do not measure task-free discovery, dynamic structure, or a budget advantage",
        ],
        "parameters": {
            "seeds": list(args.seeds),
            "sources": list(args.sources),
            "targets": list(CONTEXTS),
            "train_steps_per_source_seed": args.train_steps,
            "evaluation_episodes_per_source_target_seed": args.episodes,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "route": ROUTE_ID,
            "method": "output_mixer",
            "learning_rate": LEARNING_RATE,
            "ent_coef": ENTROPY_COEFFICIENT,
            "temperature": float(supplement.get("temperature", 2.0)),
            "exploration_epsilon": float(config["exploration_epsilon"]),
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    train_results: dict[tuple[int, str], dict[str, Any]] = {}
    baseline_results: dict[tuple[int, str], list[dict[str, Any]]] = {}
    transfer_rows: list[dict[str, Any]] = []
    baseline_csv_rows: list[dict[str, Any]] = []
    transfer_csv_rows: list[dict[str, Any]] = []
    plans = {
        (seed, target): make_eval_plan(seed, target, args.episodes)
        for seed in args.seeds
        for target in CONTEXTS
    }

    for seed in args.seeds:
        baseline_core, baseline_optimizer = _make_source_core(config, seed)
        try:
            for target in CONTEXTS:
                rows = evaluate_episodes(
                    baseline_core,
                    target=target,
                    plan=plans[(seed, target)],
                    source="W0_zero_correction",
                )
                for row in rows:
                    row["seed"] = seed
                    baseline_csv_rows.append(_evaluation_csv_row(row, seed=seed, baseline_return=None))
                baseline_results[(seed, target)] = rows
        finally:
            del baseline_core, baseline_optimizer

    for seed in args.seeds:
        for source in args.sources:
            branch_dir = args.output / "branches" / f"s{seed}" / source
            branch_result = train_source(
                seed=seed,
                source=source,
                train_steps=args.train_steps,
                config=config,
                branch_dir=branch_dir,
                smoke_test=bool(args.smoke),
            )
            train_results[(seed, source)] = branch_result
            core, _optimizer = _make_source_core(config, seed)
            try:
                with torch.no_grad():
                    core._nodes[str(ROUTE_ID)].A.copy_(torch.as_tensor(
                        branch_result["final_A_vector"], dtype=core._nodes[str(ROUTE_ID)].A.dtype
                    ).reshape_as(core._nodes[str(ROUTE_ID)].A))
                for target in CONTEXTS:
                    candidate_rows = evaluate_episodes(
                        core,
                        target=target,
                        plan=plans[(seed, target)],
                        source=source,
                    )
                    baseline_rows = baseline_results[(seed, target)]
                    if not _initial_observations_equal(candidate_rows, baseline_rows):
                        raise RuntimeError(
                            f"baseline/candidate starts diverged for seed {seed}, target {target}, source {source}"
                        )
                    paired_rows = []
                    for candidate, baseline in zip(candidate_rows, baseline_rows):
                        candidate["seed"] = seed
                        row = {
                            **candidate,
                            "baseline_return": float(baseline["episode_return"]),
                            "paired_gain": float(candidate["episode_return"] - baseline["episode_return"]),
                        }
                        paired_rows.append(row)
                        transfer_csv_rows.append(_evaluation_csv_row(
                            candidate,
                            seed=seed,
                            baseline_return=float(baseline["episode_return"]),
                        ))
                    summary = _aggregate_episode_rows(paired_rows)
                    transfer_rows.append({
                        "seed": seed,
                        "source": source,
                        "target": target,
                        "pairing_verified": True,
                        "initial_observations_equal_to_baseline": True,
                        "reset_seeds": [row["reset_seed"] for row in paired_rows],
                        "action_uniform_seed": int(plans[(seed, target)]["action_uniform_seed"]),
                        **summary,
                    })
            finally:
                del core, _optimizer

    training_interactions = sum(int(row["train_steps_actual"]) for row in train_results.values())
    baseline_evaluation_interactions = sum(
        int(row["length"]) for rows in baseline_results.values() for row in rows
    )
    candidate_evaluation_interactions = sum(int(row["length"]) for row in transfer_csv_rows)
    finished_wall = time.perf_counter() - started_wall
    finished_cpu = time.process_time() - started_cpu
    transfer_summary = {
        "rows": transfer_rows,
        "by_seed": {},
        "by_seed_source": {},
        "by_seed_target": {},
    }
    for seed in args.seeds:
        for key, selector in (
            ("by_seed", lambda row, seed=seed: row["seed"] == seed),
        ):
            selected = [row for row in transfer_rows if selector(row)]
            if selected:
                transfer_summary[key][str(seed)] = {
                    "mean_paired_return_gain": float(np.mean([row["mean_paired_return_gain"] for row in selected])),
                    "positive_source_target_cells": int(sum(row["mean_paired_return_gain"] > 0 for row in selected)),
                    "negative_source_target_cells": int(sum(row["mean_paired_return_gain"] < 0 for row in selected)),
                    "cells": len(selected),
                }
    for key, group_fields in (("by_seed_source", ("seed", "source")), ("by_seed_target", ("seed", "target"))):
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in transfer_rows:
            groups.setdefault(tuple(row[field] for field in group_fields), []).append(row)
        transfer_summary[key] = {
            "|".join(map(str, group)): {
                "mean_paired_return_gain": float(np.mean([row["mean_paired_return_gain"] for row in group_rows])),
                "positive_cells": int(sum(row["mean_paired_return_gain"] > 0 for row in group_rows)),
                "negative_cells": int(sum(row["mean_paired_return_gain"] < 0 for row in group_rows)),
                "cells": len(group_rows),
            }
            for group, group_rows in groups.items()
        }

    actual_total_interactions = (
        training_interactions
        + baseline_evaluation_interactions
        + candidate_evaluation_interactions
    )
    max_training_interactions = len(args.seeds) * len(args.sources) * args.train_steps
    max_evaluation_interactions = (
        len(args.seeds)
        * (len(CONTEXTS) + len(args.sources) * len(CONTEXTS))
        * args.episodes
        * MAX_EPISODE_STEPS
    )
    result = {
        "scope": "P0 isolated local update and cross-context transfer; not task-free discovery or a budget benchmark",
        "smoke_test": bool(args.smoke),
        "formal_defaults": formal_default,
        "seeds": list(args.seeds),
        "sources": list(args.sources),
        "targets": list(CONTEXTS),
        "train_steps_per_source_seed": args.train_steps,
        "episodes_per_target_seed": args.episodes,
        "training": list(train_results.values()),
        "transfer_summary": transfer_summary,
        "baseline_batches": len(baseline_results),
        "candidate_batches": len(transfer_rows),
        "baseline_episode_count": len(baseline_csv_rows),
        "candidate_episode_count": len(transfer_csv_rows),
        "training_interactions": training_interactions,
        "baseline_evaluation_interactions": baseline_evaluation_interactions,
        "candidate_evaluation_interactions": candidate_evaluation_interactions,
        "actual_total_interactions": actual_total_interactions,
        "max_training_interactions": max_training_interactions,
        "max_evaluation_interactions": max_evaluation_interactions,
        "max_total_interactions": max_training_interactions + max_evaluation_interactions,
        "elapsed_wall_s": finished_wall,
        "elapsed_process_cpu_s": finished_cpu,
        "output_files": {
            "baseline_episodes": "baseline_episodes.csv",
            "candidate_episodes": "transfer_episodes.csv",
            "transfer_matrix": "transfer_matrix.json",
            "provenance": "provenance.json",
        },
    }
    _write_csv(args.output / "baseline_episodes.csv", EVAL_EPISODE_FIELDS, baseline_csv_rows)
    _write_csv(args.output / "transfer_episodes.csv", EVAL_EPISODE_FIELDS, transfer_csv_rows)
    (args.output / "transfer_matrix.json").write_text(
        json.dumps(transfer_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/pre_core_p0"))
    parser.add_argument("--seeds", type=_parse_int_list, default=list(DEFAULT_SEEDS),
                        help="comma-separated deployment seeds (default: 271,283,307)")
    parser.add_argument("--sources", type=_parse_context_list, default=list(CONTEXTS),
                        help="comma-separated source contexts (default: all six C1 contexts)")
    parser.add_argument("--train-steps", type=int, default=4096,
                        help="PPO interactions per source/seed; must be a multiple of 256")
    parser.add_argument("--episodes", type=int, default=32,
                        help="complete evaluation episodes per target/source/seed")
    parser.add_argument("--smoke", action="store_true",
                        help="mark this reduced run as smoke validation, never formal evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    try:
        result = run(args)
    except Exception as exc:
        parser.exit(1, f"P0 run failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "smoke_test": result["smoke_test"],
        "actual_total_interactions": result["actual_total_interactions"],
        "elapsed_wall_s": result["elapsed_wall_s"],
        "elapsed_process_cpu_s": result["elapsed_process_cpu_s"],
        "candidate_batches": result["candidate_batches"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

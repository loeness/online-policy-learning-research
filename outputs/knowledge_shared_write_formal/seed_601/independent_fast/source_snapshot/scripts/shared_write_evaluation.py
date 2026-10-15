"""Read-only, paired evaluation for policies on the actual C1 environments.

Each episode gets independent, deterministic environment and action seeds. The
action stream is local to that episode, so calling this function does not alter
Python, NumPy, or PyTorch global random-number generators. The policy is run in
evaluation mode under ``torch.no_grad()`` and its original module training flags
are restored afterward.

``evaluate_named_cores`` is a small convenience for evaluating already-loaded
named policy objects with the same episode seeds. It does not load or clone
checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from operator import index as as_index
from typing import Any, Iterator, Protocol

import numpy as np
import torch

from online_rl.c1_envs import C1Context, C1NavigationEnv, C1_CONTEXTS
from online_rl.c1_envs import context_name as canonical_context_name


class PolicyCore(Protocol):
    """Minimal C1 policy interface used by the evaluator."""

    def get_distribution(self, obs: torch.Tensor) -> Any: ...

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor: ...


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = as_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _route_id(value: int) -> int:
    result = _nonnegative_int(value, "route_id")
    if result > 8:
        raise ValueError("route_id must be in [0, 8]")
    return result


def _episode_seeds(seed: int, episodes: int) -> list[tuple[int, int]]:
    """Derive unique reset/action seed pairs without touching global RNG state."""

    streams = np.random.SeedSequence(seed).spawn(episodes * 2)
    used: set[int] = set()
    seeds: list[int] = []
    for stream in streams:
        candidate = int(stream.generate_state(1, dtype=np.uint64)[0])
        # A collision is vanishingly unlikely, but deterministic adjustment
        # makes distinct reset/action streams an invariant of the API.
        while candidate in used:
            candidate = (candidate + 1) % (1 << 64)
        used.add(candidate)
        seeds.append(candidate)
    return [(seeds[2 * index], seeds[2 * index + 1]) for index in range(episodes)]


@contextmanager
def _evaluation_mode(core: Any) -> Iterator[None]:
    """Temporarily disable training behavior and restore every module flag."""

    modules_method = getattr(core, "modules", None)
    eval_method = getattr(core, "eval", None)
    if not callable(modules_method) or not callable(eval_method):
        raise TypeError("core must be a torch module with eval() and modules()")
    modules = list(modules_method())
    modes = [(module, bool(module.training)) for module in modules]
    try:
        eval_method()
        yield
    finally:
        # Assign flags directly to retain mixed train/eval states in children.
        for module, training in modes:
            module.training = training


def _core_device(core: Any) -> torch.device:
    parameters_method = getattr(core, "parameters", None)
    if callable(parameters_method):
        try:
            return next(parameters_method()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _observation13(observation: np.ndarray, route_id: int) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (4,):
        raise ValueError(f"C1 environment observations must have shape (4,), got {observation.shape}")
    result = np.zeros(13, dtype=np.float32)
    result[:4] = observation
    result[12] = route_id
    return result


def _probabilities(core: PolicyCore, obs: torch.Tensor) -> np.ndarray:
    policy_distribution = core.get_distribution(obs)
    distribution = getattr(policy_distribution, "distribution", None)
    probabilities = getattr(distribution, "probs", None)
    if probabilities is None:
        raise TypeError("core.get_distribution(obs) must expose distribution.probs")
    if torch.is_tensor(probabilities):
        probabilities = probabilities.detach().cpu().numpy()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape == (1, 4):
        probabilities = probabilities[0]
    if probabilities.shape != (4,) or not np.isfinite(probabilities).all():
        raise ValueError("the policy must return four finite action probabilities")
    if (probabilities < 0.0).any():
        raise ValueError("the policy returned a negative action probability")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("the policy action probabilities must have positive mass")
    return probabilities / total


def _value_prediction(core: PolicyCore, obs: torch.Tensor) -> float:
    prediction = core.predict_values(obs)
    if torch.is_tensor(prediction):
        values = prediction.detach().cpu().numpy()
    else:
        values = np.asarray(prediction)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError("core.predict_values(obs) must return one finite value for one observation")
    return float(values[0])


def evaluate_context(
    core: PolicyCore,
    context_name: str,
    route_id: int,
    *,
    seed: int,
    episodes: int = 8,
    context: C1Context | None = None,
) -> dict[str, Any]:
    """Evaluate a supplied core on one actual C1 context and fixed route.

    ``context_name`` must be one of the externally named contexts in
    ``C1_CONTEXTS``. Pass ``context`` to select an explicit ``C1Context`` spec;
    this is intended for future context registries and does not change the
    public context label in the returned metrics.

    Every episode is reset with its own recorded ``reset_seed`` and samples
    actions from the core's actual categorical probabilities with a separate
    ``np.random.default_rng(action_seed)``. Temperature and epsilon are already
    reflected in ``get_distribution`` and are not applied a second time. The
    environment owns the 64-step episode limit. Episode rows include the full
    action sequence, return, success flag, length, and observer visit counters.
    """

    if not callable(getattr(core, "get_distribution", None)):
        raise TypeError("core must implement get_distribution(obs)")
    if not callable(getattr(core, "predict_values", None)):
        raise TypeError("core must implement predict_values(obs)")
    label = canonical_context_name(context_name)
    context_spec = C1_CONTEXTS[label] if context is None else context
    if not isinstance(context_spec, C1Context):
        raise TypeError("context must be a C1Context or None")
    route = _route_id(route_id)
    evaluation_seed = _nonnegative_int(seed, "seed")
    episode_count = _nonnegative_int(episodes, "episodes")
    if episode_count == 0:
        raise ValueError("episodes must be positive")

    seed_pairs = _episode_seeds(evaluation_seed, episode_count)
    device = _core_device(core)
    rows: list[dict[str, Any]] = []

    with _evaluation_mode(core), torch.no_grad():
        for episode_index, (reset_seed, action_seed) in enumerate(seed_pairs, start=1):
            env = C1NavigationEnv(context_spec)
            action_rng = np.random.default_rng(action_seed)
            try:
                observation, _ = env.reset(seed=reset_seed)
                start_observation = [float(value) for value in observation]
                episode_return = 0.0
                actions: list[int] = []
                predicted_values: list[float] = []
                success = False
                final_observation = observation

                while True:
                    obs13 = _observation13(observation, route)
                    obs_tensor = torch.as_tensor(obs13[None, :], dtype=torch.float32, device=device)
                    probabilities = _probabilities(core, obs_tensor)
                    predicted_values.append(_value_prediction(core, obs_tensor))
                    action = int(action_rng.choice(4, p=probabilities))
                    actions.append(action)
                    final_observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    observation = final_observation
                    if terminated or truncated:
                        success = bool(info["success"])
                        break

                counters = env.observer_counters()
                steps = len(actions)
                rows.append(
                    {
                        "episode": episode_index,
                        "context": label,
                        "route_id": route,
                        "reset_seed": reset_seed,
                        "action_seed": action_seed,
                        "start_observation": start_observation,
                        "final_observation": [float(value) for value in final_observation],
                        "actions": actions,
                        "steps": steps,
                        "return": episode_return,
                        "success": success,
                        "mean_predicted_value": float(np.mean(predicted_values)),
                        "observer_region_visits": counters,
                    }
                )
            finally:
                env.close()

    total_steps = sum(row["steps"] for row in rows)
    mean_return = float(np.mean([row["return"] for row in rows]))
    mean_success = float(np.mean([float(row["success"]) for row in rows]))
    return {
        "context": label,
        "context_spec": {"family": context_spec.family, "variant": context_spec.variant},
        "route_id": route,
        "seed": evaluation_seed,
        "episodes": rows,
        "total_steps": total_steps,
        "mean_return": mean_return,
        "mean_success": mean_success,
        "success_rate": mean_success,
        "gradient_updates": 0,
    }


def evaluate_named_cores(
    cores: Mapping[str, PolicyCore],
    context_name: str,
    route_id: int,
    *,
    seed: int,
    episodes: int = 8,
    context: C1Context | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate named, already-loaded cores on the same deterministic plan.

    Each call to :func:`evaluate_context` derives identical reset/action seeds,
    making this suitable for paired policy comparisons. The mapping values are
    used as supplied; checkpoint loading and model cloning belong to callers.
    """

    return {
        name: evaluate_context(
            core,
            context_name,
            route_id,
            seed=seed,
            episodes=episodes,
            context=context,
        )
        for name, core in cores.items()
    }


__all__ = ["PolicyCore", "evaluate_context", "evaluate_named_cores"]

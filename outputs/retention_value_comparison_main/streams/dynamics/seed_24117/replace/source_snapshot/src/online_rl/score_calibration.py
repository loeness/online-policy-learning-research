"""Paired, frozen-policy evaluation helpers for navigation calibration.

Rollouts use common random numbers: candidate and baseline select actions with
the same uniform variate at every raw timestep and use the same seeded reset
stream for corresponding reset ordinals. No learner is involved in this
module.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from operator import index as as_index
from typing import Any, Callable, Mapping

import numpy as np

from .write_decision_envs import TaskSpec, WriteDecisionNavigationEnv, make_env
from .write_decision_policy import decode_actor, encode_actor


_GRID_SIZE = 9
_ACTION_COUNT = 4
_OBSERVATION_COUNT = _GRID_SIZE**4
_DISCOUNT = 0.98
_DISCOUNTED_STEPS = 64
_ACTION_SEED_STREAM = 0xA17
_RESET_SEED_STREAM = 0x5E7


def _json_value(value: Any) -> Any:
    """Convert NumPy and tuple values into JSON-serializable primitives."""

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


def _state_hash(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def snapshot_navigation(env: WriteDecisionNavigationEnv) -> dict[str, Any]:
    """Return a JSON-friendly snapshot of a navigation task's complete state.

    Detour environments are deliberately rejected because their transition
    geometry differs from the navigation domain used by this evaluator.
    """

    if not isinstance(env, WriteDecisionNavigationEnv) or env.domain != "navigation":
        raise TypeError("env must be a WriteDecisionNavigationEnv in the navigation domain")
    if len(env._observer_region_visits) != 3:
        raise ValueError("navigation observer counters must contain three bands")

    return {
        "schema_version": 1,
        "domain": "navigation",
        "task": {
            "family": env.context.family,
            "variant": int(env.context.variant),
            "preference": int(env._preference),
        },
        "grid_size": int(env.grid_size),
        "max_steps": int(env.max_steps),
        "position": [int(value) for value in env._position],
        "goal": [int(value) for value in env._goal],
        "episode_length": int(env._episode_length),
        "has_reset": bool(env._has_reset),
        "observer_region_visits": [int(value) for value in env._observer_region_visits],
        "np_random_bit_generator": type(env.np_random.bit_generator).__name__,
        "np_random_state": _json_value(deepcopy(env.np_random.bit_generator.state)),
    }


def _snapshot_int(snapshot: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = snapshot.get(key)
    if isinstance(value, bool):
        raise ValueError(f"snapshot {key} must be an integer >= {minimum}")
    try:
        result = as_index(value)
    except TypeError as exc:
        raise ValueError(f"snapshot {key} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"snapshot {key} must be an integer >= {minimum}")
    return int(result)


def _snapshot_cell(snapshot: Mapping[str, Any], key: str, grid_size: int) -> tuple[int, int]:
    value = snapshot.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"snapshot {key} must contain two grid coordinates")
    coordinates: list[int] = []
    for coordinate in value:
        if isinstance(coordinate, bool):
            raise ValueError(f"snapshot {key} coordinates must be integers inside the grid")
        try:
            integer = as_index(coordinate)
        except TypeError as exc:
            raise ValueError(f"snapshot {key} coordinates must be integers inside the grid") from exc
        if not 0 <= integer < grid_size:
            raise ValueError(f"snapshot {key} coordinates must be integers inside the grid")
        coordinates.append(int(integer))
    return coordinates[0], coordinates[1]


def restore_navigation(snapshot: Mapping[str, Any]) -> WriteDecisionNavigationEnv:
    """Reconstruct a real navigation environment from :func:`snapshot_navigation`.

    ``reset`` only initializes Gymnasium's RNG object and spaces. Every captured
    field, including its RNG state and counters, is then overwritten from the
    snapshot.
    """

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    if snapshot.get("schema_version") != 1 or snapshot.get("domain") != "navigation":
        raise ValueError("snapshot must be a version 1 navigation snapshot")
    task = snapshot.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("snapshot task must be a mapping")
    spec = TaskSpec(task.get("family"), task.get("variant"), task.get("preference"))

    grid_size = _snapshot_int(snapshot, "grid_size", minimum=2)
    if grid_size != _GRID_SIZE:
        raise ValueError(f"navigation snapshots must use grid_size {_GRID_SIZE}")
    max_steps = _snapshot_int(snapshot, "max_steps", minimum=1)
    position = _snapshot_cell(snapshot, "position", grid_size)
    goal = _snapshot_cell(snapshot, "goal", grid_size)
    episode_length = _snapshot_int(snapshot, "episode_length", minimum=0)
    has_reset = snapshot.get("has_reset")
    if not isinstance(has_reset, bool):
        raise ValueError("snapshot has_reset must be a boolean")
    raw_visits = snapshot.get("observer_region_visits")
    if not isinstance(raw_visits, (list, tuple)) or len(raw_visits) != 3:
        raise ValueError("snapshot observer_region_visits must contain three counters")
    visits: list[int] = []
    for value in raw_visits:
        if isinstance(value, bool):
            raise ValueError("snapshot observer counters must be non-negative integers")
        try:
            counter = as_index(value)
        except TypeError as exc:
            raise ValueError("snapshot observer counters must be non-negative integers") from exc
        if counter < 0:
            raise ValueError("snapshot observer counters must be non-negative integers")
        visits.append(int(counter))

    rng_state = snapshot.get("np_random_state")
    if not isinstance(rng_state, Mapping):
        raise ValueError("snapshot np_random_state must be a mapping")
    bit_generator_name = snapshot.get("np_random_bit_generator")
    if not isinstance(bit_generator_name, str):
        raise ValueError("snapshot np_random_bit_generator must be a string")
    if rng_state.get("bit_generator") != bit_generator_name:
        raise ValueError("snapshot bit generator name does not match its RNG state")

    env = make_env(spec, domain="navigation")
    try:
        env.reset(seed=0)
        if type(env.np_random.bit_generator).__name__ != bit_generator_name:
            raise ValueError(f"unsupported navigation bit generator: {bit_generator_name}")
        env.grid_size = grid_size
        env.max_steps = max_steps
        env._position = position
        env._goal = goal
        env._episode_length = episode_length
        env._has_reset = has_reset
        env._observer_region_visits = visits
        env.np_random.bit_generator.state = deepcopy(dict(rng_state))
        return env
    except BaseException:
        env.close()
        raise


def grid_observations() -> np.ndarray:
    """Return all ``9**4`` normalized navigation observations with route 1."""

    observations = np.zeros((_OBSERVATION_COUNT, 13), dtype=np.float32)
    row = 0
    denominator = np.float32(_GRID_SIZE - 1)
    for x in range(_GRID_SIZE):
        for y in range(_GRID_SIZE):
            for goal_x in range(_GRID_SIZE):
                for goal_y in range(_GRID_SIZE):
                    observations[row, :4] = (
                        x / denominator,
                        y / denominator,
                        goal_x / denominator,
                        goal_y / denominator,
                    )
                    observations[row, 12] = 1.0
                    row += 1
    return observations


def probability_index(obs4: Any) -> int:
    """Map one normalized four-coordinate observation to its table row."""

    try:
        observation = np.asarray(obs4, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("obs4 must contain four normalized grid coordinates") from exc
    if observation.shape != (4,) or not np.isfinite(observation).all():
        raise ValueError("obs4 must contain four finite normalized grid coordinates")
    if np.any(observation < 0.0) or np.any(observation > 1.0):
        raise ValueError("obs4 coordinates must be in [0, 1]")
    scaled = observation * (_GRID_SIZE - 1)
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, rtol=0.0, atol=1e-6):
        raise ValueError("obs4 coordinates must lie on the 9 by 9 grid")
    x, y, goal_x, goal_y = (int(value) for value in rounded)
    return (((x * _GRID_SIZE + y) * _GRID_SIZE + goal_x) * _GRID_SIZE) + goal_y


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = as_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _seed_value(seed: Any) -> int:
    if isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer")
    try:
        result = as_index(seed)
    except TypeError as exc:
        raise ValueError("seed must be a non-negative integer") from exc
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(result)


def _probability_table(value: Any, name: str) -> np.ndarray:
    try:
        table = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric probability table") from exc
    expected_shape = (_OBSERVATION_COUNT, _ACTION_COUNT)
    if table.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.isfinite(table).all() or np.any(table < 0.0):
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    row_sums = np.sum(table, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"each row in {name} must sum to one")
    return table / row_sums[:, None]


def _current_observation(env: Any) -> np.ndarray:
    observation = np.asarray(env._observation(), dtype=np.float32)
    if observation.shape != (4,) or not np.isfinite(observation).all():
        raise ValueError("navigation environment returned an invalid observation")
    return observation


def _rng_hash(env: Any) -> str | None:
    rng = getattr(env, "np_random", None)
    bit_generator = getattr(rng, "bit_generator", None)
    state = getattr(bit_generator, "state", None)
    return None if state is None else _state_hash(state)


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _reset_seed(seed: int, replicate: int, ordinal: int) -> int:
    sequence = np.random.SeedSequence([seed, _RESET_SEED_STREAM, replicate, ordinal])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _action_index(table: np.ndarray, observation: np.ndarray, uniform: float) -> int:
    row = probability_index(observation[:4])
    cumulative = np.cumsum(table[row])
    return min(int(np.searchsorted(cumulative, uniform, side="right")), _ACTION_COUNT - 1)


def _step_one(env: Any, table: np.ndarray, observation: np.ndarray, uniform: float) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
    action = _action_index(table, observation, uniform)
    result = env.step(action)
    if not isinstance(result, tuple) or len(result) != 5:
        raise ValueError("navigation step must return the Gymnasium five-value result")
    next_observation, reward, terminated, truncated, info = result
    next_observation = np.asarray(next_observation, dtype=np.float32)
    if next_observation.shape != (4,) or not np.isfinite(next_observation).all():
        raise ValueError("navigation environment returned an invalid observation")
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise ValueError("navigation environment returned a non-finite reward")
    if not isinstance(info, Mapping):
        raise ValueError("navigation info must be a mapping")
    return next_observation, reward_value, bool(terminated), bool(truncated), info


def _goal_reached(terminated: bool, info: Mapping[str, Any]) -> bool:
    if "goal_reached" in info:
        return bool(info["goal_reached"])
    if "success" in info:
        return bool(info["success"])
    return terminated


def paired_rollouts(
    snapshot: Mapping[str, Any],
    candidate_table: Any,
    baseline_table: Any,
    *,
    horizon: int,
    replicates: int,
    seed: int,
    env_factory: Callable[[Mapping[str, Any]], Any] = restore_navigation,
) -> dict[str, Any]:
    """Evaluate frozen policies with paired actions and reset streams.

    Tables contain one four-action probability distribution for each row from
    :func:`grid_observations`. ``step_calls`` is exactly ``2 * horizon *
    replicates`` when evaluation completes. Differences are per replicate;
    this function computes no inferential statistic.
    """

    raw_horizon = _positive_integer(horizon, "horizon")
    replicate_count = _positive_integer(replicates, "replicates")
    master_seed = _seed_value(seed)
    candidate_probabilities = _probability_table(candidate_table, "candidate_table")
    baseline_probabilities = _probability_table(baseline_table, "baseline_table")
    if not callable(env_factory):
        raise TypeError("env_factory must be callable")

    per_replicate: list[dict[str, Any]] = []
    total_step_calls = 0
    for replicate in range(replicate_count):
        candidate_env = env_factory(deepcopy(snapshot))
        baseline_env: Any | None = None
        try:
            baseline_env = env_factory(deepcopy(snapshot))
            candidate_observation = _current_observation(candidate_env)
            baseline_observation = _current_observation(baseline_env)
            candidate_first_observation = candidate_observation.astype(float).tolist()
            baseline_first_observation = baseline_observation.astype(float).tolist()
            candidate_rng_hash = _rng_hash(candidate_env)
            baseline_rng_hash = _rng_hash(baseline_env)

            action_rng = np.random.default_rng(
                np.random.SeedSequence([master_seed, _ACTION_SEED_STREAM, replicate])
            )
            returns = {"candidate": 0.0, "baseline": 0.0}
            discounted_returns = {"candidate": 0.0, "baseline": 0.0}
            goals = {"candidate": 0, "baseline": 0}
            reset_counts = {"candidate": 0, "baseline": 0}
            reset_seeds = {"candidate": [], "baseline": []}
            done = {"candidate": False, "baseline": False}
            observations = {
                "candidate": candidate_observation,
                "baseline": baseline_observation,
            }
            envs = {"candidate": candidate_env, "baseline": baseline_env}
            tables = {
                "candidate": candidate_probabilities,
                "baseline": baseline_probabilities,
            }
            for raw_step in range(raw_horizon):
                for arm in ("candidate", "baseline"):
                    if done[arm]:
                        ordinal = reset_counts[arm] + 1
                        reset_seed = _reset_seed(master_seed, replicate, ordinal)
                        observation, _ = envs[arm].reset(seed=reset_seed)
                        observation = np.asarray(observation, dtype=np.float32)
                        if observation.shape != (4,) or not np.isfinite(observation).all():
                            raise ValueError("navigation reset returned an invalid observation")
                        observations[arm] = observation
                        done[arm] = False
                        reset_counts[arm] += 1
                        reset_seeds[arm].append(reset_seed)

                uniform = float(action_rng.random())
                for arm in ("candidate", "baseline"):
                    observation, reward, terminated, truncated, info = _step_one(
                        envs[arm], tables[arm], observations[arm], uniform
                    )
                    observations[arm] = observation
                    returns[arm] += reward
                    if raw_step < _DISCOUNTED_STEPS:
                        discounted_returns[arm] += (_DISCOUNT**raw_step) * reward
                    goals[arm] += int(_goal_reached(terminated, info))
                    done[arm] = terminated or truncated
                    total_step_calls += 1

            candidate_return = float(returns["candidate"])
            baseline_return = float(returns["baseline"])
            candidate_discounted = float(discounted_returns["candidate"])
            baseline_discounted = float(discounted_returns["baseline"])
            candidate_per_step = candidate_return / raw_horizon
            baseline_per_step = baseline_return / raw_horizon
            per_replicate.append(
                {
                    "replicate": replicate,
                    "candidate": {
                        "undiscounted_return": candidate_return,
                        "discounted_first64_return": candidate_discounted,
                        "raw_steps": raw_horizon,
                        "return_per_raw_step": candidate_per_step,
                        "goals": goals["candidate"],
                        "post_termination_reset_count": reset_counts["candidate"],
                        "post_termination_reset_seeds": list(reset_seeds["candidate"]),
                        "first_observation": candidate_first_observation,
                        "initial_rng_state_sha256": candidate_rng_hash,
                    },
                    "baseline": {
                        "undiscounted_return": baseline_return,
                        "discounted_first64_return": baseline_discounted,
                        "raw_steps": raw_horizon,
                        "return_per_raw_step": baseline_per_step,
                        "goals": goals["baseline"],
                        "post_termination_reset_count": reset_counts["baseline"],
                        "post_termination_reset_seeds": list(reset_seeds["baseline"]),
                        "first_observation": baseline_first_observation,
                        "initial_rng_state_sha256": baseline_rng_hash,
                    },
                    "differences": {
                        "undiscounted_return": candidate_return - baseline_return,
                        "discounted_first64_return": candidate_discounted - baseline_discounted,
                        "return_per_raw_step": candidate_per_step - baseline_per_step,
                        "goals": goals["candidate"] - goals["baseline"],
                    },
                    "audit": {
                        "first_observations_equal": bool(
                            np.array_equal(
                                np.asarray(candidate_first_observation, dtype=np.float32),
                                np.asarray(baseline_first_observation, dtype=np.float32),
                            )
                        ),
                        "initial_rng_state_equal": (
                            None if candidate_rng_hash is None or baseline_rng_hash is None
                            else candidate_rng_hash == baseline_rng_hash
                        ),
                        "paired_action_uniform_per_raw_step": True,
                        "action_uniform_draws": raw_horizon,
                        "corresponding_reset_seeds_equal": (
                            reset_seeds["candidate"][:min(
                                len(reset_seeds["candidate"]), len(reset_seeds["baseline"])
                            )]
                            == reset_seeds["baseline"][:min(
                                len(reset_seeds["candidate"]), len(reset_seeds["baseline"])
                            )]
                        ),
                        "full_reset_sequences_equal": (
                            reset_seeds["candidate"] == reset_seeds["baseline"]
                        ),
                    },
                }
            )
        finally:
            _close_env(candidate_env)
            if baseline_env is not None:
                _close_env(baseline_env)

    expected_calls = 2 * raw_horizon * replicate_count
    if total_step_calls != expected_calls:
        raise RuntimeError(f"internal rollout step count mismatch: {total_step_calls} != {expected_calls}")
    return {
        "horizon": raw_horizon,
        "replicates": replicate_count,
        "discount_gamma": _DISCOUNT,
        "discounted_steps": min(raw_horizon, _DISCOUNTED_STEPS),
        "step_calls": total_step_calls,
        "per_replicate": per_replicate,
    }


def roundtrip_common(matrix3x3: Any) -> tuple[np.ndarray, dict[str, float]]:
    """Round-trip one shared 3 by 3 matrix through the int8 actor codec.

    The encoded actor has one zero local slot, two shared rows containing the
    input matrix and zeros, and slot zero assigned to shared group zero. Returns
    the decoded float32 matrix and its maximum absolute and root-mean-square
    quantization errors.
    """

    try:
        matrix = np.asarray(matrix3x3, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix3x3 must be a finite numeric matrix with shape (3, 3)") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix3x3 must be a finite numeric matrix with shape (3, 3)")
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    encoded = encode_actor(
        np.zeros((1, 27), dtype=np.float32),
        np.stack((matrix.reshape(9), np.zeros(9, dtype=np.float32))),
        np.asarray([0], dtype=np.int8),
        codec="int8",
    )
    _, decoded_shared, _ = decode_actor(encoded.data)
    decoded = decoded_shared[0].reshape(3, 3).astype(np.float32, copy=True)
    error = decoded.astype(np.float64) - matrix.astype(np.float64)
    diagnostics = {
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }
    return decoded, diagnostics


__all__ = [
    "grid_observations",
    "paired_rollouts",
    "probability_index",
    "restore_navigation",
    "roundtrip_common",
    "snapshot_navigation",
]

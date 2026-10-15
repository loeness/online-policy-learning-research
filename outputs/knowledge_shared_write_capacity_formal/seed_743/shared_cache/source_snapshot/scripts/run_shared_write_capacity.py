"""Run the controlled six-slot shared-write cache capacity diagnostic."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import torch

from online_rl.shared_write_capacity_env import (
    CAPACITY_CONTEXTS,
    CapacityContext,
    SharedWriteCapacityEnv,
)
from online_rl.shared_write_storage import decode_shared_knowledge, encode_shared_knowledge
from scripts import run_knowledge_loop as loop
from scripts import run_shared_write as shared_write_base


CONFIG_PATH = ROOT / "configs/shared_write_capacity.json"
SLOT_COUNT = 6


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_factory_condition(condition: str) -> str:
    return {
        "independent_cache": "independent",
        "independent_fast_cache": "independent_fast",
        "shared_cache": "oracle_active",
        "frozen_shared_cache": "oracle_active",
    }[condition]


def make_capacity_agent(condition: str, seed: int, config: Mapping[str, Any]):
    """Use the S1 policy/optimizer factory, then clear its C1 routing template."""

    if condition not in config["conditions"]:
        raise ValueError(f"unknown capacity condition: {condition!r}")
    factory_condition = _base_factory_condition(condition)
    agent, _template_slots, base = shared_write_base.make_agent(
        factory_condition, seed, dict(config)
    )
    # S1 starts with a context-to-slot template. Capacity routing begins empty;
    # only current residents are represented in the new bounded cache object.
    for slot in range(SLOT_COUNT):
        agent.core.set_slot_group(slot, -1, preserve_effective=False)
    agent.core.freeze_shared(False)
    agent.slot_generation = np.zeros(SLOT_COUNT, dtype=np.int64)
    agent.counts.update({
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_evictions": 0,
        "cache_local_optimizer_resets": 0,
        "quantization_commits": 0,
        "actor_optimizer_state_resets": 0,
    })
    cache = CapacityCache()
    agent.condition = condition
    agent.cache = cache
    return agent, base


class CapacityCache:
    """Six-key resident map with fixed-size LRU and generation arrays."""

    def __init__(self) -> None:
        self.resident_by_context: dict[str, int] = {}
        self.last_used = [0] * SLOT_COUNT
        self.generations = [0] * SLOT_COUNT
        self.clock = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "resident_by_context": dict(sorted(self.resident_by_context.items())),
            "last_used": list(self.last_used),
            "generations": list(self.generations),
            "clock": int(self.clock),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CapacityCache":
        required = {"resident_by_context", "last_used", "generations", "clock"}
        if set(state) != required:
            raise ValueError("cache checkpoint must contain only resident keys, LRU clocks, generations, and clock")
        residents = state["resident_by_context"]
        if not isinstance(residents, Mapping) or len(residents) > SLOT_COUNT:
            raise ValueError("resident cache must be a mapping with at most six entries")
        normalized: dict[str, int] = {}
        for name, slot_value in residents.items():
            if name not in CAPACITY_CONTEXTS:
                raise ValueError(f"unknown resident context key: {name!r}")
            if isinstance(slot_value, bool) or not isinstance(slot_value, (int, np.integer)):
                raise ValueError("resident slot values must be integers")
            slot = int(slot_value)
            if not 0 <= slot < SLOT_COUNT:
                raise ValueError("resident slot values must be in [0, 5]")
            normalized[str(name)] = slot
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("a cache slot cannot hold more than one resident context")

        last_used = cls._int_array(state["last_used"], "last_used", minimum=0)
        generations = cls._int_array(state["generations"], "generations", minimum=0)
        if len(last_used) != SLOT_COUNT or len(generations) != SLOT_COUNT:
            raise ValueError("LRU clocks and generations must each have six entries")
        clock = state["clock"]
        if isinstance(clock, bool) or not isinstance(clock, (int, np.integer)) or int(clock) < 0:
            raise ValueError("cache clock must be a nonnegative integer")
        if int(clock) < max(last_used, default=0):
            raise ValueError("cache clock must be at least the newest LRU clock")

        result = cls()
        result.resident_by_context = normalized
        result.last_used = last_used
        result.generations = generations
        result.clock = int(clock)
        return result

    @staticmethod
    def _int_array(values: Any, name: str, *, minimum: int) -> list[int]:
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple")
        result: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} entries must be integers")
            number = int(value)
            if number < minimum:
                raise ValueError(f"{name} entries must be at least {minimum}")
            result.append(number)
        return result

    def ensure_resident(
        self,
        core: Any,
        condition: str,
        context_name: str,
        *,
        optimizer: Any | None = None,
    ) -> dict[str, Any]:
        """Touch a hit or allocate/evict one slot for a controlled context."""

        if context_name not in CAPACITY_CONTEXTS:
            raise ValueError(f"unknown capacity context: {context_name!r}")
        if condition not in ("independent_cache", "independent_fast_cache", "shared_cache", "frozen_shared_cache"):
            raise ValueError(f"unknown capacity condition: {condition!r}")

        if context_name in self.resident_by_context:
            slot = self.resident_by_context[context_name]
            self.clock += 1
            self.last_used[slot] = self.clock
            return {
                "action": "hit",
                "context": context_name,
                "slot": slot,
                "generation": self.generations[slot],
                "evicted_context": None,
                "group": core.group_for_slot(slot),
            }

        occupied = set(self.resident_by_context.values())
        free = [slot for slot in range(SLOT_COUNT) if slot not in occupied]
        if free:
            slot = min(free)
            evicted = None
        else:
            slot = min(range(SLOT_COUNT), key=lambda candidate: (self.last_used[candidate], candidate))
            evicted = next(name for name, resident_slot in self.resident_by_context.items() if resident_slot == slot)
            del self.resident_by_context[evicted]

        core.reset_slot(slot)
        local_parameter = core._nodes[str(slot + 1)].A
        local_state_reset = bool(optimizer is not None and optimizer.state.pop(local_parameter, None) is not None)
        self.generations[slot] += 1
        self.clock += 1
        self.last_used[slot] = self.clock
        context = CAPACITY_CONTEXTS[context_name]
        group = -1
        if condition in ("shared_cache", "frozen_shared_cache"):
            group = 0 if context.family == "A" else 1
        core.set_slot_group(slot, group, preserve_effective=False)
        self.resident_by_context[context_name] = slot
        return {
            "action": "miss",
            "context": context_name,
            "slot": slot,
            "generation": self.generations[slot],
            "evicted_context": evicted,
            "group": group,
            "local_optimizer_state_reset": local_state_reset,
        }


def _actor_parameters(core: Any) -> list[torch.nn.Parameter]:
    locals_ = [core._nodes[str(slot + 1)].A for slot in range(SLOT_COUNT)]
    return [*locals_, *core.shared_parameters()]


def _quantize_and_install(agent: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    core = agent.core
    packed = encode_shared_knowledge(
        core.read_local_matrix(),
        core.read_shared_matrix(),
        core.slot_groups.detach().cpu().numpy(),
        codec=config["codec"],
        budget_bytes=config["actor_budget_bytes"],
    )
    local, shared, groups = decode_shared_knowledge(packed.data)
    core.install_local_matrix(local)
    core.install_shared_matrix(shared)
    with torch.no_grad():
        core.slot_groups.copy_(torch.as_tensor(groups, dtype=torch.long, device=core.slot_groups.device))

    actor_parameters = _actor_parameters(core)
    reset_count = 0
    for parameter in actor_parameters:
        if agent.optimizer.state.pop(parameter, None) is not None:
            reset_count += 1
    agent.counts["quantization_commits"] += 1
    agent.counts["actor_optimizer_state_resets"] += reset_count
    expected_bytes = 204 if not core.shared_groups else 230
    if packed.nbytes != expected_bytes:
        raise RuntimeError(f"expected {expected_bytes} actor bytes, encoded {packed.nbytes}")
    return {
        "actor_bytes": packed.nbytes,
        "actor_budget_bytes": int(config["actor_budget_bytes"]),
        "codec": config["codec"],
        "payload_sha256": hashlib.sha256(packed.data).hexdigest(),
        "actor_optimizer_states_reset": reset_count,
        "critic_optimizer_state_count": sum(
            1 for group in agent.optimizer.param_groups if group.get("name") == "critic"
            for parameter in group["params"] if parameter in agent.optimizer.state
        ),
    }


def save_checkpoint(
    agent: Any,
    folder: Path,
    checkpoint_name: str,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    """Save full stage-end state and check the actual W0 + checkpoint bytes."""

    packed = encode_shared_knowledge(
        agent.core.read_local_matrix(),
        agent.core.read_shared_matrix(),
        agent.core.slot_groups.detach().cpu().numpy(),
        codec=config["codec"],
        budget_bytes=config["actor_budget_bytes"],
    )
    state = agent.cache.state_dict()
    generations = np.asarray(state["generations"], dtype=np.int64)
    if not np.array_equal(agent.slot_generation, generations):
        raise RuntimeError("agent generation array and cache generations diverged")
    foundation = Path(base["foundation"])
    checkpoint = {
        "format": "shared_write_capacity_v1",
        "condition": agent.condition,
        "seed": int(agent.seed),
        "actor_binary": packed.data,
        "critic": loop._critic_state(agent.core),
        "optimizer": agent.optimizer.state_dict(),
        "slot_groups": agent.core.slot_groups.detach().cpu().clone(),
        "shared_frozen": bool(agent.core._shared_frozen),
        "cache_state": state,
        "slot_generation": agent.slot_generation.copy(),
        "counts": dict(agent.counts),
        "action_rng": copy.deepcopy(agent.action_rng.bit_generator.state),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
        "w0_sha256": sha(foundation),
        "scope": "Stage-end state; no in-flight environment or partial rollout.",
    }
    target = folder / f"checkpoint_{checkpoint_name}.pt"
    torch.save(checkpoint, target)
    (folder / f"actor_{checkpoint_name}.bin").write_bytes(packed.data)
    foundation_bytes = foundation.stat().st_size
    whole_package_bytes = foundation_bytes + target.stat().st_size
    if whole_package_bytes > int(config["whole_package_cap_bytes"]):
        raise RuntimeError(
            f"W0 plus checkpoint uses {whole_package_bytes} bytes, over "
            f"{config['whole_package_cap_bytes']}"
        )
    return {
        "checkpoint": target.name,
        "actor_bytes": packed.nbytes,
        "checkpoint_bytes": target.stat().st_size,
        "w0_bytes": foundation_bytes,
        "whole_package_bytes": whole_package_bytes,
        "whole_package_cap_bytes": int(config["whole_package_cap_bytes"]),
    }


def restore_capacity_run(folder: Path, checkpoint_name: str = "final"):
    """Restore a stage-end checkpoint and reject malformed/unbounded cache state."""

    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    payload = torch.load(folder / f"checkpoint_{checkpoint_name}.pt", map_location="cpu", weights_only=False)
    condition = payload.get("condition")
    if condition not in config["conditions"]:
        raise ValueError("checkpoint condition is not in the frozen configuration")
    agent, _base = make_capacity_agent(condition, int(payload["seed"]), config)
    base = loop.load_base_config()
    if payload.get("w0_sha256") != sha(Path(base["foundation"])):
        raise ValueError("checkpoint W0 foundation hash does not match the local foundation")
    local, shared, groups = decode_shared_knowledge(payload["actor_binary"])
    agent.core.install_local_matrix(local)
    agent.core.install_shared_matrix(shared)
    with torch.no_grad():
        installed_groups = torch.as_tensor(groups, dtype=torch.long, device=agent.core.slot_groups.device)
        if not torch.equal(installed_groups, payload["slot_groups"].to(dtype=torch.long)):
            raise ValueError("checkpoint actor assignments disagree with the explicit slot groups")
        agent.core.slot_groups.copy_(installed_groups)
    agent.core.freeze_shared(bool(payload["shared_frozen"]))
    loop._load_critic_state(agent.core, payload["critic"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    agent.counts.update(payload["counts"])
    agent.action_rng.bit_generator.state = payload["action_rng"]
    agent.cache = CapacityCache.from_state_dict(payload["cache_state"])
    generations = np.asarray(agent.cache.generations, dtype=np.int64)
    if not np.array_equal(generations, np.asarray(payload["slot_generation"], dtype=np.int64)):
        raise ValueError("checkpoint cache and agent generations disagree")
    agent.slot_generation = generations.copy()
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    return agent, config, base


def _snapshot_sources(folder: Path, config: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    relative_paths = list(loop.SOURCE_SNAPSHOT_FILES) + [Path(path) for path in (
        "scripts/run_shared_write.py",
        "scripts/shared_write_evaluation.py",
        "scripts/run_shared_write_capacity.py",
        "src/online_rl/shared_write_capacity_env.py",
        "src/online_rl/shared_write_policy.py",
        "src/online_rl/shared_write_storage.py",
        "configs/shared_write.json",
        "configs/shared_write_capacity.json",
        "docs/SHARED_WRITE_PROTOCOL.md",
        "docs/SHARED_WRITE_CAPACITY_PROTOCOL.md",
        "tests/test_shared_write_capacity.py",
    )]
    manifest: dict[str, str] = {}
    for relative in relative_paths:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        destination = folder / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[relative.as_posix()] = sha(destination)
    write_json(folder / "source_manifest.json", {
        "sources": manifest,
        "w0_sha256": sha(Path(base["foundation"])),
        "dataset_sha256": sha(Path(base["dataset"])),
    })
    write_json(folder / "config.json", dict(config))
    write_json(folder / "external_context_schedule.json", {
        "contexts": {
            name: {"family": spec.family, "band_variant": spec.variant, "local_mode": spec.local_mode}
            for name, spec in CAPACITY_CONTEXTS.items()
        },
        "bout_sequence": list(config["bout_sequence"]),
        "visibility": "Outer cache manager only; policy receives route slot, never context label or family.",
    })


def _episode_seed_pairs(seed: int, episodes: int) -> list[tuple[int, int]]:
    streams = np.random.SeedSequence(int(seed)).spawn(episodes * 2)
    values = [int(stream.generate_state(1, dtype=np.uint64)[0]) for stream in streams]
    return [(values[2 * index], values[2 * index + 1]) for index in range(episodes)]


@contextmanager
def _evaluation_mode(core: Any) -> Iterator[None]:
    modules = list(core.modules())
    modes = [(module, bool(module.training)) for module in modules]
    try:
        core.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def evaluate_capacity_context(
    core: Any,
    context_name: str,
    slot: int,
    *,
    seed: int,
    episodes: int,
) -> dict[str, Any]:
    """Evaluate one named controlled context without gradients or RNG mutation."""

    if context_name not in CAPACITY_CONTEXTS:
        raise ValueError(f"unknown capacity context: {context_name!r}")
    if not 0 <= int(slot) < SLOT_COUNT:
        raise ValueError("slot must be in [0, 5]")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    spec: CapacityContext = CAPACITY_CONTEXTS[context_name]
    route_id = int(slot) + 1
    seed_pairs = _episode_seed_pairs(seed, episodes)
    device = next(core.parameters()).device
    rows: list[dict[str, Any]] = []

    with _evaluation_mode(core), torch.no_grad():
        for episode_index, (reset_seed, action_seed) in enumerate(seed_pairs, start=1):
            env = SharedWriteCapacityEnv(spec)
            action_rng = np.random.default_rng(action_seed)
            try:
                observation, _reset_info = env.reset(seed=reset_seed)
                episode_return = 0.0
                actions: list[int] = []
                success = False
                while True:
                    policy_observation = loop.augment_observation(observation, route_id)
                    tensor = torch.as_tensor(policy_observation[None, :], dtype=torch.float32, device=device)
                    distribution = core.get_distribution(tensor).distribution
                    probabilities = distribution.probs.detach().cpu().numpy().reshape(4).astype(np.float64)
                    action = int(action_rng.choice(4, p=probabilities / probabilities.sum()))
                    actions.append(action)
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    if terminated or truncated:
                        success = bool(info["success"])
                        break
                rows.append({
                    "episode": episode_index,
                    "context": context_name,
                    "family": spec.family,
                    "band_variant": spec.variant,
                    "local_mode": spec.local_mode,
                    "route_id": route_id,
                    "reset_seed": reset_seed,
                    "action_seed": action_seed,
                    "actions": actions,
                    "steps": len(actions),
                    "return": episode_return,
                    "success": success,
                    "observer_region_visits": env.observer_counters(),
                })
            finally:
                env.close()

    total_steps = sum(row["steps"] for row in rows)
    return {
        "context": context_name,
        "context_spec": {"family": spec.family, "band_variant": spec.variant, "local_mode": spec.local_mode},
        "route_id": route_id,
        "seed": int(seed),
        "episodes": rows,
        "total_steps": total_steps,
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "gradient_updates": 0,
    }


def _evaluation_seed_for(seed: int, context_name: str) -> int:
    index = tuple(CAPACITY_CONTEXTS).index(context_name)
    return int(seed) + 50000 + index * 1000


def _record_evaluation(
    evaluation: dict[str, Any],
    *,
    stage: str,
    bout: int | str,
    condition: str,
    writer: csv.DictWriter,
) -> dict[str, Any]:
    for row in evaluation["episodes"]:
        writer.writerow({
            "stage": stage,
            "bout": bout,
            "condition": condition,
            "context": row["context"],
            "family": row["family"],
            "band_variant": row["band_variant"],
            "local_mode": row["local_mode"],
            "route_id": row["route_id"],
            "episode": row["episode"],
            "reset_seed": row["reset_seed"],
            "action_seed": row["action_seed"],
            "return": row["return"],
            "success": int(row["success"]),
            "steps": row["steps"],
            "actions": json.dumps(row["actions"]),
            "observer_region_visits": json.dumps(row["observer_region_visits"], sort_keys=True),
            "gradient_updates": 0,
        })
    return {
        "context": evaluation["context"],
        "mean_return": evaluation["mean_return"],
        "success_rate": evaluation["success_rate"],
        "environment_steps": evaluation["total_steps"],
        "gradient_updates": 0,
    }


def _evaluation_slot(agent: Any, name: str) -> tuple[Any, CapacityCache, int, dict[str, Any]]:
    """Clone one identical endpoint state and apply this target's cache lookup."""

    core_copy = copy.deepcopy(agent.core)
    cache_copy = CapacityCache.from_state_dict(agent.cache.state_dict())
    event = cache_copy.ensure_resident(core_copy, agent.condition, name)
    return core_copy, cache_copy, int(event["slot"]), event


def run_one(condition: str, seed: int, config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    folder = Path(output) / f"seed_{seed}" / condition
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Preserve existing results: {folder}")
    folder.mkdir(parents=True)
    run_config = {**dict(config), "condition": condition, "seed": int(seed)}
    agent, base = make_capacity_agent(condition, seed, run_config)
    _snapshot_sources(folder, run_config, base)

    fields = {
        "steps.csv": ["step", "bout", "context", "slot", "generation", "group", "action", "reward", "terminated", "truncated", "log_prob", "policy_updates", "phase"],
        "episodes.csv": ["step", "bout", "context", "return", "success", "length", "completed"],
        "evaluations.csv": ["stage", "bout", "condition", "context", "family", "band_variant", "local_mode", "route_id", "episode", "reset_seed", "action_seed", "return", "success", "steps", "actions", "observer_region_visits", "gradient_updates"],
        "cache_events.csv": ["bout", "context", "action", "slot", "generation", "evicted_context", "group", "local_optimizer_state_reset", "resident_count", "resident_keys", "lru_clock", "last_used", "generations"],
    }
    streams = {name: (folder / name).open("w", newline="", encoding="utf-8") for name in fields}
    writers = {name: csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore") for name, (stream, columns) in ((name, (streams[name], fields[name])) for name in fields)}
    for writer in writers.values():
        writer.writeheader()

    evaluation_episodes = int(config["pilot_evaluation_episodes"] if config.get("phase") == "pilot" else config["formal_evaluation_episodes"])
    bout_steps = int(config["pilot_bout_steps"] if config.get("phase") == "pilot" else config["formal_bout_steps"])
    if bout_steps % 1024 or bout_steps % loop.ROLLOUT_STEPS:
        raise ValueError("bout step count must be divisible by 1,024 and PPO rollout length")
    sequence = list(config["bout_sequence"])
    if any(name not in CAPACITY_CONTEXTS for name in sequence):
        raise ValueError("bout sequence contains an unknown capacity context")

    total_eval_steps = 0
    training_reward = 0.0
    online_successes = 0
    online_complete_episodes = 0
    capacity_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}
    started, cpu_started = time.perf_counter(), time.process_time()
    global_step = 0
    try:
        for bout_index, context_name in enumerate(sequence, start=1):
            context = CAPACITY_CONTEXTS[context_name]
            allocation = agent.cache.ensure_resident(
                agent.core, condition, context_name, optimizer=agent.optimizer
            )
            slot = int(allocation["slot"])
            generation = int(allocation["generation"])
            agent.slot_generation[slot] = generation
            if allocation["action"] == "hit":
                agent.counts["cache_hits"] += 1
            else:
                agent.counts["cache_misses"] += 1
                agent.counts["cache_local_optimizer_resets"] += int(allocation.get("local_optimizer_state_reset", False))
                if allocation["evicted_context"] is not None:
                    agent.counts["cache_evictions"] += 1
            cache_record = {
                "bout": bout_index,
                **allocation,
                "resident_count": len(agent.cache.resident_by_context),
                "resident_keys": json.dumps(sorted(agent.cache.resident_by_context)),
                "lru_clock": agent.cache.clock,
                "last_used": json.dumps(agent.cache.last_used),
                "generations": json.dumps(agent.cache.generations),
            }
            writers["cache_events.csv"].writerow(cache_record)
            cache_rows.append(cache_record)

            eval_seed = _evaluation_seed_for(seed, context_name)
            pre = evaluate_capacity_context(agent.core, context_name, slot, seed=eval_seed, episodes=evaluation_episodes)
            pre_key = f"bout_{bout_index:02d}_pre"
            evaluations[pre_key] = _record_evaluation(
                pre, stage="pre_bout", bout=bout_index, condition=condition,
                writer=writers["evaluations.csv"],
            )
            total_eval_steps += int(pre["total_steps"])

            env = SharedWriteCapacityEnv(context)
            observation, _ = env.reset(seed=seed + 12000 + bout_index * 1000)
            rollout: list[Any] = []
            episode_return, episode_length = 0.0, 0
            shared_before = agent.core.read_shared_matrix().copy()
            for offset in range(bout_steps):
                policy_observation = loop.augment_observation(observation, slot + 1)
                tensor = torch.as_tensor(policy_observation[None, :], dtype=torch.float32)
                with torch.no_grad():
                    probabilities = agent.core.get_distribution(tensor).distribution.probs.detach().cpu().numpy().reshape(4)
                    value = float(agent.core.predict_values(tensor).item())
                action, log_prob = loop._sample_action(probabilities, agent.action_rng)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                next_tensor = torch.as_tensor(
                    loop.augment_observation(next_observation, slot + 1)[None, :], dtype=torch.float32
                )
                with torch.no_grad():
                    next_value = float(agent.core.predict_values(next_tensor).item())
                global_step += 1
                rollout.append(loop.Transition(
                    policy_observation,
                    action,
                    float(reward),
                    value,
                    next_value,
                    log_prob,
                    probabilities,
                    done,
                    bool(terminated),
                    slot,
                    generation,
                    global_step,
                ))
                training_reward += float(reward)
                episode_return += float(reward)
                episode_length += 1
                agent.counts["environment_steps"] = global_step
                phase = "fit" if loop.rollout_number_at_step(offset + 1) <= 3 else "holdout"
                writers["steps.csv"].writerow({
                    "step": global_step,
                    "bout": bout_index,
                    "context": context_name,
                    "slot": slot,
                    "generation": generation,
                    "group": agent.core.group_for_slot(slot),
                    "action": action,
                    "reward": float(reward),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "log_prob": log_prob,
                    "policy_updates": agent.counts["minibatch_updates"],
                    "phase": phase,
                })

                if len(rollout) == loop.ROLLOUT_STEPS:
                    if loop.rollout_number_at_step(offset + 1) <= 3:
                        loop._ppo_update(agent, rollout)
                        agent.counts["fit_steps"] += len(rollout)
                    else:
                        agent.counts["holdout_steps"] += len(rollout)
                    rollout = []

                if done:
                    online_successes += int(info["success"])
                    online_complete_episodes += 1
                    writers["episodes.csv"].writerow({
                        "step": global_step,
                        "bout": bout_index,
                        "context": context_name,
                        "return": episode_return,
                        "success": int(info["success"]),
                        "length": episode_length,
                        "completed": True,
                    })
                    episode_return, episode_length = 0.0, 0
                    observation, _ = env.reset()
                else:
                    observation = next_observation

                if (offset + 1) % 1024 == 0:
                    storage = _quantize_and_install(agent, run_config)
                    cycle_name = f"step_{global_step:06d}"
                    checkpoint = save_checkpoint(agent, folder, cycle_name, run_config, base)
                    row = {"step": global_step, **storage, **checkpoint}
                    capacity_rows.append(row)

            if rollout:
                raise RuntimeError("bout ended with a partial PPO rollout")
            if episode_length:
                writers["episodes.csv"].writerow({
                    "step": global_step,
                    "bout": bout_index,
                    "context": context_name,
                    "return": episode_return,
                    "success": 0,
                    "length": episode_length,
                    "completed": False,
                })
            env.close()

            if condition == "frozen_shared_cache" and bout_index == int(config["freeze_after_bouts"]):
                agent.core.freeze_shared(True)
            if bout_index == 6:
                stage_checkpoint = save_checkpoint(agent, folder, "boot", run_config, base)
                capacity_rows.append({"step": global_step, "stage": "boot", **stage_checkpoint})
            if bout_index == 12:
                stage_checkpoint = save_checkpoint(agent, folder, "seen", run_config, base)
                capacity_rows.append({"step": global_step, "stage": "seen", **stage_checkpoint})

            post = evaluate_capacity_context(agent.core, context_name, slot, seed=eval_seed, episodes=evaluation_episodes)
            post_key = f"bout_{bout_index:02d}_post"
            evaluations[post_key] = _record_evaluation(
                post, stage="post_bout", bout=bout_index, condition=condition,
                writer=writers["evaluations.csv"],
            )
            total_eval_steps += int(post["total_steps"])

            shared_after = agent.core.read_shared_matrix()
            print(json.dumps({
                "seed": seed,
                "condition": condition,
                "bout": bout_index,
                "context": context_name,
                "steps": global_step,
                "pre_success": pre["success_rate"],
                "post_success": post["success_rate"],
                "shared_delta_l2": float(np.linalg.norm(shared_after - shared_before)),
            }), flush=True)

        final_sizes = save_checkpoint(agent, folder, "final", run_config, base)
        # Each endpoint target starts from the identical final cache/core copy;
        # a cold miss is simulated only on that private copy.
        endpoint_metrics: dict[str, Any] = {}
        endpoint_cache_events: list[dict[str, Any]] = []
        for context_name in CAPACITY_CONTEXTS:
            eval_core, eval_cache, eval_slot, event = _evaluation_slot(agent, context_name)
            endpoint_cache_events.append({"context": context_name, **event})
            endpoint = evaluate_capacity_context(
                eval_core,
                context_name,
                eval_slot,
                seed=_evaluation_seed_for(seed, context_name),
                episodes=evaluation_episodes,
            )
            endpoint_metrics[context_name] = _record_evaluation(
                endpoint,
                stage="endpoint",
                bout="endpoint",
                condition=condition,
                writer=writers["evaluations.csv"],
            )
            total_eval_steps += int(endpoint["total_steps"])
            del eval_core, eval_cache
        write_json(folder / "capacity.json", capacity_rows)
        write_json(folder / "cache_events.json", cache_rows)
        write_json(folder / "endpoint_cache_events.json", endpoint_cache_events)
        result = {
            "condition": condition,
            "seed": int(seed),
            "counts": dict(agent.counts),
            "training_reward": training_reward,
            "reward_per_training_step": training_reward / max(1, global_step),
            "online_success_rate": online_successes / max(1, online_complete_episodes),
            "online_completed_episodes": online_complete_episodes,
            "training_steps": global_step,
            "evaluation_steps": total_eval_steps,
            "total_interactions": global_step + total_eval_steps,
            "bout_steps": bout_steps,
            "bout_count": len(sequence),
            "conditions_actor_bytes": 204 if condition.startswith("independent") else 230,
            "actor_budget_bytes": int(config["actor_budget_bytes"]),
            "whole_package_cap_bytes": int(config["whole_package_cap_bytes"]),
            "final_checkpoint": final_sizes,
            "endpoint_evaluations": endpoint_metrics,
            "bout_evaluations": evaluations,
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "scope": "Known context IDs and family groups; six-slot LRU diagnostic, not unknown-context organization.",
        }
        write_json(folder / "summary.json", result)
        return result
    finally:
        for stream in streams.values():
            stream.close()


def run(phase: str, output: Path, workers: int = 1) -> dict[str, Any]:
    if phase not in ("pilot", "formal"):
        raise ValueError("phase must be pilot or formal")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["phase"] = phase
    seeds = [int(config["validation_seed"])] if phase == "pilot" else [int(value) for value in config["formal_seeds"]]
    if workers not in (1, 2) or workers > int(config["max_workers"]):
        raise ValueError("workers must be one or two")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve completed or partial cohort: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_config.json", config)
    tasks = [(condition, seed, config, output) for seed in seeds for condition in config["conditions"]]
    records: list[dict[str, Any]] = []
    if workers == 1:
        records = [run_one(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, int(config["max_workers"]))) as executor:
            pending = [executor.submit(run_one, *task) for task in tasks]
            for future in as_completed(pending):
                records.append(future.result())
    summary = {
        "phase": phase,
        "seeds": seeds,
        "conditions": list(config["conditions"]),
        "records": records,
        "training_steps": sum(record["training_steps"] for record in records),
        "evaluation_steps": sum(record["evaluation_steps"] for record in records),
        "total_interactions": sum(record["total_interactions"] for record in records),
        "scope": "The controlled diagnostic assumes outer visibility of context IDs and family groups.",
    }
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    arguments = parser.parse_args()
    run(arguments.phase, arguments.output, arguments.workers)

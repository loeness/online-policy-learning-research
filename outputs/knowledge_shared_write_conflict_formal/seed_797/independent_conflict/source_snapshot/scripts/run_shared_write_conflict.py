"""Run a controlled C1 same-dynamics reward-conflict experiment.

This module defines the runner but starts no experiment when imported. It is
an implementation diagnostic for shared writes, not a general routing or
safety claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import torch

from online_rl.shared_write_conflict_env import MODES, SharedWriteConflictEnv
from online_rl.shared_write_grouping import DynamicsGrouper, empty_profile, observe_profile
from online_rl.shared_write_storage import decode_shared_knowledge, encode_shared_knowledge
from scripts import run_knowledge_loop as loop
from scripts import run_shared_write as shared_write_base


CONFIG_PATH = ROOT / "configs" / "shared_write_conflict.json"
BASE_CONFIG_PATH = ROOT / "configs" / "shared_write.json"
PROTOCOL_PATH = ROOT / "docs" / "SHARED_WRITE_CONFLICT_PROTOCOL.md"
CONDITION_FACTORY = {
    "independent_conflict": "independent",
    "shared_conflict": "oracle_active",
    "frozen_conflict": "oracle_active",
}
SLOT_FOR_MODE = {"plus": 0, "minus": 1}
STAGES = ("initial", "after_plus", "after_minus")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def make_agent(condition: str, seed: int, config: Mapping[str, Any]):
    """Build an S1 policy through its factory, then install this test's slots."""
    if condition not in CONDITION_FACTORY:
        raise ValueError(f"unknown conflict condition: {condition!r}")
    agent, _base_slots, base = shared_write_base.make_agent(
        CONDITION_FACTORY[condition], seed, dict(config)
    )
    core = agent.core
    # The two shared matrices remain allocated in shared arms. Only group 0 is
    # used; group 1 remains zero and is still charged in the actor payload.
    for slot in range(6):
        core.set_slot_group(slot, -1, preserve_effective=False)
    if condition != "independent_conflict":
        core.set_slot_group(SLOT_FOR_MODE["plus"], 0, preserve_effective=False)
        core.set_slot_group(SLOT_FOR_MODE["minus"], 0, preserve_effective=False)
    agent.condition = condition
    agent.mode_to_slot = dict(SLOT_FOR_MODE)
    agent.grouper = DynamicsGrouper()
    agent.profiles = {mode: empty_profile() for mode in MODES}
    agent.group_decisions: dict[str, dict[str, Any]] = {}
    agent.counts.update({
        "probe_steps": 0,
        "evaluation_steps": 0,
        "training_steps": 0,
        "goal_reaches_plus": 0,
        "episodes_plus": 0,
        "goal_reaches_minus": 0,
        "episodes_minus": 0,
    })
    return agent, base


def build_probe_profile(rows: list[Mapping[str, Any]]) -> np.ndarray:
    """Build a profile from transitions, reading no reward or route metadata."""
    profile = empty_profile()
    for row in rows:
        observe_profile(profile, row["obs"], int(row["action"]), row["next_obs"])
    return profile


def _probe_seed_pair(seed: int) -> tuple[int, int]:
    return seed + 71_000, seed + 72_000


def collect_uniform_probe(
    mode: str, seed: int, steps: int
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Collect fixed-seed uniform interactions without any learner updates."""
    if mode not in MODES:
        raise ValueError(f"unknown reward mode: {mode!r}")
    if steps < 1:
        raise ValueError("probe steps must be positive")
    reset_seed, action_seed = _probe_seed_pair(seed)
    action_rng = np.random.default_rng(action_seed)
    env = SharedWriteConflictEnv(mode)
    observation, _ = env.reset(seed=reset_seed)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(steps):
            action = int(action_rng.integers(0, 4))
            next_observation, reward, terminated, truncated, info = env.step(action)
            rows.append({
                "step": index + 1,
                "mode": mode,
                "obs": np.asarray(observation, dtype=np.float32).copy(),
                "action": action,
                "reward": float(reward),
                "next_obs": np.asarray(next_observation, dtype=np.float32).copy(),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "goal_reached": bool(info["success"]),
                "interaction_cost": 1,
            })
            if terminated or truncated:
                observation, _ = env.reset()
            else:
                observation = next_observation
    finally:
        env.close()
    return rows, build_probe_profile(rows)


def _assert_probe_pair_equal(
    plus_rows: list[Mapping[str, Any]], minus_rows: list[Mapping[str, Any]]
) -> None:
    if len(plus_rows) != len(minus_rows):
        raise AssertionError("paired probe streams have different lengths")
    for plus, minus in zip(plus_rows, minus_rows):
        for field in ("action", "terminated", "truncated"):
            if plus[field] != minus[field]:
                raise AssertionError(f"paired probe {field} differs at step {plus['step']}")
        for field in ("obs", "next_obs"):
            if not np.array_equal(plus[field], minus[field]):
                raise AssertionError(f"paired probe {field} differs at step {plus['step']}")
        if not np.isclose(float(minus["reward"]), -float(plus["reward"]), rtol=0.0, atol=0.0):
            raise AssertionError(f"paired probe reward is not exactly reversed at step {plus['step']}")


def prepare_probe_groups(
    agent: Any,
    profiles: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Save both profiles; group only shared arms using arrays as input."""
    if set(profiles) != set(MODES):
        raise ValueError("profiles must contain exactly plus and minus")
    agent.profiles = {
        mode: np.asarray(profiles[mode], dtype=np.uint16).copy() for mode in MODES
    }
    decisions: dict[str, dict[str, Any]] = {}
    if agent.condition == "independent_conflict":
        for mode in MODES:
            decisions[mode] = {
                "group": -1,
                "reason": "not_applicable_independent",
                "profile_sha256": hashlib.sha256(agent.profiles[mode].tobytes()).hexdigest(),
                "scores": {},
            }
    else:
        grouper = DynamicsGrouper()
        for mode in MODES:
            # No mode, name, reward, or task metadata reaches the grouper.
            decision = grouper.assign(agent.profiles[mode])
            if decision.group != 0:
                raise RuntimeError(
                    f"{mode} probe profile did not resolve to group 0: "
                    f"{decision.reason} (profile support may be insufficient)"
                )
            slot = SLOT_FOR_MODE[mode]
            agent.core.set_slot_group(slot, int(decision.group), preserve_effective=False)
            decisions[mode] = {
                "group": int(decision.group),
                "reason": str(decision.reason),
                "profile_sha256": hashlib.sha256(agent.profiles[mode].tobytes()).hexdigest(),
                "scores": decision.scores,
            }
        agent.grouper = grouper
        if decisions["plus"]["group"] != decisions["minus"]["group"]:
            raise RuntimeError("identical transition profiles did not receive one shared group")
    agent.group_decisions = decisions
    return decisions


def _episode_seed_pairs(seed: int, episodes: int) -> list[tuple[int, int]]:
    """Create deterministic paired reset/action seeds without global RNG use."""
    if episodes < 1:
        raise ValueError("evaluation episodes must be positive")
    streams = np.random.SeedSequence(seed).spawn(episodes * 2)
    generated = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in streams]
    seen: set[int] = set()
    unique: list[int] = []
    for value in generated:
        while value in seen:
            value = (value + 1) % (1 << 64)
        seen.add(value)
        unique.append(value)
    return [(unique[2 * i], unique[2 * i + 1]) for i in range(episodes)]


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


def _action_probabilities(core: Any, policy_observation: np.ndarray) -> np.ndarray:
    observation = torch.as_tensor(policy_observation[None, :], dtype=torch.float32)
    distribution = core.get_distribution(observation).distribution.probs
    probabilities = np.asarray(distribution.detach().cpu().numpy(), dtype=np.float64).reshape(4)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("policy returned invalid action probabilities")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("policy returned zero action probability mass")
    return probabilities / total


def _evaluate_mode(
    agent: Any,
    mode: str,
    seed_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    slot = SLOT_FOR_MODE[mode]
    rows: list[dict[str, Any]] = []
    env = SharedWriteConflictEnv(mode)
    for episode, (reset_seed, action_seed) in enumerate(seed_pairs, start=1):
        action_rng = np.random.default_rng(action_seed)
        observation, _ = env.reset(seed=reset_seed)
        episode_return = 0.0
        actions: list[int] = []
        rewards: list[float] = []
        goal_reached = False
        while True:
            policy_observation = loop.augment_observation(observation, slot + 1)
            probabilities = _action_probabilities(agent.core, policy_observation)
            action = int(action_rng.choice(4, p=probabilities))
            actions.append(action)
            next_observation, reward, terminated, truncated, info = env.step(action)
            reward_value = float(reward)
            rewards.append(reward_value)
            episode_return += reward_value
            observation = next_observation
            if terminated or truncated:
                goal_reached = bool(info["success"])
                break
        rows.append({
            "episode": episode,
            "mode": mode,
            "slot": slot,
            "reset_seed": reset_seed,
            "action_seed": action_seed,
            "return": float(episode_return),
            "goal_reached": goal_reached,
            "task_success": goal_reached if mode == "plus" else None,
            "steps": len(actions),
            "actions": actions,
            "rewards": rewards,
        })
    env.close()
    mean_return = float(np.mean([row["return"] for row in rows]))
    goal_rate = float(np.mean([float(row["goal_reached"]) for row in rows]))
    return {
        "mode": mode,
        "episodes": rows,
        "total_steps": int(sum(row["steps"] for row in rows)),
        "mean_return": mean_return,
        "goal_reach_rate": goal_rate,
        "goal_success_rate": goal_rate if mode == "plus" else None,
    }


def evaluate_stage(
    agent: Any,
    stage: str,
    *,
    episodes: int,
    seed: int,
) -> tuple[dict[str, Any], int]:
    """Evaluate both routes on a repeated paired plan without touching RNG state."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    seed_pairs = _episode_seed_pairs(seed, episodes)
    results: dict[str, Any] = {}
    with _evaluation_mode(agent.core), torch.no_grad():
        for mode in MODES:
            results[mode] = _evaluate_mode(agent, mode, seed_pairs)
    return results, sum(int(results[mode]["total_steps"]) for mode in MODES)


def actor_payload(agent: Any, config: Mapping[str, Any]):
    return encode_shared_knowledge(
        agent.core.read_local_matrix(),
        agent.core.read_shared_matrix(),
        agent.core.slot_groups.detach().cpu().numpy(),
        codec=config["codec"],
        budget_bytes=config["actor_budget_bytes"],
    )


def save_checkpoint(
    agent: Any,
    folder: Path,
    name: str,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, int]:
    """Persist agent state for evaluation/restoration with hard byte checks."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    packed = actor_payload(agent, config)
    checkpoint = {
        "format": "shared_write_conflict_agent_state_v1",
        "condition": agent.condition,
        "seed": int(agent.seed),
        "mode_to_slot": dict(agent.mode_to_slot),
        "actor_binary": packed.data,
        "critic": loop._critic_state(agent.core),
        "optimizer": agent.optimizer.state_dict(),
        "optimizer_step_count": int(getattr(agent.optimizer, "step_count", 0)),
        "slot_groups": agent.core.slot_groups.detach().cpu().clone(),
        "slot_generation": np.asarray(agent.slot_generation, dtype=np.int64).copy(),
        "shared_frozen": bool(agent.core._shared_frozen),
        "counts": copy.deepcopy(agent.counts),
        "grouper": agent.grouper.state_dict(),
        "profiles": {mode: profile.copy() for mode, profile in agent.profiles.items()},
        "group_decisions": copy.deepcopy(agent.group_decisions),
        "action_rng": copy.deepcopy(agent.action_rng.bit_generator.state),
        "python_rng": random.getstate(),
        "numpy_global_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state().clone(),
        "foundation_sha256": _sha(Path(base["foundation"])),
        "dataset_sha256": _sha(Path(base["dataset"])),
        "actor_payload_bytes": packed.nbytes,
        "config": dict(config),
        "source_manifest": copy.deepcopy(getattr(agent, "source_manifest", None)),
        "scope": "Agent-state checkpoint for evaluation/restoration; not exact trajectory continuation.",
    }
    target = folder / f"checkpoint_{name}.pt"
    torch.save(checkpoint, target)
    (folder / f"actor_{name}.bin").write_bytes(packed.data)
    foundation_bytes = Path(base["foundation"]).stat().st_size
    checkpoint_bytes = target.stat().st_size
    package_bytes = foundation_bytes + checkpoint_bytes
    if packed.nbytes > int(config["actor_budget_bytes"]):
        raise AssertionError(f"actor is {packed.nbytes} bytes; cap is {config['actor_budget_bytes']}")
    if package_bytes > int(config["whole_package_cap_bytes"]):
        raise AssertionError(
            f"W0 plus checkpoint is {package_bytes} bytes; cap is {config['whole_package_cap_bytes']}"
        )
    return {
        "actor_bytes": packed.nbytes,
        "checkpoint_bytes": checkpoint_bytes,
        "foundation_bytes": foundation_bytes,
        "whole_package_bytes": package_bytes,
    }


def _source_files() -> list[Path]:
    files = set(loop.SOURCE_SNAPSHOT_FILES)
    files.update(Path(path) for path in (
        "src/online_rl/c1_supplement_experiment.py",
        "src/online_rl/shared_write_conflict_env.py",
        "src/online_rl/shared_write_grouping.py",
        "src/online_rl/shared_write_policy.py",
        "src/online_rl/shared_write_storage.py",
        "scripts/run_knowledge_loop.py",
        "scripts/run_shared_write.py",
        "scripts/run_shared_write_conflict.py",
        "scripts/shared_write_evaluation.py",
        "configs/c1.json",
        "configs/shared_write.json",
        "configs/shared_write_conflict.json",
        "docs/SHARED_WRITE_PROTOCOL.md",
        "docs/SHARED_WRITE_CONFLICT_PROTOCOL.md",
    ))
    return sorted(files, key=lambda path: path.as_posix())


def snapshot_sources(
    folder: Path, config: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    """Capture every repository source file imported or used by this runner."""
    hashes: dict[str, str] = {}
    for relative in _source_files():
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"runner dependency is missing: {relative}")
        destination = Path(folder) / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative.as_posix()] = _sha(destination)
    manifest = {
        "sources_sha256": hashes,
        "foundation_sha256": _sha(Path(base["foundation"])),
        "dataset_sha256": _sha(Path(base["dataset"])),
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "base_config_sha256": _sha(BASE_CONFIG_PATH),
    }
    write_json(Path(folder) / "source_manifest.json", manifest)
    write_json(Path(folder) / "config.json", dict(config))
    return manifest


def restore_run(
    folder: str | Path,
    stage: str = "after_minus",
    *,
    restore_global_rng: bool = True,
):
    """Restore agent state and optional process RNGs, not the environment stream."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    payload = torch.load(
        folder / f"checkpoint_{stage}.pt", map_location="cpu", weights_only=False
    )
    if payload["format"] != "shared_write_conflict_agent_state_v1":
        raise ValueError("unsupported shared-write conflict checkpoint")
    agent, base = make_agent(payload["condition"], int(payload["seed"]), config)
    if payload["foundation_sha256"] != _sha(Path(base["foundation"])):
        raise ValueError("checkpoint foundation hash does not match the configured W0")
    if payload["dataset_sha256"] != _sha(Path(base["dataset"])):
        raise ValueError("checkpoint dataset hash does not match the configured source")
    local, shared, groups = decode_shared_knowledge(payload["actor_binary"])
    agent.core.install_local_matrix(local)
    agent.core.install_shared_matrix(shared)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(groups, dtype=torch.long))
    loop._load_critic_state(agent.core, payload["critic"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    agent.optimizer.step_count = int(payload["optimizer_step_count"])
    agent.core.freeze_shared(bool(payload["shared_frozen"]))
    agent.counts.update(payload["counts"])
    agent.slot_generation = np.asarray(payload["slot_generation"], dtype=np.int64).copy()
    agent.mode_to_slot = {str(mode): int(slot) for mode, slot in payload["mode_to_slot"].items()}
    agent.grouper = DynamicsGrouper.from_state_dict(payload["grouper"])
    agent.profiles = {
        str(mode): np.asarray(profile, dtype=np.uint16).copy()
        for mode, profile in payload["profiles"].items()
    }
    agent.group_decisions = copy.deepcopy(payload["group_decisions"])
    agent.source_manifest = copy.deepcopy(payload.get("source_manifest"))
    agent.action_rng.bit_generator.state = copy.deepcopy(payload["action_rng"])
    if restore_global_rng:
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_global_rng"])
        torch.set_rng_state(payload["torch_cpu_rng"])
    agent.restore_metadata = {
        "actor_bytes": int(payload["actor_payload_bytes"]),
        "checkpoint_bytes": (folder / f"checkpoint_{stage}.pt").stat().st_size,
        "foundation_bytes": Path(base["foundation"]).stat().st_size,
        "whole_package_bytes": (
            (folder / f"checkpoint_{stage}.pt").stat().st_size
            + Path(base["foundation"]).stat().st_size
        ),
    }
    return agent, base


class _RSSSampler:
    """Sample process resident bytes while the isolated run is active."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self._process = psutil.Process()
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._peak = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._peak = max(self._peak, self._process.memory_info().rss)
            except psutil.Error:
                return

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        try:
            self._peak = max(self._peak, self._process.memory_info().rss)
        except psutil.Error:
            pass

    @property
    def peak_bytes(self) -> int:
        return int(self._peak)


def _json_vector(value: np.ndarray) -> str:
    return json.dumps([float(item) for item in value], separators=(",", ":"))


def _train_bout(
    agent: Any,
    mode: str,
    seed: int,
    bout_steps: int,
    folder: Path,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    steps_writer: csv.DictWriter,
    episodes_writer: csv.DictWriter,
    budget_rows: list[dict[str, Any]],
) -> tuple[float, int, int]:
    """Train one fresh on-policy bout and write complete transition records."""
    slot = SLOT_FOR_MODE[mode]
    env = SharedWriteConflictEnv(mode)
    observation, _ = env.reset(seed=seed + 2200 + slot * 10_000)
    rollout: list[Any] = []
    episode_return = 0.0
    episode_length = 0
    reward_total = 0.0
    goal_reaches = 0
    completed_episodes = 0
    bout_index = 0 if mode == "plus" else 1
    try:
        for offset in range(bout_steps):
            policy_observation = loop.augment_observation(observation, slot + 1)
            policy_tensor = torch.as_tensor(policy_observation).reshape(1, -1)
            with torch.no_grad():
                probabilities = (
                    agent.core.get_distribution(policy_tensor)
                    .distribution.probs.detach().cpu().numpy().reshape(4)
                )
                value = float(agent.core.predict_values(policy_tensor).item())
            action, log_prob = loop._sample_action(probabilities, agent.action_rng)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_policy_observation = loop.augment_observation(next_observation, slot + 1)
            with torch.no_grad():
                next_value = float(
                    agent.core.predict_values(
                        torch.as_tensor(next_policy_observation).reshape(1, -1)
                    ).item()
                )

            step = int(agent.counts["environment_steps"]) + 1
            rollout.append(loop.Transition(
                policy_observation,
                int(action),
                float(reward),
                value,
                next_value,
                float(log_prob),
                np.asarray(probabilities, dtype=np.float32).copy(),
                done,
                bool(terminated),
                slot,
                int(agent.slot_generation[slot]),
                step,
            ))
            fit_holdout = "fit" if loop.rollout_number_at_step(step) <= 3 else "holdout"
            if len(rollout) == loop.ROLLOUT_STEPS:
                if fit_holdout == "fit":
                    loop._ppo_update(agent, rollout)
                    agent.counts["fit_steps"] += len(rollout)
                else:
                    agent.counts["holdout_steps"] += len(rollout)
                rollout = []

            agent.counts["environment_steps"] = step
            agent.counts["training_steps"] += 1
            reward_total += float(reward)
            episode_return += float(reward)
            episode_length += 1
            steps_writer.writerow({
                "step": step,
                "bout": bout_index,
                "mode": mode,
                "slot": slot,
                "group": agent.core.group_for_slot(slot),
                "action": int(action),
                "reward": float(reward),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "done": int(done),
                "generation": int(agent.slot_generation[slot]),
                "log_prob": float(log_prob),
                "probabilities": _json_vector(np.asarray(probabilities)),
                "value": value,
                "next_value": next_value,
                "obs": _json_vector(np.asarray(observation)),
                "next_obs": _json_vector(np.asarray(next_observation)),
                "fit_or_holdout": fit_holdout,
                "policy_updates": int(agent.counts["minibatch_updates"]),
            })
            if done:
                goal_reached = bool(info["success"])
                goal_reaches += int(goal_reached)
                completed_episodes += 1
                agent.counts[f"goal_reaches_{mode}"] += int(goal_reached)
                agent.counts[f"episodes_{mode}"] += 1
                episodes_writer.writerow({
                    "step": step,
                    "bout": bout_index,
                    "mode": mode,
                    "return": episode_return,
                    "completed": 1,
                    "goal_reached": int(goal_reached),
                    "task_success": int(goal_reached) if mode == "plus" else "",
                    "length": episode_length,
                })
                episode_return = 0.0
                episode_length = 0
                observation, _ = env.reset()
            else:
                observation = next_observation

            if step % 1024 == 0:
                name = f"train_{step:08d}"
                sizes = save_checkpoint(agent, folder, name, config, base)
                budget_rows.append({"step": step, "checkpoint": name, **sizes})

        if rollout:
            raise AssertionError("bout ended with a partial 256-step PPO rollout")
        if episode_length:
            episodes_writer.writerow({
                "step": int(agent.counts["environment_steps"]),
                "bout": bout_index,
                "mode": mode,
                "return": episode_return,
                "completed": 0,
                "goal_reached": 0,
                "task_success": 0 if mode == "plus" else "",
                "length": episode_length,
            })
    finally:
        env.close()
    return reward_total, goal_reaches, completed_episodes


def _write_probe_rows(writer: csv.DictWriter, rows: list[Mapping[str, Any]]) -> None:
    for row in rows:
        writer.writerow({
            "step": row["step"],
            "mode": row["mode"],
            "obs": _json_vector(np.asarray(row["obs"])),
            "action": row["action"],
            "reward": row["reward"],
            "next_obs": _json_vector(np.asarray(row["next_obs"])),
            "terminated": int(row["terminated"]),
            "truncated": int(row["truncated"]),
            "goal_reached": int(row["goal_reached"]),
            "interaction_cost": row["interaction_cost"],
        })


def _record_evaluation(
    agent: Any,
    stage: str,
    episodes: int,
    seed: int,
    evaluation_writer: csv.DictWriter,
) -> tuple[dict[str, Any], int]:
    metrics, steps = evaluate_stage(agent, stage, episodes=episodes, seed=seed)
    agent.counts["evaluation_steps"] += steps
    for mode in MODES:
        for row in metrics[mode]["episodes"]:
            evaluation_writer.writerow({
                "stage": stage,
                "mode": mode,
                "episode": row["episode"],
                "slot": row["slot"],
                "reset_seed": row["reset_seed"],
                "action_seed": row["action_seed"],
                "return": row["return"],
                "goal_reached": int(row["goal_reached"]),
                "task_success": (
                    int(row["task_success"]) if row["task_success"] is not None else ""
                ),
                "steps": row["steps"],
                "actions": json.dumps(row["actions"], separators=(",", ":")),
                "rewards": json.dumps(row["rewards"], separators=(",", ":")),
            })
    return metrics, steps


def run_one(
    condition: str,
    seed: int,
    config: Mapping[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Run one seed/condition pair. Intended to be launched by the root agent."""
    torch.set_num_threads(1)
    folder = Path(output) / f"seed_{seed}" / condition
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Preserve existing results: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    run_config = {**dict(config), "condition": condition, "seed": int(seed)}
    agent, base = make_agent(condition, seed, run_config)
    rss_sampler = _RSSSampler()
    rss_sampler.start()
    source_manifest = snapshot_sources(folder, run_config, base)
    agent.source_manifest = source_manifest
    write_json(folder / "slot_map.json", agent.mode_to_slot)

    probe_steps = int(run_config["probe_steps_per_mode"])
    raw_probes: dict[str, list[dict[str, Any]]] = {}
    profiles: dict[str, np.ndarray] = {}
    for mode in MODES:
        raw_probes[mode], profiles[mode] = collect_uniform_probe(mode, seed, probe_steps)
        agent.counts["probe_steps"] += len(raw_probes[mode])
    _assert_probe_pair_equal(raw_probes["plus"], raw_probes["minus"])
    group_decisions = prepare_probe_groups(agent, profiles)
    if not np.array_equal(agent.profiles["plus"], agent.profiles["minus"]):
        raise AssertionError("same-dynamics reward modes produced different probe profiles")
    write_json(folder / "group_decisions.json", group_decisions)
    write_json(folder / "probe_profile_hashes.json", {
        mode: hashlib.sha256(agent.profiles[mode].tobytes()).hexdigest()
        for mode in MODES
    })

    probe_fields = [
        "step", "mode", "obs", "action", "reward", "next_obs", "terminated",
        "truncated", "goal_reached", "interaction_cost",
    ]
    step_fields = [
        "step", "bout", "mode", "slot", "group", "action", "reward",
        "terminated", "truncated", "done", "generation", "log_prob",
        "probabilities", "value", "next_value", "obs", "next_obs",
        "fit_or_holdout", "policy_updates",
    ]
    episode_fields = [
        "step", "bout", "mode", "return", "completed", "goal_reached",
        "task_success", "length",
    ]
    evaluation_fields = [
        "stage", "mode", "episode", "slot", "reset_seed", "action_seed",
        "return", "goal_reached", "task_success", "steps", "actions", "rewards",
    ]
    probe_path = folder / "probe_steps.csv"
    steps_path = folder / "steps.csv"
    episodes_path = folder / "episodes.csv"
    evaluations_path = folder / "evaluations.csv"
    budget_rows: list[dict[str, Any]] = []
    evaluation_results: dict[str, Any] = {}
    reward_by_mode = {mode: 0.0 for mode in MODES}
    goal_reaches_by_mode = {mode: 0 for mode in MODES}
    completed_by_mode = {mode: 0 for mode in MODES}
    eval_step_total = 0
    with rss_sampler:
        with probe_path.open("w", newline="", encoding="utf-8") as probe_file, \
             steps_path.open("w", newline="", encoding="utf-8") as steps_file, \
             episodes_path.open("w", newline="", encoding="utf-8") as episodes_file, \
             evaluations_path.open("w", newline="", encoding="utf-8") as eval_file:
            probe_writer = csv.DictWriter(probe_file, fieldnames=probe_fields)
            steps_writer = csv.DictWriter(steps_file, fieldnames=step_fields)
            episodes_writer = csv.DictWriter(episodes_file, fieldnames=episode_fields)
            eval_writer = csv.DictWriter(eval_file, fieldnames=evaluation_fields)
            for writer in (probe_writer, steps_writer, episodes_writer, eval_writer):
                writer.writeheader()
            for mode in MODES:
                _write_probe_rows(probe_writer, raw_probes[mode])

            evaluation_seed = seed + 50_000
            evaluation_results["initial"], cost = _record_evaluation(
                agent,
                "initial",
                int(run_config["evaluation_episodes"]),
                evaluation_seed,
                eval_writer,
            )
            eval_step_total += cost
            budget_rows.append({
                "step": 0,
                "checkpoint": "initial",
                **save_checkpoint(agent, folder, "initial", run_config, base),
            })

            bout_steps = int(run_config["bout_steps"])
            reward_by_mode["plus"], goal_reaches_by_mode["plus"], completed_by_mode["plus"] = (
                _train_bout(
                    agent, "plus", seed, bout_steps, folder, run_config, base,
                    steps_writer, episodes_writer, budget_rows,
                )
            )
            if condition == "frozen_conflict":
                agent.core.freeze_shared(True)
            evaluation_results["after_plus"], cost = _record_evaluation(
                agent,
                "after_plus",
                int(run_config["evaluation_episodes"]),
                evaluation_seed,
                eval_writer,
            )
            eval_step_total += cost
            budget_rows.append({
                "step": int(agent.counts["training_steps"]),
                "checkpoint": "after_plus",
                **save_checkpoint(agent, folder, "after_plus", run_config, base),
            })

            reward_by_mode["minus"], goal_reaches_by_mode["minus"], completed_by_mode["minus"] = (
                _train_bout(
                    agent, "minus", seed, bout_steps, folder, run_config, base,
                    steps_writer, episodes_writer, budget_rows,
                )
            )
            evaluation_results["after_minus"], cost = _record_evaluation(
                agent,
                "after_minus",
                int(run_config["evaluation_episodes"]),
                evaluation_seed,
                eval_writer,
            )
            eval_step_total += cost
            budget_rows.append({
                "step": int(agent.counts["training_steps"]),
                "checkpoint": "after_minus",
                **save_checkpoint(agent, folder, "after_minus", run_config, base),
            })

    # Preserve every periodic checkpoint's measured actor and full-package cost.
    with (folder / "capacity.csv").open("w", newline="", encoding="utf-8") as capacity_file:
        fields = [
            "step", "checkpoint", "actor_bytes", "checkpoint_bytes",
            "foundation_bytes", "whole_package_bytes",
        ]
        writer = csv.DictWriter(capacity_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(budget_rows)

    plus_after = evaluation_results["after_plus"]["plus"]
    plus_final = evaluation_results["after_minus"]["plus"]
    plus_retention = {
        "return_delta_after_minus_vs_after_plus": (
            plus_final["mean_return"] - plus_after["mean_return"]
        ),
        "goal_success_rate_delta_after_minus_vs_after_plus": (
            plus_final["goal_success_rate"] - plus_after["goal_success_rate"]
        ),
    }
    per_mode_online = {}
    for mode in MODES:
        count = max(1, bout_steps)
        per_mode_online[mode] = {
            "training_reward": float(reward_by_mode[mode]),
            "training_reward_per_step": float(reward_by_mode[mode] / count),
            "completed_episodes": int(completed_by_mode[mode]),
            "goal_reaches": int(goal_reaches_by_mode[mode]),
            "goal_success_rate": (
                float(goal_reaches_by_mode[mode] / completed_by_mode[mode])
                if mode == "plus" and completed_by_mode[mode]
                else None
            ),
            "goal_termination_rate": (
                float(goal_reaches_by_mode[mode] / completed_by_mode[mode])
                if mode == "minus" and completed_by_mode[mode]
                else None
            ),
        }
    probe_total = int(agent.counts["probe_steps"])
    train_total = int(agent.counts["training_steps"])
    total_interactions = probe_total + train_total + int(eval_step_total)
    result = {
        "seed": int(seed),
        "condition": condition,
        "counts": dict(agent.counts),
        "training_steps": train_total,
        "probe_steps": probe_total,
        "evaluation_steps": int(eval_step_total),
        "total_interactions": int(total_interactions),
        "training_reward_by_mode": per_mode_online,
        "evaluations": evaluation_results,
        "plus_retention_after_minus_vs_after_plus": plus_retention,
        "group_decisions": group_decisions,
        "probe_profile_hashes": {
            mode: hashlib.sha256(agent.profiles[mode].tobytes()).hexdigest()
            for mode in MODES
        },
        "sizes": budget_rows[-1],
        "source_manifest": source_manifest,
        "wall_s": time.perf_counter() - started_wall,
        "cpu_s": time.process_time() - started_cpu,
        "rss_bytes": int(psutil.Process().memory_info().rss),
        "sampled_peak_rss_bytes": rss_sampler.peak_bytes,
        "scope": (
            "Fixed C1 A0 transition dynamics, reward sign reversed in minus; "
            "known route boundaries, profile-only shared grouping."
        ),
    }
    write_json(folder / "summary.json", result)
    write_json(folder / "cost_ledger.json", {
        "probe_interactions": probe_total,
        "ppo_fit_and_holdout_interactions": train_total,
        "evaluation_interactions": int(eval_step_total),
        "total_environment_interactions": int(total_interactions),
        "wall_seconds": result["wall_s"],
        "cpu_seconds": result["cpu_s"],
        "rss_bytes_at_end": result["rss_bytes"],
        "sampled_peak_rss_bytes": result["sampled_peak_rss_bytes"],
        "probe_step_cost_is_one_per_transition": True,
    })
    return result


def run(phase: str, output: str | Path, workers: int = 1) -> dict[str, Any]:
    if phase not in ("pilot", "formal"):
        raise ValueError("phase must be pilot or formal")
    if workers not in (1, 2):
        raise ValueError("workers must be one or two")
    if workers > int(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["max_workers"]):
        raise ValueError("workers exceeds the configured maximum")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if phase == "pilot":
        config.update(
            bout_steps=int(config["pilot_bout_steps"]),
            evaluation_episodes=int(config["pilot_evaluation_episodes"]),
            seeds=[int(config["validation_seed"])],
        )
    else:
        config.update(
            bout_steps=int(config["formal_bout_steps"]),
            evaluation_episodes=int(config["formal_evaluation_episodes"]),
            seeds=[int(seed) for seed in config["formal_seeds"]],
        )
    config["phase"] = phase
    output_path = Path(output).resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Preserve completed or partial cohort: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(output_path / "frozen_config.json", config)
    tasks = [
        (condition, seed, config, str(output_path))
        for seed in config["seeds"]
        for condition in config["conditions"]
    ]
    records: list[dict[str, Any]] = []
    if workers == 1:
        records = [run_one(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, 2)) as executor:
            pending = [executor.submit(run_one, *task) for task in tasks]
            for future in as_completed(pending):
                records.append(future.result())
    records.sort(key=lambda row: (row["seed"], row["condition"]))
    summary = {
        "phase": phase,
        "seeds": config["seeds"],
        "conditions": config["conditions"],
        "records": records,
        "probe_steps": sum(row["probe_steps"] for row in records),
        "training_steps": sum(row["training_steps"] for row in records),
        "evaluation_steps": sum(row["evaluation_steps"] for row in records),
        "total_interactions": sum(row["total_interactions"] for row in records),
        "scope": (
            "C1 same-dynamics reward-conflict diagnostic; no general routing or "
            "policy-safety claim."
        ),
    }
    write_json(output_path / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    run(args.phase, args.output, args.workers)


if __name__ == "__main__":
    main()

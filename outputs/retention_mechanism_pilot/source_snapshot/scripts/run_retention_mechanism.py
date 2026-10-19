"""Run the frozen common-prefix retention action and horizon diagnostic."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.retention_stream_value import apply_retention_action
from online_rl.storage_decision import (
    ACTIONS,
    bootstrap_schedule,
    common_mean_effective,
    compensated_single_parent_install,
    effective_matrices,
    pair_mapping,
    semantic_hash,
    task_name,
    template_schedule,
    validate_config as validate_storage_config,
)
from scripts import run_knowledge_loop as loop
from scripts import run_storage_decision as storage
from scripts import run_write_stream as ws


CONFIG_PATH = ROOT / "configs" / "retention_mechanism.json"
PROTOCOL_PATH = ROOT / "docs" / "RETENTION_MECHANISM_PROTOCOL.md"
SUMMARY_FORMAT = "retention_mechanism_summary_v1"
SOURCE_FILES = tuple(storage.SOURCE_FILES) + (
    Path("scripts/run_retention_mechanism.py"),
    Path("src/online_rl/envs.py"),
    Path("src/online_rl/retention_stream_value.py"),
    Path("configs/retention_mechanism.json"),
    Path("docs/RETENTION_MECHANISM_PROTOCOL.md"),
)
_ACTION_RNG_STREAM = 0xA17
_BEHAVIOR_RNG_STREAM = 0xB3E


def _json_default(value: Any) -> Any:
    return storage._json_default(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the mechanism fields on top of the frozen storage harness."""
    if config.get("mechanism_format") != "retention_mechanism_v1":
        raise ValueError("mechanism_format must be retention_mechanism_v1")
    result = validate_storage_config(config)
    if tuple(result["actions"]) != ACTIONS:
        raise ValueError(f"actions must be ordered as {ACTIONS}")
    if result.get("candidate_source_steps") != 1024:
        raise ValueError("candidate_source_steps is frozen to 1024")
    if result.get("history_tail_source_bouts") != 2:
        raise ValueError("history_tail_source_bouts is frozen to two")
    if int(result.get("future_blocks", 0)) < 1:
        raise ValueError("future_blocks must be positive")
    if int(result.get("max_workers", 0)) not in (1, 2):
        raise ValueError("max_workers must be one or two")
    if int(result.get("torch_threads", 0)) != 1:
        raise ValueError("torch_threads is frozen to one")
    if int(result.get("future_seed_offset", -1)) != 40_000_000:
        raise ValueError("future_seed_offset is frozen to 40000000")
    if int(result.get("future_seed_multiplier", -1)) != 1000:
        raise ValueError("future_seed_multiplier is frozen to 1000")

    replicates = [int(value) for value in result.get("replicates", [])]
    if not replicates or len(set(replicates)) != len(replicates) or any(value < 0 for value in replicates):
        raise ValueError("replicates must be a nonempty list of unique nonnegative integers")
    result["replicates"] = replicates
    horizons = [int(value) for value in result.get("horizons", [])]
    if not horizons or len(set(horizons)) != len(horizons) or horizons != sorted(horizons):
        raise ValueError("horizons must be a nonempty, strictly increasing list")
    maximum = int(result["future_blocks"]) * 3 * int(result["bout_steps"])
    if any(value < 1 or value % int(result["rollout_steps"]) or value > maximum for value in horizons):
        raise ValueError("horizons must align to rollout boundaries within the future schedule")
    result["horizons"] = horizons
    if result.get("pair_kinds") != ["dynamics", "reward_conflict"]:
        if not result.get("pair_kinds") or not set(result["pair_kinds"]) <= {"dynamics", "reward_conflict"}:
            raise ValueError("pair_kinds must be a nonempty subset of dynamics and reward_conflict")
    if not result.get("history_regimes") or not set(result["history_regimes"]) <= {"old_major", "new_major"}:
        raise ValueError("history_regimes must be a nonempty subset of old_major and new_major")
    if not result.get("future_regimes") or not set(result["future_regimes"]) <= {"old_major", "new_major"}:
        raise ValueError("future_regimes must be a nonempty subset of old_major and new_major")
    if not result.get("seeds"):
        raise ValueError("seeds must be nonempty")
    result["seeds"] = [int(value) for value in result["seeds"]]
    return result


def _phase_config(raw_config: Mapping[str, Any], *, pilot: bool) -> dict[str, Any]:
    config = copy.deepcopy(dict(raw_config))
    phase = "pilot" if pilot else "main"
    if pilot:
        pilot_config = config.get("pilot")
        if not isinstance(pilot_config, Mapping):
            raise ValueError("pilot config must be an object")
        for name in (
            "seeds", "pair_kinds", "history_regimes", "future_regimes",
            "future_blocks", "horizons", "max_workers",
        ):
            if name not in pilot_config:
                raise ValueError(f"pilot config is missing {name}")
            config[name] = copy.deepcopy(pilot_config[name])
    normalized = validate_config(config)
    normalized["phase"] = phase
    return normalized


def _expected_interactions(config: Mapping[str, Any]) -> dict[str, int]:
    """Derive the expected ledger from the selected phase and strata."""
    case_count = (
        len(config["seeds"])
        * len(config["pair_kinds"])
        * len(config["history_regimes"])
    )
    branches_per_case = (
        len(config["future_regimes"])
        * len(config["replicates"])
        * len(config["actions"])
    )
    prefix_steps_per_case = (
        2 * int(config["bootstrap_steps"])
        + (8 - int(config["history_tail_source_bouts"])) * int(config["bout_steps"])
        + int(config["history_tail_source_bouts"]) * int(config["candidate_source_steps"])
    )
    steps_per_branch = int(config["future_blocks"]) * 3 * int(config["bout_steps"])
    prefix_steps = case_count * prefix_steps_per_case
    branch_steps = case_count * branches_per_case * steps_per_branch
    return {
        "cases": int(case_count),
        "branches": int(case_count * branches_per_case),
        "prefix_steps_per_case": int(prefix_steps_per_case),
        "steps_per_branch": int(steps_per_branch),
        "prefix_steps": int(prefix_steps),
        "branch_steps": int(branch_steps),
        "total_steps": int(prefix_steps + branch_steps),
    }


def _campaign_cpu_accounting(
    executor_mode: str, parent_cpu_s: float, case_cpu_values: Sequence[float | None],
) -> dict[str, float | None]:
    """Avoid adding inline case CPU twice while retaining per-case measurements."""
    if executor_mode not in ("inline", "process_pool"):
        raise ValueError("executor_mode must be inline or process_pool")
    case_cpu_sum = None if any(value is None for value in case_cpu_values) else float(sum(float(value) for value in case_cpu_values))
    child_cpu = 0.0 if executor_mode == "inline" else case_cpu_sum
    campaign_cpu = None if child_cpu is None else float(parent_cpu_s + child_cpu)
    return {
        "campaign_parent_cpu_s": float(parent_cpu_s),
        "campaign_case_cpu_s_sum": case_cpu_sum,
        "campaign_child_case_cpu_s": child_cpu,
        "campaign_cpu_s": campaign_cpu,
    }


def build_prefix_schedule(mapping: Mapping[str, Any], config: Mapping[str, Any], history_regime: str) -> list[dict[str, Any]]:
    """Build the common 7168-step prefix with equal source-time fits."""
    normalized = validate_config(config)
    boot = bootstrap_schedule(mapping, int(normalized["bootstrap_steps"]))
    history = template_schedule(
        mapping, history_regime, kind="history", steps=int(normalized["bout_steps"]),
        start_bout=len(boot),
    )
    tail_count = int(normalized["history_tail_source_bouts"])
    if tail_count > len(history):
        raise ValueError("history_tail_source_bouts exceeds the history template length")
    for item in history[-tail_count:]:
        item["steps"] = int(normalized["candidate_source_steps"])
    if any(int(item["group"]) != 1 or int(item["variant"]) not in (0, 1) for item in history[-tail_count:]):
        raise ValueError("the final history bouts must be group-1 variants 0 and 1")
    return boot + history


def build_future_schedule(
    mapping: Mapping[str, Any], future_regime: str, *, blocks: int, bout_steps: int, start_bout: int,
) -> list[dict[str, Any]]:
    """Build the declared minority, dominant, dominant three-bout blocks."""
    if future_regime not in ("old_major", "new_major"):
        raise ValueError("future_regime must be old_major or new_major")
    if int(blocks) < 1 or int(bout_steps) < 1:
        raise ValueError("blocks and bout_steps must be positive")
    dominant = 0 if future_regime == "old_major" else 1
    minority = 1 - dominant
    result: list[dict[str, Any]] = []
    for block in range(int(blocks)):
        variant = block % 3
        entries = ((minority, variant), (dominant, variant), (dominant, (variant + 1) % 3))
        for position, (group, current_variant) in enumerate(entries):
            result.append({
                "phase": "future",
                "bout": int(start_bout) + len(result),
                "future_block": block,
                "within_block": position,
                "group": group,
                "variant": current_variant,
                "task": task_name(mapping, group, current_variant),
                "steps": int(bout_steps),
            })
    return result


def future_seed(config: Mapping[str, Any], case_seed: int, replicate: int) -> int:
    """Return the deterministic CRN seed shared by a case/replicate's branches."""
    return int(config["future_seed_offset"]) + int(config["future_seed_multiplier"]) * int(case_seed) + int(replicate)


def _branch_specs(config: Mapping[str, Any], case_id: str) -> tuple[list[dict[str, Any]], int]:
    specs = [
        {
            "branch_id": f"{future}__r{replicate}__{action}",
            "future_regime": str(future),
            "replicate": int(replicate),
            "action": str(action),
        }
        for future in config["future_regimes"]
        for replicate in config["replicates"]
        for action in config["actions"]
    ]
    material = f"{config['execution_order_seed']}|{case_id}".encode("utf-8")
    order_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    order = np.random.default_rng(order_seed).permutation(len(specs)).tolist()
    return [specs[index] for index in order], order_seed


def _exposure_rows(bouts: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for bout in bouts:
        task = str(bout["task"])
        result.setdefault(task, []).append({
            "bout": int(bout["bout"]),
            "slot": int(bout["slot"]),
            "generation": int(bout["generation"]),
            "steps": int(bout["steps"]),
            "group": int(bout["group"]),
        })
    return result


def _slot_for_task(agent, task: str) -> int:
    matches = [index for index, value in enumerate(agent.cache.slot_to_task) if value == task]
    if len(matches) != 1:
        raise RuntimeError(f"expected one current resident for {task}, found {matches}")
    return int(matches[0])


def _source_fit_record(
    *, task: str, slot: int, generation: int, fitted_steps: int,
    source_time_steps: int, later_same_generation_steps: int,
    task_exposures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "task": str(task),
        "group": 0 if task_exposures and int(task_exposures[0]["group"]) == 0 else (1 if task_exposures else None),
        "slot": int(slot),
        "generation": int(generation),
        "fitted_steps": int(fitted_steps),
        "source_time_fitted_steps": int(source_time_steps),
        "later_same_generation_fitted_steps": int(later_same_generation_steps),
        "task_lifetime_observed_steps": int(sum(int(row["steps"]) for row in task_exposures)),
        "task_exposures": [dict(row) for row in task_exposures],
    }


def _build_source_fits(
    agent,
    mapping: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    prefix_bouts: Sequence[Mapping[str, Any]],
    source_time: Sequence[Mapping[str, Any]],
    candidate_source_steps: int,
) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    exposures = _exposure_rows(prefix_bouts)
    old_rows: list[dict[str, Any]] = []
    for source in source_time:
        task = str(source["task"])
        slot = int(source["slot"])
        generation = int(source["generation"])
        later = sum(
            int(row["steps"])
            for row in prefix_bouts
            if int(row["bout"]) > int(source["bout"])
            and int(row["slot"]) == slot
            and int(row["generation"]) == generation
        )
        old_rows.append(_source_fit_record(
            task=task,
            slot=slot,
            generation=generation,
            fitted_steps=int(source["fitted_steps"]),
            source_time_steps=int(source["fitted_steps"]),
            later_same_generation_steps=later,
            task_exposures=exposures.get(task, ()),
        ))
        if "effective_matrix" in source:
            old_rows[-1]["effective_matrix"] = copy.deepcopy(source["effective_matrix"])
    old_verified = len(old_rows) == 2 and all(
        row["source_time_fitted_steps"] == 1024 and row["group"] == 0 for row in old_rows
    )

    tail = list(schedule[-2:])
    candidate_slots: list[int] = []
    candidate_rows: list[dict[str, Any]] = []
    for source in tail:
        task = str(source["task"])
        slot = _slot_for_task(agent, task)
        generation = int(agent.slot_generation[slot])
        fitted = int(agent.fitted_steps[slot])
        candidate_slots.append(slot)
        current_generation_steps = sum(
            int(row["steps"])
            for row in prefix_bouts
            if int(row["slot"]) == slot and int(row["generation"]) == generation
        )
        candidate_rows.append(_source_fit_record(
            task=task,
            slot=slot,
            generation=generation,
            fitted_steps=fitted,
            source_time_steps=current_generation_steps,
            later_same_generation_steps=0,
            task_exposures=exposures.get(task, ()),
        ))
    candidate_verified = (
        len(set(candidate_slots)) == 2
        and len(candidate_rows) == 2
        and all(
            row["group"] == 1
            and row["fitted_steps"] == candidate_source_steps
            and row["source_time_fitted_steps"] == candidate_source_steps
            for row in candidate_rows
        )
    )
    local, shared, physical_groups = ws._actor_arrays(agent.core)
    logical_tags = np.asarray(agent.slot_group_tags, dtype=np.int8)
    effective_sources = effective_matrices(local, shared, physical_groups)
    if any(int(logical_tags[slot]) != 1 for slot in candidate_slots):
        raise RuntimeError("current candidate sources must both be logical group 1")
    candidate = common_mean_effective(local, shared, physical_groups, candidate_slots)
    for row in candidate_rows:
        row["effective_matrix"] = effective_sources[int(row["slot"])].tolist()
    fits = {
        "old_sources": old_rows,
        "candidate_sources": candidate_rows,
        "old_verified": bool(old_verified),
        "candidate_verified": bool(candidate_verified),
        "verified": bool(old_verified and candidate_verified),
        "candidate_sha256": semantic_hash(candidate),
    }
    return fits, candidate, candidate_slots


def _rng_metadata(agent, env, runtime: Mapping[str, Any]) -> dict[str, Any]:
    environment = storage.snapshot_navigation(env)
    return {
        "agent_seed": int(agent.seed),
        "action_rng_sha256": semantic_hash(agent.action_rng.bit_generator.state),
        "behavior_rng_sha256": semantic_hash(agent.behavior_rng.bit_generator.state),
        "numpy_global_rng_sha256": semantic_hash(np.random.get_state()),
        "python_global_rng_sha256": semantic_hash(random.getstate()),
        "torch_global_rng_sha256": semantic_hash(torch.get_rng_state()),
        "environment_rng_sha256": semantic_hash({
            "bit_generator": environment["np_random_bit_generator"],
            "state": environment["np_random_state"],
        }),
        "episode_reset_seed": int(runtime["episode_reset_seed"]),
        "episode_ordinal": int(runtime["episode_ordinal"]),
        "episode_length": int(runtime["episode_length"]),
    }


def _reseed_future(agent, future_value: int) -> dict[str, Any]:
    agent.seed = int(future_value)
    action_seed = int(future_value) + _ACTION_RNG_STREAM
    behavior_seed = int(future_value) + _BEHAVIOR_RNG_STREAM
    agent.action_rng = np.random.default_rng(action_seed)
    agent.behavior_rng = np.random.default_rng(behavior_seed)
    np.random.seed(int(future_value))
    random.seed(int(future_value))
    torch.manual_seed(int(future_value))
    return {
        "future_seed": int(future_value),
        "action_rng_seed": action_seed,
        "action_rng_stream": _ACTION_RNG_STREAM,
        "behavior_rng_seed": behavior_seed,
        "behavior_rng_stream": _BEHAVIOR_RNG_STREAM,
        "numpy_global_seed": int(future_value),
        "python_global_seed": int(future_value),
        "torch_global_seed": int(future_value),
        "episode_reset_seed_source": "agent.seed via run_storage_decision._reset_seed(seed, ordinal)",
        "environment_rng_action": "preserved until a normal episode reset",
    }


def _load_prefix_payloads(prefix_folder: Path, expected_hash: str) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Read fresh, nonaliased serialized prefix payloads and verify their hash."""
    learner_path = prefix_folder / "checkpoint.pt"
    simulator_path = prefix_folder / "simulator.json"
    actor_path = prefix_folder / "actor.bin"
    learner = storage._capture_learner_from_saved(learner_path)
    simulator = storage._load_json(simulator_path)
    actor = actor_path.read_bytes()
    actual = semantic_hash({"learner": learner, "simulator": simulator})
    if actual != expected_hash:
        raise RuntimeError("serialized prefix files do not match the frozen prefix hash")
    if len(actor) != 89:
        raise RuntimeError(f"prefix actor sidecar is {len(actor)} bytes, expected 89")
    return learner, simulator, actor


def _verify_restored_hash(
    expected_hash: str,
    learner_payload: Mapping[str, Any],
    simulator_payload: Mapping[str, Any],
) -> str:
    actual = semantic_hash({"learner": learner_payload, "simulator": simulator_payload})
    if actual != expected_hash:
        raise RuntimeError("restored learner and simulator do not match the common prefix hash")
    return actual


def _counter_delta(after: Mapping[str, Any], before: Mapping[str, Any], names: Sequence[str]) -> dict[str, int]:
    return {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in names
    }


def _run_schedule_counted(agent, *args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Track attempted/completed PPO calls, including a call that raises mid-update."""
    original_update = loop._ppo_update
    counts = {"rollouts_attempted": 0, "rollouts_completed": 0, "rollouts_failed": 0, "minibatch_updates": 0, "partial_minibatch_updates": 0}

    def counted_update(target, rows):
        if target is not agent:
            return original_update(target, rows)
        counts["rollouts_attempted"] += 1
        before = int(getattr(target.optimizer, "step_count", 0))
        try:
            result = original_update(target, rows)
        except Exception:
            counts["rollouts_failed"] += 1
            counts["partial_minibatch_updates"] += int(getattr(target.optimizer, "step_count", 0)) - before
            target._last_retention_ppo_call_counts = dict(counts)
            raise
        counts["rollouts_completed"] += 1
        counts["minibatch_updates"] += int(result.get("updates", 0))
        target._last_retention_ppo_call_counts = dict(counts)
        return result

    loop._ppo_update = counted_update
    try:
        bouts = storage._run_schedule(agent, *args, **kwargs)
        return bouts, counts
    finally:
        loop._ppo_update = original_update
        agent._last_retention_ppo_call_counts = dict(counts)


def _csv_record_stats(path: Path, *, reward_column: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.name, "exists": False, "sha256": None, "rows": 0}
    rows = 0
    reward_sum = 0.0
    rollout_rows = 0
    minibatch_updates = 0
    training_rows = 0
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows += 1
            if reward_column is not None:
                reward_sum += float(row.get(reward_column) or 0.0)
            if "updates" in row:
                rollout_rows += 1
                minibatch_updates += int(row.get("updates") or 0)
                training_rows += int(row.get("rows") or 0)
    result = {"path": path.name, "exists": True, "sha256": _sha(path), "rows": rows}
    if reward_column is not None:
        result["reward_sum"] = float(reward_sum)
    if path.name == "rollouts.csv":
        result.update({
            "rollout_count": rollout_rows,
            "trained_rows": training_rows,
            "minibatch_updates": minibatch_updates,
        })
    return result


def _raw_ledgers(folder: Path, root: Path | None = None) -> dict[str, Any]:
    result = {
        "steps_csv": _csv_record_stats(folder / "steps.csv", reward_column="reward"),
        "episodes_csv": _csv_record_stats(folder / "episodes.csv"),
        "rollouts_csv": _csv_record_stats(folder / "rollouts.csv"),
    }
    if root is not None:
        for record in result.values():
            if record.get("exists"):
                record["path"] = (folder / record["path"]).relative_to(root).as_posix()
    return result


def _artifact_record(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha(path),
        "bytes": int(path.stat().st_size),
    }


def _package_artifacts(folder: Path, root: Path) -> dict[str, Any]:
    return {
        "checkpoint": _artifact_record(folder / "checkpoint.pt", root),
        "actor": _artifact_record(folder / "actor.bin", root),
        "simulator": _artifact_record(folder / "simulator.json", root),
    }


def _snapshot_sources(folder: Path, config_path: Path) -> dict[str, Any]:
    snapshot = folder / "source_snapshot"
    manifest_files: dict[str, dict[str, Any]] = {}
    paths = list(dict.fromkeys(SOURCE_FILES))
    config_resolved = config_path.resolve()
    if config_resolved.is_relative_to(ROOT.resolve()):
        config_rel = Path(os.path.relpath(config_resolved, ROOT.resolve()))
        if config_rel not in paths:
            paths.append(config_rel)
    for relative in paths:
        if any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts):
            raise ValueError("private or temporary material is excluded from source snapshots")
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_digest, copied_digest = _sha(source), _sha(target)
        if source_digest != copied_digest:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        manifest_files[relative.as_posix()] = {"sha256": copied_digest, "bytes": int(target.stat().st_size)}
    base = loop.load_base_config()
    foundation = Path(base["foundation"])
    dataset = Path(base["dataset"])
    manifest = {
        "sources": manifest_files,
        "foundation": {"path": str(foundation), "sha256": _sha(foundation), "bytes": int(foundation.stat().st_size)},
        "dataset": {"path": str(dataset), "sha256": _sha(dataset), "bytes": int(dataset.stat().st_size)},
        "excluded_local_instructions": ["Agent.md", "AGENTS.md"],
    }
    _write_json(snapshot / "manifest.json", manifest)
    return manifest


def _read_persisted_counts(folder: Path) -> dict[str, Any]:
    prefix = _csv_record_stats(folder / "prefix" / "steps.csv", reward_column="reward")
    prefix_rollouts = _csv_record_stats(folder / "prefix" / "rollouts.csv")
    branch_stats: list[dict[str, Any]] = []
    branch_root = folder / "branches"
    if branch_root.is_dir():
        for child in sorted(branch_root.iterdir()):
            if child.is_dir():
                steps = _csv_record_stats(child / "steps.csv", reward_column="reward")
                rolls = _csv_record_stats(child / "rollouts.csv")
                branch_stats.append({"branch_id": child.name, "steps": steps, "rollouts": rolls})
    branch_steps = sum(int(row["steps"]["rows"]) for row in branch_stats)
    branch_rollouts = sum(int(row["rollouts"].get("rollout_count", 0)) for row in branch_stats)
    branch_updates = sum(int(row["rollouts"].get("minibatch_updates", 0)) for row in branch_stats)
    return {
        "prefix_environment_steps": int(prefix["rows"]),
        "prefix_reward_sum": float(prefix.get("reward_sum", 0.0)),
        "prefix_rollouts": int(prefix_rollouts.get("rollout_count", 0)),
        "prefix_minibatch_updates": int(prefix_rollouts.get("minibatch_updates", 0)),
        "branch_environment_steps": branch_steps,
        "branch_rollouts": branch_rollouts,
        "branch_minibatch_updates": branch_updates,
        "environment_steps_total": int(prefix["rows"]) + branch_steps,
        "branches": branch_stats,
    }


def _branch_horizons(
    bouts: Sequence[Mapping[str, Any]],
    cpu_milestones: Mapping[int, Mapping[str, Any]],
    horizons: Sequence[int],
    install_cpu_s: float,
) -> list[dict[str, Any]]:
    result = []
    for horizon in horizons:
        target = int(horizon)
        selected: list[Mapping[str, Any]] = []
        consumed = 0
        reward = 0.0
        for bout in bouts:
            steps = int(bout["steps"])
            if consumed + steps > target:
                raise RuntimeError("horizon does not align to a completed bout boundary")
            selected.append(bout)
            consumed += steps
            reward += float(bout["return_full"])
            if consumed == target:
                break
        if consumed != target:
            continue
        milestone = cpu_milestones.get(target)
        if milestone is None:
            continue
        rate = reward / target
        sim_cpu = float(milestone["simulation_cpu_s"])
        result.append({
            "steps": target,
            "reward_sum": float(reward),
            "reward_per_step": float(rate),
            "simulation_cpu_s": sim_cpu,
            "action_install_cpu_s": float(install_cpu_s),
            "utility_rate": float(rate - (install_cpu_s + sim_cpu) / target),
            "raw_environment_steps": int(target),
            "rollouts_trained": int(milestone["counts_delta"]["rollouts_trained"]),
            "minibatch_updates": int(milestone["counts_delta"]["minibatch_updates"]),
            "simulation_wall_s": float(milestone["simulation_wall_s"]),
            "cpu_sample_stage": "post_rollout_before_episode_reset_handling",
        })
    return result


def _case_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = validate_config(payload["config"])
    seed = int(payload["seed"])
    pair_kind = str(payload["pair_kind"])
    history_regime = str(payload["history_regime"])
    output = Path(payload["output"])
    case_id = f"s{seed}_{pair_kind}_{history_regime}"
    case_folder = output / "cases" / case_id
    case_folder.mkdir(parents=True, exist_ok=False)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    foundation = None
    foundation_sha = None
    env = None
    prefix_summary: dict[str, Any] | None = None
    branch_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    prefix_cpu_s = 0.0
    prefix_wall_s = 0.0
    try:
        base_config = loop.load_base_config()
        foundation = Path(base_config["foundation"])
        foundation_sha = _sha(foundation)
        mapping = pair_mapping(seed, pair_kind)
        schedule = build_prefix_schedule(mapping, config, history_regime)
        env, runtime = storage._runtime_initial(seed, str(schedule[0]["task"]))
        agent = storage._new_agent(seed, base_config, config)
        prefix_folder = case_folder / "prefix"
        prefix_folder.mkdir(parents=True, exist_ok=False)
        prefix_start_cpu, prefix_start_wall = time.process_time(), time.perf_counter()

        bootstrap_bouts, bootstrap_ppo_counts = _run_schedule_counted(
            agent, env, runtime, schedule[:2], folder=prefix_folder,
            phase="bootstrap",
        )
        old_source_time: list[dict[str, Any]] = []
        for item in schedule[:2]:
            slot = _slot_for_task(agent, str(item["task"]))
            fitted = int(agent.fitted_steps[slot])
            if fitted != int(item["steps"]):
                raise RuntimeError(f"old source {item['task']} fit {fitted} steps, expected {item['steps']}")
            old_source_time.append({
                "task": str(item["task"]), "bout": int(item["bout"]),
                "slot": slot, "generation": int(agent.slot_generation[slot]),
                "fitted_steps": fitted,
            })
        if len({row["slot"] for row in old_source_time}) != 2:
            raise RuntimeError("old source variants must occupy two distinct resident slots")

        local, shared, physical_groups = ws._actor_arrays(agent.core)
        old_effective_at_formation = effective_matrices(local, shared, physical_groups)
        for source in old_source_time:
            source["effective_matrix"] = old_effective_at_formation[int(source["slot"])].tolist()
        old_slots = [int(row["slot"]) for row in old_source_time]
        incumbent = common_mean_effective(local, shared, physical_groups, old_slots)
        new_local, new_shared, new_groups, formation = compensated_single_parent_install(
            local,
            shared,
            physical_groups,
            agent.slot_group_tags,
            parent=incumbent,
            parent_group=0,
        )
        agent.core.install_local_matrix(new_local)
        agent.core.install_shared_matrix(new_shared)
        with torch.no_grad():
            agent.core.slot_groups.copy_(torch.as_tensor(new_groups.astype(np.int64), dtype=torch.long))
        agent.parent_group = int(formation["parent_group"])
        _, formation_codec_error, formation_effective_error = storage._encode_live_actor(agent)
        ws._clear_optimizer_state(agent)

        history_bouts, history_ppo_counts = _run_schedule_counted(
            agent, env, runtime, schedule[2:], folder=prefix_folder,
            phase="prefix_history",
        )
        prefix_ppo_counts = {
            name: int(bootstrap_ppo_counts.get(name, 0)) + int(history_ppo_counts.get(name, 0))
            for name in set(bootstrap_ppo_counts) | set(history_ppo_counts)
        }
        prefix_bouts = bootstrap_bouts + history_bouts
        if len(agent.recent_groups) != 8 or agent.recent_groups[-2:] != [1, 1]:
            raise RuntimeError("prefix history must end with group-1 variants zero and one")
        source_fits, candidate, candidate_slots = _build_source_fits(
            agent,
            mapping,
            schedule,
            prefix_bouts,
            old_source_time,
            int(config["candidate_source_steps"]),
        )
        if not source_fits["verified"]:
            raise RuntimeError(f"prefix source-fit verification failed: {source_fits}")

        prefix_actor, prefix_codec_error, prefix_codec_effective_error = storage._encode_live_actor(agent)
        ws._clear_optimizer_state(agent)
        prefix_local, prefix_shared, prefix_physical_groups = ws._actor_arrays(agent.core)
        prefix_logical_tags = np.asarray(agent.slot_group_tags, dtype=np.int8)
        prefix_effective = effective_matrices(prefix_local, prefix_shared, prefix_physical_groups)
        candidate = prefix_effective[np.asarray(candidate_slots, dtype=np.int64)].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        source_fits["candidate_sha256"] = semantic_hash(candidate)
        for row in source_fits["candidate_sources"]:
            row["effective_matrix"] = prefix_effective[int(row["slot"])].tolist()
            row["fitted_steps"] = int(agent.fitted_steps[int(row["slot"])])
            row["generation"] = int(agent.slot_generation[int(row["slot"])])
        prefix_learner = storage._capture_learner(agent, foundation_sha)
        prefix_simulator = storage._capture_simulator(env, runtime)
        prefix_hash = semantic_hash({"learner": prefix_learner, "simulator": prefix_simulator})
        prefix_learner_hash = semantic_hash(prefix_learner)
        prefix_capacity = storage._save_package(prefix_folder, prefix_learner, prefix_actor, foundation, config)
        storage._write_json(prefix_folder / "simulator.json", prefix_simulator)
        observer_prefix = {
            "format": "retention_mechanism_observer_prefix_v1",
            "accessible_to_deployed_selector": False,
            "included_in_deployment_capacity": False,
            "old_source_effective_matrices_at_bootstrap": [
                {key: value for key, value in row.items() if key in ("task", "bout", "slot", "generation", "fitted_steps", "effective_matrix")}
                for row in old_source_time
            ],
            "incumbent_mean_effective_parent_at_bootstrap": incumbent.tolist(),
            "candidate_current_effective_matrices_before_averaging": [
                {key: value for key, value in row.items() if key in ("task", "slot", "generation", "fitted_steps", "effective_matrix")}
                for row in source_fits["candidate_sources"]
            ],
            "candidate_mean_effective": candidate.tolist(),
            "candidate_mean_sha256": semantic_hash(candidate),
            "prefix_actor_arrays_before_action": {
                "local": prefix_local.tolist(),
                "shared": prefix_shared.tolist(),
                "physical_groups": prefix_physical_groups.tolist(),
                "logical_group_tags": prefix_logical_tags.tolist(),
                "parent_group": int(agent.parent_group),
            },
        }
        observer_prefix_path = prefix_folder / "observer_prefix.json"
        _write_json(observer_prefix_path, observer_prefix)
        serialized_learner = storage._capture_learner_from_saved(prefix_folder / "checkpoint.pt")
        serialized_simulator = storage._load_json(prefix_folder / "simulator.json")
        _verify_restored_hash(prefix_hash, serialized_learner, serialized_simulator)
        prefix_cpu_s = float(time.process_time() - prefix_start_cpu)
        prefix_wall_s = float(time.perf_counter() - prefix_start_wall)
        prefix_elapsed = {"cpu_s": prefix_cpu_s, "wall_s": prefix_wall_s}
        prefix_summary = storage._summarize_bouts(prefix_bouts, agent, prefix_elapsed)
        prefix_summary.update({
            "steps": int(sum(int(row["steps"]) for row in prefix_bouts)),
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "prefix_state_hash": prefix_hash,
            "prefix_learner_hash": prefix_learner_hash,
            "foundation_sha256": foundation_sha,
            "formation_effective_preservation_error": formation["max_effective_preservation_error"],
            "formation_effective_errors": formation["per_slot_preservation_error"],
            "formation_codec_max_coefficient_error": formation_codec_error,
            "formation_codec_effective_errors": formation_effective_error,
            "codec_max_coefficient_error": prefix_codec_error,
            "codec_effective_errors": prefix_codec_effective_error,
            "source_fits": source_fits,
            "ppo_call_counts": prefix_ppo_counts,
            "candidate_source_slots": candidate_slots,
            "shared_frozen": bool(agent.core._shared_frozen),
            "capacity": prefix_capacity,
            "actor_payload_bytes": len(prefix_actor),
            "artifacts": _package_artifacts(prefix_folder, output),
            "raw_ledgers": _raw_ledgers(prefix_folder, output),
            "observer_artifact": _artifact_record(observer_prefix_path, output),
        })
        prefix_summary["artifacts"]["checkpoint"]["path"] = str(
            Path("cases") / case_id / "prefix" / "checkpoint.pt"
        ).as_posix()
        prefix_summary["artifacts"]["actor"]["path"] = str(
            Path("cases") / case_id / "prefix" / "actor.bin"
        ).as_posix()
        prefix_summary["artifacts"]["simulator"]["path"] = str(
            Path("cases") / case_id / "prefix" / "simulator.json"
        ).as_posix()
        _write_json(prefix_folder / "summary.json", prefix_summary)

        branch_order, execution_order_seed = _branch_specs(config, case_id)
        _write_json(case_folder / "execution_order.json", {
            "seed": execution_order_seed,
            "order": [row["branch_id"] for row in branch_order],
        })
        prefix_files_before = {
            name: {"sha256": _sha(prefix_folder / name), "bytes": int((prefix_folder / name).stat().st_size)}
            for name in ("checkpoint.pt", "actor.bin", "simulator.json")
        }
        for spec in branch_order:
            branch_folder = case_folder / "branches" / str(spec["branch_id"])
            branch_folder.mkdir(parents=True, exist_ok=False)
            branch_start_cpu, branch_start_wall = time.process_time(), time.perf_counter()
            setup_cpu_s = setup_wall_s = 0.0
            install_cpu_s = install_wall_s = 0.0
            simulation_cpu_s = simulation_wall_s = 0.0
            serialization_cpu_s = serialization_wall_s = 0.0
            restored = None
            restored_env = None
            branch_bouts: list[dict[str, Any]] = []
            cpu_milestones: dict[int, dict[str, Any]] = {}
            future_value = future_seed(config, seed, int(spec["replicate"]))
            branch_error: dict[str, Any] | None = None
            branch_summary: dict[str, Any]
            prefix_runtime_counts: dict[str, Any] | None = None
            prefix_learner_counts: dict[str, Any] | None = None
            optimizer_start_count: int | None = None
            try:
                setup_start_cpu, setup_start_wall = time.process_time(), time.perf_counter()
                serialized_learner, serialized_simulator, prefix_actor_bytes = _load_prefix_payloads(prefix_folder, prefix_hash)
                restored = storage._restore_learner(serialized_learner, base_config, config)
                restored_env, restored_runtime = storage._restore_simulator(serialized_simulator)
                restored_payload = storage._capture_learner(restored, foundation_sha)
                restored_snapshot = storage._capture_simulator(restored_env, restored_runtime)
                restored_hash = _verify_restored_hash(prefix_hash, restored_payload, restored_snapshot)
                rng_before = _rng_metadata(restored, restored_env, restored_runtime)
                environment_hash_before = rng_before["environment_rng_sha256"]
                reseed_metadata = _reseed_future(restored, future_value)
                rng_after = _rng_metadata(restored, restored_env, restored_runtime)
                if rng_after["environment_rng_sha256"] != environment_hash_before:
                    raise RuntimeError("future RNG transformation changed the restored environment RNG")
                post_transform_learner = storage._capture_learner(restored, foundation_sha)
                post_transform_simulator = storage._capture_simulator(restored_env, restored_runtime)
                post_transform_hash = semantic_hash({
                    "learner": post_transform_learner,
                    "simulator": post_transform_simulator,
                })
                setup_cpu_s = float(time.process_time() - setup_start_cpu)
                setup_wall_s = float(time.perf_counter() - setup_start_wall)
                prefix_runtime_counts = dict(restored.runtime_counts)
                prefix_learner_counts = dict(restored.counts)
                optimizer_start_count = int(getattr(restored.optimizer, "step_count", 0))

                install_start_cpu, install_start_wall = time.process_time(), time.perf_counter()
                local, shared, physical_groups = ws._actor_arrays(restored.core)
                logical_tags = np.asarray(restored.slot_group_tags, dtype=np.int8)
                current_candidate_slots = [
                    slot for slot, group in enumerate(logical_tags.tolist())
                    if int(group) == 1 and restored.cache.slot_to_task[slot] is not None
                ]
                if len(current_candidate_slots) != 2:
                    raise RuntimeError(f"expected two current candidate residents, found {current_candidate_slots}")
                candidate_groups = {int(logical_tags[slot]) for slot in current_candidate_slots}
                if len(candidate_groups) != 1:
                    raise RuntimeError("candidate sources do not share one logical group")
                candidate_group = candidate_groups.pop()
                branch_candidate = common_mean_effective(
                    local, shared, physical_groups, current_candidate_slots,
                )
                candidate_hash = semantic_hash(branch_candidate)
                if candidate_hash != source_fits["candidate_sha256"]:
                    raise RuntimeError("restored candidate source differs from the common-prefix candidate")
                incumbent_binding = int(restored.parent_group)
                fork_observation = np.asarray(restored_runtime["obs"], dtype=np.float32)
                fork_probabilities = storage._resident_probabilities(restored.core, fork_observation, restored.slots)
                before_effective = storage._effective_live(local, shared, physical_groups)
                post_local, post_shared, post_groups, intervention = apply_retention_action(
                    local,
                    shared,
                    physical_groups,
                    logical_tags,
                    incumbent_binding,
                    branch_candidate,
                    candidate_group,
                    str(spec["action"]),
                )
                if str(spec["action"]) != "keep":
                    restored.core.install_local_matrix(post_local)
                    restored.core.install_shared_matrix(post_shared)
                    with torch.no_grad():
                        restored.core.slot_groups.copy_(torch.as_tensor(post_groups.astype(np.int64), dtype=torch.long))
                restored.parent_group = int(intervention["parent_group"])
                after_effective = storage._effective_live(post_local, post_shared, post_groups)
                prequant_effective_errors = np.max(
                    np.abs(after_effective - before_effective), axis=(1, 2, 3),
                ).astype(float).tolist()
                prequant_probabilities = storage._resident_probabilities(restored.core, fork_observation, restored.slots)
                prequant_behavior_drift = storage._probability_drift(fork_probabilities, prequant_probabilities)
                if str(spec["action"]) == "keep":
                    action_actor = prefix_actor_bytes
                    action_codec_error = 0.0
                    action_codec_effective_error = [0.0, 0.0]
                else:
                    action_actor, action_codec_error, action_codec_effective_error = storage._encode_live_actor(restored)
                    ws._clear_optimizer_state(restored)
                postquant_probabilities = storage._resident_probabilities(restored.core, fork_observation, restored.slots)
                postquant_behavior_drift = storage._probability_drift(fork_probabilities, postquant_probabilities)
                action_learner = storage._capture_learner(restored, foundation_sha)
                action_simulator = storage._capture_simulator(restored_env, restored_runtime)
                action_hash = semantic_hash({"learner": action_learner, "simulator": action_simulator})
                shared_reference = ws._actor_arrays(restored.core)[1].copy()
                install_cpu_s = float(time.process_time() - install_start_cpu)
                install_wall_s = float(time.perf_counter() - install_start_wall)

                serialization_start_cpu, serialization_start_wall = time.process_time(), time.perf_counter()
                post_action_folder = branch_folder / "post_action"
                capacity_after_action = storage._save_package(
                    post_action_folder, action_learner, action_actor, foundation, config,
                )
                storage._write_json(post_action_folder / "simulator.json", action_simulator)
                post_action_artifacts = _package_artifacts(post_action_folder, output)
                serialization_cpu_s += float(time.process_time() - serialization_start_cpu)
                serialization_wall_s += float(time.perf_counter() - serialization_start_wall)

                future = build_future_schedule(
                    mapping,
                    str(spec["future_regime"]),
                    blocks=int(config["future_blocks"]),
                    bout_steps=int(config["bout_steps"]),
                    start_bout=len(schedule),
                )
                simulation_start_cpu, simulation_start_wall = time.process_time(), time.perf_counter()
                branch_start_steps = int(restored.runtime_counts["environment_steps"])
                branch_start_rollouts = int(restored.runtime_counts["rollouts_trained"])
                branch_start_updates = int(restored.runtime_counts["minibatch_updates"])
                target_steps = set(int(value) for value in config["horizons"])

                def observe_rollout(**event: Any) -> None:
                    del event
                    current_steps = int(restored.runtime_counts["environment_steps"]) - branch_start_steps
                    current_shared = ws._actor_arrays(restored.core)[1]
                    if not np.array_equal(current_shared, shared_reference):
                        raise RuntimeError("shared parent coefficients changed after branch installation")
                    if current_steps in target_steps:
                        cpu_milestones[current_steps] = {
                            "simulation_cpu_s": float(time.process_time() - simulation_start_cpu),
                            "simulation_wall_s": float(time.perf_counter() - simulation_start_wall),
                            "counts_delta": {
                                "rollouts_trained": int(restored.runtime_counts["rollouts_trained"]) - branch_start_rollouts,
                                "minibatch_updates": int(restored.runtime_counts["minibatch_updates"]) - branch_start_updates,
                            },
                        }

                restored._storage_decision_post_rollout_observer = observe_rollout
                branch_bouts, branch_ppo_counts = _run_schedule_counted(
                    restored,
                    restored_env,
                    restored_runtime,
                    future,
                    folder=branch_folder,
                    phase="future",
                    action=str(spec["action"]),
                    future_regime=str(spec["future_regime"]),
                )
                del restored._storage_decision_post_rollout_observer
                simulation_cpu_s = float(time.process_time() - simulation_start_cpu)
                simulation_wall_s = float(time.perf_counter() - simulation_start_wall)
                if not np.array_equal(ws._actor_arrays(restored.core)[1], shared_reference):
                    raise RuntimeError("shared parent coefficients changed over the branch continuation")
                final_probs_before_codec = storage._resident_probabilities(
                    restored.core, np.asarray(restored_runtime["obs"], dtype=np.float32), restored.slots,
                )

                final_serialize_start_cpu, final_serialize_start_wall = time.process_time(), time.perf_counter()
                final_actor, final_codec_error, final_codec_effective_error = storage._encode_live_actor(restored)
                final_probs_after_codec = storage._resident_probabilities(
                    restored.core, np.asarray(restored_runtime["obs"], dtype=np.float32), restored.slots,
                )
                final_codec_behavior_drift = storage._probability_drift(final_probs_before_codec, final_probs_after_codec)
                final_learner = storage._capture_learner(restored, foundation_sha)
                final_simulator = storage._capture_simulator(restored_env, restored_runtime)
                final_hash = semantic_hash({"learner": final_learner, "simulator": final_simulator})
                capacity = storage._save_package(branch_folder, final_learner, final_actor, foundation, config)
                storage._write_json(branch_folder / "simulator.json", final_simulator)
                final_artifacts = _package_artifacts(branch_folder, output)
                ledgers = _raw_ledgers(branch_folder, output)
                serialization_cpu_s += float(time.process_time() - final_serialize_start_cpu)
                serialization_wall_s += float(time.perf_counter() - final_serialize_start_wall)

                if semantic_hash(prefix_learner) != prefix_learner_hash:
                    raise RuntimeError("branch mutated the common in-memory prefix learner")
                for name, before in prefix_files_before.items():
                    target = prefix_folder / name
                    if not target.is_file() or _sha(target) != before["sha256"] or target.stat().st_size != before["bytes"]:
                        raise RuntimeError(f"branch mutated serialized prefix artifact {name}")

                horizons = _branch_horizons(branch_bouts, cpu_milestones, config["horizons"], install_cpu_s)
                runtime_names = (
                    "environment_steps", "completed_episodes", "slot_allocations", "slot_evictions",
                    "rollouts_trained", "minibatch_updates",
                )
                learner_names = (
                    "raw_steps", "environment_steps", "fit_steps", "rollouts_trained", "minibatch_updates",
                    "completed_episodes", "slot_allocations", "slot_evictions",
                )
                runtime_delta = _counter_delta(restored.runtime_counts, prefix_runtime_counts, runtime_names)
                learner_delta = _counter_delta(restored.counts, prefix_learner_counts, learner_names)
                optimizer_step_delta = int(getattr(restored.optimizer, "step_count", 0)) - int(optimizer_start_count or 0)
                if optimizer_step_delta != learner_delta["minibatch_updates"]:
                    raise RuntimeError("optimizer step counter disagrees with learner minibatch update counts")
                branch_cpu_s = float(time.process_time() - branch_start_cpu)
                branch_wall_s = float(time.perf_counter() - branch_start_wall)
                accounted_cpu = setup_cpu_s + install_cpu_s + simulation_cpu_s + serialization_cpu_s
                accounted_wall = setup_wall_s + install_wall_s + simulation_wall_s + serialization_wall_s
                branch_summary = {
                    "format": "retention_mechanism_branch_v1",
                    "branch_id": str(spec["branch_id"]),
                    "case_id": case_id,
                    "seed": seed,
                    "pair_kind": pair_kind,
                    "history_regime": history_regime,
                    "future_regime": str(spec["future_regime"]),
                    "replicate": int(spec["replicate"]),
                    "action": str(spec["action"]),
                    "future_seed": int(future_value),
                    "future_schedule": future,
                    "prefix_state_hash": prefix_hash,
                    "restored_state_hash": restored_hash,
                    "prefix_restore_verified": True,
                    "post_rng_transform_state_hash": post_transform_hash,
                    "rng_transform": {
                        "reseed": reseed_metadata,
                        "before": rng_before,
                        "after": rng_after,
                        "environment_rng_preserved": bool(
                            rng_before["environment_rng_sha256"] == rng_after["environment_rng_sha256"]
                        ),
                        "current_episode_preserved": bool(
                            int(rng_before["episode_ordinal"]) == int(rng_after["episode_ordinal"])
                            and int(rng_before["episode_length"]) == int(rng_after["episode_length"])
                            and int(rng_before["episode_reset_seed"]) == int(rng_after["episode_reset_seed"])
                        ),
                    },
                    "action_install": {
                        "incumbent_binding": incumbent_binding,
                        "candidate_group": candidate_group,
                        "parent_group": int(intervention["parent_group"]),
                        "details": intervention,
                        "candidate_sha256": candidate_hash,
                        "fork_observation": fork_observation.tolist(),
                        "pre_action_resident_probabilities": fork_probabilities.tolist(),
                        "prequantized_resident_probabilities": prequant_probabilities.tolist(),
                        "postquantized_resident_probabilities": postquant_probabilities.tolist(),
                        "prequantized_effective_errors": prequant_effective_errors,
                        "prequantized_behavior_drift": prequant_behavior_drift,
                        "postquantized_behavior_drift": postquant_behavior_drift,
                        "codec_max_coefficient_error": action_codec_error,
                        "codec_effective_errors": action_codec_effective_error,
                        "actor_payload_bytes": len(action_actor),
                        "keep_avoided_extra_install_and_codec": bool(str(spec["action"]) == "keep"),
                    },
                    "post_action_state_hash": action_hash,
                    "post_action_artifacts": post_action_artifacts,
                    "final_state_hash": final_hash,
                    "final_artifacts": final_artifacts,
                    "final_codec": {
                        "max_coefficient_error": final_codec_error,
                        "effective_errors": final_codec_effective_error,
                        "behavior_drift": final_codec_behavior_drift,
                    },
                    "shared_parent_invariant": {
                        "max_coefficient_drift": float(np.max(np.abs(ws._actor_arrays(restored.core)[1] - shared_reference), initial=0.0)),
                        "exactly_fixed": bool(np.array_equal(ws._actor_arrays(restored.core)[1], shared_reference)),
                    },
                    "per_bout": branch_bouts,
                    "horizons": horizons,
                    "raw_environment_steps": int(ledgers["steps_csv"]["rows"]),
                    "ppo_call_counts": branch_ppo_counts,
                    "counts_delta": {
                        **learner_delta,
                        "runtime_counts": runtime_delta,
                        "optimizer_step_count": optimizer_step_delta,
                        "partial_rollout_rows": int(len(restored.rollout)),
                    },
                    "raw_ledgers": ledgers,
                    "capacity": capacity,
                    "capacity_after_action": capacity_after_action,
                    "costs": {
                        "setup_cpu_s": setup_cpu_s,
                        "install_cpu_s": install_cpu_s,
                        "simulation_cpu_s": simulation_cpu_s,
                        "serialization_cpu_s": serialization_cpu_s,
                        "branch_cpu_s": branch_cpu_s,
                        "unaccounted_cpu_s": max(0.0, branch_cpu_s - accounted_cpu),
                        "setup_wall_s": setup_wall_s,
                        "install_wall_s": install_wall_s,
                        "simulation_wall_s": simulation_wall_s,
                        "serialization_wall_s": serialization_wall_s,
                        "branch_wall_s": branch_wall_s,
                        "unaccounted_wall_s": max(0.0, branch_wall_s - accounted_wall),
                    },
                    "prefix_bouts_included_in_reward": False,
                    "errors": [],
                }
            except Exception:
                branch_error = {
                    "case_id": case_id,
                    "branch_id": str(spec["branch_id"]),
                    "error_type": "branch_failure",
                    "error": traceback.format_exc(),
                }
                errors.append(branch_error)
                ledgers = _raw_ledgers(branch_folder, output)
                branch_cpu_s = float(time.process_time() - branch_start_cpu)
                branch_wall_s = float(time.perf_counter() - branch_start_wall)
                branch_summary = {
                    "format": "retention_mechanism_branch_v1",
                    "branch_id": str(spec["branch_id"]),
                    "case_id": case_id,
                    "seed": seed,
                    "pair_kind": pair_kind,
                    "history_regime": history_regime,
                    "future_regime": str(spec["future_regime"]),
                    "replicate": int(spec["replicate"]),
                    "action": str(spec["action"]),
                    "future_seed": int(future_value),
                    "prefix_state_hash": prefix_hash,
                    "prefix_restore_verified": bool(restored is not None and restored_env is not None),
                    "raw_environment_steps": int(ledgers["steps_csv"]["rows"]),
                    "raw_ledgers": ledgers,
                    "ppo_call_counts": (
                        dict(getattr(restored, "_last_retention_ppo_call_counts", {}))
                        if restored is not None else {}
                    ),
                    "per_bout": branch_bouts,
                    "horizons": [],
                    "counts_delta": None,
                    "costs": {
                        "setup_cpu_s": setup_cpu_s,
                        "install_cpu_s": install_cpu_s,
                        "simulation_cpu_s": simulation_cpu_s,
                        "serialization_cpu_s": serialization_cpu_s,
                        "branch_cpu_s": branch_cpu_s,
                        "setup_wall_s": setup_wall_s,
                        "install_wall_s": install_wall_s,
                        "simulation_wall_s": simulation_wall_s,
                        "serialization_wall_s": serialization_wall_s,
                        "branch_wall_s": branch_wall_s,
                    },
                    "errors": [branch_error],
                }
                if restored is not None and prefix_runtime_counts is not None and prefix_learner_counts is not None:
                    runtime_names = (
                        "environment_steps", "completed_episodes", "slot_allocations", "slot_evictions",
                        "rollouts_trained", "minibatch_updates",
                    )
                    learner_names = (
                        "raw_steps", "environment_steps", "fit_steps", "rollouts_trained", "minibatch_updates",
                        "completed_episodes", "slot_allocations", "slot_evictions",
                    )
                    learner_delta = _counter_delta(restored.counts, prefix_learner_counts, learner_names)
                    runtime_delta = _counter_delta(restored.runtime_counts, prefix_runtime_counts, runtime_names)
                    optimizer_step_delta = int(getattr(restored.optimizer, "step_count", 0)) - int(optimizer_start_count or 0)
                    branch_summary["counts_delta"] = {
                        **learner_delta,
                        "runtime_counts": runtime_delta,
                        "optimizer_step_count": optimizer_step_delta,
                        "partial_rollout_rows": int(len(restored.rollout)),
                        "rollouts_csv_completed": int(ledgers["rollouts_csv"].get("rollout_count", 0)),
                        "minibatch_updates_from_rollout_ledger": int(ledgers["rollouts_csv"].get("minibatch_updates", 0)),
                    }
                _write_json(branch_folder / "error.json", branch_error)
            finally:
                if restored is not None and hasattr(restored, "_storage_decision_post_rollout_observer"):
                    del restored._storage_decision_post_rollout_observer
                if restored_env is not None:
                    restored_env.close()
            _write_json(branch_folder / "summary.json", branch_summary)
            branch_results.append(branch_summary)

        foundation_unchanged = _sha(foundation) == foundation_sha
        if not foundation_unchanged:
            errors.append({"case_id": case_id, "error_type": "foundation_changed", "error": "W0 SHA changed during case"})
        total_branch_steps = sum(int(row.get("raw_environment_steps", 0)) for row in branch_results)
        branch_cpu_total = sum(float(row.get("costs", {}).get("branch_cpu_s", 0.0)) for row in branch_results)
        branch_wall_total = sum(float(row.get("costs", {}).get("branch_wall_s", 0.0)) for row in branch_results)
        result = {
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "mapping": mapping,
            "prefix": prefix_summary,
            "execution_order_seed": execution_order_seed,
            "execution_order": [row["branch_id"] for row in branch_order],
            "branches": branch_results,
            "foundation_unchanged": foundation_unchanged,
            "costs": {
                "prefix_environment_steps": int(prefix_summary["steps"]),
                "branch_environment_steps": int(total_branch_steps),
                "environment_steps_total": int(prefix_summary["steps"] + total_branch_steps),
                "prefix_cpu_s": prefix_cpu_s,
                "branch_cpu_s": branch_cpu_total,
                "case_cpu_s": float(time.process_time() - start_cpu),
                "prefix_wall_s": prefix_wall_s,
                "branch_wall_s": branch_wall_total,
                "case_wall_s": float(time.perf_counter() - start_wall),
            },
            "errors": errors,
        }
        _write_json(case_folder / "summary.json", result)
        return result
    except Exception:
        if env is not None:
            env.close()
        persisted = _read_persisted_counts(case_folder)
        error = {
            "case_id": case_id,
            "error_type": "case_failure",
            "error": traceback.format_exc(),
            "persisted_interactions": persisted,
            "count_source": "steps.csv and rollouts.csv rows",
        }
        errors.append(error)
        result = {
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "mapping": pair_mapping(seed, pair_kind),
            "prefix": prefix_summary,
            "branches": branch_results,
            "costs": {
                **persisted,
                "case_cpu_s": float(time.process_time() - start_cpu),
                "case_wall_s": float(time.perf_counter() - start_wall),
                "prefix_cpu_s": prefix_cpu_s,
                "prefix_wall_s": prefix_wall_s,
                "branch_cpu_s": sum(float(row.get("costs", {}).get("branch_cpu_s", 0.0)) for row in branch_results),
                "branch_wall_s": sum(float(row.get("costs", {}).get("branch_wall_s", 0.0)) for row in branch_results),
            },
            "errors": errors,
        }
        _write_json(case_folder / "error.json", error)
        _write_json(case_folder / "summary.json", result)
        return result


def _case_payloads(config: Mapping[str, Any], output: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(seed),
            "pair_kind": str(pair_kind),
            "history_regime": str(history_regime),
            "config": dict(config),
            "output": str(output),
        }
        for seed in seeds
        for pair_kind in config["pair_kinds"]
        for history_regime in config["history_regimes"]
    ]


def _empty_case_error(job: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed = int(job["seed"])
    pair_kind = str(job["pair_kind"])
    history_regime = str(job["history_regime"])
    case_id = f"s{seed}_{pair_kind}_{history_regime}"
    folder = output / "cases" / case_id
    persisted = _read_persisted_counts(folder)
    error = {
        "case_id": case_id,
        "error_type": "case_worker_failure",
        "error": traceback.format_exc(),
        "persisted_interactions": persisted,
        "count_source": "steps.csv and rollouts.csv rows",
    }
    return {
        "case_id": case_id,
        "seed": seed,
        "pair_kind": pair_kind,
        "history_regime": history_regime,
        "branches": [],
        "costs": {
            **persisted,
            "case_cpu_s": None,
            "case_wall_s": None,
            "prefix_cpu_s": None,
            "branch_cpu_s": None,
            "prefix_wall_s": None,
            "branch_wall_s": None,
        },
        "errors": [error],
    }


def run(
    config: Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    seeds: Sequence[int] | None = None,
    pilot: bool = False,
) -> dict[str, Any]:
    config_path = Path(config).resolve() if isinstance(config, (str, Path)) else CONFIG_PATH
    raw_config = json.loads(config_path.read_text(encoding="utf-8")) if isinstance(config, (str, Path)) else dict(config)
    normalized = _phase_config(raw_config, pilot=pilot)
    if seeds is not None:
        normalized["seeds"] = [int(value) for value in seeds]
        if not normalized["seeds"]:
            raise ValueError("at least one seed is required")
    output_path = Path(output).resolve()
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"Preserve existing mechanism output: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    source_manifest = _snapshot_sources(output_path, config_path)
    _write_json(output_path / "config.json", normalized)
    _write_json(output_path / "source_snapshot" / "config_original.json", raw_config)
    _write_json(output_path / "frozen_run.json", {
        "phase": normalized["phase"],
        "config": normalized,
        "config_sha256": _sha(config_path) if config_path.is_file() else None,
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "source_manifest": source_manifest,
    })

    jobs = _case_payloads(normalized, output_path, normalized["seeds"])
    requested_workers = int(normalized["max_workers"])
    workers = max(1, min(2, requested_workers, len(jobs)))
    executor_mode = "inline" if workers == 1 else "process_pool"
    parent_start_cpu, campaign_start_wall = time.process_time(), time.perf_counter()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if workers == 1:
        for job in jobs:
            try:
                records.append(_case_worker(job))
            except Exception:
                record = _empty_case_error(job, output_path)
                records.append(record)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_case_worker, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    records.append(future.result())
                except Exception:
                    record = _empty_case_error(job, output_path)
                    records.append(record)
    records.sort(key=lambda row: (int(row["seed"]), str(row["pair_kind"]), str(row["history_regime"])))
    errors.extend(error for row in records for error in row.get("errors", []))
    branches = [branch for row in records for branch in row.get("branches", [])]
    prefix_steps = sum(
        int(row.get("prefix", {}).get("steps", row.get("costs", {}).get("prefix_environment_steps", 0)) or 0)
        for row in records
    )
    branch_steps = sum(
        int(row.get("costs", {}).get("branch_environment_steps", 0) or 0)
        for row in records
    )
    child_cpu_values = [row.get("costs", {}).get("case_cpu_s") for row in records]
    parent_cpu = float(time.process_time() - parent_start_cpu)
    campaign_cpu = _campaign_cpu_accounting(executor_mode, parent_cpu, child_cpu_values)
    expected = _expected_interactions(normalized)
    configured_expected = normalized.get("expected_pilot" if pilot else "expected_main", {})
    expected_case_count = int(expected["cases"])
    expected_branch_count = int(expected["branches"])
    count_errors: list[dict[str, Any]] = []
    if len(records) != expected_case_count:
        count_errors.append({"error_type": "case_count_mismatch", "expected": expected_case_count, "actual": len(records)})
    if len(branches) != expected_branch_count:
        count_errors.append({"error_type": "branch_count_mismatch", "expected": expected_branch_count, "actual": len(branches)})
    expected_steps = int(expected["total_steps"])
    actual_steps = int(prefix_steps + branch_steps)
    if actual_steps != expected_steps:
        count_errors.append({"error_type": "interaction_count_mismatch", "expected": expected_steps, "actual": actual_steps})
    errors.extend(count_errors)
    costs = {
        "prefix_environment_steps": int(prefix_steps),
        "branch_environment_steps": int(branch_steps),
        "environment_steps_total": actual_steps,
        "expected_environment_steps_total": expected_steps,
        "expected_interactions": expected,
        "configured_expected_interactions": configured_expected,
        "executor_mode": executor_mode,
        "requested_workers": requested_workers,
        "actual_workers": workers,
        **campaign_cpu,
        "elapsed_wall_s": float(time.perf_counter() - campaign_start_wall),
        "prefix_cpu_s": float(sum(float(row.get("costs", {}).get("prefix_cpu_s", 0.0) or 0.0) for row in records)),
        "branch_cpu_s": float(sum(float(row.get("costs", {}).get("branch_cpu_s", 0.0) or 0.0) for row in records)),
        "prefix_wall_s": float(sum(float(row.get("costs", {}).get("prefix_wall_s", 0.0) or 0.0) for row in records)),
        "branch_wall_s": float(sum(float(row.get("costs", {}).get("branch_wall_s", 0.0) or 0.0) for row in records)),
    }
    summary = {
        "format": SUMMARY_FORMAT,
        "phase": normalized["phase"],
        "config": normalized,
        "source_manifest": source_manifest,
        "seeds": list(normalized["seeds"]),
        "case_count": len(records),
        "branch_count": len(branches),
        "cases": records,
        "branches": branches,
        "errors": errors,
        "costs": costs,
    }
    _write_json(output_path / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--pilot", action="store_true")
    arguments = parser.parse_args(argv)
    summary = run(arguments.config, arguments.output, seeds=arguments.seeds, pilot=arguments.pilot)
    print(json.dumps({
        "output": str(arguments.output),
        "phase": summary["phase"],
        "case_count": summary["case_count"],
        "branch_count": summary["branch_count"],
        "environment_steps_total": summary["costs"]["environment_steps_total"],
        "errors": len(summary["errors"]),
        "costs": summary["costs"],
    }, indent=2), flush=True)
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

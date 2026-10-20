"""Run the frozen candidate-content by state-coverage retention diagnostic."""
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

from online_rl import retention_behavioral_candidate as behavioral
from online_rl.retention_stream_value import apply_retention_action
from online_rl.storage_decision import (
    common_mean_effective,
    effective_matrices,
    pair_mapping,
    semantic_hash,
    task_name,
    template_schedule,
    compensated_single_parent_install,
    bootstrap_schedule,
)
from scripts import run_knowledge_loop as loop
from scripts import run_retention_mechanism as mechanism
from scripts import run_storage_decision as storage
from scripts import run_write_stream as ws


CONFIG_PATH = ROOT / "configs" / "retention_candidate.json"
PROTOCOL_PATH = ROOT / "docs" / "RETENTION_CANDIDATE_PROTOCOL.md"
SUMMARY_FORMAT = "retention_candidate_summary_v1"
ARMS = ("keep", "replace_mean", "replace_behavioral", "merge_mean", "merge_behavioral")
STATES = ("early", "late")
_ACTION_RNG_STREAM = mechanism._ACTION_RNG_STREAM
_BEHAVIOR_RNG_STREAM = mechanism._BEHAVIOR_RNG_STREAM

SOURCE_FILES = tuple(dict.fromkeys(
    list(mechanism.SOURCE_FILES)
    + [
        Path("scripts/run_retention_candidate.py"),
        Path("src/online_rl/retention_behavioral_candidate.py"),
        Path("configs/retention_candidate.json"),
        Path("docs/RETENTION_CANDIDATE_PROTOCOL.md"),
    ]
))


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


def _candidate_array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the candidate harness fields over the shared storage config."""
    if config.get("candidate_format") != "retention_candidate_v1":
        raise ValueError("candidate_format must be retention_candidate_v1")
    result = storage.validate_config(config)
    if result.get("history_regime_by_seed") != "old_major_if_even_else_new_major":
        raise ValueError("history_regime_by_seed must use the frozen parity rule")
    if result.get("states") != list(STATES):
        raise ValueError(f"states must be ordered as {STATES}")
    if result.get("arms") != list(ARMS):
        raise ValueError(f"arms must be ordered as {ARMS}")
    if int(result.get("bootstrap_steps", 0)) != 1024:
        raise ValueError("bootstrap_steps is frozen to 1024")
    if int(result.get("early_source_steps", 0)) != 1024:
        raise ValueError("early_source_steps is frozen to 1024")
    if int(result.get("late_source_steps", 0)) != 4096:
        raise ValueError("late_source_steps is frozen to 4096")
    if int(result.get("late_warmup_cycles", 0)) != 4:
        raise ValueError("late_warmup_cycles is frozen to four")
    if int(result.get("warmup_pair_steps", 0)) != 1024:
        raise ValueError("warmup_pair_steps is frozen to 1024")
    if result.get("warmup_write_rule") != "replace_old_group_only_after_pair":
        raise ValueError("warmup_write_rule must preserve the old group parent across new-only pairs")
    if int(result.get("evidence_rows_per_source", 0)) != 128:
        raise ValueError("evidence_rows_per_source is frozen to 128")
    if int(result.get("max_workers", 0)) not in (1, 2):
        raise ValueError("max_workers must be one or two")
    if int(result.get("torch_threads", 0)) != 1:
        raise ValueError("torch_threads is frozen to one")
    if int(result.get("future_seed_offset", -1)) != 50_000_000:
        raise ValueError("future_seed_offset is frozen to 50000000")
    if int(result.get("future_seed_multiplier", -1)) != 1000:
        raise ValueError("future_seed_multiplier is frozen to 1000")

    fit = result.get("behavioral_fit")
    if not isinstance(fit, Mapping):
        raise ValueError("behavioral_fit must be an object")
    frozen_fit = {
        "train_fraction": 0.75,
        "min_rows_per_owner": 16,
        "ridge": 0.0001,
        "maxiter": 64,
        "maxfun": 256,
        "source_weighting": "equal_mean_per_owner",
        "objective": "forward_KL_teacher_to_student_plus_ridge_to_mean",
        "candidate_selection": "best_training_objective_including_initial_mean",
        "holdout_role": "diagnostic_only_not_used_for_candidate_selection",
    }
    for name, expected in frozen_fit.items():
        if fit.get(name) != expected:
            raise ValueError(f"behavioral_fit.{name} is frozen to {expected!r}")
    result["behavioral_fit"] = dict(fit)

    seeds = [int(value) for value in result.get("seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list of unique integers")
    result["seeds"] = seeds
    replicates = [int(value) for value in result.get("replicates", [])]
    if not replicates or len(replicates) != len(set(replicates)) or any(value < 0 for value in replicates):
        raise ValueError("replicates must be a nonempty list of unique nonnegative integers")
    result["replicates"] = replicates
    result["states"] = list(result["states"])
    result["arms"] = list(result["arms"])
    horizons = [int(value) for value in result.get("horizons", [])]
    maximum = int(result.get("future_blocks", 0)) * 3 * int(result["bout_steps"])
    if not horizons or horizons != sorted(set(horizons)):
        raise ValueError("horizons must be a nonempty, strictly increasing list")
    if any(value < 1 or value % int(result["rollout_steps"]) or value > maximum for value in horizons):
        raise ValueError("horizons must align to rollout boundaries within the future schedule")
    result["horizons"] = horizons
    return result


def _phase_config(raw_config: Mapping[str, Any], *, pilot: bool) -> dict[str, Any]:
    config = copy.deepcopy(dict(raw_config))
    if pilot:
        pilot_config = config.get("pilot")
        if not isinstance(pilot_config, Mapping):
            raise ValueError("pilot config must be an object")
        for name in ("seeds", "pair_kinds", "states", "future_regimes", "replicates", "max_workers"):
            if name not in pilot_config:
                raise ValueError(f"pilot config is missing {name}")
            config[name] = copy.deepcopy(pilot_config[name])
    normalized = validate_config(config)
    normalized["phase"] = "pilot" if pilot else "main"
    return normalized


def history_regime_for_seed(seed: int) -> str:
    return "old_major" if int(seed) % 2 == 0 else "new_major"


def _prefix_step_count(config: Mapping[str, Any], state: str) -> int:
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}")
    tail_steps = int(config["early_source_steps"] if state == "early" else config["late_source_steps"])
    bootstrap_steps = 2 * int(config["bootstrap_steps"])
    history_steps = 6 * int(config["bout_steps"]) + 2 * tail_steps
    warmup_steps = 0 if state == "early" else (
        int(config["late_warmup_cycles"]) * 4 * int(config["warmup_pair_steps"])
    )
    return bootstrap_steps + warmup_steps + history_steps


def _expected_interactions(config: Mapping[str, Any]) -> dict[str, int]:
    case_count = len(config["seeds"]) * len(config["pair_kinds"]) * len(config["states"])
    branches_per_case = len(config["future_regimes"]) * len(config["replicates"]) * len(config["arms"])
    early_cases = len(config["seeds"]) * len(config["pair_kinds"]) * config["states"].count("early")
    late_cases = len(config["seeds"]) * len(config["pair_kinds"]) * config["states"].count("late")
    early_prefix_steps = _prefix_step_count(config, "early")
    late_prefix_steps = _prefix_step_count(config, "late")
    prefix_steps = early_cases * early_prefix_steps + late_cases * late_prefix_steps
    steps_per_branch = int(config["future_blocks"]) * 3 * int(config["bout_steps"])
    branch_steps = case_count * branches_per_case * steps_per_branch
    return {
        "cases": int(case_count),
        "branches": int(case_count * branches_per_case),
        "early_prefix_steps": int(early_prefix_steps),
        "late_prefix_steps": int(late_prefix_steps),
        "prefix_steps": int(prefix_steps),
        "steps_per_branch": int(steps_per_branch),
        "branch_steps": int(branch_steps),
        "total_steps": int(prefix_steps + branch_steps),
    }


def build_prefix_plan(
    mapping: Mapping[str, Any], config: Mapping[str, Any], state: str, history_regime: str,
) -> dict[str, Any]:
    """Build bootstrap, late pair warmups, and the common eight-bout history."""
    normalized = validate_config(config)
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}")
    if history_regime not in normalized["history_regimes"]:
        raise ValueError("history_regime is not configured")
    bootstrap = bootstrap_schedule(mapping, int(normalized["bootstrap_steps"]))
    next_bout = len(bootstrap)
    warmup_pairs: list[dict[str, Any]] = []
    if state == "late":
        pair_steps = int(normalized["warmup_pair_steps"])
        for cycle in range(int(normalized["late_warmup_cycles"])):
            for group in (1, 0):
                phase = "late_warmup_new" if group == 1 else "late_warmup_old"
                rows = [
                    {
                        "phase": phase,
                        "bout": next_bout + variant,
                        "group": group,
                        "variant": variant,
                        "task": task_name(mapping, group, variant),
                        "steps": pair_steps,
                        "warmup_cycle": cycle,
                        "pair_group": group,
                    }
                    for variant in (0, 1)
                ]
                warmup_pairs.append({
                    "cycle": cycle,
                    "group": group,
                    "phase": phase,
                    "write_applied_after_pair": bool(group == 0),
                    "schedule": rows,
                })
                next_bout += len(rows)

    history = template_schedule(
        mapping,
        history_regime,
        kind="history",
        steps=int(normalized["bout_steps"]),
        start_bout=next_bout,
    )
    source_steps = int(normalized["early_source_steps"] if state == "early" else normalized["late_source_steps"])
    for row in history[-2:]:
        row["steps"] = source_steps
    if any(int(row["group"]) != 1 or int(row["variant"]) not in (0, 1) for row in history[-2:]):
        raise RuntimeError("history must end with new group variants zero and one")
    schedule = bootstrap + [row for pair in warmup_pairs for row in pair["schedule"]] + history
    steps = sum(int(row["steps"]) for row in schedule)
    expected = _prefix_step_count(normalized, state)
    if steps != expected:
        raise RuntimeError(f"{state} prefix schedule has {steps} steps, expected {expected}")
    return {
        "state": state,
        "history_regime": history_regime,
        "bootstrap": bootstrap,
        "warmup_pairs": warmup_pairs,
        "history": history,
        "schedule": schedule,
        "steps": steps,
        "candidate_sources": history[-2:],
    }


def _branch_specs(config: Mapping[str, Any], case_id: str) -> tuple[list[dict[str, Any]], int]:
    specs = [
        {
            "branch_id": f"{future}__r{replicate}__{arm}",
            "future_regime": str(future),
            "replicate": int(replicate),
            "arm": str(arm),
        }
        for future in config["future_regimes"]
        for replicate in config["replicates"]
        for arm in config["arms"]
    ]
    material = f"{config['execution_order_seed']}|{case_id}".encode("utf-8")
    order_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    order = np.random.default_rng(order_seed).permutation(len(specs)).tolist()
    return [specs[index] for index in order], order_seed


def _snapshot_sources(output: Path, config_path: Path) -> dict[str, Any]:
    snapshot = output / "source_snapshot"
    files: dict[str, dict[str, Any]] = {}
    paths = list(dict.fromkeys(SOURCE_FILES))
    config_resolved = config_path.resolve()
    if config_resolved.is_relative_to(ROOT.resolve()):
        relative = Path(os.path.relpath(config_resolved, ROOT.resolve()))
        if relative not in paths:
            paths.append(relative)
    for relative in paths:
        if any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts):
            raise ValueError("private or temporary material is excluded from source snapshots")
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_digest, copy_digest = _sha(source), _sha(target)
        if source_digest != copy_digest:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        files[relative.as_posix()] = {"sha256": copy_digest, "bytes": int(target.stat().st_size)}
    base = loop.load_base_config()
    foundation, dataset = Path(base["foundation"]), Path(base["dataset"])
    manifest = {
        "sources": files,
        "foundation": {"path": str(foundation), "sha256": _sha(foundation), "bytes": int(foundation.stat().st_size)},
        "dataset": {"path": str(dataset), "sha256": _sha(dataset), "bytes": int(dataset.stat().st_size)},
        "excluded_local_instructions": ["Agent.md", "AGENTS.md"],
    }
    _write_json(snapshot / "manifest.json", manifest)
    return manifest


def _evidence_arrays(owner_rows: Sequence[Sequence[Mapping[str, Any]]], rows_per_source: int) -> dict[str, np.ndarray]:
    if len(owner_rows) != 2 or int(rows_per_source) < 1:
        raise ValueError("evidence requires two ordered owners and a positive row cap")
    ordered: list[Mapping[str, Any]] = []
    for rows in owner_rows:
        if len(rows) < int(rows_per_source):
            raise ValueError("each source must provide the configured number of post-rollout rows")
        ordered.extend(list(rows)[-int(rows_per_source):])
    observations = np.stack([np.asarray(row["obs"], dtype=np.float32) for row in ordered])
    if observations.shape != (2 * int(rows_per_source), 13):
        raise ValueError("evidence observations must have shape at most 256 by 13")
    return {
        "observations": np.ascontiguousarray(observations, dtype=np.float32),
        "owner_indices": np.asarray([owner for owner in range(2) for _ in range(int(rows_per_source))], dtype=np.int8),
        "source_slots": np.asarray([int(row["slot"]) for row in ordered], dtype=np.int8),
        "generations": np.asarray([int(row["generation"]) for row in ordered], dtype=np.int64),
        "bouts": np.asarray([int(row["bout"]) for row in ordered], dtype=np.int64),
        "global_steps": np.asarray([int(row["global_step"]) for row in ordered], dtype=np.int64),
    }


def _write_evidence_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{name: np.asarray(value) for name, value in arrays.items()})
    return {"path": path.name, "bytes": int(path.stat().st_size), "sha256": _sha(path)}


def _copy_evidence(source: Path, target: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target.stat().st_size != int(expected["bytes"]) or _sha(target) != str(expected["sha256"]):
        raise RuntimeError("evidence sidecar changed while copying to a branch")
    return {"path": target.name, "bytes": int(target.stat().st_size), "sha256": _sha(target)}


def _package_config(config: Mapping[str, Any], evidence_bytes: int) -> dict[str, Any]:
    result = dict(config)
    base_budget = int(config["package_budget_bytes"])
    if int(evidence_bytes) < 0 or int(evidence_bytes) >= base_budget:
        raise ValueError("evidence sidecar leaves no space within the full package budget")
    result["package_budget_bytes"] = base_budget - int(evidence_bytes)
    return result


def _save_candidate_package(
    folder: Path,
    learner: Mapping[str, Any],
    actor: bytes,
    foundation: Path,
    config: Mapping[str, Any],
    evidence_bytes: int,
) -> dict[str, int]:
    base = storage._save_package(folder, learner, actor, foundation, _package_config(config, evidence_bytes))
    total = int(base["full_persistent_bytes"]) + int(evidence_bytes)
    if total > int(config["package_budget_bytes"]):
        raise ValueError(
            f"complete persistent package is {total} bytes, exceeding cap {config['package_budget_bytes']}"
        )
    return {
        **base,
        "deployment_state_bytes": int(base["full_persistent_bytes"]),
        "evidence_bytes": int(evidence_bytes),
        "full_persistent_bytes": total,
        "package_budget_bytes": int(config["package_budget_bytes"]),
    }


def _artifact_record(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha(path),
        "bytes": int(path.stat().st_size),
    }


def _candidate_package_artifacts(folder: Path, root: Path) -> dict[str, Any]:
    return {
        "checkpoint": _artifact_record(folder / "checkpoint.pt", root),
        "actor": _artifact_record(folder / "actor.bin", root),
        "simulator": _artifact_record(folder / "simulator.json", root),
        "evidence": _artifact_record(folder / "evidence.npz", root),
    }


def _capture_evidence_from_post_rollout(
    agent: Any,
    *,
    targets: Mapping[int, Mapping[str, Any]],
    rows_per_source: int,
    env: Any,
    runtime: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    folder: Path,
    phase: str,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, np.ndarray]]:
    """Run one schedule and copy only the selected final-rollout evidence rows."""
    original_update = loop._ppo_update
    pending_rows: list[Any] = []
    captured: dict[int, list[dict[str, Any]]] = {}
    target_bouts = {int(key): value for key, value in targets.items()}

    def capture_after_update(target: Any, rows: Sequence[Any]):
        result = original_update(target, rows)
        if target is agent:
            bout = int(getattr(rows[0], "bout", -1)) if rows else -1
            pending_rows[:] = list(rows) if bout in target_bouts else []
        return result

    def post_rollout_observer(**event: Any) -> None:
        record = event.get("bout_record", {})
        bout = int(record.get("bout", -1))
        if bout not in target_bouts:
            return
        owner = int(target_bouts[bout]["owner_index"])
        rows = list(pending_rows)
        if len(rows) != 256:
            raise RuntimeError(f"source bout {bout} final rollout has {len(rows)} rows, expected 256")
        if not rows or any(int(getattr(row, "bout", -1)) != bout for row in rows):
            raise RuntimeError("post-rollout evidence does not match its source bout")
        selected = rows[-int(rows_per_source):]
        expected_slot = int(event.get("slot", -1))
        expected_generation = int(event.get("generation", -1))
        if any(int(row.slot) != expected_slot or int(row.generation) != expected_generation for row in selected):
            raise RuntimeError("post-rollout evidence slot or generation differs from its source")
        captured[owner] = [
            {
                "obs": np.asarray(row.obs, dtype=np.float32).copy(),
                "slot": int(row.slot),
                "generation": int(row.generation),
                "bout": int(getattr(row, "bout", bout)),
                "global_step": int(row.step),
            }
            for row in selected
        ]

    agent._storage_decision_post_rollout_observer = post_rollout_observer
    loop._ppo_update = capture_after_update
    try:
        bouts, counts = mechanism._run_schedule_counted(
            agent, env, runtime, schedule, folder=folder, phase=phase,
        )
    finally:
        loop._ppo_update = original_update
        if hasattr(agent, "_storage_decision_post_rollout_observer"):
            del agent._storage_decision_post_rollout_observer
    if set(captured) != {0, 1}:
        raise RuntimeError(f"expected evidence for owners zero and one, captured {sorted(captured)}")
    owner_rows = [captured[owner] for owner in (0, 1)]
    arrays = _evidence_arrays(owner_rows, int(rows_per_source))
    return bouts, counts, arrays


def _install_pair_mean(
    agent: Any,
    mapping: Mapping[str, Any],
    *,
    group: int,
    variants: Sequence[int] = (0, 1),
    label: str,
) -> dict[str, Any]:
    local, shared, physical_groups = ws._actor_arrays(agent.core)
    before_effective = effective_matrices(local, shared, physical_groups)
    slots = [mechanism._slot_for_task(agent, task_name(mapping, group, variant)) for variant in variants]
    if len(set(slots)) != 2:
        raise RuntimeError(f"{label} source tasks do not occupy two distinct residents")
    parent = common_mean_effective(local, shared, physical_groups, slots)
    source_matrices = before_effective[np.asarray(slots, dtype=np.int64)].copy()
    source_records = []
    for variant, slot in zip(variants, slots):
        task = task_name(mapping, group, variant)
        if int(agent.slot_group_tags[slot]) != int(group):
            raise RuntimeError(f"{label} source {task} has the wrong logical group")
        source_records.append({
            "task": task,
            "group": int(group),
            "variant": int(variant),
            "slot": int(slot),
            "generation": int(agent.slot_generation[slot]),
            "fitted_steps": int(agent.fitted_steps[slot]),
            "effective_matrix": source_matrices[len(source_records)].tolist(),
        })
    new_local, new_shared, new_groups, formation = compensated_single_parent_install(
        local,
        shared,
        physical_groups,
        agent.slot_group_tags,
        parent=parent,
        parent_group=int(group),
    )
    agent.core.install_local_matrix(new_local)
    agent.core.install_shared_matrix(new_shared)
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(new_groups.astype(np.int64), dtype=torch.long))
    agent.parent_group = int(group)
    actor, codec_error, codec_effective_error = storage._encode_live_actor(agent)
    ws._clear_optimizer_state(agent)
    post_local, post_shared, post_groups = ws._actor_arrays(agent.core)
    post_effective = effective_matrices(post_local, post_shared, post_groups)
    return {
        "label": str(label),
        "group": int(group),
        "source_slots": [int(slot) for slot in slots],
        "source_records": source_records,
        "source_effective_matrices": source_matrices,
        "mean_parent": parent.copy(),
        "mean_parent_sha256": semantic_hash(parent),
        "mean_parent_norm": float(np.linalg.norm(parent)),
        "formation": formation,
        "codec_max_coefficient_error": float(codec_error),
        "codec_effective_errors": [float(value) for value in codec_effective_error],
        "encoded_actor_bytes": int(len(actor)),
        "post_effective_matrices": post_effective[np.asarray(slots, dtype=np.int64)].copy(),
    }


def _snapshot_pair_sources(
    agent: Any,
    mapping: Mapping[str, Any],
    *,
    group: int,
) -> dict[str, Any]:
    """Record a locally trained pair without changing shared state."""
    local, shared, physical_groups = ws._actor_arrays(agent.core)
    effective = effective_matrices(local, shared, physical_groups)
    records = []
    slots = []
    for variant in (0, 1):
        task = task_name(mapping, group, variant)
        slot = mechanism._slot_for_task(agent, task)
        if int(agent.slot_group_tags[slot]) != int(group):
            raise RuntimeError(f"local-only source {task} has the wrong logical group")
        records.append({
            "task": task,
            "group": int(group),
            "variant": int(variant),
            "slot": int(slot),
            "generation": int(agent.slot_generation[slot]),
            "fitted_steps": int(agent.fitted_steps[slot]),
            "effective_matrix": effective[slot].copy().tolist(),
        })
        slots.append(slot)
    if len(set(slots)) != 2:
        raise RuntimeError(f"local-only source group {group} does not occupy two distinct residents")
    return {
        "group": int(group),
        "source_slots": slots,
        "source_records": records,
        "source_effective_matrices": effective[np.asarray(slots, dtype=np.int64)].copy(),
    }


def _current_generation_exposures(
    bouts: Sequence[Mapping[str, Any]], slot: int, generation: int,
) -> list[dict[str, Any]]:
    return [
        {
            "bout": int(row["bout"]),
            "task": str(row["task"]),
            "group": int(row["group"]),
            "steps": int(row["steps"]),
            "slot": int(row["slot"]),
            "generation": int(row["generation"]),
        }
        for row in bouts
        if int(row["slot"]) == int(slot) and int(row["generation"]) == int(generation)
    ]


def _source_fit_record(
    *,
    formation_record: Mapping[str, Any],
    prefix_bouts: Sequence[Mapping[str, Any]],
    source_time_bouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    task = str(formation_record["task"])
    slot = int(formation_record["slot"])
    generation = int(formation_record["generation"])
    source_end_bout = max(int(row["bout"]) for row in source_time_bouts)
    later_steps = sum(
        int(row["steps"])
        for row in prefix_bouts
        if int(row["bout"]) > source_end_bout
        and int(row["slot"]) == slot
        and int(row["generation"]) == generation
    )
    exposures = _current_generation_exposures(prefix_bouts, slot, generation)
    return {
        "task": task,
        "group": int(formation_record["group"]),
        "variant": int(formation_record["variant"]),
        "slot": slot,
        "generation": generation,
        "fitted_steps_at_source_time": int(formation_record["fitted_steps"]),
        "source_time_fitted_steps": int(sum(int(row["steps"]) for row in source_time_bouts)),
        "later_same_generation_steps": int(later_steps),
        "current_generation_lifetime_steps": int(sum(int(row["steps"]) for row in exposures)),
        "source_time_end_bout": int(source_end_bout),
        "effective_matrix_at_source_time": np.asarray(formation_record["effective_matrix"], dtype=np.float32).tolist(),
        "source_time_effective_sha256": semantic_hash(np.asarray(formation_record["effective_matrix"], dtype=np.float32)),
        "current_generation_exposures": exposures,
    }


def _candidate_source_records(
    agent: Any,
    mapping: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    prefix_bouts: Sequence[Mapping[str, Any]],
    expected_steps: int,
) -> tuple[list[dict[str, Any]], list[int], np.ndarray]:
    local, shared, groups = ws._actor_arrays(agent.core)
    all_effective = effective_matrices(local, shared, groups)
    records: list[dict[str, Any]] = []
    slots: list[int] = []
    matrices: list[np.ndarray] = []
    for owner, source in enumerate(sources):
        task = str(source["task"])
        slot = mechanism._slot_for_task(agent, task)
        generation = int(agent.slot_generation[slot])
        fitted = int(agent.fitted_steps[slot])
        current_rows = _current_generation_exposures(prefix_bouts, slot, generation)
        source_steps = sum(int(row["steps"]) for row in current_rows)
        if fitted != int(expected_steps) or source_steps != int(expected_steps):
            raise RuntimeError(
                f"candidate source {task} generation fit is {fitted}/{source_steps} steps, expected {expected_steps}"
            )
        if int(agent.slot_group_tags[slot]) != 1:
            raise RuntimeError(f"candidate source {task} is not logical group one")
        if not current_rows or int(current_rows[-1]["bout"]) != int(source["bout"]):
            raise RuntimeError(f"candidate source {task} is not current at the final history bout")
        effective = all_effective[slot].copy()
        records.append({
            "owner_index": int(owner),
            "task": task,
            "group": 1,
            "variant": int(source["variant"]),
            "slot": int(slot),
            "generation": generation,
            "fitted_steps": fitted,
            "source_time_fitted_steps": source_steps,
            "latest_source_bout": int(current_rows[-1]["bout"]),
            "source_bout": int(source["bout"]),
            "effective_matrix": effective.tolist(),
            "effective_sha256": semantic_hash(effective),
            "current_generation_exposures": current_rows,
        })
        slots.append(slot)
        matrices.append(effective)
    return records, slots, np.stack(matrices).astype(np.float32)


def _make_source_fits(
    incumbent_formation: Mapping[str, Any],
    mapping: Mapping[str, Any],
    bootstrap: Sequence[Mapping[str, Any]],
    warmup_pairs: Sequence[Mapping[str, Any]],
    prefix_bouts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected: Mapping[str, Any] | None = None
    for pair in warmup_pairs:
        if int(pair["group"]) == 0:
            selected = pair
    if selected is None:
        source_time_bouts = [row for row in bootstrap if int(row["group"]) == 0]
    else:
        selected_bouts = {int(row["bout"]) for row in selected["schedule"]}
        source_time_bouts = [row for row in prefix_bouts if int(row["bout"]) in selected_bouts]
    records = []
    for formation_record in incumbent_formation["source_records"]:
        task_rows = [row for row in source_time_bouts if str(row["task"]) == str(formation_record["task"])]
        records.append(_source_fit_record(
            formation_record=formation_record,
            prefix_bouts=prefix_bouts,
            source_time_bouts=task_rows,
        ))
    source_matrices = np.asarray(incumbent_formation["source_effective_matrices"], dtype=np.float32)
    return records, source_matrices


def _band_counts_by_owner(observations: np.ndarray, owners: np.ndarray) -> list[list[int]]:
    bands = np.floor(np.clip(np.asarray(observations)[:, 0], 0.0, 1.0) * 3.0).astype(np.int64).clip(0, 2)
    return [
        np.bincount(bands[np.asarray(owners) == owner], minlength=3).astype(int).tolist()
        for owner in (0, 1)
    ]


def _feature_hash(features: behavioral.FrozenPolicyFeatures) -> str:
    return semantic_hash({
        "base_logits": np.asarray(features.base_logits),
        "source_psi": np.asarray(features.source_psi),
        "contrast_basis": np.asarray(features.contrast_basis),
        "temperature": float(features.temperature),
        "epsilon": float(features.epsilon),
    })


def _fit_candidates_once(
    agent: Any,
    env: Any,
    runtime: Mapping[str, Any],
    foundation_sha: str,
    evidence: Mapping[str, np.ndarray],
    candidate_effective: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    observations = np.asarray(evidence["observations"], dtype=np.float32).copy()
    owners = np.asarray(evidence["owner_indices"], dtype=np.int8).copy()
    observations_before = observations.copy()
    owners_before = owners.copy()
    matrices_before = np.asarray(candidate_effective, dtype=np.float32).copy()
    mean_candidate = matrices_before.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    learner_before = storage._capture_learner(agent, foundation_sha)
    learner_hash_before = semantic_hash(learner_before)
    rng_before = mechanism._rng_metadata(agent, env, runtime)
    extraction_cpu, extraction_wall = time.process_time(), time.perf_counter()
    features = behavioral.extract_frozen_policy_features(agent.core, observations)
    feature_extraction_cpu = float(time.process_time() - extraction_cpu)
    feature_extraction_wall = float(time.perf_counter() - extraction_wall)
    fit_start_cpu, fit_start_wall = time.process_time(), time.perf_counter()
    behavioral_candidate, diagnostics = behavioral.fit_behavioral_candidate(
        observations,
        owners,
        matrices_before,
        features,
        min_rows_per_owner=int(config["behavioral_fit"]["min_rows_per_owner"]),
    )
    behavioral_fit_cpu = float(time.process_time() - fit_start_cpu)
    behavioral_fit_wall = float(time.perf_counter() - fit_start_wall)
    learner_after = storage._capture_learner(agent, foundation_sha)
    learner_hash_after = semantic_hash(learner_after)
    rng_after = mechanism._rng_metadata(agent, env, runtime)
    input_unchanged = bool(
        np.array_equal(observations, observations_before)
        and np.array_equal(owners, owners_before)
        and np.array_equal(matrices_before, candidate_effective)
    )
    unchanged = bool(learner_hash_before == learner_hash_after and rng_before == rng_after and input_unchanged)
    if not unchanged:
        raise RuntimeError("past-only candidate fitting changed learner/RNG/input state")
    fit_cpu = feature_extraction_cpu + behavioral_fit_cpu
    fit_wall = feature_extraction_wall + behavioral_fit_wall
    module_candidate_hash = str(diagnostics.get("parameter_hashes", {}).get("candidate_sha256", ""))
    if module_candidate_hash != _candidate_array_hash(behavioral_candidate):
        raise RuntimeError("behavioral fitter candidate hash differs from the returned float32 matrix")
    source_rows = np.bincount(owners, minlength=2).astype(int).tolist()
    result = {
        "fit_calls": 1,
        "fit_cpu_s": float(fit_cpu),
        "fit_wall_s": float(fit_wall),
        "feature_extraction_cpu_s": float(feature_extraction_cpu),
        "feature_extraction_wall_s": float(feature_extraction_wall),
        "behavioral_fit_cpu_s": float(behavioral_fit_cpu),
        "behavioral_fit_wall_s": float(behavioral_fit_wall),
        "learner_hash_before": learner_hash_before,
        "learner_hash_after": learner_hash_after,
        "rng_hashes_before": rng_before,
        "rng_hashes_after": rng_after,
        "fit_state_unchanged": unchanged,
        "feature_sha256": _feature_hash(features),
        "observations_sha256": semantic_hash(observations),
        "owner_indices_sha256": semantic_hash(owners),
        "effective_matrices_sha256": semantic_hash(matrices_before),
        "source_rows_by_owner": source_rows,
        "feature_rows_by_owner": source_rows,
        "band_counts_by_owner": _band_counts_by_owner(observations, owners),
        "mean_candidate_sha256": semantic_hash(mean_candidate),
        "mean_candidate_norm": float(np.linalg.norm(mean_candidate)),
        "behavioral_candidate_sha256": semantic_hash(behavioral_candidate),
        "behavioral_candidate_module_sha256": module_candidate_hash,
        "behavioral_candidate_norm": float(np.linalg.norm(behavioral_candidate)),
        "transient_candidate_bytes": int(mean_candidate.nbytes + behavioral_candidate.nbytes),
        "behavioral": diagnostics,
    }
    return {"mean": mean_candidate, "behavioral": behavioral_candidate}, result


def _policy_probs_for_source_owners(
    core: Any,
    observations: np.ndarray,
    owners: np.ndarray,
    source_effective: np.ndarray,
) -> np.ndarray:
    probabilities = np.empty((len(observations), 4), dtype=np.float64)
    for owner in (0, 1):
        mask = np.asarray(owners) == owner
        if np.any(mask):
            probabilities[mask] = storage._coefficient_probs(core, observations[mask], source_effective[owner])
    return probabilities


def _teacher_kl_metrics(
    core: Any,
    observations: np.ndarray,
    owners: np.ndarray,
    source_effective: np.ndarray,
    parent: np.ndarray,
    rows_per_source: int,
) -> dict[str, Any]:
    teacher = _policy_probs_for_source_owners(core, observations, owners, source_effective)
    student = storage._coefficient_probs(core, observations, np.asarray(parent, dtype=np.float32).reshape(3, 3))
    teacher_safe = np.clip(teacher, 1e-12, 1.0)
    student_safe = np.clip(student, 1e-12, 1.0)
    row_kl = np.sum(teacher_safe * (np.log(teacher_safe) - np.log(student_safe)), axis=1)
    by_owner = [float(np.mean(row_kl[np.asarray(owners) == owner])) for owner in (0, 1)]
    train_by_owner: list[float] = []
    holdout_by_owner: list[float] = []
    for owner in (0, 1):
        owner_positions = np.flatnonzero(np.asarray(owners) == owner)
        split_at = int(np.floor(0.75 * len(owner_positions)))
        train_by_owner.append(float(np.mean(row_kl[owner_positions[:split_at]])))
        holdout_by_owner.append(float(np.mean(row_kl[owner_positions[split_at:]])))
    return {
        "mean": float(sum(by_owner) / 2.0),
        "by_owner": by_owner,
        "train_mean": float(sum(train_by_owner) / 2.0),
        "train_by_owner": train_by_owner,
        "holdout_mean": float(sum(holdout_by_owner) / 2.0),
        "holdout_by_owner": holdout_by_owner,
        "rows_by_owner": [int(np.sum(np.asarray(owners) == owner)) for owner in (0, 1)],
        "row_kl_sha256": semantic_hash(row_kl.astype(np.float64)),
    }


def _capture_effective_at_slots(agent: Any, slots: Sequence[int]) -> np.ndarray:
    local, shared, groups = ws._actor_arrays(agent.core)
    effective = effective_matrices(local, shared, groups)
    return effective[np.asarray(slots, dtype=np.int64)].copy()


def _candidate_metadata_for_observer(
    incumbent_sources: Sequence[Mapping[str, Any]],
    candidate_sources: Sequence[Mapping[str, Any]],
    incumbent_effective: np.ndarray,
    candidate_effective: np.ndarray,
    candidate_fit: Mapping[str, Any],
    evidence_record: Mapping[str, Any],
    evidence: Mapping[str, np.ndarray],
    current_parent: np.ndarray,
    parent_group: int,
    inherited_states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    owner_indices = np.asarray(evidence["owner_indices"], dtype=np.int8)
    return {
        "format": "retention_candidate_observer_prefix_v1",
        "accessible_to_deployed_selector": False,
        "included_in_deployment_capacity": False,
        "incumbent_sources": [dict(row) for row in incumbent_sources],
        "candidate_sources": [dict(row) for row in candidate_sources],
        "incumbent_source_effective_matrices": np.asarray(incumbent_effective, dtype=np.float32).tolist(),
        "incumbent_source_effective_matrices_sha256": semantic_hash(np.asarray(incumbent_effective, dtype=np.float32)),
        "candidate_source_effective_matrices": np.asarray(candidate_effective, dtype=np.float32).tolist(),
        "candidate_source_effective_matrices_sha256": semantic_hash(np.asarray(candidate_effective, dtype=np.float32)),
        "current_mean_candidate_sha256": str(candidate_fit["mean_candidate_sha256"]),
        "current_behavioral_candidate_sha256": str(candidate_fit["behavioral_candidate_sha256"]),
        "current_candidate_effective_norm": float(np.linalg.norm(np.asarray(candidate_effective, dtype=np.float32))),
        "current_candidate_mean_parent_norm": float(candidate_fit["mean_candidate_norm"]),
        "current_candidate_behavioral_parent_norm": float(candidate_fit["behavioral_candidate_norm"]),
        "current_parent": np.asarray(current_parent, dtype=np.float32).reshape(3, 3).tolist(),
        "current_parent_sha256": semantic_hash(np.asarray(current_parent, dtype=np.float32).reshape(3, 3)),
        "current_parent_norm": float(np.linalg.norm(current_parent)),
        "parent_group": int(parent_group),
        "inherited_states": [dict(row) for row in inherited_states],
        "evidence_buffer": {
            **dict(evidence_record),
            "rows": int(len(owner_indices)),
            "rows_by_owner": np.bincount(owner_indices, minlength=2).astype(int).tolist(),
            "latest_source_global_steps": [
                int(np.max(np.asarray(evidence["global_steps"])[owner_indices == owner]))
                for owner in (0, 1)
            ],
            "owner_order": [str(row["task"]) for row in candidate_sources],
            "source_slots": np.asarray(evidence["source_slots"], dtype=np.int8).astype(int).tolist(),
            "generations": np.asarray(evidence["generations"], dtype=np.int64).astype(int).tolist(),
            "bouts": np.asarray(evidence["bouts"], dtype=np.int64).astype(int).tolist(),
            "global_steps": np.asarray(evidence["global_steps"], dtype=np.int64).astype(int).tolist(),
        },
        "candidate_fit": dict(candidate_fit),
    }


def _current_candidate_fit_records(
    agent: Any,
    mapping: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    prefix_bouts: Sequence[Mapping[str, Any]],
    expected_steps: int,
    evidence: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[int], np.ndarray]:
    records, slots, matrices = _candidate_source_records(
        agent, mapping, sources, prefix_bouts, expected_steps,
    )
    owner_indices = np.asarray(evidence["owner_indices"], dtype=np.int8)
    evidence_slots = np.asarray(evidence["source_slots"], dtype=np.int64)
    evidence_generations = np.asarray(evidence["generations"], dtype=np.int64)
    evidence_bouts = np.asarray(evidence["bouts"], dtype=np.int64)
    evidence_steps = np.asarray(evidence["global_steps"], dtype=np.int64)
    for owner, record in enumerate(records):
        mask = owner_indices == owner
        if not np.any(mask):
            raise RuntimeError(f"candidate owner {owner} has no bounded evidence rows")
        if (
            np.any(evidence_slots[mask] != int(record["slot"]))
            or np.any(evidence_generations[mask] != int(record["generation"]))
            or np.any(evidence_bouts[mask] != int(record["source_bout"]))
        ):
            raise RuntimeError(f"candidate evidence provenance does not match owner {owner} source")
        record["latest_source_global_step"] = int(np.max(evidence_steps[mask]))
        record["evidence_rows"] = int(np.sum(mask))
        record["evidence_first_global_step"] = int(np.min(evidence_steps[mask]))
        record["evidence_last_global_step"] = int(np.max(evidence_steps[mask]))
        if int(record["evidence_rows"]) != 128:
            raise RuntimeError(f"candidate owner {owner} evidence has {record['evidence_rows']} rows, expected 128")
    return records, slots, matrices


def _install_arm_action(
    restored: Any,
    candidate: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str, str]:
    if arm == "keep":
        operation = "keep"
        candidate_kind = "not_used"
        candidate_hash = ""
    elif arm.startswith("replace_"):
        operation = "replace"
        candidate_kind = arm.removeprefix("replace_")
        candidate_hash = semantic_hash(candidate)
    elif arm.startswith("merge_"):
        operation = "mean"
        candidate_kind = arm.removeprefix("merge_")
        candidate_hash = semantic_hash(candidate)
    else:
        raise ValueError(f"unsupported arm {arm}")
    local, shared, physical_groups = ws._actor_arrays(restored.core)
    logical_tags = np.asarray(restored.slot_group_tags, dtype=np.int8)
    current_candidate_slots = [
        slot for slot, group in enumerate(logical_tags.tolist())
        if int(group) == 1 and restored.cache.slot_to_task[slot] is not None
    ]
    if len(current_candidate_slots) != 2:
        raise RuntimeError(f"expected two current candidate residents, found {current_candidate_slots}")
    candidate_groups = {int(logical_tags[slot]) for slot in current_candidate_slots}
    if candidate_groups != {1}:
        raise RuntimeError("candidate sources do not share logical group one")
    incumbent_binding = int(restored.parent_group)
    before_effective = storage._effective_live(local, shared, physical_groups)
    post_local, post_shared, post_groups, intervention = apply_retention_action(
        local,
        shared,
        physical_groups,
        logical_tags,
        incumbent_binding,
        candidate,
        1,
        operation,
    )
    if operation != "keep":
        restored.core.install_local_matrix(post_local)
        restored.core.install_shared_matrix(post_shared)
        with torch.no_grad():
            restored.core.slot_groups.copy_(torch.as_tensor(post_groups.astype(np.int64), dtype=torch.long))
    restored.parent_group = int(intervention["parent_group"])
    return (
        before_effective,
        post_local,
        post_shared,
        {
            **intervention,
            "incumbent_binding": incumbent_binding,
            "candidate_group": 1,
            "candidate_slots": [int(slot) for slot in current_candidate_slots],
            "candidate_effective_pre_action": before_effective[np.asarray(current_candidate_slots, dtype=np.int64)].copy(),
        },
        operation,
        candidate_kind,
    )


def _attach_package_aliases(capacity: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(capacity)
    result["package_bytes"] = int(result["full_persistent_bytes"])
    result["actor_container_bytes"] = int(result["actor_bytes"])
    return result


def _run_branch(
    *,
    config: Mapping[str, Any],
    case_folder: Path,
    output: Path,
    case_id: str,
    seed: int,
    pair_kind: str,
    state: str,
    history_regime: str,
    mapping: Mapping[str, Any],
    plan: Mapping[str, Any],
    spec: Mapping[str, Any],
    base_config: Mapping[str, Any],
    foundation: Path,
    foundation_sha: str,
    prefix_folder: Path,
    prefix_hash: str,
    prefix_learner_hash: str,
    prefix_actor: bytes,
    prefix_runtime_counts: Mapping[str, Any],
    prefix_learner_counts: Mapping[str, Any],
    optimizer_start_count: int,
    prefix_files_before: Mapping[str, Mapping[str, Any]],
    evidence_record: Mapping[str, Any],
    evidence_arrays: Mapping[str, np.ndarray],
    source_effective: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    candidate_fit: Mapping[str, Any],
    candidate_slots: Sequence[int],
) -> dict[str, Any]:
    branch_id = str(spec["branch_id"])
    branch_folder = case_folder / "branches" / branch_id
    branch_folder.mkdir(parents=True, exist_ok=False)
    _copy_evidence(prefix_folder / "evidence.npz", branch_folder / "evidence.npz", evidence_record)
    branch_start_cpu, branch_start_wall = time.process_time(), time.perf_counter()
    setup_cpu_s = setup_wall_s = 0.0
    install_cpu_s = install_wall_s = 0.0
    simulation_cpu_s = simulation_wall_s = 0.0
    serialization_cpu_s = serialization_wall_s = 0.0
    restored = None
    restored_env = None
    branch_bouts: list[dict[str, Any]] = []
    cpu_milestones: dict[int, dict[str, Any]] = {}
    future_value = mechanism.future_seed(config, seed, int(spec["replicate"]))
    branch_error: dict[str, Any] | None = None
    prefix_runtime_start: dict[str, Any] | None = None
    prefix_learner_start: dict[str, Any] | None = None
    optimizer_count_start: int | None = None
    branch_ppo_counts: dict[str, int] = {}
    horizons: list[dict[str, Any]] = []
    action_install: dict[str, Any] = {}
    capacity_after_action: dict[str, Any] | None = None
    post_action_artifacts: dict[str, Any] | None = None
    action_hash: str | None = None
    restored_hash: str | None = None
    post_transform_hash: str | None = None
    rng_before: dict[str, Any] | None = None
    rng_after: dict[str, Any] | None = None
    reseed_metadata: dict[str, Any] | None = None
    try:
        setup_cpu, setup_wall = time.process_time(), time.perf_counter()
        serialized_learner, serialized_simulator, prefix_actor_bytes = mechanism._load_prefix_payloads(prefix_folder, prefix_hash)
        restored = storage._restore_learner(serialized_learner, base_config, config)
        restored_env, restored_runtime = storage._restore_simulator(serialized_simulator)
        restored_payload = storage._capture_learner(restored, foundation_sha)
        restored_snapshot = storage._capture_simulator(restored_env, restored_runtime)
        restored_hash = mechanism._verify_restored_hash(prefix_hash, restored_payload, restored_snapshot)
        rng_before = mechanism._rng_metadata(restored, restored_env, restored_runtime)
        environment_hash_before = rng_before["environment_rng_sha256"]
        reseed_metadata = mechanism._reseed_future(restored, future_value)
        rng_after = mechanism._rng_metadata(restored, restored_env, restored_runtime)
        if rng_after["environment_rng_sha256"] != environment_hash_before:
            raise RuntimeError("future RNG transformation changed the restored environment RNG")
        post_transform_learner = storage._capture_learner(restored, foundation_sha)
        post_transform_simulator = storage._capture_simulator(restored_env, restored_runtime)
        post_transform_hash = semantic_hash({"learner": post_transform_learner, "simulator": post_transform_simulator})
        setup_cpu_s = float(time.process_time() - setup_cpu)
        setup_wall_s = float(time.perf_counter() - setup_wall)
        prefix_runtime_start = dict(restored.runtime_counts)
        prefix_learner_start = dict(restored.counts)
        optimizer_count_start = int(getattr(restored.optimizer, "step_count", 0))

        install_cpu, install_wall = time.process_time(), time.perf_counter()
        arm = str(spec["arm"])
        candidate_kind = "not_used" if arm == "keep" else arm.removeprefix("replace_").removeprefix("merge_")
        candidate = np.asarray(candidates["mean" if arm == "keep" else candidate_kind], dtype=np.float32)
        observation = np.asarray(restored_runtime["obs"], dtype=np.float32)
        fork_probabilities = storage._resident_probabilities(restored.core, observation, restored.slots)
        source_before = _capture_effective_at_slots(restored, candidate_slots)
        if semantic_hash(source_before) != semantic_hash(source_effective):
            raise RuntimeError("restored candidate source matrices differ from the common-prefix source arrays")
        pre_action_parent = ws._actor_arrays(restored.core)[1].reshape(3, 3).copy()
        pre_action_teacher_kl = _teacher_kl_metrics(
            restored.core,
            np.asarray(evidence_arrays["observations"]),
            np.asarray(evidence_arrays["owner_indices"]),
            source_effective,
            pre_action_parent,
            int(config["evidence_rows_per_source"]),
        )
        before_effective, post_local, post_shared, intervention, operation, candidate_kind = _install_arm_action(
            restored,
            candidate,
            arm,
        )
        candidate_hash = None if arm == "keep" else semantic_hash(candidate)
        candidate_norm = None if arm == "keep" else float(np.linalg.norm(candidate))
        prequant_effective = storage._effective_live(post_local, post_shared, ws._actor_arrays(restored.core)[2])
        current_effective = effective_matrices(*ws._actor_arrays(restored.core))
        prequant_effective_errors = np.max(
            np.abs(prequant_effective - before_effective), axis=(1, 2, 3),
        ).astype(float).tolist()
        prequant_probabilities = storage._resident_probabilities(restored.core, observation, restored.slots)
        prequant_behavior_drift = storage._probability_drift(fork_probabilities, prequant_probabilities)
        prequant_parent = ws._actor_arrays(restored.core)[1].reshape(3, 3).copy()
        prequant_parent_hash = semantic_hash(prequant_parent)
        prequant_teacher_kl = _teacher_kl_metrics(
            restored.core,
            np.asarray(evidence_arrays["observations"]),
            np.asarray(evidence_arrays["owner_indices"]),
            source_effective,
            prequant_parent,
            int(config["evidence_rows_per_source"]),
        )
        if operation == "keep":
            action_actor = prefix_actor_bytes
            action_codec_error = 0.0
            action_codec_effective_error = [0.0, 0.0]
            ws._actor_arrays(restored.core)
        else:
            action_actor, action_codec_error, action_codec_effective_error = storage._encode_live_actor(restored)
            ws._clear_optimizer_state(restored)
        postquant_probabilities = storage._resident_probabilities(restored.core, observation, restored.slots)
        postquant_behavior_drift = storage._probability_drift(fork_probabilities, postquant_probabilities)
        postquant_effective = effective_matrices(*ws._actor_arrays(restored.core))
        postquant_parent = ws._actor_arrays(restored.core)[1].reshape(3, 3).copy()
        postquant_teacher_kl = _teacher_kl_metrics(
            restored.core,
            np.asarray(evidence_arrays["observations"]),
            np.asarray(evidence_arrays["owner_indices"]),
            source_effective,
            postquant_parent,
            int(config["evidence_rows_per_source"]),
        )
        postquant_candidate_sources = postquant_effective[np.asarray(candidate_slots, dtype=np.int64)].copy()
        action_learner = storage._capture_learner(restored, foundation_sha)
        action_simulator = storage._capture_simulator(restored_env, restored_runtime)
        action_hash = semantic_hash({"learner": action_learner, "simulator": action_simulator})
        shared_reference = ws._actor_arrays(restored.core)[1].copy()
        install_cpu_s = float(time.process_time() - install_cpu)
        install_wall_s = float(time.perf_counter() - install_wall)
        action_install = {
            "arm": arm,
            "operation": operation,
            "candidate_kind": candidate_kind,
            "incumbent_binding": int(intervention["incumbent_binding"]),
            "candidate_group": 1,
            "parent_group_after_install": int(intervention["parent_group"]),
            "bound_slots": intervention["bound_slots"],
            "details": intervention,
            "candidate_sha256": candidate_hash,
            "candidate_norm": candidate_norm,
            "pre_action_parent": pre_action_parent.tolist(),
            "pre_action_parent_norm": float(np.linalg.norm(pre_action_parent)),
            "prequantized_parent": prequant_parent.tolist(),
            "prequantized_parent_sha256": prequant_parent_hash,
            "prequantized_parent_norm": float(np.linalg.norm(prequant_parent)),
            "postquantized_parent": postquant_parent.tolist(),
            "postquantized_parent_sha256": semantic_hash(postquant_parent),
            "postquantized_parent_norm": float(np.linalg.norm(postquant_parent)),
            "candidate_source_effective_precodec": current_effective[np.asarray(candidate_slots, dtype=np.int64)].tolist(),
            "candidate_source_effective_postcodec": postquant_candidate_sources.tolist(),
            "candidate_source_effective_precodec_sha256": semantic_hash(current_effective[np.asarray(candidate_slots, dtype=np.int64)]),
            "candidate_source_effective_postcodec_sha256": semantic_hash(postquant_candidate_sources),
            "candidate_source_effective_postcodec_norm": float(np.linalg.norm(postquant_candidate_sources)),
            "prequantized_effective_errors": prequant_effective_errors,
            "prequantized_behavior_drift": prequant_behavior_drift,
            "postquantized_behavior_drift": postquant_behavior_drift,
            "codec_max_coefficient_error": float(action_codec_error),
            "codec_effective_errors": [float(value) for value in action_codec_effective_error],
            "actor_payload_bytes": int(len(action_actor)),
            "keep_avoided_extra_install_and_codec": bool(operation == "keep"),
            "pre_action_new_source_teacher_kl": pre_action_teacher_kl,
            "new_source_teacher_kl_before_codec": prequant_teacher_kl,
            "new_source_teacher_kl_after_codec": postquant_teacher_kl,
            "new_source_teacher_KL_of_installed_parent": postquant_teacher_kl,
        }

        serialize_start_cpu, serialize_start_wall = time.process_time(), time.perf_counter()
        post_action_folder = branch_folder / "post_action"
        post_action_capacity = _attach_package_aliases(_save_candidate_package(
            post_action_folder,
            action_learner,
            action_actor,
            foundation,
            config,
            int(evidence_record["bytes"]),
        ))
        storage._write_json(post_action_folder / "simulator.json", action_simulator)
        _copy_evidence(prefix_folder / "evidence.npz", post_action_folder / "evidence.npz", evidence_record)
        post_action_artifacts = _candidate_package_artifacts(post_action_folder, output)
        serialization_cpu_s += float(time.process_time() - serialize_start_cpu)
        serialization_wall_s += float(time.perf_counter() - serialize_start_wall)

        future = mechanism.build_future_schedule(
            mapping,
            str(spec["future_regime"]),
            blocks=int(config["future_blocks"]),
            bout_steps=int(config["bout_steps"]),
            start_bout=len(plan["schedule"]),
        )
        simulation_cpu, simulation_wall = time.process_time(), time.perf_counter()
        branch_start_steps = int(restored.runtime_counts["environment_steps"])
        branch_start_rollouts = int(restored.runtime_counts["rollouts_trained"])
        branch_start_updates = int(restored.runtime_counts["minibatch_updates"])
        target_steps = set(int(value) for value in config["horizons"])

        def observe_rollout(**event: Any) -> None:
            del event
            current_steps = int(restored.runtime_counts["environment_steps"]) - branch_start_steps
            if not np.array_equal(ws._actor_arrays(restored.core)[1], shared_reference):
                raise RuntimeError("shared parent coefficients changed after branch installation")
            if current_steps in target_steps:
                cpu_milestones[current_steps] = {
                    "simulation_cpu_s": float(time.process_time() - simulation_cpu),
                    "simulation_wall_s": float(time.perf_counter() - simulation_wall),
                    "counts_delta": {
                        "rollouts_trained": int(restored.runtime_counts["rollouts_trained"]) - branch_start_rollouts,
                        "minibatch_updates": int(restored.runtime_counts["minibatch_updates"]) - branch_start_updates,
                    },
                }

        restored._storage_decision_post_rollout_observer = observe_rollout
        branch_bouts, branch_ppo_counts = mechanism._run_schedule_counted(
            restored,
            restored_env,
            restored_runtime,
            future,
            folder=branch_folder,
            phase="future",
            action=arm,
            future_regime=str(spec["future_regime"]),
        )
        del restored._storage_decision_post_rollout_observer
        simulation_cpu_s = float(time.process_time() - simulation_cpu)
        simulation_wall_s = float(time.perf_counter() - simulation_wall)
        if not np.array_equal(ws._actor_arrays(restored.core)[1], shared_reference):
            raise RuntimeError("shared parent coefficients changed over the branch continuation")
        final_probabilities_before_codec = storage._resident_probabilities(
            restored.core,
            np.asarray(restored_runtime["obs"], dtype=np.float32),
            restored.slots,
        )
        final_serialize_start_cpu, final_serialize_start_wall = time.process_time(), time.perf_counter()
        final_actor, final_codec_error, final_codec_effective_error = storage._encode_live_actor(restored)
        final_probabilities_after_codec = storage._resident_probabilities(
            restored.core,
            np.asarray(restored_runtime["obs"], dtype=np.float32),
            restored.slots,
        )
        final_codec_behavior_drift = storage._probability_drift(
            final_probabilities_before_codec, final_probabilities_after_codec,
        )
        final_learner = storage._capture_learner(restored, foundation_sha)
        final_simulator = storage._capture_simulator(restored_env, restored_runtime)
        final_hash = semantic_hash({"learner": final_learner, "simulator": final_simulator})
        capacity = _attach_package_aliases(_save_candidate_package(
            branch_folder,
            final_learner,
            final_actor,
            foundation,
            config,
            int(evidence_record["bytes"]),
        ))
        storage._write_json(branch_folder / "simulator.json", final_simulator)
        branch_artifacts = _candidate_package_artifacts(branch_folder, output)
        ledgers = mechanism._raw_ledgers(branch_folder, output)
        serialization_cpu_s += float(time.process_time() - final_serialize_start_cpu)
        serialization_wall_s += float(time.perf_counter() - final_serialize_start_wall)

        if semantic_hash(serialized_learner) != prefix_learner_hash:
            raise RuntimeError("branch mutated its freshly restored common-prefix learner payload")
        for name, before in prefix_files_before.items():
            target = prefix_folder / name
            if not target.is_file() or _sha(target) != before["sha256"] or target.stat().st_size != int(before["bytes"]):
                raise RuntimeError(f"branch mutated serialized common-prefix artifact {name}")
        horizons = mechanism._branch_horizons(branch_bouts, cpu_milestones, config["horizons"], install_cpu_s)
        runtime_names = (
            "environment_steps", "completed_episodes", "slot_allocations", "slot_evictions",
            "rollouts_trained", "minibatch_updates",
        )
        learner_names = (
            "raw_steps", "environment_steps", "fit_steps", "rollouts_trained", "minibatch_updates",
            "completed_episodes", "slot_allocations", "slot_evictions",
        )
        runtime_delta = mechanism._counter_delta(restored.runtime_counts, prefix_runtime_start or {}, runtime_names)
        learner_delta = mechanism._counter_delta(restored.counts, prefix_learner_start or {}, learner_names)
        optimizer_step_delta = int(getattr(restored.optimizer, "step_count", 0)) - int(optimizer_count_start or 0)
        if optimizer_step_delta != learner_delta["minibatch_updates"]:
            raise RuntimeError("optimizer step counter disagrees with learner minibatch update counts")
        branch_cpu_s = float(time.process_time() - branch_start_cpu)
        branch_wall_s = float(time.perf_counter() - branch_start_wall)
        branch_summary = {
            "format": "retention_candidate_branch_v1",
            "branch_id": branch_id,
            "case_id": case_id,
            "seed": int(seed),
            "pair_kind": pair_kind,
            "state": state,
            "history_regime": history_regime,
            "future_regime": str(spec["future_regime"]),
            "replicate": int(spec["replicate"]),
            "arm": arm,
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
            "action_install": action_install,
            "post_action_state_hash": action_hash,
            "post_action_artifacts": post_action_artifacts,
            "capacity_after_action": post_action_capacity,
            "final_state_hash": final_hash,
            "final_artifacts": branch_artifacts,
            "final_codec": {
                "max_coefficient_error": float(final_codec_error),
                "effective_errors": [float(value) for value in final_codec_effective_error],
                "behavior_drift": final_codec_behavior_drift,
            },
            "shared_parent_invariant": {
                "max_coefficient_drift": float(np.max(np.abs(ws._actor_arrays(restored.core)[1] - shared_reference), initial=0.0)),
                "exactly_fixed": bool(np.array_equal(ws._actor_arrays(restored.core)[1], shared_reference)),
            },
            "candidate_fit_reference": {
                "mean_candidate_sha256": str(candidate_fit["mean_candidate_sha256"]),
                "behavioral_candidate_sha256": str(candidate_fit["behavioral_candidate_sha256"]),
                "fit_calls": int(candidate_fit["fit_calls"]),
            },
            "evidence_buffer_bytes": int(evidence_record["bytes"]),
            "per_bout": branch_bouts,
            "horizons": horizons,
            "raw_environment_steps": int(ledgers["steps_csv"]["rows"]),
            "ppo_call_counts": branch_ppo_counts,
            "counts_delta": {
                **learner_delta,
                "runtime_counts": runtime_delta,
                "optimizer_step_count": optimizer_step_delta,
                "partial_rollout_rows": int(len(restored.rollout)),
                "rollouts_csv_completed": int(ledgers["rollouts_csv"].get("rollout_count", 0)),
                "minibatch_updates_from_rollout_ledger": int(ledgers["rollouts_csv"].get("minibatch_updates", 0)),
            },
            "raw_ledgers": ledgers,
            "capacity": capacity,
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
                "evidence_buffer_bytes": int(evidence_record["bytes"]),
                "package_bytes": int(capacity["package_bytes"]),
                "actor_container_bytes": int(capacity["actor_container_bytes"]),
                "package_budget_bytes": int(config["package_budget_bytes"]),
            },
            "prefix_bouts_included_in_reward": False,
            "errors": [],
        }
    except Exception:
        branch_error = {
            "case_id": case_id,
            "branch_id": branch_id,
            "error_type": "branch_failure",
            "error": traceback.format_exc(),
        }
        ledgers = mechanism._raw_ledgers(branch_folder, output)
        branch_cpu_s = float(time.process_time() - branch_start_cpu)
        branch_wall_s = float(time.perf_counter() - branch_start_wall)
        branch_summary = {
            "format": "retention_candidate_branch_v1",
            "branch_id": branch_id,
            "case_id": case_id,
            "seed": int(seed),
            "pair_kind": pair_kind,
            "state": state,
            "history_regime": history_regime,
            "future_regime": str(spec["future_regime"]),
            "replicate": int(spec["replicate"]),
            "arm": str(spec["arm"]),
            "future_seed": int(future_value),
            "prefix_state_hash": prefix_hash,
            "restored_state_hash": restored_hash,
            "prefix_restore_verified": bool(restored is not None and restored_env is not None and restored_hash == prefix_hash),
            "post_rng_transform_state_hash": post_transform_hash,
            "rng_transform": {
                "reseed": reseed_metadata,
                "before": rng_before,
                "after": rng_after,
            },
            "action_install": action_install,
            "post_action_state_hash": action_hash,
            "post_action_artifacts": post_action_artifacts,
            "capacity_after_action": capacity_after_action,
            "future_schedule": [],
            "per_bout": branch_bouts,
            "horizons": horizons,
            "raw_environment_steps": int(ledgers["steps_csv"]["rows"]),
            "raw_ledgers": ledgers,
            "ppo_call_counts": dict(getattr(restored, "_last_retention_ppo_call_counts", {})) if restored is not None else {},
            "counts_delta": None,
            "evidence_buffer_bytes": int(evidence_record["bytes"]),
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
                "evidence_buffer_bytes": int(evidence_record["bytes"]),
            },
            "errors": [branch_error],
        }
        if restored is not None and prefix_runtime_start is not None and prefix_learner_start is not None:
            runtime_names = (
                "environment_steps", "completed_episodes", "slot_allocations", "slot_evictions",
                "rollouts_trained", "minibatch_updates",
            )
            learner_names = (
                "raw_steps", "environment_steps", "fit_steps", "rollouts_trained", "minibatch_updates",
                "completed_episodes", "slot_allocations", "slot_evictions",
            )
            learner_delta = mechanism._counter_delta(restored.counts, prefix_learner_start, learner_names)
            runtime_delta = mechanism._counter_delta(restored.runtime_counts, prefix_runtime_start, runtime_names)
            branch_summary["counts_delta"] = {
                **learner_delta,
                "runtime_counts": runtime_delta,
                "optimizer_step_count": int(getattr(restored.optimizer, "step_count", 0)) - int(optimizer_count_start or 0),
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
    return branch_summary


def _sum_count_maps(*maps: Mapping[str, Any]) -> dict[str, int]:
    names = set().union(*(set(value) for value in maps))
    return {name: sum(int(value.get(name, 0)) for value in maps) for name in names}


def _read_persisted_case_steps(case_folder: Path) -> dict[str, Any]:
    prefix_rows = mechanism._csv_record_stats(case_folder / "prefix" / "steps.csv", reward_column="reward")
    branch_rows: list[dict[str, Any]] = []
    root = case_folder / "branches"
    if root.is_dir():
        for branch in sorted(root.iterdir()):
            if branch.is_dir():
                rows = mechanism._csv_record_stats(branch / "steps.csv", reward_column="reward")
                branch_rows.append({"branch_id": branch.name, "rows": int(rows["rows"])})
    branch_steps = sum(row["rows"] for row in branch_rows)
    return {
        "prefix_environment_steps": int(prefix_rows["rows"]),
        "branch_environment_steps": int(branch_steps),
        "environment_steps_total": int(prefix_rows["rows"] + branch_steps),
        "branches": branch_rows,
    }


def _case_payloads(config: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(seed),
            "pair_kind": str(pair_kind),
            "state": str(state),
            "config": dict(config),
            "output": str(output),
        }
        for seed in config["seeds"]
        for pair_kind in config["pair_kinds"]
        for state in config["states"]
    ]


def _empty_case_error(job: Mapping[str, Any], output: Path, error_text: str) -> dict[str, Any]:
    seed = int(job["seed"])
    pair_kind = str(job["pair_kind"])
    state = str(job["state"])
    history_regime = history_regime_for_seed(seed)
    case_id = f"s{seed}_{pair_kind}_{state}"
    folder = output / "cases" / case_id
    folder.mkdir(parents=True, exist_ok=True)
    persisted = _read_persisted_case_steps(folder)
    error = {
        "case_id": case_id,
        "error_type": "case_worker_failure",
        "error": error_text,
        "persisted_interactions": persisted,
        "count_source": "prefix/branches steps.csv rows",
    }
    _write_json(folder / "uncaught_error.json", error)
    return {
        "case_id": case_id,
        "seed": seed,
        "pair_kind": pair_kind,
        "state": state,
        "history_regime": history_regime,
        "branches": [],
        "costs": {
            **persisted,
            "prefix_cpu_s": None,
            "candidate_fit_cpu_s": None,
            "branch_cpu_s": None,
            "case_cpu_s": None,
            "prefix_wall_s": None,
            "branch_wall_s": None,
            "case_wall_s": None,
        },
        "errors": [error],
    }


def _validate_frozen_expected(config: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    configured = config.get("expected_pilot" if config.get("phase") == "pilot" else "expected_main", {})
    if not isinstance(configured, Mapping):
        raise ValueError("expected interaction counts must be an object")
    required = {
        "cases", "branches", "early_prefix_steps", "late_prefix_steps",
        "prefix_steps", "steps_per_branch", "branch_steps", "total_steps",
    }
    if set(configured) != required:
        raise ValueError(f"configured expected interaction fields must equal {sorted(required)}")
    normalized = {name: int(value) for name, value in configured.items()}
    if normalized != dict(expected):
        raise ValueError(f"configured expected interactions differ from derived counts: {normalized} != {dict(expected)}")
    return normalized


def run(
    config: Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    seeds: Sequence[int] | None = None,
    pilot: bool = False,
) -> dict[str, Any]:
    config_path = Path(config).resolve() if isinstance(config, (str, Path)) else CONFIG_PATH
    raw_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config, (str, Path))
        else dict(config)
    )
    normalized = _phase_config(raw_config, pilot=pilot)
    if seeds is not None:
        selected_seeds = [int(value) for value in seeds]
        if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
            raise ValueError("an explicit seed override must be nonempty and unique")
        normalized["seeds"] = selected_seeds
        normalized = validate_config(normalized)
        normalized["phase"] = "pilot" if pilot else "main"
    configured_seed_pool = raw_config.get("pilot", {}).get("seeds", []) if pilot else raw_config["seeds"]
    if any(int(seed) not in set(int(value) for value in configured_seed_pool) for seed in normalized["seeds"]):
        raise ValueError("selected seeds must come from the frozen unused seed list")

    output_path = Path(output).resolve()
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"Preserve existing candidate output: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    expected = _expected_interactions(normalized)
    full_config_selection = (
        normalized["seeds"] == [int(value) for value in (raw_config.get("pilot", {}).get("seeds", []) if pilot else raw_config["seeds"])]
    )
    configured_expected = _validate_frozen_expected(normalized, expected) if full_config_selection else None

    source_manifest = _snapshot_sources(output_path, config_path)
    _write_json(output_path / "config.json", normalized)
    _write_json(output_path / "source_snapshot" / "config_original.json", raw_config)
    _write_json(output_path / "frozen_run.json", {
        "phase": normalized["phase"],
        "config": normalized,
        "config_sha256": _sha(config_path) if config_path.is_file() else None,
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "source_manifest": source_manifest,
        "expected_interactions": expected,
        "configured_expected_interactions": configured_expected,
    })

    jobs = _case_payloads(normalized, output_path)
    requested_workers = int(normalized["max_workers"])
    workers = max(1, min(2, requested_workers, len(jobs)))
    executor_mode = "inline" if workers == 1 else "process_pool"
    campaign_start_cpu, campaign_start_wall = time.process_time(), time.perf_counter()
    records: list[dict[str, Any]] = []
    if workers == 1:
        for job in jobs:
            try:
                records.append(_case_worker(job))
            except Exception:
                records.append(_empty_case_error(job, output_path, traceback.format_exc()))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_case_worker, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    records.append(future.result())
                except Exception:
                    records.append(_empty_case_error(job, output_path, traceback.format_exc()))

    records.sort(key=lambda row: (int(row["seed"]), str(row["pair_kind"]), STATES.index(str(row["state"]))))
    flat_branches = [branch for case in records for branch in case.get("branches", [])]
    errors = [error for case in records for error in case.get("errors", [])]
    prefix_steps = sum(int(case.get("costs", {}).get("prefix_environment_steps", 0) or 0) for case in records)
    branch_steps = sum(int(case.get("costs", {}).get("branch_environment_steps", 0) or 0) for case in records)
    actual_steps = prefix_steps + branch_steps
    count_errors: list[dict[str, Any]] = []
    if len(records) != expected["cases"]:
        count_errors.append({"error_type": "case_count_mismatch", "expected": expected["cases"], "actual": len(records)})
    if len(flat_branches) != expected["branches"]:
        count_errors.append({"error_type": "branch_count_mismatch", "expected": expected["branches"], "actual": len(flat_branches)})
    if actual_steps != expected["total_steps"]:
        count_errors.append({"error_type": "interaction_count_mismatch", "expected": expected["total_steps"], "actual": actual_steps})
    errors.extend(count_errors)

    case_cpu_values = [case.get("costs", {}).get("case_cpu_s") for case in records]
    parent_cpu = float(time.process_time() - campaign_start_cpu)
    campaign_cpu = mechanism._campaign_cpu_accounting(executor_mode, parent_cpu, case_cpu_values)
    costs = {
        "prefix_environment_steps": int(prefix_steps),
        "branch_environment_steps": int(branch_steps),
        "environment_steps_total": int(actual_steps),
        "expected_environment_steps_total": int(expected["total_steps"]),
        "expected_interactions": expected,
        "configured_expected_interactions": configured_expected,
        "executor_mode": executor_mode,
        "requested_workers": requested_workers,
        "actual_workers": workers,
        **campaign_cpu,
        "elapsed_wall_s": float(time.perf_counter() - campaign_start_wall),
        "prefix_cpu_s": float(sum(float(case.get("costs", {}).get("prefix_cpu_s", 0.0) or 0.0) for case in records)),
        "candidate_fit_cpu_s": float(sum(float(case.get("costs", {}).get("candidate_fit_cpu_s", 0.0) or 0.0) for case in records)),
        "branch_cpu_s": float(sum(float(case.get("costs", {}).get("branch_cpu_s", 0.0) or 0.0) for case in records)),
        "prefix_wall_s": float(sum(float(case.get("costs", {}).get("prefix_wall_s", 0.0) or 0.0) for case in records)),
        "branch_wall_s": float(sum(float(case.get("costs", {}).get("branch_wall_s", 0.0) or 0.0) for case in records)),
        "candidate_fit_wall_s": float(sum(float(case.get("costs", {}).get("candidate_fit_wall_s", 0.0) or 0.0) for case in records)),
        "serialized_evidence_bytes_prefix_once": int(sum(int(case.get("costs", {}).get("evidence_buffer_bytes", 0) or 0) for case in records)),
        "serialized_evidence_bytes_all_branch_packages": int(sum(int(branch.get("evidence_buffer_bytes", 0) or 0) for branch in flat_branches)),
    }
    summary = {
        "format": SUMMARY_FORMAT,
        "phase": normalized["phase"],
        "config": normalized,
        "config_sha256": _sha(config_path) if config_path.is_file() else None,
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "source_manifest": source_manifest,
        "seeds": list(normalized["seeds"]),
        "case_count": len(records),
        "branch_count": len(flat_branches),
        "cases": records,
        "branches": flat_branches,
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


def _case_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = validate_config(payload["config"])
    seed = int(payload["seed"])
    pair_kind = str(payload["pair_kind"])
    state = str(payload["state"])
    history_regime = history_regime_for_seed(seed)
    output = Path(payload["output"])
    case_id = f"s{seed}_{pair_kind}_{state}"
    case_folder = output / "cases" / case_id
    case_folder.mkdir(parents=True, exist_ok=False)
    case_start_cpu, case_start_wall = time.process_time(), time.perf_counter()
    foundation = None
    foundation_sha = None
    env = None
    prefix_summary: dict[str, Any] | None = None
    branch_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    prefix_cpu_s = prefix_wall_s = 0.0
    branch_cpu_total = branch_wall_total = 0.0
    candidate_fit: dict[str, Any] | None = None
    evidence_record: dict[str, Any] | None = None
    try:
        base_config = loop.load_base_config()
        foundation = Path(base_config["foundation"])
        foundation_sha = _sha(foundation)
        mapping = pair_mapping(seed, pair_kind)
        plan = build_prefix_plan(mapping, config, state, history_regime)
        env, runtime = storage._runtime_initial(seed, str(plan["schedule"][0]["task"]))
        agent = storage._new_agent(seed, base_config, config)
        prefix_folder = case_folder / "prefix"
        prefix_folder.mkdir(parents=True, exist_ok=False)
        prefix_start_cpu, prefix_start_wall = time.process_time(), time.perf_counter()

        bootstrap_bouts, bootstrap_counts = mechanism._run_schedule_counted(
            agent,
            env,
            runtime,
            plan["bootstrap"],
            folder=prefix_folder,
            phase="bootstrap",
        )
        bootstrap_source_slots = [
            mechanism._slot_for_task(agent, str(item["task"]))
            for item in plan["bootstrap"]
        ]
        if len(set(bootstrap_source_slots)) != 2:
            raise RuntimeError("bootstrap old variants must occupy two distinct resident slots")
        for slot in bootstrap_source_slots:
            if int(agent.fitted_steps[slot]) != int(config["bootstrap_steps"]):
                raise RuntimeError("bootstrap source did not receive its expected fitted steps")
        incumbent_formation = _install_pair_mean(
            agent,
            mapping,
            group=0,
            label="bootstrap_old_mean",
        )
        if any(int(row["fitted_steps"]) != int(config["bootstrap_steps"]) for row in incumbent_formation["source_records"]):
            raise RuntimeError("bootstrap incumbent source fit is not 1024 rows per owner")
        prefix_bouts = list(bootstrap_bouts)
        ppo_count_maps: list[Mapping[str, Any]] = [bootstrap_counts]
        warmup_pair_records: list[dict[str, Any]] = []
        for pair_plan in plan["warmup_pairs"]:
            _, parent_before_shared, _ = ws._actor_arrays(agent.core)
            parent_binding_before = int(agent.parent_group)
            parent_before = parent_before_shared.reshape(3, 3).copy()
            parent_before_hash = semantic_hash(parent_before)
            pair_bouts, pair_counts = mechanism._run_schedule_counted(
                agent,
                env,
                runtime,
                pair_plan["schedule"],
                folder=prefix_folder,
                phase=str(pair_plan["phase"]),
            )
            prefix_bouts.extend(pair_bouts)
            ppo_count_maps.append(pair_counts)
            group = int(pair_plan["group"])
            write_applied = bool(pair_plan["write_applied_after_pair"])
            if write_applied != (group == 0):
                raise RuntimeError("warmup write rule disagrees with the configured old-only write condition")
            if group == 0:
                install_record = _install_pair_mean(
                    agent,
                    mapping,
                    group=group,
                    label=f"cycle_{int(pair_plan['cycle'])}_{pair_plan['phase']}",
                )
                source_records = install_record["source_records"]
                source_effective_matrices = np.asarray(install_record["source_effective_matrices"], dtype=np.float32)
                formation = install_record["formation"]
                codec_max_error = install_record["codec_max_coefficient_error"]
                codec_effective_errors = install_record["codec_effective_errors"]
            else:
                install_record = None
                source_snapshot = _snapshot_pair_sources(agent, mapping, group=group)
                source_records = source_snapshot["source_records"]
                source_effective_matrices = np.asarray(source_snapshot["source_effective_matrices"], dtype=np.float32)
                formation = None
                codec_max_error = 0.0
                codec_effective_errors = [0.0, 0.0]
            if any(int(row["fitted_steps"]) != int(config["warmup_pair_steps"]) for row in source_records):
                raise RuntimeError("late warmup source fit differs from its frozen pair steps")
            _, parent_after_shared, _ = ws._actor_arrays(agent.core)
            parent_after = parent_after_shared.reshape(3, 3).copy()
            parent_after_hash = semantic_hash(parent_after)
            if group == 1 and (
                int(agent.parent_group) != parent_binding_before
                or parent_after_hash != parent_before_hash
            ):
                # Local matrices may change, but the old-only parent binding and
                # coefficients must survive every new-only warmup pair exactly.
                raise RuntimeError("new-only warmup pair changed the retained old parent")
            warmup_pair_records.append({
                "cycle": int(pair_plan["cycle"]),
                "group": group,
                "phase": str(pair_plan["phase"]),
                "write_applied": write_applied,
                "write_rule": str(config["warmup_write_rule"]),
                "parent_binding_before_after": [parent_binding_before, int(agent.parent_group)],
                "parent_sha256_before_after": [parent_before_hash, parent_after_hash],
                "parent_norm_before_after": [float(np.linalg.norm(parent_before)), float(np.linalg.norm(parent_after))],
                "parent_coefficients_unchanged": bool(np.array_equal(parent_before, parent_after)),
                "pair_bouts": pair_bouts,
                "source_records": source_records,
                "source_effective_matrices": source_effective_matrices.tolist(),
                "source_effective_sha256": semantic_hash(source_effective_matrices),
                "parent_sha256": install_record["mean_parent_sha256"] if install_record is not None else parent_after_hash,
                "parent_norm": install_record["mean_parent_norm"] if install_record is not None else float(np.linalg.norm(parent_after)),
                "formation": formation,
                "codec_max_coefficient_error": float(codec_max_error),
                "codec_effective_errors": [float(value) for value in codec_effective_errors],
            })
            if group == 0:
                incumbent_formation = install_record

        target_steps = int(config["early_source_steps"] if state == "early" else config["late_source_steps"])
        history_schedule = plan["history"]
        target_bouts = {
            int(source["bout"]): {"owner_index": owner, "task": str(source["task"])}
            for owner, source in enumerate(plan["candidate_sources"])
        }
        history_bouts, history_counts, evidence_arrays = _capture_evidence_from_post_rollout(
            agent,
            targets=target_bouts,
            rows_per_source=int(config["evidence_rows_per_source"]),
            env=env,
            runtime=runtime,
            schedule=history_schedule,
            folder=prefix_folder,
            phase="prefix_history",
        )
        prefix_bouts.extend(history_bouts)
        ppo_count_maps.append(history_counts)
        if len(agent.recent_groups) != 8 or agent.recent_groups[-2:] != [1, 1]:
            raise RuntimeError("prefix must end with two new-group history bouts")
        evidence_path = prefix_folder / "evidence.npz"
        evidence_record = _write_evidence_npz(evidence_path, evidence_arrays)

        prefix_actor, prefix_codec_error, prefix_codec_effective_error = storage._encode_live_actor(agent)
        ws._clear_optimizer_state(agent)
        candidate_sources, candidate_slots, candidate_effective = _current_candidate_fit_records(
            agent,
            mapping,
            plan["candidate_sources"],
            prefix_bouts,
            target_steps,
            evidence_arrays,
        )
        incumbent_sources, incumbent_effective = _make_source_fits(
            incumbent_formation,
            mapping,
            plan["bootstrap"],
            plan["warmup_pairs"],
            prefix_bouts,
        )
        if len(incumbent_sources) != 2 or any(
            int(row["source_time_fitted_steps"]) != int(config["bootstrap_steps"])
            if state == "early"
            else int(row["source_time_fitted_steps"]) != int(config["warmup_pair_steps"])
            for row in incumbent_sources
        ):
            raise RuntimeError("incumbent source-time fits do not match the selected formation pair")
        candidate_values, candidate_fit = _fit_candidates_once(
            agent,
            env,
            runtime,
            foundation_sha,
            evidence_arrays,
            candidate_effective,
            config,
        )
        candidate_fit["source_buffer_sha256"] = evidence_record["sha256"]
        candidate_fit["source_buffer_bytes"] = int(evidence_record["bytes"])
        candidate_fit["rows_per_source"] = int(config["evidence_rows_per_source"])
        candidate_fit["latest_source_global_steps"] = [
            int(np.max(np.asarray(evidence_arrays["global_steps"])[np.asarray(evidence_arrays["owner_indices"]) == owner]))
            for owner in (0, 1)
        ]
        candidate_fit["source_fit_records"] = candidate_sources
        candidate_fit["incumbent_source_fit_records"] = incumbent_sources

        prefix_local, prefix_shared, prefix_physical_groups = ws._actor_arrays(agent.core)
        prefix_effective = effective_matrices(prefix_local, prefix_shared, prefix_physical_groups)
        current_parent = prefix_shared.reshape(3, 3).copy()
        parent_group = int(agent.parent_group)
        prefix_learner = storage._capture_learner(agent, foundation_sha)
        prefix_simulator = storage._capture_simulator(env, runtime)
        prefix_hash = semantic_hash({"learner": prefix_learner, "simulator": prefix_simulator})
        prefix_learner_hash = semantic_hash(prefix_learner)
        prefix_capacity = _attach_package_aliases(_save_candidate_package(
            prefix_folder,
            prefix_learner,
            prefix_actor,
            foundation,
            config,
            int(evidence_record["bytes"]),
        ))
        storage._write_json(prefix_folder / "simulator.json", prefix_simulator)
        observer_prefix = _candidate_metadata_for_observer(
            incumbent_sources,
            candidate_sources,
            incumbent_effective,
            candidate_effective,
            candidate_fit,
            evidence_record,
            evidence_arrays,
            current_parent,
            parent_group,
            history_bouts,
        )
        observer_path = prefix_folder / "observer_prefix.json"
        _write_json(observer_path, observer_prefix)
        serialized_learner = storage._capture_learner_from_saved(prefix_folder / "checkpoint.pt")
        serialized_simulator = storage._load_json(prefix_folder / "simulator.json")
        mechanism._verify_restored_hash(
            prefix_hash,
            serialized_learner,
            serialized_simulator,
        )
        if semantic_hash(serialized_learner) != prefix_learner_hash:
            raise RuntimeError("serialized prefix checkpoint differs from common-prefix learner")
        prefix_cpu_s = float(time.process_time() - prefix_start_cpu)
        prefix_wall_s = float(time.perf_counter() - prefix_start_wall)
        prefix_summary = storage._summarize_bouts(prefix_bouts, agent, {"cpu_s": prefix_cpu_s, "wall_s": prefix_wall_s})
        prefix_summary.update({
            "steps": int(sum(int(row["steps"]) for row in prefix_bouts)),
            "seed": seed,
            "pair_kind": pair_kind,
            "state": state,
            "history_regime": history_regime,
            "mapping": mapping,
            "prefix_state_hash": prefix_hash,
            "prefix_learner_hash": prefix_learner_hash,
            "foundation_sha256": foundation_sha,
            "incumbent_sources": incumbent_sources,
            "candidate_sources": candidate_sources,
            "candidate_source_slots": [int(slot) for slot in candidate_slots],
            "incumbent_source_effective_sha256": semantic_hash(incumbent_effective),
            "candidate_source_effective_sha256": semantic_hash(candidate_effective),
            "candidate_source_effective_norm": float(np.linalg.norm(candidate_effective)),
            "current_mean_candidate_norm": float(np.linalg.norm(candidate_values["mean"])),
            "current_behavioral_candidate_norm": float(np.linalg.norm(candidate_values["behavioral"])),
            "current_parent": current_parent.tolist(),
            "current_parent_sha256": semantic_hash(current_parent),
            "current_parent_norm": float(np.linalg.norm(current_parent)),
            "parent_group": parent_group,
            "shared_frozen": bool(agent.core._shared_frozen),
            "inherited_states": [
                {
                    "bout": int(row["bout"]),
                    "task": str(row["task"]),
                    "group": int(row["group"]),
                    "slot": int(row["slot"]),
                    "generation": int(row["generation"]),
                    "allocated": bool(row["allocated"]),
                    "prior_inherited": bool(row["prior_inherited"]),
                    "physical_group_at_start": int(row["physical_group_at_start"]),
                    "parent_group_at_start": int(row["parent_group_at_start"]),
                }
                for row in history_bouts
            ],
            "warmup_pair_records": warmup_pair_records,
            "prefix_codec_max_coefficient_error": float(prefix_codec_error),
            "prefix_codec_effective_errors": [float(value) for value in prefix_codec_effective_error],
            "candidate_fit": candidate_fit,
            "evidence_buffer": {
                **evidence_record,
                "rows": int(len(evidence_arrays["owner_indices"])),
                "rows_by_owner": np.bincount(evidence_arrays["owner_indices"], minlength=2).astype(int).tolist(),
                "latest_source_global_steps": candidate_fit["latest_source_global_steps"],
            },
            "ppo_call_counts": _sum_count_maps(*ppo_count_maps),
            "capacity": prefix_capacity,
            "actor_payload_bytes": int(len(prefix_actor)),
            "artifacts": _candidate_package_artifacts(prefix_folder, output),
            "raw_ledgers": mechanism._raw_ledgers(prefix_folder, output),
            "observer_artifact": _artifact_record(observer_path, output),
        })
        _write_json(prefix_folder / "summary.json", prefix_summary)

        branch_order, execution_order_seed = _branch_specs(config, case_id)
        _write_json(case_folder / "execution_order.json", {
            "seed": int(execution_order_seed),
            "order": [str(row["branch_id"]) for row in branch_order],
        })
        prefix_files_before = {
            name: {"sha256": _sha(prefix_folder / name), "bytes": int((prefix_folder / name).stat().st_size)}
            for name in ("checkpoint.pt", "actor.bin", "simulator.json", "evidence.npz")
        }
        prefix_runtime_counts = dict(agent.runtime_counts)
        prefix_learner_counts = dict(agent.counts)
        optimizer_start_count = int(getattr(agent.optimizer, "step_count", 0))
        for spec in branch_order:
            result = _run_branch(
                config=config,
                case_folder=case_folder,
                output=output,
                case_id=case_id,
                seed=seed,
                pair_kind=pair_kind,
                state=state,
                history_regime=history_regime,
                mapping=mapping,
                plan=plan,
                spec=spec,
                base_config=base_config,
                foundation=foundation,
                foundation_sha=foundation_sha,
                prefix_folder=prefix_folder,
                prefix_hash=prefix_hash,
                prefix_learner_hash=prefix_learner_hash,
                prefix_actor=prefix_actor,
                prefix_runtime_counts=prefix_runtime_counts,
                prefix_learner_counts=prefix_learner_counts,
                optimizer_start_count=optimizer_start_count,
                prefix_files_before=prefix_files_before,
                evidence_record=evidence_record,
                evidence_arrays=evidence_arrays,
                source_effective=candidate_effective,
                candidates=candidate_values,
                candidate_fit=candidate_fit,
                candidate_slots=candidate_slots,
            )
            branch_results.append(result)
        env.close()
        env = None
        errors.extend(error for branch in branch_results for error in branch.get("errors", []))
        if _sha(foundation) != foundation_sha:
            errors.append({"case_id": case_id, "error_type": "foundation_changed", "error": "foundation SHA changed during case"})
        branch_steps = sum(int(row.get("raw_environment_steps", 0)) for row in branch_results)
        branch_cpu_total = sum(float(row.get("costs", {}).get("branch_cpu_s", 0.0)) for row in branch_results)
        branch_wall_total = sum(float(row.get("costs", {}).get("branch_wall_s", 0.0)) for row in branch_results)
        result = {
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "state": state,
            "history_regime": history_regime,
            "mapping": mapping,
            "prefix": prefix_summary,
            "candidate_fit": candidate_fit,
            "observer_prefix_path": (prefix_folder / "observer_prefix.json").relative_to(output).as_posix(),
            "execution_order_seed": int(execution_order_seed),
            "execution_order": [str(row["branch_id"]) for row in branch_order],
            "branches": branch_results,
            "foundation_unchanged": bool(_sha(foundation) == foundation_sha),
            "costs": {
                "prefix_environment_steps": int(prefix_summary["steps"]),
                "branch_environment_steps": int(branch_steps),
                "environment_steps_total": int(prefix_summary["steps"] + branch_steps),
                "prefix_cpu_s": float(prefix_cpu_s),
                "candidate_fit_cpu_s": float(candidate_fit["fit_cpu_s"]),
                "candidate_fit_wall_s": float(candidate_fit["fit_wall_s"]),
                "branch_cpu_s": float(branch_cpu_total),
                "case_cpu_s": float(time.process_time() - case_start_cpu),
                "prefix_wall_s": float(prefix_wall_s),
                "branch_wall_s": float(branch_wall_total),
                "case_wall_s": float(time.perf_counter() - case_start_wall),
                "package_bytes": int(prefix_capacity["package_bytes"]),
                "actor_container_bytes": int(prefix_capacity["actor_container_bytes"]),
                "package_budget_bytes": int(config["package_budget_bytes"]),
                "evidence_buffer_bytes": int(evidence_record["bytes"]),
            },
            "errors": errors,
        }
        _write_json(case_folder / "summary.json", result)
        return result
    except Exception:
        if env is not None:
            env.close()
        persisted = _read_persisted_case_steps(case_folder)
        error = {
            "case_id": case_id,
            "error_type": "case_failure",
            "error": traceback.format_exc(),
            "persisted_interactions": persisted,
            "count_source": "prefix/branches steps.csv rows",
        }
        errors.append(error)
        result = {
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "state": state,
            "history_regime": history_regime,
            "mapping": pair_mapping(seed, pair_kind),
            "prefix": prefix_summary,
            "candidate_fit": candidate_fit,
            "branches": branch_results,
            "costs": {
                **persisted,
                "prefix_cpu_s": float(prefix_cpu_s),
                "branch_cpu_s": float(branch_cpu_total),
                "case_cpu_s": float(time.process_time() - case_start_cpu),
                "prefix_wall_s": float(prefix_wall_s),
                "branch_wall_s": float(branch_wall_total),
                "candidate_fit_cpu_s": float(candidate_fit["fit_cpu_s"]) if candidate_fit is not None else 0.0,
                "candidate_fit_wall_s": float(candidate_fit["fit_wall_s"]) if candidate_fit is not None else 0.0,
                "evidence_buffer_bytes": int(evidence_record["bytes"]) if evidence_record is not None else 0,
                "case_wall_s": float(time.perf_counter() - case_start_wall),
            },
            "errors": errors,
        }
        _write_json(case_folder / "error.json", error)
        _write_json(case_folder / "summary.json", result)
        return result


if __name__ == "__main__":
    raise SystemExit(main())



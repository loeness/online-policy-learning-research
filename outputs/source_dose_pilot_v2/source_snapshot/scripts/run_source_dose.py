"""Compare 1024-step and 2048-step old-source candidates at one common fork.

The original storage-decision worker creates the shared prefix and its normal
2048-source branches. This runner observes an early old candidate during that
same prefix, then restores the saved common checkpoint for a separate set of
1024-source branches. The observer candidate is never part of learner state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_storage_decision as base


ROOT = base.ROOT
CONFIG_PATH = ROOT / "configs" / "source_dose.json"
PROTOCOL_PATH = ROOT / "docs" / "SOURCE_DOSE_PROTOCOL.md"
FORMAT = "storage_source_dose_summary_v1"
SOURCE_FILES = tuple(dict.fromkeys((
    *base.SOURCE_FILES,
    Path("scripts/run_source_dose.py"),
    Path("docs/SOURCE_DOSE_PROTOCOL.md"),
    Path("configs/source_dose.json"),
)))


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the base study settings plus the observer capture point."""
    result = base.validate_config(config)
    capture_steps = int(result.get("source_capture_steps", 0))
    rollout_steps = int(result["rollout_steps"])
    if capture_steps < 1 or capture_steps % rollout_steps:
        raise ValueError("source_capture_steps must be a positive multiple of rollout_steps")
    if capture_steps > int(result["bootstrap_steps"]):
        raise ValueError("source_capture_steps cannot exceed bootstrap_steps")
    result["source_capture_steps"] = capture_steps
    return result


def _phase_config(raw_config: Mapping[str, Any], *, pilot: bool) -> dict[str, Any]:
    phase_config = copy.deepcopy(dict(raw_config))
    if pilot:
        pilot_config = dict(phase_config["pilot"])
        for key in ("seeds", "pair_kinds", "history_regimes", "bootstrap_steps", "bout_steps"):
            phase_config[key] = pilot_config[key]
        capture_steps = phase_config.get("pilot_source_capture_steps", pilot_config.get("source_capture_steps"))
        if capture_steps is None:
            raise ValueError("pilot_source_capture_steps is required for a pilot run")
        phase_config["source_capture_steps"] = int(capture_steps)
    return validate_config(phase_config)


def _source_task_names(seed: int, pair_kind: str) -> dict[str, int]:
    mapping = base.pair_mapping(int(seed), str(pair_kind))
    return {base.task_name(mapping, 0, variant): variant for variant in (0, 1)}


def _make_source_observer(captures: dict[int, dict[str, Any]], *, seed: int, pair_kind: str, capture_steps: int):
    task_variants = _source_task_names(seed, pair_kind)

    def observe(
        *, agent, phase: str, bout_record: Mapping[str, Any], slot: int,
        generation: int, bout_step: int, rollout_index: int,
        rollout_rows: int, update: Mapping[str, Any],
    ) -> None:
        if phase != "bootstrap" or int(bout_step) != int(capture_steps):
            return
        task = str(bout_record["task"])
        if int(bout_record["group"]) != 0 or task not in task_variants:
            return
        variant = task_variants[task]
        if variant in captures:
            raise RuntimeError(f"duplicate source capture for old variant {variant}")
        local, shared, physical_groups = base.ws._actor_arrays(agent.core)
        effective = base._effective_live(local, shared, physical_groups)
        if int(slot) < 0 or int(slot) >= len(effective):
            raise RuntimeError("source capture slot is outside the actor")
        if int(agent.slot_generation[slot]) != int(generation):
            raise RuntimeError("source capture crossed a local-slot generation")
        captures[variant] = {
            "variant": int(variant),
            "task": task,
            "group": int(bout_record["group"]),
            "bout": int(bout_record["bout"]),
            "bout_step": int(bout_step),
            "global_environment_steps": int(agent.runtime_counts["environment_steps"]),
            "slot": int(slot),
            "generation": int(generation),
            "fitted_steps": int(agent.fitted_steps[slot]),
            "rollouts_trained": int(agent.runtime_counts["rollouts_trained"]),
            "minibatch_updates": int(agent.runtime_counts["minibatch_updates"]),
            "last_rollout_updates": int(update.get("updates", 0)),
            "rollout_index": int(rollout_index),
            "rollout_rows": int(rollout_rows),
            "physical_group": int(physical_groups[slot]),
            "effective_matrix_27": np.asarray(effective[slot], dtype=np.float32).reshape(27).copy(),
        }

    return observe


def _mean_captured_sources(captures: Mapping[int, Mapping[str, Any]]) -> np.ndarray:
    if set(int(key) for key in captures) != {0, 1}:
        raise ValueError("exactly old variants zero and one must be captured")
    matrices = []
    for variant in (0, 1):
        row = captures[variant]
        matrix = np.asarray(row["effective_matrix_27"], dtype=np.float32)
        if matrix.shape != (27,) or not np.isfinite(matrix).all():
            raise ValueError(f"old variant {variant} capture must contain 27 finite coefficients")
        matrices.append(matrix.reshape(3, 3, 3))
    # Match common_mean_effective: average the two source slots and all bands.
    return np.stack(matrices, axis=0).mean(axis=(0, 1), dtype=np.float64).astype(np.float32)


def _make_decision_observer(captures, frozen, original_decide):
    """Freeze both past-only decisions before the base worker runs any future."""
    def decide(history, evidence, old_parent, new_parent, score_fn):
        if frozen:
            raise RuntimeError("source-dose selector was already frozen")
        high = original_decide(history, evidence, old_parent, new_parent, score_fn)
        low = original_decide(
            history, evidence, _mean_captured_sources(captures), new_parent, score_fn,
        )
        frozen["low"] = copy.deepcopy(low)
        frozen["high"] = copy.deepcopy(high)
        return high
    return decide


def _snapshot_sources(output: Path, config_path: Path) -> dict[str, Any]:
    snapshot = output / "source_snapshot"
    sources: dict[str, dict[str, Any]] = {}
    paths = list(SOURCE_FILES)
    config_rel = Path(os.path.relpath(config_path.resolve(), ROOT.resolve()))
    if config_path.is_relative_to(ROOT) and config_rel not in paths:
        paths.append(config_rel)
    for relative in paths:
        if any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts):
            raise ValueError("private or temporary material is excluded from source snapshots")
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        digest = base._sha(source)
        if base._sha(target) != digest:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        sources[relative.as_posix()] = {"sha256": digest, "bytes": target.stat().st_size}
    base_config = base.loop.load_base_config()
    foundation, dataset = Path(base_config["foundation"]), Path(base_config["dataset"])
    manifest = {
        "format": "storage_source_dose_source_manifest_v1",
        "sources": sources,
        "foundation": {"path": str(foundation), "sha256": base._sha(foundation), "bytes": foundation.stat().st_size},
        "dataset": {"path": str(dataset), "sha256": base._sha(dataset), "bytes": dataset.stat().st_size},
        "excluded_local_instructions": ["Agent.md", "AGENTS.md"],
    }
    base._write_json(snapshot / "manifest.json", manifest)
    return manifest


def _write_observer_candidate(folder: Path, captures: Mapping[int, Mapping[str, Any]], candidate: np.ndarray) -> None:
    payload = {
        "format": "storage_source_dose_observer_v1",
        "source_capture_steps": int(next(iter(captures.values()))["bout_step"]),
        "capture_timing": "after PPO update, int8 actor roundtrip, and actor optimizer-state clear",
        "candidate_effective_matrix_3x3": candidate.tolist(),
        "candidate_coefficients": int(candidate.size),
        "candidate_storage": "observer only; never attached to learner or deployment package",
        "sources": {
            str(variant): {
                key: value for key, value in captures[variant].items()
                if key != "effective_matrix_27"
            } | {"effective_matrix_27": captures[variant]["effective_matrix_27"].tolist()}
            for variant in (0, 1)
        },
    }
    base._write_json(folder / "observer_candidate.json", payload)


def _matching_reusable_branch(
    base_branches: Sequence[Mapping[str, Any]], *, action: str,
    future_regime: str, action_state_hash: str,
) -> Mapping[str, Any] | None:
    """Reuse only replacement/none branches with a verified full action hash."""
    if action not in ("replace", "none"):
        return None
    matches = [
        row for row in base_branches
        if row.get("action") == action and row.get("future_regime") == future_regime
    ]
    if len(matches) != 1 or matches[0].get("action_state_hash") != action_state_hash:
        return None
    return matches[0]


def _run_candidate_branches(
    *, case_folder: Path, dose_folder: Path, base_record: Mapping[str, Any],
    decision: dict[str, Any], config: Mapping[str, Any], base_config: Mapping[str, Any],
    foundation: Path, old_candidate: np.ndarray, new_candidate: np.ndarray,
) -> list[dict[str, Any]]:
    case_id = str(base_record["case_id"])
    seed = int(base_record["seed"])
    pair_kind = str(base_record["pair_kind"])
    history_regime = str(base_record["history_regime"])
    foundation_sha = str(decision["foundation_sha256"])
    prefix_folder = case_folder / "prefix"
    disk_prefix_learner = base._capture_learner_from_saved(prefix_folder / "checkpoint.pt")
    disk_prefix_simulator = base._load_json(prefix_folder / "simulator.json")
    prefix_state_hash = str(decision["prefix_state_hash"])
    prefix_learner_hash = base.semantic_hash(disk_prefix_learner)
    branch_results: list[dict[str, Any]] = []
    for future_regime in config["future_regimes"]:
        future = base.template_schedule(
            decision["mapping"], str(future_regime), kind="future",
            steps=int(config["bout_steps"]),
            start_bout=len(base.prefix_schedule(decision["mapping"], config, history_regime)),
        )
        for chosen_action in base.ACTIONS:
            restore_cpu, restore_wall = time.process_time(), time.perf_counter()
            restored_env = None
            branch_folder = dose_folder / "branches" / f"{chosen_action}__{future_regime}"
            branch_folder.mkdir(parents=True, exist_ok=True)
            try:
                restored = base._restore_learner(disk_prefix_learner, base_config, config)
                restored_env, restored_runtime = base._restore_simulator(disk_prefix_simulator)
                restored_payload = base._capture_learner(restored, foundation_sha)
                restored_snapshot = base._capture_simulator(restored_env, restored_runtime)
                restored_hash = base.semantic_hash({"learner": restored_payload, "simulator": restored_snapshot})
                if restored_hash != prefix_state_hash:
                    raise RuntimeError("restored source-dose fork differs from the common prefix")

                branch_local, branch_shared, branch_groups = base.ws._actor_arrays(restored.core)
                branch_start_counts = dict(restored.runtime_counts)
                fork_observation = np.asarray(restored_runtime["obs"], dtype=np.float32)
                fork_probabilities = base._resident_probabilities(restored.core, fork_observation, restored.slots)
                before_effective = base._effective_live(branch_local, branch_shared, branch_groups)
                post_local, post_shared, post_groups, intervention = base.apply_action(
                    branch_local, branch_shared, branch_groups, restored.slot_group_tags,
                    old_candidate, new_candidate, chosen_action,
                )
                restored.core.install_local_matrix(post_local)
                restored.core.install_shared_matrix(post_shared)
                with torch.no_grad():
                    restored.core.slot_groups.copy_(torch.as_tensor(post_groups.astype(np.int64), dtype=torch.long))
                restored.parent_group = {"keep": 0, "replace": 1, "mean": -2, "none": -1}[chosen_action]
                after_prequant = base._effective_live(post_local, post_shared, post_groups)
                prequant_per_slot = np.max(np.abs(after_prequant - before_effective), axis=(1, 2, 3)).astype(float).tolist()
                prequant_probabilities = base._resident_probabilities(restored.core, fork_observation, restored.slots)
                prequant_behavior_drift = base._probability_drift(fork_probabilities, prequant_probabilities)
                action_actor, action_codec_error, action_codec_effective_error = base._encode_live_actor(restored)
                postquant_probabilities = base._resident_probabilities(restored.core, fork_observation, restored.slots)
                codec_behavior_drift = base._probability_drift(fork_probabilities, postquant_probabilities)
                base.ws._clear_optimizer_state(restored)
                action_checkpoint = base._capture_learner(restored, foundation_sha)
                action_hash = base.semantic_hash({"learner": action_checkpoint, "simulator": restored_snapshot})

                original_branch = _matching_reusable_branch(
                    base_record.get("branches", []), action=chosen_action,
                    future_regime=str(future_regime), action_state_hash=action_hash,
                )
                if original_branch is not None:
                    original_folder = case_folder / "branches" / f"{chosen_action}__{future_regime}"
                    relative_summary = Path(os.path.relpath(original_folder / "summary.json", branch_folder)).as_posix()
                    alias_elapsed = {
                        "cpu_s": float(time.process_time() - restore_cpu),
                        "wall_s": float(time.perf_counter() - restore_wall),
                    }
                    alias = {
                        "format": "storage_source_dose_branch_alias_v1",
                        "case_id": case_id,
                        "seed": seed,
                        "pair_kind": pair_kind,
                        "history_regime": history_regime,
                        "action": chosen_action,
                        "candidate_set": "old_source_1024",
                        "old_candidate_source_steps": int(config["source_capture_steps"]),
                        "future_regime": str(future_regime),
                        "primary_continuation": history_regime == str(future_regime),
                        "stress_continuation": history_regime != str(future_regime),
                        "independent_observation": False,
                        "reuse_of": relative_summary,
                        "reuse_reason": "complete post-action learner/simulator semantic hash matched the 2048-source branch",
                        "prefix_state_hash": prefix_state_hash,
                        "restored_state_hash": restored_hash,
                        "action_state_hash": action_hash,
                        "logical_final_state_hash": original_branch.get("final_state_hash"),
                        "logical_environment_steps": int(original_branch["environment_steps"]),
                        "environment_steps": 0,
                        "reward_total": None,
                        "reward_per_step": None,
                        "counts": {
                            "environment_steps": 0,
                            "completed_episodes": 0,
                            "slot_allocations": 0,
                            "slot_evictions": 0,
                            "rollouts_trained": 0,
                            "minibatch_updates": 0,
                        },
                        "logical_reward_total": original_branch.get("reward_total"),
                        "logical_reward_per_step": original_branch.get("reward_per_step"),
                        "logical_per_group": original_branch.get("per_group"),
                        "logical_per_bout": original_branch.get("per_bout"),
                        "logical_counts": original_branch.get("counts"),
                        "capacity": original_branch.get("capacity"),
                        "capacity_after_action": original_branch.get("capacity_after_action"),
                        "actual_costs": {
                            **alias_elapsed,
                            "environment_steps": 0,
                            "validation_cpu_s": alias_elapsed["cpu_s"],
                            "validation_wall_s": alias_elapsed["wall_s"],
                        },
                        "costs": {
                            **alias_elapsed,
                            "simulation_cpu_s": 0.0,
                            "simulation_wall_s": 0.0,
                            "restore_cpu_s": alias_elapsed["cpu_s"],
                            "restore_wall_s": alias_elapsed["wall_s"],
                        },
                    }
                    base._write_json(branch_folder / "summary.json", alias)
                    branch_results.append(alias)
                    decision["branches"].append({
                        "action": chosen_action,
                        "future_regime": str(future_regime),
                        "summary": f"branches/{chosen_action}__{future_regime}/summary.json",
                        "reuse_of": relative_summary,
                        "independent_observation": False,
                    })
                    base._write_json(dose_folder / "decision.json", decision)
                    continue

                post_action_capacity = base._save_package(branch_folder, action_checkpoint, action_actor, foundation, config)
                post_action_folder = branch_folder / "post_action"
                post_action_folder.mkdir(parents=True, exist_ok=True)
                (post_action_folder / "checkpoint.pt").write_bytes((branch_folder / "checkpoint.pt").read_bytes())
                (post_action_folder / "actor.bin").write_bytes((branch_folder / "actor.bin").read_bytes())
                base._write_json(branch_folder / "simulator_start.json", restored_snapshot)
                base._write_json(post_action_folder / "simulator.json", restored_snapshot)
                base._write_json(post_action_folder / "manifest.json", {
                    "role": "observer_audit_snapshot",
                    "included_in_deployment_capacity": False,
                    "action_state_hash": action_hash,
                })

                branch_start_cpu, branch_start_wall = time.process_time(), time.perf_counter()
                branch_bouts = base._run_schedule(
                    restored, restored_env, restored_runtime, future,
                    folder=branch_folder, phase="future", action=chosen_action,
                    future_regime=str(future_regime),
                )
                final_actor, final_codec_error, final_effective_error = base._encode_live_actor(restored)
                final_learner = base._capture_learner(restored, foundation_sha)
                final_simulator = base._capture_simulator(restored_env, restored_runtime)
                final_hash = base.semantic_hash({"learner": final_learner, "simulator": final_simulator})
                package = base._save_package(branch_folder, final_learner, final_actor, foundation, config)
                base._write_json(branch_folder / "simulator.json", final_simulator)
                elapsed = {
                    "cpu_s": float(time.process_time() - restore_cpu),
                    "wall_s": float(time.perf_counter() - restore_wall),
                    "simulation_cpu_s": float(time.process_time() - branch_start_cpu),
                    "simulation_wall_s": float(time.perf_counter() - branch_start_wall),
                    "restore_cpu_s": float(time.process_time() - restore_cpu),
                    "restore_wall_s": float(time.perf_counter() - restore_wall),
                }
                summary = base._summarize_bouts(branch_bouts, restored, elapsed, start_counts=branch_start_counts)
                summary.update({
                    "format": "storage_source_dose_branch_v1",
                    "case_id": case_id,
                    "seed": seed,
                    "pair_kind": pair_kind,
                    "history_regime": history_regime,
                    "action": chosen_action,
                    "candidate_set": "old_source_1024",
                    "old_candidate_source_steps": int(config["source_capture_steps"]),
                    "future_regime": str(future_regime),
                    "primary_continuation": history_regime == str(future_regime),
                    "stress_continuation": history_regime != str(future_regime),
                    "prefix_state_hash": prefix_state_hash,
                    "restored_state_hash": restored_hash,
                    "action_state_hash": action_hash,
                    "final_state_hash": final_hash,
                    "foundation_sha256": foundation_sha,
                    "foundation_unchanged": base._sha(foundation) == foundation_sha,
                    "prequantized_effective_errors": prequant_per_slot,
                    "fork_observation": fork_observation.tolist(),
                    "prequantized_behavior_drift": prequant_behavior_drift,
                    "postquantized_behavior_drift": codec_behavior_drift,
                    "prequantized_resident_probabilities": prequant_probabilities.tolist(),
                    "postquantized_resident_probabilities": postquant_probabilities.tolist(),
                    "codec_max_coefficient_error": action_codec_error,
                    "codec_effective_errors": action_codec_effective_error,
                    "final_codec_max_coefficient_error": final_codec_error,
                    "final_codec_effective_errors": final_effective_error,
                    "preservation_diagnostic": intervention,
                    "capacity": package,
                    "capacity_after_action": post_action_capacity,
                    "post_action_checkpoint": "post_action/checkpoint.pt",
                    "post_action_simulator": "post_action/simulator.json",
                    "post_action_snapshot_included_in_deployment_capacity": False,
                    "actor_payload_bytes": len(final_actor),
                    "temporary_candidate_workspace_bytes": 36,
                    "future_schedule": future,
                    "prefix_bouts_included_in_reward": False,
                })
                base._write_json(branch_folder / "summary.json", summary)
                branch_results.append(summary)
                decision["branches"].append({
                    "action": chosen_action,
                    "future_regime": str(future_regime),
                    "summary": f"branches/{chosen_action}__{future_regime}/summary.json",
                })
                base._write_json(dose_folder / "decision.json", decision)
            finally:
                if restored_env is not None:
                    restored_env.close()
    return branch_results


def _persisted_step_counts(case_folder: Path) -> dict[str, int | None]:
    prefix = base._count_step_rows(case_folder / "prefix" / "steps.csv")
    branch_values: list[int | None] = []
    for branch_root in (case_folder / "branches", case_folder / "dose_1024" / "branches"):
        if branch_root.is_dir():
            for branch in sorted(branch_root.iterdir()):
                if branch.is_dir():
                    branch_values.append(base._count_step_rows(branch / "steps.csv"))
    branch_total = None if any(value is None for value in branch_values) else int(sum(branch_values))
    total = None if prefix is None or branch_total is None else int(prefix + branch_total)
    dose_values = []
    dose_root = case_folder / "dose_1024" / "branches"
    if dose_root.is_dir():
        for branch in sorted(dose_root.iterdir()):
            if branch.is_dir():
                dose_values.append(base._count_step_rows(branch / "steps.csv"))
    dose_total = None if any(value is None for value in dose_values) else int(sum(dose_values))
    base_values = []
    base_root = case_folder / "branches"
    if base_root.is_dir():
        for branch in sorted(base_root.iterdir()):
            if branch.is_dir():
                base_values.append(base._count_step_rows(branch / "steps.csv"))
    base_total = None if any(value is None for value in base_values) else int(sum(base_values))
    return {
        "prefix_environment_steps": prefix,
        "base_branch_environment_steps": base_total,
        "source_1024_branch_environment_steps": dose_total,
        "branch_environment_steps": branch_total,
        "environment_steps_total": total,
    }


def _case_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(payload["seed"])
    pair_kind = str(payload["pair_kind"])
    history_regime = str(payload["history_regime"])
    config = validate_config(payload["config"])
    case_id = f"s{seed}_{pair_kind}_{history_regime}"
    case_folder = Path(payload["output"]) / "cases" / case_id
    dose_folder = case_folder / "dose_1024"
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    captures: dict[int, dict[str, Any]] = {}
    observer = _make_source_observer(
        captures, seed=seed, pair_kind=pair_kind,
        capture_steps=int(config["source_capture_steps"]),
    )
    original_new_agent = base._new_agent
    selector_records: dict[str, Any] = {}
    decision_observer = _make_decision_observer(captures, selector_records, base.decide_from_past)

    def observer_new_agent(*args, **kwargs):
        agent = original_new_agent(*args, **kwargs)
        agent._storage_decision_post_rollout_observer = observer
        return agent

    try:
        with patch.object(base, "_new_agent", observer_new_agent), patch.object(base, "decide_from_past", decision_observer):
            base_record = base._case_worker({
                "seed": seed,
                "pair_kind": pair_kind,
                "history_regime": history_regime,
                "config": dict(config),
                "output": str(payload["output"]),
            })
        if set(captures) != {0, 1}:
            raise RuntimeError(f"expected captures for old variants zero and one; got {sorted(captures)}")
        case_folder.mkdir(parents=True, exist_ok=True)
        dose_folder.mkdir(parents=True, exist_ok=True)
        base_decision = base._load_json(case_folder / "decision.json")
        old_candidate_1024 = _mean_captured_sources(captures)
        old_candidate_2048 = np.asarray(base_decision["old_parent"], dtype=np.float32).reshape(3, 3)
        new_candidate = np.asarray(base_decision["new_candidate"], dtype=np.float32).reshape(3, 3)
        frozen_decision = {
            "format": "storage_source_dose_case_v1",
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "prefix_state_hash": base_decision["prefix_state_hash"],
            "foundation_sha256": base_decision["foundation_sha256"],
            "mapping": base_decision["mapping"],
            "old_candidate_2048": old_candidate_2048.tolist(),
            "old_candidate_1024": old_candidate_1024.tolist(),
            "new_candidate": new_candidate.tolist(),
            "source_capture_steps": int(config["source_capture_steps"]),
            "source_capture_records": {
                str(variant): {
                    key: value for key, value in captures[variant].items()
                    if key != "effective_matrix_27"
                } | {"effective_matrix_27": captures[variant]["effective_matrix_27"].tolist()}
                for variant in (0, 1)
            },
            "selectors_frozen_before_future_expansion": True,
            "selectors": selector_records["low"]["past_only_selectors"],
            "selector_diagnostics": selector_records["low"],
            "selector_input_candidate_set": "old_candidate_1024",
            "selector_freeze_timing": "both dose rules computed before any future expansion from common prefix evidence",
            "candidate_workspace_in_learner": False,
            "learner_parent_count": 1,
            "branches": [],
        }
        base._write_json(dose_folder / "decision.json", frozen_decision)
        _write_observer_candidate(dose_folder, captures, old_candidate_1024)
        base_config = base.loop.load_base_config()
        foundation = Path(base_config["foundation"])
        dose_branches = _run_candidate_branches(
            case_folder=case_folder,
            dose_folder=dose_folder,
            base_record=base_record,
            decision=frozen_decision,
            config=config,
            base_config=base_config,
            foundation=foundation,
            old_candidate=old_candidate_1024,
            new_candidate=new_candidate,
        )
        branch_steps = sum(int(row["environment_steps"]) for row in dose_branches)
        logical_dose_steps = sum(
            int(row.get("logical_environment_steps", row["environment_steps"]))
            for row in dose_branches
        )
        branch_cpu_s = sum(float(row["costs"]["cpu_s"]) for row in dose_branches)
        branch_wall_s = sum(float(row["costs"]["wall_s"]) for row in dose_branches)
        physical_dose_branches = sum(row.get("format") != "storage_source_dose_branch_alias_v1" for row in dose_branches)
        base_costs = dict(base_record.get("costs", {}))
        prefix_steps = int(base_costs.get("prefix_environment_steps", 0))
        base_branch_steps = int(base_costs.get("branch_environment_steps", 0))
        result = {
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "prefix": base_record.get("prefix"),
            "base_branches": base_record.get("branches", []),
            "source_1024_branches": dose_branches,
            "selectors": frozen_decision["selectors"],
            "source_candidates": {
                "old_2048": old_candidate_2048.tolist(),
                "old_1024": old_candidate_1024.tolist(),
                "new": new_candidate.tolist(),
            },
            "costs": {
                "prefix_environment_steps": prefix_steps,
                "base_branch_environment_steps": base_branch_steps,
                "source_1024_branch_environment_steps": branch_steps,
                "source_1024_logical_environment_steps": logical_dose_steps,
                "source_1024_physical_branch_count": int(physical_dose_branches),
                "branch_environment_steps": base_branch_steps + branch_steps,
                "environment_steps_total": prefix_steps + base_branch_steps + branch_steps,
                "logical_environment_steps_total": prefix_steps + base_branch_steps + logical_dose_steps,
                "prefix_cpu_s": base_costs.get("prefix_cpu_s"),
                "base_branch_cpu_s": base_costs.get("branch_cpu_s"),
                "source_1024_branch_cpu_s": branch_cpu_s,
                "case_cpu_s": float(time.process_time() - start_cpu),
                "prefix_wall_s": base_costs.get("prefix_wall_s"),
                "base_branch_wall_s": base_costs.get("branch_wall_s"),
                "source_1024_branch_wall_s": branch_wall_s,
                "case_wall_s": float(time.perf_counter() - start_wall),
            },
            "errors": list(base_record.get("errors", [])),
        }
        base._write_json(dose_folder / "summary.json", {
            "format": "storage_source_dose_case_summary_v1",
            "case_id": case_id,
            "source_1024_branches": dose_branches,
            "costs": result["costs"],
            "errors": result["errors"],
        })
        return result
    except Exception:
        persisted = _persisted_step_counts(case_folder)
        error = {
            "case_id": case_id,
            "error_type": "source_dose_case_failure",
            "error": traceback.format_exc(),
            "persisted_interactions": persisted,
            "count_source": "persisted prefix and branch steps.csv rows",
        }
        case_folder.mkdir(parents=True, exist_ok=True)
        base._write_json(case_folder / "source_dose_error.json", error)
        return {
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "base_branches": [],
            "source_1024_branches": [],
            "costs": {
                **persisted,
                "case_cpu_s": float(time.process_time() - start_cpu),
                "case_wall_s": float(time.perf_counter() - start_wall),
                "prefix_cpu_s": None,
                "base_branch_cpu_s": None,
                "source_1024_branch_cpu_s": None,
                "prefix_wall_s": None,
                "base_branch_wall_s": None,
                "source_1024_branch_wall_s": None,
            },
            "errors": [error],
        }


def _jobs(config: Mapping[str, Any], output: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), "pair_kind": pair, "history_regime": history,
         "config": dict(config), "output": str(output)}
        for seed in seeds
        for pair in config["pair_kinds"]
        for history in config["history_regimes"]
    ]


def run(
    config: Mapping[str, Any] | str | Path,
    output: str | Path,
    *, seeds: Sequence[int] | None = None,
    pilot: bool = False,
) -> dict[str, Any]:
    config_path = Path(config).resolve() if isinstance(config, (str, Path)) else CONFIG_PATH
    raw_config = json.loads(config_path.read_text(encoding="utf-8")) if isinstance(config, (str, Path)) else dict(config)
    normalized = _phase_config(raw_config, pilot=pilot)
    run_seeds = tuple(int(value) for value in (seeds if seeds is not None else normalized["seeds"]))
    if not run_seeds:
        raise ValueError("at least one seed is required")
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    base._write_json(output_path / "config.json", normalized)
    base._write_json(output_path / "source_snapshot" / "config_original.json", raw_config)
    source_manifest = _snapshot_sources(output_path, config_path)
    jobs = _jobs(normalized, output_path, run_seeds)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    workers = max(1, min(int(normalized["max_workers"]), len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_case_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record = future.result()
                records.append(record)
                errors.extend(record.get("errors", []))
            except Exception:
                case_id = f"s{job['seed']}_{job['pair_kind']}_{job['history_regime']}"
                case_folder = output_path / "cases" / case_id
                persisted = _persisted_step_counts(case_folder)
                error = {
                    "case_id": case_id,
                    "error_type": "source_dose_worker_failure",
                    "error": traceback.format_exc(),
                    "persisted_interactions": persisted,
                    "count_source": "persisted prefix and branch steps.csv rows",
                }
                base._write_json(case_folder / "source_dose_error.json", error)
                errors.append(error)
                records.append({
                    "case_id": case_id,
                    "seed": int(job["seed"]),
                    "pair_kind": str(job["pair_kind"]),
                    "history_regime": str(job["history_regime"]),
                    "base_branches": [],
                    "source_1024_branches": [],
                    "costs": {
                        **persisted,
                        "case_cpu_s": None,
                        "case_wall_s": None,
                        "prefix_cpu_s": None,
                        "base_branch_cpu_s": None,
                        "source_1024_branch_cpu_s": None,
                        "prefix_wall_s": None,
                        "base_branch_wall_s": None,
                        "source_1024_branch_wall_s": None,
                    },
                    "errors": [error],
                })
    records.sort(key=lambda row: (int(row["seed"]), str(row["pair_kind"]), str(row["history_regime"])))
    base_branches = [branch for row in records for branch in row.get("base_branches", [])]
    dose_branches = [branch for row in records for branch in row.get("source_1024_branches", [])]
    physical_dose_branches = [
        branch for branch in dose_branches
        if branch.get("format") != "storage_source_dose_branch_alias_v1"
    ]

    def aggregate(key: str):
        values = [row.get("costs", {}).get(key) for row in records]
        if any(value is None for value in values):
            return None
        return sum(values)

    orchestrator_cpu = float(time.process_time() - start_cpu)
    child_cpu = aggregate("case_cpu_s")
    summary = {
        "format": FORMAT,
        "phase": "pilot" if pilot else "main",
        "config": normalized,
        "source_manifest": source_manifest,
        "seeds": list(run_seeds),
        "case_count": len(records),
        "base_branch_count": len(base_branches),
        "source_1024_branch_count": len(dose_branches),
        "source_1024_physical_branch_count": len(physical_dose_branches),
        "logical_branch_count": len(base_branches) + len(dose_branches),
        "physical_branch_count": len(base_branches) + len(physical_dose_branches),
        "branch_count": len(base_branches) + len(dose_branches),
        "cases": records,
        "base_branches": base_branches,
        "source_1024_branches": dose_branches,
        "errors": errors,
        "costs": {
            "prefix_environment_steps": aggregate("prefix_environment_steps"),
            "base_branch_environment_steps": aggregate("base_branch_environment_steps"),
            "source_1024_branch_environment_steps": aggregate("source_1024_branch_environment_steps"),
            "source_1024_logical_environment_steps": aggregate("source_1024_logical_environment_steps"),
            "branch_environment_steps": aggregate("branch_environment_steps"),
            "environment_steps_total": aggregate("environment_steps_total"),
            "logical_environment_steps_total": aggregate("logical_environment_steps_total"),
            "child_cpu_s": child_cpu,
            "orchestrator_cpu_s": orchestrator_cpu,
            "cpu_s": None if child_cpu is None else float(child_cpu + orchestrator_cpu),
            "child_wall_s": aggregate("case_wall_s"),
            "wall_s": float(time.perf_counter() - start_wall),
            "prefix_cpu_s": aggregate("prefix_cpu_s"),
            "base_branch_cpu_s": aggregate("base_branch_cpu_s"),
            "source_1024_branch_cpu_s": aggregate("source_1024_branch_cpu_s"),
            "prefix_wall_s": aggregate("prefix_wall_s"),
            "base_branch_wall_s": aggregate("base_branch_wall_s"),
            "source_1024_branch_wall_s": aggregate("source_1024_branch_wall_s"),
        },
    }
    base._write_json(output_path / "summary.json", summary)
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
        "errors": len(summary["errors"]),
        "costs": summary["costs"],
    }, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

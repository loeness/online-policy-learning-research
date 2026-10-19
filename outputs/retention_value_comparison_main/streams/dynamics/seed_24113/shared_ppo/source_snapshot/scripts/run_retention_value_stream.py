"""Run the frozen selected-feedback and repeated-retention streams.

This harness uses the existing continuous PPO stream and a single transient
candidate formed from two currently resident task generations. The learner is
given past task and group observations only; the future block schedule is
generated after the action has been selected.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
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

from online_rl.retention_stream_value import (
    BoundedRetentionStats,
    apply_retention_action,
    frequency_action,
    generalized_lower_decision,
)
from online_rl.retention_value import RetentionValueModel
from online_rl.storage_decision import (
    ACTIONS,
    BoundedGroupEvidence,
    common_mean_effective,
    effective_matrices,
    pair_mapping,
    semantic_hash,
    task_name,
)
from online_rl.write_decision_policy import decode_actor, encode_actor
from scripts import run_knowledge_loop as loop
from scripts import run_storage_decision as base
from scripts import run_write_stream as ws


CONFIG_PATH = ROOT / "configs" / "retention_value_stream.json"
PROTOCOL_PATH = ROOT / "docs" / "RETENTION_VALUE_STREAM_PROTOCOL.md"
METHODS = (
    "learned_linear", "learned_intercept", "random", "keep", "replace",
    "mean", "frequency", "lower", "shared_ppo", "independent",
)
MODEL_CONFIGS = {"n_features": 6, "ridge": 1.0, "discount": 0.97, "lambda_cpu": 1.0}
WARMUP_BOUTS = 4
WARMUP_STEPS = 3072
OUTCOME_BOUTS = 3
OUTCOME_BOUT_STEPS = 512
OUTCOME_STEPS = OUTCOME_BOUTS * OUTCOME_BOUT_STEPS


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"), default=_json_default))
        stream.write("\n")
        stream.flush()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"retention value stream config does not exist: {config_path}")
    result = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(result)
    return result


def _validate_config(config: Mapping[str, Any]) -> None:
    """Validate this protocol's v1 config without mutating it or base config."""
    if not isinstance(config, Mapping) or config.get("format") != "retention_value_stream_v1":
        raise ValueError("format must be retention_value_stream_v1")
    frozen = {
        "pair_kinds": ["dynamics", "reward_conflict"],
        "local_slots": 2,
        "shared_groups": 1,
        "rollout_steps": 256,
        "local_learning_rate": 0.003,
        "shared_learning_rate": 0.0,
        "direct_shared_learning_rate": 0.003,
        "critic_learning_rate": 0.003,
        "actor_budget_bytes": 89,
        "package_budget_bytes": 262144,
        "evidence_rows_per_group": 256,
        "recent_bouts": 8,
        "bootstrap_steps": 1024,
        "bout_steps": 512,
        "minimum_source_fitted_steps": 512,
        "outcome_bouts": 3,
        "regime_blocks": 8,
        "checkpoint_every": 8,
        "torch_threads": 1,
    }
    for key, expected in frozen.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} is frozen to {expected!r}")
    value_model = config.get("value_model")
    expected_model = {
        "n_features": 6, "ridge": 1.0, "discount": 0.97,
        "lambda_cpu": 1.0, "epsilon": 0.1, "uniform_warmup_decisions": 16,
    }
    if not isinstance(value_model, Mapping) or dict(value_model) != expected_model:
        raise ValueError("value_model does not match the frozen protocol")
    calibration = config.get("calibration")
    comparison = config.get("comparison")
    if not isinstance(calibration, Mapping) or not isinstance(comparison, Mapping):
        raise ValueError("calibration and comparison settings are required")
    if calibration.get("methods") != ["random"] or int(calibration.get("decisions", 0)) != 64:
        raise ValueError("calibration must use random for 64 decisions")
    if tuple(comparison.get("methods", ())) != METHODS or int(comparison.get("decisions", 0)) != 48:
        raise ValueError("comparison methods or decision count do not match the frozen protocol")
    for name, section in (("calibration", calibration), ("comparison", comparison)):
        pilot = section.get("pilot")
        if not isinstance(pilot, Mapping):
            raise ValueError(f"{name}.pilot settings are required")
        if int(pilot.get("decisions", 0)) != 4 or tuple(pilot.get("pair_kinds", ())) != ("dynamics",):
            raise ValueError(f"{name}.pilot must use one dynamics stream with four decisions")
        if not section.get("seeds") or not pilot.get("seeds"):
            raise ValueError(f"{name} seed lists must not be empty")


def _stream_grid(config: Mapping[str, Any], stage: str, pilot: bool) -> list[dict[str, Any]]:
    _validate_config(config)
    if stage not in ("calibration", "comparison"):
        raise ValueError("stage must be calibration or comparison")
    section = config[stage]
    selected = section["pilot"] if pilot else section
    seeds = [int(value) for value in selected["seeds"]]
    pair_kinds = [str(value) for value in (selected.get("pair_kinds", config["pair_kinds"]))]
    methods = [str(value) for value in section["methods"]]
    if stage == "calibration" and methods != ["random"]:
        raise ValueError("calibration must contain only the random method")
    if stage == "comparison" and methods != list(METHODS):
        raise ValueError("comparison method order is frozen")
    decisions = int(selected["decisions"])
    return [
        {
            "stage": stage,
            "pilot": bool(pilot),
            "pair_kind": pair_kind,
            "seed": seed,
            "method": method,
            "decisions": decisions,
        }
        for pair_kind in pair_kinds
        for seed in seeds
        for method in methods
    ]


def _agent_config(config: Mapping[str, Any], *, method: str) -> dict[str, Any]:
    return {
        "phase": "known",
        "domain": "navigation",
        "local_slots": 2,
        "shared_groups": 1,
        "local_learning_rate": float(config["local_learning_rate"]),
        "fast_local_learning_rate": float(config["local_learning_rate"]),
        "shared_learning_rate": float(
            config["direct_shared_learning_rate"] if method == "shared_ppo" else config["shared_learning_rate"]
        ),
        "critic_learning_rate": float(config["critic_learning_rate"]),
        "anchors_per_slot": 24,
        "probe_steps": 256,
        "reward_support_aware": False,
        "group_reward_min_samples": 1,
        "actor_codec": "int8",
        "actor_budget_bytes": int(config["actor_budget_bytes"]),
        "package_budget_bytes": int(config["package_budget_bytes"]),
        "evidence_rows_per_group": int(config["evidence_rows_per_group"]),
        "torch_threads": int(config["torch_threads"]),
    }


def _new_agent(seed: int, base_config: Mapping[str, Any], config: Mapping[str, Any], method: str):
    torch.set_num_threads(int(config["torch_threads"]))
    agent = ws._new_agent("reward_shared", int(seed), _agent_config(config, method=method), base_config)
    agent.core.freeze_shared(True)
    agent.cache = ws.BoundedTaskCache(2, "lru", int(seed) + 8811)
    agent.behavior_rng = np.random.default_rng(int(seed) + 1100)
    agent.evidence = BoundedGroupEvidence(int(config["evidence_rows_per_group"]))
    agent.recent_groups = []
    agent.parent_group = -1
    agent.slot_group_tags = np.full(2, -1, dtype=np.int8)
    agent.parent_version = 0
    agent.parent_source_fit = 0.0
    agent.runtime_counts = {
        "environment_steps": 0,
        "completed_episodes": 0,
        "slot_allocations": 0,
        "slot_evictions": 0,
        "rollouts_trained": 0,
        "minibatch_updates": 0,
    }
    if method == "independent":
        # Keep the shared actor container in the same 89-byte layout, but
        # permanently leave its parent zero and every local unbound.
        agent.core.install_shared_matrix(np.zeros((1, 9), dtype=np.float32))
        with torch.no_grad():
            agent.core.slot_groups.fill_(-1)
        agent.parent_group = -1
    return agent


def _task_groups(mapping: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, int]]:
    group_by_task: dict[str, int] = {}
    for group in (0, 1):
        for variant in (0, 1, 2):
            group_by_task[task_name(mapping, group, variant)] = group
    if len(group_by_task) != 6:
        raise ValueError("pair mapping did not produce six unique task IDs")
    return tuple(group_by_task), group_by_task


def _task_variant(mapping: Mapping[str, Any], task: str, group: int) -> int:
    for variant in (0, 1, 2):
        if task_name(mapping, group, variant) == task:
            return variant
    raise ValueError(f"task {task!r} is not a variant in group {group}")


def _warmup_schedule(mapping: Mapping[str, Any], *, group: int, steps: int, bout_start: int) -> list[dict[str, Any]]:
    return [
        {
            "phase": "warmup", "bout": int(bout_start + variant),
            "group": int(group), "variant": int(variant),
            "task": task_name(mapping, group, variant), "steps": int(steps),
        }
        for variant in (0, 1)
    ]


def _outcome_schedule(
    mapping: Mapping[str, Any], seed: int, decision_index: int, *, bout_start: int,
    regime_blocks: int = 8, bout_steps: int = 512,
) -> list[dict[str, Any]]:
    dominant = (int(seed) % 2 + int(decision_index) // int(regime_blocks)) % 2
    minority = 1 - dominant
    requests = (
        (minority, int(decision_index) % 3),
        (dominant, int(decision_index) % 3),
        (dominant, (int(decision_index) + 1) % 3),
    )
    return [
        {
            "phase": "outcome", "bout": int(bout_start + index),
            "group": int(group), "variant": int(variant),
            "task": task_name(mapping, group, variant), "steps": int(bout_steps),
        }
        for index, (group, variant) in enumerate(requests)
    ]


def _current_candidate(agent: Any, source_records: Sequence[Mapping[str, Any]], mapping: Mapping[str, Any], minimum_fit: int = 512):
    """Build one transient candidate, validating live generation and group."""
    if len(source_records) != 2:
        raise ValueError("a candidate requires exactly two source bouts")
    records = list(source_records)
    groups = {int(record["group"]) for record in records}
    if len(groups) != 1:
        raise ValueError("candidate sources must belong to one logical group")
    group = next(iter(groups))
    variants = [_task_variant(mapping, str(record["task"]), group) for record in records]
    if len(set(variants)) != 2:
        raise ValueError("candidate sources must use different task variants")
    slots: list[int] = []
    provenance_rows = []
    for record, variant in zip(records, variants):
        task = str(record["task"])
        slot = int(record["slot"])
        generation = int(record["generation"])
        if agent.cache.task_to_slot.get(task) != slot or agent.cache.slot_to_task[slot] != task:
            raise ValueError(f"candidate task {task!r} is no longer resident")
        if int(agent.slot_generation[slot]) != generation:
            raise ValueError(f"candidate task {task!r} generation is no longer resident")
        if int(agent.slot_group_tags[slot]) != group:
            raise ValueError(f"candidate task {task!r} logical group tag does not match its bout")
        fitted = int(agent.fitted_steps[slot])
        if fitted < int(minimum_fit):
            raise ValueError(f"candidate source {task!r} has only {fitted} fitted steps")
        slots.append(slot)
        provenance_rows.append({
            "task": task,
            "group": group,
            "variant": variant,
            "slot": slot,
            "generation": generation,
            "fitted_steps": fitted,
            "bout": int(record["bout"]),
        })
    if slots[0] == slots[1]:
        raise ValueError("candidate sources must occupy different local slots")
    local, shared, physical_groups = ws._actor_arrays(agent.core)
    candidate = common_mean_effective(local, shared, physical_groups, slots)
    source_fit = float(np.mean([row["fitted_steps"] for row in provenance_rows]))
    candidate = np.asarray(candidate, dtype=np.float32)
    provenance = {
        "candidate_group": group,
        "sources": provenance_rows,
        "source_fit_steps_mean": source_fit,
        "candidate_sha256": _sha256_bytes(np.ascontiguousarray(candidate).tobytes()),
        "candidate_matrix_persisted": False,
    }
    return candidate, group, source_fit, provenance


def _set_actor(agent: Any, local: np.ndarray, shared: np.ndarray, physical_groups: np.ndarray) -> None:
    agent.core.install_local_matrix(np.asarray(local, dtype=np.float32))
    agent.core.install_shared_matrix(np.asarray(shared, dtype=np.float32))
    with torch.no_grad():
        agent.core.slot_groups.copy_(torch.as_tensor(np.asarray(physical_groups, dtype=np.int64), dtype=torch.long))


def _codec_roundtrip(agent: Any) -> tuple[bytes, dict[str, Any]]:
    local_before, shared_before, groups_before = ws._actor_arrays(agent.core)
    effective_before = effective_matrices(local_before, shared_before, groups_before)
    encoded = encode_actor(local_before, shared_before, groups_before, codec="int8", budget_bytes=89)
    local, shared, groups = decode_actor(encoded.data)
    _set_actor(agent, local, shared, groups)
    effective_after = effective_matrices(local, shared, groups)
    per_slot = np.max(np.abs(effective_after - effective_before), axis=(1, 2, 3)).astype(float).tolist()
    coefficient = max(
        float(np.max(np.abs(local - local_before), initial=0.0)),
        float(np.max(np.abs(shared - shared_before), initial=0.0)),
    )
    return encoded.data, {
        "actor_bytes": len(encoded.data),
        "max_coefficient_error": coefficient,
        "per_slot_effective_error": per_slot,
        "max_effective_error": float(max(per_slot, default=0.0)),
    }


def _install_initial_parent(agent: Any, candidate: np.ndarray, group: int, source_fit: float, stats: BoundedRetentionStats, decision_index: int) -> dict[str, Any]:
    local, shared, groups = ws._actor_arrays(agent.core)
    local_new, shared_new, groups_new, details = apply_retention_action(
        local, shared, groups, agent.slot_group_tags, agent.parent_group,
        candidate, group, "replace",
    )
    _set_actor(agent, local_new, shared_new, groups_new)
    agent.parent_group = int(details["parent_group"])
    agent.parent_source_fit = float(source_fit)
    agent.parent_version += 1
    stats.record_parent_write(source_fit, decision_index)
    _codec_roundtrip(agent)
    return details


def _make_direct_shared(agent: Any, stats: BoundedRetentionStats, decision_index: int) -> dict[str, Any]:
    local, shared, physical = ws._actor_arrays(agent.core)
    from online_rl.storage_decision import compensated_single_parent_install

    local_new, shared_new, physical_new, details = compensated_single_parent_install(
        local, shared, physical, agent.slot_group_tags,
        parent=shared[0], parent_group=-2,
    )
    _set_actor(agent, local_new, shared_new, physical_new)
    agent.parent_group = -2
    agent.parent_version += 1
    stats.record_parent_write(agent.parent_source_fit, decision_index)
    agent.core.freeze_shared(False)
    shared_groups = [group for group in agent.optimizer.param_groups if group.get("name") == "shared"]
    if len(shared_groups) != 1 or not np.isclose(float(shared_groups[0]["lr"]), 0.003, rtol=0.0, atol=1e-12):
        raise RuntimeError("shared_ppo must have one shared optimizer group at learning rate 0.003")
    _codec_roundtrip(agent)
    return details


def _capture_checkpoint(
    agent: Any,
    foundation_sha: str,
    models: Mapping[str, RetentionValueModel],
    stats: BoundedRetentionStats,
    selector_rngs: Mapping[str, np.random.Generator],
    *,
    spec: Mapping[str, Any],
    decision_index_next: int,
    last_candidate: Mapping[str, Any] | None,
    pending_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": "retention_value_stream_checkpoint_v1",
        "stage": str(spec["stage"]),
        "pilot": bool(spec["pilot"]),
        "method": str(spec["method"]),
        "seed": int(spec["seed"]),
        "pair_kind": str(spec["pair_kind"]),
        "decision_index_next": int(decision_index_next),
        "learner": base._capture_learner(agent, foundation_sha),
        "retention_stats": stats.state_dict(),
        "value_models": {name: model.state_dict() for name, model in models.items()},
        "selector_rng_states": {name: copy.deepcopy(rng.bit_generator.state) for name, rng in selector_rngs.items()},
        "parent_metadata": {
            "binding": int(agent.parent_group),
            "version": int(agent.parent_version),
            "source_fit_steps": float(stats.parent_source_fit),
            "written_decision": int(stats.parent_written_decision),
        },
        "last_candidate_provenance": copy.deepcopy(dict(last_candidate)) if last_candidate else None,
        "pending_feedback": copy.deepcopy(dict(pending_feedback)) if pending_feedback is not None else None,
        "candidate_coefficients_persisted": False,
        "simulator_state_persisted": False,
    }


def _serialize_package(payload: Mapping[str, Any], actor_data: bytes, foundation: Path, budget: int) -> tuple[bytes, dict[str, int]]:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    checkpoint = stream.getvalue()
    total = int(foundation.stat().st_size + len(checkpoint) + len(actor_data))
    sizes = {
        "foundation_bytes": int(foundation.stat().st_size),
        "checkpoint_bytes": int(len(checkpoint)),
        "actor_bytes": int(len(actor_data)),
        "full_persistent_bytes": total,
        "package_budget_bytes": int(budget),
    }
    if total > int(budget):
        raise ValueError(f"complete persistent package is {total} bytes, exceeding cap {budget}")
    return checkpoint, sizes


def _save_package(
    directory: Path,
    payload: Mapping[str, Any],
    actor_data: bytes,
    foundation: Path,
    budget: int,
) -> dict[str, int]:
    checkpoint, sizes = _serialize_package(payload, actor_data, foundation, budget)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoint.pt").write_bytes(checkpoint)
    (directory / "actor.bin").write_bytes(actor_data)
    return sizes


def _snapshot_sources(output: Path, config_path: Path) -> dict[str, Any]:
    manifest = base._snapshot_sources(output, config_path)
    extra = (
        Path("scripts/run_retention_value_stream.py"),
        Path("src/online_rl/retention_value.py"),
        Path("src/online_rl/retention_stream_value.py"),
        Path("docs/RETENTION_VALUE_STREAM_PROTOCOL.md"),
    )
    snapshot = output / "source_snapshot"
    for relative in extra:
        if any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts):
            raise ValueError("private or temporary material is excluded from source snapshots")
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = _sha256_file(target)
        if _sha256_file(source) != digest:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        manifest["sources"][relative.as_posix()] = {"sha256": digest, "bytes": target.stat().st_size}
    _write_json(snapshot / "manifest.json", manifest)
    return manifest


def _selector_rngs(spec: Mapping[str, Any]) -> dict[str, np.random.Generator]:
    pair_offset = 37 if spec["pair_kind"] == "reward_conflict" else 11
    shared_seed = int(spec["seed"]) * 104729 + pair_offset
    method = str(spec["method"])
    if method in ("learned_linear", "learned_intercept"):
        common = shared_seed + 9001
        return {
            "exploration": np.random.default_rng(common),
            "exploration_action": np.random.default_rng(common + 1),
            "uniform": np.random.default_rng(common + 2),
        }
    salt = {"random": 5001, "calibration": 5001}.get(method, 7001)
    return {
        "exploration": np.random.default_rng(shared_seed + salt),
        "exploration_action": np.random.default_rng(shared_seed + salt + 1),
        "uniform": np.random.default_rng(shared_seed + salt + 2),
    }


def _rng_state(rngs: Mapping[str, np.random.Generator]) -> dict[str, Any]:
    return {name: copy.deepcopy(rng.bit_generator.state) for name, rng in rngs.items()}


def _greedy_action(predictions: Mapping[str, float], default_action: str = "keep") -> str:
    maximum = max(float(value) for value in predictions.values())
    tied = [action for action in ACTIONS if float(predictions[action]) == maximum]
    return default_action if default_action in tied else tied[0]


def _select_action(
    spec: Mapping[str, Any],
    decision_index: int,
    features: np.ndarray,
    models: Mapping[str, RetentionValueModel],
    rngs: Mapping[str, np.random.Generator],
    stats: BoundedRetentionStats,
    evidence: Any,
    score_fn,
    candidate: np.ndarray,
    candidate_group: int,
    incumbent_binding: int,
    incumbent_matrix: np.ndarray,
) -> dict[str, Any]:
    predictions = {
        "linear": models["linear"].predict(features),
        "intercept": models["intercept"].predict(np.asarray([1.0], dtype=np.float64)),
    }
    method = str(spec["method"])
    result: dict[str, Any] = {
        "predictions": predictions,
        "selected_action": None,
        "action_propensity": None,
        "selection_rule": "no_discrete_action",
        "explored": False,
        "lower_diagnostics": None,
        "model_name": None,
    }
    if method == "shared_ppo" or method == "independent":
        return result

    if method == "calibration" or method == "random":
        action = ACTIONS[int(rngs["uniform"].integers(len(ACTIONS)))]
        return {
            **result,
            "selected_action": action,
            "action_propensity": 0.25,
            "selection_rule": "uniform_random",
            "explored": True,
            "model_name": "uniform_random",
        }
    if method in ("learned_linear", "learned_intercept"):
        # Draw both random values every decision. This keeps the two learned
        # methods on a matched exploration stream even when their greedy
        # actions or exploration branches differ.
        explore_draw = float(rngs["exploration"].random())
        random_action = ACTIONS[int(rngs["exploration_action"].integers(len(ACTIONS)))]
        uniform_warmup = int(decision_index) < 16
        epsilon = 1.0 if uniform_warmup else 0.1
        model_name = "linear" if method == "learned_linear" else "intercept"
        model_features = features if model_name == "linear" else np.asarray([1.0], dtype=np.float64)
        greedy = _greedy_action(models[model_name].predict(model_features))
        explored = uniform_warmup or explore_draw < epsilon
        action = random_action if explored else greedy
        propensity = 0.25 if uniform_warmup else (0.025 + (0.9 if action == greedy else 0.0))
        return {
            **result,
            "selected_action": action,
            "action_propensity": float(propensity),
            "selection_rule": "uniform_warmup" if uniform_warmup else "epsilon_greedy_matched",
            "explored": bool(explored),
            "model_name": model_name,
            "greedy_action": greedy,
            "epsilon": epsilon,
        }
    if method in ("keep", "replace", "mean"):
        return {
            **result,
            "selected_action": method,
            "action_propensity": 1.0,
            "selection_rule": f"fixed_{method}",
            "model_name": f"fixed_{method}",
        }
    if method == "frequency":
        action = frequency_action(stats, candidate_group, incumbent_binding)
        return {
            **result,
            "selected_action": action,
            "action_propensity": 1.0,
            "selection_rule": "recent_group_frequency",
            "model_name": "frequency",
        }
    if method == "lower":
        action, diagnostics = _lower_action(
            evidence, score_fn, stats, incumbent_binding,
            incumbent_matrix, candidate, candidate_group,
        )
        return {
            **result,
            "selected_action": action,
            "action_propensity": 1.0,
            "selection_rule": "generalized_groupwise_lower",
            "model_name": "lower",
            "lower_diagnostics": diagnostics,
        }
    raise ValueError(f"unsupported stream method: {method}")


def _lower_action(
    evidence: Any, score_fn, stats: BoundedRetentionStats, parent_binding: int,
    incumbent: np.ndarray, candidate: np.ndarray, candidate_group: int,
) -> tuple[str, dict[str, Any]]:
    return generalized_lower_decision(
        evidence, score_fn, stats, incumbent, parent_binding, candidate, candidate_group,
    )


def _observe_bouts(
    agent: Any,
    records: Sequence[Mapping[str, Any]],
    stats: BoundedRetentionStats,
    mapping: Mapping[str, Any],
    log_path: Path,
    *,
    phase: str,
    decision_index: int | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        stats_info = stats.recurrence_info(record)
        stats.observe_bout(record)
        slot = int(record["slot"])
        fit_after = int(agent.fitted_steps[slot])
        fit_before = 0 if bool(record["allocated"]) else fit_after
        if not bool(record["allocated"]):
            fit_before = max(0, fit_after - int(record["steps"]))
        event = {
            "phase": str(phase),
            "decision_index": None if decision_index is None else int(decision_index),
            "bout_index": int(record["bout"]),
            "global_bout_index": int(record["bout"]),
            "task": str(record["task"]),
            "group": int(record["group"]),
            "variant": _task_variant(mapping, str(record["task"]), int(record["group"])),
            "steps_completed": int(record["steps"]),
            "return_first_256": float(record["return_first_256"]),
            "return_first_512": float(record["return_first_512"]),
            "return_full": float(record["return_full"]),
            "allocated": bool(record["allocated"]),
            "evicted_task": record["evicted_task"],
            "slot": slot,
            "generation": int(record["generation"]),
            "physical_group_at_start": int(record["physical_group_at_start"]),
            "parent_group_at_start": int(record["parent_group_at_start"]),
            "fitted_steps_before": int(fit_before),
            "fitted_steps_after": fit_after,
            "recurrence": stats_info,
        }
        _append_jsonl(log_path, event)
        result.append(event)
    return result


def _fitted_before_schedule(agent: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(agent.slot_generation, dtype=np.int64).copy(), np.asarray(agent.fitted_steps, dtype=np.uint64).copy()


def _candidate_from_records(
    agent: Any,
    records: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    generation_before: np.ndarray,
    fitted_before: np.ndarray,
    minimum_fit: int,
):
    candidate, group, source_fit, provenance = _current_candidate(agent, records, mapping, minimum_fit)
    for source in provenance["sources"]:
        slot = int(source["slot"])
        if int(generation_before[slot]) != int(source["generation"]):
            before = 0
        else:
            before = int(fitted_before[slot])
        if int(source["fitted_steps"]) - before < int(minimum_fit):
            raise ValueError(
                f"candidate source {source['task']!r} fitted fewer than {minimum_fit} steps in its current generation"
            )
        source["fitted_steps_before_source_pair"] = before
        source["fitted_steps_added_in_source_pair"] = int(source["fitted_steps"]) - before
    return candidate, group, source_fit, provenance


def _parent_action(
    agent: Any,
    stats: BoundedRetentionStats,
    candidate: np.ndarray,
    candidate_group: int,
    candidate_source_fit: float,
    action: str,
    decision_index: int,
) -> dict[str, Any]:
    incumbent_binding = int(agent.parent_group)
    local, shared, physical = ws._actor_arrays(agent.core)
    old_parent = np.asarray(shared[0], dtype=np.float32).reshape(3, 3)
    local_new, shared_new, physical_new, details = apply_retention_action(
        local, shared, physical, agent.slot_group_tags,
        incumbent_binding, candidate, candidate_group, action,
    )
    if action != "keep":
        _set_actor(agent, local_new, shared_new, physical_new)
        agent.parent_group = int(details["parent_group"])
    if action == "keep":
        new_source_fit = float(stats.parent_source_fit)
    elif action == "replace":
        new_source_fit = float(candidate_source_fit)
    elif action == "mean":
        new_source_fit = (float(stats.parent_source_fit) + float(candidate_source_fit)) / 2.0
    else:
        new_source_fit = 0.0
    if action != "keep":
        stats.record_parent_write(new_source_fit, int(decision_index))
        agent.parent_version += 1
        agent.parent_source_fit = new_source_fit
        actor_data, codec = _codec_roundtrip(agent)
        codec["roundtrip_applied"] = True
        ws._clear_optimizer_state(agent)
    else:
        # `keep` must not quantize/install a second time: the live actor was
        # already codec-rounded after its previous PPO rollout.
        actor_data = encode_actor(local, shared, physical, codec="int8", budget_bytes=89).data
        codec = {
            "actor_bytes": len(actor_data),
            "max_coefficient_error": 0.0,
            "per_slot_effective_error": [0.0 for _ in range(len(local))],
            "max_effective_error": 0.0,
            "roundtrip_applied": False,
        }
    details = {
        **details,
        "old_parent_sha256": _sha256_bytes(np.ascontiguousarray(old_parent).tobytes()),
        "new_parent_sha256": _sha256_bytes(np.ascontiguousarray(ws._actor_arrays(agent.core)[1][0]).tobytes()),
        "new_parent_source_fit_steps": new_source_fit,
        "codec": codec,
    }
    return {"details": details, "actor_data": actor_data}


def _models(config: Mapping[str, Any]) -> dict[str, RetentionValueModel]:
    model_config = config["value_model"]
    return {
        "linear": RetentionValueModel(
            6, ridge=float(model_config["ridge"]), discount=float(model_config["discount"]),
            lambda_cpu=float(model_config["lambda_cpu"]),
        ),
        "intercept": RetentionValueModel(
            1, ridge=float(model_config["ridge"]), discount=float(model_config["discount"]),
            lambda_cpu=float(model_config["lambda_cpu"]),
        ),
    }


def _model_state_bytes(models: Mapping[str, RetentionValueModel], stats: BoundedRetentionStats) -> int:
    return int(sum(model.numeric_nbytes() for model in models.values()) + stats.numeric_nbytes())


def _score_function(agent: Any, config: Mapping[str, Any]):
    return base._score_fn(agent.core, config)


def _worker_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(config),
        "shared_learning_rate": float(config["shared_learning_rate"]),
        "package_budget_bytes": int(config["package_budget_bytes"]),
    }


def _run_one(spec: Mapping[str, Any], config: Mapping[str, Any], config_path: Path, output_root: Path) -> dict[str, Any]:
    stream_id = f"{spec['stage']}__{spec['pair_kind']}__seed_{spec['seed']}__{spec['method']}"
    output = output_root / "streams" / str(spec["pair_kind"]) / f"seed_{int(spec['seed'])}" / str(spec["method"])
    summary_path = output / "summary.json"
    preactions_path = output / "preactions.jsonl"
    feedback_path = output / "feedback.jsonl"
    bouts_path = output / "bouts.jsonl"
    package_events_path = output / "package_events.jsonl"
    rawsteps_path = output / "rawsteps.csv"
    rollouts_path = output / "rollouts.csv"
    result: dict[str, Any] = {
        "stream_id": stream_id,
        "stage": str(spec["stage"]),
        "pilot": bool(spec["pilot"]),
        "method": str(spec["method"]),
        "seed": int(spec["seed"]),
        "pair_kind": str(spec["pair_kind"]),
        "status": "failed",
        "requested_decisions": int(spec["decisions"]),
        "completed_decisions": 0,
        "summary_path": summary_path.relative_to(output_root).as_posix(),
        "preactions_path": preactions_path.relative_to(output_root).as_posix(),
        "feedback_path": feedback_path.relative_to(output_root).as_posix(),
        "bouts_path": bouts_path.relative_to(output_root).as_posix(),
        "package_events_path": package_events_path.relative_to(output_root).as_posix(),
        "rawsteps_path": rawsteps_path.relative_to(output_root).as_posix(),
        "rollouts_path": rollouts_path.relative_to(output_root).as_posix(),
        "outcome_return_sum": 0.0,
        "warmup_return_sum": 0.0,
        "total_reward_sum": 0.0,
        "recurrence": {},
        "storage_action_counts": {action: 0 for action in ACTIONS},
        "costs": {
            "environment_steps": 0,
            "warmup_steps": 0,
            "outcome_steps": 0,
            "ppo_rollouts": 0,
            "minibatch_updates": 0,
            "storage_action_count": 0,
            "parent_write_count": 0,
            "eviction_count": 0,
            "slot_allocation_count": 0,
            "value_model_updates": 0,
            "utility_cpu_s": 0.0,
            "model_update_cpu_s": 0.0,
            "package_serialization_cpu_s": 0.0,
            "total_cpu_s": 0.0,
            "wall_s": 0.0,
        },
        "peak_package_bytes": 0,
        "actor_container_bytes": 89,
        "actual_actor_bytes": 0,
        "model_state_bytes": 0,
        "value_models_trained": str(spec["method"]) not in ("shared_ppo", "independent"),
        "failure": None,
    }
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing retention stream output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    agent = env = None
    models: dict[str, RetentionValueModel] = {}
    stats: BoundedRetentionStats | None = None
    actor_data = b""
    event_rows: list[dict[str, Any]] = []
    package_sizes: list[int] = []
    foundation = None
    try:
        manifest = _snapshot_sources(output, config_path)
        base_config = loop.load_base_config()
        foundation = Path(base_config["foundation"]).resolve()
        foundation_sha = _sha256_file(foundation)
        mapping = pair_mapping(int(spec["seed"]), str(spec["pair_kind"]))
        tasks, task_groups = _task_groups(mapping)
        stats = BoundedRetentionStats(tasks, task_groups)
        models = _models(config)
        selector_rngs = _selector_rngs(spec)
        agent = _new_agent(int(spec["seed"]), base_config, _worker_config(config), str(spec["method"]))
        env, runtime = base._runtime_initial(int(spec["seed"]), task_name(mapping, 0, 0))
        initial_local, initial_shared, initial_groups = ws._actor_arrays(agent.core)
        actor_data = encode_actor(initial_local, initial_shared, initial_groups, codec="int8", budget_bytes=89).data
        score_fn = _score_function(agent, {**config, "gate": config["gate"]})
        _write_json(output / "run_config.json", {
            "stream": dict(spec),
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "foundation_sha256": foundation_sha,
            "source_manifest": "source_snapshot/manifest.json",
            "source_count": len(manifest["sources"]),
            "future_schedule_is_external": True,
            "candidate_history_capacity": 0,
        })

        # The first pair is fitted while the parent is unbound. It creates the
        # initial group-0 parent except in the independent control.
        gen0, fit0 = _fitted_before_schedule(agent)
        group0_records = base._run_schedule(
            agent, env, runtime, _warmup_schedule(mapping, group=0, steps=1024, bout_start=0),
            folder=output, phase="warmup", action="initial_parent_source",
        )
        event_rows.extend(_observe_bouts(agent, group0_records, stats, mapping, bouts_path, phase="warmup", decision_index=None))
        group0_candidate, group0, group0_fit, group0_provenance = _candidate_from_records(
            agent, group0_records, mapping, gen0, fit0, int(config["minimum_source_fitted_steps"]),
        )
        if str(spec["method"]) != "independent":
            initial_details = _install_initial_parent(agent, group0_candidate, group0, group0_fit, stats, 0)
            result["initial_parent"] = {
                "binding": int(agent.parent_group), "source_fit_steps": group0_fit,
                "candidate_provenance": group0_provenance,
                "install": initial_details,
            }
        if str(spec["method"]) == "shared_ppo":
            direct_details = _make_direct_shared(agent, stats, 0)
            result["direct_shared_setup"] = direct_details

        # Candidate sources are the next two warmup bouts, trained at 512 steps
        # each after the group-0 parent has been formed.
        gen1, fit1 = _fitted_before_schedule(agent)
        group1_records = base._run_schedule(
            agent, env, runtime, _warmup_schedule(mapping, group=1, steps=512, bout_start=2),
            folder=output, phase="warmup", action="initial_candidate_source",
        )
        event_rows.extend(_observe_bouts(agent, group1_records, stats, mapping, bouts_path, phase="warmup", decision_index=None))
        source_records: list[Mapping[str, Any]] = group1_records[-2:]
        source_gen = np.asarray(agent.slot_generation, dtype=np.int64).copy()
        source_fit_before = np.asarray(agent.fitted_steps, dtype=np.uint64).copy()
        for record in source_records:
            slot = int(record["slot"])
            if bool(record["allocated"]) or int(gen1[slot]) != int(record["generation"]):
                source_gen[slot] = int(record["generation"])
                source_fit_before[slot] = 0
            else:
                source_gen[slot] = int(gen1[slot])
                source_fit_before[slot] = int(fit1[slot])
        # Ensure measured source growth is based on the current generation.
        _candidate_from_records(
            agent, source_records, mapping, source_gen, source_fit_before,
            int(config["minimum_source_fitted_steps"]),
        )

        # Persist the first deployable checkpoint after all 3072 real warmup steps.
        actor_data, warmup_codec = _codec_roundtrip(agent)
        result["warmup_codec"] = warmup_codec
        initial_payload = _capture_checkpoint(
            agent, foundation_sha, models, stats, selector_rngs,
            spec=spec, decision_index_next=0, last_candidate=None,
        )
        package_started = time.process_time()
        initial_checkpoint, initial_sizes = _serialize_package(
            initial_payload, actor_data, foundation, int(config["package_budget_bytes"]),
        )
        result["costs"]["package_serialization_cpu_s"] += time.process_time() - package_started
        initial_dir = output / "checkpoints" / "initial"
        initial_dir.mkdir(parents=True, exist_ok=True)
        (initial_dir / "checkpoint.pt").write_bytes(initial_checkpoint)
        (initial_dir / "actor.bin").write_bytes(actor_data)
        package_sizes.append(int(initial_sizes["full_persistent_bytes"]))
        result["initial_package"] = initial_sizes

        for decision_index in range(int(spec["decisions"])):
            decision_started_cpu = time.process_time()
            candidate, candidate_group, candidate_fit, provenance = _candidate_from_records(
                agent, source_records, mapping, source_gen, source_fit_before,
                int(config["minimum_source_fitted_steps"]),
            )
            incumbent_binding = int(agent.parent_group)
            features, feature_names = stats.features(
                candidate_group, incumbent_binding, candidate_fit, decision_index,
            )
            rng_before = _rng_state(selector_rngs)
            model_hashes = {
                name: semantic_hash(model.state_dict()) for name, model in models.items()
            }
            current_shared = ws._actor_arrays(agent.core)[1]
            incumbent_matrix = np.asarray(current_shared[0], dtype=np.float32).reshape(3, 3)
            selected = _select_action(
                spec, decision_index, features, models, selector_rngs, stats,
                agent.evidence, score_fn, candidate, candidate_group, incumbent_binding,
                incumbent_matrix,
            )
            lower_diagnostics = selected.get("lower_diagnostics")
            selection_cpu = time.process_time() - decision_started_cpu
            # This flushed record is the only pre-action selector input saved.
            preaction = {
                "stream_id": stream_id,
                "decision_index": int(decision_index),
                "model_name": selected["model_name"],
                "features": features.tolist(),
                "feature_names": list(feature_names),
                "predictions": selected["predictions"],
                "selected_action": selected["selected_action"],
                "action_propensity": selected["action_propensity"],
                "selection_rule": selected["selection_rule"],
                "explored": bool(selected["explored"]),
                "greedy_action": selected.get("greedy_action"),
                "epsilon": selected.get("epsilon"),
                "rng_state_before": rng_before,
                "model_hashes": model_hashes,
                "candidate_provenance": provenance,
                "incumbent_binding": incumbent_binding,
                "incumbent_parent_version": int(agent.parent_version),
                "incumbent_parent_source_fit_steps": float(stats.parent_source_fit),
                "incumbent_parent_sha256": _sha256_bytes(np.ascontiguousarray(current_shared[0]).tobytes()),
                "candidate_group": int(candidate_group),
                "source_fit_steps": float(candidate_fit),
                "package_bytes_before": int(package_sizes[-1]),
                "pre_action_cpu_s": float(selection_cpu),
                "lower_diagnostics": lower_diagnostics,
            }
            _append_jsonl(preactions_path, preaction)

            action_started_cpu = time.process_time()
            action_details = None
            if selected["selected_action"] is not None:
                action_result = _parent_action(
                    agent, stats, candidate, candidate_group, candidate_fit,
                    str(selected["selected_action"]), decision_index,
                )
                action_details = action_result["details"]
                actor_data = action_result["actor_data"]
                action_name = str(selected["selected_action"])
                result["storage_action_counts"][action_name] += 1
                result["costs"]["storage_action_count"] += 1
                if action_name != "keep":
                    result["costs"]["parent_write_count"] += 1
            else:
                action_name = None
            action_install_cpu = time.process_time() - action_started_cpu
            # Candidate coefficients are deliberately released before the
            # three-bout outcome horizon begins.
            del candidate

            pending_feedback = None
            if selected["selected_action"] is not None:
                pending_feedback = {
                    "decision_index": int(decision_index),
                    "selected_action": str(selected["selected_action"]),
                    "features": [float(value) for value in features],
                    "propensity": float(selected["action_propensity"]),
                    "start_step": int(agent.runtime_counts["environment_steps"]),
                }
            inflight_started = time.process_time()
            inflight_payload = _capture_checkpoint(
                agent, foundation_sha, models, stats, selector_rngs,
                spec=spec, decision_index_next=decision_index,
                last_candidate=provenance, pending_feedback=pending_feedback,
            )
            _inflight_blob, inflight_package = _serialize_package(
                inflight_payload, actor_data, foundation, int(config["package_budget_bytes"]),
            )
            inflight_cpu = time.process_time() - inflight_started
            result["costs"]["package_serialization_cpu_s"] += inflight_cpu
            package_sizes.append(int(inflight_package["full_persistent_bytes"]))
            _append_jsonl(package_events_path, {
                "stream_id": stream_id,
                "phase": "inflight",
                "decision_index": int(decision_index),
                "pending_feedback": pending_feedback,
                "package_bytes": int(inflight_package["full_persistent_bytes"]),
                "package_budget_bytes": int(inflight_package["package_budget_bytes"]),
                "serialization_cpu_s": float(inflight_cpu),
                "simulator_state_persisted": False,
                "exact_environment_resume_supported": False,
            })

            # The block schedule is generated only after selection and is never
            # passed to a value model, score function, or action selector.
            schedule = _outcome_schedule(
                mapping, int(spec["seed"]), decision_index,
                bout_start=WARMUP_BOUTS + decision_index * OUTCOME_BOUTS,
                regime_blocks=int(config["regime_blocks"]), bout_steps=int(config["bout_steps"]),
            )
            block_started = time.process_time()
            records = base._run_schedule(
                agent, env, runtime, schedule, folder=output,
                phase="outcome", action=action_name, future_regime=None,
            )
            bout_events = _observe_bouts(
                agent, records, stats, mapping, bouts_path,
                phase="outcome", decision_index=decision_index,
            )
            event_rows.extend(bout_events)
            block_cpu = time.process_time() - block_started
            utility_cpu = float(selection_cpu + action_install_cpu + block_cpu)
            block_return = float(sum(float(record["return_full"]) for record in records))
            block_steps = int(sum(int(record["steps"]) for record in records))
            if block_steps != OUTCOME_STEPS:
                raise RuntimeError(f"decision block has {block_steps} steps; expected {OUTCOME_STEPS}")
            reward_rate = block_return / float(block_steps)
            eligible_update = selected["selected_action"] is not None
            utility_target = None
            model_update_cpu = 0.0
            if eligible_update:
                utility_target = reward_rate - utility_cpu / float(block_steps)
                update_started = time.process_time()
                for model_name, model in models.items():
                    model_features = features if model_name == "linear" else np.asarray([1.0], dtype=np.float64)
                    model.update(
                        str(selected["selected_action"]), model_features,
                        reward_rate, utility_cpu, block_steps,
                    )
                model_update_cpu = time.process_time() - update_started
                result["costs"]["value_model_updates"] += 1
            result["costs"]["utility_cpu_s"] += utility_cpu
            result["costs"]["model_update_cpu_s"] += model_update_cpu
            result["outcome_return_sum"] += block_return

            # Every epoch is serialized in memory to measure the exact complete
            # package; only initial/every eighth/final snapshots are persisted.
            live_local, live_shared, live_groups = ws._actor_arrays(agent.core)
            actor_data = encode_actor(
                live_local, live_shared, live_groups,
                codec="int8", budget_bytes=int(config["actor_budget_bytes"]),
            ).data
            package_started = time.process_time()
            payload = _capture_checkpoint(
                agent, foundation_sha, models, stats, selector_rngs,
                spec=spec, decision_index_next=decision_index + 1,
                last_candidate=provenance,
            )
            checkpoint_blob, package = _serialize_package(
                payload, actor_data, foundation, int(config["package_budget_bytes"]),
            )
            package_cpu = time.process_time() - package_started
            result["costs"]["package_serialization_cpu_s"] += package_cpu
            package_sizes.append(int(package["full_persistent_bytes"]))
            if (decision_index + 1) % int(config["checkpoint_every"]) == 0 or decision_index + 1 == int(spec["decisions"]):
                checkpoint_dir = output / "checkpoints" / f"decision_{decision_index + 1:03d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (checkpoint_dir / "checkpoint.pt").write_bytes(checkpoint_blob)
                (checkpoint_dir / "actor.bin").write_bytes(actor_data)

            feedback = {
                "stream_id": stream_id,
                "decision_index": int(decision_index),
                "selected_action": selected["selected_action"],
                "block_return_sum": block_return,
                "reward_rate": reward_rate,
                "block_steps": block_steps,
                "utility_cpu_s": utility_cpu,
                "pre_action_cpu_s": float(selection_cpu),
                "action_install_cpu_s": float(action_install_cpu),
                "block_cpu_s": float(block_cpu),
                "model_update_cpu_s": float(model_update_cpu),
                "inflight_package_serialization_cpu_s": float(inflight_cpu),
                "utility_target": utility_target,
                "model_updated": bool(eligible_update),
                "target_excludes_model_update": True,
                "post_action_binding": int(agent.parent_group),
                "parent_write_details": action_details,
                "package_serialization_cpu_s": float(package_cpu),
                "package_bytes_inflight": int(inflight_package["full_persistent_bytes"]),
                "pending_feedback_cleared": True,
                "package_bytes_after": int(package["full_persistent_bytes"]),
                "model_state_bytes": _model_state_bytes(models, stats),
                "bout_returns": [float(record["return_full"]) for record in records],
                "recurrence_events": [event["recurrence"] for event in bout_events],
                "outcome_bout_indices": [int(event["bout_index"]) for event in bout_events],
            }
            _append_jsonl(feedback_path, feedback)

            result["completed_decisions"] += 1
            result["costs"]["environment_steps"] += block_steps
            result["costs"]["outcome_steps"] += block_steps
            result["costs"]["minibatch_updates"] = int(agent.runtime_counts["minibatch_updates"])
            result["costs"]["ppo_rollouts"] = int(agent.runtime_counts["rollouts_trained"])
            result["costs"]["eviction_count"] = int(agent.runtime_counts["slot_evictions"])
            result["costs"]["slot_allocation_count"] = int(agent.runtime_counts["slot_allocations"])
            source_records = records[-2:]
            source_gen, source_fit_before = _fitted_before_schedule(agent)
            for record in source_records:
                slot = int(record["slot"])
                source_gen[slot] = int(record["generation"])
                source_fit_before[slot] = 0 if bool(record["allocated"]) else max(
                    0, int(agent.fitted_steps[slot]) - int(record["steps"]),
                )
            result["actual_actor_bytes"] = len(actor_data)
            result["model_state_bytes"] = _model_state_bytes(models, stats)

        result["status"] = "completed"
    except BaseException as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        steps_path = output / "steps.csv"
        if steps_path.is_file():
            if rawsteps_path.exists():
                result["status"] = "failed"
                result["failure"] = result.get("failure") or {
                    "type": "FileExistsError",
                    "message": "rawsteps.csv already exists; preserving both files",
                    "traceback": "",
                }
            else:
                steps_path.replace(rawsteps_path)
        raw_summary = _rawstep_summary(rawsteps_path)
        result["costs"]["environment_steps"] = int(raw_summary["step_rows"])
        result["costs"]["warmup_steps"] = min(WARMUP_STEPS, int(result["costs"]["environment_steps"]))
        result["costs"]["outcome_steps"] = max(0, int(result["costs"]["environment_steps"]) - int(result["costs"]["warmup_steps"]))
        if agent is not None:
            result["costs"]["ppo_rollouts"] = int(agent.runtime_counts["rollouts_trained"])
            result["costs"]["minibatch_updates"] = int(agent.runtime_counts["minibatch_updates"])
            result["costs"]["eviction_count"] = int(agent.runtime_counts["slot_evictions"])
            result["costs"]["slot_allocation_count"] = int(agent.runtime_counts["slot_allocations"])
            result["actual_actor_bytes"] = len(actor_data)
        result["total_reward_sum"] = float(raw_summary["total_reward_sum"])
        result["warmup_return_sum"] = float(raw_summary["warmup_reward_sum"])
        result["outcome_return_sum"] = float(raw_summary["outcome_reward_sum"])
        result["rawstep_partial_bouts"] = raw_summary["partial_bouts"]
        result["recurrence"] = _recurrence_summary(event_rows)
        result["peak_package_bytes"] = max(package_sizes, default=0)
        result["model_state_bytes"] = _model_state_bytes(models, stats) if stats is not None and models else 0
        result["costs"]["total_cpu_s"] = max(0.0, time.process_time() - started_cpu)
        result["costs"]["wall_s"] = max(0.0, time.perf_counter() - started_wall)
        result["summary_path"] = summary_path.relative_to(output_root).as_posix()
        try:
            _write_json(summary_path, result)
        except Exception as write_exc:
            result["status"] = "failed"
            result["failure"] = result.get("failure") or {
                "type": type(write_exc).__name__, "message": str(write_exc), "traceback": traceback.format_exc(),
            }
    return result


def _recurrence_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = [row for row in events if row.get("phase") == "outcome"]
    recurrent = [row for row in outcomes if bool(row["recurrence"].get("recurrence"))]
    evicted = [row for row in outcomes if bool(row["recurrence"].get("natural_eviction"))]
    return {
        "outcome_bouts": len(outcomes),
        "recurred_after_eviction_count": len(recurrent),
        "recurred_after_eviction_rate": float(len(recurrent) / len(outcomes)) if outcomes else None,
        "natural_eviction_count": len(evicted),
        "recurrent_first256_reward_per_step_mean": (
            float(np.mean([row["recurrence"]["first256_reward_per_step"] for row in recurrent])) if recurrent else None
        ),
        "recurrent_first512_return_mean": float(np.mean([row["return_first_512"] for row in recurrent])) if recurrent else None,
        "recurrent_full_return_mean": float(np.mean([row["return_full"] for row in recurrent])) if recurrent else None,
    }


def _rawstep_summary(path: Path) -> dict[str, Any]:
    """Recompute real rows and reward even when a bout failed partway through."""
    summary = {
        "step_rows": 0,
        "total_reward_sum": 0.0,
        "warmup_reward_sum": 0.0,
        "outcome_reward_sum": 0.0,
        "partial_bouts": [],
    }
    if not path.is_file():
        return summary
    bout_rows: dict[tuple[str, int], int] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            reward = float(row["reward"])
            phase = str(row.get("phase", ""))
            bout = int(row["bout"])
            summary["step_rows"] += 1
            summary["total_reward_sum"] += reward
            if phase == "warmup":
                summary["warmup_reward_sum"] += reward
            elif phase == "outcome":
                summary["outcome_reward_sum"] += reward
            bout_rows[(phase, bout)] = bout_rows.get((phase, bout), 0) + 1
    partial = []
    for (phase, bout), count in sorted(bout_rows.items(), key=lambda item: (item[0][0], item[0][1])):
        expected = 1024 if phase == "warmup" and bout < 2 else 512 if phase == "warmup" else 512
        if count != expected:
            partial.append({"phase": phase, "bout_index": bout, "observed_steps": count, "expected_steps": expected})
    summary["partial_bouts"] = partial
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _merge_stream_events(row: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output = output_root / Path(row["summary_path"]).parent
    bouts = _read_jsonl(output / "bouts.jsonl")
    raw = _rawstep_summary(output / "rawsteps.csv")
    row["total_reward_sum"] = float(raw["total_reward_sum"])
    row["warmup_return_sum"] = float(raw["warmup_reward_sum"])
    row["outcome_return_sum"] = float(raw["outcome_reward_sum"])
    row["rawstep_partial_bouts"] = raw["partial_bouts"]
    row["recurrence"] = _recurrence_summary(bouts)
    return row


def _cohort_costs(
    run_wall_s: float,
    parent_process_cpu_s: float,
    child_cpu_s: float,
    workers_actual: int,
    *,
    serial_execution: bool,
) -> dict[str, float | int]:
    """Separate stream CPU from orchestration CPU without serial double-counting."""
    run_wall_s = max(0.0, float(run_wall_s))
    parent_process_cpu_s = max(0.0, float(parent_process_cpu_s))
    child_cpu_s = max(0.0, float(child_cpu_s))
    workers_actual = max(0, int(workers_actual))
    if serial_execution:
        # With one worker, _run_one ran in this process, so process_time includes
        # the same CPU already charged to its stream row.
        orchestrator_cpu_s = max(0.0, parent_process_cpu_s - child_cpu_s)
    else:
        # Process CPU time excludes child processes when the pool is parallel.
        orchestrator_cpu_s = parent_process_cpu_s
    return {
        "wall_s": run_wall_s,
        "child_cpu_s": child_cpu_s,
        "orchestrator_cpu_s": orchestrator_cpu_s,
        "total_cpu_s": child_cpu_s + orchestrator_cpu_s,
        "workers_actual": workers_actual,
    }


def run_stage(stage: str, output: str | Path, *, pilot: bool = False, config_path: str | Path = CONFIG_PATH, workers: int | None = None) -> dict[str, Any]:
    stage_started_wall = time.perf_counter()
    stage_started_cpu = time.process_time()
    config_file = Path(config_path).resolve()
    config = _load_config(config_file)
    grid = _stream_grid(config, stage, pilot)
    target = Path(output).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"Preserve existing retention stream output: {target}")
    target.mkdir(parents=True, exist_ok=True)
    frozen_path = target / "frozen_config.json"
    config_sha256 = _sha256_file(config_file)
    protocol_sha256 = _sha256_file(PROTOCOL_PATH)
    _write_json(frozen_path, {
        "format": "retention_value_stream_run_config_v1",
        "stage": stage,
        "pilot": bool(pilot),
        "config": config,
        "config_path": str(config_file),
        "config_sha256": config_sha256,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha256,
        "requested_grid": grid,
        "sources_are_frozen_before_run": True,
        "stage_one_outcomes_read": False,
    })
    worker_count = int(config["max_workers"] if workers is None else workers)
    if not 1 <= worker_count <= int(config["max_workers"]):
        raise ValueError(f"workers must be from 1 through {config['max_workers']}")
    tasks = [(spec, config, config_file, target) for spec in grid]
    if worker_count == 1:
        rows = [_run_one(*task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_one, *task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: (row["pair_kind"], int(row["seed"]), METHODS.index(row["method"])))
    rows = [_merge_stream_events(row, target) for row in rows]
    total_cost_keys = (
        "environment_steps", "warmup_steps", "outcome_steps", "ppo_rollouts",
        "minibatch_updates", "storage_action_count", "parent_write_count",
        "eviction_count", "slot_allocation_count", "value_model_updates",
    )
    totals = {
        key: int(sum(int(row["costs"].get(key, 0)) for row in rows))
        for key in total_cost_keys
    }
    for key in ("utility_cpu_s", "model_update_cpu_s", "package_serialization_cpu_s", "total_cpu_s", "wall_s"):
        totals[key] = float(sum(float(row["costs"].get(key, 0.0)) for row in rows))
    totals["aggregate_stream_wall_s"] = totals["wall_s"]
    totals["total_reward_sum"] = float(sum(float(row.get("total_reward_sum", 0.0)) for row in rows))
    totals["outcome_return_sum"] = float(sum(float(row.get("outcome_return_sum", 0.0)) for row in rows))
    totals["peak_package_bytes"] = max((int(row.get("peak_package_bytes", 0)) for row in rows), default=0)
    workers_actual = min(worker_count, len(tasks))
    cohort_costs = _cohort_costs(
        run_wall_s=time.perf_counter() - stage_started_wall,
        parent_process_cpu_s=time.process_time() - stage_started_cpu,
        child_cpu_s=totals["total_cpu_s"],
        workers_actual=workers_actual,
        serial_execution=worker_count == 1,
    )
    summary = {
        "format": "retention_value_stream_summary_v1",
        "stage": stage,
        "pilot": bool(pilot),
        "config_path": str(config_file),
        "frozen_config_path": frozen_path.name,
        "protocol_sha256": protocol_sha256,
        "streams": rows,
        "totals": totals,
        "cohort_costs": cohort_costs,
        "failures": [row for row in rows if row["status"] != "completed"],
        "completed_stream_count": sum(row["status"] == "completed" for row in rows),
        "requested_stream_count": len(rows),
    }
    _write_json(target / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "comparison"), required=True)
    parser.add_argument("--pilot", action="store_true", help="run the frozen plumbing pilot grid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    summary = run_stage(args.stage, args.output, pilot=args.pilot, config_path=args.config, workers=args.workers)
    print(json.dumps({
        "stage": summary["stage"], "pilot": summary["pilot"],
        "requested_stream_count": summary["requested_stream_count"],
        "completed_stream_count": summary["completed_stream_count"],
        "failures": len(summary["failures"]), "totals": summary["totals"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

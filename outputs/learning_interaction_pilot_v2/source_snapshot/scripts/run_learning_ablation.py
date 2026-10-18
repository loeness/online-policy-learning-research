"""Compare saved-new-parent branches with and without future PPO updates.

The source-dose run supplies the already-accounted common prefixes and the
PPO-trained replace/none branches. This runner adds only the no-gradient
replace/none branches, restoring the original prefix checkpoint for each.
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
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

# `python scripts/run_learning_ablation.py` puts only `scripts/` on sys.path.
# Bootstrap the project package paths before importing third-party or project code.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from scripts import run_source_dose as source_dose
from scripts import run_storage_decision as base


ROOT = base.ROOT
PROTOCOL_PATH = ROOT / "docs" / "LEARNING_INTERACTION_PROTOCOL.md"
FORMAT = "learning_interaction_ablation_v1"
BRANCH_FORMAT = "learning_interaction_branch_v1"
ACTIONS = ("replace", "none")
CRITIC_PREFIXES = ("mlp_extractor.value_net.", "value_net.")
W0_PREFIXES = ("mlp_extractor.policy_net.", "action_net.")
W0_EXACT_KEYS = {"source_basis"}
STORAGE_STATE_KEYS = {"slot_groups", "seen_mask"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    base._write_json(path, value)


def _excluded_local_path(relative: Path) -> bool:
    return any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts)


def _source_files() -> tuple[Path, ...]:
    files = [
        *base.SOURCE_FILES,
        *source_dose.SOURCE_FILES,
        Path("scripts/run_learning_ablation.py"),
        Path("docs/LEARNING_INTERACTION_PROTOCOL.md"),
    ]
    return tuple(dict.fromkeys(files))


def _snapshot_sources(output: Path, source_root: Path) -> dict[str, Any]:
    snapshot = output / "source_snapshot"
    source_files: dict[str, dict[str, Any]] = {}
    for relative in _source_files():
        if _excluded_local_path(relative):
            raise ValueError(f"private or temporary material is excluded from source snapshots: {relative}")
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source snapshot file is missing: {relative}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        digest = _sha(source)
        if _sha(target) != digest:
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        source_files[relative.as_posix()] = {"sha256": digest, "bytes": target.stat().st_size}

    source_summary_path = source_root / "summary.json"
    source_config_path = source_root / "config.json"
    input_snapshot = snapshot / "input_source_dose"
    input_snapshot.mkdir(parents=True, exist_ok=True)
    copied_inputs: dict[str, dict[str, Any]] = {}
    for label, path in (("summary.json", source_summary_path), ("config.json", source_config_path)):
        target = input_snapshot / label
        target.write_bytes(path.read_bytes())
        digest = _sha(path)
        if _sha(target) != digest:
            raise RuntimeError(f"source input changed while snapshotting: {path}")
        copied_inputs[label] = {"sha256": digest, "bytes": target.stat().st_size}

    source_manifest_path = source_root / "source_snapshot" / "manifest.json"
    source_manifest_record = None
    if source_manifest_path.is_file():
        target = input_snapshot / "source_manifest.json"
        target.write_bytes(source_manifest_path.read_bytes())
        source_manifest_record = {"sha256": _sha(source_manifest_path), "bytes": target.stat().st_size}

    input_summary = base._load_json(source_summary_path)
    input_source_manifest = input_summary.get("source_manifest", {})
    manifest = {
        "format": "storage_learning_interaction_manifest_v1",
        "sources": source_files,
        "source_dose_input": {
            "path": str(source_root),
            "files": copied_inputs,
            "source_manifest": source_manifest_record,
        },
        "foundation": dict(input_source_manifest.get("foundation", {})),
        "dataset": dict(input_source_manifest.get("dataset", {})),
        "excluded_local_instructions": ["Agent.md", "AGENTS.md"],
    }
    _write_json(snapshot / "manifest.json", manifest)
    return manifest


def _load_source_run(source: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source_root = Path(source).resolve()
    summary_path = source_root / "summary.json"
    config_path = source_root / "config.json"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("source input must contain summary.json and config.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_file = json.loads(config_path.read_text(encoding="utf-8"))
    if summary.get("format") != source_dose.FORMAT:
        raise ValueError(f"source summary format must be {source_dose.FORMAT!r}")
    summary_config = summary.get("config")
    if not isinstance(summary_config, Mapping) or _json_digest(summary_config) != _json_digest(config_file):
        raise ValueError("source summary config does not match its saved config.json")
    config = base.validate_config(config_file)
    if tuple(config.get("actions", ())) != tuple(base.ACTIONS):
        raise ValueError("source config does not preserve the original four-action setup")
    if not summary.get("cases"):
        raise ValueError("source summary contains no cases")
    if summary.get("errors"):
        raise ValueError("source-dose input contains failed cases; no-gradient comparison needs complete source branches")
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"required frozen protocol is missing: {PROTOCOL_PATH}")
    _validate_source_cases(source_root, summary, config)
    return source_root, summary, config


def _validate_source_cases(source_root: Path, summary: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected_pairs = {(action, future) for action in ACTIONS for future in config["future_regimes"]}
    cases = summary.get("cases", [])
    seen_case_ids: set[str] = set()
    for record in cases:
        case_id = str(record.get("case_id", ""))
        if not case_id or case_id in seen_case_ids:
            raise ValueError("source cases must have unique nonempty case_id values")
        seen_case_ids.add(case_id)
        case_folder = source_root / "cases" / case_id
        for required in (case_folder / "prefix" / "checkpoint.pt", case_folder / "prefix" / "simulator.json",
                         case_folder / "decision.json", case_folder / "dose_1024" / "decision.json"):
            if not required.is_file():
                raise FileNotFoundError(f"source case {case_id} is incomplete: {required}")
        branches = record.get("base_branches", [])
        selected: dict[tuple[str, str], Mapping[str, Any]] = {}
        for branch in branches:
            key = (str(branch.get("action")), str(branch.get("future_regime")))
            if key not in expected_pairs:
                continue
            if key in selected:
                raise ValueError(f"source case {case_id} has duplicate PPO reference branch {key}")
            if branch.get("format") == "storage_source_dose_branch_alias_v1":
                raise ValueError("PPO reference must come from the original source-dose base branch, not an alias")
            selected[key] = branch
        if set(selected) != expected_pairs:
            raise ValueError(f"source case {case_id} lacks the four replace/none PPO reference branches")
        for branch in selected.values():
            summary_path = _source_branch_summary_path(case_folder, branch)
            if not summary_path.is_file():
                raise FileNotFoundError(f"source PPO branch summary is missing: {summary_path}")
        base_decision = source_dose.base._load_json(case_folder / "decision.json")
        dose_decision = source_dose.base._load_json(case_folder / "dose_1024" / "decision.json")
        if base_decision.get("prefix_state_hash") != dose_decision.get("prefix_state_hash"):
            raise ValueError(f"source case {case_id} prefix state hash differs between source-dose records")
        if not isinstance(dose_decision.get("new_candidate"), list):
            raise ValueError(f"source case {case_id} has no saved new candidate")
        expected_steps = sum(
            _future_step_count(dose_decision["mapping"], config, str(record["history_regime"]), str(future))
            for future in config["future_regimes"]
        )
        if expected_steps <= 0:
            raise ValueError(f"source case {case_id} has an empty future schedule")


def _future_step_count(
    mapping: Mapping[str, Any], config: Mapping[str, Any], history_regime: str, future_regime: str,
) -> int:
    schedule = base.template_schedule(
        mapping, future_regime, kind="future", steps=int(config["bout_steps"]),
        start_bout=len(base.prefix_schedule(mapping, config, history_regime)),
    )
    return sum(int(row["steps"]) for row in schedule)


def _source_branch_summary_path(case_folder: Path, branch: Mapping[str, Any]) -> Path:
    relative = branch.get("summary")
    if relative:
        return case_folder / Path(str(relative))
    return case_folder / "branches" / f"{branch['action']}__{branch['future_regime']}" / "summary.json"


def _no_gradient_update(agent: Any, rows: list[Any]) -> dict[str, float]:
    """Return the exact rollout schema without touching learner or RNG state."""
    del agent, rows
    return {
        "updates": 0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
    }


def _correct_no_gradient_metadata(
    *, agent: Any,
    phase: str,
    bout_record: Mapping[str, Any],
    slot: int,
    generation: int,
    bout_step: int,
    rollout_index: int,
    rollout_rows: int,
    update: Mapping[str, Any],
) -> None:
    """Undo schedule-side training labels and record that a rollout was collected."""
    del phase, bout_record, slot, generation, bout_step, rollout_index
    if int(update.get("updates", -1)) != 0:
        raise RuntimeError("no-gradient rollout unexpectedly reported an optimizer update")
    trained = int(agent.runtime_counts.get("rollouts_trained", 0))
    fit_steps = int(agent.counts.get("fit_steps", 0))
    if trained < 1 or fit_steps < int(rollout_rows):
        raise RuntimeError("cannot safely correct no-gradient rollout metadata")
    agent.runtime_counts["rollouts_trained"] = trained - 1
    agent.runtime_counts["rollouts_collected"] = int(agent.runtime_counts.get("rollouts_collected", 0)) + 1
    agent.counts["fit_steps"] = fit_steps - int(rollout_rows)


def _never_reuse_branch(*args: Any, **kwargs: Any) -> None:
    """No-gradient outcomes are always newly executed observations."""
    del args, kwargs
    return None


@contextmanager
def _no_gradient_execution(callback=_correct_no_gradient_metadata):
    """Install scoped, reversible patches for the four no-gradient branches."""
    original_new_agent = base._new_agent

    def instrumented_new_agent(*args: Any, **kwargs: Any):
        agent = original_new_agent(*args, **kwargs)
        agent._storage_decision_post_rollout_observer = callback
        return agent

    with ExitStack() as stack:
        stack.enter_context(patch.object(base, "ACTIONS", ACTIONS))
        stack.enter_context(patch.object(source_dose, "_matching_reusable_branch", _never_reuse_branch))
        stack.enter_context(patch.object(base.loop, "_ppo_update", _no_gradient_update))
        stack.enter_context(patch.object(base, "_new_agent", instrumented_new_agent))
        yield


def _critic_payload(learner: Mapping[str, Any]) -> dict[str, Any]:
    state = learner["core_state"]
    return {key: value for key, value in state.items() if str(key).startswith(CRITIC_PREFIXES)}


def _w0_payload(learner: Mapping[str, Any]) -> dict[str, Any]:
    state = learner["core_state"]
    return {
        key: value for key, value in state.items()
        if key in W0_EXACT_KEYS or str(key).startswith(W0_PREFIXES)
    }


def _changed_core_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    left, right = before["core_state"], after["core_state"]
    if set(left) != set(right):
        return sorted(set(left) ^ set(right))
    return sorted(key for key in left if base.semantic_hash(left[key]) != base.semantic_hash(right[key]))


def _actor_change_allowed(key: str) -> bool:
    return (
        key in STORAGE_STATE_KEYS
        or key.startswith("_shared_coefficients.")
        or (key.startswith("_nodes.") and key.endswith(".A"))
    )


def _fitted_steps_not_increased(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    before_steps = np.asarray(before["fitted_steps"], dtype=np.int64)
    after_steps = np.asarray(after["fitted_steps"], dtype=np.int64)
    before_generation = np.asarray(before["slot_generation"], dtype=np.int64)
    after_generation = np.asarray(after["slot_generation"], dtype=np.int64)
    if not (before_steps.shape == after_steps.shape == before_generation.shape == after_generation.shape):
        return False
    for old_steps, new_steps, old_generation, new_generation in zip(
        before_steps, after_steps, before_generation, after_generation,
    ):
        if new_generation < old_generation:
            return False
        if new_generation == old_generation and new_steps > old_steps:
            return False
        if new_generation > old_generation and new_steps != 0:
            return False
    return True


def _audit_branch(
    branch_folder: Path,
    *,
    foundation: Path,
    expected_foundation_sha: str,
    expected_environment_steps: int,
) -> dict[str, Any]:
    action_path = branch_folder / "post_action" / "checkpoint.pt"
    final_path = branch_folder / "checkpoint.pt"
    action = base._capture_learner_from_saved(action_path)
    final = base._capture_learner_from_saved(final_path)
    critic_unchanged = base.semantic_hash(_critic_payload(action)) == base.semantic_hash(_critic_payload(final))
    w0_unchanged = base.semantic_hash(_w0_payload(action)) == base.semantic_hash(_w0_payload(final))
    optimizer_before = {
        "state": action["optimizer_state"],
        "step_count": action["optimizer_step_count"],
    }
    optimizer_after = {
        "state": final["optimizer_state"],
        "step_count": final["optimizer_step_count"],
    }
    optimizer_unchanged = base.semantic_hash(optimizer_before) == base.semantic_hash(optimizer_after)
    changed_keys = _changed_core_keys(action, final)
    actor_changes_allowed = all(_actor_change_allowed(str(key)) for key in changed_keys)
    runtime_before = action["runtime_counts"]
    runtime_after = final["runtime_counts"]
    counts_before = action["counts"]
    counts_after = final["counts"]
    fitted_steps_start = np.asarray(action["fitted_steps"], dtype=np.uint64)
    fitted_steps_end = np.asarray(final["fitted_steps"], dtype=np.uint64)
    slot_generation_start = np.asarray(action["slot_generation"], dtype=np.int64)
    slot_generation_end = np.asarray(final["slot_generation"], dtype=np.int64)
    runtime_delta = {
        key: int(runtime_after.get(key, 0)) - int(runtime_before.get(key, 0))
        for key in set(runtime_before) | set(runtime_after)
    }
    expected_rollouts = expected_environment_steps // int(base.loop.ROLLOUT_STEPS)
    environment_steps_delta = runtime_delta.get("environment_steps", 0)
    fit_steps_delta = int(counts_after.get("fit_steps", 0)) - int(counts_before.get("fit_steps", 0))
    foundation_after_sha = base._sha(foundation)
    audit = {
        "critic_unchanged": critic_unchanged,
        "w0_core_unchanged": w0_unchanged,
        "w0_unchanged": w0_unchanged,
        "foundation_unchanged": foundation_after_sha == expected_foundation_sha,
        "foundation_sha256_before": expected_foundation_sha,
        "foundation_sha256_after": foundation_after_sha,
        "optimizer_state_unchanged": optimizer_unchanged,
        "optimizer_step_count_delta": int(final["optimizer_step_count"]) - int(action["optimizer_step_count"]),
        "minibatch_updates": runtime_delta.get("minibatch_updates", 0),
        "rollouts_trained": runtime_delta.get("rollouts_trained", 0),
        "rollouts_collected": runtime_delta.get("rollouts_collected", 0),
        "expected_rollouts_collected": expected_rollouts,
        "environment_steps": environment_steps_delta,
        "expected_environment_steps": expected_environment_steps,
        "fit_steps_delta": fit_steps_delta,
        "fitted_steps_start": fitted_steps_start.tolist(),
        "fitted_steps_end": fitted_steps_end.tolist(),
        "slot_generation_start": slot_generation_start.tolist(),
        "slot_generation_end": slot_generation_end.tolist(),
        "fitted_steps_equal": bool(np.array_equal(fitted_steps_start, fitted_steps_end)),
        "fitted_steps_not_increased": _fitted_steps_not_increased(action, final),
        "changed_core_state_keys": changed_keys,
        "actor_changes_restricted_to_lru_prior_codec_state": actor_changes_allowed,
        "slot_allocations": runtime_delta.get("slot_allocations", 0),
        "slot_evictions": runtime_delta.get("slot_evictions", 0),
        "prior_bindings": runtime_delta.get("prior_bindings", 0),
    }
    audit["passed"] = bool(
        audit["critic_unchanged"]
        and audit["w0_core_unchanged"]
        and audit["foundation_unchanged"]
        and audit["optimizer_state_unchanged"]
        and audit["optimizer_step_count_delta"] == 0
        and audit["minibatch_updates"] == 0
        and audit["rollouts_trained"] == 0
        and audit["rollouts_collected"] == expected_rollouts
        and audit["environment_steps"] == expected_environment_steps
        and audit["fit_steps_delta"] == 0
        and audit["fitted_steps_not_increased"]
        and audit["actor_changes_restricted_to_lru_prior_codec_state"]
    )
    return audit


def _persisted_branch_steps(case_folder: Path) -> dict[str, Any]:
    branch_root = case_folder / "branches"
    branch_counts: dict[str, int | None] = {}
    for branch in sorted(branch_root.iterdir()) if branch_root.is_dir() else []:
        if branch.is_dir():
            branch_counts[branch.name] = base._count_step_rows(branch / "steps.csv")
    if any(value is None for value in branch_counts.values()):
        total: int | None = None
    else:
        total = int(sum(int(value) for value in branch_counts.values()))
    return {"branch_environment_steps": total, "branch_steps_by_name": branch_counts,
            "new_prefix_environment_steps": 0, "new_environment_steps_total": total}


def _branch_reference(
    source_case_folder: Path,
    branch: Mapping[str, Any],
) -> dict[str, Any]:
    path = _source_branch_summary_path(source_case_folder, branch)
    summary = base._load_json(path)
    return {
        "source_summary": str(path),
        "source_summary_sha256": base._sha(path),
        "action": str(branch["action"]),
        "future_regime": str(branch["future_regime"]),
        "action_state_hash": summary.get("action_state_hash"),
        "final_state_hash": summary.get("final_state_hash"),
        "environment_steps": int(summary.get("environment_steps", 0)),
        "reward_total": summary.get("reward_total"),
        "reward_per_step": summary.get("reward_per_step"),
        "counts": summary.get("counts", {}),
    }


def _case_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    source_root = Path(payload["source_root"])
    output_root = Path(payload["output_root"])
    source_record = dict(payload["case"])
    config = dict(payload["config"])
    seed = int(source_record["seed"])
    pair_kind = str(source_record["pair_kind"])
    history_regime = str(source_record["history_regime"])
    case_id = str(source_record["case_id"])
    source_case_folder = source_root / "cases" / case_id
    output_case_folder = output_root / "cases" / case_id
    output_case_folder.mkdir(parents=True, exist_ok=True)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    branches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    foundation: Path | None = None
    expected_foundation_sha = ""
    try:
        decision = base._load_json(source_case_folder / "dose_1024" / "decision.json")
        source_base_decision = base._load_json(source_case_folder / "decision.json")
        if str(decision.get("prefix_state_hash")) != str(source_base_decision.get("prefix_state_hash")):
            raise ValueError("source-dose decision does not point at the original common prefix")
        expected_foundation_sha = str(decision["foundation_sha256"])
        base_config = base.loop.load_base_config()
        foundation = Path(base_config["foundation"])
        if base._sha(foundation) != expected_foundation_sha:
            raise RuntimeError("W0 hash differs from the source-dose checkpoint")
        expected_steps_by_future = {
            str(future): _future_step_count(decision["mapping"], config, history_regime, str(future))
            for future in config["future_regimes"]
        }

        source_branches = source_record.get("base_branches", [])
        ppo_references = {
            (str(row["action"]), str(row["future_regime"])): _branch_reference(source_case_folder, row)
            for row in source_branches
            if str(row.get("action")) in ACTIONS
        }
        expected_branch_keys = {(action, str(future)) for action in ACTIONS for future in config["future_regimes"]}
        if set(ppo_references) != expected_branch_keys:
            raise ValueError("source case lost one or more PPO replace/none branch references")

        frozen_decision = copy.deepcopy(decision)
        frozen_decision["format"] = "storage_learning_interaction_decision_v1"
        frozen_decision["learning_mode"] = "no_gradient"
        frozen_decision["selectors_frozen_before_future_expansion"] = True
        frozen_decision["selector_information_source"] = "source-dose prefix only; no future branch feedback"
        frozen_decision["branches"] = []
        original_source_record = {
            "case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "branches": source_record.get("base_branches", []),
        }
        new_candidate = np.asarray(decision["new_candidate"], dtype=np.float32).reshape(3, 3)
        old_candidate = np.asarray(decision["old_candidate_2048"], dtype=np.float32).reshape(3, 3)

        with _no_gradient_execution():
            branches = source_dose._run_candidate_branches(
                case_folder=source_case_folder,
                dose_folder=output_case_folder,
                base_record=original_source_record,
                decision=frozen_decision,
                config=config,
                base_config=base_config,
                foundation=foundation,
                old_candidate=old_candidate,
                new_candidate=new_candidate,
            )

        if len(branches) != len(expected_branch_keys):
            raise RuntimeError(f"expected {len(expected_branch_keys)} fresh no-gradient branches, got {len(branches)}")
        if any(str(row.get("format", "")).endswith("alias_v1") for row in branches):
            raise RuntimeError("no-gradient branches must never alias PPO-trained outcomes")
        for branch in branches:
            branch_folder = output_case_folder / "branches" / f"{branch['action']}__{branch['future_regime']}"
            expected_branch_steps = expected_steps_by_future[str(branch["future_regime"])]
            if int(branch["environment_steps"]) != expected_branch_steps:
                raise RuntimeError(
                    f"raw future steps differ for {branch['action']} / {branch['future_regime']}: "
                    f"{branch['environment_steps']} != {expected_branch_steps}"
                )
            reference = ppo_references[(str(branch["action"]), str(branch["future_regime"]))]
            if branch.get("action_state_hash") != reference.get("action_state_hash"):
                raise RuntimeError(
                    f"no-gradient and PPO {branch['action']} / {branch['future_regime']} branches "
                    "do not start from the same post-action state"
                )
            audit = _audit_branch(
                branch_folder, foundation=foundation,
                expected_foundation_sha=expected_foundation_sha,
                expected_environment_steps=expected_branch_steps,
            )
            if not audit["passed"]:
                raise RuntimeError(f"no-gradient audit failed for {branch['action']} / {branch['future_regime']}: {audit}")
            branch["format"] = BRANCH_FORMAT
            branch["candidate_set"] = "shared_newC_vs_none"
            branch["old_candidate_source_steps"] = None
            branch["learning_mode"] = "no_gradient"
            matched_ppo_branch = {
                "source_case_id": case_id,
                "action": str(branch["action"]),
                "future_regime": str(branch["future_regime"]),
                **reference,
            }
            branch["ppo_reference"] = matched_ppo_branch
            branch["matched_ppo_branch"] = matched_ppo_branch
            branch["no_gradient_audit"] = audit
            branch["new_prefix_environment_steps"] = 0
            branch["new_environment_steps"] = int(branch["environment_steps"])
            branch["counts"]["rollouts_collected"] = int(audit["rollouts_collected"])
            branch["fit_steps_delta"] = int(audit["fit_steps_delta"])
            branch["source_prefix_state_hash"] = decision["prefix_state_hash"]
            branch["prefix_state_hash"] = decision["prefix_state_hash"]
            branch["source_case_id"] = case_id
            branch["source_prefix_checkpoint"] = str(source_case_folder / "prefix" / "checkpoint.pt")
            branch["minibatch_updates"] = int(audit["minibatch_updates"])
            branch["rollouts_collected"] = int(audit["rollouts_collected"])
            branch["fitted_steps_start"] = audit["fitted_steps_start"]
            branch["fitted_steps_end"] = audit["fitted_steps_end"]
            branch["fitted_steps_not_increased"] = bool(audit["fitted_steps_not_increased"])
            branch["w0_unchanged"] = bool(audit["w0_unchanged"])
            branch["critic_unchanged"] = bool(audit["critic_unchanged"])
            branch_summary_path = branch_folder / "summary.json"
            _write_json(branch_summary_path, branch)
        frozen_decision["branches"] = [
            {
                "action": str(row["action"]),
                "future_regime": str(row["future_regime"]),
                "summary": f"branches/{row['action']}__{row['future_regime']}/summary.json",
                "independent_observation": True,
                "learning_mode": "no_gradient",
            }
            for row in branches
        ]
        _write_json(output_case_folder / "decision.json", frozen_decision)
        raw_counts = _persisted_branch_steps(output_case_folder)
        result = {
            "case_id": case_id,
            "source_case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "source_prefix_state_hash": decision["prefix_state_hash"],
            "prefix_state_hash": decision["prefix_state_hash"],
            "source_prefix_environment_steps": int(source_record.get("prefix", {}).get("environment_steps", 0)),
            "new_prefix_environment_steps": 0,
            "branches": branches,
            "branch_results": branches,
            "ppo_references": list(ppo_references.values()),
            "costs": {
                **raw_counts,
                "case_cpu_s": float(time.process_time() - start_cpu),
                "case_wall_s": float(time.perf_counter() - start_wall),
                "branch_cpu_s": float(sum(float(row["costs"]["simulation_cpu_s"]) for row in branches)),
                "branch_wall_s": float(sum(float(row["costs"]["simulation_wall_s"]) for row in branches)),
            },
            "errors": errors,
        }
    except Exception:
        raw_counts = _persisted_branch_steps(output_case_folder)
        error = {
            "case_id": case_id,
            "error_type": "learning_interaction_case_failure",
            "error": traceback.format_exc(),
            "persisted_interactions": raw_counts,
            "count_source": "new no-gradient branch steps.csv rows; source prefix is not re-run",
            "source_prefix_environment_steps": int(source_record.get("prefix", {}).get("environment_steps", 0)),
            "new_prefix_environment_steps": 0,
            "cpu_s": float(time.process_time() - start_cpu),
            "wall_s": float(time.perf_counter() - start_wall),
        }
        _write_json(output_case_folder / "learning_interaction_error.json", error)
        errors.append(error)
        result = {
            "case_id": case_id,
            "source_case_id": case_id,
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history_regime,
            "source_prefix_environment_steps": int(source_record.get("prefix", {}).get("environment_steps", 0)),
            "source_prefix_state_hash": str(source_record.get("prefix", {}).get("prefix_state_hash", "")),
            "prefix_state_hash": str(source_record.get("prefix", {}).get("prefix_state_hash", "")),
            "new_prefix_environment_steps": 0,
            "branches": branches,
            "branch_results": branches,
            "ppo_references": [],
            "costs": {
                **raw_counts,
                "case_cpu_s": float(time.process_time() - start_cpu),
                "case_wall_s": float(time.perf_counter() - start_wall),
                "branch_cpu_s": None,
                "branch_wall_s": None,
            },
            "errors": errors,
        }
    _write_json(output_case_folder / "case_summary.json", result)
    return result


def _jobs(source_root: Path, output_root: Path, config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source_root": str(source_root), "output_root": str(output_root),
         "config": dict(config), "case": dict(record)}
        for record in records
    ]


def run(source: str | Path, output: str | Path) -> dict[str, Any]:
    source_root, source_summary, config = _load_source_run(Path(source))
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config.json", {
        "format": "storage_learning_interaction_config_v1",
        "source_path": str(source_root),
        "source_config": config,
        "actions": list(ACTIONS),
        "learning_mode": "no_gradient",
        "new_prefix_environment_steps": 0,
    })
    manifest = _snapshot_sources(output_root, source_root)
    source_summary_path = source_root / "summary.json"
    source_config_path = source_root / "config.json"
    source_reference = {
        "path": str(source_root),
        "summary_sha256": base._sha(source_summary_path),
        "config_sha256": base._sha(source_config_path),
        "format": source_summary["format"],
        "phase": source_summary.get("phase"),
        "source_manifest": source_summary.get("source_manifest", {}),
        "costs": source_summary.get("costs", {}),
    }
    _write_json(output_root / "source_reference.json", source_reference)
    cases = sorted(
        source_summary["cases"],
        key=lambda row: (int(row["seed"]), str(row["pair_kind"]), str(row["history_regime"])),
    )
    workers = max(1, min(int(config.get("max_workers", 4)), len(cases)))
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_case_worker, job): job for job in _jobs(source_root, output_root, config, cases)}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception:
                case_record = job["case"]
                case_id = str(case_record["case_id"])
                error = {
                    "case_id": case_id,
                    "error_type": "learning_interaction_worker_failure",
                    "error": traceback.format_exc(),
                    "persisted_interactions": _persisted_branch_steps(output_root / "cases" / case_id),
                    "new_prefix_environment_steps": 0,
                }
                _write_json(output_root / "cases" / case_id / "learning_interaction_error.json", error)
                result = {
                    "case_id": case_id,
                    "seed": int(case_record["seed"]),
                    "pair_kind": str(case_record["pair_kind"]),
                    "history_regime": str(case_record["history_regime"]),
                    "new_prefix_environment_steps": 0,
                    "branches": [],
                    "costs": {**error["persisted_interactions"], "case_cpu_s": None, "case_wall_s": None,
                              "branch_cpu_s": None, "branch_wall_s": None},
                    "errors": [error],
                }
            records.append(result)
            errors.extend(result.get("errors", []))
    records.sort(key=lambda row: (int(row["seed"]), str(row["pair_kind"]), str(row["history_regime"])))
    branches = [branch for record in records for branch in record.get("branches", [])]

    def aggregate(record_key: str):
        values = [row.get("costs", {}).get(record_key) for row in records]
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values)

    step_counts = [row.get("costs", {}).get("new_environment_steps_total") for row in records]
    if any(value is None for value in step_counts):
        new_environment_steps = None
    else:
        new_environment_steps = int(sum(int(value) for value in step_counts))
    orchestrator_cpu_s = float(time.process_time() - start_cpu)
    summary = {
        "format": FORMAT,
        "phase": "pilot" if source_summary.get("phase") == "pilot" else "main",
        "learning_mode": "no_gradient",
        "source_reference": source_reference,
        "source_manifest": manifest,
        "config": config,
        "actions": list(ACTIONS),
        "case_count": len(records),
        "branch_count": len(branches),
        "physical_branch_count": sum(bool(row.get("no_gradient_audit", {}).get("passed")) for row in branches),
        "cases": records,
        "branches": branches,
        "branch_results": branches,
        "errors": errors,
        "costs": {
            "new_prefix_environment_steps": 0,
            "new_branch_environment_steps": new_environment_steps,
            "new_environment_steps_total": new_environment_steps,
            "source_reference_prefix_environment_steps": source_summary.get("costs", {}).get("prefix_environment_steps"),
            "source_reference_branch_environment_steps": source_summary.get("costs", {}).get("branch_environment_steps"),
            "source_reference_environment_steps_total": source_summary.get("costs", {}).get("environment_steps_total"),
            "child_cpu_s": aggregate("case_cpu_s"),
            "orchestrator_cpu_s": orchestrator_cpu_s,
            "cpu_s": None if aggregate("case_cpu_s") is None else float(aggregate("case_cpu_s") + orchestrator_cpu_s),
            "child_wall_s": aggregate("case_wall_s"),
            "wall_s": float(time.perf_counter() - start_wall),
            "branch_cpu_s": aggregate("branch_cpu_s"),
            "branch_wall_s": aggregate("branch_wall_s"),
        },
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="complete source-dose output directory")
    parser.add_argument("--output", type=Path, required=True, help="new learning-interaction output directory")
    arguments = parser.parse_args(argv)
    summary = run(arguments.source, arguments.output)
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

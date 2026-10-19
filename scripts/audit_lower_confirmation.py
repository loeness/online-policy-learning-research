"""Independently audit lower-confirmation artifacts without stepping an environment.

The audit verifies saved schedules, raw transitions, selector provenance, action
branches, state hashes, persistent package bytes, and interaction cost totals.
It reports integrity findings only; it does not analyze or rank policy rewards.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.storage_decision import (
    ACTIONS,
    pair_mapping,
    prefix_schedule,
    semantic_hash,
    template_schedule,
    validate_config,
)
from online_rl.write_decision_policy import encode_actor
from scripts import audit_source_followup as source_audit
from scripts import audit_storage_decision as storage_audit


FORMAT = "lower_confirmation_audit_v1"
SUMMARY_FORMAT = "storage_decision_summary_v1"
EXPECTED_MAIN_SEEDS = (
    22101, 22102, 22103, 22104, 22105, 22106, 22107, 22108,
    22109, 22110, 22111, 22112, 22113, 22114, 22118, 22121,
)
EXPECTED_ACTIONS = ("keep", "replace", "mean", "none")
EXPECTED_HISTORIES = ("old_major", "new_major")
EXPECTED_FUTURES = ("old_major", "new_major")
PRIVATE_PARTS = {"agent.md", "agents.md", "tmp"}


def expected_plan(
    phase: str,
    *,
    seeds: Sequence[int],
    histories: Sequence[str],
    bootstrap_steps: int,
    bout_steps: int,
) -> dict[str, int]:
    """Return the frozen case, branch, and step budgets for one phase."""
    if phase not in {"pilot", "main"}:
        raise ValueError(f"unknown phase: {phase!r}")
    case_count = len(seeds) * len(histories)
    prefix_per_case = 2 * int(bootstrap_steps) + 8 * int(bout_steps)
    branch_per_case = 12 * int(bout_steps)
    branch_count = case_count * len(EXPECTED_ACTIONS)
    prefix_steps = case_count * prefix_per_case
    branch_steps = branch_count * branch_per_case
    return {
        "case_count": case_count,
        "branch_count": branch_count,
        "prefix_steps_per_case": prefix_per_case,
        "branch_steps_per_branch": branch_per_case,
        "prefix_environment_steps": prefix_steps,
        "branch_environment_steps": branch_steps,
        "environment_steps_total": prefix_steps + branch_steps,
    }


def _check(audit: storage_audit.Audit, condition: bool, code: str, where: str,
           message: str, **details: Any) -> None:
    storage_audit._check(audit, condition, code, where, message, **details)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_snapshot_path(root: Path, relative_name: str) -> Path | None:
    normalized = str(relative_name).replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or not relative.parts:
        return None
    if any(part in {"", ".", ".."} or ":" in part for part in relative.parts):
        return None
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        return None
    return candidate


def _validate_snapshot(audit: storage_audit.Audit, cohort: Path,
                       summary: Mapping[str, Any]) -> int | None:
    snapshot = cohort / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    try:
        manifest = storage_audit._json(manifest_path)
    except Exception as exc:
        audit.fail("source_manifest_read", str(manifest_path),
                   "source snapshot manifest is missing or unreadable", error=repr(exc))
        return None

    _check(audit, manifest == summary.get("source_manifest"), "source_manifest_summary",
           str(manifest_path), "summary source manifest differs from the persisted manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        audit.fail("source_manifest_empty", str(manifest_path), "source manifest lists no sources")
        return None

    required = {
        "configs/surrogate_lower_confirmation.json",
        "scripts/run_lower_confirmation.py",
        "docs/SURROGATE_LOWER_CONFIRMATION_PROTOCOL.md",
    }
    _check(audit, required <= set(map(str, sources)), "source_manifest_required_files",
           str(manifest_path), "source manifest omits a required frozen input",
           missing=sorted(required - set(map(str, sources))))

    for name, record in sources.items():
        parts = Path(str(name).replace("\\", "/")).parts
        private = any(part.casefold() in PRIVATE_PARTS for part in parts)
        _check(audit, not private, "private_source_listed", f"source_snapshot/{name}",
               "manifest lists a private or temporary path")
        copied = _safe_snapshot_path(snapshot, str(name))
        if copied is None:
            audit.fail("source_path_invalid", f"source_snapshot/{name}",
                       "manifest path is absolute, escapes the snapshot, or is malformed")
            continue
        if not copied.is_file():
            audit.fail("source_copy_missing", f"source_snapshot/{name}",
                       "manifest-listed source copy is missing")
            continue
        expected_bytes = record.get("bytes") if isinstance(record, Mapping) else None
        expected_sha = record.get("sha256") if isinstance(record, Mapping) else None
        actual_bytes = copied.stat().st_size
        actual_sha = _digest(copied)
        _check(audit, actual_bytes == expected_bytes and actual_sha == expected_sha,
               "source_copy_hash", f"source_snapshot/{name}",
               "source copy differs from its manifest size or digest",
               expected_bytes=expected_bytes, actual_bytes=actual_bytes,
               expected_sha256=expected_sha, actual_sha256=actual_sha)

    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        _check(audit, not any(part.casefold() in PRIVATE_PARTS for part in relative.parts),
               "private_source_present", f"source_snapshot/{relative.as_posix()}",
               "source snapshot contains a private or temporary path")
    excluded = {str(item).casefold() for item in manifest.get("excluded_local_instructions", [])}
    _check(audit, excluded == {"agent.md", "agents.md"}, "private_exclusions",
           str(manifest_path), "manifest does not record both local instruction exclusions",
           excluded=sorted(excluded))
    matches = source_audit._private_hash_matches(snapshot)
    _check(audit, not matches, "private_exact_hash_match", str(snapshot),
           "snapshot contains bytes identical to a local instruction file", matches=matches)

    original_path = snapshot / "config_original.json"
    frozen_copy_path = snapshot / "configs" / "surrogate_lower_confirmation.json"
    try:
        original = storage_audit._json(original_path)
        frozen_copy = storage_audit._json(frozen_copy_path)
    except Exception as exc:
        audit.fail("original_config_read", str(original_path),
                   "original run config or its frozen source copy is unreadable", error=repr(exc))
        original = None
    if original is not None:
        _check(audit, original == frozen_copy, "original_config_snapshot_match",
               str(original_path), "run config differs from its frozen source snapshot")

    foundation = manifest.get("foundation", {})
    if not isinstance(foundation, Mapping):
        audit.fail("foundation_manifest", str(manifest_path), "foundation provenance is malformed")
        return None
    try:
        foundation_bytes = int(foundation.get("bytes", -1))
    except (TypeError, ValueError):
        foundation_bytes = -1
    foundation_path = Path(str(foundation.get("path", "")))
    if foundation_bytes < 0 or not foundation_path.is_file():
        audit.fail("foundation_missing", str(foundation_path),
                   "foundation file or recorded byte count is unavailable")
        return None
    actual_foundation_sha = _digest(foundation_path)
    _check(audit, foundation_path.stat().st_size == foundation_bytes
           and actual_foundation_sha == foundation.get("sha256"),
           "foundation_hash", str(foundation_path), "foundation differs from the source manifest",
           expected_bytes=foundation_bytes, actual_bytes=foundation_path.stat().st_size,
           expected_sha256=foundation.get("sha256"), actual_sha256=actual_foundation_sha)
    return foundation_bytes


def _load_phase_config(audit: storage_audit.Audit, cohort: Path,
                       summary: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    config_path = cohort / "config.json"
    original_path = cohort / "source_snapshot" / "config_original.json"
    try:
        config = storage_audit._json(config_path)
        original = storage_audit._json(original_path)
    except Exception as exc:
        audit.fail("config_read", str(cohort), "effective or original config is unreadable", error=repr(exc))
        return None, None
    phase = str(summary.get("phase", ""))
    expected = dict(original)
    if phase == "pilot":
        pilot = original.get("pilot")
        if not isinstance(pilot, Mapping):
            audit.fail("pilot_config_missing", str(original_path), "frozen pilot config is missing")
            return None, original
        for key in ("seeds", "pair_kinds", "history_regimes", "bootstrap_steps", "bout_steps"):
            if key not in pilot:
                audit.fail("pilot_config_field_missing", str(original_path),
                           "pilot config lacks a phase override", field=key)
                return None, original
            expected[key] = pilot[key]
    elif phase != "main":
        audit.fail("phase_invalid", "summary.json", "phase must be pilot or main", phase=phase)
        return None, original
    try:
        expected = validate_config(expected)
    except Exception as exc:
        audit.fail("config_invalid", str(original_path), "frozen config fails validation", error=repr(exc))
        return None, original

    _check(audit, config == expected, "phase_config_mismatch", str(config_path),
           "effective config differs from the normalized frozen phase config")
    _check(audit, summary.get("config") == config, "summary_config_mismatch", "summary.json",
           "summary config differs from config.json")
    _check(audit, config.get("format") == "storage_decision_v1"
           and config.get("analysis_role") == "independent_surrogate_lower_confirmation",
           "config_identity", str(config_path), "config identity is not the frozen lower-confirmation run")
    _check(audit, tuple(config.get("pair_kinds", ())) == ("dynamics",), "pair_universe",
           str(config_path), "only the dynamics relation is allowed")
    _check(audit, tuple(config.get("actions", ())) == EXPECTED_ACTIONS, "action_universe",
           str(config_path), "action universe/order differs from the frozen four actions",
           actual=config.get("actions"), expected=list(EXPECTED_ACTIONS))
    _check(audit, tuple(config.get("future_regimes", ())) == EXPECTED_FUTURES, "future_config_universe",
           str(config_path), "top-level config must retain both future regimes")

    seeds = tuple(int(seed) for seed in config.get("seeds", ()))
    histories = tuple(str(item) for item in config.get("history_regimes", ()))
    if phase == "pilot":
        _check(audit, seeds == (22089,), "pilot_seed", str(config_path),
               "pilot must use the frozen independent seed")
        _check(audit, histories == ("old_major",), "pilot_history", str(config_path),
               "pilot must use only old_major history")
        _check(audit, int(config.get("bootstrap_steps", -1)) == 512
               and int(config.get("bout_steps", -1)) == 256, "pilot_step_budget", str(config_path),
               "pilot bootstrap/bout lengths differ from the frozen budget")
    else:
        _check(audit, seeds == EXPECTED_MAIN_SEEDS, "main_seed_universe", str(config_path),
               "main cohort must include the 16 frozen mapping-balanced seeds",
               actual=list(seeds), expected=list(EXPECTED_MAIN_SEEDS))
        _check(audit, histories == EXPECTED_HISTORIES, "main_history_universe", str(config_path),
               "main cohort must include both frozen history regimes",
               actual=list(histories), expected=list(EXPECTED_HISTORIES))
        _check(audit, int(config.get("bootstrap_steps", -1)) == 2048
               and int(config.get("bout_steps", -1)) == 1024, "main_step_budget", str(config_path),
               "main bootstrap/bout lengths differ from the frozen budget")
        mapping_counts = {family: 0 for family in ("A", "B")}
        for seed in seeds:
            mapping_counts[pair_mapping(seed, "dynamics")["families"]["old"]] += 1
        _check(audit, mapping_counts == {"A": 8, "B": 8}, "main_mapping_balance", str(config_path),
               "main seeds are not balanced by the frozen dynamics old-family mapping",
               mapping_counts=mapping_counts)
    return config, original


def _check_confirmation_action(audit: storage_audit.Audit, record: Mapping[str, Any],
                               decision: Mapping[str, Any], branch_records: Sequence[Mapping[str, Any]],
                               *, history: str, where: str) -> None:
    selectors = record.get("selectors")
    decision_selectors = decision.get("selectors")
    diagnostics = decision.get("selector_diagnostics", {})
    past_only = diagnostics.get("past_only_selectors", {}) if isinstance(diagnostics, Mapping) else {}
    selected = selectors.get("surrogate_lower") if isinstance(selectors, Mapping) else None
    _check(audit, selected in EXPECTED_ACTIONS, "selector_action_invalid", where,
           "surrogate_lower did not select a declared physical action", action=selected)
    _check(audit, selected == decision_selectors.get("surrogate_lower")
           and selected == past_only.get("surrogate_lower"), "selector_copy_mismatch", where,
           "case selector does not match the frozen decision and past-only diagnostics",
           case_action=selected,
           decision_action=decision_selectors.get("surrogate_lower") if isinstance(decision_selectors, Mapping) else None,
           diagnostic_action=past_only.get("surrogate_lower"))
    _check(audit, record.get("confirmation_selector") == "surrogate_lower"
           and record.get("confirmation_action") == selected, "confirmation_selector_mismatch", where,
           "confirmation action does not match its declared frozen selector",
           selector=record.get("confirmation_selector"), action=record.get("confirmation_action"),
           selected=selected)
    _check(audit, record.get("confirmation_future_regime") == history
           and record.get("confirmation_future_scope") == "history_matched_only",
           "confirmation_future_scope", where,
           "confirmation did not record the matched-history future scope",
           future=record.get("confirmation_future_regime"), history=history,
           scope=record.get("confirmation_future_scope"))
    selected_branches = [branch for branch in branch_records
                         if branch.get("action") == selected and branch.get("future_regime") == history]
    _check(audit, len(selected_branches) == 1, "selector_branch_binding", where,
           "frozen selector does not bind to exactly one completed matched-future action branch",
           action=selected, history=history, matching_branches=len(selected_branches))


def _step_rows(path: Path) -> tuple[int | None, list[dict[str, str]] | None, str | None]:
    if not path.is_file():
        return None, None, "missing"
    try:
        rows = storage_audit._csv(path)
        return len(rows), rows, None
    except Exception as exc:
        return None, None, repr(exc)


def _raw_inventory(audit: storage_audit.Audit, cohort: Path,
                   expected_paths: set[Path]) -> tuple[int | None, int]:
    """Count every persisted steps.csv, retaining partial rows for failures."""
    cases_root = cohort / "cases"
    actual_paths = {path.resolve() for path in cases_root.rglob("steps.csv") if path.is_file()} if cases_root.is_dir() else set()
    for missing in sorted(expected_paths - actual_paths):
        audit.fail("expected_steps_missing", str(missing), "expected raw step ledger is missing")
    for extra in sorted(actual_paths - expected_paths):
        audit.fail("unexpected_steps_file", str(extra),
                   "unexpected raw step ledger is included in observed interaction cost")
    partial = 0
    known = True
    for path in sorted(actual_paths):
        count, _, error = _step_rows(path)
        if count is None:
            known = False
            audit.fail("raw_steps_unreadable", str(path), "raw interaction count is unknown", error=error)
        else:
            partial += count
    if actual_paths != expected_paths:
        known = False
    return (partial if known else None), partial


def _audit_case(audit: storage_audit.Audit, cohort: Path, record: Mapping[str, Any],
                config: Mapping[str, Any], foundation_bytes: int,
                phase: str, expected_future: str) -> dict[str, Any]:
    case_id = str(record.get("case_id", ""))
    case_dir = cohort / "cases" / case_id
    where = str(case_dir)
    error_start = len(audit.errors)
    seed = int(record.get("seed", -1))
    pair_kind = str(record.get("pair_kind", ""))
    history = str(record.get("history_regime", ""))
    expected_id = f"s{seed}_{pair_kind}_{history}"
    _check(audit, case_id == expected_id, "case_identity", where,
           "case ID differs from its saved identity", expected=expected_id)
    _check(audit, pair_kind == "dynamics" and history in config["history_regimes"],
           "case_universe", where, "case pair or history is outside the frozen design")
    _check(audit, not record.get("errors"), "case_runner_errors", where,
           "runner recorded a case error", errors=record.get("errors"))
    if (case_dir / "error.json").exists():
        audit.fail("case_error_file", str(case_dir / "error.json"), "case contains a failure record")

    decision_path = case_dir / "decision.json"
    prefix_dir = case_dir / "prefix"
    try:
        decision = storage_audit._json(decision_path)
        prefix_summary = storage_audit._json(prefix_dir / "summary.json")
        prefix_checkpoint = storage_audit._checkpoint(prefix_dir / "checkpoint.pt")
        prefix_simulator = storage_audit._json(prefix_dir / "simulator.json")
    except Exception as exc:
        audit.fail("prefix_artifacts", where, "decision or prefix artifact is unreadable", error=repr(exc))
        return {"case_id": case_id, "passed": False, "errors": len(audit.errors) - error_start}

    _check(audit, decision.get("format") == "storage_decision_case_v1", "decision_format", where,
           "decision format differs from storage-decision case format")
    for field, expected_value in (("case_id", case_id), ("seed", seed), ("pair_kind", pair_kind),
                                  ("history_regime", history)):
        _check(audit, decision.get(field) == expected_value, "decision_identity", where,
               f"decision field {field} differs from its case", actual=decision.get(field), expected=expected_value)
    storage_audit._check_checkpoint_shape(audit, prefix_checkpoint, config, str(prefix_dir / "checkpoint.pt"))
    _check(audit, int(prefix_checkpoint.get("seed", -1)) == seed, "prefix_seed", where,
           "prefix checkpoint seed differs from case seed")
    mapping = pair_mapping(seed, pair_kind)
    _check(audit, storage_audit._close(decision.get("mapping"), mapping, atol=0.0, rtol=0.0),
           "pair_mapping", where, "saved task mapping differs from deterministic seed mapping")
    _check(audit, decision.get("foundation_sha256") == prefix_summary.get("foundation_sha256")
           and decision.get("foundation_sha256") == prefix_checkpoint.get("foundation_sha256"),
           "foundation_case_hash", where, "case, prefix summary, and checkpoint disagree on W0 digest")
    _check(audit, bool(decision.get("source_w0_unchanged_at_decision", False)), "w0_unchanged", where,
           "W0 was not recorded unchanged at the decision")

    try:
        prefix_hash = semantic_hash({"learner": prefix_checkpoint, "simulator": prefix_simulator})
        _check(audit, prefix_hash == decision.get("prefix_state_hash")
               and prefix_hash == prefix_summary.get("prefix_state_hash"), "prefix_state_hash", where,
               "prefix checkpoint and simulator differ from the recorded semantic hash",
               recomputed=prefix_hash, decision=decision.get("prefix_state_hash"),
               prefix_summary=prefix_summary.get("prefix_state_hash"))
    except Exception as exc:
        audit.fail("prefix_state_hash_error", where, "prefix state hash could not be recomputed", error=repr(exc))
        prefix_hash = None

    prefix_schedule_rows = prefix_schedule(mapping, config, history)
    _check(audit, storage_audit._close(decision.get("bootstrap_schedule"), prefix_schedule_rows[:2], atol=0.0, rtol=0.0)
           and storage_audit._close(decision.get("history_schedule"), prefix_schedule_rows[2:], atol=0.0, rtol=0.0),
           "prefix_schedule", where, "saved prefix schedules differ from deterministic mapping and config")
    _check(audit, prefix_summary.get("seed") == seed and prefix_summary.get("pair_kind") == pair_kind
           and prefix_summary.get("history_regime") == history, "prefix_summary_identity", where,
           "prefix summary identity differs from its case")
    _check(audit, len(prefix_summary.get("per_bout", [])) == len(prefix_schedule_rows), "prefix_bout_count", where,
           "prefix summary does not contain all scheduled bouts",
           actual=len(prefix_summary.get("per_bout", [])), expected=len(prefix_schedule_rows))
    for actual, expected in zip(prefix_summary.get("per_bout", []), prefix_schedule_rows):
        _check(audit, int(actual.get("bout", -1)) == int(expected["bout"])
               and actual.get("task") == expected["task"]
               and int(actual.get("group", -1)) == int(expected["group"])
               and int(actual.get("steps", -1)) == int(expected["steps"]),
               "prefix_bout_schedule", where, "prefix bout differs from its frozen schedule",
               actual=actual, expected=expected)

    prefix_csv = prefix_dir / "steps.csv"
    prefix_rows = []
    try:
        prefix_rows = storage_audit._csv(prefix_csv)
    except Exception as exc:
        audit.fail("prefix_steps_csv", str(prefix_csv), "prefix raw step CSV is unreadable", error=repr(exc))
    prefix_steps, prefix_reward, _ = storage_audit._check_schedule_rows(
        audit, prefix_rows, prefix_schedule_rows, phase="prefix", action="", future="",
        global_start=0, seed=seed, where=str(prefix_csv),
    )
    plan = expected_plan(phase,
                         seeds=(seed,), histories=(history,), bootstrap_steps=int(config["bootstrap_steps"]),
                         bout_steps=int(config["bout_steps"]))
    _check(audit, prefix_steps == plan["prefix_steps_per_case"], "prefix_step_budget", where,
           "prefix raw step count differs from the frozen phase budget",
           actual=prefix_steps, expected=plan["prefix_steps_per_case"])
    storage_audit._check_reward_summary(audit, prefix_rows, prefix_summary,
                                        raw_reward=prefix_reward, raw_steps=prefix_steps,
                                        where=str(prefix_dir / "summary.json"))
    storage_audit._check_rollouts(audit, prefix_dir, phase=("bootstrap", "prefix_history"),
                                  expected_rows=prefix_steps,
                                  summary={**prefix_summary, "rollout_steps": int(config["rollout_steps"])},
                                  where=str(prefix_dir / "summary.json"))
    _check(audit, int(record.get("prefix", {}).get("raw_environment_steps", -1)) == prefix_steps,
           "prefix_raw_record", where, "recorded prefix raw-row count differs from the CSV",
           observed=prefix_steps, prefix_summary=prefix_summary.get("raw_environment_steps"),
           case_record=record.get("prefix", {}).get("raw_environment_steps"))

    storage_audit._check_selector(audit, case_dir, decision, config, prefix_checkpoint)
    _check(audit, record.get("selectors") == decision.get("selectors"), "case_selector_copy", where,
           "case selector map differs from the saved decision")
    _check(audit, record.get("selector_diagnostics") == decision.get("selector_diagnostics"),
           "case_selector_diagnostics_copy", where,
           "case selector diagnostics differ from the saved decision")

    manifest = storage_audit._json(cohort / "source_snapshot" / "manifest.json")
    _check(audit, decision.get("foundation_sha256") == manifest.get("foundation", {}).get("sha256"),
           "foundation_manifest_case_hash", where, "case W0 digest differs from the cohort manifest")
    storage_audit._package_check(
        audit, folder=prefix_dir, actor_path=prefix_dir / "actor.bin",
        checkpoint_path=prefix_dir / "checkpoint.pt", recorded=prefix_summary.get("capacity"),
        foundation_bytes=foundation_bytes, package_cap=int(config["package_budget_bytes"]),
        actor_cap=int(config["actor_budget_bytes"]), where=str(prefix_dir),
    )
    try:
        local, shared, physical = storage_audit._core_arrays(prefix_checkpoint)
        encoded = encode_actor(local, shared, physical, codec="int8",
                               budget_bytes=int(config["actor_budget_bytes"])).data
        actor_path = prefix_dir / "actor.bin"
        _check(audit, actor_path.read_bytes() == encoded, "prefix_actor_checkpoint_match", str(actor_path),
               "prefix int8 actor sidecar differs from its learner checkpoint")
    except Exception as exc:
        audit.fail("prefix_actor_checkpoint_error", str(prefix_dir),
                   "prefix actor sidecar could not be checked against its checkpoint", error=repr(exc))

    branch_dir = case_dir / "branches"
    expected_branch_ids = {f"{action}__{expected_future}" for action in EXPECTED_ACTIONS}
    actual_branch_ids = {path.name for path in branch_dir.iterdir() if path.is_dir()} if branch_dir.is_dir() else set()
    _check(audit, actual_branch_ids == expected_branch_ids, "branch_directory_universe", where,
           "case branch directories differ from the four matched-future actions",
           actual=sorted(actual_branch_ids), expected=sorted(expected_branch_ids))
    branch_records = list(record.get("branches", []))
    record_branch_keys = [(str(row.get("action", "")), str(row.get("future_regime", "")))
                          for row in branch_records]
    expected_branch_keys = [(action, expected_future) for action in EXPECTED_ACTIONS]
    _check(audit, len(record_branch_keys) == len(set(record_branch_keys))
           and set(record_branch_keys) == set(expected_branch_keys), "case_branch_records", where,
           "case record does not contain exactly one branch per frozen action and matched future",
           actual=record_branch_keys, expected=expected_branch_keys)
    _check_confirmation_action(audit, record, decision, branch_records,
                               history=history, where=where)

    restored_hashes = decision.get("restored_state_hashes", {})
    _check(audit, set(restored_hashes) == {f"{action}__{expected_future}" for action in EXPECTED_ACTIONS},
           "restored_hash_universe", where, "decision restored-state hash map has unexpected branches",
           actual=sorted(restored_hashes), expected=sorted(expected_branch_ids))
    branch_rows_total = 0
    branch_cpu_values: list[float] = []
    branch_wall_values: list[float] = []
    expected_paths = {prefix_csv.resolve()}
    for action in EXPECTED_ACTIONS:
        branch_name = f"{action}__{expected_future}"
        branch = branch_dir / branch_name
        branch_where = str(branch)
        summary_path = branch / "summary.json"
        branch_record = next((row for row in branch_records
                              if row.get("action") == action and row.get("future_regime") == expected_future), None)
        try:
            branch_summary = storage_audit._json(summary_path)
            post_checkpoint_path = branch / "post_action" / "checkpoint.pt"
            post_checkpoint = storage_audit._checkpoint(post_checkpoint_path)
        except Exception as exc:
            audit.fail("branch_artifacts", branch_where, "branch or post-action checkpoint is unreadable",
                       error=repr(exc))
            continue
        _check(audit, branch_summary.get("post_action_checkpoint") == "post_action/checkpoint.pt"
               and branch_summary.get("post_action_simulator") == "post_action/simulator.json",
               "post_action_paths", branch_where,
               "branch post-action checkpoint/simulator paths differ from the frozen layout")
        for field, expected_value in (("case_id", case_id), ("seed", seed), ("pair_kind", pair_kind),
                                      ("history_regime", history), ("action", action),
                                      ("future_regime", expected_future)):
            _check(audit, branch_summary.get(field) == expected_value, "branch_identity", branch_where,
                   f"branch field {field} differs from expected identity",
                   actual=branch_summary.get(field), expected=expected_value)
        _check(audit, branch_summary.get("future_regime") == history
               and branch_summary.get("prefix_bouts_included_in_reward") is False,
               "matched_future_contract", branch_where,
               "branch future is not matched to history or includes prefix reward")
        _check(audit, bool(branch_summary.get("foundation_unchanged", False))
               and branch_summary.get("foundation_sha256") == decision.get("foundation_sha256"),
               "branch_foundation", branch_where, "branch foundation differs from the case W0")
        expected_prefix_hash = decision.get("prefix_state_hash")
        _check(audit, branch_summary.get("prefix_state_hash") == expected_prefix_hash
               and branch_summary.get("restored_state_hash") == expected_prefix_hash
               and branch_summary.get("restored_state_hash") == restored_hashes.get(branch_name),
               "branch_restore_hash", branch_where,
               "branch did not restore the same checkpoint/simulator prefix",
               expected=expected_prefix_hash, decision=restored_hashes.get(branch_name),
               branch_restore=branch_summary.get("restored_state_hash"))
        if branch_record is not None:
            record_copy = dict(branch_record)
            record_raw = record_copy.pop("raw_environment_steps", None)
            _check(audit, storage_audit._close(record_copy, branch_summary, atol=0.0, rtol=0.0),
                   "branch_record_summary_match", branch_where,
                   "case branch record differs from the branch summary file")
        storage_audit._check_checkpoint_shape(audit, post_checkpoint, config, str(post_checkpoint_path))
        post_capacity = storage_audit._package_check(
            audit, folder=branch / "post_action", actor_path=branch / "post_action" / "actor.bin",
            checkpoint_path=post_checkpoint_path, recorded=branch_summary.get("capacity_after_action"),
            foundation_bytes=foundation_bytes, package_cap=int(config["package_budget_bytes"]),
            actor_cap=int(config["actor_budget_bytes"]), where=f"{branch_where}/post_action",
        )
        final_capacity = storage_audit._package_check(
            audit, folder=branch, actor_path=branch / "actor.bin", checkpoint_path=branch / "checkpoint.pt",
            recorded=branch_summary.get("capacity"), foundation_bytes=foundation_bytes,
            package_cap=int(config["package_budget_bytes"]), actor_cap=int(config["actor_budget_bytes"]),
            where=f"{branch_where}/final",
        )
        _check(audit, branch_summary.get("post_action_snapshot_included_in_deployment_capacity") is False,
               "post_action_observer_capacity", branch_where,
               "observer-only post-action snapshot is included in deployment capacity")
        post_arrays = storage_audit._branch_core_check(
            audit, branch_dir=branch, prefix_checkpoint=prefix_checkpoint,
            post_checkpoint=post_checkpoint, prefix_sim=prefix_simulator,
            decision=decision, branch_summary=branch_summary, action=action, config=config,
        )

        schedule = template_schedule(mapping, expected_future, kind="future",
                                     steps=int(config["bout_steps"]),
                                     start_bout=len(prefix_schedule_rows))
        _check(audit, storage_audit._close(branch_summary.get("future_schedule"), schedule,
                                            atol=0.0, rtol=0.0),
               "future_schedule", branch_where,
               "branch schedule differs from its matched history regime")
        _check(audit, len(branch_summary.get("per_bout", [])) == len(schedule),
               "future_bout_count", branch_where, "branch summary omits scheduled bouts")
        for actual, expected in zip(branch_summary.get("per_bout", []), schedule):
            _check(audit, int(actual.get("bout", -1)) == int(expected["bout"])
                   and actual.get("task") == expected["task"]
                   and int(actual.get("group", -1)) == int(expected["group"])
                   and int(actual.get("steps", -1)) == int(expected["steps"]),
                   "future_bout_schedule", branch_where,
                   "branch bout differs from the frozen continuation schedule",
                   actual=actual, expected=expected)
        steps_path = branch / "steps.csv"
        expected_paths.add(steps_path.resolve())
        rows = []
        try:
            rows = storage_audit._csv(steps_path)
        except Exception as exc:
            audit.fail("branch_steps_csv", str(steps_path), "branch raw step CSV is unreadable", error=repr(exc))
        branch_steps, branch_reward, uniforms = storage_audit._check_schedule_rows(
            audit, rows, schedule, phase="future", action=action, future=expected_future,
            global_start=prefix_steps, seed=seed, where=str(steps_path),
        )
        _check(audit, branch_steps == plan["branch_steps_per_branch"], "branch_step_budget", branch_where,
               "branch raw step count differs from the frozen continuation budget",
               actual=branch_steps, expected=plan["branch_steps_per_branch"])
        storage_audit._check_reward_summary(audit, rows, branch_summary,
                                            raw_reward=branch_reward, raw_steps=branch_steps,
                                            where=str(summary_path))
        storage_audit._check_rollouts(audit, branch, phase="future", expected_rows=branch_steps,
                                      summary={**branch_summary, "rollout_steps": int(config["rollout_steps"])},
                                      where=str(summary_path))
        _check(audit, branch_record is not None
               and int(branch_record.get("raw_environment_steps", -1)) == branch_steps,
               "branch_raw_record", branch_where,
               "case record raw-row count differs from the branch CSV",
               observed=branch_steps,
               recorded=None if branch_record is None else branch_record.get("raw_environment_steps"))

        if post_arrays is not None:
            parent_group = {"keep": 0, "replace": 1, "mean": -2, "none": -1}[action]
            storage_audit._check_cache_and_inheritance(
                audit, branch_summary.get("per_bout", []), prefix_checkpoint["cache"],
                prefix_checkpoint["slot_generation"], post_arrays[2].tolist(),
                prefix_checkpoint["slot_group_tags"], parent_group=parent_group,
                config=config, where=branch_where,
            )
            storage_audit._check_allocation_probabilities(
                audit, branch_dir=branch, branch_rows=rows, branch_summary=branch_summary,
                post_checkpoint=post_checkpoint, post_arrays=post_arrays,
                config=config, where=branch_where,
            )

        try:
            final_checkpoint = storage_audit._checkpoint(branch / "checkpoint.pt")
            final_actor_path = branch / "actor.bin"
            final_actor = final_actor_path.read_bytes()
            final_local, final_shared, final_groups = storage_audit._core_arrays(final_checkpoint)
            expected_actor = encode_actor(final_local, final_shared, final_groups, codec="int8",
                                          budget_bytes=int(config["actor_budget_bytes"])).data
            _check(audit, final_actor == expected_actor, "final_actor_checkpoint_match", branch_where,
                   "final actor sidecar differs from the saved learner checkpoint")
            final_simulator = storage_audit._json(branch / "simulator.json")
            final_hash = semantic_hash({"learner": final_checkpoint, "simulator": final_simulator})
            _check(audit, final_hash == branch_summary.get("final_state_hash"), "final_state_hash", branch_where,
                   "final learner and simulator do not match the saved hash",
                   recomputed=final_hash, recorded=branch_summary.get("final_state_hash"))
            storage_audit._check_checkpoint_shape(audit, final_checkpoint, config, str(branch / "checkpoint.pt"))
            _check(audit, final_checkpoint.get("seed") == seed, "final_checkpoint_seed", branch_where,
                   "final checkpoint seed differs from case seed")
        except Exception as exc:
            audit.fail("final_package_audit", branch_where,
                       "final checkpoint, actor, or simulator could not be audited", error=repr(exc))
        _check(audit, final_capacity is not None and post_capacity is not None,
               "package_audit_complete", branch_where, "one or more saved package byte checks failed")
        if final_capacity is not None:
            _check(audit, int(branch_summary.get("actor_payload_bytes", -1)) == final_capacity["actor_bytes"],
                   "branch_actor_payload_bytes", branch_where,
                   "branch actor payload byte count differs from its actual file")
        if branch_record is not None:
            branch_cpu = branch_summary.get("costs", {}).get("cpu_s")
            branch_wall = branch_summary.get("costs", {}).get("wall_s")
            if _finite_nonnegative(branch_cpu):
                branch_cpu_values.append(float(branch_cpu))
            else:
                audit.fail("branch_cpu_unknown", branch_where,
                           "branch CPU measurement is missing or invalid", value=branch_cpu)
            if _finite_nonnegative(branch_wall):
                branch_wall_values.append(float(branch_wall))
            else:
                audit.fail("branch_wall_unknown", branch_where,
                           "branch wall measurement is missing or invalid", value=branch_wall)
        branch_rows_total += branch_steps

    _check(audit, len(branch_records) == len(EXPECTED_ACTIONS), "branch_count", where,
           "case does not contain four matched-future branches", actual=len(branch_records),
           expected=len(EXPECTED_ACTIONS))
    case_costs = record.get("costs", {})
    expected_case_total = prefix_steps + branch_rows_total
    _check(audit, case_costs.get("prefix_environment_steps") == prefix_steps
           and case_costs.get("branch_environment_steps") == branch_rows_total
           and case_costs.get("environment_steps_total") == expected_case_total,
           "case_cost_sums", where, "case cost fields differ from persisted prefix and branch raw rows",
           recorded={key: case_costs.get(key) for key in (
               "prefix_environment_steps", "branch_environment_steps", "environment_steps_total")},
           observed={"prefix_environment_steps": prefix_steps,
                     "branch_environment_steps": branch_rows_total,
                     "environment_steps_total": expected_case_total})
    prefix_costs = prefix_summary.get("costs", {})
    for actual_key, expected_value in (("prefix_cpu_s", prefix_costs.get("cpu_s")),
                                       ("prefix_wall_s", prefix_costs.get("wall_s"))):
        value = case_costs.get(actual_key)
        _check(audit, _finite_nonnegative(value) and _finite_nonnegative(expected_value)
               and math.isclose(float(value), float(expected_value), abs_tol=1e-7, rel_tol=1e-7),
               "prefix_timing_sum", where, "case prefix timing differs from its prefix summary",
               field=actual_key, recorded=value, prefix_summary=expected_value)
    for key, values in (("branch_cpu_s", branch_cpu_values), ("branch_wall_s", branch_wall_values)):
        actual = sum(values)
        _check(audit, len(values) == len(EXPECTED_ACTIONS)
               and _finite_nonnegative(case_costs.get(key))
               and math.isclose(float(case_costs[key]), actual, abs_tol=1e-7, rel_tol=1e-7),
               "branch_timing_sum", where, "case branch timing differs from branch summaries",
               field=key, recorded=case_costs.get(key), observed=actual)
    return {"case_id": case_id, "passed": len(audit.errors) == error_start,
            "errors": len(audit.errors) - error_start,
            "prefix_environment_steps": prefix_steps,
            "branch_environment_steps": branch_rows_total,
            "environment_steps_total": expected_case_total,
            "branch_count": len(branch_records)}


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
        return math.isfinite(number) and number >= 0.0
    except (TypeError, ValueError):
        return False


def audit_cohort(cohort: str | Path) -> dict[str, Any]:
    """Read and audit one completed pilot or main cohort."""
    cohort = Path(cohort).resolve()
    audit = storage_audit.Audit(cohort)
    summary_path = cohort / "summary.json"
    try:
        summary = storage_audit._json(summary_path)
    except Exception as exc:
        audit.fail("summary_read", str(summary_path), "cohort summary is missing or unreadable", error=repr(exc))
        return {"format": FORMAT, "source": str(cohort), "passed": False,
                "errors": audit.errors, "warnings": audit.warnings}
    _check(audit, summary.get("format") == SUMMARY_FORMAT, "summary_format", str(summary_path),
           "cohort summary format differs from storage-decision output")
    _check(audit, summary.get("analysis_role") == "independent_surrogate_lower_confirmation",
           "summary_analysis_role", str(summary_path), "summary analysis role is incorrect")
    phase = str(summary.get("phase", ""))
    config, original = _load_phase_config(audit, cohort, summary)
    foundation_bytes = _validate_snapshot(audit, cohort, summary)
    if config is None or foundation_bytes is None:
        return {"format": FORMAT, "source": str(cohort), "phase": phase,
                "passed": False, "errors": audit.errors, "warnings": audit.warnings}

    seeds = tuple(int(seed) for seed in config["seeds"])
    histories = tuple(str(item) for item in config["history_regimes"])
    expected = expected_plan(phase, seeds=seeds, histories=histories,
                             bootstrap_steps=int(config["bootstrap_steps"]),
                             bout_steps=int(config["bout_steps"]))
    expected_case_ids = {f"s{seed}_dynamics_{history}" for seed in seeds for history in histories}
    expected_records = list(summary.get("cases", []))
    record_ids = [str(record.get("case_id", "")) for record in expected_records]
    _check(audit, len(record_ids) == len(set(record_ids)), "duplicate_case_records", str(summary_path),
           "summary contains duplicate case records")
    _check(audit, set(record_ids) == expected_case_ids, "case_universe", str(summary_path),
           "case records differ from frozen seeds and histories",
           actual=sorted(record_ids), expected=sorted(expected_case_ids))
    _check(audit, int(summary.get("case_count", -1)) == expected["case_count"], "case_count",
           str(summary_path), "summary case count differs from frozen budget",
           actual=summary.get("case_count"), expected=expected["case_count"])
    _check(audit, int(summary.get("branch_count", -1)) == expected["branch_count"], "branch_count",
           str(summary_path), "summary branch count differs from frozen budget",
           actual=summary.get("branch_count"), expected=expected["branch_count"])
    _check(audit, not summary.get("errors"), "runner_errors", str(summary_path),
           "runner summary records failures", errors=summary.get("errors"))

    actual_case_dirs = {path.name for path in (cohort / "cases").iterdir() if path.is_dir()} if (cohort / "cases").is_dir() else set()
    _check(audit, actual_case_dirs == expected_case_ids, "case_directory_universe", str(cohort / "cases"),
           "case directories differ from the frozen case universe",
           actual=sorted(actual_case_dirs), expected=sorted(expected_case_ids))

    records_by_id = {str(row.get("case_id", "")): row for row in expected_records}
    case_reports: list[dict[str, Any]] = []
    expected_raw_paths: set[Path] = set()
    for case_id in sorted(expected_case_ids):
        record = records_by_id.get(case_id)
        if record is None:
            audit.fail("case_record_missing", case_id, "expected case has no summary record")
            continue
        case_report = _audit_case(audit, cohort, record, config, foundation_bytes,
                                  phase=phase,
                                  expected_future=str(record.get("history_regime", "")))
        case_reports.append(case_report)
        case_dir = cohort / "cases" / case_id
        expected_raw_paths.add((case_dir / "prefix" / "steps.csv").resolve())
        for action in EXPECTED_ACTIONS:
            expected_raw_paths.add((case_dir / "branches" /
                                     f"{action}__{record.get('history_regime')}" / "steps.csv").resolve())

    flattened = [branch for record in expected_records for branch in record.get("branches", [])]
    _check(audit, storage_audit._close(summary.get("branches"), flattened, atol=0.0, rtol=0.0),
           "branch_flattening", str(summary_path),
           "top-level branch records differ from case-level branch records")

    observed_total, partial_total = _raw_inventory(audit, cohort, expected_raw_paths)
    if observed_total is not None:
        _check(audit, observed_total == expected["environment_steps_total"], "protocol_raw_cost",
               str(summary_path), "all persisted raw rows differ from the frozen interaction budget",
               observed=observed_total, expected=expected["environment_steps_total"])

    def sum_case_cost(field: str) -> int | None:
        values = [row.get("costs", {}).get(field) for row in expected_records]
        if any(value is None for value in values):
            return None
        try:
            return int(sum(int(value) for value in values))
        except (TypeError, ValueError):
            return None

    observed_prefix = sum_case_cost("prefix_environment_steps")
    observed_branches = sum_case_cost("branch_environment_steps")
    observed_record_total = sum_case_cost("environment_steps_total")
    top_costs = summary.get("costs", {})
    for field, expected_value in (("prefix_environment_steps", observed_prefix),
                                  ("branch_environment_steps", observed_branches),
                                  ("environment_steps_total", observed_record_total)):
        _check(audit, top_costs.get(field) == expected_value, "cohort_cost_sum", str(summary_path),
               "cohort step-cost sum differs from its case records", field=field,
               recorded=top_costs.get(field), observed=expected_value)
    if observed_total is not None:
        _check(audit, top_costs.get("environment_steps_total") == observed_total,
               "cohort_raw_cost_sum", str(summary_path),
               "cohort interaction cost differs from all persisted raw step rows",
               recorded=top_costs.get("environment_steps_total"), observed=observed_total)
    _check(audit, observed_prefix == expected["prefix_environment_steps"], "cohort_prefix_budget",
           str(summary_path), "case prefix cost sum differs from frozen cohort budget",
           observed=observed_prefix, expected=expected["prefix_environment_steps"])
    _check(audit, observed_branches == expected["branch_environment_steps"], "cohort_branch_budget",
           str(summary_path), "case branch cost sum differs from frozen cohort budget",
           observed=observed_branches, expected=expected["branch_environment_steps"])
    _check(audit, observed_record_total == expected["environment_steps_total"], "cohort_total_budget",
           str(summary_path), "case total cost sum differs from frozen cohort budget",
           observed=observed_record_total, expected=expected["environment_steps_total"])

    case_cpu = [row.get("costs", {}).get("case_cpu_s") for row in expected_records]
    case_prefix_cpu = [row.get("costs", {}).get("prefix_cpu_s") for row in expected_records]
    case_branch_cpu = [row.get("costs", {}).get("branch_cpu_s") for row in expected_records]
    if all(_finite_nonnegative(value) for value in case_cpu + case_prefix_cpu + case_branch_cpu):
        expected_child_cpu = sum(float(value) for value in case_cpu)
        expected_prefix_cpu = sum(float(value) for value in case_prefix_cpu)
        expected_branch_cpu = sum(float(value) for value in case_branch_cpu)
        _check(audit, _close_number(top_costs.get("child_cpu_s"), expected_child_cpu),
               "cohort_child_cpu_sum", str(summary_path), "child CPU total differs from case CPU values")
        _check(audit, _close_number(top_costs.get("prefix_cpu_s"), expected_prefix_cpu),
               "cohort_prefix_cpu_sum", str(summary_path), "prefix CPU total differs from case prefix CPU values")
        _check(audit, _close_number(top_costs.get("branch_cpu_s"), expected_branch_cpu),
               "cohort_branch_cpu_sum", str(summary_path), "branch CPU total differs from case branch CPU values")
        _check(audit, _finite_nonnegative(top_costs.get("orchestrator_cpu_s"))
               and _close_number(top_costs.get("cpu_s"), expected_child_cpu + float(top_costs["orchestrator_cpu_s"])),
               "cohort_cpu_sum", str(summary_path), "cohort CPU total differs from child plus orchestrator CPU")
    else:
        audit.fail("cohort_cpu_unknown", str(summary_path),
                   "one or more case CPU measurements are missing or invalid")

    return {
        "format": FORMAT,
        "source": str(cohort),
        "phase": phase,
        "passed": not audit.errors,
        "expected": expected,
        "observed": {
            "case_count": len(record_ids),
            "branch_count": len(flattened),
            "persisted_raw_steps": observed_total,
            "persisted_raw_steps_partial": partial_total,
            "recorded_prefix_steps": observed_prefix,
            "recorded_branch_steps": observed_branches,
            "recorded_total_steps": observed_record_total,
        },
        "cases": case_reports,
        "errors": audit.errors,
        "warnings": audit.warnings,
    }


def _close_number(actual: Any, expected: float) -> bool:
    return _finite_nonnegative(actual) and math.isclose(float(actual), expected, abs_tol=1e-7, rel_tol=1e-7)


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing audit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                      encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path,
                        help="completed lower-confirmation pilot or main cohort")
    parser.add_argument("--output", required=True, type=Path,
                        help="new audit JSON path outside the cohort")
    args = parser.parse_args(argv)
    cohort = args.cohort.resolve()
    output = args.output.resolve()
    if output == cohort or cohort in output.parents:
        raise SystemExit("audit output must be outside the cohort")
    report = audit_cohort(cohort)
    _write_report(output, report)
    print(json.dumps({
        "status": "passed" if report.get("passed") else "failed",
        "phase": report.get("phase"),
        "output": str(output),
        "cases": report.get("observed", {}).get("case_count"),
        "branches": report.get("observed", {}).get("branch_count"),
        "raw_steps": report.get("observed", {}).get("persisted_raw_steps"),
        "raw_steps_partial": report.get("observed", {}).get("persisted_raw_steps_partial"),
        "errors": len(report.get("errors", [])),
    }, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

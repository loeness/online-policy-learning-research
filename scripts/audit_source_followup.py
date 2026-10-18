"""Independently audit source-dose and learning-interaction cohorts.

This audit reads saved artifacts only. It never creates or steps an environment.
It re-derives candidate means and compensated actor writes from saved tensors,
checks raw interaction ledgers, and counts every persisted step row, including
rows left by failed branches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.storage_decision import pair_mapping, prefix_schedule, semantic_hash, template_schedule
from scripts import audit_storage_decision as old_audit


FORMAT = "source_followup_audit_v1"
SOURCE_FORMAT = "storage_source_dose_summary_v1"
ALIAS_FORMAT = "storage_source_dose_branch_alias_v1"
DOSE_BRANCH_FORMAT = "storage_source_dose_branch_v1"
EXPECTED_ACTIONS = ("keep", "replace", "mean", "none")
EXPECTED_FUTURES = ("old_major", "new_major")
PRIVATE_COMPONENTS = {"agent.md", "agents.md", "tmp"}
LEARNER_LEAK_KEYS = ("old_candidate", "new_candidate", "source_candidate", "source_capture", "observer_candidate", "candidate_workspace", "displaced_parent", "backup_parent")


class Audit:
    def __init__(self, path: Path) -> None:
        self.path = path
        # The original storage-decision audit helper accepts its own Audit
        # class shape and derives case paths from ``cohort``.  Expose the same
        # attribute so it can write its findings into this audit accumulator.
        self.cohort = path
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def fail(self, code: str, where: str, message: str, **details: Any) -> None:
        row: dict[str, Any] = {"code": code, "where": where, "message": message}
        if details:
            row["details"] = details
        self.errors.append(row)

    def warn(self, code: str, where: str, message: str, **details: Any) -> None:
        row: dict[str, Any] = {"code": code, "where": where, "message": message}
        if details:
            row["details"] = details
        self.warnings.append(row)


def _check(audit: Audit, condition: bool, code: str, where: str, message: str, **details: Any) -> None:
    if not condition:
        audit.fail(code, where, message, **details)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint must contain a mapping, got {type(value).__name__}")
    return value


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_relative(root: Path, name: str) -> Path | None:
    parts = Path(str(name).replace("\\", "/")).parts
    if not parts or Path(*parts).is_absolute() or any(part in ("", ".", "..") for part in parts):
        return None
    candidate = (root / Path(*parts)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        return None
    return candidate


def _private_hash_matches(snapshot: Path) -> list[str]:
    """Return only snapshot paths whose bytes exactly match local rule files."""
    references: dict[str, str] = {}
    for name in ("Agent.md", "AGENTS.md"):
        source = ROOT / name
        if source.is_file():
            references[_digest(source)] = name
    if not references or not snapshot.is_dir():
        return []
    matches: list[str] = []
    for path in snapshot.rglob("*"):
        if not path.is_file():
            continue
        if _digest(path) in references:
            matches.append(path.relative_to(snapshot).as_posix())
    return matches


def _validate_source_snapshot(audit: Audit, cohort: Path, summary: Mapping[str, Any]) -> int | None:
    snapshot = cohort / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        audit.fail("source_manifest_missing", str(manifest_path), "source snapshot manifest is missing")
        return None
    try:
        manifest = _json(manifest_path)
    except Exception as exc:
        audit.fail("source_manifest_parse", str(manifest_path), "source snapshot manifest is unreadable", error=repr(exc))
        return None
    _check(audit, manifest == summary.get("source_manifest"), "source_manifest_mismatch", "summary.json",
           "cohort summary and persisted source manifests differ")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        audit.fail("source_manifest_empty", str(manifest_path), "source manifest has no source file records")
        return None
    for name, record in sources.items():
        rel = Path(str(name).replace("\\", "/"))
        private_path = any(part.casefold() in PRIVATE_COMPONENTS for part in rel.parts)
        _check(audit, not private_path, "private_source_listed", f"source_snapshot/{name}",
               "source manifest lists a local instruction or temporary path")
        copied = _safe_relative(snapshot, name)
        if copied is None:
            audit.fail("source_path_escape", f"source_snapshot/{name}", "source manifest path escapes the snapshot")
            continue
        if not copied.is_file():
            audit.fail("source_snapshot_file_missing", f"source_snapshot/{name}", "manifest-listed source copy is missing")
            continue
        expected_bytes = record.get("bytes") if isinstance(record, Mapping) else None
        expected_hash = record.get("sha256") if isinstance(record, Mapping) else None
        actual_hash = _digest(copied)
        _check(audit, copied.stat().st_size == expected_bytes and actual_hash == expected_hash,
               "source_snapshot_hash", f"source_snapshot/{name}",
               "manifest-listed source copy differs from its size or SHA-256")
    for path in snapshot.rglob("*"):
        rel = path.relative_to(snapshot)
        _check(audit, not any(part.casefold() in PRIVATE_COMPONENTS for part in rel.parts),
               "private_source_present", path.relative_to(cohort).as_posix(),
               "source snapshot contains a local instruction or temporary path")
    excluded = {str(item).casefold() for item in manifest.get("excluded_local_instructions", [])}
    _check(audit, excluded == {"agent.md", "agents.md"}, "private_exclusion_record", str(manifest_path),
           "source manifest does not record both local instruction exclusions")
    private_matches = _private_hash_matches(snapshot)
    _check(audit, not private_matches, "private_exact_hash_match", "source_snapshot",
           "source snapshot contains a byte-for-byte copy of a local instruction file",
           matching_paths=private_matches)
    foundation = manifest.get("foundation", {})
    if not isinstance(foundation, Mapping):
        audit.fail("foundation_manifest", str(manifest_path), "foundation provenance record is malformed")
        return None
    try:
        foundation_bytes = int(foundation.get("bytes", -1))
    except (TypeError, ValueError):
        foundation_bytes = -1
    foundation_path = Path(str(foundation.get("path", "")))
    if foundation_path.is_file():
        _check(audit, foundation_path.stat().st_size == foundation_bytes
               and _digest(foundation_path) == foundation.get("sha256"),
               "foundation_hash", str(foundation_path), "W0 foundation differs from the source manifest")
    else:
        audit.warn("foundation_source_unavailable", str(foundation_path),
                   "W0 source file is unavailable; package byte counts remain auditable from the manifest")
    return foundation_bytes if foundation_bytes >= 0 else None


def _expected_config(audit: Audit, cohort: Path, summary: Mapping[str, Any]) -> dict[str, Any] | None:
    config_path = cohort / "config.json"
    original_path = cohort / "source_snapshot" / "config_original.json"
    try:
        config = _json(config_path)
        original = _json(original_path)
    except Exception as exc:
        audit.fail("config_missing", str(cohort), "cohort config or original config cannot be read", error=repr(exc))
        return None
    phase = str(summary.get("phase", ""))
    expected = dict(original)
    capture_steps = original.get("source_capture_steps")
    if phase == "pilot":
        pilot = original.get("pilot")
        if not isinstance(pilot, Mapping):
            audit.fail("pilot_config_missing", str(original_path), "pilot config is missing")
            return None
        for key in ("seeds", "pair_kinds", "history_regimes", "bootstrap_steps", "bout_steps"):
            expected[key] = pilot[key]
        capture_steps = original.get("pilot_source_capture_steps", pilot.get("source_capture_steps"))
    elif phase != "main":
        audit.fail("phase_invalid", "summary.json", "source cohort phase must be main or pilot", phase=phase)
        return None
    expected["source_capture_steps"] = int(capture_steps or 0)
    try:
        expected = old_audit.runner.validate_config(expected)
        expected["source_capture_steps"] = int(capture_steps or 0)
    except Exception as exc:
        audit.fail("config_invalid", str(original_path), "frozen source-dose config is invalid", error=repr(exc))
        return None
    _check(audit, old_audit._close(config, expected, atol=0.0, rtol=0.0), "config_phase_mismatch", "config.json",
           "persisted phase config differs from the original frozen config")
    _check(audit, old_audit._close(summary.get("config"), config, atol=0.0, rtol=0.0),
           "summary_config_mismatch", "summary.json", "summary config differs from config.json")
    _check(audit, tuple(config.get("actions", ())) == EXPECTED_ACTIONS, "action_universe", "config.json",
           "action grid differs from the four frozen actions")
    _check(audit, tuple(config.get("future_regimes", ())) == EXPECTED_FUTURES, "future_universe", "config.json",
           "future grid differs from the two frozen futures")
    expected_capture = int(config.get("bootstrap_steps", 0)) // 2 if phase == "pilot" else 1024
    _check(audit, int(config.get("source_capture_steps", -1)) == expected_capture,
           "source_capture_budget", "config.json", "source capture step count differs from the frozen phase",
           actual=config.get("source_capture_steps"), expected=expected_capture)
    return config


def _mean_capture_27(captures: Mapping[str, Any]) -> np.ndarray:
    if set(map(str, captures)) != {"0", "1"}:
        raise ValueError("capture records must include variants 0 and 1")
    matrices: list[np.ndarray] = []
    for variant in ("0", "1"):
        record = captures[variant]
        values = np.asarray(record["effective_matrix_27"], dtype=np.float32)
        if values.shape != (27,) or not np.isfinite(values).all():
            raise ValueError(f"variant {variant} must have 27 finite captured coefficients")
        matrices.append(values.reshape(3, 3, 3))
    return np.stack(matrices, axis=0).mean(axis=(0, 1), dtype=np.float64).astype(np.float32)


def _checkpoint_effective(checkpoint: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local, shared, physical = old_audit._core_arrays(checkpoint)
    effective = local.reshape(-1, 3, 3, 3).copy()
    for slot, group in enumerate(physical.tolist()):
        if int(group) == 0:
            effective[slot] += shared.reshape(1, 3, 3)
        elif int(group) != -1:
            raise ValueError("checkpoint has an unsupported physical shared-parent id")
    tags = np.asarray(checkpoint["slot_group_tags"], dtype=np.int8)
    if tags.shape != (len(local),):
        raise ValueError("logical group tags do not match checkpoint local slots")
    return local, shared, physical, effective


def _independent_compensated_action(
    local: np.ndarray,
    shared: np.ndarray,
    physical: np.ndarray,
    logical: np.ndarray,
    old_candidate: np.ndarray,
    new_candidate: np.ndarray,
    action: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Re-derive action binding/compensation without importing apply_action."""
    current = local.reshape(-1, 3, 3, 3).astype(np.float32).copy()
    for slot, group in enumerate(physical.tolist()):
        if int(group) == 0:
            current[slot] += shared.reshape(1, 3, 3)
        elif int(group) != -1:
            raise ValueError("physical group ids must be -1 or zero")
    old = np.asarray(old_candidate, dtype=np.float32).reshape(3, 3)
    new = np.asarray(new_candidate, dtype=np.float32).reshape(3, 3)
    if action == "keep":
        target, parent_group = old, 0
        assigned = logical == 0
    elif action == "replace":
        target, parent_group = new, 1
        assigned = logical == 1
    elif action == "mean":
        target, parent_group = ((old + new) / np.float32(2.0)).astype(np.float32), -2
        assigned = np.ones(len(local), dtype=bool)
    elif action == "none":
        target, parent_group = np.zeros((3, 3), dtype=np.float32), -1
        assigned = np.zeros(len(local), dtype=bool)
    else:
        raise ValueError(f"unknown action {action!r}")
    output_shared = target.reshape(1, 9).copy()
    output_local = current.copy()
    output_local[assigned] -= target.reshape(1, 1, 3, 3)
    output_groups = np.where(assigned, 0, -1).astype(np.int8)
    before = current.copy()
    return output_local.reshape(-1, 27), output_shared, output_groups, parent_group, before


def _scan_learner_keys(audit: Audit, checkpoint: Any, where: str) -> None:
    found: list[str] = []

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                name = str(key)
                folded = name.casefold()
                if any(part in folded for part in LEARNER_LEAK_KEYS):
                    found.append(f"{prefix}.{name}" if prefix else name)
                walk(nested, f"{prefix}.{name}" if prefix else name)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                walk(nested, f"{prefix}[{index}]")

    walk(checkpoint, "")
    _check(audit, not found, "learner_candidate_backup_leak", where,
           "learner checkpoint contains candidate/observer backup fields", fields=found)


def _same_metadata(audit: Audit, actual: Any, expected: Any, code: str, where: str, message: str) -> None:
    _check(audit, old_audit._close(actual, expected, atol=0.0, rtol=0.0), code, where, message)


def _read_step_count(path: Path) -> tuple[int | None, list[dict[str, str]] | None, str | None]:
    if not path.is_file():
        return None, None, "missing"
    try:
        rows = _csv_rows(path)
        return len(rows), rows, None
    except Exception as exc:
        return None, None, repr(exc)


def _capture_checks(
    audit: Audit,
    *,
    case_dir: Path,
    config: Mapping[str, Any],
    base_decision: Mapping[str, Any],
    dose_decision: Mapping[str, Any],
    observer: Mapping[str, Any],
    prefix_rows: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    where = str(case_dir / "dose_1024")
    records = dose_decision.get("source_capture_records", {})
    observer_sources = observer.get("sources", {})
    _same_metadata(audit, observer_sources, records, "capture_record_copy", where,
                   "observer JSON and dose decision contain different capture records")
    try:
        old_1024 = _mean_capture_27(records)
        candidate = np.asarray(dose_decision.get("old_candidate_1024"), dtype=np.float32).reshape(3, 3)
        observer_candidate = np.asarray(observer.get("candidate_effective_matrix_3x3"), dtype=np.float32).reshape(3, 3)
        _check(audit, np.allclose(old_1024, candidate, atol=1e-7, rtol=0.0),
               "old_1024_capture_mean", where,
               "old 1024 candidate is not the independent mean over two 27-coefficient captures and three bands")
        _check(audit, np.allclose(old_1024, observer_candidate, atol=1e-7, rtol=0.0),
               "observer_candidate_mean", where,
               "observer candidate is not the independent source-capture mean")
    except Exception as exc:
        audit.fail("old_1024_capture_parse", where, "old 1024 capture coefficients cannot be independently reduced", error=repr(exc))
        old_1024 = None
    _check(audit, observer.get("format") == "storage_source_dose_observer_v1", "observer_format", where,
           "observer candidate format is unexpected")
    _check(audit, int(observer.get("candidate_coefficients", -1)) == 9, "observer_candidate_size", where,
           "observer candidate must contain nine stored parent coefficients")
    _check(audit, observer.get("candidate_storage") == "observer only; never attached to learner or deployment package",
           "observer_not_learner", where, "candidate must remain observer-only")
    capture_steps = int(config["source_capture_steps"])
    _check(audit, int(observer.get("source_capture_steps", -1)) == capture_steps,
           "observer_capture_steps", where, "observer capture point differs from phase config")
    _check(audit, observer.get("capture_timing") == "after PPO update, int8 actor roundtrip, and actor optimizer-state clear",
           "capture_timing", where, "capture timing differs from the frozen observer hook")
    try:
        boot = list(base_decision["bootstrap_schedule"])
        _check(audit, len(boot) == 2 and all(row.get("phase") == "bootstrap" for row in boot),
               "bootstrap_capture_schedule", where, "the source capture is not based on the two bootstrap bouts")
        rollout_steps = int(config["rollout_steps"])
        expected_rollout_index = capture_steps // rollout_steps - 1
        rollout_rows = _csv_rows(case_dir / "prefix" / "rollouts.csv")
        for variant in (0, 1):
            key = str(variant)
            record = records[key]
            expected_bout = boot[variant]
            _check(audit, int(record.get("variant", -1)) == variant
                   and int(record.get("group", -1)) == 0
                   and record.get("task") == expected_bout.get("task")
                   and int(record.get("bout", -1)) == int(expected_bout.get("bout", -2)),
                   "capture_source_identity", f"{where}/source_{variant}",
                   "capture does not identify the corresponding old bootstrap source")
            _check(audit, int(record.get("bout_step", -1)) == capture_steps,
                   "capture_bout_step", f"{where}/source_{variant}", "capture does not occur at the frozen source step")
            expected_global = variant * int(config["bootstrap_steps"]) + capture_steps
            _check(audit, int(record.get("global_environment_steps", -1)) == expected_global,
                   "capture_global_step", f"{where}/source_{variant}",
                   "capture global environment-step count differs from its bootstrap position",
                   actual=record.get("global_environment_steps"), expected=expected_global)
            _check(audit, int(record.get("rollout_rows", -1)) == rollout_steps
                   and int(record.get("rollout_index", -1)) == expected_rollout_index,
                   "capture_rollout_boundary", f"{where}/source_{variant}",
                   "capture does not land on the completed rollout boundary")
            global_index = expected_global - 1
            if 0 <= global_index < len(prefix_rows):
                row = prefix_rows[global_index]
                for field, expected in (("phase", "bootstrap"), ("taskobserver", expected_bout["task"]),
                                        ("group", "0"), ("bout", str(expected_bout["bout"])),
                                        ("bout_step", str(capture_steps)),
                                        ("slot", str(record.get("slot"))),
                                        ("generation", str(record.get("generation")))):
                    _check(audit, row.get(field) == expected, "capture_raw_source_phase",
                           f"{where}/source_{variant}",
                           "capture metadata does not match the raw bootstrap row", field=field)
            else:
                audit.fail("capture_raw_row_missing", f"{where}/source_{variant}",
                           "capture has no corresponding persisted raw step")
            matching = [row for row in rollout_rows if row.get("phase") == "bootstrap"
                        and (int(row.get("bout", -1)) < int(expected_bout["bout"])
                             or (int(row.get("bout", -1)) == int(expected_bout["bout"])
                                 and int(row.get("rollout_index", -1)) <= expected_rollout_index))]
            _check(audit, int(record.get("rollouts_trained", -1)) == len(matching),
                   "capture_rollouts_trained", f"{where}/source_{variant}",
                   "capture trained-rollout count differs from the raw rollout ledger",
                   actual=record.get("rollouts_trained"), expected=len(matching))
            expected_updates = sum(int(row.get("updates", 0)) for row in matching)
            _check(audit, int(record.get("minibatch_updates", -1)) == expected_updates,
                   "capture_minibatch_updates", f"{where}/source_{variant}",
                   "capture minibatch-update count differs from raw rollout rows",
                   actual=record.get("minibatch_updates"), expected=expected_updates)
            current_slot = int(record.get("slot", -1))
            current_generation = int(record.get("generation", -1))
            fitted_expected = 0
            for rollout in matching:
                if int(rollout.get("updates", 0)) <= 0:
                    continue
                bout = int(rollout["bout"])
                index = int(rollout["rollout_index"])
                first = index * rollout_steps + 1
                last = (index + 1) * rollout_steps
                fitted_expected += sum(
                    int(step.get("slot", -2)) == current_slot
                    and int(step.get("generation", -2)) == current_generation
                    and int(step.get("bout", -2)) == bout
                    and first <= int(step.get("bout_step", -1)) <= last
                    and step.get("phase") == "bootstrap"
                    for step in prefix_rows
                )
            _check(audit, int(record.get("fitted_steps", -1)) == fitted_expected,
                   "capture_fitted_steps", f"{where}/source_{variant}",
                   "captured fitted_steps differs from successful PPO rows for that slot generation",
                   actual=record.get("fitted_steps"), expected=fitted_expected)
            _check(audit, int(record.get("last_rollout_updates", -1)) == int(matching[-1].get("updates", -2))
                   if matching else False,
                   "capture_last_rollout", f"{where}/source_{variant}",
                   "captured update count differs from the last completed rollout")
            _check(audit, int(record.get("physical_group", -2)) in (-1, 0),
                   "capture_physical_group", f"{where}/source_{variant}",
                   "capture physical parent binding is invalid")
            _check(audit, int(record.get("generation", -1)) >= 1,
                   "capture_generation", f"{where}/source_{variant}",
                   "capture has no allocated slot generation")
            _check(audit, int(record.get("slot", -1)) in (0, 1),
                   "capture_slot", f"{where}/source_{variant}", "capture references an invalid local slot")
    except Exception as exc:
        audit.fail("capture_timing_parse", where, "capture timing evidence cannot be checked", error=repr(exc))
    new_candidate: np.ndarray | None = None
    old_2048: np.ndarray | None = None
    try:
        prefix_checkpoint = _checkpoint(case_dir / "prefix" / "checkpoint.pt")
        local, shared, physical, effective = _checkpoint_effective(prefix_checkpoint)
        tags = np.asarray(prefix_checkpoint["slot_group_tags"], dtype=np.int8)
        new_slots = np.flatnonzero(tags == 1)
        _check(audit, len(new_slots) == 2, "new_candidate_source_count", str(case_dir),
               "new candidate must be formed from two live new-source slots")
        if len(new_slots) == 2:
            new_candidate = effective[new_slots].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
            recorded_new = np.asarray(base_decision.get("new_candidate"), dtype=np.float32).reshape(3, 3)
            dose_new = np.asarray(dose_decision.get("new_candidate"), dtype=np.float32).reshape(3, 3)
            _check(audit, np.allclose(new_candidate, recorded_new, atol=1e-7, rtol=0.0)
                   and np.allclose(new_candidate, dose_new, atol=1e-7, rtol=0.0),
                   "new_candidate_source_mean", str(case_dir),
                   "new candidate differs from the two checkpoint source vectors and three-band mean")
        old_2048 = np.asarray(base_decision.get("old_parent"), dtype=np.float32).reshape(3, 3)
        recorded_old_2048 = np.asarray(dose_decision.get("old_candidate_2048"), dtype=np.float32).reshape(3, 3)
        source_shared = shared[0].reshape(3, 3)
        _check(audit, np.allclose(old_2048, recorded_old_2048, atol=0.0, rtol=0.0)
               and np.allclose(old_2048, source_shared, atol=1e-6, rtol=0.0),
               "old_2048_bootstrap_parent", str(case_dir),
               "old 2048 candidate differs from the original bootstrap parent retained in the common prefix")
        _check(audit, dose_decision.get("selector_input_candidate_set") == "old_candidate_1024",
               "selector_candidate_provenance", str(case_dir / "dose_1024"),
               "frozen low-dose selector does not record use of the independently captured 1024-step candidate")
        _check(audit, dose_decision.get("selectors_frozen_before_future_expansion") is True,
               "selector_freeze_timing", str(case_dir / "dose_1024"),
               "dose selectors were not recorded as frozen before future expansion")
        _check(audit, dose_decision.get("selector_freeze_timing") ==
               "both dose rules computed before any future expansion from common prefix evidence",
               "selector_freeze_contract", str(case_dir / "dose_1024"),
               "dose selector timing record differs from the common-prefix protocol")
        _check(audit, dose_decision.get("candidate_workspace_in_learner") is False,
               "candidate_workspace_isolated", str(case_dir / "dose_1024"),
               "observer candidates must not be stored in learner state")
        _check(audit, int(dose_decision.get("learner_parent_count", -1)) == 1,
               "learner_parent_count", str(case_dir / "dose_1024"),
               "dose decision learner must retain exactly one parent")
        return old_1024, old_2048, new_candidate
    except Exception as exc:
        audit.fail("candidate_origin_parse", str(case_dir), "candidate origin cannot be checked against saved learner", error=repr(exc))
        return old_1024, old_2048, new_candidate


def _candidate_action_check(
    audit: Audit,
    *,
    branch_dir: Path,
    prefix_checkpoint: Mapping[str, Any],
    prefix_simulator: Mapping[str, Any],
    dose_decision: Mapping[str, Any],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    foundation_bytes: int,
) -> tuple[str | None, str | None]:
    where = str(branch_dir)
    action = str(summary.get("action", ""))
    try:
        pre_local, pre_shared, pre_groups, _ = _checkpoint_effective(prefix_checkpoint)
        logical = np.asarray(prefix_checkpoint["slot_group_tags"], dtype=np.int8)
        old_candidate = np.asarray(dose_decision["old_candidate_1024"], dtype=np.float32).reshape(3, 3)
        new_candidate = np.asarray(dose_decision["new_candidate"], dtype=np.float32).reshape(3, 3)
        expected_local, expected_shared, expected_groups, expected_parent, before_effective = _independent_compensated_action(
            pre_local, pre_shared, pre_groups, logical, old_candidate, new_candidate, action,
        )
        expected_actor = old_audit.encode_actor(
            expected_local, expected_shared, expected_groups, codec="int8",
            budget_bytes=int(config["actor_budget_bytes"]),
        ).data
        decoded_local, decoded_shared, decoded_groups = old_audit.decode_actor(expected_actor)
        post_checkpoint = _checkpoint(branch_dir / str(summary.get("post_action_checkpoint", "post_action/checkpoint.pt")))
        actual_local, actual_shared, actual_groups = old_audit._core_arrays(post_checkpoint)
        _check(audit, np.array_equal(actual_local, decoded_local)
               and np.array_equal(actual_shared, decoded_shared)
               and np.array_equal(actual_groups, decoded_groups),
               "independent_compensated_action", where,
               "post-action checkpoint does not match independent action compensation and int8 roundtrip",
               action=action)
        actor_path = branch_dir / "post_action" / "actor.bin"
        _check(audit, actor_path.is_file() and actor_path.read_bytes() == expected_actor,
               "post_action_actor_exact", where, "post-action actor sidecar differs from independently encoded action")
        expected_parent_group = {"keep": 0, "replace": 1, "mean": -2, "none": -1}[action]
        _check(audit, int(post_checkpoint.get("parent_group", -99)) == expected_parent,
               "post_action_parent_group", where, "post-action parent eligibility differs from requested action")
        _check(audit, expected_parent == expected_parent_group, "action_parent_mapping", where,
               "independent action parent mapping is inconsistent")
        _check(audit, np.array_equal(np.asarray(post_checkpoint.get("slot_group_tags"), dtype=np.int8), logical),
               "post_action_logical_tags", where, "action changed known logical task tags")
        post_sim_path = branch_dir / str(summary.get("post_action_simulator", "post_action/simulator.json"))
        post_sim = _json(post_sim_path)
        _check(audit, old_audit._close(post_sim, prefix_simulator, atol=0.0, rtol=0.0),
               "post_action_simulator_restore", where, "action-time simulator state differs from the shared prefix")
        start_sim_path = branch_dir / "simulator_start.json"
        if start_sim_path.is_file():
            _check(audit, old_audit._close(_json(start_sim_path), prefix_simulator, atol=0.0, rtol=0.0),
                   "simulator_start_restore", where, "branch simulator start differs from common prefix")
        recomputed_action_hash = semantic_hash({"learner": post_checkpoint, "simulator": post_sim})
        prefix_hash = semantic_hash({"learner": prefix_checkpoint, "simulator": prefix_simulator})
        _check(audit, summary.get("prefix_state_hash") == prefix_hash
               and summary.get("restored_state_hash") == prefix_hash,
               "dose_prefix_restore_hash", where, "dose branch prefix/restore full-state hashes do not match")
        _check(audit, summary.get("action_state_hash") == recomputed_action_hash,
               "dose_action_full_hash", where, "dose branch post-action learner/simulator hash does not match")
        _check(audit, int(summary.get("actor_payload_bytes", -1)) == int(config["actor_budget_bytes"]),
               "actor_payload_bytes", where, "branch actor payload size differs from frozen budget")
        errors = np.max(np.abs(
            expected_local.reshape(-1, 3, 3, 3)
            + np.where(expected_groups[:, None, None, None] == 0, expected_shared.reshape(1, 1, 3, 3), 0.0)
            - before_effective
        ), axis=(1, 2, 3))
        recorded_errors = np.asarray(summary.get("prequantized_effective_errors", []), dtype=np.float64)
        _check(audit, recorded_errors.shape == errors.shape
               and np.allclose(recorded_errors, errors, atol=1e-6, rtol=1e-6),
               "action_compensation_record", where,
               "recorded action compensation errors differ from independent recomputation")
        _check(audit, float(np.max(errors, initial=0.0)) <= 1e-5,
               "action_compensation_preserves_residents", where,
               "action does not preserve every resident effective matrix before quantization",
               max_error=float(np.max(errors, initial=0.0)))
        _scan_learner_keys(audit, post_checkpoint, f"{where}/post_action/checkpoint.pt")
        _scan_learner_keys(audit, _checkpoint(branch_dir / "checkpoint.pt"), f"{where}/checkpoint.pt")
        old_audit._package_check(
            audit, folder=branch_dir / "post_action", actor_path=actor_path,
            checkpoint_path=branch_dir / "post_action" / str(summary.get("post_action_checkpoint", "post_action/checkpoint.pt")).split("/", 1)[-1],
            recorded=summary.get("capacity_after_action"), foundation_bytes=foundation_bytes,
            package_cap=int(config["package_budget_bytes"]), actor_cap=int(config["actor_budget_bytes"]),
            where=f"{where}/post_action",
        )
        final_checkpoint = _checkpoint(branch_dir / "checkpoint.pt")
        final_actor_path = branch_dir / "actor.bin"
        final_local, final_shared, final_groups = old_audit._core_arrays(final_checkpoint)
        final_actor_expected = old_audit.encode_actor(
            final_local, final_shared, final_groups, codec="int8",
            budget_bytes=int(config["actor_budget_bytes"]),
        ).data
        _check(audit, final_actor_path.is_file() and final_actor_path.read_bytes() == final_actor_expected,
               "final_actor_matches_checkpoint", where, "final actor sidecar differs from final checkpoint")
        final_sim = _json(branch_dir / "simulator.json")
        final_hash = semantic_hash({"learner": final_checkpoint, "simulator": final_sim})
        _check(audit, summary.get("final_state_hash") == final_hash, "final_full_state_hash", where,
               "final learner/simulator state does not match recorded hash")
        old_audit._package_check(
            audit, folder=branch_dir, actor_path=final_actor_path, checkpoint_path=branch_dir / "checkpoint.pt",
            recorded=summary.get("capacity"), foundation_bytes=foundation_bytes,
            package_cap=int(config["package_budget_bytes"]), actor_cap=int(config["actor_budget_bytes"]),
            where=f"{where}/final",
        )
        _check(audit, summary.get("foundation_unchanged") is True
               and summary.get("foundation_sha256") == dose_decision.get("foundation_sha256"),
               "dose_foundation_unchanged", where, "branch W0 digest is inconsistent or marked changed")
        return recomputed_action_hash, final_hash
    except Exception as exc:
        audit.fail("dose_action_audit_error", where,
                   "post-action state, compensation, checkpoint, or capacity could not be audited",
                   action=action, error=repr(exc))
        return None, None


def _audit_alias(
    audit: Audit,
    *,
    branch_dir: Path,
    case_dir: Path,
    summary: Mapping[str, Any],
    record: Mapping[str, Any],
    base_branch_summary: Mapping[str, Any],
    prefix_hash: str,
) -> tuple[dict[str, str] | None, int | None]:
    where = str(branch_dir)
    action = str(summary.get("action", ""))
    _check(audit, action in ("replace", "none"), "alias_action", where,
           "only candidate-independent replace/none branches may alias")
    _check(audit, summary.get("independent_observation") is False,
           "alias_observation", where, "alias must declare that it has no independent observation")
    expected_target = case_dir / "branches" / f"{action}__{summary.get('future_regime')}" / "summary.json"
    reuse = str(summary.get("reuse_of", ""))
    resolved = (branch_dir / reuse).resolve() if reuse else None
    _check(audit, resolved == expected_target.resolve(), "alias_target_path", where,
           "alias does not reference its same-case action/future source summary")
    target = base_branch_summary
    target_folder = expected_target.parent
    try:
        action_ckpt = _checkpoint(target_folder / str(target.get("post_action_checkpoint", "post_action/checkpoint.pt")))
        action_sim = _json(target_folder / str(target.get("post_action_simulator", "post_action/simulator.json")))
        action_hash = semantic_hash({"learner": action_ckpt, "simulator": action_sim})
        final_ckpt = _checkpoint(target_folder / "checkpoint.pt")
        final_sim = _json(target_folder / "simulator.json")
        final_hash = semantic_hash({"learner": final_ckpt, "simulator": final_sim})
        _check(audit, summary.get("action_state_hash") == action_hash
               and target.get("action_state_hash") == action_hash,
               "alias_complete_post_action_hash", where,
               "alias hash does not equal the complete source post-action learner and simulator hash")
        _check(audit, summary.get("prefix_state_hash") == prefix_hash
               and summary.get("restored_state_hash") == prefix_hash,
               "alias_prefix_restore_hash", where, "alias prefix/restore hashes differ from the common prefix")
        _check(audit, summary.get("logical_final_state_hash") == final_hash
               and target.get("final_state_hash") == final_hash,
               "alias_complete_final_hash", where, "alias logical final hash differs from source artifacts")
        for field, alias_field in (("environment_steps", "logical_environment_steps"),
                                   ("reward_total", "logical_reward_total"),
                                   ("reward_per_step", "logical_reward_per_step"),
                                   ("per_group", "logical_per_group"),
                                   ("per_bout", "logical_per_bout"),
                                   ("counts", "logical_counts"),
                                   ("capacity", "capacity"),
                                   ("capacity_after_action", "capacity_after_action")):
            _check(audit, old_audit._close(summary.get(alias_field), target.get(field), atol=0.0, rtol=0.0),
                   "alias_logical_copy", where, "alias logical metadata differs from the verified source branch",
                   field=alias_field)
        _check(audit, int(summary.get("environment_steps", -1)) == 0
               and summary.get("reward_total") is None and summary.get("reward_per_step") is None,
               "alias_physical_zero", where, "alias must have zero physical steps and no physical reward")
        _check(audit, summary.get("counts") == {
            "environment_steps": 0, "completed_episodes": 0, "slot_allocations": 0,
            "slot_evictions": 0, "rollouts_trained": 0, "minibatch_updates": 0,
        }, "alias_physical_counts", where, "alias physical counters must all be zero")
        costs = summary.get("costs", {})
        _check(audit, int(costs.get("simulation_cpu_s", -1)) == 0
               and float(costs.get("simulation_wall_s", -1)) == 0.0,
               "alias_simulation_cost", where, "alias must not record simulated branch cost")
        for name in ("steps.csv", "episodes.csv", "rollouts.csv"):
            _check(audit, not (branch_dir / name).exists(), "alias_raw_duplicate", str(branch_dir / name),
                   "alias contains duplicate raw branch artifacts")
        _scan_learner_keys(audit, action_ckpt, f"{target_folder}/post_action/checkpoint.pt")
        return action_ckpt, int(summary.get("logical_environment_steps", -1))
    except Exception as exc:
        audit.fail("alias_source_missing", where, "alias source artifacts are missing or unreadable", error=repr(exc))
        return None, None


def _audit_dose_case(
    audit: Audit,
    *,
    cohort: Path,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    foundation_bytes: int,
) -> dict[str, Any]:
    case_id = str(record.get("case_id", ""))
    case_dir = cohort / "cases" / case_id
    dose_dir = case_dir / "dose_1024"
    before = len(audit.errors)
    try:
        base_decision = _json(case_dir / "decision.json")
        dose_decision = _json(dose_dir / "decision.json")
        observer = _json(dose_dir / "observer_candidate.json")
        prefix_checkpoint = _checkpoint(case_dir / "prefix" / "checkpoint.pt")
        prefix_simulator = _json(case_dir / "prefix" / "simulator.json")
        prefix_rows = _csv_rows(case_dir / "prefix" / "steps.csv")
    except Exception as exc:
        audit.fail("dose_case_artifacts", case_id, "dose decision, observer, prefix, or raw rows are missing", error=repr(exc))
        return {"case_id": case_id, "passed": False, "errors": len(audit.errors) - before,
                "observed_actual_environment_steps": None, "physical_branch_count": None}
    _check(audit, dose_decision.get("format") == "storage_source_dose_case_v1", "dose_decision_format",
           str(dose_dir), "dose decision format is unexpected")
    _check(audit, int(record.get("seed", -1)) == int(dose_decision.get("seed", -2))
           and record.get("pair_kind") == dose_decision.get("pair_kind")
           and record.get("history_regime") == dose_decision.get("history_regime"),
           "dose_case_identity", case_id, "dose decision identity differs from cohort record")
    expected_prefix_hash = semantic_hash({"learner": prefix_checkpoint, "simulator": prefix_simulator})
    _check(audit, dose_decision.get("prefix_state_hash") == expected_prefix_hash
           and base_decision.get("prefix_state_hash") == expected_prefix_hash,
           "dose_prefix_full_hash", case_id, "dose and original decision do not match complete saved common prefix")
    candidates = _capture_checks(
        audit, case_dir=case_dir, config=config, base_decision=base_decision,
        dose_decision=dose_decision, observer=observer, prefix_rows=prefix_rows,
    )
    old_1024 = candidates[0]
    _scan_learner_keys(audit, prefix_checkpoint, f"{case_dir}/prefix/checkpoint.pt")
    expected_pairs = {(action, future) for action in EXPECTED_ACTIONS for future in EXPECTED_FUTURES}
    record_branches = list(record.get("source_1024_branches", []))
    result_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for branch_record in record_branches:
        key = (str(branch_record.get("action", "")), str(branch_record.get("future_regime", "")))
        if key in result_by_key:
            audit.fail("duplicate_dose_branch_record", case_id, "cohort record repeats a dose branch", branch=key)
        result_by_key[key] = branch_record
    _check(audit, set(result_by_key) == expected_pairs, "dose_branch_universe", case_id,
           "dose cohort record is missing or adding action/future branches",
           actual=sorted(result_by_key), expected=sorted(expected_pairs))
    decision_branches = list(dose_decision.get("branches", []))
    decision_keys = {(str(row.get("action", "")), str(row.get("future_regime", ""))) for row in decision_branches}
    _check(audit, decision_keys == expected_pairs and len(decision_branches) == len(expected_pairs),
           "dose_decision_branch_universe", str(dose_dir), "dose decision does not list each branch exactly once")
    mapping = pair_mapping(int(record["seed"]), str(record["pair_kind"]))
    prefix_sched = prefix_schedule(mapping, config, str(record["history_regime"]))
    prefix_steps = sum(int(row["steps"]) for row in prefix_sched)
    all_uniforms: dict[tuple[str, str], list[float]] = {}
    physical_dose_rows = 0
    physical_dose_known = True
    logical_dose_rows = 0
    logical_dose_known = True
    physical_dose_branches = 0
    logical_dose_branches = 0
    source_raw_files: set[Path] = set()
    for action, future in sorted(expected_pairs):
        key = (action, future)
        folder = dose_dir / "branches" / f"{action}__{future}"
        summary_path = folder / "summary.json"
        branch_record = result_by_key.get(key)
        decision_record = next((row for row in decision_branches
                                if row.get("action") == action and row.get("future_regime") == future), None)
        if not summary_path.is_file():
            audit.fail("dose_branch_summary_missing", str(summary_path), "expected dose branch summary is missing")
            logical_dose_known = False
            if action in ("keep", "mean"):
                physical_dose_known = False
            else:
                physical_dose_known = False
            continue
        try:
            branch_summary = _json(summary_path)
        except Exception as exc:
            audit.fail("dose_branch_summary_invalid", str(summary_path), "dose branch summary cannot be parsed", error=repr(exc))
            physical_dose_known = logical_dose_known = False
            continue
        if branch_record is not None:
            _same_metadata(audit, branch_record, branch_summary, "dose_summary_record_match", str(summary_path),
                           "case summary branch record differs from its on-disk branch summary")
        expected_rel = f"branches/{action}__{future}/summary.json"
        if decision_record is None:
            audit.fail("dose_decision_branch_missing", str(dose_dir), "dose decision omits an expected branch", branch=key)
        else:
            _check(audit, decision_record.get("summary") == expected_rel, "dose_decision_summary_path",
                   str(dose_dir), "dose decision points at a different branch summary", branch=key)
        _check(audit, branch_summary.get("case_id") == case_id
               and int(branch_summary.get("seed", -1)) == int(record["seed"])
               and branch_summary.get("pair_kind") == record.get("pair_kind")
               and branch_summary.get("history_regime") == record.get("history_regime")
               and branch_summary.get("action") == action
               and branch_summary.get("future_regime") == future,
               "dose_branch_identity", str(summary_path), "dose branch identity differs from its path")
        _check(audit, branch_summary.get("candidate_set") == "old_source_1024"
               and int(branch_summary.get("old_candidate_source_steps", -1)) == int(config["source_capture_steps"]),
               "dose_branch_candidate_label", str(summary_path), "dose branch does not identify the 1024-step old source")
        _check(audit, branch_summary.get("prefix_state_hash") == expected_prefix_hash,
               "dose_branch_prefix_hash", str(summary_path), "dose branch has a different common-prefix hash")
        logical_dose_branches += 1
        raw_path = folder / "steps.csv"
        raw_count, raw_rows, raw_error = _read_step_count(raw_path)
        if raw_path.is_file():
            source_raw_files.add(raw_path.resolve())
        is_alias = branch_summary.get("format") == ALIAS_FORMAT
        if is_alias:
            target_summary_path = case_dir / "branches" / f"{action}__{future}" / "summary.json"
            try:
                target_summary = _json(target_summary_path)
            except Exception as exc:
                audit.fail("alias_target_summary_missing", str(target_summary_path), "alias target summary is missing", error=repr(exc))
                physical_dose_known = logical_dose_known = False
                continue
            _audit_alias(audit, branch_dir=folder, case_dir=case_dir, summary=branch_summary,
                         record=branch_record or {}, base_branch_summary=target_summary,
                         prefix_hash=expected_prefix_hash)
            if raw_count is not None:
                physical_dose_rows += raw_count
                if raw_count:
                    audit.fail("alias_raw_rows_counted", str(raw_path),
                               "unexpected raw rows exist under an aliased branch and are included in physical cost")
            elif raw_error != "missing":
                audit.fail("alias_raw_read_error", str(raw_path), "unexpected alias raw artifact is unreadable", error=raw_error)
            target_raw_path = target_summary_path.parent / "steps.csv"
            target_count, target_rows, target_error = _read_step_count(target_raw_path)
            if target_count is None or target_rows is None:
                logical_dose_known = False
                audit.fail("alias_target_raw_missing", str(target_raw_path),
                           "aliased logical branch has no readable source raw rows; logical cost is unknown",
                           error=target_error)
            else:
                logical_dose_rows += target_count
                future_sched = template_schedule(mapping, future, kind="future", steps=int(config["bout_steps"]),
                                                 start_bout=len(prefix_sched))
                _, _, uniforms = old_audit._check_schedule_rows(
                    audit, target_rows, future_sched, phase="future", action=action, future=future,
                    global_start=prefix_steps, seed=int(record["seed"]), where=str(target_raw_path),
                )
                all_uniforms[key] = uniforms
        elif branch_summary.get("format") == DOSE_BRANCH_FORMAT:
            physical_dose_branches += 1
            if raw_count is None or raw_rows is None:
                physical_dose_known = logical_dose_known = False
                audit.fail("dose_physical_raw_missing", str(raw_path),
                           "physical dose branch raw steps are missing or unreadable; cost remains unknown",
                           error=raw_error)
                continue
            physical_dose_rows += raw_count
            logical_dose_rows += raw_count
            future_sched = template_schedule(mapping, future, kind="future", steps=int(config["bout_steps"]),
                                             start_bout=len(prefix_sched))
            branch_count, raw_reward, uniforms = old_audit._check_schedule_rows(
                audit, raw_rows, future_sched, phase="future", action=action, future=future,
                global_start=prefix_steps, seed=int(record["seed"]), where=str(raw_path),
            )
            all_uniforms[key] = uniforms
            expected_branch_steps = sum(int(item["steps"]) for item in future_sched)
            _check(audit, branch_count == expected_branch_steps
                   and int(branch_summary.get("environment_steps", -1)) == branch_count,
                   "dose_branch_raw_step_total", str(summary_path),
                   "physical dose branch raw row count differs from its schedule and summary",
                   raw=branch_count, expected=expected_branch_steps,
                   recorded=branch_summary.get("environment_steps"))
            old_audit._check_reward_summary(audit, raw_rows, branch_summary, raw_reward=raw_reward,
                                            raw_steps=branch_count, where=str(summary_path))
            old_audit._check_rollouts(
                audit, folder, phase="future", expected_rows=branch_count,
                summary={**branch_summary, "rollout_steps": int(config["rollout_steps"])}, where=str(summary_path),
            )
            _candidate_action_check(
                audit, branch_dir=folder, prefix_checkpoint=prefix_checkpoint,
                prefix_simulator=prefix_simulator, dose_decision=dose_decision,
                summary=branch_summary, config=config, foundation_bytes=foundation_bytes,
            )
            _check(audit, branch_summary.get("post_action_snapshot_included_in_deployment_capacity") is False,
                   "observer_snapshot_capacity", str(summary_path),
                   "post-action simulator/checkpoint audit snapshot must stay outside deployment capacity")
            if action in ("replace", "none"):
                target = _json(case_dir / "branches" / f"{action}__{future}" / "summary.json")
                expected_action_hash = target.get("action_state_hash")
                _check(audit, branch_summary.get("action_state_hash") != expected_action_hash,
                       "missed_valid_alias", str(summary_path),
                       "candidate-independent branch has the same full post-action hash as its base source but was rerun")
        else:
            audit.fail("dose_branch_format", str(summary_path), "dose branch format is not physical or alias",
                       actual=branch_summary.get("format"))
            physical_dose_known = logical_dose_known = False
        if branch_summary.get("format") not in (ALIAS_FORMAT, DOSE_BRANCH_FORMAT):
            physical_dose_known = logical_dose_known = False
    if len(all_uniforms) == len(expected_pairs):
        reference = all_uniforms[sorted(all_uniforms)[0]]
        for key, uniforms in all_uniforms.items():
            _check(audit, len(uniforms) == len(reference) and uniforms == reference,
                   "dose_paired_action_uniforms", case_id,
                   "logical dose arms do not share the same saved action-uniform stream",
                   branch=f"{key[0]}__{key[1]}")
    else:
        audit.fail("dose_paired_action_uniforms_missing", case_id,
                   "action-uniform pairing cannot be verified for all logical dose arms",
                   verified=len(all_uniforms), expected=len(expected_pairs))
    # Count any persisted raw rows even when a failed or unrecognized branch is absent from summaries.
    expected_raw_roots = {case_dir / "prefix" / "steps.csv"}
    expected_raw_roots.update(case_dir / "branches" / f"{a}__{f}" / "steps.csv" for a, f in expected_pairs)
    expected_raw_roots.update(dose_dir / "branches" / f"{a}__{f}" / "steps.csv" for a, f in expected_pairs)
    all_step_paths = {path.resolve() for path in case_dir.rglob("steps.csv") if path.is_file()}
    for extra in sorted(all_step_paths - {p.resolve() for p in expected_raw_roots}):
        extra_count, _, err = _read_step_count(extra)
        if extra_count is not None:
            physical_dose_rows += extra_count
            audit.fail("unexpected_raw_steps_file", str(extra),
                       "unexpected raw step file is included in physical interaction cost")
        else:
            physical_dose_known = False
            audit.fail("unexpected_raw_steps_unreadable", str(extra), "unexpected raw steps cannot be counted", error=err)
    prefix_count, _, prefix_error = _read_step_count(case_dir / "prefix" / "steps.csv")
    if prefix_count is None:
        audit.fail("prefix_raw_cost_unknown", str(case_dir / "prefix" / "steps.csv"),
                   "common-prefix raw steps are missing; prefix cost is unknown", error=prefix_error)
    base_rows = 0
    base_known = True
    for action, future in sorted(expected_pairs):
        path = case_dir / "branches" / f"{action}__{future}" / "steps.csv"
        count, _, error = _read_step_count(path)
        if count is None:
            base_known = False
            audit.fail("base_raw_cost_unknown", str(path), "base branch raw steps are missing; cost is unknown", error=error)
        else:
            base_rows += count
    observed_partial = (prefix_count or 0) + base_rows + physical_dose_rows
    expected_prefix = 2 * int(config["bootstrap_steps"]) + 8 * int(config["bout_steps"])
    full_known = prefix_count is not None and base_known and physical_dose_known and logical_dose_known
    actual_total: int | None = observed_partial if full_known else None
    logical_total: int | None = (prefix_count + base_rows + logical_dose_rows
                                 if prefix_count is not None and base_known and logical_dose_known else None)
    _check(audit, prefix_count == expected_prefix if prefix_count is not None else False,
           "case_prefix_cost", case_id, "prefix raw row count differs from the configured common prefix",
           actual=prefix_count, expected=expected_prefix)
    case_costs = record.get("costs", {})
    expected_case_fields = {
        "prefix_environment_steps": prefix_count,
        "base_branch_environment_steps": base_rows if base_known else None,
        "source_1024_branch_environment_steps": physical_dose_rows if physical_dose_known else None,
        "source_1024_logical_environment_steps": logical_dose_rows if logical_dose_known else None,
        "branch_environment_steps": base_rows + physical_dose_rows if base_known and physical_dose_known else None,
        "environment_steps_total": actual_total,
        "logical_environment_steps_total": logical_total,
    }
    for field, actual in expected_case_fields.items():
        _check(audit, case_costs.get(field) == actual, "case_cost_record", case_id,
               "case cost differs from all persisted raw step rows or remains incorrectly numeric when unknown",
               field=field, recorded=case_costs.get(field), persisted=actual)
    _check(audit, int(record.get("costs", {}).get("source_1024_physical_branch_count", -1)) == physical_dose_branches,
           "case_physical_branch_count", case_id, "case physical dose branch count differs from summaries",
           recorded=record.get("costs", {}).get("source_1024_physical_branch_count"), actual=physical_dose_branches)
    return {
        "case_id": case_id,
        "passed": len(audit.errors) == before,
        "errors": len(audit.errors) - before,
        "observed_prefix_rows": prefix_count,
        "observed_base_branch_rows": base_rows if base_known else None,
        "observed_source_1024_physical_rows": physical_dose_rows if physical_dose_known else None,
        "observed_source_1024_logical_rows": logical_dose_rows if logical_dose_known else None,
        "observed_actual_environment_steps": actual_total,
        "observed_actual_environment_steps_partial": observed_partial,
        "observed_logical_environment_steps": logical_total,
        "logical_branch_count": logical_dose_branches,
        "physical_dose_branch_count": physical_dose_branches,
        "missing_or_unknown_cost": not full_known,
    }


def audit_source(cohort: Path) -> dict[str, Any]:
    cohort = cohort.resolve()
    audit = Audit(cohort)
    summary_path = cohort / "summary.json"
    if not cohort.is_dir():
        audit.fail("source_cohort_missing", str(cohort), "source-dose cohort directory does not exist")
        return {"format": FORMAT, "source": str(cohort), "passed": False, "errors": audit.errors,
                "warnings": audit.warnings}
    try:
        summary = _json(summary_path)
    except Exception as exc:
        audit.fail("source_summary_missing", str(summary_path), "source-dose summary is unreadable", error=repr(exc))
        return {"format": FORMAT, "source": str(cohort), "passed": False, "errors": audit.errors,
                "warnings": audit.warnings}
    _check(audit, summary.get("format") == SOURCE_FORMAT, "source_summary_format", str(summary_path),
           "summary is not a source-dose cohort")
    config = _expected_config(audit, cohort, summary)
    foundation_bytes = _validate_source_snapshot(audit, cohort, summary)
    if config is None or foundation_bytes is None:
        return {"format": FORMAT, "source": str(cohort), "phase": summary.get("phase"), "passed": False,
                "errors": audit.errors, "warnings": audit.warnings}
    phase = str(summary.get("phase"))
    expected_cases = {
        f"s{seed}_{pair}_{history}"
        for seed in config["seeds"] for pair in config["pair_kinds"] for history in config["history_regimes"]
    }
    records = list(summary.get("cases", []))
    record_ids = [str(row.get("case_id", "")) for row in records]
    _check(audit, len(record_ids) == len(set(record_ids)), "duplicate_case_records", "summary.json",
           "source cohort contains duplicate case records")
    _check(audit, set(record_ids) == expected_cases, "case_universe", "summary.json",
           "source case universe differs from frozen config", actual=sorted(record_ids), expected=sorted(expected_cases))
    _check(audit, int(summary.get("case_count", -1)) == len(expected_cases), "case_count", "summary.json",
           "source case count differs from frozen config", actual=summary.get("case_count"), expected=len(expected_cases))
    if summary.get("errors"):
        audit.fail("source_runner_errors", "summary.json", "source runner recorded case or worker failures",
                   error_count=len(summary.get("errors", [])),
                   error_types=sorted({str(row.get("error_type", "unknown")) for row in summary.get("errors", [])}))
    costs_by_case: list[dict[str, Any]] = []
    record_by_id = {str(row.get("case_id")): row for row in records}
    for case_id in sorted(expected_cases):
        record = record_by_id.get(case_id)
        if record is None:
            audit.fail("case_record_missing", case_id, "expected source case has no summary record")
            costs_by_case.append({"case_id": case_id, "passed": False, "errors": 1,
                                  "observed_actual_environment_steps": None,
                                  "observed_actual_environment_steps_partial": 0,
                                  "missing_or_unknown_cost": True})
            continue
        base_record = dict(record)
        base_record["branches"] = list(record.get("base_branches", []))
        try:
            old_audit._audit_case(audit, case_id=case_id, case_record=base_record,
                                  config=config, foundation_bytes=foundation_bytes,
                                  main_summary=summary)
        except Exception as exc:
            audit.fail("base_case_audit_error", case_id, "original 2048-source branches could not be audited", error=repr(exc))
        costs_by_case.append(_audit_dose_case(audit, cohort=cohort, record=record,
                                              config=config, foundation_bytes=foundation_bytes))
    flattened_base = [branch for row in records for branch in row.get("base_branches", [])]
    flattened_dose = [branch for row in records for branch in row.get("source_1024_branches", [])]
    _same_metadata(audit, summary.get("base_branches"), flattened_base, "base_branch_flattening", "summary.json",
                   "top-level base branch list differs from case records")
    _same_metadata(audit, summary.get("source_1024_branches"), flattened_dose, "dose_branch_flattening", "summary.json",
                   "top-level dose branch list differs from case records")
    logical_branches = len(flattened_base) + len(flattened_dose)
    physical_dose_branches = sum(
        1 for branch in flattened_dose if branch.get("format") != ALIAS_FORMAT
    )
    physical_branches = len(flattened_base) + physical_dose_branches
    expected_logical_branches = len(expected_cases) * 16
    _check(audit, logical_branches == expected_logical_branches
           and int(summary.get("logical_branch_count", -1)) == logical_branches
           and int(summary.get("branch_count", -1)) == logical_branches,
           "logical_branch_accounting", "summary.json", "logical branch totals do not include base and both dose grids",
           actual=logical_branches, expected=expected_logical_branches)
    _check(audit, int(summary.get("source_1024_branch_count", -1)) == len(flattened_dose)
           and int(summary.get("source_1024_physical_branch_count", -1)) == physical_dose_branches,
           "dose_branch_accounting", "summary.json", "dose branch physical/logical counts differ from records",
           logical=len(flattened_dose), physical=physical_dose_branches)
    _check(audit, int(summary.get("base_branch_count", -1)) == len(flattened_base)
           and int(summary.get("physical_branch_count", -1)) == physical_branches,
           "physical_branch_accounting", "summary.json", "physical branch count differs from records",
           actual=physical_branches)
    def aggregate(field: str) -> int | None:
        values = [row.get(field) for row in costs_by_case]
        return None if any(value is None for value in values) else int(sum(values))
    aggregate_fields = {
        "prefix_environment_steps": "observed_prefix_rows",
        "base_branch_environment_steps": "observed_base_branch_rows",
        "source_1024_branch_environment_steps": "observed_source_1024_physical_rows",
        "source_1024_logical_environment_steps": "observed_source_1024_logical_rows",
        "environment_steps_total": "observed_actual_environment_steps",
        "logical_environment_steps_total": "observed_logical_environment_steps",
    }
    observed: dict[str, Any] = {}
    for summary_field, case_field in aggregate_fields.items():
        observed[summary_field] = aggregate(case_field)
        _check(audit, summary.get("costs", {}).get(summary_field) == observed[summary_field],
               "cohort_cost_accounting", "summary.json",
               "cohort cost differs from all persisted raw rows or hides an unknown as zero",
               field=summary_field, recorded=summary.get("costs", {}).get(summary_field),
               persisted=observed[summary_field])
    observed_partial = sum(int(row.get("observed_actual_environment_steps_partial", 0) or 0)
                           for row in costs_by_case)
    expected_actual = None
    if not any(row.get("missing_or_unknown_cost") for row in costs_by_case):
        expected_actual = observed.get("environment_steps_total")
    expected_by_phase = {"main": 5_111_808, "pilot": 39_936}
    if phase in expected_by_phase and expected_actual is not None:
        _check(audit, expected_actual >= expected_by_phase[phase], "minimum_actual_cost", "summary.json",
               "actual interaction cost is below the frozen minimum including possible non-reused branches",
               actual=expected_actual, expected_minimum=expected_by_phase[phase])
    return {
        "format": FORMAT,
        "source": str(cohort),
        "phase": phase,
        "passed": not audit.errors,
        "expected_case_count": len(expected_cases),
        "logical_branch_count": logical_branches,
        "physical_branch_count": physical_branches,
        "cost": {
            **observed,
            "actual_environment_steps_partial_raw_rows": observed_partial,
            "expected_minimum_environment_steps": expected_by_phase.get(phase),
        },
        "cases": costs_by_case,
        "errors": audit.errors,
        "warnings": audit.warnings,
    }


def _ablation_snapshot_checks(
    audit: Audit,
    *,
    cohort: Path,
    summary: Mapping[str, Any],
    source_root: Path,
    source_summary: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    snapshot = cohort / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    try:
        manifest = _json(manifest_path)
    except Exception as exc:
        audit.fail("ablation_manifest_missing", str(manifest_path),
                   "learning-ablation source manifest is unreadable", error=repr(exc))
        return
    _same_metadata(audit, manifest, summary.get("source_manifest"), "ablation_manifest_summary",
                   str(manifest_path), "summary and persisted ablation manifests differ")
    _check(audit, manifest.get("format") == "storage_learning_interaction_manifest_v1",
           "ablation_manifest_format", str(manifest_path), "unexpected ablation manifest format")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        audit.fail("ablation_manifest_empty", str(manifest_path), "ablation manifest has no source records")
        return
    source_names = {str(name).replace("\\", "/") for name in sources}
    required_names = {"scripts/run_learning_ablation.py", "docs/LEARNING_INTERACTION_PROTOCOL.md"}
    required_names.update(str(name).replace("\\", "/") for name in source_manifest.get("sources", {}))
    _check(audit, required_names.issubset(source_names), "ablation_manifest_coverage", str(manifest_path),
           "ablation snapshot omits its runner, protocol, or inherited source-dose source files",
           missing=sorted(required_names - source_names))
    for name, record in sources.items():
        rel = Path(str(name).replace("\\", "/"))
        private = any(part.casefold() in PRIVATE_COMPONENTS for part in rel.parts)
        _check(audit, not private, "ablation_private_source_listed", f"source_snapshot/{name}",
               "ablation manifest lists a local instruction or temporary path")
        copied = _safe_relative(snapshot, str(name))
        if copied is None:
            audit.fail("ablation_source_path_escape", f"source_snapshot/{name}",
                       "ablation manifest path escapes its snapshot")
            continue
        if not copied.is_file():
            audit.fail("ablation_source_missing", f"source_snapshot/{name}",
                       "manifest-listed ablation source copy is missing")
            continue
        expected_bytes = record.get("bytes") if isinstance(record, Mapping) else None
        expected_hash = record.get("sha256") if isinstance(record, Mapping) else None
        _check(audit, copied.stat().st_size == expected_bytes and _digest(copied) == expected_hash,
               "ablation_source_hash", f"source_snapshot/{name}",
               "manifest-listed ablation source copy differs from its size or SHA-256")
    for path in snapshot.rglob("*"):
        rel = path.relative_to(snapshot)
        _check(audit, not any(part.casefold() in PRIVATE_COMPONENTS for part in rel.parts),
               "ablation_private_source_present", path.relative_to(cohort).as_posix(),
               "ablation source snapshot contains a local instruction or temporary path")
    excluded = {str(item).casefold() for item in manifest.get("excluded_local_instructions", [])}
    _check(audit, excluded == {"agent.md", "agents.md"}, "ablation_private_exclusions",
           str(manifest_path), "ablation manifest does not record both local instruction exclusions")
    matches = _private_hash_matches(snapshot)
    _check(audit, not matches, "ablation_private_exact_hash_match", "source_snapshot",
           "ablation snapshot contains a byte-for-byte copy of a local instruction file",
           matching_paths=matches)

    input_record = manifest.get("source_dose_input", {})
    _check(audit, isinstance(input_record, Mapping), "ablation_input_manifest", str(manifest_path),
           "source-dose input provenance record is malformed")
    if not isinstance(input_record, Mapping):
        return
    try:
        input_root = Path(str(input_record.get("path", ""))).resolve()
        expected_root = source_root.resolve()
        same_root = input_root == expected_root
    except Exception:
        same_root = False
    _check(audit, same_root, "ablation_input_path", str(manifest_path),
           "ablation manifest references a different source-dose cohort")
    copied_inputs = input_record.get("files", {})
    if not isinstance(copied_inputs, Mapping):
        copied_inputs = {}
    for filename in ("summary.json", "config.json"):
        copied = snapshot / "input_source_dose" / filename
        record = copied_inputs.get(filename, {})
        if not copied.is_file():
            audit.fail("ablation_input_copy_missing", str(copied), "source-dose input copy is missing")
            continue
        _check(audit, isinstance(record, Mapping)
               and copied.stat().st_size == record.get("bytes")
               and _digest(copied) == record.get("sha256"),
               "ablation_input_copy_hash", str(copied),
               "copied source-dose input differs from its manifest size or SHA-256")
    copied_source_manifest = snapshot / "input_source_dose" / "source_manifest.json"
    source_manifest_record = input_record.get("source_manifest")
    if not copied_source_manifest.is_file():
        audit.fail("ablation_input_manifest_copy_missing", str(copied_source_manifest),
                   "source-dose manifest copy is missing")
    elif not isinstance(source_manifest_record, Mapping):
        audit.fail("ablation_input_manifest_record_missing", str(manifest_path),
                   "source-dose manifest copy has no provenance record")
    else:
        _check(audit, copied_source_manifest.stat().st_size == source_manifest_record.get("bytes")
               and _digest(copied_source_manifest) == source_manifest_record.get("sha256"),
               "ablation_input_manifest_hash", str(copied_source_manifest),
               "copied source-dose manifest differs from its provenance record")
        try:
            copied_manifest = _json(copied_source_manifest)
            _same_metadata(audit, copied_manifest, source_manifest, "ablation_input_manifest_content",
                           str(copied_source_manifest),
                           "copied source-dose manifest differs from the audited input cohort")
            _check(audit, set(source_manifest.get("sources", {})).issubset(source_names),
                   "ablation_inherited_provenance", str(manifest_path),
                   "ablation manifest does not retain all source-dose source hashes")
            _same_metadata(audit, manifest.get("foundation"), source_manifest.get("foundation"),
                           "ablation_foundation_record", str(manifest_path),
                           "ablation foundation provenance differs from source-dose input")
            _same_metadata(audit, manifest.get("dataset"), source_manifest.get("dataset"),
                           "ablation_dataset_record", str(manifest_path),
                           "ablation dataset provenance differs from source-dose input")
        except Exception as exc:
            audit.fail("ablation_input_manifest_parse", str(copied_source_manifest),
                       "copied source-dose manifest cannot be checked", error=repr(exc))


def _ablation_checkpoint_checks(
    audit: Audit,
    *,
    branch_dir: Path,
    branch: Mapping[str, Any],
    expected_environment_steps: int,
    rollout_steps: int,
    where: str,
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    action_path = branch_dir / "post_action" / "checkpoint.pt"
    final_path = branch_dir / "checkpoint.pt"
    try:
        action = _checkpoint(action_path)
        final = _checkpoint(final_path)
    except Exception as exc:
        audit.fail("ablation_checkpoint_missing", where,
                   "post-action or final learner checkpoint cannot be read", error=repr(exc))
        return None, None, None
    _scan_learner_keys(audit, action, str(action_path))
    _scan_learner_keys(audit, final, str(final_path))
    for label, checkpoint in (("post_action", action), ("final", final)):
        _check(audit, checkpoint.get("format") == "storage_decision_learner_v1",
               "ablation_checkpoint_format", f"{where}/{label}", "unexpected learner checkpoint format")
        _check(audit, isinstance(checkpoint.get("core_state"), Mapping),
               "ablation_core_state", f"{where}/{label}", "learner core state is missing")
    action_state, final_state = action.get("core_state", {}), final.get("core_state", {})
    if not isinstance(action_state, Mapping) or not isinstance(final_state, Mapping):
        return None, action, final
    keys_match = set(action_state) == set(final_state)
    _check(audit, keys_match, "ablation_core_key_set", where,
           "core state keys changed during no-gradient future collection")
    try:
        changed = sorted(
            str(key) for key in set(action_state) & set(final_state)
            if old_audit.semantic_hash(action_state[key]) != old_audit.semantic_hash(final_state[key])
        )
        changed.extend(sorted(str(key) for key in set(action_state) ^ set(final_state)))
    except Exception as exc:
        changed = []
        audit.fail("ablation_core_compare", where, "core state tensors cannot be compared", error=repr(exc))
    changed = sorted(set(changed))
    actor_allowed = re.compile(r"^(slot_groups|seen_mask|_shared_coefficients\.\d+\.C|_nodes\.\d+\.A)$")
    disallowed = [key for key in changed if not actor_allowed.fullmatch(key)]
    _check(audit, not disallowed, "ablation_actor_change_scope", where,
           "no-gradient core changes escape the LRU/prior/codec actor state", changed=changed,
           disallowed=disallowed)
    w0_keys = [key for key in action_state if key == "source_basis"
               or str(key).startswith(("mlp_extractor.policy_net.", "action_net."))]
    critic_keys = [key for key in action_state
                   if str(key).startswith(("mlp_extractor.value_net.", "value_net."))]
    _check(audit, bool(w0_keys), "ablation_w0_keys", where, "W0 actor parameters were not found")
    _check(audit, bool(critic_keys), "ablation_critic_keys", where, "critic parameters were not found")
    try:
        w0_same = bool(w0_keys) and all(
            old_audit.semantic_hash(action_state[key]) == old_audit.semantic_hash(final_state[key])
            for key in w0_keys
        )
        critic_same = bool(critic_keys) and all(
            old_audit.semantic_hash(action_state[key]) == old_audit.semantic_hash(final_state[key])
            for key in critic_keys
        )
        optimizer_same = (
            old_audit.semantic_hash(action.get("optimizer_state"))
            == old_audit.semantic_hash(final.get("optimizer_state"))
            and action.get("optimizer_step_count") == final.get("optimizer_step_count")
        )
    except Exception as exc:
        w0_same = critic_same = optimizer_same = False
        audit.fail("ablation_parameter_compare", where,
                   "W0, critic, or optimizer state cannot be compared", error=repr(exc))
    _check(audit, w0_same, "ablation_w0_changed", where, "W0 actor parameters changed during no-gradient collection")
    _check(audit, critic_same, "ablation_critic_changed", where, "critic parameters changed during no-gradient collection")
    _check(audit, optimizer_same, "ablation_optimizer_changed", where,
           "optimizer state or step count changed during no-gradient collection")
    _check(audit, int(action.get("optimizer_step_count", -1)) == int(final.get("optimizer_step_count", -2)),
           "ablation_optimizer_step_delta", where, "optimizer step count changed")

    runtime_before = action.get("runtime_counts", {})
    runtime_after = final.get("runtime_counts", {})
    counts_before = action.get("counts", {})
    counts_after = final.get("counts", {})
    if not all(isinstance(value, Mapping) for value in (runtime_before, runtime_after, counts_before, counts_after)):
        audit.fail("ablation_checkpoint_counts", where, "learner checkpoint count fields are malformed")
        return None, action, final
    runtime_delta = {
        key: int(runtime_after.get(key, 0)) - int(runtime_before.get(key, 0))
        for key in set(runtime_before) | set(runtime_after)
    }
    expected_rollouts = expected_environment_steps // rollout_steps if rollout_steps > 0 else -1
    _check(audit, expected_environment_steps % rollout_steps == 0 if rollout_steps > 0 else False,
           "ablation_rollout_alignment", where, "future budget is not an integer number of rollouts")
    _check(audit, runtime_delta.get("environment_steps", -1) == expected_environment_steps,
           "ablation_runtime_steps", where,
           "learner environment-step delta differs from the raw future rows",
           actual=runtime_delta.get("environment_steps"), expected=expected_environment_steps)
    _check(audit, runtime_delta.get("minibatch_updates", -1) == 0
           and runtime_delta.get("rollouts_trained", -1) == 0,
           "ablation_no_gradient_updates", where,
           "no-gradient branch recorded optimizer minibatch or trained-rollout updates",
           minibatch_updates=runtime_delta.get("minibatch_updates"),
           rollouts_trained=runtime_delta.get("rollouts_trained"))
    _check(audit, runtime_delta.get("rollouts_collected", -1) == expected_rollouts,
           "ablation_rollouts_collected", where,
           "collected no-gradient rollout count differs from the raw schedule",
           actual=runtime_delta.get("rollouts_collected"), expected=expected_rollouts)
    fit_before, fit_after = int(counts_before.get("fit_steps", -1)), int(counts_after.get("fit_steps", -2))
    _check(audit, fit_before == fit_after, "ablation_fit_step_count", where,
           "fit_steps changed despite disabled gradient updates", before=fit_before, after=fit_after)
    try:
        start_steps = np.asarray(action["fitted_steps"], dtype=np.int64)
        end_steps = np.asarray(final["fitted_steps"], dtype=np.int64)
        start_generation = np.asarray(action["slot_generation"], dtype=np.int64)
        end_generation = np.asarray(final["slot_generation"], dtype=np.int64)
        valid_shape = start_steps.shape == end_steps.shape == start_generation.shape == end_generation.shape
        no_growth = valid_shape
        if valid_shape:
            for old_steps, new_steps, old_gen, new_gen in zip(start_steps, end_steps, start_generation, end_generation):
                if new_gen < old_gen or (new_gen == old_gen and new_steps > old_steps):
                    no_growth = False
                if new_gen > old_gen and new_steps != 0:
                    no_growth = False
        _check(audit, no_growth, "ablation_fitted_steps_growth", where,
               "fitted_steps grew within a generation or failed to reset on generation rollover")
        _check(audit, int(counts_after.get("rollouts_trained", -1))
               - int(counts_before.get("rollouts_trained", -1)) == 0
               and int(counts_after.get("minibatch_updates", -1))
               - int(counts_before.get("minibatch_updates", -1)) == 0,
               "ablation_summary_update_counts", where,
               "checkpoint count deltas report trained rollouts or minibatch updates")
        expected_audit = branch.get("no_gradient_audit", {})
        _check(audit, expected_audit.get("critic_unchanged") is True
               and expected_audit.get("w0_unchanged") is True
               and expected_audit.get("optimizer_state_unchanged") is True
               and int(expected_audit.get("minibatch_updates", -1)) == 0
               and int(expected_audit.get("rollouts_trained", -1)) == 0
               and int(expected_audit.get("environment_steps", -1)) == expected_environment_steps
               and expected_audit.get("fitted_steps_not_increased") is True,
               "ablation_reported_checkpoint_audit", where,
               "runner no-gradient audit metadata disagrees with the required invariants")
        _check(audit, branch.get("critic_unchanged") is True and branch.get("w0_unchanged") is True
               and branch.get("fitted_steps_not_increased") is True
               and int(branch.get("minibatch_updates", -1)) == 0
               and int(branch.get("rollouts_collected", -1)) == expected_rollouts,
               "ablation_branch_audit_copy", where,
               "branch summary does not copy the independently checked no-gradient status")
        _check(audit, expected_audit.get("changed_core_state_keys") == changed,
               "ablation_changed_keys_record", where,
               "runner changed-core-key record differs from saved checkpoint tensors")
        _check(audit, expected_audit.get("actor_changes_restricted_to_lru_prior_codec_state") is True,
               "ablation_actor_change_record", where,
               "runner did not report actor changes as confined to LRU/prior/codec state")
    except Exception as exc:
        audit.fail("ablation_fitted_steps_parse", where,
                   "fitted_steps or saved no-gradient audit metadata cannot be checked", error=repr(exc))

    try:
        semantic = old_audit.semantic_hash
        simulator_path = branch_dir / "post_action" / "simulator.json"
        simulator = _json(simulator_path)
        computed_hash = semantic({"learner": action, "simulator": simulator})
        manifest_path = branch_dir / "post_action" / "manifest.json"
        action_manifest = _json(manifest_path)
        _check(audit, computed_hash == branch.get("action_state_hash")
               and action_manifest.get("action_state_hash") == computed_hash,
               "ablation_action_hash_recompute", where,
               "post-action hash does not match the saved learner/simulator state")
        start_simulator_path = branch_dir / "simulator_start.json"
        if start_simulator_path.is_file():
            _same_metadata(audit, simulator, _json(start_simulator_path), "ablation_action_simulator_copy",
                           where, "post-action simulator state differs from branch start state")
        return computed_hash, action, final
    except Exception as exc:
        audit.fail("ablation_action_hash_parse", where,
                   "post-action learner/simulator hash cannot be independently recomputed", error=repr(exc))
        return None, action, final


def _check_no_gradient_rollouts(
    audit: Audit,
    *,
    branch_dir: Path,
    summary: Mapping[str, Any],
    raw_steps: int,
    rollout_steps: int,
    where: str,
) -> int:
    path = branch_dir / "rollouts.csv"
    try:
        rows = _csv_rows(path)
    except Exception as exc:
        audit.fail("ablation_rollout_ledger_missing", str(path),
                   "no-gradient rollout ledger is unreadable", error=repr(exc))
        return 0
    submitted = updates = 0
    for index, row in enumerate(rows):
        try:
            count = int(row["rows"])
            update_count = int(row["updates"])
        except Exception as exc:
            audit.fail("ablation_rollout_row_parse", str(path), "rollout row has malformed counts",
                       row=index, error=repr(exc))
            continue
        submitted += count
        updates += update_count
        _check(audit, row.get("phase") == "future", "ablation_rollout_phase", str(path),
               "rollout ledger contains a non-future phase", row=index, actual=row.get("phase"))
        _check(audit, row.get("action") == summary.get("action")
               and row.get("future_regime") == summary.get("future_regime"),
               "ablation_rollout_identity", str(path),
               "rollout ledger identity differs from its branch", row=index)
        _check(audit, count == rollout_steps, "ablation_rollout_size", str(path),
               "no-gradient rollout row count differs from the configured size",
               row=index, actual=count, expected=rollout_steps)
        _check(audit, update_count == 0, "ablation_rollout_updates", str(path),
               "no-gradient rollout ledger records one or more minibatch updates",
               row=index, updates=update_count)
        try:
            _check(audit, int(row.get("actor_optimizer_state_cleared", "0")) == 1,
                   "ablation_actor_optimizer_marker", str(path),
                   "rollout ledger does not mark the actor optimizer state as cleared", row=index)
        except Exception:
            audit.fail("ablation_actor_optimizer_marker", str(path),
                       "rollout actor optimizer marker is malformed", row=index)
    expected_rollouts = raw_steps // rollout_steps if rollout_steps > 0 else -1
    _check(audit, raw_steps % rollout_steps == 0 if rollout_steps > 0 else False,
           "ablation_rollout_divisibility", str(path), "raw future rows are not divisible by rollout size")
    _check(audit, submitted == raw_steps and len(rows) == expected_rollouts,
           "ablation_rollout_coverage", str(path),
           "rollout ledger does not cover each persisted future row exactly once",
           submitted=submitted, raw_steps=raw_steps, rollouts=len(rows), expected_rollouts=expected_rollouts)
    _check(audit, updates == 0, "ablation_rollout_update_total", str(path),
           "rollout ledger includes gradient updates", updates=updates)
    counts = summary.get("counts", {})
    _check(audit, isinstance(counts, Mapping)
           and int(counts.get("environment_steps", -1)) == raw_steps
           and int(counts.get("rollouts_trained", -1)) == 0
           and int(counts.get("minibatch_updates", -1)) == 0,
           "ablation_summary_rollout_counts", where,
           "branch summary counts disagree with raw rows or no-gradient rollout ledger")
    _check(audit, int(summary.get("rollouts_collected", -1)) == len(rows)
           and int(summary.get("fit_steps_delta", -1)) == 0,
           "ablation_summary_collected_counts", where,
           "branch summary collected-rollout or fit-step delta is inconsistent")
    return len(rows)


def _ablation_capacity_checks(
    audit: Audit,
    *,
    branch_dir: Path,
    branch: Mapping[str, Any],
    config: Mapping[str, Any],
    foundation_bytes: int,
    where: str,
) -> None:
    actor_budget = int(config.get("actor_budget_bytes", -1))
    package_budget = int(config.get("package_budget_bytes", -1))
    for label, checkpoint_path, actor_path, record_key in (
        ("post_action", branch_dir / "post_action" / "checkpoint.pt",
         branch_dir / "post_action" / "actor.bin", "capacity_after_action"),
        ("final", branch_dir / "checkpoint.pt", branch_dir / "actor.bin", "capacity"),
    ):
        record = branch.get(record_key, {})
        if not isinstance(record, Mapping):
            audit.fail("ablation_capacity_record", f"{where}/{label}",
                       "branch capacity record is malformed")
            continue
        if not checkpoint_path.is_file() or not actor_path.is_file():
            audit.fail("ablation_capacity_file_missing", f"{where}/{label}",
                       "checkpoint or actor package file is missing")
            continue
        actual_actor = actor_path.stat().st_size
        actual_checkpoint = checkpoint_path.stat().st_size
        expected_total = foundation_bytes + actual_checkpoint + actual_actor
        _check(audit, actual_actor == actor_budget and record.get("actor_bytes") == actor_budget,
               "ablation_actor_capacity", f"{where}/{label}",
               "actor package does not match its exact frozen byte budget",
               actual=actual_actor, recorded=record.get("actor_bytes"), expected=actor_budget)
        _check(audit, record.get("foundation_bytes") == foundation_bytes
               and record.get("checkpoint_bytes") == actual_checkpoint
               and record.get("full_persistent_bytes") == expected_total
               and record.get("package_budget_bytes") == package_budget
               and expected_total <= package_budget,
               "ablation_full_package_capacity", f"{where}/{label}",
               "full persistent package byte count differs from files or exceeds its budget",
               foundation_bytes=foundation_bytes, checkpoint_bytes=actual_checkpoint,
               actor_bytes=actual_actor, expected_total=expected_total,
               recorded=record.get("full_persistent_bytes"), budget=package_budget)
    _check(audit, branch.get("actor_payload_bytes") == actor_budget,
           "ablation_actor_payload_record", where,
           "branch actor payload metadata differs from the exact frozen byte budget")


def audit_ablation(cohort: Path, source_report: Mapping[str, Any]) -> dict[str, Any]:
    """Independently audit no-gradient branches against persisted source-dose data."""
    cohort = cohort.resolve()
    audit = Audit(cohort)
    summary_path = cohort / "summary.json"
    source_root = Path(str(source_report.get("source", ""))).resolve()
    source_summary_path = source_root / "summary.json"
    source_config_path = source_root / "config.json"
    source_summary: Mapping[str, Any] = {}
    source_config: Mapping[str, Any] = {}
    source_manifest: Mapping[str, Any] = {}
    if not cohort.is_dir():
        audit.fail("ablation_cohort_missing", str(cohort), "learning-ablation cohort directory does not exist")
        return {"source": str(cohort), "passed": False, "errors": audit.errors,
                "warnings": audit.warnings, "source_audit_passed": bool(source_report.get("passed"))}
    try:
        summary = _json(summary_path)
    except Exception as exc:
        audit.fail("ablation_summary_missing", str(summary_path),
                   "learning-ablation summary is unreadable", error=repr(exc))
        return {"source": str(cohort), "passed": False, "errors": audit.errors,
                "warnings": audit.warnings, "source_audit_passed": bool(source_report.get("passed"))}
    _check(audit, source_report.get("passed") is True, "ablation_source_audit_failed", str(source_root),
           "source-dose cohort must pass its independent audit before ablation can pass")
    try:
        source_summary = _json(source_summary_path)
        source_config = _json(source_config_path)
        source_manifest = _json(source_root / "source_snapshot" / "manifest.json")
    except Exception as exc:
        audit.fail("ablation_source_input_missing", str(source_root),
                   "audited source-dose input artifacts are unreadable", error=repr(exc))
    _check(audit, summary.get("format") == "learning_interaction_ablation_v1",
           "ablation_summary_format", str(summary_path), "summary is not a learning-interaction ablation")
    phase = str(summary.get("phase", ""))
    _check(audit, phase in ("pilot", "main") and phase == str(source_summary.get("phase", "")),
           "ablation_phase", str(summary_path), "ablation phase differs from its source-dose cohort",
           actual=phase, source=source_summary.get("phase"))
    _check(audit, summary.get("learning_mode") == "no_gradient", "ablation_learning_mode",
           str(summary_path), "ablation summary does not identify no-gradient collection")
    config_path = cohort / "config.json"
    source_ref_path = cohort / "source_reference.json"
    try:
        config_record = _json(config_path)
    except Exception as exc:
        config_record = {}
        audit.fail("ablation_config_missing", str(config_path), "ablation config is unreadable", error=repr(exc))
    try:
        source_ref = _json(source_ref_path)
    except Exception as exc:
        source_ref = {}
        audit.fail("ablation_source_reference_missing", str(source_ref_path),
                   "source reference is unreadable", error=repr(exc))
    _same_metadata(audit, source_ref, summary.get("source_reference"), "ablation_source_reference_copy",
                   str(source_ref_path), "summary and persisted source references differ")
    _check(audit, config_record.get("format") == "storage_learning_interaction_config_v1"
           and config_record.get("learning_mode") == "no_gradient"
           and config_record.get("actions") == ["replace", "none"]
           and config_record.get("new_prefix_environment_steps") == 0,
           "ablation_config_contract", str(config_path),
           "ablation config differs from the frozen no-gradient branch contract")
    _same_metadata(audit, config_record.get("source_config"), source_summary.get("config"),
                   "ablation_source_config", str(config_path),
                   "ablation source config differs from the audited source-dose config")
    _same_metadata(audit, summary.get("config"), config_record.get("source_config"),
                   "ablation_summary_config", str(summary_path),
                   "ablation summary config differs from its source config")
    _check(audit, Path(str(config_record.get("source_path", ""))).resolve() == source_root,
           "ablation_config_source_path", str(config_path), "ablation config points at a different source cohort")
    ref_expected = {
        "path": str(source_root),
        "summary_sha256": _digest(source_summary_path) if source_summary_path.is_file() else None,
        "config_sha256": _digest(source_config_path) if source_config_path.is_file() else None,
        "format": source_summary.get("format"),
        "phase": source_summary.get("phase"),
        "source_manifest": source_summary.get("source_manifest", {}),
        "costs": source_summary.get("costs", {}),
    }
    _same_metadata(audit, source_ref, ref_expected, "ablation_source_reference_integrity",
                   str(source_ref_path), "ablation source reference differs from the supplied source cohort")
    _ablation_snapshot_checks(audit, cohort=cohort, summary=summary, source_root=source_root,
                              source_summary=source_summary, source_manifest=source_manifest)
    try:
        config = old_audit.validate_config(source_config)
    except Exception as exc:
        config = {}
        audit.fail("ablation_source_config_invalid", str(source_config_path),
                   "source config cannot define the ablation schedule", error=repr(exc))
    try:
        foundation_bytes = int(source_manifest.get("foundation", {}).get("bytes", -1))
    except (AttributeError, TypeError, ValueError):
        foundation_bytes = -1
    source_records = {str(row.get("case_id")): row for row in source_summary.get("cases", [])}
    case_records = list(summary.get("cases", []))
    expected_case_ids = set(source_records)
    record_ids = [str(row.get("case_id", "")) for row in case_records]
    _check(audit, len(record_ids) == len(set(record_ids)), "ablation_duplicate_case", "summary.json",
           "ablation summary contains duplicate case IDs")
    _check(audit, set(record_ids) == expected_case_ids, "ablation_case_universe", "summary.json",
           "ablation case set differs from the source-dose case set",
           actual=sorted(record_ids), expected=sorted(expected_case_ids))
    _check(audit, int(summary.get("case_count", -1)) == len(expected_case_ids),
           "ablation_case_count", "summary.json", "ablation case count differs from source-dose cohort",
           actual=summary.get("case_count"), expected=len(expected_case_ids))
    if summary.get("errors"):
        audit.fail("ablation_runner_errors", "summary.json", "ablation runner reported failed cases",
                   error_count=len(summary.get("errors", [])),
                   error_types=sorted({str(row.get("error_type", "unknown")) for row in summary.get("errors", [])}))
    actions = ("replace", "none")
    futures = tuple(str(item) for item in config.get("future_regimes", ()))
    expected_keys = {(action, future) for action in actions for future in futures}
    source_config_record = source_summary.get("config", {})
    rollout_steps = int(source_config_record.get("rollout_steps", 256)) if isinstance(source_config_record, Mapping) else 256
    case_reports: list[dict[str, Any]] = []
    observed_total = 0
    observed_partial = 0
    total_known = True
    for case_id in sorted(expected_case_ids):
        record = next((row for row in case_records if str(row.get("case_id")) == case_id), None)
        source_record = source_records[case_id]
        before_errors = len(audit.errors)
        case_dir = cohort / "cases" / case_id
        source_case_dir = source_root / "cases" / case_id
        if record is None:
            audit.fail("ablation_case_record_missing", case_id, "expected ablation case record is missing")
            case_reports.append({"case_id": case_id, "passed": False, "errors": 1,
                                 "observed_steps": None, "observed_partial_steps": 0})
            total_known = False
            continue
        case_summary_path = case_dir / "case_summary.json"
        try:
            case_summary = _json(case_summary_path)
            _same_metadata(audit, case_summary, record, "ablation_case_summary_copy", str(case_summary_path),
                           "case-level summary differs from cohort summary")
        except Exception as exc:
            audit.fail("ablation_case_summary_missing", str(case_summary_path),
                       "case summary is unreadable", error=repr(exc))
            case_summary = {}
        _check(audit, record.get("source_case_id") == case_id
               and record.get("seed") == source_record.get("seed")
               and record.get("pair_kind") == source_record.get("pair_kind")
               and record.get("history_regime") == source_record.get("history_regime"),
               "ablation_case_identity", case_id, "ablation case identity differs from source-dose input")
        source_prefix = source_record.get("prefix", {})
        prefix_steps = int(source_prefix.get("environment_steps", -1)) if isinstance(source_prefix, Mapping) else -1
        prefix_hash = str(source_record.get("prefix_state_hash") or
                          (source_prefix.get("prefix_state_hash", "") if isinstance(source_prefix, Mapping) else ""))
        _check(audit, prefix_steps > 0 and record.get("source_prefix_environment_steps") == prefix_steps
               and record.get("new_prefix_environment_steps") == 0,
               "ablation_prefix_cost", case_id,
               "ablation does not reference the already-accounted source prefix with zero new prefix cost")
        _check(audit, record.get("prefix_state_hash") == prefix_hash
               and record.get("source_prefix_state_hash") == prefix_hash,
               "ablation_prefix_hash", case_id, "ablation case does not retain the complete source prefix hash")
        dose_decision_path = source_case_dir / "dose_1024" / "decision.json"
        try:
            dose_decision = _json(dose_decision_path)
        except Exception as exc:
            dose_decision = {}
            audit.fail("ablation_dose_decision_missing", str(dose_decision_path),
                       "source-dose decision cannot be read", error=repr(exc))
        decision_path = case_dir / "decision.json"
        try:
            decision = _json(decision_path)
        except Exception as exc:
            decision = {}
            audit.fail("ablation_decision_missing", str(decision_path),
                       "ablation decision is unreadable", error=repr(exc))
        _check(audit, decision.get("format") == "storage_learning_interaction_decision_v1"
               and decision.get("learning_mode") == "no_gradient"
               and decision.get("selectors_frozen_before_future_expansion") is True
               and decision.get("selector_information_source") ==
               "source-dose prefix only; no future branch feedback",
               "ablation_decision_contract", str(decision_path),
               "ablation decision does not preserve the frozen past-only selector contract")
        _check(audit, decision.get("prefix_state_hash") == prefix_hash
               and dose_decision.get("prefix_state_hash") == prefix_hash,
               "ablation_decision_prefix_hash", str(decision_path),
               "ablation and source-dose decisions do not share the same complete prefix")
        _same_metadata(audit, decision.get("mapping"), dose_decision.get("mapping"),
                       "ablation_mapping", str(decision_path),
                       "ablation decision mapping differs from the source-dose prefix")
        source_branch_records = {
            (str(row.get("action")), str(row.get("future_regime"))): row
            for row in source_record.get("base_branches", [])
            if str(row.get("action")) in actions
        }
        _check(audit, set(source_branch_records) == expected_keys, "ablation_ppo_references", case_id,
               "source cohort does not provide all four replace/none PPO reference branches")
        no_grad_records = list(record.get("branches", []))
        _same_metadata(audit, record.get("branch_results"), no_grad_records,
                       "ablation_case_branch_results", case_id,
                       "case branch_results differs from branch summaries")
        no_grad_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for branch in no_grad_records:
            key = (str(branch.get("action")), str(branch.get("future_regime")))
            if key in no_grad_by_key:
                audit.fail("ablation_duplicate_branch", case_id,
                           "case contains duplicate no-gradient branch identity", branch=list(key))
            no_grad_by_key[key] = branch
        _check(audit, set(no_grad_by_key) == expected_keys, "ablation_branch_grid", case_id,
               "case does not contain the four fresh replace/none × future branches",
               actual=sorted(no_grad_by_key), expected=sorted(expected_keys))
        decision_branches = {
            (str(row.get("action")), str(row.get("future_regime"))): row
            for row in decision.get("branches", []) if isinstance(row, Mapping)
        }
        _check(audit, set(decision_branches) == expected_keys, "ablation_decision_branch_grid",
               str(decision_path), "decision file does not index all expected branches")
        case_path_rows: dict[str, int | None] = {}
        case_raw_known = True
        case_partial = 0
        case_uniforms: dict[str, list[float]] = {}
        for action, future in sorted(expected_keys):
            key = (action, future)
            branch = no_grad_by_key.get(key)
            branch_dir = case_dir / "branches" / f"{action}__{future}"
            raw_path = branch_dir / "steps.csv"
            raw_count, raw_rows, raw_error = _read_step_count(raw_path)
            if raw_count is None or raw_rows is None:
                case_raw_known = False
                audit.fail("ablation_raw_cost_unknown", str(raw_path),
                           "expected branch raw rows are missing or unreadable; cost remains unknown", error=raw_error)
                case_path_rows[branch_dir.name] = None
            else:
                case_path_rows[branch_dir.name] = raw_count
                case_partial += raw_count
                expected_future = template_schedule(
                    dose_decision["mapping"], future, kind="future", steps=int(config["bout_steps"]),
                    start_bout=len(prefix_schedule(dose_decision["mapping"], config,
                                                   str(source_record["history_regime"]))),
                )
                actual_steps, reward_total, uniforms = old_audit._check_schedule_rows(
                    audit, raw_rows, expected_future, phase="future", action=action, future=future,
                    global_start=prefix_steps, seed=int(source_record["seed"]), where=str(raw_path),
                )
                case_uniforms[f"{action}__{future}"] = uniforms
                if branch is None:
                    audit.fail("ablation_branch_record_missing", str(raw_path),
                               "raw branch has no corresponding summary record")
                else:
                    summary_path = branch_dir / "summary.json"
                    try:
                        branch_summary = _json(summary_path)
                        _same_metadata(audit, branch_summary, branch, "ablation_branch_summary_copy",
                                       str(summary_path), "case branch record differs from saved branch summary")
                    except Exception as exc:
                        branch_summary = {}
                        audit.fail("ablation_branch_summary_missing", str(summary_path),
                                   "branch summary is unreadable", error=repr(exc))
                    expected_count = sum(int(row["steps"]) for row in expected_future)
                    _check(audit, branch.get("format") == "learning_interaction_branch_v1"
                           and branch.get("learning_mode") == "no_gradient"
                           and branch.get("candidate_set") == "shared_newC_vs_none"
                           and branch.get("old_candidate_source_steps") is None,
                           "ablation_branch_contract", str(summary_path),
                           "branch is not a fresh no-gradient shared-newC/none observation")
                    _check(audit, branch.get("action") == action and branch.get("future_regime") == future
                           and branch.get("source_case_id") == case_id
                           and branch.get("source_prefix_state_hash") == prefix_hash
                           and branch.get("prefix_state_hash") == prefix_hash,
                           "ablation_branch_identity", str(summary_path),
                           "branch identity or complete prefix hash differs from its path")
                    _check(audit, branch.get("new_prefix_environment_steps") == 0
                           and branch.get("new_environment_steps") == actual_steps
                           and int(branch.get("environment_steps", -1)) == actual_steps
                           and int(branch.get("environment_steps", -1)) == expected_count,
                           "ablation_branch_raw_cost", str(summary_path),
                           "branch environment-step summary differs from persisted future rows and schedule",
                           raw=actual_steps, expected=expected_count)
                    _check(audit, not str(branch.get("format", "")).endswith("alias_v1")
                           and branch.get("independent_observation", True) is not False,
                           "ablation_branch_alias", str(summary_path),
                           "no-gradient branches must be independent physical observations, never aliases")
                    decision_entry = decision_branches.get(key, {})
                    _check(audit, decision_entry.get("summary") == f"branches/{action}__{future}/summary.json"
                           and decision_entry.get("independent_observation") is True
                           and decision_entry.get("learning_mode") == "no_gradient",
                           "ablation_decision_branch_path", str(decision_path),
                           "decision branch path or fresh-observation flag is inconsistent")
                    try:
                        old_audit._check_reward_summary(audit, raw_rows, branch_summary,
                                                        raw_reward=reward_total, raw_steps=actual_steps,
                                                        where=str(summary_path))
                    except Exception as exc:
                        audit.fail("ablation_reward_audit_error", str(summary_path),
                                   "reward aggregation cannot be checked", error=repr(exc))
                    _check_no_gradient_rollouts(audit, branch_dir=branch_dir, summary=branch_summary,
                                                raw_steps=actual_steps, rollout_steps=rollout_steps,
                                                where=str(summary_path))
                    _check(audit, branch_summary.get("post_action_snapshot_included_in_deployment_capacity") is False,
                           "ablation_snapshot_capacity", str(summary_path),
                           "post-action audit snapshot must stay outside deployment capacity")
                    _ablation_capacity_checks(audit, branch_dir=branch_dir, branch=branch_summary,
                                              config=config, foundation_bytes=foundation_bytes,
                                              where=str(branch_dir))
                    source_branch_record = source_branch_records.get(key)
                    if source_branch_record is None:
                        source_branch_summary = {}
                        source_branch_path = source_case_dir / "branches" / f"{action}__{future}" / "summary.json"
                    else:
                        source_branch_path = source_case_dir / "branches" / f"{action}__{future}" / "summary.json"
                        try:
                            source_branch_summary = _json(source_branch_path)
                        except Exception as exc:
                            source_branch_summary = {}
                            audit.fail("ablation_ppo_summary_missing", str(source_branch_path),
                                       "matched PPO source summary is unreadable", error=repr(exc))
                    _check(audit, source_branch_summary.get("format") == "storage_decision_branch_v1",
                           "ablation_ppo_source_format", str(source_branch_path),
                           "matched PPO reference is not a physical source branch")
                    source_hash, action_checkpoint, _ = _ablation_checkpoint_checks(
                        audit, branch_dir=branch_dir, branch=branch_summary,
                        expected_environment_steps=expected_count, rollout_steps=rollout_steps,
                        where=str(branch_dir),
                    )
                    # Recompute the matched PPO branch's complete post-action state hash from disk.
                    try:
                        source_action_path = source_case_dir / "branches" / f"{action}__{future}" / "post_action" / "checkpoint.pt"
                        source_sim_path = source_case_dir / "branches" / f"{action}__{future}" / "post_action" / "simulator.json"
                        source_action = _checkpoint(source_action_path)
                        source_sim = _json(source_sim_path)
                        source_computed_hash = old_audit.semantic_hash({"learner": source_action, "simulator": source_sim})
                    except Exception as exc:
                        source_computed_hash = None
                        audit.fail("ablation_ppo_action_hash_parse", str(source_branch_path),
                                   "matched PPO post-action state cannot be hashed", error=repr(exc))
                    reference = branch.get("ppo_reference", {})
                    _same_metadata(audit, branch.get("matched_ppo_branch"), reference,
                                   "ablation_ppo_reference_copy", str(summary_path),
                                   "matched PPO branch reference copies differ")
                    expected_source_hash = source_branch_summary.get("action_state_hash")
                    _check(audit, reference.get("source_case_id") == case_id
                           and reference.get("action") == action and reference.get("future_regime") == future
                           and reference.get("action_state_hash") == expected_source_hash
                           and reference.get("source_summary_sha256") == _digest(source_branch_path)
                           and Path(str(reference.get("source_summary", ""))).resolve() == source_branch_path.resolve(),
                           "ablation_ppo_reference_integrity", str(summary_path),
                           "branch PPO reference does not identify the exact audited source summary")
                    _check(audit, source_hash == source_computed_hash == expected_source_hash
                           and branch.get("action_state_hash") == source_hash,
                           "ablation_post_action_hash_match", str(summary_path),
                           "no-gradient and PPO post-action learner/simulator hashes differ",
                           ppo_hash=source_computed_hash, no_gradient_hash=source_hash,
                           recorded_source_hash=expected_source_hash)
                    try:
                        no_grad_actor_path = branch_dir / "post_action" / "actor.bin"
                        source_actor_path = source_case_dir / "branches" / f"{action}__{future}" / "post_action" / "actor.bin"
                        _check(audit, no_grad_actor_path.is_file() and source_actor_path.is_file()
                               and no_grad_actor_path.stat().st_size == int(config["actor_budget_bytes"])
                               and no_grad_actor_path.read_bytes() == source_actor_path.read_bytes(),
                               "ablation_post_action_actor_match", str(summary_path),
                               "post-action actor package differs from the matched PPO source actor")
                    except Exception as exc:
                        audit.fail("ablation_post_action_actor_parse", str(summary_path),
                                   "post-action actor package comparison failed", error=repr(exc))
                    _check(audit, branch.get("ppo_reference", {}).get("final_state_hash")
                           == source_branch_summary.get("final_state_hash"),
                           "ablation_ppo_final_reference", str(summary_path),
                           "branch PPO final-state reference differs from source summary")
        # Count every persisted step file once, including unexpected/failed branches.
        expected_raw = {
            (case_dir / "branches" / f"{action}__{future}" / "steps.csv").resolve()
            for action, future in expected_keys
        }
        actual_raw_paths = {path.resolve() for path in case_dir.rglob("steps.csv") if path.is_file()}
        for extra in sorted(actual_raw_paths - expected_raw):
            extra_count, _, extra_error = _read_step_count(extra)
            if extra_count is None:
                case_raw_known = False
                audit.fail("ablation_unexpected_raw_unknown", str(extra),
                           "unexpected raw step ledger is unreadable; cost remains unknown", error=extra_error)
            else:
                case_partial += extra_count
                audit.fail("ablation_unexpected_raw_steps", str(extra),
                           "unexpected raw step ledger is included in physical interaction cost")
        # Missing expected files make the cohort total unknown, even though their observed partial cost is retained.
        for expected_path in expected_raw - actual_raw_paths:
            case_raw_known = False
        observed_case = case_partial if case_raw_known else None
        observed_partial += case_partial
        total_known = total_known and case_raw_known
        if observed_case is not None:
            observed_total += observed_case
        _check(audit, len(no_grad_records) == len(expected_keys), "ablation_case_branch_count", case_id,
               "case does not have exactly four no-gradient branches", actual=len(no_grad_records),
               expected=len(expected_keys))
        for future in futures:
            uniform_replace = case_uniforms.get(f"replace__{future}")
            uniform_none = case_uniforms.get(f"none__{future}")
            _check(audit, uniform_replace is not None and uniform_none is not None
                   and uniform_replace == uniform_none,
                   "ablation_paired_uniforms", case_id,
                   "replace and none no-gradient branches do not share their saved action-uniform stream",
                   future=future)
            for action in actions:
                key = (action, future)
                source_rows, _, source_error = _read_step_count(
                    source_case_dir / "branches" / f"{action}__{future}" / "steps.csv")
                source_uniforms: list[float] | None = None
                if source_rows is not None:
                    try:
                        _, parsed_rows, _ = _read_step_count(
                            source_case_dir / "branches" / f"{action}__{future}" / "steps.csv")
                        if parsed_rows is not None:
                            source_uniforms = [old_audit._number(row["action_uniform"]) for row in parsed_rows]
                    except Exception:
                        source_uniforms = None
                _check(audit, source_uniforms is not None
                       and case_uniforms.get(f"{action}__{future}") == source_uniforms,
                       "ablation_ppo_uniform_pairing", case_id,
                       "no-gradient branch does not preserve its PPO reference action-uniform stream",
                       action=action, future=future, source_error=source_error)
        case_costs = record.get("costs", {})
        expected_branch_names = {f"{action}__{future}" for action, future in expected_keys}
        _check(audit, set(case_costs.get("branch_steps_by_name", {})) == expected_branch_names,
               "ablation_case_raw_file_index", case_id,
               "case cost index does not list exactly the four expected branch ledgers")
        for branch_name, actual_count in case_path_rows.items():
            _check(audit, case_costs.get("branch_steps_by_name", {}).get(branch_name) == actual_count,
                   "ablation_case_branch_raw_cost", case_id,
                   "case branch cost differs from its persisted step rows", branch=branch_name,
                   recorded=case_costs.get("branch_steps_by_name", {}).get(branch_name), persisted=actual_count)
        expected_record_cost = observed_case
        _check(audit, case_costs.get("new_prefix_environment_steps") == 0
               and case_costs.get("branch_environment_steps") == expected_record_cost
               and case_costs.get("new_environment_steps_total") == expected_record_cost,
               "ablation_case_costs", case_id,
               "case cost does not match all persisted no-gradient raw rows or hides unknown cost",
               expected=expected_record_cost)
        case_reports.append({"case_id": case_id, "passed": len(audit.errors) == before_errors,
                             "errors": len(audit.errors) - before_errors,
                             "observed_steps": observed_case, "observed_partial_steps": case_partial,
                             "physical_branch_count": len(no_grad_records),
                             "missing_or_unknown_cost": not case_raw_known})

    flattened = [branch for row in case_records for branch in row.get("branches", [])]
    _same_metadata(audit, summary.get("branches"), flattened, "ablation_top_branch_flattening",
                   str(summary_path), "top-level branch list differs from case-level records")
    _same_metadata(audit, summary.get("branch_results"), flattened, "ablation_top_branch_results",
                   str(summary_path), "top-level branch_results differs from case-level records")
    expected_physical = len(expected_case_ids) * len(expected_keys)
    _check(audit, len(flattened) == expected_physical
           and int(summary.get("branch_count", -1)) == expected_physical
           and int(summary.get("physical_branch_count", -1)) == expected_physical,
           "ablation_branch_accounting", str(summary_path),
           "ablation logical and physical branch totals differ from four fresh branches per case",
           observed=len(flattened), expected=expected_physical)
    expected_by_phase = {"pilot": 12_288, "main": 1_572_864}
    expected_total = expected_by_phase.get(phase)
    _check(audit, len(expected_case_ids) == (1 if phase == "pilot" else 32),
           "ablation_protocol_case_count", str(summary_path),
           "pilot/main case count differs from the frozen protocol", phase=phase,
           actual=len(expected_case_ids), expected=1 if phase == "pilot" else 32)
    if total_known:
        _check(audit, observed_total == expected_total, "ablation_protocol_cost", str(summary_path),
               "new no-gradient interaction cost differs from the frozen pilot/main budget",
               observed=observed_total, expected=expected_total)
    expected_costs = summary.get("costs", {})
    expected_new_total = observed_total if total_known else None
    _check(audit, expected_costs.get("new_prefix_environment_steps") == 0
           and expected_costs.get("new_branch_environment_steps") == expected_new_total
           and expected_costs.get("new_environment_steps_total") == expected_new_total,
           "ablation_cohort_costs", str(summary_path),
           "cohort new interaction costs differ from persisted rows or hide unknown cost",
           observed=expected_new_total)
    source_costs = source_summary.get("costs", {})
    for field, source_field in (("source_reference_prefix_environment_steps", "prefix_environment_steps"),
                                ("source_reference_branch_environment_steps", "branch_environment_steps"),
                                ("source_reference_environment_steps_total", "environment_steps_total")):
        _check(audit, expected_costs.get(field) == source_costs.get(source_field),
               "ablation_source_cost_reference", str(summary_path),
               "source prefix/PPO cost reference differs from the audited source ledger", field=field,
               recorded=expected_costs.get(field), source=source_costs.get(source_field))
    return {
        "format": str(summary.get("format", "")),
        "source": str(cohort),
        "source_dose": str(source_root),
        "phase": phase,
        "passed": not audit.errors,
        "expected_case_count": len(expected_case_ids),
        "logical_branch_count": len(flattened),
        "physical_branch_count": sum(
            1 for branch in flattened if branch.get("format") == "learning_interaction_branch_v1"
        ),
        "cost": {
            "new_prefix_environment_steps": 0,
            "new_branch_environment_steps": expected_new_total,
            "new_environment_steps_total": expected_new_total,
            "new_actual_environment_steps_partial_raw_rows": observed_partial,
            "expected_environment_steps": expected_total,
            "source_reference_prefix_environment_steps": source_costs.get("prefix_environment_steps"),
            "source_reference_branch_environment_steps": source_costs.get("branch_environment_steps"),
            "source_reference_environment_steps_total": source_costs.get("environment_steps_total"),
        },
        "cases": case_reports,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "source_audit_passed": bool(source_report.get("passed")),
    }


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing audit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="completed source-dose cohort directory")
    parser.add_argument("--ablation", type=Path, help="optional learning-ablation cohort directory")
    parser.add_argument("--output", required=True, type=Path, help="new audit JSON path outside the cohorts")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise SystemExit("audit output must be outside the source cohort")
    if args.ablation:
        ablation = args.ablation.resolve()
        if output == ablation or ablation in output.parents:
            raise SystemExit("audit output must be outside the ablation cohort")
    source_report = audit_source(source)
    report: dict[str, Any] = {"format": FORMAT, "source_dose": source_report}
    if args.ablation:
        report["learning_ablation"] = audit_ablation(args.ablation.resolve(), source_report)
    report["passed"] = bool(source_report.get("passed")) and (
        not args.ablation or bool(report["learning_ablation"].get("passed"))
    )
    _write_report(output, report)
    print(json.dumps({
        "status": "passed" if report["passed"] else "failed",
        "output": str(output),
        "source_cases": source_report.get("expected_case_count"),
        "source_logical_branches": source_report.get("logical_branch_count"),
        "source_physical_branches": source_report.get("physical_branch_count"),
        "source_environment_steps": source_report.get("cost", {}).get("environment_steps_total"),
        "source_partial_raw_rows": source_report.get("cost", {}).get("actual_environment_steps_partial_raw_rows"),
        "source_errors": len(source_report.get("errors", [])),
        "ablation_errors": len(report.get("learning_ablation", {}).get("errors", [])),
    }, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

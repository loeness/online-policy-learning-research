"""Audit a completed single-parent storage-decision cohort without replaying it.

The audit reads persisted artifacts only. Selector recomputation restores the
learner checkpoint and evaluates its bounded prefix evidence; it never calls
the experiment schedule or steps a task environment.
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

from online_rl.storage_decision import (
    ACTIONS,
    apply_action,
    common_mean_effective,
    decide_from_past,
    effective_matrices,
    pair_mapping,
    prefix_schedule,
    semantic_hash,
    template_schedule,
    validate_config,
)
from online_rl.write_decision_policy import decode_actor, encode_actor
from scripts import run_knowledge_loop as loop
from scripts import run_storage_decision as runner
from scripts import run_write_stream as ws


FORMAT = "storage_decision_audit_v1"
MAIN_SEEDS = (20101, 20107, 20113, 20117, 20123, 20129, 20143, 20149)
MAIN_PAIRS = ("dynamics", "reward_conflict")
MAIN_HISTORIES = ("old_major", "new_major")
FUTURES = ("old_major", "new_major")
EXPECTED_ACTIONS = ("keep", "replace", "mean", "none")
CHECKPOINT_KEYS = {
    "format", "seed", "foundation_sha256", "core_state", "optimizer_state",
    "optimizer_step_count", "shared_frozen", "counts", "runtime_counts",
    "slot_generation", "fitted_steps", "cache", "evidence", "recent_groups",
    "parent_group", "slot_group_tags", "action_rng", "behavior_rng", "torch_rng",
    "numpy_rng", "python_rng",
}
DISPLACED_PARENT_KEY_FRAGMENTS = (
    "backup", "displaced", "outgoing_parent", "previous_parent", "old_parent",
    "parent_before", "evicted_parent",
)
_LOCAL_KEY = re.compile(r"^_nodes\.(\d+)\.A$")
_SHARED_KEY = re.compile(r"^_shared_coefficients\.(\d+)\.C$")


class Audit:
    def __init__(self, cohort: Path) -> None:
        self.cohort = cohort
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def fail(self, code: str, where: str, message: str, **details: Any) -> None:
        row = {"code": code, "where": where, "message": message}
        if details:
            row["details"] = details
        self.errors.append(row)

    def warn(self, code: str, where: str, message: str, **details: Any) -> None:
        row = {"code": code, "where": where, "message": message}
        if details:
            row["details"] = details
        self.warnings.append(row)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint must contain a mapping, got {type(value).__name__}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, got {value!r}")
    return result


def _close(a: Any, b: Any, *, atol: float = 1e-7, rtol: float = 1e-7) -> bool:
    """Recursive comparison for JSON, NumPy, and checkpoint values."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol, equal_nan=False))
        except (TypeError, ValueError):
            try:
                return bool(np.array_equal(np.asarray(a), np.asarray(b)))
            except (TypeError, ValueError):
                return False
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_close(a[key], b[key], atol=atol, rtol=rtol) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol))
    return a == b


def _check(audit: Audit, condition: bool, code: str, where: str, message: str, **details: Any) -> None:
    if not condition:
        audit.fail(code, where, message, **details)


def _core_arrays(checkpoint: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = checkpoint["core_state"]
    local_items: list[tuple[int, np.ndarray]] = []
    shared_items: list[tuple[int, np.ndarray]] = []
    for name, value in state.items():
        local_match = _LOCAL_KEY.match(str(name))
        shared_match = _SHARED_KEY.match(str(name))
        if local_match:
            local_items.append((int(local_match.group(1)), np.asarray(value.detach().cpu()).reshape(27).astype(np.float32)))
        elif shared_match:
            shared_items.append((int(shared_match.group(1)), np.asarray(value.detach().cpu()).reshape(9).astype(np.float32)))
    local_items.sort(key=lambda item: item[0])
    shared_items.sort(key=lambda item: item[0])
    if [index for index, _ in local_items] != list(range(1, len(local_items) + 1)):
        raise ValueError("checkpoint local actor keys are not a contiguous one-based slot sequence")
    if [index for index, _ in shared_items] != list(range(len(shared_items))):
        raise ValueError("checkpoint shared actor keys are not a contiguous zero-based group sequence")
    if not local_items or not shared_items:
        raise ValueError("checkpoint has no local or shared actor coefficients")
    local = np.stack([value for _, value in local_items]).astype(np.float32)
    shared = np.stack([value for _, value in shared_items]).astype(np.float32)
    groups = np.asarray(state["slot_groups"].detach().cpu(), dtype=np.int8).copy()
    return local, shared, groups


def _check_checkpoint_shape(audit: Audit, checkpoint: Mapping[str, Any], config: Mapping[str, Any], where: str) -> None:
    missing = sorted(CHECKPOINT_KEYS - set(checkpoint))
    _check(audit, not missing, "checkpoint_missing_fields", where,
           "learner checkpoint is missing required restoration fields", missing=missing)
    if missing:
        return
    _check(audit, checkpoint.get("format") == "storage_decision_learner_v1", "checkpoint_format", where,
           "unexpected learner checkpoint format", actual=checkpoint.get("format"))
    _check(audit, int(checkpoint.get("seed", -1)) >= 0, "checkpoint_seed", where,
           "learner checkpoint has an invalid seed", actual=checkpoint.get("seed"))
    _check(audit, bool(checkpoint.get("shared_frozen")), "shared_not_frozen", where,
           "shared parent must remain frozen after the decision")
    local, shared, groups = _core_arrays(checkpoint)
    expected_local = int(config["local_slots"])
    expected_shared = int(config["shared_groups"])
    _check(audit, local.shape == (expected_local, 27), "local_layout", where,
           "checkpoint actor local layout differs from the frozen configuration", actual=list(local.shape),
           expected=[expected_local, 27])
    _check(audit, shared.shape == (expected_shared, 9), "shared_layout", where,
           "checkpoint actor shared layout differs from the frozen configuration", actual=list(shared.shape),
           expected=[expected_shared, 9])
    _check(audit, groups.shape == (expected_local,), "physical_group_shape", where,
           "checkpoint physical group tags do not match the local slot count", actual=list(groups.shape))
    _check(audit, bool(np.all((groups == -1) | (groups == 0))), "physical_group_range", where,
           "one-parent physical slot tags must be -1 or 0", actual=groups.tolist())
    _check(audit, int(checkpoint["evidence"].get("capacity", -1)) == int(config["evidence_rows_per_group"]),
           "evidence_capacity", where, "evidence capacity differs from the frozen limit",
           actual=checkpoint["evidence"].get("capacity"), expected=config["evidence_rows_per_group"])
    for group in (0, 1):
        rows = checkpoint["evidence"]["groups"].get(group, checkpoint["evidence"]["groups"].get(str(group), {}))
        count = len(rows.get("action", []))
        field_lengths = {name: len(rows.get(name, [])) for name in (
            "obs", "action", "behavior_probs", "behavior_log_prob", "advantage", "returns", "episode_id", "task", "bout",
        )}
        _check(audit, count <= int(config["evidence_rows_per_group"]), "evidence_overflow", where,
               "bounded evidence exceeds its per-group row limit", group=group, rows=count,
               limit=config["evidence_rows_per_group"])
        _check(audit, len(set(field_lengths.values())) <= 1, "evidence_field_lengths", where,
               "evidence fields have inconsistent row counts", group=group, lengths=field_lengths)
    _check(audit, len(checkpoint.get("recent_groups", [])) <= int(config["recent_bouts"]),
           "recent_group_history_overflow", where, "recent group history exceeds its frozen limit",
           count=len(checkpoint.get("recent_groups", [])), limit=config["recent_bouts"])
    for key in checkpoint:
        _scan_displaced_keys(audit, checkpoint[key], f"{where}.{key}")


def _scan_displaced_keys(audit: Audit, value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            text = str(key).casefold()
            if any(fragment in text for fragment in DISPLACED_PARENT_KEY_FRAGMENTS):
                audit.fail("displaced_parent_field", path, "learner checkpoint exposes a displaced-parent backup field",
                           field=str(key))
            _scan_displaced_keys(audit, nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _scan_displaced_keys(audit, nested, f"{path}[{index}]")


def _package_check(
    audit: Audit,
    *,
    folder: Path,
    actor_path: Path,
    checkpoint_path: Path,
    recorded: Mapping[str, Any] | None,
    foundation_bytes: int,
    package_cap: int,
    actor_cap: int,
    where: str,
) -> dict[str, int] | None:
    if not actor_path.is_file() or not checkpoint_path.is_file():
        audit.fail("package_file_missing", where, "complete persistent package is missing its actor or learner checkpoint",
                   actor_exists=actor_path.is_file(), checkpoint_exists=checkpoint_path.is_file())
        return None
    actual = {
        "foundation_bytes": int(foundation_bytes),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "actor_bytes": actor_path.stat().st_size,
    }
    actual["full_persistent_bytes"] = sum(actual.values())
    actual["package_budget_bytes"] = int(package_cap)
    _check(audit, actual["actor_bytes"] == actor_cap, "actor_capacity", where,
           "actor sidecar does not match the fixed byte budget", actual=actual["actor_bytes"], expected=actor_cap)
    _check(audit, actual["full_persistent_bytes"] <= package_cap, "package_capacity", where,
           "complete W0, checkpoint, and actor sidecar exceed the persistent package cap",
           actual=actual["full_persistent_bytes"], cap=package_cap)
    if recorded is not None:
        _check(audit, _close(actual, recorded, atol=0.0, rtol=0.0), "capacity_record_mismatch", where,
               "recorded package capacity does not match file sizes", actual=actual, recorded=dict(recorded))
    try:
        payload = actor_path.read_bytes()
        local, shared, groups = decode_actor(payload)
        _check(audit, len(payload) == actor_cap and local.shape == (2, 27) and shared.shape == (1, 9)
               and groups.shape == (2,), "actor_layout", where,
               "actor sidecar must encode two local slots and one shared parent in int8 form",
               payload_bytes=len(payload), local_shape=list(local.shape), shared_shape=list(shared.shape),
               group_shape=list(groups.shape))
    except Exception as exc:
        audit.fail("actor_decode", where, "actor sidecar could not be decoded", error=str(exc))
    return actual


def _validate_source_snapshot(audit: Audit, cohort: Path, summary: Mapping[str, Any]) -> int | None:
    manifest = summary.get("source_manifest")
    manifest_path = cohort / "source_snapshot" / "manifest.json"
    if not manifest_path.is_file():
        audit.fail("source_manifest_missing", str(manifest_path), "source snapshot manifest is missing")
        return None
    try:
        on_disk = _json(manifest_path)
    except Exception as exc:
        audit.fail("source_manifest_parse", str(manifest_path), "source manifest cannot be parsed", error=str(exc))
        return None
    _check(audit, isinstance(manifest, Mapping) and _close(manifest, on_disk), "source_manifest_mismatch",
           "summary.json", "cohort summary and persisted source manifests differ")
    sources = on_disk.get("sources", {})
    if not isinstance(sources, Mapping) or not sources:
        audit.fail("source_manifest_empty", str(manifest_path), "source manifest has no listed source files")
        return None
    for name, record in sources.items():
        parts = Path(str(name).replace("\\", "/")).parts
        _check(audit, not any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in parts),
               "private_source_listed", f"source_snapshot/{name}",
               "source manifest lists a private instruction or temporary path")
        copied = cohort / "source_snapshot" / Path(*parts)
        if not copied.is_file():
            audit.fail("source_snapshot_file_missing", str(copied), "manifest-listed source copy is missing")
            continue
        expected_sha = record.get("sha256") if isinstance(record, Mapping) else None
        expected_bytes = record.get("bytes") if isinstance(record, Mapping) else None
        _check(audit, copied.stat().st_size == expected_bytes and _sha(copied) == expected_sha,
               "source_snapshot_hash", str(copied), "manifest-listed source copy does not match its digest",
               expected_bytes=expected_bytes, actual_bytes=copied.stat().st_size,
               expected_sha256=expected_sha, actual_sha256=_sha(copied))
    for path in (cohort / "source_snapshot").rglob("*"):
        relative = path.relative_to(cohort / "source_snapshot")
        _check(audit, not any(part.casefold() in {"agent.md", "agents.md", "tmp"} for part in relative.parts),
               "private_source_present", str(path), "source snapshot contains a private instruction or temporary path")
    foundation = on_disk.get("foundation", {})
    if not isinstance(foundation, Mapping):
        audit.fail("foundation_manifest", str(manifest_path), "source manifest foundation record is malformed")
        return None
    foundation_path = Path(str(foundation.get("path", "")))
    foundation_bytes = int(foundation.get("bytes", -1))
    if not foundation_path.is_file():
        audit.fail("foundation_missing", str(foundation_path), "manifest W0 foundation file is not available for audit")
        return foundation_bytes
    actual_sha = _sha(foundation_path)
    _check(audit, foundation_path.stat().st_size == foundation_bytes and actual_sha == foundation.get("sha256"),
           "foundation_hash", str(foundation_path), "W0 foundation differs from its cohort source manifest",
           expected_bytes=foundation_bytes, actual_bytes=foundation_path.stat().st_size,
           expected_sha256=foundation.get("sha256"), actual_sha256=actual_sha)
    return foundation_bytes


def _config_expected(audit: Audit, cohort: Path, summary: Mapping[str, Any]) -> dict[str, Any] | None:
    config_path = cohort / "config.json"
    original_path = cohort / "source_snapshot" / "config_original.json"
    try:
        config = _json(config_path)
        original = _json(original_path)
    except Exception as exc:
        audit.fail("config_missing_or_invalid", str(cohort), "cohort config or original input config cannot be read", error=str(exc))
        return None
    phase = str(summary.get("phase", ""))
    phase_config = dict(original)
    if phase == "pilot":
        pilot = original.get("pilot")
        if not isinstance(pilot, Mapping):
            audit.fail("pilot_config_missing", str(original_path), "pilot cohort has no pilot configuration")
            return None
        for key in ("seeds", "pair_kinds", "history_regimes", "bootstrap_steps", "bout_steps"):
            phase_config[key] = pilot[key]
    elif phase != "main":
        audit.fail("unknown_phase", "summary.json", "cohort phase must be main or pilot", actual=phase)
        return None
    try:
        expected = validate_config(phase_config)
    except Exception as exc:
        audit.fail("config_invalid", str(original_path), "frozen cohort config is not valid", error=str(exc))
        return None
    _check(audit, _close(config, expected, atol=0.0, rtol=0.0), "config_normalization", "config.json",
           "persisted phase config differs from the normalized original experiment config")
    _check(audit, _close(summary.get("config"), config, atol=0.0, rtol=0.0), "summary_config_mismatch",
           "summary.json", "cohort summary config differs from config.json")
    if phase == "main":
        _check(audit, tuple(config.get("seeds", ())) == MAIN_SEEDS, "main_seed_universe", "config.json",
               "main cohort must include the eight frozen deployment seeds", actual=config.get("seeds"), expected=list(MAIN_SEEDS))
        _check(audit, tuple(config.get("pair_kinds", ())) == MAIN_PAIRS, "main_pair_universe", "config.json",
               "main cohort must include both frozen pair kinds", actual=config.get("pair_kinds"), expected=list(MAIN_PAIRS))
        _check(audit, tuple(config.get("history_regimes", ())) == MAIN_HISTORIES, "main_history_universe", "config.json",
               "main cohort must include both frozen history regimes", actual=config.get("history_regimes"), expected=list(MAIN_HISTORIES))
    else:
        _check(audit, tuple(config.get("seeds", ())) == (20089,), "pilot_seed", "config.json",
               "pilot must use the frozen independent seed 20089", actual=config.get("seeds"))
        _check(audit, tuple(config.get("pair_kinds", ())) == ("dynamics",), "pilot_pair", "config.json",
               "pilot must use the frozen dynamics relation", actual=config.get("pair_kinds"))
        _check(audit, tuple(config.get("history_regimes", ())) == ("old_major",), "pilot_history", "config.json",
               "pilot must use the frozen old-major history", actual=config.get("history_regimes"))
    _check(audit, tuple(config.get("actions", ())) == EXPECTED_ACTIONS, "action_universe", "config.json",
           "cohort must include keep, replace, mean, and none in the frozen order", actual=config.get("actions"))
    _check(audit, tuple(config.get("future_regimes", ())) == FUTURES, "future_universe", "config.json",
           "cohort must include both continuation futures", actual=config.get("future_regimes"))
    _check(audit, int(config.get("local_slots", -1)) == 2 and int(config.get("shared_groups", -1)) == 1,
           "actor_config_layout", "config.json", "cohort must use two local slots and one shared parent",
           local_slots=config.get("local_slots"), shared_groups=config.get("shared_groups"))
    _check(audit, int(config.get("actor_budget_bytes", -1)) == 89, "actor_config_budget", "config.json",
           "cohort actor payload cap must be 89 bytes", actual=config.get("actor_budget_bytes"))
    _check(audit, int(config.get("evidence_rows_per_group", -1)) == 256 and int(config.get("recent_bouts", -1)) == 8,
           "selector_config_limits", "config.json", "selector evidence must be capped at 256 rows per group and 8 groups",
           evidence_rows_per_group=config.get("evidence_rows_per_group"), recent_bouts=config.get("recent_bouts"))
    scope = str(config.get("scope", "")).casefold()
    _check(audit, "known task ids" in scope and "boundaries" in scope and "group supplied" in scope,
           "known_metadata_scope", "config.json", "cohort config does not declare supplied task IDs, boundaries, and group metadata",
           scope=config.get("scope"))
    return config


def _check_schedule_rows(
    audit: Audit,
    rows: Sequence[Mapping[str, str]],
    schedule: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    action: str,
    future: str,
    global_start: int,
    seed: int,
    where: str,
) -> tuple[int, float, list[float]]:
    expected_count = sum(int(item["steps"]) for item in schedule)
    _check(audit, len(rows) == expected_count, "raw_step_count", where,
           "raw steps CSV row count differs from its continuous schedule", actual=len(rows), expected=expected_count)
    reward_total = 0.0
    uniforms: list[float] = []
    bout_starts: list[int] = []
    running = 0
    for item in schedule:
        bout_starts.append(running)
        running += int(item["steps"])
    for index, row in enumerate(rows):
        schedule_index = len(schedule) - 1
        for candidate_index, item in enumerate(schedule):
            if index < bout_starts[candidate_index] + int(item["steps"]):
                schedule_index = candidate_index
                break
        expected_bout = schedule[schedule_index]
        within_bout = index - bout_starts[schedule_index] + 1
        actual_phase = row.get("phase", "")
        expected_phase = "bootstrap" if phase == "prefix" and expected_bout["phase"] == "bootstrap" else (
            "prefix_history" if phase == "prefix" else "future"
        )
        if phase == "prefix" and expected_bout["phase"] == "history":
            expected_phase = "prefix_history"
        _check(audit, actual_phase == expected_phase, "raw_step_phase", where,
               "step row has the wrong phase", row=index, actual=actual_phase, expected=expected_phase)
        _check(audit, row.get("action", "") == (action if phase == "future" else ""),
               "raw_step_action", where, "step row action label is inconsistent", row=index,
               actual=row.get("action"), expected=action if phase == "future" else "")
        _check(audit, row.get("future_regime", "") == (future if phase == "future" else ""),
               "raw_step_future", where, "step row future label is inconsistent", row=index,
               actual=row.get("future_regime"), expected=future if phase == "future" else "")
        expected_global = global_start + index + 1
        try:
            global_step = int(row["global_step"])
            bout = int(row["bout"])
            bout_step = int(row["bout_step"])
            group = int(row["group"])
            action_id = int(row["action_id"])
            episode_id = int(row["episode_id"])
            reward = _number(row["reward"])
            probabilities = np.asarray(json.loads(row["probabilities"]), dtype=np.float64)
            uniform = _number(row["action_uniform"])
            task = row["taskobserver"]
        except Exception as exc:
            audit.fail("raw_step_parse", where, "step row contains malformed numeric or JSON data", row=index, error=str(exc))
            continue
        _check(audit, global_step == expected_global, "raw_step_continuity", where,
               "raw global step sequence has a gap or duplicate", row=index, actual=global_step, expected=expected_global)
        _check(audit, bout == int(expected_bout["bout"]) and bout_step == within_bout,
               "raw_step_schedule", where, "raw step does not match its scheduled bout and within-bout position",
               row=index, bout=bout, expected_bout=expected_bout["bout"], bout_step=bout_step,
               expected_bout_step=within_bout)
        _check(audit, group == int(expected_bout["group"]) and task == str(expected_bout["task"]),
               "known_task_metadata", where, "observed task ID or supplied group differs from the schedule",
               row=index, task=task, expected_task=expected_bout["task"], group=group,
               expected_group=expected_bout["group"])
        _check(audit, probabilities.shape == (4,) and np.isfinite(probabilities).all()
               and np.all(probabilities >= 0) and math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-6),
               "behavior_probabilities", where, "action probabilities are malformed", row=index,
               probabilities=probabilities.tolist())
        _check(audit, 0.0 <= uniform < 1.0, "action_uniform_range", where,
               "raw action uniform is outside the half-open unit interval", row=index, uniform=uniform)
        if probabilities.shape == (4,) and np.isfinite(probabilities).all() and 0 <= action_id < 4:
            reconstructed = min(int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")), 3)
            _check(audit, reconstructed == action_id, "action_uniform_usage", where,
                   "recorded action is inconsistent with its probability vector and action uniform",
                   row=index, action_id=action_id, reconstructed=reconstructed)
            try:
                behavior_log_prob = _number(row["behavior_log_prob"])
                expected_log_prob = math.log(max(float(probabilities[action_id]), 1e-12))
                _check(audit, math.isclose(behavior_log_prob, expected_log_prob, abs_tol=2e-6, rel_tol=2e-6),
                       "behavior_log_probability", where, "logged behavior log probability is inconsistent",
                       row=index, actual=behavior_log_prob, expected=expected_log_prob)
            except Exception as exc:
                audit.fail("behavior_log_probability_parse", where, "behavior log probability is malformed",
                           row=index, error=str(exc))
        try:
            expected_reset_seed = runner._reset_seed(seed, episode_id)
            actual_reset_seed = int(row["reset_seed"])
            _check(audit, actual_reset_seed == expected_reset_seed, "episode_reset_seed", where,
                   "step row reset seed does not match its episode ordinal", row=index,
                   actual=actual_reset_seed, expected=expected_reset_seed)
        except (KeyError, ValueError, TypeError):
            audit.fail("episode_reset_seed_missing", where, "step row lacks a valid reset seed", row=index)
        reward_total += reward
        uniforms.append(uniform)
        environment_done = bool(int(row.get("environment_done", "0")))
        learning_boundary = bool(int(row.get("learning_boundary", "0")))
        terminated = bool(int(row.get("terminated", "0")))
        truncated = bool(int(row.get("truncated", "0")))
        _check(audit, environment_done == bool(terminated or truncated), "environment_done_flag", where,
               "environment_done must reflect termination or truncation", row=index)
        should_boundary = within_bout == int(expected_bout["steps"])
        _check(audit, learning_boundary == should_boundary, "learning_boundary_flag", where,
               "learning boundary must occur exactly at the end of each scheduled bout", row=index,
               actual=learning_boundary, expected=should_boundary)
        _check(audit, not (learning_boundary and environment_done) or should_boundary,
               "boundary_accounting", where, "environment and learning boundaries were not recorded distinctly", row=index)
        if index + 1 < len(rows) and episode_id == int(rows[index + 1].get("episode_id", -1)):
            try:
                next_obs = np.asarray(json.loads(row["next_obs"]), dtype=np.float64)
                following_obs = np.asarray(json.loads(rows[index + 1]["obs"]), dtype=np.float64)
                _check(audit, np.allclose(next_obs, following_obs, atol=1e-6, rtol=0.0),
                       "trajectory_observation_continuity", where,
                       "next observation does not match the next row in the same episode", row=index)
            except Exception as exc:
                audit.fail("trajectory_observation_parse", where, "step observation data cannot be parsed", row=index, error=str(exc))
    return len(rows), reward_total, uniforms


def _check_rollouts(
    audit: Audit,
    folder: Path,
    *,
    phase: str | Sequence[str],
    expected_rows: int,
    summary: Mapping[str, Any],
    where: str,
) -> dict[str, int]:
    path = folder / "rollouts.csv"
    try:
        rows = _csv(path)
    except Exception as exc:
        audit.fail("rollouts_csv_missing", str(path), "rollout submission ledger cannot be read", error=str(exc))
        return {"rollouts": 0, "rows": 0, "updates": 0}
    submitted = 0
    updates = 0
    expected_phases = {phase} if isinstance(phase, str) else set(phase)
    for index, row in enumerate(rows):
        try:
            count = int(row["rows"])
            update_count = int(row["updates"])
        except Exception as exc:
            audit.fail("rollout_row_parse", str(path), "rollout ledger has malformed counts", row=index, error=str(exc))
            continue
        submitted += count
        updates += update_count
        _check(audit, row.get("phase") in expected_phases, "rollout_phase", str(path),
               "rollout ledger contains a row from the wrong phase", row=index, actual=row.get("phase"))
        _check(audit, count == int(summary.get("rollout_steps", 256)), "rollout_size", str(path),
               "each submitted PPO rollout must contain the configured number of rows", row=index, actual=count,
               expected=summary.get("rollout_steps", 256))
        _check(audit, int(row.get("actor_optimizer_state_cleared", "0")) == 1, "actor_optimizer_clear", str(path),
               "actor optimizer state was not recorded as cleared after the PPO rollout", row=index)
    expected_rollouts = expected_rows // int(summary.get("rollout_steps", 256))
    _check(audit, submitted == expected_rows, "ppo_row_coverage", str(path),
           "PPO rollout ledger does not account for every fresh step exactly once",
           submitted_rows=submitted, fresh_rows=expected_rows)
    _check(audit, len(rows) == expected_rollouts, "ppo_rollout_count", str(path),
           "rollout count differs from the fresh-row budget", actual=len(rows), expected=expected_rollouts)
    counts = summary.get("counts", {})
    _check(audit, int(counts.get("environment_steps", -1)) == expected_rows, "summary_environment_steps",
           where, "summary environment-step count differs from the raw step ledger",
           actual=counts.get("environment_steps"), expected=expected_rows)
    _check(audit, int(counts.get("rollouts_trained", -1)) == len(rows), "summary_rollout_count",
           where, "summary trained-rollout count differs from the rollout ledger",
           actual=counts.get("rollouts_trained"), expected=len(rows))
    _check(audit, int(counts.get("minibatch_updates", -1)) == updates, "summary_update_count",
           where, "summary minibatch-update count differs from the rollout ledger",
           actual=counts.get("minibatch_updates"), expected=updates)
    return {"rollouts": len(rows), "rows": submitted, "updates": updates}


def _check_reward_summary(
    audit: Audit,
    raw_rows: Sequence[Mapping[str, str]],
    bout_summary: Mapping[str, Any],
    *,
    raw_reward: float,
    raw_steps: int,
    where: str,
) -> None:
    _check(audit, int(bout_summary.get("environment_steps", -1)) == raw_steps, "reward_step_total", where,
           "summary step count differs from raw steps", summary_steps=bout_summary.get("environment_steps"), raw_steps=raw_steps)
    summary_reward = _number(bout_summary.get("reward_total", float("nan")))
    _check(audit, math.isclose(summary_reward, raw_reward, abs_tol=1e-6, rel_tol=1e-8), "reward_total", where,
           "summary reward total differs from raw reward rows", summary_reward=summary_reward, raw_reward=raw_reward)
    expected_rate = raw_reward / max(1, raw_steps)
    _check(audit, math.isclose(_number(bout_summary.get("reward_per_step", float("nan"))), expected_rate,
                               abs_tol=1e-7, rel_tol=1e-7), "reward_rate", where,
           "summary reward per step differs from raw steps", expected=expected_rate,
           actual=bout_summary.get("reward_per_step"))
    per_bout = bout_summary.get("per_bout", [])
    rewards_by_bout: dict[int, float] = {}
    steps_by_bout: dict[int, int] = {}
    groups_by_bout: dict[int, int] = {}
    for row in raw_rows:
        try:
            bout = int(row["bout"])
            rewards_by_bout[bout] = rewards_by_bout.get(bout, 0.0) + _number(row["reward"])
            steps_by_bout[bout] = steps_by_bout.get(bout, 0) + 1
            groups_by_bout[bout] = int(row["group"])
        except Exception:
            continue
    _check(audit, len(per_bout) == len(rewards_by_bout), "per_bout_count", where,
           "summary bout count differs from raw step bouts", summary=len(per_bout), raw=len(rewards_by_bout))
    for bout_record in per_bout:
        try:
            bout = int(bout_record["bout"])
            expected = rewards_by_bout[bout]
            _check(audit, int(bout_record["steps"]) == steps_by_bout[bout]
                   and math.isclose(_number(bout_record["return_full"]), expected, abs_tol=1e-6, rel_tol=1e-8),
                   "per_bout_reward", where, "per-bout reward or step count differs from raw rows",
                   bout=bout, summary_steps=bout_record.get("steps"), raw_steps=steps_by_bout[bout],
                   summary_reward=bout_record.get("return_full"), raw_reward=expected)
            _check(audit, int(bout_record["group"]) == groups_by_bout[bout], "per_bout_group", where,
                   "per-bout group differs from raw step metadata", bout=bout,
                   summary_group=bout_record.get("group"), raw_group=groups_by_bout[bout])
        except Exception as exc:
            audit.fail("per_bout_parse", where, "per-bout summary record is malformed", error=str(exc), record=bout_record)
    per_group = bout_summary.get("per_group", {})
    for group in (0, 1):
        raw_group_rows = [row for row in raw_rows if int(row.get("group", -1)) == group]
        group_steps = len(raw_group_rows)
        group_reward = sum(_number(row["reward"]) for row in raw_group_rows)
        summary_group = per_group.get(str(group), {})
        _check(audit, int(summary_group.get("steps", -1)) == group_steps
               and math.isclose(_number(summary_group.get("reward_total", float("nan"))), group_reward,
                                abs_tol=1e-6, rel_tol=1e-8),
               "per_group_reward", where, "per-group reward or step count differs from raw rows",
               group=group, summary=summary_group, raw_steps=group_steps, raw_reward=group_reward)
    summary_counts = bout_summary.get("counts", {})
    allocations = sum(bool(row.get("allocated", False)) for row in per_bout)
    evictions = sum(row.get("evicted_task") is not None for row in per_bout)
    _check(audit, int(summary_counts.get("slot_allocations", -1)) == allocations,
           "allocation_count", where, "summary allocation count differs from recorded cache events",
           summary=summary_counts.get("slot_allocations"), per_bout=allocations)
    _check(audit, int(summary_counts.get("slot_evictions", -1)) == evictions,
           "eviction_count", where, "summary eviction count differs from recorded cache events",
           summary=summary_counts.get("slot_evictions"), per_bout=evictions)


def _check_cache_and_inheritance(
    audit: Audit,
    bout_records: Sequence[Mapping[str, Any]],
    start_cache: Mapping[str, Any],
    start_generation: Sequence[int],
    start_physical: Sequence[int],
    start_logical: Sequence[int],
    *,
    parent_group: int,
    config: Mapping[str, Any],
    where: str,
) -> tuple[dict[str, Any], list[int], list[int], list[int]]:
    slots = int(config["local_slots"])
    slot_to_task = list(start_cache["slot_to_task"])
    task_to_slot = {str(task): int(slot) for task, slot in start_cache["task_to_slot"].items()}
    last_used = [int(value) for value in start_cache["last_used"]]
    frequency = [int(value) for value in start_cache["frequency"]]
    clock = int(start_cache["clock"])
    generation = [int(value) for value in start_generation]
    physical = [int(value) for value in start_physical]
    logical = [int(value) for value in start_logical]
    if len(logical) != slots:
        raise ValueError("initial logical group tags do not match the two-slot cache")
    _check(audit, slots == 2, "cache_capacity", where, "known-task LRU must retain exactly two local slots", slots=slots)
    for index, record in enumerate(bout_records):
        task = str(record.get("task", ""))
        group = int(record.get("group", -1))
        clock += 1
        allocated = task not in task_to_slot
        evicted_task = None
        if not allocated:
            slot = task_to_slot[task]
            last_used[slot] = clock
            frequency[slot] += 1
            _check(audit, logical[slot] == group, "lru_logical_group", where,
                   "cache hit task's known group differs from its resident logical tag", bout=index,
                   task=task, resident_group=logical[slot], scheduled_group=group)
        else:
            available = [slot for slot, resident in enumerate(slot_to_task) if resident is None]
            slot = available[0] if available else min(range(slots), key=lambda item: (last_used[item], item))
            evicted_task = slot_to_task[slot]
            if evicted_task is not None:
                task_to_slot.pop(str(evicted_task), None)
            slot_to_task[slot] = task
            task_to_slot[task] = slot
            last_used[slot] = clock
            frequency[slot] = 1
            generation[slot] += 1
            logical[slot] = group
            physical[slot] = 0 if parent_group == -2 or parent_group == group else -1
        inherited = bool(allocated and (parent_group == -2 or parent_group == group))
        _check(audit, int(record.get("slot", -1)) == slot, "lru_slot", where,
               "scheduled task was assigned to a slot inconsistent with the two-slot LRU", bout=index,
               task=task, actual=record.get("slot"), expected=slot)
        _check(audit, bool(record.get("allocated", False)) == allocated, "lru_allocation", where,
               "task allocation indicator differs from the reconstructed cache", bout=index,
               task=task, actual=record.get("allocated"), expected=allocated)
        _check(audit, record.get("evicted_task") == evicted_task, "lru_eviction", where,
               "evicted task differs from reconstructed LRU victim", bout=index,
               task=task, actual=record.get("evicted_task"), expected=evicted_task)
        _check(audit, int(record.get("generation", -1)) == generation[slot], "slot_generation", where,
               "slot generation differs from reconstructed allocation history", bout=index,
               actual=record.get("generation"), expected=generation[slot])
        _check(audit, int(record.get("parent_group_at_start", -999)) == parent_group, "parent_group_at_start", where,
               "parent group changed during a phase or was logged incorrectly", bout=index,
               actual=record.get("parent_group_at_start"), expected=parent_group)
        _check(audit, int(record.get("physical_group_at_start", -999)) == physical[slot], "physical_group_at_start",
               where, "physical group at bout start differs from LRU/inheritance reconstruction", bout=index,
               actual=record.get("physical_group_at_start"), expected=physical[slot])
        _check(audit, bool(record.get("prior_inherited", False)) == inherited, "prior_inheritance", where,
               "parent inheritance flag differs from action and known group metadata", bout=index,
               actual=record.get("prior_inherited"), expected=inherited)
    final_cache = {
        "capacity": slots, "eviction": "lru", "task_to_slot": task_to_slot,
        "slot_to_task": slot_to_task, "last_used": last_used, "frequency": frequency,
        "clock": clock,
    }
    return final_cache, generation, physical, logical


def _check_selector(audit: Audit, case_dir: Path, decision: Mapping[str, Any], config: Mapping[str, Any], prefix_checkpoint: Mapping[str, Any]) -> None:
    where = str(case_dir / "decision.json")
    contract = decision.get("selector_contract", {})
    _check(audit, contract.get("format") == "storage_decision_past_selector_contract_v1", "selector_contract",
           where, "decision record lacks the frozen past-only selector contract", actual=contract.get("format"))
    _check(audit, contract.get("inputs", {}).get("group_evidence") == "prefix/checkpoint.pt:evidence"
           and contract.get("inputs", {}).get("surrogate_score_parameters") == "config.gate"
           and contract.get("outputs") == "selector_diagnostics.past_only_selectors",
           "selector_contract_inputs", where, "selector contract does not identify the bounded prefix inputs and output")
    checkpoint_path = case_dir / "prefix" / "checkpoint.pt"
    _check(audit, int(contract.get("evidence_capacity_rows_per_group", -1)) == int(config["evidence_rows_per_group"]),
           "selector_contract_capacity", where, "selector contract evidence limit differs from config",
           actual=contract.get("evidence_capacity_rows_per_group"), expected=config["evidence_rows_per_group"])
    try:
        base_config = loop.load_base_config()
        agent = runner._restore_learner(prefix_checkpoint, base_config, config)
        old_parent = np.asarray(decision["old_parent"], dtype=np.float32).reshape(3, 3)
        new_parent = np.asarray(decision["new_candidate"], dtype=np.float32).reshape(3, 3)
        recomputed = decide_from_past(
            agent.recent_groups,
            agent.evidence,
            old_parent,
            new_parent,
            runner._score_fn(agent.core, config),
        )
        _check(audit, _close(recomputed, decision.get("selector_diagnostics"), atol=1e-6, rtol=1e-6),
               "selector_recomputation", where,
               "past-only selector diagnostics do not recompute from the saved prefix checkpoint",
               recomputed=recomputed, recorded=decision.get("selector_diagnostics"))
        _check(audit, _close(recomputed["past_only_selectors"], decision.get("selectors"), atol=0.0, rtol=0.0),
               "selector_actions", where, "saved selector choices differ from recomputed past-only choices",
               recomputed=recomputed["past_only_selectors"], recorded=decision.get("selectors"))
        _check(audit, recomputed["recent_group_ids"] == list(prefix_checkpoint["recent_groups"]),
               "selector_recent_history", where, "selector history differs from prefix checkpoint")
        _check(audit, recomputed["support_rows"] == contract.get("evidence_support_rows"),
               "selector_support_count", where, "selector contract support counts differ from checkpoint evidence",
               recomputed=recomputed["support_rows"], recorded=contract.get("evidence_support_rows"))
        _check(audit, recomputed["support_rows"] == decision.get("evidence_rows_per_group"),
               "selector_evidence_count", where, "decision evidence counts differ from prefix checkpoint",
               recomputed=recomputed["support_rows"], recorded=decision.get("evidence_rows_per_group"))
    except Exception as exc:
        audit.fail("selector_recomputation_error", where,
                   "past-only selectors could not be recomputed from the prefix checkpoint", error=repr(exc),
                   checkpoint=str(checkpoint_path))


def _branch_core_check(
    audit: Audit,
    *,
    branch_dir: Path,
    prefix_checkpoint: Mapping[str, Any],
    post_checkpoint: Mapping[str, Any],
    prefix_sim: Mapping[str, Any],
    decision: Mapping[str, Any],
    branch_summary: Mapping[str, Any],
    action: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    where = str(branch_dir)
    try:
        pre_local, pre_shared, pre_groups = _core_arrays(prefix_checkpoint)
        actual_local, actual_shared, actual_groups = _core_arrays(post_checkpoint)
        tags = np.asarray(prefix_checkpoint["slot_group_tags"], dtype=np.int8)
        old_parent = np.asarray(decision["old_parent"], dtype=np.float32).reshape(3, 3)
        new_parent = np.asarray(decision["new_candidate"], dtype=np.float32).reshape(3, 3)
        expected_local, expected_shared, expected_groups, _ = apply_action(
            pre_local, pre_shared, pre_groups, tags, old_parent, new_parent, action,
        )
        expected_actor = encode_actor(
            expected_local, expected_shared, expected_groups, codec="int8",
            budget_bytes=int(config["actor_budget_bytes"]),
        ).data
        decoded_local, decoded_shared, decoded_groups = decode_actor(expected_actor)
        _check(audit, np.array_equal(actual_local, decoded_local)
               and np.array_equal(actual_shared, decoded_shared)
               and np.array_equal(actual_groups, decoded_groups),
               "action_state", where, "post-action checkpoint does not contain the requested action after int8 codec",
               action=action)
        actor_path = branch_dir / "post_action" / "actor.bin"
        if actor_path.is_file():
            _check(audit, actor_path.read_bytes() == expected_actor, "action_actor_payload", where,
                   "post-action actor sidecar differs from the re-encoded requested action", action=action)
        else:
            audit.fail("post_action_actor_missing", str(actor_path), "action-time actor sidecar is missing")

        pre_effective = effective_matrices(pre_local, pre_shared, pre_groups)
        action_effective = effective_matrices(expected_local, expected_shared, expected_groups)
        preservation = np.max(np.abs(action_effective - pre_effective), axis=(1, 2, 3))
        recorded_preservation = np.asarray(branch_summary.get("prequantized_effective_errors", []), dtype=np.float64)
        _check(audit, recorded_preservation.shape == preservation.shape
               and np.allclose(recorded_preservation, preservation, atol=1e-6, rtol=1e-6),
               "prequantized_preservation_record", where,
               "recorded pre-codec resident-policy preservation errors differ from recomputation",
               recorded=recorded_preservation.tolist(), recomputed=preservation.tolist())
        _check(audit, float(np.max(preservation, initial=0.0)) <= 1e-5, "prequantized_preservation",
               where, "action changed a resident effective matrix before codec quantization",
               maximum=float(np.max(preservation, initial=0.0)))

        post_effective = effective_matrices(actual_local, actual_shared, actual_groups)
        codec_effective = np.max(np.abs(post_effective - action_effective), axis=(1, 2, 3))
        recorded_codec = np.asarray(branch_summary.get("codec_effective_errors", []), dtype=np.float64)
        coefficient_error = max(
            float(np.max(np.abs(actual_local - expected_local), initial=0.0)),
            float(np.max(np.abs(actual_shared - expected_shared), initial=0.0)),
        )
        _check(audit, recorded_codec.shape == codec_effective.shape
               and np.allclose(recorded_codec, codec_effective, atol=1e-7, rtol=1e-6),
               "codec_drift_record", where, "recorded codec effective drift differs from exact int8 roundtrip",
               recorded=recorded_codec.tolist(), recomputed=codec_effective.tolist())
        _check(audit, math.isclose(float(branch_summary.get("codec_max_coefficient_error", float("nan"))),
                                   coefficient_error, abs_tol=1e-7, rel_tol=1e-6),
               "codec_coefficient_record", where, "recorded maximum coefficient drift differs from int8 roundtrip",
               recorded=branch_summary.get("codec_max_coefficient_error"), recomputed=coefficient_error)
        _check(audit, np.isfinite(codec_effective).all() and np.isfinite(coefficient_error), "codec_drift_finite",
               where, "int8 codec produced non-finite actor drift")
        # The format uses row-wise round-to-nearest int8 quantization, so every
        # coefficient must lie within half of its row scale, up to float error.
        local_scales = np.max(np.abs(expected_local), axis=1) / 127.0
        shared_scales = np.max(np.abs(expected_shared), axis=1) / 127.0
        local_scales[local_scales == 0.0] = 1.0
        shared_scales[shared_scales == 0.0] = 1.0
        bound = max(float(np.max(local_scales, initial=0.0)), float(np.max(shared_scales, initial=0.0))) / 2.0 + 2e-7
        _check(audit, coefficient_error <= bound, "codec_quantization_bound", where,
               "int8 coefficient drift exceeds the round-to-nearest quantization bound",
               actual=coefficient_error, bound=bound)

        expected_parent_group = {"keep": 0, "replace": 1, "mean": -2, "none": -1}[action]
        _check(audit, int(post_checkpoint["parent_group"]) == expected_parent_group,
               "action_parent_group", where, "saved logical parent eligibility differs from the action",
               action=action, actual=post_checkpoint["parent_group"], expected=expected_parent_group)
        _check(audit, np.array_equal(np.asarray(post_checkpoint["slot_group_tags"], dtype=np.int8), tags),
               "action_logical_tags", where, "decision action changed resident known-group tags")
        sim_path = branch_dir / str(branch_summary.get("post_action_simulator", "post_action/simulator.json"))
        post_sim = _json(sim_path)
        _check(audit, _close(post_sim, prefix_sim, atol=0.0, rtol=0.0), "fork_simulator_state", where,
               "post-action simulator snapshot differs from the common prefix simulator state")
        prefix_hash = semantic_hash({"learner": prefix_checkpoint, "simulator": prefix_sim})
        action_hash = semantic_hash({"learner": post_checkpoint, "simulator": post_sim})
        _check(audit, branch_summary.get("prefix_state_hash") == prefix_hash
               and branch_summary.get("restored_state_hash") == prefix_hash,
               "prefix_restore_hash", where, "branch restore hash does not equal the recomputed prefix hash",
               prefix_hash=prefix_hash, recorded_prefix=branch_summary.get("prefix_state_hash"),
               recorded_restore=branch_summary.get("restored_state_hash"))
        _check(audit, branch_summary.get("action_state_hash") == action_hash,
               "action_state_hash", where, "post-action checkpoint and simulator do not match the recorded action hash",
               recorded=branch_summary.get("action_state_hash"), recomputed=action_hash)

        # Action installation must not alter optimizer, evidence, cache, counters,
        # RNG streams, or non-actor core parameters. These fields are the restored
        # learner state; only the declared actor matrices and parent eligibility vary.
        excluded_top = {"core_state", "parent_group"}
        untouched_top = set(prefix_checkpoint) - excluded_top
        top_match = all(_close(prefix_checkpoint[key], post_checkpoint.get(key), atol=0.0, rtol=0.0)
                        for key in untouched_top)
        _check(audit, top_match, "restored_checkpoint_fields", where,
               "post-action learner changed a restored optimizer, counter, cache, evidence, or RNG field")
        excluded_core = {"slot_groups"}
        excluded_core.update(name for name in prefix_checkpoint["core_state"]
                             if _LOCAL_KEY.match(str(name)) or _SHARED_KEY.match(str(name)))
        static_keys = set(prefix_checkpoint["core_state"]) - excluded_core
        static_match = all(_close(prefix_checkpoint["core_state"][key], post_checkpoint["core_state"].get(key),
                                  atol=0.0, rtol=0.0) for key in static_keys)
        _check(audit, static_match, "restored_core_fields", where,
               "post-action learner changed a core field outside the actor matrices and slot assignments")
        return actual_local, actual_shared, actual_groups
    except Exception as exc:
        audit.fail("action_audit_error", where, "action, codec, simulator, or restore audit could not complete",
                   action=action, error=repr(exc))
        return None


def _check_allocation_probabilities(
    audit: Audit,
    *,
    branch_dir: Path,
    branch_rows: Sequence[Mapping[str, str]],
    branch_summary: Mapping[str, Any],
    post_checkpoint: Mapping[str, Any],
    post_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    config: Mapping[str, Any],
    where: str,
) -> None:
    if post_arrays is None:
        return
    try:
        base_config = loop.load_base_config()
        agent = runner._restore_learner(post_checkpoint, base_config, config)
        schedules = {int(record["bout"]): record for record in branch_summary.get("per_bout", [])}
        first_rows: dict[int, Mapping[str, str]] = {}
        for row in branch_rows:
            bout = int(row["bout"])
            first_rows.setdefault(bout, row)
        parent_group = int(post_checkpoint["parent_group"])
        shared = post_arrays[1][0].reshape(3, 3)
        zero = np.zeros((3, 3), dtype=np.float32)
        for bout, record in schedules.items():
            if not bool(record.get("allocated", False)):
                continue
            row = first_rows.get(bout)
            if row is None:
                audit.fail("allocation_first_row_missing", where,
                           "newly allocated future task has no first step row for inheritance verification", bout=bout)
                continue
            inherited = bool(record.get("prior_inherited", False))
            group = int(record["group"])
            expected_inherit = parent_group == -2 or parent_group == group
            _check(audit, inherited == expected_inherit, "allocation_parent_match", where,
                   "new allocation inheritance does not match the action and supplied group", bout=bout,
                   group=group, parent_group=parent_group, inherited=inherited, expected=expected_inherit)
            expected_physical = 0 if expected_inherit else -1
            _check(audit, int(record.get("physical_group_at_start", -999)) == expected_physical,
                   "allocation_physical_group", where,
                   "new allocation has the wrong physical parent binding", bout=bout,
                   actual=record.get("physical_group_at_start"), expected=expected_physical)
            raw_obs = np.asarray(json.loads(row["obs"]), dtype=np.float32)
            slot = int(record["slot"])
            policy_obs = loop.augment_observation(raw_obs, slot + 1)
            coefficients = shared if expected_inherit else zero
            expected_probs = runner._coefficient_probs(agent.core, np.asarray([policy_obs], dtype=np.float32), coefficients)[0]
            actual_probs = np.asarray(json.loads(row["probabilities"]), dtype=np.float64)
            _check(audit, actual_probs.shape == (4,) and np.allclose(actual_probs, expected_probs, atol=1e-6, rtol=1e-6),
                   "allocation_inheritance_behavior", where,
                   "first action after local reset does not reflect the expected inherited parent", bout=bout,
                   expected=expected_probs.tolist(), actual=actual_probs.tolist(), inherited=expected_inherit)
    except Exception as exc:
        audit.fail("allocation_probability_audit_error", where,
                   "allocation-time inheritance behavior could not be verified", error=repr(exc))


def _audit_case(
    audit: Audit,
    *,
    case_id: str,
    case_record: Mapping[str, Any],
    config: Mapping[str, Any],
    foundation_bytes: int,
    main_summary: Mapping[str, Any],
) -> dict[str, Any]:
    case_dir = audit.cohort / "cases" / case_id
    where = str(case_dir)
    case_errors_before = len(audit.errors)
    error_path = case_dir / "error.json"
    if error_path.is_file():
        try:
            error_record = _json(error_path)
        except Exception as exc:
            error_record = {"error": str(exc)}
        audit.fail("case_error_file", str(error_path), "case contains a recorded failure", error_record=error_record)
    for error in case_record.get("errors", []):
        audit.fail("case_summary_error", where, "cohort summary reports a case failure", error_record=error)
    expected_case_id = (
        f"s{int(case_record.get('seed', -1))}_"
        f"{case_record.get('pair_kind')}_{case_record.get('history_regime')}"
    )
    identity_match = case_id == expected_case_id
    _check(audit, identity_match, "case_identity", where, "case directory name and summary identity disagree")
    decision_path = case_dir / "decision.json"
    prefix_summary_path = case_dir / "prefix" / "summary.json"
    prefix_checkpoint_path = case_dir / "prefix" / "checkpoint.pt"
    prefix_actor_path = case_dir / "prefix" / "actor.bin"
    prefix_sim_path = case_dir / "prefix" / "simulator.json"
    try:
        decision = _json(decision_path)
        prefix_summary = _json(prefix_summary_path)
        prefix_checkpoint = _checkpoint(prefix_checkpoint_path)
        prefix_sim = _json(prefix_sim_path)
    except Exception as exc:
        audit.fail("case_prefix_artifact", where, "decision or common-prefix artifacts are missing or unreadable", error=repr(exc))
        return {"case_id": case_id, "passed": False, "errors": len(audit.errors) - case_errors_before}
    _check(audit, decision.get("format") == "storage_decision_case_v1", "decision_format", where,
           "decision record has the wrong format")
    _check_checkpoint_shape(audit, prefix_checkpoint, config, str(prefix_checkpoint_path))
    try:
        prefix_local, prefix_shared, prefix_groups = _core_arrays(prefix_checkpoint)
    except Exception as exc:
        audit.fail("prefix_actor_arrays", where, "prefix actor arrays could not be decoded", error=repr(exc))
        return {"case_id": case_id, "passed": False, "errors": len(audit.errors) - case_errors_before}
    _check(audit, int(prefix_checkpoint.get("seed", -1)) == int(case_record.get("seed", -2)),
           "prefix_seed", where, "prefix checkpoint seed differs from case identity")
    try:
        prefix_hash = semantic_hash({"learner": prefix_checkpoint, "simulator": prefix_sim})
        _check(audit, prefix_hash == decision.get("prefix_state_hash")
               and prefix_hash == prefix_summary.get("prefix_state_hash"), "prefix_state_hash", where,
               "saved prefix checkpoint and simulator do not match their recorded semantic hash",
               recomputed=prefix_hash, decision=decision.get("prefix_state_hash"), summary=prefix_summary.get("prefix_state_hash"))
    except Exception as exc:
        audit.fail("prefix_state_hash_error", where, "prefix checkpoint and simulator cannot be hashed", error=repr(exc))
        prefix_hash = None
    _check(audit, prefix_summary.get("foundation_sha256") == decision.get("foundation_sha256")
           and decision.get("foundation_sha256") == main_summary.get("source_manifest", {}).get("foundation", {}).get("sha256"),
           "case_foundation_hash", where, "case W0 digest differs from its cohort manifest")
    _check(audit, bool(decision.get("source_w0_unchanged_at_decision", False)), "w0_at_decision", where,
           "W0 was not recorded unchanged at the decision point")

    seed = int(case_record["seed"])
    pair = str(case_record["pair_kind"])
    history = str(case_record["history_regime"])
    mapping = pair_mapping(seed, pair)
    _check(audit, _close(decision.get("mapping"), mapping, atol=0.0, rtol=0.0), "pair_mapping", where,
           "recorded old/new metadata mapping differs from the frozen seed mapping",
           recorded=decision.get("mapping"), expected=mapping)
    prefix_sched = prefix_schedule(mapping, config, history)
    _check(audit, _close(decision.get("bootstrap_schedule"), prefix_sched[:2], atol=0.0, rtol=0.0),
           "bootstrap_schedule", where, "recorded bootstrap schedule differs from the frozen protocol")
    _check(audit, _close(decision.get("history_schedule"), prefix_sched[2:], atol=0.0, rtol=0.0),
           "history_schedule", where, "recorded history schedule differs from the frozen protocol")
    prefix_bouts = prefix_summary.get("per_bout", [])
    _check(audit, len(prefix_bouts) == len(prefix_sched), "prefix_bout_count", where,
           "prefix summary does not contain both bootstrap bouts and all history bouts",
           actual=len(prefix_bouts), expected=len(prefix_sched))
    for actual, expected in zip(prefix_bouts, prefix_sched):
        _check(audit, int(actual.get("bout", -1)) == int(expected["bout"])
               and actual.get("task") == expected["task"] and int(actual.get("group", -1)) == int(expected["group"])
               and int(actual.get("steps", -1)) == int(expected["steps"]),
               "prefix_bout_schedule", where, "prefix bout summary differs from the frozen schedule",
               actual=actual, expected=expected)
    try:
        prefix_rows = _csv(case_dir / "prefix" / "steps.csv")
    except Exception as exc:
        prefix_rows = []
        audit.fail("prefix_steps_csv", where, "prefix raw step CSV cannot be read", error=repr(exc))
    prefix_expected_steps = 2 * int(config["bootstrap_steps"]) + 8 * int(config["bout_steps"])
    prefix_count, prefix_reward, _ = _check_schedule_rows(
        audit, prefix_rows, prefix_sched, phase="prefix", action="", future="",
        global_start=0, seed=seed, where=str(case_dir / "prefix" / "steps.csv"),
    )
    _check(audit, prefix_count == prefix_expected_steps, "prefix_expected_steps", where,
           "prefix interaction count differs from bootstrap and history schedule",
           actual=prefix_count, expected=prefix_expected_steps)
    _check_reward_summary(audit, prefix_rows, prefix_summary, raw_reward=prefix_reward,
                          raw_steps=prefix_count, where=str(case_dir / "prefix" / "summary.json"))
    _check_rollouts(audit, case_dir / "prefix", phase=("bootstrap", "prefix_history"), expected_rows=prefix_count,
                    summary={**prefix_summary, "rollout_steps": int(config["rollout_steps"])},
                    where=str(case_dir / "prefix" / "summary.json"))
    try:
        prefix_rollout_rows = _csv(case_dir / "prefix" / "rollouts.csv")
        prefix_rollout_total = sum(int(row["rows"]) for row in prefix_rollout_rows)
        _check(audit, prefix_rollout_total == prefix_count, "prefix_ppo_coverage", where,
               "prefix PPO ledger does not include every fresh prefix row",
               rollout_rows=prefix_rollout_total, prefix_rows=prefix_count)
    except Exception as exc:
        audit.fail("prefix_rollout_ledger", where, "prefix PPO ledger cannot be audited", error=repr(exc))

    _check(audit, int(prefix_checkpoint["evidence"]["capacity"]) <= 256,
           "prefix_evidence_limit", where, "prefix evidence limit exceeds 256 rows per group")
    evidence_counts = {}
    for group in (0, 1):
        rows = prefix_checkpoint["evidence"]["groups"].get(group, prefix_checkpoint["evidence"]["groups"].get(str(group), {}))
        evidence_counts[str(group)] = len(rows.get("action", []))
        _check(audit, evidence_counts[str(group)] <= 256, "prefix_group_evidence_limit", where,
               "prefix evidence exceeds 256 rows for a group", group=group, count=evidence_counts[str(group)])
    _check(audit, len(prefix_checkpoint.get("recent_groups", [])) == 8,
           "prefix_recent_group_limit", where, "prefix recent group history must contain exactly eight groups",
           groups=prefix_checkpoint.get("recent_groups"))
    _check_selector(audit, case_dir, decision, config, prefix_checkpoint)

    # Rebuild the two-slot LRU from the observer schedule. The old shared parent
    # is formed after the bootstrap, so group 0 binds at that exact boundary.
    empty_cache = {
        "capacity": int(config["local_slots"]), "eviction": "lru", "task_to_slot": {},
        "slot_to_task": [None] * int(config["local_slots"]),
        "last_used": [0] * int(config["local_slots"]),
        "frequency": [0] * int(config["local_slots"]), "clock": 0,
    }
    initial_generation = [0] * int(config["local_slots"])
    initial_physical = [-1] * int(config["local_slots"])
    initial_logical = [-1] * int(config["local_slots"])
    cache_after_boot, generation_after_boot, physical_after_boot, logical_after_boot = _check_cache_and_inheritance(
        audit, prefix_bouts[:2], empty_cache, initial_generation, initial_physical, initial_logical,
        parent_group=-1, config=config, where=f"{case_dir}/prefix/bootstrap",
    )
    physical_after_formation = [0 if value == 0 else -1 for value in logical_after_boot]
    cache_after_history, generation_after_history, physical_after_history, logical_after_history = _check_cache_and_inheritance(
        audit, prefix_bouts[2:], cache_after_boot, generation_after_boot, physical_after_formation,
        logical_after_boot, parent_group=0, config=config, where=f"{case_dir}/prefix/history",
    )
    saved_cache = prefix_checkpoint["cache"]
    cache_fields = ("capacity", "eviction", "task_to_slot", "slot_to_task", "last_used", "frequency", "clock")
    _check(audit, all(_close(cache_after_history.get(key), saved_cache.get(key), atol=0.0, rtol=0.0)
                      for key in cache_fields), "prefix_lru_state", where,
           "reconstructed known-task LRU state differs from the common checkpoint",
           reconstructed={key: cache_after_history.get(key) for key in cache_fields},
           checkpoint={key: saved_cache.get(key) for key in cache_fields})
    _check(audit, generation_after_history == [int(value) for value in prefix_checkpoint["slot_generation"]],
           "prefix_generation_state", where, "reconstructed slot generations differ from the common checkpoint",
           reconstructed=generation_after_history, checkpoint=prefix_checkpoint["slot_generation"])
    _check(audit, logical_after_history == [int(value) for value in prefix_checkpoint["slot_group_tags"]],
           "prefix_logical_tags", where, "reconstructed logical group tags differ from the common checkpoint",
           reconstructed=logical_after_history, checkpoint=prefix_checkpoint["slot_group_tags"])
    _check(audit, physical_after_history == [int(value) for value in prefix_groups],
           "prefix_physical_groups", where, "reconstructed physical parent bindings differ from the common checkpoint",
           reconstructed=physical_after_history, checkpoint=prefix_groups.tolist())

    # The actual parent candidates are observer metadata created from the prefix.
    try:
        logical_tags = np.asarray(prefix_checkpoint["slot_group_tags"], dtype=np.int8)
        new_slots = np.flatnonzero(logical_tags == 1).tolist()
        recomputed_old = prefix_shared[0].reshape(3, 3)
        recomputed_new = common_mean_effective(prefix_local, prefix_shared, prefix_groups, new_slots)
        codec_limit = float(decision.get("prefix_codec_coefficient_error", 0.0)) + 2e-6
        _check(audit, np.max(np.abs(recomputed_old - np.asarray(decision["old_parent"], dtype=np.float32))) <= codec_limit
               and np.max(np.abs(recomputed_new - np.asarray(decision["new_candidate"], dtype=np.float32))) <= codec_limit,
               "candidate_prefix_origin", where,
               "recorded parent candidates differ from values recoverable from the prefix beyond its codec drift",
               codec_limit=codec_limit)
    except Exception as exc:
        audit.fail("candidate_prefix_origin_error", where,
                   "parent candidate values could not be checked against the prefix actor", error=repr(exc))

    _package_check(
        audit, folder=case_dir / "prefix", actor_path=prefix_actor_path,
        checkpoint_path=prefix_checkpoint_path, recorded=prefix_summary.get("capacity"),
        foundation_bytes=foundation_bytes, package_cap=int(config["package_budget_bytes"]),
        actor_cap=int(config["actor_budget_bytes"]), where=str(case_dir / "prefix"),
    )
    try:
        expected_prefix_actor = encode_actor(prefix_local, prefix_shared, prefix_groups, codec="int8",
                                             budget_bytes=int(config["actor_budget_bytes"])).data
        _check(audit, prefix_actor_path.read_bytes() == expected_prefix_actor, "prefix_actor_match", where,
               "prefix int8 actor sidecar differs from checkpoint matrices")
    except Exception as exc:
        audit.fail("prefix_actor_match_error", where, "prefix actor checkpoint correspondence cannot be checked", error=repr(exc))

    expected_branches = {(action, future) for action in EXPECTED_ACTIONS for future in FUTURES}
    expected_branch_ids = {f"{action}__{future}" for action, future in expected_branches}
    branch_dir = case_dir / "branches"
    actual_branch_ids = {path.name for path in branch_dir.iterdir() if path.is_dir()} if branch_dir.is_dir() else set()
    _check(audit, actual_branch_ids == expected_branch_ids, "branch_directory_universe", where,
           "case must contain each action by future branch exactly once", actual=sorted(actual_branch_ids),
           expected=sorted(expected_branch_ids))
    _check(audit, len(case_record.get("branches", [])) == 8, "branch_summary_count", where,
           "case summary must contain all eight action by future branches", actual=len(case_record.get("branches", [])))
    restored_hashes = decision.get("restored_state_hashes", {})
    all_uniforms: dict[str, list[float]] = {}
    branch_step_total = 0
    branch_reward_total = 0.0
    audited_branches = 0
    for action, future_regime in sorted(expected_branches):
        name = f"{action}__{future_regime}"
        branch = branch_dir / name
        branch_where = str(branch)
        branch_summary_path = branch / "summary.json"
        try:
            branch_summary = _json(branch_summary_path)
            post_checkpoint = _checkpoint(branch / str(branch_summary.get("post_action_checkpoint", "post_action/checkpoint.pt")))
        except Exception as exc:
            audit.fail("branch_artifact", branch_where, "branch summary or action-time checkpoint is missing or unreadable",
                       error=repr(exc))
            continue
        audited_branches += 1
        _check_checkpoint_shape(audit, post_checkpoint, config, str(branch_summary.get("post_action_checkpoint")))
        for field, expected_value in (("case_id", case_id), ("seed", seed), ("pair_kind", pair),
                                      ("history_regime", history), ("action", action),
                                      ("future_regime", future_regime)):
            _check(audit, branch_summary.get(field) == expected_value, "branch_identity", branch_where,
                   f"branch summary field {field} differs from its directory", actual=branch_summary.get(field),
                   expected=expected_value)
        _check(audit, bool(branch_summary.get("prefix_bouts_included_in_reward", True)) is False,
               "branch_prefix_reward", branch_where, "branch reward summary must exclude the shared prefix")
        _check(audit, bool(branch_summary.get("foundation_unchanged", False)), "branch_foundation_flag",
               branch_where, "branch did not record that W0 stayed unchanged")
        _check(audit, branch_summary.get("foundation_sha256") == decision.get("foundation_sha256"),
               "branch_foundation_hash", branch_where, "branch W0 digest differs from its case")
        _check(audit, branch_summary.get("restored_state_hash") == restored_hashes.get(name),
               "branch_restore_record", branch_where, "branch and decision restore hashes differ",
               branch=branch_summary.get("restored_state_hash"), decision=restored_hashes.get(name))
        post_capacity = _package_check(
            audit, folder=branch / "post_action", actor_path=branch / "post_action" / "actor.bin",
            checkpoint_path=branch / "post_action" / "checkpoint.pt", recorded=branch_summary.get("capacity_after_action"),
            foundation_bytes=foundation_bytes, package_cap=int(config["package_budget_bytes"]),
            actor_cap=int(config["actor_budget_bytes"]), where=f"{branch_where}/post_action",
        )
        final_capacity = _package_check(
            audit, folder=branch, actor_path=branch / "actor.bin", checkpoint_path=branch / "checkpoint.pt",
            recorded=branch_summary.get("capacity"), foundation_bytes=foundation_bytes,
            package_cap=int(config["package_budget_bytes"]), actor_cap=int(config["actor_budget_bytes"]),
            where=f"{branch_where}/final",
        )
        _check(audit, branch_summary.get("post_action_snapshot_included_in_deployment_capacity") is False,
               "observer_snapshot_capacity", branch_where,
               "post-action audit snapshot must be excluded from deployment package accounting")
        post_arrays = _branch_core_check(
            audit, branch_dir=branch, prefix_checkpoint=prefix_checkpoint, post_checkpoint=post_checkpoint,
            prefix_sim=prefix_sim, decision=decision, branch_summary=branch_summary, action=action,
            config=config,
        )
        try:
            branch_rows = _csv(branch / "steps.csv")
        except Exception as exc:
            branch_rows = []
            audit.fail("branch_steps_csv", branch_where, "branch raw step CSV cannot be read", error=repr(exc))
        future_sched = template_schedule(mapping, future_regime, kind="future", steps=int(config["bout_steps"]),
                                          start_bout=len(prefix_sched))
        _check(audit, _close(branch_summary.get("future_schedule"), future_sched, atol=0.0, rtol=0.0),
               "future_schedule", branch_where, "branch future schedule differs from its frozen continuation")
        _check(audit, len(branch_summary.get("per_bout", [])) == len(future_sched), "future_bout_count",
               branch_where, "future branch summary does not contain all scheduled bouts",
               actual=len(branch_summary.get("per_bout", [])), expected=len(future_sched))
        for actual, expected in zip(branch_summary.get("per_bout", []), future_sched):
            _check(audit, int(actual.get("bout", -1)) == int(expected["bout"])
                   and actual.get("task") == expected["task"] and int(actual.get("group", -1)) == int(expected["group"])
                   and int(actual.get("steps", -1)) == int(expected["steps"]),
                   "future_bout_schedule", branch_where, "future bout summary differs from the frozen schedule",
                   actual=actual, expected=expected)
        branch_expected_steps = sum(int(item["steps"]) for item in future_sched)
        branch_steps, branch_reward, uniforms = _check_schedule_rows(
            audit, branch_rows, future_sched, phase="future", action=action, future=future_regime,
            global_start=prefix_expected_steps, seed=seed, where=str(branch / "steps.csv"),
        )
        branch_step_total += branch_steps
        branch_reward_total += branch_reward
        all_uniforms[name] = uniforms
        _check(audit, branch_steps == branch_expected_steps, "branch_expected_steps", branch_where,
               "branch interaction count differs from the configured future schedule",
               actual=branch_steps, expected=branch_expected_steps)
        _check_reward_summary(audit, branch_rows, branch_summary, raw_reward=branch_reward,
                              raw_steps=branch_steps, where=branch_where)
        _check_rollouts(audit, branch, phase="future", expected_rows=branch_steps,
                        summary={**branch_summary, "rollout_steps": int(config["rollout_steps"])},
                        where=branch_where)
        expected_allocations = _check_cache_and_inheritance(
            audit, branch_summary.get("per_bout", []), prefix_checkpoint["cache"],
            prefix_checkpoint["slot_generation"], post_arrays[2] if post_arrays is not None else prefix_groups,
            prefix_checkpoint["slot_group_tags"],
            parent_group={"keep": 0, "replace": 1, "mean": -2, "none": -1}[action],
            config=config, where=branch_where,
        )
        _check_allocation_probabilities(
            audit, branch_dir=branch, branch_rows=branch_rows, branch_summary=branch_summary,
            post_checkpoint=post_checkpoint, post_arrays=post_arrays, config=config, where=branch_where,
        )
        # The final package must describe the final checkpoint and its own actor.
        if final_capacity is not None:
            try:
                final_checkpoint = _checkpoint(branch / "checkpoint.pt")
                final_actor = (branch / "actor.bin").read_bytes()
                final_local, final_shared, final_groups = _core_arrays(final_checkpoint)
                expected_final_actor = encode_actor(final_local, final_shared, final_groups, codec="int8",
                                                    budget_bytes=int(config["actor_budget_bytes"])).data
                _check(audit, final_actor == expected_final_actor, "final_actor_match", branch_where,
                       "final actor sidecar differs from the final checkpoint matrices")
                final_sim = _json(branch / "simulator.json")
                final_hash = semantic_hash({"learner": final_checkpoint, "simulator": final_sim})
                _check(audit, branch_summary.get("final_state_hash") == final_hash, "final_state_hash",
                       branch_where, "final checkpoint and simulator do not match their recorded hash",
                       recorded=branch_summary.get("final_state_hash"), recomputed=final_hash)
                _check_checkpoint_shape(audit, final_checkpoint, config, str(branch / "checkpoint.pt"))
                if expected_allocations is not None:
                    expected_cache, expected_generation, expected_physical, expected_logical = expected_allocations
                    final_cache = final_checkpoint["cache"]
                    cache_fields = ("capacity", "eviction", "task_to_slot", "slot_to_task", "last_used", "frequency", "clock")
                    _check(audit, all(_close(expected_cache.get(key), final_cache.get(key), atol=0.0, rtol=0.0)
                                      for key in cache_fields), "final_lru_state", branch_where,
                           "reconstructed future LRU state differs from final checkpoint",
                           reconstructed={key: expected_cache.get(key) for key in cache_fields},
                           checkpoint={key: final_cache.get(key) for key in cache_fields})
                    _check(audit, expected_generation == [int(value) for value in final_checkpoint["slot_generation"]],
                           "final_generation_state", branch_where,
                           "reconstructed future slot generations differ from final checkpoint",
                           reconstructed=expected_generation, checkpoint=final_checkpoint["slot_generation"])
                    _check(audit, expected_logical == [int(value) for value in final_checkpoint["slot_group_tags"]],
                           "final_logical_tags", branch_where,
                           "reconstructed future logical group tags differ from final checkpoint",
                           reconstructed=expected_logical, checkpoint=final_checkpoint["slot_group_tags"])
                    _check(audit, expected_physical == [int(value) for value in final_groups],
                           "final_physical_groups", branch_where,
                           "reconstructed future physical parent bindings differ from final checkpoint",
                           reconstructed=expected_physical, checkpoint=final_groups.tolist())
            except Exception as exc:
                audit.fail("final_package_audit", branch_where, "final branch package could not be audited", error=repr(exc))
        if post_capacity is not None:
            _check(audit, int(post_capacity["full_persistent_bytes"]) <= int(config["package_budget_bytes"]),
                   "post_action_capacity", branch_where, "post-action deployment state exceeds the package cap")

    _check(audit, audited_branches == 8, "audited_branch_count", where,
           "not all eight branches produced readable action-time artifacts", audited=audited_branches)
    if len(all_uniforms) == 8:
        reference_key = sorted(all_uniforms)[0]
        reference_uniforms = all_uniforms[reference_key]
        for name, values in all_uniforms.items():
            _check(audit, len(values) == len(reference_uniforms)
                   and all(a == b for a, b in zip(values, reference_uniforms)),
                   "paired_action_uniforms", where,
                   "action-uniform stream differs across the eight sibling branches",
                   reference=reference_key, branch=name, reference_steps=len(reference_uniforms), branch_steps=len(values))
    else:
        audit.fail("paired_action_uniforms_missing", where,
                   "uniform pairing cannot be established because fewer than eight branch ledgers are available",
                   branches=len(all_uniforms))
    return {
        "case_id": case_id,
        "prefix_environment_steps": prefix_count,
        "branch_environment_steps": branch_step_total,
        "environment_steps_total": prefix_count + branch_step_total,
        "prefix_reward_total": prefix_reward,
        "branch_reward_total": branch_reward_total,
        "branch_count_audited": audited_branches,
        "passed": len(audit.errors) == case_errors_before,
        "errors": len(audit.errors) - case_errors_before,
    }


def audit_cohort(cohort: Path) -> dict[str, Any]:
    cohort = cohort.resolve()
    audit = Audit(cohort)
    main_summary_path = cohort / "summary.json"
    if not cohort.is_dir():
        audit.fail("cohort_missing", str(cohort), "cohort directory does not exist")
        return {"format": FORMAT, "path": str(cohort), "passed": False, "errors": audit.errors}
    try:
        summary = _json(main_summary_path)
    except Exception as exc:
        audit.fail("cohort_summary_missing", str(main_summary_path), "cohort summary is missing or unreadable", error=repr(exc))
        return {"format": FORMAT, "path": str(cohort), "passed": False, "errors": audit.errors}
    config = _config_expected(audit, cohort, summary)
    foundation_bytes = _validate_source_snapshot(audit, cohort, summary)
    if config is None or foundation_bytes is None:
        return {"format": FORMAT, "path": str(cohort), "phase": summary.get("phase"),
                "passed": False, "errors": audit.errors, "warnings": audit.warnings}
    phase = str(summary.get("phase"))
    expected_cases = {
        f"s{seed}_{pair}_{history}"
        for seed in config["seeds"]
        for pair in config["pair_kinds"]
        for history in config["history_regimes"]
    }
    case_records = summary.get("cases", [])
    record_ids = [str(row.get("case_id", "")) for row in case_records]
    _check(audit, len(record_ids) == len(set(record_ids)), "duplicate_case_records", "summary.json",
           "cohort summary contains duplicate case records")
    _check(audit, set(record_ids) == expected_cases, "case_universe", "summary.json",
           "cohort case records differ from the frozen config", actual=sorted(record_ids), expected=sorted(expected_cases))
    _check(audit, int(summary.get("case_count", -1)) == len(expected_cases), "case_count", "summary.json",
           "cohort case count differs from the frozen config", actual=summary.get("case_count"), expected=len(expected_cases))
    expected_branch_count = len(expected_cases) * len(EXPECTED_ACTIONS) * len(FUTURES)
    _check(audit, int(summary.get("branch_count", -1)) == expected_branch_count, "cohort_branch_count", "summary.json",
           "cohort branch count differs from the expected action by future grid",
           actual=summary.get("branch_count"), expected=expected_branch_count)
    _check(audit, len(summary.get("errors", [])) == 0, "cohort_run_errors", "summary.json",
           "runner summary contains failures; these are retained in the audit report",
           errors=summary.get("errors", []))
    for error in summary.get("errors", []):
        audit.fail("runner_error", "summary.json", "runner-reported failure", error_record=error)

    expected_steps_prefix = 2 * int(config["bootstrap_steps"]) + 8 * int(config["bout_steps"])
    expected_steps_branch = 12 * int(config["bout_steps"])
    case_costs: list[dict[str, Any]] = []
    record_by_id = {str(row.get("case_id")): row for row in case_records}
    for case_id in sorted(expected_cases):
        record = record_by_id.get(case_id)
        if record is None:
            audit.fail("case_record_missing", case_id, "expected case has no summary record")
            continue
        try:
            cost = _audit_case(audit, case_id=case_id, case_record=record, config=config,
                               foundation_bytes=foundation_bytes, main_summary=summary)
        except Exception as exc:
            audit.fail("case_audit_unhandled", case_id, "case audit encountered an unexpected error", error=repr(exc))
            cost = {"case_id": case_id, "passed": False, "errors": 1,
                    "prefix_environment_steps": 0, "branch_environment_steps": 0,
                    "environment_steps_total": 0}
        case_costs.append(cost)
    observed_prefix = sum(int(row.get("prefix_environment_steps", 0)) for row in case_costs)
    observed_branches = sum(int(row.get("branch_environment_steps", 0)) for row in case_costs)
    observed_total = observed_prefix + observed_branches
    expected_prefix = len(expected_cases) * expected_steps_prefix
    expected_branches = expected_branch_count * expected_steps_branch
    costs = {
        "prefix_environment_steps_counted_once": observed_prefix,
        "branch_environment_steps_all_forks": observed_branches,
        "environment_steps_total": observed_total,
        "expected_prefix_environment_steps": expected_prefix,
        "expected_branch_environment_steps": expected_branches,
        "expected_environment_steps_total": expected_prefix + expected_branches,
    }
    _check(audit, observed_prefix == expected_prefix, "prefix_cost_accounting", "summary.json",
           "persisted prefix interactions do not match expected prefixes counted once",
           observed=observed_prefix, expected=expected_prefix)
    _check(audit, observed_branches == expected_branches, "branch_cost_accounting", "summary.json",
           "persisted branch interactions do not match all expected forks",
           observed=observed_branches, expected=expected_branches)
    _check(audit, observed_total == expected_prefix + expected_branches, "total_cost_accounting", "summary.json",
           "persisted interactions do not match the full prefix plus branch cost",
           observed=observed_total, expected=expected_prefix + expected_branches)
    recorded_costs = summary.get("costs", {})
    for field, actual in (("prefix_environment_steps", observed_prefix),
                          ("branch_environment_steps", observed_branches),
                          ("environment_steps_total", observed_total)):
        _check(audit, int(recorded_costs.get(field, -1)) == actual, "cohort_cost_record", "summary.json",
               f"cohort summary {field} differs from persisted raw CSV rows",
               recorded=recorded_costs.get(field), persisted=actual)
    report = {
        "format": FORMAT,
        "path": str(cohort),
        "phase": phase,
        "passed": not audit.errors,
        "expected_case_count": len(expected_cases),
        "expected_branch_count": expected_branch_count,
        "cases": case_costs,
        "cost": costs,
        "errors": audit.errors,
        "warnings": audit.warnings,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path, help="completed storage-decision cohort directory")
    parser.add_argument("--output", required=True, type=Path, help="path for the audit JSON report")
    args = parser.parse_args(argv)
    cohort = args.cohort.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing audit output: {output}")
    if output == cohort or cohort in output.parents:
        raise SystemExit("audit output must be outside the audited cohort directory")
    result = audit_cohort(cohort)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed" if result.get("passed") else "failed",
                      "output": str(output), "cases": result.get("expected_case_count"),
                      "branches": result.get("expected_branch_count"),
                      "environment_steps": result.get("cost", {}).get("environment_steps_total"),
                      "errors": len(result.get("errors", []))}, sort_keys=True))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

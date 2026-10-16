"""Audit and summarize known/unknown write-stream cohort outputs.

The report treats each seed as one replicate. Episodes and transitions are
aggregated within a seed and are never used as independent statistical units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASELINES = ("reward_shared", "independent7_lru")
MAX_ACTOR_BYTES = 236
MAX_PACKAGE_BYTES = 256 * 1024
PHASES = ("known", "unknown")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} must be an object: {path}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != result:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: object) -> bool:
    return value is True or value == 1 or value == "1" or value == "true"


def _mean(values: Sequence[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(usable)) if usable else None


def _std(values: Sequence[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(pstdev(usable)) if len(usable) > 1 else (0.0 if usable else None)


def _new_check(checks: list[dict[str, Any]], name: str, passed: bool,
               observed: Any = None, expected: Any = None, severity: str = "error") -> None:
    checks.append({
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "severity": severity,
    })


def _audit_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row["name"] for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row["name"] for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {"passed": not failures, "failed_checks": failures, "warnings": warnings, "checks": checks}


def _cohort_candidates(source: Path) -> list[Path]:
    if source.is_file():
        if source.name != "summary.json":
            raise ValueError("--input file must be a cohort summary.json")
        candidates = [source]
    elif (source / "summary.json").is_file():
        candidates = [source / "summary.json"]
    elif source.is_dir():
        candidates = []
        for candidate in sorted(source.rglob("summary.json")):
            try:
                summary = _read_json(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(summary.get("records"), list) and summary.get("phase") in PHASES:
                candidates.append(candidate)
    else:
        raise FileNotFoundError(source)

    result = []
    for path in candidates:
        summary = _read_json(path)
        if not isinstance(summary.get("records"), list) or summary.get("phase") not in PHASES:
            raise ValueError(f"not a write-stream cohort summary: {path}")
        result.append(path)
    if not result:
        raise ValueError(f"no known/unknown stream cohort summaries found under {source}")
    phases = [_read_json(path).get("phase") for path in result]
    if len(phases) != len(set(phases)):
        raise ValueError("input contains more than one cohort for the same phase")
    return result


def _snapshot_audit(run_dir: Path, run_record: Mapping[str, Any], frozen: Mapping[str, Any],
                    checks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = run_dir / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        _new_check(checks, "source_snapshot_manifest_present", False, str(manifest_path), "file exists")
        return {}, {}
    manifest = _read_json(manifest_path)
    _new_check(checks, "summary_manifest_matches_snapshot", manifest == run_record.get("source_manifest"),
               manifest == run_record.get("source_manifest"), True)

    sources = manifest.get("sources", {})
    source_results: dict[str, Any] = {}
    for relative, metadata in sorted(sources.items()):
        rel = PurePosixPath(str(relative))
        path = (snapshot / Path(*rel.parts)).resolve()
        try:
            path.relative_to(snapshot.resolve())
            safe = True
        except ValueError:
            safe = False
        if not safe or not path.is_file():
            _new_check(checks, f"snapshot_source_present:{relative}", False, str(path), "file within snapshot")
            source_results[str(relative)] = {"passed": False, "sha256": None}
            continue
        actual_sha = _sha256(path)
        actual_bytes = path.stat().st_size
        passed = actual_sha == metadata.get("sha256") and actual_bytes == metadata.get("bytes")
        _new_check(checks, f"snapshot_source_hash:{relative}", passed,
                   {"sha256": actual_sha, "bytes": actual_bytes}, metadata)
        source_results[str(relative)] = {"passed": passed, "sha256": actual_sha, "bytes": actual_bytes}

    for key, filename in (("selection", "selection.json"),):
        metadata = manifest.get(key, {})
        path = snapshot / filename
        passed = path.is_file()
        actual_sha = _sha256(path) if passed else None
        actual_bytes = path.stat().st_size if passed else None
        passed = bool(passed and actual_sha == metadata.get("sha256") and actual_bytes == metadata.get("bytes"))
        expected_sha = frozen.get("selection_sha256")
        passed = passed and (expected_sha is None or expected_sha == actual_sha)
        _new_check(checks, f"snapshot_{key}_hash", passed,
                   {"sha256": actual_sha, "bytes": actual_bytes}, metadata)
        source_results[key] = {"passed": passed, "sha256": actual_sha, "bytes": actual_bytes}
        external_selection = Path(str(frozen.get("selection_path", "")))
        external_ok = bool(external_selection.is_file() and _sha256(external_selection) == expected_sha)
        _new_check(checks, "frozen_selection_source_hash", external_ok,
                   str(external_selection) if external_selection.is_file() else None, expected_sha)

    external_results = {}
    for key in ("foundation", "dataset"):
        metadata = manifest.get(key, {})
        path_text = metadata.get("path")
        path = Path(str(path_text)) if path_text else None
        available = bool(path and path.is_file())
        actual_sha = _sha256(path) if available else None
        actual_bytes = path.stat().st_size if available else None
        passed = bool(available and actual_sha == metadata.get("sha256") and actual_bytes == metadata.get("bytes"))
        _new_check(checks, f"external_{key}_hash", passed,
                   {"path": path_text, "available": available, "sha256": actual_sha, "bytes": actual_bytes},
                   metadata)
        external_results[key] = {"passed": passed, "sha256": actual_sha, "bytes": actual_bytes}
    source_results.update(external_results)

    protocol_hash = frozen.get("protocol_sha256")
    copied_protocol = sources.get("docs/WRITE_STREAM_PROTOCOL.md", {}).get("sha256")
    protocol_ok = protocol_hash is None or protocol_hash == copied_protocol
    _new_check(checks, "frozen_protocol_hash_matches_snapshot", protocol_ok,
               copied_protocol, protocol_hash)
    return manifest, source_results


def _audit_config(run_dir: Path, phase: str, seed: int, condition: str,
                  cohort: Mapping[str, Any], frozen: Mapping[str, Any],
                  checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.is_file():
        _new_check(checks, "run_config_present", False, str(path), "file exists")
        return {}
    config = _read_json(path)
    expected_sequence = frozen.get("sequence")
    _new_check(checks, "run_sequence_matches_frozen", config.get("sequence") == expected_sequence,
               config.get("sequence"), expected_sequence)
    for key, expected in (("phase", phase), ("seed", seed), ("condition", condition)):
        _new_check(checks, f"run_config_{key}_matches", config.get(key) == expected,
                   config.get(key), expected)
    _new_check(checks, "run_phase_matches_cohort", phase == cohort.get("phase"), phase, cohort.get("phase"))
    return config


def _audit_capacity(run_dir: Path, config: Mapping[str, Any], manifest: Mapping[str, Any],
                    checks: list[dict[str, Any]]) -> dict[str, Any]:
    capacity_path = run_dir / "capacity.json"
    if not capacity_path.is_file():
        _new_check(checks, "capacity_ledger_present", False, str(capacity_path), "file exists")
        return {"checkpoint_count": 0, "actor_bytes": {}, "checkpoint_bytes": {}, "full_persistent_bytes": {}}
    capacity_rows = json.loads(capacity_path.read_text(encoding="utf-8"))
    if not isinstance(capacity_rows, list):
        raise ValueError(f"capacity.json must contain a list: {capacity_path}")
    foundation_bytes = _as_int(manifest.get("foundation", {}).get("bytes"))
    actor_values, checkpoint_values, persistent_values = [], [], []
    package_limit = _as_int(config.get("package_budget_bytes")) or MAX_PACKAGE_BYTES
    expected_actor_limit = min(MAX_ACTOR_BYTES, _as_int(config.get("actor_budget_bytes")) or MAX_ACTOR_BYTES)
    for index, row in enumerate(capacity_rows):
        actor = _as_int(row.get("actor_bytes"))
        actor_budget = _as_int(row.get("actor_budget_bytes"))
        checkpoint = _as_int(row.get("checkpoint_bytes"))
        foundation = _as_int(row.get("foundation_bytes"))
        persistent = _as_int(row.get("full_persistent_bytes"))
        row_package = _as_int(row.get("package_budget_bytes"))
        if actor is not None:
            actor_values.append(actor)
        if checkpoint is not None:
            checkpoint_values.append(checkpoint)
        if persistent is not None:
            persistent_values.append(persistent)
        _new_check(checks, f"actor_within_236_bytes:capacity_{index}",
                   actor is not None and actor <= MAX_ACTOR_BYTES,
                   actor, MAX_ACTOR_BYTES)
        _new_check(checks, f"actor_within_declared_budget:capacity_{index}",
                   actor is not None and actor_budget is not None and actor <= actor_budget,
                   {"actor_bytes": actor, "actor_budget_bytes": actor_budget}, "actor <= budget")
        _new_check(checks, f"int8_actor_codec:capacity_{index}", config.get("actor_codec") == "int8",
                   config.get("actor_codec"), "int8")
        _new_check(checks, f"foundation_plus_checkpoint:capacity_{index}",
                   foundation is not None and checkpoint is not None and persistent == foundation + checkpoint,
                   {"foundation_bytes": foundation, "checkpoint_bytes": checkpoint,
                    "full_persistent_bytes": persistent}, "foundation + checkpoint")
        _new_check(checks, f"foundation_size_matches_manifest:capacity_{index}",
                   foundation_bytes is not None and foundation == foundation_bytes,
                   foundation, foundation_bytes)
        _new_check(checks, f"package_budget:capacity_{index}",
                   persistent is not None and row_package is not None
                   and persistent <= row_package <= package_limit,
                   {"full_persistent_bytes": persistent, "row_budget": row_package,
                    "configured_budget": package_limit}, "full <= package <= 262144")

    actual_files = sorted(
        [path for path in (run_dir / "checkpoint_final.pt",) if path.is_file()]
        + list((run_dir / "checkpoints").glob("*.pt"))
    )
    actual_sizes = sorted(path.stat().st_size for path in actual_files)
    expected_sizes = sorted(checkpoint_values)
    physical_ok = actual_sizes == expected_sizes and len(actual_files) == len(capacity_rows)
    _new_check(checks, "physical_checkpoint_sizes_match_capacity_ledger", physical_ok,
               {"file_count": len(actual_files), "sizes": actual_sizes},
               {"ledger_count": len(capacity_rows), "sizes": expected_sizes})
    _new_check(checks, "package_cap_is_256_KiB_or_less", package_limit <= MAX_PACKAGE_BYTES,
               package_limit, MAX_PACKAGE_BYTES)
    return {
        "checkpoint_count": len(capacity_rows),
        "physical_checkpoint_count": len(actual_files),
        "actor_bytes": {"min": min(actor_values) if actor_values else None,
                        "max": max(actor_values) if actor_values else None,
                        "values": actor_values},
        "checkpoint_bytes": {"min": min(checkpoint_values) if checkpoint_values else None,
                             "max": max(checkpoint_values) if checkpoint_values else None,
                             "values": checkpoint_values},
        "foundation_bytes": foundation_bytes,
        "full_persistent_bytes": {"min": min(persistent_values) if persistent_values else None,
                                  "max": max(persistent_values) if persistent_values else None,
                                  "values": persistent_values},
        "actor_limit_bytes": MAX_ACTOR_BYTES,
        "package_limit_bytes": package_limit,
        "capacity_rows": capacity_rows,
    }


def _decision_summary(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "decisions.jsonl")
    gate_rows = [row for row in rows if isinstance(row.get("gate"), Mapping)]
    no_holdout_reasons = {
        "partial_cycle_no_fresh_holdout",
        "no_fresh_holdout_local_preserved",
        "no_fresh_holdout",
    }
    partial_no_holdout = [row for row in rows if row.get("reason") in no_holdout_reasons]
    gate_reasons: dict[str, int] = {}
    decision_reasons: dict[str, int] = {}
    accepts = rejects = 0
    for row in rows:
        reason = str(row.get("reason", "unknown"))
        decision_reasons[reason] = decision_reasons.get(reason, 0) + 1
    for row in gate_rows:
        gate = row["gate"]
        reason = str(gate.get("reason", row.get("reason", "unknown")))
        gate_reasons[reason] = gate_reasons.get(reason, 0) + 1
        if _truth(gate.get("accepted")):
            accepts += 1
        else:
            rejects += 1
    return {
        "decision_count": len(rows),
        "gate_full_count": len(gate_rows),
        "gate_accept_count": accepts,
        "gate_reject_count": rejects,
        "gate_reason_counts": dict(sorted(gate_reasons.items())),
        "partial_no_holdout_count": len(partial_no_holdout),
        "non_gate_decision_count": len(rows) - len(gate_rows),
        "decision_reason_counts": dict(sorted(decision_reasons.items())),
    }


def _route_summary(run_dir: Path, phase: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = run_dir / "routing.csv"
    if not path.is_file():
        _new_check(checks, "routing_ledger_present", phase != "unknown", str(path), "unknown requires routing.csv")
        rows: list[dict[str, str]] = []
    else:
        rows = _read_csv(path)
    reasons: dict[str, int] = {}
    evictions_by_task: dict[str, int] = {}
    unknown_by_task: dict[str, int] = {}
    task_rows: dict[str, int] = {}
    unknown_before = unknown_after = evictions = switches = 0
    eviction_events = []
    for row in rows:
        task = str(row.get("taskobserver", "unknown"))
        task_rows[task] = task_rows.get(task, 0) + 1
        reason = str(row.get("reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        before = _as_int(row.get("slot_before"))
        after = _as_int(row.get("active_after"))
        if before is None or before < 0:
            unknown_before += 1
            unknown_by_task[task] = unknown_by_task.get(task, 0) + 1
        if after is None or after < 0:
            unknown_after += 1
        if row.get("evicted_slot") not in (None, ""):
            evictions += 1
            evictions_by_task[task] = evictions_by_task.get(task, 0) + 1
            eviction_events.append({
                "raw_step": _as_int(row.get("raw_step")),
                "taskobserver": task,
                "evicted_slot": _as_int(row.get("evicted_slot")),
                "allocated_slot": _as_int(row.get("allocated_slot")),
                "slot_before": before,
                "active_after": after,
                "reason": reason,
                "switched": _truth(row.get("switched")),
            })
        switches += int(_truth(row.get("switched")))
    return {
        "training_route_rows": len(rows),
        "unknown_before_action_rows": unknown_before,
        "unknown_after_transition_rows": unknown_after,
        "unknown_before_action_fraction": unknown_before / len(rows) if rows else None,
        "unknown_after_transition_fraction": unknown_after / len(rows) if rows else None,
        "eviction_count": evictions,
        "route_switch_count": switches,
        "reason_counts": dict(sorted(reasons.items())),
        "unknown_rows_by_taskobserver": dict(sorted(unknown_by_task.items())),
        "rows_by_taskobserver": dict(sorted(task_rows.items())),
        "evictions_by_taskobserver": dict(sorted(evictions_by_task.items())),
        "eviction_events": eviction_events,
        "error_attribution": "Unknown intervals and eviction events are observable, but without an independent route label they are not false-route error counts.",
    }


def _group_summary(run_dir: Path, run_record: Mapping[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = run_dir / "observer_group_coverage.csv"
    if not path.is_file():
        _new_check(checks, "observer_group_coverage_present", False, str(path), "file exists")
        coverage_rows: list[dict[str, str]] = []
    else:
        coverage_rows = _read_csv(path)
    totals = {
        key: sum(_as_int(row.get(key)) or 0 for row in coverage_rows)
        for key in ("training_steps", "grouped_steps", "unassigned_steps", "group_full_steps", "unassigned_when_full_steps")
    }
    saturation = run_record.get("group_saturation", {})
    for source_key, target_key in (
        ("group_full_steps_total", "group_full_steps"),
        ("unassigned_when_full_steps_total", "unassigned_when_full_steps"),
        ("unassigned_steps_total", "unassigned_steps"),
    ):
        expected = _as_int(saturation.get(source_key))
        _new_check(checks, f"group_coverage_matches_summary:{source_key}", expected == totals[target_key],
                   totals[target_key], expected)
    full_steps = totals["group_full_steps"]
    unassigned_full = totals["unassigned_when_full_steps"]
    return {
        "training_steps": totals["training_steps"],
        "grouped_steps": totals["grouped_steps"],
        "unassigned_steps": totals["unassigned_steps"],
        "group_full_steps": full_steps,
        "unassigned_when_full_steps": unassigned_full,
        "unassigned_when_full_fraction": unassigned_full / full_steps if full_steps else None,
        "unassigned_step_fraction": totals["unassigned_steps"] / max(1, totals["grouped_steps"] + totals["unassigned_steps"]),
        "group_abstentions": _as_int(saturation.get("group_abstentions")) or 0,
        "max_group_abstentions": _as_int(saturation.get("max_groups_abstentions")) or 0,
        "profile_reason_frequencies": saturation.get("profile_reason_frequencies", {}),
        "by_taskobserver": [
            {key: (_as_int(row.get(key)) if key not in ("taskobserver", "resident_slots", "groups_used") else row.get(key))
             for key in ("taskobserver", "training_steps", "grouped_steps", "unassigned_steps",
                         "group_full_steps", "unassigned_when_full_steps", "resident_slots", "groups_used")}
            for row in coverage_rows
        ],
        "interpretation": "Group coverage is descriptive; low unassigned coverage does not establish algorithmic improvement or correct latent routing.",
    }


def _positive_training_outcomes(episode_rows: Sequence[Mapping[str, str]], checks: list[dict[str, Any]]) -> dict[str, Any]:
    training = [row for row in episode_rows if row.get("interaction_kind") == "training" and _truth(row.get("complete"))]
    cross_task = [row for row in training if row.get("episode_start_taskobserver") != row.get("taskobserver")]
    same_task = [row for row in training if row.get("episode_start_taskobserver") == row.get("taskobserver")]
    positive = [row for row in same_task if str(row.get("taskobserver", "")).endswith("+")]
    successes = sum(int(_truth(row.get("goal_reached"))) for row in positive)
    _new_check(checks, "episode_start_taskobserver_available",
               all("episode_start_taskobserver" in row for row in episode_rows),
               bool(episode_rows) and all("episode_start_taskobserver" in row for row in episode_rows), True)
    return {
        "positive_goal_episodes": successes,
        "positive_completed_episodes": len(positive),
        "positive_goal_rate": successes / len(positive) if positive else None,
        "cross_task_training_episodes_excluded": len(cross_task),
        "same_task_completed_training_episodes": len(same_task),
    }


def _evaluation_outcomes(eval_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    final_rows = [row for row in eval_rows if row.get("phase") == "final_clone" and _truth(row.get("complete"))]
    returns = [_as_float(row.get("return_value")) for row in final_rows]
    goals = [_as_int(row.get("goal_reached")) for row in final_rows]
    positive_rows = [row for row in final_rows if str(row.get("taskobserver", "")).endswith("+")]
    positive_goals = sum(int(_truth(row.get("goal_reached"))) for row in positive_rows)
    by_task: dict[str, list[Mapping[str, str]]] = {}
    for row in final_rows:
        by_task.setdefault(str(row.get("taskobserver", "unknown")), []).append(row)
    return {
        "final_clone_episodes": len(final_rows),
        "final_clone_signed_return_mean": _mean(returns),
        "final_clone_goal_rate_all_tasks": _mean(goals),
        "final_clone_positive_goal_episodes": positive_goals,
        "final_clone_positive_episodes": len(positive_rows),
        "final_clone_positive_goal_rate": positive_goals / len(positive_rows) if positive_rows else None,
        "evaluation_steps_from_episode_lengths": sum(_as_int(row.get("length")) or 0 for row in final_rows),
        "by_task": {
            task: {
                "episodes": len(rows),
                "mean_signed_return": _mean([_as_float(row.get("return_value")) for row in rows]),
                "goal_rate": _mean([_as_int(row.get("goal_reached")) for row in rows]),
                "positive_success_rate": (
                    _mean([_as_int(row.get("goal_reached")) for row in rows]) if task.endswith("+") else None
                ),
                "mean_length": _mean([_as_int(row.get("length")) for row in rows]),
            }
            for task, rows in sorted(by_task.items())
        },
    }


def _audit_costs(run_dir: Path, record: Mapping[str, Any], config: Mapping[str, Any],
                 checks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    steps_path = run_dir / "steps.csv"
    episodes_path = run_dir / "episodes.csv"
    evaluations_path = run_dir / "evaluations.csv"
    for name, path in (("steps_ledger_present", steps_path), ("episodes_ledger_present", episodes_path),
                       ("evaluations_ledger_present", evaluations_path)):
        _new_check(checks, name, path.is_file(), str(path), "file exists")
    step_rows, episode_rows, eval_rows = _read_csv(steps_path), _read_csv(episodes_path), _read_csv(evaluations_path)
    counts = record.get("counts", {})
    costs = record.get("costs", {})
    step_counts = {kind: sum(row.get("interaction_kind") == kind for row in step_rows)
                   for kind in ("training", "probe", "evaluation")}
    training_steps, probe_steps, eval_step_rows = step_counts["training"], step_counts["probe"], step_counts["evaluation"]
    raw_steps = training_steps + probe_steps
    count_expectations = (
        ("counts.environment_steps", _as_int(counts.get("environment_steps")), training_steps),
        ("counts.probe_steps", _as_int(counts.get("probe_steps")), probe_steps),
        ("counts.raw_steps", _as_int(counts.get("raw_steps")), raw_steps),
        ("costs.training_steps", _as_int(costs.get("training_steps")), training_steps),
        ("costs.probe_steps", _as_int(costs.get("probe_steps")), probe_steps),
        ("costs.evaluation_steps", _as_int(costs.get("evaluation_steps")), eval_step_rows),
        ("summary.evaluation_steps", _as_int(record.get("evaluation_steps")), eval_step_rows),
        ("costs.total_interactions", _as_int(costs.get("total_interactions")), raw_steps + eval_step_rows),
    )
    for name, expected, observed in count_expectations:
        _new_check(checks, f"interaction_ledger_matches:{name}", expected == observed, observed, expected)

    reward_sum = sum(_as_float(row.get("reward")) or 0.0 for row in step_rows if row.get("interaction_kind") == "training")
    reward_per_step = reward_sum / training_steps if training_steps else None
    stream_reward_rows = [
        row for row in step_rows if row.get("interaction_kind") in ("training", "probe")
    ]
    stream_reward_values = [_as_float(row.get("reward")) for row in stream_reward_rows]
    stream_reward_complete = len(stream_reward_rows) == raw_steps and all(
        value is not None for value in stream_reward_values
    )
    stream_reward_sum = (
        sum(float(value) for value in stream_reward_values if value is not None)
        if stream_reward_complete
        else None
    )
    reward_per_stream_interaction = (
        stream_reward_sum / raw_steps
        if stream_reward_sum is not None and raw_steps
        else None
    )
    summary_reward = _as_float(record.get("training_reward"))
    summary_per_step = _as_float(record.get("reward_per_step"))
    _new_check(checks, "training_reward_matches_steps_ledger",
               summary_reward is not None and math.isclose(reward_sum, summary_reward, rel_tol=1e-7, abs_tol=1e-6),
               reward_sum, summary_reward)
    _new_check(checks, "reward_per_step_matches_steps_ledger",
               summary_per_step is not None and reward_per_step is not None
               and math.isclose(reward_per_step, summary_per_step, rel_tol=1e-7, abs_tol=1e-6),
               reward_per_step, summary_per_step)

    observed_stream = record.get("per_task_observed_stream")
    if isinstance(observed_stream, Mapping):
        observed_rows = [row for row in observed_stream.values() if isinstance(row, Mapping)]
        observed_steps = [_as_int(row.get("steps")) for row in observed_rows]
        observed_rewards = [_as_float(row.get("reward")) for row in observed_rows]
        stream_summary_valid = (
            len(observed_rows) == len(observed_stream)
            and all(value is not None for value in observed_steps)
            and all(value is not None for value in observed_rewards)
        )
        summary_stream_steps = sum(value or 0 for value in observed_steps)
        summary_stream_reward = sum(value or 0.0 for value in observed_rewards)
        _new_check(checks, "observed_stream_steps_match_steps_ledger",
                   stream_summary_valid and summary_stream_steps == raw_steps,
                   summary_stream_steps if stream_summary_valid else None, raw_steps)
        _new_check(checks, "observed_stream_reward_matches_steps_ledger",
                   stream_summary_valid and stream_reward_sum is not None and math.isclose(
                       summary_stream_reward, stream_reward_sum, rel_tol=1e-7, abs_tol=1e-6
                   ),
                   summary_stream_reward if stream_summary_valid else None, stream_reward_sum)

    episode_length_sum = sum(_as_int(row.get("length")) or 0 for row in episode_rows)
    episode_lengths_ok = episode_length_sum == raw_steps
    _new_check(checks, "training_episode_lengths_match_raw_interactions", episode_lengths_ok,
               episode_length_sum, raw_steps)
    eval_lengths = sum(_as_int(row.get("length")) or 0 for row in eval_rows
                       if row.get("phase") == "final_clone" and _truth(row.get("complete")))
    _new_check(checks, "evaluation_episode_lengths_match_steps_ledger", eval_lengths == eval_step_rows,
               eval_lengths, eval_step_rows)
    expected_eval_rows = (_as_int(config.get("evaluation_targets")) or 0) * (_as_int(config.get("evaluation_episodes")) or 0)
    final_eval_rows = sum(row.get("phase") == "final_clone" and _truth(row.get("complete")) for row in eval_rows)
    _new_check(checks, "evaluation_episode_count_matches_config",
               expected_eval_rows == final_eval_rows,
               final_eval_rows, expected_eval_rows)

    training_episodes = _positive_training_outcomes(episode_rows, checks)
    reported_positive_rate = _as_float(record.get("positive_goal_rate"))
    if training_episodes["cross_task_training_episodes_excluded"]:
        summary_rate_check = None
    else:
        derived = training_episodes["positive_goal_rate"]
        summary_rate_check = (derived is not None and reported_positive_rate is not None
                              and math.isclose(derived, reported_positive_rate, rel_tol=1e-7, abs_tol=1e-7))
        _new_check(checks, "positive_goal_rate_matches_same_task_episodes", bool(summary_rate_check),
                   derived, reported_positive_rate)
    training_episodes["runner_reported_positive_goal_rate"] = reported_positive_rate
    training_episodes["runner_rate_comparison"] = (
        "not_compared_cross_task_episodes_excluded" if summary_rate_check is None
        else "matches" if summary_rate_check else "differs"
    )
    evaluations = _evaluation_outcomes(eval_rows)
    return (
        {
            "training_steps": training_steps,
            "probe_steps": probe_steps,
            "evaluation_steps": eval_step_rows,
            "raw_training_and_probe_steps": raw_steps,
            "total_interactions": raw_steps + eval_step_rows,
            "training_reward": reward_sum,
            "reward_per_train_step": reward_per_step,
            "training_and_probe_reward": stream_reward_sum,
            "reward_per_stream_interaction": reward_per_stream_interaction,
            "stream_reward_missing_rows": sum(value is None for value in stream_reward_values),
            "cpu_s": _as_float(costs.get("cpu_s")),
            "wall_s": _as_float(costs.get("wall_s")),
            "decision_cpu_s": _as_float(costs.get("decision_cpu_s")),
            "evaluation_cpu_s": _as_float(costs.get("evaluation_cpu_s")),
            "process_rss_bytes": _as_int(costs.get("process_rss_bytes")),
        },
        training_episodes,
        evaluations,
    )


def _audit_run(cohort_dir: Path, phase: str, record: Mapping[str, Any],
               cohort: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    seed, condition = int(record["seed"]), str(record["condition"])
    run_dir = cohort_dir / f"seed_{seed}" / condition
    _new_check(checks, "run_output_directory_present", run_dir.is_dir(), str(run_dir), "directory exists")
    run_config = _audit_config(run_dir, phase, seed, condition, cohort, frozen, checks)
    _new_check(checks, "frozen_phase_matches_cohort", frozen.get("phase") == phase,
               frozen.get("phase"), phase)
    manifest, snapshot_hashes = _snapshot_audit(run_dir, record, frozen, checks)
    run_summary_path = run_dir / "summary.json"
    run_summary_matches = run_summary_path.is_file() and _read_json(run_summary_path) == dict(record)
    _new_check(checks, "run_summary_matches_cohort_record", run_summary_matches,
               run_summary_matches, True)
    expected_foundation_sha = manifest.get("foundation", {}).get("sha256")
    _new_check(checks, "run_config_foundation_hash_matches_snapshot",
               run_config.get("foundation_sha256") == expected_foundation_sha,
               run_config.get("foundation_sha256"), expected_foundation_sha)
    cost, training_episodes, evaluations = _audit_costs(run_dir, record, run_config, checks)
    capacity = _audit_capacity(run_dir, run_config, manifest, checks)
    decision_path = run_dir / "decisions.jsonl"
    _new_check(checks, "decisions_ledger_present", decision_path.is_file(), str(decision_path), "file exists")
    decisions = _decision_summary(run_dir)
    routes = _route_summary(run_dir, phase, checks)
    counts = record.get("counts", {})
    routes["unknown_actor_rows"] = _as_int(counts.get("unknown_actor_rows"))
    routes["unknown_actor_row_fraction"] = (
        routes["unknown_actor_rows"] / cost["training_steps"]
        if routes["unknown_actor_rows"] is not None and cost["training_steps"] else None
    )
    groups = _group_summary(run_dir, record, checks)
    audit = _audit_summary(checks)
    return {
        "phase": phase,
        "seed": seed,
        "condition": condition,
        "domain": record.get("domain"),
        "metrics": {
            "reward_per_train_step": cost["reward_per_train_step"],
            "training_reward": cost["training_reward"],
            "reward_per_stream_interaction": cost["reward_per_stream_interaction"],
            "training_and_probe_reward": cost["training_and_probe_reward"],
            "positive_training_goal_rate": training_episodes["positive_goal_rate"],
            "positive_training_goal_episodes": training_episodes["positive_goal_episodes"],
            "positive_training_episodes": training_episodes["positive_completed_episodes"],
            "cross_task_training_episodes_excluded": training_episodes["cross_task_training_episodes_excluded"],
            "final_clone_mean_signed_return": evaluations["final_clone_signed_return_mean"],
            "final_clone_positive_goal_rate": evaluations["final_clone_positive_goal_rate"],
            "final_clone_goal_rate_all_tasks": evaluations["final_clone_goal_rate_all_tasks"],
        },
        "training_outcomes": training_episodes,
        "final_clone_evaluation": evaluations,
        "costs": cost,
        "gates": decisions,
        "routing": routes,
        "group_coverage": groups,
        "capacity": capacity,
        "snapshot_hashes": snapshot_hashes,
        "audit": audit,
    }


def _summarize_conditions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    conditions = sorted({str(row["condition"]) for row in rows})
    metrics = (
        "reward_per_train_step",
        "reward_per_stream_interaction",
        "positive_training_goal_rate",
        "final_clone_mean_signed_return",
        "final_clone_positive_goal_rate",
    )
    for condition in conditions:
        selected = sorted((row for row in rows if row["condition"] == condition), key=lambda row: int(row["seed"]))
        result[condition] = {
            "seed_count": len(selected),
            "seeds": [int(row["seed"]) for row in selected],
            "metrics": {
                metric: {
                    "per_seed": [{"seed": int(row["seed"]), "value": row["metrics"].get(metric)} for row in selected],
                    "mean": _mean([row["metrics"].get(metric) for row in selected]),
                    "descriptive_population_sd": _std([row["metrics"].get(metric) for row in selected]),
                }
                for metric in metrics
            },
            "costs": {
                key: _mean([row["costs"].get(key) for row in selected])
                for key in ("cpu_s", "wall_s", "decision_cpu_s", "evaluation_cpu_s", "process_rss_bytes")
            },
        }
    return result


def _paired_metric(rows: Sequence[Mapping[str, Any]], candidate: str, baseline: str,
                   metric: str) -> dict[str, Any]:
    candidates = {int(row["seed"]): row["metrics"].get(metric) for row in rows if row["condition"] == candidate}
    controls = {int(row["seed"]): row["metrics"].get(metric) for row in rows if row["condition"] == baseline}
    common = sorted(seed for seed in candidates.keys() & controls.keys()
                    if candidates[seed] is not None and controls[seed] is not None)
    per_seed = [{
        "seed": seed,
        "candidate": candidate,
        "baseline": baseline,
        "candidate_value": float(candidates[seed]),
        "baseline_value": float(controls[seed]),
        "difference": float(candidates[seed]) - float(controls[seed]),
    } for seed in common]
    differences = [row["difference"] for row in per_seed]
    return {
        "common_seeds": common,
        "seed_count": len(common),
        "per_seed": per_seed,
        "mean_difference": _mean(differences),
        "median_difference": float(median(differences)) if differences else None,
        "min_difference": min(differences) if differences else None,
        "max_difference": max(differences) if differences else None,
        "inference": "descriptive paired differences only; no significance test",
    }


def _paired_differences(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    conditions = sorted({str(row["condition"]) for row in rows})
    comparisons: dict[str, Any] = {}
    metrics = (
        "reward_per_train_step",
        "reward_per_stream_interaction",
        "positive_training_goal_rate",
        "final_clone_mean_signed_return",
    )
    for condition in conditions:
        comparisons[condition] = {}
        for baseline in BASELINES:
            comparisons[condition][baseline] = {
                metric: _paired_metric(rows, condition, baseline, metric) for metric in metrics
            }
    return {"comparisons": comparisons, "baseline_conditions": list(BASELINES)}


def _phase_pairing(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(int(row["seed"]), str(row["condition"]), str(row["phase"])): row for row in per_seed}
    keys = sorted({(seed, condition) for seed, condition, _phase in by_key})
    result = []
    for seed, condition in keys:
        known = by_key.get((seed, condition, "known"))
        unknown = by_key.get((seed, condition, "unknown"))
        if known is None or unknown is None:
            continue
        result.append({
            "seed": seed,
            "condition": condition,
            "unknown_minus_known_reward_per_train_step": _difference(
                unknown["metrics"].get("reward_per_train_step"), known["metrics"].get("reward_per_train_step")),
            "unknown_minus_known_reward_per_stream_interaction": _difference(
                unknown["metrics"].get("reward_per_stream_interaction"),
                known["metrics"].get("reward_per_stream_interaction")),
            "unknown_minus_known_final_clone_signed_return": _difference(
                unknown["metrics"].get("final_clone_mean_signed_return"), known["metrics"].get("final_clone_mean_signed_return")),
        })
    return {"per_seed": result, "n": len(result), "inference": "descriptive phase contrast only"}


def _difference(first: Any, second: Any) -> float | None:
    return float(first) - float(second) if first is not None and second is not None else None


def _render_figures(output: Path, per_seed: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    phases = [phase for phase in PHASES if any(row["phase"] == phase for row in per_seed)]
    conditions = sorted({str(row["condition"]) for row in per_seed})
    metrics = (
        ("reward_per_stream_interaction", "Reward per training + probe interaction"),
        ("reward_per_train_step", "Reward per training step"),
        ("final_clone_mean_signed_return", "Final clone mean signed return"),
    )
    colors = {"known": "#1b9e77", "unknown": "#7570b3"}
    figure, axes = plt.subplots(3, 1, figsize=(max(12, len(conditions) * 1.2), 12), constrained_layout=True)
    x = np.arange(len(conditions), dtype=float)
    phase_offsets = np.linspace(-0.16, 0.16, max(1, len(phases)))
    for ax, (metric, title) in zip(axes, metrics):
        for phase_index, phase in enumerate(phases):
            offset = float(phase_offsets[phase_index])
            means, sds = [], []
            for condition in conditions:
                values = [
                    row["metrics"].get(metric)
                    for row in per_seed
                    if row["phase"] == phase and row["condition"] == condition
                    and row["metrics"].get(metric) is not None
                ]
                means.append(_mean(values))
                sds.append(_std(values))
            plot_x = x + offset
            for index, value in enumerate(means):
                if value is None:
                    continue
                sd = sds[index] or 0.0
                ax.errorbar(plot_x[index], value, yerr=sd, color=colors[phase], marker="o",
                            capsize=3, linestyle="none", label=phase if index == 0 else None)
                individual = [
                    row["metrics"].get(metric)
                    for row in sorted(per_seed, key=lambda item: int(item["seed"]))
                    if row["phase"] == phase and row["condition"] == conditions[index]
                    and row["metrics"].get(metric) is not None
                ]
                for point_index, point in enumerate(individual):
                    jitter = (point_index - (len(individual) - 1) / 2.0) * 0.035
                    ax.scatter(plot_x[index] + jitter, point, color=colors[phase], alpha=0.6, s=22, zorder=3)
        ax.set_title(title)
        ax.set_xticks(x, conditions, rotation=40, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if phases:
            ax.legend(title="Stream phase")
    axes[0].set_ylabel("Reward / stream interaction")
    axes[1].set_ylabel("Reward / training step")
    axes[2].set_ylabel("Signed return")
    figure.suptitle("Write stream outcomes by phase and seed (mean ± population SD)")
    png = output / "write_stream_metrics.png"
    pdf = output / "write_stream_metrics.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [{"png": png.name, "pdf": pdf.name}]


def _fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _markdown_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Write Stream cohort analysis",
        "",
        "## 统计口径",
        "",
        "统计单位是 seed。episode 和 transition 只在 seed 内聚合，不作为独立样本。少于 4 个 seed 时只报告描述性结果；不做显著性结论。known 与 unknown phase 分开列出。",
        "",
        "在线 reward / stream interaction 由 steps.csv 中 training 与 probe 行的 reward 总和除以两类交互总数计算；final clone evaluation 不计入分子或分母。reward / train step 仍保留作 training-only 指标。比较 full-use budget baseline 与 shared 条件时优先看前者：总交互数匹配，但 baseline 把交互都用于训练，shared 条件把一部分用于 uniform probes，因此 policy dwell time 和训练交互组成不同。",
        "",
        "正向训练成功率只使用完整、起始和结束 taskobserver 相同、且 task 名以 `+` 结尾的 training episode。跨 task episode 会排除，并单独计数。评估回报按 final clone 的 signed return 汇总。",
        "",
        "步骤成本由唯一的 `steps.csv` 核对，其中已经包含 training、probe 和 evaluation；不读取独立 probe ledger。训练与评估 episode 长度和也分别与交互账本核对。",
        "",
        "## 每 seed 结果",
        "",
        "| phase | seed | condition | reward / training + probe interaction | reward / train step | 正向训练成功率 | 被排除跨 task episodes | final clone signed return | final 正向成功率 | unknown route 比例 | group 满载后未分组比例 | gate 接受 / 拒绝 | CPU s | wall s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(analysis["per_seed"], key=lambda item: (item["phase"], int(item["seed"]), item["condition"])):
        lines.append(
            f"| {row['phase']} | {row['seed']} | {row['condition']} | "
            f"{_fmt(row['metrics']['reward_per_stream_interaction'])} | "
            f"{_fmt(row['metrics']['reward_per_train_step'])} | "
            f"{_fmt(row['metrics']['positive_training_goal_rate'])} | "
            f"{row['training_outcomes']['cross_task_training_episodes_excluded']} | "
            f"{_fmt(row['metrics']['final_clone_mean_signed_return'])} | "
            f"{_fmt(row['metrics']['final_clone_positive_goal_rate'])} | "
            f"{_fmt(row['routing']['unknown_before_action_fraction'])} | "
            f"{_fmt(row['group_coverage']['unassigned_when_full_fraction'])} | "
            f"{row['gates']['gate_accept_count']} / {row['gates']['gate_reject_count']} | "
            f"{_fmt(row['costs']['cpu_s'], 2)} | {_fmt(row['costs']['wall_s'], 2)} |"
        )
    lines.extend(["", "## Paired reward / training + probe interaction differences", ""])
    for phase, phase_data in sorted(analysis["phase_analyses"].items()):
        lines.extend([f"### {phase}", "", "| condition | baseline | common seeds | 每 seed difference | mean difference |", "|---|---|---|---|---:|"])
        paired = phase_data["paired_differences"]["comparisons"]
        for condition in sorted(paired):
            for baseline in BASELINES:
                detail = paired[condition].get(baseline)
                if not detail:
                    continue
                delta = detail["reward_per_stream_interaction"]
                values = ", ".join(f"{item['seed']}: {_fmt(item['difference'])}" for item in delta["per_seed"])
                lines.append(
                    f"| {condition} | {baseline} | {', '.join(map(str, delta['common_seeds'])) or 'none'} | "
                    f"{values or 'n/a'} | {_fmt(delta['mean_difference'])} |"
                )
        lines.append("")
    lines.extend(["## Paired reward / train step differences", ""])
    for phase, phase_data in sorted(analysis["phase_analyses"].items()):
        lines.extend([f"### {phase}", "", "| condition | baseline | common seeds | 每 seed difference | mean difference |", "|---|---|---|---|---:|"])
        paired = phase_data["paired_differences"]["comparisons"]
        for condition in sorted(paired):
            for baseline in BASELINES:
                detail = paired[condition].get(baseline)
                if not detail:
                    continue
                delta = detail["reward_per_train_step"]
                values = ", ".join(f"{item['seed']}: {_fmt(item['difference'])}" for item in delta["per_seed"])
                lines.append(
                    f"| {condition} | {baseline} | {', '.join(map(str, delta['common_seeds'])) or 'none'} | "
                    f"{values or 'n/a'} | {_fmt(delta['mean_difference'])} |"
                )
        lines.append("")
    lines.extend([
        "## Per-seed gate 与成本账本",
        "",
        "| phase | seed | condition | full gate | 接受 | 拒绝 | no-holdout partial | gate reasons | train | probe | eval | total | CPU s | wall s | actor max B | checkpoint max B | W0 + checkpoint max B |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(analysis["per_seed"], key=lambda item: (item["phase"], int(item["seed"]), item["condition"])):
        gate_reasons = ", ".join(f"{key}:{value}" for key, value in row["gates"]["gate_reason_counts"].items()) or "none"
        lines.append(
            f"| {row['phase']} | {row['seed']} | {row['condition']} | {row['gates']['gate_full_count']} | "
            f"{row['gates']['gate_accept_count']} | {row['gates']['gate_reject_count']} | "
            f"{row['gates']['partial_no_holdout_count']} | {gate_reasons} | "
            f"{row['costs']['training_steps']} | {row['costs']['probe_steps']} | {row['costs']['evaluation_steps']} | "
            f"{row['costs']['total_interactions']} | {_fmt(row['costs']['cpu_s'], 2)} | {_fmt(row['costs']['wall_s'], 2)} | "
            f"{_fmt(row['capacity']['actor_bytes']['max'], 0)} | {_fmt(row['capacity']['checkpoint_bytes']['max'], 0)} | "
            f"{_fmt(row['capacity']['full_persistent_bytes']['max'], 0)} |"
        )
    lines.append("")
    lines.extend(["## 路由、分组与 gate 记录", ""])
    lines.append("unknown route 和 eviction 是观测到的路由事件；没有独立 route truth label，因此不能称为 false route/error 计数。组覆盖率是描述统计，低 unknown 或低未分组比例本身不代表算法优化。gate 的 surrogate 接受也不是共享有益的独立真值。")
    lines.append("")
    lines.append("`gate_full_count` 统计有完整 gate 诊断的决定；`partial_no_holdout_count` 统计因没有新 holdout 而跳过 gate 的部分 cycle。两者在 analysis.json 中逐 seed 保留。")
    lines.extend(["", "## 持久化容量与账本审计", ""])
    lines.append("每个 checkpoint 的物理 `.pt` 大小与 capacity ledger 对照；逐行检查 actor payload ≤ 236 B、声明 actor budget，以及 W0 foundation bytes + checkpoint bytes ≤ 256 KiB。CPU time、wall time、决策 CPU、评估 CPU 和 RSS 分开保存。")
    lines.append("")
    audit = analysis["audit"]
    lines.append(f"整体 cohort audit：**{'通过' if audit['passed'] else '未通过'}**；失败检查数：{len(audit['failed_checks'])}。")
    if audit["failed_checks"]:
        lines.append("")
        lines.append("失败项：" + ", ".join(f"`{name}`" for name in audit["failed_checks"]))
    lines.extend(["", "## 图表", "", "![Write stream metrics](write_stream_metrics.png)", ""])
    lines.append("解释限制：本报告不把低 unknown-route 比率当成算法优化，不把 gate surrogate 当成共享有益的真值，也不声称 SOTA 或已证明创新。")
    lines.append("")
    return "\n".join(lines)


def analyze_cohort(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    summaries = _cohort_candidates(source)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"preserve existing report output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []
    cohorts_audit = []
    phase_analyses: dict[str, Any] = {}
    source_signatures: dict[str, str] = {}
    for summary_path in summaries:
        cohort_dir = summary_path.parent
        cohort = _read_json(summary_path)
        phase = str(cohort["phase"])
        frozen_path = cohort_dir / "frozen_config.json"
        checks: list[dict[str, Any]] = []
        if not frozen_path.is_file():
            _new_check(checks, "frozen_config_present", False, str(frozen_path), "file exists")
            frozen: dict[str, Any] = {}
        else:
            frozen = _read_json(frozen_path)
            _new_check(checks, "frozen_conditions_match_summary",
                       list(frozen.get("conditions", [])) == list(cohort.get("conditions", [])),
                       frozen.get("conditions"), cohort.get("conditions"))
            _new_check(checks, "frozen_seeds_match_summary",
                       list(frozen.get("seeds", [])) == list(cohort.get("seeds", [])),
                       frozen.get("seeds"), cohort.get("seeds"))
            expected = cohort.get("sequence")
            _new_check(checks, "frozen_sequence_is_full_stream",
                       isinstance(expected, list) and frozen.get("sequence") == expected,
                       frozen.get("sequence"), expected)
            base_sequence = frozen.get("base_sequence")
            reconstructed = list(base_sequence) + list(reversed(base_sequence)) if isinstance(base_sequence, list) else None
            _new_check(checks, "frozen_sequence_is_base_plus_reverse",
                       reconstructed is not None and frozen.get("sequence") == reconstructed,
                       frozen.get("sequence"), reconstructed)
            _new_check(checks, "summary_base_sequence_matches_frozen",
                       cohort.get("base_sequence") == base_sequence,
                       cohort.get("base_sequence"), base_sequence)
            _new_check(checks, "cohort_protocol_hash_matches_frozen",
                       cohort.get("protocol_sha256") == frozen.get("protocol_sha256"),
                       cohort.get("protocol_sha256"), frozen.get("protocol_sha256"))

        records = cohort["records"]
        keys = [(int(row["seed"]), str(row["condition"])) for row in records]
        _new_check(checks, "cohort_records_unique", len(keys) == len(set(keys)), len(keys), len(set(keys)))
        expected_keys = {(int(seed), str(condition)) for seed in cohort.get("seeds", [])
                         for condition in cohort.get("conditions", [])}
        _new_check(checks, "cohort_record_grid_complete", set(keys) == expected_keys,
                   sorted([list(key) for key in keys]), sorted([list(key) for key in expected_keys]))
        phase_rows: list[dict[str, Any]] = []
        aggregate_cost = {key: 0 for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")}
        for record in sorted(records, key=lambda row: (int(row["seed"]), str(row["condition"]))):
            row = _audit_run(cohort_dir, phase, record, cohort, frozen)
            phase_rows.append(row)
            per_seed.append(row)
            for run_check in row["audit"]["checks"]:
                checks.append({
                    **run_check,
                    "name": f"seed_{row['seed']}/{row['condition']}:{run_check['name']}",
                })
            for key in aggregate_cost:
                aggregate_cost[key] += int(_as_int(row["costs"].get(key)) or 0)
            manifest = record.get("source_manifest", {})
            sig = json.dumps({
                "sources": {name: meta.get("sha256") for name, meta in sorted(manifest.get("sources", {}).items())},
                "selection": manifest.get("selection", {}).get("sha256"),
                "foundation": manifest.get("foundation", {}).get("sha256"),
                "dataset": manifest.get("dataset", {}).get("sha256"),
            }, sort_keys=True)
            signature_key = f"{phase}:{record['seed']}:{record['condition']}"
            source_signatures[signature_key] = sig

        declared_cost = cohort.get("cohort_cost", {})
        for key, observed in aggregate_cost.items():
            expected = _as_int(declared_cost.get(key))
            _new_check(checks, f"cohort_cost_matches_ledgers:{key}", expected == observed,
                       observed, expected)
        audit = _audit_summary(checks)
        cohorts_audit.append({"phase": phase, "path": str(cohort_dir), **audit})
        phase_analyses[phase] = {
            "phase": phase,
            "conditions": sorted({str(row["condition"]) for row in phase_rows}),
            "seeds": sorted({int(row["seed"]) for row in phase_rows}),
            "condition_summary": _summarize_conditions(phase_rows),
            "paired_differences": _paired_differences(phase_rows),
            "cohort_audit": audit,
        }

    # The snapshots should all identify one frozen program/config/dataset lineage.
    by_phase: dict[str, list[str]] = {}
    for key, signature in source_signatures.items():
        by_phase.setdefault(key.split(":", 1)[0], []).append(signature)
    shared_source_checks = []
    for phase, signatures in sorted(by_phase.items()):
        same = len(set(signatures)) <= 1
        shared_source_checks.append({"phase": phase, "passed": same, "snapshot_count": len(signatures)})
    shared_source_pass = all(row["passed"] for row in shared_source_checks)
    global_checks = [
        {"name": f"all_runs_pass:{row['phase']}", "passed": row["passed"],
         "observed": row["failed_checks"], "expected": [], "severity": "error"}
        for row in cohorts_audit
    ]
    global_checks.append({"name": "same_snapshot_lineage_within_phase", "passed": shared_source_pass,
                          "observed": shared_source_checks, "expected": "one frozen source/config hash set per phase",
                          "severity": "error"})
    audit = _audit_summary(global_checks)
    figures = _render_figures(output, per_seed)
    analysis = {
        "schema": "write_stream_analysis_v1",
        "inputs": [str(path.parent) for path in summaries],
        "statistical_unit": "seed",
        "significance_claim": False,
        "seed_count": len({int(row["seed"]) for row in per_seed}),
        "phase_analyses": phase_analyses,
        "per_seed": per_seed,
        "unknown_minus_known_phase_pairs": _phase_pairing(per_seed),
        "cohort_audits": cohorts_audit,
        "snapshot_lineage_checks": shared_source_checks,
        "audit": audit,
        "figures": figures,
        "interpretation_limits": [
            "Seeds are the statistical units; episodes and transitions are aggregated within seed.",
            "With fewer than four seeds, report descriptive paired differences only; no significance claims are made.",
            "Training positive goal rate excludes incomplete and cross-task episodes and uses same-task positive-preference episodes only.",
            "All-stream reward per interaction includes training and uniform probe rewards but excludes evaluation; equal total interaction budgets can still have different policy dwell times because a full-use baseline spends those interactions on training while shared conditions spend some on probes.",
            "A lower unknown-route rate is not, by itself, evidence of an optimization or better decisions.",
            "Unknown-route and eviction records have no independent route truth label and are not false-route error counts.",
            "Group coverage is descriptive and cannot establish that grouping is beneficial or that abstentions are solved.",
            "Gate surrogate outcomes are not independent ground truth for whether shared updates help.",
            "No SOTA or proven-innovation claim is made.",
        ],
    }
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_markdown_report(analysis), encoding="utf-8")
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="one cohort or parent containing known/unknown cohorts")
    parser.add_argument("--output", type=Path, required=True, help="new or empty report directory")
    args = parser.parse_args(argv)
    analysis = analyze_cohort(args.input, args.output)
    print(json.dumps({
        "phases": sorted(analysis["phase_analyses"]),
        "seeds": analysis["seed_count"],
        "runs": len(analysis["per_seed"]),
        "audit_passed": analysis["audit"]["passed"],
        "output": str(Path(args.output).resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Aggregate top-level write-decision and write-stream investigation cohorts.

Only each top-level write_*/summary.json records list contributes cohort
observations. Nested summaries and checkpoint metadata are never treated as
additional experiment runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_METRICS = (
    "online_reward_per_step",
    "online_reward_per_stream_step",
    "positive_goal_rate",
    "final_macro_signed_return",
    "final_positive_goal_rate",
    "gate_accepted",
    "gate_rejected",
    "group_unassigned_fraction",
    "group_unassigned_training_fraction",
    "group_abstention_rate",
    "cpu_s",
    "wall_s",
    "physical_checkpoint_plus_w0_peak_bytes",
    "actual_actor_payload_peak_bytes",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _max_int(values) -> int | None:
    present = [value for item in values if (value := _integer(item)) is not None]
    return max(present) if present else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n_seeds": 0, "mean": None, "min": None, "max": None}
    return {
        "n_seeds": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _relative_path(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace(os.sep, "/")
    except ValueError:
        return None


def _task_items(evaluations: object) -> tuple[str | None, dict[str, Any]]:
    if not isinstance(evaluations, dict):
        return None, {}
    if isinstance(evaluations.get("final"), dict):
        return "final", evaluations["final"]
    task_items = {
        str(name): value
        for name, value in evaluations.items()
        if isinstance(value, dict) and len(str(name)) == 3 and str(name)[-1] in "+-"
    }
    return ("stream_final" if task_items else None), task_items


def _task_metric(entry: object, keys: tuple[str, ...]) -> float | None:
    if not isinstance(entry, dict):
        return None
    for key in keys:
        value = _number(entry.get(key))
        if value is not None:
            return value
    episodes = entry.get("episodes")
    if isinstance(episodes, list):
        candidates = []
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            if "return_value" in keys:
                value = _number(episode.get("return_value"))
            elif any(key in keys for key in ("mean_goal_rate", "positive_success_rate", "goal_rate")):
                value = _number(episode.get("goal_reached"))
            else:
                value = None
            if value is not None:
                candidates.append(value)
        if candidates:
            return _mean(candidates)
    return None


def _final_metrics(record: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
    stage, tasks = _task_items(record.get("evaluations"))
    if not tasks:
        return stage, None, None
    returns: list[float] = []
    positive_goals: list[float] = []
    for task_name, task_result in tasks.items():
        signed_return = _task_metric(
            task_result, ("mean_signed_return", "mean_return", "return_value")
        )
        if signed_return is not None:
            returns.append(signed_return)
        # Only tasks whose hidden reward preference is positive contribute to
        # the positive-goal objective. Success on negative tasks is excluded.
        if task_name.endswith("+"):
            goal_rate = _task_metric(
                task_result, ("positive_success_rate", "mean_goal_rate", "goal_rate")
            )
            if goal_rate is not None:
                positive_goals.append(goal_rate)
    return stage, _mean(returns), _mean(positive_goals)


def _record_costs(record: dict[str, Any]) -> dict[str, float | None]:
    costs = record.get("costs") if isinstance(record.get("costs"), dict) else {}
    counts = record.get("counts") if isinstance(record.get("counts"), dict) else {}
    training = _number(costs.get("training_steps", counts.get("environment_steps")))
    if training is None:
        training = _number(counts.get("raw_steps"))
    probe = _number(costs.get("probe_steps", counts.get("probe_steps"))) or 0.0
    evaluation = _number(costs.get("evaluation_steps", counts.get("evaluation_steps"))) or 0.0
    total = _number(costs.get("total_interactions"))
    if total is None and training is not None:
        total = training + probe + evaluation
    return {
        "training_steps": training,
        "probe_steps": probe,
        "evaluation_steps": evaluation,
        "total_interactions": total,
        "cpu_s": _number(costs.get("cpu_s")),
        "wall_s": _number(costs.get("wall_s")),
    }


def _foundation_hash_sizes(root: Path) -> dict[str, dict[str, Any]]:
    candidates = [root.parent / "data" / "foundation.pt"]
    candidates.extend((root / "write_decision_foundations").glob("seed_*/foundation.pt"))
    result: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[digest] = {"bytes": path.stat().st_size, "path": path}
    return result


def _physical_artifacts(
    run_dir: Path | None,
    record: dict[str, Any],
    root: Path,
    foundation_sizes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    capacity_data: list[dict[str, Any]] = []
    capacity_path = run_dir / "capacity.json" if run_dir else None
    if capacity_path is not None and capacity_path.is_file():
        try:
            raw = _read_json(capacity_path)
            if isinstance(raw, list):
                capacity_data = [row for row in raw if isinstance(row, dict)]
            elif isinstance(raw, dict):
                capacity_data = [raw]
        except (OSError, json.JSONDecodeError):
            pass

    checkpoint_files: list[Path] = []
    actor_files: list[Path] = []
    if run_dir and run_dir.is_dir():
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            low_name = path.name.lower()
            if path.suffix.lower() == ".pt":
                checkpoint_files.append(path)
            elif path.suffix.lower() == ".bin" and "actor" in low_name:
                actor_files.append(path)

    checkpoint_peak_path = max(checkpoint_files, key=lambda p: p.stat().st_size, default=None)
    actor_peak_path = max(actor_files, key=lambda p: p.stat().st_size, default=None)
    physical_checkpoint = checkpoint_peak_path.stat().st_size if checkpoint_peak_path else None
    actor_payload = actor_peak_path.stat().st_size if actor_peak_path else None
    declared_checkpoint = _max_int(row.get("checkpoint_bytes") for row in capacity_data)
    declared_actor = _max_int(row.get("actor_bytes") for row in capacity_data)
    declared_w0 = _max_int(row.get("foundation_bytes") for row in capacity_data)

    manifest = record.get("source_manifest")
    manifest_foundation = manifest.get("foundation") if isinstance(manifest, dict) else None
    foundation_hash = record.get("foundation_sha256")
    if not foundation_hash and isinstance(manifest_foundation, dict):
        foundation_hash = manifest_foundation.get("sha256")
    measured_w0 = foundation_sizes.get(str(foundation_hash), {}).get("bytes")
    foundation_path = foundation_sizes.get(str(foundation_hash), {}).get("path")
    w0_bytes = measured_w0 if measured_w0 is not None else declared_w0
    checkpoint_for_sum = physical_checkpoint if physical_checkpoint is not None else declared_checkpoint
    checkpoint_plus_w0 = (
        checkpoint_for_sum + w0_bytes
        if checkpoint_for_sum is not None and w0_bytes is not None
        else None
    )
    declared_full_peak = _max_int(row.get("full_persistent_bytes") for row in capacity_data)

    return {
        "run_directory": _relative_path(run_dir, root) if run_dir and run_dir.exists() else None,
        "capacity_path": _relative_path(capacity_path, root) if capacity_path and capacity_path.exists() else None,
        "physical_checkpoint_peak_bytes": physical_checkpoint,
        "checkpoint_peak_size_source": "filesystem" if physical_checkpoint is not None else ("capacity_json" if declared_checkpoint is not None else None),
        "checkpoint_peak_path": (
            str(Path(os.path.relpath(checkpoint_peak_path, root))).replace(os.sep, "/")
            if checkpoint_peak_path
            else None
        ),
        "declared_checkpoint_peak_bytes": declared_checkpoint,
        "foundation_w0_bytes": w0_bytes,
        "foundation_size_source": (
            "measured_sha256_match"
            if measured_w0 is not None
            else ("capacity_json" if declared_w0 is not None else None)
        ),
        "foundation_size_path": (
            str(Path(os.path.relpath(foundation_path, root))).replace(os.sep, "/")
            if foundation_path
            else None
        ),
        "physical_checkpoint_plus_w0_peak_bytes": checkpoint_plus_w0,
        "checkpoint_plus_w0_size_source": (
            "filesystem_files"
            if physical_checkpoint is not None and measured_w0 is not None
            else ("mixed_or_capacity_fallback" if checkpoint_plus_w0 is not None else None)
        ),
        "declared_full_persistent_peak_bytes": declared_full_peak,
        "actual_actor_payload_peak_bytes": actor_payload,
        "actor_payload_peak_path": (
            str(Path(os.path.relpath(actor_peak_path, root))).replace(os.sep, "/")
            if actor_peak_path
            else None
        ),
        "actor_payload_size_source": "filesystem" if actor_payload is not None else None,
        "declared_actor_payload_peak_bytes": declared_actor,
        "checkpoint_files": len(checkpoint_files),
        "actor_payload_files": len(actor_files),
    }


def _run_source_files(run_dir: Path | None, root: Path) -> list[str]:
    if run_dir is None or not run_dir.is_dir():
        return []
    names = (
        "summary.json",
        "steps.csv",
        "episodes.csv",
        "evaluations.csv",
        "decisions.jsonl",
        "probes.csv",
        "routing.csv",
        "profile_assignments.csv",
        "observer_group_coverage.csv",
        "capacity.json",
        "config.json",
        "source_snapshot/manifest.json",
        "source_snapshot/manifest_at_execution.json",
        "source_manifest.json",
    )
    return [
        path
        for name in names
        if (path := _relative_path(run_dir / name, root)) is not None
        and (run_dir / name).is_file()
    ]


def _record_metrics(
    record: dict[str, Any],
    run_dir: Path | None,
    root: Path,
    foundation_sizes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    costs = _record_costs(record)
    final_stage, final_signed, final_positive = _final_metrics(record)
    counts = record.get("counts") if isinstance(record.get("counts"), dict) else {}
    online_reward = _number(record.get("reward_per_step"))
    if online_reward is None:
        total_reward = _number(record.get("training_reward"))
        train_steps = costs["training_steps"]
        online_reward = total_reward / train_steps if total_reward is not None and train_steps else None
    stream_reward, stream_source = None, "unavailable_probe_rewards"
    observed = record.get("per_task_observed_stream")
    if isinstance(observed, dict) and observed:
        stream_source = "unavailable_observed_stream_count_mismatch_or_missing_rewards"
        rows = list(observed.values())
        rewards = [_number(row.get("reward")) for row in rows if isinstance(row, dict)]
        steps = [_number(row.get("steps")) for row in rows if isinstance(row, dict)]
        expected = (costs["training_steps"] or 0) + (costs["probe_steps"] or 0)
        if (len(rewards) == len(rows) and all(value is not None for value in rewards + steps)
                and sum(steps) > 0 and math.isclose(sum(steps), expected, abs_tol=1e-6)):
            stream_reward = sum(rewards) / sum(steps)
            stream_source = "per_task_observed_stream_training_plus_probe"
    elif costs["probe_steps"] == 0:
        stream_reward, stream_source = online_reward, "training_only_no_probes"
    online_positive = _number(record.get("plus_goal_rate"))
    if online_positive is None:
        online_positive = _number(record.get("positive_goal_rate"))

    group_fraction = None
    group_saturation = record.get("group_saturation")
    if isinstance(group_saturation, dict):
        unassigned = _number(group_saturation.get("unassigned_active_slots"))
        active = _number(group_saturation.get("active_slots"))
        if unassigned is not None and active is not None and active > 0:
            group_fraction = unassigned / active
    if group_fraction is None and isinstance(record.get("slot_groups"), list):
        groups = record["slot_groups"]
        if groups:
            group_fraction = sum(_integer(group) == -1 for group in groups) / len(groups)

    retries = _number(counts.get("group_retries"))
    abstentions = _number(counts.get("group_abstentions"))
    if isinstance(group_saturation, dict):
        retries = _number(group_saturation.get("profile_assignments")) or retries
        abstentions = _number(group_saturation.get("group_abstentions")) or abstentions
    abstention_rate = (
        abstentions / retries
        if retries is not None and retries > 0 and abstentions is not None
        else None
    )
    physical = _physical_artifacts(run_dir, record, root, foundation_sizes)
    return {
        "condition": str(record.get("condition", "unknown")),
        "seed": _integer(record.get("seed")),
        "online_reward_per_step": online_reward,
        "online_reward_per_stream_step": stream_reward,
        "online_stream_reward_source": stream_source,
        "positive_goal_rate": online_positive,
        "final_stage": final_stage,
        "final_macro_signed_return": final_signed,
        "final_positive_goal_rate": final_positive,
        "gate_accepted": _number(counts.get("accepted_count")),
        "gate_rejected": _number(counts.get("rejected_count")),
        "group_unassigned_fraction": group_fraction,
        "group_unassigned_training_fraction": (
            _number(group_saturation.get("unassigned_step_fraction"))
            if isinstance(group_saturation, dict) else None
        ),
        "group_abstention_rate": abstention_rate,
        "training_steps": costs["training_steps"],
        "probe_steps": costs["probe_steps"],
        "evaluation_steps": costs["evaluation_steps"],
        "total_interactions": costs["total_interactions"],
        "cpu_s": costs["cpu_s"],
        "wall_s": costs["wall_s"],
        **physical,
        "source_files": _run_source_files(run_dir, root),
    }


def _seed_condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        seed = row.get("seed")
        if seed is not None:
            by_seed.setdefault(str(seed), []).append(row)
    seed_metrics: dict[str, dict[str, float | None]] = {}
    for seed, seed_rows in sorted(by_seed.items(), key=lambda item: int(item[0])):
        seed_metrics[seed] = {}
        for metric in _METRICS:
            values = [float(row[metric]) for row in seed_rows if row.get(metric) is not None]
            seed_metrics[seed][metric] = _mean(values)
    stats: dict[str, dict[str, float | int | None]] = {}
    for metric in _METRICS:
        values = [
            float(metrics[metric])
            for metrics in seed_metrics.values()
            if metrics.get(metric) is not None
        ]
        stats[metric] = _metric_stats(values)
    return {
        "n_seeds": len(seed_metrics),
        "seeds": [int(seed) for seed in seed_metrics],
        "seed_metrics": seed_metrics,
        "metrics": stats,
        "record_count": len(rows),
        "duplicate_records_within_seed": len(rows) - len(seed_metrics),
    }


def _record_run_directory(cohort_dir: Path, record: dict[str, Any], root: Path) -> Path | None:
    explicit = record.get("run_directory") or record.get("path")
    if isinstance(explicit, str):
        path = Path(explicit)
        if not path.is_absolute():
            path = cohort_dir / path
        if path.is_dir() and _relative_path(path, root) is not None:
            return path
    seed = _integer(record.get("seed"))
    condition = str(record.get("condition", ""))
    if seed is None or not condition or Path(condition).name != condition:
        return None
    path = cohort_dir / f"seed_{seed}" / condition
    return path if path.is_dir() else None


def _summarize_cohort(
    cohort_dir: Path,
    summary: dict[str, Any],
    root: Path,
    foundation_sizes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_records = summary.get("records")
    records: list[dict[str, Any]] = [r for r in raw_records if isinstance(r, dict)]
    detailed = []
    for record in records:
        run_dir = _record_run_directory(cohort_dir, record, root)
        detailed.append(_record_metrics(record, run_dir, root, foundation_sizes))
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in detailed:
        by_condition.setdefault(row["condition"], []).append(row)
    condition_rows = {
        condition: _seed_condition_summary(rows)
        for condition, rows in sorted(by_condition.items())
    }

    # Count the top-level cohort ledger once; record sums are only a check.
    declared_cost = summary.get("cohort_cost")
    computed_rows = [_record_costs(record) for record in records]
    computed_cost = {
        key: sum(float(row[key] or 0.0) for row in computed_rows)
        for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")
    }
    if isinstance(declared_cost, dict):
        root_cost = {
            key: _number(declared_cost.get(key))
            for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")
        }
        source = "top_level_cohort_cost"
    else:
        root_cost = computed_cost
        source = "sum_of_top_level_records"
    cost_check = {
        key: {
            "declared_root": root_cost[key],
            "sum_top_level_records": computed_cost[key],
            "matches": (
                None
                if root_cost[key] is None
                else math.isclose(float(root_cost[key]), float(computed_cost[key]), rel_tol=0.0, abs_tol=1e-6)
            ),
        }
        for key in computed_cost
    }
    cpu_sum = sum(float(row["cpu_s"] or 0.0) for row in computed_rows)
    wall_sum = sum(float(row["wall_s"] or 0.0) for row in computed_rows)
    expected = summary.get("seeds")
    conditions = summary.get("conditions")
    expected_records = (
        len(expected) * len(conditions)
        if isinstance(expected, list) and isinstance(conditions, list)
        else None
    )
    incomplete = expected_records is not None and len(records) < expected_records
    completion_status = (
        "pending_incomplete_declared_grid"
        if incomplete
        else str(summary.get("status", "records_present"))
    )
    return {
        "name": cohort_dir.name,
        "status": completion_status,
        "phase": summary.get("phase"),
        "source_summary": _relative_path(cohort_dir / "summary.json", root),
        "record_count": len(records),
        "expected_seeds": expected if isinstance(expected, list) else None,
        "expected_record_count": expected_records,
        "incomplete_by_declared_grid": incomplete,
        "observed_seeds": sorted({row["seed"] for row in detailed if row["seed"] is not None}),
        "conditions": list(condition_rows),
        "by_condition": condition_rows,
        "records": detailed,
        "interaction_cost": {**root_cost, "source": source, "record_sum_check": cost_check},
        "cpu_s_sum_runs": cpu_sum,
        "wall_s_sum_runs": wall_sum,
        "wall_s_note": "Sum of run wall durations; not cohort makespan.",
    }


def _failed_interactions(marker: dict[str, Any]) -> tuple[int | None, str, dict[str, int]]:
    raw = marker.get("raw_interactions")
    categories: dict[str, int] = {}
    if isinstance(raw, dict):
        categories = {
            str(key): parsed
            for key, value in raw.items()
            if (parsed := _integer(value)) is not None
        }
    total = _integer(marker.get("total_interactions"))
    if total is None:
        total = _integer(marker.get("total_interactions_recorded"))
    if total is None and categories:
        total = sum(categories.values())
    lower_bound = bool(marker.get("count_is_lower_bound", False)) or marker.get("recorded_count_exact") is False
    return total, "lower_bound" if lower_bound else "exact", categories


def _collect_inputs(root: Path, output_dir: Path) -> dict[str, Any]:
    foundation_sizes = _foundation_hash_sizes(root)
    cohorts: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    auxiliary: dict[str, Any] = {"foundations": None, "counterfactual": None}
    completed_roots: set[str] = set()

    for directory in sorted(root.glob("write_*")):
        if not directory.is_dir() or directory.resolve() == output_dir.resolve():
            continue
        summary_path = directory / "summary.json"
        failed_path = directory / "FAILED_COHORT.json"
        summary = None
        if summary_path.is_file():
            try:
                summary = _read_json(summary_path)
            except (OSError, json.JSONDecodeError):
                summary = None
        if isinstance(summary, dict):
            raw_records = summary.get("records")
            if isinstance(raw_records, list):
                if raw_records:
                    cohort = _summarize_cohort(
                        directory, summary, root, foundation_sizes
                    )
                    cohorts[directory.name] = cohort
                    completed_roots.add(directory.name)
                    if cohort["incomplete_by_declared_grid"]:
                        pending.append(
                            {
                                "name": directory.name,
                                "status": "pending_incomplete_declared_grid",
                                "observed_records": cohort["record_count"],
                                "expected_records": cohort["expected_record_count"],
                                "source_summary": cohort["source_summary"],
                            }
                        )
                else:
                    pending.append(
                        {
                            "name": directory.name,
                            "status": "pending_empty_records",
                            "source_summary": _relative_path(summary_path, root),
                        }
                    )
            elif directory.name == "write_decision_foundations":
                completed_roots.add(directory.name)
                auxiliary["foundations"] = {
                    "source_summary": _relative_path(summary_path, root),
                    "status": summary.get("status"),
                    "dataset_sha256": summary.get("dataset_sha256"),
                    "bc_updates_total": _integer(summary.get("bc_updates_total")),
                    "evaluation_steps_total": _integer(summary.get("evaluation_steps_total")),
                    "cpu_s_total": _number(summary.get("cpu_s_total")),
                    "wall_s_total": _number(summary.get("wall_s_total")),
                    "per_seed": summary.get("per_seed", []),
                }
            elif directory.name == "write_decision_counterfactual":
                completed_roots.add(directory.name)
                auxiliary["counterfactual"] = {
                    "source_summary": _relative_path(summary_path, root),
                    "evaluation_steps": _integer(summary.get("evaluation_steps")),
                    "cpu_s": _number(summary.get("cpu_s")),
                    "wall_s": _number(summary.get("wall_s")),
                    "accepted_harmful": summary.get("accepted_harmful"),
                    "rejected_beneficial": summary.get("rejected_beneficial"),
                    "results": summary.get("results", []),
                }

        if failed_path.is_file():
            try:
                marker = _read_json(failed_path)
            except (OSError, json.JSONDecodeError):
                marker = {}
            if not isinstance(marker, dict):
                marker = {}
            total, bound, categories = _failed_interactions(marker)
            deduplicated = directory.name in completed_roots
            failures.append(
                {
                    "name": directory.name,
                    "status": marker.get("status"),
                    "reason": marker.get("reason"),
                    "source_marker": _relative_path(failed_path, root),
                    "interaction_count": total,
                    "count_classification": bound,
                    "raw_interactions": categories,
                    "counted_separately": not deduplicated,
                    "deduplicated_by_top_level_summary": deduplicated,
                    "complete_runs": marker.get("complete_runs"),
                    "total_runs": marker.get("total_runs"),
                    "process_cpu_seconds_lower_bound": _number(
                        marker.get("process_cpu_seconds_lower_bound")
                    ),
                }
            )

        has_seed_dirs = any(
            child.is_dir() and child.name.startswith("seed_")
            for child in directory.iterdir()
        )
        if has_seed_dirs and summary is None and not failed_path.is_file():
            pending.append(
                {
                    "name": directory.name,
                    "status": "pending_no_top_level_summary",
                    "source_summary": None,
                }
            )

    return {
        "cohorts": cohorts,
        "failed_markers": failures,
        "pending": pending,
        "auxiliary": auxiliary,
        "completed_root_names": sorted(completed_roots),
    }


def _record_map(cohort: dict[str, Any], condition: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    duplicate_seeds: set[int] = set()
    for row in cohort.get("records", []):
        if row.get("condition") != condition or row.get("seed") is None:
            continue
        seed = int(row["seed"])
        if seed in result:
            duplicate_seeds.add(seed)
        else:
            result[seed] = row
    for seed in duplicate_seeds:
        result[seed] = {}
    return result


def _paired_rows(
    left_name: str,
    right_name: str,
    left_rows: dict[int, dict[str, Any]],
    right_rows: dict[int, dict[str, Any]],
    *,
    comparison_family: str,
    post_selection: bool,
) -> dict[str, Any]:
    common = sorted(
        seed
        for seed in set(left_rows).intersection(right_rows)
        if left_rows[seed] and right_rows[seed]
    )
    metrics = (
        "online_reward_per_step",
        "online_reward_per_stream_step",
        "positive_goal_rate",
        "final_macro_signed_return",
        "final_positive_goal_rate",
    )
    pairs = []
    for seed in common:
        deltas: dict[str, float | None] = {}
        for metric in metrics:
            left = _number(left_rows[seed].get(metric))
            right = _number(right_rows[seed].get(metric))
            deltas[metric] = left - right if left is not None and right is not None else None
        pairs.append(
            {
                "seed": seed,
                "left": left_name,
                "right": right_name,
                "delta_left_minus_right": deltas,
            }
        )
    stats = {
        metric: _metric_stats(
            [
                float(pair["delta_left_minus_right"][metric])
                for pair in pairs
                if pair["delta_left_minus_right"][metric] is not None
            ]
        )
        for metric in metrics
    }
    return {
        "comparison_family": comparison_family,
        "left": left_name,
        "right": right_name,
        "matched_seeds": common,
        "n_pairs": len(common),
        "post_selection": post_selection,
        "selection_eligible": False,
        "paired_deltas": pairs,
        "delta_stats_by_seed": stats,
        "interpretation": "Descriptive paired differences by seed; no significance claim.",
    }


def _build_pairs(root: Path, cohorts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    selection_path = root / "write_decision_selection" / "selection.json"
    selection = _read_json(selection_path) if selection_path.is_file() else {}
    selected_conditions = selection.get("selected_conditions", []) if isinstance(selection, dict) else []
    selected_conditions = [str(c) for c in selected_conditions if isinstance(c, str)]

    formal = cohorts.get("write_decision_formal")
    causal = cohorts.get("write_decision_causal_control")
    if formal:
        for selected in selected_conditions:
            for baseline in ("independent_fast", "reward_shared"):
                if selected in formal.get("conditions", []) and baseline in formal.get("conditions", []):
                    comparisons.append(
                        _paired_rows(
                            selected,
                            baseline,
                            _record_map(formal, selected),
                            _record_map(formal, baseline),
                            comparison_family="step1_formal_selected_vs_baseline",
                            post_selection=True,
                        )
                    )
            if (
                causal
                and selected in formal.get("conditions", [])
                and "trial_rollback" in causal.get("conditions", [])
            ):
                comparisons.append(
                    _paired_rows(
                        selected,
                        "trial_rollback",
                        _record_map(formal, selected),
                        _record_map(causal, "trial_rollback"),
                        comparison_family="step1_post_selection_causal_control",
                        post_selection=True,
                    )
                )

    for cohort_name, cohort in cohorts.items():
        conditions = set(cohort.get("conditions", []))
        stream_cohort = "stream" in cohort_name or any("7_lru" in c for c in conditions)
        if not stream_cohort:
            continue
        for proposed in sorted(conditions - {"independent7_lru", "reward_shared"}):
            for baseline in ("independent7_lru", "reward_shared"):
                if baseline in conditions:
                    comparisons.append(
                        _paired_rows(
                            f"{cohort_name}:{proposed}",
                            f"{cohort_name}:{baseline}",
                            _record_map(cohort, proposed),
                            _record_map(cohort, baseline),
                            comparison_family="stream_within_cohort",
                            post_selection=False,
                        )
                    )

    # These are independent cohorts with identical total online interaction
    # budgets. Known-boundary dwell times and PPO update counts still differ.
    for baseline_name, shared_names in (
        ("write_stream_known_budget_baseline", ("write_stream_known",)),
        ("write_stream_unknown_budget_baseline", ("write_stream_unknown",)),
        ("write_parent_unknown_budget_baseline", ("write_parent_cumulative_unknown", "write_parent_recycle_unknown")),
        ("write_detour_budget_w0_17", ("write_detour_w0_17",)),
        ("write_detour_budget_w0_29", ("write_detour_w0_29",)),
    ):
        baseline_cohort = cohorts.get(baseline_name)
        if not baseline_cohort:
            continue
        for shared_name in shared_names:
            shared_cohort = cohorts.get(shared_name)
            if not shared_cohort:
                continue
            for condition in shared_cohort.get("conditions", []):
                if condition not in {"reward_shared", "trial_gate_relaxed", "projected_gate"}:
                    continue
                pair = _paired_rows(
                    f"{shared_name}:{condition}", f"{baseline_name}:independent7_lru",
                    _record_map(shared_cohort, condition), _record_map(baseline_cohort, "independent7_lru"),
                    comparison_family="stream_total_interaction_budget_full_use_baseline", post_selection=True,
                )
                if pair["n_pairs"]:
                    matched = pair["matched_seeds"]
                    left = _record_map(shared_cohort, condition)
                    right = _record_map(baseline_cohort, "independent7_lru")
                    pair["online_interactions_equal_each_seed"] = all(
                        left[s]["training_steps"] + left[s]["probe_steps"] ==
                        right[s]["training_steps"] + right[s]["probe_steps"] for s in matched
                    )
                    pair["primary_metric"] = "online_reward_per_stream_step"
                    pair["budget_limitations"] = "Total training + probe interactions matched; known per-bout dwell times, computation and PPO updates are not matched."
                    comparisons.append(pair)

    recycle_cohorts = [name for name in cohorts if name.startswith("write_parent_recycle_")]
    recycle_cumulative_pairs = []
    for recycle_name in recycle_cohorts:
        cumulative_name = recycle_name.replace("write_parent_recycle_", "write_parent_cumulative_", 1)
        if cumulative_name in cohorts:
            recycle, cumulative = cohorts[recycle_name], cohorts[cumulative_name]
            common_conditions = sorted(
                set(recycle.get("conditions", [])).intersection(cumulative.get("conditions", []))
            )
            for condition in common_conditions:
                pair = _paired_rows(
                    f"{recycle_name}:{condition}",
                    f"{cumulative_name}:{condition}",
                    _record_map(recycle, condition),
                    _record_map(cumulative, condition),
                    comparison_family="recycle_fixed_vs_cumulative_same_seed_condition",
                    post_selection=False,
                )
                if pair["n_pairs"]:
                    recycle_cumulative_pairs.append(pair)
                    comparisons.append(pair)
    recycle_cumulative = {
        "status": "paired_records_available" if recycle_cumulative_pairs else "not_available",
        "pairs": recycle_cumulative_pairs,
        "note": (
            "Only matched seed and identical condition rows are paired; no favorable effect is asserted."
            if recycle_cumulative_pairs
            else "No completed recycle/cumulative cohorts share both a seed and condition; no comparison is reported."
        ),
    }
    return {
        "selection_source": _relative_path(selection_path, root) if selection_path.is_file() else None,
        "selected_conditions": selected_conditions,
        "comparisons": comparisons,
        "recycle_cumulative": recycle_cumulative,
    }


def _sum_costs(collected: dict[str, Any]) -> dict[str, Any]:
    categories = {"training_steps": 0, "probe_steps": 0, "evaluation_steps": 0}
    cost_roots = []
    cohort_component_total = 0
    cohort_declared_total = 0
    cohort_declared_total_known = True
    completed_cpu = 0.0
    completed_wall = 0.0
    for name, cohort in collected["cohorts"].items():
        root_cost = cohort["interaction_cost"]
        cost_roots.append({"root": name, "source": root_cost["source"]})
        completed_cpu += cohort["cpu_s_sum_runs"]
        completed_wall += cohort["wall_s_sum_runs"]
        cohort_component_total += sum(
            int(_integer(root_cost.get(key)) or 0)
            for key in ("training_steps", "probe_steps", "evaluation_steps")
        )
        declared_total = _integer(root_cost.get("total_interactions"))
        if declared_total is None:
            cohort_declared_total_known = False
        else:
            cohort_declared_total += declared_total
        for key in categories:
            value = _integer(root_cost.get(key))
            if value is not None:
                categories[key] += value
    foundations = collected["auxiliary"].get("foundations")
    if foundations and foundations.get("status") == "completed":
        cost_roots.append({"root": "write_decision_foundations", "source": "evaluation_steps_total"})
        categories["evaluation_steps"] += int(foundations.get("evaluation_steps_total") or 0)
        completed_cpu += float(foundations.get("cpu_s_total") or 0.0)
        completed_wall += float(foundations.get("wall_s_total") or 0.0)
    counterfactual = collected["auxiliary"].get("counterfactual")
    if counterfactual and counterfactual.get("evaluation_steps") is not None:
        cost_roots.append({"root": "write_decision_counterfactual", "source": "evaluation_steps"})
        categories["evaluation_steps"] += int(counterfactual["evaluation_steps"])
        completed_cpu += float(counterfactual.get("cpu_s") or 0.0)
        completed_wall += float(counterfactual.get("wall_s") or 0.0)

    failed_exact = 0
    failed_lower_bound = 0
    failed_cpu_lower_bound = 0.0
    for failure in collected["failed_markers"]:
        if failure.get("counted_separately"):
            failed_cpu_lower_bound += float(failure.get("process_cpu_seconds_lower_bound") or 0.0)
        if not failure.get("counted_separately") or failure.get("interaction_count") is None:
            continue
        if failure["count_classification"] == "lower_bound":
            failed_lower_bound += int(failure["interaction_count"])
        else:
            failed_exact += int(failure["interaction_count"])
    completed_total = sum(categories.values())
    return {
        "summary_roots_counted_once": cost_roots,
        "summary_roots_count": len(cost_roots),
        "summary_training_steps": categories["training_steps"],
        "summary_probe_steps": categories["probe_steps"],
        "summary_evaluation_steps": categories["evaluation_steps"],
        "summary_environment_interactions": completed_total,
        "cohort_component_interactions": cohort_component_total,
        "cohort_declared_total_interactions": cohort_declared_total if cohort_declared_total_known else None,
        "cohort_declared_total_matches_components": (
            cohort_declared_total == cohort_component_total if cohort_declared_total_known else None
        ),
        "summary_cpu_s_sum": completed_cpu,
        "summary_wall_s_sum": completed_wall,
        "failed_process_cpu_s_lower_bound": failed_cpu_lower_bound,
        "foundations_bc_updates_offline": int(foundations.get("bc_updates_total") or 0) if foundations else 0,
        "failed_exact_interactions": failed_exact,
        "failed_lower_bound_interactions": failed_lower_bound,
        "known_exact_including_failed": completed_total + failed_exact,
        "known_at_least_including_lower_bounds": completed_total + failed_exact + failed_lower_bound,
        "lower_bound_note": "Any value containing interrupted lower-bound markers must be read as >=, not an exact all-interaction total.",
    }


def _source_audit(root: Path) -> dict[str, Any] | None:
    path = root / "write_decision_formal" / "PROVENANCE_NOTE.json"
    if not path.is_file():
        return None
    note = _read_json(path)
    if not isinstance(note, dict):
        return None
    return {
        "source_file": _relative_path(path, root),
        "strict_report_cohort_source_consistency_fails": note.get(
            "strict_report_cohort_source_consistency_fails"
        ),
        "algorithm_and_config_unchanged": note.get("algorithm_and_config_unchanged"),
        "all_per_run_snapshot_hashes_remain_valid": note.get(
            "all_per_run_snapshot_hashes_remain_valid"
        ),
        "interpretation": note.get("interpretation"),
        "issue": note.get("issue"),
    }


def _source_links(
    root: Path, collected: dict[str, Any], audit: dict[str, Any] | None
) -> list[dict[str, str]]:
    links = []
    for name, cohort in collected["cohorts"].items():
        links.append({"label": f"{name} summary", "path": cohort["source_summary"]})
        frozen_config = root / name / "frozen_config.json"
        if frozen_config.is_file():
            links.append({"label": f"{name} frozen config", "path": _relative_path(frozen_config, root)})
        seen = set()
        for row in cohort["records"]:
            for path in row.get("source_files", []):
                if path not in seen:
                    links.append({"label": f"{name} raw run data", "path": path})
                    seen.add(path)
            for key in ("checkpoint_peak_path", "actor_payload_peak_path", "foundation_size_path"):
                path = row.get(key)
                if path and path not in seen:
                    links.append({"label": f"{name} measured artifact", "path": path})
                    seen.add(path)
    for failure in collected["failed_markers"]:
        links.append({"label": f"{failure['name']} failed marker", "path": failure["source_marker"]})
    selection_path = root / "write_decision_selection" / "selection.json"
    if selection_path.is_file():
        links.append({"label": "Step 1 selection record", "path": _relative_path(selection_path, root)})
    for key, label in (("foundations", "foundations summary"), ("counterfactual", "counterfactual summary")):
        item = collected["auxiliary"].get(key)
        if item:
            links.append({"label": label, "path": item["source_summary"]})
    if audit:
        links.append({"label": "provenance note", "path": audit["source_file"]})
    return links


def _analysis(root: Path, output_dir: Path) -> dict[str, Any]:
    collected = _collect_inputs(root, output_dir)
    audit = _source_audit(root)
    foundations = collected["auxiliary"].get("foundations")
    admission = {}
    for filename in ("selection.json", "formal_decision.json", "SKIPPED.json"):
        source = root / "write_parent_selection" / filename
        if source.is_file():
            admission[filename] = {"source": _relative_path(source, root), "decision": _read_json(source)}
    limitations = [
        "正式 cohort 的 PROVENANCE_NOTE 记录了 docs-only hash 变异：代码和配置相同，但 strict source audit 不是 clean pass。",
        "cold_prior_binding 未经过 gate；stream 观测到的绑定结果不能解释为已验证安全写入。",
        "当前任务环境是自定义 9×9 navigation 与 detour，不是 SOTA 结果，也不能外推为通用导航能力。",
        "正式结果最多 3 个 seed，只作描述性统计，不作显著性声称；episodes 和 transitions 不作为独立样本。",
        "报告保留失败和负结果；不会自动给出‘创新已证明’结论。",
        "报告只统计 summary/failed marker 明示的实验成本；本报告自身的合成单元测试不计入样本效率。",
        "CPU 时间是各 run CPU 秒求和；wall 时间是各 run wall duration 求和，不代表 cohort 端到端 makespan。",
        "Checkpoint、W0 与 actor 字节优先取物理文件长度；缺少可核对文件时会标记 capacity.json 回退来源，不把声明值伪装成实测。",
        "236 B actor 上限和 256 KiB 完整持久包上限是双重约束；同上限并非物理包字节恰好相等，尚未验证只约束总持久字节时的最优分配。",
        "完整数据利用基线只匹配 training + probe 交互总数，不匹配 known 每段停留时间、PPO 更新数或计算成本；跨 cohort 比较以全流奖励为准。",
        "known 固定流中的 independent7_lru 与 independent7_lfu 训练回报及训练正任务成功率相同，终点缓存诊断仍可不同；不能把二者当作额外独立验证。unknown 中二者可以不同。",
    ]
    return {
        "format": "write_investigation_report_v1",
        "input_root": str(root),
        "cohort_read_rule": "Top-level write_*/summary.json only; only its records list contributes cohort records. Nested summaries/snapshots are not counted.",
        "statistical_unit": "seed",
        "cohorts": collected["cohorts"],
        "failed_markers": collected["failed_markers"],
        "pending": collected["pending"],
        "auxiliary_roots": collected["auxiliary"],
        "interaction_ledger": _sum_costs(collected),
        "pairing": _build_pairs(root, collected["cohorts"]),
        "parent_followup_admission": admission,
        "source_audit": audit,
        "limitations": limitations,
        "source_links": _source_links(root, collected, audit),
    }


def _fmt(value: object, digits: int = 4) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _fmt_range(stats: object, digits: int = 4) -> str:
    if not isinstance(stats, dict) or stats.get("mean") is None:
        return "—"
    return f"{_fmt(stats.get('mean'), digits)} [{_fmt(stats.get('min'), digits)}, {_fmt(stats.get('max'), digits)}]"


def _md_link(path: str, root: Path, output_dir: Path) -> str:
    target = Path(os.path.relpath(root / path, output_dir)).as_posix()
    return f"[{path}]({target})"


def _write_markdown(analysis: dict[str, Any], root: Path, output_dir: Path) -> str:
    cohorts = analysis["cohorts"]
    lines = [
        "# Write Decision / Stream Investigation 汇总",
        "",
        "本报告聚合顶层 write_* 实验目录；只有各目录顶层 summary.json 的 records 列表贡献 cohort 样本。不会递归读取 snapshot 或子 run summary 来增加样本数。统计单位是 seed。",
        "",
        "## Cohort 与 condition 结果",
        "",
        "在线 reward 和正奖励任务 goal rate 均先按 seed 汇总。Train R 仅训练流；Stream R 包含训练与探测奖励，分母为 training + probe，不含评估。缺失探测奖励时全流指标留空。完整交互预算基线以 Stream R 比较；known 的每段停留时间及计算/PPO 更新数仍不匹配。final endpoint signed return 对各 task 等权取 macro mean；positive goal rate 只平均正奖励任务的成功率。方括号为 seed 间 min–max，不是置信区间。",
        "",
        "| Cohort | Condition | Seeds | Train R / step [min, max] | Stream R / interaction [min, max] | Positive goal rate | Final macro signed R | Final positive goal rate | Gate accept / reject | Endpoint unassigned slots | Training unassigned fraction | CPU sum (s) | Wall sum (s) | Peak checkpoint + W0 (B) | Actor payload peak (B) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cohort in sorted(cohorts.items()):
        for condition, summary in cohort["by_condition"].items():
            metrics = summary["metrics"]
            lines.append(
                "| {cohort} | {condition} | {n} | {online} | {stream} | {goal} | {final} | {final_goal} | {accept} / {reject} | {unassigned} | {unassigned_training} | {cpu} | {wall} | {package} | {actor} |".format(
                    cohort=name,
                    condition=condition,
                    n=summary["n_seeds"],
                    online=_fmt_range(metrics["online_reward_per_step"]),
                    stream=_fmt_range(metrics["online_reward_per_stream_step"]),
                    goal=_fmt_range(metrics["positive_goal_rate"]),
                    final=_fmt_range(metrics["final_macro_signed_return"]),
                    final_goal=_fmt_range(metrics["final_positive_goal_rate"]),
                    accept=_fmt_range(metrics["gate_accepted"], 2),
                    reject=_fmt_range(metrics["gate_rejected"], 2),
                    unassigned=_fmt_range(metrics["group_unassigned_fraction"]),
                    unassigned_training=_fmt_range(metrics["group_unassigned_training_fraction"]),
                    cpu=_fmt_range(metrics["cpu_s"], 2),
                    wall=_fmt_range(metrics["wall_s"], 2),
                    package=_fmt_range(metrics["physical_checkpoint_plus_w0_peak_bytes"], 0),
                    actor=_fmt_range(metrics["actual_actor_payload_peak_bytes"], 0),
                )
            )
    if not cohorts:
        lines.append("| 尚无已完成 cohort | — | 0 | — | — | — | — | — | — | — | — | — | — |")

    ledger = analysis["interaction_ledger"]
    lines.extend(
        [
            "",
            "## Interaction 与计算成本账本",
            "",
            "| 项目 | 数量 / 成本 |",
            "|---|---:|",
            f"| 顶层 summary cost roots（每个只计一次，含已记录的 pending partial） | {ledger['summary_roots_count']} |",
            f"| Summary roots training steps | {ledger['summary_training_steps']} |",
            f"| Summary roots probe steps | {ledger['summary_probe_steps']} |",
            f"| Summary roots evaluation steps（含 foundation 与 counterfactual） | {ledger['summary_evaluation_steps']} |",
            f"| Summary roots reported environment interactions | {ledger['summary_environment_interactions']} |",
            f"| Cohort roots declared total vs training + probe + evaluation | {ledger['cohort_declared_total_interactions']} vs {ledger['cohort_component_interactions']}（一致：{ledger['cohort_declared_total_matches_components']}） |",
            f"| Summary roots CPU 秒和 | {_fmt(ledger['summary_cpu_s_sum'], 2)} |",
            f"| Summary roots wall duration 秒和 | {_fmt(ledger['summary_wall_s_sum'], 2)} |",
            f"| Foundation 离线 BC updates（非环境交互） | {ledger['foundations_bc_updates_offline']} |",
            f"| Failed marker exact interactions（不含已由 summary 覆盖者） | {ledger['failed_exact_interactions']} |",
            f"| Failed marker lower-bound interactions（不含已由 summary 覆盖者） | ≥ {ledger['failed_lower_bound_interactions']} |",
            f"| Failed marker process CPU lower bound | ≥ {_fmt(ledger['failed_process_cpu_s_lower_bound'], 2)} s |",
            f"| 已知 exact interactions（完整 roots + exact failed） | {ledger['known_exact_including_failed']} |",
            f"| 含 lower-bound 的总计下界 | ≥ {ledger['known_at_least_including_lower_bounds']} |",
            "",
            "中断 marker 的 lower-bound 始终使用 ≥ 展示，不能解读为全部真实交互的精确总数。CPU 为各 run CPU 秒相加；wall 为各 run wall duration 相加，不是端到端 makespan。Foundation BC 更新是离线训练更新，不计入环境交互。",
            "",
            "## 配对比较",
            "",
            "所有差值按相同 seed 配对并展示 left − right；episode / transition 不作独立样本。formal 选中条件与 baselines/control 的比较发生在 selection 之后，明确标为 post-selection，不能用于 method selection。最多 3 个 seed 时只作描述性对照。",
            "",
            "| Family | Left | Right | Matched seeds | Δ Train R / step [min, max] | Δ Stream R / interaction [min, max] | Δ positive goal rate | Δ final signed R | Δ final positive goal rate | Post-selection |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    comparisons = analysis["pairing"]["comparisons"]
    for pair in comparisons:
        stats = pair["delta_stats_by_seed"]
        lines.append(
            f"| {pair['comparison_family']} | {pair['left']} | {pair['right']} | {', '.join(map(str, pair['matched_seeds'])) or '—'} | {_fmt_range(stats['online_reward_per_step'])} | {_fmt_range(stats['online_reward_per_stream_step'])} | {_fmt_range(stats['positive_goal_rate'])} | {_fmt_range(stats['final_macro_signed_return'])} | {_fmt_range(stats['final_positive_goal_rate'])} | {'是' if pair['post_selection'] else '否'} |"
        )
    if not comparisons:
        lines.append("| 暂无 seed 匹配的 paired records | — | — | — | — | — | — | — | — |")

    recycle = analysis["pairing"]["recycle_cumulative"]
    lines.extend(["", "### Recycle-fixed 与 cumulative evidence", "", recycle["note"], ""])
    if recycle["pairs"]:
        lines.append("有 matched pair 时见 analysis.json 的 pairing.recycle_cumulative.pairs；只按同 seed、同 condition 配对。")
    else:
        lines.append("当前没有可配对的完成 cohort，不报告正向或负向效果。")

    admission = analysis.get("parent_followup_admission", {})
    if admission:
        lines.extend(["", "### 预登记扩展门槛", ""])
        for filename, row in admission.items():
            lines.append(f"- {filename}：{_md_link(row['source'], root, output_dir)}；{json.dumps(row['decision'], ensure_ascii=False)}")
        lines.append("未达到准入门槛而跳过的 formal/unknown 扩展不算运行成功、运行失败或 pending，也不虚构样本成本。")

    lines.extend(["", "## Pending 与失败结果", ""])
    if analysis["pending"]:
        for item in analysis["pending"]:
            lines.append(f"- {item['name']}：{item['status']}。")
    else:
        lines.append("- 当前没有识别到 pending 的顶层 cohort。")
    if analysis["failed_markers"]:
        lines.extend(["", "失败或中断 marker 保留如下：", ""])
        for item in analysis["failed_markers"]:
            count = item["interaction_count"]
            count_text = "—" if count is None else (f"≥ {count}" if item["count_classification"] == "lower_bound" else str(count))
            dedup = "；成本已由同目录 summary 覆盖，未重复累加" if item["deduplicated_by_top_level_summary"] else ""
            lines.append(f"- {item['name']}：{item['status']}，交互计数 {count_text}{dedup}。{item['reason'] or ''}")
    else:
        lines.append("- 当前没有 failed marker。")

    lines.extend(["", "## 固定限制与审计说明", ""])
    lines.extend(f"- {limitation}" for limitation in analysis["limitations"])
    audit = analysis.get("source_audit")
    if audit:
        lines.extend(
            [
                "",
                f"正式 cohort provenance：{_md_link(audit['source_file'], root, output_dir)}。记录的 strict source audit 为 {audit['strict_report_cohort_source_consistency_fails']}；原因是 docs-only hash 变异。报告保留该记录，不修补或重标历史快照。",
            ]
        )

    lines.extend(["", "## 数据源", ""])
    for link in analysis["source_links"]:
        lines.append(f"- {link['label']}：{_md_link(link['path'], root, output_dir)}")
    lines.extend(
        [
            "",
            "图：![Seed-level cohort overview](overview.png)",
            "",
            "详细逐 seed、成本核对、物理文件峰值及 paired 差值见 analysis.json。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_plot(analysis: dict[str, Any], output_dir: Path) -> None:
    labels: list[str] = []
    reward_means: list[float] = []
    reward_lows: list[float] = []
    reward_highs: list[float] = []
    goal_means: list[float] = []
    goal_lows: list[float] = []
    goal_highs: list[float] = []
    for cohort_name, cohort in sorted(analysis["cohorts"].items()):
        for condition, summary in sorted(cohort["by_condition"].items()):
            # A common metric across all bars; absent probe rewards stay absent.
            reward = summary["metrics"]["online_reward_per_stream_step"]
            goal = summary["metrics"]["positive_goal_rate"]
            if reward["mean"] is None and goal["mean"] is None:
                continue
            labels.append(f"{cohort_name}\n{condition}")
            for stats, means, lows, highs in (
                (reward, reward_means, reward_lows, reward_highs),
                (goal, goal_means, goal_lows, goal_highs),
            ):
                mean = float(stats["mean"]) if stats["mean"] is not None else float("nan")
                low = float(stats["min"] if stats["min"] is not None else mean)
                high = float(stats["max"] if stats["max"] is not None else mean)
                means.append(mean)
                lows.append(max(0.0, mean - low))
                highs.append(max(0.0, high - mean))

    width = max(10.0, min(34.0, 0.38 * max(1, len(labels))))
    fig, axes = plt.subplots(2, 1, figsize=(width, 8), sharex=True, constrained_layout=True)
    if labels:
        x = list(range(len(labels)))
        axes[0].bar(x, reward_means, color="#3569a8", edgecolor="#1f3d5c", linewidth=0.5)
        axes[0].errorbar(x, reward_means, yerr=[reward_lows, reward_highs], fmt="none", ecolor="#202020", capsize=2, linewidth=0.7)
        axes[1].bar(x, goal_means, color="#51a879", edgecolor="#2b6045", linewidth=0.5)
        axes[1].errorbar(x, goal_means, yerr=[goal_lows, goal_highs], fmt="none", ecolor="#202020", capsize=2, linewidth=0.7)
        axes[1].set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
        axes[1].set_ylim(0, 1.05)
    else:
        axes[0].text(0.5, 0.5, "No completed cohort records", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No completed cohort records", ha="center", va="center", transform=axes[1].transAxes)
    axes[0].set_ylabel("Reward / (training + probe interaction)")
    axes[1].set_ylabel("Positive goal rate")
    axes[1].set_xlabel("Cohort / condition")
    axes[0].set_title("Seed-level cohort means; whiskers show seed min-max")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.savefig(output_dir / "overview.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "overview.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_report(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Build analysis JSON, Chinese Markdown, and PNG/PDF overview charts."""

    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"input root does not exist: {root_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    analysis = _analysis(root_path, output_path)
    with (output_path / "analysis.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    markdown = _write_markdown(analysis, root_path, output_path)
    (output_path / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    _write_plot(analysis, output_path)
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"), help="Top-level output directory to audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/write_investigation_report"),
        help="Report output directory; it is excluded from cohort discovery.",
    )
    args = parser.parse_args(argv)
    result = build_report(args.root, args.output)
    print(
        f"Wrote {args.output.resolve()} with {len(result['cohorts'])} completed cohorts, "
        f"{len(result['pending'])} pending roots, and {len(result['failed_markers'])} failure markers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit and analyze a completed write-decision cohort without rerunning it."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "independent_fast",
    "dynamics_shared",
    "reward_shared",
    "trial_gate",
    "trial_gate_relaxed",
    "projected_gate",
    "local_promote",
    "trial_rollback",
)
PROPOSED_CONDITIONS = {
    "trial_gate", "trial_gate_relaxed", "projected_gate", "local_promote", "trial_rollback"
}
TASKS = ("A0+", "A0-", "A1+", "A1-", "A2+", "A2-")
POSITIVE_TASKS = {"A0+", "A1+", "A2+"}
CONTRAST_REFERENCES = ("independent_fast", "reward_shared")
OUTCOME_METRICS = (
    "online_reward_per_train_step",
    "positive_training_goal_rate",
    "eval_final_positive_mean_return",
    "eval_final_positive_goal_rate",
    "eval_coldtransfer_after_source_A2plus_mean_return",
    "eval_coldtransfer_after_source_A2plus_goal_rate",
    "eval_final_A2plus_mean_return",
    "eval_final_A2plus_goal_rate",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as source:
        yield from csv.DictReader(source)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    items = [number for value in values if (number := _finite(value)) is not None]
    return float(np.mean(items)) if items else None


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    items = [number for value in values if (number := _finite(value)) is not None]
    if not items:
        return {"n": 0, "mean": None, "min": None, "max": None, "sd": None}
    return {
        "n": len(items),
        "mean": float(np.mean(items)),
        "min": float(np.min(items)),
        "max": float(np.max(items)),
        "sd": float(np.std(items, ddof=1)) if len(items) > 1 else 0.0,
    }


def _check(checks: list[dict[str, Any]], name: str, passed: bool | None, details: str) -> None:
    checks.append({"name": name, "passed": passed, "details": details})


def _close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    a, b = _finite(left), _finite(right)
    return a is not None and b is not None and math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def _evaluation_task(row: Mapping[str, Any]) -> dict[str, Any]:
    episode_rows = row.get("episodes", [])
    if isinstance(episode_rows, list) and episode_rows:
        returns = [float(item["return_value"]) for item in episode_rows]
        goals = [int(item.get("goal_reached", 0)) for item in episode_rows]
        lengths = [int(item.get("length", 0)) for item in episode_rows]
        return {
            "mean_return": _mean(returns),
            "mean_goal_rate": _mean(goals),
            "episodes": len(episode_rows),
            "steps": sum(lengths),
            "episode_returns": returns,
            "episode_goal_reached": goals,
        }
    return {
        "mean_return": _finite(row.get("mean_return")),
        "mean_goal_rate": _finite(row.get("mean_goal_rate")),
        "episodes": int(row.get("episodes", 0)) if isinstance(row.get("episodes"), (int, float)) else 0,
        "steps": None,
        "episode_returns": [],
        "episode_goal_reached": [],
    }


def _stage_metrics(stage: Mapping[str, Any]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    all_returns: list[float] = []
    positive_returns: list[float] = []
    positive_goals: list[int] = []
    for task in TASKS:
        value = stage.get(task)
        if not isinstance(value, Mapping):
            continue
        metrics = _evaluation_task(value)
        tasks[task] = metrics
        all_returns.extend(metrics["episode_returns"])
        if not metrics["episode_returns"] and metrics["mean_return"] is not None:
            all_returns.append(float(metrics["mean_return"]))
        if task in POSITIVE_TASKS:
            positive_returns.extend(metrics["episode_returns"])
            positive_goals.extend(metrics["episode_goal_reached"])
            if not metrics["episode_returns"] and metrics["mean_return"] is not None:
                positive_returns.append(float(metrics["mean_return"]))
            if not metrics["episode_goal_reached"] and metrics["mean_goal_rate"] is not None:
                # For fallback summaries each task has the same configured episode count.
                positive_goals.append(float(metrics["mean_goal_rate"]))
    a2plus = tasks.get("A2+", {})
    return {
        "tasks": tasks,
        "all_task_mean_return": _mean(all_returns),
        "positive_task_mean_return": _mean(positive_returns),
        "positive_task_goal_rate": _mean(positive_goals),
        "A2plus_mean_return": a2plus.get("mean_return"),
        "A2plus_goal_rate": a2plus.get("mean_goal_rate"),
    }


def _source_snapshot_audit(
    folder: Path, summary: Mapping[str, Any], checks: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest_path = folder / "source_snapshot" / "manifest.json"
    if not manifest_path.is_file():
        _check(checks, "source_snapshot_manifest", False, "source_snapshot/manifest.json is missing")
        return {"manifest_found": False, "source_files": [], "foundation": None, "dataset": None}

    manifest = _read_json(manifest_path)
    source_rows: list[dict[str, Any]] = []
    source_checks: list[bool] = []
    for relative, expected in manifest.get("sources", {}).items():
        snapshot_file = folder / "source_snapshot" / Path(relative)
        actual = _sha256(snapshot_file) if snapshot_file.is_file() else None
        passed = actual == expected.get("sha256")
        source_checks.append(passed)
        source_rows.append({
            "path": relative,
            "expected_sha256": expected.get("sha256"),
            "actual_snapshot_sha256": actual,
            "bytes": snapshot_file.stat().st_size if snapshot_file.is_file() else None,
            "passed": passed,
        })
    _check(
        checks,
        "source_snapshot_hashes",
        all(source_checks) if source_checks else False,
        f"{sum(source_checks)} / {len(source_checks)} source copies match manifest hashes",
    )

    def verify_external(name: str) -> dict[str, Any] | None:
        item = manifest.get(name)
        if not isinstance(item, Mapping):
            _check(checks, f"{name}_manifest_hash", False, f"manifest {name} entry is missing")
            return None
        path = Path(str(item.get("path", "")))
        actual = _sha256(path) if path.is_file() else None
        expected = item.get("sha256")
        passed = actual == expected
        _check(
            checks,
            f"{name}_manifest_hash",
            passed,
            f"path exists and SHA256 {'matches' if passed else 'does not match'} the run manifest",
        )
        return {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "bytes": path.stat().st_size if path.is_file() else None,
            "passed": passed,
        }

    foundation = verify_external("foundation")
    dataset = verify_external("dataset")
    summary_foundation = summary.get("foundation_sha256")
    foundation_link_ok = foundation is not None and summary_foundation == foundation.get("expected_sha256")
    _check(
        checks,
        "summary_foundation_hash_link",
        foundation_link_ok,
        "summary foundation_sha256 matches source manifest" if foundation_link_ok else "summary foundation_sha256 differs from source manifest",
    )
    return {
        "manifest_found": True,
        "source_files": source_rows,
        "foundation": foundation,
        "dataset": dataset,
        "manifest": manifest,
    }


def _decision_summary(folder: Path, condition: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = folder / "decisions.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid decisions.jsonl line {line_number}: {exc}") from exc
                if isinstance(value, dict):
                    rows.append(value)
    _check(checks, "decisions_log", path.is_file(), f"{len(rows)} decision rows" if path.is_file() else "decisions.jsonl is missing")

    reasons = Counter(str(row.get("reason", "missing_reason")) for row in rows)
    rejection_reasons = Counter(
        str(row.get("reason", "missing_reason")) for row in rows if not bool(row.get("accepted", False))
    )
    proposal_rows = [row for row in rows if condition in PROPOSED_CONDITIONS]
    gate_rows = [row for row in rows if isinstance(row.get("gate"), Mapping)]
    gate_accepts = sum(bool(row["gate"].get("accepted", row.get("accepted", False))) for row in gate_rows)
    gate_rejects = len(gate_rows) - gate_accepts
    proposal_accepts = sum(bool(row.get("accepted", False)) for row in proposal_rows)

    scopes: dict[str, list[Mapping[str, Any]]] = {
        "current_cold": [],
        "old_aggregate": [],
        "old_slots": [],
    }
    for row in gate_rows:
        gate = row["gate"]
        current = gate.get("current_cold")
        if isinstance(current, Mapping):
            scopes["current_cold"].append(current)
        old = gate.get("old_aggregate")
        if isinstance(old, Mapping):
            scopes["old_aggregate"].append(old)
        slots = gate.get("old_slots")
        if isinstance(slots, list):
            scopes["old_slots"].extend(item for item in slots if isinstance(item, Mapping))

    diagnostic_fields = (
        "ess", "behavior_kl", "mean_kl", "max_kl", "mean_delta", "episode_se", "sample_count", "episode_count"
    )
    diagnostic_stats: dict[str, Any] = {}
    for scope, values in scopes.items():
        diagnostic_stats[scope] = {
            field: _stats(item.get(field) for item in values)
            for field in diagnostic_fields
        }
        supported = [item.get("support_ok") for item in values if "support_ok" in item]
        diagnostic_stats[scope]["support_ok_rate"] = _mean(int(bool(item)) for item in supported)
        diagnostic_stats[scope]["records"] = len(values)

    decision_cpu = [float(value) for row in rows if (value := _finite(row.get("decision_cpu_s"))) is not None]
    projection = [row.get("projection") for row in rows if isinstance(row.get("projection"), Mapping)]
    return {
        "decision_records": len(rows),
        "decision_reason_counts": dict(sorted(reasons.items())),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "proposal_decisions": len(proposal_rows),
        "proposal_accepts": proposal_accepts,
        "proposal_rejects": len(proposal_rows) - proposal_accepts,
        "gate_attempts": len(gate_rows),
        "gate_accepts": gate_accepts,
        "gate_rejects": gate_rejects,
        "gate_reason_counts": dict(
            sorted(Counter(str(row.get("reason", "missing_reason")) for row in gate_rows).items())
        ),
        "gate_diagnostics": diagnostic_stats,
        "decision_cpu_s_from_log": float(sum(decision_cpu)),
        "projection_records": len(projection),
        "projection_satisfied": sum(bool(row.get("satisfied", False)) for row in projection),
        "projection_max_violation": _stats(row.get("max_violation") for row in projection),
    }


def _assignment_summary(folder: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = folder / "profile_assignments.csv"
    rows = _read_csv(path) if path.is_file() else []
    _check(
        checks,
        "profile_assignments_log",
        path.is_file(),
        f"{len(rows)} profile assignment rows" if path.is_file() else "profile_assignments.csv is missing",
    )
    reason_counts = Counter(str(row.get("reason", "missing_reason")) for row in rows)
    task_rows = []
    shadow_assigned = 0
    deployed_assigned = 0
    for row in rows:
        shadow_group = int(row.get("shadow_group", -1))
        deployed_group = int(row.get("group", -1))
        shadow_assigned += int(shadow_group >= 0)
        deployed_assigned += int(deployed_group >= 0)
        task_rows.append({
            "taskobserver": row.get("taskobserver"),
            "slot": int(row.get("slot", -1)),
            "generation": int(row.get("generation", -1)),
            "deployed_group": deployed_group,
            "shadow_group": shadow_group,
            "reason": row.get("reason", "missing_reason"),
            "profile_rows": int(row.get("profile_rows", 0)),
            "profile_bytes": int(row.get("profile_bytes", 0)),
        })
    count = len(rows)
    return {
        "assignment_count": count,
        "shadow_assigned_count": shadow_assigned,
        "shadow_abstention_count": count - shadow_assigned,
        "shadow_assignment_rate": shadow_assigned / count if count else None,
        "deployed_assigned_count": deployed_assigned,
        "deployed_abstention_count": count - deployed_assigned,
        "deployed_assignment_rate": deployed_assigned / count if count else None,
        "reason_counts": dict(sorted(reason_counts.items())),
        "task_assignments": task_rows,
    }


def _capacity_summary(folder: Path, run_config: Mapping[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = folder / "capacity.json"
    rows = _read_json(path) if path.is_file() else []
    if not isinstance(rows, list):
        raise ValueError(f"capacity.json must contain a list: {path}")
    actor_cap = int(run_config.get("actor_budget_bytes", 0))
    package_cap = int(run_config.get("package_budget_bytes", 0))
    audited: list[dict[str, Any]] = []
    file_checks: list[bool] = []
    budget_checks: list[bool] = []
    for row in rows:
        step = int(row["step"])
        stem = f"checkpoint_step_{step:08d}"
        checkpoint_path = folder / "checkpoints" / f"{stem}.pt"
        actor_path = folder / "checkpoints" / f"{stem}.actor.bin"
        checkpoint_actual = checkpoint_path.stat().st_size if checkpoint_path.is_file() else None
        actor_actual = actor_path.stat().st_size if actor_path.is_file() else None
        checkpoint_matches = checkpoint_actual == int(row.get("checkpoint_bytes", -1))
        actor_matches = actor_actual == int(row.get("actor_bytes", -1))
        file_checks.append(checkpoint_matches and actor_matches)
        actor_in_budget = actor_cap > 0 and int(row.get("actor_bytes", 0)) <= actor_cap
        package_bytes = int(row.get("full_persistent_bytes", row.get("checkpoint_bytes", 0)))
        package_in_budget = package_cap > 0 and package_bytes <= package_cap
        budget_checks.append(actor_in_budget and package_in_budget)
        audited.append({
            **row,
            "checkpoint_path": str(checkpoint_path),
            "actor_sidecar_path": str(actor_path),
            "checkpoint_physical_bytes": checkpoint_actual,
            "actor_physical_bytes": actor_actual,
            "physical_bytes_including_actor_sidecar": (
                checkpoint_actual + actor_actual
                if checkpoint_actual is not None and actor_actual is not None else None
            ),
            "checkpoint_size_matches_capacity": checkpoint_matches,
            "actor_size_matches_capacity": actor_matches,
        })
    _check(
        checks,
        "checkpoint_physical_sizes",
        bool(rows) and all(file_checks),
        f"{sum(file_checks)} / {len(rows)} checkpoint and actor sidecar sizes match capacity.json",
    )
    _check(
        checks,
        "declared_actor_and_package_budgets",
        bool(rows) and all(budget_checks),
        f"actor cap {actor_cap} B; full checkpoint cap {package_cap} B; {sum(budget_checks)} / {len(rows)} samples fit",
    )

    def sample_stats(key: str, physical_key: str | None = None) -> dict[str, Any]:
        values = [row.get(physical_key or key) for row in audited]
        stats = _stats(values)
        stats["last"] = _finite(values[-1]) if values else None
        return stats

    return {
        "sample_count": len(audited),
        "actor_budget_bytes": actor_cap,
        "package_budget_bytes": package_cap,
        "actor_bytes": sample_stats("actor_bytes", "actor_physical_bytes"),
        "checkpoint_bytes": sample_stats("checkpoint_bytes", "checkpoint_physical_bytes"),
        "full_persistent_bytes": sample_stats("full_persistent_bytes"),
        "checkpoint_plus_actor_sidecar_bytes": sample_stats("physical_bytes_including_actor_sidecar"),
        "declared_actor_bytes": sample_stats("actor_bytes"),
        "declared_checkpoint_bytes": sample_stats("checkpoint_bytes"),
        "declared_full_persistent_bytes": sample_stats("full_persistent_bytes"),
        "samples": audited,
    }


def _raw_cost_audit(
    folder: Path,
    summary: Mapping[str, Any],
    run_config: Mapping[str, Any],
    evaluations: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    steps_path = folder / "steps.csv"
    probes_path = folder / "probes.csv"
    episodes_path = folder / "episodes.csv"
    eval_path = folder / "evaluations.csv"
    counts = Counter()
    phase_counts = Counter()
    training_reward = 0.0
    if steps_path.is_file():
        for row in _iter_csv(steps_path):
            kind = row.get("interaction_kind", "")
            counts[kind] += 1
            if kind == "training":
                phase_counts[row.get("phase", "unknown")] += 1
                training_reward += float(row.get("reward", 0.0) or 0.0)
    raw_probe_steps = sum(1 for _ in _iter_csv(probes_path)) if probes_path.is_file() else 0
    eval_rows = _read_csv(eval_path) if eval_path.is_file() else []
    episodes_rows = _read_csv(episodes_path) if episodes_path.is_file() else []
    raw_eval_step_sum = sum(int(row.get("length", 0) or 0) for row in eval_rows)
    costs = summary.get("costs", {})
    if not isinstance(costs, Mapping):
        costs = {}
    raw_training_steps = counts["training"]
    raw_evaluation_steps = counts["evaluation"]
    raw_total = raw_training_steps + raw_probe_steps + raw_evaluation_steps
    training_steps = int(costs.get("training_steps", -1))
    probe_steps = int(costs.get("probe_steps", -1))
    evaluation_steps = int(costs.get("evaluation_steps", -1))
    total_interactions = int(costs.get("total_interactions", -1))
    summary_counts = summary.get("counts", {})
    if not isinstance(summary_counts, Mapping):
        summary_counts = {}

    for name, passed, details in (
        ("raw_training_steps", raw_training_steps == training_steps, f"CSV {raw_training_steps}; summary {training_steps}"),
        ("raw_probe_steps", raw_probe_steps == probe_steps, f"CSV {raw_probe_steps}; summary {probe_steps}"),
        ("raw_evaluation_steps", raw_evaluation_steps == evaluation_steps, f"CSV {raw_evaluation_steps}; summary {evaluation_steps}"),
        ("raw_total_interactions", raw_total == total_interactions, f"CSV {raw_total}; summary {total_interactions}"),
        ("evaluation_episode_lengths", raw_eval_step_sum == evaluation_steps, f"evaluation CSV episode lengths sum to {raw_eval_step_sum}; summary {evaluation_steps}"),
        ("raw_vs_summary_fit_steps", phase_counts["fit"] == int(summary_counts.get("fit_steps", -1)), f"CSV {phase_counts['fit']}; summary {summary_counts.get('fit_steps')}"),
        ("raw_vs_summary_holdout_steps", phase_counts["holdout"] == int(summary_counts.get("holdout_steps", -1)), f"CSV {phase_counts['holdout']}; summary {summary_counts.get('holdout_steps')}"),
        ("fit_plus_holdout_equals_train", int(summary_counts.get("fit_steps", 0)) + int(summary_counts.get("holdout_steps", 0)) == training_steps, f"fit {summary_counts.get('fit_steps')}; holdout {summary_counts.get('holdout_steps')}; train {training_steps}"),
        ("raw_training_reward", _close(training_reward, summary.get("training_reward"), 1e-5), f"CSV {training_reward:.8g}; summary {summary.get('training_reward')}"),
    ):
        _check(checks, name, passed, details)

    bout_steps = int(run_config.get("bout_steps", 0))
    sequence = run_config.get("sequence", [])
    expected_training = bout_steps * len(sequence) if bout_steps > 0 and isinstance(sequence, list) else None
    _check(
        checks,
        "declared_training_budget",
        expected_training is not None and expected_training == training_steps,
        f"bout_steps × sequence length = {expected_training}; summary training steps = {training_steps}",
    )
    if training_steps > 0 and training_steps % 1024 == 0:
        expected_fit = training_steps * 3 // 4
        expected_holdout = training_steps // 4
        split_passed = (
            int(summary_counts.get("fit_steps", -1)) == expected_fit
            and int(summary_counts.get("holdout_steps", -1)) == expected_holdout
        )
        split_details = f"expected fit/holdout {expected_fit}/{expected_holdout}; observed {summary_counts.get('fit_steps')}/{summary_counts.get('holdout_steps')}"
    else:
        split_passed = None
        split_details = "not checked: training budget is not aligned to complete 1,024-step cycles"
    _check(checks, "three_fit_one_holdout_split", split_passed, split_details)

    probe_size = int(run_config.get("probe_steps", 0))
    unique_tasks = len(set(sequence)) if isinstance(sequence, list) else 0
    expected_probe = probe_size * unique_tasks
    _check(
        checks,
        "declared_probe_budget",
        expected_probe == probe_steps,
        f"probe_steps × unique sequence tasks = {expected_probe}; summary probe steps = {probe_steps}",
    )

    expected_eval_episodes = int(run_config.get("evaluation_episodes", 0)) * len(TASKS) * len(evaluations)
    _check(
        checks,
        "declared_evaluation_episode_budget",
        expected_eval_episodes == len(eval_rows),
        f"configured episodes × {len(TASKS)} tasks × {len(evaluations)} stages = {expected_eval_episodes}; evaluations.csv rows = {len(eval_rows)}",
    )

    # Positive-task training success excludes the deliberately negative-goal tasks.
    completed_positive = [
        row for row in episodes_rows
        if row.get("interaction_kind") == "training"
        and row.get("complete", "0").strip().lower() in ("1", "true")
        and str(row.get("taskobserver", "")).endswith("+")
    ]
    raw_positive_rate = _mean(int(row.get("goal_reached", 0)) for row in completed_positive)
    count_denominator = int(summary_counts.get("positive_completed_episodes", 0))
    count_numerator = int(summary_counts.get("positive_goal_episodes", 0))
    count_rate = count_numerator / count_denominator if count_denominator else None
    _check(
        checks,
        "positive_training_goal_rate",
        raw_positive_rate is not None and count_rate is not None and _close(raw_positive_rate, count_rate),
        f"positive task episodes only: CSV {raw_positive_rate}; summary counters {count_numerator}/{count_denominator} = {count_rate}",
    )
    _check(
        checks,
        "negative_tasks_excluded_from_positive_success",
        all(not str(row.get("taskobserver", "")).endswith("-") for row in completed_positive),
        f"positive goal rate denominator has {len(completed_positive)} complete '+' task episodes",
    )

    decision_rows = _read_jsonl_if_exists(folder / "decisions.jsonl")
    decision_cpu = sum(float(value) for row in decision_rows if (value := _finite(row.get("decision_cpu_s"))) is not None)
    _check(
        checks,
        "decision_cpu_sum",
        _close(decision_cpu, costs.get("decision_cpu_s", 0.0), 1e-5),
        f"decision log sum {decision_cpu:.8g}; summary decision_cpu_s {costs.get('decision_cpu_s')}",
    )
    return {
        "raw_training_steps": raw_training_steps,
        "raw_probe_steps": raw_probe_steps,
        "raw_evaluation_steps": raw_evaluation_steps,
        "raw_total_interactions": raw_total,
        "phase_steps": dict(phase_counts),
        "raw_training_reward": training_reward,
        "raw_evaluation_episode_step_sum": raw_eval_step_sum,
        "raw_evaluation_episode_rows": len(eval_rows),
        "raw_episode_rows": len(episodes_rows),
        "positive_training_complete_episodes": len(completed_positive),
        "positive_training_goal_rate": raw_positive_rate,
        "decision_cpu_s_from_log": decision_cpu,
    }


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _read_seed(input_root: Path, seed: int, condition: str, config: Mapping[str, Any]) -> dict[str, Any]:
    folder = input_root / f"seed_{seed}" / condition
    checks: list[dict[str, Any]] = []
    summary_path = folder / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing run summary: {summary_path}")
    summary = _read_json(summary_path)
    run_config_path = folder / "config.json"
    run_config = _read_json(run_config_path) if run_config_path.is_file() else {}
    identity_ok = int(summary.get("seed", -1)) == seed and summary.get("condition") == condition
    _check(checks, "seed_condition_identity", identity_ok, f"summary identity is seed={summary.get('seed')}, condition={summary.get('condition')}")

    evaluations_raw = summary.get("evaluations", {})
    evaluations = {
        stage: _stage_metrics(evaluations_raw.get(stage, {}))
        for stage in ("initial", "after_source", "final")
        if isinstance(evaluations_raw.get(stage), Mapping)
    }
    positive_online_rate = None
    counts = summary.get("counts", {})
    if isinstance(counts, Mapping):
        denominator = int(counts.get("positive_completed_episodes", 0))
        numerator = int(counts.get("positive_goal_episodes", 0))
        if denominator > 0:
            positive_online_rate = numerator / denominator
    if positive_online_rate is None:
        positive_online_rate = _finite(summary.get("plus_goal_rate"))

    outcomes = {
        "online_reward_per_train_step": _finite(summary.get("reward_per_step")),
        "positive_training_goal_rate": positive_online_rate,
        "eval_final_positive_mean_return": evaluations.get("final", {}).get("positive_task_mean_return"),
        "eval_final_positive_goal_rate": evaluations.get("final", {}).get("positive_task_goal_rate"),
        "eval_coldtransfer_after_source_A2plus_mean_return": evaluations.get("after_source", {}).get("A2plus_mean_return"),
        "eval_coldtransfer_after_source_A2plus_goal_rate": evaluations.get("after_source", {}).get("A2plus_goal_rate"),
        "eval_final_A2plus_mean_return": evaluations.get("final", {}).get("A2plus_mean_return"),
        "eval_final_A2plus_goal_rate": evaluations.get("final", {}).get("A2plus_goal_rate"),
    }

    decisions = _decision_summary(folder, condition, checks)
    assignments = _assignment_summary(folder, checks)
    capacity = _capacity_summary(folder, run_config, checks)
    raw_cost = _raw_cost_audit(folder, summary, run_config, evaluations_raw, checks)
    source_audit = _source_snapshot_audit(folder, summary, checks)

    audit_passed = all(item["passed"] is True for item in checks if item["passed"] is not None)
    costs = summary.get("costs", {})
    record = {
        "seed": seed,
        "condition": condition,
        "run_directory": str(folder),
        "outcomes": outcomes,
        "evaluations": evaluations,
        "gate": decisions,
        "groups": assignments,
        "capacity": capacity,
        "cost": {
            "training_steps": int(costs.get("training_steps", 0)),
            "fit_steps": int(counts.get("fit_steps", 0)),
            "holdout_steps": int(counts.get("holdout_steps", 0)),
            "probe_steps": int(costs.get("probe_steps", 0)),
            "evaluation_steps": int(costs.get("evaluation_steps", 0)),
            "total_interactions": int(costs.get("total_interactions", 0)),
            "decision_cpu_s": _finite(costs.get("decision_cpu_s")),
            "evaluation_cpu_s": _finite(costs.get("evaluation_cpu_s")),
            "cpu_s": _finite(costs.get("cpu_s")),
            "wall_s": _finite(costs.get("wall_s")),
            "process_rss_bytes": _finite(costs.get("process_rss_bytes")),
        },
        "budgets": {
            "bout_steps": run_config.get("bout_steps"),
            "sequence": run_config.get("sequence"),
            "probe_steps_per_task": run_config.get("probe_steps"),
            "evaluation_episodes_per_task_stage": run_config.get("evaluation_episodes"),
            "actor_budget_bytes": capacity["actor_budget_bytes"],
            "package_budget_bytes": capacity["package_budget_bytes"],
            "expected_training_steps": (
                int(run_config.get("bout_steps", 0)) * len(run_config.get("sequence", []))
                if isinstance(run_config.get("sequence", []), list) else None
            ),
        },
        "raw_cost_audit": raw_cost,
        "source_snapshot_audit": source_audit,
        "audit": {"passed": audit_passed, "checks": checks},
        "runner_summary": summary,
    }
    return record


def _condition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "seed_count": len(records),
        "seeds": sorted(int(row["seed"]) for row in records),
        "metrics": {},
        "costs": {},
        "gate": {},
        "capacity": {},
        "groups": {},
        "audit_passed_seed_count": sum(bool(row["audit"]["passed"]) for row in records),
    }
    for metric in OUTCOME_METRICS:
        result["metrics"][metric] = {
            **_stats(row["outcomes"].get(metric) for row in records),
            "per_seed": [
                {"seed": row["seed"], "value": row["outcomes"].get(metric)}
                for row in sorted(records, key=lambda item: item["seed"])
            ],
        }
    for metric in (
        "training_steps", "fit_steps", "holdout_steps", "probe_steps", "evaluation_steps",
        "total_interactions", "decision_cpu_s", "evaluation_cpu_s", "cpu_s", "wall_s", "process_rss_bytes",
    ):
        result["costs"][metric] = _stats(row["cost"].get(metric) for row in records)
    for metric in ("gate_attempts", "gate_accepts", "gate_rejects", "proposal_decisions", "proposal_accepts", "proposal_rejects"):
        result["gate"][metric] = _stats(row["gate"].get(metric) for row in records)
    result["gate"]["decision_reason_counts"] = dict(
        sorted(Counter({
            key: sum(int(row["gate"]["decision_reason_counts"].get(key, 0)) for row in records)
            for row in records
            for key in row["gate"]["decision_reason_counts"]
        }).items())
    )
    for metric in (
        "assignment_count", "shadow_assigned_count", "shadow_abstention_count",
        "shadow_assignment_rate", "deployed_assigned_count", "deployed_abstention_count",
        "deployed_assignment_rate",
    ):
        result["groups"][metric] = _stats(row["groups"].get(metric) for row in records)
    result["groups"]["reason_counts"] = dict(
        sorted(Counter({
            key: sum(int(row["groups"]["reason_counts"].get(key, 0)) for row in records)
            for row in records
            for key in row["groups"]["reason_counts"]
        }).items())
    )
    for metric in ("actor_bytes", "checkpoint_bytes", "full_persistent_bytes", "checkpoint_plus_actor_sidecar_bytes"):
        result["capacity"][metric] = {
            key: _stats(row["capacity"][metric].get(key) for row in records)
            for key in ("mean", "max", "last")
        }
    return result


def _paired_differences(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition_seed = {
        (str(row["condition"]), int(row["seed"])): row for row in records
    }
    conditions = sorted({str(row["condition"]) for row in records}, key=lambda item: CONDITIONS.index(item) if item in CONDITIONS else len(CONDITIONS))
    result: dict[str, Any] = {
        "unit": "seed",
        "interpretation": "descriptive paired differences only; no significance claim",
        "comparisons": {},
    }
    for condition in conditions:
        result["comparisons"][condition] = {}
        for reference in CONTRAST_REFERENCES:
            differences: dict[str, Any] = {}
            common_seeds = sorted(
                seed for arm, seed in by_condition_seed
                if arm == condition and (reference, seed) in by_condition_seed
            )
            for metric in OUTCOME_METRICS:
                paired: list[dict[str, Any]] = []
                values: list[float] = []
                for seed in common_seeds:
                    candidate_value = _finite(by_condition_seed[(condition, seed)]["outcomes"].get(metric))
                    reference_value = _finite(by_condition_seed[(reference, seed)]["outcomes"].get(metric))
                    if candidate_value is None or reference_value is None:
                        continue
                    difference = candidate_value - reference_value
                    values.append(difference)
                    paired.append({
                        "seed": seed,
                        "condition_value": candidate_value,
                        "reference_value": reference_value,
                        "difference": difference,
                    })
                differences[metric] = {**_stats(values), "per_seed": paired}
            result["comparisons"][condition][reference] = {
                "common_seeds": common_seeds,
                "metrics": differences,
            }
    return result


def _audit_cohort(
    cohort: Mapping[str, Any], records: list[dict[str, Any]], conditions: list[str], seeds: list[int]
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    cohort_cost = cohort.get("cohort_cost", {})
    for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions"):
        observed = sum(int(row["cost"].get(key, 0)) for row in records)
        declared = cohort_cost.get(key) if isinstance(cohort_cost, Mapping) else None
        _check(checks, f"cohort_{key}", declared is not None and int(declared) == observed, f"records sum {observed}; cohort summary {declared}")

    expected_cells = len(conditions) * len(seeds)
    found_cells = len({(int(row["seed"]), str(row["condition"])) for row in records})
    _check(checks, "cohort_record_coverage", found_cells == expected_cells, f"found {found_cells} unique seed/condition cells; declared {expected_cells}")
    declared_records = cohort.get("records", [])
    declared_cells = {
        (int(row["seed"]), str(row["condition"]))
        for row in declared_records
        if isinstance(row, Mapping) and "seed" in row and "condition" in row
    }
    expected_keys = {(int(seed), str(condition)) for seed in seeds for condition in conditions}
    _check(
        checks,
        "cohort_summary_record_coverage",
        declared_cells == expected_keys and len(declared_records) == len(expected_keys),
        f"cohort summary lists {len(declared_cells)} unique cells; expected {len(expected_keys)}",
    )
    return {"passed": all(item["passed"] is True for item in checks), "checks": checks}


def _audit_cohort_snapshot_consistency(records: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    manifests = [
        row["source_snapshot_audit"].get("manifest", {})
        for row in records
        if isinstance(row["source_snapshot_audit"].get("manifest"), Mapping)
    ]
    dataset_hashes = {
        str(manifest.get("dataset", {}).get("sha256"))
        for manifest in manifests
        if isinstance(manifest.get("dataset"), Mapping)
    }
    foundation_hashes = {
        str(manifest.get("foundation", {}).get("sha256"))
        for manifest in manifests
        if isinstance(manifest.get("foundation"), Mapping)
    }
    _check(
        checks,
        "cohort_dataset_hash_consistency",
        bool(manifests) and len(dataset_hashes) == 1 and "None" not in dataset_hashes,
        f"dataset SHA256 values across run manifests: {sorted(dataset_hashes)}",
    )
    _check(
        checks,
        "cohort_foundation_hash_consistency",
        bool(manifests) and len(foundation_hashes) == 1 and "None" not in foundation_hashes,
        f"foundation SHA256 values across run manifests: {sorted(foundation_hashes)}",
    )
    source_maps = [
        manifest.get("sources", {}) if isinstance(manifest.get("sources"), Mapping) else {}
        for manifest in manifests
    ]
    names = set().union(*(mapping.keys() for mapping in source_maps)) if source_maps else set()
    source_hashes = {
        name: {
            str(mapping[name].get("sha256"))
            for mapping in source_maps
            if isinstance(mapping.get(name), Mapping)
        }
        for name in names
    }
    stable_sources = bool(source_maps) and bool(names) and all(
        all(name in mapping and isinstance(mapping[name], Mapping) for mapping in source_maps)
        and len(source_hashes[name]) == 1
        and "None" not in source_hashes[name]
        for name in names
    )
    _check(
        checks,
        "cohort_source_snapshot_consistency",
        stable_sources,
        f"{sum(len(values) == 1 for values in source_hashes.values())} / {len(names)} source file hashes agree across run manifests",
    )


def _errorbar_range(ax, x: np.ndarray, groups: list[list[float]], *, color: str, label: str) -> None:
    means = [float(np.mean(values)) if values else np.nan for values in groups]
    low = [mean - min(values) if values else 0.0 for mean, values in zip(means, groups)]
    high = [max(values) - mean if values else 0.0 for mean, values in zip(means, groups)]
    ax.errorbar(x, means, yerr=np.asarray([low, high]), marker="o", capsize=3, color=color, label=label)
    for index, values in enumerate(groups):
        if len(values) > 1:
            ax.scatter(np.full(len(values), x[index]), values, color=color, alpha=0.25, s=16, zorder=3)


def _save_figure(fig: plt.Figure, output: Path, stem: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths[suffix] = path.name
    plt.close(fig)
    return paths


def _make_figures(output: Path, records: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    x = np.arange(len(conditions))
    labels = [condition.replace("_", "\n") for condition in conditions]
    online = [
        [_finite(row["outcomes"].get("online_reward_per_train_step")) for row in records if row["condition"] == condition]
        for condition in conditions
    ]
    online = [[value for value in values if value is not None] for values in online]
    positive = [
        [_finite(row["outcomes"].get("positive_training_goal_rate")) for row in records if row["condition"] == condition]
        for condition in conditions
    ]
    positive = [[value for value in values if value is not None] for values in positive]
    fig, left = plt.subplots(figsize=(13, 5.8))
    _errorbar_range(left, x, online, color="#2864a5", label="Online signed reward / train step (mean; whisker=min–max)")
    left.set_ylabel("Online signed reward / training step")
    left.axhline(0.0, color="#777777", linewidth=0.7)
    right = left.twinx()
    _errorbar_range(right, x, positive, color="#d17a00", label="Positive-task goal rate (mean; whisker=min–max)")
    right.set_ylabel("Positive-task goal rate")
    right.set_ylim(0.0, 1.0)
    left.set_xticks(x, labels)
    left.set_title("Online outcomes by condition; seed is the unit")
    left.grid(axis="y", alpha=0.25)
    handles_a, labels_a = left.get_legend_handles_labels()
    handles_b, labels_b = right.get_legend_handles_labels()
    left.legend(handles_a + handles_b, labels_a + labels_b, loc="best", fontsize=8)
    fig.tight_layout()
    figures = {"online_outcomes": _save_figure(fig, output, "online_outcomes")}

    final_goal = [
        [_finite(row["outcomes"].get("eval_final_positive_goal_rate")) for row in records if row["condition"] == condition]
        for condition in conditions
    ]
    cold_goal = [
        [_finite(row["outcomes"].get("eval_coldtransfer_after_source_A2plus_goal_rate")) for row in records if row["condition"] == condition]
        for condition in conditions
    ]
    final_a2 = [
        [_finite(row["outcomes"].get("eval_final_A2plus_goal_rate")) for row in records if row["condition"] == condition]
        for condition in conditions
    ]
    data = [final_goal, cold_goal, final_a2]
    data = [[[_value for _value in values if _value is not None] for values in category] for category in data]
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for category_index, (category, label, color) in enumerate(zip(data, (
        "Final positive tasks", "After-source cold A2+", "Final A2+ after exposure"
    ), ("#2878b5", "#3b9b76", "#d17a00"))):
        positions = x + (category_index - 1) * width
        means = [float(np.mean(values)) if values else 0.0 for values in category]
        ax.bar(positions, means, width, color=color, alpha=0.78, label=label)
        for index, values in enumerate(category):
            if values:
                ax.scatter(np.full(len(values), positions[index]), values, color="black", alpha=0.45, s=13, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Goal rate (positive task only)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Evaluation goal rates; after-source A2+ is cold transfer")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["evaluation_goal_rates"] = _save_figure(fig, output, "evaluation_goal_rates")

    cost_names = ("training_steps", "probe_steps", "evaluation_steps")
    cost_labels = ("Training", "Probe", "Evaluation")
    cost_colors = ("#2864a5", "#d17a00", "#3b9b76")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    bottom = np.zeros(len(conditions), dtype=np.float64)
    for key, label, color in zip(cost_names, cost_labels, cost_colors):
        values = [
            sum(row["cost"][key] for row in records if row["condition"] == condition)
            / max(1, sum(1 for row in records if row["condition"] == condition))
            for condition in conditions
        ]
        axes[0].bar(x, values, bottom=bottom, color=color, label=label)
        bottom += np.asarray(values)
    axes[0].set_ylabel("Environment steps / seed")
    axes[0].set_title("Measured interaction and computation cost")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    cpu = [
        _mean(row["cost"].get("cpu_s") for row in records if row["condition"] == condition) or 0.0
        for condition in conditions
    ]
    evaluation_cpu = [
        _mean(row["cost"].get("evaluation_cpu_s") for row in records if row["condition"] == condition) or 0.0
        for condition in conditions
    ]
    decision = [
        _mean(row["cost"].get("decision_cpu_s") for row in records if row["condition"] == condition) or 0.0
        for condition in conditions
    ]
    axes[1].bar(x - 0.25, cpu, 0.25, color="#2864a5", label="Total CPU seconds")
    axes[1].bar(x, evaluation_cpu, 0.25, color="#3b9b76", label="Evaluation CPU seconds")
    axes[1].bar(x + 0.25, decision, 0.25, color="#d17a00", label="Decision CPU seconds")
    axes[1].set_ylabel("Seconds / seed")
    axes[1].set_xticks(x, labels)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    figures["costs"] = _save_figure(fig, output, "costs")

    actor = [
        _mean(row["capacity"]["actor_bytes"].get("max") for row in records if row["condition"] == condition)
        for condition in conditions
    ]
    checkpoint = [
        _mean(row["capacity"]["checkpoint_bytes"].get("max") for row in records if row["condition"] == condition)
        for condition in conditions
    ]
    package = [
        _mean(row["capacity"]["full_persistent_bytes"].get("max") for row in records if row["condition"] == condition)
        for condition in conditions
    ]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    width = 0.25
    ax.bar(x - width, [value or 0 for value in actor], width, color="#3b9b76", label="Peak actor sidecar bytes")
    ax.bar(x, [value or 0 for value in checkpoint], width, color="#7251a3", label="Peak .pt checkpoint bytes")
    ax.bar(x + width, [value or 0 for value in package], width, color="#d17a00", label="Peak W0 + checkpoint package bytes")
    ax.set_ylabel("Bytes")
    ax.set_xticks(x, labels)
    ax.set_title("Observed actor, full-state checkpoint, and complete package sizes")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    figures["capacity"] = _save_figure(fig, output, "capacity")
    return figures


def _format(value: Any, digits: int = 4) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _markdown_report(analysis: Mapping[str, Any]) -> str:
    conditions = analysis["conditions"]
    summaries = analysis["condition_summary"]
    lines = [
        "# Write-decision cohort 分析",
        "",
        f"输入 cohort：`{analysis['input_cohort']}`。阶段：`{analysis.get('phase')}`；seed：{', '.join(str(value) for value in analysis['seeds']) or '无'}。",
        "",
        "统计单位是 seed。episode、transition 和 gate 决策只用于计算每个 seed 的指标，不作为独立重复。当前 seed 数量只支持描述性比较；不报告显著性或因果结论。",
        "",
        "负奖励任务的 `goal_reached` 不计入正向 success。`after_source` 的 A2+ 是 A2 训练之前的 cold-transfer 观察；final A2+ 是完成其后续 exposure 后的观察。",
        "",
        "## 条件均值（每 seed 先汇总）",
        "",
        "| 条件 | n seeds | Online reward / train step | 正向训练 goal rate | Final 正向 eval goal | Cold A2+ goal | CPU s | Wall s | Interactions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        item = summaries.get(condition, {})
        metric = item.get("metrics", {})
        costs = item.get("costs", {})
        lines.append(
            "| {condition} | {n} | {reward} | {train_goal} | {eval_goal} | {cold_goal} | {cpu} | {wall} | {interactions} |".format(
                condition=condition,
                n=item.get("seed_count", 0),
                reward=_format(metric.get("online_reward_per_train_step", {}).get("mean")),
                train_goal=_format(metric.get("positive_training_goal_rate", {}).get("mean")),
                eval_goal=_format(metric.get("eval_final_positive_goal_rate", {}).get("mean")),
                cold_goal=_format(metric.get("eval_coldtransfer_after_source_A2plus_goal_rate", {}).get("mean")),
                cpu=_format(costs.get("cpu_s", {}).get("mean"), 2),
                wall=_format(costs.get("wall_s", {}).get("mean"), 2),
                interactions=_format(costs.get("total_interactions", {}).get("mean"), 0),
            )
        )
    missing = analysis.get("missing_conditions", [])
    if missing:
        lines.extend(["", f"缺少条件：{', '.join(missing)}。该输入不是完整的八条件 cohort。"])

    lines.extend([
        "",
        "## Paired 差值",
        "",
        "下表给出相同 seed 内的 candidate − reference 均值；括号中列出 seed 数。数值是描述性差异，不代表统计显著或已识别共享收益。完整逐 seed 差值见 `analysis.json`。",
        "",
        "| 条件 | 对照 | Online reward / step Δ | 正向训练 goal Δ | Final 正向 eval return Δ | Cold A2+ return Δ | n paired seeds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for condition in conditions:
        for reference in CONTRAST_REFERENCES:
            comparison = analysis["paired_differences"]["comparisons"].get(condition, {}).get(reference, {})
            metrics = comparison.get("metrics", {})
            lines.append(
                f"| {condition} | {reference} | {_format(metrics.get('online_reward_per_train_step', {}).get('mean'))} | "
                f"{_format(metrics.get('positive_training_goal_rate', {}).get('mean'))} | "
                f"{_format(metrics.get('eval_final_positive_mean_return', {}).get('mean'))} | "
                f"{_format(metrics.get('eval_coldtransfer_after_source_A2plus_mean_return', {}).get('mean'))} | "
                f"{len(comparison.get('common_seeds', []))} |"
            )

    lines.extend([
        "",
        "## Gate、ESS 与 policy drift",
        "",
        "| 条件 | Proposal decisions | Gate attempts | Gate accepts | Gate rejects | Main reject reasons | Cold ESS 均值 | Cold behavior KL 均值 | Old aggregate ESS 均值 | Old aggregate max KL 均值 | Decision CPU s |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for condition in conditions:
        item = summaries.get(condition, {}).get("gate", {})
        per_seed = [row for row in analysis["per_seed"] if row["condition"] == condition]
        reasons = Counter()
        for row in per_seed:
            reasons.update(row["gate"].get("rejection_reason_counts", {}))
        top_reasons = ", ".join(f"{name}:{count}" for name, count in reasons.most_common(3)) or "—"
        current_cold = [row["gate"]["gate_diagnostics"]["current_cold"] for row in per_seed]
        old_aggregate = [row["gate"]["gate_diagnostics"]["old_aggregate"] for row in per_seed]
        cold_ess = _mean(row["ess"]["mean"] for row in current_cold)
        cold_kl = _mean(row["behavior_kl"]["mean"] for row in current_cold)
        old_ess = _mean(row["ess"]["mean"] for row in old_aggregate)
        old_max_kl = _mean(row["max_kl"]["mean"] for row in old_aggregate)
        decision_cpu = summaries.get(condition, {}).get("costs", {}).get("decision_cpu_s", {}).get("mean")
        lines.append(
            f"| {condition} | {item.get('proposal_decisions', {}).get('mean', 0):.1f} | "
            f"{item.get('gate_attempts', {}).get('mean', 0):.1f} | {item.get('gate_accepts', {}).get('mean', 0):.1f} | "
            f"{item.get('gate_rejects', {}).get('mean', 0):.1f} | {top_reasons} | {_format(cold_ess)} | {_format(cold_kl)} | {_format(old_ess)} | {_format(old_max_kl)} | "
            f"{_format(decision_cpu, 4)} |"
        )
    lines.extend([
        "",
        "## Group coverage",
        "",
        "`shadow_group` 表示证据分组器的分配；`group` 表示该条件实际部署的分组。group ID 只是局部分组编号，不是任务或 reward 标签。Coverage 按每个 seed 的 assignment 数与比例汇总。",
        "",
        "| 条件 | Assignment rows / seed | Shadow assigned / seed | Shadow coverage | Deployed assigned / seed | Deployed coverage | Assignment reasons |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for condition in conditions:
        group_summary = summaries.get(condition, {}).get("groups", {})
        per_seed = [row for row in analysis["per_seed"] if row["condition"] == condition]
        reasons = Counter()
        for row in per_seed:
            reasons.update(row["groups"].get("reason_counts", {}))
        reason_text = ", ".join(f"{name}:{count}" for name, count in reasons.most_common(4)) or "—"
        lines.append(
            f"| {condition} | {_format(group_summary.get('assignment_count', {}).get('mean'), 1)} | "
            f"{_format(group_summary.get('shadow_assigned_count', {}).get('mean'), 1)} | "
            f"{_format(group_summary.get('shadow_assignment_rate', {}).get('mean'))} | "
            f"{_format(group_summary.get('deployed_assigned_count', {}).get('mean'), 1)} | "
            f"{_format(group_summary.get('deployed_assignment_rate', {}).get('mean'))} | {reason_text} |"
        )
    lines.extend([
        "",
        "Reward identification is not an oracle: reward coefficients are inferred from the finite transition/reward evidence available in each profile. A profile with no observed terminal transitions may have weak or zero terminal-feature estimates after ridge regularization, which can look like `reward_mismatch` against a better-supported profile. That reason does not establish task or reward incompatibility. The measured group coverage is limited to these probe assignments; unresolved abstentions and mismatches mean routing coverage is not shown to be solved.",
    ])
    lines.extend([
        "",
        "Gate accepts/rejects 是当前规则基于 cold-parent 与有限旧 anchor surrogate 的输出。没有独立 counterfactual 时，共享是否真实有益为 undetermined；因此 false accept 和 false reject 目前都不可测。Surrogate ESS/KL、优势估计及 episode-cluster SE 不是共享收益真值或安全保证。",
        "",
        "## 资源与物理容量",
        "",
        "| 条件 | Train steps / seed | Fit / Holdout | Probe steps / seed | Eval steps / seed | Actor peak B | Full checkpoint .pt peak B | W0 + checkpoint package peak B | CPU s | Eval CPU s | Wall s | Decision CPU s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in conditions:
        item = summaries.get(condition, {})
        costs = item.get("costs", {})
        capacity = item.get("capacity", {})
        fit = costs.get("fit_steps", {}).get("mean")
        hold = costs.get("holdout_steps", {}).get("mean")
        fit_hold = f"{_format(fit, 0)} / {_format(hold, 0)}"
        lines.append(
            f"| {condition} | {_format(costs.get('training_steps', {}).get('mean'), 0)} | {fit_hold} | "
            f"{_format(costs.get('probe_steps', {}).get('mean'), 0)} | {_format(costs.get('evaluation_steps', {}).get('mean'), 0)} | "
            f"{_format(capacity.get('actor_bytes', {}).get('max', {}).get('mean'), 0)} | "
            f"{_format(capacity.get('checkpoint_bytes', {}).get('max', {}).get('mean'), 0)} | "
            f"{_format(capacity.get('full_persistent_bytes', {}).get('max', {}).get('mean'), 0)} | "
            f"{_format(costs.get('cpu_s', {}).get('mean'), 2)} | {_format(costs.get('evaluation_cpu_s', {}).get('mean'), 2)} | "
            f"{_format(costs.get('wall_s', {}).get('mean'), 2)} | "
            f"{_format(costs.get('decision_cpu_s', {}).get('mean'), 4)} |"
        )

    audit_rows = []
    for row in analysis["per_seed"]:
        audit_rows.append((row["seed"], row["condition"], row["audit"]["passed"], row["audit"]["failed_checks"]))
    lines.extend([
        "",
        "## 数据账与审计",
        "",
        f"Dataset SHA256：`{analysis.get('dataset_sha256') or 'unknown'}`。",
        "",
        "| Seed | 条件 | 审计通过 | 失败项 |",
        "|---:|---|---|---|",
    ])
    for seed, condition, passed, failed in audit_rows:
        lines.append(f"| {seed} | {condition} | {'是' if passed else '否'} | {', '.join(failed) or '—'} |")
    lines.extend([
        "",
        "计费复核从原始 `steps.csv`、`probes.csv` 和 `evaluations.csv` 计数，并核对 summary 中的 training/probe/evaluation/total interactions、fit/holdout、decision CPU 及每个 checkpoint 与 actor sidecar 的磁盘字节。预算检查将 actor sidecar 与 actor cap 比较，并将 `full_persistent_bytes`（W0 foundation 加序列化状态）与 package cap 比较。来源快照与 W0/dataset manifest hash 会逐项验证。",
        "",
        "## 解释边界",
        "",
        "这些结果是本 cohort 上的描述性比较。两个或三个 seed 不足以支撑显著性表述。Whole-run arm 的对照可以比较平均结果，但单次 gate 的 false accept/reject 仍需要独立、预先定义的 counterfactual 结果才能判断。不得据此称该方法为 SOTA，也不能据此宣称已证明创新性。",
        "",
        "## 图表",
        "",
    ])
    for name, paths in analysis["figures"].items():
        lines.append(f"- {name}: `{paths['png']}` / `{paths['pdf']}`")
    return "\n".join(lines) + "\n"


def analyze_cohort(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source = Path(input_path).expanduser().resolve()
    if source.is_dir():
        cohort_root = source
        summary_path = source / "summary.json"
    else:
        summary_path = source
        cohort_root = source.parent
    if not summary_path.is_file():
        raise FileNotFoundError(f"cohort summary not found: {summary_path}")
    cohort = _read_json(summary_path)
    if not isinstance(cohort.get("records"), list):
        raise ValueError("cohort summary must contain a records list")
    frozen_path = cohort_root / "frozen_config.json"
    frozen = _read_json(frozen_path) if frozen_path.is_file() else {}
    conditions = list(frozen.get("conditions", cohort.get("conditions", CONDITIONS)))
    seeds = [int(value) for value in frozen.get("seeds", cohort.get("seeds", []))]
    if not conditions:
        conditions = list(dict.fromkeys(str(row.get("condition")) for row in cohort["records"]))
    if not seeds:
        seeds = sorted({int(row["seed"]) for row in cohort["records"] if "seed" in row})

    records: list[dict[str, Any]] = []
    for seed in seeds:
        for condition in conditions:
            records.append(_read_seed(cohort_root, seed, str(condition), frozen))
    conditions_present = sorted(
        {row["condition"] for row in records}, key=lambda item: CONDITIONS.index(item) if item in CONDITIONS else len(CONDITIONS)
    )
    missing_conditions = [condition for condition in CONDITIONS if condition not in conditions_present]
    per_condition: dict[str, list[dict[str, Any]]] = {
        condition: [row for row in records if row["condition"] == condition]
        for condition in conditions
    }
    condition_summary = {
        condition: _condition_summary(rows)
        for condition, rows in per_condition.items()
    }
    paired = _paired_differences(records)
    cohort_audit = _audit_cohort(cohort, records, conditions, seeds)
    _audit_cohort_snapshot_consistency(records, cohort_audit["checks"])
    cohort_audit["passed"] = all(check["passed"] is True for check in cohort_audit["checks"])
    all_dataset_hashes = sorted({
        str(row["source_snapshot_audit"].get("dataset", {}).get("expected_sha256"))
        for row in records
        if row["source_snapshot_audit"].get("dataset")
    })
    dataset_hash = all_dataset_hashes[0] if len(all_dataset_hashes) == 1 else None

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"report output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    figures = _make_figures(output, records, conditions)
    figure_hashes = {
        name: {
            suffix: {"path": filename, "sha256": _sha256(output / filename), "bytes": (output / filename).stat().st_size}
            for suffix, filename in files.items()
        }
        for name, files in figures.items()
    }

    analysis: dict[str, Any] = {
        "format": "write_decision_analysis_v1",
        "input_cohort": str(cohort_root),
        "input_summary": str(summary_path),
        "phase": cohort.get("phase"),
        "seeds": seeds,
        "conditions": conditions,
        "conditions_present": conditions_present,
        "missing_conditions": missing_conditions,
        "dataset_sha256": dataset_hash,
        "dataset_hashes_seen": all_dataset_hashes,
        "statistical_unit": "seed",
        "significance_claim": False,
        "shared_benefit_truth": "undetermined_without_independent_counterfactual",
        "false_accept_reject_measurable": False,
        "condition_summary": condition_summary,
        "paired_differences": paired,
        "cohort_audit": cohort_audit,
        "per_seed": [
            {
                **row,
                "audit": {
                    **row["audit"],
                    "failed_checks": [item["name"] for item in row["audit"]["checks"] if item["passed"] is False],
                },
            }
            for row in records
        ],
        "figures": figures,
        "figure_hashes": figure_hashes,
        "interpretation_limits": [
            "Episodes and transitions are aggregated within seed; they are not independent replicates.",
            "Two or three seeds support descriptive paired differences only, not significance claims.",
            "Gate surrogate acceptance is not an independent truth label for shared benefit.",
            "False accepts and false rejects are unmeasurable without independent counterfactual outcomes.",
            "Reward-profile matching is a finite-evidence heuristic, not an oracle; observed group coverage and reward_mismatch abstentions do not show the coverage problem is solved.",
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
    parser.add_argument("--input", type=Path, required=True, help="cohort output directory or its summary.json")
    parser.add_argument("--output", type=Path, required=True, help="empty/new report output directory")
    args = parser.parse_args(argv)
    analysis = analyze_cohort(args.input, args.output)
    print(json.dumps({
        "phase": analysis.get("phase"),
        "seeds": analysis.get("seeds"),
        "conditions": analysis.get("conditions_present"),
        "output": str(Path(args.output).resolve()),
        "cohort_audit_passed": analysis["cohort_audit"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

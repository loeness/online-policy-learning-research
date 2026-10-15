"""Audit and summarize same-dynamics reward-conflict runs.

This is a read-only postprocessor. It does not construct or interact with an
environment; it audits the saved CSV, JSON, source snapshots, and checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from online_rl.shared_write_grouping import empty_profile, observe_profile  # noqa: E402
from online_rl.shared_write_storage import decode_shared_knowledge  # noqa: E402


CONDITIONS = ("independent_conflict", "shared_conflict", "frozen_conflict")
MODES = ("plus", "minus")
STAGES = ("initial", "after_plus", "after_minus")
METRIC_KEYS = (
    "plus_after_plus_return",
    "plus_after_plus_goal_success",
    "plus_after_minus_return",
    "plus_after_minus_goal_success",
    "plus_return_retention_delta",
    "plus_goal_success_retention_delta",
    "minus_initial_signed_return",
    "minus_after_plus_signed_return",
    "minus_after_minus_signed_return",
    "minus_cold_transfer_signed_return",
    "minus_adaptation_gain_after_plus",
    "minus_training_signed_reward_per_step",
    "minus_goal_termination_rate_after_plus_metadata",
    "minus_goal_termination_rate_after_minus_metadata",
)
EXPECTED_ACTOR_BYTES = {
    "independent_conflict": 666,
    "shared_conflict": 738,
    "frozen_conflict": 738,
}
REQUIRED_CHECKPOINT_FIELDS = {
    "format",
    "condition",
    "seed",
    "actor_binary",
    "critic",
    "optimizer",
    "slot_groups",
    "slot_generation",
    "shared_frozen",
    "grouper",
    "profiles",
    "group_decisions",
    "action_rng",
    "foundation_sha256",
    "dataset_sha256",
    "source_manifest",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_check(
    checks: list[dict[str, Any]], name: str, passed: bool, details: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": str(details)})


def _close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    return float(statistics.fmean(items)) if items else None


def _parse_array(value: str, *, dtype: Any = None) -> np.ndarray:
    parsed = np.asarray(json.loads(value), dtype=dtype)
    return parsed


def _same_fields(
    left: list[Mapping[str, Any]],
    right: list[Mapping[str, Any]],
    fields: Iterable[str],
) -> tuple[bool, str]:
    if len(left) != len(right):
        return False, f"row counts differ: {len(left)} vs {len(right)}"
    for row_index, (left_row, right_row) in enumerate(zip(left, right)):
        for field in fields:
            if left_row.get(field) != right_row.get(field):
                return False, f"row {row_index + 1} field {field} differs"
    return True, f"{len(left)} rows match on {', '.join(fields)}"


def _load_checkpoint(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / f"checkpoint_{name}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _decoded_actor(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local, shared, groups = decode_shared_knowledge(payload["actor_binary"])
    return local, shared, groups


def _nested_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _nested_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_state_equal(a, b) for a, b in zip(left, right)
        )
    try:
        result = left == right
        return bool(result) if not isinstance(result, np.ndarray) else bool(result.all())
    except (TypeError, ValueError, RuntimeError):
        return False


def _eval_metrics(rows: list[Mapping[str, str]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for stage in STAGES:
        results[stage] = {}
        for mode in MODES:
            selected = [row for row in rows if row["stage"] == stage and row["mode"] == mode]
            if not selected:
                results[stage][mode] = None
                continue
            returns = [float(row["return"]) for row in selected]
            goal_rate = _mean(float(row["task_success"]) for row in selected) if mode == "plus" else None
            termination_rate = _mean(float(row["goal_reached"]) for row in selected)
            results[stage][mode] = {
                "mean_return": float(statistics.fmean(returns)),
                "goal_success_rate": goal_rate,
                "goal_termination_rate_metadata": termination_rate,
                "mean_steps": _mean(float(row["steps"]) for row in selected),
                "episodes": len(selected),
            }
    return results


def _run_metrics(
    condition: str,
    summary_record: Mapping[str, Any],
    evaluations: list[Mapping[str, str]],
    steps: list[Mapping[str, str]],
    slot_map: Mapping[str, int],
    after_plus_payload: Mapping[str, Any] | None,
    after_minus_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    eval_metrics = _eval_metrics(evaluations)
    pp = eval_metrics["after_plus"]["plus"]
    pm = eval_metrics["after_minus"]["plus"]
    mi = eval_metrics["initial"]["minus"]
    mp = eval_metrics["after_plus"]["minus"]
    mm = eval_metrics["after_minus"]["minus"]
    minus_rows = [row for row in steps if row["mode"] == "minus"]
    minus_reward_per_step = _mean(float(row["reward"]) for row in minus_rows)

    metrics: dict[str, Any] = {
        "plus_after_plus_return": pp["mean_return"],
        "plus_after_plus_goal_success": pp["goal_success_rate"],
        "plus_after_minus_return": pm["mean_return"],
        "plus_after_minus_goal_success": pm["goal_success_rate"],
        "plus_return_retention_delta": pm["mean_return"] - pp["mean_return"],
        "plus_goal_success_retention_delta": (
            pm["goal_success_rate"] - pp["goal_success_rate"]
        ),
        "minus_initial_signed_return": mi["mean_return"],
        "minus_after_plus_signed_return": mp["mean_return"],
        "minus_after_minus_signed_return": mm["mean_return"],
        "minus_cold_transfer_signed_return": (
            mp["mean_return"] - mi["mean_return"]
        ),
        "minus_adaptation_gain_after_plus": (
            mm["mean_return"] - mp["mean_return"]
        ),
        "minus_training_signed_reward_per_step": minus_reward_per_step,
        "minus_goal_termination_rate_after_plus_metadata": (
            mp["goal_termination_rate_metadata"]
        ),
        "minus_goal_termination_rate_after_minus_metadata": (
            mm["goal_termination_rate_metadata"]
        ),
    }
    if after_plus_payload is not None and after_minus_payload is not None:
        local_plus, shared_plus, _groups_plus = _decoded_actor(after_plus_payload)
        local_minus, shared_minus, _groups_minus = _decoded_actor(after_minus_payload)
        plus_slot, minus_slot = int(slot_map["plus"]), int(slot_map["minus"])
        metrics["minus_local_max_abs_after_plus"] = float(
            np.max(np.abs(local_plus[minus_slot]))
        )
        metrics["old_plus_local_unchanged_during_minus"] = bool(
            np.array_equal(local_plus[plus_slot], local_minus[plus_slot])
        )
        metrics["shared_parent_delta_l2_after_minus"] = float(
            np.linalg.norm(shared_minus.astype(np.float64) - shared_plus.astype(np.float64))
        )
    else:
        metrics["minus_local_max_abs_after_plus"] = None
        metrics["old_plus_local_unchanged_during_minus"] = None
        metrics["shared_parent_delta_l2_after_minus"] = None
    metrics["minus_goal_metric_label"] = "environment termination metadata, not task success"
    return metrics


def _audit_run(
    input_root: Path,
    summary_record: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    seed = int(summary_record["seed"])
    condition = str(summary_record["condition"])
    run_dir = input_root / f"seed_{seed}" / condition
    prefix = f"{seed}/{condition}"
    required_files = (
        "config.json",
        "slot_map.json",
        "summary.json",
        "cost_ledger.json",
        "probe_steps.csv",
        "steps.csv",
        "episodes.csv",
        "evaluations.csv",
        "capacity.csv",
        "source_manifest.json",
        "probe_profile_hashes.json",
        "group_decisions.json",
    )
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    _record_check(checks, f"{prefix}/required_artifacts", not missing, f"missing={missing}")
    if missing:
        return {"seed": seed, "condition": condition, "run_dir": str(run_dir), "error": f"missing {missing}"}

    run_summary = _read_json(run_dir / "summary.json")
    config = _read_json(run_dir / "config.json")
    slot_map = _read_json(run_dir / "slot_map.json")
    costs = _read_json(run_dir / "cost_ledger.json")
    probes = _read_csv(run_dir / "probe_steps.csv")
    steps = _read_csv(run_dir / "steps.csv")
    episodes = _read_csv(run_dir / "episodes.csv")
    evaluations = _read_csv(run_dir / "evaluations.csv")
    capacity = _read_csv(run_dir / "capacity.csv")
    if "scope" not in summary_record:
        summary_record = run_summary
    counts = summary_record["counts"]
    probe_count, training_count = len(probes), len(steps)
    evaluation_count = sum(int(row["steps"]) for row in evaluations)
    interaction_count = probe_count + training_count + evaluation_count

    _record_check(
        checks,
        f"{prefix}/raw_counts_match_summary",
        probe_count == int(counts["probe_steps"])
        and training_count == int(counts["training_steps"])
        and training_count == int(counts["environment_steps"])
        and evaluation_count == int(counts["evaluation_steps"])
        and interaction_count == int(summary_record["total_interactions"]),
        f"raw probe/train/eval/total={probe_count}/{training_count}/{evaluation_count}/{interaction_count}; "
        f"summary={counts['probe_steps']}/{counts['training_steps']}/{counts['evaluation_steps']}/{summary_record['total_interactions']}",
    )
    _record_check(
        checks,
        f"{prefix}/cost_ledger_counts_match_raw",
        probe_count == int(costs["probe_interactions"])
        and training_count == int(costs["ppo_fit_and_holdout_interactions"])
        and evaluation_count == int(costs["evaluation_interactions"])
        and interaction_count == int(costs["total_environment_interactions"]),
        f"raw={probe_count}/{training_count}/{evaluation_count}/{interaction_count}; "
        f"ledger={costs['probe_interactions']}/{costs['ppo_fit_and_holdout_interactions']}/"
        f"{costs['evaluation_interactions']}/{costs['total_environment_interactions']}",
    )

    expected_steps = 2 * int(config["bout_steps"])
    step_ids = [int(row["step"]) for row in steps]
    unique_step_ids = len(set(step_ids)) == len(step_ids)
    step_ids_sequential = sorted(step_ids) == list(range(1, expected_steps + 1))
    fit_rows = [row for row in steps if row["fit_or_holdout"] == "fit"]
    holdout_rows = [row for row in steps if row["fit_or_holdout"] == "holdout"]
    fit_fraction = len(fit_rows) / len(steps) if steps else 0.0
    fit_per_bout = int(config["bout_steps"]) * int(config["fit_rollouts_per_1024"]) / (
        int(config["fit_rollouts_per_1024"]) + int(config["holdout_rollouts_per_1024"])
    )
    bout_ids = sorted({int(row["bout"]) for row in steps})
    fit_schedule_ok = all(
        sum(1 for row in steps if int(row["bout"]) == bout and row["fit_or_holdout"] == "fit")
        == int(fit_per_bout)
        for bout in bout_ids
    )
    _record_check(
        checks,
        f"{prefix}/fresh_fit_holdout_schedule",
        unique_step_ids
        and step_ids_sequential
        and _close(fit_fraction, 0.75, 1e-12)
        and fit_schedule_ok
        and len(fit_rows) == int(counts["fit_steps"])
        and len(holdout_rows) == int(counts["holdout_steps"])
        and int(counts.get("stale_actor_rows", -1)) == 0,
        f"unique/sequential={unique_step_ids}/{step_ids_sequential}; fit={len(fit_rows)}/{len(steps)} "
        f"({fit_fraction:.3f}); per-bout fit schedule={fit_schedule_ok}; stale_actor_rows={counts.get('stale_actor_rows')}",
    )

    mode_reward_totals: dict[str, float] = {}
    training_episode_check = True
    for mode in MODES:
        mode_steps = [row for row in steps if row["mode"] == mode]
        mode_episodes = [row for row in episodes if row["mode"] == mode]
        reward_sum = math.fsum(float(row["reward"]) for row in mode_steps)
        mode_reward_totals[mode] = reward_sum
        online = summary_record["training_reward_by_mode"][mode]
        completed = [row for row in mode_episodes if row["completed"].strip().lower() in ("1", "true")]
        goals = sum(int(row["goal_reached"]) for row in completed)
        training_episode_check &= (
            int(online["completed_episodes"]) == len(completed)
            and int(online["goal_reaches"]) == goals
            and _close(online["training_reward"], reward_sum)
            and _close(online["training_reward_per_step"], reward_sum / len(mode_steps))
        )
        for episode in mode_episodes:
            episode_sum = float(episode["return"])
            # CSV episode rows include incomplete terminal records at the bout boundary.
            if not math.isfinite(episode_sum):
                training_episode_check = False
        if mode == "minus":
            training_episode_check &= all(not row["task_success"].strip() for row in mode_episodes)
            training_episode_check &= online.get("goal_success_rate") is None
        else:
            training_episode_check &= all(row["task_success"].strip() != "" for row in mode_episodes)
    _record_check(
        checks,
        f"{prefix}/signed_training_reward_and_episode_ledger",
        training_episode_check,
        f"reward totals={mode_reward_totals}; minus task_success is blank",
    )

    evaluation_metrics = _eval_metrics(evaluations)
    eval_consistency = True
    for stage in STAGES:
        for mode in MODES:
            rows = [row for row in evaluations if row["stage"] == stage and row["mode"] == mode]
            stat = evaluation_metrics[stage][mode]
            logged = summary_record["evaluations"][stage][mode]
            eval_consistency &= (
                len(rows) == int(config["evaluation_episodes"])
                and _close(stat["mean_return"], logged["mean_return"])
                and int(stat["episodes"]) == len(logged["episodes"])
                and sum(int(row["steps"]) for row in rows)
                == int(logged["total_steps"])
            )
            for row in rows:
                actions = _parse_array(row["actions"], dtype=np.int64)
                rewards = _parse_array(row["rewards"], dtype=np.float64)
                eval_consistency &= len(actions) == len(rewards) == int(row["steps"])
                eval_consistency &= _close(float(np.sum(rewards)), float(row["return"]))
                if mode == "minus":
                    eval_consistency &= not row["task_success"].strip()
                else:
                    eval_consistency &= row["task_success"].strip() == row["goal_reached"].strip()
    plans: dict[tuple[str, int], tuple[str, str]] = {}
    paired_plan_ok = True
    for row in evaluations:
        key = (row["mode"], int(row["episode"]))
        plan = (row["reset_seed"], row["action_seed"])
        if key in plans and plans[key] != plan:
            paired_plan_ok = False
        plans[key] = plan
    _record_check(
        checks,
        f"{prefix}/paired_evaluation_and_summary",
        eval_consistency and paired_plan_ok,
        f"evaluation stages={len({row['stage'] for row in evaluations})}; common mode/episode plan={paired_plan_ok}; "
        "minus task_success remains blank",
    )

    # Rebuild the profile from logged transitions using only observation, action,
    # and next observation. The reward field is deliberately never read here.
    profile_hashes = _read_json(run_dir / "probe_profile_hashes.json")
    decisions = _read_json(run_dir / "group_decisions.json")
    rebuilt_hashes: dict[str, str] = {}
    for mode in MODES:
        profile = empty_profile()
        mode_probes = [row for row in probes if row["mode"] == mode]
        for row in mode_probes:
            observe_profile(
                profile,
                _parse_array(row["obs"], dtype=np.float32),
                int(row["action"]),
                _parse_array(row["next_obs"], dtype=np.float32),
            )
        rebuilt_hashes[mode] = hashlib.sha256(profile.tobytes()).hexdigest()
    profile_ok = (
        rebuilt_hashes == profile_hashes
        and all(decisions[mode]["profile_sha256"] == rebuilt_hashes[mode] for mode in MODES)
        and rebuilt_hashes["plus"] == rebuilt_hashes["minus"]
    )
    if condition == "independent_conflict":
        profile_ok &= all(decisions[mode]["group"] == -1 for mode in MODES)
    else:
        profile_ok &= all(decisions[mode]["group"] == 0 for mode in MODES)
    _record_check(
        checks,
        f"{prefix}/reward_blind_probe_profiles",
        profile_ok,
        f"rebuilt hashes={rebuilt_hashes}; grouping={'disabled' if condition == 'independent_conflict' else 'group 0'}",
    )

    # Hash every frozen source snapshot and cross-check the per-run and checkpoint manifests.
    manifest = _read_json(run_dir / "source_manifest.json")
    manifest_match = manifest == summary_record["source_manifest"]
    missing_or_mismatched: list[str] = []
    for relative, digest in manifest.get("sources_sha256", {}).items():
        snapshot = run_dir / "source_snapshot" / Path(relative)
        if not snapshot.is_file() or _sha256(snapshot) != digest:
            missing_or_mismatched.append(relative)
    _record_check(
        checks,
        f"{prefix}/source_snapshot_hashes",
        manifest_match and not missing_or_mismatched,
        f"manifest files={len(manifest.get('sources_sha256', {}))}; bad={missing_or_mismatched}",
    )

    # Validate each saved checkpoint against measured capacity.csv bytes.
    capacity_ok = bool(capacity)
    capacity_records: list[dict[str, Any]] = []
    loaded_checkpoints: dict[str, dict[str, Any]] = {}
    actor_cap = int(config["actor_budget_bytes"])
    package_cap = int(config["whole_package_cap_bytes"])
    for row in capacity:
        name = row["checkpoint"]
        checkpoint_path = run_dir / f"checkpoint_{name}.pt"
        actor_path = run_dir / f"actor_{name}.bin"
        if not checkpoint_path.is_file() or not actor_path.is_file():
            capacity_ok = False
            capacity_records.append({"checkpoint": name, "error": "checkpoint or actor payload missing"})
            continue
        try:
            payload = _load_checkpoint(run_dir, name)
        except Exception as exc:  # corrupted files are recorded as audit failures
            capacity_ok = False
            capacity_records.append({"checkpoint": name, "error": f"load failed: {exc}"})
            continue
        loaded_checkpoints[name] = payload
        actual_actor_bytes = len(payload.get("actor_binary", b""))
        actual_checkpoint_bytes = checkpoint_path.stat().st_size
        actual_actor_file_bytes = actor_path.stat().st_size
        foundation_bytes = int(row["foundation_bytes"])
        actual_package_bytes = foundation_bytes + actual_checkpoint_bytes
        fields_ok = REQUIRED_CHECKPOINT_FIELDS.issubset(payload)
        row_ok = (
            actual_actor_bytes == int(row["actor_bytes"])
            and int(payload.get("actor_payload_bytes", -1)) == actual_actor_bytes
            and actual_actor_file_bytes == actual_actor_bytes
            and actual_checkpoint_bytes == int(row["checkpoint_bytes"])
            and actual_package_bytes == int(row["whole_package_bytes"])
            and actual_actor_bytes <= actor_cap
            and actual_package_bytes <= package_cap
            and fields_ok
            and payload.get("condition") == condition
            and int(payload.get("seed", -1)) == seed
            and payload.get("source_manifest") == manifest
        )
        capacity_ok &= row_ok
        capacity_records.append({
            "step": int(row["step"]),
            "checkpoint": name,
            "actor_bytes": actual_actor_bytes,
            "checkpoint_bytes": actual_checkpoint_bytes,
            "foundation_bytes": foundation_bytes,
            "whole_package_bytes": actual_package_bytes,
            "actor_cap_bytes": actor_cap,
            "whole_package_cap_bytes": package_cap,
            "passed": bool(row_ok),
        })
    expected_actor = EXPECTED_ACTOR_BYTES[condition]
    capacity_ok &= all(row.get("actor_bytes") == expected_actor for row in capacity_records if "actor_bytes" in row)
    _record_check(
        checks,
        f"{prefix}/actor_and_checkpoint_byte_caps",
        capacity_ok,
        f"measured checkpoints={len(capacity_records)}; actor bytes expected={expected_actor}; "
        f"actor cap={actor_cap}; W0+checkpoint cap={package_cap}",
    )

    _record_check(
        checks,
        f"{prefix}/stage_checkpoints_present",
        all(name in loaded_checkpoints for name in ("initial", "after_plus", "after_minus")),
        f"available stages={sorted(name for name in loaded_checkpoints if name in STAGES)}",
    )
    after_plus_payload = loaded_checkpoints.get("after_plus")
    after_minus_payload = loaded_checkpoints.get("after_minus")
    metrics = _run_metrics(
        condition,
        summary_record,
        evaluations,
        steps,
        slot_map,
        after_plus_payload,
        after_minus_payload,
    )

    if condition == "frozen_conflict" and after_plus_payload and after_minus_payload:
        _local_ap, shared_ap, _groups_ap = _decoded_actor(after_plus_payload)
        _local_am, shared_am, _groups_am = _decoded_actor(after_minus_payload)
        _record_check(
            checks,
            f"{prefix}/frozen_shared_parameters_unchanged",
            np.array_equal(shared_ap, shared_am),
            f"max absolute shared delta={float(np.max(np.abs(shared_am - shared_ap)))}",
        )

    minus_local_ok = False
    if after_plus_payload:
        local_ap, _shared_ap, _groups_ap = _decoded_actor(after_plus_payload)
        minus_slot = int(slot_map["minus"])
        minus_local_ok = bool(np.count_nonzero(local_ap[minus_slot]) == 0)
    _record_check(
        checks,
        f"{prefix}/minus_local_untrained_after_plus",
        minus_local_ok,
        f"maximum absolute minus local coefficient={metrics['minus_local_max_abs_after_plus']}",
    )

    source_costs = {
        "probe_interactions": int(costs["probe_interactions"]),
        "training_interactions": int(costs["ppo_fit_and_holdout_interactions"]),
        "evaluation_interactions": int(costs["evaluation_interactions"]),
        "total_interactions": int(costs["total_environment_interactions"]),
        "wall_seconds_descriptive": float(costs["wall_seconds"]),
        "cpu_seconds_descriptive": float(costs["cpu_seconds"]),
        "rss_bytes_at_end_descriptive": int(costs["rss_bytes_at_end"]),
        "sampled_peak_rss_bytes_descriptive": int(costs["sampled_peak_rss_bytes"]),
    }
    return {
        "seed": seed,
        "condition": condition,
        "run_dir": str(run_dir),
        "counts": {
            "probe": probe_count,
            "training": training_count,
            "evaluation": evaluation_count,
            "total": interaction_count,
            "fit": len(fit_rows),
            "holdout": len(holdout_rows),
        },
        "costs": source_costs,
        "metrics": metrics,
        "evaluation_metrics": evaluation_metrics,
        "capacity": capacity_records,
        "source_snapshot_file_count": len(manifest.get("sources_sha256", {})),
        "_raw": {
            "probes": probes,
            "steps": steps,
            "episodes": episodes,
            "evaluations": evaluations,
            "slot_map": slot_map,
            "checkpoints": loaded_checkpoints,
            "config": config,
        },
    }


def _audit_cross_runs(
    runs: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> None:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_seed.setdefault(int(run["seed"]), {})[str(run["condition"])] = run

    for seed, conditions in by_seed.items():
        prefix = str(seed)
        if set(conditions) != set(CONDITIONS):
            _record_check(
                checks,
                f"{prefix}/condition_coverage",
                False,
                f"found={sorted(conditions)} expected={list(CONDITIONS)}",
            )
            continue
        reference = conditions[CONDITIONS[0]]["_raw"]

        # Shared rollout plans: mode label changes only the reward sign.
        probes_match = True
        for condition in CONDITIONS[1:]:
            other = conditions[condition]["_raw"]["probes"]
            for mode in MODES:
                a = [row for row in reference["probes"] if row["mode"] == mode]
                b = [row for row in other if row["mode"] == mode]
                probes_match &= _same_fields(
                    a, b, ("step", "obs", "action", "next_obs", "terminated", "truncated")
                )[0]
        plus_probe = [row for row in reference["probes"] if row["mode"] == "plus"]
        minus_probe = [row for row in reference["probes"] if row["mode"] == "minus"]
        reward_sign_match = len(plus_probe) == len(minus_probe) and all(
            float(minus["reward"]) == -float(plus["reward"])
            for plus, minus in zip(plus_probe, minus_probe)
        )
        _record_check(
            checks,
            f"{prefix}/paired_probe_dynamics_and_opposite_rewards",
            probes_match and reward_sign_match,
            f"all-condition dynamics/actions match={probes_match}; exact reward sign reversal={reward_sign_match}",
        )

        # The evaluation plans are common across mode, stage, and condition.
        eval_plans: dict[tuple[str, str, int], tuple[str, str]] = {}
        eval_plan_match = True
        for condition, run in conditions.items():
            for row in run["_raw"]["evaluations"]:
                key = (row["stage"], row["mode"], int(row["episode"]))
                value = (row["reset_seed"], row["action_seed"])
                if key in eval_plans and eval_plans[key] != value:
                    eval_plan_match = False
                else:
                    eval_plans[key] = value
        _record_check(
            checks,
            f"{prefix}/paired_evaluation_plans_across_conditions",
            eval_plan_match,
            f"matched stage/mode/episode reset and action seeds={eval_plan_match}",
        )

        # Shared-active and frozen arms see identical plus data and policy state.
        active = conditions["shared_conflict"]["_raw"]
        frozen = conditions["frozen_conflict"]["_raw"]
        active_plus = [row for row in active["steps"] if row["mode"] == "plus"]
        frozen_plus = [row for row in frozen["steps"] if row["mode"] == "plus"]
        training_prefix_match, prefix_details = _same_fields(
            active_plus,
            frozen_plus,
            ("step", "action", "reward", "log_prob", "obs", "next_obs"),
        )
        active_ap = active["checkpoints"].get("after_plus")
        frozen_ap = frozen["checkpoints"].get("after_plus")
        checkpoint_match = bool(
            active_ap
            and frozen_ap
            and active_ap["actor_binary"] == frozen_ap["actor_binary"]
            and _nested_state_equal(active_ap["critic"], frozen_ap["critic"])
        )
        _record_check(
            checks,
            f"{prefix}/active_frozen_plus_prefix_and_after_plus_state",
            training_prefix_match and checkpoint_match,
            f"plus transition prefix: {prefix_details}; after_plus actor+critic exact={checkpoint_match}",
        )

        # Original plus local state cannot receive gradients in the minus bout.
        local_checks = True
        independent = conditions["independent_conflict"]["_raw"]
        for condition, run in conditions.items():
            plus_slot = int(run["_raw"]["slot_map"]["plus"])
            after_plus = run["_raw"]["checkpoints"].get("after_plus")
            after_minus = run["_raw"]["checkpoints"].get("after_minus")
            if not after_plus or not after_minus:
                local_checks = False
                continue
            local_ap, _shared_ap, _groups_ap = _decoded_actor(after_plus)
            local_am, _shared_am, _groups_am = _decoded_actor(after_minus)
            local_checks &= np.array_equal(local_ap[plus_slot], local_am[plus_slot])
        _record_check(
            checks,
            f"{prefix}/plus_local_parameters_unchanged_during_minus",
            local_checks,
            "compared decoded plus local row in after_plus and after_minus for all conditions",
        )

        frozen_run = conditions["frozen_conflict"]["_raw"]
        frozen_actions_unchanged = _same_eval_actions(
            frozen_run["evaluations"], "plus", "after_plus", "after_minus"
        )
        independent_actions_unchanged = _same_eval_actions(
            independent["evaluations"], "plus", "after_plus", "after_minus"
        )
        _record_check(
            checks,
            f"{prefix}/frozen_and_independent_plus_actions_unchanged",
            frozen_actions_unchanged and independent_actions_unchanged,
            f"frozen={frozen_actions_unchanged}; independent={independent_actions_unchanged}",
        )


def _same_eval_actions(
    rows: list[Mapping[str, str]], mode: str, earlier: str, later: str
) -> bool:
    first = {
        int(row["episode"]): row["actions"]
        for row in rows
        if row["mode"] == mode and row["stage"] == earlier
    }
    second = {
        int(row["episode"]): row["actions"]
        for row in rows
        if row["mode"] == mode and row["stage"] == later
    }
    return bool(first) and first == second


def _aggregate(runs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITIONS}
    for run in runs:
        by_condition[str(run["condition"])].append(run)
    means: dict[str, Any] = {}
    for condition, condition_runs in by_condition.items():
        means[condition] = {
            metric: _mean(
                run["metrics"][metric]
                for run in condition_runs
                if run["metrics"].get(metric) is not None
            )
            for metric in METRIC_KEYS
        }

    paired: dict[str, Any] = {}
    for baseline in ("independent_conflict", "frozen_conflict"):
        contrast_key = f"shared_conflict_minus_{baseline}"
        paired[contrast_key] = {}
        for metric in METRIC_KEYS:
            rows: list[dict[str, Any]] = []
            baseline_by_seed = {int(run["seed"]): run for run in by_condition[baseline]}
            active_by_seed = {int(run["seed"]): run for run in by_condition["shared_conflict"]}
            for seed in sorted(set(baseline_by_seed) & set(active_by_seed)):
                left = active_by_seed[seed]["metrics"].get(metric)
                right = baseline_by_seed[seed]["metrics"].get(metric)
                if left is None or right is None:
                    continue
                rows.append({
                    "seed": seed,
                    "active": float(left),
                    "baseline": float(right),
                    "active_minus_baseline": float(left) - float(right),
                })
            paired[contrast_key][metric] = {
                "per_seed": rows,
                "mean_paired_difference": _mean(
                    row["active_minus_baseline"] for row in rows
                ),
            }
    return means, paired


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(output_dir: Path, runs: list[dict[str, Any]], means: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "independent_conflict": "#64748b",
        "shared_conflict": "#2563eb",
        "frozen_conflict": "#d97706",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    x = np.arange(len(CONDITIONS), dtype=np.float64)
    width = 0.34
    for offset, stage, metric in (
        (-width / 2, "after_plus", "minus_after_plus_signed_return"),
        (width / 2, "after_minus", "minus_after_minus_signed_return"),
    ):
        values = [means[condition][metric] for condition in CONDITIONS]
        axes[0].bar(
            x + offset,
            values,
            width,
            label=stage.replace("_", " "),
            color=[colors[condition] for condition in CONDITIONS],
            alpha=0.68 if stage == "after_plus" else 1.0,
            edgecolor="white",
        )
        for index, condition in enumerate(CONDITIONS):
            seed_values = [
                run["metrics"][metric]
                for run in runs
                if run["condition"] == condition and run["metrics"].get(metric) is not None
            ]
            if seed_values:
                jitter = np.linspace(-0.045, 0.045, len(seed_values))
                axes[0].scatter(
                    np.full(len(seed_values), x[index] + offset) + jitter,
                    seed_values,
                    s=18,
                    color="#111827",
                    zorder=3,
                )
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_xticks(x, [name.replace("_conflict", "") for name in CONDITIONS])
    axes[0].set_ylabel("Minus route mean signed return")
    axes[0].set_title("Minus route: signed reward")
    axes[0].legend(frameon=False)

    retention_values = [means[name]["plus_return_retention_delta"] for name in CONDITIONS]
    axes[1].bar(
        x,
        retention_values,
        color=[colors[name] for name in CONDITIONS],
        edgecolor="white",
    )
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    for index, condition in enumerate(CONDITIONS):
        values = [
            run["metrics"]["plus_return_retention_delta"]
            for run in runs
            if run["condition"] == condition
        ]
        if values:
            jitter = np.linspace(-0.045, 0.045, len(values))
            axes[1].scatter(x[index] + jitter, values, s=18, color="#111827", zorder=3)
    axes[1].set_xticks(x, [name.replace("_conflict", "") for name in CONDITIONS])
    axes[1].set_ylabel("Plus return: after minus − after plus")
    axes[1].set_title("Retention on original plus objective")
    fig.suptitle("Same-dynamics reward conflict (descriptive; no significance claim)")
    fig.savefig(output_dir / "signed_reward_and_plus_retention.png", dpi=170)
    plt.close(fig)


def build_report(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    summary_path = input_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"input directory has no summary.json: {input_dir}")
    source_summary = _read_json(summary_path)
    summary_records = list(source_summary.get("records", []))
    checks: list[dict[str, Any]] = []
    expected_pairs = {
        (int(seed), str(condition))
        for seed in source_summary.get("seeds", [])
        for condition in source_summary.get("conditions", [])
    }
    actual_pairs = {(int(row["seed"]), str(row["condition"])) for row in summary_records}
    _record_check(
        checks,
        "root_summary_record_coverage",
        expected_pairs == actual_pairs and bool(actual_pairs),
        f"records={len(actual_pairs)}; missing={sorted(expected_pairs - actual_pairs)}; "
        f"unexpected={sorted(actual_pairs - expected_pairs)}",
    )
    _record_check(
        checks,
        "root_condition_set",
        tuple(source_summary.get("conditions", [])) == CONDITIONS,
        f"conditions={source_summary.get('conditions')}",
    )

    runs = [_audit_run(input_dir, row, checks) for row in summary_records]
    usable_runs = [run for run in runs if "metrics" in run]
    _audit_cross_runs(usable_runs, checks)

    aggregate_counts = {
        key: sum(int(run["counts"][name]) for run in usable_runs)
        for key, name in (
            ("probe_steps", "probe"),
            ("training_steps", "training"),
            ("evaluation_steps", "evaluation"),
            ("total_interactions", "total"),
        )
    }
    root_count_match = all(
        aggregate_counts[key] == int(source_summary.get(key, -1))
        for key in aggregate_counts
    )
    _record_check(
        checks,
        "root_aggregate_interaction_ledger",
        root_count_match,
        f"recomputed={aggregate_counts}; summary="
        f"{{'probe_steps': {source_summary.get('probe_steps')}, 'training_steps': {source_summary.get('training_steps')}, "
        f"'evaluation_steps': {source_summary.get('evaluation_steps')}, 'total_interactions': {source_summary.get('total_interactions')}}}",
    )

    condition_means, paired_contrasts = _aggregate(usable_runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_runs = [{key: value for key, value in run.items() if key != "_raw"} for run in usable_runs]
    failed = [
        {"name": row["name"], "details": row["details"]}
        for row in checks
        if not row["passed"]
    ]
    report = {
        "phase": source_summary.get("phase"),
        "input": str(input_dir),
        "conditions": list(CONDITIONS),
        "seeds": list(source_summary.get("seeds", [])),
        "interaction_totals": aggregate_counts,
        "per_seed_condition_metrics": public_runs,
        "condition_means": condition_means,
        "paired_contrasts": paired_contrasts,
        "audit": {
            "checks": checks,
            "failed_assertions": failed,
        },
        "interpretation_boundary": (
            "Minus goal_reached is environment termination metadata, not task success. "
            "Cold transfer (initial to after_plus) is reported separately from minus-stage adaptation "
            "(after_plus to after_minus). Results are descriptive; seeds are not treated as independent "
            "replications for significance claims."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    metric_rows = [
        {"seed": run["seed"], "condition": run["condition"], **run["metrics"]}
        for run in usable_runs
    ]
    _write_csv(
        output_dir / "per_seed_metrics.csv",
        metric_rows,
        ["seed", "condition", *METRIC_KEYS,
         "minus_local_max_abs_after_plus", "old_plus_local_unchanged_during_minus",
         "shared_parent_delta_l2_after_minus", "minus_goal_metric_label"],
    )
    contrast_rows: list[dict[str, Any]] = []
    for comparison, metrics in paired_contrasts.items():
        for metric, value in metrics.items():
            for row in value["per_seed"]:
                contrast_rows.append({
                    "comparison": comparison,
                    "metric": metric,
                    **row,
                })
    _write_csv(
        output_dir / "paired_contrasts.csv",
        contrast_rows,
        ["comparison", "metric", "seed", "active", "baseline", "active_minus_baseline"],
    )
    storage_rows = [
        {"seed": run["seed"], "condition": run["condition"], **row}
        for run in usable_runs
        for row in run["capacity"]
    ]
    _write_csv(
        output_dir / "capacity_audit.csv",
        storage_rows,
        ["seed", "condition", "step", "checkpoint", "actor_bytes", "checkpoint_bytes",
         "foundation_bytes", "whole_package_bytes", "actor_cap_bytes", "whole_package_cap_bytes", "passed"],
    )
    _plot_results(output_dir, usable_runs, condition_means)
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Same-dynamics reward-conflict report",
        "",
        f"Phase: **{report.get('phase')}**. Seeds: {', '.join(map(str, report.get('seeds', [])))}.",
        "",
        "Minus-route goal completion is environment termination metadata, not task success. "
        "Cold transfer (initial → after plus) is separate from adaptation during minus (after plus → after minus).",
        "",
        "## Per-condition means",
        "",
        "| Condition | Minus return after plus | Minus return after minus | Cold transfer | Minus adaptation gain | Plus return retention delta | Plus goal success after plus | Plus goal success after minus |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = report["condition_means"].get(condition, {})
        cells = [
            condition,
            values.get("minus_after_plus_signed_return"),
            values.get("minus_after_minus_signed_return"),
            values.get("minus_cold_transfer_signed_return"),
            values.get("minus_adaptation_gain_after_plus"),
            values.get("plus_return_retention_delta"),
            values.get("plus_after_plus_goal_success"),
            values.get("plus_after_minus_goal_success"),
        ]
        lines.append("| " + " | ".join(
            str(value) if isinstance(value, str) else ("—" if value is None else f"{float(value):.4f}")
            for value in cells
        ) + " |")
    lines.extend([
        "",
        "## Paired contrasts",
        "",
        "Each value is shared active minus the named baseline, paired within seed. These are descriptive contrasts; no significance claim is made.",
        "",
    ])
    for comparison, metric_map in report["paired_contrasts"].items():
        lines.append(f"### {comparison}")
        lines.append("")
        lines.append("| Metric | Mean paired difference | Per-seed differences |")
        lines.append("|---|---:|---|")
        for metric, value in metric_map.items():
            diffs = ", ".join(
                f"{row['seed']}: {row['active_minus_baseline']:.4f}"
                for row in value["per_seed"]
            )
            mean = value["mean_paired_difference"]
            lines.append(f"| {metric} | {'—' if mean is None else f'{mean:.4f}'} | {diffs} |")
        lines.append("")
    audit = report["audit"]
    lines.extend([
        "## Audit",
        "",
        f"{len(audit['checks']) - len(audit['failed_assertions'])}/{len(audit['checks'])} checks passed; "
        f"{len(audit['failed_assertions'])} failed assertions.",
        "",
        "Interaction totals: " + ", ".join(
            f"{key}={value}" for key, value in report["interaction_totals"].items()
        ) + ".",
        "",
        "CPU, wall, and memory timings are retained per run in report.json as descriptive costs.",
        "",
        "![Signed reward and plus retention](signed_reward_and_plus_retention.png)",
        "",
    ])
    if audit["failed_assertions"]:
        lines.append("### Failed assertions")
        lines.append("")
        for failure in audit["failed_assertions"]:
            lines.append(f"- `{failure['name']}`: {failure['details']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="run output directory containing summary.json")
    parser.add_argument("--output", required=True, type=Path, help="directory for report artifacts")
    args = parser.parse_args(argv)
    report = build_report(args.input, args.output)
    print(json.dumps({
        "phase": report["phase"],
        "output": str(args.output.resolve()),
        "checks": len(report["audit"]["checks"]),
        "failed_assertions": len(report["audit"]["failed_assertions"]),
    }, ensure_ascii=False))
    return 0 if not report["audit"]["failed_assertions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

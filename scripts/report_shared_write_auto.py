"""Summarize and audit completed S1b cohorts without rerunning interactions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import report_shared_write as common
from scripts import run_knowledge_loop as loop


CONDITIONS = ("independent", "oracle_active", "auto_active", "auto_frozen")
CONDITION_LABELS = {
    "independent": "Independent local",
    "oracle_active": "Oracle active parent",
    "auto_active": "Automatic active parent",
    "auto_frozen": "Automatic frozen parent",
}
CONTRASTS = {
    "auto_active_minus_independent": "independent",
    "auto_active_minus_auto_frozen": "auto_frozen",
    "auto_active_minus_oracle_active": "oracle_active",
}
PROBE_FIELDS = (
    "context", "slot", "episode", "reset_seed", "episode_action_seed",
    "episode_action_index", "obs", "action", "reward", "next_obs",
    "terminated", "truncated", "profile_counted", "interaction_cost",
)


def read_json(path: Path) -> Any:
    return common.read_json(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    return common.read_csv(path)


def mean(values: list[float]) -> float | None:
    return common.mean(values)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_pair(value: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_return": float(value["mean_return"]),
        "success_rate": float(value.get("success_rate", value.get("mean_success"))),
    }


def _context_average(evaluations: dict[str, Any], stage: str, contexts: list[str]) -> dict[str, Any]:
    return common.average_contexts(evaluations, stage, contexts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grouping_summary(summary: dict[str, Any]) -> dict[str, Any]:
    score = summary["group_accuracy"]
    context_count = int(score["context_count"])
    assigned = int(score["assigned_contexts"])
    decisions = summary.get("group_decisions", [])
    return {
        "assignment_count": len(decisions),
        "context_count": context_count,
        "assigned_contexts": assigned,
        "unresolved_contexts": max(0, context_count - assigned),
        "coverage": assigned / context_count if context_count else None,
        "correct_pairs": int(score["correct_pairs"]),
        "total_pairs": int(score["total_pairs"]),
        "pairwise_family_agreement_audit_only": _finite(score.get("pairwise_accuracy")),
        "pairwise_metric_note": "Runner pairwise accuracy; unresolved assignments count as not-same. Family labels are used only for this post-hoc audit.",
        "decisions": [
            {
                "context": row["context"],
                "slot": int(row["slot"]),
                "assigned_group": int(row["assigned_group"]),
                "reason": row["reason"],
                "observer_family_audit_only": row.get("observer_family_audit_only"),
            }
            for row in decisions
        ],
    }


def _seed_record(seed_dir: Path, condition: str, config: dict[str, Any]) -> dict[str, Any]:
    folder = seed_dir / condition
    summary = read_json(folder / "summary.json")
    seen_contexts = list(dict.fromkeys(config["seen_sequence"]))
    heldout_contexts = list(dict.fromkeys(config["heldout_sequence"]))
    all_contexts = list(dict.fromkeys([*seen_contexts, *heldout_contexts]))
    boot = _context_average(summary["evaluations"], "boot", heldout_contexts)
    seen = _context_average(summary["evaluations"], "seen", heldout_contexts)
    adapted_all = _context_average(summary["evaluations"], "adapted", all_contexts)
    adapted_old = _context_average(summary["evaluations"], "adapted", seen_contexts)
    episodes = [row for row in read_csv(folder / "episodes.csv") if row["completed"].lower() == "true"]
    successes = sum(int(row["success"]) for row in episodes)
    training_steps = int(summary["training_steps"])
    probe_steps = int(summary["probe_steps"])
    evaluation_steps = int(summary["evaluation_steps"])
    return {
        "seed": int(summary["seed"]),
        "condition": condition,
        "online": {
            "training_reward": float(summary["total_reward"]),
            "reward_per_training_step": float(summary["reward_per_step"]),
            "success_rate": float(summary["online_success"]),
            "completed_episodes": len(episodes),
            "successes": successes,
        },
        "evaluation": {
            "heldout_probe_mediated_boot": boot,
            "heldout_probe_mediated_seen": seen,
            "heldout_seen_minus_boot": {
                "mean_return": (
                    float(seen["mean_return"]) - float(boot["mean_return"])
                    if seen["mean_return"] is not None and boot["mean_return"] is not None else None
                ),
                "success_rate": (
                    float(seen["success_rate"]) - float(boot["success_rate"])
                    if seen["success_rate"] is not None and boot["success_rate"] is not None else None
                ),
            },
            "adapted_all_contexts": adapted_all,
            "adapted_seen_contexts": adapted_old,
        },
        "grouping_audit_only": _grouping_summary(summary),
        "cost": {
            "training_steps": training_steps,
            "probe_steps": probe_steps,
            "evaluation_steps": evaluation_steps,
            "total_interactions": int(summary["total_interactions"]),
            "fit_steps": int(summary["counts"]["fit_steps"]),
            "holdout_steps": int(summary["counts"]["holdout_steps"]),
            "minibatch_updates": int(summary["counts"]["minibatch_updates"]),
            "wall_seconds": _finite(summary.get("wall_s")),
            "cpu_seconds": _finite(summary.get("cpu_s")),
        },
        "capacity_samples": common.capacity_summary(folder),
        "endpoint_sizes": summary.get("sizes", {}),
        "_folder": folder,
    }


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "_folder"}


def _path(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = value[key]
    return value


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "online_reward_per_training_step": ("online", "reward_per_training_step"),
        "online_success_rate": ("online", "success_rate"),
        "probe_mediated_boot_return": ("evaluation", "heldout_probe_mediated_boot", "mean_return"),
        "probe_mediated_boot_success_rate": ("evaluation", "heldout_probe_mediated_boot", "success_rate"),
        "probe_mediated_seen_return": ("evaluation", "heldout_probe_mediated_seen", "mean_return"),
        "probe_mediated_seen_success_rate": ("evaluation", "heldout_probe_mediated_seen", "success_rate"),
        "probe_mediated_seen_minus_boot_return": ("evaluation", "heldout_seen_minus_boot", "mean_return"),
        "probe_mediated_seen_minus_boot_success_rate": ("evaluation", "heldout_seen_minus_boot", "success_rate"),
        "adapted_all_return": ("evaluation", "adapted_all_contexts", "mean_return"),
        "adapted_all_success_rate": ("evaluation", "adapted_all_contexts", "success_rate"),
        "adapted_seen_return": ("evaluation", "adapted_seen_contexts", "mean_return"),
        "adapted_seen_success_rate": ("evaluation", "adapted_seen_contexts", "success_rate"),
        "grouping_coverage": ("grouping_audit_only", "coverage"),
        "pairwise_family_agreement_audit_only": ("grouping_audit_only", "pairwise_family_agreement_audit_only"),
        "training_steps": ("cost", "training_steps"),
        "probe_steps": ("cost", "probe_steps"),
        "evaluation_steps": ("cost", "evaluation_steps"),
        "total_interactions": ("cost", "total_interactions"),
        "sampled_actor_bytes_mean": ("capacity_samples", "actor_bytes", "mean"),
        "sampled_actor_bytes_peak": ("capacity_samples", "actor_bytes", "peak"),
        "sampled_whole_package_bytes_mean": ("capacity_samples", "whole_package_bytes", "mean"),
        "sampled_whole_package_bytes_peak": ("capacity_samples", "whole_package_bytes", "peak"),
    }
    result: dict[str, Any] = {
        "seed_count": len(records),
        "seeds": [record["seed"] for record in sorted(records, key=lambda row: row["seed"])],
        "metrics": {},
        "per_seed": [_public(record) for record in sorted(records, key=lambda row: row["seed"])],
        "pooled_online": {},
        "pooled_cost": {},
    }
    for name, keys in paths.items():
        values = [_path(record, keys) for record in records]
        valid = [float(value) for value in values if value is not None]
        result["metrics"][name] = {
            "mean_across_seeds": mean(valid),
            "per_seed": [
                {"seed": record["seed"], "value": _path(record, keys)}
                for record in sorted(records, key=lambda row: row["seed"])
            ],
        }
    training_steps = sum(record["cost"]["training_steps"] for record in records)
    probe_steps = sum(record["cost"]["probe_steps"] for record in records)
    evaluation_steps = sum(record["cost"]["evaluation_steps"] for record in records)
    reward = sum(record["online"]["training_reward"] for record in records)
    successes = sum(record["online"]["successes"] for record in records)
    episodes = sum(record["online"]["completed_episodes"] for record in records)
    result["pooled_online"] = {
        "training_reward": reward,
        "training_steps": training_steps,
        "reward_per_training_step": reward / training_steps if training_steps else None,
        "successes": successes,
        "completed_episodes": episodes,
        "success_rate": successes / episodes if episodes else None,
    }
    result["pooled_cost"] = {
        "training_steps": training_steps,
        "probe_steps": probe_steps,
        "evaluation_steps": evaluation_steps,
        "total_interactions": training_steps + probe_steps + evaluation_steps,
    }
    return result


def _paired_contrasts(by_condition_seed: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    paths = {
        "online_reward_per_training_step": ("online", "reward_per_training_step"),
        "online_success_rate": ("online", "success_rate"),
        "probe_mediated_boot_return": ("evaluation", "heldout_probe_mediated_boot", "mean_return"),
        "probe_mediated_boot_success_rate": ("evaluation", "heldout_probe_mediated_boot", "success_rate"),
        "probe_mediated_seen_return": ("evaluation", "heldout_probe_mediated_seen", "mean_return"),
        "probe_mediated_seen_success_rate": ("evaluation", "heldout_probe_mediated_seen", "success_rate"),
        "probe_mediated_seen_minus_boot_return": ("evaluation", "heldout_seen_minus_boot", "mean_return"),
        "probe_mediated_seen_minus_boot_success_rate": ("evaluation", "heldout_seen_minus_boot", "success_rate"),
        "adapted_all_return": ("evaluation", "adapted_all_contexts", "mean_return"),
        "adapted_all_success_rate": ("evaluation", "adapted_all_contexts", "success_rate"),
        "adapted_seen_return": ("evaluation", "adapted_seen_contexts", "mean_return"),
        "adapted_seen_success_rate": ("evaluation", "adapted_seen_contexts", "success_rate"),
        "total_interactions": ("cost", "total_interactions"),
    }
    result: dict[str, Any] = {
        "interpretation": "Descriptive means and paired per-seed differences only; no formal statistical claims.",
        "comparisons": {},
    }
    active = by_condition_seed.get("auto_active", {})
    for name, other_condition in CONTRASTS.items():
        other = by_condition_seed.get(other_condition, {})
        seeds = sorted(set(active) & set(other))
        metrics: dict[str, Any] = {}
        for metric, path in paths.items():
            paired = []
            for seed in seeds:
                left = _path(active[seed], path)
                right = _path(other[seed], path)
                paired.append({"seed": seed, "auto_active": left, other_condition: right,
                               "difference": float(left) - float(right)})
            metrics[metric] = {
                "mean_difference": mean([row["difference"] for row in paired]),
                "paired_seed_count": len(paired),
                "per_seed": paired,
            }
        result["comparisons"][name] = {
            "reference_condition": "auto_active",
            "other_condition": other_condition,
            "paired_seed_count": len(seeds),
            "metrics": metrics,
        }
    return result


def _check(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def _raw_probe_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, list[tuple[str, ...]]]]:
    rows = read_csv(path)
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for row in rows:
        grouped.setdefault(row["context"], []).append(tuple(row[field] for field in PROBE_FIELDS))
    return rows, grouped


def _stage_episodes(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in read_csv(path):
        key = (row["stage"], row["context"])
        grouped.setdefault(key, []).append({
            "episode": int(row["episode"]),
            "reset_seed": int(row["reset_seed"]),
            "action_seed": int(row["action_seed"]),
            "actions": json.loads(row["actions"]),
            "return": float(row["return"]),
            "success": row["success"].lower() in ("1", "true"),
            "length": int(row["length"]),
        })
    for episodes in grouped.values():
        episodes.sort(key=lambda row: row["episode"])
    return grouped


def _same_decisions_ignoring_condition(left_folder: Path, right_folder: Path) -> dict[str, Any]:
    def normalized(path: Path) -> list[dict[str, str]]:
        rows = read_csv(path)
        return [{key: value for key, value in row.items() if key != "condition"} for row in rows]

    decisions_left = normalized(left_folder / "group_decisions.csv")
    decisions_right = normalized(right_folder / "group_decisions.csv")
    assignments_left = normalized(left_folder / "group_assignments.csv")
    assignments_right = normalized(right_folder / "group_assignments.csv")
    return {
        "decisions_match_ignoring_condition_label": decisions_left == decisions_right,
        "assignments_match_ignoring_condition_label": assignments_left == assignments_right,
        "decision_rows": len(decisions_left),
        "assignment_rows": len(assignments_left),
        "condition_label_excluded": True,
    }


def _verify_source_snapshot(folder: Path) -> dict[str, Any]:
    manifest = read_json(folder / "source_manifest.json")
    entries = manifest.get("sources", {})
    checked, missing, mismatched = 0, [], []
    for relative, expected in entries.items():
        captured = folder / "source_snapshot" / Path(relative)
        if not captured.is_file():
            missing.append(relative)
            continue
        checked += 1
        actual = _sha256(captured)
        if actual != expected:
            mismatched.append({"path": relative, "expected": expected, "actual": actual})
    return {
        "manifest_source_count": len(entries),
        "verified_captured_sources": checked,
        "missing_captured_sources": missing,
        "mismatched_captured_sources": mismatched,
        "foundation_sha256_recorded": manifest.get("foundation_sha256"),
        "dataset_sha256_recorded": manifest.get("dataset_sha256"),
        "comparison_basis": "Captured source_snapshot files are hashed against source_manifest.json; current runner/source files are not consulted.",
    }


def run_audits(
    input_root: Path,
    cohort_summary: dict[str, Any],
    config: dict[str, Any],
    seed_dirs: list[Path],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    seen_sequence = list(config["seen_sequence"])
    heldout = list(dict.fromkeys(config["heldout_sequence"]))
    all_contexts = list(dict.fromkeys([*seen_sequence, *config["heldout_sequence"]]))
    first_seen_bouts = min(8, len(seen_sequence))
    bout_steps = int(config["bout_steps"])
    per_context_probe_steps = int(config["probe_steps_per_context"])
    expected_probe_steps = per_context_probe_steps * len(all_contexts)
    actor_cap = int(config["actor_budget_bytes"])
    package_cap = int(config["whole_package_cap_bytes"])
    root_preboot = {int(row["seed"]): row for row in cohort_summary.get("preboot_auto_audits", [])}

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_", 1)[1])
        folders = {condition: seed_dir / condition for condition in CONDITIONS}
        summaries = {condition: read_json(folders[condition] / "summary.json") for condition in CONDITIONS}
        slot_maps = {condition: read_json(folders[condition] / "external_slot_map.json") for condition in CONDITIONS}
        slot_map_match = all(slot_maps[condition] == slot_maps[CONDITIONS[0]] for condition in CONDITIONS)
        _check(checks, f"seed_{seed}_same_slot_map_all_conditions", slot_map_match, {"slot_maps": slot_maps})

        # The fixed probe is an observed interaction: exactly 512 (or configured) steps per context.
        probe_by_condition = {condition: _raw_probe_rows(folders[condition] / "probe_steps.csv") for condition in CONDITIONS}
        probe_counts: dict[str, Any] = {}
        probe_valid = True
        for condition, (rows, grouped) in probe_by_condition.items():
            per_context = {context: len(grouped.get(context, [])) for context in all_contexts}
            row_cost = sum(int(row["interaction_cost"]) for row in rows)
            count_ok = len(rows) == expected_probe_steps and all(
                per_context[context] == per_context_probe_steps for context in all_contexts
            )
            summary_count_ok = int(summaries[condition]["probe_steps"]) == expected_probe_steps and int(
                summaries[condition]["counts"]["probe_steps"]
            ) == expected_probe_steps
            cost_ok = row_cost == expected_probe_steps and all(int(row["interaction_cost"]) == 1 for row in rows)
            probe_valid = probe_valid and count_ok and summary_count_ok and cost_ok
            probe_counts[condition] = {
                "raw_rows": len(rows),
                "per_context_rows": per_context,
                "row_interaction_cost_sum": row_cost,
                "summary_probe_steps": int(summaries[condition]["probe_steps"]),
                "summary_counts_probe_steps": int(summaries[condition]["counts"]["probe_steps"]),
                "count_ok": count_ok,
                "summary_count_ok": summary_count_ok,
                "cost_ok": cost_ok,
            }
        probe_reference = probe_by_condition[CONDITIONS[0]][1]
        probe_matches = all(
            probe_by_condition[condition][1] == probe_reference for condition in CONDITIONS[1:]
        )
        _check(checks, f"seed_{seed}_probe_count_and_interaction_cost", probe_valid,
               {"expected_probe_steps": expected_probe_steps,
                "configured_steps_per_context": per_context_probe_steps,
                "contexts": all_contexts, "conditions": probe_counts})
        _check(checks, f"seed_{seed}_probe_raw_observation_action_reward_next_observation_exact_all_conditions",
               probe_matches,
               {"conditions": list(CONDITIONS), "fields_compared": list(PROBE_FIELDS),
                "rows_per_context": {context: len(probe_reference.get(context, [])) for context in all_contexts},
                "note": "Raw 4D obs/next_obs, action, reward, seeds, episode indices, flags, and cost strings were compared; family labels are excluded."})

        # The first seen contexts never include held-out A2/B2; unknown group assignment is permitted.
        for condition in CONDITIONS:
            rows = read_csv(folders[condition] / "steps.csv")
            prefix = [row for row in rows if int(row["bout"]) < first_seen_bouts]
            by_bout: dict[int, list[dict[str, str]]] = {}
            for row in prefix:
                by_bout.setdefault(int(row["bout"]), []).append(row)
            no_heldout = not any(row["context"] in heldout for row in prefix)
            duration_ok = all(len(by_bout.get(index, [])) == bout_steps for index in range(first_seen_bouts))
            context_ok = all(
                by_bout.get(index) and all(row["context"] == seen_sequence[index] for row in by_bout[index])
                for index in range(first_seen_bouts)
            )
            _check(checks, f"seed_{seed}_{condition}_heldout_not_trained_first_seen_bouts",
                   no_heldout and duration_ok and context_ok,
                   {"checked_bouts": first_seen_bouts, "expected_steps_per_bout": bout_steps,
                    "heldout_contexts": heldout, "heldout_steps_in_prefix": sum(
                        row["context"] in heldout for row in prefix),
                    "bout_contexts": [
                        by_bout[index][0]["context"] if by_bout.get(index) else None
                        for index in range(first_seen_bouts)
                    ],
                    "bout_step_counts": {str(index): len(by_bout.get(index, [])) for index in range(first_seen_bouts)},
                    "group_resolution_is_not_a_check": True})

            # One initial decision must exist for every context; unresolved slots remain valid outcomes.
            grouping = _grouping_summary(summaries[condition])
            decision_contexts = [row["context"] for row in grouping["decisions"]]
            decisions_complete = len(decision_contexts) == len(all_contexts) and set(decision_contexts) == set(all_contexts)
            _check(checks, f"seed_{seed}_{condition}_one_group_decision_per_context", decisions_complete,
                   {"decision_contexts": decision_contexts,
                    "unresolved_contexts": grouping["unresolved_contexts"],
                    "coverage": grouping["coverage"],
                    "unresolved_assignments_are_not_failures": True})

        # The root and per-seed repaired audit are evidence; the condition field is not an algorithm input.
        preboot_file = read_json(seed_dir / "preboot_auto_audit.json")
        preboot_root = root_preboot.get(seed, {})
        preboot_fields = (
            "actor_payload_match", "critic_match", "slot_groups_match", "group_decisions_match",
            "training_action_reward_prefix_match", "source_and_heldout_probe_prefix_match",
            "passed",
        )
        preboot_consistent = all(preboot_file.get(field) == preboot_root.get(field) for field in preboot_fields)
        preboot_passed = preboot_file.get("passed") is True and preboot_root.get("passed") is True

        active_boot = (folders["auto_active"] / "actor_boot.bin").read_bytes()
        frozen_boot = (folders["auto_frozen"] / "actor_boot.bin").read_bytes()
        active_local, active_shared, active_groups = common.decode_shared_knowledge(active_boot)
        frozen_local, frozen_shared, frozen_groups = common.decode_shared_knowledge(frozen_boot)
        actor_boot_exact = active_boot == frozen_boot
        decoded_boot_equal = (
            np.array_equal(active_local, frozen_local)
            and np.array_equal(active_shared, frozen_shared)
            and np.array_equal(active_groups, frozen_groups)
        )
        decision_comparison = _same_decisions_ignoring_condition(folders["auto_active"], folders["auto_frozen"])
        decisions_same = (decision_comparison["decisions_match_ignoring_condition_label"]
                          and decision_comparison["assignments_match_ignoring_condition_label"])
        _check(checks, f"seed_{seed}_auto_preboot_audit_and_actor_payloads_match", 
               preboot_passed and preboot_consistent and actor_boot_exact and decoded_boot_equal and decisions_same,
               {"root_preboot_audit": preboot_root, "seed_preboot_audit": preboot_file,
                "preboot_audits_consistent": preboot_consistent,
                "actor_boot_binary_exact": actor_boot_exact, "decoded_actor_matrices_and_groups_equal": decoded_boot_equal,
                "normalized_group_decisions": decision_comparison,
                "condition_label_difference_ignored": True})

        # Held-out local rows must remain zero before the held-out training bouts begin.
        heldout_local_details: dict[str, Any] = {}
        all_zero = True
        for condition in CONDITIONS:
            slot_map = slot_maps[condition]
            heldout_local_details[condition] = {}
            for stage in ("boot", "seen"):
                local, shared, groups = common.decode_shared_knowledge(
                    (folders[condition] / f"actor_{stage}.bin").read_bytes()
                )
                stage_details = {}
                for context in heldout:
                    slot = int(slot_map[context])
                    is_zero = bool(np.all(local[slot] == 0.0))
                    all_zero = all_zero and is_zero
                    stage_details[context] = {"slot": slot, "zero": is_zero}
                heldout_local_details[condition][stage] = {
                    "rows": stage_details,
                    "shared_group_count": int(shared.shape[0]),
                    "assigned_groups": groups.tolist(),
                }
        _check(checks, f"seed_{seed}_heldout_local_rows_zero_at_boot_and_seen", all_zero,
               {"rows": heldout_local_details,
                "meaning": "Held-out local rows remain zero; shared parents may still provide probe-mediated behavior."})

        # After first probe assignment, frozen and independent held-out actors cannot change before Seen.
        action_checks: dict[str, Any] = {}
        action_checks_passed = True
        for condition in ("independent", "auto_frozen"):
            grouped = _stage_episodes(folders[condition] / "evaluations.csv")
            contexts_result: dict[str, Any] = {}
            for context in heldout:
                boot_rows = grouped.get(("boot", context), [])
                seen_rows = grouped.get(("seen", context), [])
                paired = []
                same = len(boot_rows) == len(seen_rows) and bool(boot_rows)
                for boot_row, seen_row in zip(boot_rows, seen_rows):
                    exact = (
                        boot_row["episode"] == seen_row["episode"]
                        and boot_row["reset_seed"] == seen_row["reset_seed"]
                        and boot_row["action_seed"] == seen_row["action_seed"]
                        and boot_row["actions"] == seen_row["actions"]
                        and boot_row["return"] == seen_row["return"]
                        and boot_row["success"] == seen_row["success"]
                    )
                    same = same and exact
                    paired.append({"boot": boot_row, "seen": seen_row, "exact_match": exact})
                contexts_result[context] = {"episode_count": len(paired), "exact_match": same, "pairs": paired}
                action_checks_passed = action_checks_passed and same
            action_checks[condition] = contexts_result
        _check(checks, f"seed_{seed}_independent_and_auto_frozen_heldout_boot_seen_actions_exact",
               action_checks_passed,
               {"heldout_contexts": action_checks,
                "compared": ["reset_seed", "action_seed", "actions", "return", "success"]})

        # Evaluation steps are measured from the persisted episode rows, not inferred from episode counts.
        eval_checks: dict[str, Any] = {}
        eval_checks_passed = True
        for condition in CONDITIONS:
            summary = summaries[condition]
            eval_rows = read_csv(folders[condition] / "evaluations.csv")
            actual_total = sum(int(row["length"]) for row in eval_rows)
            per_stage: dict[str, int] = {}
            per_context_stage: dict[str, int] = {}
            for row in eval_rows:
                key = row["stage"]
                per_stage[key] = per_stage.get(key, 0) + int(row["length"])
                context_key = f"{key}/{row['context']}"
                per_context_stage[context_key] = per_context_stage.get(context_key, 0) + int(row["length"])
            summary_stage: dict[str, int] = {}
            for stage, contexts in summary["evaluations"].items():
                for context, result in contexts.items():
                    summary_stage[f"{stage}/{context}"] = int(result["environment_steps"])
            stage_matches = all(per_context_stage.get(key) == value for key, value in summary_stage.items())
            total_matches = actual_total == int(summary["evaluation_steps"]) == sum(summary_stage.values())
            ok = stage_matches and total_matches
            eval_checks_passed = eval_checks_passed and ok
            eval_checks[condition] = {
                "csv_length_sum": actual_total,
                "summary_evaluation_steps": int(summary["evaluation_steps"]),
                "csv_stage_context_steps": per_context_stage,
                "summary_stage_context_steps": summary_stage,
                "stage_context_counts_match": stage_matches,
                "total_counts_match": total_matches,
            }
        _check(checks, f"seed_{seed}_evaluation_step_counts_match_csv_episode_lengths", eval_checks_passed,
               {"conditions": eval_checks})

        # Cost and capacity use the runner's measured counts and actual encoded artifacts.
        cost_capacity: dict[str, Any] = {}
        cost_capacity_passed = True
        base_config = loop.load_base_config()
        foundation_path = Path(base_config["foundation"])
        foundation_bytes = foundation_path.stat().st_size
        foundation_sha = _sha256(foundation_path)
        for condition in CONDITIONS:
            folder = folders[condition]
            summary = summaries[condition]
            training = int(summary["training_steps"])
            probe = int(summary["probe_steps"])
            evaluation = int(summary["evaluation_steps"])
            interaction_total = int(summary["total_interactions"])
            exact_cost = interaction_total == training + probe + evaluation
            rows = read_json(folder / "capacity.json")
            row_caps_ok = all(
                int(row["actor_bytes"]) <= actor_cap and int(row["whole_package_bytes"]) <= package_cap
                for row in rows
            )
            stage_files = {}
            stage_payloads_ok = True
            for stage in ("boot", "seen", "adapted", "final"):
                actor_path = folder / f"actor_{stage}.bin"
                payload = actor_path.read_bytes()
                local, shared, groups = common.decode_shared_knowledge(payload)
                stage_ok = len(payload) <= actor_cap
                stage_payloads_ok = stage_payloads_ok and stage_ok
                stage_files[stage] = {
                    "actor_file_bytes": len(payload),
                    "local_shape": list(local.shape),
                    "shared_shape": list(shared.shape),
                    "group_assignments": groups.tolist(),
                    "within_actor_cap": stage_ok,
                }
            final_checkpoint = folder / "checkpoint_final.pt"
            checkpoint_bytes = final_checkpoint.stat().st_size
            final_actor_matches = len((folder / "actor_final.bin").read_bytes()) == int(summary["sizes"]["actor_bytes"])
            final_checkpoint_matches = checkpoint_bytes == int(summary["sizes"]["checkpoint_bytes"])
            measured_package = checkpoint_bytes + foundation_bytes
            final_package_matches = measured_package == int(summary["sizes"]["whole_package_bytes"])
            package_ok = final_package_matches and measured_package <= package_cap
            manifest = read_json(folder / "source_manifest.json")
            foundation_hash_matches = foundation_sha == manifest.get("foundation_sha256")
            checkpoint = _load_checkpoint(final_checkpoint)
            checkpoint_hash_matches = checkpoint.get("foundation_sha256") == manifest.get("foundation_sha256")
            exact_cost = exact_cost and int(summary["counts"]["probe_steps"]) == probe
            ok = (exact_cost and row_caps_ok and stage_payloads_ok and final_actor_matches
                  and final_checkpoint_matches and package_ok and foundation_hash_matches
                  and checkpoint_hash_matches)
            cost_capacity_passed = cost_capacity_passed and ok
            cost_capacity[condition] = {
                "training_steps": training,
                "probe_steps": probe,
                "evaluation_steps": evaluation,
                "sum_training_probe_evaluation": training + probe + evaluation,
                "summary_total_interactions": interaction_total,
                "exact_interaction_ledger": exact_cost,
                "periodic_capacity_sample_count": len(rows),
                "periodic_rows_within_actor_and_package_caps": row_caps_ok,
                "actor_cap_bytes": actor_cap,
                "whole_package_cap_bytes": package_cap,
                "foundation_file_bytes": foundation_bytes,
                "foundation_hash_matches_manifest": foundation_hash_matches,
                "checkpoint_foundation_hash_matches_manifest": checkpoint_hash_matches,
                "final_checkpoint_file_bytes": checkpoint_bytes,
                "final_package_file_bytes": measured_package,
                "final_actor_size_matches_summary": final_actor_matches,
                "final_checkpoint_size_matches_summary": final_checkpoint_matches,
                "final_package_size_matches_summary": final_package_matches,
                "actor_artifacts": stage_files,
            }
        _check(checks, f"seed_{seed}_interaction_ledger_actor_sizes_and_package_caps", cost_capacity_passed,
               {"conditions": cost_capacity})

        # Verify the captured sources against their own manifest, never the mutable current source tree.
        source_results = {condition: _verify_source_snapshot(folders[condition]) for condition in CONDITIONS}
        source_hashes_passed = all(
            not result["missing_captured_sources"] and not result["mismatched_captured_sources"]
            for result in source_results.values()
        )
        _check(checks, f"seed_{seed}_captured_source_hashes_match_manifest", source_hashes_passed,
               {"conditions": source_results,
                "current_source_comparison_performed": False})

    repair_path = input_root / "POSTPROCESSING_FAILURE.json"
    if repair_path.is_file():
        repair = read_json(repair_path)
        no_new_interactions = int(repair.get("new_environment_interactions_for_repair", -1)) == 0
        training_unchanged = repair.get("training_outputs_changed") is False
        _check(checks, "postprocessing_repair_preserved_training_outputs_without_new_interactions",
               no_new_interactions and training_unchanged,
               {"artifact": str(repair_path), "metadata": repair,
                "condition_label_difference_is_not_an_algorithm_failure": True})
    return checks


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _make_figures(output: Path, by_condition_seed: dict[str, dict[int, dict[str, Any]]]) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    conditions = [condition for condition in CONDITIONS if by_condition_seed.get(condition)]
    colors = plt.get_cmap("tab10").colors
    rng = np.random.default_rng(20261015)
    paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    specs = [
        ("reward_per_training_step", "Online reward per training step"),
        ("success_rate", "Online success rate"),
    ]
    x = np.arange(len(conditions))
    for ax, (metric, ylabel) in zip(axes, specs):
        for index, condition in enumerate(conditions):
            seeds = sorted(by_condition_seed[condition])
            values = [
                float(by_condition_seed[condition][seed]["online"][
                    "reward_per_training_step" if metric == "reward_per_training_step" else "success_rate"
                ])
                for seed in seeds
            ]
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(index + jitter, values, color=colors[index], edgecolor="white", linewidth=0.7,
                       s=52, alpha=0.9, zorder=3)
            ax.plot([index - 0.19, index + 0.19], [float(np.mean(values))] * 2,
                    color="black", linewidth=2, zorder=4)
        ax.set_xticks(x, [CONDITION_LABELS[condition] for condition in conditions], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("S1b shared-write cohort: online learning", fontsize=14)
    fig.text(0.5, -0.02, "Each dot is one deployment seed; black bars mark descriptive means.", ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        path = output / f"online.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    specs = [
        ("mean_return", "Held-out A2/B2 mean return"),
        ("success_rate", "Held-out A2/B2 success rate"),
    ]
    x = np.arange(2)
    for ax, (metric, ylabel) in zip(axes, specs):
        for index, condition in enumerate(conditions):
            seed_values = []
            for seed in sorted(by_condition_seed[condition]):
                record = by_condition_seed[condition][seed]
                values = [
                    float(record["evaluation"]["heldout_probe_mediated_boot"][metric]),
                    float(record["evaluation"]["heldout_probe_mediated_seen"][metric]),
                ]
                seed_values.append(values)
                ax.plot(x, values, color=colors[index], alpha=0.22, linewidth=1.0, zorder=1)
                ax.scatter(x, values, color=colors[index], alpha=0.50, s=20, zorder=2)
            if seed_values:
                means = np.mean(np.asarray(seed_values, dtype=float), axis=0)
                ax.plot(x, means, color=colors[index], marker="o", linewidth=2.6, markersize=6,
                        label=CONDITION_LABELS[condition], zorder=3)
        ax.set_xticks(x, ["Boot\nafter held-out probes", "Seen\nafter seen sequence"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("S1b shared-write cohort: probe-mediated cold transfer", fontsize=14)
    fig.text(0.5, -0.02,
             "Each held-out context receives 512 real uniform-action probe steps before Boot; these interactions count toward cost.",
             ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        path = output / f"probe_mediated_cold_transfer.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_report(input_root: Path, output: Path) -> dict[str, Any]:
    input_root, output = input_root.resolve(), output.resolve()
    config_path, summary_path = input_root / "frozen_config.json", input_root / "summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("input must contain completed frozen_config.json and summary.json")
    config, cohort_summary = read_json(config_path), read_json(summary_path)
    if not cohort_summary.get("records"):
        raise ValueError("cohort summary.json has no completed records")
    if set(config.get("conditions", [])) != set(CONDITIONS):
        raise ValueError(f"frozen_config conditions must be {CONDITIONS}")
    seed_dirs = sorted(input_root.glob("seed_*"), key=lambda path: int(path.name.split("_", 1)[1]))
    if not seed_dirs:
        raise FileNotFoundError("input contains no seed_* result directories")

    records_by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for seed_dir in seed_dirs:
        for condition in CONDITIONS:
            if not (seed_dir / condition / "summary.json").is_file():
                raise FileNotFoundError(f"missing completed condition summary: {seed_dir / condition}")
            records_by_condition[condition].append(_seed_record(seed_dir, condition, config))
    by_condition_seed = {
        condition: {record["seed"]: record for record in rows}
        for condition, rows in records_by_condition.items()
    }
    aggregates = {condition: _aggregate(rows) for condition, rows in records_by_condition.items()}
    contrasts = _paired_contrasts(by_condition_seed)
    checks = run_audits(input_root, cohort_summary, config, seed_dirs)
    figures = _make_figures(output, by_condition_seed)
    failed = [check for check in checks if not check["passed"]]
    train_steps = sum(record["cost"]["training_steps"] for rows in records_by_condition.values() for record in rows)
    probe_steps = sum(record["cost"]["probe_steps"] for rows in records_by_condition.values() for record in rows)
    eval_steps = sum(record["cost"]["evaluation_steps"] for rows in records_by_condition.values() for record in rows)

    audit = {
        "schema_version": 1,
        "input": str(input_root),
        "phase": cohort_summary.get("phase"),
        "seeds": [int(seed_dir.name.split("_", 1)[1]) for seed_dir in seed_dirs],
        "conditions": list(CONDITIONS),
        "context_sets": {
            "seen": list(dict.fromkeys(config["seen_sequence"])),
            "heldout": list(dict.fromkeys(config["heldout_sequence"])),
            "adapted_all": list(dict.fromkeys([*config["seen_sequence"], *config["heldout_sequence"]])),
        },
        "probe_mediated_evaluation": {
            "steps_per_context": int(config["probe_steps_per_context"]),
            "heldout_context_count": len(dict.fromkeys(config["heldout_sequence"])),
            "heldout_probe_steps_before_boot": int(config["probe_steps_per_context"])
            * len(dict.fromkeys(config["heldout_sequence"])),
            "interpretation": "Boot/Seen cold-transfer evaluations follow 512-step behavior-probe interactions for each held-out context; this is probe-mediated transfer, not zero-experience transfer.",
        },
        "aggregates": aggregates,
        "paired_contrasts": contrasts,
        "cost_ledger": {
            "training_steps_all_conditions": train_steps,
            "probe_steps_all_conditions": probe_steps,
            "evaluation_steps_all_conditions": eval_steps,
            "total_interactions_all_conditions": train_steps + probe_steps + eval_steps,
            "accounting": "training + probe + evaluation interactions; verified against each condition summary and raw probe/evaluation CSVs.",
        },
        "grouping_metrics": {
            "scope": "Post-hoc family agreement only; observer family labels do not affect grouping. Unresolved assignments are reported and are not audit failures.",
            "per_condition": {
                condition: [record["grouping_audit_only"] for record in sorted(records, key=lambda row: row["seed"])]
                for condition, records in records_by_condition.items()
            },
        },
        "capacity_basis": "Actual actor and whole-package sizes from periodic capacity.json samples, plus decoded stage actor files and final checkpoint + W0 size.",
        "source_hash_basis": "Each captured source_snapshot file is checked against its own per-run source_manifest.json entry; current source files are not compared.",
        "statistical_scope": "Descriptive means and paired per-seed differences only; no formal statistical claims.",
        "figures": figures,
        "checks": checks,
        "failed_assertions": failed,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed:
        names = ", ".join(check["name"] for check in failed)
        raise AssertionError(f"S1b shared-write audit failed: {names}; details saved to {output / 'audit.json'}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed S1b output root")
    parser.add_argument("--output", type=Path, required=True, help="Report output directory")
    args = parser.parse_args()
    result = build_report(args.input, args.output)
    print(json.dumps({
        "phase": result["phase"],
        "seed_count": len(result["seeds"]),
        "check_count": len(result["checks"]),
        "failed_assertions": len(result["failed_assertions"]),
        "figures": result["figures"],
        "audit": str(args.output.resolve() / "audit.json"),
    }, indent=2))


if __name__ == "__main__":
    main()

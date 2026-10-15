"""Summarize and audit completed shared-write cohorts without rerunning them."""
from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(ROOT / "src"))

from online_rl.shared_write_storage import decode_shared_knowledge


CONDITIONS = ("independent", "independent_fast", "oracle_active", "oracle_frozen")
CONDITION_LABELS = {
    "independent": "Independent local",
    "independent_fast": "Independent, faster local LR",
    "oracle_active": "Oracle grouped, active parent",
    "oracle_frozen": "Oracle grouped, frozen parent",
}
CONTRASTS = {
    "oracle_active_minus_oracle_frozen": "oracle_frozen",
    "oracle_active_minus_independent": "independent",
    "oracle_active_minus_independent_fast": "independent_fast",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def metric_pair(result: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_return": float(result["mean_return"]),
        "success_rate": float(result.get("success_rate", result.get("mean_success"))),
    }


def average_contexts(
    evaluations: dict[str, Any], stage: str, contexts: list[str]
) -> dict[str, Any]:
    found = [context for context in contexts if context in evaluations.get(stage, {})]
    if not found:
        return {"context_count": 0, "contexts": [], "mean_return": None, "success_rate": None}
    vals = [metric_pair(evaluations[stage][context]) for context in found]
    return {
        "context_count": len(found),
        "contexts": found,
        "mean_return": mean([value["mean_return"] for value in vals]),
        "success_rate": mean([value["success_rate"] for value in vals]),
    }


def decode_actor(folder: Path, stage: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return decode_shared_knowledge((folder / f"actor_{stage}.bin").read_bytes())


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def capacity_summary(folder: Path) -> dict[str, Any]:
    rows = read_json(folder / "capacity.json")
    actor = [int(row["actor_bytes"]) for row in rows]
    package = [int(row["whole_package_bytes"]) for row in rows]
    if not rows:
        raise ValueError(f"capacity.json has no periodic samples: {folder}")
    return {
        "sample_count": len(rows),
        "actor_bytes": {"mean": mean([float(value) for value in actor]), "peak": max(actor)},
        "whole_package_bytes": {
            "mean": mean([float(value) for value in package]),
            "peak": max(package),
        },
    }


def seed_record(input_root: Path, seed_dir: Path, condition: str, config: dict[str, Any]) -> dict[str, Any]:
    folder = seed_dir / condition
    summary = read_json(folder / "summary.json")
    evaluations = summary["evaluations"]
    seen_contexts = list(dict.fromkeys(config["seen_sequence"]))
    heldout_contexts = list(dict.fromkeys(config["heldout_sequence"]))
    all_contexts = list(dict.fromkeys([*seen_contexts, *heldout_contexts]))
    boot = average_contexts(evaluations, "boot", heldout_contexts)
    seen = average_contexts(evaluations, "seen", heldout_contexts)
    adapted_all = average_contexts(evaluations, "adapted", all_contexts)
    adapted_old = average_contexts(evaluations, "adapted", seen_contexts)
    episodes = read_csv(folder / "episodes.csv")
    completed = [row for row in episodes if row["completed"].lower() == "true"]
    completed_rewards = [float(row["return"]) for row in completed]
    online_success_count = sum(int(row["success"]) for row in completed)
    training_steps = int(summary["training_steps"])
    evaluation_steps = int(summary["evaluation_steps"])
    return {
        "seed": int(summary["seed"]),
        "condition": condition,
        "online": {
            "total_reward": float(summary["total_reward"]),
            "reward_per_step": float(summary["reward_per_step"]),
            "success_rate": float(summary["online_success"]),
            "completed_episodes": len(completed),
            "successes": online_success_count,
            "episode_return_mean": mean(completed_rewards),
        },
        "evaluation": {
            "heldout_cold_boot": boot,
            "heldout_cold_seen": seen,
            "heldout_seen_minus_boot": {
                "mean_return": (
                    seen["mean_return"] - boot["mean_return"]
                    if seen["mean_return"] is not None and boot["mean_return"] is not None else None
                ),
                "success_rate": (
                    seen["success_rate"] - boot["success_rate"]
                    if seen["success_rate"] is not None and boot["success_rate"] is not None else None
                ),
            },
            "adapted_all_contexts": adapted_all,
            "adapted_seen_contexts": adapted_old,
        },
        "cost": {
            "training_steps": training_steps,
            "fit_steps": int(summary["counts"]["fit_steps"]),
            "holdout_steps": int(summary["counts"]["holdout_steps"]),
            "evaluation_steps": evaluation_steps,
            "total_interactions": int(summary["total_interactions"]),
            "minibatch_updates": int(summary["counts"]["minibatch_updates"]),
            "wall_seconds": finite_float(summary.get("wall_s")),
            "cpu_seconds": finite_float(summary.get("cpu_s")),
        },
        "capacity_samples": capacity_summary(folder),
        "endpoint_sizes": summary.get("sizes", {}),
        "_folder": folder,
    }


def condition_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(path: tuple[str, ...]) -> list[float]:
        values: list[float] = []
        for record in records:
            item: Any = record
            for key in path:
                item = item[key]
            if item is not None:
                values.append(float(item))
        return values

    metrics = {
        "online_reward_per_step": ("online", "reward_per_step"),
        "online_success_rate": ("online", "success_rate"),
        "cold_boot_return": ("evaluation", "heldout_cold_boot", "mean_return"),
        "cold_boot_success_rate": ("evaluation", "heldout_cold_boot", "success_rate"),
        "cold_seen_return": ("evaluation", "heldout_cold_seen", "mean_return"),
        "cold_seen_success_rate": ("evaluation", "heldout_cold_seen", "success_rate"),
        "cold_seen_minus_boot_return": ("evaluation", "heldout_seen_minus_boot", "mean_return"),
        "cold_seen_minus_boot_success_rate": ("evaluation", "heldout_seen_minus_boot", "success_rate"),
        "adapted_all_return": ("evaluation", "adapted_all_contexts", "mean_return"),
        "adapted_all_success_rate": ("evaluation", "adapted_all_contexts", "success_rate"),
        "adapted_old_return": ("evaluation", "adapted_seen_contexts", "mean_return"),
        "adapted_old_success_rate": ("evaluation", "adapted_seen_contexts", "success_rate"),
        "training_steps": ("cost", "training_steps"),
        "fit_steps": ("cost", "fit_steps"),
        "holdout_steps": ("cost", "holdout_steps"),
        "evaluation_steps": ("cost", "evaluation_steps"),
        "total_interactions": ("cost", "total_interactions"),
        "wall_seconds": ("cost", "wall_seconds"),
        "cpu_seconds": ("cost", "cpu_seconds"),
        "sampled_actor_bytes_mean": ("capacity_samples", "actor_bytes", "mean"),
        "sampled_actor_bytes_peak": ("capacity_samples", "actor_bytes", "peak"),
        "sampled_whole_package_bytes_mean": ("capacity_samples", "whole_package_bytes", "mean"),
        "sampled_whole_package_bytes_peak": ("capacity_samples", "whole_package_bytes", "peak"),
    }
    output: dict[str, Any] = {
        "seed_count": len(records),
        "seeds": [record["seed"] for record in sorted(records, key=lambda row: row["seed"])],
        "metrics": {},
        "per_seed": [],
        "pooled_online": {},
    }
    for name, path in metrics.items():
        values = collect(path)
        output["metrics"][name] = {
            "mean_across_seeds": mean(values),
            "per_seed": [
                {"seed": record["seed"], "value": _nested(record, path)}
                for record in sorted(records, key=lambda row: row["seed"])
            ],
        }
    total_reward = sum(float(record["online"]["total_reward"]) for record in records)
    total_steps = sum(int(record["cost"]["training_steps"]) for record in records)
    successes = sum(int(record["online"]["successes"]) for record in records)
    episodes = sum(int(record["online"]["completed_episodes"]) for record in records)
    output["pooled_online"] = {
        "total_reward": total_reward,
        "training_steps": total_steps,
        "reward_per_step": total_reward / total_steps if total_steps else None,
        "successes": successes,
        "completed_episodes": episodes,
        "success_rate": successes / episodes if episodes else None,
    }
    for record in sorted(records, key=lambda row: row["seed"]):
        output["per_seed"].append(_public_record(record))
    return output


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        value = value[key]
    return value


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "_folder"}


def paired_contrasts(records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    metric_paths = {
        "online_reward_per_step": ("online", "reward_per_step"),
        "online_success_rate": ("online", "success_rate"),
        "cold_boot_return": ("evaluation", "heldout_cold_boot", "mean_return"),
        "cold_boot_success_rate": ("evaluation", "heldout_cold_boot", "success_rate"),
        "cold_seen_return": ("evaluation", "heldout_cold_seen", "mean_return"),
        "cold_seen_success_rate": ("evaluation", "heldout_cold_seen", "success_rate"),
        "cold_seen_minus_boot_return": ("evaluation", "heldout_seen_minus_boot", "mean_return"),
        "cold_seen_minus_boot_success_rate": ("evaluation", "heldout_seen_minus_boot", "success_rate"),
        "adapted_all_return": ("evaluation", "adapted_all_contexts", "mean_return"),
        "adapted_all_success_rate": ("evaluation", "adapted_all_contexts", "success_rate"),
        "adapted_old_return": ("evaluation", "adapted_seen_contexts", "mean_return"),
        "adapted_old_success_rate": ("evaluation", "adapted_seen_contexts", "success_rate"),
    }
    result: dict[str, Any] = {"interpretation": "descriptive paired differences; no formal statistical claim", "comparisons": {}}
    active = records.get("oracle_active", {})
    for name, other_condition in CONTRASTS.items():
        other = records.get(other_condition, {})
        seeds = sorted(set(active) & set(other))
        metrics: dict[str, Any] = {}
        for metric, path in metric_paths.items():
            rows = []
            for seed in seeds:
                left = _nested(active[seed], path)
                right = _nested(other[seed], path)
                rows.append({"seed": seed, "oracle_active": left, other_condition: right,
                             "difference": float(left) - float(right)})
            metrics[metric] = {
                "mean_difference": mean([row["difference"] for row in rows]),
                "paired_seed_count": len(rows),
                "per_seed": rows,
            }
        result["comparisons"][name] = {"reference_condition": "oracle_active", "other_condition": other_condition,
                                       "paired_seed_count": len(seeds), "metrics": metrics}
    return result


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def _episode_evaluations(path: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in read_csv(path):
        key = (row["stage"], row["context"], row["episode"])
        grouped.setdefault(key, []).append({
            "stage": row["stage"], "context": row["context"], "episode": int(row["episode"]),
            "reset_seed": int(row["reset_seed"]), "action_seed": int(row["action_seed"]),
            "actions": json.loads(row["actions"]), "return": float(row["return"]),
            "success": row["success"].lower() in ("true", "1"),
        })
    # There is one row per episode; a list keeps duplicate episode keys visible.
    return grouped


def run_audits(
    input_root: Path,
    config: dict[str, Any],
    by_condition: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    seeds = sorted(set.intersection(*(set(by_condition.get(condition, {})) for condition in CONDITIONS)))
    heldout = list(dict.fromkeys(config["heldout_sequence"]))
    seen_sequence = list(config["seen_sequence"])
    duration = int(config["bout_steps"])
    freeze_bouts = int(config["freeze_after_seen_bouts"])

    for seed in sorted(set.union(*(set(by_condition.get(condition, {})) for condition in CONDITIONS))):
        folders = {condition: input_root / f"seed_{seed}" / condition for condition in CONDITIONS}
        slot_maps = {condition: read_json(folder / "external_slot_map.json") for condition, folder in folders.items()}
        add_check(checks, f"seed_{seed}_same_route_slot_map", all(slot_maps[c] == slot_maps[CONDITIONS[0]] for c in CONDITIONS),
                  {"slot_maps": slot_maps})

        step_rows = {condition: read_csv(folder / "steps.csv") for condition, folder in folders.items()}
        for condition, rows in step_rows.items():
            first_seen_bouts = [row for row in rows if int(row["bout"]) < min(8, len(seen_sequence))]
            seen_context_counts: dict[str, int] = {}
            for row in first_seen_bouts:
                seen_context_counts[row["context"]] = seen_context_counts.get(row["context"], 0) + 1
            expected_contexts = seen_sequence[:min(8, len(seen_sequence))]
            actual_by_bout: dict[int, str] = {}
            for row in first_seen_bouts:
                actual_by_bout[int(row["bout"])] = row["context"]
            expected_rows = min(8, len(seen_sequence)) * duration
            no_heldout = not any(row["context"] in heldout for row in first_seen_bouts)
            sequence_ok = all(actual_by_bout.get(index) == context for index, context in enumerate(expected_contexts))
            count_ok = len(first_seen_bouts) == expected_rows
            add_check(checks, f"seed_{seed}_{condition}_heldout_absent_first_seen_bouts",
                      no_heldout and sequence_ok and count_ok,
                      {"checked_bouts": min(8, len(seen_sequence)), "expected_rows": expected_rows,
                       "actual_rows": len(first_seen_bouts), "expected_contexts": expected_contexts,
                       "actual_contexts": [actual_by_bout.get(i) for i in range(len(expected_contexts))],
                       "heldout_contexts": heldout, "context_step_counts": seen_context_counts})

        active_rows = step_rows["oracle_active"]
        frozen_rows = step_rows["oracle_frozen"]
        active_prefix = [row for row in active_rows if int(row["bout"]) < freeze_bouts]
        frozen_prefix = [row for row in frozen_rows if int(row["bout"]) < freeze_bouts]
        expected_prefix_rows = freeze_bouts * duration
        comparable = (len(active_prefix) == len(frozen_prefix) == expected_prefix_rows and
                      all((a["bout"], a["step"], a["context"], a["action"], a["reward"],
                           a["terminated"], a["truncated"]) ==
                          (b["bout"], b["step"], b["context"], b["action"], b["reward"],
                           b["terminated"], b["truncated"])
                          for a, b in zip(active_prefix, frozen_prefix)))
        add_check(checks, f"seed_{seed}_active_frozen_first_bouts_exact_actions_rewards", comparable,
                  {"bouts": freeze_bouts, "configured_duration_per_bout": duration,
                   "expected_rows": expected_prefix_rows, "active_rows": len(active_prefix),
                   "frozen_rows": len(frozen_prefix),
                   "compared_actions": [row["action"] for row in active_prefix],
                   "compared_rewards": [float(row["reward"]) for row in active_prefix]})

        for condition in ("independent", "oracle_frozen"):
            grouped = _episode_evaluations(folders[condition] / "evaluations.csv")
            comparisons: dict[str, Any] = {}
            passed = True
            for context in heldout:
                boot_keys = sorted((key for key in grouped if key[0] == "boot" and key[1] == context),
                                   key=lambda key: int(key[2]))
                seen_keys = sorted((key for key in grouped if key[0] == "seen" and key[1] == context),
                                   key=lambda key: int(key[2]))
                boot_eps = [row for key in boot_keys for row in grouped[key]]
                seen_eps = [row for key in seen_keys for row in grouped[key]]
                match = len(boot_eps) == len(seen_eps) and len(boot_eps) > 0
                episode_pairs = []
                for boot_row, seen_row in zip(boot_eps, seen_eps):
                    same = (boot_row["episode"] == seen_row["episode"] and
                            boot_row["reset_seed"] == seen_row["reset_seed"] and
                            boot_row["action_seed"] == seen_row["action_seed"] and
                            boot_row["actions"] == seen_row["actions"] and
                            boot_row["return"] == seen_row["return"] and
                            boot_row["success"] == seen_row["success"])
                    match = match and same
                    episode_pairs.append({"boot": boot_row, "seen": seen_row, "exact_match": same})
                comparisons[context] = {"episode_count": len(episode_pairs), "pairs": episode_pairs,
                                       "exact_match": match}
                passed = passed and match
            add_check(checks, f"seed_{seed}_{condition}_heldout_boot_seen_actions_and_seeds_exact", passed,
                      {"heldout_contexts": comparisons})

        frozen_folder = folders["oracle_frozen"]
        local_boot, shared_boot, groups_boot = decode_actor(frozen_folder, "boot")
        local_seen, shared_seen, groups_seen = decode_actor(frozen_folder, "seen")
        shared_equal = np.array_equal(shared_boot, shared_seen)
        heldout_local_equal = True
        heldout_local_details = {}
        frozen_slots = read_json(frozen_folder / "external_slot_map.json")
        for context in heldout:
            slot = int(frozen_slots[context])
            equal = np.array_equal(local_boot[slot], local_seen[slot])
            heldout_local_equal = heldout_local_equal and equal
            heldout_local_details[context] = {"slot": slot, "equal": equal}
        frozen_checkpoint = _load_checkpoint_flag(frozen_folder / "checkpoint_seen.pt")
        add_check(checks, f"seed_{seed}_frozen_shared_actor_state_unchanged_boot_to_seen",
                  shared_equal and frozen_checkpoint,
                  {"shared_exact_equal": shared_equal, "shared_shape": list(shared_boot.shape),
                   "shared_max_abs_difference": float(np.max(np.abs(shared_boot - shared_seen))) if shared_boot.size else 0.0,
                   "checkpoint_shared_frozen": frozen_checkpoint,
                   "heldout_local_rows_unchanged": heldout_local_equal,
                   "heldout_local_rows": heldout_local_details,
                   "groups_boot": groups_boot.tolist(), "groups_seen": groups_seen.tolist()})

        active_folder = folders["oracle_active"]
        _, active_boot, _ = decode_actor(active_folder, "boot")
        _, active_seen, _ = decode_actor(active_folder, "seen")
        active_nonzero = bool(np.any(active_boot != 0.0))
        active_changed = not np.array_equal(active_boot, active_seen)
        add_check(checks, f"seed_{seed}_active_shared_weights_nonzero_and_changed",
                  active_nonzero and active_changed,
                  {"boot_nonzero": active_nonzero,
                   "boot_l2_norm": float(np.linalg.norm(active_boot)),
                   "boot_max_abs": float(np.max(np.abs(active_boot))) if active_boot.size else 0.0,
                   "changed_boot_to_seen": active_changed,
                   "boot_seen_l2_difference": float(np.linalg.norm(active_seen - active_boot))})

    return checks


def _load_checkpoint_flag(path: Path) -> bool:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return bool(checkpoint["shared_frozen"])


def make_figures(output: Path, by_condition: dict[str, dict[int, dict[str, Any]]]) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    conditions = [condition for condition in CONDITIONS if by_condition.get(condition)]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    metric_specs = [
        ("online", "reward_per_step", "Reward per online step"),
        ("online", "success_rate", "Online success rate"),
    ]
    rng = np.random.default_rng(20261015)
    x = np.arange(len(conditions))
    for ax, (group, metric, ylabel) in zip(axes, metric_specs):
        for index, condition in enumerate(conditions):
            records = [by_condition[condition][seed] for seed in sorted(by_condition[condition])]
            values = [float(record[group][metric]) for record in records]
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(index + jitter, values, color=colors[index], edgecolor="white", linewidth=0.7,
                       s=55, alpha=0.9, zorder=3)
            ax.plot([index - 0.19, index + 0.19], [float(np.mean(values))] * 2,
                    color="black", linewidth=2.0, zorder=4)
        ax.set_xticks(x, [CONDITION_LABELS.get(condition, condition) for condition in conditions], rotation=22, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Shared-write cohort: online learning", fontsize=14)
    fig.text(0.5, -0.02, "Each dot is one deployment seed; horizontal bars mark descriptive means.",
             ha="center", fontsize=9)
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"online.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    transfer_specs = [
        ("mean_return", "Held-out A2/B2 mean return"),
        ("success_rate", "Held-out A2/B2 success rate"),
    ]
    x = np.arange(2)
    for ax, (metric, ylabel) in zip(axes, transfer_specs):
        for index, condition in enumerate(conditions):
            seed_records = by_condition[condition]
            seed_values = []
            for seed in sorted(seed_records):
                record = seed_records[seed]
                values = [record["evaluation"]["heldout_cold_boot"][metric],
                          record["evaluation"]["heldout_cold_seen"][metric]]
                seed_values.append(values)
                ax.plot(x, values, color=colors[index], alpha=0.22, linewidth=1.0, zorder=1)
                ax.scatter(x, values, color=colors[index], alpha=0.50, s=20, zorder=2)
            if seed_values:
                means = np.mean(np.asarray(seed_values, dtype=float), axis=0)
                ax.plot(x, means, color=colors[index], marker="o", linewidth=2.6, markersize=6,
                        label=CONDITION_LABELS.get(condition, condition), zorder=3)
        ax.set_xticks(x, ["Boot\n(after first 2 bouts)", "Seen\n(after seen sequence)"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Shared-write cohort: cold held-out transfer", fontsize=14)
    fig.text(0.5, -0.02, "Thin lines are paired seeds; thick lines mark descriptive means. Held-out A2/B2 receive no updates before Seen.",
             ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        path = output / f"cold_transfer.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_report(input_root: Path, output: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    output = output.resolve()
    frozen_config_path = input_root / "frozen_config.json"
    cohort_summary_path = input_root / "summary.json"
    if not frozen_config_path.exists() or not cohort_summary_path.exists():
        raise FileNotFoundError("input must contain frozen_config.json and completed summary.json")
    config = read_json(frozen_config_path)
    cohort_summary = read_json(cohort_summary_path)
    if not cohort_summary.get("records"):
        raise ValueError("cohort summary.json has no completed records")

    seed_dirs = sorted(input_root.glob("seed_*"), key=lambda path: int(path.name.split("_", 1)[1]))
    if not seed_dirs:
        raise FileNotFoundError("input contains no seed_* result directories")
    records_by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for seed_dir in seed_dirs:
        for condition in config.get("conditions", CONDITIONS):
            if condition not in CONDITIONS:
                continue
            folder = seed_dir / condition
            if not (folder / "summary.json").exists():
                raise FileNotFoundError(f"missing completed condition summary: {folder}")
            records_by_condition[condition].append(seed_record(input_root, seed_dir, condition, config))

    by_condition_seed: dict[str, dict[int, dict[str, Any]]] = {
        condition: {record["seed"]: record for record in records}
        for condition, records in records_by_condition.items()
    }
    aggregates = {condition: condition_aggregate(records)
                  for condition, records in records_by_condition.items() if records}
    contrasts = paired_contrasts(by_condition_seed)
    checks = run_audits(input_root, config, by_condition_seed)
    figures = make_figures(output, by_condition_seed)

    all_training_steps = sum(record["cost"]["training_steps"] for rows in records_by_condition.values() for record in rows)
    all_evaluation_steps = sum(record["cost"]["evaluation_steps"] for rows in records_by_condition.values() for record in rows)
    audit = {
        "schema_version": 1,
        "input": str(input_root),
        "phase": cohort_summary.get("phase"),
        "source_seed_count": len(seed_dirs),
        "conditions": [condition for condition in CONDITIONS if records_by_condition[condition]],
        "context_sets": {
            "seen": list(dict.fromkeys(config["seen_sequence"])),
            "heldout": list(dict.fromkeys(config["heldout_sequence"])),
            "adapted_all": list(dict.fromkeys([*config["seen_sequence"], *config["heldout_sequence"]])),
        },
        "aggregates": aggregates,
        "paired_contrasts": contrasts,
        "cost_ledger": {
            "training_steps_all_conditions": all_training_steps,
            "evaluation_steps_all_conditions": all_evaluation_steps,
            "total_interactions_all_conditions": all_training_steps + all_evaluation_steps,
        },
        "capacity_basis": "Periodic rows from each condition's capacity.json; actor_bytes and whole_package_bytes are actual encoded/package sizes.",
        "statistical_scope": "Descriptive means and paired per-seed differences only; no formal statistical claims.",
        "figures": figures,
        "checks": checks,
        "failed_assertions": [check for check in checks if not check["passed"]],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["failed_assertions"]:
        names = ", ".join(check["name"] for check in audit["failed_assertions"])
        raise AssertionError(f"shared-write audit failed: {names}; details saved to {output / 'audit.json'}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed shared-write output root")
    parser.add_argument("--output", type=Path, required=True, help="New or existing report output directory")
    args = parser.parse_args()
    result = build_report(args.input, args.output)
    print(json.dumps({"phase": result["phase"], "source_seed_count": result["source_seed_count"],
                      "failed_assertions": len(result["failed_assertions"]), "figures": result["figures"],
                      "audit": str(args.output.resolve() / "audit.json")}, indent=2))


if __name__ == "__main__":
    main()

"""Summarize completed retention-stream cohorts by seed and budget."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BUDGETS = ("low", "wide")
FULL_STREAM_METRICS = ("total_reward", "reward_per_raw_step")
RECURRENCE_METRICS = (
    "first256_reward_per_step",
    "recovery512_reward_per_step",
    "recovery1024_reward_per_step",
    "first_goal_step",
    "first_gradient_step",
)
RECURRENCE_MIN_STEPS = {
    "first256_reward_per_step": 256,
    "recovery512_reward_per_step": 512,
    "recovery1024_reward_per_step": 1024,
}
CAPACITY_FIELDS = ("actor_bytes", "full_physical_bytes", "package_budget_bytes", "evidence_bytes")
COUNT_FIELDS = (
    "raw_steps", "fit_steps", "slot_evictions", "rollouts_trained", "minibatch_updates",
    "write_proposals", "write_accepts", "prior_bindings", "cpu_s", "wall_s",
)


def _declared_methods(config: Mapping[str, Any]) -> tuple[str, ...]:
    value = config.get("methods")
    if not isinstance(value, list) or not value:
        raise ValueError("config.methods must declare the complete method array")
    methods = tuple(value)
    if any(not isinstance(method, str) or not method.strip() for method in methods):
        raise ValueError("config.methods entries must be non-empty strings")
    if len(methods) != len(set(methods)):
        raise ValueError("config.methods must not contain duplicates")
    if "independent" not in methods:
        raise ValueError("config.methods must include the independent paired baseline")
    return methods


def _declared_budgets(config: Mapping[str, Any]) -> tuple[str, ...]:
    value = config.get("budgets")
    if isinstance(value, Mapping):
        budgets = tuple(value.keys())
    elif isinstance(value, list):
        budgets = tuple(value)
    else:
        raise ValueError("config.budgets must declare the complete budget array or budget mapping")
    if not budgets:
        raise ValueError("config.budgets must contain at least one budget")
    if any(not isinstance(budget, str) or budget not in BUDGETS for budget in budgets):
        raise ValueError(f"config.budgets entries must be from {BUDGETS}")
    if len(budgets) != len(set(budgets)):
        raise ValueError("config.budgets must not contain duplicates")
    # Keep the report's budget order stable even if a mapping was serialized differently.
    return tuple(budget for budget in BUDGETS if budget in budgets)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cohort summary JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite_number(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    number = _finite_number(value, name)
    assert number is not None
    if number < 0 or not number.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


def _validate_recurrence(row: Any, branch_name: str, index: int) -> dict[str, Any]:
    prefix = f"{branch_name}.recurrences[{index}]"
    if not isinstance(row, dict):
        raise ValueError(f"{prefix} must be an object")
    for key in ("task", "bout"):
        if key not in row:
            raise ValueError(f"{prefix}.{key} is required")
    if not isinstance(row["task"], str) or not row["task"]:
        raise ValueError(f"{prefix}.task must be a non-empty string")
    result = dict(row)
    result["bout"] = _nonnegative_int(row["bout"], f"{prefix}.bout")
    observed_steps = None
    if "observed_steps" in row:
        observed_steps = _nonnegative_int(row["observed_steps"], f"{prefix}.observed_steps")
        result["observed_steps"] = observed_steps
    if not isinstance(row.get("previously_evicted"), bool):
        raise ValueError(f"{prefix}.previously_evicted must be boolean")
    for metric in RECURRENCE_METRICS:
        if metric not in row:
            raise ValueError(f"{prefix}.{metric} is required")
        result[metric] = _finite_number(
            row[metric], f"{prefix}.{metric}",
            allow_none=(metric == "first_goal_step" or metric in RECURRENCE_MIN_STEPS),
        )
        minimum_steps = RECURRENCE_MIN_STEPS.get(metric)
        if observed_steps is not None and minimum_steps is not None and observed_steps < minimum_steps:
            # Some short pilot bouts serialize a shorter-window mean in the longer-window field.
            # Treat any window that was not fully observed as missing instead of relabeling it.
            result[metric] = None
    for metric in ("first_goal_step", "first_gradient_step"):
        value = result[metric]
        if value is not None and (value < 0 or not value.is_integer()):
            raise ValueError(f"{prefix}.{metric} must be a non-negative integer or null")
        if value is not None:
            result[metric] = int(value)
    result["previously_evicted"] = bool(row["previously_evicted"])
    return result


def _validate_branch(row: Any, index: int, methods: Sequence[str]) -> dict[str, Any]:
    prefix = f"records[{index}]"
    if not isinstance(row, dict):
        raise ValueError(f"{prefix} must be an object")
    for key in ("method", "budget", "seed", "status", "audit", "capacity_max", "recurrences"):
        if key not in row:
            raise ValueError(f"{prefix}.{key} is required")
    method, budget = row["method"], row["budget"]
    if method not in methods:
        raise ValueError(f"{prefix}.method must be declared in config.methods")
    if budget not in BUDGETS:
        raise ValueError(f"{prefix}.budget must be one of {BUDGETS}")
    branch = dict(row)
    branch["seed"] = _nonnegative_int(row["seed"], f"{prefix}.seed")
    if row["status"] != "completed":
        raise ValueError(f"{prefix} is incomplete: status={row['status']!r}")
    audit = row["audit"]
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError(f"{prefix}.audit.passed must be true")
    for key in ("source_unchanged", "persistent_roundtrip"):
        if audit.get(key) is not True:
            raise ValueError(f"{prefix}.audit.{key} must be true")

    for metric in ("total_reward", "reward_per_raw_step"):
        if metric not in row:
            raise ValueError(f"{prefix}.{metric} is required")
        branch[metric] = _finite_number(row[metric], f"{prefix}.{metric}")
    for field in COUNT_FIELDS:
        if field not in row:
            raise ValueError(f"{prefix}.{field} is required")
        branch[field] = _finite_number(row[field], f"{prefix}.{field}")
        if field in COUNT_FIELDS[:-2]:
            value = branch[field]
            assert value is not None
            if value < 0 or not value.is_integer():
                raise ValueError(f"{prefix}.{field} must be a non-negative integer")
            branch[field] = int(value)
        elif branch[field] is not None and branch[field] < 0:
            raise ValueError(f"{prefix}.{field} must be non-negative")
    if branch["raw_steps"] <= 0:
        raise ValueError(f"{prefix}.raw_steps must be positive")
    if branch["fit_steps"] > branch["raw_steps"]:
        raise ValueError(f"{prefix}.fit_steps cannot exceed raw_steps")
    if branch["write_accepts"] > branch["write_proposals"]:
        raise ValueError(f"{prefix}.write_accepts cannot exceed write_proposals")
    if "evaluation_steps" in row:
        branch["evaluation_steps"] = _nonnegative_int(row["evaluation_steps"], f"{prefix}.evaluation_steps")
    expected_reward_rate = float(branch["total_reward"]) / int(branch["raw_steps"])
    if not math.isclose(float(branch["reward_per_raw_step"]), expected_reward_rate, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"{prefix}.reward_per_raw_step does not match total_reward / raw_steps")

    capacity = row["capacity_max"]
    if not isinstance(capacity, dict):
        raise ValueError(f"{prefix}.capacity_max must be an object")
    clean_capacity = {}
    for field in CAPACITY_FIELDS:
        if field not in capacity:
            raise ValueError(f"{prefix}.capacity_max.{field} is required")
        clean_capacity[field] = _nonnegative_int(capacity[field], f"{prefix}.capacity_max.{field}")
    if clean_capacity["package_budget_bytes"] <= 0:
        raise ValueError(f"{prefix}.capacity_max.package_budget_bytes must be positive")
    if clean_capacity["full_physical_bytes"] > clean_capacity["package_budget_bytes"]:
        raise ValueError(f"{prefix}.capacity_max exceeds package_budget_bytes")
    branch["capacity_max"] = clean_capacity

    if not isinstance(row["recurrences"], list):
        raise ValueError(f"{prefix}.recurrences must be a list")
    if not row["recurrences"]:
        raise ValueError(f"{prefix}.recurrences must contain the completed stream's revisit observations")
    branch["recurrences"] = [
        _validate_recurrence(recurrence, prefix, recurrence_index)
        for recurrence_index, recurrence in enumerate(row["recurrences"])
    ]
    recurrence_keys = [(item["task"], item["bout"]) for item in branch["recurrences"]]
    if len(recurrence_keys) != len(set(recurrence_keys)):
        raise ValueError(f"{prefix}.recurrences has duplicate task/bout observations")
    if branch["slot_evictions"] == 0 and any(item["previously_evicted"] for item in branch["recurrences"]):
        raise ValueError(f"{prefix} marks a prior eviction despite zero slot_evictions")
    return branch


def _load_completed_cohort(
    input_path: str | Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    source = Path(input_path)
    summary_path = source / "summary.json" if source.is_dir() else source
    if summary_path.name != "summary.json" or not summary_path.is_file():
        raise FileNotFoundError(f"expected completed cohort summary.json under {source}")
    cohort = _read_json(summary_path)
    if cohort.get("format") != "retention_stream_cohort_v1":
        raise ValueError("input format must be retention_stream_cohort_v1")
    if cohort.get("status") != "completed":
        raise ValueError(f"cohort is incomplete: status={cohort.get('status')!r}")
    if cohort.get("analysis_role") != "exploratory":
        raise ValueError("cohort analysis_role must be exploratory")
    if not isinstance(cohort.get("config"), dict):
        raise ValueError("cohort config must be an object")
    methods = _declared_methods(cohort["config"])
    budgets = _declared_budgets(cohort["config"])
    if not isinstance(cohort.get("audit"), dict) or cohort["audit"].get("passed") is not True:
        raise ValueError("cohort audit.passed must be true")
    if not isinstance(cohort.get("records"), list) or not cohort["records"]:
        raise ValueError("cohort records must be a non-empty list")
    branches = [_validate_branch(row, index, methods) for index, row in enumerate(cohort["records"])]
    keys = [(row["method"], row["budget"], row["seed"]) for row in branches]
    if len(keys) != len(set(keys)):
        raise ValueError("cohort has duplicate method/budget/seed branches")

    seeds = sorted({int(row["seed"]) for row in branches})
    declared_seeds = cohort["config"].get("seeds")
    if declared_seeds is not None:
        if not isinstance(declared_seeds, list):
            raise ValueError("config.seeds must be a list when provided")
        expected_seeds = sorted(_nonnegative_int(seed, "config.seeds[]") for seed in declared_seeds)
        if len(expected_seeds) != len(set(expected_seeds)) or expected_seeds != seeds:
            raise ValueError("cohort records do not cover config.seeds exactly")
    expected = {(method, budget, seed) for method in methods for budget in budgets for seed in seeds}
    if set(keys) != expected:
        missing = sorted(expected - set(keys))
        extra = sorted(set(keys) - expected)
        raise ValueError(f"cohort branch grid is incomplete; missing={missing}, extra={extra}")

    declared_cost = cohort.get("cohort_cost")
    if not isinstance(declared_cost, dict):
        raise ValueError("cohort_cost must be an object")
    for key in ("new_raw_steps", "new_evaluation_steps", "new_total_interactions"):
        if key not in declared_cost:
            raise ValueError(f"cohort_cost.{key} is required")
        _nonnegative_int(declared_cost[key], f"cohort_cost.{key}")
    raw_sum = sum(int(row["raw_steps"]) for row in branches)
    if int(declared_cost["new_raw_steps"]) != raw_sum:
        raise ValueError("cohort_cost.new_raw_steps does not match completed branches")
    if int(declared_cost["new_total_interactions"]) < int(declared_cost["new_raw_steps"]):
        raise ValueError("cohort_cost.new_total_interactions is smaller than new_raw_steps")
    eval_sum = sum(int(row["evaluation_steps"]) for row in branches if "evaluation_steps" in row)
    eval_declared = int(declared_cost["new_evaluation_steps"])
    if any("evaluation_steps" in row for row in branches) and eval_sum != eval_declared:
        raise ValueError("cohort_cost.new_evaluation_steps does not match completed branches")
    if int(declared_cost["new_total_interactions"]) != raw_sum + eval_declared:
        raise ValueError("cohort_cost.new_total_interactions does not match raw plus evaluation steps")
    for key in ("cpu_s", "wall_s"):
        if key not in cohort:
            raise ValueError(f"cohort.{key} is required")
        _finite_number(cohort[key], f"cohort.{key}")
    method_order = {method: index for index, method in enumerate(methods)}
    return summary_path, cohort, sorted(
        branches, key=lambda row: (budgets.index(row["budget"]), method_order[row["method"]], row["seed"])
    ), methods, budgets


def _recurrence_class(branch: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    if row["previously_evicted"]:
        return "evicted_recovery"
    if int(branch["slot_evictions"]) == 0:
        return "no_eviction_revisit"
    return "resident_revisit"


def _mean_or_none(values: Sequence[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(present)) if present else None


def _range_or_none(values: Sequence[float | int | None]) -> list[float] | None:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return [min(present), max(present)] if present else None


def _seed_rows(branches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for branch in branches:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for recurrence in branch["recurrences"]:
            grouped.setdefault(_recurrence_class(branch, recurrence), []).append(recurrence)
        recurrence_summary = {}
        for category, values in sorted(grouped.items()):
            recurrence_summary[category] = {
                "task_count": len(values),
                "tasks": [str(row["task"]) for row in values],
                "metrics": {
                    metric: _mean_or_none([row[metric] for row in values])
                    for metric in RECURRENCE_METRICS
                },
            }
        rows.append({
            "method": branch["method"], "budget": branch["budget"], "seed": branch["seed"],
            "metrics": {metric: float(branch[metric]) for metric in FULL_STREAM_METRICS},
            "recurrences": recurrence_summary,
            "recurrence_observations": [
                {
                    "task": row["task"], "bout": row["bout"],
                    "category": _recurrence_class(branch, row),
                    "previously_evicted": row["previously_evicted"],
                    **({"observed_steps": row["observed_steps"]} if "observed_steps" in row else {}),
                    **{metric: row[metric] for metric in RECURRENCE_METRICS},
                }
                for row in branch["recurrences"]
            ],
            "counts": {field: branch[field] for field in COUNT_FIELDS},
            "capacity_max": dict(branch["capacity_max"]),
        })
    return rows


def _stats(per_seed: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    usable = [row for row in per_seed if row.get("value") is not None]
    values = [float(row["value"]) for row in usable]
    return {
        "seed_count": len(usable),
        "missing_seed_count": len(per_seed) - len(usable),
        "missing_seeds": [int(row["seed"]) for row in per_seed if row.get("value") is None],
        "per_seed": [{"seed": int(row["seed"]), "value": float(row["value"])} for row in usable],
        "mean": _mean_or_none(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "range": [min(values), max(values)] if values else None,
        "unit": "seed",
    }


def _summaries(
    seed_rows: Sequence[Mapping[str, Any]], methods: Sequence[str], budgets: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    categories = ("evicted_recovery", "resident_revisit", "no_eviction_revisit")
    for budget in budgets:
        result[budget] = {}
        for method in methods:
            selected = sorted(
                (row for row in seed_rows if row["budget"] == budget and row["method"] == method),
                key=lambda row: int(row["seed"]),
            )
            metrics = {}
            for metric in FULL_STREAM_METRICS:
                metrics[metric] = _stats(
                    [{"seed": row["seed"], "value": row["metrics"][metric]} for row in selected], metric
                )
            recurrence_summary = {}
            for category in categories:
                per_seed_recurrences = [
                    {"seed": row["seed"], **row["recurrences"].get(category, {})}
                    for row in selected if category in row["recurrences"]
                ]
                recurrence_summary[category] = {
                    "seed_count": len(per_seed_recurrences),
                    "task_count_total": sum(int(row["task_count"]) for row in per_seed_recurrences),
                    "per_seed": per_seed_recurrences,
                    "metrics": {
                        metric: _stats([
                            {"seed": row["seed"], "value": row["metrics"].get(metric)}
                            for row in per_seed_recurrences
                        ], metric)
                        for metric in RECURRENCE_METRICS
                    },
                }
            costs = {}
            for field in COUNT_FIELDS:
                costs[field] = _stats(
                    [{"seed": row["seed"], "value": row["counts"][field]} for row in selected], field
                )
            capacity = {}
            for field in CAPACITY_FIELDS:
                capacity[field] = _stats(
                    [{"seed": row["seed"], "value": row["capacity_max"][field]} for row in selected], field
                )
            result[budget][method] = {
                "seed_count": len(selected),
                "seeds": [int(row["seed"]) for row in selected],
                "metrics": metrics,
                "recurrences": recurrence_summary,
                "counts": costs,
                "capacity_max": capacity,
            }
    return result


def _matched_recurrence_comparison(
    seed_pairs: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    metrics: Sequence[str],
    *,
    evicted_only: bool,
) -> dict[str, Any]:
    per_seed = []
    for seed, candidate, baseline in seed_pairs:
        candidate_rows = {
            (str(row["task"]), int(row["bout"])): row
            for row in candidate["recurrence_observations"]
        }
        baseline_rows = {
            (str(row["task"]), int(row["bout"])): row
            for row in baseline["recurrence_observations"]
        }
        matched_keys = sorted(candidate_rows.keys() & baseline_rows.keys())
        if evicted_only:
            matched_keys = [
                key for key in matched_keys
                if candidate_rows[key]["previously_evicted"] is True
                and baseline_rows[key]["previously_evicted"] is True
            ]
        seed_detail: dict[str, Any] = {
            "seed": seed,
            "matched_task_bout_count": len(matched_keys),
            "matched_task_bouts": [
                {
                    "task": key[0], "bout": key[1],
                    "candidate_previously_evicted": candidate_rows[key]["previously_evicted"],
                    "independent_previously_evicted": baseline_rows[key]["previously_evicted"],
                }
                for key in matched_keys
            ],
            "metrics": {},
        }
        for metric in metrics:
            task_differences = []
            for key in matched_keys:
                candidate_value = candidate_rows[key].get(metric)
                baseline_value = baseline_rows[key].get(metric)
                difference = (
                    float(candidate_value) - float(baseline_value)
                    if candidate_value is not None and baseline_value is not None
                    else None
                )
                task_differences.append({
                    "task": key[0], "bout": key[1],
                    "candidate_value": candidate_value,
                    "independent_value": baseline_value,
                    "difference": difference,
                })
            present_differences = [
                row["difference"] for row in task_differences if row["difference"] is not None
            ]
            seed_detail["metrics"][metric] = {
                "matched_task_bout_count": len(task_differences),
                "observed_task_bout_count": len(present_differences),
                "mean_difference": _mean_or_none(present_differences),
                "per_task": task_differences,
            }
        per_seed.append(seed_detail)

    metric_summaries = {}
    for metric in metrics:
        usable = [
            {
                "seed": int(row["seed"]),
                "matched_task_bout_count": row["metrics"][metric]["observed_task_bout_count"],
                "mean_difference": row["metrics"][metric]["mean_difference"],
            }
            for row in per_seed if row["metrics"][metric]["mean_difference"] is not None
        ]
        values = [float(row["mean_difference"]) for row in usable]
        metric_summaries[metric] = {
            "seed_count": len(usable),
            "missing_seed_count": len(per_seed) - len(usable),
            "per_seed": usable,
            "matched_task_bout_count_total": sum(row["matched_task_bout_count"] for row in usable),
            "mean_difference": _mean_or_none(values),
            "difference_range": _range_or_none(values),
            "unit": "seed; each value averages matched task/bout differences within seed",
        }
    return {
        "per_seed": per_seed,
        "matched_task_bout_count_total": sum(row["matched_task_bout_count"] for row in per_seed),
        "metrics": metric_summaries,
    }


def _paired_differences(
    seed_rows: Sequence[Mapping[str, Any]], methods: Sequence[str], budgets: Sequence[str],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for budget in budgets:
        comparisons[budget] = {}
        baseline_rows = {
            int(row["seed"]): row for row in seed_rows
            if row["budget"] == budget and row["method"] == "independent"
        }
        for method in methods:
            candidates = {
                int(row["seed"]): row for row in seed_rows
                if row["budget"] == budget and row["method"] == method
            }
            common = sorted(candidates.keys() & baseline_rows.keys())
            detail: dict[str, Any] = {"baseline": "independent", "per_seed": []}
            for seed in common:
                item = {"seed": seed}
                for metric in FULL_STREAM_METRICS:
                    item[metric] = candidates[seed]["metrics"][metric] - baseline_rows[seed]["metrics"][metric]
                detail["per_seed"].append(item)
            detail["common_seeds"] = common
            detail["seed_count"] = len(common)
            detail["mean_difference"] = {
                metric: _mean_or_none([row[metric] for row in detail["per_seed"]])
                for metric in FULL_STREAM_METRICS
            }
            detail["difference_range"] = {
                metric: _range_or_none([row[metric] for row in detail["per_seed"]])
                for metric in FULL_STREAM_METRICS
            }
            seed_pairs = [
                (seed, candidates[seed], baseline_rows[seed])
                for seed in common
            ]
            detail["matched_eviction_recovery"] = _matched_recurrence_comparison(
                seed_pairs,
                ("recovery512_reward_per_step", "recovery1024_reward_per_step"),
                evicted_only=True,
            )
            detail["matched_all_recurrences"] = _matched_recurrence_comparison(
                seed_pairs, RECURRENCE_METRICS, evicted_only=False,
            )
            detail["inference"] = "descriptive seed-paired difference; no significance test"
            comparisons[budget][method] = detail
    return comparisons


def _render_figure(
    output: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    budgets: Sequence[str],
    paired_differences: Mapping[str, Any],
) -> dict[str, str]:
    colors = {
        method: matplotlib.colors.to_hex(plt.get_cmap("tab10" if len(methods) <= 10 else "tab20")(
            index % (10 if len(methods) <= 10 else 20)
        ))
        for index, method in enumerate(methods)
    }
    figure_width = max(13, len(methods) * 1.35) * len(budgets) / len(BUDGETS)
    figure, axes = plt.subplots(
        4, len(budgets), figsize=(figure_width, 12.6), squeeze=False,
    )
    metrics = (
        ("total_reward", "Full-stream total reward", None),
        ("reward_per_raw_step", "Reward per raw step", None),
        ("recovery512_reward_per_step", "512-step matched eviction recovery vs independent", "matched_evicted"),
        ("recovery1024_reward_per_step", "1,024-step matched eviction recovery vs independent", "matched_evicted"),
    )
    for column, budget in enumerate(budgets):
        selected = [row for row in seed_rows if row["budget"] == budget]
        for axis, (metric, title, category) in zip(axes[:, column], metrics):
            axis.set_title(f"{budget.title()} budget: {title}", fontsize=10, pad=8)
            axis.set_xticks(range(len(methods)), methods, rotation=22, ha="right")
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            if category == "matched_evicted":
                axis.axhline(0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
            for method_index, method in enumerate(methods):
                method_rows = [row for row in selected if row["method"] == method]
                if category is None:
                    points = [(int(row["seed"]), float(row["metrics"][metric])) for row in method_rows]
                else:
                    metric_summary = (
                        paired_differences[budget][method]["matched_eviction_recovery"]["metrics"][metric]
                    )
                    points = [
                        (int(row["seed"]), float(row["mean_difference"]))
                        for row in metric_summary["per_seed"]
                        if row["mean_difference"] is not None
                    ]
                if not points:
                    continue
                values = [value for _seed, value in points]
                average = float(mean(values))
                lower, upper = average - min(values), max(values) - average
                axis.bar(method_index, average, color=colors[method], alpha=0.72, width=0.66, zorder=2)
                axis.errorbar(method_index, average, yerr=[[lower], [upper]], color="#222222",
                              capsize=3, linewidth=1.1, linestyle="none", zorder=3)
                for point_index, (seed, value) in enumerate(sorted(points)):
                    jitter = (point_index - (len(points) - 1) / 2.0) * 0.07
                    axis.scatter(method_index + jitter, value, color=colors[method], edgecolor="#222222",
                                 linewidth=0.35, s=24, zorder=4)
            if category == "matched_evicted" and not any(
                paired_differences[budget][method]["matched_eviction_recovery"]["metrics"][metric]["per_seed"]
                for method in methods
            ):
                axis.text(0.5, 0.5, "No matched evicted task/bout observations", transform=axis.transAxes,
                          ha="center", va="center", fontsize=9, color="#555555")
            if category == "matched_evicted":
                axis.set_ylabel("Δ reward / raw step")
            else:
                axis.set_ylabel("Reward / raw step" if metric != "total_reward" else "Reward")
    handles = [Patch(facecolor=colors[method], alpha=0.72, label=method) for method in methods]
    figure.legend(handles=handles, loc="upper center", ncol=min(len(methods), 5), frameon=False,
                  bbox_to_anchor=(0.5, 0.985))
    figure.suptitle("Full-stream outcomes and matched eviction recovery", y=0.998, fontsize=14)
    figure.text(
        0.5, 0.012,
        "Bars show seed means; dots show seed values. Recovery panels show paired Δ vs independent with a zero reference.\n"
        "Matches require the same seed/task/bout and prior eviction in both arms.\n"
        "Whiskers show seed ranges, not confidence intervals; profile evidence comes from normal rollouts.\n"
        "Initial group identification uses one rollout before the first gradient update.",
        ha="center", va="bottom", fontsize=8,
    )
    figure.tight_layout(rect=(0.02, 0.11, 0.99, 0.94), h_pad=2.2, w_pad=1.6)
    png_path, pdf_path = output / "figure.png", output / "figure.pdf"
    figure.savefig(png_path, dpi=260, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"png": png_path.name, "pdf": pdf_path.name}


def _fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _fmt_mean_range(summary: Mapping[str, Any], digits: int = 4) -> str:
    value_range = summary.get("range")
    if value_range is None:
        return "n/a"
    return f"{_fmt(summary.get('mean'), digits)} [{_fmt(value_range[0], digits)}, {_fmt(value_range[1], digits)}]"


def _markdown_report(analysis: Mapping[str, Any]) -> str:
    budgets = analysis["budgets"]
    budget_names = ", ".join(budgets)
    lines = [
        "# Retention stream exploratory analysis",
        "",
        f"Analysis role: `{analysis['analysis_role']}`. Statistical unit: seed. Each seed is summarized before cross-seed means, ranges, or paired differences are calculated. Declared budgets are reported separately: {budget_names}.",
        "",
        "The full-stream outcomes are total reward and reward per raw step. Recurrence recovery is classified within each seed and is counted as eviction recovery only when the task was naturally evicted before it returned. Resident revisits and revisits with zero evictions remain separate categories. Tasks and bouts are not independent replicates.",
        "",
        "Profile evidence comes from normal rollouts; there is no independent probe, and initial group identification uses a rollout. The report does not assume an early cold-start gain. The primary recurrence comparisons are the later 512- and 1,024-step windows.",
        "",
        f"Cohort audit passed: `{analysis['audit']['passed']}`. Branches: {analysis['branch_count']}; seeds: {analysis['seed_count']}; cohort raw steps: {analysis['cohort_cost']['new_raw_steps']}; evaluation steps: {analysis['cohort_cost']['new_evaluation_steps']}; total interactions: {analysis['cohort_cost']['new_total_interactions']}; CPU s: {_fmt(analysis['cohort_runtime']['cpu_s'], 2)}; wall s: {_fmt(analysis['cohort_runtime']['wall_s'], 2)}.",
        "",
        "## Full-stream outcomes by budget",
        "",
    ]
    for budget in budgets:
        lines.extend([
            f"### {budget.title()} budget",
            "",
            "| method | seeds | total reward mean [range] | reward / raw step mean [range] |",
            "|---|---:|---:|---:|",
        ])
        for method in analysis["methods"]:
            summary = analysis["by_budget_method"][budget][method]
            total = summary["metrics"]["total_reward"]
            rate = summary["metrics"]["reward_per_raw_step"]
            lines.append(
                f"| {method} | {summary['seed_count']} | {_fmt(total['mean'])} "
                f"[{_fmt(total['min'])}, {_fmt(total['max'])}] | {_fmt(rate['mean'])} "
                f"[{_fmt(rate['min'])}, {_fmt(rate['max'])}] |"
            )
        lines.append("")

        lines.extend([
            f"#### {budget.title()} budget paired differences vs independent",
            "",
            "| method | metric | paired seeds | per-seed differences | mean difference [range] |",
            "|---|---|---|---|---:|",
        ])
        for method in analysis["methods"]:
            comparison = analysis["paired_differences"][budget][method]
            for metric in FULL_STREAM_METRICS:
                values = ", ".join(
                    f"{row['seed']}: {_fmt(row[metric])}" for row in comparison["per_seed"]
                )
                lines.append(
                    f"| {method} | {metric} | {', '.join(map(str, comparison['common_seeds'])) or 'none'} | "
                    f"{values or 'n/a'} | {_fmt(comparison['mean_difference'][metric])} "
                    f"[{_fmt((comparison['difference_range'][metric] or [None, None])[0])}, "
                    f"{_fmt((comparison['difference_range'][metric] or [None, None])[1])}] |"
                )
        lines.append("")

    lines.extend(["## Recurrence categories", ""])
    for budget in budgets:
        lines.extend([
            f"### {budget.title()} budget",
            "",
            "Each task is averaged inside its seed first. The reported cross-seed mean and range use those seed-level averages.",
            "",
            "| method | category | seed count | task count | first 256 mean [range] | recovery 512 mean [range] | recovery 1,024 mean [range] | first goal step mean [range] (n, missing) | first gradient step mean [range] |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in analysis["methods"]:
            method_summary = analysis["by_budget_method"][budget][method]
            for category in ("evicted_recovery", "resident_revisit", "no_eviction_revisit"):
                detail = method_summary["recurrences"][category]
                metric = detail["metrics"]
                lines.append(
                    f"| {method} | {category} | {detail['seed_count']} | {detail['task_count_total']} | "
                    f"{_fmt_mean_range(metric['first256_reward_per_step'])} | "
                    f"{_fmt_mean_range(metric['recovery512_reward_per_step'])} | "
                    f"{_fmt_mean_range(metric['recovery1024_reward_per_step'])} | "
                    f"{_fmt_mean_range(metric['first_goal_step'], 1)} "
                    f"(n={metric['first_goal_step']['seed_count']}, missing={metric['first_goal_step']['missing_seed_count']}) | "
                    f"{_fmt_mean_range(metric['first_gradient_step'], 1)} |"
                )
        lines.append("")

        lines.extend([
            f"#### {budget.title()} budget paired eviction-recovery differences vs independent",
            "",
            "Only the same seed and task/bout observed as previously evicted in both arms is included. Task/bout differences are averaged within seed before cross-seed summaries.",
            "",
            "| method | recovery window | matched seeds | matched task/bout observations | per-seed count and mean difference | mean per-seed difference [range] |",
            "|---|---:|---:|---:|---|---:|",
        ])
        for method in analysis["methods"]:
            comparison = analysis["paired_differences"][budget][method]
            matched = comparison["matched_eviction_recovery"]
            for metric, label in (("recovery512_reward_per_step", "512"), ("recovery1024_reward_per_step", "1,024")):
                stats = matched["metrics"][metric]
                per_seed = stats["per_seed"]
                per_seed_text = ", ".join(
                    f"{row['seed']}: n={row['matched_task_bout_count']}, {_fmt(row['mean_difference'])}"
                    for row in per_seed
                ) or "none"
                difference_range = stats["difference_range"] or [None, None]
                lines.append(
                    f"| {method} | {label} | {stats['seed_count']} | {stats['matched_task_bout_count_total']} | "
                    f"{per_seed_text} | {_fmt(stats['mean_difference'])} "
                    f"[{_fmt(difference_range[0])}, {_fmt(difference_range[1])}] |"
                )
        lines.append("")

        lines.extend([
            f"#### {budget.title()} budget all matched task/bout recurrence differences vs independent",
            "",
            "Supplementary comparison of the same task/bout in both arms regardless of eviction status. This includes visits retained in one arm and evicted in the other.",
            "",
            "| method | metric | matched seeds | matched task/bout observations | per-seed count and mean difference | mean per-seed difference [range] |",
            "|---|---|---:|---:|---|---:|",
        ])
        for method in analysis["methods"]:
            matched = analysis["paired_differences"][budget][method]["matched_all_recurrences"]
            for metric in RECURRENCE_METRICS:
                stats = matched["metrics"][metric]
                per_seed_text = ", ".join(
                    f"{row['seed']}: n={row['matched_task_bout_count']}, {_fmt(row['mean_difference'])}"
                    for row in stats["per_seed"]
                ) or "none"
                difference_range = stats["difference_range"] or [None, None]
                lines.append(
                    f"| {method} | {metric} | {stats['seed_count']} | {stats['matched_task_bout_count_total']} | "
                    f"{per_seed_text} | {_fmt(stats['mean_difference'])} "
                    f"[{_fmt(difference_range[0])}, {_fmt(difference_range[1])}] |"
                )
        lines.append("")

    lines.extend([
        "## Capacity, storage, and operation counts",
        "",
        "| budget | method | seeds | actor bytes max | full physical bytes max | package budget bytes | evidence bytes max | evictions | rollouts | minibatch updates | proposals / accepts | prior bindings | CPU s | wall s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for budget in budgets:
        for method in analysis["methods"]:
            row = analysis["by_budget_method"][budget][method]
            capacity, counts = row["capacity_max"], row["counts"]
            lines.append(
                f"| {budget} | {method} | {row['seed_count']} | "
                f"{_fmt(capacity['actor_bytes']['max'], 0)} | "
                f"{_fmt(capacity['full_physical_bytes']['max'], 0)} | "
                f"{_fmt(capacity['package_budget_bytes']['mean'], 0)} | "
                f"{_fmt(capacity['evidence_bytes']['max'], 0)} | "
                f"{_fmt(counts['slot_evictions']['mean'], 1)} | "
                f"{_fmt(counts['rollouts_trained']['mean'], 1)} | "
                f"{_fmt(counts['minibatch_updates']['mean'], 1)} | "
                f"{_fmt(counts['write_proposals']['mean'], 1)} / {_fmt(counts['write_accepts']['mean'], 1)} | "
                f"{_fmt(counts['prior_bindings']['mean'], 1)} | "
                f"{_fmt(counts['cpu_s']['mean'], 2)} | {_fmt(counts['wall_s']['mean'], 2)} |"
            )
    lines.extend([
        "",
        "## Per-seed records",
        "",
        "All negative values and task-level recurrence observations, including missing first-goal outcomes, are retained in `analysis.json`. The table below preserves each seed's full-stream reward rate.",
        "",
        "| budget | method | seed | total reward | reward / raw step | evictions |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(analysis["per_seed"], key=lambda item: (item["budget"], item["method"], item["seed"])):
        lines.append(
            f"| {row['budget']} | {row['method']} | {row['seed']} | "
            f"{_fmt(row['metrics']['total_reward'])} | {_fmt(row['metrics']['reward_per_raw_step'])} | "
            f"{row['counts']['slot_evictions']} |"
        )
    lines.extend([
        "",
        "## Figure",
        "",
        "![Full-stream reward and natural-eviction recovery](figure.png)",
        "",
        "This is an exploratory report. Seed-paired differences are descriptive; tasks, bouts, episodes, and transitions are not treated as independent seeds.",
        "",
        "## Software",
        "",
        "The visualization workflow follows the Matplotlib procedural guidance in Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065 (current arXiv version v2, revised 2 September 2026). [arXiv record](https://arxiv.org/abs/2609.00065).",
        "",
    ])
    return "\n".join(lines)


def analyze(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    summary_path, cohort, branches, methods, budgets = _load_completed_cohort(input_path)
    output = Path(output_path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preserve existing report output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    per_seed = _seed_rows(branches)
    paired_differences = _paired_differences(per_seed, methods, budgets)
    analysis: dict[str, Any] = {
        "schema": "retention_stream_analysis_v1",
        "analysis_role": "exploratory",
        "methods": list(methods),
        "budgets": list(budgets),
        "input": str(summary_path.resolve()),
        "statistical_unit": "seed",
        "seed_count": len({int(row["seed"]) for row in per_seed}),
        "branch_count": len(branches),
        "method_counts": {
            budget: {method: sum(row["budget"] == budget and row["method"] == method for row in per_seed)
                     for method in methods}
            for budget in budgets
        },
        "cohort_cost": dict(cohort["cohort_cost"]),
        "cohort_runtime": {
            "cpu_s": float(_finite_number(cohort["cpu_s"], "cohort.cpu_s")),
            "wall_s": float(_finite_number(cohort["wall_s"], "cohort.wall_s")),
        },
        "per_seed": per_seed,
        "by_budget_method": _summaries(per_seed, methods, budgets),
        "paired_differences": paired_differences,
        "audit": {"passed": True, "cohort": cohort["audit"], "all_branches_completed": True},
        "interpretation_limits": [
            "Budget arms are summarized separately; no low-versus-wide pooled estimate is produced.",
            "Each branch is one seed-level replicate. Recurrence tasks are averaged within seed before cross-seed summaries.",
            "Eviction recovery includes only recurrences marked previously_evicted. Resident and zero-eviction revisits are reported separately.",
            "Profile evidence comes from normal rollouts; there are no independent probes, and initial group identification consumes a rollout before the first gradient update.",
            "No early cold-start gain is presumed; later 512-step and 1,024-step recovery windows are the primary recurrence comparisons.",
            "Paired differences are descriptive and do not include significance tests.",
            "Exploratory cohort results do not establish confirmatory effects.",
        ],
    }
    analysis["figure"] = _render_figure(output, per_seed, methods, budgets, paired_differences)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_markdown_report(analysis), encoding="utf-8")
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed retention cohort directory or summary.json")
    parser.add_argument("--output", type=Path, required=True, help="new or empty report directory")
    args = parser.parse_args(argv)
    analysis = analyze(args.input, args.output)
    print(json.dumps({
        "analysis_role": analysis["analysis_role"],
        "seeds": analysis["seed_count"],
        "branches": analysis["branch_count"],
        "audit_passed": analysis["audit"]["passed"],
        "output": str(Path(args.output).resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

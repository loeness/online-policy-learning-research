"""Summarize a completed consolidation-retention cohort without rerunning it."""

from __future__ import annotations

import argparse
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

EXPECTED_FORMAT = "consolidation_retention_stage1_v1"
ANALYSIS_FORMAT = "consolidation_retention_analysis_v1"
METRICS = ("mean_signed_return", "mean_goal_rate", "positive_success_rate")
INVARIANCE_METRICS = (
    "raw_logit_max_error",
    "raw_probability_max_error",
    "int8_logit_max_error",
    "int8_mean_KL",
)
CAPACITY_METRICS = (
    "actor_bytes",
    "checkpoint_bytes",
    "foundation_bytes",
    "full_persistent_bytes",
    "package_budget_bytes",
)
COMPARISONS = (
    ("mean_minus_local_only", "mean", "local_only"),
    ("weighted_minus_local_only", "weighted", "local_only"),
    ("random_minus_local_only", "random", "local_only"),
    ("weighted_minus_random", "weighted", "random"),
    ("weighted_minus_mean", "weighted", "mean"),
)
METHOD_DESCRIPTIONS = {
    "local_only": "删除全部局部修正，仅剩 W0 + 零共享；这是本 assay 的基线",
    "mean": "有效参数的简单平均",
    "weighted": "按来源等权归一化的锚点最小二乘",
    "random": "随机共享矩阵；Frobenius 范数匹配 weighted",
}
METHOD_LABELS = {
    "local_only": "W0 only",
    "mean": "Mean",
    "weighted": "Evidence weighted",
    "random": "Random",
}
TARGET_ROLES = {
    "A0+": "保留",
    "A2+": "探索迁移",
    "B2+": "错误复用诊断",
    "A0-": "负目标错误复用诊断（仅看 signed return）",
}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_flags(values: Any, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != 8:
        raise ValueError(f"{label} must be a list of exactly 8 values")
    result = [
        float(value)
        if isinstance(value, (bool, np.bool_))
        else _finite_number(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    if any(value not in (0.0, 1.0) for value in result):
        raise ValueError(f"{label} values must be boolean or 0/1")
    return result


def _episode_metrics(target: dict[str, Any], label: str) -> dict[str, float]:
    returns = target.get("episode_returns")
    lengths = target.get("episode_lengths")
    if not isinstance(returns, list) or len(returns) != 8:
        raise ValueError(f"{label}.episode_returns must be a list of exactly 8 values")
    if not isinstance(lengths, list) or len(lengths) != 8:
        raise ValueError(f"{label}.episode_lengths must be a list of exactly 8 values")
    return_values = [
        _finite_number(value, f"{label}.episode_returns[{index}]")
        for index, value in enumerate(returns)
    ]
    length_values = [
        _finite_number(value, f"{label}.episode_lengths[{index}]")
        for index, value in enumerate(lengths)
    ]
    if any(value <= 0 or not value.is_integer() for value in length_values):
        raise ValueError(f"{label}.episode_lengths must contain positive integers")
    goal_flags = _positive_flags(target.get("goal_flags"), f"{label}.goal_flags")
    positive_flags = _positive_flags(
        target.get("positive_success_flags"), f"{label}.positive_success_flags"
    )
    _finite_number(target.get("evaluation_steps"), f"{label}.evaluation_steps")
    return {
        "mean_signed_return": float(np.mean(return_values)),
        "mean_goal_rate": float(np.mean(goal_flags)),
        "positive_success_rate": float(np.mean(positive_flags)),
    }


def validate_summary(summary: dict[str, Any]) -> tuple[list[int], list[str], list[str]]:
    if summary.get("format") != EXPECTED_FORMAT:
        raise ValueError(f"summary format must be {EXPECTED_FORMAT!r}")
    if summary.get("status") != "completed":
        raise ValueError("summary status must be 'completed'")
    if summary.get("analysis_role") != "exploratory":
        raise ValueError("summary analysis_role must be 'exploratory'")
    seeds = summary.get("seeds")
    methods = summary.get("methods")
    targets = summary.get("targets")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("summary.seeds must be a nonempty list of integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("summary.seeds must be unique")
    if not isinstance(methods, list) or methods != ["local_only", "mean", "weighted", "random"]:
        raise ValueError("summary.methods must list local_only, mean, weighted, random in order")
    if not isinstance(targets, list) or targets != ["A0+", "A2+", "B2+", "A0-"]:
        raise ValueError("summary.targets must list A0+, A2+, B2+, A0- in order")

    audit = summary.get("audit")
    if not isinstance(audit, dict) or not isinstance(audit.get("passed"), bool):
        raise ValueError("summary.audit must contain a boolean passed field")
    if not isinstance(audit.get("failures"), list):
        raise ValueError("summary.audit.failures must be a list")
    if not isinstance(summary.get("source_audits"), list):
        raise ValueError("summary.source_audits must be a list")
    cost = summary.get("cohort_cost")
    required_costs = (
        "source_training_steps_reused",
        "new_training_steps",
        "new_evaluation_steps",
        "new_total_interactions",
    )
    if not isinstance(cost, dict):
        raise ValueError("summary.cohort_cost must be an object")
    for key in required_costs:
        value = cost.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"summary.cohort_cost.{key} must be a nonnegative integer")

    records = summary.get("records")
    if not isinstance(records, list) or len(records) != len(seeds) * len(methods):
        raise ValueError("summary.records must contain one record for every seed and method")
    seen: set[tuple[int, str]] = set()
    for record_index, record in enumerate(records):
        label = f"records[{record_index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        seed = record.get("seed")
        method = record.get("method")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(method, str)
            or seed not in seeds
            or method not in methods
        ):
            raise ValueError(f"{label} has an unknown seed or method")
        key = (seed, method)
        if key in seen:
            raise ValueError(f"duplicate record for seed={seed}, method={method}")
        seen.add(key)
        if not isinstance(record.get("path"), str):
            raise ValueError(f"{label}.path must be a string")
        if not isinstance(record.get("proposal"), dict):
            raise ValueError(f"{label}.proposal must be an object")
        invariance = record.get("invariance")
        if not isinstance(invariance, dict):
            raise ValueError(f"{label}.invariance must be an object")
        for metric in INVARIANCE_METRICS:
            _finite_number(invariance.get(metric), f"{label}.invariance.{metric}")
        purge_audit = record.get("purge_audit")
        if not isinstance(purge_audit, dict) or not isinstance(purge_audit.get("passed"), bool):
            raise ValueError(f"{label}.purge_audit must contain a boolean passed field")
        if not isinstance(purge_audit.get("checks"), dict):
            raise ValueError(f"{label}.purge_audit.checks must be an object")
        capacity = record.get("capacity")
        if not isinstance(capacity, dict):
            raise ValueError(f"{label}.capacity must be an object")
        for metric in CAPACITY_METRICS:
            value = capacity.get(metric)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label}.capacity.{metric} must be a nonnegative integer")
        targets_data = record.get("targets")
        if not isinstance(targets_data, dict) or set(targets_data) != set(targets):
            raise ValueError(f"{label}.targets must include each target in the schema")
        for task in targets:
            if not isinstance(targets_data[task], dict):
                raise ValueError(f"{label}.targets.{task} must be an object")
            _episode_metrics(targets_data[task], f"{label}.targets.{task}")
        for key_name in ("actor_frozen_during_eval", "source_unchanged"):
            if not isinstance(record.get(key_name), bool):
                raise ValueError(f"{label}.{key_name} must be boolean")
        for key_name in ("cpu_s", "wall_s"):
            if _finite_number(record.get(key_name), f"{label}.{key_name}") < 0:
                raise ValueError(f"{label}.{key_name} must be nonnegative")
    if seen != {(seed, method) for seed in seeds for method in methods}:
        raise ValueError("summary.records is missing a seed and method pair")
    return list(seeds), list(methods), list(targets)


def _validate_postrun_audit(
    postrun_audit: dict[str, Any] | None,
    summary_records: list[dict[str, Any]],
    seeds: list[int],
    methods: list[str],
) -> dict[str, Any] | None:
    if postrun_audit is None:
        return None
    if not isinstance(postrun_audit, dict):
        raise ValueError("postrun_audit.json must contain a JSON object")
    correction = postrun_audit.get("capacity_correction", postrun_audit)
    if not isinstance(correction, dict) or not isinstance(correction.get("records"), list):
        raise ValueError("postrun_audit must contain capacity_correction.records")
    expected_records = {(seed, method) for seed in seeds for method in methods}
    original_capacity = {
        (record["seed"], record["method"]): record["capacity"]
        for record in summary_records
    }
    seen: set[tuple[int, str]] = set()
    required_bytes = (
        "checkpoint_bytes",
        "foundation_bytes",
        "actor_sidecar_bytes",
        "full_physical_bytes",
        "package_budget_bytes",
    )
    for index, record in enumerate(correction["records"]):
        label = f"postrun_audit.capacity_correction.records[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        seed, method = record.get("seed"), record.get("method")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(method, str)
            or (seed, method) not in expected_records
        ):
            raise ValueError(f"{label} has an unknown seed or method")
        key = (seed, method)
        if key in seen:
            raise ValueError(f"duplicate postrun capacity record for seed={seed}, method={method}")
        seen.add(key)
        for field in required_bytes:
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label}.{field} must be a nonnegative integer")
        if not isinstance(record.get("passed"), bool):
            raise ValueError(f"{label}.passed must be boolean")
        original_full = record.get("original_full_persistent_bytes")
        if isinstance(original_full, bool) or not isinstance(original_full, int) or original_full < 0:
            raise ValueError(f"{label}.original_full_persistent_bytes must be a nonnegative integer")
        if record["full_physical_bytes"] != sum(record[field] for field in required_bytes[:3]):
            raise ValueError(f"{label}.full_physical_bytes must equal its three physical file sizes")
        original = original_capacity[key]
        for field in ("checkpoint_bytes", "foundation_bytes", "package_budget_bytes"):
            if record[field] != original[field]:
                raise ValueError(f"{label}.{field} does not match the original capacity record")
        if original_full != original["full_persistent_bytes"]:
            raise ValueError(
                f"{label}.original_full_persistent_bytes does not match the original capacity record"
            )
        if record["actor_sidecar_bytes"] != original["actor_bytes"]:
            raise ValueError(f"{label}.actor_sidecar_bytes does not match original actor_bytes")
        if record["passed"] != (record["full_physical_bytes"] <= record["package_budget_bytes"]):
            raise ValueError(f"{label}.passed does not match its physical size and package budget")
    if seen != expected_records:
        raise ValueError("postrun_audit.capacity_correction.records must include every seed and method")
    if not isinstance(postrun_audit.get("passed"), bool):
        raise ValueError("postrun_audit.passed must be boolean")
    return postrun_audit


def _postrun_capacity_records(postrun_audit: dict[str, Any]) -> list[dict[str, Any]]:
    correction = postrun_audit.get("capacity_correction", postrun_audit)
    return correction["records"]


def analyze_summary(
    summary: dict[str, Any], postrun_audit: dict[str, Any] | None = None
) -> dict[str, Any]:
    seeds, methods, targets = validate_summary(summary)
    postrun_audit = _validate_postrun_audit(postrun_audit, summary["records"], seeds, methods)
    by_pair = {(record["seed"], record["method"]): record for record in summary["records"]}

    target_analyses: list[dict[str, Any]] = []
    for task in targets:
        method_summaries: list[dict[str, Any]] = []
        seed_values: dict[str, dict[int, dict[str, float]]] = {}
        for method in methods:
            values_for_method: dict[int, dict[str, float]] = {}
            for seed in seeds:
                source = by_pair[(seed, method)]["targets"][task]
                values_for_method[seed] = _episode_metrics(source, f"{seed}/{method}/{task}")
            seed_values[method] = values_for_method
            metrics: dict[str, Any] = {}
            for metric in METRICS:
                values = [values_for_method[seed][metric] for seed in seeds]
                metrics[metric] = {
                    "mean_across_seeds": float(np.mean(values)),
                    "min_across_seeds": float(np.min(values)),
                    "max_across_seeds": float(np.max(values)),
                    "per_seed": [
                        {"seed": seed, "value": values_for_method[seed][metric]}
                        for seed in seeds
                    ],
                }
            method_summaries.append({"method": method, "metrics": metrics})

        paired_contrasts: list[dict[str, Any]] = []
        for name, left_method, right_method in COMPARISONS:
            metric_contrasts: dict[str, Any] = {}
            for metric in METRICS:
                per_seed = [
                    {
                        "seed": seed,
                        "difference": (
                            seed_values[left_method][seed][metric]
                            - seed_values[right_method][seed][metric]
                        ),
                    }
                    for seed in seeds
                ]
                differences = [row["difference"] for row in per_seed]
                metric_contrasts[metric] = {
                    "mean_across_seeds": float(np.mean(differences)),
                    "per_seed": per_seed,
                }
            paired_contrasts.append(
                {
                    "name": name,
                    "left_method": left_method,
                    "right_method": right_method,
                    "metrics": metric_contrasts,
                }
            )
        target_analyses.append(
            {
                "target": task,
                "method_summaries": method_summaries,
                "paired_contrasts": paired_contrasts,
            }
        )

    records = summary["records"]
    invariance_max = {
        metric: max(float(record["invariance"][metric]) for record in records)
        for metric in INVARIANCE_METRICS
    }
    purge_failures = [
        {
            "seed": record["seed"],
            "method": record["method"],
            "checks": record["purge_audit"]["checks"],
        }
        for record in records
        if not record["purge_audit"]["passed"]
    ]
    purge_audits = [
        {
            "seed": record["seed"],
            "method": record["method"],
            "passed": record["purge_audit"]["passed"],
            "checks": record["purge_audit"]["checks"],
        }
        for record in records
    ]
    frozen_failures = [
        {"seed": record["seed"], "method": record["method"]}
        for record in records
        if not record["actor_frozen_during_eval"]
    ]
    unchanged_failures = [
        {"seed": record["seed"], "method": record["method"]}
        for record in records
        if not record["source_unchanged"]
    ]
    capacities = [
        {
            "seed": record["seed"],
            "method": record["method"],
            **{metric: record["capacity"][metric] for metric in CAPACITY_METRICS},
        }
        for record in records
    ]

    all_return_differences = [
        row["difference"]
        for target in target_analyses
        for row in next(
            row["metrics"]["mean_signed_return"]["per_seed"]
            for row in target["paired_contrasts"]
            if row["name"] == "weighted_minus_mean"
        )
    ]
    all_random_differences = [
        row["difference"]
        for target in target_analyses
        for row in next(
            row["metrics"]["mean_signed_return"]["per_seed"]
            for row in target["paired_contrasts"]
            if row["name"] == "weighted_minus_random"
        )
    ]

    return {
        "format": ANALYSIS_FORMAT,
        "analysis_role": summary["analysis_role"],
        "source_format": summary["format"],
        "status": summary["status"],
        "seeds": seeds,
        "methods": methods,
        "targets": target_analyses,
        "cohort_cost": dict(summary["cohort_cost"]),
        "audit": {
            "summary_passed": summary["audit"]["passed"],
            "summary_failures": list(summary["audit"]["failures"]),
            "purge_failure_count": len(purge_failures),
            "purge_failures": purge_failures,
            "purge_audits": purge_audits,
            "actor_frozen_failure_count": len(frozen_failures),
            "actor_frozen_failures": frozen_failures,
            "source_unchanged_failure_count": len(unchanged_failures),
            "source_unchanged_failures": unchanged_failures,
            "source_audits": summary["source_audits"],
        },
        "invariance_max_across_records": invariance_max,
        "capacity_by_seed_and_method": capacities,
        "postrun_audit": postrun_audit,
        "interpretation": {
            "unit": "seed; episode means are computed within each seed before aggregation",
            "aggregate_range": "minimum and maximum of the seed-level means",
            "contrasts": "paired within seed; descriptive only, no significance testing",
            "weighted_minus_mean_signed_return_mean": float(np.mean(all_return_differences)),
            "weighted_minus_random_signed_return_mean": float(np.mean(all_random_differences)),
            "random_norm_reference": "weighted Frobenius norm",
            "cross_target_aggregate_use": (
                "descriptive appendix only; includes wrong-reuse diagnostic targets and is not an algorithm selection criterion"
            ),
        },
    }


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _direction(value: float) -> str:
    if value > 0:
        return "正"
    if value < 0:
        return "负"
    return "零"


def _sign_count_text(rows: list[dict[str, Any]]) -> str:
    positive = sum(row["difference"] > 0 for row in rows)
    negative = sum(row["difference"] < 0 for row in rows)
    zero = len(rows) - positive - negative
    return f"{positive}/{len(rows)} 为正、{negative}/{len(rows)} 为负、{zero}/{len(rows)} 为零"


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _get_target(analysis: dict[str, Any], task: str) -> dict[str, Any]:
    return next(row for row in analysis["targets"] if row["target"] == task)


def _get_method(target: dict[str, Any], method: str) -> dict[str, Any]:
    return next(row for row in target["method_summaries"] if row["method"] == method)


def _get_contrast(target: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in target["paired_contrasts"] if row["name"] == name)


def _overall_contrast(analysis: dict[str, Any], name: str, metric: str) -> float:
    seed_differences = [
        row["difference"]
        for target in analysis["targets"]
        for row in _get_contrast(target, name)["metrics"][metric]["per_seed"]
    ]
    return float(np.mean(seed_differences))


def build_report_markdown(analysis: dict[str, Any]) -> str:
    seeds = analysis["seeds"]
    lines = [
        "# 共享参数巩固保留探针：Stage 1",
        "",
        "## 结果摘要",
        "",
        f"- cohort 状态：`{analysis['status']}`；分析角色：`{analysis['analysis_role']}`。",
        f"- 随机单位是 seed（{len(seeds)} 个），不是 episode 或 transition。每个 seed 内先对 8 个 episode 求均值，再跨 seed 报告均值和最小值至最大值。",
        "- 表中配对差值按同一 seed 计算，均为描述统计；不做显著性检验或因果主张。",
        "- `random` 的共享矩阵 Frobenius 范数匹配 `weighted`，不匹配 `mean`。",
        "- 方法含义：" + "；".join(f"`{method}`：{description}" for method, description in METHOD_DESCRIPTIONS.items()) + "。",
        "",
        "| 目标 | 方法 | signed return：跨 seed 均值 [最小, 最大] | goal rate：跨 seed 均值 [最小, 最大] | positive success：跨 seed 均值 [最小, 最大]（A0−仅为定义性零值） |",
        "|---|---|---:|---:|---:|",
    ]
    for target in analysis["targets"]:
        for method in target["method_summaries"]:
            metrics = method["metrics"]
            cells = []
            for metric in METRICS:
                stats = metrics[metric]
                cells.append(
                    f"{_fmt(stats['mean_across_seeds'])} "
                    f"[{_fmt(stats['min_across_seeds'])}, {_fmt(stats['max_across_seeds'])}]"
                )
            lines.append(
                f"| {target['target']} | `{method['method']}` | {cells[0]} | {cells[1]} | {cells[2]} |"
            )

    lines.extend(
        [
            "",
            "## 逐 seed 对照与目标解释",
            "",
            "下表直接比较 Mean、Evidence weighted 各自相对 W0 only 的 seed 配对 signed-return 差值；“方向一致”表示两种共享方法在该 seed 都高于、都低于或都等于基线。",
            "",
            "| 目标与 assay 角色 | Seed | Mean − W0 only | Evidence weighted − W0 only | 两方法方向一致 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for task in ("A0+", "A2+", "B2+", "A0-"):
        target = _get_target(analysis, task)
        mean_rows = _get_contrast(target, "mean_minus_local_only")["metrics"][
            "mean_signed_return"
        ]["per_seed"]
        weighted_rows = _get_contrast(target, "weighted_minus_local_only")["metrics"][
            "mean_signed_return"
        ]["per_seed"]
        mean_by_seed = {row["seed"]: row["difference"] for row in mean_rows}
        weighted_by_seed = {row["seed"]: row["difference"] for row in weighted_rows}
        for seed in seeds:
            mean_difference = mean_by_seed[seed]
            weighted_difference = weighted_by_seed[seed]
            same_direction = _direction(mean_difference) == _direction(weighted_difference)
            lines.append(
                f"| {task}（{TARGET_ROLES[task]}） | {seed} | {_signed(mean_difference, 4)} | "
                f"{_signed(weighted_difference, 4)} | {'是' if same_direction else '否'} |"
            )

    target_return_pairs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task in ("A0+", "A2+", "B2+", "A0-"):
        target = _get_target(analysis, task)
        target_return_pairs[task] = {
            "mean": _get_contrast(target, "mean_minus_local_only")["metrics"][
                "mean_signed_return"
            ]["per_seed"],
            "weighted": _get_contrast(target, "weighted_minus_local_only")["metrics"][
                "mean_signed_return"
            ]["per_seed"],
            "weighted_minus_mean": _get_contrast(target, "weighted_minus_mean")["metrics"][
                "mean_signed_return"
            ]["per_seed"],
        }
    positive_weighted_mean = [
        row
        for task in ("A0+", "A2+")
        for row in target_return_pairs[task]["weighted_minus_mean"]
    ]
    lower_count = sum(row["difference"] < 0 for row in positive_weighted_mean)
    positive_seed_cell_count = len(positive_weighted_mean)
    a0_mean_positive = sum(row["difference"] > 0 for row in target_return_pairs["A0+"]["mean"])
    a0_weighted_positive = sum(
        row["difference"] > 0 for row in target_return_pairs["A0+"]["weighted"]
    )
    a2_mean_by_seed = {
        row["seed"]: row["difference"] for row in target_return_pairs["A2+"]["mean"]
    }
    a2_weighted_by_seed = {
        row["seed"]: row["difference"] for row in target_return_pairs["A2+"]["weighted"]
    }
    a2_same_direction_seeds = [
        seed
        for seed in seeds
        if _direction(a2_mean_by_seed[seed]) == _direction(a2_weighted_by_seed[seed])
    ]
    a2_both_positive_seeds = [
        seed
        for seed in seeds
        if a2_mean_by_seed[seed] > 0 and a2_weighted_by_seed[seed] > 0
    ]
    a2_both_negative_seeds = [
        seed
        for seed in seeds
        if a2_mean_by_seed[seed] < 0 and a2_weighted_by_seed[seed] < 0
    ]
    b2_both_negative = sum(
        mean_row["difference"] < 0 and weighted_row["difference"] < 0
        for mean_row, weighted_row in zip(
            target_return_pairs["B2+"]["mean"], target_return_pairs["B2+"]["weighted"]
        )
    )
    a0_negative_both = sum(
        mean_row["difference"] < 0 and weighted_row["difference"] < 0
        for mean_row, weighted_row in zip(
            target_return_pairs["A0-"]["mean"], target_return_pairs["A0-"]["weighted"]
        )
    )
    a0_weighted_negative_seeds = [
        row["seed"]
        for row in target_return_pairs["A0+"]["weighted"]
        if row["difference"] < 0
    ]
    a0_weighted_mean_lower = sum(
        row["difference"] < 0 for row in target_return_pairs["A0+"]["weighted_minus_mean"]
    )
    a2_weighted_mean_lower = sum(
        row["difference"] < 0 for row in target_return_pairs["A2+"]["weighted_minus_mean"]
    )
    a0_weighted_negative_text = (
        ", ".join(str(seed) for seed in a0_weighted_negative_seeds)
        if a0_weighted_negative_seeds
        else "无"
    )
    a2_common_positive_text = (
        ", ".join(str(seed) for seed in a2_both_positive_seeds) if a2_both_positive_seeds else "无"
    )
    a2_common_negative_text = (
        ", ".join(str(seed) for seed in a2_both_negative_seeds) if a2_both_negative_seeds else "无"
    )
    if lower_count == positive_seed_cell_count:
        positive_weighted_mean_summary = "Mean 在每个 target-seed 配对都高于 Evidence weighted"
    elif lower_count == 0:
        positive_weighted_mean_summary = "Evidence weighted 没有在 target-seed 配对上低于 Mean"
    else:
        positive_weighted_mean_summary = (
            f"Evidence weighted 在 {lower_count}/{positive_seed_cell_count} 个 target-seed 配对低于 Mean"
        )
    lines.extend(
        [
            "",
            f"- **A0+ 保留：** Mean 相对 W0 only 的逐 seed signed-return 差值 {_sign_count_text(target_return_pairs['A0+']['mean'])}；Evidence weighted 的差值 {_sign_count_text(target_return_pairs['A0+']['weighted'])}，负向 seed 为 {a0_weighted_negative_text}。Evidence weighted 相对 Mean 在 A0+ 的 {a0_weighted_mean_lower}/{len(seeds)} seeds signed return 较低。",
            f"- **A2+ 探索迁移：** Mean 与 Evidence weighted 相对 W0 only 在 {len(a2_same_direction_seeds)}/{len(seeds)} seeds 方向一致；两者都为正的 seed 是 {a2_common_positive_text}，都为负的是 {a2_common_negative_text}。Evidence weighted 相对 Mean 在 {a2_weighted_mean_lower}/{len(seeds)} seeds 较低。A2+ 是 exploratory validation，不是独立测试。",
            f"- **B2+ 错误复用诊断：** Mean 与 Evidence weighted 相对 W0 only 在 {b2_both_negative}/{len(seeds)} seeds 同为负。Mean 的差值为 {_sign_count_text(target_return_pairs['B2+']['mean'])}；Evidence weighted 为 {_sign_count_text(target_return_pairs['B2+']['weighted'])}。该目标呈现固定 parent 的错误复用结果，不参与一般算法选择。",
            f"- **A0− 错误复用诊断：** 这是负目标；goal 不是成功。positive_success_rate 在 A0− 没有解释意义，主要看 signed return；Mean 与 Evidence weighted 相对 W0 only 在 {a0_negative_both}/{len(seeds)} seeds 同为负。Mean 的差值为 {_sign_count_text(target_return_pairs['A0-']['mean'])}；Evidence weighted 为 {_sign_count_text(target_return_pairs['A0-']['weighted'])}。",
            f"- 在正奖励目标 A0+、A2+，Evidence weighted − Mean 的 signed-return 差值在 {lower_count}/{positive_seed_cell_count} 个 target-seed 配对中为负（A0+ {a0_weighted_mean_lower}/{len(seeds)}、A2+ {a2_weighted_mean_lower}/{len(seeds)}）。{positive_weighted_mean_summary}；这是小样本描述，不作显著性主张。",
            "- B2+ 与 A0-（负目标）是刻意设置的错误复用目标。把它们与正目标合成一个跨目标平均会把保留/迁移和诊断性反例混在一起，因此**跨目标平均不用于选择算法**。",
            "",
            "## 同 seed 配对差值（完整数据）",
            "",
            "下表保留每个目标、每种对比、每项指标的全部 seed 差值和跨 seed 均值；差值定义为左侧方法减右侧方法。A0− 的 positive-success 数值为任务定义下的零值，不代表成功度量。",
        ]
    )
    for target in analysis["targets"]:
        target_note = "（负目标；主要看 signed return）" if target["target"] == "A0-" else f"（{TARGET_ROLES[target['target']]}）"
        lines.extend(
            [
                "",
                f"### {target['target']} {target_note}",
                "",
                "| 对比 | 指标 | " + " | ".join(f"seed {seed}" for seed in seeds) + " | 跨 seed 均值 |",
                "|---|---|" + "---:|" * (len(seeds) + 1),
            ]
        )
        for contrast in target["paired_contrasts"]:
            for metric in METRICS:
                entry = contrast["metrics"][metric]
                per_seed = {row["seed"]: row["difference"] for row in entry["per_seed"]}
                lines.append(
                    f"| `{contrast['name']}` | `{metric}` | "
                    + " | ".join(_signed(per_seed[seed]) for seed in seeds)
                    + f" | {_signed(entry['mean_across_seeds'])} |"
                )

    weighted_vs_mean = analysis["interpretation"]["weighted_minus_mean_signed_return_mean"]
    weighted_vs_random = analysis["interpretation"]["weighted_minus_random_signed_return_mean"]
    lines.extend(
        [
            "",
            "### 附属描述：跨目标算术平均（不用于选择算法）",
            "",
            f"将全部四个目标 × 三个 seed 配对后，weighted − Mean signed-return 算术平均为 {_signed(weighted_vs_mean)}，weighted − Random 为 {_signed(weighted_vs_random)}。这些数字仅供完整性记录；B2+、A0− 是刻意设置的错误复用诊断，不能用这个合并值判定哪种方法总体更优。Random 的矩阵范数匹配 weighted，不匹配 Mean。",
            "",
        ]
    )

    cost = analysis["cohort_cost"]
    audit = analysis["audit"]
    lines.extend(
        [
            "",
            "## 审计、成本与容量",
            "",
            f"summary audit：**{'通过' if audit['summary_passed'] else '失败'}**；purge audit 失败分支：{audit['purge_failure_count']} / {len(seeds) * len(analysis['methods'])}；eval 冻结失败分支：{audit['actor_frozen_failure_count']}；source unchanged 失败分支：{audit['source_unchanged_failure_count']}。",
            f"source audit 条目：{len(audit['source_audits'])}。最大 invariance 误差：`{_json_cell(analysis['invariance_max_across_records'])}`。",
            "Checkpoint restore 不会自动设置 `requires_grad=False`；runner 在 evaluation 前显式 freeze actor，并用 evaluation 前后权重 hash 验证未变。",
            "",
            "| Seed | 方法 | purge audit | purge checks |",
            "|---:|---|---|---|",
        ]
    )
    record_audit_rows = []
    for purge in audit["purge_audits"]:
        record_audit_rows.append(
            f"| {purge['seed']} | `{purge['method']}` | "
            f"{'通过' if purge['passed'] else '失败'} | `{_json_cell(purge['checks'])}` |"
        )
    lines.extend(record_audit_rows)
    summary_failures = audit["summary_failures"]
    if summary_failures:
        lines.extend(["", "summary audit failures:"])
        lines.extend(f"- {failure}" for failure in summary_failures)
    if audit["purge_failures"]:
        lines.extend(["", "purge audit failures:"])
        lines.extend(
            f"- seed {failure['seed']} / `{failure['method']}`: `{_json_cell(failure['checks'])}`"
            for failure in audit["purge_failures"]
        )
    if audit["source_audits"]:
        lines.extend(["", "source audit entries:"])
        lines.extend(f"- `{_json_cell(row)}`" for row in audit["source_audits"])

    lines.extend(
        [
            "",
            f"复用 source training steps：{cost['source_training_steps_reused']}；新增 training steps：{cost['new_training_steps']}；新增 evaluation steps：{cost['new_evaluation_steps']}；新增总交互：{cost['new_total_interactions']}。evaluation observer clone 的成本计入 cohort；本报告不启动新的 PPO/training。",
            "",
            "下表保留原容量日志口径的每个分支字节数；`full_persistent_bytes` 不包括同时保存的 actor `.actor.bin` sidecar。",
            "",
            "| Seed | 方法 | Actor bytes | Checkpoint bytes | Foundation bytes | Full persistent bytes（原日志口径） | Package budget bytes |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for capacity in analysis["capacity_by_seed_and_method"]:
        lines.append(
            f"| {capacity['seed']} | `{capacity['method']}` | {capacity['actor_bytes']} | "
            f"{capacity['checkpoint_bytes']} | {capacity['foundation_bytes']} | "
            f"{capacity['full_persistent_bytes']} | {capacity['package_budget_bytes']} |"
        )

    postrun_audit = analysis.get("postrun_audit")
    lines.extend(["", "## 事后物理文件核算", ""])
    if postrun_audit is None:
        lines.append("输入目录没有 `postrun_audit.json`，因此本报告只保留原始容量日志口径。")
    else:
        corrected_records = _postrun_capacity_records(postrun_audit)
        old_full_values = sorted(
            {int(row["full_persistent_bytes"]) for row in analysis["capacity_by_seed_and_method"]}
        )
        corrected_full_values = sorted(
            {int(row["full_physical_bytes"]) for row in corrected_records}
        )
        sidecar_values = sorted({int(row["actor_sidecar_bytes"]) for row in corrected_records})
        passed_count = sum(row["passed"] for row in corrected_records)
        old_text = ", ".join(str(value) for value in old_full_values)
        corrected_text = ", ".join(str(value) for value in corrected_full_values)
        sidecar_text = ", ".join(str(value) for value in sidecar_values)
        lines.append(
            f"事后核算确认：旧日志 `full_persistent_bytes` 为 {old_text} B，只计 W0 + checkpoint，未计同时生成的 actor `.actor.bin` sidecar；新 `full_physical_bytes` 为 {corrected_text} B，已计入 {sidecar_text} B sidecar。物理文件总量与 262144 B package budget 比较；新旧口径分别保留，不覆盖原始 summary。"
        )
        lines.extend(
            [
                f"事后物理文件审计：**{'通过' if postrun_audit['passed'] else '失败'}**；容量检查通过 {passed_count}/{len(corrected_records)} 个分支。",
                "",
                "| Seed | 方法 | 原日志 Full persistent bytes | Checkpoint bytes | Foundation bytes | Actor sidecar bytes | Full physical bytes | Package budget bytes | 事后审计 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in corrected_records:
            lines.append(
                f"| {row['seed']} | `{row['method']}` | {row['original_full_persistent_bytes']} | {row['checkpoint_bytes']} | "
                f"{row['foundation_bytes']} | {row['actor_sidecar_bytes']} | "
                f"{row['full_physical_bytes']} | {row['package_budget_bytes']} | "
                f"{'通过' if row['passed'] else '失败'} |"
            )
        cohort = postrun_audit.get("cohort_cost", {})
        source_cost = postrun_audit.get("source_cost", {})
        pairing = postrun_audit.get("pairing", {})
        integrity = postrun_audit.get("integrity", {})
        checkpoint_contract = postrun_audit.get("checkpoint_contract", {})
        lines.extend(
            [
                "",
                f"- 新增评估成本：主 cohort {cohort.get('main_episodes', '未提供')} episodes / {cohort.get('main_new_evaluation_steps', '未提供')} steps，pilot {cohort.get('pilot_new_evaluation_steps', '未提供')} steps；合计新增交互 {cohort.get('total_new_interactions', '未提供')}，新增训练 steps {cohort.get('new_training_steps', '未提供')}。",
                f"- 历史 source 交互复用 {source_cost.get('historical_total_interactions_reused', '未提供')}（训练 {source_cost.get('historical_training_steps_reused', '未提供')} + probe {source_cost.get('historical_probe_steps_reused', '未提供')} + 旧评估 {source_cost.get('historical_evaluation_steps_reused', '未提供')}）；这些是已发生且复用的交互，不计为本 cohort 新增。",
                f"- 配对核对：reset/action seeds 匹配为 {'是' if pairing.get('reset_and_action_seeds_match') else '否'}；共同 uniform rows {pairing.get('common_uniform_rows', '未提供')}，mismatches {pairing.get('uniform_mismatches', '未提供')}。",
                f"- 文件完整性核对：verified snapshots {integrity.get('snapshot_files_verified', '未提供')}，checkpoints {integrity.get('checkpoints_verified', '未提供')}，source files {integrity.get('source_files_verified', '未提供')}。",
                f"- Checkpoint contract：standalone restore 后需显式 freeze 为 {'是' if checkpoint_contract.get('standalone_restore_must_explicitly_freeze') else '否'}；runner 实际 freeze 且 hash 保持不变为 {'是' if checkpoint_contract.get('runtime_frozen_and_hash_unchanged') else '否'}。{checkpoint_contract.get('note', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这是 3 seeds × 每个目标每 seed 8 个 episode 的小样本 exploratory 结果；seed 是聚合与配对单位。",
            "- Source checkpoint 看过 A0/A1。A2 source 未见过，但 A2+ 用作 exploratory validation，因此结果不是 selection 后的独立测试。",
            "- 四个目标/方法都被强制使用同一个存储 parent 和新鲜执行路径。B2+ 与 A0- 用于检验错误复用，不证明自动路由或安全性。",
            "- prototype full-shared 6+2 int8 与 W0 only 同为 230 B，且使用相同容器；这是机制消融，不是容量优越证明。实际各分支字节数见上表。",
            "- 本报告给出逐 seed 结果、审计与实测成本；不据此作正式新颖性或显著性结论。",
            "",
        ]
    )
    return "\n".join(lines)


def render_figure(analysis: dict[str, Any], png_path: Path, pdf_path: Path) -> None:
    methods = analysis["methods"]
    all_target_rows = analysis["targets"]
    positive_target_rows = [row for row in all_target_rows if row["target"].endswith("+")]
    colors = {
        "local_only": "#4C78A8",
        "mean": "#F58518",
        "weighted": "#54A24B",
        "random": "#B279A2",
    }
    width = 0.18
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    figure = plt.figure(figsize=(13.8, 5.8))
    grid = figure.add_gridspec(
        1, 2, left=0.065, right=0.985, bottom=0.13, top=0.72, wspace=0.22
    )
    return_ax = figure.add_subplot(grid[0, 0])
    success_ax = figure.add_subplot(grid[0, 1])
    all_x = np.arange(len(all_target_rows), dtype=np.float64)
    positive_x = np.arange(len(positive_target_rows), dtype=np.float64)
    for method_index, method in enumerate(methods):
        returns = []
        return_low = []
        return_high = []
        successes = []
        success_low = []
        success_high = []
        for target in all_target_rows:
            stats = _get_method(target, method)["metrics"]
            ret = stats["mean_signed_return"]
            returns.append(ret["mean_across_seeds"])
            return_low.append(ret["mean_across_seeds"] - ret["min_across_seeds"])
            return_high.append(ret["max_across_seeds"] - ret["mean_across_seeds"])
        for target in positive_target_rows:
            succ = _get_method(target, method)["metrics"]["positive_success_rate"]
            successes.append(succ["mean_across_seeds"])
            success_low.append(succ["mean_across_seeds"] - succ["min_across_seeds"])
            success_high.append(succ["max_across_seeds"] - succ["mean_across_seeds"])
        return_ax.bar(
            all_x + offsets[method_index],
            returns,
            width,
            label=METHOD_LABELS.get(method, method),
            color=colors[method],
            alpha=0.9,
            yerr=np.array([return_low, return_high]),
            capsize=3,
            linewidth=0.5,
            edgecolor="white",
        )
        success_ax.bar(
            positive_x + offsets[method_index],
            successes,
            width,
            label=METHOD_LABELS.get(method, method),
            color=colors[method],
            alpha=0.9,
            yerr=np.array([success_low, success_high]),
            capsize=3,
            linewidth=0.5,
            edgecolor="white",
        )
    return_ax.axhline(0, color="#333333", linewidth=0.8)
    return_ax.set_ylabel("Seed-level mean signed return")
    return_ax.set_title("Signed return", pad=10)
    success_ax.set_ylabel("Seed-level positive success rate")
    success_ax.set_ylim(0, 1.05)
    success_ax.set_title("Positive success (A0−: signed-return only)", pad=10)
    return_ax.set_xticks(all_x, [row["target"] for row in all_target_rows])
    success_ax.set_xticks(positive_x, [row["target"] for row in positive_target_rows])
    for ax in (return_ax, success_ax):
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    handles, labels = return_ax.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(methods),
        frameon=False,
    )
    figure.suptitle("Consolidation retention probe — descriptive seed summaries", y=0.985)
    figure.text(
        0.5,
        0.805,
        "Bars show cross-seed means; whiskers show min–max of seed means.",
        ha="center",
        va="center",
        fontsize=9,
    )
    figure.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to write into nonempty report directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def generate_report(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _prepare_output(output_dir)
    summary_path = input_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"input directory has no summary.json: {input_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain a JSON object")
    postrun_audit_path = input_dir / "postrun_audit.json"
    postrun_audit = None
    if postrun_audit_path.is_file():
        postrun_audit = json.loads(postrun_audit_path.read_text(encoding="utf-8"))
        if not isinstance(postrun_audit, dict):
            raise ValueError("postrun_audit.json must contain a JSON object")
    analysis = analyze_summary(summary, postrun_audit=postrun_audit)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report_markdown(analysis), encoding="utf-8")
    render_figure(analysis, output_dir / "figure.png", output_dir / "figure.pdf")
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="directory containing summary.json")
    parser.add_argument("--output", required=True, type=Path, help="new or empty report directory")
    args = parser.parse_args(argv)
    try:
        analysis = generate_report(args.input, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        f"Wrote {args.output}: {len(analysis['targets'])} targets, "
        f"{len(analysis['seeds'])} seeds, {len(analysis['methods'])} methods"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

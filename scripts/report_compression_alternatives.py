"""Summarize completed compression-alternative runs without launching training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics as stats
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "dense", "topk50", "contrast", "center_topk50", "activation_topk50",
    "columns32", "svd2", "svd2_distilled", "bf16",
]
CONTEXTS = ["normal", "cw", "ccw", "cw_slip", "ccw_slip"]
SEED_MARKERS = {41: "o", 53: "s", 67: "^"}
METHOD_COLORS = {
    "dense": "#555555", "topk50": "#c45b57", "contrast": "#3387a3",
    "center_topk50": "#ad7c34", "activation_topk50": "#5a9665",
    "columns32": "#8b6ba8", "svd2": "#398a84",
    "svd2_distilled": "#d07836", "bf16": "#6e7ca3",
}

METHOD_NOTES = {
    "dense": "紧凑稠密参照；保留两个活跃叶的 FP32 残差，不删坐标。",
    "topk50": "原幅值 Top-K，每叶保留 130 个输入坐标；作为既有 50% 稀疏方案参照。",
    "contrast": "固定正交动作对比基，移除四个动作共同加值的 softmax 冗余；代数上精确，概率探针阈值为 2e-5。完整 Adam 状态可能超过 3,648 B 同预算。",
    "center_topk50": "先减去动作公共行均值，再按权重幅值做 50% Top-K；用于隔离中心化效果。",
    "activation_topk50": "在相同中心化后按 |w_ij| × sqrt(E[φ_j²]) 剪枝；是受 Wanda 启发的改编，不是原论文完整复现。",
    "columns32": "按中心化列范数与特征 RMS 保留 31 个特征列及必保偏置列，整列动作权重一起保存；使用紧凑稠密乘法。",
    "svd2": "在动作对比坐标中作截断 rank-2 SVD，以平方根奇异值初始化两个可训练因子；每叶 136 个浮点坐标。",
    "svd2_distilled": "与 svd2 相同表示，再用过去在线校准观测作固定 500 步教师 KL 蒸馏；lr=0.005、梯度范数上限 1，不按测试回报调参。",
    "bf16": "残差值和原生 Adam 两个动量用 BF16 持久保存，forward 临时转 FP32；基础 actor、critic 及其 Adam 仍是 FP32。状态精度变化属于方法本身，不代表硬件加速。",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    return float(stats.mean(values)) if values else None


def safe_ratio_drop(candidate: float, baseline: float) -> float | None:
    if baseline == 0 or not math.isfinite(baseline) or not math.isfinite(candidate):
        return None
    return 1.0 - candidate / baseline


def safe_relative_saving(candidate_bytes: float | None, baseline_bytes: float | None) -> float | None:
    if candidate_bytes is None or baseline_bytes in (None, 0):
        return None
    return 1.0 - candidate_bytes / baseline_bytes


def index_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], label: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate {label} record for {key}")
        result[key] = row
    return result


def expected_records(config: dict[str, Any]) -> dict[str, list[tuple[Any, ...]]]:
    seeds = config["source_seeds"]
    methods = config["methods"]
    contexts = config["contexts"]
    return {
        "constructions": [(m, s) for m in methods for s in seeds],
        "training": [(m, s) for m in methods for s in seeds],
        "before": [(m, s, c, "before") for m in methods for s in seeds for c in contexts],
        "after": [(m, s, c, "after") for m in methods for s in seeds for c in contexts],
    }


def get_seeded_rows(
    indexed: dict[tuple[Any, ...], dict[str, Any]], method: str, seeds: list[int], suffix: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    return [indexed[(method, seed, *suffix)] for seed in seeds if (method, seed, *suffix) in indexed]


def calculate_context_results(
    phase: str,
    methods: list[str],
    seeds: list[int],
    contexts: list[str],
    deployment: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for context in contexts:
        baseline_by_seed = {
            seed: deployment.get(("dense", seed, context, phase)) for seed in seeds
        }
        baseline_returns = [r["online_cumulative_return"] for r in baseline_by_seed.values() if r]
        baseline_success = [r["complete_episode_success_rate"] for r in baseline_by_seed.values() if r]
        baseline_early = [r["early_window_cumulative_reward"] for r in baseline_by_seed.values() if r]
        baseline_inference = [r["policy_inference_elapsed_s"] for r in baseline_by_seed.values() if r]
        baseline_eval_elapsed = [r["online_elapsed_s"] for r in baseline_by_seed.values() if r]
        baseline_completed = [r["completed_episodes"] for r in baseline_by_seed.values() if r]
        baseline_return_mean = mean(baseline_returns)
        baseline_success_mean = mean(baseline_success)
        for method in methods:
            candidate_rows = [deployment[(method, seed, context, phase)]
                              for seed in seeds if (method, seed, context, phase) in deployment]
            candidate_return_mean = mean([r["online_cumulative_return"] for r in candidate_rows])
            candidate_success_mean = mean([r["complete_episode_success_rate"] for r in candidate_rows])
            candidate_early_mean = mean([r["early_window_cumulative_reward"] for r in candidate_rows])
            candidate_inference_mean = mean([r["policy_inference_elapsed_s"] for r in candidate_rows])
            candidate_eval_elapsed_mean = mean([r["online_elapsed_s"] for r in candidate_rows])
            candidate_completed_mean = mean([r["completed_episodes"] for r in candidate_rows])
            relative_drop = (None if candidate_return_mean is None or baseline_return_mean is None
                             else safe_ratio_drop(candidate_return_mean, baseline_return_mean))
            success_drop_pp = (None if candidate_success_mean is None or baseline_success_mean is None
                               else (baseline_success_mean - candidate_success_mean) * 100)
            paired = []
            for seed in seeds:
                candidate = deployment.get((method, seed, context, phase))
                baseline = baseline_by_seed.get(seed)
                if not candidate or not baseline:
                    continue
                paired.append({
                    "phase": phase,
                    "source_seed": seed,
                    "return_drop_pct": (None if safe_ratio_drop(candidate["online_cumulative_return"],
                                                                  baseline["online_cumulative_return"]) is None
                                        else safe_ratio_drop(candidate["online_cumulative_return"],
                                                             baseline["online_cumulative_return"]) * 100),
                    "success_drop_percentage_points": (baseline["complete_episode_success_rate"] -
                                                       candidate["complete_episode_success_rate"]) * 100,
                    "candidate_return": candidate["online_cumulative_return"],
                    "baseline_return": baseline["online_cumulative_return"],
                    "candidate_success_rate": candidate["complete_episode_success_rate"],
                    "baseline_success_rate": baseline["complete_episode_success_rate"],
                    "candidate_early_window_cumulative_reward": candidate["early_window_cumulative_reward"],
                    "baseline_early_window_cumulative_reward": baseline["early_window_cumulative_reward"],
                    "candidate_policy_inference_elapsed_s": candidate["policy_inference_elapsed_s"],
                    "baseline_policy_inference_elapsed_s": baseline["policy_inference_elapsed_s"],
                    "candidate_online_elapsed_s": candidate["online_elapsed_s"],
                    "baseline_online_elapsed_s": baseline["online_elapsed_s"],
                    "candidate_completed_episodes": candidate.get("completed_episodes"),
                    "baseline_completed_episodes": baseline.get("completed_episodes"),
                })
            paired_returns = [r for r in paired if r["return_drop_pct"] is not None]
            paired_success = paired
            worst_return = (max(paired_returns, key=lambda r: r["return_drop_pct"])
                            if paired_returns else None)
            worst_success = (max(paired_success, key=lambda r: r["success_drop_percentage_points"])
                             if paired_success else None)
            results.append({
                "phase": phase,
                "context": context,
                "method": method,
                "n_checkpoint_runs": len(candidate_rows),
                "dense_mean_return": baseline_return_mean,
                "method_mean_return": candidate_return_mean,
                "mean_return_drop_pct": None if relative_drop is None else relative_drop * 100,
                "dense_mean_early_window_cumulative_reward": mean(baseline_early),
                "method_mean_early_window_cumulative_reward": candidate_early_mean,
                "dense_mean_policy_inference_elapsed_s": mean(baseline_inference),
                "method_mean_policy_inference_elapsed_s": candidate_inference_mean,
                "dense_mean_online_evaluation_elapsed_s": mean(baseline_eval_elapsed),
                "method_mean_online_evaluation_elapsed_s": candidate_eval_elapsed_mean,
                "dense_mean_completed_episodes": mean(baseline_completed),
                "method_mean_completed_episodes": candidate_completed_mean,
                "dense_mean_complete_episode_success_rate": baseline_success_mean,
                "method_mean_complete_episode_success_rate": candidate_success_mean,
                "mean_success_drop_percentage_points": success_drop_pp,
                "worst_paired_seed_return_drop_pct": None if worst_return is None else worst_return["return_drop_pct"],
                "worst_paired_seed_return_drop_seed": None if worst_return is None else worst_return["source_seed"],
                "worst_paired_seed_success_drop_percentage_points": None if worst_success is None else worst_success["success_drop_percentage_points"],
                "worst_paired_seed_success_drop_seed": None if worst_success is None else worst_success["source_seed"],
                "paired_seeds": paired,
            })
    return results


def calculate_byte_results(
    methods: list[str], seeds: list[int], constructions: dict[tuple[Any, ...], dict[str, Any]],
    trainings: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    dense_init = [constructions[("dense", s)]["initial_representation"]["adaptive_tensor_bytes"]
                  for s in seeds if ("dense", s) in constructions]
    dense_actual = [trainings[("dense", s)]["representation_after"]["adaptive_tensor_bytes"]
                    for s in seeds if ("dense", s) in trainings]
    dense_build_bundle = [constructions[("dense", s)]["bundle_bytes"] for s in seeds if ("dense", s) in constructions]
    dense_final_bundle = [trainings[("dense", s)]["bundle_bytes"] for s in seeds if ("dense", s) in trainings]
    topk_actual = [trainings[("topk50", s)]["representation_after"]["adaptive_tensor_bytes"]
                   for s in seeds if ("topk50", s) in trainings]
    topk_actual_by_seed = {
        s: trainings[("topk50", s)]["representation_after"]["adaptive_tensor_bytes"]
        for s in seeds if ("topk50", s) in trainings
    }
    rows = []
    for method in methods:
        per_seed = []
        for seed in seeds:
            construction = constructions.get((method, seed))
            training = trainings.get((method, seed))
            if construction is None:
                continue
            initial = construction["initial_representation"]
            actual = None if training is None else training["representation_after"]
            bytes_actual = None if actual is None else actual["adaptive_tensor_bytes"]
            topk_bytes = topk_actual_by_seed.get(seed)
            per_seed.append({
                "source_seed": seed,
                "initial_residual_value_bytes": initial.get("residual_value_bytes"),
                "initial_weight_factor_bytes": initial.get("weight_factor_bytes", initial.get("residual_weight_factor_bytes")),
                "initial_coordinate_index_bytes": initial.get("coordinate_index_bytes"),
                "initial_adaptive_tensor_bytes": initial.get("adaptive_tensor_bytes"),
                "predicted_warmed_adaptive_bytes": construction.get("predicted_warmed_adaptive_bytes"),
                "actual_warmed_residual_value_bytes": None if actual is None else actual.get("residual_value_bytes"),
                "actual_warmed_weight_factor_bytes": None if actual is None else actual.get("weight_factor_bytes", actual.get("residual_weight_factor_bytes")),
                "actual_warmed_coordinate_index_bytes": None if actual is None else actual.get("coordinate_index_bytes"),
                "actual_warmed_residual_optimizer_tensor_bytes": None if actual is None else actual.get("residual_optimizer_tensor_bytes"),
                "actual_warmed_adaptive_bytes": bytes_actual,
                "construction_bundle_bytes": construction.get("bundle_bytes"),
                "after_repair_bundle_bytes": None if training is None else training.get("bundle_bytes"),
                "actual_adaptive_saving_vs_topk_fraction": safe_relative_saving(bytes_actual, topk_bytes),
                "actual_adaptive_saving_vs_dense_fraction": safe_relative_saving(
                    bytes_actual, next((trainings[("dense", seed)]["representation_after"]["adaptive_tensor_bytes"]
                                        for _ in [0] if ("dense", seed) in trainings), None)),
                "after_repair_bundle_saving_vs_dense_fraction": safe_relative_saving(
                    None if training is None else training.get("bundle_bytes"),
                    next((trainings[("dense", seed)]["bundle_bytes"] for _ in [0]
                          if ("dense", seed) in trainings), None)),
            })
        initial_mean = mean([r["initial_adaptive_tensor_bytes"] for r in per_seed
                             if r["initial_adaptive_tensor_bytes"] is not None])
        actual_mean = mean([r["actual_warmed_adaptive_bytes"] for r in per_seed
                            if r["actual_warmed_adaptive_bytes"] is not None])
        predicted_mean = mean([r["predicted_warmed_adaptive_bytes"] for r in per_seed
                               if r["predicted_warmed_adaptive_bytes"] is not None])
        build_bundle_mean = mean([r["construction_bundle_bytes"] for r in per_seed
                                  if r["construction_bundle_bytes"] is not None])
        final_bundle_mean = mean([r["after_repair_bundle_bytes"] for r in per_seed
                                  if r["after_repair_bundle_bytes"] is not None])
        rows.append({
            "method": method,
            "n_checkpoint_runs": len(per_seed),
            "mean_initial_adaptive_bytes": initial_mean,
            "mean_predicted_warmed_adaptive_bytes": predicted_mean,
            "mean_actual_warmed_adaptive_bytes": actual_mean,
            "actual_adaptive_saving_vs_dense_fraction": safe_relative_saving(actual_mean, mean(dense_actual)),
            "actual_adaptive_saving_vs_topk_fraction": safe_relative_saving(actual_mean, mean(topk_actual)),
            "mean_construction_bundle_bytes": build_bundle_mean,
            "mean_after_repair_bundle_bytes": final_bundle_mean,
            "after_repair_bundle_saving_vs_dense_fraction": safe_relative_saving(final_bundle_mean, mean(dense_final_bundle)),
            "initial_weight_factor_bytes_by_seed": [r["initial_weight_factor_bytes"] for r in per_seed],
            "initial_index_bytes_by_seed": [r["initial_coordinate_index_bytes"] for r in per_seed],
            "actual_warmed_adaptive_bytes_by_seed": [r["actual_warmed_adaptive_bytes"] for r in per_seed],
            "actual_warmed_adaptive_bytes_delta_vs_topk_by_seed": [
                None if r["actual_warmed_adaptive_bytes"] is None or r["source_seed"] not in topk_actual_by_seed
                else r["actual_warmed_adaptive_bytes"] - topk_actual_by_seed[r["source_seed"]] for r in per_seed
            ],
            "after_repair_bundle_bytes_by_seed": [r["after_repair_bundle_bytes"] for r in per_seed],
            "after_repair_bundle_saving_vs_dense_fraction_by_seed": [r["after_repair_bundle_saving_vs_dense_fraction"] for r in per_seed],
            "per_seed": per_seed,
        })
    return rows


def calculate_training_results(
    methods: list[str], seeds: list[int], trainings: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    dense_by_seed = {s: trainings[("dense", s)] for s in seeds if ("dense", s) in trainings}
    results = []
    for method in methods:
        per_seed = []
        for seed in seeds:
            row = trainings.get((method, seed))
            dense = dense_by_seed.get(seed)
            if row is None:
                continue
            per_seed.append({
                "source_seed": seed,
                "online_steps": row.get("online_steps"),
                "online_cumulative_return": row.get("online_cumulative_return"),
                "online_elapsed_s": row.get("online_elapsed_s"),
                "optimizer_steps_added": row.get("optimizer_steps_added"),
                "return_vs_paired_dense_pct": None if dense is None else (
                    safe_ratio_drop(row["online_cumulative_return"], dense["online_cumulative_return"]) * -100
                    if safe_ratio_drop(row["online_cumulative_return"], dense["online_cumulative_return"]) is not None else None),
                "time_vs_paired_dense_pct": None if dense is None or dense["online_elapsed_s"] == 0 else
                    (row["online_elapsed_s"] / dense["online_elapsed_s"] - 1) * 100,
                "time_vs_paired_dense_ratio": None if dense is None or dense["online_elapsed_s"] == 0 else
                    row["online_elapsed_s"] / dense["online_elapsed_s"],
                "return_vs_paired_dense_ratio": None if dense is None or dense["online_cumulative_return"] == 0 else
                    row["online_cumulative_return"] / dense["online_cumulative_return"],
                "base_actor_unchanged": row.get("base_actor_unchanged"),
                "residual_optimizer_dtypes": row.get("residual_optimizer_dtypes"),
                "bundle_bytes": row.get("bundle_bytes"),
                "peak_accounted_state_bytes": row.get("peak_accounted_state_bytes"),
                "max_sampled_process_rss_bytes": row.get("max_sampled_process_rss_bytes"),
                "representation_after": row.get("representation_after"),
            })
        results.append({
            "method": method,
            "n_checkpoint_runs": len(per_seed),
            "mean_online_steps": mean([r["online_steps"] for r in per_seed if r["online_steps"] is not None]),
            "mean_online_cumulative_return": mean([r["online_cumulative_return"] for r in per_seed if r["online_cumulative_return"] is not None]),
            "mean_online_elapsed_s": mean([r["online_elapsed_s"] for r in per_seed if r["online_elapsed_s"] is not None]),
            "mean_optimizer_steps_added": mean([r["optimizer_steps_added"] for r in per_seed if r["optimizer_steps_added"] is not None]),
            "return_difference_vs_dense_pct": mean([r["return_vs_paired_dense_pct"] for r in per_seed if r["return_vs_paired_dense_pct"] is not None]),
            "time_difference_vs_dense_pct": mean([r["time_vs_paired_dense_pct"] for r in per_seed if r["time_vs_paired_dense_pct"] is not None]),
            "base_actor_unchanged_all_checkpoints": len(per_seed) == len(seeds) and all(r["base_actor_unchanged"] is True for r in per_seed),
            "residual_optimizer_dtypes_by_seed": [r["residual_optimizer_dtypes"] for r in per_seed],
            "per_seed": per_seed,
        })
    return results


def calculate_construction_results(
    methods: list[str], seeds: list[int], constructions: dict[tuple[Any, ...], dict[str, Any]],
    calibrations: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for method in methods:
        per_seed = []
        for seed in seeds:
            row = constructions.get((method, seed))
            if row is None:
                continue
            distillation = row.get("distillation") or {}
            probe = row.get("source_to_candidate_probability_probe") or {}
            per_seed.append({
                "source_seed": seed,
                "construction_elapsed_s": row.get("construction_elapsed_s"),
                "compression_optimizer_steps": row.get("compression_optimizer_steps", 0),
                "distillation_optimizer_steps": distillation.get("optimizer_steps", 0),
                "first_batch_teacher_kl": distillation.get("first_batch_teacher_KL"),
                "last_batch_teacher_kl": distillation.get("last_batch_teacher_KL"),
                "teacher_and_calibration_retained_in_bundle": distillation.get("teacher_and_calibration_retained_in_bundle"),
                "peak_explicit_live_tensor_bytes": distillation.get("peak_explicit_live_tensor_bytes"),
                "construction_explicit_live_tensor_bytes_lower_bound": row.get("construction_explicit_live_tensor_bytes_lower_bound"),
                "max_probability_difference_from_source": probe.get("max_probability_difference"),
                "mean_grid_kl_reference_to_candidate": probe.get("mean_grid_KL_reference_to_candidate"),
                "base_actor_unchanged": row.get("base_actor_unchanged"),
            })
        rows.append({
            "method": method,
            "mean_construction_elapsed_s": mean([r["construction_elapsed_s"] for r in per_seed if r["construction_elapsed_s"] is not None]),
            "mean_distillation_optimizer_steps": mean([r["distillation_optimizer_steps"] for r in per_seed if r["distillation_optimizer_steps"] is not None]),
            "mean_first_batch_teacher_kl": mean([r["first_batch_teacher_kl"] for r in per_seed if r["first_batch_teacher_kl"] is not None]),
            "mean_last_batch_teacher_kl": mean([r["last_batch_teacher_kl"] for r in per_seed if r["last_batch_teacher_kl"] is not None]),
            "mean_peak_explicit_live_tensor_bytes": mean([r["peak_explicit_live_tensor_bytes"] for r in per_seed if r["peak_explicit_live_tensor_bytes"] is not None]),
            "per_seed": per_seed,
        })
    return {
        "methods": rows,
        "calibration_replay": {
            "n_source_checkpoints": len(calibrations),
            "mean_elapsed_s": mean([r["elapsed_s"] for r in calibrations]),
            "total_elapsed_s": sum(r["elapsed_s"] for r in calibrations),
            "total_calibration_states": sum(r["states"] for r in calibrations),
            "new_environment_interactions": sum(r.get("new_environment_interactions", 0) for r in calibrations),
            "records": calibrations,
        },
        "interpretation": (
            "construction_elapsed_s includes representation construction and, for svd2_distilled, its 500-step "
            "distillation; the run schema does not record a separate distillation wall-clock timer. Distillation "
            "optimizer steps, teacher KL, and explicit live-tensor estimate are reported separately from online repair."
        ),
    }


def calculate_gate_results(
    methods: list[str], seeds: list[int], contexts: list[str], config: dict[str, Any],
    context_results: list[dict[str, Any]], byte_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return_limit_pct = float(config["maximum_relative_return_drop"]) * 100
    success_limit_pp = float(config["maximum_success_drop"]) * 100
    savings_limit_pct = float(config["minimum_adaptive_numeric_savings"]) * 100
    bundle_index = {r["method"]: r for r in byte_results}
    gates = []
    for method in methods:
        method_contexts = [r for r in context_results if r["method"] == method]
        worst_mean_return_drop = max((r["mean_return_drop_pct"] for r in method_contexts
                                      if r["mean_return_drop_pct"] is not None), default=None)
        worst_mean_success_drop = max((r["mean_success_drop_percentage_points"] for r in method_contexts
                                       if r["mean_success_drop_percentage_points"] is not None), default=None)
        paired = [seed_row for row in method_contexts for seed_row in row["paired_seeds"]]
        expected_paired_count = len(seeds) * len(contexts) * 2
        all_run_return_ok = (len(paired) == expected_paired_count and
                             all(r["return_drop_pct"] is not None and r["return_drop_pct"] <= return_limit_pct
                                 for r in paired))
        all_run_success_ok = (len(paired) == expected_paired_count and
                              all(r["success_drop_percentage_points"] <= success_limit_pp for r in paired))
        byte_row = bundle_index.get(method, {})
        saving = byte_row.get("actual_adaptive_saving_vs_dense_fraction")
        per_seed_savings = [r["actual_adaptive_saving_vs_dense_fraction"] for r in byte_row.get("per_seed", [])
                            if r["actual_adaptive_saving_vs_dense_fraction"] is not None]
        dense_bundle_by_seed = {r["source_seed"]: r["after_repair_bundle_bytes"]
                                for r in bundle_index.get("dense", {}).get("per_seed", [])}
        final_bundle_ok = bool(seeds) and all(
            (r["after_repair_bundle_bytes"] is not None and
             dense_bundle_by_seed.get(r["source_seed"]) is not None and
             r["after_repair_bundle_bytes"] < dense_bundle_by_seed[r["source_seed"]])
            for r in byte_row.get("per_seed", [])
        ) and len(byte_row.get("per_seed", [])) == len(seeds)
        return_gate = worst_mean_return_drop is not None and worst_mean_return_drop <= return_limit_pct
        success_gate = worst_mean_success_drop is not None and worst_mean_success_drop <= success_limit_pp
        savings_gate = saving is not None and saving * 100 >= savings_limit_pct
        passed = return_gate and success_gate and savings_gate and final_bundle_ok
        gates.append({
            "method": method,
            "worst_phase_context_mean_return_drop_gate_limit_pct": return_limit_pct,
            "worst_phase_context_mean_return_drop_pct": worst_mean_return_drop,
            "worst_phase_context_mean_return_gate_pass": return_gate,
            "worst_phase_context_mean_success_drop_gate_limit_percentage_points": success_limit_pp,
            "worst_phase_context_mean_success_drop_percentage_points": worst_mean_success_drop,
            "worst_phase_context_mean_success_gate_pass": success_gate,
            "adaptive_state_savings_gate_limit_pct": savings_limit_pct,
            "actual_adaptive_state_saving_vs_dense_pct": None if saving is None else saving * 100,
            "adaptive_state_savings_gate_pass": savings_gate,
            "adaptive_state_saving_vs_dense_pct_by_seed": [None if x is None else x * 100 for x in per_seed_savings],
            "after_repair_bundle_smaller_than_dense_for_each_checkpoint": final_bundle_ok,
            "mean_screen_pass": passed,
            "expected_seed_phase_context_records": expected_paired_count,
            "all_seed_phase_context_return_and_success_gates_pass": all_run_return_ok and all_run_success_ok,
            "all_seed_phase_context_return_gate_pass": all_run_return_ok,
            "all_seed_phase_context_success_gate_pass": all_run_success_ok,
            "screen_is_descriptive_only_no_candidate_selected_or_rerun": True,
        })
    return gates


def contrast_budget_summary(
    byte_results: list[dict[str, Any]], config: dict[str, Any],
    constructions: dict[tuple[Any, ...], dict[str, Any]], seeds: list[int],
) -> dict[str, Any]:
    target = int(config["adaptive_learning_state_target_bytes"])
    per_method = []
    for row in byte_results:
        run_bytes = [r["actual_warmed_adaptive_bytes"] for r in row["per_seed"]]
        per_method.append({
            "method": row["method"],
            "target_bytes": target,
            "actual_warmed_adaptive_bytes_mean": row["mean_actual_warmed_adaptive_bytes"],
            "actual_warmed_adaptive_bytes_by_seed": run_bytes,
            "within_budget_for_every_checkpoint": len(run_bytes) == len(seeds) and
                all(value is not None and value <= target for value in run_bytes),
        })
    contrast = next((r for r in per_method if r["method"] == "contrast"), None)
    contrast_builds = [constructions[("contrast", seed)] for seed in seeds if ("contrast", seed) in constructions]
    probe_errors = [r["source_to_candidate_probability_probe"]["max_probability_difference"]
                    for r in contrast_builds if r.get("source_to_candidate_probability_probe")]
    contrast_bytes = [] if contrast is None else contrast["actual_warmed_adaptive_bytes_by_seed"]
    contrast_data_complete = len(contrast_bytes) == len(seeds) and all(value is not None for value in contrast_bytes)
    contrast_within = (all(value <= target for value in contrast_bytes)
                       if contrast_data_complete else None)
    return {
        "target_bytes": target,
        "same_budget_eligibility_requires_every_checkpoint_actual_warmed_adaptive_bytes_at_or_below_target": True,
        "methods": per_method,
        "contrast_exact_lossless_diagnostic": {
            "actual_warmed_adaptive_bytes_mean": None if contrast is None else contrast["actual_warmed_adaptive_bytes_mean"],
            "actual_warmed_adaptive_bytes_by_seed": [] if contrast is None else contrast["actual_warmed_adaptive_bytes_by_seed"],
            "within_budget_for_every_checkpoint": contrast_within,
            "over_budget": None if contrast_within is None else not contrast_within,
            "max_observed_source_probability_difference": max(probe_errors, default=None),
            "declared_probability_tolerance": 2e-5,
            "lossless_by_fixed_orthogonal_action_contrast_reparameterization": True,
            "interpretation": "保留为精确代数诊断并按实际字节展示；若超 3,648 B，不列为同预算胜者。",
        },
    }


def calculate_vs_topk(
    methods: list[str], byte_results: list[dict[str, Any]], context_results: list[dict[str, Any]],
    seeds: list[int], contexts: list[str],
) -> list[dict[str, Any]]:
    byte_index = {r["method"]: r for r in byte_results}
    rows = []
    topk_bytes = byte_index.get("topk50", {}).get("mean_actual_warmed_adaptive_bytes")
    for method in methods:
        byte_row = byte_index.get(method, {})
        return_comparisons = []
        success_comparisons = []
        for context in contexts:
            result = next((r for r in context_results if r["phase"] == "after" and
                           r["context"] == context and r["method"] == method), None)
            topk = next((r for r in context_results if r["phase"] == "after" and
                         r["context"] == context and r["method"] == "topk50"), None)
            return_comparisons.append({
                "context": context,
                "mean_return_difference_vs_topk": None if not result or not topk or result["method_mean_return"] is None or topk["method_mean_return"] is None
                    else result["method_mean_return"] - topk["method_mean_return"],
                "mean_return_difference_pct_vs_topk": None if not result or not topk or topk["method_mean_return"] in (None, 0)
                    else (result["method_mean_return"] / topk["method_mean_return"] - 1) * 100,
                "mean_success_difference_percentage_points_vs_topk": None if not result or not topk or result["method_mean_complete_episode_success_rate"] is None or topk["method_mean_complete_episode_success_rate"] is None
                    else (result["method_mean_complete_episode_success_rate"] - topk["method_mean_complete_episode_success_rate"]) * 100,
            })
        rows.append({
            "method": method,
            "mean_actual_warmed_adaptive_bytes": byte_row.get("mean_actual_warmed_adaptive_bytes"),
            "actual_warmed_bytes_delta_vs_topk": None if byte_row.get("mean_actual_warmed_adaptive_bytes") is None or topk_bytes is None
                else byte_row["mean_actual_warmed_adaptive_bytes"] - topk_bytes,
            "actual_warmed_state_saving_vs_topk_pct": None if byte_row.get("actual_adaptive_saving_vs_topk_fraction") is None
                else byte_row["actual_adaptive_saving_vs_topk_fraction"] * 100,
            "actual_warmed_adaptive_bytes_delta_vs_topk_by_seed": byte_row.get("actual_warmed_adaptive_bytes_delta_vs_topk_by_seed", []),
            "after_repair_return_and_success_by_context": return_comparisons,
            "comparison_uses_actual_warmed_bytes_and_online_returns_not_parameter_count": True,
        })
    return rows


def create_figures(root: Path, methods: list[str], seeds: list[int], contexts: list[str],
                   deployment: dict[tuple[Any, ...], dict[str, Any]],
                   training: dict[tuple[Any, ...], dict[str, Any]],
                   byte_results: list[dict[str, Any]]) -> None:
    byte_by_method = {r["method"]: r for r in byte_results}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), layout="constrained", sharey=True)
    for ax, phase in zip(axes, ("before", "after")):
        for method in methods:
            color = METHOD_COLORS.get(method, "#555555")
            byte_row = byte_by_method.get(method, {})
            byte_field = "initial_adaptive_tensor_bytes" if phase == "before" else "actual_warmed_adaptive_bytes"
            byte_seed_map = {r["source_seed"]: r[byte_field]
                             for r in byte_row.get("per_seed", [])}
            for seed in seeds:
                drops = []
                for context in contexts:
                    candidate = deployment.get((method, seed, context, phase))
                    baseline = deployment.get(("dense", seed, context, phase))
                    if candidate and baseline:
                        drop = safe_ratio_drop(candidate["online_cumulative_return"],
                                               baseline["online_cumulative_return"])
                        if drop is not None:
                            drops.append(drop * 100)
                if seed not in byte_seed_map or not drops:
                    continue
                ax.scatter(byte_seed_map[seed], max(drops), color=color,
                           marker=SEED_MARKERS.get(seed, "o"), s=48, alpha=.85,
                           edgecolor="white", linewidth=.55)
        ax.axhline(5, color="#222222", linestyle="--", linewidth=1,
                   label="5% empirical return-drop gate")
        ax.axhline(0, color="#888888", linewidth=.65)
        ax.set_xlabel("Adaptive-state bytes (initial before; warmed after)")
        ax.set_title("Before repair" if phase == "before" else "After 8,192-step repair")
        ax.grid(True, alpha=.2)
    axes[0].set_ylabel("Worst context return drop vs paired dense (%)")
    method_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS[m],
                                markeredgecolor="white", markersize=8, label=m) for m in methods]
    seed_handles = [Line2D([0], [0], marker=SEED_MARKERS[s], color="#444444", linestyle="None",
                              label=f"checkpoint {s}") for s in seeds]
    fig.legend(handles=method_handles + seed_handles, loc="outside lower center", ncol=6, fontsize=8)
    fig.suptitle("Per-checkpoint worst-context return drop vs phase-appropriate adaptive state\n"
                 "Each point is one of three existing checkpoints; no confidence intervals")
    for ext in ("png", "pdf"):
        fig.savefig(root / f"compression_alternatives_pareto.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), layout="constrained", sharex=True)
    dense = {seed: training.get(("dense", seed)) for seed in seeds}
    x = np.arange(len(methods), dtype=float)
    offsets = {seed: (i - (len(seeds) - 1) / 2) * .16 for i, seed in enumerate(seeds)}
    for seed in seeds:
        reward_ratios, time_ratios, xs = [], [], []
        for ix, method in enumerate(methods):
            row = training.get((method, seed))
            base = dense.get(seed)
            if not row or not base:
                continue
            xs.append(x[ix] + offsets[seed])
            reward_ratios.append(row["online_cumulative_return"] / base["online_cumulative_return"]
                                 if base["online_cumulative_return"] else np.nan)
            time_ratios.append(row["online_elapsed_s"] / base["online_elapsed_s"]
                               if base["online_elapsed_s"] else np.nan)
        axes[0].scatter(xs, reward_ratios, marker=SEED_MARKERS.get(seed, "o"), s=43,
                        label=f"checkpoint {seed}", alpha=.85)
        axes[1].scatter(xs, time_ratios, marker=SEED_MARKERS.get(seed, "o"), s=43,
                        label=f"checkpoint {seed}", alpha=.85)
    for ax, title, ylabel in (
        (axes[0], "Online cumulative return", "Return / paired dense"),
        (axes[1], "Online wall time", "Elapsed seconds / paired dense"),
    ):
        ax.axhline(1, color="#222222", linestyle="--", linewidth=1)
        ax.set_xticks(x, methods, rotation=32, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Each point is a paired existing checkpoint; fixed 8,192-step online cohort\n"
                 "No error bars or independent-seed inference")
    for ext in ("png", "pdf"):
        fig.savefig(root / f"compression_alternatives_training.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def fnum(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def seed_tuple(rows: list[dict[str, Any]], field: str, digits: int = 2) -> str:
    by_seed = {row["source_seed"]: row.get(field) for row in rows}
    return " / ".join(fnum(by_seed.get(seed), digits) for seed in (41, 53, 67))


def optimizer_dtype_summary(rows: list[dict[str, Any]]) -> str:
    found: set[str] = set()
    for row in rows:
        for parameter_state in (row.get("residual_optimizer_dtypes") or {}).values():
            if isinstance(parameter_state, dict):
                found.update(str(dtype) for dtype in parameter_state.values())
    return ", ".join(sorted(found)) if found else "—"


def report_markdown(
    root: Path, summary: dict[str, Any], config: dict[str, Any],
    context_results: list[dict[str, Any]], byte_results: list[dict[str, Any]],
    training_results: list[dict[str, Any]], construction: dict[str, Any],
    gates: list[dict[str, Any]], budget: dict[str, Any], vs_topk: list[dict[str, Any]],
    completeness: dict[str, Any],
) -> str:
    methods = config["methods"]
    seeds = config["source_seeds"]
    contexts = config["contexts"]
    bytes_by_method = {r["method"]: r for r in byte_results}
    training_by_method = {r["method"]: r for r in training_results}
    construction_by_method = {r["method"]: r for r in construction["methods"]}
    gate_by_method = {r["method"]: r for r in gates}
    topk_by_method = {r["method"]: r for r in vs_topk}
    complete = bool(summary.get("complete"))
    expected_steps = int(config["continuation_steps"])
    lines = [
        "# 压缩替代方法对照：结果与解读", "",
        (f"**数据状态：{'全部预声明记录完成' if complete else '未全部完成，以下结果为部分记录'}。** "
         f"共 {len(seeds)} 个已有 A/B checkpoint、{len(methods)} 种表示、{len(contexts)} 个部署情境；"
         f"每个 arm 在五情境各作 2,048 步修复前和修复后部署，每个 method×seed 只进行一次 {expected_steps:,} 步在线修复。"
         "这里报告实际运行的固定比较，不做新赢家选择或重跑。"), "",
        "## 1. 比较口径与方法", "",
        "同一组 41、53、67 的 B-history_keep 终点 checkpoint 共用一个基础预训练权重。每种方法只改变两个活跃的 4×65 actor 残差；环境、路由、基础 actor、critic 和 PPO 超参数保持规程设定。全部方法（包括 dense 与原始 Top-K）都在修复前使用新 Adam，故这是新的冷优化器 cohort，不能与上一轮迁移 Adam 的 Top-K 直接作因果比较。", "",
        "校准来自此前 CW/CCW 在线轨迹，改作表示构造数据；部署前后用新的 1,500,000 随机种子偏移，训练流为相同的 8,192 步 CW→CCW 日程。修复两家族已在源 B 中出现，不称为首次新任务学习。方法根据规程预先固定，失败臂和负结果也保留；不按压缩前测试表现淘汰或调参。", "",
        "| 方法 | 表示选择与解释边界 |", "|---|---|",
    ]
    for method in methods:
        rationale = METHOD_NOTES.get(method, "")
        rationale = rationale.replace("|", "\\|")
        lines.append(f"| {method} | {rationale} |")
    lines.extend([
        "", "LoRA 提供低秩适配的概念参照 [Hu et al.](https://arxiv.org/abs/2106.09685)；Wanda 提供权重与激活感知剪枝的参照 [Sun et al.](https://arxiv.org/abs/2306.11695)；教师学生蒸馏参照 [Hinton et al.](https://arxiv.org/abs/1503.02531)。这些外部论文不构成本次强化学习结果的性能证据。", "",
        "## 2. 持久状态和整体 bundle 字节", "",
        "适应状态按实际存储的残差值/因子、坐标索引和残差 Adam 张量计算。表中初始值是在 Adam 尚未分配动量时的权重与索引；预测值按完整 Adam 状态估算；实际值来自 8,192 步修复后重新保存的优化器状态。整体 bundle 单独列出，不能把局部适应状态节省外推成整个智能体等比例节省。每格字节均为三 checkpoint 的均值；括号给出 41 / 53 / 67 的逐 checkpoint 实际暖状态。", "",
        "| 方法 | 初始残差值/因子 B | 初始索引 B | 初始适应状态 B | 预测暖状态 B | 实际暖状态 B（逐 checkpoint） | 实际暖状态节省 vs dense | 实际暖状态节省 vs Top-K | 构造 bundle B | 修复后 bundle B | bundle 节省 vs dense |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in methods:
        row = bytes_by_method.get(method, {})
        per = row.get("per_seed", [])
        avg_weight = mean([r["initial_weight_factor_bytes"] for r in per if r["initial_weight_factor_bytes"] is not None])
        avg_idx = mean([r["initial_coordinate_index_bytes"] for r in per if r["initial_coordinate_index_bytes"] is not None])
        init = mean([r["initial_adaptive_tensor_bytes"] for r in per if r["initial_adaptive_tensor_bytes"] is not None])
        lines.append(
            f"| {method} | {fnum(avg_weight,0)} | {fnum(avg_idx,0)} | {fnum(init,0)} | "
            f"{fnum(row.get('mean_predicted_warmed_adaptive_bytes'),0)} | "
            f"{fnum(row.get('mean_actual_warmed_adaptive_bytes'),0)} "
            f"({seed_tuple(per, 'actual_warmed_adaptive_bytes',0)}) | "
            f"{fnum(None if row.get('actual_adaptive_saving_vs_dense_fraction') is None else row['actual_adaptive_saving_vs_dense_fraction']*100)}% | "
            f"{fnum(None if row.get('actual_adaptive_saving_vs_topk_fraction') is None else row['actual_adaptive_saving_vs_topk_fraction']*100)}% | "
            f"{fnum(row.get('mean_construction_bundle_bytes'),0)} | {fnum(row.get('mean_after_repair_bundle_bytes'),0)} | "
            f"{fnum(None if row.get('after_repair_bundle_saving_vs_dense_fraction') is None else row['after_repair_bundle_saving_vs_dense_fraction']*100)}% |"
        )
    lines.extend([
        "", "### 每 checkpoint 实际字节账本", "",
        "| 方法 | Seed | 初始值/因子 B | 索引 B | 初始状态 B | 预测暖状态 B | 实际值/因子 B | 实际索引 B | 实际残差 Adam B | 实际暖状态 B | 构造 bundle B | 修复后 bundle B |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in methods:
        for row in bytes_by_method.get(method, {}).get("per_seed", []):
            lines.append(
                f"| {method} | {row['source_seed']} | {fnum(row['initial_weight_factor_bytes'],0)} | "
                f"{fnum(row['initial_coordinate_index_bytes'],0)} | {fnum(row['initial_adaptive_tensor_bytes'],0)} | "
                f"{fnum(row['predicted_warmed_adaptive_bytes'],0)} | {fnum(row['actual_warmed_weight_factor_bytes'],0)} | "
                f"{fnum(row['actual_warmed_coordinate_index_bytes'],0)} | "
                f"{fnum(row['actual_warmed_residual_optimizer_tensor_bytes'],0)} | "
                f"{fnum(row['actual_warmed_adaptive_bytes'],0)} | {fnum(row['construction_bundle_bytes'],0)} | "
                f"{fnum(row['after_repair_bundle_bytes'],0)} |"
            )
    lines.extend([
        "", "适应状态节省按实际暖状态字节相对 dense 计算，而不是按参数数量估算。Top-K 是同预算参照的一行；所有其它表示都同时给出相对 dense 与 Top-K 的实际字节差。整体 bundle savings 使用修复后保存的完整 agent bundle。SVD 与蒸馏 SVD 的实际数值暖状态低于 Top-K，但完整序列化 bundle 更大；因子分解会保存更多张量记录，而当前账本不提供足以精确拆分各序列化开销的字段，因此不把整体差值归因到某个单一来源。", "",
        "## 3. 同一 3,648 B 暖状态预算", "",
        "资格以每个 checkpoint 修复后的实际适应状态都不超过 3,648 B 判定，不以参数数、压缩前空 Adam 或均值替代。", "",
        "| 方法 | 实际暖状态均值 B | 41 / 53 / 67 实际 B | 每个 checkpoint 均在预算内 |", "|---|---:|---:|---|",
    ])
    for row in budget["methods"]:
        vals = " / ".join(fnum(v,0) for v in row["actual_warmed_adaptive_bytes_by_seed"])
        lines.append(f"| {row['method']} | {fnum(row['actual_warmed_adaptive_bytes_mean'],0)} | {vals} | {'是' if row['within_budget_for_every_checkpoint'] else '否'} |")
    contrast = budget["contrast_exact_lossless_diagnostic"]
    lines.extend([
        "", f"**contrast 精确诊断：**最大观测策略概率差为 {fnum(contrast['max_observed_source_probability_difference'],8)}（阈值 2e-5）；实际暖状态为 {fnum(contrast['actual_warmed_adaptive_bytes_mean'],0)} B，"
        f"{'超过' if contrast['over_budget'] else '未超过'} 3,648 B 预算。它依靠四动作公共加值不影响 softmax 概率的正交对比变换，作为无损代数诊断保留；若超预算，不将其计作同预算胜者。", "",
        "## 4. 修复前后各情境相对 dense 的回报和成功率", "",
        "回报下降先对三个 checkpoint 的累计回报求均值，再计算 `1 − 方法均值 / dense 均值`。因此表中的均值比例不是三个 seed 比例的平均。逐 checkpoint 最坏回报下降和成功率百分点下降则用相同 seed、相同情境配对后计算。负回报下降表示该方法均值高于 dense，不代表统计优势。", "",
    ])
    for phase in ("before", "after"):
        title = "修复前" if phase == "before" else "修复后"
        lines.extend([f"### {title}", "", "| 情境 | 方法 | dense 均值回报 | 方法均值回报 | 均值回报下降 | 最坏配对 seed 回报下降 | 最坏 seed | 成功率下降百分点 | 最坏配对 seed 成功率下降 pp | 最坏 seed | 前 128 步均值奖励 | 策略推理秒 | 评估秒 | 完成回合数均值 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for context in contexts:
            for method in methods:
                row = next((r for r in context_results if r["phase"] == phase and r["context"] == context and r["method"] == method), {})
                lines.append(
                    f"| {context} | {method} | {fnum(row.get('dense_mean_return'))} | {fnum(row.get('method_mean_return'))} | "
                    f"{fnum(row.get('mean_return_drop_pct'))}% | {fnum(row.get('worst_paired_seed_return_drop_pct'))}% | "
                    f"{row.get('worst_paired_seed_return_drop_seed') if row.get('worst_paired_seed_return_drop_seed') is not None else '—'} | "
                    f"{fnum(row.get('mean_success_drop_percentage_points'))} | "
                    f"{fnum(row.get('worst_paired_seed_success_drop_percentage_points'))} | "
                    f"{row.get('worst_paired_seed_success_drop_seed') if row.get('worst_paired_seed_success_drop_seed') is not None else '—'} |"
                    f" {fnum(row.get('method_mean_early_window_cumulative_reward'))} | "
                    f"{fnum(row.get('method_mean_policy_inference_elapsed_s'),3)} | "
                    f"{fnum(row.get('method_mean_online_evaluation_elapsed_s'),3)} | "
                    f"{fnum(row.get('method_mean_completed_episodes'),1)} |"
                )
        lines.append("")
    success_vals = [r["complete_episode_success_rate"] for r in summary.get("after", [])]
    if any(v >= 1.0 for v in success_vals):
        lines.append("**成功率天花板：**至少一组修复后完整回合成功率已达到 100%。达到天花板后，这个有限部署样本无法反映进一步改善；因此同时报告累计回报和配对百分点差，不将 100% 当作持续提升的证据。")
        lines.append("")
    lines.extend([
        "## 5. 在线修复回报、时间和优化器更新", "",
        f"每一行都是固定 {expected_steps:,} 步训练的三 checkpoint 均值；所有方法交互预算相同。回报和时间的逐 checkpoint 数字列在末尾的逐 seed 单元，百分比差是三个成对差的算术均值。单次顺序 CPU 运行的时间包括日志与预算审计，时间差不能解释成稳定硬件加速。", "",
        "| 方法 | 在线步 | 累计回报均值 | 回报差 vs paired dense | 时间均值 s | 时间差 vs paired dense | 新增优化器步均值 | 残差 Adam dtype | 基础 actor 每 checkpoint 不变 | Seed 41 / 53 / 67 回报 | Seed 41 / 53 / 67 时间 s | Seed 41 / 53 / 67 更新 |", "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ])
    for method in methods:
        row = training_by_method.get(method, {})
        per = row.get("per_seed", [])
        lines.append(
            f"| {method} | {fnum(row.get('mean_online_steps'),0)} | {fnum(row.get('mean_online_cumulative_return'))} | "
            f"{fnum(row.get('return_difference_vs_dense_pct'))}% | {fnum(row.get('mean_online_elapsed_s'))} | "
            f"{fnum(row.get('time_difference_vs_dense_pct'))}% | {fnum(row.get('mean_optimizer_steps_added'))} | "
            f"{optimizer_dtype_summary(per)} | {'是' if row.get('base_actor_unchanged_all_checkpoints') else '否'} | "
            f"{seed_tuple(per, 'online_cumulative_return')} | {seed_tuple(per, 'online_elapsed_s')} | "
            f"{seed_tuple(per, 'optimizer_steps_added',1)} |"
        )
    lines.extend([
        "", "全部方法都按 8,192 步报告；校准数据没有新增环境交互。不能声称某种方法用更少在线交互达成同一结果。", "",
        "## 6. 构造、蒸馏和校准成本", "",
        "构造时间与在线修复分开列示。当前运行 schema 将表示构造与蒸馏合在 `construction_elapsed_s`，没有单独的蒸馏墙钟计时；蒸馏步数、首末 batch 教师 KL、显式张量工作区另行列出。svd2_distilled 在每 seed 额外执行固定 500 步，蒸馏 optimizer 随后丢弃，在线 optimizer 重新初始化。", "",
        "| 方法 | 构造均值时间 s | 蒸馏 optimizer 步均值 | 首 batch KL | 末 batch KL | 峰值显式张量下界 B |", "|---|---:|---:|---:|---:|---:|",
    ])
    for row in construction["methods"]:
        lines.append(
            f"| {row['method']} | {fnum(row.get('mean_construction_elapsed_s'))} | "
            f"{fnum(row.get('mean_distillation_optimizer_steps'),0)} | "
            f"{fnum(row.get('mean_first_batch_teacher_kl'),6)} | {fnum(row.get('mean_last_batch_teacher_kl'),6)} | "
            f"{fnum(row.get('mean_peak_explicit_live_tensor_bytes'),0)} |"
        )
    cal = construction["calibration_replay"]
    lines.extend([
        "", f"三份过去在线校准轨迹共重放 {cal['total_calibration_states']:,} 个校准状态，重放墙钟合计 {fnum(cal['total_elapsed_s'])} 秒；新增环境交互为 {cal['new_environment_interactions']}。蒸馏只看这些过去在线校准观测的教师分布，不使用本轮测试目标、评估奖励或额外仿真交互。教师、校准张量和构造 optimizer 不保存在部署 bundle；没有持久稠密备份。显式工作区字节不含未测的 autograd / BLAS 临时区和序列化副本。", "",
        "## 7. 方法相对 Top-K：实际状态字节和修复后表现", "",
        "以下字节差来自修复后真实 warmed adaptive state；正的节省表示比 Top-K 实际字节少。参数/因子数量不代替实际索引和 Adam 状态核算。表现差按各情境的三 checkpoint 平均累计回报比 Top-K 均值计算。", "",
        "| 方法 | 实际暖状态 B | 字节差 vs Top-K B | 暖状态节省 vs Top-K | 修复后各情境回报差 vs Top-K |", "|---|---:|---:|---:|---|",
    ])
    for method in methods:
        row = topk_by_method.get(method, {})
        context_text = "; ".join(
            f"{r['context']} {fnum(r['mean_return_difference_pct_vs_topk'])}%"
            for r in row.get("after_repair_return_and_success_by_context", [])
        )
        lines.append(
            f"| {method} | {fnum(row.get('mean_actual_warmed_adaptive_bytes'),0)} | "
            f"{fnum(row.get('actual_warmed_bytes_delta_vs_topk'),0)} | "
            f"{fnum(row.get('actual_warmed_state_saving_vs_topk_pct'))}% | {context_text} |"
        )
    lines.extend([
        "", "## 8. 预声明经验筛选门槛", "",
        "此处只逐项列出规程门槛，不从本次结果挑选新赢家或发起重跑。回报与成功率的最差均值取修复前、修复后两个阶段 × 五情境的十组均值中的最大下降；同时单列全部 3 seed × 2 阶段 × 5 情境配对是否也都在 5% 界内。它们是三 checkpoint 上的经验筛选，不是统计安全保证。", "",
        "| 方法 | 最差阶段×情境均值回报下降 / 5% | 最差阶段×情境成功率下降 pp / 5 pp | 实际适应状态节省 / 10% | 每 checkpoint 修复后整体 bundle 更小 | 均值筛选四项全过 | 全部配对 seed×阶段×情境的回报及成功率均在界内 |", "|---|---:|---:|---:|---|---|---|",
    ])
    for row in gates:
        lines.append(
            f"| {row['method']} | {fnum(row['worst_phase_context_mean_return_drop_pct'])}% / {'通过' if row['worst_phase_context_mean_return_gate_pass'] else '未通过'} | "
            f"{fnum(row['worst_phase_context_mean_success_drop_percentage_points'])} / {'通过' if row['worst_phase_context_mean_success_gate_pass'] else '未通过'} | "
            f"{fnum(row['actual_adaptive_state_saving_vs_dense_pct'])}% / {'通过' if row['adaptive_state_savings_gate_pass'] else '未通过'} | "
            f"{'是' if row['after_repair_bundle_smaller_than_dense_for_each_checkpoint'] else '否'} | "
            f"{'是' if row['mean_screen_pass'] else '否'} | "
            f"{'是' if row['all_seed_phase_context_return_and_success_gates_pass'] else '否'} |"
        )
    errors = summary.get("errors", [])
    lines.extend([
        "", "## 9. 解释边界", "",
        "- 三个 checkpoint 来自同一基础预训练权重与同一任务家族阶段，不是三个独立基础模型训练种子；按 checkpoint 配对展示，不画置信区间，也不作总体统计保证。", 
        "- 每个 arm 在五情境各部署 2,048 步，修复前后各一轮；每个 method×seed 只进行一次 8,192 步 CW→CCW 在线修复。修复后轨迹因策略行为而各自分叉；随机种子安排相同不等于后续状态逐步配对。", 
        "- 完整回合成功率可能因有限完成回合数达到 100% 天花板；失败率下降空间因此被压缩。", 
        "- 所有表示固定支持，不测试删去的坐标重入、未知任务家族、共享父节点、容量耗尽后的知识整合或 Part C。没有搜索超参数，不能视作每个方法的最优调参比较。", 
        "- BF16 改变了残差与 Adam 状态的数值精度；本次单机顺序时间不是硬件加速基准。", 
        "- 状态字节是报告的持久适应状态和 bundle 口径；RSS 包含观察器审计副本，不等价于全部瞬时 RAM 上限。", 
        "- 过去在线校准轨迹现在用于校准；与本轮新测试随机流和评估奖励隔离。蒸馏没有把教师或校准数据写进最终 bundle。", 
        f"- 新增环境交互总数为 {summary.get('total_interactions', 0):,}；失败记录数为 {len(errors)}。完整汇总状态为 `complete={str(complete).lower()}`。",
        "", "## 图表", "",
        "![Before and after worst-context return loss vs phase-appropriate adaptive state](compression_alternatives_pareto.png)", "",
        "![Paired per-checkpoint training return and wall time relative to dense](compression_alternatives_training.png)", "",
        "图中每个点代表 41、53 或 67 中的一个现有 checkpoint；没有把同一 checkpoint 的情境当独立样本，也不画误差线或置信区间。", "",
        "## 软件与程序知识来源", "",
        "图由 Matplotlib 生成（版本见项目 `requirements-lock.txt`）。Matplotlib skill 对脚本绘图和 PNG/PDF 导出有贡献；引用 Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)。该软件引用不构成外部性能验证。", "",
        "[机器可读分析](analysis.json) / [原始运行汇总](summary.json) / [预注册规程](../../compression_alternatives_protocol.md) / [修复前后图 PDF](compression_alternatives_pareto.pdf) / [训练比较图 PDF](compression_alternatives_training.pdf)", "",
    ])
    if not complete:
        lines.append(f"部分记录说明：期望和已收记录对照见 `analysis.json`；错误记录：{json.dumps(errors, ensure_ascii=False)}。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/compression_alternatives",
                        help="Run directory containing summary.json (default: outputs/compression_alternatives)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Generate a partial report when summary.complete is false; default requires complete data")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing run summary: {summary_path}; no report files written")
    summary = read_json(summary_path)
    config = summary.get("config") or read_json(root / "config.json")
    methods = list(config.get("methods", METHODS))
    seeds = list(config.get("source_seeds", []))
    contexts = list(config.get("contexts", CONTEXTS))
    expected = expected_records(config)
    before_rows = list(summary.get("before", []))
    after_rows = list(summary.get("after", []))
    if before_rows and all("phase" not in row for row in before_rows):
        before_rows = [dict(row, phase="before") for row in before_rows]
    if after_rows and all("phase" not in row for row in after_rows):
        after_rows = [dict(row, phase="after") for row in after_rows]
    indexed = {
        "constructions": index_rows(summary.get("constructions", []), ("method", "source_seed"), "construction"),
        "training": index_rows(summary.get("training", []), ("method", "source_seed"), "training"),
        "before": index_rows(before_rows, ("method", "source_seed", "context", "phase"), "before evaluation"),
        "after": index_rows(after_rows, ("method", "source_seed", "context", "phase"), "after evaluation"),
    }
    missing = {}
    for name, keys in expected.items():
        got = indexed[name]
        missing[name] = [key for key in keys if key not in got]
    complete_records = not any(missing.values())
    run_complete = bool(summary.get("complete")) and not summary.get("errors") and complete_records
    completeness = {
        "summary_complete_flag": bool(summary.get("complete")),
        "error_count": len(summary.get("errors", [])),
        "records_complete": complete_records,
        "expected_record_counts": {k: len(v) for k, v in expected.items()},
        "actual_record_counts": {k: len(indexed[k]) for k in expected},
        "missing_records": {k: [list(item) for item in value] for k, value in missing.items()},
        "errors": summary.get("errors", []),
    }
    if not run_complete and not args.allow_incomplete:
        raise SystemExit(
            "summary.json is not complete; default reporting requires all constructions, before/after evaluations, "
            "training rows, and zero run errors. No report files written. Use --allow-incomplete only for a partial audit. "
            f"Record counts: {completeness['actual_record_counts']} / {completeness['expected_record_counts']}; "
            f"errors={completeness['error_count']}"
        )

    deployments = index_rows(before_rows + after_rows,
                             ("method", "source_seed", "context", "phase"), "deployment")
    context_results = [
        *calculate_context_results("before", methods, seeds, contexts, deployments),
        *calculate_context_results("after", methods, seeds, contexts, deployments),
    ]
    byte_results = calculate_byte_results(methods, seeds, indexed["constructions"], indexed["training"])
    training_results = calculate_training_results(methods, seeds, indexed["training"])
    construction_results = calculate_construction_results(
        methods, seeds, indexed["constructions"], summary.get("calibrations", []))
    gates = calculate_gate_results(methods, seeds, contexts, config, context_results, byte_results)
    budget = contrast_budget_summary(byte_results, config, indexed["constructions"], seeds)
    vs_topk = calculate_vs_topk(methods, byte_results, context_results, seeds, contexts)
    analysis = {
        "report_version": 1,
        "source_summary": str(summary_path),
        "scope": summary.get("scope", config.get("scope")),
        "complete": run_complete,
        "completeness": completeness,
        "config": config,
        "total_interactions": summary.get("total_interactions"),
        "methods_and_caveats": {method: METHOD_NOTES.get(method, "") for method in methods},
        "byte_accounting": byte_results,
        "before_after_context_performance": context_results,
        "training": training_results,
        "construction_and_distillation": construction_results,
        "same_budget_3648_bytes": budget,
        "comparison_vs_topk50_using_actual_bytes": vs_topk,
        "empirical_gates_no_winner_selected": gates,
        "figure_note": "Each plotted marker is one of three existing checkpoint seeds; no confidence intervals or independent-seed inference.",
        "caveats": [
            "All arms use a fresh Adam cohort; do not compare causally with the previous warm-Adam Top-K repair.",
            "Past online CW/CCW trajectories are repurposed as calibration; no new environment interactions or test rewards train the compressor.",
            "svd2_distilled uses 500 fixed teacher-KL updates; construction time includes distillation and the schema has no separate distillation timer.",
            "All arms use the same 8192-step continuation budget; no fewer-interactions claim.",
            "Three checkpoints share one foundation training seed; finite-sample gates are descriptive, not statistical guarantees.",
            "Failure records are retained in this analysis; methods were not fully hyperparameter tuned.",
            "No Part C, new task family, shared parent, or full-capacity consolidation is tested.",
        ],
    }
    create_figures(root, methods, seeds, contexts, deployments, indexed["training"], byte_results)
    (root / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    markdown = report_markdown(root, summary, config, context_results, byte_results, training_results,
                               construction_results, gates, budget, vs_topk, completeness)
    (root / "结果解读.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "root": str(root),
        "complete": run_complete,
        "analysis": str(root / "analysis.json"),
        "report": str(root / "结果解读.md"),
        "figures": [str(root / f"compression_alternatives_{name}.{ext}")
                    for name in ("pareto", "training") for ext in ("png", "pdf")],
        "errors_preserved": len(summary.get("errors", [])),
        "new_training_launched": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

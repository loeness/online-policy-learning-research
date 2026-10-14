"""Summarize the completed C1 fixed-structure experiment.

The script intentionally refuses to publish an interpretation until the formal
summary contains all 18 runs. It reads only saved experiment outputs and never
feeds evaluation observations back into training or routing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics as stats
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "c1"
METHODS = ("flat", "tree", "dictionary")
STAGES = ("seen", "heldout", "revisit", "history")
OLD_CONTEXTS = ("A0", "A1", "B0", "B1")
HELDOUT_CONTEXTS = ("A2", "B2")
CONTEXTS = OLD_CONTEXTS + HELDOUT_CONTEXTS
METHOD_LABEL = {"flat": "扁平", "tree": "树", "dictionary": "字典"}
STAGE_LABEL = {"seen": "已见任务", "heldout": "留出适应", "revisit": "旧任务重访", "history": "历史路由续训"}
METHOD_COLOR = {"flat": "#4C78A8", "tree": "#F58518", "dictionary": "#54A24B"}
TIER_MARKER = {"low": "o", "high": "s"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}"


def percent(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * float(value):.{digits}f}%"


def mean(values: list[float]) -> float | None:
    return stats.mean(values) if values else None


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows],
    ]


def extract_validation_trials(ledger: Any) -> list[Any]:
    if isinstance(ledger, list):
        return ledger
    if not isinstance(ledger, dict):
        return []
    preferred = ("trials", "validation_trials", "runs", "entries", "records", "experiments", "results")
    for key in preferred:
        value = ledger.get(key)
        if isinstance(value, list) and value:
            return value
    for key, value in ledger.items():
        if any(token in str(key).lower() for token in ("trial", "run", "entry", "record", "experiment")):
            if isinstance(value, list) and value:
                return value
    # Also accept a mapping from trial names to record dictionaries.
    mapping_rows = [{"trial_name": key, **value} for key, value in ledger.items() if isinstance(value, dict)]
    if mapping_rows and any(any(field in row for field in ("context", "evaluation", "success_rate", "training"))
                            for row in mapping_rows):
        return mapping_rows
    return []


def index_runs(summary: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (str(run["tier"]), str(run["method"]), int(run["seed"])): run
        for run in summary.get("runs", [])
    }


def stage_metrics(run: dict[str, Any], stage: str) -> dict[str, Any]:
    value = run.get(stage)
    return value if isinstance(value, dict) else {}


def total_optimizer_updates(run: dict[str, Any]) -> int:
    return sum(int(stage_metrics(run, stage).get("optimizer_updates", 0)) for stage in STAGES)


def total_elapsed_s(run: dict[str, Any]) -> float:
    return float(run.get("heldout_initialization_s", 0.0)) + sum(
        float(stage_metrics(run, stage).get("online_elapsed_s", 0.0)) for stage in STAGES
    )


def total_process_cpu_s(run: dict[str, Any]) -> float | None:
    values = [stage_metrics(run, stage).get("online_process_cpu_s") for stage in STAGES]
    if not any(value is not None for value in values):
        return None
    # Initialization is measured as wall time only; the experiment currently
    # records process CPU seconds for the four model.learn calls.
    return sum(float(value) for value in values if value is not None)


def adaptive_bytes(run: dict[str, Any]) -> int | None:
    # final_eval is taken immediately after revisit, before the routing-only
    # continuation. The adaptive storage shape does not change across stages.
    representation = stage_metrics(run, "revisit").get("representation", {})
    value = representation.get("adaptive_tensor_bytes")
    if value is None:
        value = stage_metrics(run, "history").get("representation", {}).get("adaptive_tensor_bytes")
    return int(value) if value is not None else None


def phase_capacity(run: dict[str, Any], field: str) -> int:
    return max((int(stage_metrics(run, phase).get(field, 0)) for phase in STAGES), default=0)


def rolling_target_for_context(output: Path, tier: str, method: str, seed: int,
                               context: str, window: int, target: float,
                               context_offset: int, censor_steps: int) -> dict[str, Any]:
    path = output / "formal" / tier / method / f"s{seed}" / "heldout" / "episodes.csv"
    if not path.exists():
        return {"context": context, "window": window, "target": target,
                "observed_episodes": 0, "event_step": None, "censored_at_step": censor_steps,
                "event_step_phase": None, "context_offset": context_offset,
                "status": "episode_csv_missing", "path": str(path.relative_to(output))}

    observed: list[int] = []
    event_step = None
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            # Mixed episodes span a context boundary; they cannot be assigned to
            # either held-out task's single-context success window.
            if row.get("context") != context or row.get("mixed_context", "0") not in ("0", "", None):
                continue
            observed.append(int(row["success"]))
            if len(observed) >= window and sum(observed[-window:]) / window >= target:
                event_step = int(row["step"])
                break
    return {
        "context": context,
        "window": window,
        "target": target,
        "observed_episodes": len(observed),
        "event_step": None if event_step is None else event_step - context_offset,
        "event_step_phase": event_step,
        "context_offset": context_offset,
        "censored_at_step": None if event_step is not None else censor_steps,
        "status": "reached" if event_step is not None else "right_censored",
        "path": str(path.relative_to(output)),
    }


def make_analysis(output: Path, summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    runs = index_runs(summary)
    seeds = [int(seed) for seed in config["seeds"]]
    target = float(config.get("target_success", 0.80))
    retention_tolerance = float(config.get("retention_tolerance", 0.05))
    heldout_bout_steps = int(config["heldout_bout_steps"])
    calibration_rows = []
    cold_rows = []
    retention_rows = []
    final_rows = []
    history_rows = []
    history_eval_rows = []
    resource_rows = []
    stage_rows = []
    rolling_rows = []

    for tier in config["tiers"]:
        for method in METHODS:
            for seed in seeds:
                key = (tier, method, seed)
                if key not in runs:
                    continue
                run = runs[key]
                oracle_steps = sum(int(stage_metrics(run, phase).get("online_steps", 0))
                                   for phase in ("seen", "heldout", "revisit"))
                calibration_rows.append({"tier": tier, "method": method, "seed": seed,
                                         "oracle_calibration_steps": oracle_steps,
                                         "history_unlabeled_steps": int(stage_metrics(run, "history").get("online_steps", 0))})

                for context in HELDOUT_CONTEXTS:
                    cold = run.get("cold", {}).get(context, {})
                    foundation = cold.get("foundation", {})
                    reuse = cold.get("shared_or_sibling_reuse", {})
                    foundation_reward = foundation.get("online_cumulative_return")
                    reuse_reward = reuse.get("online_cumulative_return")
                    cold_rows.append({
                        "tier": tier, "seed": seed, "method": method, "context": context,
                        "foundation_reward": foundation_reward, "reuse_reward": reuse_reward,
                        "reuse_minus_foundation": (float(reuse_reward) - float(foundation_reward)
                                                   if foundation_reward is not None and reuse_reward is not None else None),
                        "foundation_steps": foundation.get("online_steps"),
                        "reuse_steps": reuse.get("online_steps"),
                    })

                for context in OLD_CONTEXTS:
                    base = run.get("seen_eval", {}).get(context, {}).get("success_rate")
                    adapted = run.get("adapted_eval", {}).get(context, {}).get("success_rate")
                    final = run.get("final_eval", {}).get(context, {}).get("success_rate")
                    delta_adapted = (100 * (float(adapted) - float(base))
                                     if base is not None and adapted is not None else None)
                    delta_final = (100 * (float(final) - float(base))
                                   if base is not None and final is not None else None)
                    retention_rows.append({
                        "tier": tier, "seed": seed, "method": method, "context": context,
                        "seen_success": base, "adapted_success": adapted, "final_success": final,
                        "adapted_delta_pp": delta_adapted, "final_delta_pp": delta_final,
                        "adapted_gate_pass": bool(adapted is not None and base is not None and
                                                  float(adapted) >= float(base) - retention_tolerance),
                        "final_gate_pass": bool(final is not None and base is not None and
                                                float(final) >= float(base) - retention_tolerance),
                        "recovered_by_final": bool(
                            adapted is not None and base is not None and final is not None and
                            float(adapted) < float(base) - retention_tolerance and
                            float(final) >= float(base) - retention_tolerance),
                    })

                final_eval = run.get("final_eval", {})
                context_success = {context: final_eval.get(context, {}).get("success_rate") for context in CONTEXTS}
                passed_contexts = sum(value is not None and float(value) >= target
                                      for value in context_success.values())
                final_rows.append({
                    "tier": tier, "seed": seed, "method": method,
                    "context_success": context_success,
                    "target_success_count": passed_contexts,
                    "all_six_pass": passed_contexts == len(CONTEXTS),
                    "mean_success": mean([float(value) for value in context_success.values() if value is not None]),
                })

                history = stage_metrics(run, "history")
                history_rows.append({
                    "tier": tier, "seed": seed, "method": method,
                    "exact_accuracy": history.get("route_exact_accuracy"),
                    "family_accuracy": history.get("route_family_accuracy"),
                    "parent_fallback_fraction": history.get("parent_fallback_fraction"),
                    "foundation_fallback_fraction": history.get("foundation_fallback_fraction"),
                    "cumulative_return": history.get("online_cumulative_return"),
                    "steps": history.get("online_steps"),
                    "switch_first_correct": history.get("switch_first_correct", []),
                    "unknown_switches_not_given_to_router": history.get("unknown_switches_not_given_to_router"),
                })
                for context in CONTEXTS:
                    evaluation = run.get("history_eval", {}).get(context, {})
                    history_eval_rows.append({
                        "tier": tier, "seed": seed, "method": method, "context": context,
                        "success_rate": evaluation.get("success_rate"),
                        "cumulative_return": evaluation.get("online_cumulative_return"),
                        "completed_episodes": evaluation.get("completed_episodes"),
                        "successful_episodes": evaluation.get("successful_episodes"),
                        "parameter_hash_unchanged": evaluation.get("parameter_hash_unchanged"),
                    })

                stages = {stage: stage_metrics(run, stage) for stage in STAGES}
                stage_rows.append({
                    "tier": tier, "seed": seed, "method": method,
                    "phases": {stage: {
                        "steps": int(stages[stage].get("online_steps", 0)),
                        "cumulative_return": stages[stage].get("online_cumulative_return"),
                        "optimizer_updates": int(stages[stage].get("optimizer_updates", 0)),
                        "elapsed_s": stages[stage].get("online_elapsed_s"),
                        "route_elapsed_s": stages[stage].get("route_elapsed_s"),
                    } for stage in STAGES},
                    "total_train_steps": int(run.get("train_interactions", 0)),
                    "evaluation_steps": int(run.get("evaluation_interactions", 0)),
                    "oracle_calibration_steps": oracle_steps,
                    "optimizer_updates": total_optimizer_updates(run),
                    "elapsed_s_including_initialization": total_elapsed_s(run),
                    "process_cpu_s_training_phases": total_process_cpu_s(run),
                })

                adaptive = adaptive_bytes(run)
                resource_rows.append({
                    "tier": tier, "seed": seed, "method": method,
                    "adaptive_residual_plus_adam_bytes": adaptive,
                    "target_success_count": passed_contexts,
                    "mean_final_success": mean([float(value) for value in context_success.values() if value is not None]),
                    "final_all_six_pass": passed_contexts == len(CONTEXTS),
                    "cap_bytes": int(run.get("cap_bytes", 0)),
                    "adaptive_cap_bytes": int(summary.get("budgets", {}).get(tier, {}).get("adaptive_cap_bytes", 0)),
                    "peak_core_bytes": phase_capacity(run, "max_core_bytes"),
                    "peak_rollout_bytes": phase_capacity(run, "rollout_bytes"),
                    "peak_agent_bytes": phase_capacity(run, "max_agent_bytes"),
                    "peak_sampled_rss_bytes": phase_capacity(run, "max_sampled_process_rss_bytes"),
                    "peak_explicit_live_tensor_lower_bound_bytes": phase_capacity(run, "explicit_live_tensor_bytes_lower_bound"),
                    "final_bundle_bytes": int(stages["history"].get("bundle_bytes", 0)),
                })

                for context in HELDOUT_CONTEXTS:
                    context_offset = list(config["heldout_sequence"]).index(context) * heldout_bout_steps
                    rolling_rows.append({"tier": tier, "seed": seed, "method": method,
                        **rolling_target_for_context(output, tier, method, seed, context,
                                                     window=20, target=target,
                                                     context_offset=context_offset,
                                                     censor_steps=heldout_bout_steps)})

    final_success_count = sum(row["target_success_count"] for row in final_rows)
    all_six_count = sum(row["all_six_pass"] for row in final_rows)
    adapted_gate_pass_count = sum(row["adapted_gate_pass"] for row in retention_rows)
    final_gate_pass_count = sum(row["final_gate_pass"] for row in retention_rows)
    recovered_by_final_count = sum(row["recovered_by_final"] for row in retention_rows)
    adapted_failure_rows = [row for row in retention_rows if not row["adapted_gate_pass"]]
    max_adapted_drop = min(
        (row for row in retention_rows if row["adapted_delta_pp"] is not None),
        key=lambda row: row["adapted_delta_pp"], default=None)
    cold_pairs = [row for row in cold_rows if row["reuse_minus_foundation"] is not None]
    method_summaries = []
    for tier in config["tiers"]:
        for method in METHODS:
            final_group = [row for row in final_rows if row["tier"] == tier and row["method"] == method]
            retention_group = [row for row in retention_rows if row["tier"] == tier and row["method"] == method]
            cold_group = [row for row in cold_pairs if row["tier"] == tier and row["method"] == method]
            resource_group = [row for row in resource_rows if row["tier"] == tier and row["method"] == method]
            task_successes = sum(row["target_success_count"] for row in final_group)
            method_summaries.append({
                "tier": tier, "method": method,
                "final_task_successes": task_successes,
                "final_task_opportunities": len(final_group) * len(CONTEXTS),
                "all_six_pass_runs": sum(row["all_six_pass"] for row in final_group),
                "formal_runs": len(final_group),
                "mean_final_success": mean([float(value) for row in final_group for value in row["context_success"].values() if value is not None]),
                "adapted_retention_gate_passes": sum(row["adapted_gate_pass"] for row in retention_group),
                "final_retention_gate_passes": sum(row["final_gate_pass"] for row in retention_group),
                "retention_gate_count": len(retention_group),
                "mean_cold_reuse_minus_foundation_reward": mean([float(row["reuse_minus_foundation"]) for row in cold_group]),
                "mean_adaptive_residual_plus_adam_bytes": mean([float(row["adaptive_residual_plus_adam_bytes"]) for row in resource_group if row["adaptive_residual_plus_adam_bytes"] is not None]),
            })
    precheck_path = output / config.get("precheck_directory", "precheck") / "summary.json"
    precheck = load_json(precheck_path) if precheck_path.exists() else None
    precheck_calibration_steps = 0
    if precheck:
        for case in precheck.get("cases", []):
            precheck_calibration_steps += int(case.get("initial_training", {}).get("online_steps", 0))
            extension = case.get("extension_training")
            if extension:
                precheck_calibration_steps += int(extension.get("online_steps", 0))
    return {
        "schema_version": 1,
        "source_summary_sha256": sha256(output / "summary.json"),
        "config": config,
        "scope": summary.get("scope"),
        "run_count": len(runs),
        "headline": {
            "final_task_successes": final_success_count,
            "final_task_opportunities": len(final_rows) * len(CONTEXTS),
            "all_six_pass_runs": all_six_count,
            "formal_runs": len(final_rows),
            "adapted_old_task_retention_gate_passes": adapted_gate_pass_count,
            "final_old_task_retention_gate_passes": final_gate_pass_count,
            "old_task_retention_gate_count": len(retention_rows),
            "immediate_old_task_retention_failures": len(adapted_failure_rows),
            "recovered_old_task_retention_failures_by_final": recovered_by_final_count,
            "adapted_retention_failures": adapted_failure_rows,
            "maximum_adapted_retention_drop": max_adapted_drop,
            "cold_reuse_positive_pairs": sum(float(row["reuse_minus_foundation"]) > 0 for row in cold_pairs),
            "cold_reuse_pair_count": len(cold_pairs),
            "mean_cold_reuse_minus_foundation_reward": mean([float(row["reuse_minus_foundation"]) for row in cold_pairs]),
        },
        "method_summaries": method_summaries,
        "totals": {
            "training_interactions": int(summary.get("total_training_interactions", 0)),
            "evaluation_interactions": int(summary.get("total_evaluation_interactions", 0)),
            "validation_interactions": int(summary.get("validation_interactions", 0)),
            "selected_precheck_interactions": int(summary.get("selected_precheck_interactions", precheck.get("total_interactions", 0) if precheck else 0)),
            "oracle_labeled_calibration_interactions": sum(row["oracle_calibration_steps"] for row in calibration_rows),
            "validation_oracle_labeled_calibration_interactions": precheck_calibration_steps,
            "calibration_rows_by_run": calibration_rows,
        },
        "precheck": precheck,
        "validation_ledger": load_json(output / "validation_ledger.json") if (output / "validation_ledger.json").exists() else None,
        "validation_ledger_path": str((output / "validation_ledger.json").relative_to(output)),
        "representation_diagnostic": load_json(output / "representation_diagnostic.json") if (output / "representation_diagnostic.json").exists() else None,
        "cold_evaluations": cold_rows,
        "retention": retention_rows,
        "final_success": final_rows,
        "stages": stage_rows,
        "history_routing": history_rows,
        "history_eval": history_eval_rows,
        "resources": resource_rows,
        "rolling_20_episode_target_after_heldout": rolling_rows,
        "budget_preflight": summary.get("budgets", {}),
    }


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })


def make_plots(output: Path, analysis: dict[str, Any]) -> None:
    configure_plot_style()
    cold_rows = analysis["cold_evaluations"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True, sharey=False)
    categories = [(tier, method) for tier in ("low", "high") for method in METHODS]
    for ax, context in zip(axes, HELDOUT_CONTEXTS):
        for index, (tier, method) in enumerate(categories):
            color = METHOD_COLOR[method]
            subset = [row for row in cold_rows if row["context"] == context and row["tier"] == tier and row["method"] == method]
            for seed_index, row in enumerate(sorted(subset, key=lambda value: value["seed"])):
                jitter = (seed_index - 1) * 0.035
                x0, x1 = index - 0.09 + jitter, index + 0.09 + jitter
                foundation = row.get("foundation_reward")
                reuse = row.get("reuse_reward")
                if foundation is None or reuse is None:
                    continue
                ax.plot([x0, x1], [foundation, reuse], color=color, alpha=0.48, linewidth=1)
                ax.scatter(x0, foundation, color="#777777", marker="x", s=25, zorder=3)
                ax.scatter(x1, reuse, color=color, marker="o", s=28, zorder=3)
        ax.set_xticks(range(len(categories)), [f"{METHOD_LABEL[m]}\n{t}" for t, m in categories])
        ax.set_title(f"留出变体 {context}；同 seed 配对")
        ax.set_ylabel("1024 步累计奖励")
        ax.grid(axis="y", alpha=0.2)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color="#777777", marker="x", linestyle="", label="基础策略"),
              Line2D([0], [0], color="#444444", marker="o", linestyle="", label="父节点 / 已见兄弟复用")]
    axes[1].legend(handles=legend, loc="best")
    fig.suptitle("留出任务冷启动：基础策略与结构性复用")
    fig.savefig(output / "cold_reward.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stage_rows = analysis["stages"]
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    stage_x = np.arange(len(STAGES))
    for tier in ("low", "high"):
        for method in METHODS:
            subset = [row for row in stage_rows if row["tier"] == tier and row["method"] == method]
            if not subset:
                continue
            for row in subset:
                cumulative = []
                running = 0.0
                for stage in STAGES:
                    running += float(row["phases"][stage].get("cumulative_return") or 0.0)
                    cumulative.append(running)
                ax.plot(stage_x, cumulative, color=METHOD_COLOR[method], marker=TIER_MARKER[tier],
                        alpha=0.27, linewidth=0.9)
            points = []
            for stage in STAGES:
                values = [float(row["phases"][stage].get("cumulative_return") or 0.0)
                          for row in subset]
                points.append(sum(values) / len(values))
            total_cumulative = np.cumsum(points)
            ax.plot(stage_x, total_cumulative, color=METHOD_COLOR[method], marker=TIER_MARKER[tier],
                    linewidth=2.1, label=f"{METHOD_LABEL[method]}-{tier} 三 seed 均值")
    ax.set_xticks(stage_x, [STAGE_LABEL[name] for name in STAGES])
    ax.set_ylabel("截至该阶段的在线累计奖励总和")
    ax.set_title("在线学习各阶段累计回报（细线为单 seed，粗线为分阶段均值累加）")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, loc="best")
    fig.savefig(output / "cumulative_returns.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    resource_rows = analysis["resources"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for row in resource_rows:
        adaptive = row.get("adaptive_residual_plus_adam_bytes")
        success = row.get("mean_final_success")
        if adaptive is None or success is None:
            continue
        ax.scatter(adaptive / 1024, 100 * success, c=METHOD_COLOR[row["method"]],
                   marker=TIER_MARKER[row["tier"]], s=56, alpha=0.82,
                   edgecolor="#222222", linewidth=0.35)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=METHOD_COLOR[m], marker="o", linestyle="", label=METHOD_LABEL[m]) for m in METHODS]
    handles += [Line2D([0], [0], color="#444444", marker=TIER_MARKER[t], linestyle="", label=f"{t} 容量档") for t in ("low", "high")]
    ax.axhline(100 * float(analysis["config"].get("target_success", 0.8)), color="#666666", linestyle="--", linewidth=1,
               label="单任务目标成功率")
    ax.set_xlabel("实际残差 + 残差 Adam 状态（KiB）")
    ax.set_ylabel("final_eval 六任务平均成功率（%）")
    ax.set_title("自适应状态字节与终点成功率；每点为一个 seed")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, ncol=2, loc="best")
    fig.savefig(output / "adaptive_bytes_vs_success.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    history_rows = analysis["history_routing"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for row in history_rows:
        accuracy, reward = row.get("exact_accuracy"), row.get("cumulative_return")
        if accuracy is None or reward is None:
            continue
        ax.scatter(100 * float(accuracy), float(reward), c=METHOD_COLOR[row["method"]],
                   marker=TIER_MARKER[row["tier"]], s=58, alpha=0.82,
                   edgecolor="#222222", linewidth=0.35)
    ax.set_xlabel("隐藏标签历史路由 exact accuracy（%）")
    ax.set_ylabel("历史路由续训累计奖励（固定交互预算）")
    ax.set_title("历史识别准确率与续训奖励；两者是分开的诊断量")
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, ncol=2, loc="best")
    fig.savefig(output / "history_routing.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_report(analysis: dict[str, Any], summary: dict[str, Any]) -> str:
    config = analysis["config"]
    diagnostic = analysis["representation_diagnostic"]
    bounds = diagnostic["rank_bounds"]
    flat_errors = [float(diagnostic["contexts"][context]["flat_rank_1_relative_frobenius_error"])
                   for context in CONTEXTS]
    flat_error_range = f"{min(flat_errors):.3f}–{max(flat_errors):.3f}"
    tree_error_max = max(
        max(float(diagnostic["contexts"][context]["tree_local_rank_1_relative_frobenius_error"]),
            float(diagnostic["contexts"][context]["tree_parent_rank_1_relative_frobenius_error"]))
        for context in CONTEXTS
    )
    dictionary_k6_error = float(bounds["dictionary_K6_relative_frobenius_error_lower_bound"])
    additive_logit_error = float(diagnostic["additive_decomposition"]["maximum_full_logit_reconstruction_error"])
    lines = [
        "# C1 结果解读",
        "",
        "## 主要结果",
        "",
        f"正式结果包含 **{analysis['run_count']} 个 run**，final_eval 在 **{analysis['headline']['final_task_successes']}/{analysis['headline']['final_task_opportunities']} 个任务实例**达到单任务 80% 成功率目标；**{analysis['headline']['all_six_pass_runs']}/{analysis['headline']['formal_runs']} 个 run** 的六个任务全部达标。旧任务必须分别看适应后与重访后：adapted_eval 仅 **{analysis['headline']['adapted_old_task_retention_gate_passes']}/{analysis['headline']['old_task_retention_gate_count']}** 个 seed×任务比较保持在 seen_eval 的 5 个百分点内；**{analysis['headline']['immediate_old_task_retention_failures']}/{analysis['headline']['old_task_retention_gate_count']}** 个比较在新任务适应后暂时下降超过 5 pp，重访后其中 **{analysis['headline']['recovered_old_task_retention_failures_by_final']}** 个恢复到 5 pp 范围内。final_eval 的重访后闸门为 **{analysis['headline']['final_old_task_retention_gate_passes']}/{analysis['headline']['old_task_retention_gate_count']}**；仍有 **{analysis['headline']['old_task_retention_gate_count'] - analysis['headline']['final_old_task_retention_gate_passes']}** 个比较未通过。最大即时下降为 **{abs(analysis['headline']['maximum_adapted_retention_drop']['adapted_delta_pp']):.2f} pp**（{analysis['headline']['maximum_adapted_retention_drop']['tier']} 容量、{METHOD_LABEL[analysis['headline']['maximum_adapted_retention_drop']['method']]}、seed {analysis['headline']['maximum_adapted_retention_drop']['seed']}、{analysis['headline']['maximum_adapted_retention_drop']['context']}）。这是描述性三 seed 结果。",
        "",
        f"A2/B2 冷启动的 {analysis['headline']['cold_reuse_positive_pairs']}/{analysis['headline']['cold_reuse_pair_count']} 个配对中，父节点或同族已见兄弟均值复用的 1024 步累计奖励高于基础策略；跨配对平均差为 **{number(analysis['headline']['mean_cold_reuse_minus_foundation_reward'])}**。下面保留每个 seed、方法、容量档和留出任务的原始配对值；不以总体均值替代失败或反向结果。",
        "",
        "下表先按方法与容量档汇总三 seed 描述统计，逐 seed 数值紧随其后；它用于快速比较，不能掩盖单个失败。",
        "",
    ]
    diagnostic_lines = [
        "",
        "## 固定结构先验与表示诊断",
        "",
        "在公开且有利的固定结构先验下，tree 的 parent 修正确由在线奖励更新得到；本实验的冷启动配对和终点结果显示，已学修正能迁移到留出变体。家族分组、树层级、特征映射、gate 与秩预算均在训练前固定。因此，证据支持“在预设共享结构中在线学到可迁移知识”，不支持“从数据自动发现共享结构”。这项架构先验限制发现性主张，但不否定在线奖励确实学到了有效 parent 参数。",
        "",
        "Known-map observer 对固定 `data/foundation.pt` checkpoint 的动作重标记教师矩阵作离线可表示性诊断。它按构造分析 parent = Hᵀ(P_family − I)W_aug 与 child = Hᵀ(P_context,band − P_family)W_aug；诊断没有环境交互、agent 参数写入，也没有把教师权重灌入正式 agent。正式 agent 中的 parent 参数由在线奖励学习。",
        "",
        f"教师可精确加性分解为 family-parent rank 1 加 variant-local patch rank 1：六个上下文树父项与局部 patch 的 rank-1 相对 Frobenius 误差最大为 **{tree_error_max:.2e}**，完整 logits 重构误差为 **{additive_logit_error:.1f}**。扁平 rank 1 的教师重构误差为 **{flat_error_range}**，rank 2 可精确表示；跨六任务堆叠的字典目标 rank 为 {bounds['stacked_all_contexts_contrast_rank']}，因此 K=6 的理论最佳相对误差下界为 **{dictionary_k6_error:.5f}**，K=12 可精确表示。高容量 flat rank 2 / dictionary K=12 是表达能力控制；低容量 flat rank 1 和 dictionary K=6 对此教师目标先天地欠表达。",
        "",
        "上述 rank 误差只针对固定 W0 logits 的动作重标记教师，不是 RL 最优回报或 argmax 分类目标；别的 rank 1 策略仍可能达到相同任务行为。因此，不能从该诊断说 rank 1 方法无法解决奖励任务。已知地图教师也不是在线学得的知识或真实策略改进保证。",
        "",
    ]
    low_k6_seed71 = next((row for row in analysis["final_success"]
                          if row["tier"] == "low" and row["method"] == "dictionary" and row["seed"] == 71), None)
    if low_k6_seed71 and low_k6_seed71["all_six_pass"]:
        diagnostic_lines.extend([
            "一个边界例子：低容量 dictionary K=6 的 seed 71 在 final_eval 中六个任务均达到目标（各任务成功率见表）。这说明该诊断教师矩阵的精确可表示性不是 80% 奖励成功判据的必要条件；不能把 K=6 的矩阵逼近下界解释成任务失败预言。",
            "",
        ])
    method_table = []
    for row in analysis["method_summaries"]:
        method_table.append([row["tier"], METHOD_LABEL[row["method"]],
                             f"{row['final_task_successes']}/{row['final_task_opportunities']}",
                             f"{row['all_six_pass_runs']}/{row['formal_runs']}", percent(row["mean_final_success"]),
                             f"{row['adapted_retention_gate_passes']}/{row['retention_gate_count']} / {row['final_retention_gate_passes']}/{row['retention_gate_count']}",
                             number(row["mean_cold_reuse_minus_foundation_reward"]),
                             f"{number(row['mean_adaptive_residual_plus_adam_bytes'], 0)}"])
    lines.extend(markdown_table(["容量档", "方法", "final 达标任务数", "六任务全达标 run", "final 六任务均值成功率", "保持闸门 适应后 / 重访后", "cold 复用 − 基础平均奖励", "残差 + Adam 平均 bytes"], method_table))
    lines.extend(diagnostic_lines)
    lines.extend([
        "",
        "每个单点结果都在下表逐项列出；“六任务全部通过”要求该 run 的 A0、A1、A2、B0、B1、B2 各自达到 80%。",
        "",
        "## 冷启动：基础策略与公平结构复用",
        "",
        "冷评估使用同一 seed 与动作随机数，基础策略走 foundation route，复用组走 tree 的父节点或 flat/dictionary 的已见同族兄弟均值 route；两组都不更新参数。每格是单次 1024 步累计奖励。",
        "",
        f"正式 run 使用冻结配置：learning rate = {config['learning_rate']}, temperature = {config['temperature']}。早期 calibration-index 错误中断的 precheck 与正式 run 隔离，完整验证账本在文末逐项列出。",
    ])
    cold_rows = []
    for row in analysis["cold_evaluations"]:
        cold_rows.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]], row["context"],
                          number(row["foundation_reward"]), number(row["reuse_reward"]),
                          number(row["reuse_minus_foundation"])])
    lines.extend(markdown_table(["容量档", "seed", "方法", "留出任务", "基础策略", "结构复用", "复用 − 基础"], cold_rows))
    lines.extend(["", "## 留出适应后的学习与终点评估", "",
                  "训练阶段累计回报来自真实在线交互。seen_eval 只覆盖旧任务；adapted_eval 在 A2/B2 适应后评估六个任务；final_eval 在旧任务重访后评估六个任务。差值以同一 run、同一旧任务的 seen_eval 为基准。"])
    lines.append("")
    stage_rows = []
    for row in analysis["stages"]:
        phases = row["phases"]
        stage_rows.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]],
                           *(number(phases[phase]["cumulative_return"]) for phase in STAGES),
                           f"{row['total_train_steps']:,}", f"{row['evaluation_steps']:,}",
                           f"{row['oracle_calibration_steps']:,}", f"{row['optimizer_updates']:,}",
                           number(row["elapsed_s_including_initialization"]),
                           number(row["process_cpu_s_training_phases"])])
    lines.extend(markdown_table(["容量档", "seed", "方法", "已见在线奖赏", "留出适应在线奖赏", "旧任务重访在线奖赏", "历史续训在线奖赏", "训练步", "评估步", "oracle 校准转移", "优化器更新", "训练墙钟秒（含初始化）", "进程 CPU 秒（四训练阶段）"], stage_rows))
    lines.extend(["", "注：墙钟列将 heldout 初始化时间与各次 `model.learn` 的 `online_elapsed_s` 相加；进程 CPU 列累加四个 `model.learn` 的 `online_process_cpu_s`，不含未单独计时的 heldout 初始化。", ""])
    lines.extend(["", "### 旧任务保持闸门", "",
                  "下降差值为成功率百分点，负数表示成功率高于 seen_eval。适应后闸门与重访后闸门分别按 adapted_eval、final_eval 相对 seen_eval 判定，阈值均为下降不超过 5 pp。"])
    lines.append("")
    retention_table = []
    for row in analysis["retention"]:
        retention_table.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]], row["context"],
                                percent(row["seen_success"]), percent(row["adapted_success"]), percent(row["final_success"]),
                                number(row["adapted_delta_pp"]), number(row["final_delta_pp"]),
                                "通过" if row["adapted_gate_pass"] else "未通过",
                                "通过" if row["final_gate_pass"] else "未通过"])
    lines.extend(markdown_table(["容量档", "seed", "方法", "旧任务", "seen_eval", "adapted_eval", "final_eval", "adapted − seen (pp)", "final − seen (pp)", "适应后 5 pp 闸门", "重访后 5 pp 闸门"], retention_table))
    lines.extend(["", "### 适应后的即时下降（超过 5 pp）", "",
                  "下表逐项列出 adapted_eval 相对 seen_eval 下降超过 5 pp 的旧任务对照。列出重访后的成功率与差值，以显示哪些暂时退化后来恢复。"])
    lines.append("")
    transient_table = []
    for row in analysis["headline"]["adapted_retention_failures"]:
        transient_table.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]], row["context"],
                                percent(row["seen_success"]), percent(row["adapted_success"]),
                                number(row["adapted_delta_pp"]), percent(row["final_success"]),
                                number(row["final_delta_pp"]),
                                "恢复" if row["recovered_by_final"] else "重访后仍未恢复"])
    lines.extend(markdown_table(["容量档", "seed", "方法", "旧任务", "seen_eval", "adapted_eval", "即时差值 (pp)", "final_eval", "重访差值 (pp)", "状态"], transient_table))
    lines.extend(["", "### 六个任务的 final_eval", "",
                  "任务成功计数以每个 heldout run 中 6 个独立 final_eval 成功率是否达到 80% 计算。未达标任务仍保留在任务级表格中。", ""])
    final_table = []
    for row in analysis["final_success"]:
        final_table.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]],
                            *(percent(row["context_success"].get(context)) for context in CONTEXTS),
                            f"{row['target_success_count']}/6", "是" if row["all_six_pass"] else "否"])
    lines.extend(markdown_table(["容量档", "seed", "方法", *CONTEXTS, "达标任务", "六任务全达标"], final_table))
    lines.extend(["", "## 达到 80% 的时间：留出阶段 episodes.csv", "",
                  "分别在 A2、B2 统计首个达到 ≥80% 的滚动 20 个已完成、非 mixed_context 回合窗口；步数从各自 8,192 步 heldout bout 开始计算。B2 的原始阶段 step 会先减去 A2 bout 的 8,192 步偏移。未达到或有效回合不足 20 时在各自 8,192 步预算末右删失。这个 episode 估计依赖阶段日志，不替代固定步数 final_eval。", ""])
    rolling_table = []
    for row in analysis["rolling_20_episode_target_after_heldout"]:
        if row["status"] == "reached":
            time_cell = f"{row['event_step']:,} / 阶段 {row['event_step_phase']:,}"
        elif row["status"] == "episode_csv_missing":
            time_cell = "CSV 缺失"
        else:
            time_cell = f"≥{row['censored_at_step']:,}（删失）"
        rolling_table.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]], row["context"],
                              time_cell, row["observed_episodes"]])
    lines.extend(markdown_table(["容量档", "seed", "方法", "任务", "首次达标步 / 删失", "达标前或删失时的有效回合"], rolling_table))
    lines.extend(["", "## 自适应字节、共同容量上限与进程 RSS", "",
                  "自适应字节为 residual value + coordinate index + residual Adam tensor；不含共同冻结基础 actor 与 critic。`adaptive_cap_bytes` 是单独事先固定并强制执行的上限，完整 agent bundle 的安全余量不能挪给自适应张量。serialized core 是包含可恢复 agent/optimizer/router 状态的实测值；rollout 和进程 RSS 分开列出。容量比例比较 `peak_agent_bytes` 与事先固定的 `cap_bytes`；不把 RSS 或显式 tensor 下界当成可恢复序列化状态。", ""])
    resource_table = []
    for row in analysis["resources"]:
        cap = int(row["cap_bytes"])
        peak_agent = int(row["peak_agent_bytes"])
        headroom = cap - peak_agent
        resource_table.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]],
                               f"{row['adaptive_residual_plus_adam_bytes']:,}" if row["adaptive_residual_plus_adam_bytes"] is not None else "—",
                               f"{row['target_success_count']}/6", percent(row["mean_final_success"]),
                               f"{row['adaptive_cap_bytes']:,}",
                               f"{cap:,}", f"{peak_agent:,}", f"{headroom:,}", f"{100*peak_agent/cap:.1f}%" if cap else "—",
                               f"{row['peak_core_bytes']:,}", f"{row['peak_rollout_bytes']:,}",
                               f"{row['peak_sampled_rss_bytes']:,}", f"{row['peak_explicit_live_tensor_lower_bound_bytes']:,}"])
    lines.extend(markdown_table(["容量档", "seed", "方法", "残差 + Adam bytes", "final 达标数", "六任务均值成功率", "adaptive tensor cap bytes", "共同全 agent cap bytes", "peak agent bytes", "全 cap 余量", "peak/全 cap", "peak serialized core bytes", "peak rollout bytes", "peak sampled RSS bytes", "显式 live tensor 下界 bytes"], resource_table))
    lines.extend(["", "![实际自适应状态字节与 final success](adaptive_bytes_vs_success.png)", "",
                  "累计在线回报图将每阶段在线累计奖励顺序相加，并以细线显示每个 run、粗线显示每个方法与容量档的三 seed 分阶段均值。各阶段预算不同，表中同时列出实际交互步。", "",
                  "![在线学习分阶段累计回报](cumulative_returns.png)", "", "## 历史路由诊断", "",
                  "history 是加载已训练 agent、清空近期历史后的 calibrated-history 隐藏标签续训。它继续在六段各 1,024 步在线交互中训练；下表分开列 exact/family route accuracy 与该阶段奖励。识别置信度或准确率不是策略收益证据，本实验也没有策略改进认证门控。", ""])
    history_table = []
    for row in analysis["history_routing"]:
        history_table.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]], percent(row["exact_accuracy"]),
                              percent(row["family_accuracy"]), percent(row["parent_fallback_fraction"]),
                              percent(row["foundation_fallback_fraction"]), number(row["cumulative_return"]),
                              f"{int(row['steps'] or 0):,}"])
    lines.extend(markdown_table(["容量档", "seed", "方法", "exact accuracy", "family accuracy", "父回退占比", "基础回退占比", "history 在线奖励", "history 步"], history_table))
    lines.extend(["", "![历史路由 exact accuracy 与续训奖励](history_routing.png)", "",
                  "history_eval 在上述续训之后以已知 context route 对六个任务做独立无梯度评估。它测的是该续训后参数的条件表现，不是历史路由器在线识别表现；与主表中 history 续训前的 oracle-route final_eval 分开。", ""])
    history_eval_table = []
    for row in analysis["history_eval"]:
        history_eval_table.append([row["tier"], row["seed"], METHOD_LABEL[row["method"]], row["context"],
                                   percent(row["success_rate"]), number(row["cumulative_return"]),
                                   f"{row['successful_episodes']}/{row['completed_episodes']}" if row["completed_episodes"] is not None else "—",
                                   "是" if row["parameter_hash_unchanged"] else "否"])
    lines.extend(markdown_table(["容量档", "seed", "方法", "任务", "成功率", "评估累计奖励", "成功/完成回合", "评估未改参数"], history_eval_table))
    lines.extend(["",
                  "## 交互计数与验证前置条件", ""])
    totals = analysis["totals"]
    precheck = analysis.get("precheck") or {}
    lines.extend([
        f"正式训练交互 **{int(totals['training_interactions']):,} 步**；正式固定策略评估 **{int(totals['evaluation_interactions']):,} 步**。选定配置的六上下文 learnability precheck 另有 **{int(totals['selected_precheck_interactions']):,} 步**（含训练与独立评估）。正式阶段 seen/heldout/revisit 以 oracle 任务 ID 路由并用于转移模型校准，共 **{int(totals['oracle_labeled_calibration_interactions']):,} 个标注在线转移**；选定 precheck 训练另有 **{int(totals['validation_oracle_labeled_calibration_interactions']):,} 个**。history 的 **{sum(row['history_unlabeled_steps'] for row in totals['calibration_rows_by_run']):,} 步**使用既有校准模型并隐藏任务标签，不计入标定样本。",
        f"全部预正式验证账本共 **{int((analysis.get('validation_ledger') or {}).get('total_train_interactions', 0)):,} 个训练交互 + {int((analysis.get('validation_ledger') or {}).get('total_eval_interactions', 0)):,} 个评估交互 = {int(totals['validation_interactions']):,} 步**，包括早期失败试验、超参数验证、最终选定的六上下文前置验证及一个未记录进 CSV 的实际环境步。原始 calibration-index 错误中断的 precheck 没有用于正式学习或校准；详见下方逐条记录。",
        f"选定 precheck：{'通过' if precheck.get('passed') else '未通过'}；六个上下文均需在最多 {int(config['precheck_steps']) + int(config['precheck_extension_steps']):,} 学习步后达到 80% 成功率。",
        "",
        "验证前置六个上下文的逐项结果：",
        "",
    ])
    precheck_rows = []
    for case in precheck.get("cases", []):
        extension = case.get("extension_training")
        precheck_rows.append([case.get("context", "—"), percent(case.get("after", {}).get("success_rate")),
                              "通过" if case.get("passed") else "未通过",
                              f"{int(case.get('initial_training', {}).get('online_steps', 0)):,}",
                              f"{int(extension.get('online_steps', 0)):,}" if extension else "0",
                              f"{int(case.get('after', {}).get('completed_episodes', 0))}"])
    lines.extend(markdown_table(["上下文", "最终成功率", "80% 前置", "初始训练步", "延长训练步", "最终评估完成回合"], precheck_rows))
    ledger = analysis.get("validation_ledger")
    verification_trials = extract_validation_trials(ledger)
    selection_blindness = ((ledger or {}).get("formal_results_seen_during_selection") is False)
    lines.extend(["", "## 全部预正式验证试验", "",
                  f"以下附上 validation_ledger.json 中的全部 **{len(verification_trials)} 条试验记录**，包括失败的初始前置运行、最终选中配置前的超参数试验与选中后六上下文前置验证。记录按原始 JSON 保留，避免只报告入选的六项。账本记录正式结果是否参与筛选：{'未参与' if selection_blindness else '未确认，见原始账本'}。", ""])
    for index, trial in enumerate(verification_trials, start=1):
        lines.extend([f"### 验证记录 {index}", "", "```json",
                      json.dumps(trial, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    lines.extend([
        "",
        "![冷启动基础策略与结构复用配对回报](cold_reward.png)",
        "",
        "## 解释边界",
        "",
        "- A/B 两个家族各三个任务变体，64 维基础 actor 特征与三段可见 x gate 固定为共同 195 维叶特征；树另有 65 维全局父特征并获得父节点架构先验。前述 known-map teacher 重构分析揭示了该归纳偏置与任务设计的匹配，不是动态结构学习或自动发现抽象。",
        "- 所有正式 run 共用一个 foundation checkpoint；71、83、97 是部署学习 seed，不是三个独立预训练 seed。",
        "- seen、heldout、revisit 在线训练由 oracle 任务 ID 路由，并为转移模型提供标注校准样本；history 是另一个阶段，隐藏标签并用已校准经验模型推断路由。独立评估流无梯度更新，也不写入路由校准 bank、参数或原型。冷评估的复用只读取已见残差。",
        "- history 的非树 parent fallback 固定取同族前两个叶子的均值，即使第三个叶子后来已训练；tree parent 则通过三个叶子的 oracle 训练形成。因此 history 阶段的 route 比较不是最优后验路由或公平的三叶 fallback 基准，不能据此声称树在普遍隐藏标签路由上更优。主要共享机制证据应看 oracle-route 冷启动与受控终点评估。",
        "- 预定容量上限按参数、Adam、router、满窗口和固定 rollout 的恢复序列化测量设定，再加安全余量；本轮并非满容量淘汰实验。RSS、显式张量工作区与可恢复序列化字节不能互换。",
        "- history 使用 oracle 在线训练期间按标签校准的经验转移原型；历史推断仅接收状态、动作与下一状态。它是已校准模型上的隐藏标签部署续训，不是从零无监督情境发现。",
        "- 先前的 AB 分析性 history-only 对照强于 PPO，但不属于本次 C1 基准。局部状态置换避免单一全局旋转，状态依赖的逆动作映射仍可能成为简单且强的在线模型基线；本实验未比较该基线，也未测试全量在线微调。因此，结果不能说明参数梯度是必要的，或优于所有在线学习者。更广泛的论文主张还需加入状态感知的 history/model-based 基线与全量在线微调。",
        "- 结果是三 seed 的描述性比较；不提供显著性检验、单调变强证明、理论保证、真实机器人结论或整体算力优势。",
        "",
        "## 程序与绘图来源",
        "",
        "实验使用 Stable-Baselines3 PPO、Gymnasium 与 Matplotlib（版本见 requirements-lock.txt）。Matplotlib 技能支持本报告的静态科学图表；其程序知识来源为 Kassis, Agarwal, He, Patel 与 Brueckner（2026），[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)。该引用仅说明报告工具来源，不构成本实验的外部验证。",
        "",
        "[原始汇总](summary.json) · [逐项分析数据](analysis.json) · [全部验证试验记录](validation_ledger.json) · [结构可表示性诊断](representation_diagnostic.json) · [事先固定的协议](../../c1_protocol.md) · [配置](frozen_config.json)",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="C1 output directory (default: outputs/c1)")
    args = parser.parse_args()
    output = args.output.resolve()
    summary_path = output / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"C1 summary is not ready: {summary_path}")
    summary = load_json(summary_path)
    if not summary.get("complete") or len(summary.get("runs", [])) != 18:
        raise SystemExit(f"C1 formal results are incomplete: complete={summary.get('complete')}, runs={len(summary.get('runs', []))}/18")
    config = summary.get("config") or load_json(output / "frozen_config.json")
    expected = {(str(tier), method, int(seed)) for tier in config["tiers"] for method in METHODS for seed in config["seeds"]}
    observed = set(index_runs(summary))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SystemExit(f"C1 run keys do not match frozen design; missing={missing}, extra={extra}")
    ledger_path = output / "validation_ledger.json"
    if not ledger_path.exists():
        raise SystemExit(f"Complete validation ledger is required before reporting all verification trials: {ledger_path}")
    ledger = load_json(ledger_path)
    ledger_trials = extract_validation_trials(ledger)
    if not ledger_trials:
        raise SystemExit(f"No trial records found in validation ledger: {ledger_path}")
    if (ledger.get("total_interactions") is not None and
            summary.get("validation_interactions") is not None and
            int(ledger["total_interactions"]) != int(summary["validation_interactions"])):
        raise SystemExit("Formal summary validation_interactions does not match validation_ledger total_interactions")
    diagnostic_path = output / "representation_diagnostic.json"
    if not diagnostic_path.exists():
        raise SystemExit(f"Structural representation diagnostic is required for honest interpretation: {diagnostic_path}")

    analysis = make_analysis(output, summary, config)
    make_plots(output, analysis)
    (output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = build_report(analysis, summary)
    (output / "结果解读.md").write_text(report, encoding="utf-8")
    print(json.dumps({"runs": analysis["run_count"], "all_six_pass_runs": analysis["headline"]["all_six_pass_runs"],
                      "report": str(output / "结果解读.md"), "analysis": str(output / "analysis.json"),
                      "plots": [str(output / name) for name in ("cold_reward.png", "cumulative_returns.png", "adaptive_bytes_vs_success.png", "history_routing.png")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Aggregate the three stages without selecting hyperparameters from tests."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/knowledge_organization_report"


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def savefig(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def static_analysis():
    original = read("outputs/knowledge_static_formal/summary.json")
    global_control = read("outputs/knowledge_static_global_control/summary.json")
    distill = read("outputs/knowledge_distillation_control/summary.json")
    candidates = [r for r in original["candidate_results"] + global_control["candidate_results"]
                  if r["status"] == "complete"]
    candidates += [{**r, "method": "svd_policy_distillation", "codec_nbytes": r["actor_bytes"]}
                   for r in distill["results"]]
    result = []
    for (budget, method), rows in sorted(grouped(candidates, ("budget_bytes", "method")).items()):
        result.append({"budget": budget, "method": method, "seeds": len(rows),
                       "mean_success": mean(rows, "candidate_success_rate"),
                       "mean_return": mean(rows, "candidate_mean_return"),
                       "mean_actor_bytes": mean(rows, "codec_nbytes"),
                       "pass_count": sum(r["fixed_performance_gate_pass"] for r in rows)})
    labels = {"global_topk": "Global Top-K", "centroid2_sparse": "2 prototypes + residual",
              "truncated_svd": "Weight-SVD", "int8": "Independent int8",
              "svd_policy_distillation": "Policy-distilled low rank"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method, label in labels.items():
        values = sorted([r for r in result if r["method"] == method], key=lambda r: r["budget"])
        axes[0].plot([r["budget"] for r in values], [r["mean_success"] * 100 for r in values],
                     marker="o", markersize=4, linewidth=1.5, label=label)
    axes[0].set(xlabel="Allowed persistent actor bytes (including header)", ylabel="Success (%)",
                title="Static compression: same source basis", ylim=(0, 105))
    axes[0].legend(fontsize=8, loc="lower right")
    for method, label in [("truncated_svd", "Weight-SVD"), ("svd_policy_distillation", "Policy KL distillation")]:
        rows = [r for r in result if r["method"] == method and r["budget"] <= 448]
        axes[1].plot([r["mean_actor_bytes"] for r in rows], [r["mean_return"] for r in rows],
                     marker="o", label=label)
    axes[1].set(xlabel="Actual actor bytes", ylabel="Mean episode return",
                title="Same rank and bytes; different fitting loss")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "static_capacity")
    return result, candidates


def identification_analysis():
    data = read("outputs/knowledge_identification_formal/summary.json")
    summaries = []
    for (algorithm, window), rows in sorted(grouped(data["records"], ("algorithm", "window")).items()):
        item = {"algorithm": algorithm, "window": window}
        for key in ("transition_accuracy", "transition_logloss", "route_coverage",
                    "resolved_route_purity_by_slot_generation", "offline_ari_including_unresolved",
                    "known_context_revisit_generation_reuse", "allocations", "evictions",
                    "route_changes_inside_true_stable_segment",
                    "revisit_change_latency_mean_steps_censored_at_segment_length"):
            item[key] = mean(rows, key)
        summaries.append(item)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"robust_bank": "#0072B2", "naive_bank": "#D55E00", "no_bank_window": "#009E73"}
    for algorithm, color in colors.items():
        rows = [r for r in summaries if r["algorithm"] == algorithm]
        axes[0].plot([r["window"] for r in rows], [r["transition_accuracy"] for r in rows],
                     marker="o", label=algorithm.replace("_", " "), color=color)
    axes[0].set(xlabel="History window", ylabel="Next moving-transition accuracy",
                title="Same observable behavior stream", ylim=(0, 1))
    axes[0].legend(fontsize=9)
    rows = [r for r in data["records"] if r["algorithm"] == "robust_bank" and r["window"] == 32]
    labels = ["s" + str(r["seed"]) for r in rows]
    x = np.arange(len(rows))
    axes[1].bar(x - .2, [r["resolved_route_purity_by_slot_generation"] for r in rows], .2, label="Resolved purity")
    axes[1].bar(x, [r["offline_ari_including_unresolved"] for r in rows], .2, label="ARI")
    axes[1].bar(x + .2, [r["known_context_revisit_generation_reuse"] for r in rows], .2, label="Revisit reuse")
    axes[1].set(xticks=x, xticklabels=labels, ylim=(0, 1.05), title="High purity can hide fragmentation")
    axes[1].legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3)
    axes[1].set_title("High purity can hide fragmentation", pad=30)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "identification")
    return summaries


def online_analysis():
    path = ROOT / "outputs/knowledge_loop_formal/summary.json"
    if not path.exists():
        return None
    data = read("outputs/knowledge_loop_formal/summary.json")
    results = []
    for arm, rows in grouped(data["runs"], ("arm",)).items():
        arm = arm[0]
        commits = []
        for row in rows:
            file = ROOT / f"outputs/knowledge_loop_formal/seed_{row['seed']}/{arm}/commits.csv"
            if file.exists():
                commits += list(csv.DictReader(file.open(encoding="utf-8")))
        methods = Counter(r.get("selected_codec", "") for r in commits)
        results.append({"arm": arm, "seeds": len(rows),
                        "reward_per_step": mean(rows, "reward_per_step"),
                        "success_rate": mean(rows, "success_rate"),
                        "wall_s": mean(rows, "elapsed_seconds"),
                        "cpu_s": mean(rows, "process_cpu_seconds"),
                        "actor_bytes": mean(rows, "actor_compact_numeric_bytes"),
                        "checkpoint_bytes": mean(rows, "checkpoint_file_bytes"),
                        "mean_accepts": float(np.mean([r["counts"]["accepted"] for r in rows])),
                        "mean_rejects": float(np.mean([r["counts"]["rejected"] for r in rows])),
                        "mean_router_allocations": float(np.mean([r["counts"]["slot_allocations"] for r in rows])),
                        "training_interactions": sum(r["training_environment_steps"] for r in rows),
                        "old_revisit_interactions": sum(r["revisit_environment_steps"] for r in rows),
                        "selected_codecs": dict(methods)})
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for i, item in enumerate(results):
        ys = [r["reward_per_step"] for r in data["runs"] if r["arm"] == item["arm"]]
        ax.bar(i, np.mean(ys), color="#0072B2" if item["arm"] != "adaptive_assignment" else "#D55E00", alpha=.6)
        ax.scatter([i] * len(ys), ys, color="black", s=18, zorder=3)
    ax.set(xticks=np.arange(len(results)), xticklabels=[r["arm"].replace("_", "\n") for r in results],
           ylabel="All online reward / environment step", title="Unknown-boundary closed loop; dots = deployment seeds")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "online_return")
    return results


def corrected_revisit_analysis():
    path = ROOT / "outputs/knowledge_loop_revisit_corrected/summary.json"
    if not path.exists():
        return None
    data = read("outputs/knowledge_loop_revisit_corrected/summary.json")
    result = []
    for (arm,), rows in grouped(data["records"], ("arm",)).items():
        result.append({"arm": arm, "seeds": len(rows),
                       "paired_return_gain": mean(rows, "mean_paired_return_gain"),
                       "retain_success": mean(rows, "retain_success"),
                       "zero_success": mean(rows, "zero_success"),
                       "structural_resets": mean(rows, "structural_correction_resets")})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    static, _ = static_analysis()
    identification = identification_analysis()
    online = online_analysis()
    revisit = corrected_revisit_analysis()
    if online is None and not args.partial:
        raise RuntimeError("Formal closed loop is incomplete; no final report generated")
    value = {"static": static, "identification": identification, "online": online,
             "corrected_revisit": revisit,
             "complete_online_training": online is not None,
             "statistics": "Descriptive means across three deployment seeds, shared W0. No iid episode significance tests."}
    (OUT / "analysis.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({"static_groups": len(static), "identification_groups": len(identification),
                      "online_arms": None if online is None else len(online)}))


if __name__ == "__main__":
    main()

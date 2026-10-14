"""Report saved C1 alternative runs without starting or modifying experiments."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "c1_supplement"
REP = OUT / "representation_alternatives"
HEAD = OUT / "context_head_alternatives"
FORMAL_SEEDS = (181, 193, 211)
TENSOR_METHODS = ("tensor_learned", "output_mixer", "hierarchical_output_mixer")
ALL_CONTEXTS = ("A0", "A1", "A2", "B0", "B1", "B2")
OLD_CONTEXTS = ("A0", "A1", "B0", "B1")
EXPECTED_FORMAL = 18
EXPECTED_VALIDATION = 15

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_c1_supplement import sample as primary_sample  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _expected_paths():
    formal = []
    validation = []
    for method in TENSOR_METHODS:
        for seed in FORMAL_SEEDS:
            formal.append((method, "none", "low", seed,
                           REP / method / "formal" / method / "none" / "low" / f"s{seed}" / "result.json"))
        for candidate in range(3):
            validation.append((method, "none", "low", 919, candidate,
                               REP / method / "validation" / method / f"candidate_{candidate:02d}" / "s919" / "result.json"))
    for seed in FORMAL_SEEDS:
        formal.append(("tree_reference", "none", "low", seed,
                       REP / "tree_reference" / "formal" / "tree" / "none" / "low" / f"s{seed}" / "result.json"))
    for seed in FORMAL_SEEDS:
        formal.append(("full_context_head", "none", "high", seed,
                       HEAD / "formal" / "none" / f"s{seed}" / "result.json"))
        formal.append(("full_context_head", "ewc", "high", seed,
                       HEAD / "formal" / "ewc" / f"s{seed}" / "result.json"))
    for control in ("none", "ewc"):
        for candidate in range(3):
            validation.append(("full_context_head", control, "high", 923, candidate,
                               HEAD / "validation" / control / f"candidate_{candidate:02d}" / "s923" / "result.json"))
    return formal, validation


def _ensure_complete_inventory(formal_specs, validation_specs):
    expected_formal = {spec[-1].resolve() for spec in formal_specs}
    expected_validation = {spec[-1].resolve() for spec in validation_specs}
    found_formal = {
        path.resolve()
        for base in (REP / "tensor_learned" / "formal", REP / "output_mixer" / "formal",
                     REP / "hierarchical_output_mixer" / "formal", REP / "tree_reference" / "formal",
                     HEAD / "formal")
        for path in base.rglob("result.json")
    }
    found_validation = {
        path.resolve()
        for base in (REP / "tensor_learned" / "validation", REP / "output_mixer" / "validation",
                     REP / "hierarchical_output_mixer" / "validation", HEAD / "validation")
        for path in base.rglob("result.json")
    }
    missing_formal = sorted(str(path) for path in expected_formal - found_formal)
    extra_formal = sorted(str(path) for path in found_formal - expected_formal)
    missing_validation = sorted(str(path) for path in expected_validation - found_validation)
    extra_validation = sorted(str(path) for path in found_validation - expected_validation)
    if missing_formal or extra_formal or missing_validation or extra_validation:
        raise RuntimeError(
            "Alternative result inventory is incomplete or unexpected: "
            f"formal expected {EXPECTED_FORMAL}, found {len(found_formal)}; "
            f"validation expected {EXPECTED_VALIDATION}, found {len(found_validation)}. "
            f"Missing formal={missing_formal}; extra formal={extra_formal}; "
            f"missing validation={missing_validation}; extra validation={extra_validation}"
        )
    return expected_formal, expected_validation


def _selection_index():
    """Read the frozen validation selections, retaining every candidate."""
    tensor_frozen_path = REP / "frozen_settings.json"
    context_frozen_path = HEAD / "frozen_settings.json"
    required = [tensor_frozen_path, context_frozen_path,
                OUT / "step2_grouping" / "tree" / "selection.json"]
    missing = [str(path) for path in required if not path.is_file()]
    for method in TENSOR_METHODS:
        path = REP / method / "selection.json"
        if not path.is_file():
            missing.append(str(path))
    for name in ("selection.json", "ewc_selection.json"):
        path = HEAD / "validation" / name
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise RuntimeError("Missing frozen selection/provenance files: " + "; ".join(missing))

    index = {}
    for method in TENSOR_METHODS:
        selection = read(REP / method / "selection.json")
        for candidate in selection.get("candidates", []):
            number = int(candidate["candidate"])
            index[(method, "none", number)] = {
                "selected_candidate": selection.get("selected_candidate"),
                "saved_score": candidate.get("score"),
                "saved_status": candidate.get("status"),
                "params": candidate.get("params"),
                "coefficient": candidate.get("coefficient"),
            }
    for control, filename in (("none", "selection.json"), ("ewc", "ewc_selection.json")):
        selection = read(HEAD / "validation" / filename)
        for candidate in selection.get("candidates", []):
            number = int(candidate["candidate"])
            index[("full_context_head", control, number)] = {
                "selected_candidate": selection.get("selected_candidate"),
                "saved_score": candidate.get("score"),
                "saved_status": candidate.get("status"),
                "params": candidate.get("params"),
                "coefficient": candidate.get("coefficient"),
            }
    if len(index) != EXPECTED_VALIDATION:
        raise RuntimeError(f"Frozen selections contain {len(index)} candidate records, expected {EXPECTED_VALIDATION}")
    return index, read(tensor_frozen_path), read(context_frozen_path), read(OUT / "step2_grouping" / "tree" / "selection.json")


def _validation_score(result):
    if result.get("status") != "complete":
        return None
    final = result["final_eval"]
    adapted = result["adapted_eval"]
    final_mean = mean(final[name]["success_rate"] for name in ALL_CONTEXTS)
    old_mean = mean(adapted[name]["success_rate"] for name in OLD_CONTEXTS)
    heldout_mean = mean(adapted[name]["success_rate"] for name in ("A2", "B2"))
    return 0.50 * final_mean + 0.25 * old_mean + 0.25 * heldout_mean


def _label(row):
    label = row["group"]
    if row["control"] != "none":
        label += " + " + row["control"].upper()
    return label


def _capacity_peaks(result):
    capacities = []
    for phase in result.get("training", {}).values():
        if isinstance(phase, dict) and isinstance(phase.get("capacity"), dict):
            capacities.append(phase["capacity"])
    history = result.get("history", {})
    if isinstance(history, dict):
        capacities.append(history)
    return {
        "max_serialized_core_bytes": max((c.get("max_serialized_core_bytes", 0) for c in capacities), default=0),
        "max_rollout_bytes": max((c.get("max_rollout_bytes", 0) for c in capacities), default=0),
        "max_complete_agent_bytes_from_phases": max((c.get("max_complete_agent_bytes", 0) for c in capacities), default=0),
    }


def _optimizer_audit_index():
    audit_path = OUT / "optimizer_step_audit.json"
    if not audit_path.is_file():
        return {}
    audit = read(audit_path)
    indexed = {}
    for item in audit.get("runs", []):
        result_path = Path(item["result"])
        if not result_path.is_absolute():
            result_path = ROOT / result_path
        indexed[result_path.resolve()] = item
    return indexed


def _make_row(result_path, result, *, run_type, group, control, tier, seed,
              candidate=None, selections=None, optimizer_audits=None):
    row = {
        "run_type": run_type,
        "group": group,
        "method": result.get("method", group),
        "control": result.get("control", control),
        "tier": result.get("tier", tier),
        "seed": result.get("seed", seed),
        "candidate": candidate,
        "status": result.get("status", "missing_status"),
        "result_path": str(result_path.relative_to(ROOT)),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
        "params": result.get("params"),
        "coefficient": result.get("coefficient"),
        "budget_classification": result.get("budget_classification"),
        "complete_agent_cap_bytes": result.get("complete_agent_cap_bytes"),
        "adaptive_parameter_optimizer_cap_bytes": result.get("adaptive_parameter_optimizer_cap_bytes"),
    }
    if run_type == "validation":
        key = (group, control, int(candidate))
        selection = selections[key]
        score = _validation_score(result)
        stored_score = selection.get("saved_score")
        if score is not None and stored_score is not None and math.isfinite(float(stored_score)):
            if abs(score - float(stored_score)) > 1e-10:
                raise RuntimeError(f"Saved and recomputed selection scores differ for {result_path}: {stored_score} vs {score}")
        row["validation_selection_score"] = score
        row["selected_candidate"] = selection.get("selected_candidate")
        row["candidate_selected"] = (int(candidate) == selection.get("selected_candidate"))
        row["selection_status"] = selection.get("saved_status")
        row["selected_params"] = selection.get("params")
        if result.get("params"):
            row["params"] = result["params"]
        if selection.get("coefficient") is not None:
            row["coefficient"] = selection["coefficient"]
    else:
        row.update({"validation_selection_score": None, "selected_candidate": None,
                    "candidate_selected": None, "selection_status": None, "selected_params": None})
    if result.get("status") != "complete":
        return row

    metrics = primary_sample(result)
    row.update(metrics)
    audit = (optimizer_audits or {}).get(result_path.resolve(), {})
    corrected_updates = result.get("corrected_optimizer_steps", audit.get("corrected_optimizer_steps"))
    if corrected_updates is not None:
        row["raw_optimizer_steps"] = audit.get("raw_optimizer_steps", metrics["optimizer_steps"])
        row["optimizer_steps"] = int(corrected_updates)
        row["optimizer_step_count_source"] = "checkpoint_audit" if audit else "checkpoint_audit_injected"
    else:
        row["raw_optimizer_steps"] = metrics["optimizer_steps"]
        row["optimizer_step_count_source"] = "runner_counter"
    row["hidden_route_family_accuracy"] = result["history"]["route_family_accuracy"]
    row["seen_mask_bytes"] = result.get("seen_mask_bytes", result.get("representation", {}).get("coordinate_index_bytes", 0))
    # Compressed storage counts its mask in the adaptive allocation. Full
    # actor accounting already excludes that mask from the adaptive actor
    # subtotal, while the complete serialized state still includes it.
    row["adaptive_bytes_excluding_seen_mask"] = max(
        0, row["adaptive_bytes"] - (row["seen_mask_bytes"] if row["adaptive_parameter_optimizer_cap_bytes"] is not None else 0)
    )
    row["trainable_parameter_bytes"] = result.get("representation", {}).get("trainable_parameter_bytes")
    row["source_basis_bytes"] = result.get("representation", {}).get("source_basis_bytes", 0)
    row["source_basis_trainable"] = result.get("representation", {}).get("source_basis_trainable", False)
    row["context_head_parameter_bytes"] = result.get("representation", {}).get("context_head_parameter_bytes", 0)
    row["ewc_fisher_bytes"] = result.get("continual_state", {}).get("ewc_fisher_bytes", 0)
    row["ewc_reference_bytes"] = result.get("continual_state", {}).get("ewc_reference_bytes", 0)
    row["capacity_margin_bytes"] = row["complete_agent_cap_bytes"] - row["complete_agent_bytes"]
    row["capacity_pass"] = row["complete_agent_bytes"] <= row["complete_agent_cap_bytes"]
    row.update(_capacity_peaks(result))
    for name in ALL_CONTEXTS:
        row[f"final_{name}_success"] = result["final_eval"][name]["success_rate"]
    for name in OLD_CONTEXTS:
        row[f"seen_{name}_success"] = result["seen_eval"][name]["success_rate"]
        row[f"adapted_{name}_success"] = result["adapted_eval"][name]["success_rate"]
    for name in ("A2", "B2"):
        cold = result["cold"][name]
        row[f"cold_{name}_pristine_return"] = cold["online_cumulative_return"]
        row[f"cold_{name}_adapted_path_return"] = cold["adapted_path"]["online_cumulative_return"]
        row[f"cold_{name}_actual_path_gain"] = cold["adapted_path"]["online_cumulative_return"] - cold["online_cumulative_return"]
        row[f"cold_{name}_pristine_success"] = cold["success_rate"]
        row[f"cold_{name}_adapted_path_success"] = cold["adapted_path"]["success_rate"]
    row["formal_cohort"] = "seeds_181_193_211"
    return row


def _gather(formal_specs, validation_specs, selections):
    rows = []
    optimizer_audits = _optimizer_audit_index()
    for method, control, tier, seed, path in formal_specs:
        result = read(path)
        if int(result.get("seed", seed)) != seed or bool(result.get("validation", False)):
            raise RuntimeError(f"Formal metadata mismatch in {path}")
        if result.get("control", control) != control or result.get("tier", tier) != tier:
            raise RuntimeError(f"Formal control/tier mismatch in {path}")
        rows.append(_make_row(path, result, run_type="formal", group=method, control=control,
                              tier=tier, seed=seed, optimizer_audits=optimizer_audits))
    for method, control, tier, seed, candidate, path in validation_specs:
        result = read(path)
        if int(result.get("seed", seed)) != seed or not bool(result.get("validation", True)):
            raise RuntimeError(f"Validation metadata mismatch in {path}")
        if result.get("control", control) != control or result.get("tier", tier) != tier:
            raise RuntimeError(f"Validation control/tier mismatch in {path}")
        rows.append(_make_row(path, result, run_type="validation", group=method, control=control,
                              tier=tier, seed=seed, candidate=candidate, selections=selections,
                              optimizer_audits=optimizer_audits))
    if sum(row["run_type"] == "formal" for row in rows) != EXPECTED_FORMAL:
        raise RuntimeError("Formal cohort must contain exactly 18 rows")
    if sum(row["run_type"] == "validation" for row in rows) != EXPECTED_VALIDATION:
        raise RuntimeError("Validation cohort must contain exactly 15 rows")
    return rows


def _attach_source_retention(rows):
    """Join the optional forced-route-0 source probe to its same-seed W0."""
    path = OUT / "source_retention.json"
    if not path.is_file():
        return {"available": False, "reason": "source_retention.json is not present"}
    report = read(path)
    baseline = {int(row["seed"]): row for row in report.get("baseline", [])
                if int(row["seed"]) in FORMAL_SEEDS}
    if set(baseline) != set(FORMAL_SEEDS):
        raise RuntimeError(f"Source-retention report lacks matched-seed W0 controls for {FORMAL_SEEDS}")
    allowed_cohorts = {"representation_alternatives", "context_head_alternatives"}
    source_runs = [row for row in report.get("runs", [])
                   if row.get("cohort") in allowed_cohorts and int(row.get("seed", -1)) in FORMAL_SEEDS]
    keyed = {}
    for row in source_runs:
        key = (row["cohort"], row["method"], row.get("control", "none"), int(row["seed"]))
        if key in keyed:
            raise RuntimeError(f"Duplicate source-retention probe row for {key}")
        keyed[key] = row
    matches = 0
    for row in rows:
        if row["run_type"] != "formal":
            continue
        for name in ("source_probe_return", "source_probe_success", "source_w0_return",
                     "source_w0_success", "source_return_gain_vs_seed_w0", "source_success_gain_pp_vs_seed_w0"):
            row[name] = None
        row["source_probe_route"] = None
        row["source_probe_hash_unchanged"] = None
        if row["status"] != "complete":
            continue
        cohort = ("context_head_alternatives" if row["group"] == "full_context_head"
                  else "representation_alternatives")
        key = (cohort, row["method"], row["control"], int(row["seed"]))
        probe = keyed.get(key)
        w0 = baseline.get(int(row["seed"]))
        if w0 is None:
            raise RuntimeError(f"Source-retention report has no same-seed W0 baseline for seed {row['seed']}")
        if probe is None:
            continue
        row["source_probe_return"] = probe.get("return")
        row["source_probe_success"] = probe.get("success_rate")
        row["source_w0_return"] = w0.get("return")
        row["source_w0_success"] = w0.get("success_rate")
        row["source_return_gain_vs_seed_w0"] = probe.get("return", 0.0) - w0.get("return", 0.0)
        row["source_success_gain_pp_vs_seed_w0"] = 100 * (probe.get("success_rate", 0.0) - w0.get("success_rate", 0.0))
        row["source_probe_route"] = probe.get("route")
        row["source_probe_hash_unchanged"] = probe.get("parameter_hash_unchanged")
        matches += 1
    return {
        "available": True,
        "scope": report.get("scope"),
        "root_route_is_forced": True,
        "is_router_accuracy_test": False,
        "alternative_probe_rows": len(source_runs),
        "alternative_rows_matched": matches,
        "w0_baseline_seeds": sorted(baseline),
        "additional_evaluation_interactions": 512 * (len(source_runs) + len(baseline)),
        "unmatched_complete_formal_rows": [
            {"method": _label(row), "seed": row["seed"]}
            for row in rows if row["run_type"] == "formal" and row["status"] == "complete"
            and row.get("source_probe_return") is None
        ],
    }


def _summary(formal_rows):
    groups = defaultdict(list)
    for row in formal_rows:
        groups[_label(row)].append(row)
    summaries = []
    for label, members in sorted(groups.items()):
        complete = [row for row in members if row["status"] == "complete"]
        metrics = ("final_success", "immediate_old_success", "mean_old_drop_pp", "retained_old_tasks",
                   "cold_reward_gain", "hidden_reward", "hidden_success", "hidden_route_accuracy",
                   "hidden_route_family_accuracy", "adaptive_bytes_excluding_seen_mask", "complete_agent_bytes",
                   "cpu_s", "optimizer_steps", "training_interactions", "evaluation_interactions")
        means = {key: mean(row[key] for row in complete) for key in metrics} if complete else {}
        sample_sd = stdev(row["final_success"] for row in complete) if len(complete) > 1 else None
        summaries.append({
            "method": label,
            "tier": members[0]["tier"],
            "budget_classification": members[0]["budget_classification"],
            "expected_n": len(members),
            "complete_n": len(complete),
            "failed_n": len(members) - len(complete),
            "formal_seeds": [row["seed"] for row in members],
            "complete_seeds": [row["seed"] for row in complete],
            "final_success_sample_sd": sample_sd,
            "all_six_at_least_80pct_count": sum(bool(row["all_six_at_least_80pct"]) for row in complete),
            "all_six_at_least_80pct_denominator": len(complete),
            "retained_old_tasks_total": sum(int(row["retained_old_tasks"]) for row in complete),
            "retained_old_tasks_denominator": 4 * len(complete),
            "max_complete_agent_bytes": max((row["complete_agent_bytes"] for row in complete), default=None),
            "complete_agent_cap_bytes": members[0].get("complete_agent_cap_bytes"),
            "means": means,
            "source_probe_complete_n": sum(row.get("source_return_gain_vs_seed_w0") is not None for row in complete),
            "source_probe_mean_return_gain_vs_seed_w0": (
                mean(row["source_return_gain_vs_seed_w0"] for row in complete
                     if row.get("source_return_gain_vs_seed_w0") is not None)
                if any(row.get("source_return_gain_vs_seed_w0") is not None for row in complete) else None
            ),
            "source_probe_mean_success_gain_pp_vs_seed_w0": (
                mean(row["source_success_gain_pp_vs_seed_w0"] for row in complete
                     if row.get("source_success_gain_pp_vs_seed_w0") is not None)
                if any(row.get("source_success_gain_pp_vs_seed_w0") is not None for row in complete) else None
            ),
            "seed_rows": [{key: row.get(key) for key in (
                "seed", "status", "final_success", "all_six_at_least_80pct", "immediate_old_success",
                "mean_old_drop_pp", "retained_old_tasks", "cold_reward_gain", "hidden_reward", "hidden_success",
                "hidden_route_accuracy", "hidden_route_family_accuracy", "adaptive_bytes_excluding_seen_mask",
                "complete_agent_bytes", "complete_agent_cap_bytes", "cpu_s", "optimizer_steps",
                "raw_optimizer_steps", "optimizer_step_count_source", "source_probe_return", "source_probe_success",
                "source_w0_return", "source_w0_success", "source_return_gain_vs_seed_w0",
                "source_success_gain_pp_vs_seed_w0", "training_interactions", "evaluation_interactions")}
                          for row in members],
        })
    return summaries


def _write_csv(rows):
    fields = [
        "run_type", "formal_cohort", "group", "method", "control", "tier", "seed", "candidate", "status",
        "result_path", "error_type", "error", "params", "coefficient", "validation_selection_score",
        "selected_candidate", "candidate_selected", "selection_status", "budget_classification",
        "complete_agent_cap_bytes", "adaptive_parameter_optimizer_cap_bytes", "final_success",
        "all_six_at_least_80pct", "immediate_old_success", "mean_old_drop_pp", "retained_old_tasks",
        "cold_reward_gain", "hidden_reward", "hidden_success", "hidden_route_accuracy",
        "hidden_route_family_accuracy", "adaptive_bytes", "seen_mask_bytes", "adaptive_bytes_excluding_seen_mask",
        "source_basis_bytes", "source_basis_trainable", "context_head_parameter_bytes", "ewc_fisher_bytes",
        "ewc_reference_bytes", "complete_agent_bytes", "complete_agent_cap_bytes", "capacity_margin_bytes",
        "capacity_pass", "max_serialized_core_bytes", "max_rollout_bytes", "cpu_s", "optimizer_steps",
        "raw_optimizer_steps", "optimizer_step_count_source", "training_interactions", "evaluation_interactions",
        "source_probe_route", "source_probe_hash_unchanged", "source_probe_return", "source_probe_success",
        "source_w0_return", "source_w0_success", "source_return_gain_vs_seed_w0",
        "source_success_gain_pp_vs_seed_w0",
    ] + [f"final_{name}_success" for name in ALL_CONTEXTS]
    with (OUT / "alternative_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_clean(rows))


def _plot(formal_rows):
    methods = ("tree_reference", "tensor_learned", "output_mixer", "hierarchical_output_mixer",
               "full_context_head", "full_context_head + EWC")
    labels = ("tree low", "tensor learned low", "output mixer low", "hierarchical mixer low",
              "full context head", "full context head + EWC")
    color_map = dict(zip(methods, plt.get_cmap("tab10").colors[:len(methods)]))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    xjitter = (-0.12, 0.0, 0.12)
    complete = [row for row in formal_rows if row["status"] == "complete"]
    for x, (method, label) in enumerate(zip(methods, labels)):
        members = sorted((row for row in complete if _label(row) == method), key=lambda row: row["seed"])
        color = color_map[method]
        for index, row in enumerate(members):
            offset = xjitter[index % len(xjitter)]
            axes[0, 0].scatter(x + offset, row["final_success"] * 100, color=color, s=38, zorder=3)
            axes[0, 1].scatter(x + offset, row["mean_old_drop_pp"], color=color, s=38, zorder=3)
            axes[1, 0].scatter(x + offset, row["hidden_success"] * 100, color=color, s=38, zorder=3)
            axes[1, 1].scatter(row["adaptive_bytes_excluding_seen_mask"] / 1024,
                               row["final_success"] * 100, color=color, s=38, zorder=3)
        if members:
            for ax, key, factor in ((axes[0, 0], "final_success", 100),
                                    (axes[0, 1], "mean_old_drop_pp", 1),
                                    (axes[1, 0], "hidden_success", 100)):
                center = mean(row[key] for row in members) * factor
                ax.plot([x - .20, x + .20], [center, center], color=color, lw=2.2)
            adaptive = mean(row["adaptive_bytes_excluding_seen_mask"] for row in members) / 1024
            final = mean(row["final_success"] for row in members) * 100
            axes[1, 1].scatter(adaptive, final, color=color, marker="D", edgecolor="black", s=62, zorder=4)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(range(len(methods)), labels, rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=.22)
    axes[0, 0].set(title="Final success across all six contexts", ylabel="Success (%)", ylim=(-2, 103))
    axes[0, 1].set(title="Immediate old-context forgetting", ylabel="Mean positive drop (percentage points)")
    axes[1, 0].set(title="Actual hidden-router history stream", ylabel="Completed-episode success (%)", ylim=(-2, 103))
    axes[1, 1].set(title="Adaptive actor state and final success", xlabel="Adaptive parameters and optimizer (KiB, seen mask excluded; log scale)", ylabel="Mean six-context success (%)", ylim=(-2, 103), xscale="log")
    axes[1, 1].grid(alpha=.22, which="both")
    handles = [plt.Line2D([], [], color=color_map[method], marker="o", linestyle="None", label=label)
               for method, label in zip(methods, labels)]
    axes[1, 1].legend(handles=handles, loc="lower left", ncol=2, fontsize=8, framealpha=.92)
    fig.suptitle("C1 alternative methods · three formal seeds per method · points are seeds, lines/diamonds are means")
    fig.savefig(OUT / "alternative_comparisons.png", dpi=220)
    fig.savefig(OUT / "alternative_comparisons.pdf")
    plt.close(fig)


def _markdown(analysis):
    formal = analysis["formal_rows"]
    validation = analysis["validation_rows"]
    summaries = analysis["formal_summaries"]
    table = lambda headers, rows: "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
         *["| " + " | ".join(str(value) for value in row) + " |" for row in rows]])
    complete_n = sum(row["status"] == "complete" for row in formal)
    validation_complete = sum(row["status"] == "complete" for row in validation)
    lines = [
        "# C1 alternative methods: saved-data report", "",
        "This report reads the saved tensor and context-head runs only. It starts no experiment and makes no benchmark-wide or causal claim. The original Step 3 seeds 131, 149, and 167 are not pooled with this separate formal cohort (181, 193, and 211).",
        "", "## Scope and comparison conditions", "",
        "The six formal methods use the same single trained source checkpoint W0; seeds 181, 193, and 211 vary deployment runs, not pretraining. The methods are the frozen Step 2 low-tier tree reference; a trainable W0 source-basis tensor; frozen-basis output and hierarchical output mixers; a full shared actor with compact context heads; and that same context-head policy with EWC. The full actor uses the protocol amendment's 3 × 64 visible-band input bias. The four low-tier representations use the original compressed adaptive/complete caps. Full context-head runs are measured Pareto references outside the original adaptive caps and must fit the 393,216-byte complete-agent cap.",
        "", "During the seen, held-out, and revisit oracle phases, endpoint IDs and the phase schedule are supplied. Endpoint IDs and switch cues are hidden only during later history continuation; its router uses the transition bank calibrated during the declared oracle schedule, and that bank is frozen during the hidden stream. The fixed family structure is also supplied in this C1 setting. The source-retention probe forces root route 0 and tests neither router accuracy nor routing. The tensor basis is derived from W0's centered action head and spans action-row permutations; that inductive bias matches this synthetic action-remapping family. These runs do not establish transfer outside that source row space or from-zero latent-context discovery.",
        "", "The comparisons keep all saved candidates, including low-scoring candidates and failed runs. Failed result rows count toward the declared 18 formal and 15 validation files, but are excluded from performance averages. The validation score is the frozen `0.50 × final six-context success + 0.25 × immediate old-context success + 0.25 × held-out success` rule. The stored score is checked against the saved result before reporting.",
        "", "## Validation candidates and frozen selections", "",
        table(["Method / control", "Candidate", "LR", "Entropy", "EWC λ", "Validation score", "Status", "Selected"], [
            [_label(row), row["candidate"], row.get("params", {}).get("learning_rate", "—"),
             row.get("params", {}).get("ent_coef", "—"), row.get("coefficient", "—"),
             "—" if row.get("validation_selection_score") is None else f"{row['validation_selection_score']:.4f}",
             row["status"], "yes" if row.get("candidate_selected") else "no"]
            for row in validation]),
        "", "The tree reference reuses the already-frozen Step 2 tree selection; it was not retuned in this supplement. Tensor candidates use seed 919. Context-head learning-rate and EWC candidates use seed 923; EWC candidates reuse the ordinary context-head learning rate selected on that same validation seed. Candidate data are retained in `outputs/c1_supplement/alternative_summary.csv`.",
        "", "## Formal results", "",
        f"The saved inventory contains {len(formal)} formal rows ({complete_n} complete, {len(formal)-complete_n} failed) and {len(validation)} validation rows ({validation_complete} complete, {len(validation)-validation_complete} failed). Performance summaries below aggregate only complete formal rows.",
        "", "### Per-seed final success", "",
        table(["Method", "Seed", "A0", "A1", "A2", "B0", "B1", "B2", "Mean", "All six ≥ 80%"], [
            [row["group"] + (" + EWC" if row["control"] == "ewc" else ""), row["seed"],
             *[f"{100*row[f'final_{name}_success']:.1f}%" for name in ALL_CONTEXTS],
             f"{100*row['final_success']:.1f}%", "yes" if row["all_six_at_least_80pct"] else "no"]
            for row in formal if row["status"] == "complete"]),
        "", "### Across-seed summary", "",
        table(["Method", "Complete / 3", "Mean final success ± sample SD", "All six ≥ 80%", "Immediate old success", "Positive old drop (pp)", "Retained old contexts"], [
            [row["method"], f"{row['complete_n']}/3",
             ("—" if not row["complete_n"] else f"{100*row['means']['final_success']:.1f}% ± {100*row['final_success_sample_sd']:.1f} pp" if row["final_success_sample_sd"] is not None else f"{100*row['means']['final_success']:.1f}% ± —"),
             f"{row['all_six_at_least_80pct_count']}/{row['all_six_at_least_80pct_denominator']}",
             "—" if not row["complete_n"] else f"{100*row['means']['immediate_old_success']:.1f}%",
             "—" if not row["complete_n"] else f"{row['means']['mean_old_drop_pp']:.2f}",
             f"{row['retained_old_tasks_total']}/{row['retained_old_tasks_denominator']}"]
            for row in summaries]),
        "", "### Hidden-router continuation and deployment cost", "",
        table(["Method", "Hidden return", "Hidden success", "Route exact / family", "Adaptive actor + Adam (B)", "Peak complete state + rollout / cap (B)", "CPU (s)", "Actual optimizer steps", "Train / eval interactions"], [
            [row["method"], "—" if not row["complete_n"] else f"{row['means']['hidden_reward']:.1f}",
             "—" if not row["complete_n"] else f"{100*row['means']['hidden_success']:.1f}%",
             "—" if not row["complete_n"] else f"{100*row['means']['hidden_route_accuracy']:.1f}% / {100*row['means']['hidden_route_family_accuracy']:.1f}%",
             "—" if not row["complete_n"] else f"{row['means']['adaptive_bytes_excluding_seen_mask']:.0f}",
             "—" if row["max_complete_agent_bytes"] is None else f"{row['max_complete_agent_bytes']} / {row['complete_agent_cap_bytes']}",
             "—" if not row["complete_n"] else f"{row['means']['cpu_s']:.1f}",
             "—" if not row["complete_n"] else f"{row['means']['optimizer_steps']:.0f}",
             "—" if not row["complete_n"] else f"{row['means']['training_interactions']:.0f} / {row['means']['evaluation_interactions']:.0f}"]
            for row in summaries]),
        "", "Adaptive actor bytes exclude the endpoint seen mask; complete-state accounting keeps it and the 780-byte W0 source basis (frozen for the output mixers and trainable for `tensor_learned`), as well as optimizer state, router, continual state, and rollout buffer. Optimizer storage includes Adam moment tensors and scalar step state. The full context-head row includes all actor/critic, six heads, EWC/Fisher state when enabled, and rollout storage in the complete-agent cap. Per-run values and cap outcomes are in the CSV.",
        "", "### Actual-path cold gain over pristine W0", "",
        table(["Method", "A2 reward gain", "B2 reward gain", "Mean gain"], [
            [row["method"], "—" if not row["complete_n"] else f"{mean(r['cold_A2_actual_path_gain'] for r in formal if r['status']=='complete' and _label(r)==row['method']):.2f}",
             "—" if not row["complete_n"] else f"{mean(r['cold_B2_actual_path_gain'] for r in formal if r['status']=='complete' and _label(r)==row['method']):.2f}",
             "—" if not row["complete_n"] else f"{row['means']['cold_reward_gain']:.2f}"]
            for row in summaries]),
        "", "### Failed runs", "",
    ]
    source = analysis.get("source_retention", {})
    if source.get("available"):
        lines[-2:-2] = [
            "", "### Source retention versus matched-seed W0", "",
            f"This post-training diagnostic forces root route 0 for 512 NavigationEnv steps, makes no policy updates, and compares each saved policy with the pristine W0 built for the same seed. The alternative cohort contains {source['alternative_probe_rows']} trained checkpoints and {len(source['w0_baseline_seeds'])} matched-seed W0 controls ({source['additional_evaluation_interactions']:,} evaluation interactions). Root routing is fixed, so these values measure source-task behavior and do not measure router accuracy or routing.",
            "",
            table(["Method", "Matched probes", "Return gain vs same-seed W0", "Success gain (pp)"], [
                [row["method"], f"{row['source_probe_complete_n']}/{row['complete_n']}",
                 "—" if row["source_probe_mean_return_gain_vs_seed_w0"] is None else f"{row['source_probe_mean_return_gain_vs_seed_w0']:.2f}",
                 "—" if row["source_probe_mean_success_gain_pp_vs_seed_w0"] is None else f"{row['source_probe_mean_success_gain_pp_vs_seed_w0']:.2f}"]
                for row in summaries]),
            "",
        ]
    else:
        lines[-2:-2] = ["", "Source-retention results were not present when this report was generated; no source-retention claim is made.", ""]
    failures = [row for row in formal + validation if row["status"] != "complete"]
    if failures:
        lines.append(table(["Cohort", "Method / control", "Seed", "Candidate", "Status", "Error"], [
            [row["run_type"], _label(row), row["seed"], row.get("candidate", "—"), row["status"], row.get("error") or "—"]
            for row in failures]))
    else:
        lines.append("No saved formal or validation run failed.")
    lines += [
        "", "## Interpretation", "",
        "These are finite-seed C1 comparisons, not a ranking across unrelated budgets. The full actor is a higher-capacity reference, and the low-tier alternatives share a structured action-remapping prior. A strong tensor or mixer result would support this representation under the declared synthetic prior; it would not show that freezing a source basis helps general tasks. A weak or failed method remains visible and narrows the result to this protocol.",
        "", "Unknown boundaries occur in history continuation, but that stage is supplied a calibrated transition bank and fixed family structure. This is not from-zero latent-context discovery or dynamic-structure learning. The six arms share one W0, so source-pretraining variability is untested; the three formal seeds vary deployment runs only. Capacity replacement after the cap is full is also untested. EWC is the declared task-free diagonal empirical-Fisher adaptation. The context-head comparison does not reproduce a published continual-learning benchmark method. No CLEAR reimplementation is included in this supplemental report.",
        "", "## Protocol and source pointers", "",
        "- Protocol and evaluation schedule: `docs/C1_SUPPLEMENT_PROTOCOL.md` and `docs/C1_SUPPLEMENT_AMENDMENTS.md`.",
        "- Tensor architecture: `docs/C1_TENSOR_ALTERNATIVES.md`, `src/online_rl/c1_tensor_controls.py`, and `src/online_rl/c1_tensor_experiment.py`.",
        "- Context-head and EWC controls: `docs/C1_CONTEXT_HEAD_ALTERNATIVE.md`, `src/online_rl/c1_context_head_controls.py`, and `src/online_rl/c1_context_head_experiment.py`.",
        "- Shared runner and capacity accounting: `src/online_rl/c1_supplement_experiment.py`.",
        "- Data products: `outputs/c1_supplement/alternative_analysis.json`, `alternative_summary.csv`, `alternative_comparisons.png`, and `alternative_comparisons.pdf`.",
        "", "![Six-method C1 comparison](alternative_comparisons.png)", ""
    ]
    return "\n".join(lines)


def main():
    formal_specs, validation_specs = _expected_paths()
    _ensure_complete_inventory(formal_specs, validation_specs)
    selections, tensor_frozen, context_frozen, tree_selection = _selection_index()
    rows = _gather(formal_specs, validation_specs, selections)
    formal_rows = [row for row in rows if row["run_type"] == "formal"]
    validation_rows = [row for row in rows if row["run_type"] == "validation"]
    source_retention = _attach_source_retention(rows)
    summaries = _summary(formal_rows)
    analysis = {
        "scope": "C1 saved-data alternatives; no experiments started by this script",
        "formal_expected": EXPECTED_FORMAL,
        "formal_result_files": len(formal_rows),
        "formal_complete": sum(row["status"] == "complete" for row in formal_rows),
        "formal_failed": sum(row["status"] != "complete" for row in formal_rows),
        "formal_seeds": list(FORMAL_SEEDS),
        "validation_expected": EXPECTED_VALIDATION,
        "validation_result_files": len(validation_rows),
        "validation_complete": sum(row["status"] == "complete" for row in validation_rows),
        "validation_failed": sum(row["status"] != "complete" for row in validation_rows),
        "validation_seeds": {"tensor": 919, "context_head": 923},
        "tensor_frozen_settings": tensor_frozen,
        "context_head_frozen_settings": context_frozen,
        "tree_selection": tree_selection,
        "source_retention": source_retention,
        "formal_summaries": summaries,
        "formal_rows": formal_rows,
        "validation_rows": validation_rows,
        "formal_training_interactions": sum(row.get("training_interactions", 0) for row in formal_rows if row["status"] == "complete"),
        "formal_evaluation_interactions": sum(row.get("evaluation_interactions", 0) for row in formal_rows if row["status"] == "complete"),
        "validation_training_interactions": sum(row.get("training_interactions", 0) for row in validation_rows if row["status"] == "complete"),
        "validation_evaluation_interactions": sum(row.get("evaluation_interactions", 0) for row in validation_rows if row["status"] == "complete"),
        "source_retention_evaluation_interactions": source_retention.get("additional_evaluation_interactions", 0),
        "failed_runs": [row for row in rows if row["status"] != "complete"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alternative_analysis.json").write_text(json.dumps(_clean(analysis), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(rows)
    (OUT / "alternative_results.md").write_text(_markdown(analysis), encoding="utf-8")
    _plot(formal_rows)
    print(json.dumps({"formal": len(formal_rows), "formal_complete": analysis["formal_complete"],
                      "validation": len(validation_rows), "validation_complete": analysis["validation_complete"],
                      "outputs": ["alternative_analysis.json", "alternative_results.md", "alternative_summary.csv",
                                  "alternative_comparisons.png", "alternative_comparisons.pdf"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

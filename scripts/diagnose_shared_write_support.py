"""Offline support diagnostics for the S1b transition-profile grouper.

This module reads saved probe rows and never creates or steps an environment.
Family names are used only after grouping, for pairwise audit summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from online_rl.shared_write_grouping import (
    DynamicsGrouper,
    empty_profile,
    observe_profile,
    supported_directions,
)


DEFAULT_INPUT = ROOT / "outputs" / "knowledge_shared_write_auto_formal"
DEFAULT_OUTPUT = ROOT / "outputs" / "knowledge_shared_write_support_diagnostic"
CONFIG_PATH = ROOT / "configs" / "shared_write_auto.json"
SEEDS = (677, 691, 709)
CONTEXT_ORDER = ("A0", "B0", "A2", "B2", "A1", "B1")
PREFIXES = (16, 32, 64, 128, 256, 512)
PROFILE_CELLS = 3 * 4
PAIR_COUNT = 6 * 5 // 2
EXPECTED_GROUPER_CONFIG: dict[str, int | float] = {
    "max_groups": 2,
    "min_overlap": 4,
    "min_agreement": 0.6,
    "min_cell_support": 2,
    "novelty_max_agreement": 0.3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_probe_rows(path: Path) -> tuple[dict[str, list[dict[str, str]]], int]:
    by_context = {context: [] for context in CONTEXT_ORDER}
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"context", "obs", "action", "next_obs"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} is missing required probe columns: {sorted(required)}")
        for row in reader:
            context = row["context"]
            if context not in by_context:
                raise ValueError(f"unexpected context {context!r} in {path}")
            by_context[context].append(row)
            row_count += 1

    wrong_counts = {context: len(rows) for context, rows in by_context.items() if len(rows) != 512}
    if wrong_counts:
        raise ValueError(f"expected 512 saved probe rows per context in {path}; got {wrong_counts}")
    return by_context, row_count


def rebuild_profile(
    rows: list[dict[str, str]],
    *,
    action_filter: frozenset[int] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rebuild one profile from saved transition rows, without environment access."""

    profile = empty_profile()
    rows_scanned = 0
    rows_used = 0
    movement_transitions = 0
    for row in rows:
        rows_scanned += 1
        action = int(row["action"])
        if action_filter is not None and action not in action_filter:
            continue
        obs = json.loads(row["obs"])
        next_obs = json.loads(row["next_obs"])
        rows_used += 1
        movement_transitions += int(observe_profile(profile, obs, action, next_obs))
    return profile, {
        "rows_scanned_for_profile": rows_scanned,
        "rows_used_for_profile": rows_used,
        "movement_transitions_counted": movement_transitions,
    }


def profile_support(profile: np.ndarray, rows_used: int) -> dict[str, int | float | list[int]]:
    directions = supported_directions(profile)
    supported_by_band = np.count_nonzero(directions >= 0, axis=1)
    supported_cells = int(np.count_nonzero(directions >= 0))
    return {
        "rows_used_for_profile": int(rows_used),
        "movement_transitions_counted": int(np.sum(profile, dtype=np.uint64)),
        "supported_cells": supported_cells,
        "unsupported_cells": PROFILE_CELLS - supported_cells,
        "supported_cell_coverage": supported_cells / PROFILE_CELLS,
        "supported_cells_by_x_band": [int(value) for value in supported_by_band],
    }


def pairwise_family_audit(
    assignments: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score family agreement only for resolved pairs; count unresolved pairs separately."""

    cells: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    resolved = 0
    unresolved = 0
    for left_index, left in enumerate(CONTEXT_ORDER):
        for right in CONTEXT_ORDER[left_index + 1 :]:
            left_group = int(assignments[left])
            right_group = int(assignments[right])
            same_family = left[0] == right[0]
            if left_group < 0 or right_group < 0:
                pair_status = "unresolved"
                unresolved += 1
                correct: bool | None = None
            else:
                predicted_same = left_group == right_group
                correct = predicted_same == same_family
                pair_status = "correct" if correct else "error"
                resolved += 1
                if predicted_same and same_family:
                    cells["true_positive"] += 1
                elif predicted_same:
                    cells["false_positive"] += 1
                elif same_family:
                    cells["false_negative"] += 1
                else:
                    cells["true_negative"] += 1
            rows.append(
                {
                    "context_left": left,
                    "family_left_audit_only": left[0],
                    "group_left": left_group,
                    "context_right": right,
                    "family_right_audit_only": right[0],
                    "group_right": right_group,
                    "pair_status": pair_status,
                    "family_pair_correct": correct,
                }
            )

    errors = cells["false_positive"] + cells["false_negative"]
    audit = {
        "total_context_pairs": PAIR_COUNT,
        "resolved_pairs": resolved,
        "unresolved_pairs": unresolved,
        "resolved_pair_coverage": resolved / PAIR_COUNT,
        "pairwise_accuracy_over_resolved_pairs": (resolved - errors) / resolved if resolved else None,
        "pairwise_error_rate_over_resolved_pairs": errors / resolved if resolved else None,
        "confusion_counts": {
            "true_positive": cells["true_positive"],
            "true_negative": cells["true_negative"],
            "false_positive": cells["false_positive"],
            "false_negative": cells["false_negative"],
        },
    }
    return audit, rows


def summarize_assignments(
    decisions: list[dict[str, Any]],
    grouper: DynamicsGrouper,
    *,
    evidence_mode: str,
    input_rows_loaded_for_seed: int,
    profile_rows_scanned: int,
    profile_rows_reprocessed: int,
    source_elapsed_steps_by_context: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assignments = {row["context"]: int(row["assigned_group"]) for row in decisions}
    audit, pair_rows = pairwise_family_audit(assignments)
    assigned_contexts = sum(group >= 0 for group in assignments.values())
    supported_coverage = float(np.mean([row["supported_cell_coverage"] for row in decisions]))
    return (
        {
            "evidence_mode": evidence_mode,
            "context_order": list(CONTEXT_ORDER),
            "input_rows_loaded_for_seed": int(input_rows_loaded_for_seed),
            "profile_source_rows_scanned_across_six_contexts": int(profile_rows_scanned),
            "profile_rows_reprocessed_across_six_contexts": int(profile_rows_reprocessed),
            "historical_source_probe_steps_by_context": {
                key: int(value) for key, value in source_elapsed_steps_by_context.items()
            },
            "assigned_contexts": assigned_contexts,
            "unresolved_contexts": len(CONTEXT_ORDER) - assigned_contexts,
            "assignment_coverage": assigned_contexts / len(CONTEXT_ORDER),
            "mean_supported_cell_coverage": supported_coverage,
            "reason_counts": dict(Counter(row["reason"] for row in decisions)),
            "numeric_state_bytes": grouper.nbytes,
            "pairwise_family_audit": audit,
        },
        pair_rows,
    )


def _new_fixed_grouper(config: Mapping[str, Any]) -> DynamicsGrouper:
    return DynamicsGrouper(**dict(config))


def _run_sequence(
    seed: int,
    rows_by_context: Mapping[str, list[dict[str, str]]],
    grouper_config: Mapping[str, Any],
    *,
    evidence_mode: str,
    input_rows_loaded_for_seed: int,
    profile_builder: Callable[[str], tuple[np.ndarray, dict[str, int]]],
    source_elapsed_steps_by_context: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # Context strings determine only which offline rows are read and output order.
    # The call to assign receives a numeric profile and no route/family metadata.
    grouper = _new_fixed_grouper(grouper_config)
    decisions: list[dict[str, Any]] = []
    profile_rows_scanned = 0
    profile_rows_reprocessed = 0
    for context in CONTEXT_ORDER:
        profile, row_meta = profile_builder(context)
        profile_rows_scanned += int(row_meta["rows_scanned_for_profile"])
        profile_rows_reprocessed += int(row_meta["rows_used_for_profile"])
        assignment = grouper.assign(profile)
        summary = profile_support(profile, row_meta["rows_used_for_profile"])
        row = {
            "seed": int(seed),
            "evidence_mode": evidence_mode,
            "context": context,
            "family_audit_only": context[0],
            "assigned_group": int(assignment.group),
            "reason": str(assignment.reason),
            "scores": assignment.scores,
            "scores_json": json.dumps(assignment.scores, sort_keys=True),
            **row_meta,
            **summary,
            "source_elapsed_probe_steps": int(source_elapsed_steps_by_context[context]),
            "assignment_coverage_so_far": sum(item["assigned_group"] >= 0 for item in decisions) + int(assignment.group >= 0),
            "unresolved_contexts_so_far": sum(item["assigned_group"] < 0 for item in decisions) + int(assignment.group < 0),
            "numeric_state_bytes": grouper.nbytes,
        }
        decisions.append(row)

    summary, pair_rows = summarize_assignments(
        decisions,
        grouper,
        evidence_mode=evidence_mode,
        input_rows_loaded_for_seed=input_rows_loaded_for_seed,
        profile_rows_scanned=profile_rows_scanned,
        profile_rows_reprocessed=profile_rows_reprocessed,
        source_elapsed_steps_by_context=source_elapsed_steps_by_context,
    )
    for pair_row in pair_rows:
        pair_row["seed"] = int(seed)
        pair_row["evidence_mode"] = evidence_mode
    return decisions, summary, pair_rows


def evaluate_tied_prototypes(
    seed: int,
    profiles: Mapping[str, np.ndarray],
    grouper_config: Mapping[str, Any],
    *,
    source_rows_used: Mapping[str, int],
) -> dict[str, Any]:
    """Force an explicit tie from offline A0/A1 references and an A2 cell subset."""

    a0 = profiles["A0"]
    a1 = profiles["A1"]
    a2 = profiles["A2"]
    d0 = supported_directions(a0)
    d1 = supported_directions(a1)
    d2 = supported_directions(a2)
    common_equal = (d0 >= 0) & (d1 >= 0) & (d2 >= 0) & (d0 == d1) & (d1 == d2)
    common_cells = int(np.count_nonzero(common_equal))
    if common_cells < int(grouper_config["min_overlap"]):
        raise ValueError(
            f"seed {seed}: only {common_cells} common A0/A1/A2 cells are available for tied prototype test"
        )

    # The two aggregate prototypes are deliberately installed as independent
    # references in a directly edited state_dict. They are not discovered groups.
    state = _new_fixed_grouper(grouper_config).state_dict()
    references = np.stack((a0, a1)).astype(np.uint16, copy=True)
    state["group_counts"] = references
    state["nbytes"] = int(references.nbytes)
    grouper = DynamicsGrouper.from_state_dict(state)

    query = a2.copy()
    query[~common_equal, :] = 0
    assignment = grouper.assign(query, commit=False)
    scores = assignment.scores
    tied_scores = (
        0 in scores
        and 1 in scores
        and int(scores[0]["overlap"]) == int(scores[1]["overlap"])
        and np.isclose(float(scores[0]["agreement"]), float(scores[1]["agreement"]), rtol=0.0, atol=1e-12)
    )
    if assignment.group != -1 or assignment.reason != "ambiguous_match" or not tied_scores:
        raise AssertionError(
            f"seed {seed}: explicit tie failed; got group={assignment.group}, "
            f"reason={assignment.reason}, scores={scores}"
        )
    return {
        "seed": int(seed),
        "diagnostic_type": "synthetic_tied_prototype_pressure_test",
        "uses_saved_probe_profiles_offline": True,
        "not_a_training_discovery_or_measured_group_assignment": True,
        "state_construction": "directly populated DynamicsGrouper.state_dict group_counts from full A0 and A1 profiles as separate reference groups",
        "query_construction": "A2 full profile masked to cells where full A0, A1, and A2 all have the same supported direction",
        "reference_contexts": ["A0", "A1"],
        "query_context": "A2",
        "source_rows_reused_from_full_profile_audit": {
            key: int(value) for key, value in source_rows_used.items()
        },
        "profile_rows_reused_from_full_profile_audit": int(sum(source_rows_used.values())),
        "additional_profile_rows_reprocessed": 0,
        "common_equal_supported_cells": common_cells,
        "query_supported_cells": int(np.count_nonzero(supported_directions(query) >= 0)),
        "scores": scores,
        "returned_group": int(assignment.group),
        "reason": str(assignment.reason),
        "numeric_state_bytes": grouper.nbytes,
        "new_environment_interactions": 0,
    }


def audit_local_context_references(
    seed: int,
    full_profiles: Mapping[str, np.ndarray],
    censored_profiles: Mapping[str, np.ndarray],
    censored_decisions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare censored profiles with same-seed full profiles after grouping.

    Context names are observer-only reference labels. The function is called
    after the grouper has made all decisions and does not feed back into it.
    """

    decisions_by_context = {row["context"]: row for row in censored_decisions}
    rows: list[dict[str, Any]] = []
    for query_context in CONTEXT_ORDER:
        query_directions = supported_directions(censored_profiles[query_context])
        query_mask = query_directions >= 0
        query_supported_cells = int(np.count_nonzero(query_mask))
        query_full_directions = supported_directions(full_profiles[query_context])
        unprobed_full_support = (~query_mask) & (query_full_directions >= 0)
        unprobed_actions = [
            action
            for action in range(4)
            if bool(np.any(unprobed_full_support[:, action]))
        ]

        no_conflict_references: list[str] = []
        fully_covered_no_conflict_references: list[str] = []
        differing_references_on_unprobed_cells: list[str] = []
        reference_details: dict[str, dict[str, int | bool | None]] = {}
        for reference_context in CONTEXT_ORDER:
            reference_directions = supported_directions(full_profiles[reference_context])
            overlap = query_mask & (reference_directions >= 0)
            conflicts = overlap & (reference_directions != query_directions)
            overlap_cells = int(np.count_nonzero(overlap))
            conflict_cells = int(np.count_nonzero(conflicts))
            no_conflict = overlap_cells > 0 and conflict_cells == 0
            covers_all_query_support = overlap_cells == query_supported_cells
            if no_conflict:
                no_conflict_references.append(reference_context)
                if covers_all_query_support:
                    fully_covered_no_conflict_references.append(reference_context)

            comparable_unprobed = unprobed_full_support & (reference_directions >= 0)
            hidden_differences = int(
                np.count_nonzero(
                    comparable_unprobed
                    & (reference_directions != query_full_directions)
                )
            )
            if hidden_differences and no_conflict:
                differing_references_on_unprobed_cells.append(reference_context)
            reference_details[reference_context] = {
                "overlap_cells_on_censored_support": overlap_cells,
                "conflicting_cells_on_censored_support": conflict_cells,
                "no_conflict_on_observed_overlap": no_conflict,
                "covers_all_query_supported_cells": covers_all_query_support,
                "full_profile_differences_on_unprobed_cells": hidden_differences,
            }

        decision = decisions_by_context[query_context]
        same_family_no_conflict = [
            name for name in fully_covered_no_conflict_references if name[0] == query_context[0]
        ]
        other_family_no_conflict = [
            name for name in fully_covered_no_conflict_references if name[0] != query_context[0]
        ]
        rows.append(
            {
                "seed": int(seed),
                "query_context_audit_label_only": query_context,
                "observer_family_audit_only": query_context[0],
                "action_censoring_mode": "offline_action_censoring_0_1",
                "censored_assigned_group": int(decision["assigned_group"]),
                "censored_assignment_reason": str(decision["reason"]),
                "censored_supported_cells": query_supported_cells,
                "unprobed_actions_with_full_profile_support": unprobed_actions,
                "unprobed_full_profile_cells": int(np.count_nonzero(unprobed_full_support)),
                "no_conflict_full_reference_names": no_conflict_references,
                "no_conflict_full_reference_count": len(no_conflict_references),
                "fully_covered_no_conflict_reference_names": fully_covered_no_conflict_references,
                "fully_covered_no_conflict_reference_count": len(fully_covered_no_conflict_references),
                "same_family_fully_covered_no_conflict_reference_names_audit_only": same_family_no_conflict,
                "other_family_fully_covered_no_conflict_reference_names_audit_only": other_family_no_conflict,
                "local_context_identifiable_from_censored_profile": len(fully_covered_no_conflict_references) == 1,
                "local_context_ambiguity_present": len(fully_covered_no_conflict_references) > 1,
                "candidate_references_with_differences_hidden_on_unprobed_cells": differing_references_on_unprobed_cells,
                "reference_details": reference_details,
            }
        )

    summary = {
        "diagnostic_type": "offline_action_censoring_local_context_reference_audit",
        "oracle_observer_reference_only": True,
        "reference_context_names_passed_to_grouper": False,
        "reference_profiles_by_seed": "full saved probe profiles from the same seed, constructed after censored assignments",
        "query_profiles": "the same action 0/1 censored profiles used for the grouper decisions",
        "comparison_rule": "A reference is a fully covered no-conflict candidate when it supports every cell supported by the censored query and its direction agrees on each such cell.",
        "contexts_with_multiple_fully_covered_no_conflict_references": sum(
            len(row["fully_covered_no_conflict_reference_names"]) > 1 for row in rows
        ),
        "contexts_with_hidden_differences_on_unprobed_cells": sum(
            bool(row["candidate_references_with_differences_hidden_on_unprobed_cells"])
            for row in rows
        ),
        "new_environment_interactions": 0,
        "per_context": rows,
    }
    return summary, rows


def _build_plot(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    prefix_rows = [row for row in summary_rows if row["plot_category"] == "prefix"]
    censored_rows = [row for row in summary_rows if row["plot_category"] == "action_censored"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    metrics = (
        ("assignment_coverage", "Assigned context coverage"),
        ("pairwise_family_audit", "Pairwise audit coverage and error"),
    )
    x_values = np.asarray(PREFIXES, dtype=np.float64)
    for axis, (metric, title) in zip(axes, metrics):
        def draw_series(
            value_getter: Callable[[dict[str, Any]], float | None],
            *,
            label: str,
            color: str,
            linestyle: str = "-",
            marker: str = "o",
        ) -> None:
            values = []
            spreads = []
            for prefix in PREFIXES:
                rows = [row for row in prefix_rows if row["x_value"] == prefix]
                samples = [value_getter(row) for row in rows]
                samples = [float(value) for value in samples if value is not None]
                values.append(float(np.mean(samples)) if samples else float("nan"))
                spreads.append(float(np.std(samples)) if samples else 0.0)
            axis.errorbar(
                x_values,
                values,
                yerr=spreads,
                marker=marker,
                capsize=3,
                linewidth=1.7,
                linestyle=linestyle,
                color=color,
                label=label,
            )

        if metric == "assignment_coverage":
            draw_series(
                lambda row: float(row["assignment_coverage"]),
                label="Assigned context coverage",
                color="#2878B5",
            )
            censored_values = [float(row["assignment_coverage"]) for row in censored_rows]
            axis.errorbar(
                [512],
                [float(np.mean(censored_values))],
                yerr=[float(np.std(censored_values))],
                marker="D",
                markersize=6,
                capsize=3,
                linestyle="none",
                color="#D95F02",
                label="_nolegend_",
            )
        else:
            draw_series(
                lambda row: float(row["pairwise_family_audit"]["resolved_pair_coverage"]),
                label="Resolved pair coverage",
                color="#6A3D9A",
                linestyle="--",
                marker="s",
            )
            draw_series(
                lambda row: (
                    None
                    if row["pairwise_family_audit"]["pairwise_error_rate_over_resolved_pairs"] is None
                    else float(row["pairwise_family_audit"]["pairwise_error_rate_over_resolved_pairs"])
                ),
                label="Family error among resolved pairs",
                color="#2878B5",
            )
            censored_coverage = [
                float(row["pairwise_family_audit"]["resolved_pair_coverage"])
                for row in censored_rows
            ]
            censored_errors = [
                float(row["pairwise_family_audit"]["pairwise_error_rate_over_resolved_pairs"])
                for row in censored_rows
                if row["pairwise_family_audit"]["pairwise_error_rate_over_resolved_pairs"] is not None
            ]
            for censored_values in (censored_coverage, censored_errors):
                if censored_values:
                    axis.errorbar(
                        [512],
                        [float(np.mean(censored_values))],
                        yerr=[float(np.std(censored_values))],
                        marker="D",
                        markersize=6,
                        capsize=3,
                        linestyle="none",
                        color="#D95F02",
                        label="_nolegend_",
                    )
        axis.set_title(title)
        axis.set_xlabel("Offline source probe rows per context")
        axis.set_xlim(0, 540)
        axis.set_ylim(-0.04, 1.04)
        axis.set_xticks(PREFIXES)
        axis.grid(True, alpha=0.28)

    axes[0].set_ylabel("Fraction")
    axes[1].set_ylabel("Fraction")
    fig.suptitle("S3 support diagnostic — offline replay; 0 new environment interactions")
    legend_handles = [
        Line2D([0], [0], color="#2878B5", marker="o", linewidth=1.7, label="Prefix evidence (mean ± SD, 3 seeds)"),
        Line2D([0], [0], color="#6A3D9A", marker="s", linestyle="--", linewidth=1.7, label="Resolved pair coverage"),
        Line2D([0], [0], color="#2878B5", marker="o", linewidth=1.7, label="Family error among resolved pairs"),
        Line2D([0], [0], color="#D95F02", marker="D", linestyle="none", label="Actions 0/1 retained; source elapsed 512"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(value, sort_keys=True)
            writer.writerow(row)


def run_diagnostic(input_root: Path, output_root: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    config_document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    grouper_config = dict(config_document["grouper"])
    if grouper_config != EXPECTED_GROUPER_CONFIG:
        raise ValueError(
            "the saved S1b grouper config differs from this fixed diagnostic configuration: "
            f"{grouper_config!r}"
        )

    input_paths = {
        seed: input_root / f"seed_{seed}" / "independent" / "probe_steps.csv"
        for seed in SEEDS
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing independent probe inputs: " + ", ".join(missing))
    expected_outputs = {
        "summary.json",
        "assignments.csv",
        "pairwise_audit.csv",
        "tied_prototype_checks.json",
        "support_coverage_error.png",
        "local_reference_audit.json",
        "local_reference_audit.csv",
    }
    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing:
            names = {path.name for path in existing}
            summary_path = output_root / "summary.json"
            is_own_report = False
            if names.issubset(expected_outputs) and summary_path.is_file():
                try:
                    previous = json.loads(summary_path.read_text(encoding="utf-8"))
                    is_own_report = (
                        previous.get("diagnostic")
                        == "S3 offline shared-write evidence support and ambiguity diagnostic"
                        and previous.get("input_root") == str(input_root)
                    )
                except (OSError, json.JSONDecodeError):
                    is_own_report = False
            if not is_own_report:
                raise FileExistsError(
                    f"refusing to overwrite a non-empty or unrelated output directory: {output_root}"
                )
    output_root.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        "shared_write_auto_config": {
            "path": str(CONFIG_PATH.resolve()),
            "sha256": sha256_file(CONFIG_PATH),
        }
    }
    loaded: dict[int, dict[str, list[dict[str, str]]]] = {}
    file_row_counts: dict[int, int] = {}
    for seed, path in input_paths.items():
        by_context, row_count = _load_probe_rows(path)
        loaded[seed] = by_context
        file_row_counts[seed] = row_count
        input_hashes[f"seed_{seed}_independent_probe_steps"] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": row_count,
        }

    assignment_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    tied_checks: list[dict[str, Any]] = []
    local_reference_checks: list[dict[str, Any]] = []
    local_reference_rows: list[dict[str, Any]] = []
    local_reference_profile_rows_scanned = 0
    local_reference_profile_rows_used = 0

    for seed in SEEDS:
        rows_by_context = loaded[seed]
        source_elapsed = {context: len(rows_by_context[context]) for context in CONTEXT_ORDER}
        for prefix in PREFIXES:
            decisions, summary, pairs = _run_sequence(
                seed,
                rows_by_context,
                grouper_config,
                evidence_mode=f"prefix_{prefix}",
                input_rows_loaded_for_seed=file_row_counts[seed],
                profile_builder=lambda context, n=prefix: rebuild_profile(rows_by_context[context][:n]),
                source_elapsed_steps_by_context=source_elapsed,
            )
            assignment_rows.extend(decisions)
            pairwise_rows.extend(pairs)
            scenario_summaries.append(
                {
                    "seed": seed,
                    "scenario": f"prefix_{prefix}",
                    "plot_category": "prefix",
                    "x_value": prefix,
                    **summary,
                }
            )

        action_filter = frozenset((0, 1))
        censored_profiles: dict[str, np.ndarray] = {}

        def build_censored_profile(context: str) -> tuple[np.ndarray, dict[str, int]]:
            profile, metadata = rebuild_profile(
                rows_by_context[context], action_filter=action_filter
            )
            censored_profiles[context] = profile.copy()
            return profile, metadata

        censored_decisions, censored_summary, censored_pairs = _run_sequence(
            seed,
            rows_by_context,
            grouper_config,
            evidence_mode="offline_action_censoring_0_1",
            input_rows_loaded_for_seed=file_row_counts[seed],
            profile_builder=build_censored_profile,
            source_elapsed_steps_by_context=source_elapsed,
        )
        for row in censored_decisions:
            row["offline_evidence_censoring"] = True
            row["retained_action_set"] = [0, 1]
            row["full_source_elapsed_probe_steps"] = 512
        assignment_rows.extend(censored_decisions)
        pairwise_rows.extend(censored_pairs)
        scenario_summaries.append(
            {
                "seed": seed,
                "scenario": "offline_action_censoring_0_1",
                "plot_category": "action_censored",
                "x_value": 512,
                "offline_evidence_censoring": True,
                "retained_action_set": [0, 1],
                "source_elapsed_probe_steps_per_context": 512,
                **censored_summary,
            }
        )

        full_profiles: dict[str, np.ndarray] = {}
        full_profile_rows_used: dict[str, int] = {}
        for context in CONTEXT_ORDER:
            full_profile, full_meta = rebuild_profile(rows_by_context[context])
            full_profiles[context] = full_profile
            full_profile_rows_used[context] = int(full_meta["rows_used_for_profile"])
            local_reference_profile_rows_scanned += int(full_meta["rows_scanned_for_profile"])
            local_reference_profile_rows_used += int(full_meta["rows_used_for_profile"])

        tie_contexts = ("A0", "A1", "A2")
        tied_checks.append(
            evaluate_tied_prototypes(
                seed,
                full_profiles,
                grouper_config,
                source_rows_used={context: len(rows_by_context[context]) for context in tie_contexts},
            )
        )
        local_summary, per_context_rows = audit_local_context_references(
            seed, full_profiles, censored_profiles, censored_decisions
        )
        local_summary["full_reference_profile_rows_scanned"] = sum(
            int(full_profile_rows_used[context]) for context in CONTEXT_ORDER
        )
        local_summary["full_reference_profile_rows_reused"] = sum(
            int(full_profile_rows_used[context]) for context in CONTEXT_ORDER
        )
        local_summary["censored_query_profiles_reused_from_assignment"] = sum(
            int(row["rows_used_for_profile"]) for row in censored_decisions
        )
        local_reference_checks.append(local_summary)
        local_reference_rows.extend(per_context_rows)

    prefix_rows_scanned = sum(
        int(row["profile_source_rows_scanned_across_six_contexts"])
        for row in scenario_summaries
        if row["plot_category"] == "prefix"
    )
    prefix_rows_used = sum(
        int(row["profile_rows_reprocessed_across_six_contexts"])
        for row in scenario_summaries
        if row["plot_category"] == "prefix"
    )
    censored_rows_scanned = sum(
        int(row["profile_source_rows_scanned_across_six_contexts"])
        for row in scenario_summaries
        if row["plot_category"] == "action_censored"
    )
    censored_rows_used = sum(
        int(row["profile_rows_reprocessed_across_six_contexts"])
        for row in scenario_summaries
        if row["plot_category"] == "action_censored"
    )
    tie_rows_reused = sum(
        int(row["profile_rows_reused_from_full_profile_audit"]) for row in tied_checks
    )

    aggregate = {
        "diagnostic": "S3 offline shared-write evidence support and ambiguity diagnostic",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "inputs": input_hashes,
        "fixed_grouper_config": EXPECTED_GROUPER_CONFIG,
        "formal_seeds": list(SEEDS),
        "contexts_and_assignment_order": list(CONTEXT_ORDER),
        "prefix_rows_per_context": list(PREFIXES),
        "analysis_constraints": {
            "only_independent_condition_probe_steps_csv": True,
            "family_labels_used_only_for_pairwise_observer_audit_after_assignment": True,
            "grouper_receives_numeric_profile_only": True,
            "local_reference_names_used_only_after_all_assignments": True,
            "local_reference_names_passed_to_grouper": False,
            "local_reference_audit_is_observer_oracle_analysis": True,
            "no_policy_reward_estimation": True,
            "no_claim_of_safe_matching": True,
            "no_new_environment_interactions": 0,
            "offline_data_row_reads_are_not_new_interactions": True,
        },
        "offline_data_accounting": {
            "saved_probe_rows_per_seed": file_row_counts,
            "saved_probe_rows_total_read": int(sum(file_row_counts.values())),
            "historical_saved_probe_steps_total": int(sum(file_row_counts.values())),
            "prefix_profile_rows_scanned_offline": prefix_rows_scanned,
            "prefix_profile_rows_used_offline": prefix_rows_used,
            "action_censoring_rows_scanned_offline": censored_rows_scanned,
            "action_censoring_rows_used_in_profiles_offline": censored_rows_used,
            "local_reference_full_profile_rows_scanned_offline": local_reference_profile_rows_scanned,
            "local_reference_full_profile_rows_used_offline": local_reference_profile_rows_used,
            "local_reference_censored_query_profiles_reused_from_assignment": censored_rows_used,
            "tie_test_profile_rows_reused_from_local_reference_audit": tie_rows_reused,
            "tie_test_additional_profile_rows_reprocessed_offline": 0,
            "total_profile_source_rows_scanned_offline": prefix_rows_scanned + censored_rows_scanned + local_reference_profile_rows_scanned,
            "total_profile_rows_used_offline": prefix_rows_used + censored_rows_used + local_reference_profile_rows_used,
            "new_environment_interactions": 0,
            "note": "Profile rows are replayed from saved CSV observations; reading or reprocessing them does not step an environment.",
        },
        "prefix_and_censoring_results": scenario_summaries,
        "tied_prototype_pressure_tests": tied_checks,
        "action_censored_local_context_reference_audit": local_reference_checks,
        "interpretation_boundary": "This reports available transition support, unresolved decisions, and family agreement in these saved probes. The action-censored family grouping does not identify the individual local context when multiple full-profile references agree on observed actions and differ only on untried actions. It is not a policy-benefit estimate, a guarantee of safe sharing, or evidence of general family discovery.",
    }

    (output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(
        output_root / "assignments.csv",
        assignment_rows,
        [
            "seed",
            "evidence_mode",
            "context",
            "family_audit_only",
            "assigned_group",
            "reason",
            "scores_json",
            "rows_used_for_profile",
            "rows_scanned_for_profile",
            "movement_transitions_counted",
            "supported_cells",
            "unsupported_cells",
            "supported_cell_coverage",
            "supported_cells_by_x_band",
            "source_elapsed_probe_steps",
            "full_source_elapsed_probe_steps",
            "offline_evidence_censoring",
            "retained_action_set",
            "assignment_coverage_so_far",
            "unresolved_contexts_so_far",
            "numeric_state_bytes",
        ],
    )
    _write_csv(
        output_root / "pairwise_audit.csv",
        pairwise_rows,
        [
            "seed",
            "evidence_mode",
            "context_left",
            "family_left_audit_only",
            "group_left",
            "context_right",
            "family_right_audit_only",
            "group_right",
            "pair_status",
            "family_pair_correct",
        ],
    )
    (output_root / "tied_prototype_checks.json").write_text(
        json.dumps(tied_checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "local_reference_audit.json").write_text(
        json.dumps(
            {
                "input_hashes": input_hashes,
                "fixed_grouper_config": EXPECTED_GROUPER_CONFIG,
                "analysis": local_reference_checks,
                "new_environment_interactions": 0,
                "not_used_for_group_decisions": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_csv(
        output_root / "local_reference_audit.csv",
        local_reference_rows,
        [
            "seed",
            "query_context_audit_label_only",
            "observer_family_audit_only",
            "action_censoring_mode",
            "censored_assigned_group",
            "censored_assignment_reason",
            "censored_supported_cells",
            "unprobed_actions_with_full_profile_support",
            "unprobed_full_profile_cells",
            "no_conflict_full_reference_names",
            "no_conflict_full_reference_count",
            "fully_covered_no_conflict_reference_names",
            "fully_covered_no_conflict_reference_count",
            "same_family_fully_covered_no_conflict_reference_names_audit_only",
            "other_family_fully_covered_no_conflict_reference_names_audit_only",
            "local_context_identifiable_from_censored_profile",
            "local_context_ambiguity_present",
            "candidate_references_with_differences_hidden_on_unprobed_cells",
            "reference_details",
        ],
    )
    _build_plot(scenario_summaries, output_root / "support_coverage_error.png")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="S1b formal output directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new diagnostic output directory")
    args = parser.parse_args()
    report = run_diagnostic(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "saved_rows_read": report["offline_data_accounting"]["saved_probe_rows_total_read"],
                "new_environment_interactions": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

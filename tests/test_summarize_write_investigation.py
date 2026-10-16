"""Synthetic audits for write-investigation report accounting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_write_investigation import build_report


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _record(condition: str, reward: float, final_return: float) -> dict:
    return {
        "condition": condition,
        "seed": 7,
        "reward_per_step": reward,
        "plus_goal_rate": 0.25,
        "training_reward": reward * 10,
        "counts": {
            "environment_steps": 10,
            "accepted_count": 2,
            "rejected_count": 1,
            "group_retries": 4,
            "group_abstentions": 1,
        },
        "costs": {
            "training_steps": 10,
            "probe_steps": 2,
            "evaluation_steps": 5,
            "total_interactions": 17,
            "cpu_s": 3.0,
            "wall_s": 5.0,
        },
        "slot_groups": [0, -1],
        "evaluations": {
            "initial": {},
            "final": {
                "A0+": {"mean_return": final_return, "mean_goal_rate": 0.25},
                # Success on a negative-reward task must not count as a
                # positive-goal success.
                "A0-": {"mean_return": 100.0, "mean_goal_rate": 1.0},
            },
        },
    }


def test_synthetic_top_level_audit_deduplicates_and_labels_lower_bounds(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    output = tmp_path / "report"

    formal_dir = root / "write_decision_formal"
    records = [
        _record("selected", 0.3, 2.0),
        _record("independent_fast", 0.2, 1.0),
        _record("reward_shared", 0.1, 0.0),
    ]
    cohort_cost = {
        "training_steps": 30,
        "probe_steps": 6,
        "evaluation_steps": 15,
        "total_interactions": 51,
    }
    _dump(
        formal_dir / "summary.json",
        {
            "phase": "formal",
            "conditions": ["selected", "independent_fast", "reward_shared"],
            "seeds": [7],
            "records": records,
            "cohort_cost": cohort_cost,
        },
    )
    # A nested summary is deliberately huge; it must not be counted.
    _dump(
        formal_dir / "seed_7" / "selected" / "nested" / "summary.json",
        {"records": [_record("nested_not_a_cohort", 99.0, 99.0)], "cohort_cost": {"total_interactions": 99_999}},
    )
    # This stale marker is retained but deduplicated against the completed
    # summary in the same top-level cohort root.
    _dump(
        formal_dir / "FAILED_COHORT.json",
        {"status": "stale marker", "total_interactions": 51, "reason": "already summarized"},
    )

    causal_dir = root / "write_decision_causal_control"
    _dump(
        causal_dir / "summary.json",
        {
            "conditions": ["trial_rollback"],
            "seeds": [7],
            "records": [_record("trial_rollback", 0.15, 1.5)],
            "cohort_cost": {
                "training_steps": 10,
                "probe_steps": 0,
                "evaluation_steps": 0,
                "total_interactions": 10,
            },
        },
    )
    _dump(root / "write_decision_selection" / "selection.json", {"selected_conditions": ["selected"]})
    _dump(
        root / "write_decision_foundations" / "summary.json",
        {
            "status": "completed",
            "bc_updates_total": 4,
            "evaluation_steps_total": 6,
            "cpu_s_total": 2.0,
            "wall_s_total": 2.5,
        },
    )
    _dump(
        root / "write_decision_counterfactual" / "summary.json",
        {"evaluation_steps": 7, "cpu_s": 1.0, "wall_s": 1.2, "results": []},
    )
    _dump(
        root / "write_stream_interrupted" / "FAILED_COHORT.json",
        {
            "status": "stopped",
            "reason": "short synthetic failure",
            "raw_interactions": {"training": 6, "probe": 2},
            "total_interactions_recorded": 8,
            "recorded_count_exact": False,
            "count_is_lower_bound": True,
        },
    )
    (root / "write_stream_pending" / "seed_99").mkdir(parents=True)

    run_dir = formal_dir / "seed_7" / "selected"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "step.pt").write_bytes(b"1234567890")
    (run_dir / "checkpoints" / "step.actor.bin").write_bytes(b"actor")
    _dump(
        run_dir / "capacity.json",
        [
            {
                "step": 1,
                "actor_bytes": 5,
                "checkpoint_bytes": 10,
                "foundation_bytes": 3,
                "full_persistent_bytes": 13,
            }
        ],
    )
    (run_dir / "steps.csv").write_text("step,reward\n1,0.3\n", encoding="utf-8")

    analysis = build_report(root, output)

    ledger = analysis["interaction_ledger"]
    # Top-level summary ledger is 51, causal control is 10, foundation eval
    # is 6, counterfactual eval is 7. Nested and stale-marker values are not
    # added; the interrupted marker remains a separate lower bound.
    assert ledger["summary_environment_interactions"] == 74
    assert ledger["summary_training_steps"] == 40
    assert ledger["summary_probe_steps"] == 6
    assert ledger["summary_evaluation_steps"] == 28
    assert ledger["cohort_declared_total_interactions"] == 61
    assert ledger["cohort_component_interactions"] == 61
    assert ledger["cohort_declared_total_matches_components"] is True
    assert ledger["failed_exact_interactions"] == 0
    assert ledger["failed_lower_bound_interactions"] == 8
    assert ledger["known_exact_including_failed"] == 74
    assert ledger["known_at_least_including_lower_bounds"] == 82
    assert ledger["foundations_bc_updates_offline"] == 4
    assert ledger["summary_roots_count"] == 4

    formal = analysis["cohorts"]["write_decision_formal"]
    assert formal["record_count"] == 3
    selected = formal["by_condition"]["selected"]
    assert selected["n_seeds"] == 1
    selected_row = formal["records"][0]
    assert selected_row["final_macro_signed_return"] == 51.0
    assert selected_row["final_positive_goal_rate"] == 0.25
    assert selected_row["group_unassigned_fraction"] == 0.5
    assert selected_row["physical_checkpoint_plus_w0_peak_bytes"] == 13
    assert selected_row["actual_actor_payload_peak_bytes"] == 5
    assert "steps.csv" in selected_row["source_files"][0] or any(
        path.endswith("steps.csv") for path in selected_row["source_files"]
    )

    deduplicated_marker = next(
        row for row in analysis["failed_markers"] if row["name"] == "write_decision_formal"
    )
    assert deduplicated_marker["deduplicated_by_top_level_summary"] is True
    interrupted = next(
        row for row in analysis["failed_markers"] if row["name"] == "write_stream_interrupted"
    )
    assert interrupted["count_classification"] == "lower_bound"
    assert interrupted["interaction_count"] == 8
    assert any(row["name"] == "write_stream_pending" for row in analysis["pending"])

    comparisons = analysis["pairing"]["comparisons"]
    selected_vs_reward = next(
        row
        for row in comparisons
        if row["comparison_family"] == "step1_formal_selected_vs_baseline"
        and row["right"] == "reward_shared"
    )
    assert selected_vs_reward["matched_seeds"] == [7]
    assert abs(
        selected_vs_reward["delta_stats_by_seed"]["online_reward_per_step"]["mean"] - 0.2
    ) < 1e-12
    assert selected_vs_reward["post_selection"] is True
    assert selected_vs_reward["selection_eligible"] is False
    assert analysis["pairing"]["recycle_cumulative"]["status"] == "not_available"

    for filename in ("analysis.json", "report.md", "overview.png", "overview.pdf"):
        assert (output / filename).is_file()
        assert (output / filename).stat().st_size > 0
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "不作显著性声称" in report
    assert "≥ 8" in report


def test_stream_dict_eval_uses_positive_tasks_and_pairs_with_lru_baselines(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    output = tmp_path / "report"
    conditions = ["trial_gate_relaxed", "independent7_lru", "reward_shared"]
    records = []
    reward_values = {"trial_gate_relaxed": 0.3, "independent7_lru": 0.2, "reward_shared": 0.1}
    for condition in conditions:
        record = _record(condition, reward_values[condition], 0.0)
        record["evaluations"] = {
            "A0+": {"mean_signed_return": 3.0, "positive_success_rate": 0.6},
            "A0-": {"mean_signed_return": -4.0, "positive_success_rate": 1.0},
            "B0+": {"mean_signed_return": 1.0, "positive_success_rate": 0.4},
            "B0-": {"mean_signed_return": -2.0, "positive_success_rate": 1.0},
        }
        records.append(record)
    _dump(
        root / "write_stream_known_formal" / "summary.json",
        {
            "phase": "formal",
            "conditions": conditions,
            "seeds": [7],
            "records": records,
            "cohort_cost": {
                "training_steps": 30,
                "probe_steps": 6,
                "evaluation_steps": 15,
                "total_interactions": 51,
            },
        },
    )

    analysis = build_report(root, output)
    stream = analysis["cohorts"]["write_stream_known_formal"]
    candidate = stream["records"][0]
    assert candidate["final_stage"] == "stream_final"
    assert candidate["final_macro_signed_return"] == -0.5
    assert candidate["final_positive_goal_rate"] == 0.5
    assert len(analysis["pairing"]["comparisons"]) == 2
    assert all(pair["comparison_family"] == "stream_within_cohort" for pair in analysis["pairing"]["comparisons"])
    vs_lru = next(pair for pair in analysis["pairing"]["comparisons"] if "independent7_lru" in pair["right"])
    assert vs_lru["matched_seeds"] == [7]
    assert abs(vs_lru["delta_stats_by_seed"]["online_reward_per_step"]["mean"] - 0.1) < 1e-12


def test_partial_top_level_records_are_reported_as_pending(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    output = tmp_path / "report"
    _dump(
        root / "write_stream_partial" / "summary.json",
        {
            "conditions": ["trial_gate_relaxed", "reward_shared"],
            "seeds": [7, 8],
            "records": [_record("trial_gate_relaxed", 0.1, 0.0)],
            "cohort_cost": {
                "training_steps": 10,
                "probe_steps": 0,
                "evaluation_steps": 0,
                "total_interactions": 10,
            },
        },
    )

    analysis = build_report(root, output)
    partial = analysis["cohorts"]["write_stream_partial"]
    assert partial["status"] == "pending_incomplete_declared_grid"
    assert partial["expected_record_count"] == 4
    assert partial["record_count"] == 1
    pending = next(item for item in analysis["pending"] if item["name"] == "write_stream_partial")
    assert pending["observed_records"] == 1
    assert pending["expected_records"] == 4


@pytest.mark.parametrize("shared_name,baseline_name", [
    ("write_stream_known", "write_stream_known_budget_baseline"),
    ("write_detour_w0_17", "write_detour_budget_w0_17"),
    ("write_detour_w0_29", "write_detour_budget_w0_29"),
])
def test_stream_probe_reward_and_full_use_cross_cohort_pair(tmp_path: Path, shared_name, baseline_name) -> None:
    root, output = tmp_path / "outputs", tmp_path / "report"
    shared = _record("trial_gate_relaxed", 0.3, 0.0)
    shared["per_task_observed_stream"] = {
        "A0+": {"reward": 3.0, "steps": 10},
        "A0-": {"reward": -1.8, "steps": 2},
    }
    shared["group_saturation"] = {"unassigned_step_fraction": 0.75}
    baseline = _record("independent7_lru", 0.2, 0.0)
    baseline["costs"].update(training_steps=12, probe_steps=0)
    for name, record in ((shared_name, shared), (baseline_name, baseline)):
        _dump(root / name / "summary.json", {
            "conditions": [record["condition"]], "seeds": [7], "records": [record],
        })
    analysis = build_report(root, output)
    row = analysis["cohorts"][shared_name]["records"][0]
    assert abs(row["online_reward_per_stream_step"] - 0.1) < 1e-12
    assert row["online_reward_per_step"] == 0.3
    assert row["group_unassigned_training_fraction"] == 0.75
    pair, = analysis["pairing"]["comparisons"]
    assert pair["online_interactions_equal_each_seed"] is True
    assert pair["primary_metric"] == "online_reward_per_stream_step"
    # Probe losses reverse the ranking; training-only R cannot stand in for
    # a comparison at equal actual interaction budget.
    assert pair["delta_stats_by_seed"]["online_reward_per_step"]["mean"] > 0
    assert abs(pair["delta_stats_by_seed"]["online_reward_per_stream_step"]["mean"] + 0.1) < 1e-12


def test_missing_probe_rewards_are_not_invented_and_raw_reward_is_normalized(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    record = _record("trial_gate_relaxed", 0.3, 0.0)
    record.pop("reward_per_step")
    _dump(root / "write_stream_known" / "summary.json", {
        "conditions": [record["condition"]], "seeds": [7], "records": [record],
    })
    analysis = build_report(root, tmp_path / "report")
    row = analysis["cohorts"]["write_stream_known"]["records"][0]
    assert row["online_reward_per_step"] == 0.3
    assert row["online_reward_per_stream_step"] is None


def test_parent_pairing_uses_same_stage_and_omits_unmatched_seeds(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    for variant in ("cumulative", "recycle"):
        for stage, seed in (("validation", 7), ("formal", 8)):
            record = _record("reward_shared", 0.3, 0.0)
            record["seed"] = seed
            _dump(root / f"write_parent_{variant}_{stage}" / "summary.json", {
                "conditions": ["reward_shared"], "seeds": [seed], "records": [record],
            })
    _dump(root / "write_parent_selection" / "SKIPPED.json", {"formal_and_unknown_followup": "skipped"})
    analysis = build_report(root, tmp_path / "report")
    pairs = analysis["pairing"]["recycle_cumulative"]["pairs"]
    assert len(pairs) == 2
    assert all(pair["n_pairs"] == 1 for pair in pairs)
    assert all(pair["left"].split(":")[0].split("_")[-1] ==
               pair["right"].split(":")[0].split("_")[-1] for pair in pairs)
    assert analysis["parent_followup_admission"]["SKIPPED.json"]["decision"]["formal_and_unknown_followup"] == "skipped"

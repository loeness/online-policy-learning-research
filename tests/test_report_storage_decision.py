import json
from pathlib import Path

import pytest

from scripts.report_storage_decision import analyze_summary, generate_report


SEEDS = (20101, 20107)
PAIR_KINDS = ("dynamics", "reward_conflict")
REGIMES = ("old_major", "new_major")
ACTIONS = ("keep", "replace", "mean", "none")


def _action_map(history):
    recent = [0, 0, 0, 0, 0, 0, 1, 1] if history == "old_major" else [0, 1, 1, 1, 0, 1, 1, 1]
    frequency = "replace" if recent.count(1) > recent.count(0) else "keep"
    latest_old = max(index for index, group in enumerate(recent) if group == 0)
    latest_new = max(index for index, group in enumerate(recent) if group == 1)
    recency = "replace" if latest_new > latest_old else "keep"
    return {
        "keep": "keep", "replace": "replace", "mean": "mean", "none": "none",
        "frequency": frequency, "recency": recency,
        "surrogate_gain": "replace", "surrogate_lower": "keep",
    }, recent


def _write_cohort(root: Path, *, omit_branch=None, add_failure=False):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "seeds": list(SEEDS), "pair_kinds": list(PAIR_KINDS),
        "history_regimes": list(REGIMES), "future_regimes": list(REGIMES),
        "actions": list(ACTIONS), "bootstrap_steps": 1, "bout_steps": 1,
        "package_budget_bytes": 262144,
        "report": {"bootstrap_resamples": 128, "bootstrap_seed": 20261018,
                   "practical_reward_per_step": 0.002, "direction_consistency_minimum": 6},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({
        "analysis_role": "test fixture",
        "errors": ([{"case_id": "20101_dynamics_old_major", "message": "simulated branch failure"}] if add_failure else []),
        "costs": {"environment_steps": 1},
    }), encoding="utf-8")

    for seed in SEEDS:
        for pair_index, pair_kind in enumerate(PAIR_KINDS):
            for history_index, history in enumerate(REGIMES):
                case_id = f"{seed}_{pair_kind}_{history}"
                case_dir = root / "cases" / case_id
                decision_dir = case_dir
                action_map, recent = _action_map(history)
                decision = {
                    "case_id": case_id,
                    "seed": seed,
                    "pair_kind": pair_kind,
                    "history_regime": history,
                    "selectors": action_map,
                    "selector_diagnostics": {
                        "recent_group_ids": recent,
                        "group_frequency": {"0": recent.count(0) / 8, "1": recent.count(1) / 8},
                        "past_only_selectors": action_map,
                        "surrogate": {
                            "old_zero_vs_old_parent": {"gain": 0.0},
                            "new_parent_vs_zero": {"gain": 0.0},
                            "frequency_weighted_gain": 1.0,
                            "frequency_weighted_lower": -1.0,
                        },
                    },
                    "evidence_rows_per_group": {"old": 20, "new": 20},
                    "prefix_state_hash": "same-prefix-hash",
                    "restored_state_hashes": {"keep__old_major": "same-prefix-hash"},
                    # Branch schedule metadata is written after freezing the
                    # prefix decision and must not be mistaken for selector input.
                    "branches": [{"future_regime": "old_major"}],
                }
                decision_dir.mkdir(parents=True, exist_ok=True)
                (decision_dir / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
                prefix_dir = decision_dir / "prefix"
                prefix_dir.mkdir()
                (prefix_dir / "summary.json").write_text(json.dumps({
                    "environment_steps": 10, "costs": {"cpu_s": 0.5, "wall_s": 0.6},
                    "capacity": {"foundation_bytes": 1000, "checkpoint_bytes": 10000,
                                 "actor_bytes": 89, "full_persistent_bytes": 11089},
                }), encoding="utf-8")
                for action in ACTIONS:
                    for future in REGIMES:
                        branch = f"{action}__{future}"
                        if omit_branch == (case_id, branch):
                            continue
                        branch_dir = decision_dir / "branches" / branch
                        branch_dir.mkdir(parents=True)
                        # The replace-minus-keep recurrence effect varies by
                        # history. The two history values must be averaged
                        # before the seed enters bootstrap resampling.
                        base = seed * 0.01 + pair_index * 0.2 + history_index * 0.4
                        if future == "new_major":
                            base += 5.0
                        if history == "old_major" and future == "new_major":
                            base += 2.0
                        if action == "replace":
                            delta_by_history = {
                                "old_major": {"old_major": 1.0, "new_major": 3.0},
                                "new_major": {"old_major": -1.0, "new_major": 5.0},
                            }
                            base += delta_by_history[history][future]
                        elif action == "mean":
                            base += 0.25
                        elif action == "none":
                            base -= 0.25
                        total_steps = 12
                        summary = {
                            "reward_total": base,
                            "reward_per_step": base / total_steps,
                            "environment_steps": total_steps,
                            "counts": {"environment_steps": 12, "slot_allocations": 2,
                                       "slot_evictions": 1, "rollouts_trained": 1,
                                       "minibatch_updates": 4},
                            "rollouts_trained": 1,
                            "minibatch_updates": 4,
                            "costs": {"cpu_s": 0.25, "wall_s": 0.3},
                            "capacity": {"foundation_bytes": 1000, "checkpoint_bytes": 10000,
                                         "actor_bytes": 89, "full_persistent_bytes": 11089,
                                         "package_budget_bytes": 262144},
                            "capacity_after_action": {"foundation_bytes": 1000, "checkpoint_bytes": 9000,
                                                      "actor_bytes": 89, "full_persistent_bytes": 10089},
                            "per_group": {"0": {"reward_total": 2.0, "steps": 6, "reward_per_step": 1/3},
                                          "1": {"reward_total": 3.0, "steps": 6, "reward_per_step": 1/2}},
                            "per_bout": [{"bout": 0, "group": 0, "steps": 1, "allocated": True,
                                          "evicted_task": None, "return_first_256": 1.0,
                                          "return_first_512": 1.0, "return_full": 1.0}],
                        }
                        (branch_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_interaction_merges_history_within_seed_before_bootstrap(tmp_path):
    _write_cohort(tmp_path / "cohort")
    analysis = analyze_summary(tmp_path / "cohort")

    result = analysis["interaction_I"]["dynamics"]["reward_total"]
    assert result["n_seeds"] == 2
    assert [row["seed"] for row in result["per_seed"]] == list(SEEDS)
    # I is 2 in old_major history and 6 in new_major history, averaged to 4
    # per deployment seed before resampling.
    assert [row["value"] for row in result["per_seed"]] == pytest.approx([4.0, 4.0])
    assert result["mean"] == pytest.approx(4.0)
    assert result["percentile_ci95"] == pytest.approx([4.0, 4.0])
    assert analysis["preference_crossover"]["dynamics"]["reward_total"]["per_case_crossover_count"] == 2
    assert analysis["preference_crossover"]["dynamics"]["reward_total"]["per_case_count_with_complete_D"] == 4
    assert analysis["preference_crossover"]["dynamics"]["reward_total"]["per_seed_D_old_future_summary"]["mean"] == pytest.approx(0.0)
    assert analysis["preference_crossover"]["dynamics"]["reward_total"]["per_seed_D_new_future_summary"]["mean"] == pytest.approx(4.0)


def test_primary_and_reversal_stress_use_distinct_future_branches(tmp_path):
    _write_cohort(tmp_path / "cohort")
    analysis = analyze_summary(tmp_path / "cohort")
    assert analysis["static_checks"]["selector_prefix_only"]["all_passed"] is True
    primary = analysis["continuation_conditions"]["primary"]["dynamics"]["methods"]["keep"]["reward_total"]
    stress = analysis["continuation_conditions"]["stress"]["dynamics"]["methods"]["keep"]["reward_total"]

    assert primary["mean"] != stress["mean"]
    assert primary["n_seeds"] == stress["n_seeds"] == len(SEEDS)


def test_realized_selector_outcome_looks_up_its_selected_full_branch(tmp_path):
    cohort = tmp_path / "cohort"
    _write_cohort(cohort)
    analysis = analyze_summary(cohort)
    case = cohort / "cases" / f"{SEEDS[0]}_dynamics_old_major"
    selectors = json.loads((case / "decision.json").read_text(encoding="utf-8"))["selectors"]
    expected = json.loads((case / "branches" / f"{selectors['surrogate_gain']}__old_major" / "summary.json").read_text(encoding="utf-8"))["reward_total"]
    realized = analysis["continuation_conditions"]["primary"]["dynamics"]["methods"]["surrogate_gain"]["reward_total"]
    seed_value = next(row["value"] for row in realized["per_seed"] if row["seed"] == SEEDS[0])
    # Per-seed method result is the mean of both history-specific selector
    # branch outcomes; the old-major half is the branch selected above.
    new_case = cohort / "cases" / f"{SEEDS[0]}_dynamics_new_major"
    new_selectors = json.loads((new_case / "decision.json").read_text(encoding="utf-8"))["selectors"]
    new_expected = json.loads((new_case / "branches" / f"{new_selectors['surrogate_gain']}__new_major" / "summary.json").read_text(encoding="utf-8"))["reward_total"]
    assert seed_value == pytest.approx((expected + new_expected) / 2)
    contrast = analysis["continuation_conditions"]["primary"]["dynamics"]["selector_paired_contrasts"]["surrogate_gain_minus_frequency"]["reward_total"]
    assert contrast["n_seeds"] == len(SEEDS)


def test_runner_actor_bytes_populate_physical_actor_sidecar_range(tmp_path):
    cohort = tmp_path / "cohort"
    _write_cohort(cohort)

    analysis = analyze_summary(cohort)
    sidecar = analysis["costs"]["observed_leaf_summaries"]["physical_bytes_ranges"]["sidecar_bytes"]

    # The runner records the emitted actor.bin size as capacity.actor_bytes.
    assert sidecar == {"min": 89, "max": 89, "n": 64}


def test_missing_branch_and_runner_failure_are_visible_and_reduce_paired_n(tmp_path):
    cohort = tmp_path / "cohort"
    missing_case = f"{SEEDS[0]}_dynamics_old_major"
    _write_cohort(cohort, omit_branch=(missing_case, "replace__new_major"), add_failure=True)
    analysis = analyze_summary(cohort)

    audit = analysis["static_checks"]["branch_completeness"]
    assert audit["all_expected_cases_complete"] is False
    assert audit["observed_branch_summary_count"] == audit["expected_branch_count"] - 1
    incomplete = next(row for row in audit["incomplete_cases"] if row.get("case_id") == missing_case)
    assert incomplete["missing_branches"] == ["replace__new_major"]
    assert analysis["interaction_I"]["dynamics"]["reward_total"]["n_seeds"] == 1
    failures = analysis["static_checks"]["runner_and_parse_failures"]
    assert any(row["type"] == "runner_recorded_failure" for row in failures)


def test_report_writes_machine_analysis_markdown_and_publication_figures(tmp_path):
    cohort = tmp_path / "cohort"
    _write_cohort(cohort)
    output = tmp_path / "report"

    analysis = generate_report(cohort, output)

    assert analysis["format"] == "storage_decision_analysis_v1"
    assert {path.name for path in output.iterdir()} == {"analysis.json", "report.md", "figure.png", "figure.pdf"}
    assert (output / "figure.png").stat().st_size > 1000
    assert (output / "figure.pdf").stat().st_size > 1000
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "history matches future" in report
    assert "history differs from future" in report
    assert "does not automatically label" in report.lower()

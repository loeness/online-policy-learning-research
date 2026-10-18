import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import report_retention_stream as report  # noqa: E402


METHODS = ("independent", "direct_shared", "mean", "mean_reuse_gate", "retention_gate")


def _recurrence(
    *, evicted: bool, first256: float, recovery512: float, recovery1024: float,
    task: str | None = None, bout: int = 13,
) -> dict:
    return {
        "task": task or ("A0+" if evicted else "B0+"),
        "bout": bout,
        "previously_evicted": evicted,
        "first256_reward_per_step": first256,
        "recovery512_reward_per_step": recovery512,
        "recovery1024_reward_per_step": recovery1024,
        "first_goal_step": None if evicted else 128,
        "first_gradient_step": 256,
    }


def _cohort() -> dict:
    records = []
    for budget in report.BUDGETS:
        for method_index, method in enumerate(METHODS):
            for seed in (1, 2):
                seed_delta = seed - 1
                total_reward = -10 + method_index * 4 + seed
                raw_steps = 1000
                evictions = 0 if budget == "wide" else (4 if method != "mean" else 0)
                if budget == "low" and evictions:
                    if seed == 1:
                        recurrences = [
                            _recurrence(evicted=True, first256=0.1, recovery512=0.2 + method_index, recovery1024=0.4 + method_index),
                            _recurrence(evicted=True, first256=0.5, recovery512=0.4 + method_index, recovery1024=0.6 + method_index, task="A0-", bout=14),
                        ]
                    else:
                        recurrences = [
                            _recurrence(evicted=True, first256=0.8, recovery512=0.8 + method_index, recovery1024=1.0 + method_index),
                        ]
                else:
                    recurrences = [
                        _recurrence(evicted=False, first256=2.1, recovery512=2.2, recovery1024=2.4),
                    ]
                records.append({
                    "method": method,
                    "budget": budget,
                    "seed": seed,
                    "status": "completed",
                    "raw_steps": raw_steps,
                    "evaluation_steps": 100,
                    "fit_steps": 700,
                    "total_reward": total_reward,
                    "reward_per_raw_step": total_reward / raw_steps,
                    "slot_evictions": evictions,
                    "rollouts_trained": 16 + seed_delta,
                    "minibatch_updates": 8 + method_index,
                    "write_proposals": method_index + 2,
                    "write_accepts": method_index + 1,
                    "prior_bindings": method_index,
                    "cpu_s": 2.0 + method_index,
                    "wall_s": 3.0 + method_index,
                    "capacity_max": {
                        "actor_bytes": 180 + method_index,
                        "full_physical_bytes": 2000 + method_index,
                        "package_budget_bytes": 5000,
                        "evidence_bytes": 400 + method_index,
                    },
                    "recurrences": recurrences,
                    "audit": {
                        "passed": True,
                        "source_unchanged": True,
                        "persistent_roundtrip": True,
                    },
                })
    return {
        "format": "retention_stream_cohort_v1",
        "status": "completed",
        "analysis_role": "exploratory",
        "config": {"methods": list(METHODS), "budgets": list(report.BUDGETS), "seeds": [1, 2]},
        "records": records,
        "audit": {"passed": True},
        "cohort_cost": {
            "new_raw_steps": 20_000,
            "new_evaluation_steps": 2_000,
            "new_total_interactions": 22_000,
        },
        "cpu_s": 50.0,
        "wall_s": 60.0,
    }


def _write_cohort(root: Path, cohort: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "summary.json"
    path.write_text(json.dumps(cohort or _cohort()), encoding="utf-8")
    return root


def test_report_keeps_declared_five_methods_budgets_and_seed_unit(tmp_path: Path) -> None:
    source = _write_cohort(tmp_path / "cohort")
    output = tmp_path / "report"

    analysis = report.analyze(source, output)

    assert analysis["methods"] == list(METHODS)
    assert analysis["seed_count"] == 2
    assert analysis["branch_count"] == 20
    assert analysis["method_counts"]["low"]["mean_reuse_gate"] == 2
    low_independent = analysis["by_budget_method"]["low"]["independent"]["metrics"]["total_reward"]
    assert low_independent["mean"] == pytest.approx(-8.5)
    assert low_independent["range"] == pytest.approx([-9.0, -8.0])
    paired = analysis["paired_differences"]["low"]["direct_shared"]
    assert paired["mean_difference"]["total_reward"] == pytest.approx(4.0)
    assert paired["difference_range"]["total_reward"] == pytest.approx([4.0, 4.0])
    assert "mean_reuse_gate" in (output / "report.md").read_text(encoding="utf-8")
    assert "cold-start gain" in (output / "report.md").read_text(encoding="utf-8")
    assert json.loads((output / "analysis.json").read_text(encoding="utf-8"))["statistical_unit"] == "seed"
    for filename in ("analysis.json", "report.md", "figure.png", "figure.pdf"):
        assert (output / filename).is_file()


def test_low_only_pilot_uses_declared_budget_keys_everywhere(tmp_path: Path) -> None:
    cohort = _cohort()
    cohort["records"] = [row for row in cohort["records"] if row["budget"] == "low"]
    cohort["config"]["budgets"] = {"low": {"local_slots": 3}}
    raw_steps = sum(row["raw_steps"] for row in cohort["records"])
    evaluation_steps = sum(row["evaluation_steps"] for row in cohort["records"])
    cohort["cohort_cost"] = {
        "new_raw_steps": raw_steps,
        "new_evaluation_steps": evaluation_steps,
        "new_total_interactions": raw_steps + evaluation_steps,
    }
    output = tmp_path / "pilot-report"

    analysis = report.analyze(_write_cohort(tmp_path / "pilot-cohort", cohort), output)

    assert analysis["budgets"] == ["low"]
    assert analysis["branch_count"] == len(METHODS) * 2
    assert set(analysis["by_budget_method"]) == {"low"}
    assert set(analysis["paired_differences"]) == {"low"}
    assert set(analysis["method_counts"]) == {"low"}
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "Declared budgets are reported separately: low." in markdown
    assert "### Wide budget" not in markdown
    assert (output / "figure.png").is_file()
    assert (output / "figure.pdf").is_file()


def test_512_step_pilot_window_is_not_reported_as_1024_step_recovery(tmp_path: Path) -> None:
    cohort = _cohort()
    cohort["records"] = [row for row in cohort["records"] if row["budget"] == "low"]
    cohort["config"]["budgets"] = {"low": {"local_slots": 3}}
    for branch in cohort["records"]:
        for recurrence in branch["recurrences"]:
            recurrence["observed_steps"] = 512
    raw_steps = sum(row["raw_steps"] for row in cohort["records"])
    evaluation_steps = sum(row["evaluation_steps"] for row in cohort["records"])
    cohort["cohort_cost"] = {
        "new_raw_steps": raw_steps,
        "new_evaluation_steps": evaluation_steps,
        "new_total_interactions": raw_steps + evaluation_steps,
    }

    analysis = report.analyze(_write_cohort(tmp_path / "pilot-cohort", cohort), tmp_path / "pilot-report")

    independent_recurrences = analysis["by_budget_method"]["low"]["independent"]["recurrences"]["evicted_recovery"]
    assert independent_recurrences["metrics"]["recovery512_reward_per_step"]["seed_count"] == 2
    assert independent_recurrences["metrics"]["recovery1024_reward_per_step"]["seed_count"] == 0
    assert independent_recurrences["metrics"]["recovery1024_reward_per_step"]["mean"] is None
    observation = next(
        row for row in analysis["per_seed"]
        if row["budget"] == "low" and row["method"] == "independent" and row["seed"] == 1
    )["recurrence_observations"][0]
    assert observation["observed_steps"] == 512
    assert observation["recovery1024_reward_per_step"] is None


@pytest.mark.parametrize("budgets", [{"pilot": {}}, ["low", "pilot"], [], ["low", "low"]])
def test_unsupported_or_duplicate_declared_budgets_are_rejected(budgets: object, tmp_path: Path) -> None:
    cohort = _cohort()
    cohort["config"]["budgets"] = budgets

    with pytest.raises(ValueError, match="config.budgets"):
        report.analyze(_write_cohort(tmp_path / "cohort", cohort), tmp_path / "report")

    assert not (tmp_path / "report").exists()


def test_recurrence_is_averaged_within_seed_and_zero_eviction_is_not_recovery(tmp_path: Path) -> None:
    analysis = report.analyze(_write_cohort(tmp_path / "cohort"), tmp_path / "report")
    low_independent = analysis["by_budget_method"]["low"]["independent"]["recurrences"]
    eviction = low_independent["evicted_recovery"]
    # Seed 1 has two task rows averaging to 0.3; seed 2 has one row at 0.8.
    # The across-seed mean is therefore 0.55, not a task-weighted 0.4667.
    assert eviction["metrics"]["first256_reward_per_step"]["mean"] == pytest.approx(0.55)
    assert eviction["metrics"]["first256_reward_per_step"]["seed_count"] == 2

    wide = analysis["by_budget_method"]["wide"]["direct_shared"]["recurrences"]
    assert wide["evicted_recovery"]["seed_count"] == 0
    assert wide["evicted_recovery"]["task_count_total"] == 0
    assert wide["no_eviction_revisit"]["seed_count"] == 2
    assert wide["no_eviction_revisit"]["task_count_total"] == 2
    paired_wide = analysis["paired_differences"]["wide"]["direct_shared"]
    assert paired_wide["matched_eviction_recovery"]["metrics"]["recovery512_reward_per_step"]["seed_count"] == 0


def test_paired_eviction_recovery_matches_same_task_bouts_and_reports_all_recurrences(tmp_path: Path) -> None:
    cohort = _cohort()
    for branch in cohort["records"]:
        if branch["budget"] == "low" and branch["method"] == "direct_shared":
            for recurrence in branch["recurrences"]:
                recurrence["previously_evicted"] = False
    source = _write_cohort(tmp_path / "cohort", cohort)
    output = tmp_path / "report"

    analysis = report.analyze(source, output)

    paired = analysis["paired_differences"]["low"]["direct_shared"]
    matched_eviction = paired["matched_eviction_recovery"]["metrics"]["recovery512_reward_per_step"]
    assert matched_eviction["seed_count"] == 0
    assert matched_eviction["matched_task_bout_count_total"] == 0
    assert matched_eviction["mean_difference"] is None

    matched_all = paired["matched_all_recurrences"]["metrics"]["recovery512_reward_per_step"]
    assert matched_all["seed_count"] == 2
    assert matched_all["matched_task_bout_count_total"] == 3
    assert matched_all["mean_difference"] == pytest.approx(1.0)
    assert [row["matched_task_bout_count"] for row in matched_all["per_seed"]] == [2, 1]
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "Only the same seed and task/bout observed as previously evicted in both arms is included." in markdown
    assert "regardless of eviction status" in markdown


@pytest.mark.parametrize("defect", ["missing_branch", "incomplete_branch", "failed_audit"])
def test_incomplete_cohort_is_rejected_before_report_output(defect: str, tmp_path: Path) -> None:
    cohort = _cohort()
    if defect == "missing_branch":
        cohort["records"].pop()
    elif defect == "incomplete_branch":
        cohort["records"][0]["status"] = "running"
    else:
        cohort["records"][0]["audit"]["persistent_roundtrip"] = False
    source = _write_cohort(tmp_path / "cohort", cohort)
    output = tmp_path / "report"

    with pytest.raises(ValueError):
        report.analyze(source, output)
    assert not output.exists()

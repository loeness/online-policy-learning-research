import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report_score_calibration as report  # noqa: E402


NOISE = (-0.007, -0.005, -0.003, -0.001, 0.001, 0.003, 0.005, 0.007)


def _outcome(mean_gain: float, *, scale: float = 1.0) -> dict[str, object]:
    replicates = [mean_gain + scale * value for value in NOISE]
    mean = sum(replicates) / len(replicates)
    sample_sd = (sum((value - mean) ** 2 for value in replicates) / (len(replicates) - 1)) ** 0.5
    stderr = sample_sd / (len(replicates) ** 0.5)
    margin = report.T_CRITICAL_DF_7_95 * stderr
    low, high = mean - margin, mean + margin
    sign = "beneficial" if low > 0 else "harmful" if high < 0 else "ambiguous"
    weighted = [0.5 * value for value in replicates]
    return {
        "replicate_gains": replicates,
        "prototype_weighted_replicate_gains": weighted,
        "mean_gain": mean,
        "stderr": stderr,
        "ci_low": low,
        "ci_high": high,
        "sign": sign,
    }


def _candidate(index: int, seed: int, budget: str, *, missing_score: bool = False) -> dict[str, object]:
    score = float(index - 3.5)
    gains = (-0.3, -0.2, -0.001, 0.0, 0.001, 0.2, 0.3, 0.4)
    return {
        "id": f"candidate-{index}",
        "seed": seed,
        "budget": budget,
        "group": index % 2,
        "step": 1024 + 64 * index,
        "accepted": index >= 1 and index != 5,
        "predictors": {
            "original_mean_lower": None if missing_score else score,
            "original_mean_gain": score + 0.1,
            "codec_mean_lower": None if index == 2 else score * 1.2,
            "fresh_lower": -score,
            "negative_mean_stderr": -abs(score),
            "time_step": float(index),
        },
        "contexts": [f"task-{index}"],
        "prototype_weights": {f"task-{index}": 1.0},
        "outcome": _outcome(gains[index]),
    }


def _excluded(seed: int, budget: str) -> dict[str, object]:
    return {
        "id": "excluded-no-source-context",
        "seed": seed,
        "budget": budget,
        "step": 12000,
        "reason": "no_archive_source_context",
        "predictors": _candidate(0, seed, budget)["predictors"],
        "contexts": [],
    }


def _cohort() -> dict[str, object]:
    records = []
    for seed in report.EXPECTED_SEEDS:
        for budget in report.EXPECTED_BUDGETS:
            records.append({
                "seed": seed,
                "budget": budget,
                "status": "completed",
                "learner_steps": 24576,
                "observer_steps": 8 * report.OBSERVER_STEPS_PER_CONTEXT,
                "candidates": [
                    _candidate(index, seed, budget, missing_score=(seed == 19727 and budget == "wide" and index == 0))
                    for index in range(8)
                ],
                "excluded_candidates": [_excluded(seed, budget)],
                "captured_tests": 9,
                "audit": {"passed": True},
            })
    return {
        "format": "score_calibration_cohort_v1",
        "status": "completed",
        "analysis_role": "diagnostic",
        "config": {
            "mode": "main",
            "seeds": list(report.EXPECTED_SEEDS),
            "budgets": list(report.EXPECTED_BUDGETS),
            "bout_steps": 2048,
            "replicates": report.REPLICATES,
        },
        "records": records,
        "audit": {"passed": True},
        "cost": {
            "learner_steps": 147456,
            "observer_steps": 6 * 8 * report.OBSERVER_STEPS_PER_CONTEXT,
            "new_total_interactions": 147456 + 6 * 8 * report.OBSERVER_STEPS_PER_CONTEXT,
        },
        "cpu_s": 12.0,
        "wall_s": 10.0,
    }


def _write_cohort(directory: Path, cohort: dict[str, object] | None = None) -> Path:
    directory.mkdir(parents=True)
    summary = directory / "summary.json"
    summary.write_text(json.dumps(cohort or _cohort(), allow_nan=False), encoding="utf-8")
    return directory


def test_report_separates_seed_level_primary_prototype_and_gate_diagnostics(tmp_path: Path) -> None:
    source = _write_cohort(tmp_path / "cohort")
    output = tmp_path / "report"

    analysis = report.analyze(source, output)

    assert analysis["cohort"]["branch_count"] == 6
    assert analysis["cohort"]["candidate_count"] == 48
    assert analysis["cohort"]["excluded_candidate_count"] == 6
    assert analysis["cohort"]["captured_test_count"] == 54
    cell = next(row for row in analysis["cells"] if row["seed"] == 19703 and row["budget"] == "low")
    assert cell["predictor_rho_vs_mean_gain"]["original_mean_lower"]["rho"] == pytest.approx(1.0)
    assert cell["primary_partial_rho_controlling_group"]["rho"] == pytest.approx(1.0)
    assert cell["primary_rho_vs_prototype_weighted_mean_gain"]["rho"] == pytest.approx(1.0)
    assert cell["predictor_missing_counts"]["codec_mean_lower"] == 1
    assert cell["primary_score_approval_gate"]["n_ambiguous_outcomes"] == 3
    assert cell["primary_score_approval_gate"]["approved_but_mc_harmful_count"] == 0
    assert cell["accepted_gate"]["accepted_but_mc_harmful_count"] == 1
    assert cell["accepted_gate"]["not_accepted_but_mc_beneficial_count"] == 1
    assert analysis["aggregate_by_budget"]["low"]["seed_level_primary_rho"]["expected_count"] == 3
    assert "exploratory heuristic" in analysis["favorable_heuristic"]["criterion"]
    assert "conditional 128-step frozen-policy gain" in analysis["estimand"]
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "does not estimate adaptation, final return" in markdown
    assert "Score not approved but MC beneficial" in markdown
    assert "Gate accepted but MC harmful" in markdown
    assert "Plotting software: Matplotlib" in markdown
    assert "doi:10.1109/MCSE.2007.55" in markdown
    assert "Procedural plotting guidance: Kassis" in markdown
    assert "arXiv:2609.00065v2" not in markdown
    saved = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert saved["format"] == "score_calibration_analysis_v1"
    assert (output / "figure.png").stat().st_size > 1000
    assert (output / "figure.pdf").stat().st_size > 1000


def test_rank_and_group_partial_rank_are_descriptive_and_handle_ties_and_constants() -> None:
    assert report.spearman([1, 2, 2, 4, 5], [2, 4, 4, 8, 10])["rho"] == pytest.approx(1.0)
    assert report.spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])["status"] == "not_estimable_constant_input"
    assert report.spearman([1, 2, 3, 4], [4, 3, 2, 1])["status"] == "not_estimable_too_few_events"

    result = report.partial_spearman_by_group(
        [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], ["a", "a", "b", "b", "c", "c"]
    )
    assert result["rho"] == pytest.approx(1.0)
    degenerate = report.partial_spearman_by_group(
        [1, 1, 1, 1, 1], [1, 2, 3, 4, 5], ["a"] * 5
    )
    assert degenerate["status"] == "not_estimable_degenerate_group_residuals"


def test_missing_score_is_counted_and_excluded_from_score_sign_metrics(tmp_path: Path) -> None:
    cohort = _cohort()
    source = _write_cohort(tmp_path / "cohort", cohort)
    analysis = report.analyze_cohort(json.loads((source / "summary.json").read_text(encoding="utf-8")))
    cell = next(row for row in analysis["cells"] if row["seed"] == 19727 and row["budget"] == "wide")

    assert cell["predictor_missing_counts"]["original_mean_lower"] == 1
    assert cell["predictor_rho_vs_mean_gain"]["original_mean_lower"]["n"] == 7
    assert cell["primary_score_approval_gate"]["n_missing_score_among_resolved"] == 1
    assert cell["accepted_gate"]["n_missing_score_among_resolved"] == 0


def test_pilot_subset_is_reported_without_applying_main_cohort_heuristic(tmp_path: Path) -> None:
    cohort = _cohort()
    records = cohort["records"]
    assert isinstance(records, list)
    branch = next(row for row in records if row["seed"] == 19703 and row["budget"] == "low")
    branch["seed"] = report.PILOT_SEED
    branch["learner_steps"] = 12288
    branch["candidates"] = [_candidate(index, report.PILOT_SEED, "low") for index in range(8)]
    branch["excluded_candidates"] = [_excluded(report.PILOT_SEED, "low")]
    cohort["records"] = [branch]
    cohort["config"] = {
        "mode": "pilot", "seeds": [report.PILOT_SEED], "budgets": ["low"],
        "bout_steps": 1024, "replicates": report.REPLICATES,
    }
    observer_steps = 8 * report.OBSERVER_STEPS_PER_CONTEXT
    cohort["cost"] = {
        "learner_steps": 12288, "observer_steps": observer_steps,
        "new_total_interactions": 12288 + observer_steps,
    }
    source = _write_cohort(tmp_path / "pilot-cohort", cohort)
    output = tmp_path / "pilot-report"
    analysis = report.analyze(source, output)

    assert analysis["cohort"]["mode"] == "pilot"
    assert analysis["cohort"]["branch_count"] == 1
    assert analysis["aggregate_by_budget"]["low"]["seed_level_primary_rho"]["expected_count"] == 1
    assert analysis["favorable_heuristic"]["applicable"] is False
    assert analysis["favorable_heuristic"]["met_for_all_budgets"] is None
    assert "not evaluated on the pilot cohort" in (output / "report.md").read_text(encoding="utf-8")
    assert (output / "figure.png").is_file()
    assert (output / "figure.pdf").is_file()


@pytest.mark.parametrize("defect", ["missing_branch", "duplicate_branch", "incomplete_branch", "bad_audit", "bad_cost", "bad_count", "bad_replicates", "bad_sign"])
def test_invalid_grid_audit_cost_and_outcomes_are_rejected_before_output(defect: str, tmp_path: Path) -> None:
    cohort = _cohort()
    records = cohort["records"]
    assert isinstance(records, list)
    if defect == "missing_branch":
        records.pop()
    elif defect == "duplicate_branch":
        records.append(dict(records[0]))
    elif defect == "incomplete_branch":
        records[0]["status"] = "running"
    elif defect == "bad_audit":
        cohort["audit"]["passed"] = False
    elif defect == "bad_cost":
        cohort["cost"]["observer_steps"] += 1
    elif defect == "bad_count":
        records[0]["captured_tests"] -= 1
    elif defect == "bad_replicates":
        records[0]["candidates"][0]["outcome"]["replicate_gains"].pop()
    else:
        records[0]["candidates"][0]["outcome"]["sign"] = "beneficial"
    source = _write_cohort(tmp_path / "cohort", cohort)
    output = tmp_path / "report"

    with pytest.raises(ValueError):
        report.analyze(source, output)
    assert not output.exists()


def test_nonempty_report_path_is_refused(tmp_path: Path) -> None:
    source = _write_cohort(tmp_path / "cohort")
    output = tmp_path / "report"
    output.mkdir()
    (output / "keep.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        report.analyze(source, output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "existing"

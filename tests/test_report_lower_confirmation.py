import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import report_lower_confirmation as report  # noqa: E402


SEEDS_A = list(range(22101, 22109))
SEEDS_B = list(range(22201, 22209))


def _config(seeds, histories):
    return {
        "format": "storage_decision_v1",
        "seeds": list(seeds),
        "pair_kinds": ["dynamics"],
        "history_regimes": list(histories),
        "future_regimes": ["old_major", "new_major"],
        "actions": list(report.ACTIONS),
        "report": {
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20261019,
            "practical_reward_per_step": 0.002,
            "secondary": ["replace", "mean", "frequency"],
        },
    }


def _case(seed, family, history, lower_minus_keep, *, frequency="mean", row_base=100):
    base = 0.01 + (0.001 if history == "new_major" else 0.0) + (seed % 7) / 10000
    rewards = {
        "keep": base,
        "replace": base + lower_minus_keep,
        "mean": base + lower_minus_keep - 0.003,
        "none": base - 0.004,
    }
    branches = []
    for action in report.ACTIONS:
        rate = rewards[action]
        branches.append({
            "action": action,
            "future_regime": history,
            "environment_steps": 10,
            "reward_total": rate * 10,
            "reward_per_step": rate,
            "raw_environment_steps": row_base,
        })
    return {
        "case_id": f"s{seed}_dynamics_{history}",
        "seed": seed,
        "pair_kind": "dynamics",
        "history_regime": history,
        "confirmation_action": "replace",
        "confirmation_future_regime": history,
        "confirmation_future_scope": "history_matched_only",
        "selectors": {
            "surrogate_lower": "replace",
            "frequency": frequency,
            "replace": "replace",
            "mean": "mean",
            "none": "none",
            "keep": "keep",
        },
        "prefix": {
            "raw_environment_steps": row_base // 2,
            "per_bout": [
                {"phase": "bootstrap", "group": 0, "task": f"{family}0+"},
                {"phase": "bootstrap", "group": 0, "task": f"{family}1+"},
            ],
        },
        "branches": branches,
        "costs": {
            "prefix_environment_steps": 5,
            "branch_environment_steps": 40,
            "environment_steps_total": 45,
            "case_cpu_s": 1.25,
            "case_wall_s": 1.5,
        },
        "errors": [],
    }


def _summary(*, deltas=None, histories=report.HISTORIES, phase="main", frequency="mean"):
    if deltas is None:
        deltas = {seed: (0.01 if seed in SEEDS_A else 0.03) for seed in SEEDS_A + SEEDS_B}
    seeds = list(deltas)
    cases = []
    for seed in seeds:
        family = "A" if seed in SEEDS_A else "B"
        for history in histories:
            delta = deltas[seed]
            if isinstance(delta, dict):
                delta = delta[history]
            cases.append(_case(seed, family, history, delta, frequency=frequency, row_base=100))
    main_cohort = phase == "main" and tuple(histories) == report.HISTORIES
    return {
        "format": "storage_decision_summary_v1",
        "phase": phase,
        "analysis_role": "independent_surrogate_lower_confirmation",
        "confirmation": {
            "selector": "surrogate_lower",
            "selectors_are_past_only": True,
            "physical_actions": list(report.ACTIONS),
        },
        "config": _config(seeds, histories),
        "seeds": seeds,
        "case_count": len(cases),
        "branch_count": len(cases) * 4,
        "cases": cases,
        "errors": [],
        "costs": {
            "environment_steps_total": len(cases) * 45,
            "prefix_environment_steps": len(cases) * 5,
            "branch_environment_steps": len(cases) * 40,
            "cpu_s": None,
            "wall_s": len(cases) * 1.5,
        },
    }


def _write_summary(path: Path, value: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_complete_confirmation_uses_seed_average_stratified_bootstrap_and_observer_branches(tmp_path: Path):
    deltas = {seed: (0.01 if seed in SEEDS_A else 0.03) for seed in SEEDS_A + SEEDS_B}
    deltas[SEEDS_A[0]] = {"old_major": 0.04, "new_major": -0.02}
    source = _write_summary(tmp_path / "input", _summary(deltas=deltas, frequency="mean"))

    result = report.analyze(source, tmp_path / "report")

    assert result["status"] == "complete_confirmation"
    assert result["bootstrap"]["computed"] is True
    assert result["bootstrap"]["resamples"] == 10000
    assert result["bootstrap"]["seed"] == 20261019
    assert result["primary"]["estimate"]["n"] == 16
    # The first seed averages (+0.04, -0.02) before entering the seed mean.
    first = next(row for row in result["per_seed"] if row["seed"] == SEEDS_A[0])
    assert first["metrics"]["lower_minus_keep"] == pytest.approx(0.01)
    # frequency selects mean, so its contrast is evaluated from the paid mean branch.
    assert result["secondary"]["frequency"]["estimate"]["mean"] == pytest.approx(0.003)
    assert result["family_seed_counts"] == {"A": 8, "B": 8}
    assert result["primary"]["estimate"]["ci95"] == pytest.approx([0.02, 0.02])
    assert result["primary"]["directional_status"] == "directionally_confirmed"
    assert result["practical_threshold"]["status"] == "supports_practical_threshold"
    assert result["costs"]["actual_rows"] == len(report.HISTORIES) * 16 * (50 + 4 * 100)
    assert result["costs"]["actual_rows_complete"] is True
    assert result["scoring"]["branch_outcomes_feed_back_to_learner"] is False

    for filename in ("analysis.json", "report.md", "per_seed.csv"):
        assert (tmp_path / "report" / filename).is_file()
    markdown = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "Original practical threshold" in markdown
    assert "observer-only" in markdown
    with (tmp_path / "report" / "per_seed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert rows[0]["observed_history_mean_lower_minus_keep"] != ""


def test_ties_stay_in_all_seed_denominators(tmp_path: Path):
    values = {}
    for seed in SEEDS_A:
        values[seed] = 0.01
    for index, seed in enumerate(SEEDS_B):
        values[seed] = 0.0 if index < 4 else -0.01
    source = _write_summary(tmp_path / "input", _summary(deltas=values))

    result = report.analyze(source, tmp_path / "report")
    estimate = result["primary"]["estimate"]

    assert estimate["positive"] == 8
    assert estimate["negative"] == 4
    assert estimate["tie"] == 4
    assert estimate["denominator"] == 16
    assert estimate["n"] == 16


def test_one_history_pilot_is_explicitly_incomplete_and_has_no_confirmation_interval(tmp_path: Path):
    pilot = _summary(
        deltas={22089: 0.01},
        histories=("old_major",),
        phase="pilot",
    )
    source = _write_summary(tmp_path / "pilot", pilot)

    result = report.analyze(source, tmp_path / "report")

    assert result["status"] == "incomplete_pilot"
    assert result["primary"]["estimate"] is None
    assert result["bootstrap"]["computed"] is False
    assert result["per_seed"][0]["matched_history_count"] == 1
    assert result["costs"]["actual_rows"] == 50 + 4 * 100
    assert result["costs"]["cpu_s"] is None
    markdown = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "incomplete pilot" in markdown
    assert "not confirmation evidence" in markdown


def test_missing_main_case_marks_report_incomplete_without_bootstrap(tmp_path: Path):
    summary = _summary()
    summary["cases"].pop()
    summary["case_count"] -= 1
    summary["branch_count"] -= 4
    source = _write_summary(tmp_path / "input", summary)

    result = report.analyze(source, tmp_path / "report")

    assert result["status"] == "incomplete_cohort"
    assert result["primary"]["estimate"] is None
    assert result["bootstrap"]["computed"] is False
    assert any("missing case" in reason for reason in result["incomplete_reasons"])


def test_bootstrap_and_report_json_are_reproducible(tmp_path: Path):
    source = _write_summary(tmp_path / "input", _summary())

    first = report.analyze(source, tmp_path / "report_a")
    second = report.analyze(source, tmp_path / "report_b")

    assert first["primary"]["estimate"]["ci95"] == second["primary"]["estimate"]["ci95"]
    first_json = json.loads((tmp_path / "report_a" / "analysis.json").read_text(encoding="utf-8"))
    second_json = json.loads((tmp_path / "report_b" / "analysis.json").read_text(encoding="utf-8"))
    assert first_json == second_json

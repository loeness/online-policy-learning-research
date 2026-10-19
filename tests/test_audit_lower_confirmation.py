from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import audit_lower_confirmation as audit_lower
from scripts import audit_storage_decision as storage_audit


def test_expected_plan_matches_frozen_pilot_and_main_budgets() -> None:
    pilot = audit_lower.expected_plan(
        "pilot", seeds=(22089,), histories=("old_major",),
        bootstrap_steps=512, bout_steps=256,
    )
    assert pilot == {
        "case_count": 1,
        "branch_count": 4,
        "prefix_steps_per_case": 3072,
        "branch_steps_per_branch": 3072,
        "prefix_environment_steps": 3072,
        "branch_environment_steps": 12288,
        "environment_steps_total": 15360,
    }

    main = audit_lower.expected_plan(
        "main", seeds=tuple(range(16)), histories=("old_major", "new_major"),
        bootstrap_steps=2048, bout_steps=1024,
    )
    assert main == {
        "case_count": 32,
        "branch_count": 128,
        "prefix_steps_per_case": 12288,
        "branch_steps_per_branch": 12288,
        "prefix_environment_steps": 393216,
        "branch_environment_steps": 1572864,
        "environment_steps_total": 1966080,
    }


def _confirmation_fixture() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    record: dict[str, object] = {
        "selectors": {"surrogate_lower": "replace"},
        "confirmation_selector": "surrogate_lower",
        "confirmation_action": "replace",
        "confirmation_future_regime": "old_major",
        "confirmation_future_scope": "history_matched_only",
    }
    decision: dict[str, object] = {
        "selectors": {"surrogate_lower": "replace"},
        "selector_diagnostics": {"past_only_selectors": {"surrogate_lower": "replace"}},
    }
    branches: list[dict[str, object]] = [
        {"action": action, "future_regime": "old_major"}
        for action in ("keep", "replace", "mean", "none")
    ]
    return record, decision, branches


def test_confirmation_action_binds_past_only_selection_to_matched_branch() -> None:
    audit = storage_audit.Audit(Path("unused"))
    record, decision, branches = _confirmation_fixture()

    audit_lower._check_confirmation_action(
        audit, record, decision, branches, history="old_major", where="case"
    )

    assert audit.errors == []


@pytest.mark.parametrize(
    ("field", "replacement", "expected_code"),
    [
        ("confirmation_action", "keep", "confirmation_selector_mismatch"),
        ("confirmation_future_scope", "all_futures", "confirmation_future_scope"),
    ],
)
def test_confirmation_action_rejects_recorded_verdict_mismatch(
    field: str, replacement: str, expected_code: str,
) -> None:
    audit = storage_audit.Audit(Path("unused"))
    record, decision, branches = _confirmation_fixture()
    record[field] = replacement

    audit_lower._check_confirmation_action(
        audit, record, decision, branches, history="old_major", where="case"
    )

    assert expected_code in {row["code"] for row in audit.errors}


def test_raw_inventory_counts_extra_step_ledgers_and_marks_total_incomplete(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort"
    expected = cohort / "cases" / "s1" / "prefix" / "steps.csv"
    extra = cohort / "cases" / "s1" / "unexpected" / "steps.csv"
    for path, rewards in ((expected, (1.0, 2.0)), (extra, (3.0,))):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("reward",))
            writer.writeheader()
            for reward in rewards:
                writer.writerow({"reward": reward})

    audit = storage_audit.Audit(cohort)
    observed, partial = audit_lower._raw_inventory(audit, cohort, {expected.resolve()})

    assert observed is None
    assert partial == 3
    assert "unexpected_steps_file" in {row["code"] for row in audit.errors}


@pytest.mark.skipif(
    not (Path("outputs/lower_confirmation_pilot") / "summary.json").is_file(),
    reason="completed lower-confirmation pilot artifacts are unavailable",
)
def test_completed_pilot_artifacts_pass_read_only_audit() -> None:
    report = audit_lower.audit_cohort("outputs/lower_confirmation_pilot")

    assert report["passed"] is True, report["errors"]
    assert report["phase"] == "pilot"
    assert report["observed"]["case_count"] == 1
    assert report["observed"]["branch_count"] == 4
    assert report["observed"]["persisted_raw_steps"] == 15360

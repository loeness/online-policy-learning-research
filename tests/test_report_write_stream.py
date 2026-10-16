"""Synthetic output-ledger audit checks for write-stream reporting."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_write_stream import analyze_cohort, _snapshot_audit  # noqa: E402


SEEDS = (17, 29)
CONDITIONS = ("reward_shared", "independent7_lru", "trial_gate")
BASE_SEQUENCE = (
    "A0+", "A0-", "B0+", "B0-", "A1+", "A1-", "B1+", "B1-", "A2+", "A2-", "B2+", "B2-",
)
SEQUENCE = BASE_SEQUENCE + tuple(reversed(BASE_SEQUENCE))


@pytest.mark.parametrize("flag,protocol", [
    ({"budget_baseline": True}, "WRITE_BUDGET_BASELINE.md"),
    ({"parent_recycle": False}, "WRITE_RECYCLE_PROTOCOL.md"),
    ({"parent_recycle": True}, "WRITE_RECYCLE_PROTOCOL.md"),
])
def test_wrapper_snapshot_audit_uses_the_declared_wrapper_protocol(tmp_path, flag, protocol):
    snapshot = tmp_path / "source_snapshot"
    sources = {}
    for name in ("WRITE_STREAM_PROTOCOL.md", protocol):
        path = snapshot / "docs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        sources[f"docs/{name}"] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    manifest = {"sources": sources}
    _write_json(snapshot / "manifest.json", manifest)
    frozen = {**flag, "protocol_sha256": sources[f"docs/{protocol}"]["sha256"]}
    checks = []
    _snapshot_audit(tmp_path, {"source_manifest": manifest}, frozen, checks)
    result = next(check for check in checks if check["name"] == "frozen_protocol_hash_matches_snapshot")
    assert result["passed"] is True
    frozen["protocol_sha256"] = "wrong_hash"
    checks = []
    _snapshot_audit(tmp_path, {"source_manifest": manifest}, frozen, checks)
    result = next(check for check in checks if check["name"] == "frozen_protocol_hash_matches_snapshot")
    assert result["passed"] is False


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(cohort_dir: Path, foundation: Path, dataset: Path, selection: Path,
              phase: str, seed: int, condition: str, condition_index: int) -> dict[str, object]:
    run = cohort_dir / f"seed_{seed}" / condition
    run.mkdir(parents=True)
    phase_bonus = 0.5 if phase == "unknown" else 0.0
    seed_bonus = 0.25 if seed == 29 else 0.0
    reward_per_step = float(condition_index + 1) + phase_bonus + seed_bonus
    train_steps, probe_steps, eval_steps = 14, 2, 5

    step_rows = []
    for index in range(train_steps):
        step_rows.append({
            "interaction_kind": "training", "raw_step": index + 1,
            "taskobserver": "A0+", "phase": "train", "reward": reward_per_step,
        })
    for index in range(probe_steps):
        step_rows.append({
            "interaction_kind": "probe", "raw_step": train_steps + index + 1,
            "taskobserver": "A0+", "phase": "uniform_probe", "reward": 2.0,
        })
    for index in range(eval_steps):
        step_rows.append({
            "interaction_kind": "evaluation", "raw_step": train_steps + probe_steps + index + 1,
            "taskobserver": "A0+", "phase": "final_clone", "reward": 0.0,
        })
    _write_csv(run / "steps.csv", ["interaction_kind", "raw_step", "taskobserver", "phase", "reward"], step_rows)

    episode_rows = [
        {"interaction_kind": "training", "episode_start_taskobserver": "A0+", "taskobserver": "A0+",
         "complete": 1, "goal_reached": 1, "positive_success": 1, "length": 3},
        {"interaction_kind": "training", "episode_start_taskobserver": "A0+", "taskobserver": "A0+",
         "complete": 1, "goal_reached": 0, "positive_success": 0, "length": 2},
        # This ended in a positive task after crossing the boundary from A0-.
        # It is deliberately excluded from the positive-goal denominator.
        {"interaction_kind": "training", "episode_start_taskobserver": "A0-", "taskobserver": "A0+",
         "complete": 1, "goal_reached": 1, "positive_success": 1, "length": 4},
        {"interaction_kind": "training", "episode_start_taskobserver": "A0-", "taskobserver": "A0-",
         "complete": 1, "goal_reached": 1, "positive_success": 0, "length": 5},
        {"interaction_kind": "probe", "episode_start_taskobserver": "A0+", "taskobserver": "A0+",
         "complete": 1, "goal_reached": 0, "positive_success": 0, "length": 2},
    ]
    _write_csv(run / "episodes.csv", [
        "interaction_kind", "episode_start_taskobserver", "taskobserver", "complete",
        "goal_reached", "positive_success", "length",
    ], episode_rows)

    eval_rows = [
        {"phase": "final_clone", "taskobserver": "A0+", "return_value": reward_per_step + 2,
         "goal_reached": 1, "positive_success": 1, "length": 2, "complete": 1},
        {"phase": "final_clone", "taskobserver": "A0-", "return_value": -reward_per_step,
         "goal_reached": 1, "positive_success": 0, "length": 3, "complete": 1},
    ]
    _write_csv(run / "evaluations.csv", [
        "phase", "taskobserver", "return_value", "goal_reached", "positive_success", "length", "complete",
    ], eval_rows)

    route_rows = []
    if phase == "unknown":
        route_rows = [
            {"raw_step": 1, "bout": 0, "taskobserver": "A0+", "slot_before": -1,
             "active_after": -1, "allocated_slot": "", "evicted_slot": "", "switched": 0,
             "reason": "insufficient_support", "confidence": 0.0},
            {"raw_step": 2, "bout": 0, "taskobserver": "A0+", "slot_before": -1,
             "active_after": 0, "allocated_slot": 0, "evicted_slot": "", "switched": 1,
             "reason": "first_profile", "confidence": 0.5},
            {"raw_step": 3, "bout": 1, "taskobserver": "A0-", "slot_before": 0,
             "active_after": 2, "allocated_slot": 2, "evicted_slot": 2, "switched": 1,
             "reason": "novel_profile", "confidence": 0.5},
        ]
    _write_csv(run / "routing.csv", [
        "raw_step", "bout", "taskobserver", "slot_before", "active_after", "allocated_slot",
        "evicted_slot", "switched", "reason", "confidence",
    ], route_rows)
    _write_csv(run / "profile_assignments.csv", ["phase", "bout", "slot", "generation", "group", "reason", "profile_rows", "scores", "committed"], [])

    decisions = []
    if condition == "trial_gate":
        decisions = [
            {"accepted": False, "reason": "cold_gain_below_threshold",
             "gate": {"accepted": False, "reason": "cold_gain_below_threshold"}},
            {"accepted": False, "reason": "partial_cycle_no_fresh_holdout", "gate": None},
        ]
    (run / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8")

    coverage = [{
        "taskobserver": "A0+", "training_steps": train_steps, "grouped_steps": 8,
        "unassigned_steps": 6, "group_full_steps": 4, "unassigned_when_full_steps": 1,
        "resident_slots": "[0]", "groups_used": "[0, 1]",
    }]
    _write_csv(run / "observer_group_coverage.csv", list(coverage[0]), coverage)

    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_size, actor_size, foundation_size = 64, 100, foundation.stat().st_size
    (run / "checkpoint_final.pt").write_bytes(b"c" * checkpoint_size)
    _write_json(run / "capacity.json", [{
        "step": train_steps, "actor_bytes": actor_size, "actor_budget_bytes": 236,
        "checkpoint_bytes": checkpoint_size, "foundation_bytes": foundation_size,
        "full_persistent_bytes": foundation_size + checkpoint_size,
        "package_budget_bytes": 262144,
    }])

    snapshot = run / "source_snapshot"
    snapshot.mkdir()
    source_file = snapshot / "dummy.py"
    source_file.write_text("frozen runner source\n", encoding="utf-8")
    protocol_source = ROOT / "docs" / "WRITE_STREAM_PROTOCOL.md"
    protocol_copy = snapshot / "docs" / "WRITE_STREAM_PROTOCOL.md"
    protocol_copy.parent.mkdir(parents=True)
    shutil.copy2(protocol_source, protocol_copy)
    shutil.copy2(selection, snapshot / "selection.json")
    sources = {}
    for relative, path in (("dummy.py", source_file), ("docs/WRITE_STREAM_PROTOCOL.md", protocol_copy)):
        sources[relative] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    manifest = {
        "sources": sources,
        "selection": {"sha256": _sha(snapshot / "selection.json"), "bytes": (snapshot / "selection.json").stat().st_size},
        "foundation": {"path": str(foundation), "sha256": _sha(foundation), "bytes": foundation_size},
        "dataset": {"path": str(dataset), "sha256": _sha(dataset), "bytes": dataset.stat().st_size},
    }
    _write_json(snapshot / "manifest.json", manifest)

    run_config = {
        "phase": phase, "seed": seed, "condition": condition,
        "sequence": list(SEQUENCE), "actor_codec": "int8", "actor_budget_bytes": 236,
        "package_budget_bytes": 262144, "evaluation_targets": 2, "evaluation_episodes": 1,
        "foundation_sha256": _sha(foundation),
    }
    _write_json(run / "config.json", run_config)
    summary = {
        "phase": phase, "condition": condition, "seed": seed, "domain": "navigation",
        "sequence": list(SEQUENCE), "counts": {
            "environment_steps": train_steps, "probe_steps": probe_steps,
            "raw_steps": train_steps + probe_steps, "evaluation_steps": eval_steps,
            "positive_completed_episodes": 3, "positive_goal_episodes": 2,
            "unknown_actor_rows": 2 if phase == "unknown" else 0,
        },
        "training_reward": reward_per_step * train_steps, "reward_per_step": reward_per_step,
        "per_task_observed_stream": {
            "A0+": {"reward": reward_per_step * train_steps + 4.0,
                    "steps": train_steps + probe_steps},
        },
        # This mimics the observer summary before cross-task episodes are removed.
        "positive_goal_rate": 2 / 3,
        "evaluation_steps": eval_steps,
        "costs": {
            "training_steps": train_steps, "probe_steps": probe_steps,
            "evaluation_steps": eval_steps, "total_interactions": train_steps + probe_steps + eval_steps,
            "cpu_s": 1.25, "wall_s": 2.5, "decision_cpu_s": 0.1,
            "evaluation_cpu_s": 0.2, "process_rss_bytes": 4096,
        },
        "group_saturation": {
            "group_abstentions": 3, "max_groups_abstentions": 1,
            "profile_reason_frequencies": {"matched": 2, "max_groups_reached": 1},
            "group_full_steps_total": 4, "unassigned_when_full_steps_total": 1,
            "unassigned_steps_total": 6,
        },
        "source_manifest": manifest,
    }
    _write_json(run / "summary.json", summary)
    return summary


def _make_cohort(root: Path, phase: str) -> None:
    cohort = root / phase
    cohort.mkdir(parents=True)
    foundation = root / "foundation.pt"
    dataset = root / "dataset.npz"
    selection = root / "selection.json"
    if not foundation.exists():
        foundation.write_bytes(b"w0")
        dataset.write_bytes(b"dataset")
        _write_json(selection, {"format": "write_decision_selection_v1", "selected_conditions": ["trial_gate", "local_promote"]})
    records = []
    for seed in SEEDS:
        for index, condition in enumerate(CONDITIONS):
            records.append(_make_run(cohort, foundation, dataset, selection, phase, seed, condition, index))
    protocol_sha = _sha(ROOT / "docs" / "WRITE_STREAM_PROTOCOL.md")
    _write_json(cohort / "frozen_config.json", {
        "phase": phase, "conditions": list(CONDITIONS), "seeds": list(SEEDS),
        "base_sequence": list(BASE_SEQUENCE), "sequence": list(SEQUENCE),
        "selection_path": str(selection), "selection_sha256": _sha(selection),
        "protocol_sha256": protocol_sha,
    })
    _write_json(cohort / "summary.json", {
        "phase": phase, "base_sequence": list(BASE_SEQUENCE), "sequence": list(SEQUENCE),
        "conditions": list(CONDITIONS), "seeds": list(SEEDS), "records": records,
        "protocol_sha256": protocol_sha,
        "cohort_cost": {
            "training_steps": 6 * 14, "probe_steps": 6 * 2,
            "evaluation_steps": 6 * 5, "total_interactions": 6 * 21,
        },
    })


def test_stream_report_pairs_seeds_and_audits_ledgers_capacity_and_snapshots(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohorts"
    cohort_root.mkdir()
    _make_cohort(cohort_root, "known")
    _make_cohort(cohort_root, "unknown")

    report_dir = tmp_path / "report"
    analysis = analyze_cohort(cohort_root, report_dir)

    assert set(analysis["phase_analyses"]) == {"known", "unknown"}
    assert analysis["statistical_unit"] == "seed"
    assert analysis["significance_claim"] is False
    assert len(analysis["per_seed"]) == 12
    assert analysis["audit"]["passed"] is True

    known_trial = next(row for row in analysis["per_seed"]
                       if row["phase"] == "known" and row["seed"] == 17 and row["condition"] == "trial_gate")
    assert known_trial["metrics"]["reward_per_train_step"] == 3.0
    assert known_trial["metrics"]["training_and_probe_reward"] == 46.0
    assert known_trial["metrics"]["reward_per_stream_interaction"] == 46.0 / 16
    assert known_trial["costs"]["raw_training_and_probe_steps"] == 16
    assert "observed_stream_steps_match_steps_ledger" not in known_trial["audit"]["failed_checks"]
    assert "observed_stream_reward_matches_steps_ledger" not in known_trial["audit"]["failed_checks"]
    assert known_trial["training_outcomes"]["positive_goal_rate"] == 0.5
    assert known_trial["training_outcomes"]["cross_task_training_episodes_excluded"] == 1
    assert known_trial["training_outcomes"]["runner_rate_comparison"] == "not_compared_cross_task_episodes_excluded"
    assert known_trial["final_clone_evaluation"]["evaluation_steps_from_episode_lengths"] == 5
    assert known_trial["gates"]["gate_full_count"] == 1
    assert known_trial["gates"]["gate_reject_count"] == 1
    assert known_trial["gates"]["partial_no_holdout_count"] == 1
    assert known_trial["group_coverage"]["unassigned_when_full_steps"] == 1
    assert known_trial["capacity"]["actor_bytes"]["max"] == 100
    assert known_trial["capacity"]["checkpoint_bytes"]["max"] == 64

    unknown_shared = next(row for row in analysis["per_seed"]
                          if row["phase"] == "unknown" and row["seed"] == 17 and row["condition"] == "reward_shared")
    assert unknown_shared["routing"]["unknown_before_action_rows"] == 2
    assert unknown_shared["routing"]["unknown_actor_rows"] == 2
    assert unknown_shared["routing"]["eviction_count"] == 1
    assert "not false-route error counts" in unknown_shared["routing"]["error_attribution"]

    trial_comparison = analysis["phase_analyses"]["known"]["paired_differences"]["comparisons"]["trial_gate"]
    reward_diff = trial_comparison["reward_shared"]["reward_per_train_step"]
    assert reward_diff["common_seeds"] == [17, 29]
    assert reward_diff["per_seed"][0]["difference"] == 2.0
    assert reward_diff["mean_difference"] == 2.0
    stream_reward_diff = trial_comparison["reward_shared"]["reward_per_stream_interaction"]
    assert stream_reward_diff["per_seed"][0]["difference"] == 1.75
    independent_diff = trial_comparison["independent7_lru"]["reward_per_train_step"]
    assert independent_diff["per_seed"][0]["difference"] == 1.0
    independent_stream_diff = trial_comparison["independent7_lru"]["reward_per_stream_interaction"]
    assert independent_stream_diff["per_seed"][0]["difference"] == 0.875
    assert analysis["unknown_minus_known_phase_pairs"]["n"] == 6
    assert analysis["unknown_minus_known_phase_pairs"]["per_seed"][0][
        "unknown_minus_known_reward_per_stream_interaction"
    ] == 0.4375
    assert (report_dir / "analysis.json").is_file()
    assert {path.suffix for path in report_dir.glob("*.png")} == {".png"}
    assert {path.suffix for path in report_dir.glob("*.pdf")} == {".pdf"}
    report = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "统计单位是 seed" in report
    assert "跨 task episode 会排除" in report
    assert "低 unknown 或低未分组比例本身不代表算法优化" in report
    assert "reward / training + probe interaction differences" in report
    assert "policy dwell time" in report


def test_stream_report_detects_a_physical_checkpoint_size_mismatch(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    _make_cohort(cohort_root, "known")
    checkpoint = cohort_root / "known" / "seed_17" / "reward_shared" / "checkpoint_final.pt"
    checkpoint.write_bytes(b"wrong physical size")

    analysis = analyze_cohort(cohort_root / "known", tmp_path / "report")
    row = next(item for item in analysis["per_seed"] if item["seed"] == 17 and item["condition"] == "reward_shared")

    assert "physical_checkpoint_sizes_match_capacity_ledger" in row["audit"]["failed_checks"]
    assert analysis["audit"]["passed"] is False


def test_stream_reward_is_missing_when_any_training_or_probe_reward_is_missing(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    _make_cohort(cohort_root, "known")

    steps_path = cohort_root / "known" / "seed_17" / "reward_shared" / "steps.csv"
    with steps_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    missing_probe = next(row for row in rows if row["interaction_kind"] == "probe")
    missing_probe["reward"] = ""
    _write_csv(steps_path, list(rows[0]), rows)

    analysis = analyze_cohort(cohort_root / "known", tmp_path / "report")
    row = next(item for item in analysis["per_seed"]
               if item["seed"] == 17 and item["condition"] == "reward_shared")

    assert row["metrics"]["reward_per_train_step"] == 1.0
    assert row["metrics"]["reward_per_stream_interaction"] is None
    assert row["costs"]["stream_reward_missing_rows"] == 1
    assert "observed_stream_reward_matches_steps_ledger" in row["audit"]["failed_checks"]

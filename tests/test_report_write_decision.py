"""End-to-end checks for the write-decision cohort analysis artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_write_decision import analyze_cohort  # noqa: E402


CONDITIONS = (
    "independent_fast",
    "dynamics_shared",
    "reward_shared",
    "trial_gate",
    "trial_gate_relaxed",
    "projected_gate",
    "local_promote",
    "trial_rollback",
)
SEEDS = (17, 29)
TASKS = ("A0+", "A0-", "A1+", "A1-", "A2+", "A2-")
STAGES = ("initial", "after_source", "final")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(root: Path, seed: int, condition: str, condition_rank: int) -> dict[str, object]:
    run = root / f"seed_{seed}" / condition
    run.mkdir(parents=True)
    seed_offset = 0.1 if seed == 29 else 0.0
    base = float(condition_rank) + seed_offset
    training_reward = 1024.0 * base

    steps = []
    for step in range(1024):
        steps.append({
            "interaction_kind": "training",
            "phase": "fit" if step < 768 else "holdout",
            "reward": f"{base:.8f}",
        })
    for stage in STAGES:
        steps.extend({
            "interaction_kind": "evaluation",
            "phase": stage,
            "reward": "0.0",
        } for _ in TASKS)
    _write_csv(run / "steps.csv", ["interaction_kind", "phase", "reward"], steps)
    _write_csv(run / "probes.csv", ["interaction_kind"], [{"interaction_kind": "probe"}])
    _write_csv(
        run / "episodes.csv",
        ["interaction_kind", "taskobserver", "complete", "goal_reached", "length"],
        [
            {"interaction_kind": "training", "taskobserver": "A0+", "complete": 1, "goal_reached": 0, "length": 1},
            # Negative-task success must not enter the positive-task denominator.
            {"interaction_kind": "training", "taskobserver": "A0-", "complete": 1, "goal_reached": 1, "length": 1},
        ],
    )

    eval_rows = []
    eval_summary = {}
    for stage in STAGES:
        stage_data = {}
        for task in TASKS:
            is_positive = task.endswith("+")
            task_return = base + 0.5 if is_positive else -base
            if stage == "after_source" and task == "A2+":
                task_return = base + 2.0
            if stage == "final" and task == "A2+":
                task_return = base + 3.0
            goal = 0 if is_positive else 1
            episode = {
                "return_value": task_return,
                "goal_reached": goal,
                "length": 1,
            }
            stage_data[task] = {
                "mean_return": task_return,
                "mean_goal_rate": goal,
                "episodes": [episode],
                "negative_goal": not is_positive,
            }
            eval_rows.append({
                "stage": stage,
                "taskobserver": task,
                "return_value": task_return,
                "goal_reached": goal,
                "length": 1,
                "complete": 1,
            })
        eval_summary[stage] = stage_data
    _write_csv(
        run / "evaluations.csv",
        ["stage", "taskobserver", "return_value", "goal_reached", "length", "complete"],
        eval_rows,
    )

    decision_cpu = 0.002
    if condition == "trial_gate":
        decision = {
            "accepted": False,
            "reason": "cold_gain_below_threshold",
            "decision_cpu_s": decision_cpu,
            "gate": {
                "accepted": False,
                "reason": "cold_gain_below_threshold",
                "current_cold": {
                    "ess": 12.0,
                    "behavior_kl": 0.03,
                    "max_kl": 0.04,
                    "mean_delta": -0.01,
                    "episode_se": 0.02,
                    "sample_count": 16,
                    "episode_count": 4,
                    "support_ok": True,
                },
                "old_aggregate": {
                    "ess": 10.0,
                    "behavior_kl": 0.02,
                    "max_kl": 0.05,
                    "mean_delta": 0.0,
                    "episode_se": 0.01,
                    "sample_count": 12,
                    "episode_count": 3,
                    "support_ok": True,
                },
            },
        }
    else:
        reason = "independent_local_commit" if condition == "independent_fast" else "direct_joint_ppo_commit"
        decision = {"accepted": True, "reason": reason, "decision_cpu_s": decision_cpu}
    (run / "decisions.jsonl").write_text(json.dumps(decision) + "\n", encoding="utf-8")

    if condition == "reward_shared":
        deployed_group, shadow_group, assignment_reason = -1, -1, "reward_mismatch"
    elif condition == "independent_fast":
        deployed_group, shadow_group, assignment_reason = -1, 0, "first_group"
    else:
        deployed_group, shadow_group, assignment_reason = 0, 0, "first_group"
    _write_csv(run / "profile_assignments.csv", [
        "bout", "taskobserver", "slot", "generation", "group", "shadow_group",
        "reason", "profile_rows", "profile_bytes", "scores",
    ], [{
        "bout": 0,
        "taskobserver": "A0+",
        "slot": 0,
        "generation": 1,
        "group": deployed_group,
        "shadow_group": shadow_group,
        "reason": assignment_reason,
        "profile_rows": 128,
        "profile_bytes": 256,
        "scores": "{}",
    }])

    actor_bytes, checkpoint_bytes = 16, 64
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint_step_00001024.pt").write_bytes(b"c" * checkpoint_bytes)
    (checkpoint_dir / "checkpoint_step_00001024.actor.bin").write_bytes(b"a" * actor_bytes)
    _write_json(run / "capacity.json", [{
        "step": 1024,
        "actor_bytes": actor_bytes,
        "actor_budget_bytes": 128,
        "checkpoint_bytes": checkpoint_bytes,
        "foundation_bytes": 10,
        "full_persistent_bytes": checkpoint_bytes + 10,
        "complete_agent_state_bytes": checkpoint_bytes,
        "package_budget_bytes": 512,
    }])

    source_snapshot = run / "source_snapshot"
    source_snapshot.mkdir()
    snap_source = source_snapshot / "dummy.py"
    snap_source.write_text("frozen source\n", encoding="utf-8")
    foundation = root / "foundation.pt"
    dataset = root / "dataset.npz"
    foundation.write_bytes(b"foundation")
    dataset.write_bytes(b"dataset")
    _write_json(source_snapshot / "manifest.json", {
        "sources": {
            "dummy.py": {"sha256": _hash(snap_source), "bytes": snap_source.stat().st_size},
        },
        "foundation": {"path": str(foundation), "sha256": _hash(foundation)},
        "dataset": {"path": str(dataset), "sha256": _hash(dataset)},
    })

    training_steps = 1024
    probe_steps = 1
    evaluation_steps = len(STAGES) * len(TASKS)
    total = training_steps + probe_steps + evaluation_steps
    summary = {
        "condition": condition,
        "seed": seed,
        "sequence": ["A0+"],
        "bout_steps": training_steps,
        "counts": {
            "environment_steps": training_steps,
            "probe_steps": probe_steps,
            "evaluation_steps": evaluation_steps,
            "fit_steps": 768,
            "holdout_steps": 256,
            "positive_completed_episodes": 1,
            "positive_goal_episodes": 0,
        },
        "training_reward": training_reward,
        "reward_per_step": base,
        "plus_goal_rate": 0.0,
        "evaluations": eval_summary,
        "costs": {
            "training_steps": training_steps,
            "probe_steps": probe_steps,
            "evaluation_steps": evaluation_steps,
            "total_interactions": total,
            "decision_cpu_s": decision_cpu,
            "evaluation_cpu_s": 0.1,
            "cpu_s": 0.5,
            "wall_s": 1.0,
            "process_rss_bytes": 4096,
        },
        "foundation_sha256": _hash(foundation),
    }
    _write_json(run / "summary.json", summary)
    _write_json(run / "config.json", {
        "bout_steps": training_steps,
        "sequence": ["A0+"],
        "probe_steps": probe_steps,
        "evaluation_episodes": 1,
        "actor_budget_bytes": 128,
        "package_budget_bytes": 512,
    })
    return summary


def test_report_keeps_seed_pairing_and_audits_the_source_cost_and_capacity(tmp_path):
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    conditions = list(CONDITIONS)
    seeds = list(SEEDS)
    summaries = []
    costs = {key: 0 for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")}
    for seed in seeds:
        for rank, condition in enumerate(conditions):
            summary = _make_run(cohort, seed, condition, rank)
            summaries.append(summary)
            for key in costs:
                costs[key] += int(summary["costs"][key])

    _write_json(cohort / "frozen_config.json", {"conditions": conditions, "seeds": seeds})
    _write_json(cohort / "summary.json", {
        "phase": "validation",
        "conditions": conditions,
        "seeds": seeds,
        "records": summaries,
        "cohort_cost": costs,
    })

    analysis = analyze_cohort(cohort, tmp_path / "report")

    assert analysis["statistical_unit"] == "seed"
    assert analysis["significance_claim"] is False
    assert analysis["shared_benefit_truth"] == "undetermined_without_independent_counterfactual"
    assert analysis["false_accept_reject_measurable"] is False
    assert analysis["cohort_audit"]["passed"] is True
    assert len(analysis["per_seed"]) == len(seeds) * len(conditions)
    assert all(row["audit"]["passed"] for row in analysis["per_seed"])

    # With negative-task episodes succeeding and all positive-task episodes failing,
    # positive success remains zero.
    trial = next(row for row in analysis["per_seed"] if row["seed"] == 17 and row["condition"] == "trial_gate")
    assert trial["outcomes"]["positive_training_goal_rate"] == 0.0
    assert trial["outcomes"]["eval_final_positive_goal_rate"] == 0.0
    assert trial["outcomes"]["eval_coldtransfer_after_source_A2plus_mean_return"] == 5.0
    assert trial["gate"]["gate_rejects"] == 1
    assert trial["gate"]["gate_reason_counts"] == {"cold_gain_below_threshold": 1}
    assert trial["gate"]["gate_diagnostics"]["current_cold"]["ess"]["mean"] == 12.0
    reward_shared = next(row for row in analysis["per_seed"] if row["seed"] == 17 and row["condition"] == "reward_shared")
    assert reward_shared["groups"]["shadow_abstention_count"] == 1
    assert reward_shared["groups"]["reason_counts"] == {"reward_mismatch": 1}
    assert trial["raw_cost_audit"]["raw_total_interactions"] == 1043
    assert trial["capacity"]["actor_bytes"]["max"] == 16.0
    assert trial["capacity"]["checkpoint_bytes"]["max"] == 64.0
    assert trial["capacity"]["full_persistent_bytes"]["max"] == 74.0

    comparisons = analysis["paired_differences"]["comparisons"]["trial_gate"]
    assert comparisons["independent_fast"]["common_seeds"] == [17, 29]
    assert comparisons["independent_fast"]["metrics"]["online_reward_per_train_step"]["mean"] == 3.0
    assert comparisons["reward_shared"]["metrics"]["online_reward_per_train_step"]["mean"] == 1.0
    assert len(comparisons["independent_fast"]["metrics"]["online_reward_per_train_step"]["per_seed"]) == 2

    output = tmp_path / "report"
    assert (output / "analysis.json").is_file()
    report_text = (output / "report.md").read_text(encoding="utf-8")
    assert "false accept 和 false reject 目前都不可测" in report_text
    assert "Reward identification is not an oracle" in report_text
    assert "routing coverage is not shown to be solved" in report_text
    assert "统计单位是 seed" in report_text
    assert {path.suffix for path in output.glob("*.png")} == {".png"}
    assert {path.suffix for path in output.glob("*.pdf")} == {".pdf"}

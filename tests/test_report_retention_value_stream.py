from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import report_retention_value_stream as report


METHODS = [
    "learned_linear", "learned_intercept", "random", "keep", "replace",
    "mean", "frequency", "lower", "shared_ppo", "independent",
]


def _config() -> dict[str, Any]:
    return {
        "format": "retention_value_stream_v1",
        "pair_kinds": ["dynamics", "reward_conflict"],
        "bootstrap_steps": 1,
        "bout_steps": 1,
        "outcome_bouts": 3,
        "package_budget_bytes": 262144,
        "value_model": {"lambda_cpu": 1.0},
        "calibration": {
            "seeds": [101, 202], "decisions": 64, "methods": ["random"],
            "pilot": {"seeds": [901], "pair_kinds": ["dynamics"], "decisions": 4},
        },
        "comparison": {
            "seeds": [301, 302], "decisions": 48, "methods": METHODS,
            "pilot": {"seeds": [903], "pair_kinds": ["dynamics"], "decisions": 4},
        },
        "report": {"bootstrap_resamples": 10000, "bootstrap_seed": 20261019},
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_run(
    root: Path,
    *,
    stage: str,
    pilot: bool = False,
    failed_key: tuple[int, str, str] | None = None,
    cohort_costs: dict[str, Any] | None = None,
    summary_protocol_sha256: str = "test-protocol-hash",
) -> Path:
    config = _config()
    design = config[stage]
    if pilot:
        design = {**design, **design["pilot"]}
    warmup_steps = 2 * config["bootstrap_steps"] + 2 * config["bout_steps"]
    decisions = int(design["decisions"])
    outcome_steps = decisions * config["outcome_bouts"] * config["bout_steps"]
    total_steps = warmup_steps + outcome_steps
    methods = design["methods"]
    pair_kinds = design.get("pair_kinds", config["pair_kinds"])
    streams = []

    for seed in design["seeds"]:
        for relation_index, relation in enumerate(pair_kinds):
            for method in methods:
                key = (seed, relation, method)
                stream_id = f"{stage}-{seed}-{relation}-{method}"
                stream_dir = root / "streams" / stream_id
                stream_dir.mkdir(parents=True, exist_ok=True)
                failed = failed_key == key
                requested = decisions
                completed = 0 if failed else requested
                status = "failed" if failed else "completed"
                preactions = []
                feedback = []
                outcome_rates: list[float] = []
                for index in range(requested if not failed else 1):
                    selected_action = None if method in ("shared_ppo", "independent") else report.ACTIONS[index % 4]
                    target = 0.2 + (seed % 5) * 0.01 + relation_index * 0.02 + index * 0.0001
                    if stage == "calibration":
                        linear = target if index >= requested - 32 else target + 10.0
                        intercept = target + 0.5
                        outcome_delta = 0.0
                    else:
                        linear = target + 0.1
                        intercept = target - 0.1
                        outcome_delta = 0.1 if method == "learned_linear" else -0.1 if method == "learned_intercept" else 0.0
                    predictions = {
                        "linear": {action: linear for action in report.ACTIONS},
                        "intercept": {action: intercept for action in report.ACTIONS},
                    }
                    preactions.append({
                        "decision_index": index,
                        "predictions": predictions,
                        "selected_action": selected_action,
                        "action_propensity": 0.25 if selected_action is not None else 1.0,
                        "features": [1.0],
                    })
                    block_steps = config["outcome_bouts"] * config["bout_steps"]
                    utility_cpu = 0.3
                    reward_rate = target + outcome_delta + utility_cpu / block_steps
                    utility_target = target + outcome_delta if selected_action is not None else None
                    feedback.append({
                        "decision_index": index,
                        "selected_action": selected_action,
                        "reward_rate": reward_rate,
                        "block_return_sum": reward_rate * block_steps,
                        "block_steps": block_steps,
                        "utility_cpu_s": utility_cpu,
                        "model_update_cpu_s": 0.001,
                        "utility_target": utility_target,
                        "model_updated": selected_action is not None,
                        "target_excludes_model_update": True,
                    })
                    outcome_rates.append(reward_rate)

                if failed:
                    total_rows = 2
                    warmup_rows = 2
                    observed_outcomes = 0
                else:
                    total_rows = total_steps
                    warmup_rows = warmup_steps
                    observed_outcomes = outcome_steps
                preactions_path = stream_dir / "preactions.jsonl"
                feedback_path = stream_dir / "feedback.jsonl"
                summary_path = stream_dir / "summary.json"
                rawsteps_path = stream_dir / "rawsteps.csv"
                bouts_path = stream_dir / "bouts.jsonl"
                _write_jsonl(preactions_path, preactions)
                _write_jsonl(feedback_path, feedback)
                _write_json(summary_path, {"stream_id": stream_id, "status": status})

                # Write a compact step log whose rewards agree with the block
                # rates, so all return metrics are recomputed from observed rows.
                with rawsteps_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["phase", "reward"])
                    for _ in range(warmup_rows):
                        writer.writerow(["warmup", 0.05])
                    for index in range(observed_outcomes):
                        writer.writerow([
                            "outcome",
                            outcome_rates[index // (config["outcome_bouts"] * config["bout_steps"])],
                        ])

                bouts = []
                expected_bouts = 4 + requested * config["outcome_bouts"]
                for bout_index in range(expected_bouts if not failed else 4):
                    phase = "warmup" if bout_index < 4 else "outcome"
                    task = f"task-{bout_index % 6}"
                    evicted = "task-0" if bout_index == 1 else None
                    recurrence = phase == "outcome" and bout_index == 4
                    decision_index = None if phase == "warmup" else (bout_index - 4) // config["outcome_bouts"]
                    steps = config["bootstrap_steps"] if phase == "warmup" and bout_index < 2 else config["bout_steps"]
                    bout_reward = 0.05 * steps if phase == "warmup" else outcome_rates[decision_index] * steps
                    bouts.append({
                        "phase": phase,
                        "decision_index": decision_index,
                        "bout_index": bout_index,
                        "task": task,
                        "steps_completed": steps,
                        "return_first_256": 0.0,
                        "return_full": bout_reward,
                        "evicted_task": evicted,
                        "recurrence": {
                            "recurrence": recurrence,
                            "seen_before": recurrence,
                            "allocated": recurrence,
                            "evicted_task": evicted,
                        },
                    })
                _write_jsonl(bouts_path, bouts)

                action_counts = {action: sum(row["selected_action"] == action for row in feedback) for action in report.ACTIONS}
                action_count = sum(action_counts.values())
                costs = {
                    "environment_steps": total_rows,
                    "warmup_steps": warmup_rows,
                    "outcome_steps": total_rows - warmup_rows,
                    "ppo_rollouts": 2,
                    "minibatch_updates": 3,
                    "storage_action_count": action_count,
                    "parent_write_count": sum(row["selected_action"] not in (None, "keep") for row in feedback),
                    "eviction_count": 1,
                    "slot_allocation_count": 2,
                    "value_model_updates": action_count,
                    "package_serialization_cpu_s": 0.2,
                    "total_cpu_s": 1.5,
                    "wall_s": 1.7,
                    "utility_cpu_s": utility_cpu * len(feedback),
                    "model_update_cpu_s": 0.001 * len(feedback),
                }
                streams.append({
                    "stream_id": stream_id,
                    "stage": stage,
                    "method": method,
                    "seed": seed,
                    "pair_kind": relation,
                    "status": status,
                    "completed_decisions": completed,
                    "requested_decisions": requested,
                    "summary_path": summary_path.relative_to(root).as_posix(),
                    "preactions_path": preactions_path.relative_to(root).as_posix(),
                    "feedback_path": feedback_path.relative_to(root).as_posix(),
                    "bouts_path": bouts_path.relative_to(root).as_posix(),
                    "rawsteps_path": rawsteps_path.relative_to(root).as_posix(),
                    "storage_action_counts": action_counts,
                    "costs": costs,
                    "peak_package_bytes": 4096,
                    "actor_container_bytes": 89,
                    "actual_actor_bytes": 84,
                    "model_state_bytes": 2048,
                    "failure": {"type": "SyntheticError", "message": "partial fixture"} if failed else None,
                })

    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "frozen_config.json", {
        "format": "run_config", "config": config, "protocol_sha256": "test-protocol-hash",
    })
    summary_path = root / "summary.json"
    _write_json(summary_path, {
        "format": report.SUMMARY_FORMAT,
        "stage": stage,
        "pilot": pilot,
        "frozen_config_path": "frozen_config.json",
        "protocol_sha256": summary_protocol_sha256,
        "streams": streams,
        "totals": {"environment_steps": sum(row["costs"]["environment_steps"] for row in streams)},
        "failures": [row for row in streams if row["status"] != "completed"],
        **({} if cohort_costs is None else {"cohort_costs": cohort_costs}),
    })
    return summary_path


def test_calibration_uses_only_final_tail_and_seed_clustered_relations(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration"
    _write_run(calibration, stage="calibration")
    output = tmp_path / "report"

    result = report.analyze(calibration=calibration, output=output)

    detail = result["calibration"]
    assert detail["status"] == "complete_exploratory_cohort"
    estimate = detail["primary"]["estimate"]
    assert estimate["n"] == 2
    assert estimate["mean"] == pytest.approx(-0.25)
    assert estimate["ci95"] == pytest.approx([-0.25, -0.25])
    seed_mean = next(row for row in detail["per_seed"] if row["seed"] == 101 and row["pair_kind"] == "all_relations")
    assert seed_mean["linear_minus_intercept_mse"] == pytest.approx(-0.25)
    assert detail["selected_action_support"]["tail"] == {"keep": 32, "replace": 32, "mean": 32, "none": 32}
    assert detail["costs"]["actual_step_rows"] == 2 * 2 * (4 + 64 * 3)
    assert detail["costs"]["aggregate_stream_cpu_s"]["sum"] == pytest.approx(6.0)
    assert (output / "analysis.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "per_seed.csv").is_file()
    assert (output / "prequential.csv").is_file()


def test_comparison_pilot_is_descriptive_and_uses_complete_bout_returns(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison_pilot"
    _write_run(comparison, stage="comparison", pilot=True)
    output = tmp_path / "report"

    result = report.analyze(comparison=comparison, output=output)

    detail = result["comparison"]
    assert detail["status"] == "pilot_descriptive_only"
    assert detail["cohort_complete"]
    assert detail["by_relation"]["dynamics"]["primary"]["whole_stream"]["estimate"] is None
    assert detail["streams"][0]["prequential"]["tail_expected_decisions"] == 32
    assert detail["streams"][0]["prequential"]["tail_observed_decisions"] == 0
    assert detail["by_method"]["learned_linear"]["action_support_counts"]["none"] == 1
    assert detail["by_method"]["independent"]["action_support_counts"]["no_retention_operation"] == 4
    assert detail["costs"]["actual_step_rows"] == 10 * (4 + 4 * 3)
    assert all(stream["return_metrics_source"] == "bouts.jsonl" for stream in detail["streams"])
    assert detail["costs"]["campaign_wall_s"] is None
    assert detail["costs"]["campaign_cost_source"] == "unknown_not_logged"
    assert detail["costs"]["aggregate_stream_wall_s"]["sum"] is not None
    assert "Pilot results are not a confirmation conclusion." in (output / "report.md").read_text(encoding="utf-8")


def test_comparison_reports_primary_thresholds_controls_and_ties(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison"
    _write_run(
        comparison,
        stage="comparison",
        cohort_costs={
            "wall_s": 20.0, "child_cpu_s": 10.0, "orchestrator_cpu_s": 1.0,
            "total_cpu_s": 11.0, "workers_actual": 6,
        },
    )
    result = report.analyze(comparison=comparison, output=tmp_path / "report")

    detail = result["comparison"]
    assert detail["status"] == "complete_exploratory_cohort"
    primary = detail["by_relation"]["dynamics"]["primary"]["postwarmup"]
    assert primary["estimate"]["n"] == 2
    assert primary["estimate"]["mean"] == pytest.approx(0.2)
    assert primary["thresholds"]["zero"]["ci95_lower_above_threshold"] is True
    assert primary["thresholds"]["practical_reward_per_step"]["threshold"] == pytest.approx(0.002)
    assert primary["thresholds"]["practical_reward_per_step"]["ci95_lower_above_threshold"] is True
    ties = detail["by_relation"]["dynamics"]["all_method_vs_control"]["keep"]["frequency"]["postwarmup"]["estimate"]
    assert ties["mean"] == pytest.approx(0.0)
    assert ties["tie"] == 2
    assert ties["denominator"] == 2
    assert "frequency" in detail["by_method"]
    assert detail["costs"]["campaign_wall_s"] == pytest.approx(20.0)
    assert detail["costs"]["campaign_total_cpu_s"] == pytest.approx(11.0)
    assert detail["costs"]["workers_actual"] == 6


def test_failed_stream_costs_are_preserved_and_block_bootstrap(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration_failed"
    _write_run(calibration, stage="calibration", failed_key=(101, "dynamics", "random"))
    result = report.analyze(calibration=calibration, output=tmp_path / "report")

    detail = result["calibration"]
    assert detail["status"] == "incomplete_cohort"
    assert detail["primary"]["estimate"] is None
    assert not detail["cohort_complete"]
    failed = next(stream for stream in detail["streams"] if stream["status"] == "failed")
    assert failed["raw_step_metrics"]["actual_step_rows"] == 2
    assert failed["return_metrics_source"] == "rawsteps"
    assert failed["costs"]["total_cpu_s"] == pytest.approx(1.5)
    assert detail["costs"]["actual_step_rows"] == 2 + 3 * (4 + 64 * 3)
    assert detail["costs"]["aggregate_stream_cpu_s"]["sum"] == pytest.approx(6.0)
    assert detail["costs"]["aggregate_stream_cpu_s"]["observed_subtotal"] == pytest.approx(6.0)


def test_protocol_hash_mismatch_is_reported_as_incomplete(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration_pilot"
    _write_run(
        calibration,
        stage="calibration",
        pilot=True,
        summary_protocol_sha256="different-protocol-hash",
    )
    result = report.analyze(calibration=calibration, output=tmp_path / "report")

    detail = result["calibration"]
    assert not detail["protocol"]["hashes_match"]
    assert not detail["cohort_complete"]
    assert any("protocol_sha256 does not match" in reason for reason in detail["incomplete_reasons"])


def test_recurrence_marks_only_after_a_prior_natural_eviction() -> None:
    bouts = [
        {"task": "task-0", "steps_completed": 256, "return_first_256": 64.0,
         "evicted_task": None, "recurrence": {"recurrence": False}},
        {"task": "task-1", "steps_completed": 256, "return_first_256": 128.0,
         "evicted_task": "task-0", "recurrence": {"recurrence": False}},
        {"task": "task-0", "steps_completed": 256, "return_first_256": 192.0,
         "evicted_task": None, "recurrence": {"recurrence": True}},
    ]

    summary = report._recurrence_summary(bouts)

    assert summary["recurrence_bout_count"] == 1
    assert summary["after_eviction_bout_count"] == 1
    assert summary["after_eviction_first256_mean_reward_per_step"] == pytest.approx(0.75)
    assert summary["interpretation"] == "factual natural-recurrence outcomes; not causal recovery loss"

"""Offline reports for selected-feedback calibration and retention streams.

The report reads frozen run summaries and their actual JSON/CSV logs. It does
not launch an environment or feed any observer result back to a learner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACTIONS = ("keep", "replace", "mean", "none")
HISTORY_METHODS = ("learned_linear", "learned_intercept")
CONTROL_METHODS = (
    "random", "keep", "replace", "mean", "frequency", "lower", "shared_ppo", "independent",
)
METRICS = ("whole_stream", "postwarmup")
STAGES = ("calibration", "comparison")
SUMMARY_FORMAT = "retention_value_stream_summary_v1"
REPORT_FORMAT = "retention_value_stream_report_v1"
TAIL_DECISIONS = 32
_COST_FIELDS = (
    "environment_steps", "warmup_steps", "outcome_steps", "ppo_rollouts",
    "minibatch_updates", "storage_action_count", "parent_write_count",
    "eviction_count", "slot_allocation_count", "value_model_updates",
    "package_serialization_cpu_s", "total_cpu_s", "wall_s", "utility_cpu_s",
    "model_update_cpu_s",
)


def _finite(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    number = _finite(value, name)
    assert number is not None
    if number < 0 or not number.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    if path is None or not path.is_file():
        return [], f"missing JSONL log: {path}"
    result = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return [], f"{path}:{line_number} is not a JSON object"
                result.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL log: {path}") from exc
    return result, None


def _read_raw_steps(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """Stream over the per-step log, retaining only reward and row totals."""

    if path is None or not path.is_file():
        return None, f"missing raw step log: {path}"
    suffix = path.suffix.casefold()
    total_steps = 0
    total_reward = 0.0
    outcome_steps = 0
    outcome_reward = 0.0
    phase_steps: dict[str, int] = {}
    phase_rewards: dict[str, float] = {}
    try:
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not {"phase", "reward"} <= set(reader.fieldnames):
                    return None, f"raw step CSV must contain phase and reward columns: {path}"
                for line_number, row in enumerate(reader, start=2):
                    phase = row.get("phase")
                    reward = _finite(row.get("reward"), f"{path}:{line_number}.reward")
                    assert reward is not None
                    if not isinstance(phase, str) or not phase:
                        return None, f"raw step row {line_number} has no phase"
                    total_steps += 1
                    total_reward += reward
                    phase_steps[phase] = phase_steps.get(phase, 0) + 1
                    phase_rewards[phase] = phase_rewards.get(phase, 0.0) + reward
                    if phase == "outcome":
                        outcome_steps += 1
                        outcome_reward += reward
        elif suffix in (".jsonl", ".json"):
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        return None, f"raw step row {line_number} is not an object"
                    phase = row.get("phase")
                    reward = _finite(row.get("reward"), f"{path}:{line_number}.reward")
                    assert reward is not None
                    if not isinstance(phase, str) or not phase:
                        return None, f"raw step row {line_number} has no phase"
                    total_steps += 1
                    total_reward += reward
                    phase_steps[phase] = phase_steps.get(phase, 0) + 1
                    phase_rewards[phase] = phase_rewards.get(phase, 0.0) + reward
                    if phase == "outcome":
                        outcome_steps += 1
                        outcome_reward += reward
        else:
            return None, f"unsupported raw step log format: {path.suffix}"
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read raw step log: {path}") from exc

    whole_rate = total_reward / total_steps if total_steps else None
    outcome_rate = outcome_reward / outcome_steps if outcome_steps else None
    return {
        "actual_step_rows": int(total_steps),
        "whole_stream_reward_sum": float(total_reward),
        "whole_stream_steps": int(total_steps),
        "whole_stream_return_per_step": whole_rate,
        "postwarmup_reward_sum": float(outcome_reward),
        "postwarmup_steps": int(outcome_steps),
        "postwarmup_return_per_step": outcome_rate,
        "phase_steps": phase_steps,
        "phase_reward_sums": phase_rewards,
        "phases_valid": set(phase_steps) <= {"warmup", "outcome"},
    }, None


def _bout_return_metrics(bouts: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Recompute complete-stream returns from the runner's compact bout log."""

    phase_steps = {"warmup": 0, "outcome": 0}
    phase_rewards = {"warmup": 0.0, "outcome": 0.0}
    for index, bout in enumerate(bouts):
        phase = bout.get("phase")
        if phase not in phase_steps:
            return None
        steps = _integer(bout.get("steps_completed"), f"bouts[{index}].steps_completed", allow_none=True)
        reward = _finite(bout.get("return_full"), f"bouts[{index}].return_full", allow_none=True)
        if steps is None or steps <= 0 or reward is None:
            return None
        phase_steps[phase] += steps
        phase_rewards[phase] += reward
    whole_steps = sum(phase_steps.values())
    whole_reward = sum(phase_rewards.values())
    return {
        "whole_stream_reward_sum": float(whole_reward),
        "whole_stream_steps": int(whole_steps),
        "whole_stream_return_per_step": float(whole_reward / whole_steps) if whole_steps else None,
        "postwarmup_reward_sum": float(phase_rewards["outcome"]),
        "postwarmup_steps": int(phase_steps["outcome"]),
        "postwarmup_return_per_step": (
            float(phase_rewards["outcome"] / phase_steps["outcome"]) if phase_steps["outcome"] else None
        ),
        "phase_steps": phase_steps,
        "phase_reward_sums": phase_rewards,
        "metrics_source": "bouts.jsonl",
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON input contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return str(value)


def _sum_known(values: Sequence[int | float | None]) -> int | float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    result = sum(present)
    return int(result) if all(isinstance(value, int) for value in present) else float(result)


def _sum_complete(values: Sequence[int | float | None]) -> int | float | None:
    if not values or any(value is None for value in values):
        return None
    result = sum(values)
    return int(result) if all(isinstance(value, int) for value in values) else float(result)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _resolve_run_path(run_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (run_root / path).resolve()


def _load_config(run_root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    embedded = summary.get("config")
    if isinstance(embedded, Mapping):
        return dict(embedded)
    frozen_path = summary.get("frozen_config_path")
    config_path = _resolve_run_path(run_root, frozen_path)
    if config_path is None or not config_path.is_file():
        raise ValueError("summary must contain config or an existing frozen_config_path")
    frozen = _read_json(config_path)
    # The runner snapshots a wrapper containing the exact config and provenance.
    # Accept a bare config as well, which is useful for older/local fixtures.
    nested = frozen.get("config")
    return dict(nested) if isinstance(nested, Mapping) else frozen


def _stage_design(config: Mapping[str, Any], stage: str, pilot: bool) -> dict[str, Any]:
    raw = config.get(stage)
    if not isinstance(raw, Mapping):
        raise ValueError(f"frozen config.{stage} must be an object")
    design = dict(raw)
    if pilot:
        pilot_config = design.get("pilot")
        if not isinstance(pilot_config, Mapping):
            raise ValueError(f"frozen config.{stage}.pilot must be an object for pilot input")
        for key, value in pilot_config.items():
            design[key] = value
    if not isinstance(design.get("seeds"), list) or not design["seeds"]:
        raise ValueError(f"frozen config.{stage}.seeds must be nonempty")
    if "pair_kinds" not in design:
        design["pair_kinds"] = config.get("pair_kinds", [])
    if "methods" not in design:
        design["methods"] = raw.get("methods", [])
    for key in ("pair_kinds", "methods"):
        if not isinstance(design.get(key), list) or not design[key]:
            raise ValueError(f"frozen config.{stage}.{key} must be nonempty")
    decisions = _integer(design.get("decisions"), f"frozen config.{stage}.decisions")
    if decisions is None or decisions <= 0:
        raise ValueError(f"frozen config.{stage}.decisions must be positive")
    return {
        "seeds": [int(_integer(seed, f"frozen config.{stage}.seeds[]")) for seed in design["seeds"]],
        "pair_kinds": [str(kind) for kind in design["pair_kinds"]],
        "methods": [str(method) for method in design["methods"]],
        "decisions": int(decisions),
    }


def _read_run(input_path: str | Path, expected_stage: str) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    source = Path(input_path)
    summary_path = source / "summary.json" if source.is_dir() else source
    if summary_path.name != "summary.json" or not summary_path.is_file():
        raise FileNotFoundError(f"expected summary.json under {source}")
    summary = _read_json(summary_path)
    if summary.get("format") != SUMMARY_FORMAT:
        raise ValueError(f"input format must be {SUMMARY_FORMAT}")
    stage = summary.get("stage")
    if stage != expected_stage:
        raise ValueError(f"input stage must be {expected_stage}, got {stage!r}")
    pilot = summary.get("pilot")
    if not isinstance(pilot, bool):
        raise ValueError("summary.pilot must be boolean")
    run_root = summary_path.parent
    frozen_path = _resolve_run_path(run_root, summary.get("frozen_config_path"))
    frozen_document = _read_json(frozen_path) if frozen_path is not None and frozen_path.is_file() else {}
    summary_protocol_sha256 = summary.get("protocol_sha256")
    frozen_protocol_sha256 = frozen_document.get("protocol_sha256")
    config = _load_config(run_root, summary)
    design = _stage_design(config, expected_stage, pilot)
    protocol_hash_match = (
        isinstance(summary_protocol_sha256, str)
        and bool(summary_protocol_sha256)
        and isinstance(frozen_protocol_sha256, str)
        and bool(frozen_protocol_sha256)
        and summary_protocol_sha256 == frozen_protocol_sha256
    )
    return summary, config, run_root, {
        **design,
        "pilot": pilot,
        "summary_path": str(summary_path.resolve()),
        "summary_protocol_sha256": summary_protocol_sha256,
        "frozen_protocol_sha256": frozen_protocol_sha256,
        "protocol_hash_match": protocol_hash_match,
    }


def _action_name(value: Any, context: str, *, allow_none: bool = True) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or value not in ACTIONS:
        raise ValueError(f"{context} must be one of {ACTIONS}" + (" or null" if allow_none else ""))
    return value


def _recurrence_summary(bouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evicted_tasks: set[str] = set()
    recurrence_rates: list[float] = []
    after_eviction_rates: list[float] = []
    recurrence_bout_count = 0
    after_eviction_bout_count = 0
    for bout in bouts:
        task = bout.get("task")
        if not isinstance(task, str):
            task = None
        details = bout.get("recurrence")
        details = details if isinstance(details, Mapping) else {}
        recurrence_flag = details.get("recurrence")
        if not isinstance(recurrence_flag, bool):
            recurrence_flag = bool(details.get("seen_before") and details.get("allocated"))
        steps = _integer(bout.get("steps_completed"), "bout.steps_completed", allow_none=True)
        first_total = _finite(bout.get("return_first_256"), "bout.return_first_256", allow_none=True)
        rate = None
        if steps is not None and steps >= 256 and first_total is not None:
            rate = float(first_total / 256.0)
        after_eviction = bool(task is not None and task in evicted_tasks and recurrence_flag)
        if recurrence_flag:
            recurrence_bout_count += 1
            if rate is not None:
                recurrence_rates.append(rate)
        if after_eviction:
            after_eviction_bout_count += 1
            if rate is not None:
                after_eviction_rates.append(rate)

        victim = bout.get("evicted_task")
        if not isinstance(victim, str):
            victim = details.get("evicted_task")
        if isinstance(victim, str) and victim:
            evicted_tasks.add(victim)

    return {
        "recurrence_bout_count": int(recurrence_bout_count),
        "recurrence_first256_valid_bout_count": len(recurrence_rates),
        "recurrence_first256_mean_reward_per_step": _mean(recurrence_rates),
        "after_eviction_bout_count": int(after_eviction_bout_count),
        "after_eviction_first256_valid_bout_count": len(after_eviction_rates),
        "after_eviction_first256_mean_reward_per_step": _mean(after_eviction_rates),
        "interpretation": "factual natural-recurrence outcomes; not causal recovery loss",
    }


def _prequential_rows(
    stream: Mapping[str, Any],
    preactions: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
    *,
    tail_start: int,
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    reasons: list[str] = []
    pre_by_index: dict[int, Mapping[str, Any]] = {}
    feedback_by_index: dict[int, Mapping[str, Any]] = {}
    for prefix, rows, target in (
        ("preactions", preactions, pre_by_index), ("feedback", feedback, feedback_by_index)
    ):
        for offset, row in enumerate(rows):
            index = _integer(row.get("decision_index"), f"{prefix}[{offset}].decision_index")
            if index is None:
                reasons.append(f"{stream['stream_id']}: {prefix} row has no decision_index")
                continue
            if index in target:
                reasons.append(f"{stream['stream_id']}: duplicate {prefix} decision_index {index}")
            else:
                target[index] = row

    output_rows: list[dict[str, Any]] = []
    squared: dict[str, list[float]] = {"linear": [], "intercept": []}
    biases: dict[str, list[float]] = {"linear": [], "intercept": []}
    action_support: dict[str, int] = {action: 0 for action in ACTIONS}
    tail_support: dict[str, int] = {action: 0 for action in ACTIONS}
    matched_count = 0
    tail_count = 0
    mismatched = []
    expected = set(range(int(stream["requested_decisions"])))

    for index in sorted(set(pre_by_index) | set(feedback_by_index)):
        pre = pre_by_index.get(index)
        observed = feedback_by_index.get(index)
        if index not in expected:
            reasons.append(f"{stream['stream_id']}: unexpected decision_index {index}")
        is_tail = int(stream["requested_decisions"]) >= TAIL_DECISIONS and index >= tail_start
        predicted_action = _action_name(
            pre.get("selected_action") if pre is not None else None,
            f"{stream['stream_id']}.preactions[{index}].selected_action",
        )
        actual_action = _action_name(
            observed.get("selected_action") if observed is not None else None,
            f"{stream['stream_id']}.feedback[{index}].selected_action",
        )
        action = actual_action if actual_action is not None else predicted_action
        target = _finite(
            observed.get("utility_target") if observed is not None else None,
            f"{stream['stream_id']}.feedback[{index}].utility_target",
            allow_none=True,
        )
        predictions = pre.get("predictions") if pre is not None else None
        predictions = predictions if isinstance(predictions, Mapping) else {}
        selected_predictions: dict[str, float | None] = {}
        errors: dict[str, float | None] = {}
        if pre is not None and observed is not None and predicted_action != actual_action:
            mismatched.append(index)
        if pre is not None and observed is not None and predicted_action == actual_action and target is not None:
            matched_count += 1
            if action in action_support:
                action_support[action] += 1
                if is_tail:
                    tail_support[action] += 1
            if is_tail:
                tail_count += 1
        for model_name in ("linear", "intercept"):
            model_values = predictions.get(model_name)
            value = None
            if isinstance(model_values, Mapping) and predicted_action is not None:
                raw_prediction = model_values.get(predicted_action)
                value = _finite(raw_prediction, f"{stream['stream_id']}.predictions.{model_name}.{predicted_action}", allow_none=True)
            selected_predictions[model_name] = value
            if pre is not None and observed is not None and predicted_action == actual_action and target is not None and value is not None:
                error = float(value - target)
                if is_tail:
                    squared[model_name].append(error * error)
                    biases[model_name].append(error)
                errors[model_name] = error * error
            else:
                errors[model_name] = None
        status = "matched" if pre is not None and observed is not None and predicted_action == actual_action and target is not None else (
            "unmatched_preaction" if pre is not None and observed is None else
            "unmatched_feedback" if pre is None and observed is not None else
            "action_mismatch" if pre is not None and observed is not None else
            "missing_label"
        )
        if stage == "calibration":
            output_rows.append({
                "stage": stage,
                "stream_id": stream["stream_id"],
                "seed": stream["seed"],
                "pair_kind": stream["pair_kind"],
                "decision_index": index,
                "tail_final_32_of_64": bool(is_tail),
                "status": status,
                "selected_action": action,
                "action_propensity": None if pre is None else pre.get("action_propensity"),
                "features": None if pre is None else json.dumps(pre.get("features"), separators=(",", ":"), ensure_ascii=False),
                "linear_selected_prediction": selected_predictions["linear"],
                "intercept_selected_prediction": selected_predictions["intercept"],
                "utility_target": target,
                "linear_squared_error": errors["linear"],
                "intercept_squared_error": errors["intercept"],
                "utility_cpu_s": None if observed is None else observed.get("utility_cpu_s"),
                "model_update_cpu_s": None if observed is None else observed.get("model_update_cpu_s"),
            })

    for index in sorted(expected - set(pre_by_index)):
        reasons.append(f"{stream['stream_id']}: missing preaction decision_index {index}")
    for index in sorted(expected - set(feedback_by_index)):
        reasons.append(f"{stream['stream_id']}: missing feedback decision_index {index}")
    for index in mismatched:
        reasons.append(f"{stream['stream_id']}: preaction/feedback selected_action mismatch at decision {index}")

    tail_expected_count = TAIL_DECISIONS
    tail_complete = (
        int(stream["requested_decisions"]) >= TAIL_DECISIONS
        and tail_expected_count == TAIL_DECISIONS
        and all(len(squared[model]) == TAIL_DECISIONS for model in squared)
    )
    mse = {
        model: float(np.mean(squared[model])) if squared[model] else None
        for model in ("linear", "intercept")
    }
    bias = {
        model: float(np.mean(biases[model])) if biases[model] else None
        for model in ("linear", "intercept")
    }
    stats = {
        "matched_decisions": int(matched_count),
        "tail_expected_decisions": int(tail_expected_count),
        "tail_observed_decisions": int(tail_count),
        "tail_complete": bool(tail_complete),
        "tail_mse": mse,
        "tail_prediction_bias": bias,
        "action_support_counts": action_support,
        "tail_action_support_counts": tail_support,
    }
    if stage == "calibration" and not tail_complete:
        reasons.append(f"{stream['stream_id']}: final-32-of-64 selected-label tail is incomplete")
    return output_rows, stats, reasons


def _stream_record(
    row: Any,
    index: int,
    *,
    run_root: Path,
    stage: str,
    design: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    if not isinstance(row, Mapping):
        raise ValueError(f"streams[{index}] must be an object")
    stream_id = str(row.get("stream_id", f"stream-{index}"))
    method = row.get("method")
    pair_kind = row.get("pair_kind")
    seed_value = _integer(row.get("seed"), f"streams[{index}].seed")
    if seed_value is None:
        raise ValueError(f"streams[{index}].seed is required")
    if not isinstance(method, str) or not method:
        raise ValueError(f"streams[{index}].method is required")
    if not isinstance(pair_kind, str) or not pair_kind:
        raise ValueError(f"streams[{index}].pair_kind is required")
    status = str(row.get("status", "unknown"))
    requested = _integer(row.get("requested_decisions"), f"streams[{index}].requested_decisions", allow_none=True)
    completed = _integer(row.get("completed_decisions"), f"streams[{index}].completed_decisions", allow_none=True)
    requested = int(requested if requested is not None else design["decisions"])
    costs_value = row.get("costs")
    costs_value = costs_value if isinstance(costs_value, Mapping) else {}
    costs = {key: _json_safe(costs_value.get(key)) for key in _COST_FIELDS}
    cost_action_count = _json_safe(costs_value.get("action_count"))
    package_fields = {}
    for key in ("peak_package_bytes", "actor_container_bytes", "actual_actor_bytes", "model_state_bytes"):
        package_fields[key] = _integer(row.get(key), f"streams[{index}].{key}", allow_none=True)

    paths = {
        name: _resolve_run_path(run_root, row.get(f"{name}_path"))
        for name in ("summary", "preactions", "feedback", "rawsteps", "rollouts")
    }
    stream_summary = None
    if paths["summary"] is not None and paths["summary"].is_file():
        stream_summary = _read_json(paths["summary"])
    stream_dir = paths["summary"].parent if paths["summary"] is not None else (
        paths["preactions"].parent if paths["preactions"] is not None else run_root
    )
    bouts_path_value = row.get("bouts_path")
    bouts_path = _resolve_run_path(run_root, bouts_path_value) if bouts_path_value else stream_dir / "bouts.jsonl"

    preactions, preaction_error = _read_jsonl(paths["preactions"])
    feedback, feedback_error = _read_jsonl(paths["feedback"])
    bouts, bouts_error = _read_jsonl(bouts_path)
    raw, raw_error = _read_raw_steps(paths["rawsteps"])
    reasons: list[str] = []
    for error in (preaction_error, feedback_error, raw_error):
        if error:
            reasons.append(f"{stream_id}: {error}")
    if stream_summary is None:
        reasons.append(f"{stream_id}: missing stream summary JSON")
    if bouts_error:
        reasons.append(f"{stream_id}: {bouts_error}")
    if status != "completed":
        reasons.append(f"{stream_id}: runner status is {status}")
    if completed != requested:
        reasons.append(f"{stream_id}: completed_decisions {completed} does not equal requested_decisions {requested}")

    expected_steps = (
        2 * int(config.get("bootstrap_steps", 1024))
        + 2 * int(config.get("bout_steps", 512))
        + requested * int(config.get("outcome_bouts", 3)) * int(config.get("bout_steps", 512))
    )
    if raw is not None:
        if not raw["phases_valid"]:
            reasons.append(f"{stream_id}: raw steps contain an unexpected phase")
        if status == "completed" and raw["actual_step_rows"] != expected_steps:
            reasons.append(f"{stream_id}: actual raw rows {raw['actual_step_rows']} != expected {expected_steps}")
        if costs.get("environment_steps") is not None:
            reported_steps = _integer(costs["environment_steps"], f"{stream_id}.costs.environment_steps")
            if reported_steps != raw["actual_step_rows"]:
                reasons.append(f"{stream_id}: cost environment_steps disagrees with raw step rows")

    bouts_valid = status == "completed" and bouts_error is None
    bout_metrics = None
    if status == "completed":
        expected_bouts = 4 + requested * int(config.get("outcome_bouts", 3))
        if len(bouts) != expected_bouts:
            reasons.append(f"{stream_id}: bout log has {len(bouts)} rows; expected {expected_bouts}")
            bouts_valid = False
        expected_outcome_bouts = int(config.get("outcome_bouts", 3))
        expected_bout_steps = int(config.get("bout_steps", 512))
        expected_warmup_steps = int(config.get("bootstrap_steps", 1024))
        for bout_index, bout in enumerate(bouts):
            observed_index = _integer(bout.get("bout_index"), f"{stream_id}.bouts[{bout_index}].bout_index", allow_none=True)
            if observed_index != bout_index:
                reasons.append(f"{stream_id}: bout rows are missing, duplicated, or out of order at index {bout_index}")
                bouts_valid = False
            expected_phase = "warmup" if bout_index < 4 else "outcome"
            if bout.get("phase") != expected_phase:
                reasons.append(f"{stream_id}: bout {bout_index} has phase {bout.get('phase')!r}, expected {expected_phase!r}")
                bouts_valid = False
            expected_decision = None if expected_phase == "warmup" else (bout_index - 4) // expected_outcome_bouts
            if bout.get("decision_index") != expected_decision:
                reasons.append(f"{stream_id}: bout {bout_index} has an incorrect decision_index")
                bouts_valid = False
            expected_steps_for_bout = expected_warmup_steps if expected_phase == "warmup" and bout_index < 2 else expected_bout_steps
            observed_steps = _integer(bout.get("steps_completed"), f"{stream_id}.bouts[{bout_index}].steps_completed", allow_none=True)
            if observed_steps != expected_steps_for_bout:
                reasons.append(f"{stream_id}: bout {bout_index} has {observed_steps} steps; expected {expected_steps_for_bout}")
                bouts_valid = False
        bout_metrics = _bout_return_metrics(bouts)
        if bout_metrics is None:
            reasons.append(f"{stream_id}: complete bout log lacks a valid full return or step count")
            bouts_valid = False
        if bouts_valid and raw is not None and bout_metrics is not None:
            if bout_metrics["whole_stream_steps"] != raw["actual_step_rows"]:
                reasons.append(f"{stream_id}: complete bout steps disagree with raw step rows")
                bouts_valid = False
            if not math.isclose(bout_metrics["whole_stream_reward_sum"], raw["whole_stream_reward_sum"], rel_tol=1e-8, abs_tol=1e-6):
                reasons.append(f"{stream_id}: complete bout rewards disagree with raw step rewards")
                bouts_valid = False
            if bout_metrics["postwarmup_steps"] != raw["postwarmup_steps"] or not math.isclose(
                bout_metrics["postwarmup_reward_sum"], raw["postwarmup_reward_sum"], rel_tol=1e-8, abs_tol=1e-6,
            ):
                reasons.append(f"{stream_id}: complete outcome-bout totals disagree with raw step rows")
                bouts_valid = False

    return_metrics = bout_metrics if bouts_valid and bout_metrics is not None else raw

    lambda_cpu = float(config.get("value_model", {}).get("lambda_cpu", 0.0))
    utility_cpu_values: list[float] = []
    model_update_cpu_values: list[float] = []
    utility_penalties: list[float] = []
    utility_mismatches = []
    action_support = {action: 0 for action in ACTIONS}
    action_support["no_retention_operation"] = 0
    expected_block_steps = int(config.get("outcome_bouts", 3)) * int(config.get("bout_steps", 512))
    for feedback_index, item in enumerate(feedback):
        action = _action_name(
            item.get("selected_action"), f"{stream_id}.feedback[{feedback_index}].selected_action"
        )
        action_support[action if action is not None else "no_retention_operation"] += 1
        utility_cpu = _finite(item.get("utility_cpu_s"), f"{stream_id}.feedback[{feedback_index}].utility_cpu_s", allow_none=True)
        update_cpu = _finite(item.get("model_update_cpu_s"), f"{stream_id}.feedback[{feedback_index}].model_update_cpu_s", allow_none=True)
        if utility_cpu is not None:
            utility_cpu_values.append(utility_cpu)
        if update_cpu is not None:
            model_update_cpu_values.append(update_cpu)
        steps = _integer(item.get("block_steps"), f"{stream_id}.feedback[{feedback_index}].block_steps", allow_none=True)
        reward_rate = _finite(item.get("reward_rate"), f"{stream_id}.feedback[{feedback_index}].reward_rate", allow_none=True)
        target = _finite(item.get("utility_target"), f"{stream_id}.feedback[{feedback_index}].utility_target", allow_none=True)
        if steps != expected_block_steps:
            reasons.append(f"{stream_id}: feedback {feedback_index} has {steps} block steps; expected {expected_block_steps}")
        if item.get("target_excludes_model_update") is not True:
            reasons.append(f"{stream_id}: feedback {feedback_index} does not exclude model-update CPU from its target")
        if item.get("model_updated") is not (action is not None):
            reasons.append(f"{stream_id}: feedback {feedback_index} model_updated flag disagrees with its selected action")
        if utility_cpu is not None and steps and reward_rate is not None:
            if utility_cpu < 0:
                utility_mismatches.append(feedback_index)
            penalty = lambda_cpu * utility_cpu / steps
            utility_penalties.append(penalty)
            if target is not None and not math.isclose(target, reward_rate - penalty, rel_tol=1e-6, abs_tol=1e-8):
                utility_mismatches.append(feedback_index)
        if action is not None and (steps is None or steps <= 0 or reward_rate is None or target is None or utility_cpu is None):
            utility_mismatches.append(feedback_index)
    if utility_mismatches:
        reasons.append(f"{stream_id}: utility_target disagrees with reward_rate minus measured CPU term")

    reported_action_counts = row.get("storage_action_counts")
    if not isinstance(reported_action_counts, Mapping):
        reasons.append(f"{stream_id}: missing storage_action_counts")
    else:
        for action in ACTIONS:
            reported_count = _integer(reported_action_counts.get(action), f"{stream_id}.storage_action_counts.{action}", allow_none=True)
            if reported_count != action_support[action]:
                reasons.append(f"{stream_id}: logged {action} count disagrees with selected feedback rows")
    if cost_action_count is None:
        cost_action_count = _json_safe(costs_value.get("storage_action_count"))
    if cost_action_count is not None:
        reported_count = _integer(cost_action_count, f"{stream_id}.costs.storage_action_count")
        if reported_count != sum(action_support[action] for action in ACTIONS):
            reasons.append(f"{stream_id}: storage_action_count disagrees with selected feedback rows")
    if status == "completed":
        missing_cost_fields = [key for key in _COST_FIELDS if costs.get(key) is None]
        if missing_cost_fields:
            reasons.append(f"{stream_id}: completed stream lacks cost fields {missing_cost_fields}")
        for name, value in package_fields.items():
            if value is None:
                reasons.append(f"{stream_id}: completed stream lacks {name}")

    prequential_rows, prequential_stats, prequential_reasons = _prequential_rows(
        {
            "stream_id": stream_id,
            "seed": int(seed_value),
            "pair_kind": pair_kind,
            "requested_decisions": requested,
        },
        preactions,
        feedback,
        tail_start=max(0, requested - TAIL_DECISIONS),
        stage=stage,
    )
    reasons.extend(prequential_reasons)
    recurrence = _recurrence_summary(bouts) if stage == "comparison" else None

    cap_bytes = _integer(config.get("package_budget_bytes"), "config.package_budget_bytes", allow_none=True)
    if package_fields["peak_package_bytes"] is not None and cap_bytes is not None:
        if package_fields["peak_package_bytes"] > cap_bytes:
            reasons.append(f"{stream_id}: peak package exceeds the frozen package cap")

    record = {
        "stream_id": stream_id,
        "stage": stage,
        "method": method,
        "seed": int(seed_value),
        "pair_kind": pair_kind,
        "status": status,
        "requested_decisions": requested,
        "completed_decisions": completed,
        "failure": _json_safe(row.get("failure")),
        "log_paths": {name: None if path is None else str(path) for name, path in paths.items()},
        "raw_step_metrics": raw,
        "return_metrics": return_metrics,
        "return_metrics_source": None if return_metrics is None else return_metrics.get("metrics_source", "rawsteps"),
        "action_support_counts": action_support,
        "logged_storage_action_counts": _json_safe(row.get("storage_action_counts")),
        "logged_cost_action_count": cost_action_count,
        "prequential": prequential_stats,
        "utility_cost": {
            "lambda_cpu": lambda_cpu,
            "utility_cpu_s_observed": _sum_known(utility_cpu_values),
            "utility_cpu_s_count": len(utility_cpu_values),
            "model_update_cpu_s_observed": _sum_known(model_update_cpu_values),
            "model_update_cpu_s_count": len(model_update_cpu_values),
            "mean_cpu_deduction_per_step": _mean(utility_penalties),
            "sum_blockwise_cpu_deductions_per_step": float(sum(utility_penalties)) if utility_penalties else None,
            "utility_target_mismatch_count": len(utility_mismatches),
        },
        "recurrence": recurrence,
        "costs": costs,
        "peak_package_bytes": package_fields["peak_package_bytes"],
        "package_budget_bytes": cap_bytes,
        "actor_container_bytes": package_fields["actor_container_bytes"],
        "actual_actor_bytes": package_fields["actual_actor_bytes"],
        "model_state_bytes": package_fields["model_state_bytes"],
        "stream_summary": None if stream_summary is None else _json_safe(stream_summary),
        "complete": status == "completed" and not reasons,
        "incomplete_reasons": list(dict.fromkeys(reasons)),
    }
    return record, prequential_rows, reasons


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    positive = sum(value > 1e-12 for value in values)
    negative = sum(value < -1e-12 for value in values)
    return {
        "positive": int(positive),
        "negative": int(negative),
        "tie": int(len(values) - positive - negative),
        "denominator": int(len(values)),
    }


def _bootstrap_summary(values: Sequence[float], *, resamples: int, seed: int, allowed: bool) -> dict[str, Any] | None:
    if not allowed or not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.shape[0], size=(resamples, array.shape[0]))
    draws = array[indices].mean(axis=1)
    bounds = np.quantile(draws, [0.025, 0.975])
    return {
        "n": int(array.shape[0]),
        "mean": float(array.mean()),
        "ci95": [float(bounds[0]), float(bounds[1])],
        **_sign_counts(values),
        "interval_interpretation": "descriptive seed bootstrap; exploratory cohort",
    }


def _aggregate_costs(
    streams: Sequence[Mapping[str, Any]],
    runner_totals: Any,
    cap: int | None,
    cohort_costs: Any = None,
) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for field in _COST_FIELDS:
        values = []
        missing = 0
        for stream in streams:
            raw = stream["costs"].get(field)
            if raw is None or isinstance(raw, Mapping):
                missing += 1
                continue
            number = _finite(raw, f"costs.{field}")
            assert number is not None
            values.append(number)
        if field in (
            "environment_steps", "warmup_steps", "outcome_steps", "ppo_rollouts",
            "minibatch_updates", "storage_action_count", "parent_write_count",
            "eviction_count", "slot_allocation_count", "value_model_updates",
        ):
            observed = int(sum(values)) if values and all(float(value).is_integer() for value in values) else (sum(values) if values else None)
        else:
            observed = float(sum(values)) if values else None
        report_field = {
            "total_cpu_s": "aggregate_stream_cpu_s",
            "wall_s": "aggregate_stream_wall_s",
        }.get(field, field)
        totals[report_field] = {"sum": observed if missing == 0 else None, "observed_subtotal": observed, "missing_stream_count": missing}

    raw_rows = [stream["raw_step_metrics"]["actual_step_rows"] if stream["raw_step_metrics"] is not None else None for stream in streams]
    reported_steps = [stream["costs"].get("environment_steps") for stream in streams]
    totals.update({
        "actual_step_rows": _sum_known(raw_rows),
        "actual_step_rows_complete": bool(streams) and all(value is not None for value in raw_rows),
        "raw_rows_match_reported_environment_steps": bool(streams)
        and all(raw is not None and reported is not None and int(raw) == int(reported) for raw, reported in zip(raw_rows, reported_steps)),
        "peak_package_bytes_max": max(
            (int(stream["peak_package_bytes"]) for stream in streams if stream["peak_package_bytes"] is not None),
            default=None,
        ),
        "package_budget_bytes": cap,
        "package_cap_passed": bool(streams)
        and all(stream["peak_package_bytes"] is not None and cap is not None and stream["peak_package_bytes"] <= cap for stream in streams),
        "model_state_bytes_max": max(
            (int(stream["model_state_bytes"]) for stream in streams if stream["model_state_bytes"] is not None),
            default=None,
        ),
        "actor_container_bytes": sorted({stream["actor_container_bytes"] for stream in streams if stream["actor_container_bytes"] is not None}),
        "actual_actor_bytes_max": max(
            (int(stream["actual_actor_bytes"]) for stream in streams if stream["actual_actor_bytes"] is not None),
            default=None,
        ),
        "campaign_wall_s": None,
        "campaign_total_cpu_s": None,
        "campaign_child_cpu_s": None,
        "campaign_orchestrator_cpu_s": None,
        "workers_actual": None,
    })
    if isinstance(cohort_costs, Mapping):
        totals["campaign_wall_s"] = _finite(cohort_costs.get("wall_s"), "cohort_costs.wall_s", allow_none=True)
        totals["campaign_total_cpu_s"] = _finite(cohort_costs.get("total_cpu_s"), "cohort_costs.total_cpu_s", allow_none=True)
        totals["campaign_child_cpu_s"] = _finite(cohort_costs.get("child_cpu_s"), "cohort_costs.child_cpu_s", allow_none=True)
        totals["campaign_orchestrator_cpu_s"] = _finite(
            cohort_costs.get("orchestrator_cpu_s"), "cohort_costs.orchestrator_cpu_s", allow_none=True,
        )
        totals["workers_actual"] = _integer(cohort_costs.get("workers_actual"), "cohort_costs.workers_actual", allow_none=True)
    reported_totals = dict(runner_totals) if isinstance(runner_totals, Mapping) else runner_totals
    if isinstance(reported_totals, dict):
        # Legacy runner totals are sums over child streams, not campaign elapsed time.
        for source, target in (("wall_s", "aggregate_stream_wall_s"), ("total_cpu_s", "aggregate_stream_cpu_s")):
            if source in reported_totals:
                reported_totals[target] = reported_totals.pop(source)
    totals["runner_totals_reported"] = _json_safe(reported_totals)
    totals["campaign_cost_source"] = "cohort_costs" if isinstance(cohort_costs, Mapping) else "unknown_not_logged"
    return totals


def _calibration_analysis(
    streams: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    *,
    resamples: int,
    seed: int,
    cohort_complete: bool,
) -> dict[str, Any]:
    expected_seeds = list(design["seeds"])
    relations = list(design["pair_kinds"])
    by_key = {(stream["seed"], stream["pair_kind"]): stream for stream in streams if stream["method"] == "random"}
    expected_keys = {(seed_value, relation) for seed_value in expected_seeds for relation in relations}
    tail_complete = cohort_complete and all(
        by_key.get(key) is not None and by_key[key]["prequential"]["tail_complete"] for key in expected_keys
    )
    per_seed = []
    for seed_value in expected_seeds:
        relation_metrics: dict[str, Any] = {}
        deltas = []
        linear_errors = []
        intercept_errors = []
        for relation in relations:
            stream = by_key.get((seed_value, relation))
            stats = None if stream is None else stream["prequential"]
            mse = None if stats is None else stats["tail_mse"]
            difference = None
            if stats is not None and stats["tail_complete"]:
                linear_errors.append(float(mse["linear"]))
                intercept_errors.append(float(mse["intercept"]))
                difference = float(mse["linear"] - mse["intercept"])
                deltas.append(difference)
            relation_metrics[relation] = {
                "linear_mse": None if mse is None else mse["linear"],
                "intercept_mse": None if mse is None else mse["intercept"],
                "linear_minus_intercept_mse": difference,
                "tail_complete": bool(stats and stats["tail_complete"]),
            }
        per_seed.append({
            "seed": seed_value,
            "relation_count": len(relations),
            "observed_relation_count": len(deltas),
            "linear_mse_mean_over_relations": _mean(linear_errors) if len(linear_errors) == len(relations) else None,
            "intercept_mse_mean_over_relations": _mean(intercept_errors) if len(intercept_errors) == len(relations) else None,
            "linear_minus_intercept_mse_mean_over_relations": _mean(deltas) if len(deltas) == len(relations) else None,
            "by_relation": relation_metrics,
        })

    overall_values = [row["linear_minus_intercept_mse_mean_over_relations"] for row in per_seed]
    overall_values = [float(value) for value in overall_values if value is not None]
    overall = _bootstrap_summary(
        overall_values,
        resamples=resamples,
        seed=seed,
        allowed=tail_complete and len(overall_values) == len(expected_seeds) and not design["pilot"],
    )
    by_relation = {}
    for relation_index, relation in enumerate(relations):
        relation_values = [
            float(row["by_relation"][relation]["linear_minus_intercept_mse"])
            for row in per_seed
            if row["by_relation"][relation]["linear_minus_intercept_mse"] is not None
        ]
        by_relation[relation] = {
            "estimate": _bootstrap_summary(
                relation_values,
                resamples=resamples,
                seed=seed + relation_index + 1,
                allowed=tail_complete and len(relation_values) == len(expected_seeds) and not design["pilot"],
            ),
            "observed_seed_count": len(relation_values),
            "expected_seed_count": len(expected_seeds),
        }

    support = {
        "overall": {action: sum(stream["prequential"]["action_support_counts"][action] for stream in streams) for action in ACTIONS},
        "tail": {action: sum(stream["prequential"]["tail_action_support_counts"][action] for stream in streams) for action in ACTIONS},
    }
    bias_by_model = {}
    for model in ("linear", "intercept"):
        values = [
            float(stream["prequential"]["tail_prediction_bias"][model])
            for stream in streams
            if stream["prequential"]["tail_prediction_bias"][model] is not None
        ]
        bias_by_model[model] = {"observed_stream_mean": _mean(values), "stream_count": len(values)}

    return {
        "status": "pilot_descriptive_only" if design["pilot"] else ("complete_exploratory_cohort" if cohort_complete and tail_complete else "incomplete_cohort"),
        "cohort_complete": bool(cohort_complete),
        "tail_complete": bool(tail_complete),
        "expected_stream_count": len(expected_keys),
        "observed_stream_count": len(streams),
        "primary": {
            "contrast": "prequential MSE(linear) minus MSE(intercept)",
            "tail_decisions": TAIL_DECISIONS,
            "decisions_per_stream": int(design["decisions"]),
            "score_target": "feedback.utility_target for the selected action only",
            "direction": "negative favors linear prediction on selected labels",
            "estimate": overall,
            "observed_seed_count": len(overall_values),
            "expected_seed_count": len(expected_seeds),
        },
        "by_relation": by_relation,
        "per_seed": per_seed,
        "selected_action_support": support,
        "tail_prediction_bias": bias_by_model,
        "exploratory_scope": "seed bootstrap with relations averaged within seed; this measures selected-label prediction only and does not identify unchosen-action outcomes",
    }


def _comparison_analysis(
    streams: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    *,
    resamples: int,
    seed: int,
    practical_threshold: float,
    cohort_complete: bool,
) -> dict[str, Any]:
    methods = list(design["methods"])
    controls = [method for method in methods if method not in HISTORY_METHODS]
    expected_seeds = list(design["seeds"])
    relations = list(design["pair_kinds"])
    by_key = {(stream["seed"], stream["pair_kind"], stream["method"]): stream for stream in streams}
    per_seed = []
    relation_results: dict[str, Any] = {}

    for relation_index, relation in enumerate(relations):
        relation_results[relation] = {
            "expected_seed_count": len(expected_seeds),
            "primary": {},
            "all_method_vs_control": {},
        }
        for metric in METRICS:
            left_method, right_method = "learned_linear", "learned_intercept"
            diffs = []
            for seed_value in expected_seeds:
                left = by_key.get((seed_value, relation, left_method))
                right = by_key.get((seed_value, relation, right_method))
                if left is not None and right is not None:
                    left_rate = _stream_metric(left, metric)
                    right_rate = _stream_metric(right, metric)
                    if left_rate is not None and right_rate is not None:
                        diffs.append(float(left_rate - right_rate))
            estimate = _bootstrap_summary(
                diffs,
                resamples=resamples,
                seed=seed + relation_index * 101 + (0 if metric == "whole_stream" else 1),
                allowed=cohort_complete and len(diffs) == len(expected_seeds) and not design["pilot"],
            )
            thresholds = {}
            for threshold_name, threshold_value in (
                ("zero", 0.0), ("practical_reward_per_step", practical_threshold),
            ):
                thresholds[threshold_name] = {
                    "threshold": float(threshold_value),
                    "mean_minus_threshold": None if estimate is None else estimate["mean"] - threshold_value,
                    "ci95_lower_above_threshold": None if estimate is None else estimate["ci95"][0] > threshold_value,
                }
            relation_results[relation]["primary"][metric] = {
                "estimate": estimate,
                "thresholds": thresholds,
                "observed_seed_count": len(diffs),
                "expected_seed_count": len(expected_seeds),
            }

        for method in methods:
            relation_results[relation]["all_method_vs_control"][method] = {}
            for control in controls:
                if method == control:
                    continue
                relation_results[relation]["all_method_vs_control"][method][control] = {}
                for metric_index, metric in enumerate(METRICS):
                    diffs = []
                    for seed_value in expected_seeds:
                        left = by_key.get((seed_value, relation, method))
                        right = by_key.get((seed_value, relation, control))
                        if left is None or right is None:
                            continue
                        left_rate = _stream_metric(left, metric)
                        right_rate = _stream_metric(right, metric)
                        if left_rate is not None and right_rate is not None:
                            diffs.append(float(left_rate - right_rate))
                    relation_results[relation]["all_method_vs_control"][method][control][metric] = {
                        "estimate": _bootstrap_summary(
                            diffs,
                            resamples=resamples,
                            seed=seed + relation_index * 101 + metric_index,
                            allowed=cohort_complete and len(diffs) == len(expected_seeds) and not design["pilot"],
                        ),
                        "observed_seed_count": len(diffs),
                        "expected_seed_count": len(expected_seeds),
                    }

    for seed_value in expected_seeds:
        for relation in relations:
            seed_streams = [by_key.get((seed_value, relation, method)) for method in methods]
            row: dict[str, Any] = {
                "stage": "comparison",
                "seed": seed_value,
                "pair_kind": relation,
                "row_level": "seed_relation",
                "status": "complete" if all(stream is not None and stream["complete"] for stream in seed_streams) else "incomplete",
                "stream_count": sum(stream is not None for stream in seed_streams),
                "expected_stream_count": len(methods),
            }
            for method in methods:
                stream = by_key.get((seed_value, relation, method))
                row[f"{method}_whole_stream"] = None if stream is None else _stream_metric(stream, "whole_stream")
                row[f"{method}_postwarmup"] = None if stream is None else _stream_metric(stream, "postwarmup")
            for method in methods:
                for control in controls:
                    if method == control:
                        continue
                    left = by_key.get((seed_value, relation, method))
                    right = by_key.get((seed_value, relation, control))
                    for metric in METRICS:
                        left_rate = None if left is None else _stream_metric(left, metric)
                        right_rate = None if right is None else _stream_metric(right, metric)
                        row[f"{method}_minus_{control}_{metric}"] = (
                            None if left_rate is None or right_rate is None else float(left_rate - right_rate)
                        )
            pair_counts = _sum_known([
                stream["raw_step_metrics"]["actual_step_rows"]
                for stream in seed_streams if stream is not None and stream["raw_step_metrics"] is not None
            ])
            row["actual_step_rows"] = pair_counts
            row["failed_stream_count"] = sum(stream is not None and not stream["complete"] for stream in seed_streams)
            per_seed.append(row)

    method_aggregates = {}
    for method in methods:
        method_streams = [stream for stream in streams if stream["method"] == method]
        recurrence_rows = [stream["recurrence"] for stream in method_streams if stream["recurrence"] is not None]
        support = {action: sum(stream["action_support_counts"].get(action, 0) for stream in method_streams) for action in ACTIONS}
        support["no_retention_operation"] = sum(
            stream["action_support_counts"].get("no_retention_operation", 0) for stream in method_streams
        )
        by_method = {
            "expected_stream_count": len(expected_seeds) * len(relations),
            "observed_stream_count": len(method_streams),
            "completed_stream_count": sum(stream["status"] == "completed" for stream in method_streams),
            "whole_stream_return_per_step_observed_mean": _mean([
                float(_stream_metric(stream, "whole_stream"))
                for stream in method_streams if _stream_metric(stream, "whole_stream") is not None
            ]),
            "postwarmup_return_per_step_observed_mean": _mean([
                float(_stream_metric(stream, "postwarmup"))
                for stream in method_streams if _stream_metric(stream, "postwarmup") is not None
            ]),
            "action_support_counts": support,
            "recurrence": {
                "stream_count": len(recurrence_rows),
                "recurrence_bout_count": sum(row["recurrence_bout_count"] for row in recurrence_rows),
                "recurrence_first256_mean_of_stream_means": _mean([
                    float(row["recurrence_first256_mean_reward_per_step"])
                    for row in recurrence_rows if row["recurrence_first256_mean_reward_per_step"] is not None
                ]),
                "after_eviction_bout_count": sum(row["after_eviction_bout_count"] for row in recurrence_rows),
                "after_eviction_first256_mean_of_stream_means": _mean([
                    float(row["after_eviction_first256_mean_reward_per_step"])
                    for row in recurrence_rows if row["after_eviction_first256_mean_reward_per_step"] is not None
                ]),
                "interpretation": "factual natural-recurrence outcomes; not causal recovery loss",
            },
            "costs": _aggregate_costs(
                method_streams,
                runner_totals=None,
                cap=next((stream["package_budget_bytes"] for stream in method_streams if stream["package_budget_bytes"] is not None), None),
            ),
        }
        # Keep this compact aggregate next to the comparison analysis for all controls.
        method_aggregates[method] = by_method

    return {
        "status": "pilot_descriptive_only" if design["pilot"] else ("complete_exploratory_cohort" if cohort_complete else "incomplete_cohort"),
        "cohort_complete": bool(cohort_complete),
        "expected_stream_count": len(expected_seeds) * len(relations) * len(methods),
        "observed_stream_count": len(streams),
        "methods": methods,
        "control_methods": controls,
        "primary_method_contrast": "learned_linear minus learned_intercept",
        "practical_reward_per_step_threshold": float(practical_threshold),
        "by_relation": relation_results,
        "per_seed": per_seed,
        "by_method": method_aggregates,
        "exploration_note": "learned_linear and learned_intercept use matched 16-decision uniform warmup exploration and then the frozen epsilon-greedy schedule; deterministic keep/replace/mean/frequency/lower controls do not have the same exploration cost; random remains uniform",
        "whole_stream_note": "whole-stream reward/step includes warmup and all outcome blocks; postwarmup is outcome blocks only",
    }


def _stream_metric(stream: Mapping[str, Any], metric: str) -> float | None:
    raw = stream.get("return_metrics")
    if not isinstance(raw, Mapping):
        return None
    key = "whole_stream_return_per_step" if metric == "whole_stream" else "postwarmup_return_per_step"
    value = raw.get(key)
    return None if value is None else float(value)


def _method_per_seed_rows(
    stage: str,
    streams: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if stage == "calibration":
        by_key = {(stream["seed"], stream["pair_kind"]): stream for stream in streams}
        relations = list(design["pair_kinds"])
        for seed_value in design["seeds"]:
            relation_rows = []
            for relation in relations:
                stream = by_key.get((seed_value, relation))
                stats = None if stream is None else stream["prequential"]
                tail_mse = {} if stats is None else stats["tail_mse"]
                relation_row = {
                    "stage": stage, "seed": seed_value, "pair_kind": relation,
                    "row_level": "seed_relation",
                    "status": "complete" if stats and stats["tail_complete"] and stream["complete"] else "incomplete",
                    "linear_mse": tail_mse.get("linear"),
                    "intercept_mse": tail_mse.get("intercept"),
                    "linear_minus_intercept_mse": (
                        None if not stats or not stats["tail_complete"] else tail_mse["linear"] - tail_mse["intercept"]
                    ),
                    "action_support_counts": None if stats is None else stats["action_support_counts"],
                    "tail_action_support_counts": None if stats is None else stats["tail_action_support_counts"],
                    "actual_step_rows": None if stream is None or stream["raw_step_metrics"] is None else stream["raw_step_metrics"]["actual_step_rows"],
                }
                relation_rows.append(relation_row)
                rows.append(relation_row)
            all_complete = len(relation_rows) == len(relations) and all(row["linear_minus_intercept_mse"] is not None for row in relation_rows)
            rows.append({
                "stage": stage, "seed": seed_value, "pair_kind": "all_relations", "row_level": "seed_mean_relations",
                "status": "complete" if all_complete else "incomplete",
                "linear_mse": _mean([float(row["linear_mse"]) for row in relation_rows]) if all_complete else None,
                "intercept_mse": _mean([float(row["intercept_mse"]) for row in relation_rows]) if all_complete else None,
                "linear_minus_intercept_mse": _mean([float(row["linear_minus_intercept_mse"]) for row in relation_rows]) if all_complete else None,
                "action_support_counts": None,
                "tail_action_support_counts": None,
                "actual_step_rows": _sum_known([row["actual_step_rows"] for row in relation_rows]),
            })
        return rows

    if comparison is None:
        return rows
    by_key = {(stream["seed"], stream["pair_kind"], stream["method"]): stream for stream in streams}
    methods = list(design["methods"])
    controls = comparison["control_methods"]
    for seed_value in design["seeds"]:
        relation_rows = []
        for relation in design["pair_kinds"]:
            row: dict[str, Any] = {
                "stage": stage, "seed": seed_value, "pair_kind": relation, "row_level": "seed_relation",
            }
            relevant = [by_key.get((seed_value, relation, method)) for method in methods]
            row["status"] = "complete" if all(stream is not None and stream["complete"] for stream in relevant) else "incomplete"
            row["stream_count"] = sum(stream is not None for stream in relevant)
            row["failed_stream_count"] = sum(stream is not None and not stream["complete"] for stream in relevant)
            for method in methods:
                stream = by_key.get((seed_value, relation, method))
                row[f"{method}_whole_stream_return_per_step"] = None if stream is None else _stream_metric(stream, "whole_stream")
                row[f"{method}_postwarmup_return_per_step"] = None if stream is None else _stream_metric(stream, "postwarmup")
            for method in methods:
                for control in controls:
                    if method == control:
                        continue
                    left = by_key.get((seed_value, relation, method))
                    right = by_key.get((seed_value, relation, control))
                    for metric in METRICS:
                        left_rate = None if left is None else _stream_metric(left, metric)
                        right_rate = None if right is None else _stream_metric(right, metric)
                        row[f"{method}_minus_{control}_{metric}"] = None if left_rate is None or right_rate is None else left_rate - right_rate
            row["actual_step_rows"] = _sum_known([
                stream["raw_step_metrics"]["actual_step_rows"]
                for stream in relevant if stream is not None and stream["raw_step_metrics"] is not None
            ])
            relation_rows.append(row)
            rows.append(row)
        all_complete = len(relation_rows) == len(design["pair_kinds"]) and all(row["status"] == "complete" for row in relation_rows)
        aggregate: dict[str, Any] = {
            "stage": stage, "seed": seed_value, "pair_kind": "all_relations", "row_level": "seed_mean_relations",
            "status": "complete" if all_complete else "incomplete",
            "stream_count": sum(row["stream_count"] for row in relation_rows),
            "failed_stream_count": sum(row["failed_stream_count"] for row in relation_rows),
            "actual_step_rows": _sum_known([row["actual_step_rows"] for row in relation_rows]),
        }
        for method in methods:
            for metric in METRICS:
                key = f"{method}_{metric}_return_per_step"
                values = [row.get(key) for row in relation_rows]
                aggregate[key] = _mean([float(value) for value in values if value is not None]) if all_complete else None
        for method in methods:
            for control in controls:
                if method == control:
                    continue
                for metric in METRICS:
                    key = f"{method}_minus_{control}_{metric}"
                    values = [row.get(key) for row in relation_rows]
                    aggregate[key] = _mean([float(value) for value in values if value is not None]) if all_complete else None
        rows.append(aggregate)
    return rows


def _analysis_for_stage(
    input_path: str | Path,
    stage: str,
    *,
    resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, config, run_root, design = _read_run(input_path, stage)
    raw_streams = summary.get("streams")
    if not isinstance(raw_streams, list):
        raise ValueError("summary.streams must be a list")
    streams = []
    prequential_rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    if not design["protocol_hash_match"]:
        reasons.append(
            "root summary protocol_sha256 does not match frozen_config.json protocol_sha256"
        )
    seen_keys: set[tuple[int, str, str]] = set()
    for index, raw_stream in enumerate(raw_streams):
        stream, rows, stream_reasons = _stream_record(
            raw_stream,
            index,
            run_root=run_root,
            stage=stage,
            design=design,
            config=config,
        )
        key = (stream["seed"], stream["pair_kind"], stream["method"])
        if key in seen_keys:
            reasons.append(f"duplicate stream key {key}")
        seen_keys.add(key)
        streams.append(stream)
        prequential_rows.extend(rows)
        reasons.extend(stream_reasons)

    expected_keys = {
        (seed_value, pair_kind, method)
        for seed_value in design["seeds"]
        for pair_kind in design["pair_kinds"]
        for method in design["methods"]
    }
    for missing in sorted(expected_keys - seen_keys):
        reasons.append(f"missing expected stream {missing}")
    for extra in sorted(seen_keys - expected_keys):
        reasons.append(f"unexpected stream {extra}")
    failures = summary.get("failures")
    if failures:
        reasons.append("root summary contains failed stream records")

    # A completed main cohort must contain every expected stream and every stream log.
    reasons = list(dict.fromkeys(reasons))
    cohort_complete = not reasons and len(streams) == len(expected_keys) and all(
        stream["status"] == "completed" and stream["complete"] for stream in streams
    )
    totals = _aggregate_costs(
        streams,
        runner_totals=summary.get("totals"),
        cap=_integer(config.get("package_budget_bytes"), "config.package_budget_bytes", allow_none=True),
        cohort_costs=summary.get("cohort_costs"),
    )
    if stage == "calibration":
        detail = _calibration_analysis(
            streams,
            design,
            resamples=resamples,
            seed=bootstrap_seed,
            cohort_complete=cohort_complete,
        )
        comparison_detail = None
    else:
        detail = _comparison_analysis(
            streams,
            design,
            resamples=resamples,
            seed=bootstrap_seed,
            practical_threshold=float(_finite(
                config.get("report", {}).get("practical_reward_per_step", 0.002),
                "config.report.practical_reward_per_step",
            )),
            cohort_complete=cohort_complete,
        )
        comparison_detail = detail

    detail.update({
        "stage": stage,
        "pilot": bool(design["pilot"]),
        "protocol": {
            "summary_sha256": design["summary_protocol_sha256"],
            "frozen_start_sha256": design["frozen_protocol_sha256"],
            "hashes_match": bool(design["protocol_hash_match"]),
        },
        "expected_seeds": list(design["seeds"]),
        "expected_pair_kinds": list(design["pair_kinds"]),
        "expected_methods": list(design["methods"]),
        "decisions_per_stream": int(design["decisions"]),
        "incomplete_reasons": reasons,
        "costs": totals,
        "streams": streams,
    })
    detail["per_seed"] = _method_per_seed_rows(stage, streams, design, comparison_detail)
    return detail, prequential_rows


def _markdown(analysis: Mapping[str, Any]) -> str:
    lines = ["# Selected-Feedback Retention Stream Report", ""]
    calibration = analysis.get("calibration")
    comparison = analysis.get("comparison")
    if calibration is None:
        lines.extend(["Calibration input: not supplied.", ""])
    else:
        label = "PILOT · descriptive only" if calibration["pilot"] else calibration["status"]
        lines.extend([f"Calibration status: **{label}**.", ""])
        if not calibration["protocol"]["hashes_match"]:
            lines.extend(["Protocol audit: root summary and frozen-start protocol hashes do not match; treat this input as incomplete.", ""])
        if calibration["pilot"]:
            lines.extend(["Pilot results are plumbing and accounting observations; they do not support a confirmation conclusion.", ""])
        primary = calibration["primary"]
        estimate = primary["estimate"]
        lines.extend(["## Calibration", ""])
        if estimate is None:
            lines.extend(["No complete-cohort bootstrap estimate is reported. Per-stream observed scores, failures, and costs remain in `analysis.json` and `per_seed.csv`.", ""])
        else:
            ci = estimate["ci95"]
            lines.extend([
                f"Final 32 of 64 selected labels: mean linear − intercept prequential MSE = **{estimate['mean']:.8g}**; descriptive seed-bootstrap 95% CI [{ci[0]:.8g}, {ci[1]:.8g}] (n = {estimate['n']} seeds). Negative favors linear on observed selected labels.",
                "Relations are averaged within seed before the overall seed bootstrap; relation-specific results are also in the JSON report.",
                "",
            ])
        lines.extend([
            "Only the selected action's pre-action prediction is scored against `feedback.utility_target`. Unchosen actions have no observed label and are not scored.",
            "",
        ])

    if comparison is None:
        lines.extend(["Comparison input: not supplied.", ""])
    else:
        label = "PILOT · descriptive only" if comparison["pilot"] else comparison["status"]
        lines.extend([f"Comparison status: **{label}**.", ""])
        if not comparison["protocol"]["hashes_match"]:
            lines.extend(["Protocol audit: root summary and frozen-start protocol hashes do not match; treat this input as incomplete.", ""])
        if comparison["pilot"]:
            lines.extend(["Pilot results are not a confirmation conclusion.", ""])
        lines.extend(["## Repeated-retention comparison", ""])
        lines.extend([
            "Whole-stream return/step includes warmup and all outcome blocks. Postwarmup return/step uses outcome blocks only. The primary contrast is learned_linear − learned_intercept within seed and relation; its four-seed intervals are descriptive. The lower confidence bound is checked separately against zero and the frozen practical threshold.",
            "",
        ])
        if comparison["cohort_complete"] and not comparison["pilot"]:
            lines.extend(["| Relation | Metric | Mean learned_linear − learned_intercept | 95% seed-bootstrap CI | Positive / negative / tie seeds | Lower CI > 0 / +practical threshold |", "|---|---|---:|---:|---:|---|"])
            for relation, result in comparison["by_relation"].items():
                for metric in METRICS:
                    estimate = result["primary"][metric]["estimate"]
                    if estimate is None:
                        continue
                    ci = estimate["ci95"]
                    thresholds = result["primary"][metric]["thresholds"]
                    above_zero = "yes" if thresholds["zero"]["ci95_lower_above_threshold"] else "no"
                    practical = "yes" if thresholds["practical_reward_per_step"]["ci95_lower_above_threshold"] else "no"
                    lines.append(
                        f"| {relation} | {metric} | {estimate['mean']:.8g} | [{ci[0]:.8g}, {ci[1]:.8g}] | {estimate['positive']} / {estimate['negative']} / {estimate['tie']} of {estimate['denominator']} | {above_zero} / {practical} |"
                    )
            lines.append("")
            lines.append(f"The practical threshold is +{comparison['practical_reward_per_step_threshold']:.8g} reward per step; JSON also records the point estimate minus each threshold.")
            lines.append("")
        else:
            lines.extend(["No paired bootstrap interval is reported because the input is a pilot or does not cover the complete frozen cohort. Failed and missing streams remain listed with their recorded costs.", ""])
        lines.extend([
            "Learned_linear and learned_intercept share the first 16 uniform warmup decisions and then use the frozen epsilon-greedy schedule. Random stays uniform. Deterministic keep/replace/mean/frequency/lower controls have different exploration costs.",
            "",
            "All methods versus every control are included in `analysis.json`; stream and seed-level values are in `per_seed.csv`.",
            "",
        ])

    lines.extend(["## Cost, retention, and scope", ""])
    for stage, detail in (("calibration", calibration), ("comparison", comparison)):
        if detail is None:
            continue
        costs = detail["costs"]
        lines.append(f"### {stage}")
        lines.append("")
        lines.append("| Quantity | Total / maximum |")
        lines.append("|---|---:|")
        lines.append(f"| Actual raw step rows | {_display(costs.get('actual_step_rows'))} |")
        for field, label in (
            ("environment_steps", "Reported environment steps"),
            ("ppo_rollouts", "PPO rollouts"),
            ("minibatch_updates", "Minibatch updates"),
            ("campaign_wall_s", "Campaign elapsed wall seconds"),
            ("campaign_total_cpu_s", "Campaign total CPU seconds"),
            ("campaign_child_cpu_s", "Campaign child CPU seconds"),
            ("campaign_orchestrator_cpu_s", "Campaign orchestrator CPU seconds"),
            ("aggregate_stream_cpu_s", "Sum of per-stream CPU seconds"),
            ("aggregate_stream_wall_s", "Sum of per-stream wall seconds (not elapsed)"),
            ("utility_cpu_s", "CPU seconds charged to utility"),
            ("model_update_cpu_s", "Model update CPU seconds"),
            ("parent_write_count", "Parent writes"),
            ("storage_action_count", "Retention operations"),
            ("workers_actual", "Workers actually used"),
        ):
            value = costs.get(field, {})
            value = value.get("sum") if isinstance(value, Mapping) else value
            lines.append(f"| {label} | {_display(value)} |")
        lines.append(f"| Peak package bytes | {_display(costs.get('peak_package_bytes_max'))} / cap {_display(costs.get('package_budget_bytes'))} |")
        lines.append(f"| Maximum model state bytes | {_display(costs.get('model_state_bytes_max'))} |")
        lines.append("")

    lines.extend([
        "Comparison recurrence metrics are factual natural-recurrence first-256 reward rates, including after prior natural eviction. They are reported with bout counts and are not causal recovery loss estimates.",
        "",
        "Campaign elapsed and total CPU values are shown only when the root summary logs `cohort_costs`; legacy per-stream wall totals are labeled as sums and are not campaign elapsed time.",
        "",
        "This is selected-feedback utility prediction and a four-seed exploratory policy comparison. It does not observe unchosen-action returns, identify per-write causal benefit, or establish novelty of the ridge/discounting component.",
        "",
        f"Bootstrap settings: {analysis['bootstrap']['resamples']} resamples, seed {analysis['bootstrap']['seed']}; intervals are descriptive. Report generation itself runs no RL and adds no environment steps.",
        "",
    ])
    return "\n".join(lines)


def _csv_outputs(
    per_seed_rows: Sequence[Mapping[str, Any]],
    prequential_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    per_seed_columns: list[str] = []
    for row in per_seed_rows:
        for key in row:
            if key not in per_seed_columns:
                per_seed_columns.append(key)
    with (output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_columns or ["stage", "seed"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_seed_rows)

    prequential_columns = [
        "stage", "stream_id", "seed", "pair_kind", "decision_index", "tail_final_32_of_64", "status",
        "selected_action", "action_propensity", "features", "linear_selected_prediction",
        "intercept_selected_prediction", "utility_target", "linear_squared_error", "intercept_squared_error",
        "utility_cpu_s", "model_update_cpu_s",
    ]
    with (output_dir / "prequential.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prequential_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prequential_rows)


def _display(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, Mapping):
        value = value.get("sum")
        if value is None:
            return "unknown"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def analyze(
    *,
    calibration: str | Path | None = None,
    comparison: str | Path | None = None,
    output: str | Path,
) -> dict[str, Any]:
    """Analyze either run independently, or combine both saved run outputs."""

    if calibration is None and comparison is None:
        raise ValueError("provide at least one of calibration or comparison")
    # The frozen report config is the source of these values; if both inputs are
    # provided, verify they agree rather than silently choosing one.
    analyses: dict[str, Any] = {"calibration": None, "comparison": None}
    all_prequential: list[dict[str, Any]] = []
    bootstrap_configs = []
    stage_specs = (("calibration", calibration), ("comparison", comparison))
    for stage, input_path in stage_specs:
        if input_path is None:
            continue
        summary_path = Path(input_path) / "summary.json" if Path(input_path).is_dir() else Path(input_path)
        summary_for_config = _read_json(summary_path)
        root = summary_path.parent
        config = _load_config(root, summary_for_config)
        report_config = config.get("report")
        if not isinstance(report_config, Mapping):
            raise ValueError("frozen config.report must be an object")
        resamples = _integer(report_config.get("bootstrap_resamples"), "config.report.bootstrap_resamples")
        bootstrap_seed = _integer(report_config.get("bootstrap_seed"), "config.report.bootstrap_seed")
        if resamples != 10000 or bootstrap_seed != 20261019:
            raise ValueError("frozen bootstrap settings must be 10000 resamples and seed 20261019")
        bootstrap_configs.append((int(resamples), int(bootstrap_seed)))
        detail, prequential_rows = _analysis_for_stage(
            input_path,
            stage,
            resamples=int(resamples),
            bootstrap_seed=int(bootstrap_seed),
        )
        analyses[stage] = detail
        all_prequential.extend(prequential_rows)
    if len(set(bootstrap_configs)) > 1:
        raise ValueError("calibration and comparison inputs must use the same frozen bootstrap settings")
    resamples, bootstrap_seed = bootstrap_configs[0]
    result = {
        "format": REPORT_FORMAT,
        "status": "report_generated",
        "bootstrap": {
            "resamples": resamples,
            "seed": bootstrap_seed,
            "method": "paired seed bootstrap; calibration averages relations within seed, comparison estimates each relation separately",
            "interval_interpretation": "descriptive exploratory intervals; the four-seed comparison is not a high-power confirmation",
        },
        "calibration": analyses["calibration"],
        "comparison": analyses["comparison"],
        "files": {
            "analysis_json": "analysis.json",
            "markdown": "report.md",
            "per_seed_csv": "per_seed.csv",
            "prequential_csv": "prequential.csv",
        },
    }
    per_seed_rows: list[dict[str, Any]] = []
    if analyses["calibration"] is not None:
        per_seed_rows.extend(analyses["calibration"]["per_seed"])
    if analyses["comparison"] is not None:
        per_seed_rows.extend(analyses["comparison"]["per_seed"])

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown(result), encoding="utf-8")
    _csv_outputs(per_seed_rows, all_prequential, output_dir)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, help="calibration output directory or summary.json")
    parser.add_argument("--comparison", type=Path, help="comparison output directory or summary.json")
    parser.add_argument("--output", type=Path, required=True, help="report output directory")
    args = parser.parse_args(argv)
    result = analyze(calibration=args.calibration, comparison=args.comparison, output=args.output)
    print(json.dumps({
        "output": str(args.output),
        "calibration_status": None if result["calibration"] is None else result["calibration"]["status"],
        "comparison_status": None if result["comparison"] is None else result["comparison"]["status"],
        "calibration_actual_rows": None if result["calibration"] is None else result["calibration"]["costs"]["actual_step_rows"],
        "comparison_actual_rows": None if result["comparison"] is None else result["comparison"]["costs"]["actual_step_rows"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

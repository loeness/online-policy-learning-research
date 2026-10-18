"""Read-only audit of frozen score-calibration experiment artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from online_rl.score_calibration import roundtrip_common  # noqa: E402


FORMAT = "score_calibration_audit_v1"
GRID_SIZE = 9
GRID_ROWS = GRID_SIZE**4
PAIRED_T_CRITICAL_7 = 2.3646242515927844
PRIVATE_PARTS = {"agent.md", "agents.md", "tmp"}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def close(self, actual: Any, expected: Any, message: str, *, atol: float = 1e-10) -> None:
        try:
            equal = math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=atol)
        except (TypeError, ValueError, OverflowError):
            equal = False
        self.check(equal, message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def state_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return sha256_bytes(encoded)


def has_private_component(path: str | Path) -> bool:
    return any(part.casefold() in PRIVATE_PARTS for part in Path(path).parts)


def matrix(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 3 by 3 matrix")
    return array


def arrays_equal(a: Any, b: Any) -> bool:
    try:
        left = np.asarray(a, dtype=np.float32)
        right = np.asarray(b, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        return False
    return left.shape == right.shape and np.array_equal(left, right)


def check_source_snapshot(audit: Audit, cohort: Path) -> dict[str, Any]:
    snapshot = cohort / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    audit.check(manifest_path.is_file(), "source snapshot manifest is missing")
    if not manifest_path.is_file():
        return {}
    manifest = read_json(manifest_path)
    audit.check(isinstance(manifest, dict), "source snapshot manifest must be an object")
    if not isinstance(manifest, dict):
        return {}

    expected_files: set[str] = {"manifest.json"}
    for relative, details in manifest.items():
        audit.check(not has_private_component(relative), f"private path in source manifest: {relative}")
        source = snapshot / relative
        expected_files.add(Path(relative).as_posix())
        if not source.is_file():
            audit.check(False, f"source snapshot file is missing: {relative}")
            continue
        audit.check(source.stat().st_size == details.get("bytes"), f"source size mismatch: {relative}")
        audit.check(sha256_file(source) == details.get("sha256"), f"source digest mismatch: {relative}")

    for source in snapshot.rglob("*"):
        relative = source.relative_to(snapshot)
        audit.check(not has_private_component(relative), f"private path in source snapshot: {relative.as_posix()}")
        if source.is_file():
            audit.check(relative.as_posix() in expected_files,
                        f"unmanifested source snapshot file: {relative.as_posix()}")

    helper = snapshot / "src/online_rl/score_calibration.py"
    runner = snapshot / "scripts/run_score_calibration.py"
    if helper.is_file():
        source = helper.read_text(encoding="utf-8")
        audit.check(bool(re.search(r"_GRID_SIZE\s*=\s*9\b", source)),
                    "frozen helper does not declare a 9 by 9 navigation grid")
        audit.check(bool(re.search(r"_OBSERVATION_COUNT\s*=\s*_GRID_SIZE\s*\*\*\s*4", source)),
                    "frozen helper does not define a complete four-coordinate grid")
    if runner.is_file():
        source = runner.read_text(encoding="utf-8")
        audit.check("grid = grid_observations()" in source,
                    "frozen runner does not build the full observation grid")
        audit.check("coefficient_probs(actor.core, grid," in source,
                    "frozen runner does not score the full grid table")
        audit.check("probability_index(obs[:4])" in source,
                    "frozen runner does not use four-coordinate lookup indexing")

    helper_entry = manifest.get("src/online_rl/score_calibration.py", {})
    live_helper = ROOT / "src/online_rl/score_calibration.py"
    if live_helper.is_file() and helper_entry:
        audit.check(sha256_file(live_helper) == helper_entry.get("sha256"),
                    "live roundtrip helper differs from the frozen helper used for this audit")
    return manifest


def check_full_grid(audit: Audit) -> None:
    # Reconstruct the expected index set with plain arithmetic; no environment
    # instance, policy, or learner is created by this check.
    indices = {
        (((x * GRID_SIZE + y) * GRID_SIZE + gx) * GRID_SIZE + gy)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        for gx in range(GRID_SIZE)
        for gy in range(GRID_SIZE)
    }
    audit.check(len(indices) == GRID_ROWS and indices == set(range(GRID_ROWS)),
                "full-grid index reconstruction did not cover all 9^4 rows")


def compare_prediction_event(audit: Audit, prediction: dict[str, Any], event: dict[str, Any]) -> None:
    step = int(event["step"])
    prefix = f"write_test step {step}"
    audit.check(int(prediction.get("step", -1)) == step, f"{prefix}: prediction step mismatch")
    audit.check(prediction.get("accepted") is event.get("accepted"), f"{prefix}: accepted flag mismatch")
    audit.check(prediction.get("decision_reason") == event.get("reason"), f"{prefix}: reason mismatch")
    audit.check(int(prediction.get("proposal_step", -1)) == int(event.get("proposal_step", -2)),
                f"{prefix}: proposal step mismatch")
    audit.check(int(prediction.get("group", -1)) == int(event.get("group", -2)),
                f"{prefix}: group mismatch")
    audit.check(prediction.get("candidate_raw") == event.get("common"), f"{prefix}: common matrix mismatch")
    provenance = prediction.get("historical_provenance", [])
    original_scores = [item.get("original_score") for item in provenance]
    audit.check(original_scores == event.get("historical_scores"), f"{prefix}: historical scores mismatch")
    audit.check(prediction.get("fresh_score") == event.get("fresh_score"), f"{prefix}: fresh score mismatch")


def audit_parent_chain(
    audit: Audit,
    events: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    current = np.zeros((2, 3, 3), dtype=np.float32)
    versions = [0, 0]
    by_step: dict[int, list[dict[str, Any]]] = {}
    update_by_step: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event") == "write_test":
            by_step.setdefault(int(event["step"]), []).append(event)
    for update in updates:
        update_by_step.setdefault(int(update["step"]), []).append(update)
    prediction_by_step = {int(item["step"]): item for item in predictions}

    for step in sorted(set(by_step) | set(update_by_step)):
        step_events = by_step.get(step, [])
        audit.check(len(step_events) <= 1, f"multiple write_test events at step {step}")
        for event in step_events:
            prediction = prediction_by_step.get(step)
            if prediction is None:
                continue
            group = int(event["group"])
            if group not in (0, 1):
                audit.check(False, f"write_test step {step}: invalid group {group}")
                continue
            audit.check(int(prediction.get("proposal_parent_version", -1)) == versions[group],
                        f"write_test step {step}: parent version mismatch")
            audit.check(arrays_equal(prediction.get("old_parent_raw"), current[group]),
                        f"write_test step {step}: old parent does not match recovered chain")
            candidate = matrix(event["common"], f"write_test {step} common")
            expected_candidate_q, _ = roundtrip_common(candidate)
            expected_parent_q, _ = roundtrip_common(current[group])
            audit.check(arrays_equal(prediction.get("candidate_int8"), expected_candidate_q),
                        f"write_test step {step}: candidate int8 roundtrip mismatch")
            audit.check(arrays_equal(prediction.get("old_parent_int8"), expected_parent_q),
                        f"write_test step {step}: old parent int8 roundtrip mismatch")
            if event.get("accepted") is True:
                current[group] = candidate
                versions[group] += 1

        # The learner records one actor row-codec operation after each update.
        # Shared rows are quantized independently by the frozen int8 format.
        for update in update_by_step.get(step, []):
            for group in range(2):
                current[group], _ = roundtrip_common(current[group])


def audit_anchors(
    audit: Audit,
    run_dir: Path,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    anchor_path = run_dir / "recurrence_anchors.json"
    anchors = read_json(anchor_path)
    anchor_digest = sha256_file(anchor_path)
    expected_offsets = [int(value) for value in config["anchor_offsets"]]
    schedule = read_json(run_dir / "learner/schedule.json")
    sequence = schedule["sequence"]
    bout_steps = int(schedule["bout_steps"])
    initial = sequence[: int(config["capture_initial_bouts"])]
    recurrent = sequence[int(config["capture_initial_bouts"]):]
    audit.check(len(set(initial)) == int(config["capture_initial_bouts"]),
                f"{run_dir}: initial schedule does not contain unique contexts")
    audit.check(set(initial) == set(recurrent), f"{run_dir}: recurrence schedule does not replay the initial contexts")
    audit.check(set(anchors) == set(initial), f"{run_dir}: anchor contexts differ from scheduled recurrences")

    for task, records in anchors.items():
        audit.check(len(records) == len(expected_offsets), f"{run_dir}: {task} has wrong anchor count")
        audit.check(sorted(int(item["offset"]) for item in records) == sorted(expected_offsets),
                    f"{run_dir}: {task} anchor offsets mismatch")
        for item in records:
            bout, offset = int(item["bout"]), int(item["offset"])
            audit.check(bout >= int(config["capture_initial_bouts"]),
                        f"{run_dir}: {task} anchor is not a natural recurrence")
            audit.check(bout < len(sequence) and sequence[bout] == task,
                        f"{run_dir}: {task} anchor has the wrong scheduled bout")
            audit.check(int(item["raw_step"]) == bout * bout_steps + offset + 1,
                        f"{run_dir}: {task} anchor raw step is inconsistent")
            audit.check(item.get("task") == task, f"{run_dir}: anchor task label mismatch")
            audit.check(item.get("snapshot_sha256") == state_hash(item.get("snapshot")),
                        f"{run_dir}: {task} anchor snapshot digest mismatch")

    for prediction in predictions:
        step = int(prediction["step"])
        for task in prediction.get("contexts", []):
            future = anchors.get(task, [])
            audit.check(bool(future), f"{run_dir}: prediction {step} has no recurrence anchors for {task}")
            audit.check(all(int(item["raw_step"]) > step for item in future),
                        f"{run_dir}: prediction {step} uses a nonfuture anchor for {task}")
    return anchors, schedule


def audit_rollout_file(
    audit: Audit,
    path: Path,
    prediction: dict[str, Any],
    anchor: dict[str, Any],
    config: dict[str, Any],
) -> list[float]:
    data = read_json(path)
    prefix = f"{path.parent.name}/{path.name}"
    horizon = int(config["horizon"])
    replicates = int(config["replicates"])
    audit.check(data.get("candidate_id") == prediction.get("id"), f"{prefix}: candidate id mismatch")
    audit.check(data.get("task") == anchor.get("task"), f"{prefix}: task mismatch")
    audit.check(data.get("anchor") == anchor, f"{prefix}: stored anchor differs from natural anchor")
    audit.check(int(data.get("horizon", -1)) == horizon, f"{prefix}: horizon mismatch")
    audit.check(int(data.get("replicates", -1)) == replicates, f"{prefix}: replicate count mismatch")
    audit.check(int(data.get("step_calls", -1)) == 2 * horizon * replicates,
                f"{prefix}: paid fork step count mismatch")
    trials = data.get("per_replicate", [])
    audit.check(len(trials) == replicates, f"{prefix}: per-replicate count mismatch")

    differences: list[float] = []
    for trial_index, trial in enumerate(trials):
        candidate, baseline = trial.get("candidate", {}), trial.get("baseline", {})
        diff, pair_audit = trial.get("differences", {}), trial.get("audit", {})
        tag = f"{prefix} replicate {trial_index}"
        audit.check(int(trial.get("replicate", -1)) == trial_index, f"{tag}: replicate index mismatch")
        for arm_name, arm in (("candidate", candidate), ("baseline", baseline)):
            audit.check(int(arm.get("raw_steps", -1)) == horizon, f"{tag}: {arm_name} raw steps mismatch")
            seeds = arm.get("post_termination_reset_seeds", [])
            audit.check(int(arm.get("post_termination_reset_count", -1)) == len(seeds),
                        f"{tag}: {arm_name} reset seed count mismatch")
        for key in ("first_observations_equal", "initial_rng_state_equal", "paired_action_uniform_per_raw_step",
                    "corresponding_reset_seeds_equal"):
            audit.check(pair_audit.get(key) is True, f"{tag}: paired audit failed at {key}")
        audit.check(int(pair_audit.get("action_uniform_draws", -1)) == horizon,
                    f"{tag}: action uniform draw count mismatch")
        candidate_seeds = candidate.get("post_termination_reset_seeds", [])
        baseline_seeds = baseline.get("post_termination_reset_seeds", [])
        common_prefix = min(len(candidate_seeds), len(baseline_seeds))
        audit.check(candidate_seeds[:common_prefix] == baseline_seeds[:common_prefix],
                    f"{tag}: corresponding reset seed prefix mismatch")
        audit.check(candidate.get("first_observation") == baseline.get("first_observation"),
                    f"{tag}: initial observations differ")
        audit.check(candidate.get("initial_rng_state_sha256") == baseline.get("initial_rng_state_sha256"),
                    f"{tag}: initial environment RNG digests differ")
        for field in ("undiscounted_return", "discounted_first64_return", "return_per_raw_step", "goals"):
            audit.close(diff.get(field), float(candidate[field]) - float(baseline[field]),
                        f"{tag}: {field} difference mismatch")
        differences.append(float(diff["return_per_raw_step"]))
    return differences


def audit_outcomes(
    audit: Audit,
    run_dir: Path,
    config: dict[str, Any],
    anchors: dict[str, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> int:
    included = run_summary.get("candidates", [])
    excluded = run_summary.get("excluded_candidates", [])
    pred_by_id = {item["id"]: item for item in predictions}
    included_ids = [item.get("id") for item in included]
    excluded_ids = [item.get("id") for item in excluded]
    audit.check(len(set(included_ids)) == len(included_ids), f"{run_dir}: duplicate included candidate")
    audit.check(len(set(excluded_ids)) == len(excluded_ids), f"{run_dir}: duplicate excluded candidate")
    audit.check(not (set(included_ids) & set(excluded_ids)), f"{run_dir}: included/excluded candidates overlap")
    audit.check(set(included_ids) | set(excluded_ids) == set(pred_by_id),
                f"{run_dir}: candidates and exclusions do not cover every captured prediction")

    expected_excluded: dict[str, str] = {}
    for prediction in predictions:
        if prediction.get("missing_source_lineage"):
            expected_excluded[prediction["id"]] = "untraceable_historical_source"
        elif not prediction.get("contexts"):
            expected_excluded[prediction["id"]] = "no_historical_source_context"
        elif any(task not in anchors or any(int(item["raw_step"]) <= int(prediction["step"])
                                           for item in anchors[task])
                 for task in prediction.get("contexts", [])):
            expected_excluded[prediction["id"]] = "missing_or_nonfuture_natural_recurrence"
    actual_excluded = {item.get("id"): item.get("reason") for item in excluded}
    audit.check(actual_excluded == expected_excluded, f"{run_dir}: candidate exclusion reasons mismatch")

    forks_dir = run_dir / "forks"
    expected_files: set[str] = set()
    observer_steps = 0
    for candidate in included:
        prediction = pred_by_id.get(candidate.get("id"))
        if prediction is None:
            audit.check(False, f"{run_dir}: outcome references unknown prediction {candidate.get('id')}")
            continue
        for field in ("step", "accepted", "candidate_raw", "old_parent_raw", "candidate_int8", "old_parent_int8",
                      "historical_provenance", "prototype_weights", "contexts"):
            audit.check(candidate.get(field) == prediction.get(field),
                        f"{run_dir}: outcome candidate {prediction['id']} changed prediction field {field}")
        context_gains: dict[str, list[float]] = {}
        for task in prediction["contexts"]:
            task_anchors = anchors.get(task, [])
            anchor_replicates = []
            for anchor in task_anchors:
                offset = int(anchor["offset"])
                name = f"step_{int(prediction['step']):08d}_{task}_{offset:03d}.json"
                expected_files.add(name)
                path = forks_dir / name
                audit.check(path.is_file(), f"{run_dir}: missing fork {name}")
                if not path.is_file():
                    continue
                differences = audit_rollout_file(audit, path, prediction, anchor, config)
                anchor_replicates.append(differences)
                observer_steps += 2 * int(config["horizon"]) * int(config["replicates"])
            if anchor_replicates:
                context_gains[task] = np.asarray(anchor_replicates, dtype=np.float64).mean(axis=0).tolist()

        saved_context_gains = candidate.get("context_replicate_gains", {})
        audit.check(set(saved_context_gains) == set(context_gains),
                    f"{run_dir}: context replicate-gain keys mismatch for {prediction['id']}")
        for task, values in context_gains.items():
            audit.check(np.allclose(saved_context_gains.get(task, []), values, rtol=1e-9, atol=1e-12),
                        f"{run_dir}: anchor mean mismatch for {prediction['id']} context {task}")

        if not context_gains:
            continue
        context_matrix = np.asarray(list(context_gains.values()), dtype=np.float64)
        equal_gains = context_matrix.mean(axis=0)
        weights = prediction.get("prototype_weights", {})
        audit.close(math.fsum(float(value) for value in weights.values()), 1.0,
                    f"{run_dir}: prototype weights do not sum to one for {prediction['id']}")
        weighted_gains = np.zeros(int(config["replicates"]), dtype=np.float64)
        for task, gains in context_gains.items():
            audit.check(task in weights, f"{run_dir}: missing prototype weight for {task}")
            weighted_gains += float(weights.get(task, 0.0)) * np.asarray(gains, dtype=np.float64)
        outcome = candidate.get("outcome", {})
        audit.check(np.allclose(outcome.get("replicate_gains", []), equal_gains, rtol=1e-9, atol=1e-12),
                    f"{run_dir}: equal-context replicate gains mismatch for {prediction['id']}")
        audit.check(np.allclose(outcome.get("prototype_weighted_replicate_gains", []), weighted_gains,
                                rtol=1e-9, atol=1e-12),
                    f"{run_dir}: prototype-weighted replicate gains mismatch for {prediction['id']}")

        if int(config.get("replicates", 0)) == 8 and float(config.get("mc_interval_level", 0.0)) == 0.95:
            mean_gain = float(equal_gains.mean())
            stderr = float(equal_gains.std(ddof=1) / math.sqrt(len(equal_gains)))
            margin = float(config.get("benefit_margin", 0.0))
            low = mean_gain - PAIRED_T_CRITICAL_7 * stderr
            high = mean_gain + PAIRED_T_CRITICAL_7 * stderr
            sign = "beneficial" if low > margin else "harmful" if high < -margin else "ambiguous"
            audit.close(outcome.get("mean_gain"), mean_gain,
                        f"{run_dir}: mean gain mismatch for {prediction['id']}")
            audit.close(outcome.get("stderr"), stderr,
                        f"{run_dir}: Monte Carlo standard error mismatch for {prediction['id']}")
            audit.close(outcome.get("ci_low"), low, f"{run_dir}: CI lower bound mismatch for {prediction['id']}")
            audit.close(outcome.get("ci_high"), high, f"{run_dir}: CI upper bound mismatch for {prediction['id']}")
            audit.check(outcome.get("sign") == sign, f"{run_dir}: CI sign label mismatch for {prediction['id']}")

        expected_candidate_steps = 2 * int(config["horizon"]) * int(config["replicates"]) * \
            len(config["anchor_offsets"]) * len(prediction.get("contexts", []))
        audit.check(int(candidate.get("observer_steps", -1)) == expected_candidate_steps,
                    f"{run_dir}: candidate observer interaction count mismatch for {prediction['id']}")

    actual_files = {path.name for path in forks_dir.glob("*.json")} if forks_dir.exists() else set()
    audit.check(actual_files == expected_files, f"{run_dir}: fork file set differs from candidate/context anchors")
    audit.check(observer_steps == int(run_summary.get("observer_steps", -1)),
                f"{run_dir}: paid observer steps do not equal fork ledger")
    return observer_steps


def audit_run(
    audit: Audit,
    cohort: Path,
    record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    seed, budget = int(record["seed"]), str(record["budget"])
    run_dir = cohort / f"seed_{seed}" / budget
    run_tag = f"seed {seed} budget {budget}"
    local_errors_before = len(audit.errors)
    audit.check(record.get("status") == "completed", f"{run_tag}: run status is not completed")
    audit.check(record.get("audit", {}).get("passed") is True, f"{run_tag}: run audit flag failed")
    audit.check(not (run_dir / "FAILED_RUN.json").exists(), f"{run_tag}: failed-run artifact is present")
    disk_summary_path = run_dir / "summary.json"
    if not disk_summary_path.is_file():
        audit.check(False, f"{run_tag}: run summary is missing")
        return {"seed": seed, "budget": budget, "passed": False}
    disk_summary = read_json(disk_summary_path)
    audit.check(disk_summary == record, f"{run_tag}: cohort record differs from run summary")

    learner_dir = run_dir / "learner"
    learner_summary = read_json(learner_dir / "summary.json")
    learner_config = read_json(learner_dir / "run_config.json")
    schedule = read_json(learner_dir / "schedule.json")
    raw_rows = list(csv.DictReader((learner_dir / "steps.csv").open(newline="", encoding="utf-8")))
    updates = read_jsonl(learner_dir / "updates.jsonl")
    events = read_jsonl(learner_dir / "events.jsonl")
    bout_steps = int(schedule["bout_steps"])
    expected_steps = int(schedule["raw_steps"])
    expected_sequence = schedule.get("sequence", [])
    audit.check(expected_steps == len(expected_sequence) * bout_steps,
                f"{run_tag}: schedule raw-step total mismatch")
    audit.check(len(expected_sequence) == 12, f"{run_tag}: expected twelve-bout schedule")
    audit.check(learner_summary.get("status") == "completed", f"{run_tag}: learner status is not completed")
    audit.check(record.get("learner_steps") == expected_steps, f"{run_tag}: observer and schedule raw steps differ")
    audit.check(learner_summary.get("raw_steps") == expected_steps, f"{run_tag}: learner raw steps mismatch")
    audit.check(len(raw_rows) == expected_steps, f"{run_tag}: raw CSV row count mismatch")
    audit.check([int(row["raw_step"]) for row in raw_rows] == list(range(1, expected_steps + 1)),
                f"{run_tag}: raw CSV steps are not contiguous")
    reward_sum = math.fsum(float(row["reward"]) for row in raw_rows)
    audit.close(learner_summary.get("total_reward"), reward_sum, f"{run_tag}: CSV reward total mismatch")
    audit.close(learner_summary.get("reward_per_raw_step"), reward_sum / expected_steps,
                f"{run_tag}: reward per raw step mismatch")
    fit_steps = sum(int(item.get("rows", 0)) for item in updates)
    audit.check(learner_summary.get("fit_steps") == fit_steps,
                f"{run_tag}: learner fit steps do not equal update rows")
    audit.check(record.get("learner_summary", {}).get("fit_steps") == fit_steps,
                f"{run_tag}: outer fit-step summary mismatch")
    cumulative = 0
    for update in updates:
        cumulative += int(update.get("rows", 0))
        audit.check(int(update.get("step", -1)) == cumulative,
                    f"{run_tag}: update step does not equal cumulative fitted rows")
    audit.check(fit_steps == expected_steps, f"{run_tag}: raw rows and fit rows differ")
    capacity = read_json(learner_dir / "capacity.json")
    audit.check(bool(capacity) and [int(item.get("step", -1)) for item in capacity] ==
                list(range(bout_steps, expected_steps + 1, bout_steps)),
                f"{run_tag}: capacity ledger does not cover every bout boundary")

    predictions_path = run_dir / "predictions.json"
    predictions = read_json(predictions_path)
    anchor_path = run_dir / "recurrence_anchors.json"
    audit.check(record.get("prediction_sha256") == sha256_file(predictions_path),
                f"{run_tag}: prediction SHA-256 mismatch")
    audit.check(record.get("anchor_sha256") == sha256_file(anchor_path),
                f"{run_tag}: recurrence-anchor SHA-256 mismatch")
    capture_cutoff = int(config["capture_initial_bouts"]) * bout_steps
    write_events = [item for item in events if item.get("event") == "write_test"]
    prefix_events = [item for item in write_events if int(item["step"]) <= capture_cutoff]
    later_events = [item for item in write_events if int(item["step"]) > capture_cutoff]
    event_steps = [int(item["step"]) for item in prefix_events]
    prediction_steps = [int(item["step"]) for item in predictions]
    audit.check(len(event_steps) == len(set(event_steps)), f"{run_tag}: duplicate prefix write-test step")
    audit.check(prediction_steps == event_steps,
                f"{run_tag}: captured predictions are not exactly the ordered prefix write tests")
    audit.check(not any(step > capture_cutoff for step in prediction_steps),
                f"{run_tag}: prediction recorded after the frozen prefix cutoff")
    event_by_step = {int(item["step"]): item for item in prefix_events}
    for prediction in predictions:
        event = event_by_step.get(int(prediction["step"]))
        if event is not None:
            compare_prediction_event(audit, prediction, event)
    audit_parent_chain(audit, events, updates, predictions)

    anchors, _ = audit_anchors(audit, run_dir, config, predictions)
    run_observer_steps = audit_outcomes(audit, run_dir, config, anchors, predictions, record)
    table_errors = [item.get("lookup_max_probability_error") for item in record.get("candidates", [])]
    audit.check(record.get("audit", {}).get("lookup_distribution_verified") is True,
                f"{run_tag}: lookup-distribution audit flag is false")
    audit.check(all(value is not None and math.isfinite(float(value)) and float(value) <= 1e-6
                    for value in table_errors), f"{run_tag}: full-grid lookup error is absent or too large")
    audit.check(int(record.get("captured_tests", -1)) == len(predictions),
                f"{run_tag}: captured test count mismatch")
    return {"seed": seed, "budget": budget, "passed": len(audit.errors) == local_errors_before,
            "learner_steps": int(record.get("learner_steps", 0)),
            "observer_steps": run_observer_steps, "write_tests": len(write_events),
            "captured_prefix_tests": len(predictions), "later_write_tests": len(later_events),
            "fork_files": len(list((run_dir / "forks").glob("*.json")))}


def audit_cohort(path: Path) -> dict[str, Any]:
    cohort = path.resolve()
    audit = Audit()
    report: dict[str, Any] = {"path": str(cohort), "passed": False, "errors": [], "runs": [],
                              "failed_costs": [], "cost": {"learner_steps": 0, "observer_steps": 0,
                                                              "new_total_interactions": 0}}
    if not cohort.is_dir():
        report["errors"] = ["cohort directory does not exist"]
        return report
    summary_path = cohort / "summary.json"
    if not summary_path.is_file():
        report["errors"] = ["cohort summary.json is missing"]
        report["failed_costs"] = [read_json(p) for p in cohort.glob("seed_*/*/FAILED_RUN.json")]
        report["cost"] = {
            "learner_steps": sum(int(item.get("learner_steps", 0)) for item in report["failed_costs"]),
            "observer_steps": sum(int(item.get("observer_steps", 0)) for item in report["failed_costs"]),
        }
        report["cost"]["new_total_interactions"] = sum(report["cost"].values())
        return report

    summary = read_json(summary_path)
    audit.check(summary.get("format") == "score_calibration_cohort_v1", "unexpected cohort summary format")
    config = summary.get("config", {})
    audit.check(summary.get("analysis_role") == "diagnostic", "cohort analysis role is not diagnostic")
    audit.check(config.get("feedback_to_learner") is False, "config enables observer feedback to learner")
    audit.check(config.get("retain_all_prefix_write_tests") is True,
                "config does not declare all prefix write tests retained")
    audit.check(config.get("tune_from_pilot_or_main") is False, "config permits tuning from calibration outcomes")
    frozen_path = cohort / "frozen_config.json"
    audit.check(frozen_path.is_file(), "frozen_config.json is missing")
    if frozen_path.is_file():
        frozen = read_json(frozen_path)
        audit.check(frozen.get("calibration") == config, "cohort config differs from frozen calibration config")
    check_source_snapshot(audit, cohort)
    check_full_grid(audit)

    expected_pairs = {(int(seed), str(budget)) for seed in config.get("seeds", [])
                      for budget in config.get("budgets", [])}
    records = summary.get("records", [])
    failed_costs = summary.get("failed_costs", [])
    report["failed_costs"] = failed_costs
    record_pairs = [(int(item.get("seed", -1)), str(item.get("budget", ""))) for item in records]
    failed_pairs = [(int(item.get("seed", -1)), str(item.get("budget", ""))) for item in failed_costs]
    audit.check(len(record_pairs) == len(set(record_pairs)), "duplicate completed run in cohort summary")
    audit.check(len(failed_pairs) == len(set(failed_pairs)), "duplicate failed run in cohort summary")
    audit.check(not (set(record_pairs) & set(failed_pairs)), "run appears as both completed and failed")
    audit.check(set(record_pairs) | set(failed_pairs) == expected_pairs,
                "completed and failed run records do not cover the frozen cohort")
    audit.check(summary.get("status") == "completed" and summary.get("audit", {}).get("passed") is True,
                "cohort did not complete successfully")
    audit.check(not summary.get("audit", {}).get("failures"), "cohort runner recorded failures")

    failed_files = {path.resolve(): read_json(path) for path in cohort.glob("seed_*/*/FAILED_RUN.json")}
    failed_file_payloads = list(failed_files.values())
    audit.check(failed_file_payloads == failed_costs,
                "FAILED_RUN.json artifacts differ from preserved cohort failed_costs")
    for run_path in cohort.glob("seed_*/*/FAILED_RUN.json"):
        audit.check(not has_private_component(run_path.relative_to(cohort)),
                    f"private component in run artifact path: {run_path.relative_to(cohort)}")

    total_learner = sum(int(item.get("learner_steps", 0)) for item in records + failed_costs)
    total_observer = sum(int(item.get("observer_steps", 0)) for item in records + failed_costs)
    computed_cost = {"learner_steps": total_learner, "observer_steps": total_observer,
                     "new_total_interactions": total_learner + total_observer}
    report["cost"] = computed_cost
    audit.check(summary.get("cost") == computed_cost, "cohort interaction cost does not match preserved run costs")

    for record in records:
        try:
            run_report = audit_run(audit, cohort, record, config)
            report["runs"].append(run_report)
        except Exception as error:  # Continue auditing other run folders and preserve the failure detail.
            audit.errors.append(f"run {record.get('seed')}/{record.get('budget')}: {type(error).__name__}: {error}")

    discovered_pairs = set()
    for seed_dir in cohort.glob("seed_*"):
        match = re.fullmatch(r"seed_(\d+)", seed_dir.name)
        if not match:
            continue
        for budget_dir in seed_dir.iterdir():
            if budget_dir.is_dir():
                discovered_pairs.add((int(match.group(1)), budget_dir.name))
    audit.check(discovered_pairs == expected_pairs, "run folders differ from the frozen seed/budget matrix")
    report["passed"] = not audit.errors
    report["errors"] = audit.errors
    report["status"] = summary.get("status")
    report["expected_runs"] = len(expected_pairs)
    report["completed_runs"] = len(records)
    report["failed_runs"] = len(failed_costs)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", required=True, type=Path,
                        help="cohort output directory; may be repeated")
    parser.add_argument("--output", required=True, type=Path,
                        help="new JSON report path outside all audited cohorts")
    args = parser.parse_args()
    cohorts = [path.resolve() for path in args.cohort]
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing audit output: {output}")
    if any(output == cohort or cohort in output.parents for cohort in cohorts):
        raise SystemExit("audit output must be outside all input cohort directories")

    reports = [audit_cohort(path) for path in cohorts]
    total_learner = sum(int(item["cost"].get("learner_steps", 0)) for item in reports)
    total_observer = sum(int(item["cost"].get("observer_steps", 0)) for item in reports)
    result = {
        "format": FORMAT,
        "status": "passed" if all(item.get("passed") for item in reports) else "failed",
        "cohorts": reports,
        "errors": [f"{item['path']}: {error}" for item in reports for error in item.get("errors", [])],
        "cost": {"learner_steps": total_learner, "observer_steps": total_observer,
                 "new_total_interactions": total_learner + total_observer},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "cost": result["cost"],
                      "errors": len(result["errors"])}, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

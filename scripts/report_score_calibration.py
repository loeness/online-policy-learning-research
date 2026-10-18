"""Report whether frozen retention-surplus scores predict later recurrence benefit.

This reader intentionally depends only on the saved cohort summary schema. It
does not import the experiment runner or its policy and environment modules.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_SEEDS = (19703, 19717, 19727)
EXPECTED_BUDGETS = ("low", "wide")
PILOT_SEED = 19701
REPLICATES = 8
ANCHORS_PER_CONTEXT = 4
POLICY_ARMS = 2
RAW_STEPS_PER_ARM = 128
OBSERVER_STEPS_PER_CONTEXT = REPLICATES * ANCHORS_PER_CONTEXT * POLICY_ARMS * RAW_STEPS_PER_ARM
PREDICTORS = (
    "original_mean_lower",
    "original_mean_gain",
    "codec_mean_lower",
    "fresh_lower",
    "negative_mean_stderr",
    "time_step",
)
PRIMARY_PREDICTOR = "original_mean_lower"
T_CRITICAL_DF_7_95 = 2.364624251
NUMERIC_TOLERANCE = 1e-9


def _finite(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    number = _finite(value, name)
    assert number is not None
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _check_finite_tree(value: Any, name: str = "summary") -> None:
    """Reject non-finite JSON numeric values, including in preserved metadata."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _finite(value, name)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite_tree(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite_tree(item, f"{name}.{key}")


def _rankdata(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks, including average ranks for ties."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    sum_xx = sum(value * value for value in centered_x)
    sum_yy = sum(value * value for value in centered_y)
    if sum_xx <= 0.0 or sum_yy <= 0.0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / math.sqrt(sum_xx * sum_yy)


def spearman(x: Sequence[float], y: Sequence[float], *, minimum_events: int = 5) -> dict[str, Any]:
    """Descriptive Spearman correlation; no p-value is calculated."""
    if len(x) != len(y):
        raise ValueError("Spearman inputs must have equal length")
    n = len(x)
    if n < minimum_events:
        return {"n": n, "rho": None, "status": "not_estimable_too_few_events"}
    rho = _pearson(_rankdata(x), _rankdata(y))
    if rho is None:
        return {"n": n, "rho": None, "status": "not_estimable_constant_input"}
    return {"n": n, "rho": rho, "status": "estimable"}


def partial_spearman_by_group(
    x: Sequence[float], y: Sequence[float], groups: Sequence[str], *, minimum_events: int = 5,
) -> dict[str, Any]:
    """Correlate global ranks after removing each group's rank mean."""
    if not (len(x) == len(y) == len(groups)):
        raise ValueError("partial Spearman inputs must have equal length")
    n = len(x)
    if n < minimum_events:
        return {"n": n, "rho": None, "status": "not_estimable_too_few_events"}
    rank_x = _rankdata(x)
    rank_y = _rankdata(y)
    group_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_indices.setdefault(group, []).append(index)
    residual_x = [0.0] * n
    residual_y = [0.0] * n
    for indices in group_indices.values():
        mean_x = statistics.fmean(rank_x[index] for index in indices)
        mean_y = statistics.fmean(rank_y[index] for index in indices)
        for index in indices:
            residual_x[index] = rank_x[index] - mean_x
            residual_y[index] = rank_y[index] - mean_y
    rho = _pearson(residual_x, residual_y)
    if rho is None:
        return {"n": n, "rho": None, "status": "not_estimable_degenerate_group_residuals"}
    return {"n": n, "rho": rho, "status": "estimable"}


def _outcome(replicate_values: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(replicate_values, list) or len(replicate_values) != REPLICATES:
        raise ValueError(f"{prefix} must contain exactly {REPLICATES} paired replicate gains")
    values = [_finite(value, f"{prefix}[{index}]") for index, value in enumerate(replicate_values)]
    clean = [float(value) for value in values if value is not None]
    mean_gain = statistics.fmean(clean)
    stderr = statistics.stdev(clean) / math.sqrt(REPLICATES)
    margin = T_CRITICAL_DF_7_95 * stderr
    ci_low = mean_gain - margin
    ci_high = mean_gain + margin
    if ci_low > 0.0:
        sign = "beneficial"
    elif ci_high < 0.0:
        sign = "harmful"
    else:
        sign = "ambiguous"
    return {
        "replicate_gains": clean,
        "mean_gain": mean_gain,
        "stderr": stderr,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sign": sign,
    }


def _close(actual: Any, expected: float, name: str) -> float:
    number = _finite(actual, name)
    assert number is not None
    if not math.isclose(number, expected, rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE):
        raise ValueError(f"{name} does not match its replicate-level value")
    return number


def _validate_candidate(row: Any, branch_seed: int, branch_budget: str, prefix: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{prefix} must be an object")
    required = ("id", "seed", "budget", "group", "step", "accepted", "predictors", "contexts", "prototype_weights", "outcome")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{prefix} is missing required fields: {', '.join(missing)}")
    if not isinstance(row["id"], (str, int)) or isinstance(row["id"], bool) or str(row["id"]).strip() == "":
        raise ValueError(f"{prefix}.id must be a non-empty string or integer")
    seed = _integer(row["seed"], f"{prefix}.seed")
    if seed != branch_seed or row["budget"] != branch_budget:
        raise ValueError(f"{prefix} seed/budget do not match the containing branch")
    if isinstance(row["group"], bool) or not isinstance(row["group"], (str, int)) or str(row["group"]).strip() == "":
        raise ValueError(f"{prefix}.group must be a non-empty string or integer label")
    step = _integer(row["step"], f"{prefix}.step")
    if step > 12288:
        raise ValueError(f"{prefix}.step exceeds the pre-recurrence score freeze window")
    if not isinstance(row["accepted"], bool):
        raise ValueError(f"{prefix}.accepted must be boolean")

    predictors = row["predictors"]
    if not isinstance(predictors, dict):
        raise ValueError(f"{prefix}.predictors must be an object")
    clean_predictors: dict[str, float | None] = {}
    for key in PREDICTORS:
        if key not in predictors:
            raise ValueError(f"{prefix}.predictors.{key} is required")
        clean_predictors[key] = _finite(predictors[key], f"{prefix}.predictors.{key}", allow_none=True)

    contexts = row["contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"{prefix}.contexts must be a non-empty array")
    if any(not isinstance(context, str) or not context.strip() for context in contexts):
        raise ValueError(f"{prefix}.contexts entries must be non-empty strings")
    if len(contexts) != len(set(contexts)):
        raise ValueError(f"{prefix}.contexts must be unique")
    weights = row["prototype_weights"]
    if not isinstance(weights, dict):
        raise ValueError(f"{prefix}.prototype_weights must be an object")
    clean_weights: dict[str, float] = {}
    for task, weight in weights.items():
        if task not in contexts:
            raise ValueError(f"{prefix}.prototype_weights contains a context outside contexts")
        clean_weight = _finite(weight, f"{prefix}.prototype_weights[{task!r}]")
        assert clean_weight is not None
        if clean_weight < 0:
            raise ValueError(f"{prefix}.prototype_weights values must be non-negative")
        clean_weights[task] = clean_weight
    if clean_weights and sum(clean_weights.values()) <= 0:
        raise ValueError(f"{prefix}.prototype_weights must have positive total weight")

    outcome = row["outcome"]
    if not isinstance(outcome, dict):
        raise ValueError(f"{prefix}.outcome must be an object")
    primary = _outcome(outcome.get("replicate_gains"), f"{prefix}.outcome.replicate_gains")
    _close(outcome.get("mean_gain"), primary["mean_gain"], f"{prefix}.outcome.mean_gain")
    _close(outcome.get("stderr"), primary["stderr"], f"{prefix}.outcome.stderr")
    _close(outcome.get("ci_low"), primary["ci_low"], f"{prefix}.outcome.ci_low")
    _close(outcome.get("ci_high"), primary["ci_high"], f"{prefix}.outcome.ci_high")
    if outcome.get("sign") != primary["sign"]:
        raise ValueError(f"{prefix}.outcome.sign does not match the pointwise t7 interval")
    weighted = _outcome(
        outcome.get("prototype_weighted_replicate_gains"),
        f"{prefix}.outcome.prototype_weighted_replicate_gains",
    )
    return {
        "id": str(row["id"]), "seed": seed, "budget": branch_budget,
        "group": str(row["group"]), "step": step, "accepted": row["accepted"],
        "predictors": clean_predictors, "contexts": list(contexts),
        "prototype_weights": clean_weights,
        "outcome": {**primary, "prototype_weighted_replicate_gains": weighted["replicate_gains"],
                    "prototype_weighted_mean_gain": weighted["mean_gain"],
                    "prototype_weighted_stderr": weighted["stderr"],
                    "prototype_weighted_ci_low": weighted["ci_low"],
                    "prototype_weighted_ci_high": weighted["ci_high"],
                    "prototype_weighted_sign": weighted["sign"]},
    }


def _validate_excluded(row: Any, branch_seed: int, branch_budget: str, prefix: str) -> dict[str, Any]:
    """Validate the preserved prediction record for a test with no source context."""
    if isinstance(row, (str, int)) and not isinstance(row, bool):
        raise ValueError(f"{prefix} must preserve prediction data, source provenance, and reason")
    if not isinstance(row, dict):
        raise ValueError(f"{prefix} must be an object")
    for key in ("id", "seed", "budget", "step", "reason", "predictors"):
        if key not in row:
            raise ValueError(f"{prefix}.{key} is required")
    if not isinstance(row["id"], (str, int)) or isinstance(row["id"], bool) or str(row["id"]).strip() == "":
        raise ValueError(f"{prefix}.id must be a non-empty string or integer")
    if _integer(row["seed"], f"{prefix}.seed") != branch_seed:
        raise ValueError(f"{prefix}.seed does not match the containing branch")
    if row["budget"] != branch_budget:
        raise ValueError(f"{prefix}.budget does not match the containing branch")
    step = _integer(row["step"], f"{prefix}.step")
    if step > 12288:
        raise ValueError(f"{prefix}.step exceeds the pre-recurrence score freeze window")
    if not isinstance(row["reason"], str) or not row["reason"].strip():
        raise ValueError(f"{prefix}.reason must be a non-empty string")
    if "outcome" in row:
        raise ValueError(f"{prefix} cannot include an outcome because the test had no evaluable source context")
    predictors = row["predictors"]
    if not isinstance(predictors, dict):
        raise ValueError(f"{prefix}.predictors must be an object")
    for key in PREDICTORS:
        if key not in predictors:
            raise ValueError(f"{prefix}.predictors.{key} is required")
        _finite(predictors[key], f"{prefix}.predictors.{key}", allow_none=True)
    if "contexts" in row:
        contexts = row["contexts"]
        if not isinstance(contexts, list) or any(not isinstance(item, str) for item in contexts):
            raise ValueError(f"{prefix}.contexts must be an array of strings")
        if contexts:
            raise ValueError(f"{prefix} has source contexts and cannot be excluded")
    return dict(row)


def _read_cohort(input_path: str | Path) -> dict[str, Any]:
    source = Path(input_path)
    summary_path = source / "summary.json" if source.is_dir() else source
    if summary_path.name != "summary.json" or not summary_path.is_file():
        raise FileNotFoundError(f"expected cohort summary.json under {source}")
    try:
        cohort = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cohort summary JSON: {summary_path}") from exc
    if not isinstance(cohort, dict):
        raise ValueError("cohort summary must be a JSON object")
    return cohort


def validate_cohort(cohort: Any) -> dict[str, Any]:
    """Validate the complete predeclared cohort and return normalized records."""
    if not isinstance(cohort, dict):
        raise ValueError("cohort summary must be an object")
    _check_finite_tree(cohort)
    if cohort.get("format") != "score_calibration_cohort_v1":
        raise ValueError("input format must be score_calibration_cohort_v1")
    if cohort.get("status") != "completed":
        raise ValueError("cohort status must be completed")
    if cohort.get("analysis_role") != "diagnostic":
        raise ValueError("cohort analysis_role must be diagnostic")
    config = cohort.get("config")
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    seeds = config.get("seeds")
    budgets = config.get("budgets")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("config.seeds must be a non-empty array")
    clean_seeds = [_integer(seed, f"config.seeds[{index}]") for index, seed in enumerate(seeds)]
    if len(clean_seeds) != len(set(clean_seeds)):
        raise ValueError("config.seeds must not contain duplicates")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("config.budgets must be a non-empty array")
    clean_budgets = list(budgets)
    if any(not isinstance(budget, str) or budget not in EXPECTED_BUDGETS for budget in clean_budgets):
        raise ValueError(f"config.budgets entries must be from {EXPECTED_BUDGETS}")
    if len(clean_budgets) != len(set(clean_budgets)):
        raise ValueError("config.budgets must not contain duplicates")
    mode = config.get("mode")
    if mode not in ("main", "pilot"):
        raise ValueError("config.mode must be 'main' or 'pilot'")
    bout_steps = _integer(config.get("bout_steps"), "config.bout_steps", minimum=1)
    if mode == "main":
        if tuple(clean_seeds) != EXPECTED_SEEDS or tuple(clean_budgets) != EXPECTED_BUDGETS:
            raise ValueError("main config must contain the complete predeclared 3-seed by 2-budget grid")
        if bout_steps != 2048:
            raise ValueError("main config.bout_steps must be 2048")
    else:
        if tuple(clean_seeds) != (PILOT_SEED,) or tuple(clean_budgets) != ("low",):
            raise ValueError("pilot config must contain only seed 19701 and the low budget")
        if bout_steps != 1024:
            raise ValueError("pilot config.bout_steps must be 1024")
    if _integer(config.get("replicates"), "config.replicates") != REPLICATES:
        raise ValueError(f"config.replicates must be {REPLICATES}")
    expected_branch_learner_steps = 12 * bout_steps

    audit = cohort.get("audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError("cohort.audit.passed must be true")
    cost = cohort.get("cost")
    if not isinstance(cost, dict):
        raise ValueError("cost must be an object")
    clean_cost = {field: _integer(cost.get(field), f"cost.{field}") for field in
                  ("learner_steps", "observer_steps", "new_total_interactions")}
    if clean_cost["new_total_interactions"] != clean_cost["learner_steps"] + clean_cost["observer_steps"]:
        raise ValueError("cost.new_total_interactions must equal learner_steps + observer_steps")
    cpu_s = _finite(cohort.get("cpu_s"), "cpu_s")
    wall_s = _finite(cohort.get("wall_s"), "wall_s")
    if cpu_s is None or wall_s is None or cpu_s < 0 or wall_s < 0:
        raise ValueError("cpu_s and wall_s must be non-negative finite numbers")

    rows = cohort.get("records")
    if not isinstance(rows, list):
        raise ValueError("records must be an array")
    expected_grid = {(seed, budget) for seed in clean_seeds for budget in clean_budgets}
    actual_grid: set[tuple[int, str]] = set()
    clean_records: list[dict[str, Any]] = []
    summed_learner_steps = 0
    summed_observer_steps = 0
    for index, row in enumerate(rows):
        prefix = f"records[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{prefix} must be an object")
        seed = _integer(row.get("seed"), f"{prefix}.seed")
        budget = row.get("budget")
        if budget not in EXPECTED_BUDGETS:
            raise ValueError(f"{prefix}.budget must be one of {EXPECTED_BUDGETS}")
        key = (seed, budget)
        if key not in expected_grid:
            raise ValueError(f"{prefix} is outside the declared seed/budget grid")
        if key in actual_grid:
            raise ValueError(f"duplicate branch for seed={seed}, budget={budget}")
        actual_grid.add(key)
        if row.get("status") != "completed":
            raise ValueError(f"{prefix} is incomplete: status={row.get('status')!r}")
        learner_steps = _integer(row.get("learner_steps"), f"{prefix}.learner_steps")
        observer_steps = _integer(row.get("observer_steps"), f"{prefix}.observer_steps")
        if learner_steps != expected_branch_learner_steps:
            raise ValueError(
                f"{prefix}.learner_steps must equal 12 bouts x config.bout_steps ({expected_branch_learner_steps})"
            )
        summed_learner_steps += learner_steps
        summed_observer_steps += observer_steps
        candidates_value = row.get("candidates")
        excluded_value = row.get("excluded_candidates")
        if not isinstance(candidates_value, list) or not isinstance(excluded_value, list):
            raise ValueError(f"{prefix}.candidates and excluded_candidates must be arrays")
        candidates = [
            _validate_candidate(candidate, seed, budget, f"{prefix}.candidates[{candidate_index}]")
            for candidate_index, candidate in enumerate(candidates_value)
        ]
        excluded = [
            _validate_excluded(candidate, seed, budget, f"{prefix}.excluded_candidates[{candidate_index}]")
            for candidate_index, candidate in enumerate(excluded_value)
        ]
        ids = [candidate["id"] for candidate in candidates + excluded]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{prefix} has duplicate candidate ids")
        declared_captured = len(candidates) + len(excluded)
        if _integer(row.get("captured_tests"), f"{prefix}.captured_tests") != declared_captured:
            raise ValueError(f"{prefix}.captured_tests must equal candidates + excluded_candidates")
        expected_observer_steps = sum(len(candidate["contexts"]) for candidate in candidates) * OBSERVER_STEPS_PER_CONTEXT
        if observer_steps != expected_observer_steps:
            raise ValueError(
                f"{prefix}.observer_steps must equal {OBSERVER_STEPS_PER_CONTEXT} per evaluated context"
            )
        branch_audit = row.get("audit")
        if branch_audit is not None and (not isinstance(branch_audit, dict) or branch_audit.get("passed") is not True):
            raise ValueError(f"{prefix}.audit.passed must be true when branch audit is present")
        clean_records.append({
            "seed": seed, "budget": budget, "status": "completed",
            "learner_steps": learner_steps, "observer_steps": observer_steps,
            "captured_tests": declared_captured,
            "candidates": candidates, "excluded_candidates": excluded,
        })
    if actual_grid != expected_grid:
        missing = sorted(expected_grid - actual_grid)
        raise ValueError(f"records do not contain the complete seed/budget grid; missing {missing}")
    if summed_learner_steps != clean_cost["learner_steps"]:
        raise ValueError("cost.learner_steps does not equal the sum of branch learner_steps")
    if summed_observer_steps != clean_cost["observer_steps"]:
        raise ValueError("cost.observer_steps does not equal the sum of branch observer_steps")
    actual_total_tests = sum(record["captured_tests"] for record in clean_records)
    expected_learner_steps = len(expected_grid) * expected_branch_learner_steps
    if clean_cost["learner_steps"] != expected_learner_steps:
        raise ValueError(f"cost.learner_steps must equal the declared-grid total {expected_learner_steps}")
    return {
        "config": dict(config), "mode": mode, "seeds": clean_seeds, "budgets": clean_budgets,
        "audit": dict(audit), "cost": clean_cost,
        "cpu_s": cpu_s, "wall_s": wall_s, "records": clean_records,
        "captured_tests": actual_total_tests,
    }


def _confusion(rows: Sequence[Mapping[str, Any]], prediction_key: str) -> dict[str, Any]:
    resolved = [row for row in rows if row["outcome"]["sign"] in ("beneficial", "harmful")]
    ambiguous_count = sum(row["outcome"]["sign"] == "ambiguous" for row in rows)
    missing = [row for row in resolved if row["predictors"][PRIMARY_PREDICTOR] is None] if prediction_key == "score_sign" else []
    if prediction_key == "score_sign":
        eligible = [row for row in resolved if row["predictors"][PRIMARY_PREDICTOR] is not None]
        positive = {row["id"] for row in eligible if row["predictors"][PRIMARY_PREDICTOR] > 0.0}
    else:
        eligible = resolved
        positive = {row["id"] for row in eligible if row["accepted"]}
    actual_positive = {row["id"] for row in eligible if row["outcome"]["sign"] == "beneficial"}
    actual_negative = {row["id"] for row in eligible if row["outcome"]["sign"] == "harmful"}
    approved_but_harmful = sorted(positive & actual_negative)
    not_approved_but_beneficial = sorted((set(row["id"] for row in eligible) - positive) & actual_positive)
    n_positive_predictions = len(positive)
    n_actual_positive = len(actual_positive)
    precision = len(positive & actual_positive) / n_positive_predictions if n_positive_predictions else None
    recall = len(positive & actual_positive) / n_actual_positive if n_actual_positive else None
    positive_label = "accepted" if prediction_key == "accepted_gate" else "approved"
    negative_label = "not_accepted" if prediction_key == "accepted_gate" else "not_approved"
    return {
        "classification": "accepted_gate" if prediction_key == "accepted_gate" else "original_mean_lower_gt_zero_approval_gate",
        "n_total": len(rows), "n_resolved_outcomes": len(resolved), "n_classified": len(eligible),
        "n_resolved": len(eligible),
        "n_ambiguous_outcomes": ambiguous_count,
        "n_missing_score_among_resolved": len(missing),
        "positive_prediction_count": n_positive_predictions,
        "beneficial_count": n_actual_positive,
        "harmful_count": len(actual_negative),
        f"{positive_label}_but_mc_harmful_count": len(approved_but_harmful),
        f"{positive_label}_but_mc_harmful_ids": approved_but_harmful,
        f"{negative_label}_but_mc_beneficial_count": len(not_approved_but_beneficial),
        f"{negative_label}_but_mc_beneficial_ids": not_approved_but_beneficial,
        "precision": precision, "precision_status": "estimable" if precision is not None else "not_estimable_no_positive_predictions",
        "recall": recall, "recall_status": "estimable" if recall is not None else "not_estimable_no_beneficial_outcomes",
    }


def _auc(rows: Sequence[Mapping[str, Any]], predictor: str) -> float | None:
    resolved = [row for row in rows if row["outcome"]["sign"] in ("beneficial", "harmful") and row["predictors"][predictor] is not None]
    labels = [row["outcome"]["sign"] == "beneficial" for row in resolved]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _rankdata([float(row["predictors"][predictor]) for row in resolved])
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _correlation_for(rows: Sequence[Mapping[str, Any]], predictor: str, target: str) -> dict[str, Any]:
    selected = [row for row in rows if row["predictors"][predictor] is not None]
    x = [float(row["predictors"][predictor]) for row in selected]
    if target == "mean_gain":
        y = [float(row["outcome"]["mean_gain"]) for row in selected]
    elif target == "prototype_weighted_mean_gain":
        y = [float(row["outcome"]["prototype_weighted_mean_gain"]) for row in selected]
    else:
        raise ValueError(f"unknown correlation target: {target}")
    result = spearman(x, y)
    result["predictor"] = predictor
    result["target"] = target
    result["missing_predictor_count"] = len(rows) - len(selected)
    return result


def _cell_analysis(branch: Mapping[str, Any]) -> dict[str, Any]:
    rows = branch["candidates"]
    diagnostics = {predictor: _correlation_for(rows, predictor, "mean_gain") for predictor in PREDICTORS}
    primary = diagnostics[PRIMARY_PREDICTOR]
    selected = [row for row in rows if row["predictors"][PRIMARY_PREDICTOR] is not None]
    partial = partial_spearman_by_group(
        [float(row["predictors"][PRIMARY_PREDICTOR]) for row in selected],
        [float(row["outcome"]["mean_gain"]) for row in selected],
        [str(row["group"]) for row in selected],
    )
    partial["missing_predictor_count"] = len(rows) - len(selected)
    prototype = _correlation_for(rows, PRIMARY_PREDICTOR, "prototype_weighted_mean_gain")
    return {
        "seed": branch["seed"], "budget": branch["budget"],
        "candidate_count": len(rows), "excluded_candidate_count": len(branch["excluded_candidates"]),
        "captured_tests": branch["captured_tests"],
        "predictor_missing_counts": {
            predictor: sum(row["predictors"][predictor] is None for row in rows)
            for predictor in PREDICTORS
        },
        "predictor_rho_vs_mean_gain": diagnostics,
        "primary_partial_rho_controlling_group": partial,
        "primary_rho_vs_prototype_weighted_mean_gain": prototype,
        "primary_score_approval_gate": _confusion(rows, "score_sign"),
        "accepted_gate": _confusion(rows, "accepted_gate"),
        "primary_auc_resolved_sign": _auc(rows, PRIMARY_PREDICTOR),
        "candidate_outcomes": [
            {"id": row["id"], "step": row["step"], "group": row["group"],
             "predictors": dict(row["predictors"]), "contexts": list(row["contexts"]),
             "prototype_weights": dict(row["prototype_weights"]),
             "original_mean_lower": row["predictors"][PRIMARY_PREDICTOR],
             "mean_gain": row["outcome"]["mean_gain"], "stderr": row["outcome"]["stderr"],
             "ci_low": row["outcome"]["ci_low"], "ci_high": row["outcome"]["ci_high"],
             "sign": row["outcome"]["sign"], "replicate_gains": row["outcome"]["replicate_gains"],
             "prototype_weighted_replicate_gains": row["outcome"]["prototype_weighted_replicate_gains"],
             "prototype_weighted_mean_gain": row["outcome"]["prototype_weighted_mean_gain"],
             "prototype_weighted_sign": row["outcome"]["prototype_weighted_sign"],
             "accepted": row["accepted"]}
            for row in rows
        ],
        "excluded_candidate_records": [
            dict(row)
            for row in branch["excluded_candidates"]
        ],
    }


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    return {
        "estimable_count": len(present), "expected_count": len(values),
        "mean": statistics.fmean(present) if present else None,
        "range": [min(present), max(present)] if present else None,
    }


def analyze_cohort(cohort: Any) -> dict[str, Any]:
    clean = validate_cohort(cohort)
    branch_lookup = {(row["seed"], row["budget"]): row for row in clean["records"]}
    cells = [
        _cell_analysis(branch_lookup[(seed, budget)])
        for budget in clean["budgets"] for seed in clean["seeds"]
    ]
    aggregate_by_budget: dict[str, Any] = {}
    main_heuristic_applicable = clean["mode"] == "main"
    for budget in clean["budgets"]:
        budget_cells = [cell for cell in cells if cell["budget"] == budget]
        primary_summary = _summary([cell["predictor_rho_vs_mean_gain"][PRIMARY_PREDICTOR]["rho"] for cell in budget_cells])
        partial_summary = _summary([cell["primary_partial_rho_controlling_group"]["rho"] for cell in budget_cells])
        prototype_summary = _summary([cell["primary_rho_vs_prototype_weighted_mean_gain"]["rho"] for cell in budget_cells])
        positive_rho_count = sum(
            cell["predictor_rho_vs_mean_gain"][PRIMARY_PREDICTOR]["rho"] is not None
            and cell["predictor_rho_vs_mean_gain"][PRIMARY_PREDICTOR]["rho"] > 0.0
            for cell in budget_cells
        )
        criterion = {
            "applicable": main_heuristic_applicable,
            "all_three_cells_estimable": primary_summary["estimable_count"] == 3 if main_heuristic_applicable else None,
            "mean_rho_at_least_0_3": primary_summary["mean"] is not None and primary_summary["mean"] >= 0.3 if main_heuristic_applicable else None,
            "at_least_two_of_three_seed_rhos_positive": positive_rho_count >= 2 if main_heuristic_applicable else None,
            "mean_partial_rho_positive": partial_summary["mean"] is not None and partial_summary["mean"] > 0.0 if main_heuristic_applicable else None,
        }
        aggregate_by_budget[budget] = {
            "seed_level_primary_rho": primary_summary,
            "seed_level_partial_rho_controlling_group": partial_summary,
            "seed_level_prototype_weighted_rho": prototype_summary,
            "positive_seed_primary_rho_count": positive_rho_count,
            "favorable_heuristic_checks": criterion,
            "favorable_heuristic_met": all(value for key, value in criterion.items() if key != "applicable") if main_heuristic_applicable else None,
        }
    analysis = {
        "format": "score_calibration_analysis_v1", "status": "completed", "analysis_role": "diagnostic",
        "estimand": "conditional 128-step frozen-policy gain on later natural recurrence",
        "statistical_unit": "seed within budget; candidates and recurrence replicates are not independent seeds",
        "protocol": {
            "score_freeze": "all write_tests within the first 6 bouts and step <= 12288, before any recurrence",
            "primary_target_weighting": "equal weight over unique source contexts in the scored archive",
            "secondary_target_weighting": "prototype-reservoir weights",
            "recurrence_design": "four actual recurrence state anchors at offsets 0, 64, 128, and 192; 8 paired runs x 128 raw steps per policy arm",
            "common_randomness": "common per-step action uniforms and reset-ordinal seeds",
            "intervals": "pointwise 95% Student t intervals with 7 degrees of freedom over paired replicate aggregates",
            "interval_scope": "conditional on the observed contexts and recurrence anchors; pointwise intervals are not simultaneous across candidates",
            "sign_threshold_epsilon": 0.0,
            "sign_rule": "beneficial if CI low > 0; harmful if CI high < 0; otherwise ambiguous",
            "scope_limit": "conditional frozen-policy gain only; not adaptation, final return, complete write benefit, or long-term online effect",
        "inference_limit": "no p-values, no confidence interval across seeds, and no pooling seed-by-budget cells as independent seeds",
        },
        "cohort": {
            "config": clean["config"], "audit": clean["audit"], "cost": clean["cost"],
            "mode": clean["mode"], "seeds": clean["seeds"], "budgets": clean["budgets"],
            "cpu_s": clean["cpu_s"], "wall_s": clean["wall_s"],
            "branch_count": len(clean["records"]),
            "captured_test_count": clean["captured_tests"],
            "candidate_count": sum(len(row["candidates"]) for row in clean["records"]),
            "excluded_candidate_count": sum(len(row["excluded_candidates"]) for row in clean["records"]),
        },
        "cells": cells,
        "aggregate_by_budget": aggregate_by_budget,
        "favorable_heuristic": {
            "criterion": "exploratory heuristic: both budgets must have all 3 seed cells estimable, mean primary rho >= 0.3, at least 2 of 3 positive seed rhos, and mean partial rho controlling group > 0",
            "applicable": main_heuristic_applicable,
            "met_for_all_budgets": all(aggregate_by_budget[budget]["favorable_heuristic_met"] for budget in clean["budgets"]) if main_heuristic_applicable else None,
            "is_significance_test": False,
        },
    }
    return analysis


def _fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not estimable"
    return f"{float(value):.{digits}f}"


def render_markdown(analysis: Mapping[str, Any]) -> str:
    is_pilot = analysis["cohort"]["mode"] == "pilot"
    pilot_banner = "This is a single-seed pilot pipeline diagnostic, not main-cohort evidence."
    scope_note = (
        "This pilot has one seed; candidate and replicate rows do not add independent seeds."
        if is_pilot else
        "The 3 seeds within each budget are the statistical units; candidates and the 6 seed-by-budget cells are not treated as independent seeds."
    )
    lines = [
        "# Frozen-policy score calibration",
        "",
        *( [pilot_banner, ""] if is_pilot else [] ),
        "This diagnostic tests whether the frozen retention-surplus score predicts actual benefit from inserting the candidate into a frozen policy at a later natural recurrence.",
        "",
        "The target is the conditional 128-step frozen-policy gain. It does not estimate adaptation, final return, the complete write benefit, or long-term online effect.",
        "",
        "Scores are frozen for all `write_tests` in the first 6 bouts at steps up to 12,288, before any recurrence. The primary outcome weights unique source contexts in the scored archive equally. Prototype-reservoir weighting is secondary. Each candidate is evaluated from four observed recurrence states at offsets 0, 64, 128, and 192, with 8 paired runs of 128 raw steps per policy arm and common action uniforms and reset-ordinal seeds.",
        "",
        "Outcome labels use pointwise 95% Student t intervals with 7 degrees of freedom over the 8 paired replicate aggregates. Epsilon is 0: beneficial means the lower endpoint is above zero; harmful means the upper endpoint is below zero; all other cases are ambiguous. These labels are conditional on the observed contexts and recurrence anchors; the intervals are not simultaneous across candidates. No p-values or confidence intervals across seeds are reported. " + scope_note,
        "",
        "Ambiguous outcomes are retained and counted separately. `original_mean_lower > 0` is a conservative approval gate; a non-positive lower score means not approved, not a prediction of harm. The report counts approvals with MC-labeled harm and non-approvals with MC-labeled benefit (opportunity loss). Writer acceptance is reported as a separate gate classification. Precision and recall are descriptive and marked not estimable when their denominator is zero. AUROC is descriptive over resolved beneficial/harmful outcomes only.",
        "",
        "## Seed-level results",
        "",
        "| Budget | Seed | Candidates | Excluded | Primary rho | Partial rho by group | Prototype-weighted rho | Score approved but MC harmful | Score not approved but MC beneficial | Score precision / recall | Ambiguous | Gate accepted but MC harmful | Gate not accepted but MC beneficial | Gate precision / recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    cells = analysis["cells"]
    for cell in cells:
        score = cell["primary_score_approval_gate"]
        gate = cell["accepted_gate"]
        lines.append(
            f"| {cell['budget']} | {cell['seed']} | {cell['candidate_count']} | {cell['excluded_candidate_count']} | "
            f"{_fmt_number(cell['predictor_rho_vs_mean_gain'][PRIMARY_PREDICTOR]['rho'])} | "
            f"{_fmt_number(cell['primary_partial_rho_controlling_group']['rho'])} | "
            f"{_fmt_number(cell['primary_rho_vs_prototype_weighted_mean_gain']['rho'])} | "
            f"{score['approved_but_mc_harmful_count']} | {score['not_approved_but_mc_beneficial_count']} | "
            f"{_fmt_number(score['precision'])} / {_fmt_number(score['recall'])} | {score['n_ambiguous_outcomes']} | "
            f"{gate['accepted_but_mc_harmful_count']} | {gate['not_accepted_but_mc_beneficial_count']} | "
            f"{_fmt_number(gate['precision'])} / {_fmt_number(gate['recall'])} |"
        )
    lines.extend([
        "",
        "## Seed summaries",
        "",
        f"Means and ranges below summarize the {len(analysis['cohort']['seeds'])} declared seed-level values separately within each budget. They are descriptive, with no across-seed interval or p-value.",
        "",
        "| Budget | Primary rho mean (range) | Estimable seeds | Partial rho mean (range) | Prototype rho mean (range) | Positive primary rho seeds | Heuristic checks all met |",
        "| --- | --- | ---: | --- | --- | ---: | --- |",
    ])
    for budget, result in analysis["aggregate_by_budget"].items():
        def describe(summary: Mapping[str, Any]) -> str:
            value = _fmt_number(summary["mean"])
            limits = summary["range"]
            span = "not estimable" if limits is None else f"[{_fmt_number(limits[0])}, {_fmt_number(limits[1])}]"
            return f"{value} ({span})"
        lines.append(
            f"| {budget} | {describe(result['seed_level_primary_rho'])} | "
            f"{result['seed_level_primary_rho']['estimable_count']}/{result['seed_level_primary_rho']['expected_count']} | "
            f"{describe(result['seed_level_partial_rho_controlling_group'])} | "
            f"{describe(result['seed_level_prototype_weighted_rho'])} | "
            f"{result['positive_seed_primary_rho_count']}/{result['seed_level_primary_rho']['expected_count']} | {result['favorable_heuristic_met']} |"
        )
    lines.extend([
        "",
        "## Predictor diagnostics",
        "",
        "These descriptive Spearman correlations compare each recorded predictor with the equal-context mean gain within each seed and budget. Missing predictor values are excluded for that predictor and reported in `analysis.json`. No threshold is fitted or tuned.",
        "",
        "| Budget | Seed | " + " | ".join(PREDICTORS) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in PREDICTORS) + " |",
    ])
    for cell in cells:
        values = [cell["predictor_rho_vs_mean_gain"][predictor]["rho"] for predictor in PREDICTORS]
        lines.append(f"| {cell['budget']} | {cell['seed']} | " + " | ".join(_fmt_number(value) for value in values) + " |")
    lines.extend([
        "",
        "## Exploratory favorable heuristic",
        "",
        analysis["favorable_heuristic"]["criterion"].capitalize() + ".",
        "",
        (
            f"Met for both budgets: **{analysis['favorable_heuristic']['met_for_all_budgets']}**. This is an exploratory screening rule, not a significance test or confirmatory claim."
            if analysis["favorable_heuristic"]["applicable"]
            else "The main-cohort exploratory heuristic is not evaluated on the pilot cohort."
        ),
        "",
        "## Cohort audit and cost",
        "",
        f"Validated {analysis['cohort']['branch_count']} completed seed-by-budget branches, {analysis['cohort']['candidate_count']} evaluated candidates, {analysis['cohort']['excluded_candidate_count']} excluded tests, and {analysis['cohort']['captured_test_count']} captured write tests. Cohort audit passed: **{analysis['cohort']['audit']['passed']}**.",
        "",
        f"Interaction cost: {analysis['cohort']['cost']['learner_steps']} learner steps + {analysis['cohort']['cost']['observer_steps']} observer steps = {analysis['cohort']['cost']['new_total_interactions']} total new interactions. CPU time: {analysis['cohort']['cpu_s']:.3f} s; wall time: {analysis['cohort']['wall_s']:.3f} s.",
        "",
        "The figure shows candidate-level primary scores and conditional gains. Zero lines mark the score and outcome sign boundaries. It does not show candidate-level confidence intervals; the inferential labels use the 8 paired replicate aggregates described above.",
        "",
        "Plotting software: Matplotlib, cited as J. D. Hunter, \u201cMatplotlib: A 2D Graphics Environment,\u201d *Computing in Science & Engineering* 9(3), 90\u201395 (2007). [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55). See [Matplotlib\u2019s citation guidance](https://matplotlib.org/stable/project/citing.html).",
        "Procedural plotting guidance: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).",
        "",
    ])
    return "\n".join(lines)


def render_figure(analysis: Mapping[str, Any]) -> tuple[bytes, bytes]:
    seeds = analysis["cohort"]["seeds"]
    budgets = analysis["cohort"]["budgets"]
    colors = {seed: plt.get_cmap("tab10")(index) for index, seed in enumerate(seeds)}
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.75 * len(budgets), 5.2), constrained_layout=True, sharex=False, sharey=True, squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        for seed in seeds:
            cell = next(row for row in analysis["cells"] if row["budget"] == budget and row["seed"] == seed)
            points = [row for row in cell["candidate_outcomes"] if row["original_mean_lower"] is not None]
            ax.scatter(
                [row["original_mean_lower"] for row in points],
                [row["mean_gain"] for row in points],
                color=colors[seed], label=f"seed {seed}", alpha=0.78, edgecolor="white", linewidth=0.5, s=42,
            )
        ax.axhline(0.0, color="#505050", linewidth=0.9, linestyle="--")
        ax.axvline(0.0, color="#505050", linewidth=0.9, linestyle="--")
        ax.set_title(f"{budget.title()} budget")
        ax.set_xlabel("Frozen original mean lower score")
        ax.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("Conditional 128-step frozen-policy gain")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
    png_buffer, pdf_buffer = io.BytesIO(), io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_buffer.getvalue(), pdf_buffer.getvalue()


def analyze(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    cohort = _read_cohort(input_path)
    analysis = analyze_cohort(cohort)
    output = Path(output_path)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"refusing to write into existing non-empty report path: {output}")
    markdown = render_markdown(analysis)
    png, pdf = render_figure(analysis)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(markdown, encoding="utf-8")
    (output / "figure.png").write_bytes(png)
    (output / "figure.pdf").write_bytes(pdf)
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="completed cohort directory containing summary.json")
    parser.add_argument("--output", required=True, type=Path, help="new empty directory for analysis and figures")
    args = parser.parse_args(argv)
    try:
        analyze(args.input, args.output)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote score calibration report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

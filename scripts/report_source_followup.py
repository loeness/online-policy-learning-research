"""Summarize the frozen source-dose and learning-interaction follow-up cohorts.

This is an analysis-only entry point. It reads runner summaries, decisions, and
branch summaries; it never imports the environment or executes a runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FORMAT = "source_followup_analysis_v1"
SOURCE_FORMAT = "storage_source_dose_summary_v1"
ACTIONS = ("keep", "replace", "mean", "none")
SELECTORS = ("frequency", "recency", "surrogate_gain", "surrogate_lower")
RELATIONS = ("dynamics", "reward_conflict")
REGIMES = ("old_major", "new_major")
DOSE_BRANCH_KEYS = {"old2048": "base_branches", "old1024": "source_1024_branches"}
PRACTICAL = 0.002
BOOTSTRAP_REPS = 10000
COLORS = {"dynamics": "#336B87", "reward_conflict": "#D17A22"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{where} must be finite numeric data") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite numeric data")
    return result


def _integer(value: Any, where: str) -> int:
    number = _finite(value, where)
    if int(number) != number:
        raise ValueError(f"{where} must be an integer")
    return int(number)


def _source_dir(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    summary_path = source if source.is_file() else source / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"source summary not found: {summary_path}")
    summary = _json(summary_path)
    if summary.get("format") != SOURCE_FORMAT:
        raise ValueError(f"expected {SOURCE_FORMAT}, got {summary.get('format')!r}")
    return summary_path.parent, summary


def _branch_reward(row: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    fmt = str(row.get("format", ""))
    is_alias = fmt == "storage_source_dose_branch_alias_v1" or row.get("independent_observation") is False
    if is_alias:
        total_value = row.get("logical_reward_total")
        step_value = row.get("logical_environment_steps")
        if step_value is None:
            counts = row.get("logical_counts")
            step_value = counts.get("environment_steps") if isinstance(counts, Mapping) else None
        if total_value is None or step_value is None:
            raise ValueError(f"{where}: alias must carry logical reward total and logical environment steps")
        actual_steps_value = row.get("environment_steps")
        if actual_steps_value is None:
            actual_counts = row.get("counts")
            actual_steps_value = actual_counts.get("environment_steps") if isinstance(actual_counts, Mapping) else None
        if actual_steps_value is None:
            raise ValueError(f"{where}: alias is missing measured actual environment steps")
        actual_steps = _integer(actual_steps_value, f"{where}.environment_steps")
        if actual_steps != 0:
            raise ValueError(f"{where}: an alias must record zero actual environment steps")
        total = _finite(total_value, f"{where}.logical_reward_total")
        logical_steps = _integer(step_value, f"{where}.logical_environment_steps")
        if logical_steps <= 0:
            raise ValueError(f"{where}: alias logical duration must be positive")
        return {
            "reward_total": total,
            "reward_per_step": total / logical_steps,
            "logical_environment_steps": logical_steps,
            "actual_environment_steps": actual_steps,
            "alias": True,
            "reuse_of": row.get("reuse_of"),
            "independent_observation": False,
            "costs": row.get("actual_costs", row.get("costs")),
            "capacity": row.get("capacity"),
            "counts": row.get("logical_counts"),
            "codec_max_coefficient_error": row.get("codec_max_coefficient_error"),
            "codec_effective_errors": row.get("codec_effective_errors"),
            "final_codec_max_coefficient_error": row.get("final_codec_max_coefficient_error"),
            "final_codec_effective_errors": row.get("final_codec_effective_errors"),
            "prequantized_behavior_drift": row.get("prequantized_behavior_drift"),
            "postquantized_behavior_drift": row.get("postquantized_behavior_drift"),
        }

    total_value = row.get("reward_total")
    step_value = row.get("environment_steps")
    counts = row.get("counts")
    if step_value is None and isinstance(counts, Mapping):
        step_value = counts.get("environment_steps")
    if total_value is None or step_value is None:
        raise ValueError(f"{where}: branch is missing reward_total or measured environment_steps")
    total = _finite(total_value, f"{where}.reward_total")
    steps = _integer(step_value, f"{where}.environment_steps")
    if steps <= 0:
        raise ValueError(f"{where}: branch duration must be positive")
    return {
        "reward_total": total,
        "reward_per_step": total / steps,
        "logical_environment_steps": steps,
        "actual_environment_steps": steps,
        "alias": False,
        "reuse_of": None,
        "independent_observation": True,
        "costs": row.get("costs"),
        "capacity": row.get("capacity"),
        "counts": counts,
        "codec_max_coefficient_error": row.get("codec_max_coefficient_error"),
        "codec_effective_errors": row.get("codec_effective_errors"),
        "final_codec_max_coefficient_error": row.get("final_codec_max_coefficient_error"),
        "final_codec_effective_errors": row.get("final_codec_effective_errors"),
        "prequantized_behavior_drift": row.get("prequantized_behavior_drift"),
        "postquantized_behavior_drift": row.get("postquantized_behavior_drift"),
    }


def _branch_index(rows: Any, *, where: str) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    issues: list[str] = []
    if not isinstance(rows, list):
        return result, [f"{where} must be a branch list"]
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            issues.append(f"{where}[{index}] is not an object")
            continue
        action, future = raw.get("action"), raw.get("future_regime")
        if action not in ACTIONS or future not in REGIMES:
            issues.append(f"{where}[{index}] has an invalid action or future_regime")
            continue
        key = (str(action), str(future))
        if key in result:
            issues.append(f"{where} contains duplicate branch {key}")
            continue
        try:
            result[key] = _branch_reward(raw, where=f"{where}[{index}]")
        except ValueError as exc:
            issues.append(str(exc))
    return result, issues


def _selector_map(decision: Mapping[str, Any] | None) -> tuple[dict[str, str], str | None]:
    if not isinstance(decision, Mapping):
        return {}, None
    candidates = [
        decision.get("past_only_selectors"),
        decision.get("selector_actions"),
        decision.get("selectors"),
    ]
    diagnostics = decision.get("selector_diagnostics")
    if isinstance(diagnostics, Mapping):
        candidates.append(diagnostics.get("past_only_selectors"))
    chosen = next((item for item in candidates if isinstance(item, Mapping)), None)
    if chosen is None:
        return {}, None
    actions = {name: str(chosen[name]) for name in SELECTORS if chosen.get(name) in ("keep", "replace")}
    candidate_source = decision.get("selector_input_candidate_set")
    if candidate_source is None:
        candidate_source = decision.get("selector_candidate_set")
    return actions, None if candidate_source is None else str(candidate_source)


def _load_decisions(root: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("case_id", ""))
    case_dir = root / "cases" / case_id
    base_path = case_dir / "decision.json"
    dose_path = case_dir / "dose_1024" / "decision.json"
    base_decision = _json(base_path) if base_path.is_file() else None
    dose_decision = _json(dose_path) if dose_path.is_file() else None
    return {"old2048": base_decision, "old1024": dose_decision}


def _normalize_case(root: Path, raw: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(raw.get("case_id", ""))
    if not case_id:
        raise ValueError("case is missing case_id")
    seed = _integer(raw.get("seed"), f"{case_id}.seed")
    relation, history = raw.get("pair_kind"), raw.get("history_regime")
    if relation not in RELATIONS or history not in REGIMES:
        raise ValueError(f"{case_id}: pair_kind/history_regime is outside the frozen protocol")
    branches: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    issues = [str(item) for item in raw.get("errors", [])]
    for dose, key in DOSE_BRANCH_KEYS.items():
        branch_map, branch_issues = _branch_index(raw.get(key), where=f"{case_id}.{key}")
        branches[dose] = branch_map
        issues.extend(branch_issues)
        for action in ACTIONS:
            for future in REGIMES:
                if (action, future) not in branch_map:
                    issues.append(f"{case_id}.{dose}: missing {action}/{future} logical branch")
    decisions = _load_decisions(root, raw)
    selectors: dict[str, Any] = {}
    for dose in DOSE_BRANCH_KEYS:
        action_map, candidate_source = _selector_map(decisions.get(dose))
        selectors[dose] = {"actions": action_map, "candidate_source": candidate_source}
    # The low-dose runner may not yet emit a fresh selector map. Frequency and
    # recency depend only on the realized prefix history, while gain/lower do
    # depend on the old candidate. Only carry those two across an explicitly
    # old2048-sourced map; never relabel gain/lower as a 1024-dose decision.
    low = selectors["old1024"]
    high = selectors["old2048"]
    low_actions = low["actions"]
    if low["candidate_source"] == "old_candidate_2048":
        for name in ("surrogate_gain", "surrogate_lower"):
            low_actions.pop(name, None)
        low["selection_note"] = "gain/lower selector omitted: saved map was evaluated with old_candidate_2048"
        for name in ("frequency", "recency"):
            if name not in low_actions and name in high["actions"]:
                low_actions[name] = high["actions"][name]
        low["selection_source"] = "history-only selectors copied from old2048; gain/lower unavailable"
    else:
        low["selection_note"] = None
        low["selection_source"] = "dose decision map"
    high["selection_source"] = "old2048 decision map"

    return {
        "case_id": case_id,
        "seed": seed,
        "relation": str(relation),
        "history": str(history),
        "branches": branches,
        "selectors": selectors,
        "prefix": raw.get("prefix"),
        "costs": raw.get("costs"),
        "errors": issues,
    }


def _bootstrap(values: Mapping[int, float], *, seed: int, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": None, "n_seeds": 0, "positive_seeds": 0, "per_seed": {}}
    per_seed = {str(key): float(values[key]) for key in sorted(values)}
    ordered = np.asarray([values[key] for key in sorted(values)], dtype=np.float64)
    mean = float(ordered.mean())
    positive = int(np.count_nonzero(ordered > 0.0))
    if len(ordered) < 2:
        ci = None
    else:
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(ordered), size=(reps, len(ordered)))
        means = ordered[sampled].mean(axis=1)
        low, high = np.quantile(means, (0.025, 0.975))
        ci = [float(low), float(high)]
    return {"mean": mean, "ci95": ci, "n_seeds": len(ordered), "positive_seeds": positive, "per_seed": per_seed}


def _estimate(values: Mapping[int, float], *, seed: int, label: str) -> dict[str, Any]:
    result = _bootstrap(values, seed=seed)
    result["interpretation"] = label
    return result


def _dose_r(case: Mapping[str, Any], dose: str, action: str, future: str) -> float:
    return float(case["branches"][dose][(action, future)]["reward_per_step"])


def _complete_cases_by_relation(
    cases: Sequence[Mapping[str, Any]], history_regimes: Sequence[str] = REGIMES
) -> tuple[dict[str, dict[int, dict[str, Mapping[str, Any]]]], dict[str, list[int]]]:
    grouped: dict[str, dict[int, dict[str, Mapping[str, Any]]]] = {relation: {} for relation in RELATIONS}
    for case in cases:
        grouped[case["relation"]].setdefault(case["seed"], {})[case["history"]] = case
    complete: dict[str, dict[int, dict[str, Mapping[str, Any]]]] = {relation: {} for relation in RELATIONS}
    missing: dict[str, list[int]] = {relation: [] for relation in RELATIONS}
    for relation, by_seed in grouped.items():
        for seed, by_history in by_seed.items():
            good = True
            for history in history_regimes:
                case = by_history.get(history)
                if case is None or case["errors"]:
                    good = False
                    break
                if any(len(case["branches"][dose]) != len(ACTIONS) * len(REGIMES) for dose in DOSE_BRANCH_KEYS):
                    good = False
                    break
            if good:
                complete[relation][seed] = by_history
            else:
                missing[relation].append(seed)
    return complete, missing


def _dose_estimands(
    complete: Mapping[int, Mapping[str, Mapping[str, Any]]], *,
    rng_seed: int, history_regimes: Sequence[str] = REGIMES,
) -> dict[str, Any]:
    result: dict[str, Any] = {"D_by_future": {}, "I": {}, "B": {}, "Q": {}}
    for dose in ("old1024", "old2048"):
        for future in REGIMES:
            values: dict[int, float] = {}
            for seed, by_history in complete.items():
                values[seed] = float(np.mean([
                    _dose_r(by_history[history], dose, "replace", future)
                    - _dose_r(by_history[history], dose, "keep", future)
                    for history in history_regimes
                ]))
            label = f"D_{dose}_{future} = R_replace - R_keep, averaged over histories within seed"
            result["D_by_future"][f"{dose}/{future}"] = _estimate(values, seed=rng_seed, label=label)

    i_values: dict[str, dict[int, float]] = {dose: {} for dose in ("old1024", "old2048")}
    for dose in ("old1024", "old2048"):
        for seed, by_history in complete.items():
            contrasts = []
            for history in history_regimes:
                case = by_history[history]
                d_new = _dose_r(case, dose, "replace", "new_major") - _dose_r(case, dose, "keep", "new_major")
                d_old = _dose_r(case, dose, "replace", "old_major") - _dose_r(case, dose, "keep", "old_major")
                contrasts.append(d_new - d_old)
            i_values[dose][seed] = float(np.mean(contrasts))
        result["I"][dose] = _estimate(
            i_values[dose], seed=rng_seed,
            label=f"I_{dose}: future new-major minus old-major D, averaged over histories",
        )
    result["Q"] = _estimate(
        {seed: i_values["old1024"][seed] - i_values["old2048"][seed] for seed in complete},
        seed=rng_seed, label="Q = I_old1024 - I_old2048; no directional hypothesis",
    )

    for kind, future_condition in (("all", None), ("primary", "same"), ("stress", "different")):
        b_values: dict[int, float] = {}
        for seed, by_history in complete.items():
            cells = []
            for history in history_regimes:
                for future in REGIMES:
                    if future_condition == "same" and future != history:
                        continue
                    if future_condition == "different" and future == history:
                        continue
                    case = by_history[history]
                    d_low = _dose_r(case, "old1024", "replace", future) - _dose_r(case, "old1024", "keep", future)
                    d_high = _dose_r(case, "old2048", "replace", future) - _dose_r(case, "old2048", "keep", future)
                    cells.append(d_low - d_high)
            if cells:
                b_values[seed] = float(np.mean(cells))
        est = _estimate(
            b_values, seed=rng_seed,
            label=f"B_{kind}: mean_h,f[(replace-keep)_old1024 - (replace-keep)_old2048]",
        )
        if kind != "stress" and est["mean"] is not None:
            if est["mean"] >= PRACTICAL and est["positive_seeds"] >= 6 and est["ci95"] is not None and est["ci95"][0] > 0:
                classification = "supports the predeclared positive magnitude and direction"
            elif est["ci95"] is not None and est["ci95"][1] < PRACTICAL:
                classification = "does not support the predeclared +0.002 reward/step magnitude"
            else:
                classification = "uncertain under the predeclared criterion"
        else:
            classification = "descriptive stress condition" if kind == "stress" else "not estimable"
        est["predeclared_classification"] = classification
        result["B"][kind] = est

    flips: dict[str, Any] = {}
    for dose in ("old1024", "old2048"):
        old_result = result["D_by_future"][f"{dose}/old_major"]
        new_result = result["D_by_future"][f"{dose}/new_major"]
        old, new = old_result["mean"], new_result["mean"]
        old_per_seed, new_per_seed = old_result["per_seed"], new_result["per_seed"]
        paired_seeds = sorted(set(old_per_seed) & set(new_per_seed))
        per_seed_reversals = {
            seed: float(old_per_seed[seed]) * float(new_per_seed[seed]) < 0.0
            for seed in paired_seeds
        }
        flips[dose] = {
            "old_major_sign": None if old is None else int(np.sign(old)),
            "new_major_sign": None if new is None else int(np.sign(new)),
            "preference_reversal": bool(old is not None and new is not None and old * new < 0.0),
            "per_seed_reversal_count": sum(per_seed_reversals.values()),
            "per_seed_count": len(per_seed_reversals),
            "per_seed_reversals": per_seed_reversals,
        }
    result["preference_reversal_by_dose"] = flips
    return result


def _selector_estimands(
    complete: Mapping[int, Mapping[str, Mapping[str, Any]]], *,
    relation: str, rng_seed: int, history_regimes: Sequence[str] = REGIMES,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dose in ("old2048", "old1024"):
        output[dose] = {}
        available = {
            selector
            for case_set in complete.values()
            for case in case_set.values()
            for selector in case["selectors"][dose]["actions"]
        }
        for selector in SELECTORS:
            output[dose][selector] = {}
            if selector not in available:
                output[dose][selector] = {
                    "available": False, "reason": "no valid dose-specific selector action map"
                }
                continue
            for continuation, primary in (("primary", True), ("stress", False)):
                output[dose][selector][continuation] = {}
                for comparator in ("keep", "replace", "mean"):
                    values: dict[int, float] = {}
                    for seed, by_history in complete.items():
                        cells: list[float] = []
                        valid = True
                        for history in history_regimes:
                            case = by_history[history]
                            selection = case["selectors"][dose]
                            action = selection["actions"].get(selector)
                            if action not in ("keep", "replace"):
                                valid = False
                                break
                            for future in REGIMES:
                                if (future == history) != primary:
                                    continue
                                cells.append(
                                    _dose_r(case, dose, action, future)
                                    - _dose_r(case, dose, comparator, future)
                                )
                        if valid and cells:
                            values[seed] = float(np.mean(cells))
                    if not values:
                        output[dose][selector][continuation][comparator] = {
                            "available": False, "reason": "selector data incomplete"
                        }
                    else:
                        output[dose][selector][continuation][comparator] = {
                            "available": True,
                            "estimate": _estimate(
                                values, seed=rng_seed,
                                label=f"{selector} selected action minus {comparator}; {continuation} continuations",
                            ),
                            "selection_sources": sorted({
                                str(case["selectors"][dose].get("selection_source", "unknown"))
                                for case_set in complete.values()
                                for case in case_set.values()
                                if selector in case["selectors"][dose]["actions"]
                            }),
                        }
    return output

def _read_capture(root: Path, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    candidate_coeff_counts: list[int] = []
    observer_json_bytes: list[int] = []
    capture_timing: list[str] = []
    for case in cases:
        case_id = str(case["case_id"])
        path = root / "cases" / case_id / "dose_1024" / "observer_candidate.json"
        if not path.is_file():
            missing.append(case_id)
            continue
        payload = _json(path)
        candidate_coeff_counts.append(_integer(payload.get("candidate_coefficients"), f"{case_id}.candidate_coefficients"))
        observer_json_bytes.append(path.stat().st_size)
        if payload.get("capture_timing") is not None:
            capture_timing.append(str(payload["capture_timing"]))
        sources = payload.get("sources")
        if not isinstance(sources, Mapping):
            missing.append(f"{case_id}:sources")
            continue
        for source, raw in sources.items():
            if not isinstance(raw, Mapping):
                continue
            records.append({
                "case_id": case_id,
                "source_variant": str(source),
                "task": raw.get("task"),
                "requested_bout_step": payload.get("source_capture_steps"),
                "actual_bout_step": raw.get("bout_step"),
                "global_environment_steps": raw.get("global_environment_steps"),
                "fitted_steps": raw.get("fitted_steps"),
                "rollouts_trained": raw.get("rollouts_trained"),
                "minibatch_updates": raw.get("minibatch_updates"),
                "last_rollout_updates": raw.get("last_rollout_updates"),
                "rollout_index": raw.get("rollout_index"),
                "rollout_rows": raw.get("rollout_rows"),
                "slot": raw.get("slot"),
                "generation": raw.get("generation"),
            })
    numeric_fields = ("actual_bout_step", "global_environment_steps", "fitted_steps", "rollouts_trained", "minibatch_updates")
    ranges: dict[str, Any] = {}
    for field in numeric_fields:
        vals = [_finite(row[field], f"capture.{field}") for row in records if row.get(field) is not None]
        ranges[field] = None if not vals else {"min": min(vals), "max": max(vals), "mean": statistics.mean(vals)}
    return {
        "capture_timing": sorted(set(capture_timing)) or None,
        "records": records,
        "ranges": ranges,
        "requested_capture_steps": sorted({row["requested_bout_step"] for row in records if row.get("requested_bout_step") is not None}),
        "candidate_coefficients": sorted(set(candidate_coeff_counts)),
        "candidate_float32_bytes_per_case": 36 if candidate_coeff_counts and set(candidate_coeff_counts) == {9} else None,
        "observer_candidate_json_bytes_range": None if not observer_json_bytes else [min(observer_json_bytes), max(observer_json_bytes)],
        "missing_case_ids": missing,
    }


def _capacity_summary(root: Path, cases: Sequence[Mapping[str, Any]], package_budget: Any, actor_budget: Any) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    for case in cases:
        case_dir = root / "cases" / str(case["case_id"])
        for rel in ("prefix/summary.json",):
            path = case_dir / rel
            if path.is_file():
                data = _json(path)
                capacity = data.get("capacity")
                if isinstance(capacity, Mapping):
                    observed.append({"case_id": case["case_id"], "branch": "prefix", **capacity})
        for dose, dirname in (("old2048", "branches"), ("old1024", "dose_1024/branches")):
            branch_root = case_dir / dirname
            if not branch_root.is_dir():
                continue
            for branch_dir in branch_root.iterdir():
                path = branch_dir / "summary.json"
                if not path.is_file():
                    continue
                data = _json(path)
                capacity = data.get("capacity_after_action", data.get("capacity"))
                if isinstance(capacity, Mapping):
                    observed.append({"case_id": case["case_id"], "branch": f"{dose}/{branch_dir.name}", **capacity})
    key = "full_persistent_bytes"
    values = [int(item[key]) for item in observed if item.get(key) is not None]
    budgets = [int(item["package_budget_bytes"]) for item in observed if item.get("package_budget_bytes") is not None]
    actor_values = [int(item["actor_bytes"]) for item in observed if item.get("actor_bytes") is not None]
    return {
        "package_budget_bytes_config": package_budget,
        "observed_package_count": len(observed),
        "capacity_records": observed,
        "full_persistent_bytes_range": None if not values else [min(values), max(values)],
        "package_budget_bytes_observed_range": None if not budgets else [min(budgets), max(budgets)],
        "actor_payload_budget_bytes_config": actor_budget,
        "actor_payload_bytes_range": None if not actor_values else [min(actor_values), max(actor_values)],
        "over_budget_count": None if not values else sum(
            1 for item in observed
            if item.get(key) is not None and item.get("package_budget_bytes") is not None
            and int(item[key]) > int(item["package_budget_bytes"])
        ),
        "missing_capacity_count": sum(key not in item for item in observed),
    }


def _cost_integrity(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    costs = summary.get("costs", {})
    if not isinstance(costs, Mapping):
        costs = {}
    branch_alias_count = sum(
        1 for case in cases
        for dose in DOSE_BRANCH_KEYS
        for branch in case["branches"][dose].values()
        if branch["alias"]
    )
    branch_logical_count = sum(len(case["branches"][dose]) for case in cases for dose in DOSE_BRANCH_KEYS)
    branch_physical_count = branch_logical_count - branch_alias_count
    errors = [
        {"case_id": case["case_id"], "errors": case["errors"]}
        for case in cases if case["errors"]
    ]
    top_errors = summary.get("errors", [])
    return {
        "top_level_costs": dict(costs),
        "case_costs": {case["case_id"]: case.get("costs") for case in cases},
        "logical_future_branch_count_observed": branch_logical_count,
        "physical_future_branch_count_observed": branch_physical_count,
        "verified_alias_branch_count_observed": branch_alias_count,
        "alias_rule": "alias rewards use logical branch reward and H; actual envsteps must be exactly zero; aliases are not independent observations",
        "errors": errors,
        "top_level_errors": top_errors,
        "complete_case_count": sum(not case["errors"] for case in cases),
    }


def _distribution(values: Sequence[Any]) -> dict[str, Any]:
    numeric = sorted(_finite(value, "distribution value") for value in values if value is not None)
    if not numeric:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "n": len(numeric),
        "min": float(numeric[0]),
        "median": float(np.median(numeric)),
        "p95": float(np.percentile(numeric, 95)),
        "max": float(numeric[-1]),
    }


def _codec_drift_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        ("codec_max_coefficient_error", "codec_max_coefficient_error"),
        ("final_codec_max_coefficient_error", "final_codec_max_coefficient_error"),
        ("prequantized_max_probability_absolute_drift", "prequantized_behavior_drift.max_probability_absolute_drift"),
        ("prequantized_max_kl", "prequantized_behavior_drift.max_kl_reference_to_candidate"),
        ("postquantized_max_probability_absolute_drift", "postquantized_behavior_drift.max_probability_absolute_drift"),
        ("postquantized_max_kl", "postquantized_behavior_drift.max_kl_reference_to_candidate"),
    )
    result: dict[str, Any] = {"definition": "Distributions use physical post-action branch records only; dose aliases are excluded from counts.", "doses": {}}
    for dose in ("old2048", "old1024"):
        action_result: dict[str, Any] = {}
        for action in ACTIONS:
            branches = [
                case["branches"][dose][(action, future)]
                for case in cases
                for future in REGIMES
                if (action, future) in case["branches"][dose]
                and not case["branches"][dose][(action, future)]["alias"]
            ]
            observed: dict[str, list[Any]] = {name: [] for name, _ in metrics}
            effective_errors: list[Any] = []
            final_effective_errors: list[Any] = []
            for branch in branches:
                for name, path in metrics:
                    if "." in path:
                        parent, child = path.split(".", 1)
                        payload = branch.get(parent)
                        value = payload.get(child) if isinstance(payload, Mapping) else None
                    else:
                        value = branch.get(path)
                    if value is not None:
                        observed[name].append(value)
                raw_errors = branch.get("codec_effective_errors")
                if isinstance(raw_errors, Sequence) and not isinstance(raw_errors, (str, bytes)):
                    effective_errors.extend(raw_errors)
                raw_final_errors = branch.get("final_codec_effective_errors")
                if isinstance(raw_final_errors, Sequence) and not isinstance(raw_final_errors, (str, bytes)):
                    final_effective_errors.extend(raw_final_errors)
            action_result[action] = {
                "physical_branch_count": len(branches),
                **{name: _distribution(values) for name, values in observed.items()},
                "codec_effective_error_components": _distribution(effective_errors),
                "final_codec_effective_error_components": _distribution(final_effective_errors),
            }
        result["doses"][dose] = action_result
    return result


def analyze_source(source: str | Path) -> dict[str, Any]:
    root, summary = _source_dir(source)
    config = summary.get("config")
    if not isinstance(config, Mapping):
        cfg_path = root / "config.json"
        config = _json(cfg_path) if cfg_path.is_file() else None
    if not isinstance(config, Mapping):
        raise ValueError("source summary must include a config object or config.json")
    raw_cases = summary.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("source summary.cases must be a list")
    cases = [_normalize_case(root, row) for row in raw_cases if isinstance(row, Mapping)]
    expected_seeds = config.get("seeds", summary.get("seeds", []))
    expected_relations = config.get("pair_kinds", list(RELATIONS))
    expected_histories = config.get("history_regimes", list(REGIMES))
    if not isinstance(expected_histories, list) or not expected_histories or not set(expected_histories) <= set(REGIMES):
        raise ValueError("config.history_regimes must be a non-empty subset of the frozen regimes")
    if not isinstance(expected_seeds, list) or not expected_seeds:
        expected_seeds = sorted({case["seed"] for case in cases})
    expected_case_ids = {
        f"s{int(seed)}_{relation}_{history}"
        for seed in expected_seeds for relation in expected_relations for history in expected_histories
    }
    observed_case_ids = {case["case_id"] for case in cases}
    missing_case_ids = sorted(expected_case_ids - observed_case_ids)
    complete, incomplete_seeds = _complete_cases_by_relation(cases, history_regimes=expected_histories)
    bootstrap_seed = int((config.get("report") or {}).get("bootstrap_seed", 20261018)) if isinstance(config.get("report"), Mapping) else 20261018
    relations: dict[str, Any] = {}
    for relation in expected_relations:
        relation_cases = [case for case in cases if case["relation"] == relation]
        eligible = complete.get(str(relation), {})
        mean_action_rates: dict[str, Any] = {}
        for dose in ("old1024", "old2048"):
            mean_action_rates[dose] = {}
            for action in ACTIONS:
                mean_action_rates[dose][action] = {}
                for future in REGIMES:
                    values = {
                        seed: float(np.mean([
                            _dose_r(by_history[history], dose, action, future)
                            for history in expected_histories
                        ]))
                        for seed, by_history in eligible.items()
                    }
                    mean_action_rates[dose][action][future] = _estimate(
                        values,
                        seed=bootstrap_seed,
                        label=f"mean reward rate: {dose}/{action}/{future}; histories averaged within seed",
                    )
        relations[str(relation)] = {
            "case_count": len(relation_cases),
            "complete_seed_count": len(eligible),
            "expected_seed_count": len(expected_seeds),
            "incomplete_or_missing_seeds": sorted(set(incomplete_seeds.get(str(relation), [])) | {
                int(seed) for seed in expected_seeds if int(seed) not in eligible
            }),
            "estimands": _dose_estimands(eligible, rng_seed=bootstrap_seed, history_regimes=expected_histories),
            "selectors": _selector_estimands(eligible, relation=str(relation), rng_seed=bootstrap_seed, history_regimes=expected_histories),
            "mean_action_reward_rates": mean_action_rates,
        }
    package_budget = config.get("package_budget_bytes")
    replace_alias_count = sum(
        1
        for case in cases
        for future in REGIMES
        if ("replace", future) in case["branches"]["old1024"]
        and case["branches"]["old1024"][("replace", future)]["alias"]
    )
    replace_branch_count = sum(
        1
        for case in cases
        for future in REGIMES
        if ("replace", future) in case["branches"]["old1024"]
    )
    return {
        "format": FORMAT,
        "source_root": str(root),
        "source_format": summary.get("format"),
        "phase": summary.get("phase", config.get("phase", "unknown")),
        "config": dict(config),
        "protocol_estimators": {
            "reward": "cumulative future reward / logical future environment steps",
            "dose_labels": {"old1024": 1024, "old2048": 2048},
            "bootstrap": {"unit": "seed cluster", "resamples": BOOTSTRAP_REPS, "interval": "95% paired percentile", "seed": bootstrap_seed},
            "history_weighting": f"equal weight over configured histories {list(expected_histories)} within each seed before seed-cluster bootstrap",
            "aliases": "logical reward retained, measured actual envsteps reported as zero, never resampled as independent units",
            "candidate_scope": "old_source_1024 is installed only at the decision fork; the common prefix still uses 2048 steps per old source",
        },
        "expected_case_count": len(expected_case_ids),
        "observed_case_count": len(cases),
        "expected_history_regimes": list(expected_histories),
        "missing_case_ids": missing_case_ids,
        "relations": relations,
        "case_integrity": [
            {"case_id": case["case_id"], "seed": case["seed"], "relation": case["relation"], "history": case["history"], "complete": not bool(case["errors"]), "issues": case["errors"]}
            for case in cases
        ],
        "costs_and_integrity": _cost_integrity(summary, cases),
        "replace_dose_reuse": {
            "observed_old1024_replace_branches": replace_branch_count,
            "alias_old1024_replace_branches": replace_alias_count,
            "physical_old1024_replace_branches": replace_branch_count - replace_alias_count,
            "all_observed_replace_branches_reused": replace_branch_count > 0 and replace_alias_count == replace_branch_count,
        },
        "source_capture": _read_capture(root, cases),
        "capacity": _capacity_summary(root, cases, package_budget, config.get("actor_budget_bytes")),
        "codec_probability_drift": _codec_drift_summary(cases),
        "cautions": [
            "This is a fitted-content dose intervention at a common full prefix, not an end-to-end process in which old sources were trained for only 1024 steps.",
            "The dose contrast does not assert equal old/new candidate quality.",
            "Bootstrap intervals are descriptive and do not establish adequate statistical power.",
            "Selector outcomes are evaluated on future branches; future outcomes do not enter selector decisions.",
        ],
    }


def _fmt_estimate(value: Mapping[str, Any]) -> str:
    mean = value.get("mean")
    if mean is None:
        return "not estimable"
    ci = value.get("ci95")
    interval = "no interval (fewer than 2 seed clusters)" if ci is None else f"[{ci[0]:+.5f}, {ci[1]:+.5f}]"
    return f"{mean:+.5f} [{interval}]; n={value.get('n_seeds')}, positive={value.get('positive_seeds')}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _show(value: Any) -> Any:
    return "unknown" if value is None else value


def _fmt_distribution(value: Mapping[str, Any]) -> str:
    if not value.get("n"):
        return "unavailable"
    return (
        f"n={value['n']}; median={value['median']:.3g}; "
        f"p95={value['p95']:.3g}; max={value['max']:.3g}"
    )


def _render_markdown(analysis: Mapping[str, Any], *, ablation: Mapping[str, Any] | None = None) -> str:
    rels = analysis["relations"]
    replace_reuse = analysis.get("replace_dose_reuse", {})
    if replace_reuse.get("all_observed_replace_branches_reused"):
        replace_identity_note = (
            f"All {replace_reuse['alias_old1024_replace_branches']} observed 1024-step `replace` branches are hash-verified aliases of their 2048-step counterparts. Therefore B algebraically reduces to `mean_h,f(R_keep,2048 − R_keep,1024)`: this is an old-candidate-content effect under `keep`, not independent evidence that replace or automatic organization improves reward."
        )
    else:
        replace_identity_note = (
            "Where the alias audit confirms `replace` is reused across source-dose branches, B algebraically reduces to `mean_h,f(R_keep,2048 − R_keep,1024)`. This is an old-candidate-content effect under `keep`, not independent evidence that replace or automatic organization improves reward."
        )
    lines = [
        "# Source-dose and learning-interaction follow-up",
        "",
        f"- Cohort phase: `{analysis['phase']}`",
        f"- Source directory: `{analysis['source_root']}`",
        f"- Cases: {analysis['observed_case_count']} observed / {analysis['expected_case_count']} expected",
        "- Report-generation environment steps: 0; this script only reads existing outputs.",
        "- Independent unit: seed; histories are averaged within seed before the 10,000-resample paired seed-cluster bootstrap.",
        "- Intervals are descriptive 95% percentile intervals. Aliases retain logical reward and duration but have zero actual environment steps and are not independent samples.",
        "",
        "## Frozen source-dose estimands",
        "",
        "Reward rate is cumulative future reward divided by logical future environment steps. `D_c = R_replace,c − R_keep,c`; `B = mean_h,f(D_1024 − D_2048)`. The primary B averages only history=future cells; the stress B averages history≠future cells.",
        "",
        replace_identity_note,
        "",
    ]
    for relation in ("dynamics", "reward_conflict"):
        if relation not in rels:
            continue
        item = rels[relation]
        est = item["estimands"]
        lines += [f"### Relation: `{relation}`", "", f"Complete paired seed clusters: {item['complete_seed_count']} / {item['expected_seed_count']}.", ""]
        b_rows = []
        for key, title in (("all", "B, future-balanced"), ("primary", "B, history=future"), ("stress", "B, history≠future")):
            e = est["B"][key]
            b_rows.append((title, _fmt_estimate(e), e.get("predeclared_classification", "")))
        lines += [_md_table(("Estimand", "Estimate [95% interval]", "Predeclared reading"), b_rows), ""]
        dose_rows = []
        for dose in ("old1024", "old2048"):
            for future in REGIMES:
                dose_rows.append((f"D {dose}, future {future}", _fmt_estimate(est["D_by_future"][f"{dose}/{future}"]), f"reversal={est['preference_reversal_by_dose'][dose]['preference_reversal']}"))
            dose_rows.append((f"I {dose}", _fmt_estimate(est["I"][dose]), "new-major future minus old-major future"))
        dose_rows.append(("Q = I_1024 − I_2048", _fmt_estimate(est["Q"]), "two-sided; no fixed directional hypothesis"))
        lines += [_md_table(("Contrast", "Estimate [95% interval]", "Definition / note"), dose_rows), ""]
        rate_rows = []
        for dose in ("old1024", "old2048"):
            for action in ACTIONS:
                for future in REGIMES:
                    rate_rows.append((
                        dose,
                        action,
                        future,
                        _fmt_estimate(item["mean_action_reward_rates"][dose][action][future]),
                    ))
        lines += ["All action mean reward rates", "", _md_table(("Dose", "Action", "Future", "Reward rate [95% interval]"), rate_rows), ""]
        lines += ["Per-seed estimates are in `analysis.json`; branches and histories are not resampled independently.", ""]

    lines += ["## Past-only selector contrasts", "", "Each contrast is selected-action reward rate minus the fixed comparator, averaged within seed. `primary` uses history=future; `stress` uses history≠future. A missing selector map is reported as unavailable, never as a zero effect.", ""]
    for relation, item in rels.items():
        selector_rows = []
        for dose in ("old2048", "old1024"):
            for selector in SELECTORS:
                for continuation in ("primary", "stress"):
                    stats = item["selectors"][dose].get(selector, {}).get(continuation, {})
                    for comparator in ("keep", "replace", "mean"):
                        entry = stats.get(comparator)
                        if not isinstance(entry, Mapping) or not entry.get("available"):
                            result = "unavailable"
                            source = ""
                        else:
                            result = _fmt_estimate(entry["estimate"])
                            source = ", ".join(entry.get("selection_sources", []))
                        selector_rows.append((relation, dose, selector, continuation, f"vs {comparator}", result, source))
        lines += [f"### `{relation}`", "", _md_table(("Relation", "Dose", "Selector", "Continuation", "Comparator", "Estimate [95% interval]", "Decision source"), selector_rows), ""]

    if ablation is not None:
        lines += ["## Persistence × future learning", "", "`T = (R_replace − R_none)_PPO − (R_replace − R_none)_frozen`. Each cell is first averaged over histories within seed and then bootstrapped over paired seed clusters. `T < 0` means future learning narrowed the initial replace/none gap; it does not by itself reject persistence, learning, or sample-efficiency value.", ""]
        for relation, result in ablation.get("relations", {}).items():
            rows = []
            for continuation in ("primary", "all_future", "stress"):
                for future in (*REGIMES, "balanced"):
                    estimate = result.get("T", {}).get(continuation, {}).get(future)
                    if estimate is None:
                        continue
                    rows.append((
                        continuation,
                        future,
                        _fmt_estimate(estimate),
                        _fmt_estimate(estimate["S_train"]),
                        _fmt_estimate(estimate["S_frozen"]),
                        estimate.get("practical_classification", ""),
                    ))
            lines += [f"### Ablation relation: `{relation}`", "", _md_table(("Condition", "Future", "T estimate [95% interval]", "S_train", "S_frozen", "±0.002 reading"), rows or [("—", "—", "unavailable", "", "", "")]), ""]

        mechanism = ablation.get("mechanism", {})
        lines += [
            "## Reward-trajectory diagnostics",
            "",
            "Short-horizon and per-bout contrasts are computed from the source PPO and no-gradient branch `steps.csv` files. Only rows marked `future` are read; the common prefix logs are excluded. These curves explain initialization and relearning and do not replace the full-future estimand.",
            "",
        ]
        early_rows = []
        for relation, relation_data in mechanism.get("relations", {}).items():
            for future in REGIMES:
                for horizon in (256, 512):
                    metrics = relation_data.get("early_steps", {}).get(future, {}).get(str(horizon), {})
                    if not metrics:
                        continue
                    early_rows.append((
                        relation,
                        future,
                        horizon,
                        _fmt_estimate(metrics.get("S_train", {})),
                        _fmt_estimate(metrics.get("S_frozen", {})),
                        _fmt_estimate(metrics.get("T", {})),
                    ))
        lines += [
            _md_table(("Relation", "Future", "First N steps", "S_train", "S_frozen", "T"), early_rows or [("—", "—", "—", "not available", "not available", "not available")]),
            "",
            f"Future-only step logs read: {mechanism.get('future_step_logs_read', 'unknown')}; prefix included: `{mechanism.get('prefix_included', 'unknown')}`; step-curve issues: {len(mechanism.get('issues', []))}.",
            "",
            "Per-bout seed-cluster estimates are in `analysis.json`; the figure plots `S_train` and `S_frozen` separately for each future regime.",
            "",
            "![Short-horizon and per-bout reward contrasts](learning_mechanism_curves.png)",
            "",
        ]
        if mechanism.get("issues"):
            lines += ["Step-curve issues:", ""]
            lines.extend(f"- {issue}" for issue in mechanism["issues"])
            lines.append("")

    costs = analysis["costs_and_integrity"]
    cost = costs["top_level_costs"]
    cost_rows = [
            ("Actual environment steps", _show(cost.get("environment_steps_total"))),
            ("Logical environment steps", _show(cost.get("logical_environment_steps_total"))),
            ("Prefix environment steps", _show(cost.get("prefix_environment_steps"))),
            ("Future branch environment steps (actual)", _show(cost.get("branch_environment_steps"))),
            ("CPU seconds (runner aggregate)", _show(cost.get("cpu_s"))),
            ("Wall seconds (runner aggregate)", _show(cost.get("wall_s"))),
            ("Child CPU seconds", _show(cost.get("child_cpu_s"))),
            ("Child wall seconds", _show(cost.get("child_wall_s"))),
            ("Logical / physical future branches", f"{costs['logical_future_branch_count_observed']} / {costs['physical_future_branch_count_observed']}"),
            ("Verified alias branches", costs["verified_alias_branch_count_observed"]),
            ("Failed or incomplete cases", len(costs["errors"]) + len(analysis["missing_case_ids"])),
        ]
    capture = analysis["source_capture"]
    capture_ranges = capture.get("ranges", {})

    def capture_range(field: str) -> str:
        value = capture_ranges.get(field)
        if not isinstance(value, Mapping):
            return "unknown"
        return f"{_show(value.get('min'))}–{_show(value.get('max'))}"

    lines += [
        "## Costs, storage, and integrity",
        "",
        _md_table(("Measure", "Observed value"), cost_rows),
        "",
        f"Capture records: {len(capture['records'])}; timing `{', '.join(capture.get('capture_timing') or []) or 'unknown'}`; requested bout-step values `{capture['requested_capture_steps']}`; actual bout steps `{capture_range('actual_bout_step')}`; global environment steps `{capture_range('global_environment_steps')}`. Fitted steps range `{capture_range('fitted_steps')}`, trained rollouts `{capture_range('rollouts_trained')}`, and minibatch updates `{capture_range('minibatch_updates')}`. Candidate coefficient counts `{capture['candidate_coefficients']}`; candidate storage `{_show(capture['candidate_float32_bytes_per_case'])}` bytes per case when 9 float32 coefficients are present.",
        "",
        f"Storage: actor payload budget `{_show(analysis['capacity']['actor_payload_budget_bytes_config'])}` bytes and observed payload range `{analysis['capacity']['actor_payload_bytes_range']}`; full package budget `{_show(analysis['capacity']['package_budget_bytes_config'])}` bytes and observed full-persistent range `{analysis['capacity']['full_persistent_bytes_range']}`. Observed over-budget packages `{_show(analysis['capacity']['over_budget_count'])}`. Missing capacity remains unknown.",
        "",
        "Codec and probability drift distributions below count only physical post-action branches; aliases are not counted again. Prequantized probability drift reflects compensation before int8 encoding, while postquantized drift measures behavior after the codec roundtrip.",
        "",
    ]
    drift_rows = []
    drift = analysis["codec_probability_drift"]
    for dose in ("old2048", "old1024"):
        for action in ACTIONS:
            values = drift["doses"][dose][action]
            drift_rows.append((
                dose,
                action,
                values["physical_branch_count"],
                _fmt_distribution(values["codec_max_coefficient_error"]),
                _fmt_distribution(values["final_codec_max_coefficient_error"]),
                _fmt_distribution(values["prequantized_max_probability_absolute_drift"]),
                _fmt_distribution(values["postquantized_max_probability_absolute_drift"]),
                _fmt_distribution(values["postquantized_max_kl"]),
            ))
    lines += [
        _md_table(("Dose", "Action", "Physical branches", "Action codec error", "Final codec error", "Pre-int8 max |Δp|", "Post-int8 max |Δp|", "Post-int8 max KL"), drift_rows),
        "",
    ]
    if ablation is not None:
        ac = ablation.get("costs", {})
        incremental = ac.get("ablation_incremental") or {}
        lines += [
            "### Learning-ablation incremental and merged costs",
            "",
            _md_table(("Measure", "Observed value"), [
                ("New prefix environment steps", _show(ac.get("new_prefix_environment_steps"))),
                ("New future environment steps", _show(ac.get("new_future_environment_steps"))),
                ("New logical future environment steps", _show(ac.get("new_logical_environment_steps"))),
                ("New total environment steps", _show(ac.get("new_environment_steps_total"))),
                ("Merged actual environment steps, source + new", _show(ac.get("merged_actual_environment_steps"))),
                ("Merged logical environment steps, source + new", _show(ac.get("merged_logical_environment_steps"))),
                ("Source PPO reference branches", _show(ac.get("source_ppo_reference_branch_count"))),
                ("Source PPO reference environment steps", _show(ac.get("source_ppo_reference_environment_steps"))),
                ("Source PPO reference CPU / wall seconds", f"{_show(ac.get('source_ppo_reference_cpu_s'))} / {_show(ac.get('source_ppo_reference_wall_s'))}"),
                ("Ablation incremental CPU / wall seconds", f"{_show(incremental.get('cpu_s', incremental.get('worker_cpu_s')))} / {_show(incremental.get('wall_s', incremental.get('worker_wall_s')))}"),
                ("Ablation child CPU / wall seconds", f"{_show(incremental.get('child_cpu_s'))} / {_show(incremental.get('child_wall_s'))}"),
            ]),
            "",
            "Source prefix and PPO comparator costs are references to the source-dose ledger; they are not added a second time to incremental ablation costs.",
            "",
        ]
    lines += ["## Scope and software", ""]
    lines.extend(f"- {caution}" for caution in analysis["cautions"])
    if analysis.get("phase") == "pilot":
        lines.append("- Pilot results are integration diagnostics only; they are not inferential evidence and were not used to tune the experiment.")
    lines += [
        "- Semantic alias hash verification is delegated to the separate source-follow-up audit; this report consumes runner alias records and does not repeat the full checkpoint audit.",
        "- Figures use Matplotlib (Hunter, 2007). The analysis workflow cites Scientific Agent Skills for procedural context only (Kassis et al., 2026); it is not a task-performance reference.",
        "",
        "![Frozen source-dose estimands](source_followup_estimands.png)",
        "",
    ]
    if ablation is not None:
        lines.append("![Persistence and future-learning interaction](learning_interaction_T.png)")
        lines.append("")
    return "\n".join(lines)


def _plot_source_estimands(analysis: Mapping[str, Any], output: Path) -> None:
    relations = [name for name in RELATIONS if name in analysis["relations"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), constrained_layout=True)
    labels = (("all", "all future cells"), ("primary", "history = future"), ("stress", "history ≠ future"))
    x_labels: list[str] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    colors: list[str] = []
    raw_seed_points: list[tuple[int, float, str]] = []
    for rel_index, relation in enumerate(relations):
        estimands = analysis["relations"][relation]["estimands"]
        for b_key, b_label in labels:
            value = estimands["B"][b_key]
            if value["mean"] is None:
                continue
            x_labels.append(f"{relation}\nB {b_label}")
            means.append(value["mean"])
            ci = value.get("ci95")
            lows.append(value["mean"] if ci is None else ci[0])
            highs.append(value["mean"] if ci is None else ci[1])
            colors.append(COLORS.get(relation, "#4C78A8"))
    ax = axes[0]
    xpos = np.arange(len(means), dtype=float)
    for x, mean, low, high, color in zip(xpos, means, lows, highs, colors):
        yerr = None if low == high else [[mean - low], [high - mean]]
        ax.errorbar(x, mean, yerr=yerr, fmt="o", color=color, capsize=3, markersize=6)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axhline(PRACTICAL, color="#777777", linestyle="--", linewidth=1, label="+0.002 practical threshold")
    ax.set_xticks(xpos, x_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Reward per environment step")
    ax.set_title("Old-source dose effect B")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    b_extent = max([PRACTICAL, *(abs(value) for value in lows + highs)] if means else [PRACTICAL]) * 1.25
    ax.set_ylim(-b_extent, b_extent)

    ax = axes[1]
    i_positions, i_values, i_err_low, i_err_high, i_colors, i_labels = [], [], [], [], [], []
    pos = 0
    for relation in relations:
        estimands = analysis["relations"][relation]["estimands"]
        for dose in ("old1024", "old2048"):
            value = estimands["I"][dose]
            if value["mean"] is None:
                continue
            ci = value.get("ci95")
            i_positions.append(pos)
            i_values.append(value["mean"])
            i_err_low.append(0.0 if ci is None else value["mean"] - ci[0])
            i_err_high.append(0.0 if ci is None else ci[1] - value["mean"])
            i_colors.append(COLORS.get(relation, "#4C78A8") if dose == "old1024" else "#9E9E9E")
            i_labels.append(f"{relation}\n{dose}")
            pos += 1
    if i_values:
        ax.errorbar(i_positions, i_values, yerr=np.array([i_err_low, i_err_high]), fmt="none", ecolor="#555555", capsize=3, linewidth=1)
        ax.scatter(i_positions, i_values, c=i_colors, s=38, zorder=3)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(i_positions, i_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Reward per environment step")
    ax.set_title("I by old-source dose")
    ax.grid(axis="y", alpha=0.25)
    i_extent = max([PRACTICAL, *(abs(value) for value in i_values + [v - e for v, e in zip(i_values, i_err_low)] + [v + e for v, e in zip(i_values, i_err_high)])] if i_values else [PRACTICAL]) * 1.25
    ax.set_ylim(-i_extent, i_extent)

    ax = axes[2]
    q_positions, q_values, q_lows, q_highs, q_colors, q_labels = [], [], [], [], [], []
    for i, relation in enumerate(relations):
        value = analysis["relations"][relation]["estimands"]["Q"]
        if value["mean"] is None:
            continue
        ci = value.get("ci95")
        q_positions.append(i)
        q_values.append(value["mean"])
        q_lows.append(0.0 if ci is None else value["mean"] - ci[0])
        q_highs.append(0.0 if ci is None else ci[1] - value["mean"])
        q_colors.append(COLORS.get(relation, "#4C78A8"))
        q_labels.append(relation)
    if q_values:
        ax.errorbar(q_positions, q_values, yerr=np.array([q_lows, q_highs]), fmt="none", ecolor="#555555", capsize=3, linewidth=1)
        ax.scatter(q_positions, q_values, c=q_colors, s=42, zorder=3)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(q_positions, q_labels, rotation=20, ha="right")
    ax.set_ylabel("Reward per environment step")
    ax.set_title("Q = I₁₀₂₄ − I₂₀₄₈")
    ax.grid(axis="y", alpha=0.25)
    q_extent = max([PRACTICAL, *(abs(value) for value in q_values + [v - e for v, e in zip(q_values, q_lows)] + [v + e for v, e in zip(q_values, q_highs)])] if q_values else [PRACTICAL]) * 1.25
    ax.set_ylim(-q_extent, q_extent)
    fig.suptitle("Frozen old-source dose estimands (seed-cluster intervals)", fontsize=13)
    fig.savefig(output / "source_followup_estimands.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "source_followup_estimands.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _ablation_branch_rows(root: Path, case: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = case.get("branch_results", case.get("branches", []))
    if not isinstance(rows, list):
        return []
    resolved: list[dict[str, Any]] = []
    case_dir = root / "cases" / str(case.get("source_case_id", case.get("case_id", "")))
    for item in rows:
        if isinstance(item, Mapping):
            summary_ref = item.get("summary")
            if summary_ref and not item.get("reward_total") and not item.get("logical_reward_total"):
                ref_path = Path(str(summary_ref))
                candidates = (
                    root / ref_path,
                    case_dir / ref_path,
                    root / "cases" / str(case.get("case_id", "")) / ref_path,
                )
                found = next((candidate for candidate in candidates if candidate.is_file()), None)
                if found is not None:
                    resolved.append(_json(found))
                    continue
            resolved.append(dict(item))
        elif isinstance(item, str):
            candidates = (root / item, case_dir / item)
            found = next((candidate for candidate in candidates if candidate.is_file()), None)
            if found is not None:
                resolved.append(_json(found))
    return resolved


def _ablation_branch_audit(branch: Mapping[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    audit = branch.get("no_gradient_audit")
    if not isinstance(audit, Mapping):
        audit = branch.get("audit")
    if not isinstance(audit, Mapping):
        audit = {}
    counts = branch.get("counts", {})
    updates = branch.get("minibatch_updates")
    if updates is None and isinstance(counts, Mapping):
        updates = counts.get("minibatch_updates")
    if updates is None or _integer(updates, "ablation.minibatch_updates") != 0:
        issues.append("no-gradient branch minibatch_updates is missing or nonzero")
    fitted_ok = branch.get("fitted_steps_not_increased")
    if fitted_ok is None:
        fitted_ok = audit.get("fitted_steps_not_increased")
    if fitted_ok is not True:
        issues.append("no-gradient branch lacks a passing fitted_steps_not_increased audit")
    for key, aliases in (
        ("w0_unchanged", ("w0_unchanged", "source_w0_unchanged")),
        ("critic_unchanged", ("critic_unchanged",)),
    ):
        value = next((branch.get(name) for name in aliases if branch.get(name) is not None), None)
        if value is None:
            value = next((audit.get(name) for name in aliases if audit.get(name) is not None), None)
        if value is not True:
            issues.append(f"no-gradient branch lacks a passing {key} audit")
    return not issues, issues


def _read_future_trace(path: Path) -> dict[str, Any]:
    """Read a future branch's raw rewards; reject any non-future/prefix rows."""
    required = {"phase", "global_step", "bout", "taskobserver", "reward"}
    rewards: list[float] = []
    bout_keys: list[tuple[str, str]] = []
    bout_rewards: dict[tuple[str, str], list[float]] = {}
    previous_step: int | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: steps.csv lacks required future-step columns")
        for row_number, row in enumerate(reader, start=2):
            if row.get("phase") != "future":
                raise ValueError(f"{path}:{row_number}: expected a future-only step log")
            step = _integer(row.get("global_step"), f"{path}:{row_number}.global_step")
            if previous_step is not None and step <= previous_step:
                raise ValueError(f"{path}:{row_number}: global_step is not strictly increasing")
            previous_step = step
            reward = _finite(row.get("reward"), f"{path}:{row_number}.reward")
            rewards.append(reward)
            key = (str(row.get("bout", "")), str(row.get("taskobserver", "")))
            if key not in bout_rewards:
                bout_keys.append(key)
                bout_rewards[key] = []
            bout_rewards[key].append(reward)
    if not rewards:
        raise ValueError(f"{path}: steps.csv has no future rows")
    return {
        "step_count": len(rewards),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "bout_keys": bout_keys,
        "bout_reward_rates": [float(np.mean(bout_rewards[key])) for key in bout_keys],
    }


def _mechanism_curves(
    source_root: Path,
    ablation_root: Path,
    by_relation: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    history_regimes: Sequence[str],
) -> dict[str, Any]:
    """Build paired PPO/no-gradient short-horizon and per-bout reward gaps.

    Only branch-local future steps.csv files are read. The common prefix and its
    raw steps are not opened, reused, or included in any curve.
    """
    trace_cache: dict[Path, dict[str, Any]] = {}
    issues: list[str] = []
    cells: dict[str, dict[str, dict[int, dict[str, dict[str, Any]]]]] = {
        relation: {future: {} for future in REGIMES} for relation in RELATIONS
    }
    for relation, seeds in by_relation.items():
        for seed, histories in seeds.items():
            for history in history_regimes:
                cell = histories[history]
                case_id = str(cell["source_case_id"])
                for future in REGIMES:
                    traces: dict[tuple[str, str], dict[str, Any]] = {}
                    branch_rows = {
                        ("ppo", action): cell["ppo"].get((action, future))
                        for action in ("replace", "none")
                    }
                    branch_rows.update({
                        ("frozen", action): cell["no_gradient"]["branches"].get((action, future))
                        for action in ("replace", "none")
                    })
                    failed = False
                    for (mode, action), branch in branch_rows.items():
                        if not isinstance(branch, Mapping):
                            issues.append(f"{case_id}/{mode}/{action}/{future}: missing branch summary for mechanism curves")
                            failed = True
                            continue
                        branch_root = source_root if mode == "ppo" else ablation_root
                        path = branch_root / "cases" / case_id / "branches" / f"{action}__{future}" / "steps.csv"
                        try:
                            resolved = path.resolve()
                            if resolved not in trace_cache:
                                trace_cache[resolved] = _read_future_trace(resolved)
                            trace = trace_cache[resolved]
                            expected_steps = branch.get("actual_environment_steps")
                            if expected_steps is not None and trace["step_count"] != _integer(expected_steps, f"{case_id}.{mode}.{action}.{future}.actual_environment_steps"):
                                raise ValueError(
                                    f"{path}: {trace['step_count']} future rows differ from branch actual_environment_steps={expected_steps}"
                                )
                            traces[(mode, action)] = trace
                        except (OSError, ValueError) as exc:
                            issues.append(f"{case_id}/{mode}/{action}/{future}: {exc}")
                            failed = True
                    if failed or len(traces) != 4:
                        continue
                    early: dict[int, dict[str, float]] = {}
                    for horizon in (256, 512):
                        if any(trace["step_count"] < horizon for trace in traces.values()):
                            continue
                        s_train = float(
                            np.mean(traces[("ppo", "replace")]["rewards"][:horizon])
                            - np.mean(traces[("ppo", "none")]["rewards"][:horizon])
                        )
                        s_frozen = float(
                            np.mean(traces[("frozen", "replace")]["rewards"][:horizon])
                            - np.mean(traces[("frozen", "none")]["rewards"][:horizon])
                        )
                        early[horizon] = {"S_train": s_train, "S_frozen": s_frozen, "T": s_train - s_frozen}

                    bout_key_sets = [traces[key]["bout_keys"] for key in traces]
                    per_bout: dict[int, dict[str, float]] = {}
                    if all(keys == bout_key_sets[0] for keys in bout_key_sets[1:]):
                        for index in range(len(bout_key_sets[0])):
                            s_train = float(
                                traces[("ppo", "replace")]["bout_reward_rates"][index]
                                - traces[("ppo", "none")]["bout_reward_rates"][index]
                            )
                            s_frozen = float(
                                traces[("frozen", "replace")]["bout_reward_rates"][index]
                                - traces[("frozen", "none")]["bout_reward_rates"][index]
                            )
                            per_bout[index + 1] = {"S_train": s_train, "S_frozen": s_frozen, "T": s_train - s_frozen}
                    else:
                        issues.append(f"{case_id}/{future}: PPO and no-gradient branches have different bout/task schedules; per-bout contrast omitted")
                    cells[relation][future].setdefault(seed, {})[history] = {
                        "early_steps": {str(horizon): value for horizon, value in early.items()},
                        "per_bout": {str(index): value for index, value in per_bout.items()},
                    }

    relations: dict[str, Any] = {}
    for relation in RELATIONS:
        relation_result: dict[str, Any] = {"early_steps": {}, "per_bout": {}}
        for future in REGIMES:
            seed_histories = cells[relation][future]
            early_result: dict[str, Any] = {}
            for horizon in (256, 512):
                metric_result: dict[str, Any] = {}
                for metric in ("S_train", "S_frozen", "T"):
                    seed_values: dict[int, float] = {}
                    for seed, histories in seed_histories.items():
                        values = [
                            histories.get(history, {}).get("early_steps", {}).get(str(horizon), {}).get(metric)
                            for history in history_regimes
                        ]
                        if values and all(value is not None for value in values):
                            seed_values[seed] = float(np.mean(values))
                    metric_result[metric] = _estimate(seed_values, seed=20261018, label=f"{metric}, first {horizon} future steps")
                early_result[str(horizon)] = metric_result
            relation_result["early_steps"][future] = early_result

            bout_ids = sorted({
                int(bout_id)
                for histories in seed_histories.values()
                for payload in histories.values()
                for bout_id in payload.get("per_bout", {})
            })
            bout_result = []
            for bout_id in bout_ids:
                metric_result = {}
                for metric in ("S_train", "S_frozen", "T"):
                    seed_values = {}
                    for seed, histories in seed_histories.items():
                        values = [
                            histories.get(history, {}).get("per_bout", {}).get(str(bout_id), {}).get(metric)
                            for history in history_regimes
                        ]
                        if values and all(value is not None for value in values):
                            seed_values[seed] = float(np.mean(values))
                    metric_result[metric] = _estimate(seed_values, seed=20261018, label=f"{metric}, future bout {bout_id}")
                bout_result.append({"bout_order": bout_id, **metric_result})
            relation_result["per_bout"][future] = bout_result
        relations[relation] = relation_result

    return {
        "format": "learning_interaction_mechanism_v1",
        "definition": "S_train and S_frozen are replace-minus-none reward rates from branch-local future steps; T is their difference. First-horizon values use the first N future steps; per-bout values use each branch's bout reward rate. Histories are averaged within seed before seed-cluster bootstrap.",
        "prefix_included": False,
        "future_step_logs_read": len(trace_cache),
        "issues": issues,
        "relations": relations,
    }


def _analyze_ablation(source_root: Path, ablation_input: str | Path) -> dict[str, Any]:
    """Join no-gradient branches to trained source-dose PPO reference branches."""
    path = Path(ablation_input).resolve()
    summary_path = path if path.is_file() else path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"ablation summary not found: {summary_path}")
    summary = _json(summary_path)
    if summary.get("format") not in ("learning_interaction_ablation_v1", "learning_ablation_summary_v1"):
        raise ValueError(f"unsupported ablation summary format: {summary.get('format')!r}")
    source_root, source_summary = _source_dir(source_root)
    source_config = source_summary.get("config", {})
    history_regimes = source_config.get("history_regimes", list(REGIMES))
    source_case_rows = source_summary.get("cases", [])
    source_cases: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for source_raw in source_case_rows:
        if not isinstance(source_raw, Mapping):
            continue
        try:
            normalized = _normalize_case(source_root, source_raw)
            decision_path = source_root / "cases" / normalized["case_id"] / "decision.json"
            decision = _json(decision_path) if decision_path.is_file() else {}
            normalized["prefix_state_hash"] = decision.get("prefix_state_hash")
            normalized["source_decision"] = decision
            source_cases[normalized["case_id"]] = normalized
        except Exception as exc:
            issues.append(f"{source_raw.get('case_id', 'unknown')}: cannot normalize source PPO branches: {exc}")

    raw_cases = summary.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("ablation summary.cases must be a list")
    no_gradient: dict[str, dict[str, Any]] = {}
    for case in raw_cases:
        if not isinstance(case, Mapping):
            continue
        source_case_id = str(case.get("source_case_id", case.get("case_id", "")))
        if source_case_id not in source_cases:
            issues.append(f"{source_case_id}: no matching source-dose case")
            continue
        source_case = source_cases[source_case_id]
        expected_hash = source_case.get("prefix_state_hash")
        case_hash = case.get("prefix_state_hash")
        if case_hash != expected_hash or not expected_hash:
            issues.append(f"{source_case_id}: common-prefix state hash is missing or mismatched")
            continue
        seed = _integer(case.get("seed", source_case["seed"]), f"{source_case_id}.seed")
        if seed != source_case["seed"]:
            issues.append(f"{source_case_id}: seed differs from source-dose case")
            continue
        branch_map: dict[tuple[str, str], dict[str, Any]] = {}
        rows = _ablation_branch_rows(summary_path.parent, case)
        for raw_branch in rows:
            action, future = raw_branch.get("action"), raw_branch.get("future_regime")
            if action not in ("replace", "none") or future not in REGIMES:
                issues.append(f"{source_case_id}: no-gradient branch has invalid action/future")
                continue
            if raw_branch.get("learning_mode", "no_gradient") != "no_gradient":
                issues.append(f"{source_case_id}/{action}/{future}: expected learning_mode=no_gradient")
                continue
            branch_case_id = str(raw_branch.get("source_case_id", source_case_id))
            if branch_case_id != source_case_id:
                issues.append(f"{source_case_id}/{action}/{future}: branch references a different source case")
                continue
            branch_hash = raw_branch.get("prefix_state_hash", expected_hash)
            if branch_hash != expected_hash:
                issues.append(f"{source_case_id}/{action}/{future}: branch prefix hash mismatch")
                continue
            match = raw_branch.get("matched_ppo_branch")
            if not isinstance(match, Mapping) or str(match.get("source_case_id", "")) != source_case_id or match.get("action") != action or match.get("future_regime") != future:
                issues.append(f"{source_case_id}/{action}/{future}: matched PPO reference does not identify the same source case/action/future")
                continue
            audit_ok, audit_issues = _ablation_branch_audit(raw_branch)
            if not audit_ok:
                issues.extend(f"{source_case_id}/{action}/{future}: {issue}" for issue in audit_issues)
                continue
            try:
                reward = _branch_reward(raw_branch, where=f"no_gradient.{source_case_id}.{action}.{future}")
            except ValueError as exc:
                issues.append(str(exc))
                continue
            key = (str(action), str(future))
            if key in branch_map:
                issues.append(f"{source_case_id}: duplicate no-gradient branch {key}")
                continue
            branch_map[key] = {
                "reward_per_step": reward["reward_per_step"],
                "logical_environment_steps": reward["logical_environment_steps"],
                "actual_environment_steps": reward["actual_environment_steps"],
                "costs": reward["costs"],
                "counts": reward["counts"],
                "matched_ppo_branch": dict(match),
                "rollouts_collected": raw_branch.get("rollouts_collected"),
                "fitted_steps_start": raw_branch.get("fitted_steps_start"),
                "fitted_steps_end": raw_branch.get("fitted_steps_end"),
            }
        expected_keys = {(action, future) for action in ("replace", "none") for future in REGIMES}
        if set(branch_map) != expected_keys:
            issues.append(f"{source_case_id}: incomplete or unaudited no-gradient branch set")
        no_gradient[source_case_id] = {
            "seed": seed,
            "relation": source_case["relation"],
            "history": source_case["history"],
            "prefix_state_hash": expected_hash,
            "branches": branch_map,
            "complete": set(branch_map) == expected_keys,
        }

    # Keep only seed clusters with every configured history and every tested
    # action/future pair in both PPO and no-gradient conditions.
    by_relation: dict[str, dict[int, dict[str, dict[str, Any]]]] = {relation: {} for relation in RELATIONS}
    expected_by_seed: dict[str, dict[int, dict[str, str]]] = {relation: {} for relation in RELATIONS}
    for case_id, source_case in source_cases.items():
        expected_by_seed[source_case["relation"]].setdefault(source_case["seed"], {})[source_case["history"]] = case_id
    for relation, seeds in expected_by_seed.items():
        for seed, histories in seeds.items():
            if any(history not in histories for history in history_regimes):
                continue
            complete_seed = True
            selected_cases: dict[str, dict[str, Any]] = {}
            for history in history_regimes:
                case_id = histories[history]
                ng_case = no_gradient.get(case_id)
                if ng_case is None:
                    complete_seed = False
                    break
                source_case = source_cases[case_id]
                if not ng_case["complete"] or source_case["errors"]:
                    complete_seed = False
                    break
                ppo = source_case["branches"]["old2048"]
                if any((action, future) not in ppo for action in ACTIONS for future in REGIMES):
                    complete_seed = False
                    break
                selected_cases[history] = {"source_case_id": case_id, "no_gradient": ng_case, "ppo": ppo}
            if complete_seed:
                by_relation[relation][seed] = selected_cases

    results: dict[str, Any] = {}
    for relation, seed_cases in by_relation.items():
        t_cells: dict[str, Any] = {}
        for continuation in ("primary", "all_future", "stress"):
            t_cells[continuation] = {}
            for future in (*REGIMES, "balanced"):
                included_futures = list(REGIMES) if future == "balanced" else [future]
                t_values: dict[int, float] = {}
                train_values: dict[int, float] = {}
                frozen_values: dict[int, float] = {}
                for seed, histories in seed_cases.items():
                    ts, train, frozen = [], [], []
                    for history in history_regimes:
                        for current_future in included_futures:
                            if continuation == "primary" and history != current_future:
                                continue
                            if continuation == "stress" and history == current_future:
                                continue
                            cell = histories[history]
                            ppo = cell["ppo"]
                            ng = cell["no_gradient"]["branches"]
                            s_train = (
                                ppo[("replace", current_future)]["reward_per_step"]
                                - ppo[("none", current_future)]["reward_per_step"]
                            )
                            s_frozen = (
                                ng[("replace", current_future)]["reward_per_step"]
                                - ng[("none", current_future)]["reward_per_step"]
                            )
                            train.append(s_train)
                            frozen.append(s_frozen)
                            ts.append(s_train - s_frozen)
                    if ts:
                        t_values[seed] = float(np.mean(ts))
                        train_values[seed] = float(np.mean(train))
                        frozen_values[seed] = float(np.mean(frozen))
                estimate = _estimate(
                    t_values, seed=20261018,
                    label=f"T {continuation}, future={future}",
                )
                estimate["S_train"] = _estimate(train_values, seed=20261018, label="replace minus none under PPO")
                estimate["S_frozen"] = _estimate(frozen_values, seed=20261018, label="replace minus none without gradients")
                ci = estimate.get("ci95")
                if ci is None:
                    estimate["practical_classification"] = "not enough seed clusters to assess ±0.002"
                elif ci[0] > PRACTICAL:
                    estimate["practical_classification"] = "interval entirely above +0.002"
                elif ci[1] < -PRACTICAL:
                    estimate["practical_classification"] = "interval entirely below −0.002"
                else:
                    estimate["practical_classification"] = "uncertain at the ±0.002 practical threshold"
                t_cells[continuation][future] = estimate
        results[relation] = {
            "T": t_cells,
            "n_seed_clusters": len(seed_cases),
            "incomplete_seeds": sorted(set(expected_by_seed[relation]) - set(seed_cases)),
        }

    mechanism = _mechanism_curves(source_root, summary_path.parent, by_relation, history_regimes)
    issues.extend(mechanism["issues"])

    source_costs = source_summary.get("costs", {})
    ablation_costs = summary.get("costs", {})
    if not isinstance(source_costs, Mapping):
        source_costs = {}
    if not isinstance(ablation_costs, Mapping):
        ablation_costs = {}

    # Source PPO comparators are already included in the source-dose ledger.
    # Count and sum only the matched replace/none old2048 branches here so the
    # report can identify the reference cost without adding it a second time.
    ppo_reference_branches = [
        branch
        for source_case in source_cases.values()
        for action in ("replace", "none")
        for future in REGIMES
        if (branch := source_case["branches"]["old2048"].get((action, future))) is not None
    ]

    def complete_sum(values: Sequence[Any]) -> float | None:
        if not values or any(value is None for value in values):
            return None
        try:
            numeric = [_finite(value, "cost value") for value in values]
        except ValueError:
            return None
        return float(sum(numeric))

    ppo_reference_costs = [
        branch.get("costs") if isinstance(branch.get("costs"), Mapping) else {}
        for branch in ppo_reference_branches
    ]
    no_gradient_branches = [
        branch
        for case in no_gradient.values()
        for branch in case["branches"].values()
    ]
    new_actual_steps = complete_sum([branch.get("actual_environment_steps") for branch in no_gradient_branches])
    expected_no_gradient_branch_count = len(raw_cases) * len(("replace", "none")) * len(REGIMES)
    new_logical_steps = (
        complete_sum([branch.get("logical_environment_steps") for branch in no_gradient_branches])
        if len(no_gradient_branches) == expected_no_gradient_branch_count and not issues
        else None
    )
    reported_new_steps = ablation_costs.get(
        "new_branch_environment_steps", ablation_costs.get("new_future_env_steps")
    )
    source_actual_steps = source_costs.get("environment_steps_total")
    source_logical_steps = source_costs.get("logical_environment_steps_total")
    merged_actual_steps = complete_sum([source_actual_steps, reported_new_steps])
    merged_logical_steps = complete_sum([source_logical_steps, new_logical_steps])
    if new_actual_steps is not None and reported_new_steps is not None and new_actual_steps != reported_new_steps:
        issues.append("ablation branch environment-step summary does not match audited branch totals")
    return {
        "format": "learning_interaction_analysis_v1",
        "source_root": str(source_root),
        "ablation_root": str(summary_path.parent),
        "relations": results,
        "mechanism": mechanism,
        "issues": issues + [str(error) for error in summary.get("errors", [])],
        "case_count": len(raw_cases),
        "costs": {
            "ablation_incremental": dict(ablation_costs),
            "source_cohort_merged_once": dict(source_costs),
            "new_prefix_environment_steps": ablation_costs.get(
                "new_prefix_environment_steps", ablation_costs.get("new_prefix_env_steps")
            ),
            "new_future_environment_steps": reported_new_steps if reported_new_steps is not None else new_actual_steps,
            "new_logical_environment_steps": new_logical_steps,
            "new_environment_steps_total": ablation_costs.get(
                "new_environment_steps_total", ablation_costs.get("new_total_environment_steps")
            ),
            "source_cohort_actual_environment_steps": source_actual_steps,
            "source_cohort_logical_environment_steps": source_logical_steps,
            "merged_actual_environment_steps": merged_actual_steps,
            "merged_logical_environment_steps": merged_logical_steps,
            "source_ppo_reference_branch_count": len(ppo_reference_branches),
            "source_ppo_reference_environment_steps": complete_sum(
                [branch.get("actual_environment_steps") for branch in ppo_reference_branches]
            ),
            "source_ppo_reference_cpu_s": complete_sum([cost.get("cpu_s") for cost in ppo_reference_costs]),
            "source_ppo_reference_wall_s": complete_sum([cost.get("wall_s") for cost in ppo_reference_costs]),
        },
        "cautions": [
            "No-gradient branches preserve actions, rewards, and natural eviction/inheritance; zero gradients do not mean no cache activity.",
            "Rollouts collected are not trained rollouts; update counts and fitted_steps are checked separately.",
            "T < 0 means PPO narrowed the initial replace/none gap and is not a rejection of persistence or learning in general.",
        ],
    }

def _plot_ablation(ablation: Mapping[str, Any], output: Path) -> None:
    relations = [name for name in RELATIONS if name in ablation.get("relations", {})]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    positions, means, lower, upper, colors, labels = [], [], [], [], [], []
    position = 0
    for relation in relations:
        for continuation in ("primary", "all_future", "stress"):
            for future in (*REGIMES, "balanced"):
                estimate = ablation["relations"][relation]["T"].get(continuation, {}).get(future)
                if not estimate or estimate.get("mean") is None:
                    continue
                ci = estimate.get("ci95")
                positions.append(position)
                means.append(estimate["mean"])
                lower.append(0.0 if ci is None else estimate["mean"] - ci[0])
                upper.append(0.0 if ci is None else ci[1] - estimate["mean"])
                colors.append(COLORS.get(relation, "#4C78A8"))
                labels.append(f"{relation}\n{continuation}, {future}")
                position += 1
    if positions:
        ax.errorbar(positions, means, yerr=np.array([lower, upper]), fmt="none", ecolor="#555555", capsize=3)
        ax.scatter(positions, means, c=colors, s=42, zorder=3)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axhline(PRACTICAL, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(-PRACTICAL, color="#777777", linestyle="--", linewidth=1, label="±0.002 practical thresholds")
    ax.set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("T: reward per environment step")
    ax.set_title("Persistence × future PPO interaction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    extent = max([PRACTICAL, *(abs(value) for value in means + [value - error for value, error in zip(means, lower)] + [value + error for value, error in zip(means, upper)])] if means else [PRACTICAL]) * 1.25
    ax.set_ylim(-extent, extent)
    fig.savefig(output / "learning_interaction_T.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "learning_interaction_T.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_mechanism(mechanism: Mapping[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey="row", constrained_layout=True)
    condition_styles = {
        "S_train": ("#336B87", "PPO"),
        "S_frozen": ("#D17A22", "No gradient"),
    }
    future_styles = {"old_major": "-", "new_major": "--"}
    for row, relation in enumerate(RELATIONS):
        relation_data = mechanism.get("relations", {}).get(relation, {})
        has_data = False
        extent_values: list[float] = []
        early_ax, bout_ax = axes[row]
        for future in REGIMES:
            early = relation_data.get("early_steps", {}).get(future, {})
            for metric, (color, condition_label) in condition_styles.items():
                estimates = [early.get(str(horizon), {}).get(metric) for horizon in (256, 512)]
                points = [(h, item) for h, item in zip((256, 512), estimates) if isinstance(item, Mapping) and item.get("mean") is not None]
                if points:
                    has_data = True
                    x_values = [point[0] for point in points]
                    y_values = [float(point[1]["mean"]) for point in points]
                    label = f"{future}, {condition_label}"
                    early_ax.plot(x_values, y_values, marker="o", color=color, linestyle=future_styles[future], label=label)
                    for x, point in points:
                        ci = point.get("ci95")
                        if ci is not None:
                            early_ax.errorbar([x], [point["mean"]], yerr=[[point["mean"] - ci[0]], [ci[1] - point["mean"]]], fmt="none", ecolor=color, capsize=2)
                            extent_values.extend([abs(ci[0]), abs(ci[1])])
                        extent_values.append(abs(float(point["mean"])))

                bouts = relation_data.get("per_bout", {}).get(future, [])
                bout_points = [item for item in bouts if isinstance(item.get(metric), Mapping) and item[metric].get("mean") is not None]
                if bout_points:
                    has_data = True
                    x_values = [int(point["bout_order"]) for point in bout_points]
                    y_values = [float(point[metric]["mean"]) for point in bout_points]
                    label = f"{future}, {condition_label}"
                    bout_ax.plot(x_values, y_values, marker="o", markersize=3.5, color=color, linestyle=future_styles[future], label=label)
                    for x, point in zip(x_values, bout_points):
                        ci = point[metric].get("ci95")
                        if ci is not None:
                            bout_ax.errorbar([x], [point[metric]["mean"]], yerr=[[point[metric]["mean"] - ci[0]], [ci[1] - point[metric]["mean"]]], fmt="none", ecolor=color, capsize=2)
                            extent_values.extend([abs(ci[0]), abs(ci[1])])
                        extent_values.append(abs(float(point[metric]["mean"])))
        for ax in (early_ax, bout_ax):
            if has_data:
                ax.axhline(0, color="#333333", linewidth=0.9)
                ax.grid(axis="y", alpha=0.22)
                ax.legend(frameon=False, fontsize=7, ncol=2)
            else:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No complete curve data", transform=ax.transAxes, ha="center", va="center", color="#777777")
        if extent_values:
            extent = max(extent_values) * 1.2
            early_ax.set_ylim(-extent, extent)
        early_ax.set_title(f"{relation}: first future steps")
        early_ax.set_xlabel("Steps from future-branch start")
        early_ax.set_xticks((256, 512))
        bout_ax.set_title(f"{relation}: per-bout reward gap")
        bout_ax.set_xlabel("Future bout order")
        early_ax.set_ylabel("Replace − none, reward per step")
        bout_ax.set_ylabel("Replace − none, reward per step")
    fig.suptitle("Short-horizon and per-bout reward gaps: PPO vs no-gradient", fontsize=13)
    fig.savefig(output / "learning_mechanism_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "learning_mechanism_curves.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(source: str | Path, output: str | Path, *, ablation: str | Path | None = None) -> dict[str, Any]:
    analysis = analyze_source(source)
    source_root = Path(analysis["source_root"])
    ablation_analysis = _analyze_ablation(source_root, ablation) if ablation is not None else None
    out = Path(output).resolve()
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise FileExistsError(f"output must be a new or empty directory: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
    _plot_source_estimands(analysis, out)
    if ablation_analysis is not None:
        _plot_ablation(ablation_analysis, out)
        _plot_mechanism(ablation_analysis["mechanism"], out)
    analysis["ablation"] = ablation_analysis
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_render_markdown(analysis, ablation=ablation_analysis), encoding="utf-8")
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source-dose output directory or summary.json")
    parser.add_argument("--ablation", type=Path, help="optional no-gradient learning-ablation output directory or summary.json")
    parser.add_argument("--output", type=Path, required=True, help="new or empty report directory")
    args = parser.parse_args(argv)
    write_report(args.source, args.output, ablation=args.ablation)
    print(f"wrote source follow-up report to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Report the frozen surrogate-lower matched-future confirmation cohort.

This is an offline observer over saved summaries. It does not run an
environment, score a new action, or feed branch outcomes back to a learner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACTIONS = ("keep", "replace", "mean", "none")
HISTORIES = ("old_major", "new_major")
COMPARATORS = ("keep", "replace", "mean", "frequency", "none")
REPORT_FORMAT = "surrogate_lower_confirmation_report_v1"
EXPECTED_SEED_COUNT = 16
EXPECTED_FAMILY_COUNTS = {"A": 8, "B": 8}
_TASK_FAMILY = re.compile(r"^([AB])[0-2][+-]$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read input summary JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("input summary must be a JSON object")
    return value


def _finite_number(value: Any, name: str, *, allow_none: bool = False) -> float | None:
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


def _nonnegative_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    number = _finite_number(value, name)
    assert number is not None
    if number < 0 or not number.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


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


def _old_family(case: Mapping[str, Any]) -> str | None:
    prefix = case.get("prefix")
    bouts = prefix.get("per_bout") if isinstance(prefix, Mapping) else None
    if not isinstance(bouts, list):
        return None
    families = set()
    for row in bouts:
        if not isinstance(row, Mapping) or row.get("phase") != "bootstrap":
            continue
        task = row.get("task")
        match = _TASK_FAMILY.fullmatch(task) if isinstance(task, str) else None
        if row.get("group") == 0 and match is not None:
            families.add(match.group(1))
    return next(iter(families)) if len(families) == 1 else None


def _branch_rate(branch: Mapping[str, Any], context: str) -> float:
    steps = _nonnegative_int(branch.get("environment_steps"), f"{context}.environment_steps")
    if steps is None or steps <= 0:
        raise ValueError(f"{context}.environment_steps must be positive")
    total = _finite_number(branch.get("reward_total"), f"{context}.reward_total")
    assert total is not None
    rate = total / steps
    if "reward_per_step" in branch:
        reported = _finite_number(branch["reward_per_step"], f"{context}.reward_per_step")
        assert reported is not None
        if not math.isclose(reported, rate, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"{context}.reward_per_step does not match reward_total / environment_steps")
    return float(rate)


def _selector_action(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in ACTIONS:
        raise ValueError(f"{context} must be one of {ACTIONS}")
    return value


def _case_record(case: Any, index: int) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(case, Mapping):
        raise ValueError(f"cases[{index}] must be an object")
    context = f"cases[{index}]"
    seed = _nonnegative_int(case.get("seed"), f"{context}.seed")
    if seed is None:
        raise ValueError(f"{context}.seed is required")
    history = case.get("history_regime")
    pair_kind = case.get("pair_kind")
    if not isinstance(history, str) or not history:
        raise ValueError(f"{context}.history_regime is required")
    if not isinstance(pair_kind, str) or not pair_kind:
        raise ValueError(f"{context}.pair_kind is required")

    reasons: list[str] = []
    if pair_kind != "dynamics":
        reasons.append(f"{context}: pair_kind is not dynamics")
    family = _old_family(case)
    if family is None:
        reasons.append(f"{context}: cannot derive one old family from bootstrap task IDs")

    case_errors = case.get("errors", [])
    if case_errors:
        reasons.append(f"{context}: runner recorded case errors")

    selectors = case.get("selectors")
    selectors = selectors if isinstance(selectors, Mapping) else {}
    lower_from_record = case.get("confirmation_action")
    lower_from_selectors = selectors.get("surrogate_lower")
    if lower_from_record is not None and lower_from_selectors is not None and lower_from_record != lower_from_selectors:
        reasons.append(f"{context}: confirmation_action disagrees with selectors.surrogate_lower")
    lower_action = _selector_action(
        lower_from_record if lower_from_record is not None else lower_from_selectors,
        f"{context}.confirmation_action",
    )
    frequency_action = _selector_action(selectors.get("frequency"), f"{context}.selectors.frequency")
    if lower_action is None:
        reasons.append(f"{context}: missing frozen surrogate_lower selection")
    if frequency_action is None:
        reasons.append(f"{context}: missing frequency selection")

    branches_value = case.get("branches")
    if not isinstance(branches_value, list):
        branches_value = []
        reasons.append(f"{context}: branches must be a list")
    branch_map: dict[tuple[str, str], float] = {}
    matched_count = 0
    branch_raw_rows: list[int | None] = []
    for branch_index, branch in enumerate(branches_value):
        if not isinstance(branch, Mapping):
            reasons.append(f"{context}: branches[{branch_index}] is not an object")
            continue
        action = branch.get("action")
        future = branch.get("future_regime")
        if action not in ACTIONS or not isinstance(future, str):
            reasons.append(f"{context}: branches[{branch_index}] has an invalid action/future")
            continue
        branch_raw_rows.append(
            _nonnegative_int(
                branch.get("raw_environment_steps"),
                f"{context}.branches[{branch_index}].raw_environment_steps",
                allow_none=True,
            )
        )
        if future == history:
            matched_count += 1
            key = (str(action), str(future))
            if key in branch_map:
                reasons.append(f"{context}: duplicate matched branch {key}")
                continue
            branch_map[key] = _branch_rate(branch, f"{context}.branches[{branch_index}]")

    expected_keys = {(action, history) for action in ACTIONS}
    if set(branch_map) != expected_keys:
        missing = sorted(expected_keys - set(branch_map))
        reasons.append(f"{context}: missing matched action branches {missing}")
    if len(branches_value) != len(ACTIONS) or matched_count != len(ACTIONS):
        reasons.append(f"{context}: expected exactly four history-matched observer branches")

    metrics: dict[str, float] | None = None
    if set(branch_map) == expected_keys and lower_action is not None and frequency_action is not None:
        by_role = {
            "surrogate_lower": branch_map[(lower_action, history)],
            "keep": branch_map[("keep", history)],
            "replace": branch_map[("replace", history)],
            "mean": branch_map[("mean", history)],
            "frequency": branch_map[(frequency_action, history)],
            "none": branch_map[("none", history)],
        }
        metrics = dict(by_role)
        for comparator in COMPARATORS:
            metrics[f"lower_minus_{comparator}"] = float(by_role["surrogate_lower"] - by_role[comparator])

    prefix = case.get("prefix")
    prefix = prefix if isinstance(prefix, Mapping) else {}
    prefix_rows = _nonnegative_int(
        prefix.get("raw_environment_steps"), f"{context}.prefix.raw_environment_steps", allow_none=True
    )
    if prefix_rows is None:
        reasons.append(f"{context}: missing prefix raw_environment_steps")
    if any(value is None for value in branch_raw_rows):
        reasons.append(f"{context}: missing branch raw_environment_steps")

    costs = case.get("costs")
    costs = costs if isinstance(costs, Mapping) else {}
    clean_costs = {}
    for name in (
        "prefix_environment_steps",
        "branch_environment_steps",
        "environment_steps_total",
        "case_cpu_s",
        "case_wall_s",
        "prefix_cpu_s",
        "branch_cpu_s",
        "prefix_wall_s",
        "branch_wall_s",
    ):
        clean_costs[name] = _finite_number(costs.get(name), f"{context}.costs.{name}", allow_none=True)

    return {
        "case_id": str(case.get("case_id", f"seed-{seed}-{history}")),
        "seed": int(seed),
        "family": family,
        "history": history,
        "pair_kind": pair_kind,
        "metrics": metrics,
        "prefix_rows": prefix_rows,
        "branch_rows": _sum_known(branch_raw_rows),
        "all_branch_rows_present": bool(branch_raw_rows) and all(value is not None for value in branch_raw_rows),
        "environment_steps_total": clean_costs["environment_steps_total"],
        "costs": clean_costs,
        "raw_case": case,
    }, reasons


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    positive = sum(value > 1e-12 for value in values)
    negative = sum(value < -1e-12 for value in values)
    ties = len(values) - positive - negative
    return {
        "positive": int(positive),
        "negative": int(negative),
        "tie": int(ties),
        "denominator": int(len(values)),
    }


def _bootstrap_means(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rows_a = [row for row in per_seed if row["old_family"] == "A"]
    rows_b = [row for row in per_seed if row["old_family"] == "B"]
    values_a = np.asarray([[row["metrics"][metric] for metric in metrics] for row in rows_a], dtype=np.float64)
    values_b = np.asarray([[row["metrics"][metric] for metric in metrics] for row in rows_b], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices_a = rng.integers(0, len(rows_a), size=(resamples, len(rows_a)))
    indices_b = rng.integers(0, len(rows_b), size=(resamples, len(rows_b)))
    draws = 0.5 * values_a[indices_a].mean(axis=1) + 0.5 * values_b[indices_b].mean(axis=1)
    return {metric: draws[:, index] for index, metric in enumerate(metrics)}


def _summarize_values(values: Sequence[float], bootstrap: np.ndarray | None = None) -> dict[str, Any]:
    mean_value = _mean(values)
    if mean_value is None:
        return {"n": 0, "mean": None, "ci95": None, **_sign_counts([])}
    ci = None
    if bootstrap is not None:
        bounds = np.quantile(bootstrap, [0.025, 0.975])
        ci = [float(bounds[0]), float(bounds[1])]
    return {"n": len(values), "mean": mean_value, "ci95": ci, **_sign_counts(values)}


def _family_summary(per_seed: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("A", "B"):
        members = [row for row in per_seed if row["old_family"] == family]
        result[family] = {"seed_count": len(members), "metrics": {}}
        for metric in metrics:
            values = [float(row["metrics"][metric]) for row in members]
            result[family]["metrics"][metric] = _summarize_values(values)
    return result


def _history_summary(
    cases: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for history in HISTORIES:
        chosen: dict[int, Mapping[str, Any]] = {}
        for case in cases:
            if case["history"] == history and case["metrics"] is not None:
                chosen[int(case["seed"])] = case
        result[history] = {}
        for metric in metrics:
            values = [float(row["metrics"][metric]) for row in chosen.values()]
            result[history][metric] = _summarize_values(values)
    return result


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "N/A" if not math.isfinite(number) else f"{number:.{digits}f}"


def _markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Surrogate-Lower Confirmation Report",
        "",
        f"**Status: `{analysis['status']}`**",
        "",
    ]
    if analysis["status"] == "incomplete_pilot":
        observed = ", ".join(analysis["observed_histories"] or ["none"])
        lines.extend([
            f"This is an incomplete pilot with history condition(s) {observed}. It is not confirmation evidence; no confirmation interval is calculated.",
            "",
        ])
    elif analysis["status"] == "incomplete_cohort":
        lines.extend(["The input does not meet the frozen 16-seed, two-history confirmation design. No confirmation interval is calculated.", ""])
    else:
        lines.extend(["This report evaluates the frozen, matched-future confirmation conditions.", ""])

    if analysis["incomplete_reasons"]:
        lines.extend(["## Completeness notes", ""])
        lines.extend(f"- {reason}" for reason in analysis["incomplete_reasons"])
        lines.append("")

    lines.extend([
        "## Primary contrast",
        "",
        "The unit is the seed: within each seed, average the two matched history=future conditions, then average the 16 seeds equally. The bootstrap resamples eight seeds within old-family A and eight within B. Ties remain in the denominator.",
        "",
    ])
    primary = analysis["primary"]
    if primary["estimate"] is None:
        lines.extend(["No primary confirmation estimate is available for this incomplete input.", ""])
    else:
        estimate = primary["estimate"]
        ci = estimate["ci95"]
        ci_text = f"[{_format_number(ci[0])}, {_format_number(ci[1])}]" if ci else "N/A"
        lines.extend([
            f"Mean `surrogate_lower − keep`: **{_format_number(estimate['mean'])} reward/step**; stratified percentile 95% CI {ci_text}.",
            f"Directional result: **{primary['directional_status']}**. Positive / negative / tie seeds: {estimate['positive']} / {estimate['negative']} / {estimate['tie']} of {estimate['denominator']}.",
            "",
        ])

    practical = analysis["practical_threshold"]
    lines.extend([
        "## Original practical threshold",
        "",
        f"The frozen threshold remains **{_format_number(practical['threshold'])} reward/step** and is assessed separately from direction. Status: **{practical['status']}**.",
    ])
    if practical["estimate"] is not None:
        p_estimate = practical["estimate"]
        p_ci = p_estimate["ci95"]
        p_ci_text = f"[{_format_number(p_ci[0])}, {_format_number(p_ci[1])}]" if p_ci else "N/A"
        lines.append(f"Primary mean: {_format_number(p_estimate['mean'])}; 95% CI: {p_ci_text}.")
    lines.append("")

    lines.extend(["## Secondary contrasts", "", "Each contrast is `surrogate_lower` minus the named comparator. `frequency` uses the frozen prefix-only frequency choice. All scored values come from the already-paid physical action branches; branch scores are observer-only and do not update the learner.", ""])
    lines.extend([
        "| Comparator | Mean reward/step | Stratified 95% CI | Positive / negative / tie seeds |",
        "|---|---:|---:|---:|",
    ])
    for comparator in COMPARATORS[1:]:
        item = analysis["secondary"][comparator]
        if item["estimate"] is None:
            mean_text, ci_text, signs = "N/A", "N/A", "N/A"
        else:
            estimate = item["estimate"]
            ci = estimate["ci95"]
            mean_text = _format_number(estimate["mean"])
            ci_text = f"[{_format_number(ci[0])}, {_format_number(ci[1])}]" if ci else "N/A"
            signs = f"{estimate['positive']} / {estimate['negative']} / {estimate['tie']} of {estimate['denominator']}"
        lines.append(f"| {comparator} | {mean_text} | {ci_text} | {signs} |")
    lines.append("")

    lines.extend(["## Primary contrast by history and old family", "", "These are descriptive paired seed means; the overall confirmation interval is stratified by old family.", ""])
    lines.extend(["| Group | Seeds | Mean lower − keep | Positive / negative / tie |", "|---|---:|---:|---:|"])
    for history in HISTORIES:
        row = analysis["by_history"][history]["lower_minus_keep"]
        lines.append(f"| {history} | {row['n']} | {_format_number(row['mean'])} | {row['positive']} / {row['negative']} / {row['tie']} of {row['denominator']} |")
    for family in ("A", "B"):
        row = analysis["by_old_family"][family]["metrics"]["lower_minus_keep"]
        lines.append(f"| old family {family} | {row['n']} | {_format_number(row['mean'])} | {row['positive']} / {row['negative']} / {row['tie']} of {row['denominator']} |")
    lines.append("")

    costs = analysis["costs"]
    lines.extend(["## Cost and raw-row accounting", "", "| Quantity | Total |", "|---|---:|"])
    for label, key in (
        ("Environment steps from summary", "environment_steps_total"),
        ("Prefix environment steps", "prefix_environment_steps"),
        ("Branch environment steps", "branch_environment_steps"),
        ("Actual persisted step rows", "actual_rows"),
        ("CPU seconds", "cpu_s"),
        ("Wall seconds", "wall_s"),
    ):
        lines.append(f"| {label} | {_format_number(costs.get(key), digits=3)} |")
    lines.extend([
        "",
        f"Actual raw-row count complete: **{costs['actual_rows_complete']}**. Raw rows are summed from each case's saved `prefix.raw_environment_steps` and branch `raw_environment_steps` fields.",
        "",
        "## Reproducibility and scoring",
        "",
        f"Bootstrap: {analysis['bootstrap']['resamples']} stratified seed resamples, RNG seed {analysis['bootstrap']['seed']}, percentile interval. Input phase: `{analysis['input_phase']}`; observed histories: {', '.join(analysis['observed_histories']) or 'none'}.",
        "",
        "The report reads saved summaries only. It runs no RL, environment steps, or new branch evaluations.",
        "",
    ])
    return "\n".join(lines)


def _csv_rows(per_seed: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    columns = [
        "seed", "old_family", "case_count", "matched_history_count", "expected_history_count",
        "prefix_raw_rows", "branch_raw_rows", "actual_raw_rows", "environment_steps_total",
        "case_cpu_s", "case_wall_s",
    ]
    roles = ("surrogate_lower", "keep", "replace", "mean", "frequency", "none")
    deltas = tuple(f"lower_minus_{name}" for name in COMPARATORS)
    for history in HISTORIES:
        for name in roles + deltas:
            columns.append(f"{history}_{name}")
    for name in roles:
        columns.append(f"observed_history_mean_{name}")
    for name in deltas:
        columns.append(f"observed_history_mean_{name}")

    rows = []
    for seed_row in per_seed:
        row = {key: seed_row.get(key) for key in columns}
        for history in HISTORIES:
            history_metrics = seed_row["history_metrics"].get(history, {})
            for name in roles + deltas:
                row[f"{history}_{name}"] = history_metrics.get(name)
        for name in roles + deltas:
            row[f"observed_history_mean_{name}"] = seed_row["metrics"].get(name)
        rows.append(row)
    return columns, rows


def analyze(input_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Create JSON, Markdown, and per-seed CSV from a saved run summary."""

    source = Path(input_path)
    summary_path = source / "summary.json" if source.is_dir() else source
    if summary_path.name != "summary.json" or not summary_path.is_file():
        raise FileNotFoundError(f"expected summary.json under {source}")
    summary = _read_json(summary_path)
    if summary.get("format") != "storage_decision_summary_v1":
        raise ValueError("input format must be storage_decision_summary_v1")
    if summary.get("analysis_role") != "independent_surrogate_lower_confirmation":
        raise ValueError("input analysis_role must be independent_surrogate_lower_confirmation")
    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("input config must be an object")
    report_config = config.get("report")
    if not isinstance(report_config, Mapping):
        raise ValueError("config.report must be an object")
    resamples_value = _nonnegative_int(report_config.get("bootstrap_resamples"), "config.report.bootstrap_resamples")
    bootstrap_seed_value = _nonnegative_int(report_config.get("bootstrap_seed"), "config.report.bootstrap_seed")
    practical_threshold = _finite_number(report_config.get("practical_reward_per_step"), "config.report.practical_reward_per_step")
    assert resamples_value is not None and bootstrap_seed_value is not None and practical_threshold is not None
    if resamples_value != 10000 or bootstrap_seed_value != 20261019 or not math.isclose(practical_threshold, 0.002):
        raise ValueError("bootstrap settings and practical threshold must match the frozen protocol")
    if tuple(config.get("actions", ACTIONS)) != ACTIONS:
        raise ValueError(f"config.actions must be ordered as {ACTIONS}")

    raw_cases = summary.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("input cases must be a list")
    case_rows: list[dict[str, Any]] = []
    completeness_reasons: list[str] = []
    for index, case in enumerate(raw_cases):
        clean_case, case_reasons = _case_record(case, index)
        case_rows.append(clean_case)
        completeness_reasons.extend(case_reasons)

    phase = str(summary.get("phase", "unknown"))
    input_histories = tuple(str(value) for value in config.get("history_regimes", ()))
    observed_histories = sorted({row["history"] for row in case_rows if row["history"] in HISTORIES})
    is_pilot = phase == "pilot" or len(input_histories) < 2
    expected_seeds_raw = config.get("seeds", summary.get("seeds", []))
    if not isinstance(expected_seeds_raw, list):
        raise ValueError("config.seeds must be a list")
    expected_seeds = [_nonnegative_int(value, "config.seeds[]") for value in expected_seeds_raw]
    expected_seeds = [int(value) for value in expected_seeds if value is not None]
    summary_seeds_raw = summary.get("seeds", [])
    if not isinstance(summary_seeds_raw, list):
        raise ValueError("summary.seeds must be a list")
    summary_seeds = [_nonnegative_int(value, "summary.seeds[]") for value in summary_seeds_raw]
    summary_seeds = [int(value) for value in summary_seeds if value is not None]
    if not is_pilot:
        if tuple(input_histories) != HISTORIES:
            completeness_reasons.append("config.history_regimes must be old_major and new_major in frozen order")
        if len(expected_seeds) != EXPECTED_SEED_COUNT or len(set(expected_seeds)) != EXPECTED_SEED_COUNT:
            completeness_reasons.append("config.seeds must declare the 16 unique frozen confirmation seeds")
        if sorted(summary_seeds) != sorted(expected_seeds):
            completeness_reasons.append("summary.seeds do not match config.seeds")
        if len(set(summary_seeds)) != len(summary_seeds):
            completeness_reasons.append("summary.seeds contains duplicates")

    by_seed_history: dict[tuple[int, str], dict[str, Any]] = {}
    for row in case_rows:
        key = (row["seed"], row["history"])
        if key in by_seed_history:
            completeness_reasons.append(f"duplicate case for seed {key[0]} and history {key[1]}")
        else:
            by_seed_history[key] = row

    expected_histories = ("old_major",) if is_pilot else HISTORIES
    seed_universe = sorted(set(expected_seeds) | set(summary_seeds) | {row["seed"] for row in case_rows})
    if not is_pilot:
        expected_pairs = {(seed, history) for seed in expected_seeds for history in HISTORIES}
        observed_pairs = set(by_seed_history)
        for missing_seed, missing_history in sorted(expected_pairs - observed_pairs):
            completeness_reasons.append(f"missing case for seed {missing_seed} and history {missing_history}")
        for extra_seed, extra_history in sorted(observed_pairs - expected_pairs):
            completeness_reasons.append(f"unexpected case for seed {extra_seed} and history {extra_history}")

    family_by_seed: dict[int, str | None] = {}
    for seed in seed_universe:
        found = {row["family"] for row in case_rows if row["seed"] == seed and row["family"] is not None}
        if len(found) == 1:
            family_by_seed[seed] = next(iter(found))
        else:
            family_by_seed[seed] = None
            if not is_pilot:
                completeness_reasons.append(f"seed {seed} does not have one consistent old family")
    family_counts = {family: sum(value == family for value in family_by_seed.values()) for family in ("A", "B")}
    if not is_pilot and family_counts != EXPECTED_FAMILY_COUNTS:
        completeness_reasons.append(f"old-family strata must contain eight seeds each; got {family_counts}")

    input_errors = summary.get("errors", [])
    if input_errors:
        completeness_reasons.append("runner summary contains errors")
    confirmation = summary.get("confirmation")
    if not isinstance(confirmation, Mapping) or confirmation.get("selectors_are_past_only") is not True:
        completeness_reasons.append("confirmation metadata does not certify past-only selectors")
    if not is_pilot and int(summary.get("case_count", len(case_rows))) != EXPECTED_SEED_COUNT * len(HISTORIES):
        completeness_reasons.append("summary.case_count does not equal 32")
    if not is_pilot and int(summary.get("branch_count", 0)) != EXPECTED_SEED_COUNT * len(HISTORIES) * len(ACTIONS):
        completeness_reasons.append("summary.branch_count does not equal 128")

    # Preserve first occurrence and keep the report stable if multiple checks share a reason.
    completeness_reasons = list(dict.fromkeys(completeness_reasons))
    full_main = not is_pilot and not completeness_reasons
    status = "complete_confirmation" if full_main else ("incomplete_pilot" if is_pilot else "incomplete_cohort")

    metrics_names = tuple(
        ["surrogate_lower", "keep", "replace", "mean", "frequency", "none"]
        + [f"lower_minus_{name}" for name in COMPARATORS]
    )
    per_seed_rows: list[dict[str, Any]] = []
    expected_history_count = len(expected_histories)
    for seed in seed_universe:
        history_metrics: dict[str, dict[str, float] | dict[str, Any]] = {}
        seen_rows = [row for row in case_rows if row["seed"] == seed]
        seed_metrics_by_history = 0
        for history in HISTORIES:
            case = by_seed_history.get((seed, history))
            if case is None or case["metrics"] is None:
                history_metrics[history] = {}
                continue
            history_metrics[history] = dict(case["metrics"])
            seed_metrics_by_history += 1
        mean_metrics: dict[str, float | None] = {}
        for name in metrics_names:
            values = [
                float(history_metrics[history][name])
                for history in HISTORIES
                if name in history_metrics[history]
            ]
            mean_metrics[name] = _mean(values)

        seed_cases = [row for row in case_rows if row["seed"] == seed]
        prefix_rows = _sum_known([row["prefix_rows"] for row in seed_cases])
        branch_rows = _sum_known([row["branch_rows"] for row in seed_cases])
        actual_rows = _sum_known([prefix_rows, branch_rows])
        environment_steps = _sum_complete([row["environment_steps_total"] for row in seed_cases])
        costs_by_seed: dict[str, int | float | None] = {}
        for cost_name in (
            "case_cpu_s", "case_wall_s", "prefix_cpu_s", "branch_cpu_s", "prefix_wall_s", "branch_wall_s"
        ):
            costs_by_seed[cost_name] = _sum_complete([row["costs"].get(cost_name) for row in seed_cases])

        per_seed_rows.append({
            "seed": seed,
            "old_family": family_by_seed.get(seed),
            "case_count": len(seed_cases),
            "matched_history_count": seed_metrics_by_history,
            "expected_history_count": expected_history_count,
            "history_metrics": history_metrics,
            "metrics": mean_metrics,
            "prefix_raw_rows": prefix_rows,
            "branch_raw_rows": branch_rows,
            "actual_raw_rows": actual_rows,
            "environment_steps_total": environment_steps,
            **costs_by_seed,
        })

    observed_rows = [row for row in per_seed_rows if row["matched_history_count"] > 0]
    complete_metric_rows = [
        row for row in per_seed_rows
        if row["matched_history_count"] == len(HISTORIES)
        and all(row["metrics"][name] is not None for name in metrics_names)
    ]
    if full_main and len(complete_metric_rows) != EXPECTED_SEED_COUNT:
        full_main = False
        status = "incomplete_cohort"
        completeness_reasons.append("not every seed has both complete matched histories and observer scores")
    if full_main:
        # The seed rows are the inferential units. Keep any one-history pilot rows descriptive only.
        inference_rows = complete_metric_rows
        bootstrap_draws = _bootstrap_means(
            inference_rows,
            metrics=tuple(f"lower_minus_{name}" for name in COMPARATORS),
            resamples=int(resamples_value),
            seed=int(bootstrap_seed_value),
        )
    else:
        inference_rows = []
        bootstrap_draws = {}

    contrast_names = tuple(f"lower_minus_{name}" for name in COMPARATORS)
    primary_values = [float(row["metrics"]["lower_minus_keep"]) for row in inference_rows]
    primary_estimate = _summarize_values(
        primary_values,
        bootstrap_draws.get("lower_minus_keep"),
    ) if full_main else None
    if primary_estimate is None:
        directional_status = "not_evaluated_incomplete_input"
        practical_status = "not_evaluated_incomplete_input"
    else:
        lower_ci, upper_ci = primary_estimate["ci95"]
        if primary_estimate["mean"] > 0 and lower_ci > 0:
            directional_status = "directionally_confirmed"
        elif upper_ci <= 0:
            directional_status = "not_supported_by_this_cohort"
        else:
            directional_status = "uncertain"
        if lower_ci > practical_threshold:
            practical_status = "supports_practical_threshold"
        elif upper_ci <= practical_threshold:
            practical_status = "does_not_support_practical_threshold"
        else:
            practical_status = "uncertain_at_practical_threshold"

    secondary: dict[str, Any] = {}
    for comparator in COMPARATORS[1:]:
        key = f"lower_minus_{comparator}"
        values = [float(row["metrics"][key]) for row in inference_rows]
        estimate = _summarize_values(values, bootstrap_draws.get(key)) if full_main else None
        secondary[comparator] = {
            "estimate": estimate,
            "interpretation": "descriptive_secondary_contrast",
            "positive_negative_tie_denominator": None if estimate is None else estimate["denominator"],
        }

    history_metrics = _history_summary(case_rows, contrast_names)
    family_summary = _family_summary(complete_metric_rows, contrast_names) if full_main else {
        family: {"seed_count": 0, "metrics": {metric: _summarize_values([]) for metric in contrast_names}}
        for family in ("A", "B")
    }
    if full_main:
        family_summary = _family_summary(inference_rows, contrast_names)

    # Check actual row count coverage independently from statistical completeness.
    raw_row_complete = bool(case_rows) and all(
        row["prefix_rows"] is not None
        and row["branch_rows"] is not None
        and row["all_branch_rows_present"]
        for row in case_rows
    )
    actual_prefix_rows = _sum_known([row["prefix_rows"] for row in case_rows])
    actual_branch_rows = _sum_known([row["branch_rows"] for row in case_rows])
    actual_rows = _sum_known([actual_prefix_rows, actual_branch_rows])
    costs_input = summary.get("costs")
    costs_input = costs_input if isinstance(costs_input, Mapping) else {}
    total_costs = {
        name: _finite_number(costs_input.get(name), f"summary.costs.{name}", allow_none=True)
        for name in (
            "environment_steps_total", "prefix_environment_steps", "branch_environment_steps", "cpu_s",
            "child_cpu_s", "orchestrator_cpu_s", "wall_s", "prefix_cpu_s", "branch_cpu_s",
            "prefix_wall_s", "branch_wall_s",
        )
    }
    total_costs.update({
        "actual_prefix_rows": actual_prefix_rows,
        "actual_branch_rows": actual_branch_rows,
        "actual_rows": actual_rows,
        "actual_rows_complete": raw_row_complete,
    })

    primary = {
        "contrast": "surrogate_lower minus keep",
        "statistical_unit": "seed",
        "history_aggregation": "equal mean of old_major and new_major matched history=future conditions within seed",
        "estimate": primary_estimate,
        "directional_status": directional_status,
        "interpretation_rule": "direction confirmed only if mean > 0 and the stratified percentile 95% CI lower endpoint > 0",
    }
    practical = {
        "threshold": float(practical_threshold),
        "status": practical_status,
        "estimate": primary_estimate,
        "rule": "reported separately; support requires the 95% CI lower endpoint to exceed the original threshold",
    }
    analysis = {
        "format": REPORT_FORMAT,
        "status": status,
        "input_summary": str(summary_path.resolve()),
        "input_phase": phase,
        "observed_histories": observed_histories,
        "expected_histories": list(expected_histories),
        "seed_count": len(seed_universe),
        "case_count": len(case_rows),
        "family_seed_counts": family_counts,
        "incomplete_reasons": completeness_reasons,
        "scoring": {
            "source": "saved, already-paid four physical-action observer branches",
            "observer_only": True,
            "branch_outcomes_feed_back_to_learner": False,
            "primary_selected_action": "case.confirmation_action / selectors.surrogate_lower",
            "frequency_selected_action": "case.selectors.frequency",
            "matched_future_only": True,
        },
        "bootstrap": {
            "resamples": int(resamples_value),
            "seed": int(bootstrap_seed_value),
            "method": "stratified percentile cluster bootstrap over seed; sample 8 old-family A and 8 old-family B with replacement",
            "confidence_level": 0.95,
            "computed": bool(full_main),
        },
        "tie_rule": "per-seed differences within 1e-12 reward/step of zero count as ties and remain in the full seed denominator",
        "primary": primary,
        "practical_threshold": practical,
        "secondary": secondary,
        "by_history": history_metrics,
        "by_old_family": family_summary,
        "costs": total_costs,
        "per_seed": per_seed_rows,
        "per_seed_csv": "per_seed.csv",
    }

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "report.md").write_text(_markdown(analysis), encoding="utf-8")
    columns, rows = _csv_rows(per_seed_rows)
    with (output_path / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="run output directory or its summary.json")
    parser.add_argument("--output", type=Path, required=True, help="directory for analysis.json, report.md, and per_seed.csv")
    args = parser.parse_args(argv)
    result = analyze(args.input, args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "seed_count": result["seed_count"],
        "case_count": result["case_count"],
        "environment_steps_total": result["costs"]["environment_steps_total"],
        "actual_rows": result["costs"]["actual_rows"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

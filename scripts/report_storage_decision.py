"""Aggregate and report the frozen single shared-slot storage decision cohort.

The reporter reads saved runner JSON and leaf summaries only. It does not
import or execute the online runner, policy, or environment code.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FORMAT = "storage_decision_analysis_v1"
ACTIONS = ("keep", "replace", "mean", "none")
PAST_SELECTORS = ("frequency", "recency", "surrogate_gain", "surrogate_lower")
METHODS = ACTIONS + PAST_SELECTORS
PAIR_KINDS = ("dynamics", "reward_conflict")
REGIMES = ("old_major", "new_major")
METRICS = ("reward_total", "reward_per_step")
DEFAULT_CONFIG = {
    "seeds": [20101, 20107, 20113, 20117, 20123, 20129, 20143, 20149],
    "pair_kinds": list(PAIR_KINDS),
    "history_regimes": list(REGIMES),
    "future_regimes": list(REGIMES),
    "actions": list(ACTIONS),
    "bootstrap_steps": 2048,
    "bout_steps": 1024,
    "report": {
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20261018,
        "practical_reward_per_step": 0.002,
        "direction_consistency_minimum": 6,
    },
}
COLORS = {
    "dynamics": "#336B87",
    "reward_conflict": "#D17A22",
    "keep": "#5470C6",
    "replace": "#C44536",
    "mean": "#5B8E7D",
    "none": "#8A6FA8",
    "frequency": "#333333",
    "recency": "#4C9F70",
    "surrogate_gain": "#C07A18",
    "surrogate_lower": "#B04A87",
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite numeric data")
    return number


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _find_key(value: Any, key: str) -> Any | None:
    """Find the first matching key in a small decision metadata tree."""
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def _contains_key(value: Any, forbidden: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                found.append(str(key))
            found.extend(_contains_key(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.extend(_contains_key(child, forbidden))
    return found


def _config_from_input(input_dir: Path, root_summary: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (input_dir / "config.json", input_dir / "config" / "storage_decision.json")
    config: Any = None
    for path in candidates:
        if path.is_file():
            config = _read_json(path)
            break
    if config is None:
        config = root_summary.get("config", DEFAULT_CONFIG)
    if not isinstance(config, dict):
        raise ValueError("cohort config must be a JSON object")
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in config.items():
        merged[key] = value
    report = dict(DEFAULT_CONFIG["report"])
    if isinstance(merged.get("report"), dict):
        report.update(merged["report"])
    merged["report"] = report
    for field in ("seeds", "pair_kinds", "history_regimes", "future_regimes", "actions"):
        value = merged.get(field)
        if not isinstance(value, list) or not value:
            raise ValueError(f"config.{field} must be a non-empty list")
    if tuple(merged["actions"]) != ACTIONS:
        raise ValueError(f"config.actions must be ordered as {ACTIONS}")
    for field, allowed in (("pair_kinds", PAIR_KINDS), ("history_regimes", REGIMES), ("future_regimes", REGIMES)):
        if set(merged[field]) != set(allowed):
            raise ValueError(f"config.{field} must contain {allowed}")
    return merged


def _case_factors(decision: Mapping[str, Any]) -> tuple[int, str, str]:
    seed = decision.get("seed")
    pair_kind = decision.get("pair_kind")
    history = decision.get("history_regime")
    if seed is None or pair_kind is None or history is None:
        factors = decision.get("case")
        if isinstance(factors, dict):
            seed = seed if seed is not None else factors.get("seed")
            pair_kind = pair_kind if pair_kind is not None else factors.get("pair_kind")
            history = history if history is not None else factors.get("history_regime")
    if isinstance(seed, bool):
        raise ValueError("decision.seed must be an integer")
    try:
        seed_int = int(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("decision.seed must be an integer") from exc
    if seed_int != seed:
        raise ValueError("decision.seed must be an integer")
    if pair_kind not in PAIR_KINDS or history not in REGIMES:
        raise ValueError("decision pair_kind/history_regime is outside the frozen protocol")
    return seed_int, str(pair_kind), str(history)


def _selector_actions(decision: Mapping[str, Any]) -> dict[str, str]:
    candidates = (
        decision.get("past_only_selectors"),
        decision.get("selector_actions"),
        decision.get("selectors"),
        _find_key(decision, "past_only_selectors"),
    )
    selected: Any = next((item for item in candidates if isinstance(item, dict)), None)
    if selected is None:
        raise ValueError("decision is missing its past_only_selectors action map")
    actions: dict[str, str] = {}
    for name in PAST_SELECTORS:
        action = selected.get(name)
        if action not in ("keep", "replace"):
            raise ValueError(f"past selector {name} must choose keep or replace")
        actions[name] = str(action)
    for action in ACTIONS:
        if selected.get(action) != action:
            raise ValueError(f"past_only_selectors.{action} must be the fixed {action} action")
    return actions


def _selector_audit(decision: Mapping[str, Any], actions: Mapping[str, str]) -> dict[str, Any]:
    recent = _find_key(decision, "recent_group_ids")
    score = _find_key(decision, "surrogate")
    if not isinstance(score, dict):
        score = {}
    duplicate_map = _find_key(decision, "selectors")
    selector_inputs = _find_key(decision, "selector_diagnostics")
    if not isinstance(selector_inputs, dict):
        selector_inputs = {"recent_group_ids": recent, "surrogate": score}
    forbidden = _contains_key(selector_inputs, {"future_regime", "future_regimes", "future_schedule", "future_actions"})
    checked: dict[str, Any] = {
        "decision_contains_future_schedule_keys": forbidden,
        "prefix_history_available": isinstance(recent, list),
        "selector_maps_consistent": None,
        "frequency_recomputed": None,
        "recency_recomputed": None,
        "surrogate_gain_recomputed": None,
        "surrogate_lower_recomputed": None,
    }
    if isinstance(duplicate_map, dict):
        checked["selector_maps_consistent"] = all(duplicate_map.get(name) == action for name, action in actions.items())
    if isinstance(recent, list) and len(recent) == 8 and all(item in (0, 1) for item in recent):
        expected_frequency = "replace" if recent.count(1) > recent.count(0) else "keep"
        latest_old = max((i for i, group in enumerate(recent) if group == 0), default=-1)
        latest_new = max((i for i, group in enumerate(recent) if group == 1), default=-1)
        expected_recency = "replace" if latest_new > latest_old else "keep"
        checked["frequency_recomputed"] = actions["frequency"] == expected_frequency
        checked["recency_recomputed"] = actions["recency"] == expected_recency
    for name, fields in (("surrogate_gain", ("frequency_weighted_gain", "weighted_gain", "gain")),
                         ("surrogate_lower", ("frequency_weighted_lower", "weighted_lower", "lower"))):
        field = next((candidate for candidate in fields if candidate in score), fields[0])
        raw = score.get(field)
        if raw is None:
            checked[f"{name}_recomputed"] = actions[name] == "keep"
        else:
            value = _finite(raw, f"decision.surrogate.{field}")
            expected = "replace" if value > 0.0 else "keep"
            checked[f"{name}_recomputed"] = actions[name] == expected
    values = [
        checked["prefix_history_available"],
        checked["selector_maps_consistent"] is True,
        checked["frequency_recomputed"] is True,
        checked["recency_recomputed"] is True,
        checked["surrogate_gain_recomputed"] is True,
        checked["surrogate_lower_recomputed"] is True,
        not forbidden,
    ]
    checked["passed"] = all(values)
    return checked


def _hash_audit(decision: Mapping[str, Any]) -> dict[str, Any]:
    prefix_hash = _find_key(decision, "prefix_state_hash")
    restored = _find_key(decision, "restored_state_hashes")
    restored_values: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            restored_values.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    if restored is not None:
        collect(restored)
    matches = None
    if prefix_hash is not None and restored_values:
        matches = all(value == str(prefix_hash) for value in restored_values)
    return {"prefix_hash_present": prefix_hash is not None,
            "restored_hash_count": len(restored_values),
            "all_restores_match_prefix": matches}


def _summary_steps(summary: Mapping[str, Any]) -> int | None:
    for key in ("environment_steps", "total_steps", "online_steps", "steps"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    per_bout = summary.get("per_bout")
    if isinstance(per_bout, list) and per_bout:
        vals = [row.get("steps") for row in per_bout if isinstance(row, dict) and row.get("steps") is not None]
        if vals:
            return sum(int(value) for value in vals)
    return None


def _summary_cost(summary: Mapping[str, Any], kind: str) -> float | None:
    keys = ("cpu_seconds", "cpu_s", "cpu_seconds_total") if kind == "cpu" else ("wall_seconds", "wall_s", "elapsed_seconds", "elapsed_s")
    for key in keys:
        if summary.get(key) is not None:
            return _finite(summary[key], f"summary.{key}")
    nested = summary.get("costs")
    if isinstance(nested, dict):
        keys = ("cpu_s", "cpu_seconds") if kind == "cpu" else ("wall_s", "wall_seconds")
        for key in keys:
            if nested.get(key) is not None:
                return _finite(nested[key], f"summary.costs.{key}")
    return None


def _load_leaf(path: Path, errors: list[dict[str, Any]], *, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append({"type": "invalid_json", "path": str(path), "label": label, "message": str(exc)})
        return None


def _load_cases(input_dir: Path, config: Mapping[str, Any], root_summary: Mapping[str, Any]) -> tuple[dict[tuple[int, str, str], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cases: dict[tuple[int, str, str], dict[str, Any]] = {}
    parse_errors: list[dict[str, Any]] = []
    cases_root = input_dir / "cases"
    if not cases_root.is_dir():
        return cases, parse_errors, []
    observed_case_dirs = sorted(path.parent for path in cases_root.glob("*/decision.json"))
    extras: list[dict[str, Any]] = []
    main_seeds = {int(seed) for seed in config["seeds"]}
    pilot_seeds = set()
    pilot = config.get("pilot")
    if isinstance(pilot, dict) and isinstance(pilot.get("seeds"), list):
        pilot_seeds = {int(seed) for seed in pilot["seeds"]}
    for case_dir in observed_case_dirs:
        decision_path = case_dir / "decision.json"
        decision = _load_leaf(decision_path, parse_errors, label="decision")
        if decision is None:
            continue
        try:
            seed, pair_kind, history = _case_factors(decision)
            selector_actions = _selector_actions(decision)
            audit = _selector_audit(decision, selector_actions)
        except (TypeError, ValueError) as exc:
            parse_errors.append({"type": "invalid_decision", "path": str(decision_path), "message": str(exc)})
            continue
        prefix = _load_leaf(case_dir / "prefix" / "summary.json", parse_errors, label="prefix")
        branches: dict[tuple[str, str], dict[str, Any]] = {}
        for action in ACTIONS:
            for future in REGIMES:
                branch_path = case_dir / "branches" / f"{action}__{future}" / "summary.json"
                branch = _load_leaf(branch_path, parse_errors, label="branch")
                if branch is not None:
                    branches[(action, future)] = branch
        record = {
            "case_id": str(decision.get("case_id", case_dir.name)),
            "seed": seed,
            "pair_kind": pair_kind,
            "history_regime": history,
            "directory": str(case_dir),
            "decision": decision,
            "selector_actions": selector_actions,
            "selector_audit": audit,
            "hash_audit": _hash_audit(decision),
            "prefix": prefix,
            "branches": branches,
            "is_pilot": seed in pilot_seeds and seed not in main_seeds,
        }
        key = (seed, pair_kind, history)
        if key in cases:
            parse_errors.append({"type": "duplicate_case", "case_key": list(key), "paths": [cases[key]["directory"], str(case_dir)]})
        else:
            if seed not in main_seeds:
                extras.append({"case_id": record["case_id"], "seed": seed, "pair_kind": pair_kind,
                               "history_regime": history, "classified_as": "pilot_or_auxiliary" if seed in pilot_seeds else "auxiliary",
                               "directory": str(case_dir)})
            cases[key] = record
    # Retain failures/root records exactly as recorded. They may represent a
    # case that died before it could write decision.json.
    root_failures = root_summary.get("errors", root_summary.get("failures", []))
    if root_failures:
        if isinstance(root_failures, list):
            parse_errors.extend({"type": "runner_recorded_failure", "record": row} for row in root_failures)
        else:
            parse_errors.append({"type": "runner_recorded_failure", "record": root_failures})
    return cases, parse_errors, extras


def _complete_branch(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    status = str(summary.get("status", "complete")).lower()
    if summary.get("failed") is True or status in {"failed", "error", "incomplete"}:
        return False
    return all(summary.get(field) is not None for field in METRICS)


def _seed_summary(values: Mapping[int, float], *, resamples: int, bootstrap_seed: int) -> dict[str, Any]:
    rows = [{"seed": int(seed), "value": float(value)} for seed, value in sorted(values.items())]
    vals = [row["value"] for row in rows]
    if not vals:
        return {"n_seeds": 0, "per_seed": [], "mean": None, "min": None, "max": None, "percentile_ci95": None,
                "bootstrap": {"resamples": int(resamples), "seed": int(bootstrap_seed), "unit": "deployment seed cluster"}}
    mean = statistics.fmean(vals)
    if resamples > 0:
        rng = np.random.default_rng(int(bootstrap_seed))
        indices = rng.integers(0, len(vals), size=(int(resamples), len(vals)))
        boot_means = np.mean(np.asarray(vals, dtype=np.float64)[indices], axis=1)
        ci = [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]
    else:
        ci = None
    return {
        "n_seeds": len(vals), "per_seed": rows, "mean": mean,
        "min": min(vals), "max": max(vals), "percentile_ci95": ci,
        "bootstrap": {"resamples": int(resamples), "seed": int(bootstrap_seed), "unit": "deployment seed cluster"},
    }


def _future_for(history: str, condition: str) -> str:
    return history if condition == "primary" else ("new_major" if history == "old_major" else "old_major")


def _case_value(case: Mapping[str, Any], method: str, condition: str, metric: str) -> float | None:
    future = _future_for(str(case["history_regime"]), condition)
    action = method if method in ACTIONS else case["selector_actions"].get(method)
    if action not in ACTIONS:
        return None
    summary = case["branches"].get((action, future))
    if not _complete_branch(summary):
        return None
    try:
        return _finite(summary[metric], f"branch.{metric}")
    except ValueError:
        return None


def _paired_seed_values(
    cases: Mapping[tuple[int, str, str], Mapping[str, Any]],
    pair_kind: str,
    condition: str,
    method: str,
    metric: str,
    *,
    baseline: str | None = None,
) -> dict[int, float]:
    result: dict[int, float] = {}
    seeds = sorted({key[0] for key in cases if key[1] == pair_kind})
    for seed in seeds:
        history_values: list[float] = []
        valid = True
        for history in REGIMES:
            case = cases.get((seed, pair_kind, history))
            if case is None:
                valid = False
                break
            value = _case_value(case, method, condition, metric)
            if value is None:
                valid = False
                break
            if baseline is not None:
                base = _case_value(case, baseline, condition, metric)
                if base is None:
                    valid = False
                    break
                value -= base
            history_values.append(value)
        if valid and len(history_values) == len(REGIMES):
            # The protocol's analysis unit is the deployment seed. Merge its
            # correlated history conditions before any cross-seed bootstrap.
            result[seed] = statistics.fmean(history_values)
    return result


def _interaction_seed_values(cases: Mapping[tuple[int, str, str], Mapping[str, Any]], pair_kind: str, metric: str) -> dict[int, float]:
    result: dict[int, float] = {}
    seeds = sorted({key[0] for key in cases if key[1] == pair_kind})
    for seed in seeds:
        history_interactions: list[float] = []
        valid = True
        for history in REGIMES:
            case = cases.get((seed, pair_kind, history))
            if case is None:
                valid = False
                break
            outcomes: dict[tuple[str, str], float] = {}
            for action in ("keep", "replace"):
                for future in REGIMES:
                    branch = case["branches"].get((action, future))
                    if not _complete_branch(branch):
                        valid = False
                        break
                    try:
                        outcomes[(action, future)] = _finite(branch[metric], f"branch.{metric}")
                    except ValueError:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            d_new = outcomes[("replace", "new_major")] - outcomes[("keep", "new_major")]
            d_old = outcomes[("replace", "old_major")] - outcomes[("keep", "old_major")]
            history_interactions.append(d_new - d_old)
        if valid and len(history_interactions) == len(REGIMES):
            result[seed] = statistics.fmean(history_interactions)
    return result


def _d_preference_crossover(
    cases: Mapping[tuple[int, str, str], Mapping[str, Any]], pair_kind: str, metric: str,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Describe where replace is worse under old recurrence but better under new."""
    per_case: list[dict[str, Any]] = []
    by_seed: dict[int, list[tuple[float, float]]] = {}
    seeds = sorted(int(seed) for seed in expected_seeds) if expected_seeds is not None else sorted({key[0] for key in cases if key[1] == pair_kind})
    for seed in seeds:
        seed_rows: list[tuple[float, float]] = []
        for history in REGIMES:
            case = cases.get((seed, pair_kind, history))
            if case is None:
                continue
            branches = case["branches"]
            required = (("replace", "old_major"), ("keep", "old_major"),
                        ("replace", "new_major"), ("keep", "new_major"))
            if any(not _complete_branch(branches.get(key)) for key in required):
                continue
            d_old = _finite(branches[("replace", "old_major")][metric], f"branch.{metric}") - _finite(branches[("keep", "old_major")][metric], f"branch.{metric}")
            d_new = _finite(branches[("replace", "new_major")][metric], f"branch.{metric}") - _finite(branches[("keep", "new_major")][metric], f"branch.{metric}")
            crossover = d_old < 0.0 and d_new > 0.0
            row = {"seed": seed, "history_regime": history, "D_old_future": d_old,
                   "D_new_future": d_new, "replace_preference_reverses": crossover}
            per_case.append(row)
            seed_rows.append((d_old, d_new))
        if len(seed_rows) == len(REGIMES):
            by_seed[seed] = seed_rows
    d_old_seed = {seed: statistics.fmean(row[0] for row in rows) for seed, rows in by_seed.items()}
    d_new_seed = {seed: statistics.fmean(row[1] for row in rows) for seed, rows in by_seed.items()}
    crossovers = sum(row["replace_preference_reverses"] for row in per_case)
    return {
        "per_case_crossover_count": crossovers,
        "per_case_count_with_complete_D": len(per_case),
        "expected_per_case_count": len(seeds) * len(REGIMES),
        "per_case_crossover_fraction": crossovers / len(per_case) if per_case else None,
        "per_case": per_case,
        "per_seed_D_old_future": d_old_seed,
        "per_seed_D_new_future": d_new_seed,
    }


def _condition_analysis(cases: Mapping[tuple[int, str, str], Mapping[str, Any]], config: Mapping[str, Any], condition: str) -> dict[str, Any]:
    resamples = int(config["report"]["bootstrap_resamples"])
    bootstrap_seed = int(config["report"]["bootstrap_seed"])
    result: dict[str, Any] = {}
    for pair_kind in PAIR_KINDS:
        methods: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            methods[method] = {
                metric: _seed_summary(
                    _paired_seed_values(cases, pair_kind, condition, method, metric),
                    resamples=resamples, bootstrap_seed=bootstrap_seed,
                ) for metric in METRICS
            }
        contrasts: dict[str, dict[str, Any]] = {}
        for selector in PAST_SELECTORS:
            baselines = ACTIONS + (("frequency",) if selector != "frequency" else ())
            for baseline in baselines:
                if selector == baseline:
                    continue
                contrast_name = f"{selector}_minus_{baseline}"
                contrasts[contrast_name] = {
                    metric: _seed_summary(
                        _paired_seed_values(cases, pair_kind, condition, selector, metric, baseline=baseline),
                        resamples=resamples, bootstrap_seed=bootstrap_seed,
                    ) for metric in METRICS
                }
        result[pair_kind] = {"methods": methods, "selector_paired_contrasts": contrasts}
    return result


def _cost_summary(cases: Mapping[tuple[int, str, str], Mapping[str, Any]], config: Mapping[str, Any], root_summary: Mapping[str, Any], expected_case_count: int) -> dict[str, Any]:
    def empty_costs() -> dict[str, Any]:
        return {"summaries": 0, "steps": 0, "steps_unknown": 0, "cpu_seconds": 0.0, "cpu_summaries": 0,
                "wall_seconds": 0.0, "wall_summaries": 0,
                "rollouts": 0, "rollouts_summaries": 0,
                "minibatch_updates": 0, "minibatch_updates_summaries": 0,
                "allocations": 0, "allocations_summaries": 0, "evictions": 0, "evictions_summaries": 0}

    prefix_costs = empty_costs()
    branch_costs = empty_costs()
    physical_sizes: dict[str, list[int]] = {key: [] for key in ("actor_bytes", "checkpoint_bytes", "w0_bytes", "sidecar_bytes", "package_bytes")}
    package_cap = config.get("package_budget_bytes")
    counter_aliases = {
        "rollouts": ("rollouts", "rollout_count", "n_rollouts", "rollouts_trained"),
        "minibatch_updates": ("minibatch_updates", "minibatches", "minibatch_count", "n_minibatches", "optimizer_updates", "update_count", "ppo_updates", "updates"),
        "allocations": ("allocations", "allocation_count", "allocations_count", "slot_allocations"),
        "evictions": ("evictions", "eviction_count", "evictions_count", "slot_evictions"),
    }

    def absorb(summary: Mapping[str, Any], target: dict[str, Any]) -> None:
        target["summaries"] += 1
        steps = _summary_steps(summary)
        if steps is None:
            target["steps_unknown"] += 1
        else:
            target["steps"] += steps
        cpu = _summary_cost(summary, "cpu")
        wall = _summary_cost(summary, "wall")
        if cpu is not None:
            target["cpu_seconds"] += cpu
            target["cpu_summaries"] += 1
        if wall is not None:
            target["wall_seconds"] += wall
            target["wall_summaries"] += 1
        for metric, aliases in counter_aliases.items():
            raw = next((summary[key] for key in aliases if summary.get(key) is not None), None)
            counts = summary.get("counts")
            if raw is None and isinstance(counts, dict):
                count_aliases = {
                    "rollouts": ("rollouts_trained", "rollouts"),
                    "minibatch_updates": ("minibatch_updates", "minibatches", "optimizer_updates", "updates"),
                    "allocations": ("slot_allocations", "allocations"),
                    "evictions": ("slot_evictions", "evictions"),
                }[metric]
                raw = next((counts[key] for key in count_aliases if counts.get(key) is not None), None)
            if raw is not None:
                target[metric] += int(raw)
                target[f"{metric}_summaries"] += 1
    for case in cases.values():
        summary = case.get("prefix")
        if isinstance(summary, dict):
            absorb(summary, prefix_costs)
        for summary in case["branches"].values():
            absorb(summary, branch_costs)
            capacity = summary.get("capacity") if isinstance(summary.get("capacity"), dict) else {}
            for metric, aliases in {
                "actor_bytes": ("actor_bytes", "actor_package_bytes", "actor_size_bytes", "actor_payload_bytes"),
                "checkpoint_bytes": ("checkpoint_bytes", "full_checkpoint_bytes"),
                "w0_bytes": ("w0_bytes", "foundation_bytes"),
                "sidecar_bytes": ("sidecar_bytes", "actor_sidecar_bytes", "actor_bytes"),
                "package_bytes": ("package_bytes", "whole_package_bytes", "checkpoint_package_bytes", "full_persistent_bytes"),
            }.items():
                raw = next((summary[key] for key in aliases if summary.get(key) is not None), None)
                if raw is None:
                    raw = next((capacity[key] for key in aliases if capacity.get(key) is not None), None)
                if raw is not None:
                    physical_sizes[metric].append(int(raw))
    prefix_expected_per_case = 2 * int(config["bootstrap_steps"]) + 8 * int(config["bout_steps"])
    branch_expected_per_branch = 12 * int(config["bout_steps"])
    expected_branches = expected_case_count * len(ACTIONS) * len(REGIMES)
    package_sizes = physical_sizes["package_bytes"]
    observed_package_cap_violations = [value for value in package_sizes if package_cap is not None and value > int(package_cap)]
    runner_costs = root_summary.get("costs", root_summary.get("cost_totals"))
    return {
        "expected_main": {
            "case_count": expected_case_count,
            "prefix_steps_per_case": prefix_expected_per_case,
            "branch_steps_per_branch": branch_expected_per_branch,
            "branch_count": expected_branches,
            "steps_total": expected_case_count * prefix_expected_per_case + expected_branches * branch_expected_per_branch,
        },
        "observed_leaf_summaries": {
            "prefix": prefix_costs,
            "branches": branch_costs,
            "steps_total": prefix_costs["steps"] + branch_costs["steps"],
            "cpu_seconds_total": prefix_costs["cpu_seconds"] + branch_costs["cpu_seconds"],
            "wall_seconds_sum": prefix_costs["wall_seconds"] + branch_costs["wall_seconds"],
            "physical_bytes_ranges": {key: {"min": min(values), "max": max(values), "n": len(values)} if values else {"min": None, "max": None, "n": 0}
                                      for key, values in physical_sizes.items()},
            "package_cap_bytes": package_cap,
            "package_cap_violation_count": len(observed_package_cap_violations),
            "actor_cap_bytes": config.get("actor_budget_bytes"),
            "actor_cap_violation_count": (
                sum(value > int(config["actor_budget_bytes"]) for value in physical_sizes["actor_bytes"])
                if config.get("actor_budget_bytes") is not None else 0
            ),
        },
        "runner_recorded_totals": runner_costs,
    }


def analyze_summary(input_dir: Path, root_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build per-seed paired estimates, bootstrap summaries and completeness audit."""
    input_dir = Path(input_dir)
    if root_summary is None:
        root_summary = _read_json(input_dir / "summary.json")
    if not isinstance(root_summary, Mapping):
        raise ValueError("root summary must be a JSON object")
    config = _config_from_input(input_dir, root_summary)
    cases, parse_errors, extras = _load_cases(input_dir, config, root_summary)
    expected_seeds = [int(seed) for seed in config["seeds"]]
    expected_case_keys = {
        (seed, pair_kind, history)
        for seed in expected_seeds
        for pair_kind in PAIR_KINDS
        for history in REGIMES
    }
    expected_branch_count = len(expected_case_keys) * len(ACTIONS) * len(REGIMES)
    incomplete_cases: list[dict[str, Any]] = []
    for key in sorted(expected_case_keys):
        case = cases.get(key)
        if case is None:
            incomplete_cases.append({"case_key": list(key), "missing_case_record": True, "missing_branches": []})
            continue
        missing: list[str] = []
        for action in ACTIONS:
            for future in REGIMES:
                if not _complete_branch(case["branches"].get((action, future))):
                    missing.append(f"{action}__{future}")
        if missing:
            incomplete_cases.append({"case_id": case["case_id"], "case_key": list(key), "missing_case_record": False, "missing_branches": missing})
    actual_expected_cases = {key: cases[key] for key in cases if key in expected_case_keys}
    complete_case_keys = [key for key, case in actual_expected_cases.items() if all(
        _complete_branch(case["branches"].get((action, future))) for action in ACTIONS for future in REGIMES
    )]
    selector_audits = [case["selector_audit"] for case in actual_expected_cases.values()]
    selector_audit = {
        "decision_records_checked": len(selector_audits),
        "prefix_only_pass_count": sum(bool(row["passed"]) for row in selector_audits),
        "all_passed": bool(selector_audits) and all(bool(row["passed"]) for row in selector_audits),
        "records": [
            {"case_id": case["case_id"], "seed": case["seed"], "pair_kind": case["pair_kind"],
             "history_regime": case["history_regime"], **case["selector_audit"]}
            for case in actual_expected_cases.values()
        ],
    }
    hash_audit_records = [
        {"case_id": case["case_id"], "seed": case["seed"], "pair_kind": case["pair_kind"],
         "history_regime": case["history_regime"], **case["hash_audit"]}
        for case in actual_expected_cases.values()
    ]
    recorded_hash_rows = [row for row in hash_audit_records if row["prefix_hash_present"] and row["restored_hash_count"] > 0]
    prefix_count = sum(case["prefix"] is not None for case in actual_expected_cases.values())
    branch_summary_count = sum(len(case["branches"]) for case in actual_expected_cases.values())
    interaction: dict[str, Any] = {}
    preference_crossover: dict[str, Any] = {}
    mechanism_check: dict[str, Any] = {}
    for pair_kind in PAIR_KINDS:
        interaction[pair_kind] = {
            metric: _seed_summary(
                _interaction_seed_values(actual_expected_cases, pair_kind, metric),
                resamples=int(config["report"]["bootstrap_resamples"]),
                bootstrap_seed=int(config["report"]["bootstrap_seed"]),
            ) for metric in METRICS
        }
        preference_crossover[pair_kind] = {}
        for metric in METRICS:
            crossover = _d_preference_crossover(actual_expected_cases, pair_kind, metric, expected_seeds)
            crossover["per_seed_D_old_future_summary"] = _seed_summary(
                crossover.pop("per_seed_D_old_future"),
                resamples=int(config["report"]["bootstrap_resamples"]),
                bootstrap_seed=int(config["report"]["bootstrap_seed"]),
            )
            crossover["per_seed_D_new_future_summary"] = _seed_summary(
                crossover.pop("per_seed_D_new_future"),
                resamples=int(config["report"]["bootstrap_resamples"]),
                bootstrap_seed=int(config["report"]["bootstrap_seed"]),
            )
            preference_crossover[pair_kind][metric] = crossover
        per_step = interaction[pair_kind]["reward_per_step"]
        positive_count = sum(row["value"] > 0.0 for row in per_step["per_seed"])
        complete_n = per_step["n_seeds"]
        minimum = int(config["report"]["direction_consistency_minimum"])
        expected_n = len(expected_seeds)
        if complete_n < expected_n:
            status = "not_assessable_incomplete_seed_cohort"
        elif per_step["mean"] is not None and per_step["mean"] > 0.0 and positive_count >= minimum:
            status = "predeclared_direction_rule_met"
        else:
            status = "predeclared_direction_rule_not_met"
        mechanism_check[pair_kind] = {
            "mean_I_positive": per_step["mean"] is not None and per_step["mean"] > 0.0,
            "positive_seed_count": positive_count,
            "observed_seed_count": complete_n,
            "expected_seed_count": expected_n,
            "minimum_positive_seeds": minimum,
            "status": status,
        }
    conditions = {name: _condition_analysis(actual_expected_cases, config, name) for name in ("primary", "stress")}
    case_recorded = [
        {"case_id": case["case_id"], "seed": case["seed"], "pair_kind": case["pair_kind"],
         "history_regime": case["history_regime"], "selector_actions": case["selector_actions"],
         "prefix_restore_hashes": case["hash_audit"],
         "branch_summaries": len(case["branches"]), "prefix_summary_present": case["prefix"] is not None,
         "complete": all(_complete_branch(case["branches"].get((action, future))) for action in ACTIONS for future in REGIMES)}
        for case in actual_expected_cases.values()
    ]
    branch_diagnostics: list[dict[str, Any]] = []
    for case in actual_expected_cases.values():
        for action in ACTIONS:
            for future in REGIMES:
                summary = case["branches"].get((action, future))
                if not isinstance(summary, dict):
                    continue
                per_bout = summary.get("per_bout", [])
                branch_diagnostics.append({
                    "case_id": case["case_id"], "seed": case["seed"],
                    "pair_kind": case["pair_kind"], "history_regime": case["history_regime"],
                    "action": action, "future_regime": future,
                    "continuation_condition": "primary" if case["history_regime"] == future else "stress",
                    "reward_total": summary.get("reward_total"),
                    "reward_per_step": summary.get("reward_per_step"),
                    "per_group": summary.get("per_group"),
                    "per_bout": [
                        {key: row.get(key) for key in (
                            "bout", "task", "group", "steps", "allocated", "evicted_task",
                            "return_first_256", "return_first_512", "return_full",
                        ) if key in row}
                        for row in per_bout if isinstance(row, dict)
                    ] if isinstance(per_bout, list) else [],
                    "counts": summary.get("counts"),
                    "costs": summary.get("costs"),
                    "capacity": summary.get("capacity"),
                    "capacity_after_action": summary.get("capacity_after_action"),
                    "preservation_diagnostic": summary.get("preservation_diagnostic"),
                    "prequantized_behavior_drift": summary.get("prequantized_behavior_drift"),
                    "postquantized_behavior_drift": summary.get("postquantized_behavior_drift"),
                    "foundation_unchanged": summary.get("foundation_unchanged"),
                })
    report_cfg = config["report"]
    return {
        "format": FORMAT,
        "cohort": {
            "analysis_role": root_summary.get("analysis_role", "predeclared controlled mechanism"),
            "inference_seed_ids": expected_seeds,
            "pair_kinds": list(PAIR_KINDS),
            "history_regimes": list(REGIMES),
            "future_regimes": list(REGIMES),
            "expected_case_count": len(expected_case_keys),
            "observed_main_case_records": len(actual_expected_cases),
            "complete_main_case_records": len(complete_case_keys),
            "expected_branch_count": expected_branch_count,
            "observed_branch_summaries": branch_summary_count,
            "pilot_or_auxiliary_cases_excluded_from_inference": extras,
            "case_records": case_recorded,
        },
        "branch_diagnostics": branch_diagnostics,
        "static_checks": {
            "prefix_completeness": {
                "expected_prefix_count": len(expected_case_keys),
                "observed_prefix_summary_count": prefix_count,
                "all_prefix_summaries_present": prefix_count == len(expected_case_keys),
            },
            "branch_completeness": {
                "expected_branch_count": expected_branch_count,
                "observed_branch_summary_count": branch_summary_count,
                "all_expected_cases_complete": len(complete_case_keys) == len(expected_case_keys),
                "complete_case_count": len(complete_case_keys),
                "incomplete_cases": incomplete_cases,
            },
            "selector_prefix_only": selector_audit,
            "prefix_restore_hashes": {
                "decision_records_checked": len(hash_audit_records),
                "hashes_recorded": len(recorded_hash_rows),
                "matched_records": sum(row["all_restores_match_prefix"] is True for row in recorded_hash_rows),
                "all_recorded_hashes_match": bool(recorded_hash_rows) and all(row["all_restores_match_prefix"] is True for row in recorded_hash_rows),
                "records": hash_audit_records,
            },
            "runner_and_parse_failures": parse_errors,
        },
        "interaction_I": interaction,
        "preference_crossover": preference_crossover,
        "predeclared_mechanism_direction_check": mechanism_check,
        "continuation_conditions": conditions,
        "costs": _cost_summary(actual_expected_cases, config, root_summary, len(expected_case_keys)),
        "decision_thresholds": {
            "practical_reward_per_step_reference": float(report_cfg["practical_reward_per_step"]),
            "mechanism_direction_consistency_minimum": int(report_cfg["direction_consistency_minimum"]),
            "bootstrap_resamples": int(report_cfg["bootstrap_resamples"]),
            "bootstrap_seed": int(report_cfg["bootstrap_seed"]),
            "interpretation": "Descriptive percentile intervals; no automatic threshold classification or significance claim.",
        },
        "limits": [
            "Only the predeclared eight deployment seeds enter inference; pilot and auxiliary seeds remain separately listed.",
            "Histories are averaged within deployment seed and pair kind before bootstrap resampling.",
            "The study tests one known-group retention decision with continued local PPO, not unknown-context discovery, repeated adaptive allocation, or globally optimal storage.",
            "A realized selector result is the outcome of the corresponding complete PPO branch; any across-action maximum is selection-biased and is not an oracle bound.",
            "No Spearman or rank-correlation statistic is used as an acceptance criterion.",
        ],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _summary_line(summary: Mapping[str, Any], digits: int = 4) -> str:
    ci = summary.get("percentile_ci95")
    ci_text = "NA" if ci is None else f"[{_fmt(ci[0], digits)}, {_fmt(ci[1], digits)}]"
    seeds = ", ".join(f"{row['seed']}: {_fmt(row['value'], digits)}" for row in summary.get("per_seed", []))
    return f"n={summary.get('n_seeds', 0)}; mean={_fmt(summary.get('mean'), digits)}; range=[{_fmt(summary.get('min'), digits)}, {_fmt(summary.get('max'), digits)}]; 95% percentile CI={ci_text}; seeds: {seeds}"


def build_report_markdown(analysis: Mapping[str, Any]) -> str:
    cfg = analysis["decision_thresholds"]
    cohort = analysis["cohort"]
    lines = [
        "# Single shared-slot storage decision: descriptive report", "",
        f"Completed {cohort['complete_main_case_records']} of {cohort['expected_case_count']} expected main cases and "
        f"{cohort['observed_branch_summaries']} of {cohort['expected_branch_count']} expected action-by-future summaries.", "",
        "All estimates use deployment seed as the unit. The two histories are averaged within seed and pair kind before the 10,000-resample percentile bootstrap (seed 20261018). Intervals are descriptive; this is not a significance or power claim.", "",
        "## Mechanism interaction I", "",
        "I = (replace − keep under new-major future) − (replace − keep under old-major future). Each seed value first averages its two histories.", "",
        "| Pair kind | Metric | Per-seed values, mean, range, and 95% CI | Positive seeds / direction check |", "|---|---|---|---|",
    ]
    for pair_kind in PAIR_KINDS:
        for metric in METRICS:
            check = analysis["predeclared_mechanism_direction_check"][pair_kind]
            direction = f"{check['positive_seed_count']}/{check['expected_seed_count']}; {check['status']}" if metric == "reward_per_step" else "—"
            lines.append(f"| {pair_kind} | {metric} | {_summary_line(analysis['interaction_I'][pair_kind][metric], 5)} | {direction} |")
    practical = float(cfg["practical_reward_per_step_reference"])
    lines.extend([
        "", f"The predeclared practical reference is {practical:.3f} reward per step. Report the interval alongside it; the threshold does not automatically label a result as strong or conclusive.",
        "", "## Preference crossover diagnostic", "",
        "I > 0 alone can occur when one action is preferred under both future regimes. The table separately counts per-case reversals where D(old-major future) < 0 and D(new-major future) > 0; Dold and Dnew summaries average both histories within seed.",
        "", "| Pair kind | Metric | Per-case crossovers / complete cases | Dold per-seed | Dnew per-seed |", "|---|---|---|---|---|",
    ])
    for pair_kind in PAIR_KINDS:
        for metric in METRICS:
            crossover = analysis["preference_crossover"][pair_kind][metric]
            count = f"{crossover['per_case_crossover_count']}/{crossover['per_case_count_with_complete_D']} (expected {crossover['expected_per_case_count']})"
            d_old = _summary_line(crossover["per_seed_D_old_future_summary"], 5)
            d_new = _summary_line(crossover["per_seed_D_new_future_summary"], 5)
            lines.append(f"| {pair_kind} | {metric} | {count} | {d_old} | {d_new} |")
    lines.extend(["", "## Primary continuation: history matches future", ""])
    lines.extend(_condition_markdown(analysis["continuation_conditions"]["primary"]))
    lines.extend(["", "## Reversal stress: history differs from future", ""])
    lines.extend(_condition_markdown(analysis["continuation_conditions"]["stress"]))
    checks = analysis["static_checks"]
    branch = checks["branch_completeness"]
    selector = checks["selector_prefix_only"]
    prefix_hashes = checks["prefix_restore_hashes"]
    prefixes = checks["prefix_completeness"]
    lines.extend([
        "", "## Integrity and completeness", "",
        f"Branch completeness: {branch['observed_branch_summary_count']} / {branch['expected_branch_count']} summaries; all expected cases complete = {branch['all_expected_cases_complete']}.",
        f"Prefix summaries: {prefixes['observed_prefix_summary_count']} / {prefixes['expected_prefix_count']}.",
        f"Selector prefix audit: {selector['prefix_only_pass_count']} / {selector['decision_records_checked']} decision records passed the no-future-key, bounded-history, frequency/recency recomputation, and frozen-surrogate action checks.",
        f"Prefix/restore hashes: {prefix_hashes['matched_records']} / {prefix_hashes['hashes_recorded']} records with recorded hashes matched, out of {prefix_hashes['decision_records_checked']} decision records.",
        f"Recorded runner or parse failures: {len(checks['runner_and_parse_failures'])}.",
        "", "### Costs", "",
        f"Expected main interactions from protocol: {analysis['costs']['expected_main']['steps_total']:,} "
        f"({analysis['costs']['expected_main']['case_count']} prefixes and {analysis['costs']['expected_main']['branch_count']} branches).",
        f"Observed prefix interactions: {analysis['costs']['observed_leaf_summaries']['prefix']['steps']:,} across "
        f"{analysis['costs']['observed_leaf_summaries']['prefix']['summaries']} prefixes; observed branch interactions: "
        f"{analysis['costs']['observed_leaf_summaries']['branches']['steps']:,} across "
        f"{analysis['costs']['observed_leaf_summaries']['branches']['summaries']} branches; total leaf-summary interactions: "
        f"{analysis['costs']['observed_leaf_summaries']['steps_total']:,}; "
        f"CPU seconds {analysis['costs']['observed_leaf_summaries']['cpu_seconds_total']:.2f}; "
        f"summed wall seconds {analysis['costs']['observed_leaf_summaries']['wall_seconds_sum']:.2f}.",
        f"Observed rollouts/minibatch updates: prefixes "
        f"{analysis['costs']['observed_leaf_summaries']['prefix']['rollouts']}/"
        f"{analysis['costs']['observed_leaf_summaries']['prefix']['minibatch_updates']}; branches "
        f"{analysis['costs']['observed_leaf_summaries']['branches']['rollouts']}/"
        f"{analysis['costs']['observed_leaf_summaries']['branches']['minibatch_updates']}.",
        f"Physical byte ranges: actor {analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['actor_bytes']['min']}–"
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['actor_bytes']['max']}; checkpoint "
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['checkpoint_bytes']['min']}–"
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['checkpoint_bytes']['max']}; W0 "
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['w0_bytes']['min']}–"
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['w0_bytes']['max']}; sidecar "
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['sidecar_bytes']['min']}–"
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['sidecar_bytes']['max']}; full package "
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['package_bytes']['min']}–"
        f"{analysis['costs']['observed_leaf_summaries']['physical_bytes_ranges']['package_bytes']['max']} "
        f"under cap {analysis['costs']['observed_leaf_summaries']['package_cap_bytes']}.",
        "Per-bout and per-group returns, along with branch cost and capacity diagnostics, are retained in analysis.json.",
        "", "### Scope", "",
    ])
    lines.extend(f"- {item}" for item in analysis["limits"])
    lines.extend([
        "", "Plotting uses Matplotlib (Hunter, 2007; see `references/references.bib`, key `Hunter2007Matplotlib`). The procedural visualization workflow draws on Kassis, Agarwal, He, Patel, and Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065, [current arXiv record](https://arxiv.org/abs/2609.00065).",
        "", "No claim of statistical significance, state-of-the-art performance, global storage optimality, or a new core algorithm follows from this report.", "",
    ])
    if branch["incomplete_cases"]:
        lines.extend(["## Incomplete cases", ""])
        for row in branch["incomplete_cases"]:
            lines.append(f"- `{row.get('case_id', row.get('case_key'))}`: missing {', '.join(row.get('missing_branches', [])) or 'decision record'}.")
        lines.append("")
    if checks["runner_and_parse_failures"]:
        lines.extend(["## Recorded failures", ""])
        for row in checks["runner_and_parse_failures"]:
            lines.append(f"- `{json.dumps(row, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines)


def _condition_markdown(condition: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for pair_kind in PAIR_KINDS:
        block = condition[pair_kind]
        lines.extend([f"### {pair_kind}", "", "Action and selector returns are listed as cumulative reward and per-step reward. Selector contrasts are paired at each prefix and averaged over both histories per seed.", "", "| Method | Cumulative reward | Reward per step |", "|---|---|---|"])
        for method in METHODS:
            total = block["methods"][method]["reward_total"]
            per_step = block["methods"][method]["reward_per_step"]
            lines.append(f"| {method} | {_summary_line(total, 3)} | {_summary_line(per_step, 5)} |")
        lines.extend(["", "Selector minus baseline paired contrasts:", "", "| Contrast | Cumulative reward difference | Per-step reward difference |", "|---|---|---|"])
        for name, metrics in block["selector_paired_contrasts"].items():
            lines.append(f"| {name} | {_summary_line(metrics['reward_total'], 3)} | {_summary_line(metrics['reward_per_step'], 5)} |")
        lines.append("")
    return lines


def _plot_forest(ax: Any, condition: Mapping[str, Any], title: str) -> None:
    contrasts = [
        "frequency_minus_keep", "frequency_minus_replace", "frequency_minus_mean", "frequency_minus_none",
        "recency_minus_frequency", "surrogate_gain_minus_frequency", "surrogate_lower_minus_frequency",
    ]
    labels = [name.replace("_minus_", " − ") for name in contrasts]
    y = np.arange(len(contrasts), dtype=float)
    offsets = {"dynamics": -0.12, "reward_conflict": 0.12}
    for pair_kind in PAIR_KINDS:
        for index, name in enumerate(contrasts):
            summary = condition[pair_kind]["selector_paired_contrasts"][name]["reward_per_step"]
            if summary["mean"] is None:
                continue
            ci = summary["percentile_ci95"]
            xerr = None if ci is None else [[summary["mean"] - ci[0]], [ci[1] - summary["mean"]]]
            ax.errorbar(
                summary["mean"], y[index] + offsets[pair_kind], xerr=xerr,
                fmt="o", markersize=4, capsize=2, color=COLORS[pair_kind],
                label=pair_kind if index == 0 else None,
            )
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_title(title)
    ax.set_xlabel("Paired difference in reward per step vs baseline")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower right")


def render_figure(analysis: Mapping[str, Any], png_path: Path, pdf_path: Path) -> None:
    figure = plt.figure(figsize=(13.5, 10.2), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(1.05, 1.45, 1.45))
    interaction_ax = figure.add_subplot(grid[0, 0])
    y_positions = {"dynamics": 0.0, "reward_conflict": 1.0}
    for pair_kind in PAIR_KINDS:
        summary = analysis["interaction_I"][pair_kind]["reward_per_step"]
        values = summary["per_seed"]
        y = y_positions[pair_kind]
        x = [row["value"] for row in values]
        interaction_ax.scatter(x, [y] * len(x), color=COLORS[pair_kind], s=35, alpha=0.9, zorder=3, label=pair_kind)
        if summary["mean"] is not None:
            lo, hi = summary["percentile_ci95"] or (summary["mean"], summary["mean"])
            interaction_ax.errorbar(summary["mean"], y, xerr=[[summary["mean"] - lo], [hi - summary["mean"]]],
                                    fmt="D", color=COLORS[pair_kind], capsize=4, markersize=5, zorder=4)
    interaction_ax.axvline(0, color="#444444", linewidth=0.8)
    interaction_ax.axvline(float(analysis["decision_thresholds"]["practical_reward_per_step_reference"]), color="#777777", linewidth=0.8, linestyle="--")
    interaction_ax.set_yticks([0.0, 1.0], ["dynamics", "reward_conflict"])
    interaction_ax.set_xlabel("I in reward per step; dots are deployment seeds, diamonds are means with 95% seed-cluster CI")
    interaction_ax.set_title("A. Replace-versus-keep effect varies with future recurrence")
    interaction_ax.grid(axis="x", alpha=0.25)
    interaction_ax.legend(frameon=False, ncol=2, loc="lower right")
    _plot_forest(figure.add_subplot(grid[1, 0]), analysis["continuation_conditions"]["primary"],
                 "B. Primary continuation: paired selector contrasts with fixed actions and frequency")
    _plot_forest(figure.add_subplot(grid[2, 0]), analysis["continuation_conditions"]["stress"],
                 "C. Reversal stress: paired selector contrasts with fixed actions and frequency")
    figure.suptitle("Single shared-slot storage decision — seed-level descriptive estimates", fontsize=14)
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to write into nonempty report directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def generate_report(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _prepare_output(output_dir)
    summary_path = input_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"input directory has no summary.json: {input_dir}")
    root_summary = _read_json(summary_path)
    analysis = analyze_summary(input_dir, root_summary)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(build_report_markdown(analysis), encoding="utf-8")
    render_figure(analysis, output_dir / "figure.png", output_dir / "figure.pdf")
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="runner cohort directory containing summary.json and cases/")
    parser.add_argument("--output", required=True, type=Path, help="new or empty report directory")
    args = parser.parse_args(argv)
    try:
        analysis = generate_report(args.input, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    cohort = analysis["cohort"]
    print(f"Wrote {args.output}: {cohort['complete_main_case_records']}/{cohort['expected_case_count']} cases complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

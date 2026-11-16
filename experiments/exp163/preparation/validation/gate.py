"""Pure scalar gate for E163; scientific mapping is frozen in gate_mapping.json.

This module accepts only an internal validation record containing the five
already-produced holdout scalars and explicit C/D fallback booleans.  The
schema is bound by runner.py to the frozen h036_stage0_score_v1 outputs; it does
not define scientific fitting, codec, KL, or prefix behavior.
It performs scalar arithmetic only.
"""

from __future__ import annotations

import json
import math
import numbers
import sys
from collections.abc import Mapping
from typing import Any


SCHEMA_ID = "E163_INTERNAL_SCALAR_GATE_V1"
SCHEMA_STATUS = "FROZEN_SCALAR_MAPPING_H036_DELIVERY_V1"
CANONICAL_SEEDS = tuple(range(26201, 26214)) + (26215, 26216, 26218)
REQUIRED_PASS_COUNT = 12
GUARD_EPSILON = 1e-10
SCALAR_FIELDS = ("K_old_C", "K_old_D", "K_new_W0", "K_new_C", "K_new_D")
FALLBACK_KEYS = ("C", "D")
PROVISIONAL_ADAPTER_CONTRACT = {
    "container": "list_or_tuple_of_row_objects; order is ignored",
    "row_count": 16,
    "seed_field": "seed",
    "canonical_seeds": list(CANONICAL_SEEDS),
    "scalar_fields": list(SCALAR_FIELDS),
    "scalar_semantics": (
        "already-produced encoded fresh-local=0 holdout scalar values; this "
        "module does not calculate or verify them"
    ),
    "scalar_constraints": "real, finite, nonnegative",
    "fallback_flags": {
        "field": "fallback_flags",
        "shape": {"C": "boolean", "D": "boolean"},
        "meaning": "whether each candidate used a legal fallback; annotation only",
    },
    "whole_batch_policy": (
        "exactly one valid row for every canonical seed; any invalid or "
        "incomplete input suppresses all scoring"
    ),
    "mapping_status": SCHEMA_STATUS,
}


def _empty_report(status: str, errors: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_ID,
        "schema_status": SCHEMA_STATUS,
        "provisional_adapter_contract": PROVISIONAL_ADAPTER_CONTRACT,
        "metric_scope": (
            "Supplied encoded fresh-local=0 holdout behavior-KL scalars screen "
            "behavior compatibility only; reward is not measured or inferred."
        ),
        "status": status,
        "expected_seeds": list(CANONICAL_SEEDS),
        "gate_denominator": len(CANONICAL_SEEDS),
        "required_pass_count": REQUIRED_PASS_COUNT,
        "scored_count": 0,
        "pass_count": None,
        "evaluations": [],
        "validation_errors": errors,
        "fallback_policy": (
            "C/D fallback flags are annotations; every canonical seed remains "
            "in the denominator and is scored by the same thresholds."
        ),
        "statistical_inference": {
            "p_values": "not computed",
            "confidence_intervals": "not computed",
            "basis": "deterministic per-seed thresholds only",
        },
    }


def evaluate_batch(records: object) -> dict[str, Any]:
    """Validate and score one complete batch of 16 provisional scalar records.

    Each record has ``seed``, the five names in ``SCALAR_FIELDS``, and
    ``fallback_flags`` exactly shaped as ``{"C": bool, "D": bool}``.
    Missing rows or fields yield INCOMPLETE. Duplicate/unexpected seeds,
    malformed values, negative/non-finite scalars, and invalid flags yield
    INVALID. Either result invalidates the whole batch; partial rows are never
    scored. For a complete valid batch, all 16 rows count, including rows with
    a true fallback flag.
    """
    invalid: list[Any] = []
    incomplete: list[Any] = []

    if records is None:
        incomplete.append({"problem": "record_array_missing"})
        return _empty_report("INCOMPLETE", {"invalid": invalid, "incomplete": incomplete})
    if not isinstance(records, (list, tuple)):
        invalid.append({"problem": "record_array_must_be_list_or_tuple"})
        return _empty_report("INVALID", {"invalid": invalid, "incomplete": incomplete})

    expected = set(CANONICAL_SEEDS)
    seen: set[int] = set()
    parsed: dict[int, dict[str, Any]] = {}

    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            invalid.append({"row_index": row_index, "problem": "record_must_be_object"})
            continue

        if "seed" not in record:
            incomplete.append({"row_index": row_index, "problem": "seed_missing"})
            continue
        seed = record["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            invalid.append({"row_index": row_index, "problem": "seed_must_be_integer", "value": repr(seed)})
            continue
        if seed not in expected:
            invalid.append({"row_index": row_index, "problem": "unexpected_seed", "seed": seed})
            continue
        if seed in seen:
            invalid.append({"row_index": row_index, "problem": "duplicate_seed", "seed": seed})
            continue
        seen.add(seed)

        row_values: dict[str, Any] = {}
        row_has_invalid = False
        missing_fields = [field for field in SCALAR_FIELDS if field not in record]
        if missing_fields:
            incomplete.append({"seed": seed, "problem": "scalar_fields_missing", "fields": missing_fields})

        for field in SCALAR_FIELDS:
            if field not in record:
                continue
            raw = record[field]
            if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
                invalid.append({"seed": seed, "field": field, "problem": "scalar_must_be_real_number", "value": repr(raw)})
                row_has_invalid = True
                continue
            try:
                value = float(raw)
            except (OverflowError, TypeError, ValueError):
                invalid.append({"seed": seed, "field": field, "problem": "scalar_not_representable_as_float"})
                row_has_invalid = True
                continue
            if not math.isfinite(value):
                invalid.append({"seed": seed, "field": field, "problem": "scalar_must_be_finite", "value": repr(raw)})
                row_has_invalid = True
                continue
            if value < 0.0:
                invalid.append({"seed": seed, "field": field, "problem": "scalar_must_be_nonnegative", "value": value})
                row_has_invalid = True
                continue
            row_values[field] = value

        if "fallback_flags" not in record:
            incomplete.append({"seed": seed, "problem": "fallback_flags_missing"})
            row_has_invalid = True
        else:
            flags = record["fallback_flags"]
            if not isinstance(flags, Mapping):
                invalid.append({"seed": seed, "problem": "fallback_flags_must_be_object"})
                row_has_invalid = True
            else:
                flag_keys = set(flags)
                missing_flags = set(FALLBACK_KEYS) - flag_keys
                extra_flags = flag_keys - set(FALLBACK_KEYS)
                if missing_flags:
                    incomplete.append({"seed": seed, "problem": "fallback_flags_incomplete", "missing": sorted(missing_flags)})
                    row_has_invalid = True
                if extra_flags:
                    invalid.append({"seed": seed, "problem": "fallback_flags_have_unknown_keys", "keys": sorted(map(str, extra_flags))})
                    row_has_invalid = True
                for key in FALLBACK_KEYS:
                    if key in flags and not isinstance(flags[key], bool):
                        invalid.append({"seed": seed, "problem": "fallback_flag_must_be_boolean", "candidate": key})
                        row_has_invalid = True
                if not missing_flags and not extra_flags and all(isinstance(flags[key], bool) for key in FALLBACK_KEYS):
                    row_values["fallback_flags"] = {key: flags[key] for key in FALLBACK_KEYS}

        if not row_has_invalid and len(row_values) == len(SCALAR_FIELDS) + 1:
            parsed[seed] = row_values

    missing_seeds = sorted(expected - seen)
    if missing_seeds:
        incomplete.append({"problem": "canonical_seeds_missing", "seeds": missing_seeds})

    if invalid:
        return _empty_report("INVALID", {"invalid": invalid, "incomplete": incomplete})
    if incomplete:
        return _empty_report("INCOMPLETE", {"invalid": invalid, "incomplete": incomplete})

    evaluations: list[dict[str, Any]] = []
    for seed in CANONICAL_SEEDS:
        row = parsed[seed]
        old_c = row["K_old_C"]
        old_d = row["K_old_D"]
        new_w0 = row["K_new_W0"]
        new_c = row["K_new_C"]
        new_d = row["K_new_D"]
        gain_c = new_w0 - new_c
        gain_d = new_w0 - new_d
        conditions = {
            "K_old_C_gt_1e-10": old_c > GUARD_EPSILON,
            "G_new_C_gt_1e-10": gain_c > GUARD_EPSILON,
            "K_old_D_le_0.75_K_old_C": old_d <= 0.75 * old_c,
            "G_new_D_ge_0.5_G_new_C": gain_d >= 0.5 * gain_c,
        }
        failed_conditions = [name for name, passed in conditions.items() if not passed]
        evaluations.append(
            {
                "seed": seed,
                **{field: row[field] for field in SCALAR_FIELDS},
                "G_new_C": gain_c,
                "G_new_D": gain_d,
                "old_KL_reduction_C_minus_D": old_c - old_d,
                "fallback_flags": dict(row["fallback_flags"]),
                "conditions": conditions,
                "passed": not failed_conditions,
                "failed_conditions": failed_conditions,
            }
        )

    pass_count = sum(1 for item in evaluations if item["passed"])
    fallback_seed_ids = {
        candidate: [item["seed"] for item in evaluations if item["fallback_flags"][candidate]]
        for candidate in FALLBACK_KEYS
    }
    return {
        "schema": SCHEMA_ID,
        "schema_status": SCHEMA_STATUS,
        "provisional_adapter_contract": PROVISIONAL_ADAPTER_CONTRACT,
        "metric_scope": (
            "Supplied encoded fresh-local=0 holdout behavior-KL scalars screen "
            "behavior compatibility only; reward is not measured or inferred."
        ),
        "status": "MECHANISM_SCREEN_PASSED_EXPLORATORY" if pass_count >= REQUIRED_PASS_COUNT else "NO_ADVANCEMENT",
        "expected_seeds": list(CANONICAL_SEEDS),
        "gate_denominator": len(CANONICAL_SEEDS),
        "required_pass_count": REQUIRED_PASS_COUNT,
        "scored_count": len(evaluations),
        "pass_count": pass_count,
        "evaluations": evaluations,
        "validation_errors": {"invalid": [], "incomplete": []},
        "fallback_policy": (
            "C/D fallback flags are annotations; every canonical seed remains "
            "in the denominator and is scored by the same thresholds."
        ),
        "fallback_seed_ids": fallback_seed_ids,
        "denominator_guard_failure_seed_ids": {
            "K_old_C_gt_1e-10": [
                item["seed"] for item in evaluations if not item["conditions"]["K_old_C_gt_1e-10"]
            ],
            "G_new_C_gt_1e-10": [
                item["seed"] for item in evaluations if not item["conditions"]["G_new_C_gt_1e-10"]
            ],
        },
        "statistical_inference": {
            "p_values": "not computed",
            "confidence_intervals": "not computed",
            "basis": "deterministic per-seed thresholds only",
        },
    }


def _synthetic_rows() -> list[dict[str, Any]]:
    """Build synthetic scalar fixtures; these contain no candidate outputs."""
    return [
        {
            "seed": seed,
            "K_old_C": 4.0,
            "K_old_D": 3.0,
            "K_new_W0": 10.0,
            "K_new_C": 6.0,
            "K_new_D": 8.0,
            "fallback_flags": {"C": False, "D": False},
        }
        for seed in CANONICAL_SEEDS
    ]


def run_synthetic_checks() -> dict[str, Any]:
    """Run the bounded arithmetic and whole-batch validation checks."""
    checks: list[dict[str, Any]] = []

    def record_check(name: str, expected: Any, actual: Any) -> None:
        result = "PASS" if expected == actual else "FAIL"
        checks.append({"name": name, "expected": expected, "actual": actual, "result": result})
        if result != "PASS":
            raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")

    boundary = evaluate_batch(_synthetic_rows())
    boundary_row = boundary["evaluations"][0]
    record_check(
        "inclusive_25_percent_old_reduction_and_50_percent_new_gain",
        {"status": "MECHANISM_SCREEN_PASSED_EXPLORATORY", "passed": True, "old_condition": True, "new_condition": True},
        {
            "status": boundary["status"],
            "passed": boundary_row["passed"],
            "old_condition": boundary_row["conditions"]["K_old_D_le_0.75_K_old_C"],
            "new_condition": boundary_row["conditions"]["G_new_D_ge_0.5_G_new_C"],
        },
    )

    old_guard_rows = _synthetic_rows()
    old_guard_rows[0]["K_old_C"] = GUARD_EPSILON
    old_guard_rows[0]["K_old_D"] = 0.0
    old_guard = evaluate_batch(old_guard_rows)
    old_guard_row = old_guard["evaluations"][0]
    record_check(
        "old_denominator_guard_is_strict",
        {"condition": False, "failed": True, "pass_count": 15, "guard_failure_seeds": [26201]},
        {
            "condition": old_guard_row["conditions"]["K_old_C_gt_1e-10"],
            "failed": not old_guard_row["passed"],
            "pass_count": old_guard["pass_count"],
            "guard_failure_seeds": old_guard["denominator_guard_failure_seed_ids"]["K_old_C_gt_1e-10"],
        },
    )

    gain_guard_rows = _synthetic_rows()
    gain_guard_rows[0]["K_new_W0"] = 2.0 * GUARD_EPSILON
    gain_guard_rows[0]["K_new_C"] = GUARD_EPSILON
    gain_guard_rows[0]["K_new_D"] = 0.0
    gain_guard = evaluate_batch(gain_guard_rows)
    gain_guard_row = gain_guard["evaluations"][0]
    record_check(
        "new_gain_denominator_guard_is_strict",
        {"gain": GUARD_EPSILON, "condition": False, "failed": True, "pass_count": 15, "guard_failure_seeds": [26201]},
        {
            "gain": gain_guard_row["G_new_C"],
            "condition": gain_guard_row["conditions"]["G_new_C_gt_1e-10"],
            "failed": not gain_guard_row["passed"],
            "pass_count": gain_guard["pass_count"],
            "guard_failure_seeds": gain_guard["denominator_guard_failure_seed_ids"]["G_new_C_gt_1e-10"],
        },
    )

    twelve_rows = _synthetic_rows()
    eleven_rows = _synthetic_rows()
    for row in twelve_rows[12:]:
        row["K_old_D"] = 3.01
    for row in eleven_rows[11:]:
        row["K_old_D"] = 3.01
    twelve = evaluate_batch(twelve_rows)
    eleven = evaluate_batch(eleven_rows)
    record_check(
        "12_of_16_threshold_is_inclusive_at_12",
        {"at_12": ("MECHANISM_SCREEN_PASSED_EXPLORATORY", 12), "at_11": ("NO_ADVANCEMENT", 11)},
        {"at_12": (twelve["status"], twelve["pass_count"]), "at_11": (eleven["status"], eleven["pass_count"])},
    )

    fallback_rows = _synthetic_rows()
    for row in fallback_rows[12:]:
        row["K_old_D"] = 3.01
    fallback_rows[0]["fallback_flags"] = {"C": True, "D": False}
    fallback_rows[1]["fallback_flags"] = {"C": False, "D": True}
    fallback_rows[15]["fallback_flags"] = {"C": True, "D": True}
    fallback = evaluate_batch(fallback_rows)
    record_check(
        "fallback_rows_are_carried_and_scored_without_filtering",
        {"scored_count": 16, "pass_count": 12, "C_fallback_seeds": [26201, 26218], "D_fallback_seeds": [26202, 26218]},
        {
            "scored_count": fallback["scored_count"],
            "pass_count": fallback["pass_count"],
            "C_fallback_seeds": fallback["fallback_seed_ids"]["C"],
            "D_fallback_seeds": fallback["fallback_seed_ids"]["D"],
        },
    )

    missing_row = evaluate_batch(_synthetic_rows()[:-1])
    record_check(
        "missing_seed_invalidates_whole_batch_as_incomplete",
        {"status": "INCOMPLETE", "scored_count": 0, "pass_count": None},
        {"status": missing_row["status"], "scored_count": missing_row["scored_count"], "pass_count": missing_row["pass_count"]},
    )

    missing_field_rows = _synthetic_rows()
    missing_field_rows[0].pop("K_old_C")
    missing_field = evaluate_batch(missing_field_rows)
    record_check(
        "missing_scalar_invalidates_whole_batch_as_incomplete",
        {"status": "INCOMPLETE", "scored_count": 0},
        {"status": missing_field["status"], "scored_count": missing_field["scored_count"]},
    )

    duplicate_rows = _synthetic_rows()
    duplicate_rows.append(dict(duplicate_rows[0]))
    duplicate = evaluate_batch(duplicate_rows)
    record_check(
        "duplicate_seed_invalidates_whole_batch",
        {"status": "INVALID", "scored_count": 0},
        {"status": duplicate["status"], "scored_count": duplicate["scored_count"]},
    )

    nonfinite_rows = _synthetic_rows()
    nonfinite_rows[0]["K_old_C"] = math.nan
    nonfinite = evaluate_batch(nonfinite_rows)
    record_check(
        "nonfinite_scalar_invalidates_whole_batch",
        {"status": "INVALID", "scored_count": 0},
        {"status": nonfinite["status"], "scored_count": nonfinite["scored_count"]},
    )

    return {
        "suite": "E163 scalar gate synthetic arithmetic checks",
        "scope": "software preparation only; synthetic scalar fixtures only",
        "schema": SCHEMA_ID,
        "schema_status": SCHEMA_STATUS,
        "real_fits": 0,
        "environment_interactions": 0,
        "execution_authorized": False,
        "overall": "PASS",
        "check_count": len(checks),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["--self-check"]:
        print("Usage: python -B gate.py --self-check", file=sys.stderr)
        return 2
    print(json.dumps(run_synthetic_checks(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline audit and exploratory report for the retention-candidate run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SUMMARY_FORMAT = "retention_candidate_summary_v1"
REPORT_FORMAT = "retention_candidate_report_v1"
ARMS = ("keep", "replace_mean", "replace_behavioral", "merge_mean", "merge_behavioral")
MAIN_SEEDS = (25201, 25202, 25203, 25204, 25205, 25206)
PILOT_SEED = 25189
OLD_FAMILIES = ("A", "B")
E161_ARMS = ("keep", "replace_mean", "replace_behavioral")
E161_FUTURES = ("old_major", "new_major", "balanced")
E161_PRIMARY_FUTURES = ("old_major", "new_major")
E161_PRIMARY_CONTRASTS = (
    "replace_behavioral-minus-replace_mean",
    "replace_behavioral-minus-keep",
)
OUTCOME_CONTRASTS = (
    ("replace_behavioral-minus-replace_mean", "replace_behavioral", "replace_mean"),
    ("replace_behavioral-minus-keep", "replace_behavioral", "keep"),
    ("replace_mean-minus-keep", "replace_mean", "keep"),
    ("merge_behavioral-minus-merge_mean", "merge_behavioral", "merge_mean"),
    ("merge_behavioral-minus-keep", "merge_behavioral", "keep"),
    ("merge_mean-minus-keep", "merge_mean", "keep"),
)
INTERACTION_NAME = "candidate_by_operation_interaction"


def _is_e161(config: Mapping[str, Any]) -> bool:
    return "e161" in str(config.get("analysis_role", "")).casefold()


def _available_contrasts(arms: Sequence[str]) -> tuple[tuple[str, str, str], ...]:
    arm_set = set(arms)
    return tuple(row for row in OUTCOME_CONTRASTS if row[1] in arm_set and row[2] in arm_set)


def _number(value: Any, name: str, *, integer: bool = False, nullable: bool = False) -> float | int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if integer:
        if result < 0 or not result.is_integer():
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(result)
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in report")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return str(value)


def _canonical_hash(value: Any) -> str:
    body = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {where}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain an object")
    return value


def _load_summary(input_path: str | Path) -> tuple[dict[str, Any], Path, str]:
    source = Path(input_path)
    path = source / "summary.json" if source.is_dir() else source
    if path.name != "summary.json" or not path.is_file():
        raise FileNotFoundError(f"expected summary.json under {source}")
    summary = _read_json(path, "summary.json")
    return summary, path.resolve(), _file_hash(path)


def _load_config(run_root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    embedded = summary.get("config")
    if isinstance(embedded, Mapping):
        return dict(embedded)
    for key in ("frozen_config_path", "config_path"):
        relative = summary.get(key)
        if isinstance(relative, str) and relative.strip():
            path = Path(relative)
            path = path.resolve() if path.is_absolute() else (run_root / path).resolve()
            if path.is_file():
                value = _read_json(path, "frozen config")
                nested = value.get("config")
                return dict(nested) if isinstance(nested, Mapping) else value
    config_path = run_root / "config.json"
    if config_path.is_file():
        value = _read_json(config_path, "config.json")
        nested = value.get("config")
        return dict(nested) if isinstance(nested, Mapping) else value
    raise ValueError("summary must include config or point to a saved config.json")


def _active_config(config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    result = dict(config)
    if phase == "pilot" and isinstance(config.get("pilot"), Mapping):
        for key, value in config["pilot"].items():
            result[key] = value
    return result


def _case_prefix(case: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    prefix = case.get("prefix")
    if not isinstance(prefix, Mapping):
        raise ValueError(f"{where}.prefix must be an object")
    return prefix


def _old_family(case: Mapping[str, Any], where: str) -> str | None:
    mapping = case.get("mapping")
    if not isinstance(mapping, Mapping):
        return None
    families = mapping.get("families")
    if isinstance(families, Mapping):
        value = families.get("old")
    else:
        value = mapping.get("old_family")
    return str(value) if value in OLD_FAMILIES else None


def _design(summary: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, str | None]]:
    if summary.get("format") != SUMMARY_FORMAT:
        raise ValueError(f"summary.format must be {SUMMARY_FORMAT}")
    phase = summary.get("phase")
    if phase not in ("pilot", "main"):
        raise ValueError("summary.phase must be pilot or main")
    config = _mapping(summary.get("config"), "summary.config") if isinstance(summary.get("config"), Mapping) else None
    if config is None:
        raise ValueError("summary.config must be embedded for design validation")
    active = _active_config(config, str(phase))
    try:
        seeds = [int(value) for value in active["seeds"]]
        pairs = [str(value) for value in active["pair_kinds"]]
        states = [str(value) for value in active["states"]]
        futures = [str(value) for value in active["future_regimes"]]
        replicates = [int(value) for value in active["replicates"]]
        arms = [str(value) for value in active["arms"]]
        horizons = [int(value) for value in active["horizons"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"incomplete {phase} dimensions in summary.config") from exc
    is_e161 = _is_e161(config)
    if is_e161:
        if phase != "main":
            raise ValueError("E161 confirmatory analysis must use phase main")
        if arms != list(E161_ARMS):
            raise ValueError(f"E161 arms must equal the frozen order {E161_ARMS}")
        if pairs != ["dynamics"] or states != ["late"]:
            raise ValueError("E161 must use dynamics / late only")
        if len(seeds) != 16:
            raise ValueError("E161 requires sixteen configured independent seeds")
        if set(futures) != set(E161_FUTURES) or len(futures) != len(E161_FUTURES):
            raise ValueError("E161 futures must contain old_major, new_major, and balanced exactly once")
        if replicates != [0, 1]:
            raise ValueError("E161 requires paired replicas 0 and 1")
        report_config = _mapping(config.get("report"), "config.report")
        if report_config.get("primary_future_regimes") != list(E161_PRIMARY_FUTURES):
            raise ValueError("E161 primary_future_regimes must be old_major then new_major")
        if [str(value) for value in report_config.get("primary_contrasts", [])] != list(E161_PRIMARY_CONTRASTS):
            raise ValueError("E161 primary_contrasts must contain the two frozen behavioral contrasts")
        if horizons != [1536, 6144] or int(report_config.get("primary_horizon", -1)) != 6144:
            raise ValueError("E161 must retain the 1536-step window and use 6144 as primary horizon")
        if int(report_config.get("bootstrap_resamples", -1)) != 10000 or int(report_config.get("bootstrap_seed", -1)) != 2026102102:
            raise ValueError("E161 bootstrap sensitivity settings differ from the frozen 10000 / 2026102102 plan")
        threshold = _number(report_config.get("practical_reward_per_step"),
                            "config.report.practical_reward_per_step")
        if not math.isclose(float(threshold), 0.002, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("E161 practical_reward_per_step must equal 0.002")
    elif arms != list(ARMS):
        raise ValueError(f"arms must equal the frozen order {ARMS}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("active seeds must be nonempty and unique")
    if not pairs or not states or not futures or not replicates:
        raise ValueError("active design dimensions must be nonempty")
    if not is_e161 and states != ["early", "late"]:
        raise ValueError("states must be early then late")
    if horizons != sorted(set(horizons)) or not horizons or horizons[0] <= 0:
        raise ValueError("horizons must be increasing positive integers")
    if phase == "main" and not is_e161 and tuple(seeds) != MAIN_SEEDS:
        raise ValueError(f"main seeds must equal the frozen set {MAIN_SEEDS}")
    if phase == "main" and is_e161 and config.get("history_regime_by_seed") != "old_major_if_even_else_new_major":
        raise ValueError("E161 history regime must follow the frozen seed-parity assignment")
    if phase == "pilot" and (seeds != [PILOT_SEED] or pairs != ["dynamics"] or futures != ["old_major"] or replicates != [0]):
        raise ValueError("pilot dimensions differ from the frozen one-seed design")
    if not is_e161 and config.get("history_regime_by_seed") != "old_major_if_even_else_new_major":
        raise ValueError("history regime must follow the frozen seed-parity rule")
    if config.get("warmup_write_rule") != "replace_old_group_only_after_pair":
        raise ValueError("late warmup must use the frozen old-group-only write rule")

    expected_cases = {(seed, pair, state) for seed in seeds for pair in pairs for state in states}
    expected_branches = {
        (seed, pair, state, future, replicate, arm)
        for seed, pair, state in expected_cases
        for future in futures for replicate in replicates for arm in arms
    }
    raw_cases, raw_branches = summary.get("cases"), summary.get("branches")
    if not isinstance(raw_cases, list) or not isinstance(raw_branches, list):
        raise ValueError("summary.cases and summary.branches must be arrays")
    cases = [dict(_mapping(value, f"summary.cases[{index}]")) for index, value in enumerate(raw_cases)]
    branches = [dict(_mapping(value, f"summary.branches[{index}]")) for index, value in enumerate(raw_branches)]
    if summary.get("seeds") is not None and [int(value) for value in summary["seeds"]] != seeds:
        raise ValueError("summary.seeds differs from the active frozen seed list")
    if summary.get("errors"):
        raise ValueError("summary contains run errors")
    if summary.get("case_count") is not None and int(_number(summary.get("case_count"), "summary.case_count", integer=True)) != len(cases):
        raise ValueError("summary.case_count disagrees with cases")
    if summary.get("branch_count") is not None and int(_number(summary.get("branch_count"), "summary.branch_count", integer=True)) != len(branches):
        raise ValueError("summary.branch_count disagrees with branches")

    case_index: dict[tuple[int, str, str], dict[str, Any]] = {}
    cases_by_id: dict[str, dict[str, Any]] = {}
    families: dict[int, str | None] = {}
    for index, case in enumerate(cases):
        where = f"cases[{index}]"
        try:
            key = (int(_number(case.get("seed"), f"{where}.seed", integer=True)),
                   str(case.get("pair_kind")), str(case.get("state")))
        except ValueError as exc:
            raise ValueError(f"invalid identity at {where}") from exc
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or key in case_index or case_id in cases_by_id or key not in expected_cases:
            raise ValueError(f"duplicate, missing, or unexpected case identity at {where}")
        expected_history = "old_major" if key[0] % 2 == 0 else "new_major"
        if case.get("history_regime") != expected_history:
            raise ValueError(f"case {case_id} history regime violates seed parity")
        if case.get("errors"):
            raise ValueError(f"case {case_id} contains errors")
        prefix = _case_prefix(case, f"case {case_id}")
        if prefix.get("errors"):
            raise ValueError(f"case {case_id} prefix contains errors")
        family = _old_family(case, f"case {case_id}")
        if family is None:
            raise ValueError(f"case {case_id} has no valid old-family label")
        if key[0] in families and families[key[0]] != family:
            raise ValueError(f"seed {key[0]} family mapping varies between cases")
        families[key[0]] = family
        case_index[key] = case
        cases_by_id[case_id] = case
    if set(case_index) != expected_cases:
        raise ValueError(f"case grid mismatch; missing {sorted(expected_cases - set(case_index))[:4]}")
    if phase == "main":
        expected_stratum_size = 8 if is_e161 else 3
        expected_counts = {family: expected_stratum_size for family in OLD_FAMILIES}
        parity_count = expected_stratum_size
        if ({family: list(families.values()).count(family) for family in OLD_FAMILIES} != expected_counts
                or sum(seed % 2 == 0 for seed in seeds) != parity_count
                or sum(seed % 2 for seed in seeds) != parity_count):
            raise ValueError(f"main seed strata must contain {expected_stratum_size} seeds per old family and parity")

    branch_index: dict[tuple[int, str, str, str, int, str], dict[str, Any]] = {}
    branch_by_id: dict[tuple[str, str], tuple[int, str, str, str, int, str]] = {}
    schedules: dict[tuple[str, str, int], str] = {}
    for index, branch in enumerate(branches):
        where = f"branches[{index}]"
        try:
            key = (int(_number(branch.get("seed"), f"{where}.seed", integer=True)),
                   str(branch.get("pair_kind")), str(branch.get("state")),
                   str(branch.get("future_regime")),
                   int(_number(branch.get("replicate"), f"{where}.replicate", integer=True)),
                   str(branch.get("arm")))
        except ValueError as exc:
            raise ValueError(f"invalid identity at {where}") from exc
        branch_id, case_id = branch.get("branch_id"), branch.get("case_id")
        if not isinstance(branch_id, str) or not branch_id or (str(case_id), branch_id) in branch_by_id or key in branch_index:
            raise ValueError(f"duplicate or missing branch identity at {where}")
        if key not in expected_branches or case_id not in cases_by_id:
            raise ValueError(f"unexpected branch grid cell at {where}: {key}")
        linked = cases_by_id[str(case_id)]
        linked_key = (int(linked["seed"]), str(linked["pair_kind"]), str(linked["state"]))
        if linked_key != key[:3] or branch.get("history_regime") != linked.get("history_regime"):
            raise ValueError(f"branch {branch_id} case link disagrees with its identity")
        if branch.get("errors"):
            raise ValueError(f"branch {branch_id} contains errors")
        if branch.get("prefix_restore_verified") is not True or branch.get("prefix_state_hash") != branch.get("restored_state_hash"):
            raise ValueError(f"branch {branch_id} failed common-prefix restore verification")
        if branch.get("prefix_bouts_included_in_reward") is not False:
            raise ValueError(f"branch {branch_id} reward includes prefix bouts")
        if int(_number(branch.get("future_seed"), f"branch {branch_id}.future_seed", integer=True)) != int(config["future_seed_offset"]) + int(config["future_seed_multiplier"]) * key[0] + key[4]:
            raise ValueError(f"branch {branch_id} future seed differs from the frozen transform")
        schedule = branch.get("future_schedule")
        if not isinstance(schedule, list) or not schedule:
            raise ValueError(f"branch {branch_id} requires a future schedule")
        schedule_key = (str(case_id), key[3], key[4])
        schedule_hash = _canonical_hash(schedule)
        if schedule_key in schedules and schedules[schedule_key] != schedule_hash:
            raise ValueError(f"branch {branch_id} future schedule differs across arms")
        schedules[schedule_key] = schedule_hash
        fit_ref = _mapping(branch.get("candidate_fit_reference"), f"branch {branch_id}.candidate_fit_reference")
        prefix = _case_prefix(linked, f"case {case_id}")
        fit = _mapping(prefix.get("candidate_fit"), f"case {case_id}.prefix.candidate_fit")
        for field in ("mean_candidate_sha256", "behavioral_candidate_sha256"):
            if fit_ref.get(field) != fit.get(field):
                raise ValueError(f"branch {branch_id} references a different fitted candidate ({field})")
        branch_index[key] = branch
        branch_by_id[(str(case_id), branch_id)] = key
    if set(branch_index) != expected_branches:
        raise ValueError(f"branch grid mismatch; missing {sorted(expected_branches - set(branch_index))[:4]}")

    for case_id, case in cases_by_id.items():
        embedded = case.get("branches")
        if isinstance(embedded, list):
            embedded_index: dict[str, Mapping[str, Any]] = {}
            for item in embedded:
                item = _mapping(item, f"case {case_id}.branches[]")
                branch_id = item.get("branch_id")
                if not isinstance(branch_id, str) or branch_id in embedded_index:
                    raise ValueError(f"case {case_id} has duplicate embedded branch IDs")
                embedded_index[branch_id] = item
            expected_ids = {branch_id for (linked_case_id, branch_id) in branch_by_id if linked_case_id == case_id}
            if set(embedded_index) != expected_ids:
                raise ValueError(f"case {case_id} embedded branch grid is incomplete")
            for branch_id, item in embedded_index.items():
                flat = branches_by_pair_id(branch_by_id, branch_index, case_id, branch_id)
                if _canonical_hash(item) != _canonical_hash(flat):
                    raise ValueError(f"case {case_id} embedded branch differs from flat branch {branch_id}")
        prefix = _case_prefix(case, f"case {case_id}")
        if not isinstance(prefix.get("prefix_state_hash"), str) or not prefix.get("prefix_state_hash"):
            raise ValueError(f"case {case_id} is missing its prefix state hash")
        if len(prefix.get("warmup_pair_records", [])) != (8 if case.get("state") == "late" else 0):
            raise ValueError(f"case {case_id} warmup-pair record count disagrees with its state")
        if case.get("state") == "late":
            records = prefix["warmup_pair_records"]
            expected_groups = [1, 0] * 4
            if [int(_number(row.get("group"), f"case {case_id}.warmup.group", integer=True)) for row in records] != expected_groups:
                raise ValueError(f"case {case_id} warmup pair ordering is not new then old for four cycles")
            if [int(_number(row.get("cycle"), f"case {case_id}.warmup.cycle", integer=True)) for row in records] != [cycle for cycle in range(4) for _ in (0, 1)]:
                raise ValueError(f"case {case_id} warmup records have an invalid cycle order")
            for row in records:
                expected_write = int(row["group"]) == 0
                if row.get("write_applied") is not expected_write:
                    raise ValueError(f"case {case_id} warmup write rule was not followed")
                if row.get("write_rule") != "replace_old_group_only_after_pair":
                    raise ValueError(f"case {case_id} warmup record does not name the frozen write rule")
                binding = row.get("parent_binding_before_after")
                hashes = row.get("parent_sha256_before_after")
                if not isinstance(binding, list) or len(binding) != 2 or binding != [0, 0]:
                    raise ValueError(f"case {case_id} warmup changed or omitted the old parent binding")
                if not isinstance(hashes, list) or len(hashes) != 2 or any(not isinstance(value, str) or not value for value in hashes):
                    raise ValueError(f"case {case_id} warmup is missing before/after parent hashes")
                if not expected_write and hashes[0] != hashes[1]:
                    raise ValueError(f"case {case_id} local-only new-pair warmup changed the shared parent")
                if not expected_write and row.get("parent_coefficients_unchanged") is not True:
                    raise ValueError(f"case {case_id} local-only new-pair warmup did not preserve parent coefficients")
                if not expected_write and row.get("codec_max_coefficient_error") not in (None, 0, 0.0):
                    raise ValueError(f"case {case_id} new-pair local-only warmup unexpectedly encoded the shared actor")
        # Every flat branch must restore this one saved prefix.
        for (linked_case_id, branch_id), key in branch_by_id.items():
            if linked_case_id == case_id and branch_index[key].get("prefix_state_hash") != prefix["prefix_state_hash"]:
                raise ValueError(f"branch {branch_id} prefix hash differs from case {case_id}")

    active.update({"seeds": seeds, "pair_kinds": pairs, "states": states,
                   "future_regimes": futures, "replicates": replicates,
                   "arms": arms, "horizons": horizons})
    expected = _expected_counts(config, active)
    return ({"phase": str(phase), "config": dict(config), "active": active,
             "expected_counts": expected}, cases, branches, families)


def branches_by_pair_id(
    by_id: Mapping[tuple[str, str], tuple[int, str, str, str, int, str]],
    index: Mapping[tuple[int, str, str, str, int, str], dict[str, Any]],
    case_id: str,
    branch_id: str,
) -> Mapping[str, Any]:
    return index[by_id[(case_id, branch_id)]]


def _expected_counts(config: Mapping[str, Any], active: Mapping[str, Any]) -> dict[str, int]:
    case_count = len(active["seeds"]) * len(active["pair_kinds"]) * len(active["states"])
    branch_count = case_count * len(active["future_regimes"]) * len(active["replicates"]) * len(active["arms"])
    early_steps = 2 * int(config["bootstrap_steps"]) + 6 * int(config["bout_steps"]) + 2 * int(config["early_source_steps"])
    late_steps = early_steps + 4 * 2 * 2 * int(config["warmup_pair_steps"]) + 2 * (int(config["late_source_steps"]) - int(config["early_source_steps"]))
    prefix_steps = len(active["seeds"]) * len(active["pair_kinds"]) * sum(
        late_steps if str(state) == "late" else early_steps for state in active["states"]
    )
    branch_steps_per_branch = max(int(value) for value in active["horizons"])
    branch_steps = branch_count * branch_steps_per_branch
    return {"cases": case_count, "branches": branch_count, "early_prefix_steps": early_steps,
            "late_prefix_steps": late_steps, "prefix_steps": prefix_steps,
            "steps_per_branch": branch_steps_per_branch, "branch_steps": branch_steps,
            "total_steps": prefix_steps + branch_steps}


def _resolve_ledger(root: Path, meta: Any, where: str) -> tuple[Path, Mapping[str, Any]]:
    record = _mapping(meta, f"{where} metadata")
    name = record.get("path")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}.path is required")
    path = Path(name)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{where} path leaves the run directory") from exc
    if not path.is_file() or record.get("exists") is not True:
        raise ValueError(f"{where} must exist")
    actual_hash = _file_hash(path)
    if record.get("sha256") != actual_hash:
        raise ValueError(f"{where} SHA-256 mismatch")
    return path, record


def _read_csv_rows(path: Path, where: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"{where} is missing a CSV header")
            return list(reader.fieldnames), [dict(row) for row in reader]
    except (OSError, csv.Error) as exc:
        raise ValueError(f"cannot read {where}") from exc


def _audit_steps(
    root: Path,
    meta: Any,
    where: str,
    checkpoints: Sequence[int],
    *,
    expected_action: str | None = None,
    expected_future: str | None = None,
    require_bout: bool = False,
) -> tuple[dict[str, Any], dict[int, dict[str, float | int]], dict[int, dict[str, float | int]]]:
    path, record = _resolve_ledger(root, meta, where)
    row_count = 0
    marks: dict[int, dict[str, float | int]] = {}
    targets = set(int(value) for value in checkpoints)
    reward_total = 0.0
    previous_global: int | None = None
    previous_bout_step: tuple[int, int] | None = None
    closed_bouts: set[int] = set()
    bout_metrics: dict[int, dict[str, float | int]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"reward"}
            if require_bout:
                required |= {"bout", "bout_step"}
            if not reader.fieldnames or not required <= set(reader.fieldnames):
                raise ValueError(f"{where} requires columns {sorted(required)}")
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                reward = float(_number(row.get("reward"), f"{where}:{row_number}.reward"))
                reward_total += reward
                if expected_action is not None and row.get("action") != expected_action:
                    raise ValueError(f"{where}:{row_number} action differs from its branch")
                if expected_future is not None and row.get("future_regime") != expected_future:
                    raise ValueError(f"{where}:{row_number} future differs from its branch")
                raw_global = row.get("global_step")
                if raw_global not in (None, ""):
                    global_step = int(_number(raw_global, f"{where}:{row_number}.global_step", integer=True))
                    if previous_global is not None and global_step != previous_global + 1:
                        raise ValueError(f"{where} global_step is not contiguous")
                    previous_global = global_step
                raw_bout, raw_bout_step = row.get("bout"), row.get("bout_step")
                if raw_bout not in (None, "") and raw_bout_step not in (None, ""):
                    bout_id = int(_number(raw_bout, f"{where}:{row_number}.bout", integer=True))
                    bout_step = int(_number(raw_bout_step, f"{where}:{row_number}.bout_step", integer=True))
                    pair = (bout_id, bout_step)
                    if previous_bout_step is None or pair[0] != previous_bout_step[0]:
                        if pair[0] in closed_bouts or pair[1] != 1:
                            raise ValueError(f"{where} bout order repeats an earlier bout or does not start at step 1")
                        if previous_bout_step is not None:
                            closed_bouts.add(previous_bout_step[0])
                    metrics = bout_metrics.setdefault(bout_id, {
                        "steps": 0, "reward_full": 0.0, "return_first_256": 0.0, "first_256_rows": 0,
                    })
                    metrics["steps"] = int(metrics["steps"]) + 1
                    metrics["reward_full"] = float(metrics["reward_full"]) + reward
                    if int(metrics["first_256_rows"]) < 256:
                        metrics["return_first_256"] = float(metrics["return_first_256"]) + reward
                        metrics["first_256_rows"] = int(metrics["first_256_rows"]) + 1
                    if previous_bout_step is not None and (pair[0] < previous_bout_step[0] or
                                                            (pair[0] == previous_bout_step[0] and pair[1] != previous_bout_step[1] + 1)):
                        raise ValueError(f"{where} is not in contiguous bout order")
                    previous_bout_step = pair
                if row_count in targets:
                    marks[row_count] = {"steps": row_count, "reward_sum": reward_total,
                                        "reward_per_step": reward_total / row_count}
    except (OSError, csv.Error) as exc:
        raise ValueError(f"cannot read {where}") from exc
    if int(_number(record.get("rows"), f"{where}.rows", integer=True)) != row_count:
        raise ValueError(f"{where} row count differs from ledger manifest")
    declared_reward = float(_number(record.get("reward_sum"), f"{where}.reward_sum"))
    if not math.isclose(declared_reward, reward_total, rel_tol=1e-11, abs_tol=1e-8):
        raise ValueError(f"{where} reward total differs from ledger manifest")
    return ({"path": path.relative_to(root.resolve()).as_posix(), "sha256": _file_hash(path),
             "rows": row_count, "reward_sum": reward_total}, marks, bout_metrics)


def _audit_count_ledger(root: Path, meta: Any, where: str, *, rollouts: bool = False) -> dict[str, Any]:
    path, record = _resolve_ledger(root, meta, where)
    fields, rows = _read_csv_rows(path, where)
    if rollouts:
        if not {"rows", "updates"} <= set(fields):
            raise ValueError(f"{where} requires rows and updates columns")
        trained_rows = sum(int(_number(row.get("rows"), f"{where}.rows", integer=True)) for row in rows)
        updates = sum(int(_number(row.get("updates"), f"{where}.updates", integer=True)) for row in rows)
        if int(_number(record.get("rollout_count"), f"{where}.rollout_count", integer=True)) != len(rows):
            raise ValueError(f"{where} rollout count differs from ledger")
        if int(_number(record.get("trained_rows"), f"{where}.trained_rows", integer=True)) != trained_rows:
            raise ValueError(f"{where} trained row count differs from ledger")
        if int(_number(record.get("minibatch_updates"), f"{where}.minibatch_updates", integer=True)) != updates:
            raise ValueError(f"{where} minibatch update count differs from ledger")
        if int(_number(record.get("rows"), f"{where}.rows", integer=True)) != len(rows):
            raise ValueError(f"{where} rollout ledger row count differs from manifest")
        checkpoints: dict[int, dict[str, int]] = {}
        cumulative_rows = cumulative_updates = 0
        for row in rows:
            cumulative_rows += int(_number(row.get("rows"), f"{where}.rows", integer=True))
            cumulative_updates += int(_number(row.get("updates"), f"{where}.updates", integer=True))
            checkpoints[cumulative_rows] = {"rollouts_trained": len(checkpoints) + 1,
                                            "minibatch_updates": cumulative_updates}
        return {"path": path.relative_to(root.resolve()).as_posix(), "sha256": _file_hash(path),
                "rows": len(rows), "trained_rows": trained_rows, "minibatch_updates": updates,
                "checkpoints": checkpoints}
    if int(_number(record.get("rows"), f"{where}.rows", integer=True)) != len(rows):
        raise ValueError(f"{where} row count differs from ledger")
    return {"path": path.relative_to(root.resolve()).as_posix(), "sha256": _file_hash(path), "rows": len(rows)}


def _ledger_map(record: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    value = record.get("raw_ledgers")
    return _mapping(value, f"{where}.raw_ledgers")


def _audit_evidence_buffer(
    run_root: Path, case: Mapping[str, Any], prefix: Mapping[str, Any],
    fit: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    observer_relative = case.get("observer_prefix_path")
    if not isinstance(observer_relative, str) or not observer_relative:
        raise ValueError(f"case {case_id} has no observer_prefix_path")
    observer_path = (run_root / observer_relative).resolve()
    try:
        observer_path.relative_to(run_root.resolve())
    except ValueError as exc:
        raise ValueError(f"case {case_id} observer prefix path leaves the run directory") from exc
    observer = _read_json(observer_path, f"case {case_id} observer prefix")
    if observer.get("format") != "retention_candidate_observer_prefix_v1":
        raise ValueError(f"case {case_id} observer prefix format is invalid")
    if observer.get("accessible_to_deployed_selector") is not False or observer.get("included_in_deployment_capacity") is not False:
        raise ValueError(f"case {case_id} observer-only metadata is incorrectly marked deployable")
    evidence = _mapping(observer.get("evidence_buffer"), f"case {case_id}.observer.evidence_buffer")
    filename = evidence.get("path")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"case {case_id} evidence buffer path must be a sidecar filename")
    evidence_path = (observer_path.parent / filename).resolve()
    if not evidence_path.is_file():
        raise ValueError(f"case {case_id} evidence buffer file is missing")
    expected_bytes = int(_number(evidence.get("bytes"), f"case {case_id}.evidence.bytes", integer=True))
    expected_sha = evidence.get("sha256")
    if evidence_path.stat().st_size != expected_bytes or _file_hash(evidence_path) != expected_sha:
        raise ValueError(f"case {case_id} evidence buffer hash or byte count differs")
    prefix_evidence = _mapping(prefix.get("evidence_buffer"), f"case {case_id}.prefix.evidence_buffer")
    if (prefix_evidence.get("sha256") != expected_sha
            or int(_number(prefix_evidence.get("bytes"), f"case {case_id}.prefix.evidence_buffer.bytes", integer=True)) != expected_bytes):
        raise ValueError(f"case {case_id} prefix evidence metadata differs from observer sidecar")
    if int(_number(fit.get("source_buffer_bytes"), f"case {case_id}.fit.source_buffer_bytes", integer=True)) != expected_bytes:
        raise ValueError(f"case {case_id} fit buffer byte count differs from evidence sidecar")
    if fit.get("source_buffer_sha256") != expected_sha:
        raise ValueError(f"case {case_id} fit buffer hash differs from evidence sidecar")
    try:
        with np.load(evidence_path, allow_pickle=False) as archive:
            required = {"observations", "owner_indices", "source_slots", "generations", "bouts", "global_steps"}
            if set(archive.files) != required:
                raise ValueError(f"case {case_id} evidence arrays differ from the frozen sidecar schema")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load case {case_id} evidence sidecar") from exc
    observations = arrays["observations"]
    owners = arrays["owner_indices"]
    if observations.shape != (2 * int(config["evidence_rows_per_source"]), 13) or observations.dtype != np.float32:
        raise ValueError(f"case {case_id} evidence observations have the wrong shape or dtype")
    expected_dtypes = {"owner_indices": np.dtype(np.int8), "source_slots": np.dtype(np.int8),
                       "generations": np.dtype(np.int64), "bouts": np.dtype(np.int64),
                       "global_steps": np.dtype(np.int64)}
    for name, expected_dtype in expected_dtypes.items():
        if arrays[name].shape != (len(observations),) or arrays[name].dtype != expected_dtype:
            raise ValueError(f"case {case_id} evidence {name} has the wrong shape or dtype")
    if owners.shape != (len(observations),) or set(np.unique(owners).tolist()) != {0, 1}:
        raise ValueError(f"case {case_id} evidence owner indices are invalid")
    if not np.isfinite(observations).all():
        raise ValueError(f"case {case_id} evidence observations contain nonfinite values")
    counts = np.bincount(owners.astype(np.int64), minlength=2).astype(int).tolist()
    if counts != [int(config["evidence_rows_per_source"])] * 2:
        raise ValueError(f"case {case_id} evidence rows do not split evenly by source")
    bands = np.floor(np.clip(observations[:, 0], 0.0, 1.0) * 3.0).astype(np.int64).clip(0, 2)
    recomputed_bands = [np.bincount(bands[owners == owner], minlength=3).astype(int).tolist() for owner in (0, 1)]
    if recomputed_bands != fit.get("band_counts_by_owner"):
        raise ValueError(f"case {case_id} source band coverage differs from the preserved evidence rows")
    if int(_number(evidence.get("rows"), f"case {case_id}.evidence.rows", integer=True)) != len(observations):
        raise ValueError(f"case {case_id} evidence row count metadata differs")
    if evidence.get("rows_by_owner") != counts:
        raise ValueError(f"case {case_id} evidence owner counts metadata differs")
    sources = prefix.get("candidate_sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError(f"case {case_id} must identify two fitted source generations")
    for owner, source in enumerate(sources):
        source = _mapping(source, f"case {case_id}.candidate_sources[{owner}]")
        mask = owners == owner
        for array_name, source_name in (("source_slots", "slot"), ("generations", "generation"), ("bouts", "source_bout")):
            if array_name not in arrays or source_name not in source:
                raise ValueError(f"case {case_id} evidence provenance omits {array_name}/{source_name}")
            if np.any(arrays[array_name][mask] != int(source[source_name])):
                raise ValueError(f"case {case_id} evidence rows do not match owner {owner} source generation")
        expected_steps = int(config["early_source_steps"] if case.get("state") == "early" else config["late_source_steps"])
        if int(_number(source.get("source_time_fitted_steps"), f"case {case_id}.source_time_fitted_steps", integer=True)) != expected_steps:
            raise ValueError(f"case {case_id} owner {owner} source-time fit differs from its state")
        if int(_number(source.get("fitted_steps"), f"case {case_id}.fitted_steps", integer=True)) != expected_steps:
            raise ValueError(f"case {case_id} owner {owner} fitted steps differ from its state")
        if int(np.max(arrays["global_steps"][mask])) != int(source.get("latest_source_global_step")):
            raise ValueError(f"case {case_id} owner {owner} latest evidence step differs from source metadata")
    observer_fit = _mapping(observer.get("candidate_fit"), f"case {case_id}.observer.candidate_fit")
    if observer_fit.get("behavioral_candidate_sha256") != fit.get("behavioral_candidate_sha256"):
        raise ValueError(f"case {case_id} observer candidate hash differs from the fitted candidate")
    if observer.get("current_mean_candidate_sha256") != fit.get("mean_candidate_sha256") or observer.get("current_behavioral_candidate_sha256") != fit.get("behavioral_candidate_sha256"):
        raise ValueError(f"case {case_id} observer candidate hashes differ from the fitted candidates")
    if _canonical_hash(observer.get("candidate_sources")) != _canonical_hash(prefix.get("candidate_sources")):
        raise ValueError(f"case {case_id} observer candidate sources differ from the prefix source records")
    source_arrays = _mapping(observer.get("evidence_buffer"), f"case {case_id}.observer.evidence_buffer")
    for array_name in ("source_slots", "generations", "bouts", "global_steps"):
        if source_arrays.get(array_name) != arrays[array_name].astype(np.int64).tolist():
            raise ValueError(f"case {case_id} observer {array_name} differs from the binary evidence sidecar")
    if len(np.unique(arrays["global_steps"])) != len(observations):
        raise ValueError(f"case {case_id} evidence rows repeat a source global step")
    prefix_artifacts = _mapping(prefix.get("artifacts"), f"case {case_id}.prefix.artifacts")
    evidence_artifact = _mapping(prefix_artifacts.get("evidence"), f"case {case_id}.prefix.artifacts.evidence")
    prefix_artifact_path = _audit_evidence_artifact(
        run_root, evidence_artifact, f"case {case_id}.prefix.artifacts.evidence", expected_sha, expected_bytes,
    )
    if evidence_artifact.get("sha256") != expected_sha or int(evidence_artifact.get("bytes", -1)) != expected_bytes:
        raise ValueError(f"case {case_id} prefix package evidence differs from observer evidence")
    return {"observer_prefix_path": observer_relative, "evidence_path": str(evidence_path.relative_to(run_root.resolve()).as_posix()),
            "sha256": str(expected_sha), "bytes": expected_bytes, "rows": len(observations),
            "rows_by_owner": counts, "band_counts_by_owner": recomputed_bands,
            "source_generations_verified": True,
            "prefix_package_evidence_path": prefix_artifact_path.relative_to(run_root.resolve()).as_posix()}


def _audit_evidence_artifact(
    run_root: Path, artifact: Mapping[str, Any], where: str,
    expected_sha: str, expected_bytes: int,
) -> Path:
    relative = artifact.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{where}.path is required")
    path = Path(relative)
    path = path.resolve() if path.is_absolute() else (run_root / path).resolve()
    try:
        path.relative_to(run_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{where} path leaves the run directory") from exc
    declared_bytes = int(_number(artifact.get("bytes"), f"{where}.bytes", integer=True))
    if not path.is_file() or declared_bytes != expected_bytes or path.stat().st_size != expected_bytes:
        raise ValueError(f"{where} is missing or has a different sidecar size")
    if artifact.get("sha256") != expected_sha or _file_hash(path) != expected_sha:
        raise ValueError(f"{where} differs from the frozen source evidence sidecar")
    return path


def _check_capacity_evidence(
    capacity: Mapping[str, Any], expected_evidence_bytes: int, where: str,
) -> None:
    evidence_bytes = int(_number(capacity.get("evidence_bytes"), f"{where}.evidence_bytes", integer=True))
    deployment_bytes = int(_number(capacity.get("deployment_state_bytes"), f"{where}.deployment_state_bytes", integer=True))
    full_bytes = int(_number(capacity.get("full_persistent_bytes"), f"{where}.full_persistent_bytes", integer=True))
    package_bytes = int(_number(capacity.get("package_bytes"), f"{where}.package_bytes", integer=True))
    if evidence_bytes != expected_evidence_bytes or full_bytes != deployment_bytes + evidence_bytes or package_bytes != full_bytes:
        raise ValueError(f"{where} package byte totals do not include the full evidence sidecar exactly once")


def _audit_run(
    cases: Sequence[Mapping[str, Any]],
    branches: Sequence[Mapping[str, Any]],
    run_root: Path,
    design: Mapping[str, Any],
    families: Mapping[int, str | None],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    active, config, phase = design["active"], design["config"], design["phase"]
    is_e161 = _is_e161(config)
    horizons = [int(value) for value in active["horizons"]]
    max_horizon = max(horizons)
    case_by_key = {
        (int(row["seed"]), str(row["pair_kind"]), str(row["state"])): row
        for row in cases
    }
    case_observations: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    evidence_by_case: dict[str, dict[str, Any]] = {}
    branch_final_evidence_bytes = branch_post_action_evidence_bytes = 0
    prefix_steps_total = branch_steps_total = prefix_reward_total = branch_reward_total = 0
    prefix_rollouts = branch_rollouts = 0
    prefix_updates = branch_updates = 0
    branch_observations: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for case_key, case in case_by_key.items():
        case_id = str(case["case_id"])
        prefix = _case_prefix(case, f"case {case_id}")
        prefix_ledgers = _ledger_map(prefix, f"case {case_id}.prefix")
        prefix_steps_expected = int(design["expected_counts"]["early_prefix_steps" if case_key[2] == "early" else "late_prefix_steps"])
        prefix_record, _, _ = _audit_steps(run_root, prefix_ledgers.get("steps_csv"),
                                           f"case {case_id} prefix steps.csv", [prefix_steps_expected])
        if prefix_record["rows"] != prefix_steps_expected or int(_number(prefix.get("steps"), f"case {case_id}.prefix.steps", integer=True)) != prefix_steps_expected:
            raise ValueError(f"case {case_id} prefix actual steps differ from its frozen state")
        episode_record = _audit_count_ledger(run_root, prefix_ledgers.get("episodes_csv"), f"case {case_id} prefix episodes.csv")
        rollout_record = _audit_count_ledger(run_root, prefix_ledgers.get("rollouts_csv"), f"case {case_id} prefix rollouts.csv", rollouts=True)
        for item in (prefix_record, episode_record, rollout_record):
            raw_hashes[item["path"]] = item["sha256"]
        prefix_steps_total += prefix_record["rows"]
        prefix_reward_total += prefix_record["reward_sum"]
        prefix_rollouts += rollout_record["rows"]
        prefix_updates += rollout_record["minibatch_updates"]
        ppo = _mapping(prefix.get("ppo_call_counts"), f"case {case_id}.prefix.ppo_call_counts")
        if int(_number(ppo.get("rollouts_completed"), f"case {case_id}.prefix.rollouts_completed", integer=True)) != rollout_record["rows"]:
            raise ValueError(f"case {case_id} prefix rollouts disagree with rollouts.csv")
        if int(_number(ppo.get("minibatch_updates"), f"case {case_id}.prefix.minibatch_updates", integer=True)) != rollout_record["minibatch_updates"]:
            raise ValueError(f"case {case_id} prefix minibatch updates disagree with rollouts.csv")

        fit = _mapping(prefix.get("candidate_fit"), f"case {case_id}.prefix.candidate_fit")
        behavioral = _mapping(fit.get("behavioral"), f"case {case_id}.prefix.candidate_fit.behavioral")
        if fit.get("fit_state_unchanged") is not True or behavioral.get("finite_candidate") is not True:
            raise ValueError(f"case {case_id} candidate fit did not preserve its inputs/state or produced a nonfinite candidate")
        if int(_number(fit.get("fit_calls"), f"case {case_id}.candidate_fit.fit_calls", integer=True)) != 1:
            raise ValueError(f"case {case_id} candidate must be fit exactly once per prefix")
        for owner_key in ("train_rows_by_owner", "holdout_rows_by_owner"):
            values = behavioral.get(owner_key)
            if not isinstance(values, list) or len(values) != 2 or any(int(_number(x, f"case {case_id}.{owner_key}", integer=True)) <= 0 for x in values):
                raise ValueError(f"case {case_id} candidate-fit diagnostics require two nonempty owners in {owner_key}")
        band_counts = fit.get("band_counts_by_owner")
        if not isinstance(band_counts, list) or len(band_counts) != 2 or any(not isinstance(row, list) or len(row) != 3 for row in band_counts):
            raise ValueError(f"case {case_id} source band coverage must be a two-by-three table")
        for owner, counts in enumerate(band_counts):
            if sum(int(_number(value, f"case {case_id}.band_counts_by_owner[{owner}]", integer=True)) for value in counts) != int(config["evidence_rows_per_source"]):
                raise ValueError(f"case {case_id} owner {owner} band counts do not sum to the evidence budget")
        if int(_number(fit.get("source_buffer_bytes"), f"case {case_id}.source_buffer_bytes", integer=True)) != int(_number(prefix.get("evidence_buffer", {}).get("bytes"), f"case {case_id}.evidence_buffer.bytes", integer=True)):
            raise ValueError(f"case {case_id} fit buffer bytes disagree with the prefix evidence package")
        if fit.get("source_buffer_sha256") != prefix.get("evidence_buffer", {}).get("sha256"):
            raise ValueError(f"case {case_id} fit buffer hash differs from the prefix evidence package")
        evidence_record = _audit_evidence_buffer(run_root, case, prefix, fit, config)
        evidence_by_case[case_id] = evidence_record

        case_capacity = _check_capacity(prefix.get("capacity"), config, f"case {case_id} prefix capacity")
        _check_capacity_evidence(case_capacity, int(evidence_record["bytes"]), f"case {case_id} prefix capacity")
        case_costs = _mapping(case.get("costs"), f"case {case_id}.costs")
        fit_cpu = float(_number(fit.get("fit_cpu_s"), f"case {case_id}.candidate_fit.fit_cpu_s"))
        prefix_cpu = float(_number(case_costs.get("prefix_cpu_s"), f"case {case_id}.costs.prefix_cpu_s"))
        declared_fit_cpu = float(_number(case_costs.get("candidate_fit_cpu_s"), f"case {case_id}.costs.candidate_fit_cpu_s"))
        if not math.isclose(fit_cpu, declared_fit_cpu, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"case {case_id} fit CPU differs between fit and cost records")
        if prefix_cpu + 1e-9 < fit_cpu:
            raise ValueError(f"case {case_id} prefix CPU is smaller than its included fit CPU")
        for field, expected in (("package_bytes", case_capacity["package_bytes"]),
                                ("actor_container_bytes", case_capacity["actor_container_bytes"]),
                                ("package_budget_bytes", case_capacity["package_budget_bytes"])):
            if int(_number(case_costs.get(field), f"case {case_id}.costs.{field}", integer=True)) != expected:
                raise ValueError(f"case {case_id} package capacity differs between prefix and case cost records")
        case_observation = {
            "case_id": case_id, "seed": case_key[0], "family": families.get(case_key[0]),
            "pair_kind": case_key[1], "state": case_key[2], "history_regime": str(case["history_regime"]),
            "prefix_steps": prefix_record["rows"], "prefix_reward_sum": prefix_record["reward_sum"],
            "prefix_cpu_s_including_fit": prefix_cpu, "fit_cpu_s": fit_cpu,
            "prefix_cpu_s_excluding_fit_estimate": max(0.0, prefix_cpu - fit_cpu),
            "fit_wall_s": float(_number(fit.get("fit_wall_s"), f"case {case_id}.fit_wall_s")),
            "prefix_wall_s": float(_number(case_costs.get("prefix_wall_s"), f"case {case_id}.prefix_wall_s")),
            "evidence_buffer_bytes": int(fit["source_buffer_bytes"]),
            "candidate_fit": dict(fit), "behavioral_fit_diagnostics": dict(behavioral),
            "band_counts_by_owner": band_counts, "prefix_capacity": case_capacity,
            "evidence_buffer": evidence_record,
            "warmup_pair_records": _json_safe(prefix.get("warmup_pair_records", [])),
        }
        if is_e161:
            case_observation["prefix_rollouts_trained"] = rollout_record["rows"]
            case_observation["prefix_minibatch_updates"] = rollout_record["minibatch_updates"]
        case_observations.append(case_observation)

    for branch in branches:
        bid = str(branch["branch_id"])
        key = (int(branch["seed"]), str(branch["pair_kind"]), str(branch["state"]),
               str(branch["future_regime"]), int(branch["replicate"]), str(branch["arm"]))
        case = case_by_key[key[:3]]
        ledgers = _ledger_map(branch, f"branch {bid}")
        step_record, step_marks, step_rows = _audit_steps(
            run_root, ledgers.get("steps_csv"), f"branch {bid} steps.csv", horizons,
            expected_action=str(branch["arm"]), expected_future=str(branch["future_regime"]),
            require_bout=True,
        )
        if step_record["rows"] != max_horizon or int(_number(branch.get("raw_environment_steps"), f"branch {bid}.raw_environment_steps", integer=True)) != max_horizon:
            raise ValueError(f"branch {bid} actual step count differs from the longest horizon")
        episode_record = _audit_count_ledger(run_root, ledgers.get("episodes_csv"), f"branch {bid} episodes.csv")
        rollout_record = _audit_count_ledger(run_root, ledgers.get("rollouts_csv"), f"branch {bid} rollouts.csv", rollouts=True)
        for item in (step_record, episode_record, rollout_record):
            raw_hashes[item["path"]] = item["sha256"]
        branch_steps_total += step_record["rows"]
        branch_reward_total += step_record["reward_sum"]
        branch_rollouts += rollout_record["rows"]
        branch_updates += rollout_record["minibatch_updates"]
        branch_ppo = _mapping(branch.get("ppo_call_counts"), f"branch {bid}.ppo_call_counts")
        for field, expected in (("rollouts_completed", rollout_record["rows"]),
                                ("minibatch_updates", rollout_record["minibatch_updates"])):
            if int(_number(branch_ppo.get(field), f"branch {bid}.ppo_call_counts.{field}", integer=True)) != expected:
                raise ValueError(f"branch {bid} {field} disagrees with rollouts.csv")
        hrows = branch.get("horizons")
        if not isinstance(hrows, list):
            raise ValueError(f"branch {bid}.horizons must be an array")
        declared: dict[int, Mapping[str, Any]] = {}
        for hrow in hrows:
            hrow = _mapping(hrow, f"branch {bid}.horizons[]")
            horizon = int(_number(hrow.get("steps"), f"branch {bid}.horizon.steps", integer=True))
            if horizon in declared:
                raise ValueError(f"branch {bid} has duplicate horizon {horizon}")
            declared[horizon] = hrow
        if set(declared) != set(horizons):
            raise ValueError(f"branch {bid} horizon grid is incomplete")
        costs = _mapping(branch.get("costs"), f"branch {bid}.costs")
        branch_cpu = float(_number(costs.get("branch_cpu_s"), f"branch {bid}.costs.branch_cpu_s"))
        if branch_cpu < 0:
            raise ValueError(f"branch {bid} branch CPU must be nonnegative")
        capacity = _check_capacity(branch.get("capacity"), config, f"branch {bid} final capacity")
        capacity_after = _check_capacity(branch.get("capacity_after_action"), config, f"branch {bid} post-action capacity")
        evidence_record = evidence_by_case[str(branch["case_id"])]
        evidence_bytes = int(evidence_record["bytes"])
        _check_capacity_evidence(capacity, evidence_bytes, f"branch {bid} final capacity")
        _check_capacity_evidence(capacity_after, evidence_bytes, f"branch {bid} post-action capacity")
        branch_evidence_bytes = int(_number(branch.get("evidence_buffer_bytes"), f"branch {bid}.evidence_buffer_bytes", integer=True))
        if branch_evidence_bytes != evidence_bytes:
            raise ValueError(f"branch {bid} evidence buffer bytes differ from its prefix source")
        for artifacts_key, capacity_value, total_name in (
            ("final_artifacts", capacity, "final"),
            ("post_action_artifacts", capacity_after, "post-action"),
        ):
            artifacts = _mapping(branch.get(artifacts_key), f"branch {bid}.{artifacts_key}")
            artifact = _mapping(artifacts.get("evidence"), f"branch {bid}.{artifacts_key}.evidence")
            artifact_path = _audit_evidence_artifact(
                run_root, artifact, f"branch {bid}.{total_name} evidence artifact",
                str(evidence_record["sha256"]), evidence_bytes,
            )
            if total_name == "final":
                branch_final_evidence_bytes += int(artifact_path.stat().st_size)
            else:
                branch_post_action_evidence_bytes += int(artifact_path.stat().st_size)
            if total_name == "final":
                for capacity_field in ("package_bytes", "actor_container_bytes", "package_budget_bytes"):
                    if int(_number(costs.get(capacity_field), f"branch {bid}.costs.{capacity_field}", integer=True)) != capacity[capacity_field]:
                        raise ValueError(f"branch {bid} final package capacity differs from its cost record")
        action_install = _mapping(branch.get("action_install"), f"branch {bid}.action_install")
        counts_delta = branch.get("counts_delta")
        slot_evictions = runtime_slot_evictions = None
        if is_e161:
            counts_delta = _mapping(counts_delta, f"branch {bid}.counts_delta")
            slot_evictions = int(_number(counts_delta.get("slot_evictions"),
                                         f"branch {bid}.counts_delta.slot_evictions", integer=True))
            runtime_counts = _mapping(counts_delta.get("runtime_counts"),
                                      f"branch {bid}.counts_delta.runtime_counts")
            runtime_slot_evictions = int(_number(runtime_counts.get("slot_evictions"),
                                                 f"branch {bid}.counts_delta.runtime_counts.slot_evictions",
                                                 integer=True))
            if slot_evictions != runtime_slot_evictions:
                raise ValueError(f"branch {bid} learner and runtime eviction counters differ")
        install = float(_number(costs.get("install_cpu_s"), f"branch {bid}.costs.install_cpu_s"))
        if install < 0:
            raise ValueError(f"branch {bid} install CPU must be nonnegative")
        for horizon in horizons:
            hrow = declared[horizon]
            raw = step_marks.get(horizon)
            if raw is None:
                raise ValueError(f"branch {bid} raw steps are missing horizon {horizon}")
            for field, observed, tolerance in (("reward_sum", raw["reward_sum"], 1e-8),
                                               ("reward_per_step", raw["reward_per_step"], 1e-10)):
                value = float(_number(hrow.get(field), f"branch {bid}.{horizon}.{field}"))
                if not math.isclose(value, float(observed), rel_tol=1e-11, abs_tol=tolerance):
                    raise ValueError(f"branch {bid} {field} at {horizon} differs from raw steps.csv")
            if int(_number(hrow.get("raw_environment_steps"), f"branch {bid}.{horizon}.raw_environment_steps", integer=True)) != horizon:
                raise ValueError(f"branch {bid} raw step count is wrong at horizon {horizon}")
            simulation_cpu = float(_number(hrow.get("simulation_cpu_s"), f"branch {bid}.{horizon}.simulation_cpu_s"))
            utility_rate = float(_number(hrow.get("utility_rate"), f"branch {bid}.{horizon}.utility_rate"))
            if simulation_cpu < 0:
                raise ValueError(f"branch {bid} has negative simulation CPU")
            checkpoint = rollout_record["checkpoints"].get(horizon)
            if checkpoint is None:
                raise ValueError(f"branch {bid} rollouts.csv has no trained-row checkpoint at {horizon} steps")
            for field in ("rollouts_trained", "minibatch_updates"):
                if int(_number(hrow.get(field), f"branch {bid}.{horizon}.{field}", integer=True)) != int(checkpoint[field]):
                    raise ValueError(f"branch {bid} {field} at {horizon} differs from rollouts.csv")
            declared_install_cpu = float(_number(
                hrow.get("action_install_cpu_s"), f"branch {bid}.{horizon}.action_install_cpu_s",
            ))
            if not math.isclose(declared_install_cpu, install, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(f"branch {bid} action-install CPU differs between horizon and branch costs")
            expected_utility = float(raw["reward_per_step"]) - (install + simulation_cpu) / horizon
            if not math.isclose(utility_rate, expected_utility, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"branch {bid} utility rate at {horizon} differs from raw reward and measured CPU")
            observation = {
                "branch_id": bid, "case_id": str(branch["case_id"]), "seed": key[0],
                "family": families.get(key[0]), "pair_kind": key[1], "state": key[2],
                "history_regime": str(branch["history_regime"]), "future_regime": key[3],
                "replicate": key[4], "arm": key[5], "horizon": horizon,
                "reward_sum": float(raw["reward_sum"]), "reward_per_step": float(raw["reward_per_step"]),
                "simulation_cpu_s": simulation_cpu, "action_install_cpu_s": install,
                "utility_rate": utility_rate, "branch_cpu_s": branch_cpu,
                "install_cpu_s": install, "capacity": capacity,
                "capacity_after_action": capacity_after,
                "action_install": dict(action_install),
                "new_source_teacher_kl_before_codec": _json_safe(action_install.get("new_source_teacher_kl_before_codec")),
                "new_source_teacher_kl_after_codec": _json_safe(action_install.get("new_source_teacher_kl_after_codec")),
            }
            if is_e161:
                observation["rollouts_trained"] = int(_number(
                    hrow.get("rollouts_trained"), f"branch {bid}.{horizon}.rollouts_trained", integer=True,
                ))
                observation["minibatch_updates"] = int(_number(
                    hrow.get("minibatch_updates"), f"branch {bid}.{horizon}.minibatch_updates", integer=True,
                ))
                observation["slot_evictions"] = slot_evictions
                observation["runtime_slot_evictions"] = runtime_slot_evictions
                observation["counts_delta"] = _json_safe(counts_delta)
            branch_observations.append(observation)
        per_bout = branch.get("per_bout")
        if not isinstance(per_bout, list) or not per_bout:
            raise ValueError(f"branch {bid} per_bout must be a nonempty array")
        seen_bouts: set[int] = set()
        selected: dict[str, list[dict[str, float]]] = {"allocated_only": [], "all_bouts": []}
        for bout_index, raw_bout in enumerate(per_bout):
            bout = _mapping(raw_bout, f"branch {bid}.per_bout[{bout_index}]")
            bout_id = int(_number(bout.get("bout"), f"branch {bid}.per_bout.bout", integer=True))
            if bout_id in seen_bouts:
                raise ValueError(f"branch {bid} repeats bout {bout_id}")
            seen_bouts.add(bout_id)
            if not isinstance(bout.get("allocated"), bool):
                raise ValueError(f"branch {bid} bout {bout_id} must state allocated explicitly")
            bout_steps = int(_number(bout.get("steps"), f"branch {bid}.per_bout.steps", integer=True))
            first_return = float(_number(bout.get("return_first_256"), f"branch {bid}.per_bout.return_first_256"))
            full_return = float(_number(bout.get("return_full"), f"branch {bid}.per_bout.return_full"))
            if bout_steps < 256:
                raise ValueError(f"branch {bid} bout {bout_id} is shorter than the recovery window")
            raw_bout_metrics = step_rows.get(bout_id)
            if raw_bout_metrics is None:
                raise ValueError(f"branch {bid} bout {bout_id} is missing from steps.csv")
            if int(raw_bout_metrics["steps"]) != bout_steps:
                raise ValueError(f"branch {bid} bout {bout_id} step count differs from steps.csv")
            if int(raw_bout_metrics["first_256_rows"]) != 256:
                raise ValueError(f"branch {bid} bout {bout_id} lacks a full first-256 recovery window")
            for field, declared_value, raw_value in (
                ("return_first_256", first_return, float(raw_bout_metrics["return_first_256"])),
                ("return_full", full_return, float(raw_bout_metrics["reward_full"])),
            ):
                if not math.isclose(declared_value, raw_value, rel_tol=1e-11, abs_tol=1e-8):
                    raise ValueError(f"branch {bid} bout {bout_id} {field} differs from steps.csv")
            row = {"return_first_256_per_bout": first_return, "return_first_256_per_step": first_return / 256.0}
            selected["all_bouts"].append(row)
            if bout["allocated"]:
                selected["allocated_only"].append(row)
        if not selected["all_bouts"] or not selected["allocated_only"]:
            raise ValueError(f"branch {bid} must contain both allocated-only and all-bout recovery observations")
        if set(step_rows) != seen_bouts:
            raise ValueError(f"branch {bid} per_bout summary does not cover every steps.csv bout")
        for population, rows in selected.items():
            recovery_rows.append({
                "seed": key[0], "family": families.get(key[0]), "pair_kind": key[1], "state": key[2],
                "future_regime": key[3], "replicate": key[4], "arm": key[5],
                "population": population, "bout_count": len(rows),
                "return_first_256_per_bout": float(np.mean([row["return_first_256_per_bout"] for row in rows])),
                "return_first_256_per_step": float(np.mean([row["return_first_256_per_step"] for row in rows])),
            })

    expected = design["expected_counts"]
    total_steps = prefix_steps_total + branch_steps_total
    if (len(cases), len(branches), prefix_steps_total, branch_steps_total, total_steps) != (
        expected["cases"], expected["branches"], expected["prefix_steps"],
        expected["branch_steps"], expected["total_steps"],
    ):
        raise ValueError(
            f"recounted grid/step totals differ from frozen design: cases={len(cases)}, branches={len(branches)}, "
            f"prefix={prefix_steps_total}, branch={branch_steps_total}, total={total_steps}"
        )
    return branch_observations, {
        "prefix_environment_steps": prefix_steps_total,
        "branch_environment_steps": branch_steps_total,
        "total_environment_steps": total_steps,
        "prefix_reward_sum": prefix_reward_total,
        "branch_reward_sum": branch_reward_total,
        "prefix_rollouts_trained": prefix_rollouts,
        "branch_rollouts_trained": branch_rollouts,
        "total_rollouts_trained": prefix_rollouts + branch_rollouts,
        "prefix_minibatch_updates": prefix_updates,
        "branch_minibatch_updates": branch_updates,
        "total_minibatch_updates": prefix_updates + branch_updates,
        "raw_ledger_count": len(raw_hashes),
        "raw_ledger_sha256": dict(sorted(raw_hashes.items())),
        "raw_ledger_hash_manifest_sha256": _canonical_hash(raw_hashes),
        "evidence_buffer_bytes": {
            "prefix_sidecars_once": int(sum(int(row["bytes"]) for row in evidence_by_case.values())),
            "final_branch_packages": branch_final_evidence_bytes,
            "post_action_packages": branch_post_action_evidence_bytes,
            "all_persisted_prefix_and_branch_package_copies": int(
                sum(int(row["bytes"]) for row in evidence_by_case.values())
                + branch_final_evidence_bytes + branch_post_action_evidence_bytes
            ),
            "prefix_count": len(evidence_by_case),
            "final_branch_package_count": len(branches),
            "post_action_package_count": len(branches),
        },
        "case_costs": case_observations,
        "recovery_branch_rows": recovery_rows,
        "branch_cpu_s": float(sum(float(row["branch_cpu_s"]) for row in branch_observations if row["horizon"] == max_horizon)),
    }, recovery_rows


def _check_capacity(value: Any, config: Mapping[str, Any], where: str) -> dict[str, Any]:
    capacity = _mapping(value, where)
    actor_budget = int(_number(config.get("actor_budget_bytes"), "config.actor_budget_bytes", integer=True))
    package_budget = int(_number(config.get("package_budget_bytes"), "config.package_budget_bytes", integer=True))
    actor_bytes = int(_number(capacity.get("actor_container_bytes"), f"{where}.actor_container_bytes", integer=True))
    package_bytes = int(_number(capacity.get("package_bytes"), f"{where}.package_bytes", integer=True))
    declared_budget = int(_number(capacity.get("package_budget_bytes"), f"{where}.package_budget_bytes", integer=True))
    if actor_budget != 89 or package_budget != 262144:
        raise ValueError("frozen byte budgets must be exactly 89 actor bytes and 262144 package bytes")
    if actor_bytes != actor_budget or declared_budget != package_budget or package_bytes > package_budget:
        raise ValueError(f"{where} violates the frozen actor/package byte budget")
    return {**dict(capacity), "actor_container_bytes": actor_bytes, "package_bytes": package_bytes,
            "package_budget_bytes": declared_budget, "within_package_budget": package_bytes <= package_budget}


def _audit_campaign_costs(
    summary: Mapping[str, Any], design: Mapping[str, Any], recount: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]], branches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    costs = _mapping(summary.get("costs"), "summary.costs")
    expected = design["expected_counts"]
    declared_interactions = _mapping(costs.get("expected_interactions"), "summary.costs.expected_interactions")
    if {str(key): int(_number(value, f"summary.costs.expected_interactions.{key}", integer=True))
        for key, value in declared_interactions.items()} != expected:
        raise ValueError("runner expected_interactions differs from the frozen derived grid")
    for field, expected_value in (
        ("prefix_environment_steps", recount["prefix_environment_steps"]),
        ("branch_environment_steps", recount["branch_environment_steps"]),
        ("environment_steps_total", recount["total_environment_steps"]),
        ("expected_environment_steps_total", expected["total_steps"]),
    ):
        if int(_number(costs.get(field), f"summary.costs.{field}", integer=True)) != int(expected_value):
            raise ValueError(f"runner {field} differs from the raw step recount")

    prefix_cpu = sum(float(_number(_mapping(case.get("costs"), f"case {case.get('case_id')}.costs").get("prefix_cpu_s"),
                                   f"case {case.get('case_id')}.prefix_cpu_s")) for case in cases)
    fit_cpu = sum(float(_number(_mapping(case.get("costs"), f"case {case.get('case_id')}.costs").get("candidate_fit_cpu_s"),
                                f"case {case.get('case_id')}.candidate_fit_cpu_s")) for case in cases)
    case_branch_cpu = sum(float(_number(_mapping(case.get("costs"), f"case {case.get('case_id')}.costs").get("branch_cpu_s"),
                                        f"case {case.get('case_id')}.branch_cpu_s")) for case in cases)
    branch_cpu = sum(float(_number(_mapping(branch.get("costs"), f"branch {branch.get('branch_id')}.costs").get("branch_cpu_s"),
                                   f"branch {branch.get('branch_id')}.branch_cpu_s")) for branch in branches)
    for field, expected_value in (("prefix_cpu_s", prefix_cpu), ("candidate_fit_cpu_s", fit_cpu),
                                  ("branch_cpu_s", branch_cpu)):
        value = float(_number(costs.get(field), f"summary.costs.{field}"))
        if not math.isclose(value, expected_value, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"runner {field} differs from per-case/branch CPU records")
    if not math.isclose(case_branch_cpu, branch_cpu, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("case branch CPU totals differ from individual branch CPU records")
    for case in cases:
        case_costs = _mapping(case.get("costs"), f"case {case.get('case_id')}.costs")
        case_cpu = float(_number(case_costs.get("case_cpu_s"), f"case {case.get('case_id')}.case_cpu_s"))
        included = float(_number(case_costs.get("prefix_cpu_s"), f"case {case.get('case_id')}.prefix_cpu_s"))
        included += float(_number(case_costs.get("branch_cpu_s"), f"case {case.get('case_id')}.branch_cpu_s"))
        if case_cpu + 1e-8 < included:
            raise ValueError(f"case {case.get('case_id')} CPU total is below its measured prefix plus branch work")
    evidence = _mapping(recount.get("evidence_buffer_bytes"), "recount.evidence_buffer_bytes")
    prefix_bytes = int(evidence["prefix_sidecars_once"])
    final_branch_bytes = int(evidence["final_branch_packages"])
    if int(_number(costs.get("serialized_evidence_bytes_prefix_once"),
                   "summary.costs.serialized_evidence_bytes_prefix_once", integer=True)) != prefix_bytes:
        raise ValueError("runner prefix evidence byte count differs from the sidecar files")
    if int(_number(costs.get("serialized_evidence_bytes_all_branch_packages"),
                   "summary.costs.serialized_evidence_bytes_all_branch_packages", integer=True)) != final_branch_bytes:
        raise ValueError("runner branch evidence byte count differs from final package sidecars")
    return {
        "passed": True,
        "prefix_cpu_s": prefix_cpu,
        "candidate_fit_cpu_s_included_in_prefix": fit_cpu,
        "branch_cpu_s": branch_cpu,
        "case_branch_cpu_sum_matches_individual_branches": True,
        "environment_steps_match_raw_ledgers": True,
        "prefix_sidecar_bytes_match": True,
        "final_branch_package_sidecar_bytes_match": True,
        "post_action_sidecar_bytes_recounted": int(evidence["post_action_packages"]),
    }


def _branch_value_index(observations: Sequence[Mapping[str, Any]], metric: str) -> dict[tuple[Any, ...], float]:
    result: dict[tuple[Any, ...], float] = {}
    for row in observations:
        key = (int(row["seed"]), str(row["pair_kind"]), str(row["state"]),
               str(row["future_regime"]), int(row["replicate"]), str(row["arm"]), int(row.get("horizon", 0)))
        value = _number(row.get(metric), f"observation.{metric}", nullable=True)
        if value is None:
            continue
        if key in result:
            raise ValueError(f"duplicate observation for {key}")
        result[key] = float(value)
    return result


def _bootstrap_stat(
    per_seed: Mapping[int, float], families: Mapping[int, str | None],
    *, phase: str, resamples: int, bootstrap_seed: int,
) -> dict[str, Any]:
    items = [{"seed": int(seed), "family": families.get(int(seed)), "value": float(value)}
             for seed, value in sorted(per_seed.items())]
    mean = float(np.mean([row["value"] for row in items])) if items else None
    interval = None
    method = "pilot single seed; interval not estimated" if phase == "pilot" else "stratified paired seed bootstrap"
    if phase == "main":
        if len(items) != 6 or any(families.get(seed) not in OLD_FAMILIES for seed in per_seed):
            raise ValueError("main bootstrap requires six independent seeds with old-family labels")
        strata = {family: [int(seed) for seed in per_seed if families.get(int(seed)) == family]
                  for family in OLD_FAMILIES}
        if any(len(strata[family]) != 3 for family in OLD_FAMILIES):
            raise ValueError("main bootstrap strata must have three seeds from each old family")
        rng = np.random.default_rng(int(bootstrap_seed))
        draws = np.empty(int(resamples), dtype=np.float64)
        for index in range(int(resamples)):
            sampled: list[float] = []
            for family in OLD_FAMILIES:
                choices = rng.choice(strata[family], size=3, replace=True)
                sampled.extend(float(per_seed[int(seed)]) for seed in choices)
            draws[index] = float(np.mean(sampled))
        interval = [float(value) for value in np.quantile(draws, [0.025, 0.975], method="linear")]
    return {"mean": mean, "ci95": interval, "seed_count": len(items), "per_seed": items,
            "bootstrap_method": method, "bootstrap_resamples": int(resamples) if phase == "main" else 0}


def _student_t_quantile(probability: float, degrees_freedom: float) -> float:
    """Return a Student-t quantile using the already locked mpmath dependency."""
    from mpmath import mp

    if not 0.0 < probability < 1.0 or degrees_freedom <= 0.0:
        raise ValueError("Student-t probability and degrees of freedom must be valid")
    with mp.workdps(45):
        target = mp.mpf(str(probability))
        if math.isinf(degrees_freedom):
            return float(mp.sqrt(2) * mp.erfinv(2 * target - 1))
        df = mp.mpf(str(degrees_freedom))

        def cdf(value: Any) -> Any:
            if value == 0:
                return mp.mpf("0.5")
            tail = mp.betainc(df / 2, mp.mpf("0.5"), 0,
                              df / (df + value * value), regularized=True) / 2
            return 1 - tail if value > 0 else tail

        if probability == 0.5:
            return 0.0
        sign = 1 if probability > 0.5 else -1
        low, high = mp.mpf("0"), mp.mpf("1")
        while cdf(high) < max(target, 1 - target):
            high *= 2
        for _ in range(180):
            middle = (low + high) / 2
            if cdf(middle) < max(target, 1 - target):
                low = middle
            else:
                high = middle
        return float(sign * (low + high) / 2)


def _e161_stratified_stat(
    per_seed: Mapping[int, float], families: Mapping[int, str | None],
    *, resamples: int, bootstrap_seed: int,
) -> dict[str, Any]:
    """Equal-family Welch inference with a stratified bootstrap sensitivity interval."""
    strata = {
        family: [(int(seed), float(value)) for seed, value in sorted(per_seed.items())
                 if families.get(int(seed)) == family]
        for family in OLD_FAMILIES
    }
    if set(per_seed) != {int(seed) for rows in strata.values() for seed, _ in rows}:
        raise ValueError("E161 effects require an A/B family label for every independent seed")
    if any(len(strata[family]) != 8 for family in OLD_FAMILIES):
        raise ValueError("E161 statistics require eight independent seeds in each A/B stratum")
    values_a = np.asarray([value for _, value in strata["A"]], dtype=np.float64)
    values_b = np.asarray([value for _, value in strata["B"]], dtype=np.float64)
    mean_a, mean_b = float(np.mean(values_a)), float(np.mean(values_b))
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    estimate = (mean_a + mean_b) / 2.0
    component_a, component_b = var_a / len(values_a), var_b / len(values_b)
    standard_error = math.sqrt(component_a + component_b) / 2.0
    denominator = component_a * component_a / (len(values_a) - 1) + component_b * component_b / (len(values_b) - 1)
    degrees_freedom = math.inf if denominator == 0.0 else (component_a + component_b) ** 2 / denominator
    if standard_error == 0.0:
        lower = estimate
        ci95 = [estimate, estimate]
        t_statistic: float | None = 0.0 if estimate == 0.0 else None
        t_statistic_degenerate = "zero" if estimate == 0.0 else (
            "positive_infinity" if estimate > 0.0 else "negative_infinity"
        )
    else:
        t_statistic = estimate / standard_error
        t_statistic_degenerate = None
        lower = estimate - _student_t_quantile(0.95, degrees_freedom) * standard_error
        half_width = _student_t_quantile(0.975, degrees_freedom) * standard_error
        ci95 = [estimate - half_width, estimate + half_width]

    rng = np.random.default_rng(int(bootstrap_seed))
    draws = np.empty(int(resamples), dtype=np.float64)
    for family_index, values in enumerate((values_a, values_b)):
        sampled_indices = rng.integers(0, len(values), size=(int(resamples), len(values)))
        family_means = np.mean(values[sampled_indices], axis=1)
        if family_index == 0:
            draws[:] = family_means / 2.0
        else:
            draws += family_means / 2.0
    bootstrap_ci95 = [float(value) for value in np.quantile(draws, [0.025, 0.975], method="linear")]
    items = [{"seed": int(seed), "family": families[int(seed)], "value": float(value)}
             for seed, value in sorted(per_seed.items())]
    return {
        "mean": estimate,
        "ci95": ci95,
        "ci95_method": "two-sided 95% A/B-stratified Welch-t interval",
        "lower95_one_sided": lower,
        "one_sided_lower_method": "one-sided 95% A/B-stratified Welch-t lower bound",
        "standard_error": standard_error,
        "welch_degrees_freedom": None if math.isinf(degrees_freedom) else degrees_freedom,
        "welch_degrees_freedom_infinite": math.isinf(degrees_freedom),
        "welch_t_statistic": t_statistic,
        "welch_t_statistic_degenerate": t_statistic_degenerate,
        "family_summary": {
            family: {"seed_count": len(strata[family]), "mean": float(np.mean(values)),
                     "sample_sd": float(np.std(values, ddof=1))}
            for family, values in (("A", values_a), ("B", values_b))
        },
        "seed_count": len(items),
        "per_seed": items,
        "bootstrap_ci95_sensitivity": bootstrap_ci95,
        "bootstrap_method": "A/B-stratified paired-seed bootstrap sensitivity only",
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(bootstrap_seed),
    }


def _contrast_seed_values(
    index: Mapping[tuple[Any, ...], float],
    *, seed: int, pair: str, state: str, horizon: int,
    futures: Sequence[str], replicates: Sequence[int],
    left: str, right: str,
) -> float:
    differences: list[float] = []
    for future in futures:
        for replicate in replicates:
            akey = (seed, pair, state, future, replicate, left, horizon)
            bkey = (seed, pair, state, future, replicate, right, horizon)
            if akey not in index or bkey not in index:
                raise ValueError(f"paired contrast is missing branch cell(s): {akey}, {bkey}")
            differences.append(index[akey] - index[bkey])
    return float(np.mean(differences))


def _e161_contrast_seed_values(
    index: Mapping[tuple[Any, ...], float],
    *, seed: int, pair: str, state: str, horizon: int,
    futures: Sequence[str], replicates: Sequence[int],
    left: str, right: str,
) -> float:
    """Average replicas within each regime, then weight each regime equally."""
    regime_means = []
    for future in futures:
        differences = []
        for replicate in replicates:
            akey = (seed, pair, state, future, replicate, left, horizon)
            bkey = (seed, pair, state, future, replicate, right, horizon)
            if akey not in index or bkey not in index:
                raise ValueError(f"paired contrast is missing branch cell(s): {akey}, {bkey}")
            differences.append(index[akey] - index[bkey])
        regime_means.append(float(np.mean(differences)))
    if not regime_means:
        raise ValueError("E161 contrast requires at least one primary future regime")
    return float(np.mean(regime_means))


def _compute_effects(
    observations: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    families: Mapping[int, str | None],
    active: Mapping[str, Any],
    config: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    report_config = _mapping(config.get("report"), "config.report")
    resamples = int(_number(report_config.get("bootstrap_resamples"), "config.report.bootstrap_resamples", integer=True))
    bootstrap_seed = int(_number(report_config.get("bootstrap_seed"), "config.report.bootstrap_seed", integer=True))
    is_e161 = _is_e161(config)
    horizons = [int(value) for value in active["horizons"]]
    primary_horizon = int(_number(report_config.get("primary_horizon"), "config.report.primary_horizon", integer=True))
    if primary_horizon not in horizons:
        raise ValueError("primary_horizon is not one of the saved horizons")
    seeds, pairs, states = active["seeds"], active["pair_kinds"], active["states"]
    futures, replicates = active["future_regimes"], active["replicates"]
    arms_set = set(active.get("arms", ARMS))
    contrasts = _available_contrasts(active["arms"]) if is_e161 else OUTCOME_CONTRASTS
    primary_futures = ([str(value) for value in report_config["primary_future_regimes"]]
                       if is_e161 else [str(value) for value in futures])

    def contrast_seed(*, seed: int, pair: str, state: str, horizon: int,
                      futures: Sequence[str], left: str, right: str) -> float:
        implementation = _e161_contrast_seed_values if is_e161 else _contrast_seed_values
        return implementation(reward_index, seed=seed, pair=pair, state=state, horizon=horizon,
                               futures=futures, replicates=replicates, left=left, right=right)

    def summarize(per_seed: Mapping[int, float]) -> dict[str, Any]:
        if is_e161:
            return _e161_stratified_stat(per_seed, families, resamples=resamples,
                                         bootstrap_seed=bootstrap_seed)
        return _bootstrap_stat(per_seed, families, phase=phase, resamples=resamples,
                               bootstrap_seed=bootstrap_seed)

    reward_index = _branch_value_index(observations, "reward_per_step")
    effects: list[dict[str, Any]] = []
    future_effects: list[dict[str, Any]] = []
    interaction_effects: list[dict[str, Any]] = []
    for horizon in horizons:
        for pair in pairs:
            for state in states:
                for name, left, right in contrasts:
                    per_seed = {
                        int(seed): contrast_seed(seed=int(seed), pair=str(pair), state=str(state),
                                                 horizon=horizon, futures=primary_futures,
                                                 left=left, right=right)
                        for seed in seeds
                    }
                    effects.append({"pair_kind": pair, "state": state, "horizon": horizon,
                                    "contrast": name, "left_arm": left, "right_arm": right,
                                    "reward_per_step": summarize(per_seed)})
                    for future in futures:
                        scoped = {
                            int(seed): contrast_seed(seed=int(seed), pair=str(pair), state=str(state),
                                                     horizon=horizon, futures=[str(future)],
                                                     left=left, right=right)
                            for seed in seeds
                        }
                        future_effects.append({"pair_kind": pair, "state": state, "future_regime": future,
                                               "horizon": horizon, "contrast": name,
                                               "left_arm": left, "right_arm": right,
                                               "reward_per_step": summarize(scoped)})
                # Candidate-by-operation interaction: (merge behavioral - merge mean) - (replace behavioral - replace mean).
                if {"replace_behavioral", "replace_mean", "merge_behavioral", "merge_mean"} <= arms_set:
                    per_seed_interaction = {}
                    per_future_interaction: dict[str, dict[int, float]] = {str(future): {} for future in futures}
                    for seed in seeds:
                        replace_gap = contrast_seed(
                            seed=int(seed), pair=str(pair), state=str(state), horizon=horizon,
                            futures=primary_futures, left="replace_behavioral", right="replace_mean",
                        )
                        merge_gap = contrast_seed(
                            seed=int(seed), pair=str(pair), state=str(state), horizon=horizon,
                            futures=primary_futures, left="merge_behavioral", right="merge_mean",
                        )
                        per_seed_interaction[int(seed)] = merge_gap - replace_gap
                        for future in futures:
                            replace_future = contrast_seed(
                                seed=int(seed), pair=str(pair), state=str(state), horizon=horizon,
                                futures=[str(future)], left="replace_behavioral", right="replace_mean",
                            )
                            merge_future = contrast_seed(
                                seed=int(seed), pair=str(pair), state=str(state), horizon=horizon,
                                futures=[str(future)], left="merge_behavioral", right="merge_mean",
                            )
                            per_future_interaction[str(future)][int(seed)] = merge_future - replace_future
                    interaction_effects.append({"pair_kind": pair, "state": state, "horizon": horizon,
                                                "contrast": INTERACTION_NAME,
                                                "reward_per_step": summarize(per_seed_interaction)})
                    for future in futures:
                        future_effects.append({"pair_kind": pair, "state": state, "future_regime": future,
                                               "horizon": horizon, "contrast": INTERACTION_NAME,
                                               "reward_per_step": summarize(per_future_interaction[str(future)])})

    recovery_effects: list[dict[str, Any]] = []
    recovery_future_effects: list[dict[str, Any]] = []
    recovery_index: dict[tuple[Any, ...], float] = {}
    for row in recovery:
        key = (int(row["seed"]), str(row["pair_kind"]), str(row["state"]), str(row["future_regime"]),
               int(row["replicate"]), str(row["arm"]), str(row["population"]))
        metric_value = _number(row.get("return_first_256_per_step"), f"recovery {key}.return_first_256_per_step", nullable=True)
        if metric_value is not None:
            if key in recovery_index:
                raise ValueError(f"duplicate recovery observation {key}")
            recovery_index[key] = float(metric_value)
    for population in ("allocated_only", "all_bouts"):
        for pair in pairs:
            for state in states:
                for name, left, right in contrasts:
                    per_seed: dict[int, float] = {}
                    for seed in seeds:
                        regime_means = []
                        all_differences = []
                        for future in primary_futures:
                            differences = []
                            for replicate in replicates:
                                akey = (int(seed), str(pair), str(state), str(future), int(replicate), left, population)
                                bkey = (int(seed), str(pair), str(state), str(future), int(replicate), right, population)
                                if akey not in recovery_index or bkey not in recovery_index:
                                    raise ValueError(f"recovery contrast is missing paired bout population: {akey}, {bkey}")
                                differences.append(recovery_index[akey] - recovery_index[bkey])
                                all_differences.append(recovery_index[akey] - recovery_index[bkey])
                            regime_means.append(float(np.mean(differences)))
                        per_seed[int(seed)] = float(np.mean(regime_means) if is_e161 else np.mean(all_differences))
                    recovery_effects.append({"population": population, "pair_kind": pair, "state": state,
                                             "contrast": name, "metric": "return_first_256_per_step",
                                             "stat": summarize(per_seed)})
                    for future in futures:
                        by_seed_future: dict[int, float] = {}
                        for seed in seeds:
                            differences = []
                            for replicate in replicates:
                                akey = (int(seed), str(pair), str(state), str(future), int(replicate), left, population)
                                bkey = (int(seed), str(pair), str(state), str(future), int(replicate), right, population)
                                if akey not in recovery_index or bkey not in recovery_index:
                                    raise ValueError(f"recovery contrast is missing paired bout population: {akey}, {bkey}")
                                differences.append(recovery_index[akey] - recovery_index[bkey])
                            by_seed_future[int(seed)] = float(np.mean(differences))
                        recovery_future_effects.append({"population": population, "pair_kind": pair, "state": state,
                                                        "future_regime": future, "contrast": name,
                                                        "metric": "return_first_256_per_step",
                                                        "stat": summarize(by_seed_future)})
                if {"replace_behavioral", "replace_mean", "merge_behavioral", "merge_mean"} <= arms_set:
                    per_seed_interaction = {}
                    for seed in seeds:
                        replace_gap = []
                        merge_gap = []
                        for future in primary_futures:
                            for replicate in replicates:
                                get = lambda arm: recovery_index[(int(seed), str(pair), str(state), str(future), int(replicate), arm, population)]
                                replace_gap.append(get("replace_behavioral") - get("replace_mean"))
                                merge_gap.append(get("merge_behavioral") - get("merge_mean"))
                        per_seed_interaction[int(seed)] = float(np.mean(merge_gap) - np.mean(replace_gap))
                    recovery_effects.append({"population": population, "pair_kind": pair, "state": state,
                                             "contrast": INTERACTION_NAME, "metric": "return_first_256_per_step",
                                             "stat": summarize(per_seed_interaction)})
    future_interactions = []
    if "old_major" in futures and "new_major" in futures:
        for pair in pairs:
            for state in states:
                for name, left, right in contrasts:
                    by_seed = {}
                    for seed in seeds:
                        old_value = contrast_seed(seed=int(seed), pair=str(pair), state=str(state),
                                                  horizon=primary_horizon, futures=["old_major"],
                                                  left=left, right=right)
                        new_value = contrast_seed(seed=int(seed), pair=str(pair), state=str(state),
                                                  horizon=primary_horizon, futures=["new_major"],
                                                  left=left, right=right)
                        by_seed[int(seed)] = new_value - old_value
                    future_interactions.append({"pair_kind": pair, "state": state, "horizon": primary_horizon,
                                                "contrast": f"new_major_minus_old_major:{name}",
                                                "reward_per_step": summarize(by_seed)})
    result = {"primary_horizon": primary_horizon, "effects": effects,
            "future_scoped_effects": future_effects, "candidate_by_operation_interaction": interaction_effects,
            "future_prevalence_interaction": future_interactions, "recovery_effects": recovery_effects}
    if is_e161:
        result["primary_future_regimes"] = primary_futures
        result["recovery_future_effects"] = recovery_future_effects
    return result


def _e161_scientific_decision(effects: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    report_config = _mapping(config.get("report"), "config.report")
    threshold = float(_number(report_config.get("practical_reward_per_step"),
                              "config.report.practical_reward_per_step"))
    primary_horizon = int(effects["primary_horizon"])
    wanted = set(E161_PRIMARY_CONTRASTS)
    rows = [row for row in effects["effects"]
            if row["horizon"] == primary_horizon and row["pair_kind"] == "dynamics"
            and row["state"] == "late" and row["contrast"] in wanted]
    found = {str(row["contrast"]): row for row in rows}
    if set(found) != wanted or len(rows) != len(wanted):
        raise ValueError("E161 primary horizon is missing one of its two frozen contrasts")
    contrasts = {}
    for name in E161_PRIMARY_CONTRASTS:
        stat = found[name]["reward_per_step"]
        mean = float(stat["mean"])
        lower = float(stat["lower95_one_sided"])
        contrasts[name] = {
            "estimate": mean,
            "one_sided_95_lower": lower,
            "direction_pass": lower > 0.0,
            "practical_threshold": threshold,
            "point_estimate_at_or_above_threshold": mean >= threshold,
            "welch_degrees_freedom": stat["welch_degrees_freedom"],
            "two_sided_95_ci": stat["ci95"],
        }
    direction_pass = all(row["direction_pass"] for row in contrasts.values())
    threshold_pass = all(row["point_estimate_at_or_above_threshold"] for row in contrasts.values())
    if direction_pass and threshold_pass:
        status = "ADVANCE_TO_FULL_STREAM_VALIDATION"
    elif direction_pass:
        status = "SMALL_CONDITION_SPECIFIC_GAIN"
    elif all(row["estimate"] > 0.0 for row in contrasts.values()):
        status = "INCONCLUSIVE"
    else:
        status = "NOT_REPEATED"
    return {
        "status": status,
        "technical_audit_status": "PASS",
        "primary_horizon": primary_horizon,
        "primary_future_regimes": list(E161_PRIMARY_FUTURES),
        "balanced_in_primary_estimand": False,
        "independent_unit": "deployment seed; all sixteen configured seeds retained",
        "contrast_rule": "both one-sided 95% Welch lower bounds must exceed zero",
        "advancement_rule": "direction rule and both point estimates at or above 0.002 reward/step",
        "intersection_union_direction_pass": direction_pass,
        "advancement_threshold_pass": threshold_pass,
        "contrasts": contrasts,
    }


def _diagnostics(
    cases: Sequence[Mapping[str, Any]], branches: Sequence[Mapping[str, Any]],
    families: Mapping[int, str | None],
) -> dict[str, Any]:
    fit_rows = []
    for case in cases:
        prefix = _case_prefix(case, f"case {case['case_id']}")
        fit = _mapping(prefix.get("candidate_fit"), f"case {case['case_id']}.candidate_fit")
        behavior = _mapping(fit.get("behavioral"), f"case {case['case_id']}.candidate_fit.behavioral")
        fit_rows.append({
            "case_id": str(case["case_id"]), "seed": int(case["seed"]),
            "family": families.get(int(case["seed"])), "pair_kind": str(case["pair_kind"]),
            "state": str(case["state"]), "candidate_fit": _json_safe(fit),
            "precodec_fit_split_kl": {
                "train_mean_initial": behavior.get("train_kl_mean_init"),
                "train_mean_fitted": behavior.get("train_kl_mean_fit"),
                "train_by_owner_initial": behavior.get("train_kl_by_owner_init"),
                "train_by_owner_fitted": behavior.get("train_kl_by_owner_fit"),
                "holdout_mean_initial": behavior.get("holdout_kl_mean_init"),
                "holdout_mean_fitted": behavior.get("holdout_kl_mean_fit"),
                "holdout_by_owner_initial": behavior.get("holdout_kl_by_owner_init"),
                "holdout_by_owner_fitted": behavior.get("holdout_kl_by_owner_fit"),
            },
            "source_band_counts_by_owner": fit.get("band_counts_by_owner"),
            "candidate_norms": {"mean": fit.get("mean_candidate_norm"),
                                "behavioral": fit.get("behavioral_candidate_norm")},
            "fit_cpu_s": fit.get("fit_cpu_s"), "fit_wall_s": fit.get("fit_wall_s"),
            "source_buffer_bytes": fit.get("source_buffer_bytes"),
            "transient_candidate_bytes": fit.get("transient_candidate_bytes"),
            "optimizer_workspace_bytes_estimate": behavior.get("optimizer_workspace_bytes_estimate"),
            "temporary_array_bytes_estimate": behavior.get("temporary_array_bytes_estimate"),
        })
    codec_rows = []
    for branch in branches:
        action = _mapping(branch.get("action_install"), f"branch {branch.get('branch_id')}.action_install")
        codec_rows.append({
            "branch_id": str(branch["branch_id"]), "seed": int(branch["seed"]),
            "family": families.get(int(branch["seed"])), "pair_kind": str(branch["pair_kind"]),
            "state": str(branch["state"]), "future_regime": str(branch["future_regime"]),
            "replicate": int(branch["replicate"]), "arm": str(branch["arm"]),
            "candidate_kind": action.get("candidate_kind"), "operation": action.get("operation"),
            "new_source_teacher_kl_before_codec": _json_safe(action.get("new_source_teacher_kl_before_codec")),
            "new_source_teacher_kl_after_codec": _json_safe(action.get("new_source_teacher_kl_after_codec")),
            "pre_action_new_source_teacher_kl": _json_safe(action.get("pre_action_new_source_teacher_kl")),
            "candidate_norm": action.get("candidate_norm"),
            "pre_action_parent_norm": action.get("pre_action_parent_norm"),
            "prequantized_parent_norm": action.get("prequantized_parent_norm"),
            "postquantized_parent_norm": action.get("postquantized_parent_norm"),
            "candidate_source_effective_precodec_norm": float(np.linalg.norm(np.asarray(action.get("candidate_source_effective_precodec", []), dtype=np.float64))),
            "candidate_source_effective_postcodec_norm": action.get("candidate_source_effective_postcodec_norm"),
            "prequantized_behavior_drift": _json_safe(action.get("prequantized_behavior_drift")),
            "postquantized_behavior_drift": _json_safe(action.get("postquantized_behavior_drift")),
            "codec_max_coefficient_error": action.get("codec_max_coefficient_error"),
            "actor_payload_bytes": action.get("actor_payload_bytes"),
        })
    return {"candidate_fit_by_case": fit_rows, "branch_codec_kl_by_arm": codec_rows}


def _costs(cases: Sequence[Mapping[str, Any]], branches: Sequence[Mapping[str, Any]], recount: Mapping[str, Any],
           *, arms: Sequence[str] = ARMS, include_evictions: bool = False) -> dict[str, Any]:
    unique_cases = []
    for case in cases:
        prefix = _case_prefix(case, f"case {case['case_id']}")
        fit = _mapping(prefix.get("candidate_fit"), f"case {case['case_id']}.candidate_fit")
        costs = _mapping(case.get("costs"), f"case {case['case_id']}.costs")
        unique_cases.append({
            "case_id": str(case["case_id"]), "seed": int(case["seed"]),
            "pair_kind": str(case["pair_kind"]), "state": str(case["state"]),
            "prefix_steps": int(_number(prefix.get("steps"), f"case {case['case_id']}.prefix.steps", integer=True)),
            "prefix_cpu_s_including_fit": float(_number(costs.get("prefix_cpu_s"), f"case {case['case_id']}.prefix_cpu_s")),
            "candidate_fit_cpu_s": float(_number(costs.get("candidate_fit_cpu_s"), f"case {case['case_id']}.candidate_fit_cpu_s")),
            "candidate_fit_wall_s": float(_number(fit.get("fit_wall_s"), f"case {case['case_id']}.fit_wall_s")),
            "branch_cpu_s": float(_number(costs.get("branch_cpu_s"), f"case {case['case_id']}.branch_cpu_s")),
            "case_cpu_s": float(_number(costs.get("case_cpu_s"), f"case {case['case_id']}.case_cpu_s")),
            "evidence_buffer_bytes": int(_number(fit.get("source_buffer_bytes"), f"case {case['case_id']}.source_buffer_bytes", integer=True)),
            "transient_candidate_bytes": int(_number(fit.get("transient_candidate_bytes"), f"case {case['case_id']}.transient_candidate_bytes", integer=True)),
            "package_bytes": int(_number(costs.get("package_bytes"), f"case {case['case_id']}.package_bytes", integer=True)),
            "actor_container_bytes": int(_number(costs.get("actor_container_bytes"), f"case {case['case_id']}.actor_container_bytes", integer=True)),
            "package_budget_bytes": int(_number(costs.get("package_budget_bytes"), f"case {case['case_id']}.package_budget_bytes", integer=True)),
        })
    branch_costs = []
    by_arm: dict[str, list[float]] = {str(arm): [] for arm in arms}
    for branch in branches:
        cost = _mapping(branch.get("costs"), f"branch {branch['branch_id']}.costs")
        cpu = float(_number(cost.get("branch_cpu_s"), f"branch {branch['branch_id']}.branch_cpu_s"))
        by_arm[str(branch["arm"])].append(cpu)
        branch_record = {
            "branch_id": str(branch["branch_id"]), "seed": int(branch["seed"]),
            "pair_kind": str(branch["pair_kind"]), "state": str(branch["state"]),
            "future_regime": str(branch["future_regime"]), "replicate": int(branch["replicate"]),
            "arm": str(branch["arm"]), "branch_cpu_s": cpu,
            "setup_cpu_s": cost.get("setup_cpu_s"), "install_cpu_s": cost.get("install_cpu_s"),
            "simulation_cpu_s": cost.get("simulation_cpu_s"), "serialization_cpu_s": cost.get("serialization_cpu_s"),
            "package_bytes_final": branch.get("capacity", {}).get("package_bytes"),
            "package_bytes_after_action": branch.get("capacity_after_action", {}).get("package_bytes"),
            "actor_container_bytes": branch.get("capacity", {}).get("actor_container_bytes"),
            "package_budget_bytes": branch.get("capacity", {}).get("package_budget_bytes"),
        }
        if include_evictions:
            counts_delta = _mapping(branch.get("counts_delta"), f"branch {branch['branch_id']}.counts_delta")
            branch_record["slot_evictions"] = int(_number(
                counts_delta.get("slot_evictions"), f"branch {branch['branch_id']}.counts_delta.slot_evictions",
                integer=True,
            ))
        branch_costs.append(branch_record)
    prefix_cpu = sum(row["prefix_cpu_s_including_fit"] for row in unique_cases)
    fit_cpu = sum(row["candidate_fit_cpu_s"] for row in unique_cases)
    branch_cpu = sum(row["branch_cpu_s"] for row in branch_costs)
    return {
        "prefix_cpu_s_including_fit": prefix_cpu,
        "candidate_fit_cpu_s_component_of_prefix": fit_cpu,
        "prefix_cpu_s_excluding_fit_estimate": max(0.0, prefix_cpu - fit_cpu),
        "branch_cpu_s": branch_cpu,
        "measured_unique_prefix_plus_branch_cpu_s": prefix_cpu + branch_cpu,
        "branch_cpu_s_by_arm": {arm: {"branch_count": len(values), "mean": float(np.mean(values)) if values else None,
                                       "sum": float(sum(values))} for arm, values in by_arm.items()},
        "cases": unique_cases,
        "branches": branch_costs,
        "buffer_bytes_are_charged_in_each_package": True,
        "environment_step_recount": {
            "prefix": recount["prefix_environment_steps"], "branch": recount["branch_environment_steps"],
            "total": recount["total_environment_steps"],
        },
        "byte_budgets": {"actor_container_bytes": 89, "package_budget_bytes": 262144,
                         "verified_all_prefix_and_branch_packages": True},
    }


def _source_hashes(run_root: Path, summary: Mapping[str, Any], summary_path: Path, summary_sha: str) -> dict[str, Any]:
    manifest = summary.get("source_manifest")
    if not isinstance(manifest, Mapping):
        manifest_path = run_root / "source_snapshot" / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path, "source snapshot manifest")
        else:
            manifest = {}
    manifest = dict(manifest)
    excluded = manifest.get("excluded_local_instructions")
    if not isinstance(excluded, list) or not {"agent.md", "agents.md"} <= {str(value).casefold() for value in excluded}:
        raise ValueError("source snapshot does not record exclusion of machine-local agent instructions")
    source_files = manifest.get("sources", {})
    verified_files: dict[str, Any] = {}
    if isinstance(source_files, Mapping):
        for relative, record_value in source_files.items():
            if Path(str(relative)).name.casefold() in {"agent.md", "agents.md"}:
                raise ValueError("source snapshot manifest includes machine-local agent instructions")
            record = _mapping(record_value, f"source_manifest.sources.{relative}")
            path = run_root / "source_snapshot" / str(relative)
            digest = record.get("sha256")
            if not path.is_file():
                raise ValueError(f"source snapshot file is missing: {relative}")
            actual = _file_hash(path)
            actual_bytes = int(path.stat().st_size)
            if actual != digest or actual_bytes != int(_number(record.get("bytes"), f"source_manifest.sources.{relative}.bytes", integer=True)):
                raise ValueError(f"source snapshot hash or byte count mismatch: {relative}")
            verified_files[str(relative)] = {"sha256": actual, "bytes": actual_bytes, "verified": True}
    runner_hash = None
    fit_module_hash = None
    for relative, record_value in source_files.items() if isinstance(source_files, Mapping) else []:
        record = record_value if isinstance(record_value, Mapping) else {}
        if str(relative).endswith("scripts/run_retention_candidate.py"):
            runner_hash = record.get("sha256")
        if str(relative).endswith("src/online_rl/retention_behavioral_candidate.py"):
            fit_module_hash = record.get("sha256")
    return {
        "summary_sha256": summary_sha,
        "config_canonical_sha256": _canonical_hash(summary.get("config", {})),
        "source_manifest_canonical_sha256": _canonical_hash(manifest),
        "source_manifest": _json_safe(manifest),
        "source_files_verified": verified_files,
        "runner_sha256": runner_hash,
        "behavioral_fit_module_sha256": fit_module_hash,
        "reporter_sha256": _file_hash(Path(__file__).resolve()),
        "summary_path": str(summary_path),
    }


def _audit_record_history(
    run_root: Path, phase: str, expected: Mapping[str, int], source_manifest_sha256: str,
) -> list[dict[str, Any]]:
    audit_root = run_root.parent / "retention_candidate_audit"
    records = sorted(audit_root.glob(f"{phase}*.json"), key=lambda path: (path.stat().st_mtime_ns, path.name))
    if not records:
        raise ValueError(f"no external {phase} audit record is available under {audit_root}")
    output: list[dict[str, Any]] = []
    for index, path in enumerate(records):
        record = _read_json(path, f"external {phase} audit record")
        if record.get("format") != "retention_candidate_audit_v1" or record.get("phase") != phase:
            raise ValueError(f"external audit record has an unexpected format or phase: {path.name}")
        cohort = record.get("cohort")
        if not isinstance(cohort, str) or Path(cohort).resolve() != run_root.resolve():
            raise ValueError(f"external audit record points at a different run: {path.name}")
        if record.get("source_manifest_sha256") != source_manifest_sha256:
            raise ValueError(f"external audit record uses a different source manifest: {path.name}")
        coverage = _mapping(record.get("coverage"), f"{path.name}.coverage")
        expected_coverage = {
            "cases": expected["cases"],
            "branches": expected["branches"],
            "expected_cases": expected["cases"],
            "expected_branches": expected["branches"],
        }
        for field, expected_count in expected_coverage.items():
            if int(_number(coverage.get(field), f"{path.name}.coverage.{field}", integer=True)) != int(expected_count):
                raise ValueError(f"external audit coverage differs from the frozen grid: {path.name}")
        raw_counts = _mapping(record.get("raw_counts"), f"{path.name}.raw_counts")
        if int(_number(raw_counts.get("environment_steps_total"), f"{path.name}.raw_counts.environment_steps_total", integer=True)) != int(expected["total_steps"]):
            raise ValueError(f"external audit step count differs from the frozen grid: {path.name}")
        output.append({
            "path": path.relative_to(run_root.parent.resolve()).as_posix(),
            "sha256": _file_hash(path),
            "bytes": int(path.stat().st_size),
            "format": record["format"],
            "phase": record["phase"],
            "declared_status": record.get("status"),
            "coverage": _json_safe(coverage),
            "raw_counts": _json_safe(raw_counts),
            "integrity_errors": _json_safe(record.get("integrity_errors", [])),
            "diagnostic_warnings": _json_safe(record.get("diagnostic_warnings", [])),
            "superseded_by_later_record": index < len(records) - 1,
            "review_context": (
                "This earlier audit output is preserved as written. The parent reviewer reported auditor contract/replay defects and required a repaired follow-up; this marks an audit-tool revision, not an experiment failure."
                if index < len(records) - 1 else None
            ),
        })
    if output[-1]["declared_status"] != "passed":
        raise ValueError(f"latest external {phase} audit record does not declare a pass: {output[-1]['path']}")
    return output


def _mean_by(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str], value: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        metric = _number(row.get(value), f"diagnostic.{value}", nullable=True)
        if metric is None:
            continue
        key = tuple(row.get(field) for field in key_fields)
        buckets.setdefault(key, []).append(float(metric))
    return [{**dict(zip(key_fields, key)), "mean": float(np.mean(values)), "record_count": len(values)}
            for key, values in sorted(buckets.items(), key=lambda item: tuple(str(v) for v in item[0]))]


def _render_figure(output: Path, analysis: Mapping[str, Any]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "legend.fontsize": 8, "figure.titlesize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    ax_reward, ax_kl, ax_cost = axes
    primary = [row for row in analysis["effects"] if row["horizon"] == analysis["primary_horizon"]
               and row["contrast"] in ("replace_behavioral-minus-replace_mean", "replace_behavioral-minus-keep")]
    labels, values, lows, highs, colors = [], [], [], [], []
    for row in primary:
        stat = row["reward_per_step"]
        if stat["mean"] is None:
            continue
        labels.append(f"{row['pair_kind']} / {row['state']}\n{row['contrast'].replace('-', '−')}")
        values.append(float(stat["mean"]))
        ci = stat.get("ci95")
        if isinstance(ci, list) and len(ci) == 2:
            lows.append(float(stat["mean"]) - float(ci[0]))
            highs.append(float(ci[1]) - float(stat["mean"]))
        else:
            lows.append(0.0)
            highs.append(0.0)
        colors.append("#0072B2" if row["contrast"] == "replace_behavioral-minus-replace_mean" else "#D55E00")
    y = np.arange(len(values))
    if values:
        ax_reward.errorbar(values, y, xerr=[lows, highs], fmt="o", color="#0072B2", ecolor="#555555",
                           capsize=2, markersize=4, linewidth=1)
        for yi, color in zip(y, colors):
            ax_reward.scatter(values[yi], yi, color=color, s=25, zorder=3)
        ax_reward.set_yticks(y, labels)
        ax_reward.invert_yaxis()
    ax_reward.axvline(0, color="#333333", linewidth=0.8)
    ax_reward.set_xlabel("Reward / step difference")
    ax_reward.set_title(f"Paired contrasts at {analysis['primary_horizon']:,} steps")
    ax_reward.grid(axis="x", alpha=0.25)

    fit_rows = analysis["diagnostics"]["candidate_fit_by_case"]
    fit_values = [("Train initial", "train_mean_initial"), ("Train fitted", "train_mean_fitted"),
                  ("Holdout initial", "holdout_mean_initial"), ("Holdout fitted", "holdout_mean_fitted")]
    fit_means = []
    fit_labels = []
    for label, metric in fit_values:
        values_for_metric = [float(row["precodec_fit_split_kl"][metric]) for row in fit_rows
                             if row["precodec_fit_split_kl"].get(metric) is not None]
        if values_for_metric:
            fit_labels.append(label)
            fit_means.append(float(np.mean(values_for_metric)))
    if fit_means:
        bars = ax_kl.bar(np.arange(len(fit_means)), fit_means,
                         color=("#999999", "#0072B2", "#BBBBBB", "#D55E00")[:len(fit_means)])
        ax_kl.set_xticks(np.arange(len(fit_means)), fit_labels, rotation=25, ha="right")
        ax_kl.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)
    ax_kl.set_ylabel("Forward KL")
    ax_kl.set_title("Candidate fit split diagnostics\n(float32 candidate, before action codec)")
    ax_kl.grid(axis="y", alpha=0.25)

    cost = analysis["costs"]
    values = [cost["prefix_cpu_s_excluding_fit_estimate"], cost["candidate_fit_cpu_s_component_of_prefix"], cost["branch_cpu_s"]]
    labels = ["Prefix\nexcluding fit", "Candidate\nfit component", "Branch\ncontinuations"]
    bars = ax_cost.bar(np.arange(3), values, color=("#56B4E9", "#CC79A7", "#009E73"))
    ax_cost.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax_cost.set_xticks(np.arange(3), labels)
    ax_cost.set_ylabel("Measured process CPU seconds")
    ax_cost.set_title("Measured CPU by actual work")
    ax_cost.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Retention candidate diagnostic — {analysis['phase']}")
    paths: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = output / f"figure.{suffix}"
        options: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            options["dpi"] = 300
        fig.savefig(path, **options)
        paths[suffix] = str(path.resolve())
    plt.close(fig)
    return paths


def _format(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _interval(stat: Mapping[str, Any]) -> str:
    interval = stat.get("ci95")
    if not isinstance(interval, list) or len(interval) != 2:
        return "not estimated"
    return f"[{_format(interval[0])}, {_format(interval[1])}]"


def _markdown_e161(analysis: Mapping[str, Any]) -> str:
    decision = analysis["scientific_decision"]
    lines = [
        "# E161 retention candidate confirmation report", "",
        f"Role: **{analysis['analysis_role']}**. Scientific status: **{decision['status']}**.",
        f"Technical audit: **{analysis['audit']['passed']}**. The technical integrity result is reported separately from the scientific decision.", "",
        "This offline report reads the saved cohort, step and rollout ledgers, checkpoint metadata, and bounded evidence artifacts. It makes no environment calls.", "",
        "## Design and main decision", "",
        f"- Independent units: {len(analysis['seeds'])} deployment seeds, all retained; A/B family labels are balanced.",
        f"- Cases: {analysis['case_count']}; branches: {analysis['branch_count']}; measured environment interactions: {analysis['costs_recount']['total_environment_steps']:,}.",
        f"- Main futures: {', '.join(decision['primary_future_regimes'])}. Balanced is a secondary boundary and does not enter the main estimate.",
        f"- Main estimand: average the two paired replicas within each future regime, then give old_major and new_major equal weight within seed.",
        f"- IUT direction passes: **{decision['intersection_union_direction_pass']}**; practical point threshold passes for both contrasts: **{decision['advancement_threshold_pass']}**.",
        f"- Advancement requires both one-sided 95% Welch lower bounds above zero and both point estimates at least {decision['contrasts'][E161_PRIMARY_CONTRASTS[0]]['practical_threshold']:.3f} reward/step.",
        "- Welch inference uses equal A/B family means and the Welch-Satterthwaite degrees of freedom. The stratified seed bootstrap is a sensitivity analysis only.", "",
        "| primary contrast | estimate | one-sided 95% lower | two-sided 95% Welch interval | direction | point threshold | bootstrap sensitivity interval |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for name in E161_PRIMARY_CONTRASTS:
        row = decision["contrasts"][name]
        stat = next(item["reward_per_step"] for item in analysis["effects"]
                    if item["horizon"] == analysis["primary_horizon"] and item["contrast"] == name)
        boot = stat.get("bootstrap_ci95_sensitivity")
        boot_text = "not estimated" if not isinstance(boot, list) else f"[{_format(boot[0])}, {_format(boot[1])}]"
        lines.append(
            f"| {name} | {_format(row['estimate'])} | {_format(row['one_sided_95_lower'])} | "
            f"[{_format(row['two_sided_95_ci'][0])}, {_format(row['two_sided_95_ci'][1])}] | "
            f"{row['direction_pass']} | {row['point_estimate_at_or_above_threshold']} | {boot_text} |"
        )
    lines.extend(["", "Every configured seed remains in `analysis.json`; no old E130 result is pooled and no seed is excluded based on its significance or reward.",
                  "", "## Per-regime reward contrasts", "",
                  "These per-regime estimates include all three configured futures. `balanced` remains secondary.", "",
                  "| future | contrast | reward / step | two-sided 95% Welch interval | seeds |",
                  "|---|---|---:|---|---:|"])
    for row in analysis["future_scoped_effects"]:
        if row["horizon"] == analysis["primary_horizon"]:
            stat = row["reward_per_step"]
            lines.append(f"| {row['future_regime']} | {row['contrast']} | {_format(stat['mean'])} | {_interval(stat)} | {stat['seed_count']} |")
    lines.extend(["", "## Nested window and first-256-step recovery", "",
                  "The 1,536-step window and 6,144-step primary horizon are both retained by seed and regime. First-256-step recovery is reported per regime and in `analysis.json`.", "",
                  "| horizon | contrast | reward / step | two-sided 95% Welch interval |",
                  "|---:|---|---:|---|"])
    for row in analysis["effects"]:
        if row["horizon"] != analysis["primary_horizon"]:
            stat = row["reward_per_step"]
            lines.append(f"| {row['horizon']:,} | {row['contrast']} | {_format(stat['mean'])} | {_interval(stat)} |")
    lines.extend(["", "| recovery population | future | contrast | reward / step |", "|---|---|---|---:|"])
    for row in analysis.get("effects_summary", {}).get("recovery_future_effects", []):
        stat = row["stat"]
        lines.append(f"| {row['population']} | {row['future_regime']} | {row['contrast']} | {_format(stat['mean'])} |")
    lines.extend(["", "## Fit, codec, eviction, update, byte, and cost evidence", "",
                  "Training and holdout KL, source band counts, codec drift, source buffer and package bytes, slot evictions, rollout updates, and measured CPU costs are preserved in `analysis.json` by case, seed, future, replica, arm, and horizon where applicable.",
                  f"- Raw CSV ledgers checked: {analysis['costs_recount']['raw_ledger_count']}; rollouts: {analysis['costs_recount']['total_rollouts_trained']:,}; minibatch updates: {analysis['costs_recount']['total_minibatch_updates']:,}.",
                  f"- Evidence sidecar copies: {analysis['costs_recount']['evidence_buffer_bytes']['all_persisted_prefix_and_branch_package_copies']:,} bytes; complete package limit: {analysis['costs']['byte_budgets']['package_budget_bytes']:,} bytes.",
                  f"- Measured campaign CPU: {analysis['costs']['measured_unique_prefix_plus_branch_cpu_s']:.3f} seconds.",
                  "- Fit KL is diagnostic; it does not replace reward as the primary outcome.", "",
                  "## Figure", "", "![E161 retention candidate analysis](figure.png)", "",
                  "## Provenance", "",
                  f"Summary SHA-256: {analysis['input_hashes']['summary_sha256']}  ",
                  f"Frozen config canonical SHA-256: {analysis['input_hashes']['config_canonical_sha256']}  ",
                  f"Source manifest canonical SHA-256: {analysis['input_hashes']['source_manifest_canonical_sha256']}  ",
                  f"Runner SHA-256: {analysis['input_hashes']['runner_sha256'] or 'unavailable'}  ",
                  f"Reporter SHA-256: {analysis['input_hashes']['reporter_sha256']}", ""])
    return "\n".join(lines)


def _markdown(analysis: Mapping[str, Any]) -> str:
    if "e161" in str(analysis.get("analysis_role", "")).casefold():
        return _markdown_e161(analysis)
    primary = analysis["primary_horizon"]
    uncertainty = (
        "Six deployment seeds are the independent units. Future regimes and replicas were paired within each seed, then averaged before the old-family-stratified bootstrap."
        if analysis["phase"] == "main" else
        "This one-seed pilot has no uncertainty interval. The main report uses six deployment seeds, averaging paired futures and replicas within seed before old-family-stratified bootstrap."
    )
    lines = [
        "# Retention candidate report", "",
        f"Phase: **{analysis['phase']}**. Role: **{analysis['analysis_role']}**. This is an exploratory fixed-candidate, one-write diagnostic.", "",
        "The reporter reads saved summaries and hashed ledgers only. It makes no environment calls, does not adapt candidates or analysis choices from reward outcomes, and does not send results back to a learner.", "",
        "## Design and audit", "",
        f"- Prefixes: {analysis['case_count']}; branches: {analysis['branch_count']}; actual environment steps: {analysis['costs_recount']['total_environment_steps']:,}.",
        f"- Common prefix steps: {analysis['costs_recount']['prefix_environment_steps']:,}; branch continuation steps: {analysis['costs_recount']['branch_environment_steps']:,}.",
        f"- Recounted training: {analysis['costs_recount']['total_rollouts_trained']:,} rollouts and {analysis['costs_recount']['total_minibatch_updates']:,} minibatch updates.",
        f"- SHA-256 checked raw CSV ledgers: {analysis['costs_recount']['raw_ledger_count']}; complete grid, restore, pairing, reward, cost, and capacity checks passed: **{analysis['audit']['passed']}**.",
        f"- Actor container: {analysis['costs']['byte_budgets']['actor_container_bytes']} bytes; complete package cap: {analysis['costs']['byte_budgets']['package_budget_bytes']:,} bytes.",
        "- Early and late are joint state-coverage conditions. They do not identify a causal effect of age, fit volume, or maturity alone.",
        f"- {uncertainty}",
        "- All five arms, including keep, remain in the report. A lower fit KL by itself is not evidence of improved return.", "",
        f"## Primary reward contrasts at {primary:,} steps", "",
        "Every arm contrast is paired within prefix, future regime, and replicate. Future regimes and replicates are averaged within seed before the descriptive bootstrap.", "",
        "| relation | state | contrast | seeds | reward / step difference | 95% descriptive interval |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in analysis["effects"]:
        if row["horizon"] != primary:
            continue
        stat = row["reward_per_step"]
        lines.append(f"| {row['pair_kind']} | {row['state']} | {row['contrast']} | {stat['seed_count']} | {_format(stat['mean'])} | {_interval(stat)} |")
    replica_count = len(analysis["replicates"])
    replica_word = "replica" if replica_count == 1 else "replicas"
    lines.extend(["", "## Future-scoped contrasts", "",
                  f"These estimates keep the future regime fixed and average the {replica_count} configured {replica_word} within seed.", "",
                  "| relation | state | future | contrast | reward / step difference | 95% descriptive interval | seeds |",
                  "|---|---|---|---|---:|---|---:|"])
    for row in analysis["future_scoped_effects"]:
        if row["horizon"] != primary:
            continue
        stat = row["reward_per_step"]
        lines.append(f"| {row['pair_kind']} | {row['state']} | {row['future_regime']} | {row['contrast']} | {_format(stat['mean'])} | {_interval(stat)} | {stat['seed_count']} |")
    lines.extend(["", "## Candidate by operation interaction", "",
                  "The interaction is (merge behavioral − merge mean) − (replace behavioral − replace mean). It is computed within each seed after averaging paired futures and replicas.", "",
                  "| relation | state | horizon | interaction / step | 95% descriptive interval | seeds |",
                  "|---|---|---:|---:|---|---:|"])
    for row in analysis["candidate_by_operation_interaction"]:
        if row["horizon"] == primary:
            stat = row["reward_per_step"]
            lines.append(f"| {row['pair_kind']} | {row['state']} | {row['horizon']:,} | {_format(stat['mean'])} | {_interval(stat)} | {stat['seed_count']} |")
    lines.extend(["", "## First-256-step recovery", "",
                  "Recovery is reported as mean return per 256-step bout, converted to reward per step. The allocated-only view includes bouts that allocated a slot; the all-bouts view includes every future bout. The same within-seed pairing is used.", "",
                  "| population | relation | state | contrast | reward / step difference | 95% descriptive interval | seeds |",
                  "|---|---|---|---|---:|---|---:|"])
    for row in analysis["effects_summary"]["recovery_effects"]:
        stat = row["stat"]
        lines.append(f"| {row['population']} | {row['pair_kind']} | {row['state']} | {row['contrast']} | {_format(stat['mean'])} | {_interval(stat)} | {stat['seed_count']} |")
    lines.extend(["", "## Candidate fit and codec diagnostics", "",
                  "The behavioral candidate was fit once per prefix from the bounded source buffer. Training KL drove candidate selection; holdout KL is diagnostic only. Branch-specific KL after the action is listed separately from the fit-split KL.", "",
                  "| relation | state | seed | train KL initial | train KL fitted | holdout KL initial | holdout KL fitted | band counts owner 0 / 1 | mean norm | behavioral norm | fit CPU s | buffer bytes |",
                  "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"])
    for row in analysis["diagnostics"]["candidate_fit_by_case"]:
        kl, norms = row["precodec_fit_split_kl"], row["candidate_norms"]
        bands = json.dumps(row["source_band_counts_by_owner"], separators=(",", ":"))
        lines.append(f"| {row['pair_kind']} | {row['state']} | {row['seed']} | {_format(kl['train_mean_initial'])} | {_format(kl['train_mean_fitted'])} | {_format(kl['holdout_mean_initial'])} | {_format(kl['holdout_mean_fitted'])} | `{bands}` | {_format(norms['mean'])} | {_format(norms['behavioral'])} | {_format(row['fit_cpu_s'])} | {row['source_buffer_bytes']:,} |")
    lines.extend(["", "Branch-level before/after int8-codec source-teacher KL, owner-specific counts, action norms, and behavior drift are preserved in `analysis.json`.", "",
                  "## Actual costs and capacity", "",
                  f"The common prefix cost was {analysis['costs']['prefix_cpu_s_including_fit']:.3f} CPU seconds including fit; the candidate fit component was {analysis['costs']['candidate_fit_cpu_s_component_of_prefix']:.3f} seconds and is not added twice. Branch setup, action install, continuation, and serialization used {analysis['costs']['branch_cpu_s']:.3f} CPU seconds. Unique measured prefix plus branch time was {analysis['costs']['measured_unique_prefix_plus_branch_cpu_s']:.3f} seconds.", "",
                  "| cost component | CPU seconds |", "|---|---:|"])
    for name, value in (("Prefix including fit", analysis["costs"]["prefix_cpu_s_including_fit"]),
                        ("Candidate fit component of prefix", analysis["costs"]["candidate_fit_cpu_s_component_of_prefix"]),
                        ("Prefix excluding fit estimate", analysis["costs"]["prefix_cpu_s_excluding_fit_estimate"]),
                        ("Branch continuations", analysis["costs"]["branch_cpu_s"]),
                        ("Unique prefix plus branch", analysis["costs"]["measured_unique_prefix_plus_branch_cpu_s"])):
        lines.append(f"| {name} | {value:.3f} |")
    evidence_bytes = analysis["costs_recount"]["evidence_buffer_bytes"]
    lines.extend(["", f"Every checked prefix and branch package used the {analysis['costs']['byte_budgets']['actor_container_bytes']}-byte actor container and stayed within the {analysis['costs']['byte_budgets']['package_budget_bytes']:,}-byte complete-package cap. Source-buffer, transient-candidate, package, and branch CPU records are retained in `analysis.json`.",
                  f"Evidence sidecars used {evidence_bytes['prefix_sidecars_once']:,} bytes in prefixes, {evidence_bytes['final_branch_packages']:,} bytes in final branch packages, and {evidence_bytes['post_action_packages']:,} bytes in post-action packages. Those measured copies are included in the corresponding full-package caps.", "",
                  "## Figure", "", "![Retention candidate report](figure.png)", "",
                  "## Interpretation limits", "",
                  "This is a known-identity, fixed-candidate, one-write comparison. It does not test a deployed selector, repeated learned organizer, unknown-context discovery, tree or DAG comparisons, or C2. The late state changes online history and state coverage jointly. Lower fitted KL does not establish a reward gain. These results do not support claims of SOTA performance or organizer success.", "",
                  "## Provenance", "",
                  f"Summary SHA-256: {analysis['input_hashes']['summary_sha256']}  ",
                  f"Frozen config canonical SHA-256: {analysis['input_hashes']['config_canonical_sha256']}  ",
                  f"Source manifest canonical SHA-256: {analysis['input_hashes']['source_manifest_canonical_sha256']}  ",
                  f"Runner SHA-256: {analysis['input_hashes']['runner_sha256'] or 'unavailable'}  ",
                  f"Behavioral fit module SHA-256: {analysis['input_hashes']['behavioral_fit_module_sha256'] or 'unavailable'}  ",
                  f"Reporter SHA-256: {analysis['input_hashes']['reporter_sha256']}", "",
                  "## External audit history", "",
                  "Audit files are preserved by SHA-256, including superseded records. Their declared statuses and raw counts are shown verbatim; the report's own ledger checks are recorded separately.", "",
                  "| file | declared status | cases / branches | raw steps | superseded | SHA-256 |",
                  "|---|---|---:|---:|---|---|"])
    for record in analysis["external_audit_records"]:
        coverage = record["coverage"]
        raw_counts = record["raw_counts"]
        lines.append(
            f"| `{record['path']}` | {record['declared_status']} | {coverage['cases']} / {coverage['branches']} | "
            f"{raw_counts['environment_steps_total']:,} | {'yes' if record['superseded_by_later_record'] else 'no'} | `{record['sha256']}` |"
        )
        if record.get("review_context"):
            lines.extend(["", record["review_context"]])
    lines.extend(["", "## Software", "",
                  "Static figures use Matplotlib's object-oriented interface and export PNG, PDF, and SVG. The report preserves the exact reporter source beside its hash.", ""])
    return "\n".join(lines)


def analyze(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    summary, summary_path, summary_sha = _load_summary(input_path)
    run_root = summary_path.parent.resolve()
    if not isinstance(summary.get("config"), Mapping):
        summary["config"] = _load_config(run_root, summary)
    design, cases, branches, families = _design(summary)
    observations, recount, recovery = _audit_run(cases, branches, run_root, design, families)
    campaign_cost_audit = _audit_campaign_costs(summary, design, recount, cases, branches)
    effects = _compute_effects(observations, recovery, families, design["active"], design["config"], design["phase"])
    diagnostics = _diagnostics(cases, branches, families)
    is_e161 = _is_e161(design["config"])
    costs = _costs(cases, branches, recount, arms=design["active"]["arms"], include_evictions=is_e161)
    source_hashes = _source_hashes(run_root, summary, summary_path, summary_sha)
    active = design["active"]
    expected = design["expected_counts"]
    manifest_path = run_root / "source_snapshot" / "manifest.json"
    audit_records = _audit_record_history(run_root, design["phase"], expected, _file_hash(manifest_path))
    analysis: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "analysis_role": str(design["config"].get("analysis_role", "exploratory_candidate_content_and_state_coverage")),
        "phase": design["phase"], "seeds": list(active["seeds"]),
        "pair_kinds": list(active["pair_kinds"]), "states": list(active["states"]),
        "future_regimes": list(active["future_regimes"]), "replicates": list(active["replicates"]),
        "arms": list(active["arms"]), "horizons": list(active["horizons"]),
        "primary_horizon": effects["primary_horizon"],
        "case_count": len(cases), "branch_count": len(branches),
        "seed_families": {str(seed): families.get(int(seed)) for seed in active["seeds"]},
        "statistical_unit": "deployment seed after paired future regimes and replicas are averaged within seed",
        "effects": effects["effects"],
        "future_scoped_effects": effects["future_scoped_effects"],
        "candidate_by_operation_interaction": effects["candidate_by_operation_interaction"],
        "future_prevalence_interaction": effects["future_prevalence_interaction"],
        "effects_summary": {"recovery_effects": effects["recovery_effects"]},
        "branch_horizon_observations": observations,
        "recovery_branch_rows": recovery,
        "diagnostics": diagnostics,
        "costs": costs,
        "costs_recount": recount,
        "audit": {
            "passed": True, "case_and_branch_grids_complete": True,
            "semantic_pairing_verified": True, "branches_not_independent_units": True,
            "future_schedule_matched_across_arms": True,
            "common_prefix_restore_hashes_verified": True,
            "step_and_reward_ledgers_recounted": True,
            "rollout_and_update_ledgers_recounted": True,
            "source_fit_and_band_coverage_verified": True,
            "late_old_group_only_write_rule_verified": True,
            "exact_actor_and_full_package_caps_verified": True,
            "evidence_sidecar_provenance_and_package_copies_verified": True,
            "runner_aggregate_costs_reconciled": campaign_cost_audit["passed"],
            "external_audit_record_history_captured": True,
            "actual_steps_match_frozen_design": recount["total_environment_steps"] == expected["total_steps"],
        },
        "campaign_cost_audit": campaign_cost_audit,
        "expected_counts": expected,
        "interpretation_limits": [
            ("The confirmation compares three fixed arms under known identity and one saved write; it is not a deployed selector."
             if is_e161 else "The diagnostic compares five fixed arms under known identity and one saved write; it is not a deployed selector."),
            ("The result applies to the frozen dynamics / late condition and does not identify a causal effect of age or fit volume."
             if is_e161 else "The early and late conditions jointly vary online history and state coverage, so their contrast is not an isolated age or fit-volume effect."),
            "Fitted KL is a diagnostic and does not imply reward improvement.",
            "Future regimes and replicas are paired within seed; branches are not independent statistical units.",
            "No outcomes were used to adapt the candidate, analysis choices, or future conditions.",
            ("Balanced is a secondary boundary; the primary estimand equally averages only old_major and new_major within seed."
             if is_e161 else "The run does not establish SOTA performance, a new organizer success, unknown-context discovery, repeated organizer learning, or tree/DAG superiority."),
        ],
        "input_hashes": source_hashes,
        "external_audit_records": audit_records,
        "reporter_sha256": source_hashes["reporter_sha256"],
        "runner_costs": _json_safe(summary.get("costs", {})),
    }
    if is_e161:
        primary_futures = list(effects["primary_future_regimes"])
        analysis["primary_future_regimes"] = primary_futures
        analysis["secondary_future_regimes"] = [future for future in design["active"]["future_regimes"]
                                                 if future not in primary_futures]
        analysis["effects_summary"]["recovery_future_effects"] = effects["recovery_future_effects"]
        analysis["scientific_decision"] = _e161_scientific_decision(effects, design["config"])
        analysis["e161_statistical_plan"] = {
            "independent_unit": "deployment seed",
            "family_weighting": "equal mean of A and B",
            "main_futures": primary_futures,
            "secondary_futures": analysis["secondary_future_regimes"],
            "future_replicate_aggregation": "mean paired difference over replicas within regime, then equal mean over main regimes",
            "primary_contrasts": list(E161_PRIMARY_CONTRASTS),
            "historical_results_included": False,
            "significance_based_seed_exclusions": False,
        }
    output = Path(output_path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"preserve existing report output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    analysis["figure"] = _render_figure(output, analysis)
    reporter_copy = output / "reporter_sources" / Path(__file__).name
    reporter_copy.parent.mkdir(parents=True, exist_ok=True)
    reporter_copy.write_bytes(Path(__file__).resolve().read_bytes())
    if _file_hash(reporter_copy) != source_hashes["reporter_sha256"]:
        raise RuntimeError("reporter source changed while preserving its exact executed version")
    analysis["reporter_source_path"] = str(reporter_copy.resolve())
    (output / "analysis.json").write_text(
        json.dumps(_json_safe(analysis), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_markdown(analysis), encoding="utf-8")
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed pilot/main run directory or summary.json")
    parser.add_argument("--output", type=Path, required=True, help="new or empty report directory")
    args = parser.parse_args(argv)
    analysis = analyze(args.input, args.output)
    print(json.dumps({
        "phase": analysis["phase"], "cases": analysis["case_count"],
        "branches": analysis["branch_count"], "steps": analysis["costs_recount"]["total_environment_steps"],
        "audit_passed": analysis["audit"]["passed"], "output": str(Path(args.output).resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

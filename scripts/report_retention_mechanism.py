"""Offline audit and report for the frozen retention mechanism diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SUMMARY_FORMAT = "retention_mechanism_summary_v1"
REPORT_FORMAT = "retention_mechanism_report_v1"
ACTIONS = ("keep", "replace", "mean", "none")
MAIN_SEEDS = (25101, 25102, 25103, 25104, 25107, 25108)
PILOT_SEED = 25089
MAIN_COUNTS = {"cases": 24, "branches": 384, "prefix_steps": 172032,
               "branch_steps": 4718592, "total_steps": 4890624}
PILOT_COUNTS = {"cases": 1, "branches": 8, "prefix_steps": 7168,
                "branch_steps": 49152, "total_steps": 56320}
TIE_TOLERANCE = 1e-12


def _number(value: Any, name: str, *, integer: bool = False, nullable: bool = False) -> float | int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if integer:
        if number < 0 or not number.is_integer():
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(number)
    return number


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
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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


def _load_summary(input_path: str | Path) -> tuple[dict[str, Any], Path, str]:
    source = Path(input_path)
    path = source / "summary.json" if source.is_dir() else source
    if path.name != "summary.json" or not path.is_file():
        raise FileNotFoundError(f"expected summary.json under {source}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read summary.json: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("summary.json must contain an object")
    return value, path.resolve(), _file_hash(path)


def _active_config(config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    result = dict(config)
    if phase == "pilot":
        pilot = _mapping(config.get("pilot"), "config.pilot")
        for key in ("seeds", "pair_kinds", "history_regimes", "future_regimes", "future_blocks", "horizons", "max_workers"):
            if key in pilot:
                result[key] = pilot[key]
    return result


def _design(summary: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, str | None]]:
    if summary.get("format") != SUMMARY_FORMAT:
        raise ValueError(f"input format must be {SUMMARY_FORMAT}")
    phase = summary.get("phase")
    if phase not in ("pilot", "main"):
        raise ValueError("summary.phase must be pilot or main")
    config = _mapping(summary.get("config"), "summary.config")
    active = _active_config(config, str(phase))
    try:
        seeds = [int(x) for x in active["seeds"]]
        pairs = [str(x) for x in active["pair_kinds"]]
        histories = [str(x) for x in active["history_regimes"]]
        futures = [str(x) for x in active["future_regimes"]]
        replicas = [int(x) for x in active["replicates"]]
        actions = [str(x) for x in active["actions"]]
        horizons = [int(x) for x in active["horizons"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"incomplete {phase} dimensions in summary.config") from exc
    if actions != list(ACTIONS):
        raise ValueError(f"actions must be {ACTIONS}")
    if not horizons or horizons != sorted(set(horizons)) or horizons[0] <= 0:
        raise ValueError("horizons must be increasing positive integers")
    if phase == "main" and tuple(seeds) != MAIN_SEEDS:
        raise ValueError(f"main seeds must equal {MAIN_SEEDS}")
    if phase == "pilot" and (seeds != [PILOT_SEED] or pairs != ["dynamics"] or histories != ["old_major"] or futures != ["old_major"]):
        raise ValueError("pilot dimensions differ from the frozen design")

    expected_cases = {(s, p, h) for s in seeds for p in pairs for h in histories}
    expected_branches = {(s, p, h, f, r, a) for s, p, h in expected_cases
                         for f in futures for r in replicas for a in actions}
    raw_cases, raw_branches = summary.get("cases"), summary.get("branches")
    if not isinstance(raw_cases, list) or not isinstance(raw_branches, list):
        raise ValueError("summary.cases and summary.branches must be arrays")
    cases = [_mapping(x, f"cases[{i}]") for i, x in enumerate(raw_cases)]
    branches = [_mapping(x, f"branches[{i}]") for i, x in enumerate(raw_branches)]
    if list(int(s) for s in summary.get("seeds", [])) != seeds:
        raise ValueError("summary.seeds disagrees with the active frozen design")
    if int(_number(summary.get("case_count"), "summary.case_count", integer=True)) != len(cases):
        raise ValueError("summary.case_count disagrees with cases")
    if int(_number(summary.get("branch_count"), "summary.branch_count", integer=True)) != len(branches):
        raise ValueError("summary.branch_count disagrees with branches")
    if summary.get("errors"):
        raise ValueError("summary has errors")

    case_index: dict[tuple[int, str, str], dict[str, Any]] = {}
    cases_by_id: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(cases):
        key = (int(_number(row.get("seed"), f"cases[{i}].seed", integer=True)), str(row.get("pair_kind")), str(row.get("history_regime")))
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or key in case_index or case_id in cases_by_id or key not in expected_cases:
            raise ValueError(f"duplicate, missing, or unexpected case identity at cases[{i}]")
        if row.get("errors"):
            raise ValueError(f"case {case_id} has errors")
        mapping = _mapping(row.get("mapping"), f"case {case_id}.mapping")
        family_map = _mapping(mapping.get("families"), f"case {case_id}.mapping.families")
        old_family, new_family = family_map.get("old"), family_map.get("new")
        expected_new = ("B" if old_family == "A" else "A") if key[1] == "dynamics" else old_family
        if old_family not in ("A", "B") or new_family != expected_new:
            raise ValueError(f"case {case_id} family map is invalid for {key[1]}")
        case_index[key] = dict(row)
        cases_by_id[case_id] = dict(row)
    if set(case_index) != expected_cases:
        raise ValueError(f"case grid mismatch; missing {sorted(expected_cases - set(case_index))[:4]}")

    flat_index: dict[tuple[int, str, str, str, int, str], dict[str, Any]] = {}
    branch_id_index: dict[tuple[str, str], tuple[int, str, str, str, int, str]] = {}
    for i, row in enumerate(branches):
        try:
            key = (int(_number(row.get("seed"), f"branches[{i}].seed", integer=True)),
                   str(row.get("pair_kind")), str(row.get("history_regime")), str(row.get("future_regime")),
                   int(_number(row.get("replicate"), f"branches[{i}].replicate", integer=True)), str(row.get("action")))
        except ValueError as exc:
            raise ValueError(f"invalid branch identity at branches[{i}]") from exc
        branch_id, case_id = row.get("branch_id"), row.get("case_id")
        identity = (str(case_id), str(branch_id))
        if not isinstance(branch_id, str) or not branch_id or identity in branch_id_index or key in flat_index:
            raise ValueError(f"duplicate or missing branch identity at branches[{i}]")
        if key not in expected_branches or case_id not in cases_by_id:
            raise ValueError(f"unexpected branch key or case link: {key}")
        linked_case = cases_by_id[case_id]
        if (int(linked_case["seed"]), str(linked_case["pair_kind"]), str(linked_case["history_regime"])) != key[:3]:
            raise ValueError(f"branch {branch_id} case link disagrees with branch dimensions")
        if row.get("errors"):
            raise ValueError(f"branch {branch_id} has errors")
        if row.get("prefix_restore_verified") is not True or row.get("prefix_state_hash") != row.get("restored_state_hash"):
            raise ValueError(f"branch {branch_id} failed common-prefix restore verification")
        flat_index[key] = dict(row)
        branch_id_index[identity] = key
    if set(flat_index) != expected_branches:
        raise ValueError(f"branch grid mismatch; missing {sorted(expected_branches - set(flat_index))[:4]}")

    for case_id, case in cases_by_id.items():
        embedded = case.get("branches")
        if not isinstance(embedded, list):
            raise ValueError(f"case {case_id}.branches must be an array")
        embedded_ids: set[str] = set()
        for row in embedded:
            row = _mapping(row, f"case {case_id}.branches[]")
            branch_id = row.get("branch_id")
            identity = (case_id, str(branch_id))
            if not isinstance(branch_id, str) or branch_id in embedded_ids or identity not in branch_id_index:
                raise ValueError(f"case {case_id} has missing or duplicate embedded branch IDs")
            embedded_ids.add(branch_id)
            flat = flat_index[branch_id_index[identity]]
            if _canonical_hash(row) != _canonical_hash(flat):
                raise ValueError(f"case {case_id} embedded branch differs from flattened record {branch_id}")
        expected_ids = {branch_id for (linked_case_id, branch_id), key in branch_id_index.items()
                        if linked_case_id == case_id and key[:3] == (
                            int(case["seed"]), str(case["pair_kind"]), str(case["history_regime"]))}
        if embedded_ids != expected_ids:
            raise ValueError(f"case {case_id} has an incomplete embedded branch grid")
        prefix = _mapping(case.get("prefix"), f"case {case_id}.prefix")
        prefix_hash = prefix.get("prefix_state_hash")
        for branch_id in embedded_ids:
            flat = flat_index[branch_id_index[(case_id, branch_id)]]
            if flat.get("prefix_state_hash") != prefix_hash:
                raise ValueError(f"branch {branch_id} prefix hash differs from case prefix")

    families: dict[int, str | None] = {}
    for seed in seeds:
        old_families = {case["mapping"]["families"]["old"] for case in cases if int(case["seed"]) == seed}
        if len(old_families) != 1:
            raise ValueError(f"seed {seed} family mapping varies across cases")
        families[seed] = next(iter(old_families))
    if phase == "main" and ({v: list(families.values()).count(v) for v in ("A", "B")} != {"A": 3, "B": 3}
                            or sum(s % 2 == 0 for s in seeds) != 3 or sum(s % 2 == 1 for s in seeds) != 3):
        raise ValueError("main seed strata must be three old-family A/B and three even/odd")
    expected_counts = MAIN_COUNTS if phase == "main" else PILOT_COUNTS
    if len(cases) != expected_counts["cases"] or len(branches) != expected_counts["branches"]:
        raise ValueError("case/branch count differs from frozen design")
    if max(horizons) * len(branches) != expected_counts["branch_steps"]:
        raise ValueError("branch horizon total differs from frozen physical step count")
    active.update({"seeds": seeds, "pair_kinds": pairs, "history_regimes": histories, "future_regimes": futures,
                   "replicates": replicas, "actions": actions, "horizons": horizons})
    return {"phase": str(phase), "config": dict(config), "active": active, "expected_counts": expected_counts}, \
        [dict(x) for x in cases], [dict(x) for x in branches], families


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
    if not path.is_file():
        raise ValueError(f"missing ledger {path}")
    if record.get("exists") is not True:
        raise ValueError(f"{where}.exists must be true for every audited ledger")
    actual_hash = _file_hash(path)
    if record.get("sha256") != actual_hash:
        raise ValueError(f"{where} SHA-256 mismatch")
    return path, record


def _steps(root: Path, meta: Any, where: str, checkpoints: Sequence[int],
           action: str | None = None, future: str | None = None) -> tuple[dict[str, Any], dict[int, dict[str, float | int]]]:
    path, record = _resolve_ledger(root, meta, where)
    targets = set(int(x) for x in checkpoints)
    row_count = 0
    reward_total = 0.0
    marks: dict[int, dict[str, float | int]] = {}
    previous_global: int | None = None
    previous_bout: int | None = None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "reward" not in reader.fieldnames:
                raise ValueError(f"{where} requires reward column")
            for line_no, row in enumerate(reader, start=2):
                row_count += 1
                reward = float(_number(row.get("reward"), f"{where}:{line_no}.reward"))
                reward_total += reward
                if action is not None and row.get("action") != action:
                    raise ValueError(f"{where}:{line_no} action mismatch")
                if future is not None and row.get("future_regime") != future:
                    raise ValueError(f"{where}:{line_no} future mismatch")
                raw_global = row.get("global_step")
                if raw_global not in (None, ""):
                    global_step = int(_number(raw_global, f"{where}:{line_no}.global_step", integer=True))
                    if previous_global is not None and global_step != previous_global + 1:
                        raise ValueError(f"{where} global_step is not contiguous")
                    previous_global = global_step
                raw_bout = row.get("bout")
                if raw_bout not in (None, ""):
                    bout = int(_number(raw_bout, f"{where}:{line_no}.bout", integer=True))
                    if previous_bout is not None and bout < previous_bout:
                        raise ValueError(f"{where} rows are not in bout order")
                    previous_bout = bout
                if row_count in targets:
                    marks[row_count] = {"steps": row_count, "reward_sum": reward_total,
                                        "reward_per_step": reward_total / row_count}
    except (OSError, csv.Error) as exc:
        raise ValueError(f"cannot read {where}") from exc
    declared_rows = int(_number(record.get("rows"), f"{where}.rows", integer=True))
    declared_sum = float(_number(record.get("reward_sum"), f"{where}.reward_sum"))
    if declared_rows != row_count or not math.isclose(declared_sum, reward_total, rel_tol=1e-11, abs_tol=1e-8):
        raise ValueError(f"{where} raw rows/reward disagree with ledger manifest")
    return {"path": path.relative_to(root.resolve()).as_posix(), "sha256": _file_hash(path),
            "rows": row_count, "reward_sum": reward_total}, marks


def _rollouts(root: Path, meta: Any, where: str, checkpoints: Sequence[int]) -> tuple[dict[str, Any], dict[int, dict[str, int]]]:
    path, record = _resolve_ledger(root, meta, where)
    targets = set(int(x) for x in checkpoints)
    row_count = total_rows = total_updates = 0
    marks: dict[int, dict[str, int]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"rows", "updates"} <= set(reader.fieldnames):
                raise ValueError(f"{where} requires rows and updates columns")
            for line_no, row in enumerate(reader, start=2):
                row_count += 1
                nrows = int(_number(row.get("rows"), f"{where}:{line_no}.rows", integer=True))
                updates = int(_number(row.get("updates"), f"{where}:{line_no}.updates", integer=True))
                if nrows <= 0:
                    raise ValueError(f"{where}:{line_no}.rows must be positive")
                total_rows += nrows
                total_updates += updates
                if total_rows in targets:
                    marks[total_rows] = {"rollouts_trained": row_count, "minibatch_updates": total_updates}
    except (OSError, csv.Error) as exc:
        raise ValueError(f"cannot read {where}") from exc
    if int(_number(record.get("rows"), f"{where}.rows", integer=True)) != row_count:
        raise ValueError(f"{where} rollout row count disagrees with manifest")
    if total_rows != max(targets):
        raise ValueError(f"{where} sums {total_rows} rollout rows, expected {max(targets)}")
    if int(_number(record.get("trained_rows"), f"{where}.trained_rows", integer=True)) != total_rows:
        raise ValueError(f"{where} trained_rows disagrees with its rollout ledger")
    if int(_number(record.get("minibatch_updates"), f"{where}.minibatch_updates", integer=True)) != total_updates:
        raise ValueError(f"{where} minibatch_updates disagrees with its rollout ledger")
    if int(_number(record.get("rollout_count"), f"{where}.rollout_count", integer=True)) != row_count:
        raise ValueError(f"{where} rollout_count disagrees with its rollout ledger")
    return {"path": path.relative_to(root.resolve()).as_posix(), "sha256": _file_hash(path),
            "rows": row_count, "rollout_count": row_count, "trained_rows": total_rows,
            "minibatch_updates": total_updates}, marks


def _simple_ledger(root: Path, meta: Any, where: str) -> dict[str, Any]:
    path, record = _resolve_ledger(root, meta, where)
    count = 0
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            count = sum(1 for _ in reader)
    except (OSError, csv.Error) as exc:
        raise ValueError(f"cannot read {where}") from exc
    if int(_number(record.get("rows"), f"{where}.rows", integer=True)) != count:
        raise ValueError(f"{where} row count disagrees with manifest")
    return {"path": path.relative_to(root.resolve()).as_posix(), "sha256": _file_hash(path), "rows": count}


def _ledger_map(record: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    return _mapping(record.get("raw_ledgers"), f"{where}.raw_ledgers")


def _audit_ledgers(cases: Sequence[Mapping[str, Any]], branches: Sequence[Mapping[str, Any]],
                   root: Path, prefix_steps: int, horizons: Sequence[int], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    prefix_rows = branch_rows = prefix_rollouts = branch_rollouts = 0
    prefix_updates = branch_updates = 0
    prefix_reward = branch_reward = 0.0
    source_fits_audit: list[dict[str, Any]] = []
    schedules: dict[tuple[str, str], str] = {}
    for case in cases:
        cid = str(case["case_id"])
        prefix = _mapping(case.get("prefix"), f"case {cid}.prefix")
        if int(_number(prefix.get("steps"), f"case {cid}.prefix.steps", integer=True)) != prefix_steps:
            raise ValueError(f"case {cid} prefix step count is not {prefix_steps}")
        source_fits = _mapping(prefix.get("source_fits"), f"case {cid}.prefix.source_fits")
        if not all(source_fits.get(flag) is True for flag in ("old_verified", "candidate_verified", "verified")):
            raise ValueError(f"case {cid} source-fit verification failed")
        source_rows: dict[str, list[dict[str, Any]]] = {}
        for role in ("old_sources", "candidate_sources"):
            rows = source_fits.get(role)
            if not isinstance(rows, list) or len(rows) != 2:
                raise ValueError(f"case {cid} requires two {role} source-time matrices")
            normalized_rows = []
            for source in rows:
                source = _mapping(source, f"case {cid}.prefix.source_fits.{role}[]")
                expected_group = 0 if role == "old_sources" else 1
                group = int(_number(source.get("group"),
                                    f"case {cid}.prefix.source_fits.{role}.group", integer=True))
                fitted_steps = int(_number(source.get("fitted_steps"),
                                           f"case {cid}.prefix.source_fits.{role}.fitted_steps", integer=True))
                source_steps = int(_number(source.get("source_time_fitted_steps"),
                                           f"case {cid}.prefix.source_fits.{role}.source_time_fitted_steps",
                                           integer=True))
                matrix = source.get("effective_matrix")
                if group != expected_group or fitted_steps != 1024 or source_steps != 1024:
                    raise ValueError(f"case {cid} {role} source fit must be 1024 steps in group {expected_group}")
                if not isinstance(matrix, list):
                    raise ValueError(f"case {cid} {role} source record is missing its effective matrix")
                normalized_rows.append({
                    "task": str(source.get("task")), "group": group, "slot": source.get("slot"),
                    "generation": source.get("generation"), "fitted_steps": fitted_steps,
                    "source_time_fitted_steps": source_steps, "effective_matrix": matrix,
                    "effective_matrix_canonical_sha256": _canonical_hash(matrix),
                    "later_same_generation_fitted_steps": source.get("later_same_generation_fitted_steps"),
                    "task_lifetime_observed_steps": source.get("task_lifetime_observed_steps"),
                })
            source_rows[role] = normalized_rows
        source_fits_audit.append({
            "case_id": cid, "seed": int(case["seed"]), "pair_kind": str(case["pair_kind"]),
            "history_regime": str(case["history_regime"]), "verified": True,
            "old_source_time_steps": [row["source_time_fitted_steps"] for row in source_rows["old_sources"]],
            "candidate_source_time_steps": [row["source_time_fitted_steps"] for row in source_rows["candidate_sources"]],
            "old_sources": source_rows["old_sources"], "candidate_sources": source_rows["candidate_sources"],
            "comparison_basis": "source-time effective matrices fitted from 1024 steps in old parent group 0 and fresh candidate group 1; task lifetime may include later same-generation exposure",
        })
        ledgers = _ledger_map(prefix, f"case {cid}.prefix")
        step_meta, _ = _steps(root, ledgers.get("steps_csv"), f"case {cid} prefix steps.csv", [prefix_steps])
        if not math.isclose(step_meta["reward_sum"], float(_number(prefix.get("reward_total"), f"case {cid}.prefix.reward_total")), rel_tol=1e-11, abs_tol=1e-8):
            raise ValueError(f"case {cid} prefix reward differs from steps.csv")
        rollout_meta, rollout_marks = _rollouts(root, ledgers.get("rollouts_csv"), f"case {cid} prefix rollouts.csv", [prefix_steps])
        episodes = _simple_ledger(root, ledgers.get("episodes_csv"), f"case {cid} prefix episodes.csv")
        if prefix_steps not in rollout_marks:
            raise ValueError(f"case {cid} prefix has no completed-rollout checkpoint")
        prefix_rows += step_meta["rows"]
        prefix_reward += step_meta["reward_sum"]
        prefix_rollouts += rollout_marks[prefix_steps]["rollouts_trained"]
        prefix_updates += rollout_marks[prefix_steps]["minibatch_updates"]
        prefix_counts = _mapping(prefix.get("ppo_call_counts"), f"case {cid}.prefix.ppo_call_counts")
        if int(_number(prefix_counts.get("rollouts_completed"), f"case {cid}.prefix.ppo_call_counts.rollouts_completed", integer=True)) != rollout_meta["rollout_count"]:
            raise ValueError(f"case {cid} prefix completed-rollout count disagrees with rollouts.csv")
        if int(_number(prefix_counts.get("minibatch_updates"), f"case {cid}.prefix.ppo_call_counts.minibatch_updates", integer=True)) != rollout_meta["minibatch_updates"]:
            raise ValueError(f"case {cid} prefix PPO update count disagrees with rollouts.csv")
        for item in (step_meta, rollout_meta, episodes):
            hashes[item["path"]] = item["sha256"]

    for branch in branches:
        bid = str(branch["branch_id"])
        seed, rep = int(branch["seed"]), int(branch["replicate"])
        expected_future_seed = int(config.get("future_seed_offset", 40000000)) + int(config.get("future_seed_multiplier", 1000)) * seed + rep
        if int(_number(branch.get("future_seed"), f"branch {bid}.future_seed", integer=True)) != expected_future_seed:
            raise ValueError(f"branch {bid} future seed differs from frozen transform")
        if int(_number(branch.get("raw_environment_steps"), f"branch {bid}.raw_environment_steps", integer=True)) != max(horizons):
            raise ValueError(f"branch {bid} raw_environment_steps differs from its continuation horizon")
        schedule = branch.get("future_schedule")
        if not isinstance(schedule, list):
            raise ValueError(f"branch {bid} future_schedule must be an array")
        schedule_hash = _canonical_hash(schedule)
        schedule_key = (str(branch["case_id"]), str(branch["future_regime"]))
        if schedule_key in schedules and schedules[schedule_key] != schedule_hash:
            raise ValueError(f"branch {bid} future schedule differs across actions or replicas")
        schedules[schedule_key] = schedule_hash
        rng = _mapping(branch.get("rng_transform"), f"branch {bid}.rng_transform")
        if rng.get("environment_rng_preserved") is not True or rng.get("current_episode_preserved") is not True:
            raise ValueError(f"branch {bid} changed the environment RNG or current episode during reseeding")
        parent_audit = _mapping(branch.get("shared_parent_invariant"), f"branch {bid}.shared_parent_invariant")
        if parent_audit.get("exactly_fixed") is not True or float(_number(parent_audit.get("max_coefficient_drift"),
                                                                           f"branch {bid}.shared_parent_invariant.max_coefficient_drift")) != 0.0:
            raise ValueError(f"branch {bid} shared parent changed during continuation")
        if branch.get("prefix_bouts_included_in_reward") is not False:
            raise ValueError(f"branch {bid} reward includes its prefix")
        ledgers = _ledger_map(branch, f"branch {bid}")
        step_meta, step_marks = _steps(root, ledgers.get("steps_csv"), f"branch {bid} steps.csv", horizons,
                                       str(branch["action"]), str(branch["future_regime"]))
        rollout_meta, rollout_marks = _rollouts(root, ledgers.get("rollouts_csv"), f"branch {bid} rollouts.csv", horizons)
        episodes = _simple_ledger(root, ledgers.get("episodes_csv"), f"branch {bid} episodes.csv")
        if step_meta["rows"] != max(horizons):
            raise ValueError(f"branch {bid} raw step count is incomplete")
        ppo_counts = _mapping(branch.get("ppo_call_counts"), f"branch {bid}.ppo_call_counts")
        if int(_number(ppo_counts.get("rollouts_completed"), f"branch {bid}.ppo_call_counts.rollouts_completed", integer=True)) != rollout_meta["rollout_count"]:
            raise ValueError(f"branch {bid} completed-rollout count disagrees with rollouts.csv")
        if int(_number(ppo_counts.get("minibatch_updates"), f"branch {bid}.ppo_call_counts.minibatch_updates", integer=True)) != rollout_meta["minibatch_updates"]:
            raise ValueError(f"branch {bid} PPO update count disagrees with rollouts.csv")
        if int(_number(ppo_counts.get("rollouts_failed"), f"branch {bid}.ppo_call_counts.rollouts_failed", integer=True)) != 0:
            raise ValueError(f"branch {bid} reports failed PPO rollouts")
        if int(_number(ppo_counts.get("rollouts_attempted"), f"branch {bid}.ppo_call_counts.rollouts_attempted", integer=True)) != rollout_meta["rollout_count"]:
            raise ValueError(f"branch {bid} attempted/completed rollout counts differ")
        if int(_number(ppo_counts.get("partial_minibatch_updates"), f"branch {bid}.ppo_call_counts.partial_minibatch_updates", integer=True)) != 0:
            raise ValueError(f"branch {bid} has partial PPO minibatch updates")
        counts_delta = _mapping(branch.get("counts_delta"), f"branch {bid}.counts_delta")
        if int(_number(counts_delta.get("optimizer_step_count"), f"branch {bid}.counts_delta.optimizer_step_count", integer=True)) != rollout_meta["minibatch_updates"]:
            raise ValueError(f"branch {bid} optimizer step count disagrees with rollouts.csv")
        hrows = branch.get("horizons")
        if isinstance(hrows, Mapping):
            hrows = list(hrows.values())
        if not isinstance(hrows, list):
            raise ValueError(f"branch {bid}.horizons must be an array")
        declared: dict[int, Mapping[str, Any]] = {}
        for hrow in hrows:
            hrow = _mapping(hrow, f"branch {bid}.horizons[]")
            h = int(_number(hrow.get("steps"), f"branch {bid}.horizon.steps", integer=True))
            if h in declared:
                raise ValueError(f"branch {bid} has duplicate horizon {h}")
            declared[h] = hrow
        if set(declared) != set(int(h) for h in horizons):
            raise ValueError(f"branch {bid} horizon rows incomplete")
        for h in horizons:
            raw, roll = step_marks.get(int(h)), rollout_marks.get(int(h))
            if raw is None or roll is None:
                raise ValueError(f"branch {bid} raw ledgers miss horizon {h}")
            hrow = declared[int(h)]
            for field, actual, tol in (
                ("reward_sum", raw["reward_sum"], 1e-8),
                ("reward_per_step", raw["reward_per_step"], 1e-10),
            ):
                expected_value = float(_number(hrow.get(field), f"branch {bid}.{h}.{field}"))
                if not math.isclose(float(actual), expected_value, rel_tol=1e-11, abs_tol=tol):
                    raise ValueError(f"branch {bid} {field} at horizon {h} differs from raw steps")
            if int(_number(hrow.get("raw_environment_steps"), f"branch {bid}.{h}.raw_environment_steps", integer=True)) != h:
                raise ValueError(f"branch {bid} raw environment steps differ at horizon {h}")
            for field in ("rollouts_trained", "minibatch_updates"):
                if int(_number(hrow.get(field), f"branch {bid}.{h}.{field}", integer=True)) != roll[field]:
                    raise ValueError(f"branch {bid} {field} at horizon {h} differs from rollouts.csv")
            if int(roll["rollouts_trained"]) != int(h) // 256:
                raise ValueError(f"branch {bid} did not train one completed PPO rollout per 256 steps at horizon {h}")
            cpu = float(_number(hrow.get("simulation_cpu_s"), f"branch {bid}.{h}.simulation_cpu_s"))
            utility = _number(hrow.get("utility_rate"), f"branch {bid}.{h}.utility_rate", nullable=True)
            install = _number(hrow.get("action_install_cpu_s"), f"branch {bid}.{h}.action_install_cpu_s", nullable=True)
            if cpu < 0:
                raise ValueError(f"branch {bid} has negative simulation CPU")
            observations.append({
                "branch_id": bid, "case_id": str(branch["case_id"]), "seed": seed,
                "pair_kind": str(branch["pair_kind"]), "history_regime": str(branch["history_regime"]),
                "future_regime": str(branch["future_regime"]), "replicate": rep, "action": str(branch["action"]),
                "horizon": int(h), "reward_sum": float(raw["reward_sum"]),
                "reward_per_step": float(raw["reward_per_step"]), "simulation_cpu_s": cpu,
                "cpu_seconds_per_1000_steps": cpu * 1000 / int(h), "action_install_cpu_s": install,
                "utility_rate": utility, "rollouts_trained": roll["rollouts_trained"],
                "minibatch_updates": roll["minibatch_updates"], "raw_environment_steps": int(h),
            })
        prefix_rows += 0
        branch_rows += step_meta["rows"]
        branch_reward += step_meta["reward_sum"]
        branch_rollouts += rollout_marks[max(horizons)]["rollouts_trained"]
        branch_updates += rollout_marks[max(horizons)]["minibatch_updates"]
        for item in (step_meta, rollout_meta, episodes):
            hashes[item["path"]] = item["sha256"]

    expected = MAIN_COUNTS if len(branches) == MAIN_COUNTS["branches"] else PILOT_COUNTS
    total = prefix_rows + branch_rows
    if (prefix_rows, branch_rows, total) != (expected["prefix_steps"], expected["branch_steps"], expected["total_steps"]):
        raise ValueError(f"actual raw steps mismatch: prefix={prefix_rows}, branches={branch_rows}, total={total}")
    return observations, {
        "prefix_environment_steps": prefix_rows, "branch_environment_steps": branch_rows,
        "total_environment_steps": total, "prefix_reward_sum": prefix_reward,
        "branch_reward_sum": branch_reward, "prefix_rollouts_trained": prefix_rollouts,
        "branch_rollouts_trained": branch_rollouts, "total_rollouts_trained": prefix_rollouts + branch_rollouts,
        "prefix_minibatch_updates": prefix_updates, "branch_minibatch_updates": branch_updates,
        "total_minibatch_updates": prefix_updates + branch_updates, "raw_ledger_count": len(hashes),
        "raw_ledger_hashes": dict(sorted(hashes.items())), "source_fits": source_fits_audit,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _stat(values: Mapping[int, float], families: Mapping[int, str | None],
          config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    rows = sorted((int(seed), float(value)) for seed, value in values.items())
    result: dict[str, Any] = {
        "seed_count": len(rows),
        "mean": _mean([v for _, v in rows]),
        "per_seed": [{"seed": s, "family": families.get(s), "value": v} for s, v in rows],
    }
    if phase == "pilot":
        result.update({"interval": None, "interval_method": "not estimated for single-seed pilot"})
        return result
    report = _mapping(config.get("report"), "config.report")
    nboot = int(_number(report.get("bootstrap_resamples"), "config.report.bootstrap_resamples", integer=True))
    rng_seed = int(_number(report.get("bootstrap_seed"), "config.report.bootstrap_seed", integer=True))
    strata = {family: [s for s, _ in rows if families.get(s) == family] for family in ("A", "B")}
    if nboot <= 0 or strata != {"A": [s for s in strata["A"]], "B": [s for s in strata["B"]]}:
        raise ValueError("invalid stratified bootstrap configuration")
    if any(len(strata[f]) != 3 for f in ("A", "B")) or set(dict(rows)) != set(MAIN_SEEDS):
        raise ValueError("main bootstrap requires all six seeds and three in each family stratum")
    value_by_seed = dict(rows)
    rng = np.random.default_rng(rng_seed)
    draws = np.empty(nboot, dtype=np.float64)
    for index in range(nboot):
        sampled: list[int] = []
        for family in ("A", "B"):
            sampled.extend(int(x) for x in rng.choice(strata[family], size=3, replace=True))
        draws[index] = sum(value_by_seed[seed] for seed in sampled) / len(sampled)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    result.update({
        "interval": [float(low), float(high)],
        "interval_method": "stratified percentile bootstrap over independent seeds",
        "bootstrap_resamples": nboot,
        "bootstrap_seed": rng_seed,
        "strata": {family: list(strata[family]) for family in ("A", "B")},
    })
    return result


def _annotate_reward_reference(stat: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    reference = float(_number(
        _mapping(config.get("report"), "config.report").get("practical_reward_per_step"),
        "config.report.practical_reward_per_step",
    ))
    interval = stat.get("interval")
    if not isinstance(interval, list) or len(interval) != 2:
        interpretation = "not estimated for the single-seed pilot"
    elif float(interval[0]) < reference <= float(interval[1]):
        interpretation = "inconclusive: interval spans effects below and at or above the practical reference"
    elif float(interval[0]) >= reference:
        interpretation = "interval is at or above the practical reference"
    else:
        interpretation = "interval is below the practical reference"
    stat["practical_reward_per_step_reference"] = reference
    stat["practical_interval_interpretation"] = interpretation
    return stat


def _cell_means(observations: Sequence[Mapping[str, Any]], horizon: int, metric: str) -> dict[tuple[Any, ...], float]:
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in observations:
        if int(row["horizon"]) != horizon or row.get(metric) is None:
            continue
        key = (int(row["seed"]), str(row["pair_kind"]), str(row["history_regime"]),
               str(row["future_regime"]), int(row["replicate"]), str(row["action"]))
        buckets.setdefault(key, []).append(float(row[metric]))
    return {key: float(sum(values) / len(values)) for key, values in buckets.items()}


def _effects(observations: Sequence[Mapping[str, Any]], families: Mapping[int, str | None],
             config: Mapping[str, Any], active: Mapping[str, Any], phase: str
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    seeds = [int(s) for s in active["seeds"]]
    metrics = ("reward_per_step", "cpu_seconds_per_1000_steps", "utility_rate")
    for horizon in active["horizons"]:
        horizon = int(horizon)
        cells = {metric: _cell_means(observations, horizon, metric) for metric in metrics}
        for pair in active["pair_kinds"]:
            pair = str(pair)
            for history in active["history_regimes"]:
                history = str(history)
                for future in active["future_regimes"]:
                    future = str(future)
                    for action in ACTIONS[1:]:
                        stats: dict[str, Any] = {}
                        for metric in metrics:
                            values = {}
                            cell = cells[metric]
                            for seed in seeds:
                                akey = (seed, pair, history, future, 0, action)
                                # Average replicas equally within each condition.
                                contrasts = []
                                for rep in active["replicates"]:
                                    ak = (seed, pair, history, future, int(rep), action)
                                    kk = (seed, pair, history, future, int(rep), "keep")
                                    if ak in cell and kk in cell:
                                        contrasts.append(cell[ak] - cell[kk])
                                if contrasts:
                                    values[seed] = float(sum(contrasts) / len(contrasts))
                            stats[metric] = _stat(values, families, config, phase)
                            if metric == "reward_per_step":
                                _annotate_reward_reference(stats[metric], config)
                        subgroup_rows.append({
                            "pair_kind": pair, "history_regime": history, "future_regime": future,
                            "horizon": horizon, "action": action, "contrast": f"{action} - keep", **stats,
                        })

            for action in ACTIONS[1:]:
                aggregate_stats: dict[str, Any] = {}
                for metric in metrics:
                    cell = cells[metric]
                    values: dict[int, float] = {}
                    for seed in seeds:
                        history_values = []
                        for history in active["history_regimes"]:
                            future_values = []
                            for future in active["future_regimes"]:
                                contrasts = []
                                for rep in active["replicates"]:
                                    ak = (seed, pair, str(history), str(future), int(rep), action)
                                    kk = (seed, pair, str(history), str(future), int(rep), "keep")
                                    if ak in cell and kk in cell:
                                        contrasts.append(cell[ak] - cell[kk])
                                if contrasts:
                                    future_values.append(float(sum(contrasts) / len(contrasts)))
                            if future_values:
                                history_values.append(float(sum(future_values) / len(future_values)))
                        if len(history_values) == len(active["history_regimes"]):
                            values[seed] = float(sum(history_values) / len(history_values))
                    aggregate_stats[metric] = _stat(values, families, config, phase)
                    if metric == "reward_per_step":
                        _annotate_reward_reference(aggregate_stats[metric], config)
                all_rows.append({
                    "pair_kind": pair, "horizon": horizon, "action": action,
                    "contrast": f"{action} - keep", **aggregate_stats,
                })

            if set(active["future_regimes"]) == {"old_major", "new_major"}:
                interaction_values: dict[int, float] = {}
                cell = cells["reward_per_step"]
                for seed in seeds:
                    history_interactions = []
                    for history in active["history_regimes"]:
                        future_deltas = {}
                        for future in ("old_major", "new_major"):
                            contrasts = []
                            for rep in active["replicates"]:
                                rk = (seed, pair, str(history), future, int(rep), "replace")
                                kk = (seed, pair, str(history), future, int(rep), "keep")
                                if rk in cell and kk in cell:
                                    contrasts.append(cell[rk] - cell[kk])
                            if contrasts:
                                future_deltas[future] = float(sum(contrasts) / len(contrasts))
                        if set(future_deltas) == {"old_major", "new_major"}:
                            history_interactions.append(future_deltas["new_major"] - future_deltas["old_major"])
                    if len(history_interactions) == len(active["history_regimes"]):
                        interaction_values[seed] = float(sum(history_interactions) / len(history_interactions))
                interaction_stat = _stat(interaction_values, families, config, phase)
                _annotate_reward_reference(interaction_stat, config)
                interactions.append({
                    "pair_kind": pair, "horizon": horizon,
                    "formula": "(replace - keep) new_major future - (replace - keep) old_major future",
                    "reward_per_step": interaction_stat,
                })
    return all_rows, subgroup_rows, interactions


def _choose(scores: Mapping[str, float]) -> tuple[str, list[str], float]:
    high = max(scores.values())
    tied = [action for action in ACTIONS if abs(scores[action] - high) <= TIE_TOLERANCE]
    chosen = "keep" if "keep" in tied else tied[0]
    sorted_scores = sorted(scores.values(), reverse=True)
    return chosen, tied, float(sorted_scores[0] - sorted_scores[1])


def _cross_replica(observations: Sequence[Mapping[str, Any]], families: Mapping[int, str | None],
                   config: Mapping[str, Any], active: Mapping[str, Any], phase: str) -> dict[str, Any]:
    seeds = [int(s) for s in active["seeds"]]
    pairs = [str(p) for p in active["pair_kinds"]]
    histories = [str(h) for h in active["history_regimes"]]
    futures = [str(f) for f in active["future_regimes"]]
    reps = [int(r) for r in active["replicates"]]
    if reps != [0, 1]:
        raise ValueError("cross-replica analysis requires replicas 0 and 1")
    raw: list[dict[str, Any]] = []
    for horizon in active["horizons"]:
        horizon = int(horizon)
        cells = _cell_means(observations, horizon, "reward_per_step")
        for seed in seeds:
            for pair in pairs:
                for history in histories:
                    for future in futures:
                        scores = {
                            rep: {action: cells[(seed, pair, history, future, rep, action)] for action in ACTIONS}
                            for rep in reps
                        }
                        chosen: dict[int, tuple[str, list[str], float]] = {
                            rep: _choose(scores[rep]) for rep in reps
                        }
                        agreement = chosen[0][0] == chosen[1][0]
                        for train_rep, eval_rep in ((0, 1), (1, 0)):
                            action, ties, margin = chosen[train_rep]
                            evaluated = scores[eval_rep]
                            raw.append({
                                "seed": seed, "family": families.get(seed), "pair_kind": pair,
                                "history_regime": history, "future_regime": future, "horizon": horizon,
                                "selection_replica": train_rep, "evaluation_replica": eval_rep,
                                "selected_action": action, "tied_max_actions": ties,
                                "tie_default_keep": len(ties) > 1 and action == "keep",
                                "selection_margin_reward_per_step": margin,
                                "replica_argmax_agreement": agreement,
                                "selected_minus_keep_reward_per_step": evaluated[action] - evaluated["keep"],
                                "selected_minus_replace_reward_per_step": evaluated[action] - evaluated["replace"],
                                "fixed_replace_minus_keep_reward_per_step": evaluated["replace"] - evaluated["keep"],
                            })

    metric_names = ("selected_minus_keep_reward_per_step", "selected_minus_replace_reward_per_step",
                    "fixed_replace_minus_keep_reward_per_step", "selection_margin_reward_per_step")

    def summarize(group_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_metric: dict[str, Any] = {}
        for name in metric_names:
            values: dict[int, float] = {}
            for seed in seeds:
                selected = [float(row[name]) for row in group_rows if int(row["seed"]) == seed]
                if selected:
                    values[seed] = float(sum(selected) / len(selected))
            by_metric[name] = _stat(values, families, config, phase)
            if "reward_per_step" in name:
                _annotate_reward_reference(by_metric[name], config)
        agreements: dict[int, float] = {}
        for seed in seeds:
            # Agreement is recorded twice, once per direction, so retain one copy per case/future.
            seen: dict[tuple[str, str, str], float] = {}
            for row in group_rows:
                if int(row["seed"]) == seed:
                    seen[(str(row["history_regime"]), str(row["future_regime"]), str(row["pair_kind"]))] = float(row["replica_argmax_agreement"])
            if seen:
                agreements[seed] = float(sum(seen.values()) / len(seen))
        choices = {action: sum(row["selected_action"] == action for row in group_rows) for action in ACTIONS}
        total = sum(choices.values())
        return {
            "metrics": by_metric,
            "replica_argmax_agreement": _stat(agreements, families, config, phase),
            "choice_counts": {action: {"count": count, "fraction": count / total if total else None}
                              for action, count in choices.items()},
            "per_seed": [
                {"seed": seed, "family": families.get(seed),
                 **{name: next((r["value"] for r in by_metric[name]["per_seed"] if r["seed"] == seed), None)
                    for name in metric_names},
                 "argmax_agreement": next((r["value"] for r in _stat(agreements, families, config, phase)["per_seed"]
                                           if r["seed"] == seed), None),
                 "choice_counts": {action: sum(row["seed"] == seed and row["selected_action"] == action
                                               for row in group_rows) for action in ACTIONS}}
                for seed in seeds
            ],
        }

    primary = int(_mapping(config.get("report"), "config.report").get("primary_horizon", max(active["horizons"])))
    if primary not in active["horizons"]:
        primary = max(active["horizons"])
    primary_rows = [row for row in raw if row["horizon"] == primary]
    overall: dict[str, Any] = {}
    by_condition: dict[str, Any] = {}
    for pair in pairs:
        overall[pair] = summarize([row for row in primary_rows if row["pair_kind"] == pair])
        by_condition[pair] = {}
        for horizon in active["horizons"]:
            by_condition[pair][str(horizon)] = {}
            for future in futures:
                group = [row for row in raw if row["pair_kind"] == pair and row["horizon"] == horizon
                         and row["future_regime"] == future]
                by_condition[pair][str(horizon)][future] = summarize(group)
    return {
        "primary_horizon": primary,
        "observer_only": True,
        "stable_past_only_ranking": False,
        "same_replica_maximum_reported": False,
        "tie_rule": "ties within 1e-12 prefer keep, then declared action order",
        "metrics_by_pair_kind": overall,
        "by_horizon_relation_future": by_condition,
        "per_case_future_decisions": raw,
    }


def _outcomes(observations: Sequence[Mapping[str, Any]], families: Mapping[int, str | None],
              active: Mapping[str, Any], config: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    records = []
    seeds = [int(s) for s in active["seeds"]]
    for pair in active["pair_kinds"]:
        for horizon in active["horizons"]:
            for action in ACTIONS:
                for metric in ("reward_per_step", "cpu_seconds_per_1000_steps", "utility_rate"):
                    values: dict[int, float] = {}
                    for seed in seeds:
                        # First average replicas/futures within each history, then histories within seed.
                        by_history: dict[str, list[float]] = {}
                        for row in observations:
                            if (row["seed"], row["pair_kind"], row["horizon"], row["action"]) == (seed, pair, horizon, action) and row.get(metric) is not None:
                                by_history.setdefault(str(row["history_regime"]), []).append(float(row[metric]))
                        if by_history:
                            values[seed] = float(np.mean([np.mean(v) for v in by_history.values()]))
                    records.append({
                        "pair_kind": str(pair), "horizon": int(horizon), "action": action,
                        "metric": metric, "summary": _stat(values, families, config, phase),
                    })
    return records


def _find_effect(rows: Sequence[Mapping[str, Any]], pair: str, horizon: int, action: str) -> Mapping[str, Any]:
    return next(row for row in rows
                if row["pair_kind"] == pair and int(row["horizon"]) == horizon and row["action"] == action)


def _render_figure(output: Path, analysis: Mapping[str, Any]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = list(analysis["pair_kinds"])
    horizons = [int(h) for h in analysis["horizons"]]
    primary = int(analysis["primary_horizon"])
    colors = {"replace": "#0072B2", "mean": "#D55E00", "none": "#009E73"}
    markers = {pair: marker for pair, marker in zip(pairs, ("o", "s"))}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    ax_effect, ax_seed, ax_cross = axes

    for pair in pairs:
        for action in ACTIONS[1:]:
            rows = [_find_effect(analysis["action_effects"], pair, h, action) for h in horizons]
            means = [row["reward_per_step"]["mean"] for row in rows]
            ax_effect.plot(horizons, means, marker=markers[pair], color=colors[action],
                           linewidth=1.8, label=f"{pair}: {action} - keep")
            intervals = [row["reward_per_step"].get("interval") for row in rows]
            if all(isinstance(interval, list) and len(interval) == 2 for interval in intervals):
                ax_effect.fill_between(horizons, [v[0] for v in intervals], [v[1] for v in intervals],
                                       color=colors[action], alpha=0.12)
    ax_effect.axhline(0, color="#444444", linewidth=1)
    ax_effect.axhline(float(analysis["practical_reward_per_step_reference"]), color="#777777",
                      linestyle="--", linewidth=1, label="+0.002 reward reference")
    ax_effect.set_xticks(horizons, [f"{h:,}" for h in horizons])
    ax_effect.set_title("Paired reward-rate effect by horizon")
    ax_effect.set_xlabel("Nested continuation horizon (steps)")
    ax_effect.set_ylabel("Reward per step versus keep")
    ax_effect.grid(axis="y", alpha=0.25)
    ax_effect.legend(fontsize=7, frameon=False)

    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for pi, pair in enumerate(pairs):
        pair_label = pair.replace("_", "\n")
        for ai, action in enumerate(ACTIONS[1:]):
            x = pi * 4 + ai
            tick_positions.append(x)
            tick_labels.append(f"{pair_label}\n{action}")
            row = _find_effect(analysis["action_effects"], pair, primary, action)
            for item in row["reward_per_step"]["per_seed"]:
                color = "#4D4D4D" if item.get("family") == "A" else "#999999"
                jitter = (int(item["seed"]) % 3 - 1) * 0.045
                ax_seed.scatter([x + jitter], [item["value"]], marker=markers[pair], color=color, s=25, alpha=0.86)
    ax_seed.axhline(0, color="#444444", linewidth=1)
    ax_seed.axhline(float(analysis["practical_reward_per_step_reference"]), color="#777777", linestyle="--", linewidth=1)
    ax_seed.set_xticks(tick_positions, tick_labels)
    ax_seed.set_title(f"Per-seed gaps at {primary:,} steps")
    ax_seed.set_ylabel("Reward per step versus keep")
    ax_seed.grid(axis="y", alpha=0.25)
    if analysis["phase"] == "main":
        family_note = "dark: old family A; light: old family B"
    else:
        pilot_family = next(iter(analysis["seed_families"].values()))
        family_note = f"pilot old family {pilot_family}; no family interval"
    ax_seed.text(0.01, 0.99, family_note, transform=ax_seed.transAxes,
                 va="top", fontsize=8)

    cross = analysis["cross_replica_choice"]["metrics_by_pair_kind"]
    names = ("selected_minus_keep_reward_per_step", "selected_minus_replace_reward_per_step",
             "fixed_replace_minus_keep_reward_per_step")
    labels = ("Selected - keep", "Selected - fixed replace", "Fixed replace - keep")
    ccolors = ("#0072B2", "#D55E00", "#009E73")
    for pi, pair in enumerate(pairs):
        for ci, (name, label) in enumerate(zip(names, labels)):
            stat = cross[pair]["metrics"][name]
            if stat["mean"] is None:
                continue
            x = pi + (ci - 1) * 0.2
            ax_cross.scatter([x], [stat["mean"]], color=ccolors[ci], s=35,
                             label=label if pi == 0 else None, zorder=3)
            interval = stat.get("interval")
            if isinstance(interval, list):
                ax_cross.vlines(x, interval[0], interval[1], color=ccolors[ci], linewidth=2)
    ax_cross.axhline(0, color="#444444", linewidth=1)
    ax_cross.set_xticks(range(len(pairs)), pairs)
    ax_cross.set_title("Held-out replica choice comparison")
    ax_cross.set_ylabel("Reward-rate difference")
    ax_cross.grid(axis="y", alpha=0.25)
    ax_cross.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Retention mechanism diagnostic - {analysis['phase']}", fontsize=14)

    paths: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = output / f"figure.{suffix}"
        options: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            options["dpi"] = 240
        fig.savefig(path, **options)
        paths[suffix] = str(path.resolve())
    plt.close(fig)
    return paths


def _format(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "NA"


def _interval(stat: Mapping[str, Any]) -> str:
    interval = stat.get("interval")
    if not isinstance(interval, list) or len(interval) != 2:
        return "not estimated"
    return f"[{_format(interval[0])}, {_format(interval[1])}]"


def _markdown(analysis: Mapping[str, Any]) -> str:
    phase = analysis["phase"]
    audit = analysis["costs_recount"]
    uncertainty_text = (
        "Main intervals resample the six independent seeds within old-family A/B strata (three each). "
        "Histories, futures and replicas are averaged within each seed."
        if phase == "main" else
        "This one-seed pilot has no uncertainty interval. Main intervals use six independent seeds, "
        "stratified by old-family A/B (three each); histories, futures and replicas are averaged within seed."
    )
    lines = [
        "# Retention mechanism report", "",
        f"Phase: **{phase}**. Role: {analysis['analysis_role']}. This is an observer-only action and horizon diagnosis.", "",
        "The report reads frozen summaries and raw ledgers, makes no environment calls, and never feeds results to a learner.", "",
        "## Design and audit", "",
        f"- Cases: {analysis['case_count']}; branches: {analysis['branch_count']}; audited environment steps: {audit['total_environment_steps']:,}.",
        f"- Prefix steps: {audit['prefix_environment_steps']:,}; branch continuation steps: {audit['branch_environment_steps']:,}.",
        f"- SHA-256 checked raw ledger files: {audit['raw_ledger_count']}.",
        f"- Recounted PPO training: {audit['total_rollouts_trained']:,} completed rollouts and {audit['total_minibatch_updates']:,} minibatch updates.",
        f"- Restore, grid, horizon reward, rollout and optimizer-update audits passed: **{analysis['audit']['passed']}**.",
        "- Horizons are nested checkpoints within one continuation. They are not independent runs.",
        f"- {uncertainty_text}",
        "- Reward per step is primary. CPU seconds per 1,000 environment steps and utility rate remain separate outcomes.", "",
        "## Paired action effects", "",
        "Each operation is paired with keep from the same prefix, future and stochastic replica. The +0.002 reference is for reward per step only.", "",
        "| relation | horizon | contrast | seeds | reward / step difference | 95% descriptive interval | interval interpretation | CPU seconds / 1,000 step difference | utility-rate difference |",
        "|---|---:|---|---:|---:|---|---|---:|---:|",
    ]
    for row in analysis["action_effects"]:
        reward, cpu, utility = row["reward_per_step"], row["cpu_seconds_per_1000_steps"], row["utility_rate"]
        lines.append(f"| {row['pair_kind']} | {row['horizon']:,} | {row['contrast']} | {reward['seed_count']} | {_format(reward['mean'])} | {_interval(reward)} | {reward['practical_interval_interpretation']} | {_format(cpu['mean'])} | {_format(utility['mean'])} |")
    lines.extend(["", "## Future-prevalence interaction", "",
                  "For replace versus keep, this is the new-major future contrast minus the old-major future contrast, averaged across history within seed.", "",
                  "| relation | horizon | reward / step interaction | 95% descriptive interval | interval interpretation | seeds |",
                  "|---|---:|---:|---|---|---:|"])
    for row in analysis["future_prevalence_interaction"]:
        stat = row["reward_per_step"]
        lines.append(f"| {row['pair_kind']} | {row['horizon']:,} | {_format(stat['mean'])} | {_interval(stat)} | {stat['practical_interval_interpretation']} | {stat['seed_count']} |")
    lines.extend(["", "## All condition subgroups", "",
                  "Every relation, history, future, horizon and action contrast is retained below and in analysis.json.", "",
                  "| relation | history | future | horizon | contrast | seeds | reward / step difference | interval | interval interpretation |",
                  "|---|---|---|---:|---|---:|---:|---|---|"])
    for row in analysis["subgroup_effects"]:
        stat = row["reward_per_step"]
        lines.append(f"| {row['pair_kind']} | {row['history_regime']} | {row['future_regime']} | {row['horizon']:,} | {row['contrast']} | {stat['seed_count']} | {_format(stat['mean'])} | {_interval(stat)} | {stat['practical_interval_interpretation']} |")

    lines.extend(["", "## Cross-replica selection diagnostic", "",
                  "The best action is selected using reward on one stochastic continuation and scored on the other continuation from the same prefix. The direction is swapped. The fixed comparator is replace, frozen before this run. The choice is future-aware and diagnostic; it does not estimate a deployed past-only selector or policy value. No same-replica maximum or global oracle bound is reported.",
                  "This ranking should not be treated as a stable past-only action ranking. Ties within 1e-12 prefer keep, then config action order. Selection margins and replica argmax agreement describe repeatability; six seeds do not support a formal negative conclusion.", "",
                  "| relation | horizon | future | chosen - keep | interval | interval interpretation | chosen - fixed replace | interval | fixed replace - keep | interval | argmax agreement |",
                  "|---|---:|---|---:|---|---|---:|---|---:|---|---:|"])
    for pair, horizons_by_value in analysis["cross_replica_choice"]["by_horizon_relation_future"].items():
        for horizon, futures in horizons_by_value.items():
            for future, item in futures.items():
                stats = item["metrics"]
                a = stats["selected_minus_keep_reward_per_step"]
                b = stats["selected_minus_replace_reward_per_step"]
                c = stats["fixed_replace_minus_keep_reward_per_step"]
                g = item["replica_argmax_agreement"]
                lines.append(f"| {pair} | {int(horizon):,} | {future} | {_format(a['mean'])} | {_interval(a)} | {a['practical_interval_interpretation']} | {_format(b['mean'])} | {_interval(b)} | {_format(c['mean'])} | {_interval(c)} | {_format(g['mean'])} |")

    lines.extend(["", "## Per-seed cross-replica comparisons", "",
                  "Each seed contributes after averaging the two selection/evaluation directions and histories within its condition. Family labels are shown to make the six-seed distribution inspectable.", "",
                  "| relation | future | horizon | seed | old family | selected - keep | selected - fixed replace | fixed replace - keep | argmax agreement |",
                  "|---|---|---:|---:|---|---:|---:|---:|---:|"])
    for pair, horizons_by_value in analysis["cross_replica_choice"]["by_horizon_relation_future"].items():
        for horizon, futures in horizons_by_value.items():
            for future, item in futures.items():
                for seed_row in item["per_seed"]:
                    lines.append(
                        f"| {pair} | {future} | {int(horizon):,} | {seed_row['seed']} | "
                        f"{seed_row.get('family') or 'pilot'} | "
                        f"{_format(seed_row['selected_minus_keep_reward_per_step'])} | "
                        f"{_format(seed_row['selected_minus_replace_reward_per_step'])} | "
                        f"{_format(seed_row['fixed_replace_minus_keep_reward_per_step'])} | "
                        f"{_format(seed_row['argmax_agreement'])} |"
                    )

    lines.extend(["", "## Per-seed long-horizon records", "",
                  f"Each row reports a seed's paired reward-rate gap at {analysis['primary_horizon']:,} steps.", "",
                  "| relation | seed | old family | contrast | reward / step difference |",
                  "|---|---:|---|---|---:|"])
    for row in analysis["action_effects"]:
        if row["horizon"] == analysis["primary_horizon"]:
            for item in row["reward_per_step"]["per_seed"]:
                family = item.get("family") or "pilot"
                lines.append(f"| {row['pair_kind']} | {item['seed']} | {family} | {row['contrast']} | {_format(item['value'])} |")

    lines.extend(["", "## Cost ledger", "",
                  "Runner timing fields are preserved below; raw transition and rollout totals were independently recounted.", "",
                  "<details><summary>Runner cost JSON</summary>", "", "<pre>",
                  json.dumps(analysis["runner_costs"], indent=2, ensure_ascii=False), "</pre>", "</details>", "",
                  "## Figure", "", "![Retention mechanism diagnostic](figure.png)", "",
                  "## Interpretation", "",
                  f"The frozen practical reference is +{_format(analysis['practical_reward_per_step_reference'])} reward per step. An interval spanning useful and negligible effects is inconclusive, not evidence of no action value.",
                  "If no reproducible action gap appears under these candidates, prioritize candidate or representation diagnostics. If action gaps appear but cross-replica choices do not predict the other continuation reliably, prioritize an action-contrast diagnostic. The root records the gate decision after reviewing these results.", "",
                  "## Input hashes", "",
                  f"Summary SHA-256: {analysis['input_hashes']['summary_sha256']}  ",
                  f"Config canonical SHA-256: {analysis['input_hashes']['config_canonical_sha256']}  ",
                  f"Source manifest canonical SHA-256: {analysis['input_hashes']['source_manifest_canonical_sha256']}", "",
                  "## Software", "",
                  "The static multi-panel figure follows the object-oriented Matplotlib and vector-export guidance in Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, arXiv:2609.00065 (current arXiv version v2, revised 2 September 2026). https://arxiv.org/abs/2609.00065", ""])
    return "\n".join(lines)


def _input_hashes(summary_path: Path, summary_sha: str, summary: Mapping[str, Any],
                  config: Mapping[str, Any], raw_ledger_hashes: Mapping[str, str]) -> dict[str, Any]:
    del summary_path
    return {
        "summary_sha256": summary_sha,
        "config_canonical_sha256": _canonical_hash(config),
        "source_manifest_canonical_sha256": _canonical_hash(summary.get("source_manifest", {})),
        "raw_ledger_sha256": dict(sorted(raw_ledger_hashes.items())),
        "raw_ledger_hash_manifest_sha256": _canonical_hash(raw_ledger_hashes),
    }


def analyze(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    summary, summary_path, summary_sha = _load_summary(input_path)
    design, cases, branches, families = _design(summary)
    config, active, phase = design["config"], design["active"], design["phase"]
    horizons = [int(h) for h in active["horizons"]]
    prefix_steps = int(config.get("expected_main", {}).get("prefix_steps_per_case", 7168))
    observations, recount = _audit_ledgers(cases, branches, summary_path.parent, prefix_steps, horizons, config)
    runner_costs = _json_safe(_mapping(summary.get("costs"), "summary.costs"))
    recount["runner_costs"] = runner_costs
    action_effects, subgroup_effects, interactions = _effects(observations, families, config, active, phase)
    cross_replica = _cross_replica(observations, families, config, active, phase)
    report_config = _mapping(config.get("report"), "config.report")
    primary = int(report_config.get("primary_horizon", max(horizons)))
    if primary not in horizons:
        primary = max(horizons)
    expected = design["expected_counts"]
    analysis: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "analysis_role": str(config.get("analysis_role", "exploratory")),
        "phase": phase, "seeds": list(active["seeds"]), "pair_kinds": list(active["pair_kinds"]),
        "history_regimes": list(active["history_regimes"]), "future_regimes": list(active["future_regimes"]),
        "replicates": list(active["replicates"]), "actions": list(ACTIONS), "horizons": horizons,
        "primary_horizon": primary, "case_count": len(cases), "branch_count": len(branches),
        "seed_families": {str(seed): family for seed, family in sorted(families.items())},
        "statistical_unit": "independent seed after averaging histories, futures, and replicas",
        "action_effects": action_effects, "subgroup_effects": subgroup_effects,
        "future_prevalence_interaction": interactions, "cross_replica_choice": cross_replica,
        "outcomes_by_seed": _outcomes(observations, families, active, config, phase),
        "branch_horizon_rows": observations, "costs_recount": recount, "runner_costs": runner_costs,
        "audit": {
            "passed": True, "raw_rows_and_rewards_recounted": True, "raw_ledger_sha256_verified": True,
            "rollout_rows_and_optimizer_updates_recounted": True, "prefix_restore_hashes_verified": True,
            "case_and_branch_grids_complete": True,
            "actual_steps_match_frozen_design": recount["total_environment_steps"] == expected["total_steps"],
        },
        "practical_reward_per_step_reference": float(_number(
            report_config.get("practical_reward_per_step"), "config.report.practical_reward_per_step")),
        "input_hashes": _input_hashes(summary_path, summary_sha, summary, config, recount["raw_ledger_hashes"]),
        "reporter_sha256": _file_hash(Path(__file__).resolve()),
        "interpretation_limits": [
            "The report diagnoses fixed operations and horizons; it does not train a selector.",
            "The four actions use the specified constructed candidate and do not bound global candidate quality.",
            "Replicas, histories, futures, and nested horizons are not additional independent deployment seeds.",
            "Cross-replica future-aware choices are observer diagnostics, not a deployed past-only selector or off-policy value estimate.",
            "Cross-replica action rankings are not evidence of stable past-only rankings.",
            "No same-replica naive maximum or global oracle upper bound is reported.",
            "A six-seed interval spanning useful and negligible effects is inconclusive, not formal evidence of no action value.",
            "The +0.002 reference applies to reward per step only, not CPU rate or utility rate.",
        ],
    }
    output = Path(output_path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"preserve existing report output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    analysis["figure"] = _render_figure(output, analysis)
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
        "phase": analysis["phase"], "cases": analysis["case_count"], "branches": analysis["branch_count"],
        "steps": analysis["costs_recount"]["total_environment_steps"],
        "audit_passed": analysis["audit"]["passed"], "output": str(Path(args.output).resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

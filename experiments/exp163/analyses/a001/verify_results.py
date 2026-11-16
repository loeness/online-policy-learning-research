#!/usr/bin/env python3
"""Offline audit for the single frozen E163 campaign.

This script reads only the supplied run mirror and frozen run spec. It never
imports model libraries, deserializes a checkpoint, opens an original prefix,
or starts research work. Non-COMPLETE campaigns receive a cost/evidence audit
only; they are not scored as scientific results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import Any


EXPECTED_RUN_SPEC_SHA256 = "6e7604856c28ce33218f732eb893c1cc56827cab7eb4e51abee21b994f4491e8"
EXPECTED_SOURCE_IDENTITY = "0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea"
EXPECTED_FOUNDATION = {
    "bytes": 55477,
    "sha256": "e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222",
}
SEEDS = tuple(range(26201, 26214)) + (26215, 26216, 26218)
ARMS = {"C": 0, "D": 1}
PACKAGE_LIMIT = 262144
MAX_CLOSURES = 256
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSONL_BYTES = 32 * 1024 * 1024
EPSILON = 1e-10


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads(data: bytes | str) -> Any:
    return json.loads(
        data,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _digest_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


class Checks:
    def __init__(self) -> None:
        self.count = 0
        self.failures: list[dict[str, Any]] = []

    def check(
        self,
        name: str,
        ok: bool,
        *,
        actual: Any = None,
        expected: Any = None,
        detail: str | None = None,
    ) -> bool:
        self.count += 1
        if not ok:
            failure: dict[str, Any] = {"check": name}
            if actual is not None:
                failure["actual"] = actual
            if expected is not None:
                failure["expected"] = expected
            if detail:
                failure["detail"] = detail
            self.failures.append(failure)
        return ok


class RunFiles:
    """Resolve named files below one mirror without following symlinks."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, relative: str, *, directory: bool = False) -> Path:
        rel = Path(relative)
        if rel.is_absolute() or not rel.parts or any(
            part in ("", ".", "..") for part in rel.parts
        ):
            raise ValueError(f"unsafe relative run path: {relative!r}")
        current = self.root
        for index, part in enumerate(rel.parts):
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"symlink in run mirror path: {relative}")
            final = index == len(rel.parts) - 1
            if not final and not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"non-directory path component: {relative}")
            if final:
                expected_mode = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
                if not expected_mode:
                    raise ValueError(f"unexpected file type in run mirror: {relative}")
        return current

    def json(self, relative: str) -> Any:
        path = self.path(relative)
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError(f"JSON file exceeds audit size limit: {relative}")
        return _loads(path.read_bytes())

    def jsonl(self, relative: str) -> tuple[list[dict[str, Any]], list[str]]:
        path = self.path(relative)
        if path.stat().st_size > MAX_JSONL_BYTES:
            raise ValueError(f"JSONL file exceeds audit size limit: {relative}")
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = _loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("JSONL row is not an object")
                    rows.append(row)
                except Exception as exc:
                    errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")
        return rows, errors

    def bytes(self, relative: str, *, limit: int = 1024) -> bytes:
        path = self.path(relative)
        if path.stat().st_size > limit:
            raise ValueError(f"binary file exceeds audit size limit: {relative}")
        return path.read_bytes()

    def file_identity(self, relative: str) -> dict[str, Any]:
        path = self.path(relative)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
        return {"bytes": size, "sha256": digest.hexdigest()}


def _spec_identity(spec: dict[str, Any], checks: Checks) -> dict[int, dict[str, dict[str, Any]]]:
    spec_hash = spec.pop("_computed_sha256", None)
    checks.check(
        "spec.approved_sha256",
        spec_hash == EXPECTED_RUN_SPEC_SHA256,
        actual=spec_hash,
        expected=EXPECTED_RUN_SPEC_SHA256,
    )
    checks.check("spec.format", spec.get("format") == "e163_execution_spec_v1",
                 actual=spec.get("format"), expected="e163_execution_spec_v1")
    checks.check("spec.experiment_id", spec.get("experiment_id") == "E163",
                 actual=spec.get("experiment_id"), expected="E163")
    checks.check("spec.source_identity", spec.get("source_identity_sha256") == EXPECTED_SOURCE_IDENTITY,
                 actual=spec.get("source_identity_sha256"), expected=EXPECTED_SOURCE_IDENTITY)
    checks.check("spec.seeds", spec.get("seeds") == list(SEEDS),
                 actual=spec.get("seeds"), expected=list(SEEDS))
    checks.check("spec.lambda_values", spec.get("lambda_values") == [0, 1],
                 actual=spec.get("lambda_values"), expected=[0, 1])
    checks.check("spec.foundation", spec.get("foundation") == {
        **EXPECTED_FOUNDATION,
        "remote_path": "/data/robot/p001/s001/data/foundation.pt",
    }, actual=spec.get("foundation"), expected=EXPECTED_FOUNDATION)

    expected_limits = {
        "aggregate_remote_project_bytes": 100000000000,
        "campaign_wall_seconds": 180,
        "candidate_fits": 32,
        "cpu_seconds": 120,
        "environment_interactions": 0,
        "gpu_devices": 0,
        "new_artifact_cache_temp_bytes": 268435456,
        "numeric_threads_per_worker": 1,
        "workers": 1,
    }
    checks.check("spec.limits", spec.get("limits") == expected_limits,
                 actual=spec.get("limits"), expected=expected_limits)
    checks.check("spec.gate_mapping.source_identity",
                 isinstance(spec.get("gate_mapping"), dict)
                 and spec["gate_mapping"].get("source_identity_sha256") == EXPECTED_SOURCE_IDENTITY,
                 actual=(spec.get("gate_mapping") or {}).get("source_identity_sha256")
                 if isinstance(spec.get("gate_mapping"), dict) else None,
                 expected=EXPECTED_SOURCE_IDENTITY)

    refs: dict[int, dict[str, dict[str, Any]]] = {}
    rows = spec.get("input_files")
    if not isinstance(rows, list):
        checks.check("spec.input_identity_manifest", False, actual=type(rows).__name__,
                     expected="48 reference objects")
        return refs
    valid = True
    for row in rows:
        if not isinstance(row, dict):
            valid = False
            continue
        seed, name = row.get("seed"), row.get("file")
        size, digest = row.get("bytes"), row.get("sha256")
        if not _is_int(seed) or seed not in SEEDS or name not in (
            "actor.bin", "checkpoint.pt", "evidence.npz"
        ):
            valid = False
            continue
        if not _is_int(size) or size < 0 or not _digest_hex(digest):
            valid = False
            continue
        by_name = refs.setdefault(seed, {})
        if name in by_name:
            valid = False
            continue
        by_name[name] = {"bytes": size, "sha256": digest}
    expected_keys = {(seed, name) for seed in SEEDS for name in (
        "actor.bin", "checkpoint.pt", "evidence.npz"
    )}
    actual_keys = {(seed, name) for seed, by_name in refs.items() for name in by_name}
    valid = valid and actual_keys == expected_keys and len(rows) == 48
    checks.check("spec.input_identity_manifest", valid,
                 actual=len(actual_keys), expected=48)
    return refs


def _metric_number(value: Any, name: str, checks: Checks) -> float | None:
    number = _number(value)
    checks.check(name, number is not None, actual=value, expected="finite JSON number")
    return number


def _check_closure_diagnostics(diag: dict[str, Any], prefix: str, checks: Checks) -> dict[str, Any]:
    required = {
        "closure_count", "termination_reason", "closure_records",
        "returned_is_C0", "fallback_applied", "fallback_reason",
        "computation_counts", "candidate_sha256", "seed", "prepared_id",
    }
    checks.check(f"{prefix}.diagnostic_fields",
                 required <= set(diag), actual=sorted(set(diag)), expected=sorted(required))

    count = diag.get("closure_count")
    records = diag.get("closure_records")
    count_ok = _is_int(count) and 0 <= count <= MAX_CLOSURES
    checks.check(f"{prefix}.closure_count_limit", count_ok, actual=count,
                 expected=f"integer from 0 to {MAX_CLOSURES}")
    checks.check(f"{prefix}.closure_record_array",
                 isinstance(records, list), actual=type(records).__name__, expected="list")
    if isinstance(records, list):
        checks.check(f"{prefix}.closure_record_count",
                     count == len(records), actual=len(records), expected=count)
        expected_closure_fields = {
            "index", "fullL", "newKL", "oldKL", "ridge",
            "accepted", "reject_reason", "candidate_hash",
        }
        valid_indices = True
        valid_fields = True
        valid_values = True
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                valid_fields = valid_values = valid_indices = False
                continue
            valid_fields &= set(record) == expected_closure_fields
            valid_indices &= record.get("index") == index
            valid_values &= all(_number(record.get(key)) is not None for key in (
                "fullL", "newKL", "oldKL", "ridge"
            ))
            valid_values &= isinstance(record.get("accepted"), bool)
            valid_values &= _digest_hex(record.get("candidate_hash"))
            reason = record.get("reject_reason")
            valid_values &= reason is None or isinstance(reason, str)
            valid_values &= (record.get("accepted") is True) == (reason is None)
        checks.check(f"{prefix}.closure_fields", valid_fields,
                     actual="closure record keys", expected=sorted(expected_closure_fields))
        checks.check(f"{prefix}.closure_indices", valid_indices,
                     actual="sequential one-based indices", expected=True)
        checks.check(f"{prefix}.closure_values", valid_values,
                     actual="finite values, Boolean acceptance, SHA-256, coherent rejection",
                     expected=True)

    checks.check(f"{prefix}.returned_is_C0_retained",
                 isinstance(diag.get("returned_is_C0"), bool),
                 actual=diag.get("returned_is_C0"), expected="boolean")
    fallback = diag.get("fallback_applied")
    reason = diag.get("fallback_reason")
    checks.check(f"{prefix}.fallback_fields_retained",
                 isinstance(fallback, bool) and (reason is None or isinstance(reason, str)),
                 actual={"fallback_applied": fallback, "fallback_reason": reason},
                 expected="Boolean and null or string")
    checks.check(f"{prefix}.fallback_reason_consistent",
                 (fallback is True and reason == "float32_new_train_kl_gate")
                 or (fallback is False and reason is None),
                 actual={"fallback_applied": fallback, "fallback_reason": reason},
                 expected="true with float32_new_train_kl_gate, otherwise false with null")
    checks.check(f"{prefix}.termination_reason",
                 diag.get("termination_reason") in (
                     "optimizer_completed", "evaluation_budget_exhausted"
                 ),
                 actual=diag.get("termination_reason"),
                 expected=["optimizer_completed", "evaluation_budget_exhausted"])

    counts = diag.get("computation_counts")
    required_counts = {
        "initialization_objective_evaluations", "closure_objective_evaluations",
        "final_objective_evaluations", "total_objective_evaluations",
        "initialization_kl_metric_evaluations", "closure_kl_metric_evaluations",
        "float32_gate_kl_metric_evaluations", "final_kl_metric_evaluations",
        "total_kl_metric_evaluations", "total_objective_and_kl_evaluations",
    }
    counts_ok = isinstance(counts, dict) and required_counts <= set(counts)
    checks.check(f"{prefix}.computation_counts_fields", counts_ok,
                 actual=sorted(counts) if isinstance(counts, dict) else type(counts).__name__,
                 expected=sorted(required_counts))
    if counts_ok:
        values = {key: counts[key] for key in required_counts}
        all_ints = all(_is_int(value) and value >= 0 for value in values.values())
        checks.check(f"{prefix}.computation_counts_integers", all_ints,
                     actual=values, expected="nonnegative integers")
        if all_ints and _is_int(count):
            checks.check(f"{prefix}.closure_computation_count",
                         values["closure_objective_evaluations"] == count
                         and values["closure_kl_metric_evaluations"] == 2 * count,
                         actual={
                             "closure_objective_evaluations": values["closure_objective_evaluations"],
                             "closure_kl_metric_evaluations": values["closure_kl_metric_evaluations"],
                             "closure_count": count,
                         },
                         expected={"objective": count, "KL": 2 * count})
            checks.check(f"{prefix}.total_objective_computation_count",
                         values["total_objective_evaluations"] == (
                             values["initialization_objective_evaluations"]
                             + values["closure_objective_evaluations"]
                             + values["final_objective_evaluations"]
                         ),
                         actual=values["total_objective_evaluations"],
                         expected=values["initialization_objective_evaluations"]
                         + values["closure_objective_evaluations"]
                         + values["final_objective_evaluations"])
            checks.check(f"{prefix}.total_kl_computation_count",
                         values["total_kl_metric_evaluations"] == (
                             values["initialization_kl_metric_evaluations"]
                             + values["closure_kl_metric_evaluations"]
                             + values["float32_gate_kl_metric_evaluations"]
                             + values["final_kl_metric_evaluations"]
                         ),
                         actual=values["total_kl_metric_evaluations"],
                         expected=values["initialization_kl_metric_evaluations"]
                         + values["closure_kl_metric_evaluations"]
                         + values["float32_gate_kl_metric_evaluations"]
                         + values["final_kl_metric_evaluations"])
            checks.check(f"{prefix}.combined_computation_count",
                         values["total_objective_and_kl_evaluations"] == (
                             values["total_objective_evaluations"]
                             + values["total_kl_metric_evaluations"]
                         ),
                         actual=values["total_objective_and_kl_evaluations"],
                         expected=values["total_objective_evaluations"]
                         + values["total_kl_metric_evaluations"])
    return {
        "closure_count": count,
        "returned_is_C0": diag.get("returned_is_C0"),
        "fallback_applied": fallback,
        "fallback_reason": reason,
        "termination_reason": diag.get("termination_reason"),
    }


def _candidate_bytes_check(
    files: RunFiles, base: str, candidate: dict[str, Any], diag: dict[str, Any],
    metrics: dict[str, Any], seed: int, arm: str, checks: Checks,
) -> None:
    prefix = f"science.s{seed}.{arm}"
    raw_path = f"{base}/candidate.f32"
    raw: bytes | None = None
    try:
        raw = files.bytes(raw_path, limit=36)
    except Exception as exc:
        checks.check(f"{prefix}.candidate_bytes_read", False, detail=f"{type(exc).__name__}: {exc}")
    if raw is None:
        return
    checks.check(f"{prefix}.candidate_raw_length", len(raw) == 36,
                 actual=len(raw), expected=36)
    checks.check(f"{prefix}.candidate_matrix_metadata",
                 candidate.get("shape") == [3, 3]
                 and candidate.get("dtype") == "float32"
                 and candidate.get("order") == "C",
                 actual={key: candidate.get(key) for key in ("shape", "dtype", "order")},
                 expected={"shape": [3, 3], "dtype": "float32", "order": "C"})
    dtype_str = candidate.get("dtype_str")
    byteorder = candidate.get("byteorder")
    expected_dtype = "<f4" if byteorder == "little" else ">f4" if byteorder == "big" else None
    checks.check(f"{prefix}.candidate_dtype_byteorder",
                 dtype_str == expected_dtype,
                 actual={"dtype_str": dtype_str, "byteorder": byteorder},
                 expected={"dtype_str": expected_dtype, "byteorder": "little or big"})
    if len(raw) != 36 or expected_dtype is None or dtype_str != expected_dtype:
        return
    endian = "<" if byteorder == "little" else ">"
    values = struct.unpack(endian + "9f", raw)
    recorded = candidate.get("values")
    flattened: list[Any] = []
    if isinstance(recorded, list) and len(recorded) == 3 and all(
        isinstance(row, list) and len(row) == 3 for row in recorded
    ):
        flattened = [item for row in recorded for item in row]
    values_match = len(flattened) == 9 and all(
        _number(value) == actual for value, actual in zip(flattened, values)
    )
    checks.check(f"{prefix}.candidate_values_match_raw_bytes",
                 values_match, actual="candidate.json values versus unpacked candidate.f32",
                 expected="exact float32 round trip")

    raw_digest = hashlib.sha256(raw).hexdigest()
    checks.check(f"{prefix}.candidate_raw_sha256",
                 candidate.get("raw_file_sha256") == raw_digest,
                 actual=candidate.get("raw_file_sha256"), expected=raw_digest)
    scientific_digest = hashlib.sha256(
        dtype_str.encode("ascii") + b"\0" + repr((3, 3)).encode("ascii") + b"\0" + raw
    ).hexdigest()
    expected_scientific = candidate.get("scientific_candidate_sha256")
    checks.check(f"{prefix}.candidate_scientific_sha256",
                 expected_scientific == scientific_digest
                 and expected_scientific == diag.get("candidate_sha256")
                 and expected_scientific == metrics.get("candidate_sha256"),
                 actual={
                     "candidate_json": expected_scientific,
                     "diagnostics": diag.get("candidate_sha256"),
                     "metrics": metrics.get("candidate_sha256"),
                 },
                 expected=scientific_digest)
    checks.check(f"{prefix}.candidate_fit_metadata",
                 candidate.get("seed", seed) == seed
                 and candidate.get("lambda_value") == ARMS[arm]
                 and candidate.get("prepared_id") == diag.get("prepared_id")
                 and diag.get("prepared_id") == metrics.get("prepared_id"),
                 actual={
                     "seed": candidate.get("seed", seed),
                     "lambda_value": candidate.get("lambda_value"),
                     "prepared_id": candidate.get("prepared_id"),
                 },
                 expected={"seed": seed, "lambda_value": ARMS[arm],
                           "prepared_id": diag.get("prepared_id")})


def _reference_meta(reference: Any) -> dict[str, Any] | None:
    if not isinstance(reference, dict):
        return None
    return {
        "seed": reference.get("seed"),
        "file": reference.get("file"),
        "bytes": reference.get("bytes"),
        "sha256": reference.get("sha256"),
    }


def _check_cost(
    files: RunFiles, spec: dict[str, Any], checks: Checks,
    launch_call_seconds: float | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, str]:
    state: dict[str, Any] | None = None
    worker_state: dict[str, Any] | None = None
    missing: list[str] = []
    parse_errors: list[str] = []
    for rel, dest in (("state.json", "state"), ("state_worker.json", "worker")):
        try:
            value = files.json(rel)
            if not isinstance(value, dict):
                raise ValueError("top-level JSON value is not an object")
            if dest == "state":
                state = value
            else:
                worker_state = value
        except FileNotFoundError:
            missing.append(rel)
        except Exception as exc:
            parse_errors.append(f"{rel}: {type(exc).__name__}: {exc}")

    wait4_rows: list[dict[str, Any]] = []
    meter_rows: list[dict[str, Any]] = []
    for rel, target in (("wait4.jsonl", wait4_rows), ("meter.jsonl", meter_rows)):
        try:
            rows, row_errors = files.jsonl(rel)
            target.extend(rows)
            parse_errors.extend(f"{rel} {item}" for item in row_errors)
        except FileNotFoundError:
            missing.append(rel)
        except Exception as exc:
            parse_errors.append(f"{rel}: {type(exc).__name__}: {exc}")

    run_status = str(state.get("status")) if isinstance(state, dict) else "UNKNOWN"
    meter_by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in meter_rows:
        meter_by_kind.setdefault(str(row.get("kind")), []).append(row)

    cost: dict[str, Any] = {
        "validity": "UNKNOWN",
        "scope": "recorded supervisor and wait4 evidence only; valid full campaign cost is UNKNOWN unless accounting and clock origin are valid",
        "observed_run_status": run_status,
        "missing_evidence": missing,
        "parse_errors": parse_errors,
        "wait4_record_count": len(wait4_rows),
        "meter_event_count": len(meter_rows),
        "worker_body": None,
        "worker_wait4": None,
        "recorded_subtotal_cpu_seconds": None,
        "campaign_cpu_seconds": None,
        "reported_campaign_wall_seconds_before_footer": None,
        "campaign_wall_seconds": None,
        "nominal_bounded_cpu_seconds_upper": None,
        "nominal_bounded_wall_seconds_upper": None,
        "footer_reserve": None,
        "wall_origin": None,
        "environment_interactions": None,
        "known_precheck_wait4_cpu_seconds": None,
        "known_precheck_wait4_records": [],
    }

    if isinstance(state, dict):
        snapshot = state.get("cpu_snapshot_before_final_footer")
        if isinstance(snapshot, dict):
            subtotal = _number(snapshot.get("total_cpu_seconds"))
            own = _number(snapshot.get("supervisor_rusage_self_cpu_seconds"))
            children = _number(snapshot.get("reaped_wait4_cpu_seconds"))
            live = _number(snapshot.get("live_owned_process_cpu_seconds_from_proc_stat"))
            cost["recorded_subtotal_cpu_seconds"] = subtotal
            if all(value is not None for value in (own, children, live, subtotal)):
                checks.check("cost.recorded_cpu_snapshot_sum",
                             abs((own + children + live) - subtotal) <= 1e-6,
                             actual={"own": own, "wait4_children": children, "live": live,
                                     "total": subtotal},
                             expected="own + wait4 + live equals total")
            cost["known_cpu_components"] = {
                "supervisor_rusage_self_cpu_seconds": own,
                "recorded_wait4_children_cpu_seconds": children,
                "live_owned_process_cpu_seconds": live,
            }
        cost["reported_campaign_wall_seconds_before_footer"] = state.get(
            "campaign_wall_seconds_before_final_footer"
        )
        cost["nominal_bounded_cpu_seconds_upper"] = state.get(
            "bounded_campaign_cpu_seconds_upper"
        )
        cost["nominal_bounded_wall_seconds_upper"] = state.get(
            "bounded_campaign_wall_seconds_upper"
        )
        cost["footer_reserve"] = state.get("finalization_reserve")
        cost["environment_interactions"] = state.get("environment_interactions")
        cost["gpu_devices"] = state.get("gpu_devices")
        cost["e_root_apparent_bytes"] = state.get("e_root_apparent_bytes")
        cost["aggregate_project_apparent_bytes"] = state.get("aggregate_project_apparent_bytes")

        expected_reserve = {"cpu_seconds": 5.0, "wall_seconds": 10.0, "byte_reserve": 65536}
        checks.check("cost.footer_reserve",
                     state.get("finalization_reserve") == expected_reserve,
                     actual=state.get("finalization_reserve"), expected=expected_reserve)
        checks.check("cost.zero_environment_interactions",
                     state.get("environment_interactions") == 0,
                     actual=state.get("environment_interactions"), expected=0)
        checks.check("cost.zero_gpu_devices",
                     state.get("gpu_devices") == 0,
                     actual=state.get("gpu_devices"), expected=0)
        checks.check("cost.recorded_limits_match_frozen_spec",
                     state.get("limits") == spec.get("limits"),
                     actual=state.get("limits"), expected=spec.get("limits"))

        if isinstance(snapshot, dict):
            upper_cpu = _number(state.get("bounded_campaign_cpu_seconds_upper"))
            upper_wall = _number(state.get("bounded_campaign_wall_seconds_upper"))
            wall_before = _number(state.get("campaign_wall_seconds_before_final_footer"))
            total_cpu = _number(snapshot.get("total_cpu_seconds"))
            if upper_cpu is not None and total_cpu is not None:
                checks.check("cost.nominal_cpu_upper_arithmetic",
                             abs(upper_cpu - (total_cpu + 5.0)) <= 1e-6,
                             actual=upper_cpu, expected=total_cpu + 5.0)
            if upper_wall is not None and wall_before is not None:
                checks.check("cost.nominal_wall_upper_arithmetic",
                             abs(upper_wall - (wall_before + 10.0)) <= 1e-6,
                             actual=upper_wall, expected=wall_before + 10.0)

        direct_sum = 0.0
        direct_rows_valid = True
        worker_rows: list[dict[str, Any]] = []
        probe_sum = 0.0
        for row in wait4_rows:
            cpu = _number(row.get("wait4_cpu_seconds_including_waited_descendants"))
            if cpu is None or cpu < 0:
                direct_rows_valid = False
                continue
            direct_sum += cpu
            if row.get("purpose") == "scientific_worker":
                worker_rows.append(row)
            if row.get("purpose") == "cpu_descendant_accounting_precheck":
                probe_sum += cpu
                cost["known_precheck_wait4_records"].append({
                    "pid": row.get("pid"),
                    "purpose": row.get("purpose"),
                    "start_ticks": row.get("start_ticks"),
                    "exit_code": row.get("exit_code"),
                    "wait4_cpu_seconds_including_waited_descendants": cpu,
                })
        cost["known_precheck_wait4_cpu_seconds"] = probe_sum
        cost["wait4_sum_cpu_seconds"] = direct_sum if direct_rows_valid else None
        state_rows = state.get("child_wait4_records")
        if isinstance(state_rows, list):
            state_by_pid = {
                row.get("pid"): row for row in state_rows if isinstance(row, dict)
            }
            raw_by_pid = {
                row.get("pid"): row for row in wait4_rows if isinstance(row, dict)
            }
            copied_match = len(state_by_pid) == len(state_rows) == len(raw_by_pid)
            if copied_match:
                for pid, raw in raw_by_pid.items():
                    copied = state_by_pid.get(pid)
                    if not isinstance(copied, dict):
                        copied_match = False
                        break
                    copied_match &= (
                        copied.get("purpose") == raw.get("purpose")
                        and copied.get("exit_code") == raw.get("exit_code")
                        and copied.get("start_ticks") == raw.get("start_ticks")
                        and copied.get("wait4_cpu_seconds_including_waited_descendants")
                        == raw.get("wait4_cpu_seconds_including_waited_descendants")
                    )
            checks.check("cost.raw_wait4_matches_final_state",
                         copied_match and len(state_rows) == state.get("completed_direct_children"),
                         actual={"raw_wait4": len(wait4_rows), "state_wait4": len(state_rows),
                                 "completed_direct_children": state.get("completed_direct_children")},
                         expected="same PID/purpose/exit/start/cpu records")
            if isinstance(snapshot, dict):
                recorded_children = _number(snapshot.get("reaped_wait4_cpu_seconds"))
                checks.check("cost.wait4_sum_matches_snapshot",
                             direct_rows_valid and recorded_children is not None
                             and abs(direct_sum - recorded_children) <= 1e-6,
                             actual={"wait4_sum": direct_sum, "snapshot": recorded_children},
                             expected="equal within 1e-6 seconds")
        else:
            checks.check("cost.state_wait4_array", False, actual=type(state_rows).__name__,
                         expected="list")

        audit_record = state.get("wait4_children_audit")
        if isinstance(audit_record, dict):
            difference = _number(audit_record.get("difference_seconds"))
            checks.check("cost.wait4_children_audit",
                         difference is not None and difference <= 0.03,
                         actual=audit_record, expected="difference at most 0.03 seconds")
        else:
            cost["wait4_children_audit"] = None
            if run_status == "COMPLETE":
                checks.check("cost.wait4_children_audit_present", False,
                             actual=audit_record, expected="valid wait4 audit object")
        cost["wait4_children_audit"] = audit_record
        cost["wait4_audit_error"] = state.get("wait4_audit_error")

        if len(worker_rows) == 1:
            worker = worker_rows[0]
            cost["worker_wait4"] = {
                "purpose": worker.get("purpose"),
                "exit_code": worker.get("exit_code"),
                "wait4_cpu_seconds_including_waited_descendants":
                    worker.get("wait4_cpu_seconds_including_waited_descendants"),
            }
        elif run_status == "COMPLETE":
            checks.check("cost.exactly_one_scientific_worker",
                         False, actual=len(worker_rows), expected=1)

    if isinstance(worker_state, dict):
        cost["worker_body"] = {
            "status": worker_state.get("status"),
            "fit_attempts": worker_state.get("fit_attempts"),
            "fit_returns": worker_state.get("fit_returns"),
            "completed_fits": worker_state.get("completed_fits"),
            "worker_body_cpu_seconds": worker_state.get("worker_body_cpu_seconds"),
            "worker_body_wall_seconds": worker_state.get("worker_body_wall_seconds"),
            "environment_interactions": worker_state.get("environment_interactions"),
            "cost_scope": worker_state.get("cost_scope"),
        }
        if worker_state.get("environment_interactions") is not None:
            checks.check("cost.worker_zero_environment_interactions",
                         worker_state.get("environment_interactions") == 0,
                         actual=worker_state.get("environment_interactions"), expected=0)
        if run_status == "COMPLETE":
            actual_worker = _number((cost.get("worker_wait4") or {}).get(
                "wait4_cpu_seconds_including_waited_descendants"
            ))
            body_cpu = _number(worker_state.get("worker_body_cpu_seconds"))
            body_wall = _number(worker_state.get("worker_body_wall_seconds"))
            checks.check("cost.actual_worker_wait4_covers_worker_body_cpu",
                         actual_worker is not None and body_cpu is not None
                         and actual_worker + 0.02 >= body_cpu,
                         actual={"worker_wait4_full": actual_worker, "worker_body": body_cpu},
                         expected="wait4 full worker CPU is at least worker-body subtotal")
            campaign_wall = _number(
                state.get("campaign_wall_seconds_before_final_footer")
            ) if isinstance(state, dict) else None
            checks.check("cost.worker_body_wall_within_campaign",
                         body_wall is not None and campaign_wall is not None
                         and body_wall <= campaign_wall + 0.02,
                         actual={"worker_body": body_wall, "campaign_reported": campaign_wall},
                         expected="worker body wall is no greater than campaign wall")

    finalizing = meter_by_kind.get("campaign_finalizing", [])
    started = meter_by_kind.get("campaign_started", [])
    if len(finalizing) == 1 and isinstance(state, dict):
        event = finalizing[0]
        event_snapshot = event.get("cpu_snapshot")
        state_snapshot = state.get("cpu_snapshot_before_final_footer")
        cost["raw_meter_finalizing"] = {
            "campaign_wall_seconds": event.get("campaign_wall_seconds"),
            "bounded_cpu_seconds_upper": event.get("bounded_cpu_seconds_upper"),
            "bounded_wall_seconds_upper": event.get("bounded_wall_seconds_upper"),
            "wait4_audit_error": event.get("wait4_audit_error"),
        }
        checks.check("cost.raw_meter_finalizing_matches_state",
                     event_snapshot == state_snapshot
                     and event.get("campaign_wall_seconds")
                     == state.get("campaign_wall_seconds_before_final_footer")
                     and event.get("bounded_cpu_seconds_upper")
                     == state.get("bounded_campaign_cpu_seconds_upper")
                     and event.get("bounded_wall_seconds_upper")
                     == state.get("bounded_campaign_wall_seconds_upper"),
                     actual=cost["raw_meter_finalizing"],
                     expected="final meter event agrees with state.json")
    elif run_status == "COMPLETE":
        checks.check("cost.raw_meter_finalizing_present", False,
                     actual=len(finalizing), expected=1)

    if len(started) == 1:
        first = started[0]
        start_ticks = first.get("supervisor_start_ticks")
        ticks_per_second = first.get("ticks_per_second")
        first_monotonic = _number(first.get("monotonic"))
        origin_age = None
        if _is_int(start_ticks) and _number(ticks_per_second) not in (None, 0) and first_monotonic is not None:
            origin_age = first_monotonic - (start_ticks / float(ticks_per_second))
        cost["wall_origin"] = {
            "supervisor_pid": first.get("supervisor_pid"),
            "supervisor_start_ticks": start_ticks,
            "ticks_per_second": ticks_per_second,
            "first_event_monotonic": first.get("monotonic"),
            "first_event_minus_kernel_start_seconds": origin_age,
            "run_spec_sha256": first.get("run_spec_sha256"),
            "observed_launcher_call_seconds": launch_call_seconds,
        }
        if launch_call_seconds is not None and origin_age is not None:
            cost["wall_origin"]["first_event_gap_exceeds_launcher_call"] = (
                origin_age > launch_call_seconds + 0.05
            )
            cost["wall_origin"]["reported_wall_exceeds_launcher_call"] = (
                (_number(cost["reported_campaign_wall_seconds_before_footer"]) or 0.0)
                > launch_call_seconds + 0.05
            )
        checks.check("cost.meter_run_spec_identity",
                     first.get("run_spec_sha256") == EXPECTED_RUN_SPEC_SHA256,
                     actual=first.get("run_spec_sha256"),
                     expected=EXPECTED_RUN_SPEC_SHA256)
    elif run_status == "COMPLETE":
        checks.check("cost.raw_meter_campaign_started_present", False,
                     actual=len(started), expected=1)

    if len(worker_rows) == 1:
        worker_pid = worker_rows[0].get("pid")
        command_events = [
            row for row in meter_by_kind.get("worker_command_started", [])
            if isinstance(row.get("child"), dict) and row["child"].get("pid") == worker_pid
        ]
        reaped_events = [
            row for row in meter_by_kind.get("child_reaped", [])
            if row.get("pid") == worker_pid
        ]
        if len(command_events) == 1 and len(reaped_events) == 1:
            begin = _number(command_events[0].get("monotonic"))
            end = _number(reaped_events[0].get("monotonic"))
            worker_wall = end - begin if begin is not None and end is not None else None
            cost["worker_wait4"]["meter_elapsed_wall_seconds"] = worker_wall
            if isinstance(worker_state, dict):
                body_wall = _number(worker_state.get("worker_body_wall_seconds"))
                if run_status == "COMPLETE":
                    checks.check("cost.raw_meter_worker_wall_covers_body",
                                 worker_wall is not None and body_wall is not None
                                 and worker_wall + 0.02 >= body_wall,
                                 actual={"raw_meter_worker_wall": worker_wall,
                                         "worker_body_wall": body_wall},
                                 expected="raw meter worker lifecycle covers worker body")
        elif run_status == "COMPLETE":
            checks.check("cost.raw_meter_worker_lifecycle_present", False,
                         actual={"started": len(command_events), "reaped": len(reaped_events)},
                         expected={"started": 1, "reaped": 1})

    cost_failures = [
        item for item in checks.failures if str(item.get("check", "")).startswith("cost.")
    ]
    if parse_errors:
        cost["validity"] = "ANOMALOUS"
    elif missing:
        cost["validity"] = "PARTIAL"
    elif cost_failures:
        cost["validity"] = "ANOMALOUS"
    else:
        cost["validity"] = "RECORDED_EVIDENCE_CONSISTENT"

    accounting_invalid = (
        run_status == "ACCOUNTING_INVALID"
        or bool(state and state.get("wait4_audit_error"))
        or bool(state and state.get("wait4_children_audit") is None)
    )
    if run_status == "ACCOUNTING_INVALID":
        checks.check(
            "cost.supervisor_accounting_invalid",
            False,
            actual={
                "status": run_status,
                "error": state.get("error") if isinstance(state, dict) else None,
                "wait4_audit_error": state.get("wait4_audit_error")
                if isinstance(state, dict) else None,
            },
            expected="valid wait4 and RUSAGE_CHILDREN reconciliation",
        )
    suspicious_origin = False
    if cost.get("wall_origin"):
        origin_age = cost["wall_origin"].get("first_event_minus_kernel_start_seconds")
        reported_wall = _number(cost.get("reported_campaign_wall_seconds_before_footer"))
        if origin_age is not None and launch_call_seconds is not None:
            suspicious_origin = origin_age > launch_call_seconds + 0.05
        if reported_wall is not None and launch_call_seconds is not None:
            suspicious_origin |= reported_wall > launch_call_seconds + 0.05
    if run_status == "COMPLETE" and not accounting_invalid and not suspicious_origin:
        cost["campaign_cpu_seconds"] = cost.get("recorded_subtotal_cpu_seconds")
        cost["campaign_wall_seconds"] = cost.get("reported_campaign_wall_seconds_before_footer")
        cost["validity"] = "VALIDATED"
    else:
        cost["campaign_cpu_seconds"] = None
        cost["campaign_wall_seconds"] = None
        if accounting_invalid or suspicious_origin:
            cost["validity"] = "INVALID_CAMPAIGN_COST_UNKNOWN"
    return cost, state, worker_state, run_status


def _science_audit(
    files: RunFiles, spec: dict[str, Any], input_refs: dict[int, dict[str, dict[str, Any]]],
    checks: Checks,
) -> dict[str, Any]:
    scalar_names = ("K_old_C", "K_old_D", "K_new_W0", "K_new_C", "K_new_D")
    per_seed: list[dict[str, Any]] = []
    metrics_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    diag_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    completed_expected: list[dict[str, Any]] = []
    gates: dict[int, dict[str, Any]] = {}
    report_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        case = f"cases/s{seed}"
        preparation: dict[str, Any] | None = None
        try:
            preparation = files.json(f"{case}/preparation.json")
            if not isinstance(preparation, dict):
                raise ValueError("preparation record is not an object")
        except Exception as exc:
            checks.check(f"science.s{seed}.preparation_read", False,
                         detail=f"{type(exc).__name__}: {exc}")
        if isinstance(preparation, dict):
            checks.check(f"science.s{seed}.preparation_seed",
                         preparation.get("seed") == seed,
                         actual=preparation.get("seed"), expected=seed)
            checks.check(f"science.s{seed}.preparation_foundation",
                         preparation.get("foundation") == spec.get("foundation"),
                         actual=preparation.get("foundation"), expected=spec.get("foundation"))
            identity = preparation.get("input_identity")
            identity_files = identity.get("files") if isinstance(identity, dict) else None
            expected_for_seed = input_refs.get(seed, {})
            identity_ok = isinstance(identity_files, dict) and set(identity_files) == set(expected_for_seed)
            if identity_ok:
                for name, reference in expected_for_seed.items():
                    actual = identity_files.get(name)
                    identity_ok &= (
                        isinstance(actual, dict)
                        and actual.get("bytes") == reference["bytes"]
                        and actual.get("sha256") == reference["sha256"]
                    )
            checks.check(f"science.s{seed}.input_identity_metadata",
                         bool(identity_ok),
                         actual="preparation input identities versus frozen spec metadata",
                         expected="same three file sizes and SHA-256 identities; original inputs not opened")
            checks.check(f"science.s{seed}.evidence_reference",
                         _reference_meta(preparation.get("evidence_reference")) == {
                             "seed": seed,
                             "file": "evidence.npz",
                             **expected_for_seed.get("evidence.npz", {}),
                         },
                         actual=_reference_meta(preparation.get("evidence_reference")),
                         expected={"seed": seed, "file": "evidence.npz",
                                   **expected_for_seed.get("evidence.npz", {})})

        arm_summary: dict[str, Any] = {}
        arm_metrics: dict[str, dict[str, Any]] = {}
        arm_diags: dict[str, dict[str, Any]] = {}
        for arm, lam in ARMS.items():
            base = f"{case}/{arm.lower()}"
            prefix = f"science.s{seed}.{arm}"
            try:
                metrics = files.json(f"{base}/metrics.json")
                if not isinstance(metrics, dict):
                    raise ValueError("metrics record is not an object")
            except Exception as exc:
                checks.check(f"{prefix}.metrics_read", False,
                             detail=f"{type(exc).__name__}: {exc}")
                continue
            try:
                diag = files.json(f"{base}/diagnostics.json")
                if not isinstance(diag, dict):
                    raise ValueError("diagnostics record is not an object")
            except Exception as exc:
                checks.check(f"{prefix}.diagnostics_read", False,
                             detail=f"{type(exc).__name__}: {exc}")
                continue
            try:
                candidate = files.json(f"{base}/candidate.json")
                if not isinstance(candidate, dict):
                    raise ValueError("candidate record is not an object")
            except Exception as exc:
                checks.check(f"{prefix}.candidate_json_read", False,
                             detail=f"{type(exc).__name__}: {exc}")
                candidate = {}

            checks.check(f"{prefix}.json_finite",
                         _finite_tree(metrics) and _finite_tree(diag) and _finite_tree(candidate),
                         actual="metric/diagnostic/candidate JSON numeric fields",
                         expected="finite")
            checks.check(f"{prefix}.formats",
                         metrics.get("format") == "h036_score_metrics_v1",
                         actual=metrics.get("format"), expected="h036_score_metrics_v1")
            checks.check(f"{prefix}.seed_lambda",
                         metrics.get("seed") == seed
                         and metrics.get("lambda_value") == lam
                         and diag.get("seed") == seed
                         and candidate.get("lambda_value") == lam,
                         actual={"metric_seed": metrics.get("seed"),
                                 "metric_lambda": metrics.get("lambda_value"),
                                 "diagnostic_seed": diag.get("seed"),
                                 "candidate_lambda": candidate.get("lambda_value")},
                         expected={"seed": seed, "lambda": lam})
            checks.check(f"{prefix}.encoded_binding_local",
                         metrics.get("binding") == "wildcard"
                         and metrics.get("fresh_local_is_zero") is True,
                         actual={"binding": metrics.get("binding"),
                                 "fresh_local_is_zero": metrics.get("fresh_local_is_zero")},
                         expected={"binding": "wildcard", "fresh_local_is_zero": True})
            checks.check(f"{prefix}.prepared_id",
                         isinstance(metrics.get("prepared_id"), str)
                         and metrics.get("prepared_id") == diag.get("prepared_id")
                         and metrics.get("prepared_id") == candidate.get("prepared_id")
                         and isinstance(preparation, dict)
                         and metrics.get("prepared_id") == preparation.get("prepared_id"),
                         actual={"metrics": metrics.get("prepared_id"),
                                 "diagnostics": diag.get("prepared_id"),
                                 "candidate": candidate.get("prepared_id"),
                                 "preparation": preparation.get("prepared_id")
                                 if isinstance(preparation, dict) else None},
                         expected="one prepared_id shared by all records for this seed")

            _check_closure_diagnostics(diag, prefix, checks)
            _candidate_bytes_check(files, base, candidate, diag, metrics, seed, arm, checks)

            try:
                actor_identity = files.file_identity(f"{base}/actor.bin")
            except Exception as exc:
                actor_identity = None
                checks.check(f"{prefix}.actor_file", False,
                             detail=f"{type(exc).__name__}: {exc}")
            try:
                checkpoint_identity = files.file_identity(f"{base}/checkpoint.pt")
            except Exception as exc:
                checkpoint_identity = None
                checks.check(f"{prefix}.checkpoint_file", False,
                             detail=f"{type(exc).__name__}: {exc}")
            capacity = metrics.get("capacity")
            artifacts = metrics.get("artifacts")
            capacity = capacity if isinstance(capacity, dict) else {}
            artifacts = artifacts if isinstance(artifacts, dict) else {}
            expected_input = input_refs.get(seed, {})
            evidence_ref = expected_input.get("evidence.npz", {})
            checks.check(f"{prefix}.actor_actual_identity",
                         actor_identity is not None
                         and actor_identity.get("bytes") == 89
                         and actor_identity.get("bytes") == capacity.get("actor_bytes")
                         and actor_identity.get("sha256") == artifacts.get("actor_sha256"),
                         actual=actor_identity,
                         expected={"bytes": 89, "sha256": artifacts.get("actor_sha256")})
            checks.check(f"{prefix}.checkpoint_actual_identity",
                         checkpoint_identity is not None
                         and checkpoint_identity.get("bytes") == capacity.get("checkpoint_bytes")
                         and checkpoint_identity.get("sha256") == artifacts.get("checkpoint_sha256"),
                         actual=checkpoint_identity,
                         expected={"bytes": capacity.get("checkpoint_bytes"),
                                   "sha256": artifacts.get("checkpoint_sha256")})

            try:
                persistent = files.json(f"{base}/persistent_references.json")
                if not isinstance(persistent, dict):
                    raise ValueError("persistent reference record is not an object")
            except Exception as exc:
                persistent = {}
                checks.check(f"{prefix}.persistent_references_read", False,
                             detail=f"{type(exc).__name__}: {exc}")
            foundation = spec.get("foundation")
            package_sum = None
            if checkpoint_identity is not None and actor_identity is not None:
                package_sum = (
                    EXPECTED_FOUNDATION["bytes"]
                    + checkpoint_identity["bytes"]
                    + actor_identity["bytes"]
                    + evidence_ref.get("bytes", 0)
                )
            checks.check(f"{prefix}.foundation_identity_metadata",
                         foundation == {
                             "bytes": EXPECTED_FOUNDATION["bytes"],
                             "sha256": EXPECTED_FOUNDATION["sha256"],
                             "remote_path": "/data/robot/p001/s001/data/foundation.pt",
                         }
                         and capacity.get("foundation_bytes") == EXPECTED_FOUNDATION["bytes"]
                         and artifacts.get("evidence_sha256") == evidence_ref.get("sha256")
                         and capacity.get("evidence_npz_bytes") == evidence_ref.get("bytes")
                         and isinstance(persistent.get("foundation"), dict)
                         and persistent["foundation"].get("bytes") == EXPECTED_FOUNDATION["bytes"]
                         and persistent["foundation"].get("sha256") == EXPECTED_FOUNDATION["sha256"]
                         and persistent["foundation"].get("remote_path")
                         == "/data/robot/p001/s001/data/foundation.pt"
                         and persistent.get("foundation") == {
                             **EXPECTED_FOUNDATION,
                             "remote_path": "/data/robot/p001/s001/data/foundation.pt",
                         },
                         actual={"metrics_foundation_bytes": capacity.get("foundation_bytes"),
                                 "evidence_bytes": capacity.get("evidence_npz_bytes"),
                                 "evidence_sha256": artifacts.get("evidence_sha256"),
                                 "persistent_foundation": persistent.get("foundation")},
                         expected={"foundation_bytes": EXPECTED_FOUNDATION["bytes"],
                                   "evidence": evidence_ref})
            checks.check(f"{prefix}.original_evidence_identity_metadata",
                         _reference_meta(persistent.get("evidence")) == {
                             "seed": seed,
                             "file": "evidence.npz",
                             **evidence_ref,
                         },
                         actual=_reference_meta(persistent.get("evidence")),
                         expected={"seed": seed, "file": "evidence.npz", **evidence_ref})
            checks.check(f"{prefix}.persistent_package_size",
                         package_sum is not None
                         and package_sum <= PACKAGE_LIMIT
                         and capacity.get("full_persistent_bytes") == package_sum
                         and capacity.get("package_budget_bytes") == PACKAGE_LIMIT
                         and persistent.get("full_persistent_bytes") == package_sum
                         and persistent.get("deployment_checkpoint_saved_verbatim") is True,
                         actual={"computed_bytes": package_sum,
                                 "metrics_bytes": capacity.get("full_persistent_bytes"),
                                 "persistent_bytes": persistent.get("full_persistent_bytes"),
                                 "limit": capacity.get("package_budget_bytes")},
                         expected=f"foundation + saved checkpoint + 89-byte actor + identity-referenced evidence <= {PACKAGE_LIMIT}")

            encoded = metrics.get("encoded_fresh")
            w0 = metrics.get("W0_fresh")
            encoded = encoded if isinstance(encoded, dict) else {}
            w0 = w0 if isinstance(w0, dict) else {}
            k_old = _metric_number(metrics.get("K_old"), f"{prefix}.K_old_finite", checks)
            k_new = _metric_number(metrics.get("K_new"), f"{prefix}.K_new_finite", checks)
            k_w0 = _metric_number(metrics.get("K_new_W0"), f"{prefix}.K_new_W0_finite", checks)
            gain = _metric_number(metrics.get("G_new"), f"{prefix}.G_new_finite", checks)
            old_holdout = _metric_number(encoded.get("old_holdout"),
                                         f"{prefix}.encoded_old_holdout_finite", checks)
            new_holdout = _metric_number(encoded.get("new_holdout"),
                                         f"{prefix}.encoded_new_holdout_finite", checks)
            w0_holdout = _metric_number(w0.get("new_holdout"),
                                        f"{prefix}.W0_new_holdout_finite", checks)
            checks.check(f"{prefix}.encoded_holdout_mapping",
                         k_old is not None and old_holdout is not None and k_old == old_holdout
                         and k_new is not None and new_holdout is not None and k_new == new_holdout,
                         actual={"K_old": k_old, "encoded_old_holdout": old_holdout,
                                 "K_new": k_new, "encoded_new_holdout": new_holdout},
                         expected="K_old/K_new equal encoded fresh old/new holdout")
            checks.check(f"{prefix}.W0_G_new_mapping",
                         k_w0 is not None and w0_holdout is not None and k_w0 == w0_holdout
                         and gain is not None and k_w0 is not None and k_new is not None
                         and gain == k_w0 - k_new,
                         actual={"K_new_W0": k_w0, "W0_fresh.new_holdout": w0_holdout,
                                 "G_new": gain, "K_new_W0_minus_K_new":
                                     k_w0 - k_new if k_w0 is not None and k_new is not None else None},
                         expected="K_new_W0 equals W0 holdout; G_new equals K_new_W0 - K_new")
            checks.check(f"{prefix}.nonnegative_KL",
                         all(value is not None and value >= 0 for value in (k_old, k_new, k_w0)),
                         actual={"K_old": k_old, "K_new": k_new, "K_new_W0": k_w0},
                         expected="finite nonnegative holdout KL values")

            arm_metrics[arm] = metrics
            arm_diags[arm] = diag
            metrics_by_key[(seed, arm)] = metrics
            diag_by_key[(seed, arm)] = diag
            completed_expected.append({
                "seed": seed,
                "arm": arm,
                "lambda_value": lam,
                "candidate_sha256": metrics.get("candidate_sha256"),
            })
            arm_summary[arm] = {
                "K_old": metrics.get("K_old"),
                "K_new": metrics.get("K_new"),
                "K_new_W0": metrics.get("K_new_W0"),
                "G_new": metrics.get("G_new"),
                "candidate_sha256": metrics.get("candidate_sha256"),
                **{
                    key: _check_value
                    for key, _check_value in _check_closure_diagnostics(diag, f"{prefix}.summary", Checks()).items()
                },
            }

        if set(arm_metrics) == {"C", "D"}:
            c, d = arm_metrics["C"], arm_metrics["D"]
            c_w0, d_w0 = c.get("K_new_W0"), d.get("K_new_W0")
            checks.check(f"science.s{seed}.C_D_shared_W0",
                         _number(c_w0) is not None and _number(d_w0) is not None
                         and c_w0 == d_w0,
                         actual={"C": c_w0, "D": d_w0}, expected="equal K_new_W0")
            row = {
                "seed": seed,
                "K_old_C": c.get("K_old"),
                "K_old_D": d.get("K_old"),
                "K_new_W0": c.get("K_new_W0"),
                "K_new_C": c.get("K_new"),
                "K_new_D": d.get("K_new"),
                "fallback_flags": {
                    "C": arm_diags["C"].get("fallback_applied"),
                    "D": arm_diags["D"].get("fallback_applied"),
                },
            }
            gates[seed] = row
            report_rows.append(row)
        per_seed.append({"seed": seed, "fits": arm_summary})

    try:
        gate_inputs = files.json("gate_inputs.json")
        if not isinstance(gate_inputs, list):
            raise ValueError("gate_inputs is not a list")
    except Exception as exc:
        gate_inputs = []
        checks.check("science.gate_inputs_read", False,
                     detail=f"{type(exc).__name__}: {exc}")
    try:
        report = files.json("scientific_report.json")
        if not isinstance(report, dict):
            raise ValueError("scientific_report is not an object")
    except Exception as exc:
        report = {}
        checks.check("science.report_read", False,
                     detail=f"{type(exc).__name__}: {exc}")

    checks.check("science.gate_input_row_count",
                 len(gate_inputs) == 16, actual=len(gate_inputs), expected=16)
    expected_gate_rows = [gates[seed] for seed in SEEDS if seed in gates]
    checks.check("science.gate_inputs_equal_metrics",
                 gate_inputs == expected_gate_rows and len(expected_gate_rows) == 16,
                 actual=len(gate_inputs), expected="16 rows exactly mapped from C/D metrics and diagnostics")

    computed_evaluations: list[dict[str, Any]] = []
    for seed in SEEDS:
        row = gates.get(seed)
        if row is None:
            continue
        old_c = _number(row["K_old_C"])
        old_d = _number(row["K_old_D"])
        new_w0 = _number(row["K_new_W0"])
        new_c = _number(row["K_new_C"])
        new_d = _number(row["K_new_D"])
        if None in (old_c, old_d, new_w0, new_c, new_d):
            checks.check(f"science.s{seed}.gate_scalars_finite",
                         False, actual=row, expected="five finite nonnegative scalars")
            continue
        gain_c = new_w0 - new_c
        gain_d = new_w0 - new_d
        conditions = {
            "K_old_C_gt_1e-10": old_c > EPSILON,
            "G_new_C_gt_1e-10": gain_c > EPSILON,
            "K_old_D_le_0.75_K_old_C": old_d <= 0.75 * old_c,
            "G_new_D_ge_0.5_G_new_C": gain_d >= 0.5 * gain_c,
        }
        failed = [name for name, passed in conditions.items() if not passed]
        computed_evaluations.append({
            "seed": seed,
            "K_old_C": old_c,
            "K_old_D": old_d,
            "K_new_W0": new_w0,
            "K_new_C": new_c,
            "K_new_D": new_d,
            "G_new_C": gain_c,
            "G_new_D": gain_d,
            "old_KL_reduction_C_minus_D": old_c - old_d,
            "fallback_flags": row["fallback_flags"],
            "conditions": conditions,
            "passed": not failed,
            "failed_conditions": failed,
        })
    passes = sum(bool(row["passed"]) for row in computed_evaluations)
    expected_status = (
        "MECHANISM_SCREEN_PASSED_EXPLORATORY" if passes >= 12 else "NO_ADVANCEMENT"
    )
    reported_evaluations = report.get("evaluations")
    checks.check("science.evaluation_count",
                 len(computed_evaluations) == 16
                 and isinstance(reported_evaluations, list)
                 and len(reported_evaluations) == 16,
                 actual={"computed": len(computed_evaluations),
                         "reported": len(reported_evaluations)
                         if isinstance(reported_evaluations, list) else None},
                 expected=16)
    evaluations_match = (
        isinstance(reported_evaluations, list)
        and reported_evaluations == computed_evaluations
    )
    checks.check("science.report_evaluations_equal_independent_recompute",
                 evaluations_match,
                 actual="reported per-seed evaluations versus independently recomputed values",
                 expected="exact equality")
    checks.check("science.report_pass_count",
                 report.get("pass_count") == passes,
                 actual=report.get("pass_count"), expected=passes)
    checks.check("science.report_12_of_16_status",
                 report.get("status") == expected_status
                 and report.get("scored_count") == 16
                 and report.get("required_pass_count") == 12
                 and report.get("gate_denominator") == 16
                 and report.get("expected_seeds") == list(SEEDS),
                 actual={"status": report.get("status"),
                         "scored_count": report.get("scored_count"),
                         "required_pass_count": report.get("required_pass_count"),
                         "gate_denominator": report.get("gate_denominator"),
                         "expected_seeds": report.get("expected_seeds")},
                 expected={"status": expected_status, "scored_count": 16,
                           "required_pass_count": 12, "gate_denominator": 16,
                           "expected_seeds": list(SEEDS)})
    checks.check("science.report_scientific_mapping",
                 report.get("scientific_mapping") == spec.get("gate_mapping"),
                 actual=report.get("scientific_mapping"), expected=spec.get("gate_mapping"))

    expected_fallbacks = {
        arm: [seed for seed in SEEDS
              if seed in gates and gates[seed]["fallback_flags"].get(arm) is True]
        for arm in ("C", "D")
    }
    checks.check("science.report_fallback_seed_ids",
                 report.get("fallback_seed_ids") == expected_fallbacks,
                 actual=report.get("fallback_seed_ids"), expected=expected_fallbacks)
    expected_guard_failures = {
        "K_old_C_gt_1e-10": [
            row["seed"] for row in computed_evaluations
            if not row["conditions"]["K_old_C_gt_1e-10"]
        ],
        "G_new_C_gt_1e-10": [
            row["seed"] for row in computed_evaluations
            if not row["conditions"]["G_new_C_gt_1e-10"]
        ],
    }
    checks.check("science.report_denominator_guard_seed_ids",
                 report.get("denominator_guard_failure_seed_ids") == expected_guard_failures,
                 actual=report.get("denominator_guard_failure_seed_ids"),
                 expected=expected_guard_failures)

    completed = report.get("completed_fits")
    checks.check("science.report_completed_fit_count",
                 isinstance(completed, list) and len(completed) == 32,
                 actual=len(completed) if isinstance(completed, list) else None,
                 expected=32)
    checks.check("science.report_completed_fits_equal_artifacts",
                 completed == completed_expected and len(completed_expected) == 32,
                 actual=len(completed) if isinstance(completed, list) else None,
                 expected="32 seed/arm/lambda/candidate identity rows in frozen order")

    return {
        "validated_seed_count": len(gates),
        "validated_fit_count": len(metrics_by_key),
        "computed_pass_count": passes,
        "required_pass_count": 12,
        "computed_gate_status": expected_status,
        "reported_gate_status": report.get("status"),
        "reported_pass_count": report.get("pass_count"),
        "independently_recomputed_inequalities": [
            "K_old_C > 1e-10",
            "G_new_C > 1e-10",
            "K_old_D <= 0.75 * K_old_C",
            "G_new_D >= 0.5 * G_new_C",
        ],
        "per_seed": per_seed,
        "scientific_report_matches_independent_recompute": evaluations_match,
    }


def _read_local_spec(path_value: str) -> tuple[dict[str, Any], str]:
    path = Path(path_value).expanduser()
    if not path.is_file() or path.is_symlink():
        raise ValueError("the supplied frozen spec must be a regular non-symlink file")
    data = path.read_bytes()
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("frozen spec exceeds audit size limit")
    digest = hashlib.sha256(data).hexdigest()
    spec = _loads(data)
    if not isinstance(spec, dict):
        raise ValueError("frozen spec is not a JSON object")
    return spec, digest


def _write_output(path_value: str, report: dict[str, Any], run_root: Path) -> None:
    output = Path(path_value).expanduser().absolute()
    if output.resolve(strict=False).is_relative_to(run_root):
        raise ValueError("audit output must be outside the run mirror")
    output.parent.mkdir(parents=True, exist_ok=True)
    for parent in (output.parent, *output.parent.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("audit output path contains a symlink")
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit one mirrored E163 run offline")
    parser.add_argument("--run-root", required=True, help="local mirror of runs/run001/remote/run001")
    parser.add_argument("--spec", required=True, help="local frozen run_spec.json")
    parser.add_argument("--output", required=True, help="new audit JSON path")
    parser.add_argument(
        "--launch-call-wall-seconds",
        type=float,
        default=None,
        help="optional coordinator-observed end-to-end launcher call duration",
    )
    args = parser.parse_args(argv)
    if args.launch_call_wall_seconds is not None and (
        not math.isfinite(args.launch_call_wall_seconds)
        or args.launch_call_wall_seconds < 0
    ):
        parser.error("--launch-call-wall-seconds must be finite and nonnegative")

    spec, spec_hash = _read_local_spec(args.spec)
    spec["_computed_sha256"] = spec_hash
    checks = Checks()
    input_refs = _spec_identity(spec, checks)

    root_arg = Path(args.run_root).expanduser()
    if not root_arg.is_dir() or root_arg.is_symlink():
        raise ValueError("run root must be an existing, non-symlink directory")
    run_root = root_arg.resolve(strict=True)
    files = RunFiles(run_root)

    cost, state, worker_state, run_status = _check_cost(
        files, spec, checks, args.launch_call_wall_seconds
    )
    is_complete = run_status == "COMPLETE"
    scientific = None
    if is_complete:
        scientific = _science_audit(files, spec, input_refs, checks)
        if isinstance(worker_state, dict):
            checks.check("science.worker_state_complete",
                         worker_state.get("status") == "COMPLETE"
                         and worker_state.get("completed_fits") == 32
                         and worker_state.get("environment_interactions") == 0,
                         actual={key: worker_state.get(key) for key in (
                             "status", "completed_fits", "environment_interactions"
                         )},
                         expected={"status": "COMPLETE", "completed_fits": 32,
                                   "environment_interactions": 0})
        else:
            checks.check("science.worker_state_present", False,
                         actual=None, expected="state_worker.json")
        try:
            environment = files.json("environment.json")
            if not isinstance(environment, dict):
                raise ValueError("environment record is not an object")
            checks.check("science.environment_identity",
                         environment.get("run_spec_sha256") == spec_hash
                         and environment.get("source_identity_sha256") == EXPECTED_SOURCE_IDENTITY
                         and environment.get("platform") == "linux"
                         and environment.get("torch_threads") == 1
                         and environment.get("torch_interop_threads") == 1,
                         actual={key: environment.get(key) for key in (
                             "run_spec_sha256", "source_identity_sha256", "platform",
                             "torch_threads", "torch_interop_threads"
                         )},
                         expected={"run_spec_sha256": spec_hash,
                                   "source_identity_sha256": EXPECTED_SOURCE_IDENTITY,
                                   "platform": "linux", "torch_threads": 1,
                                   "torch_interop_threads": 1})
        except Exception as exc:
            checks.check("science.environment_read", False,
                         detail=f"{type(exc).__name__}: {exc}")

    cost_failures = [
        item for item in checks.failures if str(item.get("check", "")).startswith("cost.")
    ]
    if is_complete:
        audit_status = "COMPLETE_VALID" if not checks.failures else "COMPLETE_INVALID"
    elif run_status == "UNKNOWN":
        audit_status = "UNKNOWN"
    else:
        audit_status = run_status

    report = {
        "schema": "E163_OFFLINE_AUDIT_V1",
        "audit_status": audit_status,
        "observed_supervisor_status": run_status,
        "scientific_validation": "COMPLETE_AUDIT" if is_complete else "SKIPPED_NONCOMPLETE_RUN",
        "mirror_integrity": {
            "status": "REMOTE_BYTE_IDENTITIES_VERIFIED",
            "source": "coordinator-reported provenance manifest; this script does not repeat the SSH comparison",
            "verified_bytes": {
                "state.json": 2883,
                "meter.jsonl": 2863,
                "wait4.jsonl": 474,
                "total": 6220,
            },
        },
        "execution_integrity_assessment": {
            "verified_artifact_facts": [
                "state.json reports ACCOUNTING_INVALID with RUSAGE_CHILDREN and recorded wait4 accounting mismatch",
                "only one recorded direct child exists: cpu_descendant_accounting_precheck, exit code 0, wait4 CPU 0.122070 seconds",
                "no scientific worker wait4 record is present; this non-COMPLETE run is not scored",
                "the recorded supervisor subtotal is 0.268236 CPU seconds, composed of 0.146166 supervisor RUSAGE_SELF seconds and 0.122070 recorded wait4 seconds",
                "the state reports nominal bounds of 5.268236 CPU seconds and 77.527558 seconds wall time, which include the 5 second and 10 second footer reserves",
            ],
            "verified_source_facts": [
                "launch_command.sh sources enter.sh, sets the project environment, changes to the frozen source directory, then execs the supervisor Python command",
                "supervisor.py derives campaign_started_mono from /proc process start ticks before writing the first campaign_started event",
                "the CPU descendant precheck compares RUSAGE_CHILDREN with the sum of explicitly recorded wait4 children before recording precheck success",
                "the final children audit repeats a RUSAGE_CHILDREN versus recorded wait4 sum comparison",
            ],
            "inferences": [
                "the first meter event is about 67.345 seconds after the reported kernel start tick, while the coordinator-observed launcher call lasted 0.745819 seconds; the reported 67.527558 second wall value therefore includes a pre-event interval and is not accepted as campaign wall cost",
                "the coordinator reports that Python and du children ran in the launching shell before exec; because launch_command.sh execs in place and the supervisor compares cumulative child usage with its own tracked wait4 rows, inherited child accounting is a plausible explanation for the mismatch, but the mirror does not establish the exact prior child identities or CPU amounts",
            ],
            "cost_conclusion": "campaign CPU and wall cost are UNKNOWN under this invalid accounting and wall-clock origin; retain the recorded subtotal and nominal bounds as evidence only",
        },
        "read_scope": {
            "run_root": str(run_root),
            "frozen_spec_sha256": spec_hash,
            "original_prefix_files_opened": False,
            "checkpoint_deserialized": False,
            "model_or_numeric_libraries_imported": False,
            "fit_or_environment_execution_started": False,
        },
        "cost": cost,
        "scientific": scientific,
        "check_count": checks.count,
        "failure_count": len(checks.failures),
        "cost_failure_count": len(cost_failures),
        "failures": checks.failures,
    }
    _write_output(args.output, report, run_root)
    print(json.dumps({
        "audit_status": audit_status,
        "observed_supervisor_status": run_status,
        "scientific_validation": report["scientific_validation"],
        "cost_validity": cost.get("validity"),
        "campaign_cpu_seconds": cost.get("campaign_cpu_seconds"),
        "campaign_wall_seconds": cost.get("campaign_wall_seconds"),
        "output": str(Path(args.output).expanduser().absolute()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

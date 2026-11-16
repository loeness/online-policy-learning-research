#!/usr/bin/env python3
"""Linux-only, single-worker meter for the three E165 validation stages.

The supervisor is intentionally small and standard-library-only. It validates
the exact stage authorization before sampling runtime resources or launching a
worker. Its output contract remains frozen until the coordinator binds the
stage workers and their schemas.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import select
import signal
import stat
import sys
import time
import traceback


PROJECT = Path("/data/robot/p001")
E_ROOT = PROJECT / "e165"
ORCHESTRATION_ROOT = E_ROOT / "o001"
SOURCE_CWD = E_ROOT / "s001"
PYTHON = PROJECT / "env/bin/python"
STAGE_ROOTS = {
    "CHECKS": E_ROOT / "c001",
    "DIAGNOSTIC": E_ROOT / "run001",
    "AUDIT": E_ROOT / "a001",
}
STAGE_WORKERS = {
    "CHECKS": ORCHESTRATION_ROOT / "checks_entry.py",
    "DIAGNOSTIC": ORCHESTRATION_ROOT / "runner.py",
    "AUDIT": ORCHESTRATION_ROOT / "audit_entry.py",
}
STAGE_CAPS = {
    "CHECKS": {"cpu_seconds": 20, "wall_seconds": 60},
    "DIAGNOSTIC": {"cpu_seconds": 80, "wall_seconds": 140},
    "AUDIT": {"cpu_seconds": 20, "wall_seconds": 40},
}
ROUND_LIMITS = {
    "cpu_seconds": 120,
    "cumulative_controlled_wall_seconds": 240,
    "new_artifact_cache_temp_bytes": 134217728,
    "aggregate_remote_project_bytes": 100000000000,
    "workers": 1,
    "numeric_threads_per_worker": 1,
    "gpu_devices": 0,
    "environment_interactions": 0,
    "real_candidate_fit_calls": 0,
    "candidate_recovery_calls": 0,
    "real_full_training_oracle_evaluations": 336,
    "new_complete_score_packages": 33,
    "logical_paired_prefixes": 16,
}
FOOTER_CPU_RESERVE_SECONDS = 5.0
FOOTER_WALL_RESERVE_SECONDS = 10.0
FOOTER_BYTE_RESERVE = 65536
METER_INTERVAL_SECONDS = 0.25
POLL_INTERVAL_SECONDS = 0.025
STOP_GRACE_SECONDS = 0.75
REAP_TIMEOUT_SECONDS = 1.0
CPU_PROBE_WALL_TIMEOUT_SECONDS = 5.0
# CHECKS control schema is fixed; each stage still needs its own exact
# coordinator authorization, source manifest, worker and specification.
OUTPUT_CONTRACT_FROZEN = True

EXPECTED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "TMPDIR": str(E_ROOT / "tmp"),
    "TMP": str(E_ROOT / "tmp"),
    "TEMP": str(E_ROOT / "tmp"),
    "XDG_CACHE_HOME": str(E_ROOT / "cache"),
    "XDG_CONFIG_HOME": str(E_ROOT / "config"),
    "XDG_DATA_HOME": str(E_ROOT / "state"),
    "XDG_STATE_HOME": str(E_ROOT / "state"),
    "MPLCONFIGDIR": str(E_ROOT / "cache/mpl"),
    "TORCH_HOME": str(E_ROOT / "cache/torch"),
    "TORCH_EXTENSIONS_DIR": str(E_ROOT / "cache/ext"),
}
SAFE_SPEC_ENV_KEYS = frozenset(EXPECTED_ENVIRONMENT) | {
    "VECLIB_MAXIMUM_THREADS", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_MODE",
}
PRIOR_STAGES = {
    "CHECKS": (),
    "DIAGNOSTIC": ("CHECKS",),
    "AUDIT": ("CHECKS", "DIAGNOSTIC"),
}
STAGE_SPEC_PATHS = {
    stage: ORCHESTRATION_ROOT / f"stage_{stage.lower()}.json"
    for stage in STAGE_CAPS
}
STAGE_AUTH_PATHS = {
    stage: ORCHESTRATION_ROOT / f"authorization_{stage.lower()}.json"
    for stage in STAGE_CAPS
}
CAMPAIGN_START_SOURCE = (
    "new supervisor process start ticks from /proc/self/stat; includes Python interpreter "
    "startup; launch shell and SSH parent are recorded separately and excluded from the campaign clock"
)
CPU_TOTAL_SCOPE = (
    "supervisor RUSAGE_SELF + completed direct-child wait4 totals + live owned /proc counters; "
    "child wait4 descendants checked by in-campaign precheck"
)
FINAL_FOOTER_DISCLOSURE = (
    "not observed after supervisor exit; explicitly bounded by the reported CPU and wall reserves"
)


class SupervisorError(RuntimeError):
    pass


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _is_int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _is_finite_nonnegative(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _exact_mapping(value: object, expected: dict) -> bool:
    if not isinstance(value, dict) or set(value) != set(expected):
        return False
    return all(type(value[key]) is type(expected[key]) and value[key] == expected[key]
               for key in expected)


def _safe_path(value: str, label: str, *, root: Path,
               must_exist: bool = True, allow_internal_symlink: bool = False) -> Path:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise SupervisorError(f"{label} must be an absolute POSIX path")
    path = Path(value)
    if str(path) != value or ".." in path.parts or "." in path.parts:
        raise SupervisorError(f"{label} is not in canonical path form")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_project = PROJECT.resolve(strict=True)
        resolved = path.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise SupervisorError(f"{label} cannot be resolved: {exc}") from exc
    if resolved_project != PROJECT or resolved_root != root:
        raise SupervisorError("allocated project roots must be canonical real paths")
    if not _is_below(resolved_root, resolved_project) or not _is_below(resolved, resolved_root):
        raise SupervisorError(f"{label} resolves outside the allocated project")
    if not allow_internal_symlink and resolved != path:
        raise SupervisorError(f"{label} contains a symlink or path alias")

    try:
        root_info = root.lstat()
    except OSError as exc:
        raise SupervisorError(f"{label} root cannot be inspected: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise SupervisorError(f"{label} root is not a real directory")
    relative = path.relative_to(root)
    cursor = root
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            if must_exist:
                raise SupervisorError(f"{label} component is missing: {cursor}")
            continue
        except OSError as exc:
            raise SupervisorError(f"{label} component cannot be inspected: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) and not allow_internal_symlink:
            raise SupervisorError(f"{label} contains a symlink: {cursor}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise SupervisorError(f"{label} parent is not a directory: {cursor}")
    return path


def _read_json(path: Path, label: str) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupervisorError(f"{label} must contain a JSON object")
    return value, raw


def _validate_environment(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          for k, v in value.items()):
        raise SupervisorError("environment must be a string-to-string object")
    if "HOME" in value:
        raise SupervisorError("the spec may not change HOME")
    unknown = set(value) - SAFE_SPEC_ENV_KEYS
    if unknown:
        raise SupervisorError(f"environment contains unapproved keys: {sorted(unknown)}")
    for key, expected in EXPECTED_ENVIRONMENT.items():
        if value.get(key) != expected:
            raise SupervisorError(f"environment override mismatch for {key}")
    for key in ("VECLIB_MAXIMUM_THREADS",):
        if key in value and value[key] != "1":
            raise SupervisorError(f"{key} must be 1")
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if key in value and value[key] != "":
            raise SupervisorError(f"{key} must be empty because GPU use is disabled")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if key in value and value[key] != "1":
            raise SupervisorError(f"{key} must be 1 when supplied")
    if "WANDB_MODE" in value and value["WANDB_MODE"] != "offline":
        raise SupervisorError("WANDB_MODE must be offline when supplied")
    return dict(value)


def _expected_worker_argv(stage: str, spec_path: Path, authorization_path: Path) -> list[str]:
    return ["-B", str(STAGE_WORKERS[stage]), "--spec", str(spec_path),
            "--authorization", str(authorization_path)]


def _validate_descriptor_list(value: object, label: str, *, root: Path) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise SupervisorError(f"{label} must be a nonempty descriptor list")
    seen: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise SupervisorError(f"{label}[{index}] must be an object")
        path_value = row.get("path")
        if not isinstance(path_value, str):
            raise SupervisorError(f"{label}[{index}].path must be a string")
        path = _safe_path(path_value, f"{label}[{index}].path", root=root, must_exist=False)
        if path_value in seen:
            raise SupervisorError(f"{label} contains a duplicate path")
        seen.add(path_value)
        if not _is_int(row.get("bytes")) or not _is_sha256(row.get("sha256")):
            raise SupervisorError(f"{label}[{index}] needs exact bytes and sha256")
    return value


def _verify_descriptor_list(rows: list[dict], label: str, *, root: Path) -> None:
    for index, row in enumerate(rows):
        path = _safe_path(row["path"], f"{label}[{index}].path", root=root, must_exist=True)
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SupervisorError(f"{label}[{index}] is not a single-link regular file")
            raw = path.read_bytes()
        except OSError as exc:
            raise SupervisorError(f"cannot verify {label}[{index}]: {exc}") from exc
        if len(raw) != row["bytes"] or _sha256(raw) != row["sha256"]:
            raise SupervisorError(f"{label}[{index}] bytes or sha256 differ from the exact manifest")


def _validate_prior_descriptors(value: object, stage: str) -> dict[str, dict]:
    expected_stages = PRIOR_STAGES[stage]
    if not isinstance(value, dict) or set(value) != set(expected_stages):
        raise SupervisorError(f"prior_stage_states keys do not match the {stage} predecessor list")
    result: dict[str, dict] = {}
    for prior_stage in expected_stages:
        row = value.get(prior_stage)
        expected_path = STAGE_ROOTS[prior_stage] / "state.json"
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise SupervisorError(f"prior_stage_states.{prior_stage} must be an object")
        if row.get("path") != str(expected_path):
            raise SupervisorError(f"prior_stage_states.{prior_stage}.path differs from its controlled state path")
        if not _is_int(row.get("bytes")) or not _is_sha256(row.get("sha256")):
            raise SupervisorError(f"prior_stage_states.{prior_stage} needs exact bytes and sha256")
        _safe_path(row["path"], f"prior_stage_states.{prior_stage}.path",
                   root=E_ROOT, must_exist=True)
        result[prior_stage] = dict(row)
    return result


def _validate_spec_and_authorization(stage: str, spec_path_value: str,
                                     authorization_path_value: str
                                     ) -> tuple[dict, dict, Path, Path, Path]:
    if stage not in STAGE_CAPS:
        raise SupervisorError("stage must be CHECKS, DIAGNOSTIC, or AUDIT")
    spec_path = _safe_path(spec_path_value, "spec path", root=E_ROOT, must_exist=True)
    authorization_path = _safe_path(authorization_path_value, "authorization path",
                                    root=E_ROOT, must_exist=True)
    if spec_path != STAGE_SPEC_PATHS[stage] or authorization_path != STAGE_AUTH_PATHS[stage]:
        raise SupervisorError("spec and authorization must be the exact o001 files for this stage")
    spec, spec_raw = _read_json(spec_path, "stage spec")
    authorization, _ = _read_json(authorization_path, "stage authorization")

    # Complete the authorization binding before any runtime resource sampling.
    if authorization.get("experiment_id") != "E165" or authorization.get("stage") != stage:
        raise PermissionError("coordinator authorization stage does not match E165 and the CLI")
    if authorization.get("resource_authorized") is not True:
        raise PermissionError("coordinator authorization does not enable E165 resources")
    if authorization.get("execution_authorized") is not True:
        raise PermissionError("coordinator has not enabled this E165 stage")
    if authorization.get("run_spec_sha256") != _sha256(spec_raw):
        raise PermissionError("coordinator authorization is not bound to these exact spec bytes")
    if not _exact_mapping(authorization.get("limits"), STAGE_CAPS[stage]):
        raise PermissionError("authorization stage limits differ from the approved allocation")
    reservation = authorization.get("single_attempt_reservation_id")
    if not isinstance(reservation, str) or not reservation.strip():
        raise PermissionError("single_attempt_reservation_id is missing")
    directive = authorization.get("coordinator_directive")
    if not isinstance(directive, str) or not directive.strip():
        raise PermissionError("coordinator_directive is missing")

    if spec.get("experiment_id") != "E165" or spec.get("stage") != stage:
        raise SupervisorError("stage spec identity does not match E165 and the requested stage")
    if spec.get("project_root") != str(PROJECT) or spec.get("e_root") != str(E_ROOT):
        raise SupervisorError("spec project roots differ from the allocated paths")
    if spec.get("orchestration_root") != str(ORCHESTRATION_ROOT):
        raise SupervisorError("spec orchestration_root differs from e165/o001")
    if spec.get("source_cwd") != str(SOURCE_CWD):
        raise SupervisorError("spec source_cwd differs from e165/s001")
    if spec.get("run_root") != str(STAGE_ROOTS[stage]):
        raise SupervisorError("spec run_root differs from the fixed output path for this stage")
    if spec.get("python") != str(PYTHON):
        raise SupervisorError("spec python differs from the allocated project interpreter")
    if not _exact_mapping(spec.get("limits"), STAGE_CAPS[stage]):
        raise SupervisorError("spec stage limits differ from the fixed budget allocation")
    if not _exact_mapping(spec.get("round_limits"), ROUND_LIMITS):
        raise SupervisorError("spec round_limits differ from the approved E165 planned limits")
    if not _is_int(spec.get("project_bytes_before")):
        raise SupervisorError("project_bytes_before must be the non-E165 integer baseline")
    if not _is_sha256(spec.get("source_identity_sha256")):
        raise SupervisorError("source_identity_sha256 must be a lowercase SHA256")
    source_files = _validate_descriptor_list(spec.get("source_files"), "source_files", root=SOURCE_CWD)
    orchestration_files = _validate_descriptor_list(
        spec.get("orchestration_files"), "orchestration_files", root=ORCHESTRATION_ROOT)
    prior = _validate_prior_descriptors(spec.get("prior_stage_states"), stage)
    expected_worker_argv = _expected_worker_argv(stage, spec_path, authorization_path)
    if spec.get("worker_argv") != expected_worker_argv:
        raise SupervisorError("worker_argv differs from the fixed stage worker command")
    if stage == "CHECKS":
        expected_nodeids = spec.get("expected_test_nodeids")
        if (not isinstance(expected_nodeids, list) or not expected_nodeids
                or any(not isinstance(item, str) or not item for item in expected_nodeids)
                or len(set(expected_nodeids)) != len(expected_nodeids)):
            raise SupervisorError("CHECKS spec expected_test_nodeids must be nonempty and unique")
    environment = _validate_environment(spec.get("environment"))
    spec = dict(spec)
    spec["source_files"] = source_files
    spec["orchestration_files"] = orchestration_files
    spec["prior_stage_states"] = prior
    spec["environment"] = environment

    _safe_path(spec["source_cwd"], "source_cwd", root=E_ROOT, must_exist=True)
    _safe_path(spec["orchestration_root"], "orchestration_root", root=E_ROOT, must_exist=True)
    _safe_path(spec["python"], "compute interpreter", root=PROJECT,
               must_exist=True, allow_internal_symlink=True)
    run_root = _safe_path(spec["run_root"], "run_root", root=E_ROOT, must_exist=False)
    if run_root.exists() or run_root.is_symlink():
        raise SupervisorError("stage output root already exists; refusing to overwrite evidence")
    if not E_ROOT.is_dir() or not SOURCE_CWD.is_dir() or not ORCHESTRATION_ROOT.is_dir():
        raise SupervisorError("allocated E165 source or orchestration directory is missing")
    return spec, authorization, spec_path, authorization_path, run_root


def _verify_input_manifests(spec: dict) -> None:
    _verify_descriptor_list(spec["source_files"], "source_files", root=SOURCE_CWD)
    _verify_descriptor_list(spec["orchestration_files"], "orchestration_files",
                            root=ORCHESTRATION_ROOT)


def _expected_state_limits(stage: str) -> dict:
    return {
        "cpu_seconds": STAGE_CAPS[stage]["cpu_seconds"],
        "campaign_wall_seconds": STAGE_CAPS[stage]["wall_seconds"],
        "new_artifact_cache_temp_bytes": ROUND_LIMITS["new_artifact_cache_temp_bytes"],
        "workers": ROUND_LIMITS["workers"],
        "numeric_threads_per_worker": ROUND_LIMITS["numeric_threads_per_worker"],
        "gpu_devices": ROUND_LIMITS["gpu_devices"],
        "environment_interactions": ROUND_LIMITS["environment_interactions"],
        "real_candidate_fit_calls": ROUND_LIMITS["real_candidate_fit_calls"],
        "candidate_recovery_calls": ROUND_LIMITS["candidate_recovery_calls"],
        "real_full_training_oracle_evaluations": ROUND_LIMITS["real_full_training_oracle_evaluations"],
        "new_complete_score_packages": ROUND_LIMITS["new_complete_score_packages"],
        "logical_paired_prefixes": ROUND_LIMITS["logical_paired_prefixes"],
        "aggregate_remote_project_bytes": ROUND_LIMITS["aggregate_remote_project_bytes"],
    }


def _cost_row(state: dict, stage: str) -> dict:
    snapshot = state.get("cpu_snapshot_before_final_footer")
    if not isinstance(snapshot, dict):
        raise SupervisorError(f"prior {stage} state lacks its final CPU snapshot")
    observed_cpu = snapshot.get("total_cpu_seconds")
    observed_wall = state.get("campaign_wall_seconds_before_final_footer")
    upper_cpu = state.get("bounded_campaign_cpu_seconds_upper")
    upper_wall = state.get("bounded_campaign_wall_seconds_upper")
    for value, label in ((observed_cpu, "observed CPU"), (observed_wall, "observed wall"),
                         (upper_cpu, "bounded CPU"), (upper_wall, "bounded wall")):
        if not _is_finite_nonnegative(value):
            raise SupervisorError(f"prior {stage} state has invalid {label}")
    if abs(float(upper_cpu) - (float(observed_cpu) + FOOTER_CPU_RESERVE_SECONDS)) > 1e-6:
        raise SupervisorError(f"prior {stage} bounded CPU does not include the fixed footer reserve")
    if abs(float(upper_wall) - (float(observed_wall) + FOOTER_WALL_RESERVE_SECONDS)) > 1e-6:
        raise SupervisorError(f"prior {stage} bounded wall does not include the fixed footer reserve")
    return {
        "stage": stage,
        "observed_cpu_seconds": float(observed_cpu),
        "bounded_cpu_seconds_upper": float(upper_cpu),
        "observed_wall_seconds": float(observed_wall),
        "bounded_wall_seconds_upper": float(upper_wall),
    }


def _validate_round_ledger(state: dict, expected_rows: list[dict], stage: str) -> None:
    ledger = state.get("round_cost_ledger")
    if not isinstance(ledger, dict) or ledger.get("stages") != expected_rows:
        raise SupervisorError(f"prior {stage} round_cost_ledger does not match its recorded stages")
    expected_totals = {
        "observed_cpu_seconds": sum(row["observed_cpu_seconds"] for row in expected_rows),
        "bounded_cpu_seconds_upper": sum(row["bounded_cpu_seconds_upper"] for row in expected_rows),
        "observed_wall_seconds": sum(row["observed_wall_seconds"] for row in expected_rows),
        "bounded_wall_seconds_upper": sum(row["bounded_wall_seconds_upper"] for row in expected_rows),
    }
    for field, expected in expected_totals.items():
        actual = ledger.get(field)
        if not _is_finite_nonnegative(actual) or abs(float(actual) - expected) > 1e-6:
            raise SupervisorError(f"prior {stage} round_cost_ledger total {field} is inconsistent")


def _load_and_validate_prior_states(spec: dict) -> tuple[list[dict], list[dict]]:
    stage = spec["stage"]
    prior_stages = PRIOR_STAGES[stage]
    states: list[dict] = []
    costs: list[dict] = []
    source_identity = spec["source_identity_sha256"]
    for prior_stage in prior_stages:
        descriptor = spec["prior_stage_states"][prior_stage]
        state_path = _safe_path(descriptor["path"], f"prior {prior_stage} state path",
                                root=E_ROOT, must_exist=True)
        try:
            info = state_path.lstat()
            raw = state_path.read_bytes()
        except OSError as exc:
            raise SupervisorError(f"cannot read prior {prior_stage} state: {exc}") from exc
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or len(raw) != descriptor["bytes"] or _sha256(raw) != descriptor["sha256"]):
            raise SupervisorError(f"prior {prior_stage} state bytes or SHA differ from its descriptor")
        try:
            state = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SupervisorError(f"prior {prior_stage} state is invalid JSON") from exc
        if not isinstance(state, dict):
            raise SupervisorError(f"prior {prior_stage} state must be a JSON object")
        if (state.get("experiment_id") != "E165" or state.get("stage") != prior_stage
                or state.get("status") != "COMPLETE" or state.get("source_identity_sha256") != source_identity
                or not _is_int(state.get("return_code")) or state["return_code"] != 0
                or not _is_int(state.get("worker_return_code")) or state["worker_return_code"] != 0):
            raise SupervisorError(f"prior {prior_stage} state is not a matching COMPLETE result")
        if not _exact_mapping(state.get("limits"), _expected_state_limits(prior_stage)):
            raise SupervisorError(f"prior {prior_stage} state limits differ from its fixed allocation")
        row = _cost_row(state, prior_stage)
        cap = STAGE_CAPS[prior_stage]
        if (row["bounded_cpu_seconds_upper"] > cap["cpu_seconds"]
                or row["bounded_wall_seconds_upper"] > cap["wall_seconds"]
                or row["observed_cpu_seconds"] >= cap["cpu_seconds"] - FOOTER_CPU_RESERVE_SECONDS
                or row["observed_wall_seconds"] >= cap["wall_seconds"] - FOOTER_WALL_RESERVE_SECONDS):
            raise SupervisorError(f"prior {prior_stage} cost exceeded its own hard or stop limit")
        states.append(state)
        costs.append(row)
        _validate_round_ledger(state, costs, prior_stage)
    previous_cpu_upper = sum(row["bounded_cpu_seconds_upper"] for row in costs)
    previous_wall_upper = sum(row["bounded_wall_seconds_upper"] for row in costs)
    current = STAGE_CAPS[stage]
    if previous_cpu_upper + current["cpu_seconds"] > ROUND_LIMITS["cpu_seconds"]:
        raise SupervisorError("prior stage upper costs plus this stage allocation exceed round CPU")
    if previous_wall_upper + current["wall_seconds"] > ROUND_LIMITS["cumulative_controlled_wall_seconds"]:
        raise SupervisorError("prior stage upper costs plus this stage allocation exceed round wall time")
    return states, costs


def _proc_stat(pid: int) -> dict[str, int | str]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise SupervisorError(f"cannot parse owned /proc/{pid}/stat")
    fields = raw[close + 2:].split()
    if len(fields) < 20:
        raise SupervisorError(f"short owned /proc/{pid}/stat record")
    return {
        "state": fields[0],
        "utime_ticks": int(fields[11]),
        "stime_ticks": int(fields[12]),
        "cutime_ticks": int(fields[13]),
        "cstime_ticks": int(fields[14]),
        "start_ticks": int(fields[19]),
    }


def _proc_uptime() -> float:
    return float(Path("/proc/uptime").read_text(encoding="ascii").split()[0])


def _ru_cpu(usage: resource.struct_rusage) -> float:
    return float(usage.ru_utime + usage.ru_stime)


def _ru_record(usage: resource.struct_rusage) -> dict[str, float | int]:
    return {
        "user_seconds": float(usage.ru_utime),
        "system_seconds": float(usage.ru_stime),
        "max_rss_kib": int(usage.ru_maxrss),
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "input_blocks": int(usage.ru_inblock),
        "output_blocks": int(usage.ru_oublock),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
    }


def _capture_startup_proof() -> tuple[dict, list[str]]:
    """Capture launch identity and process-local accounting before any fork."""
    pid = os.getpid()
    parent_pid = os.getppid()
    expected_parent_raw = os.environ.get("E165_LAUNCH_PARENT_PID")
    proof: dict = {
        "supervisor_pid": pid,
        "parent_pid": parent_pid,
        "launch_parent_pid_env": expected_parent_raw,
        "captured_before_supervisor_fork": True,
    }
    errors: list[str] = []
    try:
        proof["supervisor_start_ticks"] = int(_proc_stat(pid)["start_ticks"])
    except (OSError, SupervisorError, ValueError) as exc:
        proof["supervisor_start_ticks_error"] = f"{type(exc).__name__}: {exc}"
        errors.append("supervisor start ticks could not be read")
    try:
        expected_parent_pid = int(expected_parent_raw) if expected_parent_raw is not None else None
    except ValueError:
        expected_parent_pid = None
    proof["expected_parent_pid"] = expected_parent_pid
    if expected_parent_pid is None or expected_parent_pid <= 1:
        errors.append("E165_LAUNCH_PARENT_PID is missing or invalid")
    elif expected_parent_pid != parent_pid:
        errors.append("actual parent PID differs from E165_LAUNCH_PARENT_PID")
    else:
        try:
            proof["parent_start_ticks"] = int(_proc_stat(expected_parent_pid)["start_ticks"])
        except (OSError, SupervisorError, ValueError) as exc:
            proof["parent_start_ticks_error"] = f"{type(exc).__name__}: {exc}"
            errors.append("known launch parent start ticks could not be read")
    if pid == parent_pid:
        errors.append("supervisor replaced the launch shell instead of starting as a child")
    if ("supervisor_start_ticks" in proof and "parent_start_ticks" in proof
            and proof["supervisor_start_ticks"] < proof["parent_start_ticks"]):
        errors.append("supervisor start ticks precede the launch parent")
    try:
        proof["rusage_sampling_phase"] = "POST_AUTHORIZATION_PRE_FORK"
        proof["rusage_self_post_authorization_pre_fork"] = _ru_record(
            resource.getrusage(resource.RUSAGE_SELF))
        proof["rusage_children_post_authorization_pre_fork"] = _ru_record(
            resource.getrusage(resource.RUSAGE_CHILDREN))
    except (OSError, ValueError) as exc:
        proof["rusage_start_error"] = f"{type(exc).__name__}: {exc}"
        errors.append("initial process resource usage could not be read")
    else:
        child_usage = proof["rusage_children_post_authorization_pre_fork"]
        if child_usage["user_seconds"] != 0.0 or child_usage["system_seconds"] != 0.0:
            errors.append("initial RUSAGE_CHILDREN CPU must be exactly zero")
    return proof, errors


def _storage_bytes(root: Path) -> int:
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise SupervisorError("E165 root is not a real directory")
    total = 0
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        parent = Path(directory)
        for name in list(names):
            path = parent / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SupervisorError(f"non-directory or symlink under E165 root: {path}")
        for name in files:
            path = parent / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SupervisorError(f"non-regular file or symlink under E165 root: {path}")
            if info.st_nlink != 1:
                raise SupervisorError(f"hard-linked file under E165 root: {path}")
            total += info.st_size
    return total


def _validate_capacity(spec: dict, *, anticipated_bytes: int = 0,
                       reserve_footer: bool = True) -> int:
    e_bytes = _storage_bytes(E_ROOT)
    cap = ROUND_LIMITS["new_artifact_cache_temp_bytes"]
    reserve = FOOTER_BYTE_RESERVE if reserve_footer else 0
    if e_bytes + anticipated_bytes + reserve > cap:
        raise SupervisorError("E165 new artifact/cache/temp allowance or footer reserve would be exceeded")
    if spec["project_bytes_before"] + e_bytes + anticipated_bytes > ROUND_LIMITS["aggregate_remote_project_bytes"]:
        raise SupervisorError("aggregate project byte cap would be exceeded")
    return e_bytes


def _make_worker_environment(spec_environment: dict[str, str], stage: str) -> dict[str, str]:
    # Pass only ordinary runtime settings; never forward SSH agents or credentials.
    child_env: dict[str, str] = {}
    for key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV"):
        if key in os.environ:
            child_env[key] = os.environ[key]
    child_env.update(spec_environment)
    child_env.update({
        "E165_SUPERVISED": "1",
        "E165_STAGE": stage,
        "CUDA_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    if "HOME" in os.environ and child_env.get("HOME") != os.environ["HOME"]:
        raise SupervisorError("worker environment changed HOME")
    return child_env


class Campaign:
    def __init__(self, stage: str, spec: dict, authorization: dict,
                 spec_path: Path, authorization_path: Path, run_root: Path,
                 startup_proof: dict, prior_states: list[dict], prior_costs: list[dict]):
        self.stage = stage
        self.spec = spec
        self.authorization = authorization
        self.spec_path = spec_path
        self.authorization_path = authorization_path
        self.run_root = run_root
        self.startup_proof = startup_proof
        self.prior_states = prior_states
        self.prior_costs = prior_costs
        self.ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        if self.ticks_per_second <= 0:
            raise SupervisorError("invalid kernel clock tick rate")
        own_stat = _proc_stat(os.getpid())
        self.own_start_ticks = int(own_stat["start_ticks"])
        self.campaign_started_mono = time.monotonic() - (
            _proc_uptime() - self.own_start_ticks / self.ticks_per_second)
        self.handles: dict[int, dict] = {}
        self.completed: list[dict] = []
        self.stop_signal: int | None = None
        self.last_meter_mono = 0.0
        self.last_storage_bytes = 0
        self.worker_pid: int | None = None
        self.worker_process_group: int | None = None
        self.worker_returncode: int | None = None
        self.worker_state: dict | None = None
        self.worker_state_sha256: str | None = None
        self.worker_log_path = self.run_root / "worker.stdout.log"
        self.meter_path = self.run_root / "meter.jsonl"
        self.wait_path = self.run_root / "wait4.jsonl"
        self.meter_stream = None
        self.wait_stream = None
        self.worker_env = _make_worker_environment(spec["environment"], stage)

    @property
    def cpu_limit(self) -> float:
        return float(STAGE_CAPS[self.stage]["cpu_seconds"])

    @property
    def wall_limit(self) -> float:
        return float(STAGE_CAPS[self.stage]["wall_seconds"])

    @property
    def cpu_stop(self) -> float:
        return self.cpu_limit - FOOTER_CPU_RESERVE_SECONDS

    @property
    def wall_stop(self) -> float:
        return self.wall_limit - FOOTER_WALL_RESERVE_SECONDS

    def start(self) -> None:
        if sys.platform != "linux" or not sys.flags.isolated or not sys.dont_write_bytecode:
            raise SupervisorError("supervisor must use isolated, bytecode-disabled Linux Python")
        try:
            actual_python = Path(sys.executable).resolve(strict=True)
            expected_python = PYTHON.resolve(strict=True)
        except OSError as exc:
            raise SupervisorError(f"compute interpreter cannot be resolved: {exc}") from exc
        if actual_python != expected_python:
            raise SupervisorError("supervisor is not running under the allocated project interpreter")
        for key, expected in EXPECTED_ENVIRONMENT.items():
            if os.environ.get(key) != expected:
                raise SupervisorError(f"launch environment differs from the frozen project value for {key}")
        _validate_capacity(self.spec)
        self.run_root.mkdir(mode=0o700)
        self._ensure_runtime_directories()
        self.meter_stream = self.meter_path.open("x", encoding="utf-8", buffering=1)
        self.wait_stream = self.wait_path.open("x", encoding="utf-8", buffering=1)
        self.last_storage_bytes = _storage_bytes(E_ROOT)
        self.record_event(
            "campaign_started",
            experiment_id="E165",
            stage=self.stage,
            campaign_start_source="new supervisor /proc/self/stat start ticks",
            supervisor_pid=os.getpid(),
            supervisor_start_ticks=self.own_start_ticks,
            parent_pid=self.startup_proof["parent_pid"],
            parent_start_ticks=self.startup_proof["parent_start_ticks"],
            ticks_per_second=self.ticks_per_second,
            startup_proof=self.startup_proof,
            self_rusage_at_record=_ru_record(resource.getrusage(resource.RUSAGE_SELF)),
            rusage_children_at_campaign_start=_ru_record(resource.getrusage(resource.RUSAGE_CHILDREN)),
            authorization_sha256=_sha256(self.authorization_path.read_bytes()),
            run_spec_sha256=_sha256(self.spec_path.read_bytes()),
            single_attempt_reservation_id=self.authorization["single_attempt_reservation_id"],
            stage_limits=STAGE_CAPS[self.stage],
            round_limits=ROUND_LIMITS,
        )

    def _ensure_runtime_directories(self) -> None:
        requested = sorted({Path(self.worker_env[key]) for key in (
            "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
            "XDG_DATA_HOME", "XDG_STATE_HOME", "MPLCONFIGDIR", "TORCH_HOME",
            "TORCH_EXTENSIONS_DIR")})
        for path in requested:
            _safe_path(str(path), "worker cache or temp directory", root=E_ROOT,
                       must_exist=False)
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SupervisorError(f"worker cache or temp path is not a real directory: {path}")
        self.last_storage_bytes = _validate_capacity(self.spec)

    def elapsed_wall(self) -> float:
        return max(0.0, time.monotonic() - self.campaign_started_mono)

    def self_cpu(self) -> float:
        return _ru_cpu(resource.getrusage(resource.RUSAGE_SELF))

    def _children_cpu(self) -> float:
        return sum(float(row["rusage"]["user_seconds"] + row["rusage"]["system_seconds"])
                   for row in self.completed)

    def _live_cpu_rows(self) -> list[dict]:
        rows = []
        for pid, handle in list(self.handles.items()):
            try:
                current = _proc_stat(pid)
            except FileNotFoundError:
                self._reap(pid, block=False)
                continue
            if int(current["start_ticks"]) != handle["start_ticks"]:
                raise SupervisorError(f"owned PID {pid} changed start identity")
            ticks = sum(int(current[key]) for key in
                        ("utime_ticks", "stime_ticks", "cutime_ticks", "cstime_ticks"))
            rows.append({
                "pid": pid,
                "start_ticks": handle["start_ticks"],
                "process_group": handle["process_group"],
                "purpose": handle["purpose"],
                "state": current["state"],
                "own_and_waited_descendant_cpu_ticks": ticks,
                "cpu_seconds_from_proc_stat": ticks / self.ticks_per_second,
            })
        return rows

    def cpu_snapshot(self) -> dict:
        live = self._live_cpu_rows()
        live_cpu = sum(float(row["cpu_seconds_from_proc_stat"]) for row in live)
        own = self.self_cpu()
        children = self._children_cpu()
        return {
            "supervisor_rusage_self_cpu_seconds": own,
            "reaped_wait4_cpu_seconds": children,
            "live_owned_process_cpu_seconds_from_proc_stat": live_cpu,
            "live_owned_processes": live,
            "total_cpu_seconds": own + children + live_cpu,
        }

    def _write_line(self, stream, path: Path, value: dict, *, footer: bool = False) -> None:
        data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        _validate_capacity(self.spec, anticipated_bytes=len(data), reserve_footer=not footer)
        stream.write(data.decode("utf-8"))
        stream.flush()
        self.last_storage_bytes = _storage_bytes(E_ROOT)

    def record_event(self, kind: str, **fields) -> None:
        row = {"kind": kind, "monotonic": time.monotonic(),
               "campaign_elapsed_wall_seconds": self.elapsed_wall()}
        row.update(fields)
        self._write_line(self.meter_stream, self.meter_path, row)

    def record_wait4(self, handle: dict, status: int, usage: resource.struct_rusage) -> dict:
        row = {
            "pid": handle["pid"],
            "process_group": handle["process_group"],
            "start_ticks": handle["start_ticks"],
            "purpose": handle["purpose"],
            "wait_status": status,
            "exit_code": os.waitstatus_to_exitcode(status),
            "rusage": _ru_record(usage),
            "wait4_cpu_seconds_including_waited_descendants": _ru_cpu(usage),
            "reaped_monotonic": time.monotonic(),
        }
        self.completed.append(row)
        self.handles.pop(handle["pid"], None)
        self._write_line(self.wait_stream, self.wait_path, row)
        self.record_event("child_reaped", pid=handle["pid"], purpose=handle["purpose"],
                          exit_code=row["exit_code"], wait4_cpu_seconds=row["wait4_cpu_seconds_including_waited_descendants"])
        return row

    def _reap(self, pid: int, *, block: bool) -> dict | None:
        handle = self.handles.get(pid)
        if handle is None:
            return None
        flags = 0 if block else os.WNOHANG
        try:
            waited, status, usage = os.wait4(pid, flags)
        except InterruptedError:
            return self._reap(pid, block=block)
        if waited == 0:
            return None
        return self.record_wait4(handle, status, usage)

    def reap_all_nonblocking(self) -> None:
        for pid in list(self.handles):
            self._reap(pid, block=False)

    def _set_process_group(self, pid: int) -> None:
        try:
            os.setpgid(pid, pid)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise
        if os.getpgid(pid) != pid:
            raise SupervisorError("owned process group identity was not established")

    def _fork_gated(self, purpose: str, child_entry) -> tuple[dict, int]:
        gate_read, gate_write = os.pipe()
        try:
            pid = os.fork()
        except BaseException:
            os.close(gate_read)
            os.close(gate_write)
            raise
        if pid == 0:
            try:
                os.close(gate_write)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                os.setpgid(0, 0)
                if os.read(gate_read, 1) != b"1":
                    os._exit(125)
                os.close(gate_read)
                child_entry()
                os._exit(0)
            except BaseException:
                os._exit(126)
        os.close(gate_read)
        try:
            start_ticks = int(_proc_stat(pid)["start_ticks"])
            handle = {
                "pid": pid,
                "process_group": pid,
                "start_ticks": start_ticks,
                "purpose": purpose,
                "spawned_monotonic": time.monotonic(),
            }
            self.handles[pid] = handle
            self._set_process_group(pid)
            self.record_event("child_started", **handle)
            os.write(gate_write, b"1")
            os.close(gate_write)
            return handle, pid
        except BaseException:
            os.close(gate_write)
            handle = self.handles.get(pid)
            if handle is not None:
                try:
                    self._signal_owned_group(handle, signal.SIGKILL)
                except BaseException:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                try:
                    self._wait_for_reap(pid, timeout=REAP_TIMEOUT_SECONDS)
                except BaseException:
                    pass
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                deadline = time.monotonic() + REAP_TIMEOUT_SECONDS
                while time.monotonic() < deadline:
                    try:
                        waited, _status, _usage = os.wait4(pid, os.WNOHANG)
                    except OSError:
                        break
                    if waited == pid:
                        break
                    time.sleep(0.01)
            raise

    def _launch_cpu_tree_precheck(self) -> None:
        report_read, report_write = os.pipe()

        def burn_cpu(duration: float) -> None:
            deadline = time.process_time() + duration
            value = 0
            while time.process_time() < deadline:
                value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
            if value == -1:
                os._exit(124)

        def child_entry() -> None:
            os.close(report_read)
            grandchild = os.fork()
            if grandchild == 0:
                burn_cpu(0.06)
                os._exit(0)
            grandchild_start = int(_proc_stat(grandchild)["start_ticks"])
            waited, status, grandchild_usage = os.wait4(grandchild, 0)
            if waited != grandchild or status != 0:
                os._exit(123)
            burn_cpu(0.06)
            own_usage = resource.getrusage(resource.RUSAGE_SELF)
            report = {
                "grandchild": {"pid": grandchild, "start_ticks": grandchild_start,
                               "wait_status": status, "rusage": _ru_record(grandchild_usage)},
                "child_self_rusage": _ru_record(own_usage),
                "reported_tree_cpu_seconds": _ru_cpu(own_usage) + _ru_cpu(grandchild_usage),
                "reported_descendant_cpu_seconds": _ru_cpu(grandchild_usage),
            }
            os.write(report_write, _json_bytes(report))
            os.close(report_write)

        handle, pid = self._fork_gated("cpu_descendant_accounting_precheck", child_entry)
        os.close(report_write)
        report_data = bytearray()
        deadline = time.monotonic() + CPU_PROBE_WALL_TIMEOUT_SECONDS
        try:
            while True:
                self._check_limits()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SupervisorError("CPU descendant precheck exceeded its five-second bound")
                try:
                    readable, _, _ = select.select([report_read], [], [], min(POLL_INTERVAL_SECONDS, remaining))
                except InterruptedError:
                    continue
                if not readable:
                    continue
                chunk = os.read(report_read, 4096)
                if not chunk:
                    break
                report_data.extend(chunk)
                if len(report_data) > 65536:
                    raise SupervisorError("CPU precheck report exceeded its fixed bound")
        finally:
            os.close(report_read)
        waited = self._wait_for_reap(pid, timeout=REAP_TIMEOUT_SECONDS)
        if waited is None:
            raise SupervisorError("CPU precheck child was not reaped")
        try:
            report = json.loads(report_data)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SupervisorError("CPU descendant precheck did not return valid evidence") from exc
        direct = waited["wait4_cpu_seconds_including_waited_descendants"]
        reported = float(report["reported_tree_cpu_seconds"])
        tolerance = max(0.02, reported * 0.20)
        if waited["exit_code"] != 0 or reported < 0.08 or abs(direct - reported) > tolerance:
            raise SupervisorError("wait4 descendant accounting precheck failed")
        if self._group_exists(handle["process_group"]):
            self._cleanup_group_after_reap(handle["process_group"])
            raise SupervisorError("CPU descendant precheck left a process in its owned group")
        children_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        if abs(_ru_cpu(children_rusage) - self._children_cpu()) > 0.02:
            raise SupervisorError("RUSAGE_CHILDREN and recorded wait4 accounting differ")
        self.record_event("cpu_descendant_precheck_passed", child=handle, report=report,
                          supervisor_wait4_cpu_seconds=direct,
                          comparison_tolerance_seconds=tolerance,
                          rusage_children_cpu_seconds=_ru_cpu(children_rusage))
        self._check_limits()

    def _launch_stop_precheck(self) -> None:
        def sleep_until_signalled() -> None:
            while True:
                time.sleep(0.1)

        handle, pid = self._fork_gated("owned_process_group_stop_precheck", sleep_until_signalled)
        self._wait_for_owned_live(pid, timeout=0.25)
        self._signal_owned_group(handle, signal.SIGTERM)
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        waited = None
        while time.monotonic() < deadline:
            waited = self._reap(pid, block=False)
            if waited is not None:
                break
            time.sleep(0.01)
        if waited is None:
            self._signal_owned_group(handle, signal.SIGKILL)
            waited = self._wait_for_reap(pid, timeout=REAP_TIMEOUT_SECONDS)
        if waited is None or waited["exit_code"] != -signal.SIGTERM or waited["wait_status"] != signal.SIGTERM:
            raise SupervisorError("recorded process-group stop precheck failed")
        if self._group_exists(handle["process_group"]):
            self._cleanup_group_after_reap(handle["process_group"])
            raise SupervisorError("stop precheck left a process in its owned group")
        self.record_event("owned_process_group_stop_precheck_passed", child=handle,
                          exit_code=waited["exit_code"], wait_status=waited["wait_status"],
                          wait4_cpu_seconds=waited["wait4_cpu_seconds_including_waited_descendants"])
        self._check_limits()

    def _wait_for_owned_live(self, pid: int, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_limits()
            if pid not in self.handles:
                raise SupervisorError("owned precheck child exited before observation")
            try:
                current = _proc_stat(pid)
            except FileNotFoundError:
                self._reap(pid, block=False)
                continue
            if int(current["start_ticks"]) != self.handles[pid]["start_ticks"]:
                raise SupervisorError("owned precheck PID changed identity")
            if current["state"] not in ("Z", "X"):
                return
        raise SupervisorError("owned precheck child did not become live in time")

    def _signal_owned_group(self, handle: dict, sig: int) -> None:
        pid = int(handle["pid"])
        if pid not in self.handles:
            raise SupervisorError("refusing to signal a process without an active owned handle")
        current = _proc_stat(pid)
        if int(current["start_ticks"]) != int(handle["start_ticks"]):
            raise SupervisorError("refusing to signal a reused PID")
        if os.getpgid(pid) != int(handle["process_group"]):
            raise SupervisorError("refusing to signal a changed process group")
        os.killpg(int(handle["process_group"]), sig)
        try:
            self.record_event("owned_process_group_signalled", pid=pid,
                              process_group=handle["process_group"], signal=sig,
                              start_ticks=handle["start_ticks"])
        except BaseException:
            pass

    def _group_exists(self, process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _cleanup_group_after_reap(self, process_group: int) -> None:
        if not self._group_exists(process_group):
            return
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        while time.monotonic() < deadline and self._group_exists(process_group):
            time.sleep(0.01)
        if self._group_exists(process_group):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + REAP_TIMEOUT_SECONDS
        while time.monotonic() < deadline and self._group_exists(process_group):
            time.sleep(0.01)
        if self._group_exists(process_group):
            raise SupervisorError("owned process group remained live after bounded stop and reap")

    def _check_limits(self) -> dict:
        self.reap_all_nonblocking()
        snapshot = self.cpu_snapshot()
        wall = self.elapsed_wall()
        if self.stop_signal is not None:
            raise SupervisorError(f"supervisor received stop signal {self.stop_signal}")
        if snapshot["total_cpu_seconds"] >= self.cpu_stop:
            raise SupervisorError("stage reached its CPU safety stop")
        if wall >= self.wall_stop:
            raise SupervisorError("stage reached its wall safety stop")
        now = time.monotonic()
        if now - self.last_meter_mono >= METER_INTERVAL_SECONDS:
            self.record_event("meter_sample", cpu_snapshot=snapshot, stage_wall_seconds=wall,
                              e165_apparent_bytes=_storage_bytes(E_ROOT))
            self.last_meter_mono = now
        return snapshot

    def _wait_for_reap(self, pid: int, *, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self._reap(pid, block=False)
            if result is not None:
                return result
            time.sleep(0.01)
        return self._reap(pid, block=False)

    def _terminate_owned(self, handle: dict, reason: str) -> None:
        pid = int(handle["pid"])
        if pid not in self.handles:
            return
        self.record_event("owned_child_termination_requested", pid=pid,
                          purpose=handle["purpose"], reason=reason)
        self._signal_owned_group(handle, signal.SIGTERM)
        waited = self._wait_for_reap(pid, timeout=STOP_GRACE_SECONDS)
        if waited is None and pid in self.handles:
            self._signal_owned_group(handle, signal.SIGKILL)
            waited = self._wait_for_reap(pid, timeout=REAP_TIMEOUT_SECONDS)
        if waited is None:
            raise SupervisorError(f"owned child {pid} could not be reaped")
        self._cleanup_group_after_reap(int(handle["process_group"]))
        self.record_event("owned_child_terminated", pid=pid, purpose=handle["purpose"],
                          exit_code=waited["exit_code"], reason=reason)

    def _terminate_worker(self, reason: str) -> None:
        if self.worker_pid is not None and self.worker_pid in self.handles:
            self._terminate_owned(self.handles[self.worker_pid], reason)
        elif self.worker_process_group is not None and self._group_exists(self.worker_process_group):
            self._cleanup_group_after_reap(self.worker_process_group)

    def _launch_worker(self) -> tuple[dict, int, int]:
        worker = STAGE_WORKERS[self.stage]
        log_stream = self.worker_log_path.open("xb", buffering=0)
        log_stream.close()
        read_fd, write_fd = os.pipe()

        def child_entry() -> None:
            os.close(read_fd)
            os.dup2(write_fd, 1)
            os.dup2(write_fd, 2)
            if write_fd not in (1, 2):
                os.close(write_fd)
            os.chdir(str(SOURCE_CWD))
            os.closerange(3, 65536)
            argv = [str(PYTHON)] + self.spec["worker_argv"]
            os.execve(str(PYTHON), argv, self.worker_env)

        try:
            handle, pid = self._fork_gated("scientific_worker", child_entry)
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            raise
        os.close(write_fd)
        self.worker_pid = pid
        self.worker_process_group = int(handle["process_group"])
        self.record_event("worker_command_started", stage=self.stage, handle=handle,
                          argv=self.spec["worker_argv"], run_root=str(self.run_root),
                          worker_log_path=str(self.worker_log_path))
        return handle, pid, read_fd

    def _monitor_worker(self, pid: int, output_fd: int) -> dict:
        waited = None
        eof = False
        log_path = self.worker_log_path
        try:
            with log_path.open("ab", buffering=0) as log_stream:
                while waited is None or not eof:
                    self._check_limits()
                    if not eof:
                        try:
                            readable, _, _ = select.select([output_fd], [], [], POLL_INTERVAL_SECONDS)
                        except InterruptedError:
                            readable = []
                        if readable:
                            chunk = os.read(output_fd, 65536)
                            if not chunk:
                                eof = True
                            else:
                                _validate_capacity(self.spec, anticipated_bytes=len(chunk))
                                log_stream.write(chunk)
                    if waited is None:
                        waited = self._reap(pid, block=False)
                    if waited is None and eof:
                        time.sleep(POLL_INTERVAL_SECONDS)
        finally:
            os.close(output_fd)
        if waited is None:
            waited = self._wait_for_reap(pid, timeout=REAP_TIMEOUT_SECONDS)
        if waited is None:
            raise SupervisorError("worker did not produce a wait4 result")
        if self.worker_process_group is not None and self._group_exists(self.worker_process_group):
            self._cleanup_group_after_reap(self.worker_process_group)
            raise SupervisorError("worker exited while leaving a process in its owned group")
        self.worker_returncode = waited["exit_code"]
        self.record_event("worker_lifecycle_closed", stage=self.stage,
                          exit_code=waited["exit_code"], wait_status=waited["wait_status"],
                          wait4_cpu_seconds=waited["wait4_cpu_seconds_including_waited_descendants"],
                          worker_log_path=str(log_path))
        return waited

    def _validate_worker_state(self, exit_code: int) -> dict:
        if exit_code != 0:
            raise SupervisorError(f"stage worker exited with code {exit_code}")
        state_path = _safe_path(str(self.run_root / "state_worker.json"),
                                "state_worker.json", root=self.run_root, must_exist=True)
        try:
            info = state_path.lstat()
            raw = state_path.read_bytes()
            worker = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SupervisorError(f"cannot read worker completion state: {exc}") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not isinstance(worker, dict):
            raise SupervisorError("state_worker.json must be a single-link regular JSON object")
        if (worker.get("format") != "e165_worker_state_v1" or worker.get("stage") != self.stage
                or worker.get("status") != "COMPLETE"):
            raise SupervisorError("worker completion state identity or status is invalid")
        if worker.get("source_identity_sha256") != self.spec["source_identity_sha256"]:
            raise SupervisorError("worker source_identity_sha256 differs from the frozen spec")
        for field in ("environment_interactions", "gpu_devices", "real_candidate_fit_calls",
                      "candidate_recovery_calls"):
            if not _is_int(worker.get(field)) or worker[field] != 0:
                raise SupervisorError(f"worker state {field} must be the exact integer zero")
        if self.stage == "CHECKS":
            expected = self.spec.get("expected_test_nodeids")
            executed = worker.get("executed_test_nodeids")
            if (not isinstance(expected, list) or not expected
                    or any(not isinstance(item, str) or not item for item in expected)
                    or len(set(expected)) != len(expected)):
                raise SupervisorError("CHECKS spec expected_test_nodeids must be nonempty and unique")
            if not isinstance(executed, list) or executed != expected:
                raise SupervisorError("CHECKS executed_test_nodeids differ from the frozen expected list")
            for field in ("failed", "errors", "skipped"):
                if not _is_int(worker.get(field)) or worker[field] != 0:
                    raise SupervisorError(f"CHECKS worker state {field} must be the exact integer zero")
        elif self.stage == "DIAGNOSTIC":
            if not _is_int(worker.get("logical_prefixes")) or worker["logical_prefixes"] != 16:
                raise SupervisorError("DIAGNOSTIC requires logical_prefixes=16")
            oracle_count = worker.get("full_training_oracle_evaluations")
            package_count = worker.get("new_complete_score_packages")
            if not _is_int(oracle_count) or oracle_count > ROUND_LIMITS["real_full_training_oracle_evaluations"]:
                raise SupervisorError("DIAGNOSTIC full_training_oracle_evaluations exceeds its fixed cap")
            if not _is_int(package_count) or package_count > ROUND_LIMITS["new_complete_score_packages"]:
                raise SupervisorError("DIAGNOSTIC new_complete_score_packages exceeds its fixed cap")
            if not _is_int(worker.get("anchor_confirmed_count")) or worker["anchor_confirmed_count"] != 1:
                raise SupervisorError("DIAGNOSTIC requires anchor_confirmed_count=1")
            scores = worker.get("completed_scores")
            if not isinstance(scores, list) or len(scores) != package_count:
                raise SupervisorError("DIAGNOSTIC completed_scores length differs from successful packages")
            if not isinstance(worker.get("cohort_lock_sha256"), str) or not worker["cohort_lock_sha256"].strip():
                raise SupervisorError("DIAGNOSTIC cohort_lock_sha256 must be nonempty")
        else:
            audit_result = worker.get("audit_result")
            if not isinstance(audit_result, dict) or audit_result.get("status") != "COMPLETE_VALID":
                raise SupervisorError("AUDIT audit_result.status must be COMPLETE_VALID")
            if worker.get("remote_statistics_completed") is not True:
                raise SupervisorError("AUDIT remote_statistics_completed must be true")
            audit_path = _safe_path(str(self.run_root / "audit.json"), "audit.json",
                                    root=self.run_root, must_exist=True)
            audit_info = audit_path.lstat()
            if not stat.S_ISREG(audit_info.st_mode) or audit_info.st_nlink != 1:
                raise SupervisorError("AUDIT audit.json must be a single-link regular file")
            audit, _ = _read_json(audit_path, "saved audit.json")
            if audit.get("status") != audit_result["status"]:
                raise SupervisorError("AUDIT audit_result does not match the saved audit.json")
        self.worker_state = worker
        self.worker_state_sha256 = _sha256(raw)
        return worker

    def _children_usage_audit(self) -> dict:
        if self.handles:
            raise SupervisorError("cannot finalize while a recorded child remains unreaped")
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        initial = self.startup_proof["rusage_children_post_authorization_pre_fork"]
        initial_user = float(initial["user_seconds"])
        initial_system = float(initial["system_seconds"])
        final_user = float(children.ru_utime)
        final_system = float(children.ru_stime)
        wait4_user = sum(float(row["rusage"]["user_seconds"]) for row in self.completed)
        wait4_system = sum(float(row["rusage"]["system_seconds"]) for row in self.completed)
        delta_user = final_user - initial_user
        delta_system = final_system - initial_system
        if initial_user != 0.0 or initial_system != 0.0:
            raise SupervisorError("initial RUSAGE_CHILDREN CPU was not zero")
        if abs(wait4_user - delta_user) > 0.03 or abs(wait4_system - delta_system) > 0.03:
            raise SupervisorError("final wait4 sum differs from RUSAGE_CHILDREN")
        return {
            "initial_rusage_children_cpu_seconds": {"user": initial_user, "system": initial_system},
            "final_rusage_children_cpu_seconds": {"user": final_user, "system": final_system},
            "rusage_children_delta_cpu_seconds": {"user": delta_user, "system": delta_system},
            "wait4_sum_cpu_seconds": {"user": wait4_user, "system": wait4_system},
            "difference_seconds": {"user": abs(wait4_user - delta_user),
                                   "system": abs(wait4_system - delta_system)},
        }

    def _current_cost_row(self, snapshot: dict, wall: float) -> dict:
        return {
            "stage": self.stage,
            "observed_cpu_seconds": float(snapshot["total_cpu_seconds"]),
            "bounded_cpu_seconds_upper": float(snapshot["total_cpu_seconds"] + FOOTER_CPU_RESERVE_SECONDS),
            "observed_wall_seconds": float(wall),
            "bounded_wall_seconds_upper": float(wall + FOOTER_WALL_RESERVE_SECONDS),
        }

    def _round_cost_ledger(self, current_row: dict) -> dict:
        rows = list(self.prior_costs) + [current_row]
        return {
            "stages": rows,
            "observed_cpu_seconds": sum(row["observed_cpu_seconds"] for row in rows),
            "bounded_cpu_seconds_upper": sum(row["bounded_cpu_seconds_upper"] for row in rows),
            "observed_wall_seconds": sum(row["observed_wall_seconds"] for row in rows),
            "bounded_wall_seconds_upper": sum(row["bounded_wall_seconds_upper"] for row in rows),
        }

    def _effective_limits(self) -> dict:
        stage_limit = STAGE_CAPS[self.stage]
        return {
            "cpu_seconds": stage_limit["cpu_seconds"],
            "campaign_wall_seconds": stage_limit["wall_seconds"],
            "new_artifact_cache_temp_bytes": ROUND_LIMITS["new_artifact_cache_temp_bytes"],
            "workers": ROUND_LIMITS["workers"],
            "numeric_threads_per_worker": ROUND_LIMITS["numeric_threads_per_worker"],
            "gpu_devices": ROUND_LIMITS["gpu_devices"],
            "environment_interactions": ROUND_LIMITS["environment_interactions"],
            "real_candidate_fit_calls": ROUND_LIMITS["real_candidate_fit_calls"],
            "candidate_recovery_calls": ROUND_LIMITS["candidate_recovery_calls"],
            "real_full_training_oracle_evaluations": ROUND_LIMITS["real_full_training_oracle_evaluations"],
            "new_complete_score_packages": ROUND_LIMITS["new_complete_score_packages"],
            "logical_paired_prefixes": ROUND_LIMITS["logical_paired_prefixes"],
            "aggregate_remote_project_bytes": ROUND_LIMITS["aggregate_remote_project_bytes"],
        }

    def finalize(self, status: str, *, return_code: int, error: dict | None = None) -> int:
        for handle in list(self.handles.values()):
            try:
                self._terminate_owned(handle, "finalization cleanup")
            except BaseException:
                status = "OWNED_PROCESS_STOP_FAILED"
                return_code = 2
        try:
            self._terminate_worker("finalization group cleanup")
        except BaseException:
            status = "OWNED_PROCESS_STOP_FAILED"
            return_code = 2
        self.reap_all_nonblocking()
        audit = None
        audit_error = None
        try:
            audit = self._children_usage_audit()
        except BaseException as exc:
            audit_error = f"{type(exc).__name__}: {exc}"
            status = "ACCOUNTING_INVALID"
            return_code = 2
        snapshot = self.cpu_snapshot()
        wall = self.elapsed_wall()
        cpu_upper = snapshot["total_cpu_seconds"] + FOOTER_CPU_RESERVE_SECONDS
        wall_upper = wall + FOOTER_WALL_RESERVE_SECONDS
        if status == "COMPLETE" and (cpu_upper > self.cpu_limit or wall_upper > self.wall_limit):
            status = "FINALIZATION_RESERVE_EXCEEDED"
            return_code = 2
        current_cost = self._current_cost_row(snapshot, wall)
        ledger = self._round_cost_ledger(current_cost)
        if (ledger["bounded_cpu_seconds_upper"] > ROUND_LIMITS["cpu_seconds"]
                or ledger["bounded_wall_seconds_upper"] > ROUND_LIMITS["cumulative_controlled_wall_seconds"]):
            status = "ROUND_LIMIT_EXCEEDED"
            return_code = 2
        worker_counts = {
            "environment_interactions": None,
            "gpu_devices": None,
            "real_candidate_fit_calls": None,
            "candidate_recovery_calls": None,
            "worker_count_status": "UNKNOWN_NO_STATE_WORKER",
        }
        if self.worker_pid is None:
            worker_counts.update({
                "environment_interactions": 0,
                "gpu_devices": 0,
                "real_candidate_fit_calls": 0,
                "candidate_recovery_calls": 0,
                "worker_count_status": "NO_WORKER_STARTED_ZERO",
            })
        elif self.worker_state is not None:
            for field in ("environment_interactions", "gpu_devices", "real_candidate_fit_calls",
                          "candidate_recovery_calls"):
                value = self.worker_state.get(field)
                if _is_int(value):
                    worker_counts[field] = value
            worker_counts["worker_count_status"] = "KNOWN_FROM_STATE_WORKER"
        if (self.completed and len(self.completed) != 3) or self.handles:
            status = "ACCOUNTING_INVALID"
            return_code = 2
        expected_purposes = {"cpu_descendant_accounting_precheck",
                             "owned_process_group_stop_precheck", "scientific_worker"}
        if status == "COMPLETE" and {row["purpose"] for row in self.completed} != expected_purposes:
            status = "ACCOUNTING_INVALID"
            return_code = 2
        try:
            self.record_event("campaign_finalizing", status=status, return_code=return_code,
                              cpu_snapshot=snapshot, campaign_wall_seconds=wall,
                              bounded_cpu_seconds_upper=cpu_upper,
                              bounded_wall_seconds_upper=wall_upper,
                              wait4_children_audit=audit, wait4_audit_error=audit_error,
                              finalization_reserve={"cpu_seconds": FOOTER_CPU_RESERVE_SECONDS,
                                                    "wall_seconds": FOOTER_WALL_RESERVE_SECONDS,
                                                    "byte_reserve": FOOTER_BYTE_RESERVE})
            _validate_capacity(self.spec, reserve_footer=False)
            e_bytes = _storage_bytes(E_ROOT)
            state = {
                "experiment_id": "E165",
                "stage": self.stage,
                "status": status,
                "return_code": return_code,
                "worker_return_code": self.worker_returncode,
                "run_spec_sha256": _sha256(self.spec_path.read_bytes()),
                "authorization_sha256": _sha256(self.authorization_path.read_bytes()),
                "single_attempt_reservation_id": self.authorization["single_attempt_reservation_id"],
                "source_identity_sha256": self.spec["source_identity_sha256"],
                "owned_child_handles_final": sorted(self.handles),
                "completed_direct_children": len(self.completed),
                "child_wait4_records": [
                    {"pid": row["pid"], "purpose": row["purpose"], "exit_code": row["exit_code"],
                     "wait_status": row["wait_status"], "process_group": row["process_group"],
                     "start_ticks": row["start_ticks"],
                     "wait4_cpu_seconds_including_waited_descendants": row["wait4_cpu_seconds_including_waited_descendants"]}
                    for row in self.completed
                ],
                "cpu_snapshot_before_final_footer": snapshot,
                "bounded_campaign_cpu_seconds_upper": cpu_upper,
                "cpu_total_scope": CPU_TOTAL_SCOPE,
                "campaign_wall_seconds_before_final_footer": wall,
                "bounded_campaign_wall_seconds_upper": wall_upper,
                "campaign_start_source": CAMPAIGN_START_SOURCE,
                "campaign_cost_scope": {
                    "origin": "fresh supervisor process birth ticks; includes Python startup, same-stage prechecks, worker, stop, reap and finalization",
                    "cpu": "supervisor lifetime RUSAGE_SELF plus direct-child wait4 including waited descendants, with live owned counters during execution",
                    "unmeasured_tail": "5 CPU / 10 wall second conservative finalization reserve; 65536 byte footer reserve",
                    "outside_campaign": "external SSH transport, source transfer, admission and parent launch-shell operations; operational elapsed reported separately, unmeasured CPU UNKNOWN",
                },
                "limits": self._effective_limits(),
                "stage_limits": STAGE_CAPS[self.stage],
                "round_limits": ROUND_LIMITS,
                "safety_stops": {"cpu_seconds": self.cpu_stop, "wall_seconds": self.wall_stop},
                "finalization_reserve": {"cpu_seconds": FOOTER_CPU_RESERVE_SECONDS,
                                          "wall_seconds": FOOTER_WALL_RESERVE_SECONDS,
                                          "byte_reserve": FOOTER_BYTE_RESERVE},
                "e_root_apparent_bytes": e_bytes,
                "project_bytes_before_non_e165_baseline": self.spec["project_bytes_before"],
                "aggregate_project_apparent_bytes": self.spec["project_bytes_before"] + e_bytes,
                "wait4_children_audit": audit,
                "wait4_audit_error": audit_error,
                "final_footer_tail_measurement": FINAL_FOOTER_DISCLOSURE,
                **worker_counts,
                "worker_state_sha256": self.worker_state_sha256,
                "prior_stage_state_descriptors": self.spec["prior_stage_states"],
                "prior_stage_costs": self.prior_costs,
                "stage_cost": current_cost,
                "round_cost_ledger": ledger,
                "external_operational_elapsed_seconds": None,
                "external_operational_cpu_seconds": "UNKNOWN",
                "error": error,
                "evidence_note": "The final footer is conservatively reserved, not measured. Parent launch shell, SSH, transfer and admission are outside the stage origin; disclose operational elapsed separately and unmeasured CPU as UNKNOWN.",
            }
            state_path = self.run_root / "state.json"
            state_bytes = _json_bytes(state)
            _validate_capacity(self.spec, anticipated_bytes=len(state_bytes), reserve_footer=False)
            with state_path.open("x", encoding="utf-8") as stream:
                stream.write(state_bytes.decode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as exc:
            try:
                self.record_event("finalization_error", exception_type=type(exc).__name__, message=str(exc))
            except BaseException:
                pass
            status = "FINALIZATION_INVALID"
            return_code = 2
        finally:
            for stream in (self.meter_stream, self.wait_stream):
                if stream is not None:
                    try:
                        stream.flush()
                        os.fsync(stream.fileno())
                        stream.close()
                    except OSError:
                        pass
        return return_code

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        try:
            self.start()
            self.record_event("precheck_started", precheck="zero_model_metering")
            self._launch_cpu_tree_precheck()
            self.record_event("precheck_started", precheck="owned_group_stop")
            self._launch_stop_precheck()
            self.record_event("all_prechecks_passed", cpu_snapshot=self._check_limits(),
                              campaign_wall_seconds=self.elapsed_wall(),
                              configured_numeric_threads=1, configured_gpu_devices=0,
                              actual_compute_interpreter=str(PYTHON.resolve(strict=True)))
            _validate_capacity(self.spec)
            handle, pid, output_fd = self._launch_worker()
            worker_result = self._monitor_worker(pid, output_fd)
            _validate_capacity(self.spec)
            self._validate_worker_state(worker_result["exit_code"])
            return self.finalize("COMPLETE", return_code=0)
        except BaseException as exc:
            error = {"exception_type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc()}
            try:
                if self.meter_stream is not None:
                    self.record_event("campaign_error", **error)
            except BaseException:
                pass
            try:
                self._terminate_worker("supervisor error or failed precheck")
            except BaseException:
                pass
            status = "PRECHECK_FAILED" if self.worker_pid is None else "INCOMPLETE"
            return self.finalize(status, return_code=2, error=error)

    def _request_stop(self, signum, frame) -> None:
        self.stop_signal = int(signum)


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise one authorized Linux E165 stage")
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_CAPS))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    try:
        spec, authorization, spec_path, authorization_path, run_root = _validate_spec_and_authorization(
            args.stage, args.spec, args.authorization)
        if not OUTPUT_CONTRACT_FROZEN:
            raise PermissionError("E165 worker output contract is not frozen")
        _verify_input_manifests(spec)
        prior_states, prior_costs = _load_and_validate_prior_states(spec)
    except PermissionError as exc:
        print(_json_bytes({"kind": "authorization_refused", "message": str(exc)}).decode("utf-8"),
              file=sys.stderr)
        return 3
    except BaseException as exc:
        print(_json_bytes({"kind": "preflight_refused", "exception_type": type(exc).__name__,
                           "message": str(exc)}).decode("utf-8"), file=sys.stderr)
        return 3
    startup_proof = None
    try:
        startup_proof, startup_errors = _capture_startup_proof()
        if startup_errors:
            raise SupervisorError("; ".join(startup_errors))
    except BaseException as exc:
        failure = {"kind": "startup_precheck_failed", "exception_type": type(exc).__name__,
                   "message": str(exc)}
        if startup_proof is not None:
            failure["startup_proof"] = startup_proof
        print(_json_bytes(failure).decode("utf-8"), file=sys.stderr)
        return 3
    try:
        campaign = Campaign(args.stage, spec, authorization, spec_path, authorization_path,
                            run_root, startup_proof, prior_states, prior_costs)
    except BaseException as exc:
        print(_json_bytes({"kind": "supervisor_initialization_failed",
                           "startup_proof": startup_proof,
                           "exception_type": type(exc).__name__, "message": str(exc)}).decode("utf-8"),
              file=sys.stderr)
        return 3
    return campaign.run()


if __name__ == "__main__":
    raise SystemExit(main())

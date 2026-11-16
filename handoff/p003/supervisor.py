#!/usr/bin/env python3
"""H037 one-phase Linux supervisor for an explicitly authorized CHECKS or RUN.

This executable is not itself authorization to run research. It requires exact
per-stage specs and authorization records, then records best-effort accounting
for its own process and the one process group it starts. It does not provide a
filesystem quota, cgroup, GPU device-node, environment-interaction, or escape-
proof process-tree boundary; those limits are disclosed in the state and README.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pwd
import resource
import select
import signal
import stat
import sys
import time
import traceback
from typing import Any
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path("/data/robot/p001")
FORMAT = "h037_phase_spec_v1"
AUTH_FORMAT = "h037_phase_authorization_v1"
STATE_FORMAT = "h037_supervisor_state_v1"
WORKER_STATE_FORMAT = "h037_worker_state_v1"
PROJECT_BYTE_CAP = 100_000_000_000
IDENTITY_ALGORITHM = (
    "SHA256 of UTF-8 rows sorted by ordinal path, each lower-case file sha256 TAB "
    "decimal bytes TAB path LF"
)
PHASES = {"CHECKS", "RUN"}
FORBIDDEN_BASENAMES = {"AGENTS.md", "Agent.md"}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
EMPTY_GPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
}
METER_INTERVAL_SECONDS = 0.25
POLL_INTERVAL_SECONDS = 0.025
STOP_GRACE_SECONDS = 0.75
REAP_TIMEOUT_SECONDS = 1.0
CACHE_ENV = {
    "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
    "XDG_DATA_HOME", "XDG_STATE_HOME", "MPLCONFIGDIR", "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
}


class Refusal(RuntimeError):
    """A static or dynamic admission condition failed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"cannot read {label} as JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Refusal(f"{label} must be a JSON object")
    return value, raw


def _is_int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _is_finite(value: object, *, minimum: float = 0.0) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= minimum
    except OverflowError:
        return False


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _no_symlink_components(
    path: Path, *, allow_missing_leaf: bool = False, allow_missing_tail: bool = False
) -> Path:
    """Resolve a canonical path while refusing symlink components."""
    if not path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise Refusal(f"path is not absolute and canonical: {path}")
    current = Path(path.anchor)
    parts = path.parts[1:]
    missing_tail = False
    for index, part in enumerate(parts):
        current = current / part
        if missing_tail:
            continue
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing_tail or (allow_missing_leaf and index == len(parts) - 1):
                missing_tail = True
                break
            raise Refusal(f"required path component is missing: {current}")
        if stat.S_ISLNK(info.st_mode):
            raise Refusal(f"symlink path component is forbidden: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise Refusal(f"non-directory path component: {current}")
    return path


def _safe_project_path(value: object, label: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str):
        raise Refusal(f"{label} must be a path string")
    path = Path(value)
    if not _beneath(path, PROJECT_ROOT) or path == PROJECT_ROOT:
        raise Refusal(f"{label} must be below the exact allocated project root")
    _no_symlink_components(path, allow_missing_tail=not must_exist)
    if must_exist and not path.exists():
        raise Refusal(f"{label} does not exist: {path}")
    if not must_exist and path.exists():
        raise Refusal(f"{label} already exists; refusing to overwrite: {path}")
    return path


def _verify_manifest(value: object, label: str, root: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Refusal(f"{label} must be a nonempty exact-file manifest")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise Refusal(f"{label}[{index}] must contain only path, bytes, sha256")
        path_text = row["path"]
        if not isinstance(path_text, str) or Path(path_text).name in FORBIDDEN_BASENAMES:
            raise Refusal(f"{label}[{index}] has a forbidden or invalid path")
        path = Path(path_text)
        if not _beneath(path, root) or path == root:
            raise Refusal(f"{label}[{index}] is outside its exact manifest root")
        if path_text in seen:
            raise Refusal(f"{label} contains a duplicate path")
        seen.add(path_text)
        _no_symlink_components(path)
        if not _is_int(row["bytes"]) or not _is_sha256(row["sha256"]):
            raise Refusal(f"{label}[{index}] lacks exact byte and SHA values")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise Refusal(f"{label}[{index}] is not a single-link regular file")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise Refusal(f"cannot read allowlisted {label}[{index}]: {exc}") from exc
        if len(raw) != row["bytes"] or _sha256(raw) != row["sha256"]:
            raise Refusal(f"{label}[{index}] byte count or SHA differs")
        result.append(dict(row))
    return result


def _validate_environment(value: object, phase_root: Path) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
    ):
        raise Refusal("environment must be a string-to-string object")
    if "HOME" in value:
        raise Refusal("the supervisor must not change HOME")
    expected = {**THREAD_ENV, **EMPTY_GPU_ENV}
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise Refusal(f"{key} must be fixed to {wanted!r}")
    for key in CACHE_ENV:
        item = value.get(key)
        if item is None:
            raise Refusal(f"{key} must be explicitly set under the fresh phase output root")
        path = Path(item)
        if not path.is_absolute() or not _beneath(path, phase_root):
            raise Refusal(f"{key} must point under the fresh phase output root")
    if not value.get("PATH"):
        raise Refusal("PATH must be explicit in the phase environment")
    return dict(value)


def _validate_limits(spec: dict[str, Any], auth: dict[str, Any], phase: str) -> None:
    phase_limits = spec.get("phase_limits")
    round_limits = spec.get("round_limits")
    if not isinstance(phase_limits, dict) or auth.get("phase_limits") != phase_limits:
        raise Refusal("phase limits are not byte-bound to the coordinator authorization")
    if not isinstance(round_limits, dict) or auth.get("round_limits") != round_limits:
        raise Refusal("round limits are not byte-bound to the coordinator authorization")
    for field in (
        "cpu_seconds", "wall_seconds", "footer_cpu_seconds", "footer_wall_seconds",
        "new_artifact_cache_temp_bytes",
    ):
        if not _is_finite(phase_limits.get(field), minimum=0.001):
            raise Refusal(f"phase_limits.{field} must be finite and positive")
    if phase_limits["footer_cpu_seconds"] >= phase_limits["cpu_seconds"]:
        raise Refusal("CPU footer reserve must fit inside the phase allocation")
    if phase_limits["footer_wall_seconds"] >= phase_limits["wall_seconds"]:
        raise Refusal("wall footer reserve must fit inside the phase allocation")
    if not _is_int(phase_limits.get("footer_bytes"), minimum=1):
        raise Refusal("phase_limits.footer_bytes must reserve a positive integer footer allowance")
    if phase_limits["footer_bytes"] >= phase_limits["new_artifact_cache_temp_bytes"]:
        raise Refusal("footer byte reserve must fit inside the phase output allowance")
    required_round = (
        "cpu_seconds", "wall_seconds", "new_artifact_cache_temp_bytes",
        "workers", "numeric_threads_per_worker", "gpu_devices",
        "max_environment_interactions", "max_persistent_package_bytes",
    )
    if any(not _is_int(round_limits.get(key)) for key in required_round):
        raise Refusal("round limits need nonnegative integer caps")
    if round_limits["workers"] != 1 or round_limits["numeric_threads_per_worker"] != 1:
        raise Refusal("only one worker and one numerical thread are in scope")
    if round_limits["gpu_devices"] != 0:
        raise Refusal("GPU use must be disabled")
    if phase == "CHECKS" and round_limits["max_environment_interactions"] != 0:
        raise Refusal("CHECKS must have a zero interaction cap")
    if phase == "RUN" and round_limits["max_environment_interactions"] > 3584:
        raise Refusal("RUN cannot exceed the H037 3584-interaction ceiling")
    if phase_limits["cpu_seconds"] > round_limits["cpu_seconds"]:
        raise Refusal("phase CPU allocation exceeds the new-round CPU allocation")
    if phase_limits["wall_seconds"] > round_limits["wall_seconds"]:
        raise Refusal("phase wall allocation exceeds the new-round wall allocation")
    if phase_limits["new_artifact_cache_temp_bytes"] > round_limits["new_artifact_cache_temp_bytes"]:
        raise Refusal("phase storage allocation exceeds the new-round storage allocation")


def _validate_authorization(
    phase: str, spec_path: Path, auth_path: Path
) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    spec, spec_raw = _json_file(spec_path, "phase spec")
    auth, auth_raw = _json_file(auth_path, "phase authorization")
    if spec.get("format") != FORMAT or spec.get("experiment_id") != "H037":
        raise Refusal("phase spec identity is not H037 v1")
    if spec.get("phase") != phase or phase not in PHASES:
        raise Refusal("the CLI phase and exact phase spec differ")
    if auth.get("format") != AUTH_FORMAT or auth.get("experiment_id") != "H037":
        raise Refusal("coordinator authorization identity is invalid")
    if auth.get("phase") != phase:
        raise Refusal("coordinator authorization is for another phase")
    if auth.get("execution_authorized") is not True or auth.get("resource_authorized") is not True:
        raise Refusal("this phase has no exact execution and resource authorization")
    if auth.get("run_spec_sha256") != _sha256(spec_raw):
        raise Refusal("authorization does not bind these exact spec bytes")
    if not isinstance(auth.get("single_attempt_allocation_id"), str) or not auth["single_attempt_allocation_id"]:
        raise Refusal("new single-attempt allocation ID is required")
    if auth.get("automatic_retry") is not False or auth.get("reuse_historical_budget") is not False:
        raise Refusal("automatic retries and prior-round budget reuse are forbidden")
    if not isinstance(auth.get("coordinator_directive"), str) or not auth["coordinator_directive"]:
        raise Refusal("an exact coordinator directive is required")
    if spec.get("single_attempt_allocation_id") != auth["single_attempt_allocation_id"]:
        raise Refusal("spec and authorization allocation IDs differ")
    _validate_limits(spec, auth, phase)
    return spec, auth, spec_raw, auth_raw


def _validate_spec(spec: dict[str, Any], phase: str) -> dict[str, Any]:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        raise Refusal("paths mapping is missing")
    required_paths = {
        "project_root", "round_root", "phase_root", "orchestration_root",
        "source_root", "input_root", "cwd", "python", "config", "worker_state",
        "spec", "authorization", "stdout_log",
    }
    if phase == "CHECKS":
        required_paths |= {"collection_log", "junit_xml", "state"}
    else:
        required_paths |= {"output", "state"}
    if not required_paths.issubset(paths):
        raise Refusal("paths mapping is incomplete")
    if paths["project_root"] != str(PROJECT_ROOT):
        raise Refusal("project root differs from the exact allocated /data/robot/p001")
    round_root = _safe_project_path(paths["round_root"], "round root", must_exist=phase == "RUN")
    phase_root = _safe_project_path(paths["phase_root"], "phase root", must_exist=False)
    if phase_root.parent != round_root:
        raise Refusal("phase root must be a fresh direct child of the round root")
    orchestration_root = _safe_project_path(paths["orchestration_root"], "orchestration root", must_exist=True)
    source_root = _safe_project_path(paths["source_root"], "source root", must_exist=True)
    input_root = _safe_project_path(paths["input_root"], "input root", must_exist=True)
    cwd = _safe_project_path(paths["cwd"], "worker cwd", must_exist=True)
    python = _safe_project_path(paths["python"], "interpreter", must_exist=True)
    config = _safe_project_path(paths["config"], "config", must_exist=True)
    if paths["spec"] == paths["authorization"]:
        raise Refusal("spec and authorization must be separate exact files")
    if not config.is_file() or not python.is_file():
        raise Refusal("config and interpreter must be regular files")
    if not _beneath(cwd, source_root):
        raise Refusal("worker cwd must be inside the exact source root")
    manifests = spec.get("manifests")
    if not isinstance(manifests, dict):
        raise Refusal("exact file manifests are required")
    source_files = _verify_manifest(manifests.get("source_files"), "source_files", source_root)
    input_files = _verify_manifest(manifests.get("input_files"), "input_files", input_root)
    orchestration_files = _verify_manifest(
        manifests.get("orchestration_files"), "orchestration_files", orchestration_root
    )
    config_rows = [row for row in source_files if row["path"] == str(config)]
    if len(config_rows) != 1:
        raise Refusal("the exact H037 config must occur once in source_files")
    if not any(row["path"] == str(Path(__file__).resolve()) for row in orchestration_files):
        raise Refusal("the exact supervisor source must be bound in orchestration_files")
    config_obj, config_raw = _json_file(config, "H037 config")
    if config_obj.get("seed") != spec.get("seed") or not _is_int(spec.get("seed")):
        raise Refusal("phase seed must match the exact frozen H037 config")
    if spec.get("config_sha256") != _sha256(config_raw):
        raise Refusal("config SHA does not match the phase spec")
    runtime = spec.get("runtime_dependencies")
    if (not isinstance(runtime, dict) or set(runtime) != {"python_version", "packages"}
            or not isinstance(runtime.get("python_version"), str)
            or not isinstance(runtime.get("packages"), dict) or not runtime["packages"]
            or any(not isinstance(k, str) or not isinstance(v, str) or not v
                   for k, v in runtime["packages"].items())):
        raise Refusal("runtime_dependencies must freeze the interpreter and exact package versions")
    if spec.get("identity_algorithm") != IDENTITY_ALGORITHM:
        raise Refusal("phase spec must explicitly use the accepted v3 manifest identity algorithm")
    calculated_source_identity = _source_manifest_identity(source_files, source_root)
    if spec.get("source_identity_sha256") != calculated_source_identity:
        raise Refusal("source_identity_sha256 differs from the frozen code/config manifest subset")
    calculated_payload_identity = _manifest_identity(source_files, source_root)
    if spec.get("payload_identity_sha256") != calculated_payload_identity:
        raise Refusal("payload_identity_sha256 differs from the exact algorithm delivery manifest")
    command = spec.get("command")
    if not isinstance(command, dict) or set(command) != {"argv", "cwd"}:
        raise Refusal("command must bind an argv vector and cwd; shell strings are forbidden")
    argv = command["argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise Refusal("command argv must be a nonempty exact string list")
    if Path(argv[0]) != python or command["cwd"] != str(cwd):
        raise Refusal("interpreter or worker cwd differs from the bound command")
    if "-c" in argv or any(";" in part or "&&" in part or "|" in part for part in argv):
        raise Refusal("shell or inline-code command forms are forbidden")
    if phase == "CHECKS":
        collect_argv = spec.get("collection_argv")
        expected_nodes = spec.get("expected_test_nodeids")
        if (not isinstance(collect_argv, list) or not collect_argv
                or any(not isinstance(x, str) or not x for x in collect_argv)):
            raise Refusal("CHECKS requires one exact collection argv")
        if Path(collect_argv[0]) != python or "--collect-only" not in collect_argv:
            raise Refusal("CHECKS collection argv must use the bound interpreter and --collect-only")
        if "-c" in collect_argv or any(";" in part or "&&" in part or "|" in part for part in collect_argv):
            raise Refusal("CHECKS collection argv cannot use a shell or inline-code form")
        if "-v" not in argv or not any(x == f"--junitxml={paths['junit_xml']}" for x in argv):
            raise Refusal("CHECKS execution argv must be verbose and write the bound JUnit file")
        if (not isinstance(expected_nodes, list) or not expected_nodes
                or any(not isinstance(x, str) or not x for x in expected_nodes)
                or len(set(expected_nodes)) != len(expected_nodes)):
            raise Refusal("CHECKS expected_test_nodeids must be a nonempty unique accepted list")
    else:
        seed_positions = [i for i, part in enumerate(argv[:-1]) if part == "--seed"]
        if len(seed_positions) != 1 or argv[seed_positions[0] + 1] != str(spec.get("seed")):
            raise Refusal("RUN argv must contain exactly one --seed equal to the frozen seed")
        for bound_path in (str(config), str(paths["output"]), str(paths["worker_state"])):
            if bound_path not in argv and not any(part.endswith("=" + bound_path) for part in argv):
                raise Refusal(f"RUN argv must name its exact config/output/worker-state path: {bound_path}")
        script_paths = [Path(part) for part in argv[1:] if part.endswith(".py")]
        if not script_paths or any(
            not _beneath(path if path.is_absolute() else cwd / path, source_root)
            for path in script_paths
        ):
            raise Refusal("RUN argv must execute only a manifest-bound Python source script")
    environment = _validate_environment(spec.get("environment"), phase_root)
    if not _is_int(spec.get("minimum_free_bytes")) or not _is_int(spec.get("minimum_free_inodes")):
        raise Refusal("minimum_free_bytes and minimum_free_inodes must be exact nonnegative integers")
    for name in ("worker_state", "state", "stdout_log", "collection_log", "junit_xml", "output"):
        value = paths.get(name)
        if value is None:
            continue
        candidate = _safe_project_path(value, name, must_exist=False)
        if not _beneath(candidate, phase_root) or candidate == phase_root:
            raise Refusal(f"{name} must be a new child of this phase output root")
    output_names = [paths[name] for name in ("worker_state", "state", "stdout_log", "collection_log", "junit_xml", "output")
                    if name in paths]
    if len(set(output_names)) != len(output_names):
        raise Refusal("phase output, state, log, and JUnit paths must be distinct")
    if phase == "RUN":
        expected_interactions = spec.get("expected_environment_interactions")
        if (not _is_int(expected_interactions)
                or expected_interactions > spec["round_limits"]["max_environment_interactions"]):
            raise Refusal("RUN expected_environment_interactions must fit its admitted cap")
    if spec.get("phase_identity_sha256") != _phase_identity(
        phase, spec, source_files, input_files, orchestration_files
    ):
        raise Refusal("phase identity does not match its exact file and command binding")
    return {
        **spec,
        "paths": dict(paths),
        "source_files": source_files,
        "input_files": input_files,
        "orchestration_files": orchestration_files,
        "environment": environment,
        "round_root_path": round_root,
        "phase_root_path": phase_root,
        "orchestration_root_path": orchestration_root,
        "source_root_path": source_root,
        "input_root_path": input_root,
        "cwd_path": cwd,
        "python_path": python,
        "config_path": config,
        "command_argv": list(argv),
        "collection_argv": list(spec["collection_argv"]) if phase == "CHECKS" else None,
    }


def _phase_identity(
    phase: str, spec: dict[str, Any], source_files: list[dict[str, Any]],
    input_files: list[dict[str, Any]], orchestration_files: list[dict[str, Any]],
) -> str:
    identity = {
        "experiment_id": "H037",
        "phase": phase,
        "seed": spec.get("seed"),
        "config_sha256": spec.get("config_sha256"),
        "command": spec.get("command"),
        "paths": spec.get("paths"),
        "phase_limits": spec.get("phase_limits"),
        "round_limits": spec.get("round_limits"),
        "source_files": source_files,
        "input_files": input_files,
        "identity_algorithm": spec.get("identity_algorithm"),
        "source_identity_sha256": spec.get("source_identity_sha256"),
        "payload_identity_sha256": spec.get("payload_identity_sha256"),
        "orchestration_files": orchestration_files,
        "collection_argv": spec.get("collection_argv"),
        "expected_test_nodeids": spec.get("expected_test_nodeids"),
        "minimum_free_bytes": spec.get("minimum_free_bytes"),
        "minimum_free_inodes": spec.get("minimum_free_inodes"),
        "expected_environment_interactions": spec.get("expected_environment_interactions"),
        "runtime_dependencies": spec.get("runtime_dependencies"),
        "single_attempt_allocation_id": spec.get("single_attempt_allocation_id"),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256(raw)


def _manifest_identity(rows: list[dict[str, Any]], root: Path) -> str:
    seen: set[str] = set()
    ordered = []
    for row in rows:
        path = row["path"]
        if path in seen:
            raise Refusal("source and input manifests contain the same path twice")
        seen.add(path)
        ordered.append(row)
    normalized = [(row, Path(row["path"]).relative_to(root).as_posix()) for row in ordered]
    text = "".join(f"{row['sha256']}\t{row['bytes']}\t{relative}\n"
                   for row, relative in sorted(normalized, key=lambda item: item[1]))
    return _sha256(text.encode("utf-8"))


def _source_manifest_identity(rows: list[dict[str, Any]], source_root: Path) -> str:
    prefixes = ("src/", "scripts/", "tests/", "configs/")
    selected = []
    for row in rows:
        relative = Path(row["path"]).relative_to(source_root).as_posix()
        if relative.startswith(prefixes) or relative.startswith("requirements"):
            selected.append(row)
    if not selected:
        raise Refusal("the frozen source identity subset is empty")
    return _manifest_identity(selected, source_root)


def _verify_check_receipt(spec: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if spec.get("phase") != "RUN":
        raise Refusal("CHECKS must not consume a predecessor receipt")
    prior = spec.get("predecessor_receipt")
    if not isinstance(prior, dict) or set(prior) != {"path", "bytes", "sha256"}:
        raise Refusal("RUN requires the exact CHECKS receipt descriptor")
    path = _safe_project_path(prior["path"], "CHECKS receipt", must_exist=True)
    if not _beneath(path, spec["round_root_path"]):
        raise Refusal("CHECKS receipt must stay under this H037 round root")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise Refusal("CHECKS receipt is not a single-link regular file")
    raw = path.read_bytes()
    if len(raw) != prior["bytes"] or _sha256(raw) != prior["sha256"]:
        raise Refusal("CHECKS receipt bytes or SHA differs")
    state = json.loads(raw)
    if (not isinstance(state, dict) or state.get("format") != STATE_FORMAT
            or state.get("phase") != "CHECKS" or state.get("status") != "COMPLETE"):
        raise Refusal("predecessor receipt is not a COMPLETE CHECKS supervisor state")
    if state.get("state_path") != str(path) or state.get("round_limits") != spec.get("round_limits"):
        raise Refusal("CHECKS receipt state path or round limits differ from RUN")
    if state.get("phase_identity_sha256") != spec.get("predecessor_phase_identity_sha256"):
        raise Refusal("CHECKS receipt is for a different frozen phase identity")
    if state.get("single_attempt_allocation_id") != spec.get("single_attempt_allocation_id"):
        raise Refusal("CHECKS receipt belongs to a different allocation")
    if (state.get("source_identity_sha256") != spec.get("source_identity_sha256")
            or state.get("config_sha256") != spec.get("config_sha256")
            or state.get("seed") != spec.get("seed")):
        raise Refusal("CHECKS receipt is not bound to the same source, config, and seed")
    if state.get("source_identity_sha256") != spec.get("source_identity_sha256"):
        raise Refusal("CHECKS receipt source identity differs from RUN")
    costs = state.get("bounded_cost")
    if not isinstance(costs, dict) or not _is_finite(costs.get("cpu_seconds")) or not _is_finite(costs.get("wall_seconds")):
        raise Refusal("CHECKS receipt lacks bounded phase costs")
    phase_limits = spec["phase_limits"]
    round_limits = spec["round_limits"]
    if (costs["cpu_seconds"] + phase_limits["cpu_seconds"] > round_limits["cpu_seconds"]
            or costs["wall_seconds"] + phase_limits["wall_seconds"] > round_limits["wall_seconds"]):
        raise Refusal("CHECKS bounded costs plus RUN allocation exceed the new round")
    prior_bytes = state.get("round_apparent_bytes_after")
    if not _is_int(prior_bytes) or prior_bytes + phase_limits["new_artifact_cache_temp_bytes"] > round_limits["new_artifact_cache_temp_bytes"]:
        raise Refusal("CHECKS output plus RUN byte reservation exceeds the new round")
    return state, _sha256(raw)


def _apparent_bytes(root: Path) -> int:
    """Count bytes in one explicitly allocated project subtree; never follow links."""
    if not root.exists():
        return 0
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode):
        raise Refusal("round root is not a real directory")
    total = 0
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in list(names):
            info = (base / name).lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise Refusal("non-directory or symlink inside the allocated H037 root")
        for name in files:
            info = (base / name).lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise Refusal("non-regular or linked file inside the allocated H037 root")
            total += info.st_size
    return total


def _aggregate_project_bytes() -> int:
    """Metadata-only aggregate size; callers must already be authorized for p001."""
    total = 0
    for directory, names, files in os.walk(PROJECT_ROOT, topdown=True, followlinks=False):
        base = Path(directory)
        for name in list(names):
            info = (base / name).lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise Refusal("unexpected link or non-directory inside allocated project")
        for name in files:
            info = (base / name).lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise Refusal("unexpected link or non-regular file inside allocated project")
            total += info.st_size
    return total


def _check_capacity(spec: dict[str, Any], *, phase: str) -> dict[str, int]:
    limits = spec["round_limits"]
    new_cap = limits["new_artifact_cache_temp_bytes"]
    root = spec["round_root_path"]
    used = _apparent_bytes(root) if root.exists() else 0
    if used > new_cap:
        raise Refusal("current H037 outputs/cache/temp already exceed the approved round cap")
    project_used = _aggregate_project_bytes()
    remaining_round = max(0, new_cap - used)
    if project_used + remaining_round > PROJECT_BYTE_CAP:
        raise Refusal("the entire new H037 byte allowance does not fit the project cap")
    parent = root.parent if phase == "CHECKS" else spec["phase_root_path"].parent
    if not parent.exists() or not parent.is_dir():
        raise Refusal("authorized output parent directory is unavailable")
    usage = os.statvfs(parent)
    free_bytes = usage.f_bavail * usage.f_frsize
    free_inodes = usage.f_favail
    required_free_bytes = remaining_round + spec["minimum_free_bytes"]
    if free_bytes < required_free_bytes or free_inodes < spec["minimum_free_inodes"]:
        raise Refusal("authorized destination lacks admitted free bytes or inodes")
    return {
        "project_apparent_bytes": project_used,
        "round_apparent_bytes": used,
        "destination_free_bytes": free_bytes,
        "destination_free_inodes": free_inodes,
    }


def _check_phase_result(spec: dict[str, Any], state_path: Path, phase: str) -> None:
    """Validate a frozen worker-result contract after a future approved run."""
    state, raw = _json_file(state_path, "worker state")
    if state.get("format") != WORKER_STATE_FORMAT or state.get("phase") != phase:
        raise Refusal("worker state format or phase differs")
    binding = state.get("binding")
    expected_binding = {
        "phase_identity_sha256": spec["phase_identity_sha256"],
        "config_sha256": spec["config_sha256"],
        "seed": spec["seed"],
        "command_argv": spec["command_argv"],
    }
    if phase == "CHECKS":
        expected_binding["collection_argv"] = spec["collection_argv"]
    if binding != expected_binding:
        raise Refusal("worker state does not echo the exact phase binding")
    counters = state.get("counters")
    if not isinstance(counters, dict):
        raise Refusal("worker state lacks auditable counters")
    if (counters.get("gpu_devices") != 0 or counters.get("workers") != 1
            or counters.get("numeric_threads_per_worker") != 1):
        raise Refusal("worker state violates the zero-GPU, single-worker/thread envelope")
    interactions = counters.get("environment_interactions")
    cap = spec["round_limits"]["max_environment_interactions"]
    if not _is_int(interactions) or interactions > cap:
        raise Refusal("worker's audited interaction count is absent or over cap")
    if phase == "CHECKS" and interactions != 0:
        raise Refusal("CHECKS worker state reports environment interactions")
    if phase == "RUN" and interactions != spec.get("expected_environment_interactions"):
        raise Refusal("RUN interactions differ from the accepted fixed schedule")
    if state.get("status") != "COMPLETE":
        raise Refusal("worker did not report COMPLETE")
    if phase == "CHECKS":
        checks = state.get("checks")
        expected_nodes = spec.get("expected_test_nodeids")
        if (not isinstance(checks, dict)
                or checks.get("collected_test_nodeids") != expected_nodes
                or checks.get("executed_test_nodeids") != expected_nodes):
            raise Refusal("CHECKS collected or executed test nodes differ from the accepted list")
        if any(checks.get(key) != 0 for key in ("failed", "errors", "skipped")):
            raise Refusal("CHECKS contains failures, errors, or skipped tests")
    packages = state.get("persistent_packages", [] if phase == "CHECKS" else None)
    if not isinstance(packages, list) or (phase == "RUN" and not packages):
        raise Refusal("RUN worker state must list each persistent learner package")
    cap_bytes = spec["round_limits"]["max_persistent_package_bytes"]
    seen: set[str] = set()
    for index, row in enumerate(packages):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise Refusal(f"persistent_packages[{index}] is malformed")
        path = _safe_project_path(row["path"], f"persistent package {index}", must_exist=True)
        if not _beneath(path, spec["phase_root_path"]) or str(path) in seen:
            raise Refusal("persistent package path escapes the RUN output root or is duplicated")
        seen.add(str(path))
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != row["bytes"]:
            raise Refusal("persistent package is not a matching regular file")
        if row["bytes"] > cap_bytes or _sha256(path.read_bytes()) != row["sha256"]:
            raise Refusal("persistent package exceeds its approved size or SHA")


def _proc_stat(pid: int) -> dict[str, int | str]:
    """Read only one PID already owned by this supervisor; never enumerate /proc."""
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise Refusal("malformed owned process stat")
    fields = raw[close + 2:].split()
    if len(fields) < 20:
        raise Refusal("owned process stat lacks required fields")
    return {
        "state": fields[0], "ppid": int(fields[1]), "pgrp": int(fields[2]),
        "session": int(fields[3]), "utime": int(fields[11]), "stime": int(fields[12]),
        "cutime": int(fields[13]), "cstime": int(fields[14]), "start_ticks": int(fields[19]),
    }


def _proc_uptime() -> float:
    return float(Path("/proc/uptime").read_text(encoding="ascii").split()[0])


def _ru_cpu(value: resource.struct_rusage) -> float:
    return float(value.ru_utime + value.ru_stime)


def _ru_record(value: resource.struct_rusage) -> dict[str, float | int]:
    return {
        "user_seconds": float(value.ru_utime), "system_seconds": float(value.ru_stime),
        "max_rss_kib": int(value.ru_maxrss), "minor_faults": int(value.ru_minflt),
        "major_faults": int(value.ru_majflt), "voluntary_context_switches": int(value.ru_nvcsw),
        "involuntary_context_switches": int(value.ru_nivcsw),
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


class PhaseSupervisor:
    """Run only exact argv vectors and keep bounded, append-only local evidence."""

    def __init__(self, phase: str, spec: dict[str, Any], auth: dict[str, Any],
                 spec_path: Path, auth_path: Path, spec_raw: bytes, auth_raw: bytes,
                 predecessor_sha256: str | None):
        if sys.platform != "linux" or not sys.flags.isolated or not sys.dont_write_bytecode:
            raise Refusal("the phase supervisor requires isolated, bytecode-disabled Linux Python")
        if os.geteuid() != os.getuid() or os.geteuid() == 0:
            raise Refusal("the phase supervisor must run as the authorized unprivileged account")
        if pwd.getpwuid(os.geteuid()).pw_name != "robot":
            raise Refusal("the H037 supervisor may run only as the robot account")
        if Path(sys.executable).resolve(strict=True) != spec["python_path"].resolve(strict=True):
            raise Refusal("fresh supervisor interpreter differs from the exact project interpreter")
        runtime = spec["runtime_dependencies"]
        if sys.version.split()[0] != runtime["python_version"]:
            raise Refusal("Python runtime version differs from the exact accepted phase spec")
        self.actual_dependencies: dict[str, str] = {}
        for name, expected in runtime["packages"].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise Refusal(f"required distribution is absent: {name}") from exc
            if actual != expected:
                raise Refusal(f"runtime distribution version differs: {name}")
            self.actual_dependencies[name] = actual
        self.phase, self.spec, self.auth = phase, spec, auth
        self.spec_path, self.auth_path = spec_path, auth_path
        self.spec_sha256, self.auth_sha256 = _sha256(spec_raw), _sha256(auth_raw)
        self.predecessor_sha256 = predecessor_sha256
        self.paths = spec["paths"]
        self.root = spec["phase_root_path"]
        self.handles: dict[int, dict[str, Any]] = {}
        self.wait_rows: list[dict[str, Any]] = []
        self.stop_signal: int | None = None
        self.reason: str | None = None
        self.last_meter = 0.0
        self.peak_round_bytes = 0
        self.minimum_free_bytes_seen: int | None = None
        self.minimum_free_inodes_seen: int | None = None
        self.commands: list[dict[str, Any]] = []
        hz = int(os.sysconf("SC_CLK_TCK"))
        if hz <= 0:
            raise Refusal("kernel clock tick rate is invalid")
        self.ticks_per_second = hz
        own = _proc_stat(os.getpid())
        self.supervisor_pid = os.getpid()
        self.supervisor_start_ticks = int(own["start_ticks"])
        self.campaign_start_mono = time.monotonic() - (
            _proc_uptime() - self.supervisor_start_ticks / self.ticks_per_second)
        self.initial_children_rusage = _ru_record(resource.getrusage(resource.RUSAGE_CHILDREN))
        if self.initial_children_rusage["user_seconds"] != 0 or self.initial_children_rusage["system_seconds"] != 0:
            raise Refusal("fresh supervisor has nonzero pre-fork RUSAGE_CHILDREN; refusing contaminated accounting")

    @property
    def phase_limits(self) -> dict[str, Any]:
        return self.spec["phase_limits"]

    @property
    def stop_cpu(self) -> float:
        return float(self.phase_limits["cpu_seconds"] - self.phase_limits["footer_cpu_seconds"])

    @property
    def stop_wall(self) -> float:
        return float(self.phase_limits["wall_seconds"] - self.phase_limits["footer_wall_seconds"])

    def elapsed_wall(self) -> float:
        return max(0.0, time.monotonic() - self.campaign_start_mono)

    def cpu_snapshot(self) -> dict[str, Any]:
        live_rows = []
        for pid, handle in list(self.handles.items()):
            try:
                row = _proc_stat(pid)
            except FileNotFoundError:
                continue
            if int(row["start_ticks"]) != handle["start_ticks"] or int(row["pgrp"]) != handle["process_group"]:
                raise Refusal("owned process identity changed while supervised")
            ticks = sum(int(row[key]) for key in ("utime", "stime", "cutime", "cstime"))
            live_rows.append({"pid": pid, "start_ticks": handle["start_ticks"],
                              "process_group": handle["process_group"], "purpose": handle["purpose"],
                              "state": row["state"], "own_and_waited_descendant_cpu_ticks": ticks,
                              "cpu_seconds": ticks / self.ticks_per_second})
        self_cpu = _ru_cpu(resource.getrusage(resource.RUSAGE_SELF))
        reaped_cpu = sum(float(x["wait4_cpu_seconds_including_waited_descendants"]) for x in self.wait_rows)
        live_cpu = sum(float(x["cpu_seconds"]) for x in live_rows)
        return {"supervisor_rusage_self_cpu_seconds": self_cpu,
                "reaped_wait4_cpu_seconds": reaped_cpu,
                "live_owned_leader_cpu_seconds_from_proc_stat": live_cpu,
                "live_owned_processes": live_rows,
                "total_cpu_seconds": self_cpu + reaped_cpu + live_cpu,
                "unobserved_descendant_cpu_scope": "live descendants not waited by the recorded direct child are not individually enumerated"}

    def _capacity(self) -> dict[str, int]:
        reading = _check_capacity(self.spec, phase=self.phase)
        self.peak_round_bytes = max(self.peak_round_bytes, reading["round_apparent_bytes"])
        self.minimum_free_bytes_seen = (reading["destination_free_bytes"] if self.minimum_free_bytes_seen is None
                                        else min(self.minimum_free_bytes_seen, reading["destination_free_bytes"]))
        self.minimum_free_inodes_seen = (reading["destination_free_inodes"] if self.minimum_free_inodes_seen is None
                                         else min(self.minimum_free_inodes_seen, reading["destination_free_inodes"]))
        return reading

    def _write_json_new(self, path: Path, value: object, *, footer: bool = False) -> str:
        data = _json_bytes(value)
        if footer:
            if len(data) > self.phase_limits["footer_bytes"]:
                raise Refusal("final state is larger than its admitted footer byte reserve")
        else:
            reading = self._capacity()
            used = reading["round_apparent_bytes"]
            if used + len(data) + self.phase_limits["footer_bytes"] > self.spec["round_limits"]["new_artifact_cache_temp_bytes"]:
                raise Refusal("phase output/cache/temp reached its approved byte safety stop")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.peak_round_bytes = max(self.peak_round_bytes, _apparent_bytes(self.spec["round_root_path"]))
        return _sha256(data)

    def _event(self, kind: str, **fields: Any) -> None:
        path = self.root / "supervisor_events.jsonl"
        data = (json.dumps({"kind": kind, "monotonic": time.monotonic(),
                            "campaign_elapsed_wall_seconds": self.elapsed_wall(), **fields},
                           sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        reading = self._capacity()
        if reading["round_apparent_bytes"] + len(data) + self.phase_limits["footer_bytes"] > self.spec["round_limits"]["new_artifact_cache_temp_bytes"]:
            raise Refusal("event log reached the approved phase byte safety stop")
        mode = "ab" if path.exists() else "xb"
        with path.open(mode) as stream:
            stream.write(data)
            stream.flush()
        self.peak_round_bytes = max(self.peak_round_bytes, _apparent_bytes(self.spec["round_root_path"]))

    def _record_reap(self, pid: int, status: int, usage: resource.struct_rusage) -> dict[str, Any]:
        handle = self.handles.pop(pid)
        row = {"pid": pid, "process_group": handle["process_group"],
               "start_ticks": handle["start_ticks"], "purpose": handle["purpose"],
               "wait_status": status, "exit_code": os.waitstatus_to_exitcode(status),
               "rusage": _ru_record(usage),
               "wait4_cpu_seconds_including_waited_descendants": _ru_cpu(usage),
               "reaped_monotonic": time.monotonic()}
        self.wait_rows.append(row)
        self._event("child_reaped", wait4=row)
        return row

    def _reap_nonblocking(self, pid: int) -> dict[str, Any] | None:
        try:
            waited, status, usage = os.wait4(pid, os.WNOHANG)
        except InterruptedError:
            return None
        if waited == 0:
            return None
        return self._record_reap(pid, status, usage)

    def _group_exists(self, pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _signal_group(self, handle: dict[str, Any], signum: int) -> None:
        pid, pgid = int(handle["pid"]), int(handle["process_group"])
        if pid in self.handles:
            try:
                current = _proc_stat(pid)
            except FileNotFoundError:
                if not self._group_exists(pgid):
                    return
            else:
                if int(current["start_ticks"]) != handle["start_ticks"] or int(current["pgrp"]) != pgid:
                    raise Refusal("refusing to signal a changed or reused process identity")
        elif not self._group_exists(pgid):
            return
        os.killpg(pgid, signum)
        self._event("process_group_signal", pid=pid, process_group=pgid, signal=signum)

    def _stop_group(self, handle: dict[str, Any], reason: str) -> dict[str, Any] | None:
        self.reason = reason
        self._signal_group(handle, signal.SIGTERM)
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        result = None
        while time.monotonic() < deadline:
            result = self._reap_nonblocking(handle["pid"]) if handle["pid"] in self.handles else result
            if result is not None and not self._group_exists(handle["process_group"]):
                return result
            time.sleep(POLL_INTERVAL_SECONDS)
        if self._group_exists(handle["process_group"]):
            self._signal_group(handle, signal.SIGKILL)
        deadline = time.monotonic() + REAP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            result = self._reap_nonblocking(handle["pid"]) if handle["pid"] in self.handles else result
            if result is not None and not self._group_exists(handle["process_group"]):
                return result
            time.sleep(POLL_INTERVAL_SECONDS)
        if handle["pid"] in self.handles:
            raise Refusal("supervised child did not reap within the bounded stop window")
        if self._group_exists(handle["process_group"]):
            raise Refusal("recorded process group remains live after bounded SIGTERM/SIGKILL")
        return result

    def _launch(self, purpose: str, argv: list[str], cwd: Path, output_path: Path) -> dict[str, Any]:
        self._capacity()
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        gate_read, gate_write = os.pipe()
        log_fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            pid = os.fork()
        except BaseException:
            os.close(gate_read); os.close(gate_write); os.close(log_fd)
            raise
        if pid == 0:
            try:
                os.close(gate_write)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                os.setsid()
                if os.read(gate_read, 1) != b"1":
                    os._exit(125)
                os.close(gate_read)
                os.dup2(log_fd, 1); os.dup2(log_fd, 2); os.close(log_fd)
                os.chdir(cwd)
                soft_cpu = max(1, math.ceil(self.stop_cpu))
                resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu, soft_cpu + 1))
                child_env = dict(self.spec["environment"])
                if "HOME" in os.environ:
                    child_env["HOME"] = os.environ["HOME"]
                os.execve(argv[0], argv, child_env)
            except BaseException:
                os._exit(126)
        os.close(gate_read); os.close(log_fd)
        try:
            start_ticks = int(_proc_stat(pid)["start_ticks"])
            deadline = time.monotonic() + 1.0
            while os.getpgid(pid) != pid:
                if time.monotonic() >= deadline:
                    raise Refusal("worker did not establish its unique owned process group")
                time.sleep(0.005)
            handle = {"pid": pid, "process_group": pid, "start_ticks": start_ticks,
                      "purpose": purpose, "spawned_monotonic": time.monotonic(),
                      "argv": list(argv), "log_path": str(output_path)}
            self.handles[pid] = handle
            command_record = {"purpose": purpose, "argv": list(argv), "cwd": str(cwd),
                              "log_path": str(output_path), "pid": pid,
                              "spawned_monotonic": handle["spawned_monotonic"], "status": "RUNNING"}
            self.commands.append(command_record)
            self._event("child_started", handle=handle)
            os.write(gate_write, b"1")
            os.close(gate_write)
        except BaseException:
            os.close(gate_write)
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            try:
                waited, status, usage = os.wait4(pid, 0)
                if waited == pid and pid in self.handles:
                    try:
                        self._record_reap(pid, status, usage)
                    except BaseException:
                        self.handles.pop(pid, None)
            except OSError:
                pass
            self.handles.pop(pid, None)
            raise
        result = None
        while result is None:
            if self.stop_signal is not None:
                self._stop_group(handle, f"supervisor received signal {self.stop_signal}")
                raise Refusal(self.reason or "supervisor interrupted")
            wall = self.elapsed_wall()
            cpu = self.cpu_snapshot()
            if wall >= self.stop_wall:
                self._stop_group(handle, "phase wall safety stop reached")
                raise Refusal(self.reason or "wall safety stop")
            if cpu["total_cpu_seconds"] >= self.stop_cpu:
                self._stop_group(handle, "phase CPU safety stop reached")
                raise Refusal(self.reason or "CPU safety stop")
            if time.monotonic() - self.last_meter >= METER_INTERVAL_SECONDS:
                capacity = self._capacity()
                self._event("meter_sample", cpu_snapshot=cpu, wall_seconds=wall, capacity=capacity)
                self.last_meter = time.monotonic()
            try:
                result = self._reap_nonblocking(pid)
            except ChildProcessError as exc:
                raise Refusal("owned worker disappeared before wait4 accounting") from exc
            if result is None:
                time.sleep(POLL_INTERVAL_SECONDS)
        if self._group_exists(handle["process_group"]):
            self._stop_group(handle, "worker exited while an owned process group member remained")
            command_record["status"] = "PROCESS_GROUP_REMAINS_AFTER_LEADER_EXIT"
            raise Refusal("worker left a live process in its owned group; phase stopped")
        command_record["wait4"] = result
        command_record["status"] = "COMPLETE" if result["exit_code"] == 0 else "NONZERO_EXIT"
        return result

    def _ensure_directories(self) -> None:
        round_root = self.spec["round_root_path"]
        if self.phase == "CHECKS" and not round_root.exists():
            round_root.mkdir(mode=0o700)
        if round_root.is_symlink() or not round_root.is_dir():
            raise Refusal("round root is not a real directory")
        self.root.mkdir(mode=0o700)
        for key in CACHE_ENV:
            path = Path(self.spec["environment"][key])
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise Refusal(f"cache/temp directory is not a real directory: {path}")
        self._capacity()

    def _run_checks(self) -> dict[str, Any]:
        collect_log = Path(self.paths["collection_log"])
        collected_result = self._launch("pytest_collection", self.spec["collection_argv"],
                                        self.spec["cwd_path"], collect_log)
        if collected_result["exit_code"] != 0:
            raise Refusal("pytest collection command failed")
        collected_lines = collect_log.read_text(encoding="utf-8", errors="replace").splitlines()
        expected = self.spec["expected_test_nodeids"]
        test_files = {node.split("::", 1)[0] for node in expected}
        command_test_files = {part for part in self.spec["collection_argv"]
                              if part.endswith(".py") and "::" not in part}
        if command_test_files != test_files:
            raise Refusal("pytest collection argv test-file list differs from the accepted node manifest")
        collected = [line.strip() for line in collected_lines
                     if any(line.strip().startswith(file_name + "::") for file_name in test_files)]
        if collected != expected:
            raise Refusal("actual pytest collection node sequence differs from the accepted spec")
        run_log = Path(self.paths["stdout_log"])
        run_test_files = {part for part in self.spec["command_argv"]
                          if part.endswith(".py") and "::" not in part}
        if run_test_files != test_files:
            raise Refusal("pytest execution argv test-file list differs from the accepted node manifest")
        run_result = self._launch("pytest_execution", self.spec["command_argv"],
                                  self.spec["cwd_path"], run_log)
        junit_path = Path(self.paths["junit_xml"])
        if not junit_path.is_file() or junit_path.is_symlink():
            raise Refusal("pytest did not produce the exact regular JUnit output")
        junit_raw = junit_path.read_bytes()
        xml_root = ET.fromstring(junit_raw)
        cases = list(xml_root.iter("testcase"))
        failures = list(xml_root.iter("failure")); errors = list(xml_root.iter("error")); skipped = list(xml_root.iter("skipped"))
        execution_lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        statuses: dict[str, str] = {}
        for line in execution_lines:
            clean = line.strip()
            matched_file = next((name for name in test_files if clean.startswith(name + "::")), None)
            if matched_file is None:
                continue
            tokens = clean.split()
            status_index = next((i for i in range(len(tokens) - 1, 0, -1)
                                 if tokens[i] in {"PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS", "RERUN"}), None)
            if status_index is None:
                continue
            node = " ".join(tokens[:status_index])
            status_text = tokens[status_index]
            if node in statuses:
                raise Refusal("pytest verbose report repeats a test node")
            statuses[node] = status_text
        executed = [node for node in expected if node in statuses]
        if (run_result["exit_code"] != 0 or executed != expected or set(statuses) != set(expected)
                or len(cases) != len(expected)
                or failures or errors or skipped
                or any(status != "PASSED" for status in statuses.values())):
            raise Refusal("pytest execution did not complete every accepted node without failures or skips")
        return {"collected_test_nodeids": collected, "executed_test_nodeids": executed,
                "statuses": statuses, "failed": len(failures), "errors": len(errors),
                "skipped": len(skipped), "junit_path": str(junit_path),
                "junit_bytes": len(junit_raw), "junit_sha256": _sha256(junit_raw),
                "collection_log_path": str(collect_log), "collection_log_sha256": _sha256(collect_log.read_bytes()),
                "execution_log_path": str(run_log), "execution_log_sha256": _sha256(run_log.read_bytes()),
                "junit_testcase_count": len(cases), "exit_code": run_result["exit_code"]}

    def _make_checks_worker_state(self, checks: dict[str, Any]) -> dict[str, Any]:
        return {"format": WORKER_STATE_FORMAT, "phase": "CHECKS", "status": "COMPLETE",
                "binding": {"phase_identity_sha256": self.spec["phase_identity_sha256"],
                            "config_sha256": self.spec["config_sha256"], "seed": self.spec["seed"],
                            "command_argv": self.spec["command_argv"],
                            "collection_argv": self.spec["collection_argv"]},
                "counters": {"environment_interactions": 0, "gpu_devices": 0, "workers": 1,
                             "numeric_threads_per_worker": 1},
                "checks": checks, "persistent_packages": []}

    def _final_state(self, status: str, *, worker_state_sha256: str | None,
                     worker_state: dict[str, Any] | None, error: dict[str, Any] | None) -> dict[str, Any]:
        cpu = self.cpu_snapshot()
        wall = self.elapsed_wall()
        bounded_cpu = cpu["total_cpu_seconds"] + float(self.phase_limits["footer_cpu_seconds"])
        bounded_wall = wall + float(self.phase_limits["footer_wall_seconds"])
        try:
            capacity = self._capacity()
            capacity_error = None
        except BaseException as exc:
            capacity_error = f"{type(exc).__name__}: {exc}"
            capacity = {"project_apparent_bytes": None, "round_apparent_bytes": None,
                        "destination_free_bytes": None, "destination_free_inodes": None}
            if status == "COMPLETE":
                status = "INCOMPLETE"
                self.reason = self.reason or "final capacity audit failed"
        return {"format": STATE_FORMAT, "experiment_id": "H037", "phase": self.phase,
                "status": status, "spec_sha256": self.spec_sha256, "authorization_sha256": self.auth_sha256,
                "phase_identity_sha256": self.spec["phase_identity_sha256"],
                "source_identity_sha256": self.spec["source_identity_sha256"],
                "payload_identity_sha256": self.spec["payload_identity_sha256"],
                "identity_algorithm": self.spec["identity_algorithm"],
                "runtime_dependency_versions": self.actual_dependencies,
                "config_sha256": self.spec["config_sha256"], "seed": self.spec["seed"],
                "single_attempt_allocation_id": self.auth["single_attempt_allocation_id"],
                "state_path": self.paths["state"],
                "phase_limits": self.phase_limits,
                "round_limits": self.spec["round_limits"],
                "predecessor_receipt_sha256": self.predecessor_sha256,
                "operator_confirmation": {"confirmed_spec_sha256": self.spec_sha256,
                                           "cryptographic_identity": False},
                "coordinator_authorization": {"directive_sha256": _sha256(
                                                   self.auth["coordinator_directive"].encode("utf-8")),
                                               "cryptographic_signature_verified": False},
                "command_argv": self.spec["command_argv"], "collection_argv": self.spec.get("collection_argv"),
                "worker_state_sha256": worker_state_sha256,
                "worker_state_path": self.paths["worker_state"],
                "worker_state_status": worker_state.get("status") if isinstance(worker_state, dict) else None,
                "error": error, "stop_reason": self.reason,
                "supervisor_pid": self.supervisor_pid,
                "supervisor_start_ticks": self.supervisor_start_ticks,
                "campaign_start_source": "fresh supervisor /proc/self/stat start ticks; Python startup included",
                "campaign_wall_seconds_observed_before_footer": wall,
                "campaign_wall_seconds_bounded_upper": bounded_wall,
                "wall_bound_disclosure": "footer is a conservative reserve; time after supervisor exit is not directly observed",
                "cpu_snapshot_before_footer": cpu, "campaign_cpu_seconds_observed_before_footer": cpu["total_cpu_seconds"],
                "campaign_cpu_seconds_bounded_upper": bounded_cpu,
                "bounded_cost": {"cpu_seconds": bounded_cpu, "wall_seconds": bounded_wall},
                "cpu_bound_disclosure": "RUSAGE_SELF + direct-child wait4 + live direct-child /proc stat; un-waited live descendants are not individually measured",
                "child_wait4_records": self.wait_rows,
                "process_boundary_disclosure": "only the one recorded session/process group is signalled; a descendant that escapes with setsid/setpgid cannot be guaranteed stopped",
                "interaction_boundary_disclosure": "RUN interactions are checked against fixed schedule and result logs after exit; no kernel-level interaction counter is enforced",
                "gpu_thread_boundary_disclosure": "environment masks GPU visibility and fixes common numerical thread variables; worker receipt is post-hoc, not a device or thread quota",
                "storage_boundary_disclosure": "round files and destination free bytes/inodes are sampled; no filesystem quota prevents a burst or out-of-root write between samples",
                "storage_peak_sampled_bytes": self.peak_round_bytes,
                "capacity_finalization_error": capacity_error,
                "project_apparent_bytes_after": capacity["project_apparent_bytes"],
                "round_apparent_bytes_after": (capacity["round_apparent_bytes"] + self.phase_limits["footer_bytes"]
                                               if capacity["round_apparent_bytes"] is not None else None),
                "destination_free_bytes_after": capacity["destination_free_bytes"],
                "destination_free_inodes_after": capacity["destination_free_inodes"],
                "minimum_destination_free_bytes_sampled": self.minimum_free_bytes_seen,
                "minimum_destination_free_inodes_sampled": self.minimum_free_inodes_seen,
                "meter_interval_seconds": METER_INTERVAL_SECONDS,
                "poll_interval_seconds": POLL_INTERVAL_SECONDS,
                "footer_reserve": {"cpu_seconds": self.phase_limits["footer_cpu_seconds"],
                                   "wall_seconds": self.phase_limits["footer_wall_seconds"],
                                   "bytes": self.phase_limits["footer_bytes"]},
                "pre_fork_rusage_children": self.initial_children_rusage,
                "commands": self.commands}

    def _request_stop(self, signum: int, frame: Any) -> None:
        self.stop_signal = int(signum)

    def run(self) -> int:
        status, worker, worker_digest, error = "INCOMPLETE", None, None, None
        try:
            if _sha256(self.spec_path.read_bytes()) != self.spec_sha256:
                raise Refusal("phase spec bytes changed after admission")
            if _sha256(self.auth_path.read_bytes()) != self.auth_sha256:
                raise Refusal("authorization bytes changed after admission")
            _verify_manifest(self.spec["source_files"], "source_files", self.spec["source_root_path"])
            _verify_manifest(self.spec["input_files"], "input_files", self.spec["input_root_path"])
            _verify_manifest(self.spec["orchestration_files"], "orchestration_files",
                             self.spec["orchestration_root_path"])
            self._ensure_directories()
            self._event("phase_started", phase=self.phase, spec_sha256=self.spec_sha256,
                        authorization_sha256=self.auth_sha256, phase_limits=self.phase_limits,
                        round_limits=self.spec["round_limits"], operator_account="robot")
            for key, expected in {**THREAD_ENV, **EMPTY_GPU_ENV}.items():
                if self.spec["environment"].get(key) != expected:
                    raise Refusal(f"runtime environment lost required {key} setting")
            if self.phase == "CHECKS":
                checks = self._run_checks()
                worker = self._make_checks_worker_state(checks)
                worker_digest = self._write_json_new(Path(self.paths["worker_state"]), worker)
            else:
                outcome = self._launch("scientific_worker", self.spec["command_argv"],
                                       self.spec["cwd_path"], Path(self.paths["stdout_log"]))
                if outcome["exit_code"] != 0:
                    raise Refusal("RUN worker exited with a nonzero status")
                worker, raw = _json_file(Path(self.paths["worker_state"]), "RUN worker state")
                worker_digest = _sha256(raw)
            _check_phase_result(self.spec, Path(self.paths["worker_state"]), self.phase)
            if self.phase == "RUN":
                expected_interactions = self.spec.get("expected_environment_interactions")
                actual = worker["counters"]["environment_interactions"]
                if not _is_int(expected_interactions) or actual != expected_interactions:
                    raise Refusal("RUN interaction count differs from the accepted fixed schedule")
            status = "COMPLETE"
        except BaseException as exc:
            self.reason = self.reason or str(exc)
            error = {"type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc(limit=8)}
            for handle in list(self.handles.values()):
                try:
                    self._stop_group(handle, "supervisor exception cleanup")
                except BaseException as cleanup_exc:
                    error["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            status = "FAILED" if not self.handles else "INCOMPLETE"
        try:
            state = self._final_state(status, worker_state_sha256=worker_digest,
                                      worker_state=worker, error=error)
            digest = self._write_json_new(Path(self.paths["state"]), state, footer=True)
            print(json.dumps({"status": status, "state_path": self.paths["state"],
                              "state_sha256": digest}, sort_keys=True))
        except BaseException as exc:
            print(json.dumps({"status": "FINALIZATION_FAILED", "reason": str(exc)}, sort_keys=True),
                  file=sys.stderr)
            return 4
        return 0 if status == "COMPLETE" else 2


def _validate_predecessor(args: argparse.Namespace, spec: dict[str, Any]) -> str | None:
    if args.phase == "CHECKS":
        if args.receipt is not None or args.receipt_sha256 is not None or "predecessor_receipt" in spec:
            raise Refusal("CHECKS is the first phase and cannot consume a receipt")
        return None
    if not args.receipt or not _is_sha256(args.receipt_sha256):
        raise Refusal("RUN requires the exact successful CHECKS receipt path and SHA")
    if Path(args.receipt) != Path(spec.get("predecessor_receipt", {}).get("path", "")):
        raise Refusal("CLI predecessor receipt path differs from the exact phase spec")
    state, digest = _verify_check_receipt(spec)
    if digest != args.receipt_sha256:
        raise Refusal("CLI predecessor receipt SHA differs")
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--confirm-spec-sha256", required=True,
                        help="manual operator confirmation: exact admitted spec SHA256")
    parser.add_argument("--receipt")
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    try:
        spec_path = Path(args.spec)
        auth_path = Path(args.authorization)
        if not _beneath(spec_path, PROJECT_ROOT) or not _beneath(auth_path, PROJECT_ROOT):
            raise Refusal("spec and authorization must be exact files under the allocated project")
        _no_symlink_components(spec_path)
        _no_symlink_components(auth_path)
        spec, auth, spec_raw, auth_raw = _validate_authorization(args.phase, spec_path, auth_path)
        if not _is_sha256(args.confirm_spec_sha256) or args.confirm_spec_sha256 != _sha256(spec_raw):
            raise Refusal("manual operator confirmation does not match the exact spec bytes")
        if spec.get("paths", {}).get("spec") != str(spec_path):
            raise Refusal("phase spec path differs from its exact path binding")
        if spec.get("paths", {}).get("authorization") != str(auth_path):
            raise Refusal("authorization path differs from its exact path binding")
        validated = _validate_spec(spec, args.phase)
        validated["predecessor_receipt_sha256"] = _validate_predecessor(args, validated)
        _check_capacity(validated, phase=args.phase)
        supervisor = PhaseSupervisor(args.phase, validated, auth, spec_path, auth_path,
                                     spec_raw, auth_raw,
                                     validated["predecessor_receipt_sha256"])
        signal.signal(signal.SIGTERM, supervisor._request_stop)
        signal.signal(signal.SIGINT, supervisor._request_stop)
        return supervisor.run()
    except (Refusal, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

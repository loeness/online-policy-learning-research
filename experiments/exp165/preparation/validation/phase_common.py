"""Shared fail-closed bootstrap for the E165 remote stage workers.

This module validates the stage authorization and frozen input bytes before it
adds the scientific source tree to ``sys.path``.  It intentionally has no
scientific imports and performs no work at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any


PROJECT_ROOT = Path("/data/robot/p001")
E_ROOT = PROJECT_ROOT / "e165"
ORCHESTRATION_ROOT = E_ROOT / "o001"
SOURCE_ROOT = E_ROOT / "s001"
PYTHON_PATH = PROJECT_ROOT / "env/bin/python"
STAGE_ROOTS = {
    "CHECKS": E_ROOT / "c001",
    "DIAGNOSTIC": E_ROOT / "run001",
    "AUDIT": E_ROOT / "a001",
}
PYTEST_BASETEMP = STAGE_ROOTS["CHECKS"] / "t001"
CHECKS_SYNTHETIC_CALL_LIMITS = {
    "full_training_oracle_evaluations_upper_bound": 1,
    "complete_score_calls_upper_bound": 0,
}
CHECKS_MOCK_CALL_DECLARATION = {
    "training_oracle_calls_upper_bound": 102,
    "complete_score_calls_upper_bound": 2,
}
CHECKS_REAL_CALL_LIMITS = {
    "fit_calls_upper_bound": 0,
    "real_protected_fit_property": "e165.real_protected_fit_calls",
    "real_behavioral_fit_property": "e165.real_behavioral_fit_calls",
    "real_fit_total_property": "e165.real_fit_calls",
}
ALLOWED_PATH_ROOTS = frozenset({E_ROOT, ORCHESTRATION_ROOT, SOURCE_ROOT, *STAGE_ROOTS.values()})
STAGE_LIMITS = {
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
WORKERS = {
    "CHECKS": ORCHESTRATION_ROOT / "checks_entry.py",
    "DIAGNOSTIC": ORCHESTRATION_ROOT / "runner.py",
    "AUDIT": ORCHESTRATION_ROOT / "audit_entry.py",
}
STAGE_SPEC_PATHS = {
    stage: ORCHESTRATION_ROOT / f"stage_{stage.lower()}.json"
    for stage in STAGE_LIMITS
}
STAGE_AUTH_PATHS = {
    stage: ORCHESTRATION_ROOT / f"authorization_{stage.lower()}.json"
    for stage in STAGE_LIMITS
}
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
OPTIONAL_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "ROCR_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
}
class PhaseError(RuntimeError):
    """A frozen E165 stage precondition or output contract was violated."""


@dataclass(frozen=True)
class PhaseContext:
    stage: str
    spec_path: Path
    authorization_path: Path
    spec: dict[str, Any]
    authorization: dict[str, Any]
    run_root: Path
    source_cwd: Path
    source_identity_sha256: str


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_int(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _exact_json_equal(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int equality shortcut."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(_exact_json_equal(actual[key], value)
                        for key, value in expected.items()))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PhaseError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PhaseError(f"{label} contains invalid number {token}")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PhaseError(f"{label} must be a JSON object")
    return value


def safe_project_path(value: str, *, root: Path, must_exist: bool = True,
                      directory: bool | None = None) -> Path:
    """Return a canonical path under an approved root, rejecting symlink hops."""
    if root not in ALLOWED_PATH_ROOTS:
        raise PhaseError(f"path root is outside the E165 allowlist: {root}")
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise PhaseError("project paths must be absolute POSIX paths")
    path = Path(value)
    if str(path) != value or "." in path.parts or ".." in path.parts:
        raise PhaseError(f"path is not canonical: {value}")
    if path != root and root not in path.parents:
        raise PhaseError(f"path is outside its approved root: {value}")

    # Check every component from /; this rejects an internal link before resolve.
    cursor = Path("/")
    relative_parts = path.parts[1:]
    for index, part in enumerate(relative_parts):
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            if must_exist:
                raise PhaseError(f"path component is missing: {cursor}")
            continue
        except OSError as exc:
            raise PhaseError(f"cannot inspect path component {cursor}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise PhaseError(f"symlink path component is not allowed: {cursor}")
        if index < len(relative_parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise PhaseError(f"path parent is not a directory: {cursor}")

    try:
        root_info = root.lstat()
        resolved = path.resolve(strict=must_exist)
    except OSError as exc:
        raise PhaseError(f"cannot resolve path {value}: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise PhaseError(f"approved path root is not a real directory: {root}")
    if resolved != path:
        raise PhaseError(f"path alias is not allowed: {value}")
    if must_exist:
        info = path.lstat()
        if directory is True and not stat.S_ISDIR(info.st_mode):
            raise PhaseError(f"path must be a directory: {value}")
        if directory is False and not stat.S_ISREG(info.st_mode):
            raise PhaseError(f"path must be a regular file: {value}")
    return path


def check_pytest_basetemp() -> Path:
    """Validate the fixed pytest basetemp before pytest can clean or create it."""
    path = safe_project_path(str(PYTEST_BASETEMP), root=STAGE_ROOTS["CHECKS"],
                             must_exist=False, directory=False)
    try:
        path.lstat()
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise PhaseError(f"cannot inspect fixed pytest basetemp {path}: {exc}") from exc
    raise PhaseError(f"fixed pytest basetemp already exists; refusing pytest cleanup: {path}")


def _read_json_path(path_value: str, label: str) -> tuple[Path, dict[str, Any], bytes]:
    path = safe_project_path(path_value, root=ORCHESTRATION_ROOT,
                             must_exist=True, directory=False)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PhaseError(f"cannot read {label}: {exc}") from exc
    return path, _json_object(raw, label), raw


def read_identity(path: str, *, root: Path, expected_bytes: int,
                  expected_sha256: str) -> dict[str, Any]:
    """Read one declared JSON identity after checking its exact byte descriptor."""
    safe_path = safe_project_path(path, root=root, must_exist=True, directory=False)
    try:
        raw = safe_path.read_bytes()
    except OSError as exc:
        raise PhaseError(f"cannot read identity {path}: {exc}") from exc
    if not _is_int(expected_bytes) or len(raw) != expected_bytes:
        raise PhaseError(f"identity byte count differs from its descriptor: {path}")
    if not _is_sha256(expected_sha256) or _sha256(raw) != expected_sha256:
        raise PhaseError(f"identity SHA256 differs from its descriptor: {path}")
    return _json_object(raw, f"identity {path}")


def _validate_auth(stage: str, auth: dict[str, Any], spec_raw: bytes) -> None:
    if auth.get("experiment_id") != "E165" or auth.get("stage") != stage:
        raise PhaseError("authorization identity does not match E165 and this stage")
    if auth.get("resource_authorized") is not True or auth.get("execution_authorized") is not True:
        raise PhaseError("stage authorization is not explicitly enabled")
    if auth.get("run_spec_sha256") != _sha256(spec_raw):
        raise PhaseError("authorization is not bound to these exact spec bytes")
    if not _exact_json_equal(auth.get("limits"), STAGE_LIMITS[stage]):
        raise PhaseError("authorization limits differ from the frozen stage allocation")
    for field in ("coordinator_directive", "single_attempt_reservation_id"):
        if not isinstance(auth.get(field), str) or not auth[field].strip():
            raise PhaseError(f"authorization {field} is missing")


def _validate_environment_spec(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value.items()):
        raise PhaseError("spec environment must be a string-to-string object")
    if "HOME" in value:
        raise PhaseError("spec may not change HOME")
    allowed = set(EXPECTED_ENVIRONMENT) | set(OPTIONAL_ENVIRONMENT)
    if set(value) - allowed:
        raise PhaseError("spec environment contains an unapproved variable")
    for key, expected in EXPECTED_ENVIRONMENT.items():
        if value.get(key) != expected:
            raise PhaseError(f"spec environment differs from the frozen value for {key}")
    for key, expected in OPTIONAL_ENVIRONMENT.items():
        if key in value and value[key] != expected:
            raise PhaseError(f"spec environment differs from the frozen value for {key}")
    for key, expected in value.items():
        if os.environ.get(key) != expected:
            raise PhaseError(f"process environment differs from the frozen value for {key}")
    for key in ("TMP", "TEMP", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
                "XDG_DATA_HOME", "XDG_STATE_HOME", "MPLCONFIGDIR", "TORCH_HOME",
                "TORCH_EXTENSIONS_DIR"):
        path = safe_project_path(value[key], root=E_ROOT, must_exist=True, directory=True)
        if E_ROOT not in path.parents:
            raise PhaseError(f"cache or temporary path is not inside e165: {key}")
    return dict(value)


def _validate_descriptor_rows(value: object, label: str, *, root: Path,
                              allow_empty: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise PhaseError(f"{label} must be a descriptor list")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise PhaseError(f"{label}[{index}] must be an object")
        path_value = row.get("path")
        if not isinstance(path_value, str) or path_value in seen:
            raise PhaseError(f"{label}[{index}] path is missing or duplicated")
        seen.add(path_value)
        path = safe_project_path(path_value, root=root, must_exist=True, directory=False)
        if not _is_int(row.get("bytes")) or not _is_sha256(row.get("sha256")):
            raise PhaseError(f"{label}[{index}] must declare exact bytes and lowercase sha256")
        try:
            info = path.lstat()
            raw = path.read_bytes()
        except OSError as exc:
            raise PhaseError(f"cannot verify {label}[{index}]: {exc}") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PhaseError(f"{label}[{index}] is not a single-link regular file")
        if len(raw) != row["bytes"] or _sha256(raw) != row["sha256"]:
            raise PhaseError(f"{label}[{index}] bytes or sha256 differ from the frozen descriptor")
        rows.append(dict(row))
    return rows


def _validate_stage_spec(stage: str, spec: dict[str, Any], spec_path: Path,
                         authorization_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed = {
        "format": "e165_execution_spec_v1",
        "experiment_id": "E165",
        "stage": stage,
        "preparation_state": "FROZEN",
        "resource_authorized": True,
        "project_root": str(PROJECT_ROOT),
        "e_root": str(E_ROOT),
        "orchestration_root": str(ORCHESTRATION_ROOT),
        "source_cwd": str(SOURCE_ROOT),
        "run_root": str(STAGE_ROOTS[stage]),
        "python": str(PYTHON_PATH),
        "limits": STAGE_LIMITS[stage],
        "round_limits": ROUND_LIMITS,
    }
    for key, expected in fixed.items():
        if not _exact_json_equal(spec.get(key), expected):
            raise PhaseError(f"spec {key} differs from the fixed E165 contract")
    if not _is_int(spec.get("project_bytes_before")) or spec["project_bytes_before"] > ROUND_LIMITS["aggregate_remote_project_bytes"]:
        raise PhaseError("project_bytes_before must be a non-E165 byte baseline within the project cap")
    source_identity = spec.get("source_identity_sha256")
    if not _is_sha256(source_identity):
        raise PhaseError("source_identity_sha256 must be a lowercase SHA256")

    expected_worker_argv = ["-B", str(WORKERS[stage]), "--spec", str(spec_path),
                            "--authorization", str(authorization_path)]
    if spec.get("worker_argv") != expected_worker_argv:
        raise PhaseError("worker_argv differs from the exact supervisor stage command")
    actual_argv = [sys.argv[0], *sys.argv[1:]]
    if actual_argv != expected_worker_argv[1:]:
        raise PhaseError("running worker arguments differ from the authorized worker_argv")

    source_files = _validate_descriptor_rows(spec.get("source_files"), "source_files",
                                             root=SOURCE_ROOT)
    orchestration_files = _validate_descriptor_rows(
        spec.get("orchestration_files"), "orchestration_files", root=ORCHESTRATION_ROOT)
    declared_control = {row["path"] for row in orchestration_files}
    if str(WORKERS[stage]) not in declared_control or str(ORCHESTRATION_ROOT / "phase_common.py") not in declared_control:
        raise PhaseError("stage worker and phase_common.py must both be byte-bound orchestration files")

    scientific = spec.get("scientific_manifest")
    if not isinstance(scientific, dict) or not isinstance(scientific.get("path"), str):
        raise PhaseError("scientific_manifest descriptor is missing")
    if not _is_int(scientific.get("bytes")) or not _is_sha256(scientific.get("sha256")):
        raise PhaseError("scientific_manifest must bind exact bytes and sha256")
    all_rows = source_files + orchestration_files
    if not any(row["path"] == scientific["path"] and row["bytes"] == scientific["bytes"]
               and row["sha256"] == scientific["sha256"] for row in all_rows):
        raise PhaseError("scientific_manifest must match a verified source/control descriptor")

    packages = spec.get("expected_packages")
    if not isinstance(packages, dict) or len(packages) != 33 or any(
            not isinstance(name, str) or not name or not isinstance(version, str) or not version
            for name, version in packages.items()):
        raise PhaseError("expected_packages must contain exactly 33 pinned versions")
    if spec.get("expected_python_version") != "3.12.15":
        raise PhaseError("expected_python_version must be the frozen CPython 3.12.15")
    _validate_environment_spec(spec.get("environment"))

    if stage == "CHECKS":
        pytest_config = spec.get("pytest_config")
        expected_config = str(ORCHESTRATION_ROOT / "pytest.ini")
        if pytest_config != expected_config:
            raise PhaseError("CHECKS pytest_config must be the exact frozen /o001/pytest.ini path")
        safe_project_path(pytest_config, root=ORCHESTRATION_ROOT,
                          must_exist=True, directory=False)
        if pytest_config not in declared_control:
            raise PhaseError("pytest_config must be present in verified orchestration_files")
        targets = spec.get("pytest_target")
        nodeids = spec.get("expected_test_nodeids")
        if not isinstance(targets, list) or not targets or any(not isinstance(item, str) for item in targets):
            raise PhaseError("CHECKS pytest_target must be a nonempty frozen path list")
        if len(set(targets)) != len(targets):
            raise PhaseError("CHECKS pytest_target contains duplicate paths")
        source_paths = {row["path"] for row in source_files}
        for target in targets:
            safe_project_path(target, root=SOURCE_ROOT, must_exist=True, directory=False)
            if target not in source_paths:
                raise PhaseError("each pytest_target must be present in the verified source descriptors")
        if not isinstance(nodeids, list) or not nodeids or any(
                not isinstance(nodeid, str) or not nodeid for nodeid in nodeids):
            raise PhaseError("CHECKS expected_test_nodeids must be a nonempty frozen list")
        if len(set(nodeids)) != len(nodeids):
            raise PhaseError("CHECKS expected_test_nodeids contains duplicates")
        envelope = spec.get("synthetic_call_envelope")
        if not isinstance(envelope, dict) or not envelope:
            raise PhaseError("CHECKS synthetic_call_envelope must be explicitly frozen")
        if envelope.get("actual_count_source") != "pytest_user_properties":
            raise PhaseError("CHECKS synthetic_call_envelope must use pytest teardown user properties")
        for key, expected_bound in CHECKS_SYNTHETIC_CALL_LIMITS.items():
            if not _exact_json_equal(envelope.get(key), expected_bound):
                raise PhaseError(f"CHECKS synthetic_call_envelope.{key} differs from its frozen limit")

        mock_call_declaration = spec.get("mock_call_declaration")
        if not isinstance(mock_call_declaration, dict):
            raise PhaseError("CHECKS mock_call_declaration must be an object")
        for key, expected_value in CHECKS_MOCK_CALL_DECLARATION.items():
            if not _exact_json_equal(mock_call_declaration.get(key), expected_value):
                raise PhaseError(f"CHECKS mock_call_declaration.{key} differs from its frozen limit")
        real_call_limits = spec.get("real_call_limits")
        if not isinstance(real_call_limits, dict):
            raise PhaseError("CHECKS real_call_limits must be an object")
        for key, expected_value in CHECKS_REAL_CALL_LIMITS.items():
            if not _exact_json_equal(real_call_limits.get(key), expected_value):
                raise PhaseError(f"CHECKS real_call_limits.{key} differs from its frozen limit")

    prior = spec.get("prior_stage_states")
    required_prior = {"CHECKS": set(), "DIAGNOSTIC": {"CHECKS"},
                      "AUDIT": {"CHECKS", "DIAGNOSTIC"}}[stage]
    if not isinstance(prior, dict) or set(prior) != required_prior:
        raise PhaseError("prior_stage_states must exactly match the stage predecessor list")
    for prior_stage, row in prior.items():
        expected_path = str(STAGE_ROOTS[prior_stage] / "state.json")
        if not isinstance(row, dict) or row.get("path") != expected_path:
            raise PhaseError(f"prior_stage_states.{prior_stage} has the wrong path")
        if not _is_int(row.get("bytes")) or not _is_sha256(row.get("sha256")):
            raise PhaseError(f"prior_stage_states.{prior_stage} needs exact bytes and sha256")
        safe_project_path(expected_path, root=E_ROOT, must_exist=True, directory=False)
    return source_files, orchestration_files


def load_context(stage: str, spec: str, authorization: str) -> PhaseContext:
    """Authorize a worker, verify its declared bytes, then expose frozen src."""
    if stage not in STAGE_LIMITS:
        raise PhaseError("stage must be CHECKS, DIAGNOSTIC, or AUDIT")
    spec_path, spec_value, spec_raw = _read_json_path(spec, "stage spec")
    auth_path, auth_value, _ = _read_json_path(authorization, "stage authorization")
    if spec_path != STAGE_SPEC_PATHS[stage] or auth_path != STAGE_AUTH_PATHS[stage]:
        raise PhaseError("spec and authorization must be the exact o001 files for this stage")

    # Nothing below this gate may inspect runtime environment or import science.
    _validate_auth(stage, auth_value, spec_raw)
    if os.environ.get("E165_SUPERVISED") != "1" or os.environ.get("E165_STAGE") != stage:
        raise PhaseError("worker must be launched by the matching E165 supervisor stage")

    source_files, orchestration_files = _validate_stage_spec(
        stage, spec_value, spec_path, auth_path)
    safe_project_path(str(SOURCE_ROOT), root=E_ROOT, must_exist=True, directory=True)
    safe_project_path(str(ORCHESTRATION_ROOT), root=E_ROOT, must_exist=True, directory=True)
    run_root = safe_project_path(str(STAGE_ROOTS[stage]), root=E_ROOT,
                                 must_exist=True, directory=True)

    # These current control files must be among the verified bytes.
    declared_control = {row["path"] for row in orchestration_files}
    common_path = str(ORCHESTRATION_ROOT / "phase_common.py")
    if common_path not in declared_control:
        raise PhaseError("phase_common.py is absent from orchestration_files")
    active_common = safe_project_path(str(Path(__file__)), root=ORCHESTRATION_ROOT,
                                      must_exist=True, directory=False)
    if str(active_common) != common_path:
        raise PhaseError("the running phase_common.py is outside its exact orchestration path")
    common_raw = active_common.read_bytes()
    common_row = next(row for row in orchestration_files if row["path"] == common_path)
    if len(common_raw) != common_row["bytes"] or _sha256(common_raw) != common_row["sha256"]:
        raise PhaseError("the running phase_common.py differs from its frozen descriptor")

    frozen_src = safe_project_path(str(SOURCE_ROOT / "src"), root=SOURCE_ROOT,
                                   must_exist=True, directory=True)
    if str(frozen_src) not in sys.path:
        sys.path.insert(0, str(frozen_src))
    return PhaseContext(
        stage=stage,
        spec_path=spec_path,
        authorization_path=auth_path,
        spec=spec_value,
        authorization=auth_value,
        run_root=run_root,
        source_cwd=SOURCE_ROOT,
        source_identity_sha256=spec_value["source_identity_sha256"],
    )


def environment_record(context: PhaseContext) -> dict[str, Any]:
    """Record and validate the authorized Python, package, Linux, and CPU build."""
    if sys.platform != "linux" or platform.system() != "Linux" or sys.byteorder != "little":
        raise PhaseError("E165 requires Linux with little-endian byte order")
    expected_executable = str(PYTHON_PATH)
    try:
        actual_real = Path(sys.executable).resolve(strict=True)
        expected_real = PYTHON_PATH.resolve(strict=True)
    except OSError as exc:
        raise PhaseError(f"cannot resolve the active Python executable: {exc}") from exc
    if sys.executable != expected_executable or actual_real != expected_real:
        raise PhaseError("sys.executable differs from the allocated project interpreter")
    actual_python = platform.python_version()
    if actual_python != context.spec["expected_python_version"]:
        raise PhaseError("active Python version differs from the frozen spec")

    package_versions: dict[str, str] = {}
    for package, expected in sorted(context.spec["expected_packages"].items()):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PhaseError(f"pinned dependency is not installed: {package}") from exc
        if actual != expected:
            raise PhaseError(f"installed {package} version {actual!r} differs from {expected!r}")
        package_versions[package] = actual

    torch = importlib.import_module("torch")
    torch_version = str(torch.__version__)
    cuda_version = getattr(torch.version, "cuda", None)
    if torch_version != context.spec["expected_packages"].get("torch") or "+cpu" not in torch_version:
        raise PhaseError("torch is not the exact frozen CPU-only build")
    if cuda_version is not None:
        raise PhaseError("torch reports CUDA support; this E165 stage is CPU-only")
    initial_torch_threads = int(torch.get_num_threads())
    initial_interop_threads = int(torch.get_num_interop_threads())
    try:
        torch.set_num_threads(1)
        if initial_interop_threads != 1:
            torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise PhaseError(f"cannot set torch thread counts before the stage workload: {exc}") from exc
    torch_threads = int(torch.get_num_threads())
    interop_threads = int(torch.get_num_interop_threads())
    if torch_threads != 1 or interop_threads != 1:
        raise PhaseError("torch thread counts differ from the one-thread allocation")

    return {
        "format": "e165_environment_v1",
        "stage": context.stage,
        "python_executable": sys.executable,
        "python_version": actual_python,
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "byteorder": sys.byteorder,
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": package_versions,
        "torch": {
            "version": torch_version, "cuda_version": cuda_version,
            "cpu_only_build": True,
            "initial_threads": initial_torch_threads,
            "initial_interop_threads": initial_interop_threads,
            "threads": torch_threads, "interop_threads": interop_threads,
        },
        "environment": {
            key: os.environ.get(key)
            for key in sorted(set(context.spec["environment"]) | {
                "E165_SUPERVISED", "E165_STAGE", "VECLIB_MAXIMUM_THREADS",
                "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            }) if key in os.environ
        },
        "gpu_devices_used": 0,
    }


def _output_path(context: PhaseContext, name: str) -> Path:
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None or name in {".", ".."}:
        raise PhaseError("stage output name must be a single safe filename")
    root = safe_project_path(str(context.run_root), root=E_ROOT,
                             must_exist=True, directory=True)
    return safe_project_path(str(root / name), root=root,
                             must_exist=False, directory=False)


def write_bytes(context: PhaseContext, name: str, data: bytes) -> Path:
    """Create one file under this stage root without following or replacing it."""
    if not isinstance(data, bytes):
        raise PhaseError("write_bytes data must be bytes")
    path = _output_path(context, name)
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise PhaseError(f"refusing to overwrite stage output {name}") from exc
    except OSError as exc:
        raise PhaseError(f"cannot create stage output {name}: {exc}") from exc
    return path


def write_json(context: PhaseContext, name: str, value: dict[str, Any]) -> Path:
    """Serialize JSON with stable keys and exclusive stage-local creation."""
    try:
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseError(f"cannot serialize stage JSON {name}: {exc}") from exc
    return write_bytes(context, name, raw)


def base_state(context: PhaseContext) -> dict[str, Any]:
    """Return the common worker-state identity with every prohibited counter zero."""
    state: dict[str, Any] = {
        "format": "e165_worker_state_v1",
        "stage": context.stage,
        "status": "INCOMPLETE",
        "source_identity_sha256": context.source_identity_sha256,
        "environment_interactions": 0,
        "gpu_devices": 0,
        "real_candidate_fit_calls": 0,
        "candidate_recovery_calls": 0,
    }
    if context.stage == "CHECKS":
        expected = list(context.spec["expected_test_nodeids"])
        state.update({
            "expected_test_nodeids": expected,
            "collected_test_nodeids": [],
            "executed_test_nodeids": [],
            "passed": 0,
            "failed": 0,
            "error": 0,
            "errors": 0,
            "skipped": 0,
            "xfail": 0,
            "collection_errors": 0,
        })
    return state


def finish_state(context: PhaseContext, state: dict[str, Any], status: str) -> dict[str, Any]:
    """Validate and exclusively write the final worker state for this stage."""
    if status not in {"COMPLETE", "INVALID", "INCOMPLETE"}:
        raise PhaseError("worker status must be COMPLETE, INVALID, or INCOMPLETE")
    required = base_state(context)
    for key in ("format", "stage", "source_identity_sha256", "environment_interactions",
                "gpu_devices", "real_candidate_fit_calls", "candidate_recovery_calls"):
        if state.get(key) != required[key]:
            raise PhaseError(f"worker state core field {key} differs from the authorized base")
    result = dict(state)
    result["status"] = status
    if status == "COMPLETE" and context.stage == "CHECKS":
        expected = result.get("expected_test_nodeids")
        collected = result.get("collected_test_nodeids")
        executed = result.get("executed_test_nodeids")
        if expected != required["expected_test_nodeids"] or collected != expected or executed != expected:
            raise PhaseError("CHECKS COMPLETE requires exact expected, collected, and executed nodeids")
        for field in ("failed", "error", "errors", "skipped", "xfail", "collection_errors"):
            if type(result.get(field)) is not int or result[field] != 0:
                raise PhaseError(f"CHECKS COMPLETE requires {field}=0")
        if type(result.get("passed")) is not int or result["passed"] != len(expected):
            raise PhaseError("CHECKS COMPLETE requires every expected test to pass")
    write_json(context, "state_worker.json", result)
    return result


"""API for stage workers:

* ``load_context(stage, spec, authorization) -> PhaseContext`` validates the
  exact stage authorization, inputs, control files, Python command and frozen
  source descriptors before adding ``/e165/s001/src`` to ``sys.path``.
* ``safe_project_path(value, root=..., must_exist=True, directory=None)``
  enforces canonical paths and rejects symlinks at every component.
* ``check_pytest_basetemp()`` validates the fixed CHECKS temporary path, rejects
  symlink components, and refuses to start pytest if that path already exists.
* ``read_identity(path, root=..., expected_bytes=..., expected_sha256=...)``
  verifies exact JSON bytes before returning a project identity object.
* ``environment_record(context)`` verifies and records the actual interpreter,
  all 33 package pins, Linux identity, and CPU-only one-thread torch build.
* ``base_state(context)`` initializes the common counters and CHECKS fields;
  ``finish_state(context, state, status)`` validates and exclusively writes
  ``state_worker.json``.
* ``write_json(context, name, value)`` and ``write_bytes(context, name, data)``
  create single-name files only inside this stage's already-created output root.
"""

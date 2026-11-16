#!/usr/bin/env python3
"""Linux-only, single-worker E163 campaign supervisor.

This file deliberately uses only the standard library. It does not start until
the coordinator authorization is bound to the exact run-spec bytes.
"""
from __future__ import annotations

import argparse
import errno
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
E_ROOT = PROJECT / "e163"
ORCHESTRATION_ROOT = E_ROOT / "o001"
SOURCE_CWD = E_ROOT / "s001"
RUN_ROOT = E_ROOT / "run001"
PYTHON = PROJECT / "env/bin/python"
RUNNER = ORCHESTRATION_ROOT / "runner.py"
SPEC_PATH = ORCHESTRATION_ROOT / "run_spec.json"
AUTHORIZATION_PATH = ORCHESTRATION_ROOT / "authorization.json"

EXPECTED_LIMITS = {
    "cpu_seconds": 120,
    "campaign_wall_seconds": 180,
    "new_artifact_cache_temp_bytes": 268435456,
    "workers": 1,
    "numeric_threads_per_worker": 1,
    "gpu_devices": 0,
    "environment_interactions": 0,
    "candidate_fits": 32,
    "aggregate_remote_project_bytes": 100000000000,
}
CPU_STOP_SECONDS = 115.0
WALL_STOP_SECONDS = 170.0
FOOTER_CPU_RESERVE_SECONDS = 5.0
FOOTER_WALL_RESERVE_SECONDS = 10.0
FOOTER_BYTE_RESERVE = 65536
METER_INTERVAL_SECONDS = 0.25
POLL_INTERVAL_SECONDS = 0.025
STOP_GRACE_SECONDS = 0.75
CPU_PROBE_WALL_TIMEOUT_SECONDS = 5.0
REAP_TIMEOUT_SECONDS = 1.0

EXPECTED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
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
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE", "WANDB_MODE",
}


class SupervisorError(RuntimeError):
    pass


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _canonical(path_value: str, label: str, *, must_exist: bool = True,
               root: Path = PROJECT, allow_internal_symlink: bool = False) -> Path:
    if not isinstance(path_value, str) or not path_value.startswith("/"):
        raise SupervisorError(f"{label} must be an absolute POSIX path")
    path = Path(path_value)
    if str(path) != path_value:
        raise SupervisorError(f"{label} is not in canonical path form")
    try:
        resolved = path.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise SupervisorError(f"{label} cannot be resolved: {exc}") from exc
    resolved_root = root.resolve(strict=True)
    resolved_project = PROJECT.resolve(strict=True)
    if resolved_project != PROJECT or resolved_root != root:
        raise SupervisorError("allocated project roots must be canonical real paths")
    if not _is_below(resolved_root, resolved_project):
        raise SupervisorError(f"{label} root resolves outside the allocated project")
    if not _is_below(resolved, resolved_root):
        raise SupervisorError(f"{label} resolves outside the allocated project")
    if not allow_internal_symlink and resolved != path:
        raise SupervisorError(f"{label} contains a symlink or path alias")
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
    for key in ("NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if key in value and value[key] != "1":
            raise SupervisorError(f"{key} must be 1")
    for key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if key in value and value[key] != "":
            raise SupervisorError(f"{key} must be empty because GPU use is disabled")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if key in value and value[key] != "1":
            raise SupervisorError(f"{key} must be 1 when supplied")
    if "WANDB_MODE" in value and value["WANDB_MODE"] != "offline":
        raise SupervisorError("WANDB_MODE must be offline when supplied")
    return dict(value)


def _validate_spec_and_authorization(spec_path_value: str, authorization_path_value: str
                                     ) -> tuple[dict, dict, Path, Path]:
    spec_path = _canonical(spec_path_value, "spec path", root=E_ROOT)
    authorization_path = _canonical(authorization_path_value, "authorization path", root=E_ROOT)
    if spec_path != SPEC_PATH or authorization_path != AUTHORIZATION_PATH:
        raise SupervisorError("spec and authorization must be the frozen o001 files")
    spec, spec_raw = _read_json(spec_path, "run spec")
    authorization, _ = _read_json(authorization_path, "authorization")

    if spec.get("experiment_id") != "E163":
        raise SupervisorError("run spec is not for E163")
    if spec.get("project_root") != str(PROJECT) or spec.get("e_root") != str(E_ROOT):
        raise SupervisorError("run spec project roots differ from the allocated paths")
    if spec.get("orchestration_root") != str(ORCHESTRATION_ROOT):
        raise SupervisorError("run spec orchestration root differs from o001")
    if spec.get("source_cwd") != str(SOURCE_CWD) or spec.get("run_root") != str(RUN_ROOT):
        raise SupervisorError("run spec source or run directory differs from the frozen paths")
    if spec.get("python") != str(PYTHON):
        raise SupervisorError("run spec must use the allocated project Python interpreter")
    if Path(sys.executable).resolve(strict=True) != PYTHON.resolve(strict=True):
        raise SupervisorError("supervisor is not running under the run spec compute interpreter")
    if sys.platform != "linux" or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise SupervisorError("supervisor must run on Linux with isolated, bytecode-disabled Python")

    expected_argv = ["-B", str(RUNNER), "--spec", str(spec_path),
                     "--authorization", str(authorization_path)]
    if spec.get("worker_argv") != expected_argv:
        raise SupervisorError("worker_argv does not match the frozen runner command")
    if spec.get("limits") != EXPECTED_LIMITS:
        raise SupervisorError("run spec limits differ from the fixed E163 authorization")
    if not isinstance(spec.get("project_bytes_before"), int) or spec["project_bytes_before"] < 0:
        raise SupervisorError("project_bytes_before must be a nonnegative integer baseline")
    if isinstance(spec["project_bytes_before"], bool):
        raise SupervisorError("project_bytes_before must be an integer byte count")
    if authorization.get("experiment_id") != "E163":
        raise PermissionError("coordinator authorization is not for E163")
    if authorization.get("execution_authorized") is not True:
        raise PermissionError("coordinator has not enabled E163 execution")
    if authorization.get("run_spec_sha256") != _sha256(spec_raw):
        raise PermissionError("coordinator authorization is not bound to these run-spec bytes")
    if authorization.get("limits") != EXPECTED_LIMITS:
        raise PermissionError("coordinator authorization limits differ from E163")
    directive = authorization.get("coordinator_directive")
    if not isinstance(directive, str) or not directive.strip():
        raise PermissionError("coordinator directive is missing")

    _canonical(spec["source_cwd"], "source cwd", root=E_ROOT)
    _canonical(spec["orchestration_root"], "orchestration root", root=E_ROOT)
    _canonical(spec["python"], "compute interpreter", root=PROJECT,
               allow_internal_symlink=True)
    if not SOURCE_CWD.is_dir() or not ORCHESTRATION_ROOT.is_dir() or not RUNNER.is_file():
        raise SupervisorError("source or orchestration directory is missing")
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise SupervisorError("run001 already exists; refusing to overwrite campaign evidence")
    _validate_environment(spec.get("environment"))
    return spec, authorization, spec_path, authorization_path


def _proc_stat(pid: int) -> dict[str, int | str]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise SupervisorError(f"cannot parse /proc/{pid}/stat")
    fields = raw[close + 2:].split()
    if len(fields) < 20:
        raise SupervisorError(f"short /proc/{pid}/stat record")
    result: dict[str, int | str] = {
        "state": fields[0],
        "utime_ticks": int(fields[11]),
        "stime_ticks": int(fields[12]),
        "cutime_ticks": int(fields[13]),
        "cstime_ticks": int(fields[14]),
        "start_ticks": int(fields[19]),
    }
    return result


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


def _storage_bytes(root: Path) -> int:
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise SupervisorError("E163 root is not a real directory")
    total = 0
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        parent = Path(directory)
        for name in list(names):
            path = parent / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SupervisorError(f"non-directory or symlink under E163 root: {path}")
        for name in files:
            path = parent / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SupervisorError(f"non-regular file or symlink under E163 root: {path}")
            if info.st_nlink != 1:
                raise SupervisorError(f"hard-linked file under E163 root: {path}")
            total += info.st_size
    return total


def _validate_capacity(spec: dict, *, anticipated_bytes: int = 0, reserve_footer: bool = True) -> int:
    e_bytes = _storage_bytes(E_ROOT)
    cap = EXPECTED_LIMITS["new_artifact_cache_temp_bytes"]
    reserve = FOOTER_BYTE_RESERVE if reserve_footer else 0
    if e_bytes + anticipated_bytes + reserve > cap:
        raise SupervisorError("E163 byte allowance or reserved footer space would be exceeded")
    if spec["project_bytes_before"] + e_bytes + anticipated_bytes > EXPECTED_LIMITS["aggregate_remote_project_bytes"]:
        raise SupervisorError("aggregate project byte cap would be exceeded")
    return e_bytes


def _make_worker_environment(spec_environment: dict[str, str]) -> dict[str, str]:
    # Pass only ordinary runtime settings; never forward SSH agents or credentials.
    child_env: dict[str, str] = {}
    for key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV"):
        if key in os.environ:
            child_env[key] = os.environ[key]
    child_env.update(spec_environment)
    child_env.update({
        "E163_SUPERVISED": "1",
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
    def __init__(self, spec: dict, authorization: dict, spec_path: Path,
                 authorization_path: Path):
        self.spec = spec
        self.authorization = authorization
        self.spec_path = spec_path
        self.authorization_path = authorization_path
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
        self.worker_returncode: int | None = None
        self.status = "PRECHECK"
        self.error: dict | None = None
        self.worker_env = _make_worker_environment(spec["environment"])
        self.run_root = RUN_ROOT
        self.meter_path = RUN_ROOT / "meter.jsonl"
        self.wait_path = RUN_ROOT / "wait4.jsonl"
        self.meter_stream = None
        self.wait_stream = None

    def start(self) -> None:
        # run001 is unique; no campaign file is replaced.
        _validate_capacity(self.spec)
        self.run_root.mkdir(mode=0o700)
        self._ensure_runtime_directories()
        self.meter_stream = self.meter_path.open("x", encoding="utf-8", buffering=1)
        self.wait_stream = self.wait_path.open("x", encoding="utf-8", buffering=1)
        self.last_storage_bytes = _storage_bytes(E_ROOT)
        self.record_event("campaign_started", campaign_start_source="supervisor /proc/self/stat start ticks",
                          supervisor_pid=os.getpid(), supervisor_start_ticks=self.own_start_ticks,
                          ticks_per_second=self.ticks_per_second,
                          self_rusage_at_record=_ru_record(resource.getrusage(resource.RUSAGE_SELF)),
                          authorization_sha256=_sha256(self.authorization_path.read_bytes()),
                          run_spec_sha256=_sha256(self.spec_path.read_bytes()),
                          limits=EXPECTED_LIMITS)

    def _ensure_runtime_directories(self) -> None:
        requested = sorted({Path(self.worker_env[key]) for key in
                            ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
                             "XDG_DATA_HOME", "XDG_STATE_HOME", "MPLCONFIGDIR", "TORCH_HOME",
                             "TORCH_EXTENSIONS_DIR")})
        for path in requested:
            _canonical(str(path), "worker cache or temp directory", must_exist=False, root=E_ROOT)
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
                # A child normally remains as a zombie until wait4. Retry only this
                # recorded PID; never enumerate or inspect unrelated processes.
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
        row = {"kind": kind, "monotonic": time.monotonic(), "campaign_elapsed_wall_seconds": self.elapsed_wall()}
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

    def _reap_untracked_fork_failure(self, pid: int, purpose: str) -> None:
        """Bound cleanup if /proc identity could not be captured after fork."""
        deadline = time.monotonic() + REAP_TIMEOUT_SECONDS
        while True:
            waited, status, usage = os.wait4(pid, os.WNOHANG)
            if waited == pid:
                row = {
                    "pid": pid,
                    "process_group": None,
                    "start_ticks": None,
                    "purpose": purpose,
                    "identity_state": "start_tick_unavailable; direct fork-return PID killed before gate release",
                    "wait_status": status,
                    "exit_code": os.waitstatus_to_exitcode(status),
                    "rusage": _ru_record(usage),
                    "wait4_cpu_seconds_including_waited_descendants": _ru_cpu(usage),
                    "reaped_monotonic": time.monotonic(),
                }
                self.completed.append(row)
                if self.wait_stream is not None:
                    self._write_line(self.wait_stream, self.wait_path, row)
                return
            if time.monotonic() >= deadline:
                raise SupervisorError("untracked fork failure child did not exit within bounded cleanup")
            time.sleep(0.01)

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

    def _fork_gated(self, purpose: str, child_entry, *, process_group: bool = True) -> tuple[dict, int]:
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
                if process_group:
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
                "process_group": pid if process_group else os.getpgid(pid),
                "start_ticks": start_ticks,
                "purpose": purpose,
                "spawned_monotonic": time.monotonic(),
            }
            self.handles[pid] = handle
            if process_group:
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
                try:
                    self._reap_untracked_fork_failure(pid, purpose)
                except BaseException:
                    pass
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

        # The pipe used for the child report remains open across the fork, while the
        # gate pipe in _fork_gated is independent and closes on both sides.
        handle, pid = self._fork_gated("cpu_descendant_accounting_precheck", child_entry)
        os.close(report_write)
        report_data = bytearray()
        probe_deadline = time.monotonic() + CPU_PROBE_WALL_TIMEOUT_SECONDS
        try:
            while True:
                self._check_limits()
                remaining = probe_deadline - time.monotonic()
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
        descendant = float(report["reported_descendant_cpu_seconds"])
        tolerance = max(0.02, reported * 0.20)
        if waited["exit_code"] != 0 or reported < 0.08 or abs(direct - reported) > tolerance:
            raise SupervisorError("wait4 descendant accounting precheck failed")
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
        if waited is None or waited["exit_code"] != -signal.SIGTERM:
            raise SupervisorError("recorded process-group stop precheck failed")
        self.record_event("owned_process_group_stop_precheck_passed", child=handle,
                          exit_code=waited["exit_code"], wait4_cpu_seconds=waited["wait4_cpu_seconds_including_waited_descendants"])
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
            # The signal has already been delivered. Keep cleanup independent of
            # logging capacity so a log failure cannot leave an owned child live.
            pass

    def _check_limits(self) -> dict:
        self.reap_all_nonblocking()
        snapshot = self.cpu_snapshot()
        wall = self.elapsed_wall()
        if self.stop_signal is not None:
            raise SupervisorError(f"supervisor received stop signal {self.stop_signal}")
        if snapshot["total_cpu_seconds"] >= CPU_STOP_SECONDS:
            raise SupervisorError("campaign reached the 115 CPU-second safety stop")
        if wall >= WALL_STOP_SECONDS:
            raise SupervisorError("campaign reached the 170-second wall safety stop")
        return {**snapshot, "campaign_wall_seconds": wall}

    def _wait_for_reap(self, pid: int, *, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while True:
            waited = self._reap(pid, block=False)
            if waited is not None:
                return waited
            if pid not in self.handles:
                return next((row for row in reversed(self.completed) if row["pid"] == pid), None)
            snapshot = self.cpu_snapshot()
            if pid not in self.handles:
                return next((row for row in reversed(self.completed) if row["pid"] == pid), None)
            wall = self.elapsed_wall()
            if (snapshot["total_cpu_seconds"] >= EXPECTED_LIMITS["cpu_seconds"]
                    or wall >= EXPECTED_LIMITS["campaign_wall_seconds"]):
                raise SupervisorError("hard campaign limit reached while reaping an owned child")
            if time.monotonic() >= deadline:
                raise SupervisorError("owned child did not exit within its bounded reap window")
            time.sleep(0.01)

    def _worker_cpu_limit(self) -> int:
        snapshot = self._check_limits()
        return max(0, math.floor(CPU_STOP_SECONDS - snapshot["total_cpu_seconds"]))

    def _launch_worker(self) -> tuple[dict, int]:
        cpu_limit = self._worker_cpu_limit()
        if cpu_limit < 1:
            raise SupervisorError("less than one CPU second remains before the safe worker limit")
        environment = self.worker_env
        argv = [str(PYTHON), *self.spec["worker_argv"]]
        runner_path = RUNNER
        source_cwd = SOURCE_CWD
        file_limit = EXPECTED_LIMITS["new_artifact_cache_temp_bytes"] - self.last_storage_bytes - FOOTER_BYTE_RESERVE
        if file_limit < 1:
            raise SupervisorError("insufficient remaining E163 bytes for worker output and final evidence")
        console_path = RUN_ROOT / "worker_console.log"
        _validate_capacity(self.spec, anticipated_bytes=0)
        console_fd = os.open(console_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

        def exec_worker() -> None:
            try:
                os.dup2(console_fd, 1)
                os.dup2(console_fd, 2)
                if console_fd > 2:
                    os.close(console_fd)
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
                os.chdir(source_cwd)
                os.execve(str(PYTHON), argv, environment)
            except BaseException as exc:
                message = f"E163 worker exec failed: {type(exc).__name__}: {exc}\n".encode("utf-8", "replace")
                try:
                    os.write(2, message[:2048])
                finally:
                    os._exit(127)

        try:
            handle, pid = self._fork_gated("scientific_worker", exec_worker)
        finally:
            os.close(console_fd)
        self.worker_pid = pid
        self.record_event("worker_command_started", child=handle, executable=str(PYTHON),
                          argv=argv, cwd=str(source_cwd), environment_keys=sorted(environment),
                          home_preserved=(environment.get("HOME") == os.environ.get("HOME")),
                          soft_and_hard_cpu_limit_seconds=cpu_limit,
                          per_file_size_limit_bytes=file_limit,
                          runner=str(runner_path), console_log=str(console_path))
        return handle, pid

    def _terminate_worker(self, reason: str) -> None:
        pid = self.worker_pid
        if pid is None or pid not in self.handles:
            return
        self._terminate_owned(self.handles[pid], reason)

    def _terminate_owned(self, handle: dict, reason: str) -> None:
        pid = int(handle["pid"])
        if pid not in self.handles:
            return
        try:
            self.record_event("owned_child_stop_requested", reason=reason, child=handle)
        except BaseException:
            pass
        try:
            self._signal_owned_group(handle, signal.SIGTERM)
        except BaseException:
            # The direct PID and start identity still provide a bounded fallback
            # for this one child owned by this supervisor.
            current = _proc_stat(pid)
            if int(current["start_ticks"]) != int(handle["start_ticks"]):
                raise SupervisorError("refusing fallback signal to a reused PID")
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        while time.monotonic() < deadline:
            self.reap_all_nonblocking()
            if pid not in self.handles:
                return
            time.sleep(0.01)
        if pid in self.handles:
            try:
                self._signal_owned_group(handle, signal.SIGKILL)
            except BaseException:
                current = _proc_stat(pid)
                if int(current["start_ticks"]) != int(handle["start_ticks"]):
                    raise SupervisorError("refusing fallback kill to a reused PID")
                os.kill(pid, signal.SIGKILL)
            self._wait_for_reap(pid, timeout=REAP_TIMEOUT_SECONDS)

    def _monitor_worker(self, pid: int) -> dict:
        next_storage_check = time.monotonic()
        next_meter = time.monotonic()
        result = None
        stop_reason = None
        while True:
            self.reap_all_nonblocking()
            if pid not in self.handles:
                result = next((row for row in reversed(self.completed) if row["pid"] == pid), None)
                break
            snapshot = self.cpu_snapshot()
            wall = self.elapsed_wall()
            if self.stop_signal is not None:
                stop_reason = f"supervisor received signal {self.stop_signal}"
            elif snapshot["total_cpu_seconds"] >= CPU_STOP_SECONDS:
                stop_reason = "115 CPU-second safety stop"
            elif wall >= WALL_STOP_SECONDS:
                stop_reason = "170-second wall safety stop"
            if stop_reason:
                self.record_event("safety_stop_triggered", reason=stop_reason,
                                  cpu_snapshot=snapshot, campaign_wall_seconds=wall)
                self._terminate_worker(stop_reason)
                break
            now = time.monotonic()
            if now >= next_storage_check:
                self.last_storage_bytes = _validate_capacity(self.spec)
                if now >= next_meter:
                    self.record_event("meter_sample", cpu_snapshot=snapshot,
                                      campaign_wall_seconds=wall,
                                      e_root_apparent_bytes=self.last_storage_bytes,
                                      project_baseline_bytes=self.spec["project_bytes_before"],
                                      aggregate_project_apparent_bytes=self.spec["project_bytes_before"] + self.last_storage_bytes)
                    next_meter = now + METER_INTERVAL_SECONDS
                next_storage_check = now + METER_INTERVAL_SECONDS
            time.sleep(POLL_INTERVAL_SECONDS)
        self.reap_all_nonblocking()
        if result is None:
            result = next((row for row in reversed(self.completed) if row["pid"] == pid), None)
        if result is None:
            raise SupervisorError("scientific worker lifecycle lacks a wait4 record")
        self.worker_returncode = int(result["exit_code"])
        self.record_event("worker_lifecycle_closed", child=result,
                          cpu_snapshot=self.cpu_snapshot(), campaign_wall_seconds=self.elapsed_wall())
        return result

    def _validate_worker_result(self, exit_code: int) -> None:
        if exit_code != 0:
            raise SupervisorError(f"scientific worker exited with code {exit_code}")
        state_path = RUN_ROOT / "state_worker.json"
        env_path = RUN_ROOT / "environment.json"
        report_path = RUN_ROOT / "scientific_report.json"
        if not state_path.is_file() or not env_path.is_file() or not report_path.is_file():
            raise SupervisorError("worker omitted required terminal evidence")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        environment = json.loads(env_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (state.get("status") != "COMPLETE" or state.get("completed_fits") != 32
                or state.get("environment_interactions") != 0):
            raise SupervisorError("worker terminal state is incomplete or outside the fixed budget")
        if (environment.get("torch_threads") != 1 or environment.get("torch_interop_threads") != 1
                or environment.get("platform") != "linux"):
            raise SupervisorError("worker did not report the fixed Linux single-thread configuration")
        if Path(environment.get("executable", "")).resolve(strict=True) != PYTHON.resolve(strict=True):
            raise SupervisorError("compute worker executable differs from the allocated interpreter")
        if report.get("scored_count") != 16 or len(report.get("completed_fits", [])) != 32:
            raise SupervisorError("worker report does not cover the complete fixed batch")

    def _children_usage_audit(self) -> dict:
        if self.handles:
            raise SupervisorError("cannot finalize while a recorded child remains unreaped")
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        wait4_total = self._children_cpu()
        rusage_total = _ru_cpu(children)
        if abs(wait4_total - rusage_total) > 0.03:
            raise SupervisorError("final wait4 sum differs from RUSAGE_CHILDREN")
        return {"wait4_sum_cpu_seconds": wait4_total,
                "rusage_children_cpu_seconds": rusage_total,
                "difference_seconds": abs(wait4_total - rusage_total)}

    def finalize(self, status: str, *, return_code: int, error: dict | None = None) -> int:
        for handle in list(self.handles.values()):
            if int(handle["pid"]) in self.handles:
                try:
                    self._terminate_owned(handle, "finalization cleanup")
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
        if status == "COMPLETE" and (cpu_upper > EXPECTED_LIMITS["cpu_seconds"]
                                     or wall_upper > EXPECTED_LIMITS["campaign_wall_seconds"]):
            status = "FINALIZATION_RESERVE_EXCEEDED"
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
            state = {
                "experiment_id": "E163",
                "status": status,
                "return_code": return_code,
                "worker_return_code": self.worker_returncode,
                "completed_direct_children": len(self.completed),
                "child_wait4_records": [
                    {"pid": row["pid"], "purpose": row["purpose"], "exit_code": row["exit_code"],
                     "start_ticks": row["start_ticks"],
                     "wait4_cpu_seconds_including_waited_descendants": row["wait4_cpu_seconds_including_waited_descendants"]}
                    for row in self.completed
                ],
                "cpu_snapshot_before_final_footer": snapshot,
                "bounded_campaign_cpu_seconds_upper": cpu_upper,
                "cpu_total_scope": "supervisor RUSAGE_SELF + completed direct-child wait4 totals + live owned /proc counters; child wait4 descendants checked by in-campaign precheck",
                "campaign_wall_seconds_before_final_footer": wall,
                "bounded_campaign_wall_seconds_upper": wall_upper,
                "campaign_start_source": "supervisor process start ticks from /proc/self/stat; includes Python interpreter startup, excludes any unobserved shell or SSH launcher",
                "limits": EXPECTED_LIMITS,
                "safety_stops": {"cpu_seconds": CPU_STOP_SECONDS, "wall_seconds": WALL_STOP_SECONDS},
                "finalization_reserve": {"cpu_seconds": FOOTER_CPU_RESERVE_SECONDS,
                                          "wall_seconds": FOOTER_WALL_RESERVE_SECONDS,
                                          "byte_reserve": FOOTER_BYTE_RESERVE},
                "e_root_apparent_bytes": _storage_bytes(E_ROOT),
                "project_bytes_before_non_e163_baseline": self.spec["project_bytes_before"],
                "aggregate_project_apparent_bytes": self.spec["project_bytes_before"] + _storage_bytes(E_ROOT),
                "wait4_children_audit": audit,
                "wait4_audit_error": audit_error,
                "final_footer_tail_measurement": "not observed after supervisor exit; explicitly bounded by the reported CPU and wall reserves",
                "environment_interactions": 0,
                "gpu_devices": 0,
                "real_fit_count": 32 if status == "COMPLETE" else None,
                "error": error,
                "evidence_note": "The final tiny footer is covered by the explicit reserve; command-lifetime cost excludes no claimed launcher because no external launcher counters are available.",
            }
            state_path = RUN_ROOT / "state.json"
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
            handle, pid = self._launch_worker()
            worker_result = self._monitor_worker(pid)
            _validate_capacity(self.spec)
            self._validate_worker_result(worker_result["exit_code"])
            return self.finalize("COMPLETE", return_code=0)
        except BaseException as exc:
            error = {"exception_type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc()}
            self.error = error
            try:
                self.record_event("campaign_error", **error)
            except BaseException:
                pass
            self._terminate_worker("supervisor error or failed precheck")
            status = "PRECHECK_FAILED" if self.worker_pid is None else "INCOMPLETE"
            return self.finalize(status, return_code=2, error=error)

    def _request_stop(self, signum, frame) -> None:
        self.stop_signal = int(signum)


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise the authorized Linux E163 campaign")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    try:
        spec, authorization, spec_path, authorization_path = _validate_spec_and_authorization(
            args.spec, args.authorization)
    except PermissionError as exc:
        print(f"E163 authorization refused: {exc}", file=sys.stderr)
        return 3
    except BaseException as exc:
        print(f"E163 preflight refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    campaign = None
    try:
        campaign = Campaign(spec, authorization, spec_path, authorization_path)
    except BaseException as exc:
        print(f"E163 supervisor initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    return campaign.run()


if __name__ == "__main__":
    raise SystemExit(main())

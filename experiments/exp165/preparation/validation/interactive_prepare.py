"""Prepare the E165 CHECKS inputs through one user-driven SSH session.

This helper never runs the CHECKS stage.  It verifies the exact local manifest,
waits for remote capacity admission, then streams only the approved 42 files.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import queue
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


REMOTE_USER_HOST = "robot@100.111.191.97"
PROJECT_ROOT = "/data/robot/p001"
E_ROOT = "/data/robot/p001/e165"
STAGE_SPEC_PATH = "/data/robot/p001/e165/o001/stage_checks.json"
PREPARATION_DIRECTIVE = "E165_CHECKS_PREPARE001_20261028_APPROVED"
TRANSFER_MANIFEST_SHA256 = "1076fadb7ac7503d5fc0dcc21be49c57ee17cce83fb86d989bab677507d58e2b"
FROZEN_CANDIDATE_SHA256 = "5f4741444ee0d2ee326c93c35d95680c91020fdc08e3d79b8adbe0e118f19d77"
SCIENTIFIC_SOURCE_IDENTITY_SHA256 = "4e96df16ca207ea961bafaed6b92dfbee07d7cdd55dc7cf986b5fb2b48189f0b"
SCIENTIFIC_MANIFEST_SHA256 = "64f9f5ef31d7e3157ddaa7c59fa21e4acb901002c4046566dfa43e6e314d4ec3"
TRANSFER_FILE_COUNT = 42
TRANSFER_BYTES = 1_202_701
PROJECT_CAP_BYTES = 100_000_000_000
E165_ALLOWANCE_BYTES = 134_217_728
INITIAL_FREE_INODE_FLOOR = 256
POST_TRANSFER_FREE_INODE_FLOOR = 128
SSH_READY_TIMEOUT_SECONDS = 180
SSH_COMPLETION_TIMEOUT_SECONDS = 120
MAX_PROTOCOL_LINE_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_TAR_BYTES = 16 * 1024 * 1024
WINDOWS_COMMAND_LINE_SAFE_MAX = 30_000
ALLOWED_SUFFIXES = {".py", ".json", ".md", ".txt", ".sh", ".ini"}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREPARATION_DIR = HERE.parent
MANIFEST_PATH = HERE / "checks_transfer_manifest.json"
FROZEN_CANDIDATE_PATH = HERE / "checks_frozen_candidate.json"
COORDINATOR_PREP_PATH = PREPARATION_DIR / "coordinator_prepare001.json"


REMOTE_SCRIPT = r'''import base64, copy, datetime, hashlib, io, json, os, pwd, stat, sys, tarfile

PROJECT_ROOT = "/data/robot/p001"
E_ROOT = "/data/robot/p001/e165"
STAGE_SPEC_PATH = "/data/robot/p001/e165/o001/stage_checks.json"
TRANSFER_SHA256 = "1076fadb7ac7503d5fc0dcc21be49c57ee17cce83fb86d989bab677507d58e2b"
CANDIDATE_SHA256 = "5f4741444ee0d2ee326c93c35d95680c91020fdc08e3d79b8adbe0e118f19d77"
SOURCE_IDENTITY_SHA256 = "4e96df16ca207ea961bafaed6b92dfbee07d7cdd55dc7cf986b5fb2b48189f0b"
SCIENTIFIC_MANIFEST_SHA256 = "64f9f5ef31d7e3157ddaa7c59fa21e4acb901002c4046566dfa43e6e314d4ec3"
EXPECTED_COUNT = 42
EXPECTED_TRANSFER_BYTES = 1202701
PROJECT_CAP_BYTES = 100000000000
E165_ALLOWANCE_BYTES = 134217728
INITIAL_FREE_INODE_FLOOR = 256
POST_TRANSFER_FREE_INODE_FLOOR = 128
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_TAR_BYTES = 16 * 1024 * 1024
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def send(record):
    sys.stdout.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        block = stream.read(min(1024 * 1024, remaining))
        if not block:
            raise ValueError("protocol stream ended early")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def open_dir_component(parent_fd, name):
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("unsafe path component")
    flags = os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
    return os.open(name, flags, dir_fd=parent_fd)


def project_fd_open():
    if not O_NOFOLLOW or not O_DIRECTORY:
        raise ValueError("filesystem does not support no-follow directory opens")
    fd = os.open("/", os.O_RDONLY | O_DIRECTORY | O_CLOEXEC)
    try:
        for component in ("data", "robot", "p001"):
            next_fd = open_dir_component(fd, component)
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


def open_e_root(project_fd, create=False):
    try:
        return open_dir_component(project_fd, "e165")
    except FileNotFoundError:
        if not create:
            return None
        os.mkdir("e165", 0o700, dir_fd=project_fd)
        return open_dir_component(project_fd, "e165")


def sum_tree(fd, *, strict_e165=False, exclude_e165=False):
    total = 0
    stack = [(os.dup(fd), "")]
    try:
        while stack:
            directory_fd, prefix = stack.pop()
            try:
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        name = entry.name
                        if prefix == "" and name == "e165" and exclude_e165:
                            continue
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            child_fd = open_dir_component(directory_fd, name)
                            stack.append((child_fd, name if not prefix else prefix + "/" + name))
                        elif strict_e165 and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
                            raise ValueError("E165 contains a symlink, non-regular file, or hard link")
                        else:
                            total += info.st_size
            finally:
                os.close(directory_fd)
    finally:
        for directory_fd, _ in stack:
            try:
                os.close(directory_fd)
            except OSError:
                pass
    return total


def measure(project_fd):
    baseline = sum_tree(project_fd, exclude_e165=True)
    e_fd = open_e_root(project_fd, create=False)
    try:
        e_bytes = sum_tree(e_fd, strict_e165=True) if e_fd is not None else 0
    finally:
        if e_fd is not None:
            os.close(e_fd)
    fs = os.statvfs(project_fd)
    block_size = fs.f_frsize or fs.f_bsize
    free_bytes = fs.f_bavail * block_size
    free_inodes = fs.f_favail
    if free_inodes < 0:
        raise ValueError("filesystem does not report available inodes")
    remaining = E165_ALLOWANCE_BYTES - e_bytes
    if remaining < 0:
        raise ValueError("existing E165 usage exceeds its 128 MiB allowance")
    if baseline + E165_ALLOWANCE_BYTES > PROJECT_CAP_BYTES:
        raise ValueError("the full E165 allowance exceeds the 100 GB project cap")
    if free_bytes < remaining:
        raise ValueError("free bytes do not fit the remaining E165 allowance")
    return {
        "project_bytes_before": baseline,
        "project_total_bytes_before": baseline + e_bytes,
        "e165_bytes_before": e_bytes,
        "e165_allowance_remaining_bytes": remaining,
        "project_free_bytes_before": free_bytes,
        "project_free_inodes_before": free_inodes,
        "project_cap_bytes": PROJECT_CAP_BYTES,
        "e165_allowance_bytes": E165_ALLOWANCE_BYTES,
        "initial_free_inode_floor": INITIAL_FREE_INODE_FLOOR,
        "admission_measured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def validate_destination(path):
    if not isinstance(path, str) or "\\" in path or "//" in path:
        raise ValueError("invalid destination path")
    if not path.startswith(E_ROOT + "/"):
        raise ValueError("destination is outside E165")
    relative = path[len(PROJECT_ROOT) + 1:]
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("destination contains traversal")
    if len(parts) < 3 or parts[0] != "e165" or parts[1] not in ("s001", "o001"):
        raise ValueError("destination is outside the exact E165 staging roots")
    if any(part.lower() in (".git", "agent.md", "agents.md") for part in parts):
        raise ValueError("forbidden destination path")
    return parts


def inspect_parent(project_fd, parts):
    fd = os.dup(project_fd)
    missing = False
    try:
        for component in parts[:-1]:
            if missing:
                continue
            try:
                child = open_dir_component(fd, component)
            except FileNotFoundError:
                missing = True
                continue
            os.close(fd)
            fd = child
        if not missing:
            try:
                os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("destination already exists; preserving existing evidence")
    finally:
        os.close(fd)


def required_directory_set(paths):
    result = set()
    for parts in paths:
        for end in range(1, len(parts)):
            result.add(tuple(parts[:end]))
    return result


def mkdir_open_chain(project_fd, directory_parts):
    fd = os.dup(project_fd)
    try:
        for component in directory_parts:
            try:
                os.mkdir(component, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child = open_dir_component(fd, component)
            os.close(fd)
            fd = child
        return fd
    except Exception:
        os.close(fd)
        raise


def validate_metadata(envelope, initial):
    if envelope.get("manifest_sha256") != TRANSFER_SHA256 or envelope.get("candidate_sha256") != CANDIDATE_SHA256:
        raise ValueError("approved manifest or frozen candidate SHA-256 binding mismatch")
    if envelope.get("scientific_source_identity_sha256") != SOURCE_IDENTITY_SHA256:
        raise ValueError("scientific source identity mismatch")
    if envelope.get("scientific_manifest_sha256") != SCIENTIFIC_MANIFEST_SHA256:
        raise ValueError("scientific manifest SHA-256 mismatch")
    manifest_raw = base64.b64decode(envelope.get("manifest_b64", ""), validate=True)
    if digest(manifest_raw) != TRANSFER_SHA256:
        raise ValueError("approved transfer manifest SHA-256 mismatch")
    manifest = json.loads(manifest_raw)
    if (manifest.get("format") != "e165_checks_transfer_candidate_v1"
            or manifest.get("state") != "REVIEW_ONLY_NO_REMOTE_AUTHORIZATION"
            or manifest.get("remote_operations_authorized") is not False
            or manifest.get("remote_references") != []
            or manifest.get("scientific_source_identity_sha256") != SOURCE_IDENTITY_SHA256
            or manifest.get("scientific_manifest_sha256") != SCIENTIFIC_MANIFEST_SHA256
            or manifest.get("file_count") != EXPECTED_COUNT
            or manifest.get("total_bytes") != EXPECTED_TRANSFER_BYTES
            or manifest.get("stage_spec_path_after_admission") != STAGE_SPEC_PATH):
        raise ValueError("approved transfer manifest metadata differs from the reviewed scope")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list) or len(manifest_files) != EXPECTED_COUNT:
        raise ValueError("approved manifest does not contain exactly 42 files")
    manifest_map = {}
    for item in manifest_files:
        if not isinstance(item, dict) or set(item) != {"local_path", "destination", "bytes", "sha256"}:
            raise ValueError("approved manifest descriptor has unexpected fields")
        parts = validate_destination(item["destination"])
        name = "/".join(parts)
        if name in manifest_map:
            raise ValueError("approved manifest contains duplicate destinations")
        manifest_map[name] = {"bytes": item["bytes"], "sha256": item["sha256"]}
    files = envelope.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_COUNT:
        raise ValueError("metadata does not contain exactly 42 file descriptors")
    expected = {}
    total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("file descriptor has unexpected fields")
        parts = validate_destination(item["path"])
        size = item.get("bytes")
        sha = item.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("invalid manifest byte length or digest")
        if any(char not in "0123456789abcdef" for char in sha):
            raise ValueError("invalid manifest SHA-256 encoding")
        name = "/".join(parts)
        if name in expected:
            raise ValueError("duplicate destination in transfer metadata")
        expected[name] = {"parts": parts, "bytes": size, "sha256": sha}
        total += size
    if total != EXPECTED_TRANSFER_BYTES:
        raise ValueError("metadata total does not equal the approved transfer size")
    if {name: {"bytes": row["bytes"], "sha256": row["sha256"]} for name, row in expected.items()} != manifest_map:
        raise ValueError("transfer descriptors differ from the exact approved manifest")
    final_spec_raw = base64.b64decode(envelope.get("final_spec_b64", ""), validate=True)
    if digest(final_spec_raw) != envelope.get("final_spec_sha256"):
        raise ValueError("dynamic final CHECKS spec SHA-256 mismatch")
    final_spec = json.loads(final_spec_raw)
    if (final_spec.get("format") != "e165_execution_spec_v1"
            or final_spec.get("experiment_id") != "E165"
            or final_spec.get("stage") != "CHECKS"):
        raise ValueError("unexpected dynamic CHECKS spec")
    if final_spec.get("preparation_state") != "FROZEN" or final_spec.get("resource_authorized") is not True:
        raise ValueError("dynamic CHECKS spec lacks the exact frozen/resource state")
    if final_spec.get("source_identity_sha256") != SOURCE_IDENTITY_SHA256:
        raise ValueError("dynamic CHECKS spec source identity mismatch")
    if final_spec.get("project_bytes_before") != initial["project_bytes_before"]:
        raise ValueError("dynamic CHECKS spec baseline differs from remote admission")
    if final_spec.get("admission_measured_utc") != initial["admission_measured_utc"]:
        raise ValueError("dynamic CHECKS spec UTC differs from remote admission")
    if final_spec.get("execution_authorized") is not None:
        raise ValueError("execution authorization must remain separate from the CHECKS spec")
    if (final_spec.get("project_root") != PROJECT_ROOT
            or final_spec.get("e_root") != E_ROOT
            or final_spec.get("orchestration_root") != E_ROOT + "/o001"
            or final_spec.get("source_cwd") != E_ROOT + "/s001"
            or final_spec.get("run_root") != E_ROOT + "/c001"):
        raise ValueError("dynamic CHECKS spec path roots differ from the fixed E165 contract")
    if (final_spec.get("limits") != {"cpu_seconds": 20, "wall_seconds": 60}
            or final_spec.get("round_limits", {}).get("new_artifact_cache_temp_bytes") != E165_ALLOWANCE_BYTES
            or final_spec.get("round_limits", {}).get("aggregate_remote_project_bytes") != PROJECT_CAP_BYTES):
        raise ValueError("dynamic CHECKS spec resource limits differ from the approved allocation")
    worker_argv = final_spec.get("worker_argv")
    if (not isinstance(worker_argv, list) or len(worker_argv) != 6
            or worker_argv[0] != "-B"
            or worker_argv[1] != E_ROOT + "/o001/checks_entry.py"
            or worker_argv[2] != "--spec"
            or worker_argv[3] != STAGE_SPEC_PATH
            or worker_argv[4] != "--authorization"
            or worker_argv[5] != E_ROOT + "/o001/authorization_checks.json"):
        raise ValueError("dynamic CHECKS spec worker arguments differ from the frozen contract")
    candidate_rows = final_spec.get("source_files", []) + final_spec.get("orchestration_files", [])
    candidate_map = {}
    for row in candidate_rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("dynamic spec descriptor has unexpected fields")
        candidate_map[row["path"]] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    metadata_map = {item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]} for item in files}
    if candidate_map != metadata_map:
        raise ValueError("dynamic spec descriptors do not match the exact transfer metadata")
    scientific = final_spec.get("scientific_manifest")
    if (not isinstance(scientific, dict)
            or scientific.get("sha256") != SCIENTIFIC_MANIFEST_SHA256
            or scientific.get("path") not in metadata_map
            or scientific.get("bytes") != metadata_map[scientific["path"]]["bytes"]):
        raise ValueError("dynamic spec scientific manifest descriptor is invalid")
    archive_size = envelope.get("tar_bytes")
    if type(archive_size) is not int or not 0 < archive_size <= MAX_TAR_BYTES:
        raise ValueError("invalid bounded tar stream length")
    return final_spec, final_spec_raw, expected, archive_size


def validate_tar(raw_tar, expected):
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive:
        members = archive.getmembers()
        if len(members) != EXPECTED_COUNT:
            raise ValueError("tar stream does not contain exactly 42 entries")
        for member in members:
            if not member.isreg() or member.islnk() or member.issym() or member.isdir():
                raise ValueError("tar stream contains a non-regular member")
            descriptor = expected.get(member.name)
            if descriptor is None or member.name in contents:
                raise ValueError("tar stream contains an unexpected or duplicate path")
            if member.size != descriptor["bytes"]:
                raise ValueError("tar member byte length differs from manifest")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("tar member cannot be read")
            data = source.read(member.size + 1)
            if len(data) != member.size or digest(data) != descriptor["sha256"]:
                raise ValueError("tar member bytes or SHA-256 differ from manifest")
            contents[member.name] = data
    if set(contents) != set(expected):
        raise ValueError("tar stream does not match the exact manifest path set")
    return contents


def write_exclusive(project_fd, parts, data):
    parent_fd = mkdir_open_chain(project_fd, parts[:-1])
    fd = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | O_CLOEXEC
        fd = os.open(parts[-1], flags, 0o600, dir_fd=parent_fd)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def verify_written(project_fd, parts, expected_size, expected_sha):
    parent_fd = mkdir_open_chain(project_fd, parts[:-1])
    try:
        fd = os.open(parts[-1], os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != expected_size:
                raise ValueError("written destination metadata differs")
            hasher = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                hasher.update(chunk)
            if total != expected_size or hasher.hexdigest() != expected_sha:
                raise ValueError("written destination SHA-256 differs")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def main():
    if pwd.getpwuid(os.geteuid()).pw_name != "robot":
        raise ValueError("remote process is not running as robot")
    project_fd = project_fd_open()
    try:
        initial = measure(project_fd)
        if initial["project_free_inodes_before"] < INITIAL_FREE_INODE_FLOOR:
            raise ValueError("free inode count is below the 256-inode admission floor")
        send({"type": "READY", **initial})
        prefix = read_exact(sys.stdin.buffer, 4)
        metadata_size = int.from_bytes(prefix, "big")
        if not 0 < metadata_size <= MAX_METADATA_BYTES:
            raise ValueError("metadata frame length is invalid")
        envelope = json.loads(read_exact(sys.stdin.buffer, metadata_size))
        final_spec, final_spec_raw, expected, archive_size = validate_metadata(envelope, initial)
        target_parts = {name: row["parts"] for name, row in expected.items()}
        spec_parts = validate_destination(STAGE_SPEC_PATH)
        for parts in list(target_parts.values()) + [spec_parts]:
            inspect_parent(project_fd, parts)
        required_dirs = required_directory_set(list(target_parts.values()) + [spec_parts])
        required_inode_count = max(INITIAL_FREE_INODE_FLOOR, EXPECTED_COUNT + 1 + len(required_dirs) + 64)
        current = measure(project_fd)
        if (current["project_bytes_before"] != initial["project_bytes_before"]
                or current["e165_bytes_before"] != initial["e165_bytes_before"]):
            raise ValueError("project usage changed after initial capacity admission")
        if current["project_free_bytes_before"] < current["e165_allowance_remaining_bytes"]:
            raise ValueError("free bytes no longer fit the full remaining allowance")
        if current["project_free_inodes_before"] < required_inode_count:
            raise ValueError("free inodes do not fit the exact transfer plus reserve")
        send({"type": "TRANSFER_READY", "admission_measured_utc": initial["admission_measured_utc"],
              "project_bytes_before": initial["project_bytes_before"],
              "project_free_bytes_before": current["project_free_bytes_before"],
              "project_free_inodes_before": current["project_free_inodes_before"],
              "required_free_inodes": required_inode_count,
              "manifest_sha256": TRANSFER_SHA256, "candidate_sha256": CANDIDATE_SHA256,
              "file_count": EXPECTED_COUNT, "transfer_bytes": EXPECTED_TRANSFER_BYTES})
        raw_tar = read_exact(sys.stdin.buffer, archive_size)
        if sys.stdin.buffer.read(1):
            raise ValueError("tar stream contains trailing bytes")
        contents = validate_tar(raw_tar, expected)
        before_write = measure(project_fd)
        if (before_write["project_bytes_before"] != initial["project_bytes_before"]
                or before_write["e165_bytes_before"] != initial["e165_bytes_before"]):
            raise ValueError("project usage changed before the first destination write")
        if before_write["project_free_inodes_before"] < required_inode_count:
            raise ValueError("free inode count changed below admission")
        receipts = []
        for name, descriptor in expected.items():
            data = contents[name]
            write_exclusive(project_fd, descriptor["parts"], data)
            verify_written(project_fd, descriptor["parts"], descriptor["bytes"], descriptor["sha256"])
            receipts.append({"path": descriptor["parts"] and PROJECT_ROOT + "/" + name,
                             "bytes": descriptor["bytes"], "sha256": descriptor["sha256"]})
        after_files = measure(project_fd)
        if after_files["project_bytes_before"] != initial["project_bytes_before"]:
            raise ValueError("non-E165 baseline changed during transfer")
        if after_files["project_bytes_before"] + E165_ALLOWANCE_BYTES > PROJECT_CAP_BYTES:
            raise ValueError("project cap no longer fits the full E165 allowance")
        if after_files["project_free_bytes_before"] < after_files["e165_allowance_remaining_bytes"]:
            raise ValueError("free bytes do not fit the remaining E165 allowance")
        if after_files["e165_bytes_before"] > E165_ALLOWANCE_BYTES:
            raise ValueError("E165 transfer exceeded its 128 MiB allowance")
        if after_files["project_free_inodes_before"] < POST_TRANSFER_FREE_INODE_FLOOR + 1:
            raise ValueError("free inode count is too low for the stage spec and remaining work")
        final_spec_sha = digest(final_spec_raw)
        before_spec = measure(project_fd)
        remaining_after_spec = E165_ALLOWANCE_BYTES - before_spec["e165_bytes_before"] - len(final_spec_raw)
        if remaining_after_spec < 0 or before_spec["project_bytes_before"] + E165_ALLOWANCE_BYTES > PROJECT_CAP_BYTES:
            raise ValueError("stage spec does not fit the project or E165 allowance")
        if before_spec["project_free_bytes_before"] < len(final_spec_raw) + remaining_after_spec:
            raise ValueError("free bytes do not fit the stage spec and remaining allowance")
        if before_spec["project_free_inodes_before"] < POST_TRANSFER_FREE_INODE_FLOOR + 1:
            raise ValueError("free inodes do not fit the stage spec and reserve")
        write_exclusive(project_fd, spec_parts, final_spec_raw)
        verify_written(project_fd, spec_parts, len(final_spec_raw), final_spec_sha)
        final_measure = measure(project_fd)
        if final_measure["project_bytes_before"] != initial["project_bytes_before"]:
            raise ValueError("non-E165 baseline changed before final receipt")
        if final_measure["project_bytes_before"] + final_measure["e165_bytes_before"] > PROJECT_CAP_BYTES:
            raise ValueError("aggregate project cap exceeded after transfer")
        if final_measure["e165_bytes_before"] > E165_ALLOWANCE_BYTES:
            raise ValueError("E165 allowance exceeded after transfer")
        if final_measure["project_free_inodes_before"] < POST_TRANSFER_FREE_INODE_FLOOR:
            raise ValueError("free inode count fell below the post-transfer floor")
        send({
            "type": "DONE",
            "state": "PREPARED_FOR_COORDINATOR_REVIEW",
            "remote_account": "robot",
            "preparation_directive": "E165_CHECKS_PREPARE001_20261028_APPROVED",
            "transfer_manifest_sha256": TRANSFER_SHA256,
            "frozen_candidate_sha256": CANDIDATE_SHA256,
            "scientific_source_identity_sha256": SOURCE_IDENTITY_SHA256,
            "scientific_manifest_sha256": SCIENTIFIC_MANIFEST_SHA256,
            "file_count": len(receipts), "transfer_bytes": EXPECTED_TRANSFER_BYTES,
            "files": receipts,
            "stage_spec_path": STAGE_SPEC_PATH,
            "stage_spec_bytes": len(final_spec_raw),
            "stage_spec_sha256": final_spec_sha,
            "resource_authorized": True,
            "execution_authorized": False,
            "execution_authorization_file_created": False,
            "stage_execution_started": False,
            "checks_started": False,
            "project_bytes_before": initial["project_bytes_before"],
            "admission_measured_utc": initial["admission_measured_utc"],
            "project_total_bytes_before": initial["project_total_bytes_before"],
            "e165_bytes_before": initial["e165_bytes_before"],
            "e165_bytes_after": final_measure["e165_bytes_before"],
            "aggregate_project_bytes_after": final_measure["project_total_bytes_before"],
            "project_free_bytes_before": initial["project_free_bytes_before"],
            "project_free_bytes_after": final_measure["project_free_bytes_before"],
            "project_free_inodes_before": initial["project_free_inodes_before"],
            "project_free_inodes_after": final_measure["project_free_inodes_before"],
            "e165_allowance_bytes": E165_ALLOWANCE_BYTES,
            "project_cap_bytes": PROJECT_CAP_BYTES,
            "tar_bytes": archive_size,
        })
    finally:
        os.close(project_fd)


try:
    main()
except BaseException as exc:
    try:
        send({"type": "ERROR", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
    except BaseException:
        pass
    sys.exit(2)
'''


class PreparationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_reparse_point(path: Path) -> bool:
    info = path.lstat()
    attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_no_reparse_components(path: Path, *, root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(resolved_root)
    except ValueError as exc:
        raise PreparationError("manifest source is outside this workspace") from exc
    current = resolved_root
    if _is_reparse_point(current):
        raise PreparationError("workspace root is a reparse point")
    for component in relative.parts:
        current = current / component
        if _is_reparse_point(current):
            raise PreparationError("manifest source contains a symlink or reparse point")


def validate_local_inputs() -> tuple[bytes, bytes, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if os.name != "nt":
        raise PreparationError("run this helper from Windows PowerShell with native OpenSSH")
    if not ROOT.is_dir() or not HERE.is_dir():
        raise PreparationError("helper workspace paths are unavailable")
    for controlled_path in (MANIFEST_PATH, FROZEN_CANDIDATE_PATH, COORDINATOR_PREP_PATH):
        _assert_no_reparse_components(controlled_path, root=ROOT)
    manifest_raw = MANIFEST_PATH.read_bytes()
    candidate_raw = FROZEN_CANDIDATE_PATH.read_bytes()
    coordinator_raw = COORDINATOR_PREP_PATH.read_bytes()
    if sha256_bytes(manifest_raw) != TRANSFER_MANIFEST_SHA256:
        raise PreparationError("approved transfer manifest SHA-256 mismatch")
    if sha256_bytes(candidate_raw) != FROZEN_CANDIDATE_SHA256:
        raise PreparationError("frozen CHECKS candidate SHA-256 mismatch")
    manifest = json.loads(manifest_raw)
    candidate = json.loads(candidate_raw)
    coordinator = json.loads(coordinator_raw)
    if coordinator.get("directive") != PREPARATION_DIRECTIVE:
        raise PreparationError("coordinator preparation directive does not match")
    if coordinator.get("remote_account") != "robot" or coordinator.get("project_root") != PROJECT_ROOT:
        raise PreparationError("coordinator account or project root does not match")
    if coordinator.get("new_work_root") != E_ROOT:
        raise PreparationError("coordinator E165 path does not match")
    if coordinator.get("checks_transfer_manifest_sha256") != TRANSFER_MANIFEST_SHA256:
        raise PreparationError("coordinator did not approve this exact transfer manifest")
    if coordinator.get("checks_frozen_candidate_sha256") != FROZEN_CANDIDATE_SHA256:
        raise PreparationError("coordinator did not approve this exact frozen candidate")
    if coordinator.get("scientific_source_identity_sha256") != SCIENTIFIC_SOURCE_IDENTITY_SHA256:
        raise PreparationError("coordinator scientific source identity mismatch")
    if coordinator.get("scientific_manifest_sha256") != SCIENTIFIC_MANIFEST_SHA256:
        raise PreparationError("coordinator scientific manifest SHA-256 mismatch")
    limits = coordinator.get("round_limits", {})
    if (limits.get("new_artifact_cache_temp_bytes") != E165_ALLOWANCE_BYTES
            or limits.get("aggregate_project_bytes") != PROJECT_CAP_BYTES):
        raise PreparationError("coordinator capacity limits do not match")
    if coordinator.get("resource_authorized") is not True or coordinator.get("execution_authorized") is not False:
        raise PreparationError("coordinator authorization state differs from preparation-only scope")
    if manifest.get("file_count") != TRANSFER_FILE_COUNT or manifest.get("total_bytes") != TRANSFER_BYTES:
        raise PreparationError("approved transfer count or byte total mismatch")
    if manifest.get("stage_spec_path_after_admission") != STAGE_SPEC_PATH:
        raise PreparationError("manifest stage-spec destination does not match")
    items = manifest.get("files")
    if not isinstance(items, list) or len(items) != TRANSFER_FILE_COUNT:
        raise PreparationError("manifest must contain exactly 42 files")
    if candidate.get("stage") != "CHECKS" or candidate.get("resource_authorized") is not False:
        raise PreparationError("frozen CHECKS candidate state mismatch")
    if candidate.get("project_bytes_before") is not None or candidate.get("admission_measured_utc") is not None:
        raise PreparationError("frozen candidate already contains an admission measurement")
    worker_argv = candidate.get("worker_argv")
    if (not isinstance(worker_argv, list) or len(worker_argv) != 6
            or worker_argv[3] != STAGE_SPEC_PATH
            or worker_argv[5] != "/data/robot/p001/e165/o001/authorization_checks.json"):
        raise PreparationError("frozen CHECKS worker arguments do not bind the approved remote paths")
    manifest_descriptors: dict[str, dict[str, Any]] = {}
    clean_items: list[dict[str, Any]] = []
    total = 0
    for raw_item in items:
        if not isinstance(raw_item, dict) or set(raw_item) != {"local_path", "destination", "bytes", "sha256"}:
            raise PreparationError("manifest row has unexpected fields")
        destination = raw_item["destination"]
        pure_destination = PurePosixPath(destination)
        if (not isinstance(destination, str) or "\\" in destination or "//" in destination
                or pure_destination.as_posix() != destination or ".." in pure_destination.parts
                or not destination.startswith(E_ROOT + "/")):
            raise PreparationError("manifest destination is not a strict path under E165")
        parts = pure_destination.parts
        if len(parts) < 7 or parts[:5] != ("/", "data", "robot", "p001", "e165"):
            raise PreparationError("manifest destination is outside the exact project path")
        if parts[5] not in ("s001", "o001"):
            raise PreparationError("manifest destination is outside s001 and o001")
        if any(part.lower() in (".git", "agent.md", "agents.md") for part in parts):
            raise PreparationError("manifest includes a forbidden instruction or Git path")
        suffix = PurePosixPath(destination).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise PreparationError("manifest includes a non-source or model-like file type")
        size = raw_item["bytes"]
        digest = raw_item["sha256"]
        if type(size) is not int or size < 0 or not isinstance(digest, str) or len(digest) != 64:
            raise PreparationError("manifest byte length or SHA-256 is malformed")
        if any(char not in "0123456789abcdef" for char in digest):
            raise PreparationError("manifest SHA-256 encoding is malformed")
        if destination in manifest_descriptors:
            raise PreparationError("manifest has duplicate destination paths")
        local_raw = raw_item["local_path"]
        if not isinstance(local_raw, str):
            raise PreparationError("manifest local path is malformed")
        local_windows = PureWindowsPath(local_raw)
        if not local_windows.is_absolute():
            raise PreparationError("manifest local path must be absolute")
        local_path = Path(local_raw)
        _assert_no_reparse_components(local_path, root=ROOT)
        resolved_local = local_path.resolve(strict=True)
        if not resolved_local.is_file():
            raise PreparationError("manifest source is not a regular file")
        info = resolved_local.stat()
        if not stat.S_ISREG(info.st_mode):
            raise PreparationError("manifest source is not a regular file")
        hasher = hashlib.sha256()
        observed = 0
        with resolved_local.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                hasher.update(chunk)
        if observed != size or hasher.hexdigest() != digest:
            raise PreparationError("local source bytes differ from the approved manifest")
        normalized = {"local_path": str(resolved_local), "destination": destination,
                      "bytes": size, "sha256": digest}
        clean_items.append(normalized)
        manifest_descriptors[destination] = {"bytes": size, "sha256": digest}
        total += size
    if total != TRANSFER_BYTES:
        raise PreparationError("manifest source sizes do not equal the approved total")
    candidate_rows = candidate.get("source_files", []) + candidate.get("orchestration_files", [])
    candidate_descriptors = {row.get("path"): {"bytes": row.get("bytes"), "sha256": row.get("sha256")}
                             for row in candidate_rows if isinstance(row, dict)}
    if candidate_descriptors != manifest_descriptors:
        raise PreparationError("frozen candidate descriptors differ from the transfer manifest")
    return manifest_raw, candidate_raw, manifest, candidate, clean_items


def build_tar(items: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for item in items:
            path = Path(item["local_path"])
            info = tarfile.TarInfo(name=item["destination"][len(PROJECT_ROOT) + 1:])
            info.size = item["bytes"]
            info.mode = 0o600
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            hasher = hashlib.sha256()
            data = bytearray()
            with path.open("rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    data.extend(chunk)
                    hasher.update(chunk)
            if len(data) != item["bytes"] or hasher.hexdigest() != item["sha256"]:
                raise PreparationError("local source changed after manifest validation")
            archive.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    if len(raw) > MAX_TAR_BYTES:
        raise PreparationError("in-memory archive exceeds its strict size bound")
    return raw


def _reader_thread(stream: Any, output: queue.Queue[bytes | None]) -> None:
    try:
        while True:
            line = stream.readline(MAX_PROTOCOL_LINE_BYTES + 1)
            if not line:
                break
            output.put(line)
    finally:
        output.put(None)


def _read_record(output: queue.Queue[bytes | None], timeout: float) -> dict[str, Any]:
    try:
        line = output.get(timeout=timeout)
    except queue.Empty as exc:
        raise PreparationError("native SSH did not return the required remote protocol frame in time") from exc
    if line is None:
        raise PreparationError("SSH ended before the required remote protocol frame")
    if len(line) > MAX_PROTOCOL_LINE_BYTES:
        raise PreparationError("remote protocol frame exceeds its size bound")
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError("remote stdout was not a structured JSON protocol frame") from exc
    if not isinstance(record, dict):
        raise PreparationError("remote protocol frame is not a JSON object")
    if record.get("type") == "ERROR":
        raise PreparationError("remote preparation stopped: " + str(record.get("error", "unknown remote error")))
    return record


def _build_ssh_command(ssh: str) -> list[str]:
    compressed = zlib.compress(REMOTE_SCRIPT.encode("utf-8"), level=9)
    encoded = base64.b64encode(compressed).decode("ascii")
    fixed_args = [
        ssh, "-T", "-S", "none",
        "-o", "BatchMode=no",
        "-o", "NumberOfPasswordPrompts=1",
        "-o", "RequestTTY=no",
        "-o", "RemoteCommand=none",
        "-o", "PermitLocalCommand=no",
        "-o", "ClearAllForwardings=yes",
        "-o", "ForwardAgent=no",
        "-o", "ForwardX11=no",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "UpdateHostKeys=no",
        REMOTE_USER_HOST,
    ]
    # Reserve room for quoting, the remote command wrapper and Windows argv
    # escaping before constructing the final CreateProcess command line.
    conservative_length = sum(len(arg) + 4 for arg in fixed_args) + len(encoded) + 512
    if conservative_length >= WINDOWS_COMMAND_LINE_SAFE_MAX:
        raise PreparationError("compressed remote command exceeds the safe Windows command-line bound")
    code = "exec(__import__('zlib').decompress(__import__('base64').b64decode('" + encoded + "')))"
    args = fixed_args + ["python3 -c " + shlex.quote(code)]
    command_line = subprocess.list2cmdline(args)
    if len(command_line) + 1 >= WINDOWS_COMMAND_LINE_SAFE_MAX:
        raise PreparationError("final OpenSSH command line exceeds the safe Windows command-line bound")
    return args


def _terminate_child(process: subprocess.Popen[bytes]) -> None:
    if process.stdin and not process.stdin.closed:
        try:
            process.stdin.close()
        except OSError:
            pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def _save_receipt(receipt: dict[str, Any]) -> Path:
    raw = (json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        tail = "" if suffix == 0 else f"_{suffix:03d}"
        path = HERE / f"interactive_prepare_receipt_{stamp}{tail}.json"
        try:
            with path.open("xb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            return path
        except FileExistsError:
            continue
    raise PreparationError("could not create a unique local receipt without overwriting evidence")


def run() -> int:
    receipt: dict[str, Any] = {
        "format": "e165_interactive_preparation_receipt_v1",
        "started_utc": utc_now(),
        "preparation_directive": PREPARATION_DIRECTIVE,
        "target": REMOTE_USER_HOST,
        "project_root": PROJECT_ROOT,
        "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA256,
        "frozen_candidate_sha256": FROZEN_CANDIDATE_SHA256,
        "source_identity_sha256": SCIENTIFIC_SOURCE_IDENTITY_SHA256,
        "execution_authorization_file_created": False,
        "checks_started": False,
        "archive_sent": False,
        "state": "NOT_STARTED",
    }
    process: subprocess.Popen[bytes] | None = None
    reader: threading.Thread | None = None
    records: queue.Queue[bytes | None] | None = None
    try:
        manifest_raw, candidate_raw, _manifest, candidate, items = validate_local_inputs()
        receipt["state"] = "LOCAL_INPUTS_VALIDATED"
        receipt["transfer_file_count"] = len(items)
        receipt["transfer_bytes"] = sum(item["bytes"] for item in items)
        ssh = shutil.which("ssh.exe") or shutil.which("ssh")
        if not ssh:
            raise PreparationError("native OpenSSH ssh.exe was not found on PATH")
        command = _build_ssh_command(ssh)
        child_env = {}
        for name in ("PATH", "SystemRoot", "WINDIR", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
                     "HOME", "SSH_AUTH_SOCK", "SSH_AGENT_PID", "TEMP", "TMP", "LANG", "LC_ALL", "TERM"):
            value = os.environ.get(name)
            if value is not None:
                child_env[name] = value
        print("Starting one native OpenSSH session. If SSH requests authentication, enter it only in the native SSH console prompt.")
        print("Python will not read the PowerShell console; SSH stderr remains attached to that console. No file bytes are sent before remote capacity READY.")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=None, shell=False, close_fds=True, env=child_env)
        assert process.stdout is not None and process.stdin is not None
        records = queue.Queue()
        reader = threading.Thread(target=_reader_thread, args=(process.stdout, records), daemon=True)
        reader.start()
        ready = _read_record(records, SSH_READY_TIMEOUT_SECONDS)
        if ready.get("type") != "READY":
            raise PreparationError("remote did not complete capacity and inode admission")
        if (ready.get("project_cap_bytes") != PROJECT_CAP_BYTES
                or ready.get("e165_allowance_bytes") != E165_ALLOWANCE_BYTES
                or ready.get("project_bytes_before", -1) + E165_ALLOWANCE_BYTES > PROJECT_CAP_BYTES
                or ready.get("e165_bytes_before", E165_ALLOWANCE_BYTES + 1) > E165_ALLOWANCE_BYTES
                or ready.get("e165_allowance_remaining_bytes") != E165_ALLOWANCE_BYTES - ready.get("e165_bytes_before", 0)
                or ready.get("project_free_bytes_before", 0) < ready.get("e165_allowance_remaining_bytes", E165_ALLOWANCE_BYTES)
                or ready.get("project_free_inodes_before", 0) < INITIAL_FREE_INODE_FLOOR):
            raise PreparationError("remote READY did not satisfy the exact storage admission bounds")
        receipt["remote_ready"] = ready
        receipt["state"] = "REMOTE_CAPACITY_READY"
        archive = build_tar(items)
        dynamic_spec = dict(candidate)
        dynamic_spec["resource_authorized"] = True
        dynamic_spec["project_bytes_before"] = ready["project_bytes_before"]
        dynamic_spec["admission_measured_utc"] = ready["admission_measured_utc"]
        final_spec_raw = (json.dumps(dynamic_spec, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        final_spec_sha = sha256_bytes(final_spec_raw)
        envelope = {
            "manifest_sha256": TRANSFER_MANIFEST_SHA256,
            "manifest_b64": base64.b64encode(manifest_raw).decode("ascii"),
            "candidate_sha256": FROZEN_CANDIDATE_SHA256,
            "scientific_source_identity_sha256": SCIENTIFIC_SOURCE_IDENTITY_SHA256,
            "scientific_manifest_sha256": SCIENTIFIC_MANIFEST_SHA256,
            "files": [{"path": item["destination"], "bytes": item["bytes"], "sha256": item["sha256"]}
                      for item in items],
            "final_spec_b64": base64.b64encode(final_spec_raw).decode("ascii"),
            "final_spec_sha256": final_spec_sha,
            "tar_bytes": len(archive),
        }
        metadata = (json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        if len(metadata) > MAX_METADATA_BYTES:
            raise PreparationError("manifest metadata exceeds the strict protocol bound")
        process.stdin.write(len(metadata).to_bytes(4, "big"))
        process.stdin.write(metadata)
        process.stdin.flush()
        transfer_ready = _read_record(records, SSH_READY_TIMEOUT_SECONDS)
        if transfer_ready.get("type") != "TRANSFER_READY":
            raise PreparationError("remote did not revalidate storage and destinations before transfer")
        if (transfer_ready.get("admission_measured_utc") != ready.get("admission_measured_utc")
                or transfer_ready.get("project_bytes_before") != ready.get("project_bytes_before")
                or transfer_ready.get("manifest_sha256") != TRANSFER_MANIFEST_SHA256
                or transfer_ready.get("candidate_sha256") != FROZEN_CANDIDATE_SHA256
                or transfer_ready.get("file_count") != TRANSFER_FILE_COUNT
                or transfer_ready.get("transfer_bytes") != TRANSFER_BYTES
                or transfer_ready.get("project_free_inodes_before", 0) < transfer_ready.get("required_free_inodes", INITIAL_FREE_INODE_FLOOR)):
            raise PreparationError("remote transfer READY did not bind the approved manifest and admitted capacity")
        receipt["transfer_ready"] = transfer_ready
        receipt["dynamic_stage_spec_sha256"] = final_spec_sha
        receipt["state"] = "TRANSFER_READY"
        receipt["archive_send_started"] = True
        process.stdin.write(archive)
        process.stdin.flush()
        process.stdin.close()
        receipt["archive_sent"] = True
        receipt["tar_bytes"] = len(archive)
        del archive
        done = _read_record(records, SSH_COMPLETION_TIMEOUT_SECONDS)
        if done.get("type") != "DONE":
            raise PreparationError("remote did not return a completed structured receipt")
        receipt["remote_receipt"] = done
        receipt["state"] = done.get("state", "REMOTE_COMPLETED")
        try:
            exit_code = process.wait(timeout=SSH_COMPLETION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise PreparationError("SSH did not exit after the remote receipt") from exc
        receipt["ssh_exit_code"] = exit_code
        if exit_code != 0:
            raise PreparationError("SSH returned a nonzero exit code after the remote receipt")
        if (done.get("resource_authorized") is not True
                or done.get("execution_authorized") is not False
                or done.get("execution_authorization_file_created") is not False
                or done.get("checks_started") is not False
                or done.get("stage_execution_started") is not False
                or done.get("stage_spec_path") != STAGE_SPEC_PATH
                or done.get("stage_spec_sha256") != receipt.get("dynamic_stage_spec_sha256")
                or done.get("stage_spec_bytes", 0) <= 0
                or done.get("file_count") != TRANSFER_FILE_COUNT
                or done.get("transfer_bytes") != TRANSFER_BYTES
                or done.get("transfer_manifest_sha256") != TRANSFER_MANIFEST_SHA256
                or done.get("frozen_candidate_sha256") != FROZEN_CANDIDATE_SHA256
                or done.get("scientific_source_identity_sha256") != SCIENTIFIC_SOURCE_IDENTITY_SHA256
                or done.get("scientific_manifest_sha256") != SCIENTIFIC_MANIFEST_SHA256
                or done.get("project_bytes_before") != ready.get("project_bytes_before")
                or done.get("admission_measured_utc") != ready.get("admission_measured_utc")
                or done.get("aggregate_project_bytes_after", PROJECT_CAP_BYTES + 1) > PROJECT_CAP_BYTES
                or done.get("e165_bytes_after", E165_ALLOWANCE_BYTES + 1) > E165_ALLOWANCE_BYTES
                or done.get("project_free_bytes_after", 0) < max(0, E165_ALLOWANCE_BYTES - done.get("e165_bytes_after", E165_ALLOWANCE_BYTES))
                or done.get("project_free_inodes_after", 0) < POST_TRANSFER_FREE_INODE_FLOOR):
            raise PreparationError("remote receipt does not match preparation-only acceptance conditions")
        expected_files = {item["destination"]: {"bytes": item["bytes"], "sha256": item["sha256"]} for item in items}
        observed_files = {row.get("path"): {"bytes": row.get("bytes"), "sha256": row.get("sha256")}
                          for row in done.get("files", []) if isinstance(row, dict)}
        if observed_files != expected_files:
            raise PreparationError("remote receipt file hashes differ from the approved 42-file manifest")
        receipt["state"] = "PREPARED_FOR_COORDINATOR_REVIEW"
        receipt["completed_utc"] = utc_now()
        receipt_path = _save_receipt(receipt)
        print("Preparation completed. No CHECKS code, test, model, or supervisor was run.")
        print(f"Local receipt: {receipt_path}")
        print(f"Remote CHECKS spec SHA-256: {done['stage_spec_sha256']}")
        return 0
    except KeyboardInterrupt:
        receipt["state"] = "INTERRUPTED"
        receipt["error"] = "user interrupted the local helper"
        print("Interrupted. No password was read by Python; no automatic retry will occur.", file=sys.stderr)
        return_code = 130
    except Exception as exc:
        receipt["state"] = "FAILED_CLOSED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Preparation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not receipt.get("archive_send_started"):
            print("No 42-file archive was sent. Preserve any remote evidence; do not retry automatically.", file=sys.stderr)
        else:
            print("The archive was sent. Any remote partial evidence is preserved; do not retry automatically.", file=sys.stderr)
        return_code = 1
    finally:
        if process is not None and process.poll() is None:
            receipt["ssh_terminated_by_helper"] = True
            _terminate_child(process)
        if process is not None:
            receipt.setdefault("ssh_exit_code", process.poll())
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
    if "receipt_path" not in locals():
        receipt["completed_utc"] = utc_now()
        try:
            receipt_path = _save_receipt(receipt)
            print(f"Local receipt: {receipt_path}")
        except Exception as save_exc:
            print(f"Could not save a local receipt: {type(save_exc).__name__}: {save_exc}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(run())

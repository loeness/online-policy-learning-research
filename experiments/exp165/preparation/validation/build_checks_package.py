"""Build an inert CHECKS review package from frozen bytes; no research imports.

This is a local static manifest/JSON operation. It never connects remotely,
executes tests, grants resources, or replaces an earlier generated package.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DELIVERY = ROOT / "research/appendix/H036/delivery_budgeted_v1"
REMOTE = "/data/robot/p001/e165"
CONTROLS = ("supervisor.py", "phase_common.py", "checks_entry.py",
            "launch_checks.sh", "pytest.ini")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def descriptor(path: Path, destination: str) -> dict:
    raw = path.read_bytes()
    return {"path": destination, "bytes": len(raw), "sha256": digest(raw)}


def main() -> None:
    outputs = (HERE / "checks_frozen_candidate.json", HERE / "checks_transfer_manifest.json")
    if any(p.exists() for p in outputs):
        raise FileExistsError("Preserve existing preparation evidence; version any repair")
    manifest_path = DELIVERY / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest["state"] != "SOURCE_FROZEN_REMOTE_CHECKS_PENDING":
        raise ValueError("Scientific source is not in the required frozen state")
    rows = manifest["snapshot_files"]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    if rows != sorted(rows, key=lambda row: row["path"]) or digest(canonical) != manifest["source_identity_sha256"]:
        raise ValueError("Scientific snapshot identity is inconsistent")
    if len(rows) != 36 or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Unexpected scientific file count or duplicate")
    snapshot = DELIVERY / "source_snapshot"
    transfer = []
    source = []
    for row in rows:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts or relative.name.lower() in {"agent.md", "agents.md"}:
            raise ValueError("Forbidden source path")
        local = snapshot.joinpath(*relative.parts)
        if local.is_symlink() or not local.resolve().is_relative_to(snapshot.resolve()):
            raise ValueError("Snapshot path escapes its root")
        target = f"{REMOTE}/s001/{relative.as_posix()}"
        item = descriptor(local, target)
        if item["bytes"] != row["bytes"] or item["sha256"] != row["sha256"]:
            raise ValueError("Frozen source bytes differ: " + str(relative))
        source.append(item)
        transfer.append({"local_path": str(local), "destination": target,
                         "bytes": item["bytes"], "sha256": item["sha256"]})
    manifest_target = f"{REMOTE}/s001/delivery_manifest.json"
    manifest_desc = descriptor(manifest_path, manifest_target)
    source.append(manifest_desc)
    transfer.append({"local_path": str(manifest_path), "destination": manifest_target,
                     "bytes": manifest_desc["bytes"], "sha256": manifest_desc["sha256"]})
    controls = []
    for name in CONTROLS:
        local = HERE / name
        target = f"{REMOTE}/o001/{name}"
        item = descriptor(local, target)
        controls.append(item)
        transfer.append({"local_path": str(local), "destination": target,
                         "bytes": item["bytes"], "sha256": item["sha256"]})
    by_relative = {row["path"]: row for row in rows}
    checks_rel = manifest["checks_spec"]["path"]
    checks_path = snapshot.joinpath(*PurePosixPath(checks_rel).parts)
    checks = json.loads(checks_path.read_bytes())
    if by_relative[checks_rel]["sha256"] != manifest["checks_spec"]["sha256"]:
        raise ValueError("Checks spec is not the frozen scientific checks spec")
    if checks["expected_count"] != 9 or len(checks["expected_test_nodeids"]) != 9:
        raise ValueError("Unexpected frozen CHECKS node count")
    lock = (snapshot / "requirements-lock.txt").read_text(encoding="utf-8")
    packages = {}
    for line in lock.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        packages[name] = version
    if len(packages) != 33:
        raise ValueError("Expected exactly 33 pinned dependencies")
    spec = json.loads((HERE / "stage_checks.template.json").read_bytes())
    spec.update({
        "preparation_state": "FROZEN",
        "resource_authorized": False,
        "source_identity_sha256": manifest["source_identity_sha256"],
        "scientific_manifest": manifest_desc,
        "anchor_manifest": next(row for row in source if row["path"] == f"{REMOTE}/s001/{manifest['anchor_manifest']['path']}"),
        "source_files": source,
        "orchestration_files": controls,
        "expected_packages": packages,
        "expected_test_nodeids": checks["expected_test_nodeids"],
        "pytest_target": [f"{REMOTE}/s001/{checks['test_file']}"],
        "pytest_config": f"{REMOTE}/o001/pytest.ini",
        "synthetic_call_envelope": checks["synthetic_call_envelope"],
        "mock_call_declaration": checks["mock_call_declaration"],
        "real_call_limits": checks["real_call_limits"],
        "project_bytes_before": None,
        "admission_measured_utc": None,
        "prior_stage_states": {},
    })
    package = {
        "format": "e165_checks_transfer_candidate_v1",
        "state": "REVIEW_ONLY_NO_REMOTE_AUTHORIZATION",
        "scientific_source_identity_sha256": manifest["source_identity_sha256"],
        "scientific_manifest_sha256": digest(manifest_raw),
        "stage_spec_path_after_admission": f"{REMOTE}/o001/stage_checks.json",
        "stage_spec_is_dynamic": "Resource flag and measured project baseline require coordinator directive and new exact SHA.",
        "files": transfer,
        "total_bytes": sum(item["bytes"] for item in transfer),
        "file_count": len(transfer),
        "remote_references": [],
        "remote_operations_authorized": False,
    }
    for out, value in zip(outputs, (spec, package)):
        out.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

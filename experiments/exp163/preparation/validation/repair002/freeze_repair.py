"""Freeze the local-only E163 launcher repair; never connect or execute it."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

HERE = Path(__file__).resolve().parent
OLD = HERE.parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    spec = json.loads((OLD / "run_spec.json").read_text())
    old_freeze = json.loads((OLD / "execution_freeze001.json").read_text())
    rows = []
    assert b"\r" not in (HERE / "launch_command.sh").read_bytes(), "POSIX launcher must use LF bytes"
    for name in ("runner.py", "supervisor.py", "gate.py", "gate_mapping.json", "launch_command.sh"):
        data = (HERE / name).read_bytes()
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8"))
        rows.append({"path": name, "bytes": len(data), "sha256": sha(data)})
    for name in ("gate.py", "gate_mapping.json"):
        assert (HERE/name).read_bytes() == (OLD/name).read_bytes()
    spec.update({
        "format": "e163_execution_spec_v2_local_repair",
        "orchestration_root": "/data/robot/p001/e163/o002",
        "run_root": "/data/robot/p001/e163/run002",
        "worker_argv": ["-B", "/data/robot/p001/e163/o002/runner.py", "--spec",
                        "/data/robot/p001/e163/o002/run_spec.json", "--authorization",
                        "/data/robot/p001/e163/o002/authorization.json"],
        "orchestration_files": rows[:4],
        "launcher_file": rows[4],
        "repair_of_run_spec_sha256": old_freeze["run_spec_sha256"],
        "fresh_process_contract": "launch_command.sh forks a new dedicated supervisor and explicitly waits; supervisor verifies known parent PID and zero initial waited-child usage. No inherited-counter subtraction.",
        "admission_status": "LOCAL_DRAFT_ONLY; historic non-e163 baseline retained; fresh remote admission and new resource authorization required before any execution",
    })
    dump(HERE / "run_spec.json", spec)
    spec_bytes = (HERE / "run_spec.json").read_bytes()
    auth = {"experiment_id": "E163", "execution_authorized": False, "limits": spec["limits"],
            "coordinator_directive": "", "run_spec_sha256": sha(spec_bytes)}
    dump(HERE / "authorization.template.json", auth)
    # Reuse exactly the original scientific manifest bytes; no scientific snapshot copy.
    online = OLD.parents[3]
    mbytes = (online / "research/appendix/H036/delivery_v1/manifest.json").read_bytes()
    assert sha(mbytes) == old_freeze["scientific_manifest_sha256"]
    files = {"o002/"+row["path"]: (HERE/row["path"]).read_bytes() for row in rows}
    files["o002/run_spec.json"] = spec_bytes
    files["o002/scientific_manifest.json"] = mbytes
    members = [{"path": name, "bytes": len(data), "sha256": sha(data)} for name,data in sorted(files.items())]
    dump(HERE / "transfer_manifest.json", {"state": "LOCAL_ONLY_NOT_TRANSFERRED", "destination_root": "/data/robot/p001/e163",
                                         "members": members, "member_count": len(members),
                                         "uncompressed_bytes": sum(r["bytes"] for r in members)})
    with zipfile.ZipFile(HERE / "repair002.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026,10,22,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    identity = sha(json.dumps(rows,sort_keys=True,separators=(",",":")).encode())
    freeze = {
        "state": "LOCAL_REPAIR_FROZEN_PENDING_INDEPENDENT_REVIEW_AND_COORDINATOR_RESOURCE_DECISION",
        "experiment_id": "E163", "repair": "002", "previous_attempt": "run001 ACCOUNTING_INVALID; retained unchanged",
        "scientific_source_identity_sha256": old_freeze["source_identity_sha256"],
        "scientific_manifest_sha256": sha(mbytes), "scientific_source_changed": False,
        "orchestration_identity_sha256": identity,
        "identity_algorithm": "SHA256 compact sorted-key JSON of ordered runner,supervisor,gate,mapping,launcher file rows; launcher included because launch lifecycle is the repair",
        "orchestration_files_and_launcher": rows, "run_spec_sha256": sha(spec_bytes),
        "archive_sha256": sha((HERE/"repair002.zip").read_bytes()),
        "archive_bytes": (HERE/"repair002.zip").stat().st_size,
        "remote_entry_argv": ["sh", "/data/robot/p001/e163/o002/launch_command.sh"],
        "supervisor_argv": [spec["python"], "-I", "-B", "/data/robot/p001/e163/o002/supervisor.py", "--spec",
                            "/data/robot/p001/e163/o002/run_spec.json", "--authorization",
                            "/data/robot/p001/e163/o002/authorization.json"],
        "source_cwd": spec["source_cwd"], "run_root": spec["run_root"], "environment": spec["environment"],
        "numeric_limits": spec["limits"], "numeric_limits_status": "UNCHANGED_METHOD_CAPS_ONLY; NO_NEW_RESERVATION_OR_EXECUTION_AUTHORITY",
        "input_files_unchanged": spec["input_files"] == json.loads((OLD/"run_spec.json").read_text())["input_files"],
        "transferred": False, "ssh_connections_for_repair": 0, "runtime_prechecks": 0, "real_fits": 0,
        "execution_authorized": False,
    }
    dump(HERE / "freeze.json", freeze)
    print(json.dumps({k:freeze[k] for k in ("state","orchestration_identity_sha256","run_spec_sha256","archive_sha256","archive_bytes","execution_authorized")}))


if __name__ == "__main__":
    main()

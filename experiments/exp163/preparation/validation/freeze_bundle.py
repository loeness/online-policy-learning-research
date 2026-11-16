"""Build an exact E163 source/orchestration transfer, without running science."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
DELIVERY = ROOT / "research/appendix/H036/delivery_v1"
REMOTE = "/data/robot/p001/e163"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def main():
    mbytes = (DELIVERY / "manifest.json").read_bytes()
    assert digest(mbytes) == "b837222aff57ab45abee226e9aa8bc9d2cd3444807fa348dee0c0c1ece70dc54"
    manifest = json.loads(mbytes)
    source = DELIVERY / "source_snapshot"
    source_files = [{k: row[k] for k in ("path", "bytes", "sha256")} for row in manifest["snapshot_files"]]
    content = {}
    for row in source_files:
        data = (source / row["path"]).read_bytes()
        assert len(data) == row["bytes"] and digest(data) == row["sha256"]
        content["s001/"+row["path"]] = data
    orch = []
    for name in ("runner.py", "supervisor.py", "gate.py", "gate_mapping.json"):
        data = (BASE / name).read_bytes()
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8"))
        orch.append({"path": name, "bytes": len(data), "sha256": digest(data)})
        content["o001/"+name] = data
    inputs = [{"seed": r["seed"], "file": r["file"], "bytes": r["bytes"], "sha256": r["sha256"],
               "remote_path": f"/data/robot/p001/e161/run001/producer/cases/s{r['seed']}_dynamics_late/prefix/{r['file']}"}
              for r in manifest["input_files"]]
    admission = json.loads((BASE / "admission002.json").read_text())
    environment = {
        **{k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        **{k: REMOTE+"/tmp" for k in ("TMPDIR", "TMP", "TEMP")},
        "XDG_CACHE_HOME": REMOTE+"/cache", "XDG_CONFIG_HOME": REMOTE+"/config",
        "XDG_DATA_HOME": REMOTE+"/state", "XDG_STATE_HOME": REMOTE+"/state",
        "MPLCONFIGDIR": REMOTE+"/cache/mpl", "TORCH_HOME": REMOTE+"/cache/torch",
        "TORCH_EXTENSIONS_DIR": REMOTE+"/cache/ext",
    }
    limits = {"cpu_seconds": 120, "campaign_wall_seconds": 180, "new_artifact_cache_temp_bytes": 268435456,
              "workers": 1, "numeric_threads_per_worker": 1, "gpu_devices": 0, "environment_interactions": 0,
              "candidate_fits": 32, "aggregate_remote_project_bytes": 100000000000}
    spec = {
        "experiment_id": "E163", "format": "e163_execution_spec_v1",
        "source_identity_sha256": manifest["source_identity_sha256"],
        "scientific_manifest_sha256": digest(mbytes), "git_parent": manifest["git_commit"],
        "project_root": "/data/robot/p001", "e_root": REMOTE,
        "orchestration_root": REMOTE+"/o001", "source_cwd": REMOTE+"/s001", "run_root": REMOTE+"/run001",
        "python": "/data/robot/p001/env/bin/python",
        "worker_argv": ["-B", REMOTE+"/o001/runner.py", "--spec", REMOTE+"/o001/run_spec.json",
                        "--authorization", REMOTE+"/o001/authorization.json"],
        "environment": environment, "limits": limits,
        "project_bytes_before": admission["project_bytes_before_non_e163"],
        "baseline_definition": "non-e163 project bytes; supervisor adds all e_root bytes including archive and sources",
        "admission_measured_utc": admission["measured_utc"],
        "seeds": list(range(26201,26214))+[26215,26216,26218], "lambda_values": [0,1],
        "source_files": source_files, "orchestration_files": orch, "input_files": inputs,
        "foundation": {"remote_path": "/data/robot/p001/s001/data/foundation.pt",
                       "bytes": manifest["foundation"]["bytes"], "sha256": manifest["foundation"]["sha256"]},
        "gate_mapping": json.loads((BASE / "gate_mapping.json").read_text()),
    }
    dump(BASE / "run_spec.json", spec)
    spec_data = (BASE / "run_spec.json").read_bytes()
    content["o001/run_spec.json"] = spec_data
    content["o001/scientific_manifest.json"] = mbytes
    members = [{"path": name, "bytes": len(data), "sha256": digest(data)} for name,data in sorted(content.items())]
    transfer = {"format": "E163_EXPLICIT_TRANSFER_MANIFEST_V1", "destination_root": REMOTE,
                "members": members, "member_count": len(members), "uncompressed_bytes": sum(r["bytes"] for r in members),
                "scientific_source_identity_sha256": manifest["source_identity_sha256"],
                "run_spec_sha256": digest(spec_data)}
    dump(BASE / "transfer_manifest.json", transfer)
    with zipfile.ZipFile(BASE / "transfer001.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(content.items()):
            info = zipfile.ZipInfo(name, (2026,10,22,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    bundle = (BASE / "transfer001.zip").read_bytes()
    capacity = {"source_and_orchestration_bytes": transfer["uncompressed_bytes"], "retained_archive_bytes": len(bundle),
                "planned_raw_deployment_upper_bytes": 32*262144,
                "planned_artifacts_logs_cache_temp_reserve_bytes": 96*1024*1024,
                "new_bytes_planning_upper": transfer["uncompressed_bytes"]+len(bundle)+32*262144+96*1024*1024,
                "project_bytes_before": spec["project_bytes_before"],
                "available_bytes_at_admission": admission["available_bytes"],
                "planning_note": "Generous write allowance, not an inference of measured runtime or a fit smoke. Fresh admission rechecked before launch."}
    assert capacity["new_bytes_planning_upper"] < limits["new_artifact_cache_temp_bytes"]
    assert spec["project_bytes_before"]+capacity["new_bytes_planning_upper"] < 100000000000
    dump(BASE / "capacity_plan.json", capacity)
    auth = {"experiment_id": "E163", "execution_authorized": False, "run_spec_sha256": digest(spec_data),
            "limits": limits, "coordinator_directive": ""}
    dump(BASE / "authorization.template.json", auth)
    command = [spec["python"], "-I", "-B", REMOTE+"/o001/supervisor.py", "--spec", REMOTE+"/o001/run_spec.json",
               "--authorization", REMOTE+"/o001/authorization.json"]
    freeze = {"state": "FROZEN_PENDING_REMOTE_TRANSFER_VERIFICATION_AND_COORDINATOR_EXECUTION_STATE",
              "experiment_id": "E163", "scientific_manifest_sha256": digest(mbytes),
              "source_identity_sha256": manifest["source_identity_sha256"], "git_parent": manifest["git_commit"],
              "orchestration_files": orch,
              "orchestration_identity_sha256": digest(json.dumps(orch,sort_keys=True,separators=(",",":")).encode()),
              "run_spec_sha256": digest(spec_data), "transfer_manifest_sha256": digest((BASE/"transfer_manifest.json").read_bytes()),
              "archive_sha256": digest(bundle), "archive_bytes": len(bundle), "source_files": len(source_files),
              "input_files": len(inputs), "campaign_argv": command, "campaign_cwd": spec["source_cwd"],
              "environment": environment, "authorization_file": REMOTE+"/o001/authorization.json",
              "authorization_file_current_state": "NOT_CREATED; execution_authorized remains false",
              "precheck_status": "NOT_RUN; will be inside the single authorized campaign", "fits": 0}
    dump(BASE / "execution_freeze001.json", freeze)
    launch = "set -eu\n. /data/robot/p001/enter.sh\n"
    launch += "\n".join("export "+key+"='"+val+"'" for key,val in environment.items())+"\n"
    launch += "cd '"+spec["source_cwd"]+"'\nexec "+" ".join("'"+arg+"'" for arg in command)+"\n"
    (BASE/"launch_command.sh").write_text(launch,encoding="utf-8")
    print(json.dumps({"members":len(members),"source_files":len(source_files),"archive_bytes":len(bundle),
                      "orchestration_identity_sha256":freeze["orchestration_identity_sha256"],
                      "run_spec_sha256":digest(spec_data),"fits":0,"precheck":"NOT_RUN"}))


if __name__ == "__main__":
    main()

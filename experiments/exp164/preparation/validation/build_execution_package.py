"""Build local metadata allowlists only; never import or run scientific code."""
from pathlib import Path, PurePosixPath
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DELIVERY = REPO / "research/appendix/H036/delivery_acceptance_v1"
SOURCE = DELIVERY / "source_snapshot"
REMOTE = "/data/robot/p001/e164"
SOURCE_ID = "baae9a6a766518105a6d5b2e9bc64fef3488ff581870931cf83ff2d7b23a4c9a"
MANIFEST_SHA = "76073211ec4ff37af2677e0d3cae33c87e46a703a25c306ccb0257c03a2ef379"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    (HERE / name).write_bytes((json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode())


def identify(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha(data)}


def collect(value, result):
    if isinstance(value, dict):
        key = value.get("key") or value.get("remote_path") or value.get("path")
        if "sha256" in value and "bytes" in value and key:
            if key in result:
                assert all(result[key][k] == value[k] for k in ("bytes", "sha256"))
            result[key] = value
        for item in value.values():
            collect(item, result)
    elif isinstance(value, list):
        for item in value:
            collect(item, result)


def main():
    raw_manifest = (DELIVERY / "manifest.json").read_bytes()
    assert sha(raw_manifest) == MANIFEST_SHA
    manifest = json.loads(raw_manifest)
    assert manifest["source_identity_sha256"] == SOURCE_ID
    source_files = [{k: row[k] for k in ("path", "bytes", "sha256")} for row in manifest["snapshot_files"]]
    assert len(source_files) == 27 and sum(row["bytes"] for row in source_files) == 762031
    source_paths = {row["path"] for row in source_files}
    assert not {"Agent.md", "AGENTS.md"}.intersection(source_paths)
    binding_relative = "research/appendix/H036/ACCEPTANCE_BINDINGS_V1.json"
    bindings_path = SOURCE / binding_relative
    bindings = json.loads(bindings_path.read_bytes())
    descriptors = {}
    collect(bindings, descriptors)
    reference_map, staged = {}, {}
    for key, descriptor in sorted(descriptors.items()):
        if descriptor.get("remote_path"):
            actual = descriptor["remote_path"]
        elif descriptor.get("path") in source_paths:
            actual = REMOTE + "/s001/" + descriptor["path"]
        else:
            # These are only the exact local provenance/mirror/v1-manifest JSONs.
            local = (REPO / descriptor["path"]).resolve(strict=True)
            assert local.is_relative_to(REPO) and local.suffix == ".json"
            assert identify(local) == {k: descriptor[k] for k in ("bytes", "sha256")}
            actual = REMOTE + "/i001/" + descriptor["sha256"] + ".json"
            staged[actual] = {"source_path": local.relative_to(REPO).as_posix(), "remote_target": actual,
                              **{k: descriptor[k] for k in ("bytes", "sha256")}, "role": "exact_provenance_json"}
        path = PurePosixPath(actual)
        assert path.is_relative_to(PurePosixPath("/data/robot/p001")) and ".." not in path.parts
        reference_map[key] = {"actual_path": actual, **{k: descriptor[k] for k in ("bytes", "sha256")}}
    write("reference_map.json", {"format": "e164_reference_map_v1", "references": reference_map,
          "reference_count": len(reference_map), "current_access_authorized": False,
          "usage": "Exact opaque integrity checks; only the five recovery prefixes are deserialized after admission."})
    methods = ["METHOD_ACCEPTANCE_V1.md", "HANDOFF_ACCEPTANCE_V1.md", "ACCEPTANCE_BINDINGS_V1.json"]
    write("source_binding.json", {"format": "e164_scientific_binding_v1", "state": "BOUND_FROZEN_SOURCE",
          "source_identity_sha256": SOURCE_ID, "scientific_manifest_sha256": MANIFEST_SHA,
          "methods": [{"path": name, **identify(SOURCE / "research/appendix/H036" / name)} for name in methods],
          "scientific_modules": ["online_rl.retention_acceptance_replay", "online_rl.retention_acceptance_diagnostic",
                                  "online_rl.retention_protected_inputs"],
          "api_schema_review": "api_binding_snapshot.json", "implementation_review": "frozen_implementation_review.json",
          "execution_or_resource_authorization": False})
    plan = json.loads((HERE / "execution_plan.json").read_bytes())
    plan["state"] = "BOUND_FROZEN_SOURCE_PENDING_RESOURCE_AUTHORIZATION"
    plan["scientific_api_mapping"] = "source_binding.json"
    write("execution_plan.json", plan)
    report_contract = json.loads((HERE / "report_contract.json").read_bytes())
    report_contract["state"] = "BOUND_FROZEN_SOURCE"
    report_contract["runner_binding_status_required_now"] = "BOUND_FROZEN_SOURCE"
    report_contract["binding_note"] = "Runner normalizes the accepted legacy score.metrics into this experiment-owned contract; resource and execution admission remain separate."
    write("report_contract.json", report_contract)
    controls = ["supervisor.py", "runner.py", "report.py", "launch_command.sh", "report_contract.json",
                "execution_plan.json", "source_binding.json", "reference_map.json"]
    orchestration_files = [{"path": name, **identify(HERE / name)} for name in controls]
    spec = json.loads((HERE / "run_spec.draft.json").read_bytes())
    spec.update({"format": "e164_execution_spec_v1", "preparation_state": "FROZEN", "resource_authorized": False,
                 "source_identity_sha256": SOURCE_ID, "scientific_manifest_sha256": MANIFEST_SHA,
                 "source_files": source_files, "orchestration_files": orchestration_files,
                 "scientific_manifest": {"path": REMOTE + "/i001/scientific_manifest.json", **identify(DELIVERY / "manifest.json")},
                 "bindings": {"path": REMOTE + "/s001/" + binding_relative, **identify(bindings_path)},
                 "reference_map": reference_map, "source_binding_status": "BOUND_FROZEN_SOURCE",
                 "output_contract_status": "BOUND_FROZEN_SOURCE", "expected_python_version": "3.12.15",
                 "project_bytes_before": None, "admission_measured_utc": None})
    spec["campaign_cost_scope"] = {
        "origin": "fresh supervisor process birth ticks",
        "included": "Python startup; in-campaign prechecks; worker preparation/fits/scoring/logging; stop/reap and reserved footer",
        "cpu": "lifetime supervisor RUSAGE_SELF plus direct-child wait4 including waited descendants; live owned counters during execution",
        "safety_stops": {"cpu_seconds": 55, "wall_seconds": 110},
        "finalization_reserve": {"cpu_seconds": 5, "wall_seconds": 10, "byte_reserve": 65536},
        "outside_origin": ["parent launch shell", "SSH transport", "source/input transfer", "capacity/admission operations"],
        "outside_cost_reporting": "separate operational elapsed; unmeasured CPU UNKNOWN, never zero or included in measured campaign subtotal",
        "coordinator_scope_acceptance": "2026-10-23 explicit thread directive; numerical resource authorization remains pending",
    }
    pins = {}
    for line in (SOURCE / "requirements-lock.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            pins[name] = version
    spec["expected_packages"] = pins
    write("run_spec.json", spec)
    members = [{"source_path": (HERE / row["path"]).relative_to(REPO).as_posix(),
                "remote_target": REMOTE + "/o001/" + row["path"], "role": "orchestration", **row}
               for row in orchestration_files]
    members.append({"source_path": (HERE / "run_spec.json").relative_to(REPO).as_posix(),
                    "remote_target": REMOTE + "/o001/run_spec.json", "role": "resource_false_spec", **identify(HERE / "run_spec.json")})
    members.extend({"source_path": (SOURCE / row["path"]).relative_to(REPO).as_posix(),
                    "remote_target": REMOTE + "/s001/" + row["path"], "role": "frozen_scientific_source", **row}
                   for row in source_files)
    members.extend(staged.values())
    members.append({"source_path": (DELIVERY / "manifest.json").relative_to(REPO).as_posix(),
                    "remote_target": REMOTE + "/i001/scientific_manifest.json", "role": "accepted_scientific_manifest", **identify(DELIVERY / "manifest.json")})
    assert len({row["remote_target"] for row in members}) == len(members)
    write("transfer_manifest.json", {"format": "e164_transfer_allowlist_v1", "state": "STATIC_BOUND_NOT_AUTHORIZED_TO_TRANSFER",
          "transfer_authorized": False, "members": members, "member_count": len(members),
          "total_bytes": sum(row["bytes"] for row in members), "existing_reference_map": "reference_map.json",
          "authorization_member": None, "archive_created": False,
          "admission_fields_pending": ["resource_authorized", "project_bytes_before", "admission_measured_utc", "new coordinator directive and authorization payload"]})
    identity_rows = sorted(orchestration_files, key=lambda row: row["path"])
    orchestration_id = sha(json.dumps(identity_rows, sort_keys=True, separators=(",", ":")).encode())
    write("execution_freeze.json", {"format": "e164_local_execution_binding_v1", "state": "STATIC_BINDING_PENDING_FINAL_ENGINEERING_REVIEW",
          "source_identity_sha256": SOURCE_ID, "scientific_manifest_sha256": MANIFEST_SHA,
          "orchestration_identity_sha256": orchestration_id, "orchestration_identity_algorithm": "SHA256 compact sorted-key JSON of path/bytes/SHA rows sorted by path",
          "run_spec_sha256": identify(HERE / "run_spec.json")["sha256"],
          "transfer_manifest_sha256": identify(HERE / "transfer_manifest.json")["sha256"],
          "entry_argv": ["sh", REMOTE + "/o001/launch_command.sh"], "cwd": REMOTE + "/s001",
          "supervisor_argv": ["/data/robot/p001/env/bin/python", "-I", "-B", REMOTE + "/o001/supervisor.py", "--spec", REMOTE + "/o001/run_spec.json", "--authorization", REMOTE + "/o001/authorization.json"],
          "worker_argv": ["/data/robot/p001/env/bin/python", *spec["worker_argv"]],
          "environment": spec["environment"], "planned_limits": spec["limits"],
          "resource_authorized": False, "execution_authorized": False, "runtime_prechecks_executed": False,
          "reference_count": len(reference_map), "transfer_member_count": len(members),
          "admission_mutation_note": "Resource admission and fresh non-E164 project baseline require a new exact spec hash and separately bound coordinator directive before any launch."})
    print(json.dumps({"source_files": len(source_files), "reference_keys": len(reference_map),
                      "transfer_members": len(members), "transfer_bytes": sum(row["bytes"] for row in members),
                      "orchestration_identity_sha256": orchestration_id, "run_spec_sha256": identify(HERE / "run_spec.json")["sha256"],
                      "resource_authorized": False, "execution_authorized": False}))


if __name__ == "__main__":
    main()

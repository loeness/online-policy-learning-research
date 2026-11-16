"""E163 orchestration only: compose frozen scientific APIs, never fit on import."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback


SEEDS = tuple(range(26201, 26214)) + (26215, 26216, 26218)
SOURCE_ID = "0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea"
MANIFEST_SHA = "b837222aff57ab45abee226e9aa8bc9d2cd3444807fa348dee0c0c1ece70dc54"
PROJECT = Path("/data/robot/p001")
E_ROOT = PROJECT / "e163"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(path, root=PROJECT):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"path outside allocation: {path}")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"symlink rejected: {component}")
    if path.resolve() != path:
        raise ValueError(f"noncanonical path: {path}")
    return path


def read_json(path):
    return json.loads(safe(path, E_ROOT).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    write_bytes(path, data)


def write_bytes(path, data):
    path = safe(path, E_ROOT)
    with path.open("xb") as stream:
        stream.write(data)


def verify_file(path, expected):
    path = safe(path)
    data = path.read_bytes()
    if len(data) != expected["bytes"] or sha(data) != expected["sha256"]:
        raise ValueError(f"file identity mismatch: {path}")
    return data


def validate_authorization(spec_path, authorization_path):
    if (Path(spec_path) != E_ROOT / "o002/run_spec.json"
            or Path(authorization_path) != E_ROOT / "o002/authorization.json"):
        raise PermissionError("spec/authorization paths differ from frozen o002 placement")
    spec_bytes = safe(spec_path, E_ROOT).read_bytes()
    spec = json.loads(spec_bytes)
    auth = read_json(authorization_path)
    if (auth.get("experiment_id") != "E163" or auth.get("execution_authorized") is not True
            or auth.get("run_spec_sha256") != sha(spec_bytes)):
        raise PermissionError("coordinator execution authorization/spec binding missing")
    if auth.get("limits") != spec["limits"] or not auth.get("coordinator_directive"):
        raise PermissionError("authorization limits/directive missing")
    if os.environ.get("E163_SUPERVISED") != "1":
        raise PermissionError("worker requires the frozen supervisor")
    if (spec["source_identity_sha256"] != SOURCE_ID
            or spec["project_root"] != str(PROJECT) or spec["e_root"] != str(E_ROOT)
            or spec["seeds"] != list(SEEDS) or spec["lambda_values"] != [0, 1]):
        raise ValueError("run spec changes the frozen phase")
    return spec, auth


class CampaignStopped(Exception):
    pass


class BatchStateError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.scientific_status = status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    spec, auth = validate_authorization(args.spec, args.authorization)
    source = safe(spec["source_cwd"], E_ROOT)
    run = safe(spec["run_root"], E_ROOT)
    orchestration = safe(spec["orchestration_root"], E_ROOT)
    if source != E_ROOT / "s001" or run != E_ROOT / "run002" or orchestration != E_ROOT / "o002":
        raise ValueError("unexpected frozen workspace placement")
    run.mkdir(exist_ok=True)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    rows, completed, attempts, fit_returns, active, stage = [], [], [], [], None, "identity"

    def event(kind, **fields):
        with safe(run / "events_worker.jsonl", E_ROOT).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, "monotonic": time.monotonic(), **fields},
                                    sort_keys=True, allow_nan=False) + "\n")

    def stop(signum, frame):
        raise CampaignStopped(f"supervisor stop signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    try:
        if os.getcwd() != str(source):
            raise ValueError("unexpected scientific working directory")
        manifest_bytes = safe(orchestration / "scientific_manifest.json", E_ROOT).read_bytes()
        if sha(manifest_bytes) != MANIFEST_SHA:
            raise ValueError("frozen scientific manifest identity mismatch")
        manifest = json.loads(manifest_bytes)
        identity = lambda row: {k: row[k] for k in ("path", "bytes", "sha256")}
        if (manifest["source_identity_sha256"] != SOURCE_ID
                or spec["source_files"] != [identity(r) for r in manifest["snapshot_files"]]):
            raise ValueError("source inventory is not exactly the frozen scientific manifest")
        exact_inputs = []
        for row in manifest["input_files"]:
            seed, name = row["seed"], row["file"]
            if seed not in SEEDS or name not in ("actor.bin", "checkpoint.pt", "evidence.npz"):
                raise ValueError("unexpected frozen input identifier")
            exact_inputs.append({"seed": seed, "file": name, "bytes": row["bytes"], "sha256": row["sha256"],
                                 "remote_path": str(PROJECT / "e161" / "run001" / "producer" / "cases"
                                                    / f"s{seed}_dynamics_late" / "prefix" / name)})
        if spec["input_files"] != exact_inputs or len(exact_inputs) != 48:
            raise ValueError("input references are not the exact manifest-bound paths and identities")
        expected_foundation = {"remote_path": str(PROJECT / "s001" / "data" / "foundation.pt"),
                               "bytes": manifest["foundation"]["bytes"], "sha256": manifest["foundation"]["sha256"]}
        if spec["foundation"] != expected_foundation:
            raise ValueError("foundation reference is not the exact allowlisted identity")
        if {r["path"] for r in spec["orchestration_files"]} != {"runner.py", "supervisor.py", "gate.py", "gate_mapping.json"}:
            raise ValueError("orchestration inventory mismatch")
        for row in spec["source_files"]:
            if Path(row["path"]).is_absolute() or ".." in Path(row["path"]).parts:
                raise ValueError("nonrelative scientific source member")
            verify_file(source / row["path"], row)
        for row in spec["orchestration_files"]:
            verify_file(safe(spec["orchestration_root"], E_ROOT) / row["path"], row)
        for row in spec["input_files"]:
            verify_file(row["remote_path"], row)
        verify_file(spec["foundation"]["remote_path"], spec["foundation"])
        expected_inputs = {(seed, name) for seed in SEEDS for name in ("actor.bin", "checkpoint.pt", "evidence.npz")}
        if {(r["seed"], r["file"]) for r in spec["input_files"]} != expected_inputs or len(spec["input_files"]) != 48:
            raise ValueError("input manifest does not contain exactly the fixed 48 references")

        deps = {}
        for line in (source / "requirements-lock.txt").read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, version = line.split("==")
            actual = importlib.metadata.version(name)
            deps[name] = actual
            if actual != version:
                raise ValueError(f"dependency mismatch: {name} expected {version}, actual {actual}")
        sys.path.insert(0, str(source / "src"))
        import numpy as np
        import torch
        from online_rl.retention_protected_inputs import prepare_inputs, validate_config, encode_and_score_stage0
        from online_rl.retention_protected_candidate import fit_protected_candidate
        from gate import evaluate_batch
        if Path(sys.modules["gate"].__file__).resolve() != orchestration / "gate.py":
            raise ValueError("gate import escaped frozen orchestration")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        config = validate_config()
        imports = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                   if (name == "online_rl" or name.startswith("online_rl.")) and getattr(module, "__file__", None)}
        if not imports or any(not Path(p).is_relative_to(source / "src") for p in imports.values()):
            raise ValueError("scientific import escaped frozen source")
        write_json(run / "environment.json", {
            "python": sys.version, "executable": sys.executable, "dependencies": deps,
            "scientific_modules": imports, "numeric_environment": spec["environment"],
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "source_identity_sha256": SOURCE_ID, "run_spec_sha256": sha(Path(args.spec).read_bytes()),
            "authorization": auth, "platform": sys.platform,
        })

        # Validate every fixed prefix before beginning any fit. No seed selection.
        stage, prepared_all = "prepare", []
        for seed in SEEDS:
            entries = [r for r in spec["input_files"] if r["seed"] == seed]
            parents = {str(Path(r["remote_path"]).parent) for r in entries}
            if len(parents) != 1:
                raise ValueError("prefix reference parents differ")
            prepared = prepare_inputs(Path(parents.pop()), config=config, expected_seed=seed,
                                      expected_identity={r["file"]: r["sha256"] for r in entries},
                                      logical_case=f"s{seed}_dynamics_late")
            case = run / "cases" / f"s{seed}"
            case.mkdir(parents=True, exist_ok=False)
            split_keys = ("new_train_indices", "new_holdout_indices", "old_train_indices", "old_holdout_indices")
            write_json(case / "preparation.json", {
                "seed": seed, "prepared_id": prepared["prepared_id"], "input_identity": prepared["input_identity"],
                "owner_slots": prepared["owner_slots"].tolist(), "owner_tasks": prepared["owner_tasks"],
                "splits": {key: prepared[key].tolist() for key in split_keys},
                "memory_cpu_wall": prepared["memory_cpu_wall"],
                "foundation": spec["foundation"],
                "evidence_reference": next(r for r in entries if r["file"] == "evidence.npz"),
                "transient_arrays": {key: {"dtype": str(prepared[key].dtype), "shape": list(prepared[key].shape),
                                            "raw_C_order_sha256": sha(prepared[key].tobytes(order="C"))}
                                     for key in ("new_observations", "old_observations", "new_teacher", "old_teacher", "mean64")},
            })
            prepared_all.append(prepared)
            event("prefix_validated", seed=seed, prepared_id=prepared["prepared_id"])

        stage = "fit_and_score"
        for prepared in prepared_all:
            seed, candidates = prepared["seed"], {}
            for label, lam in (("C", 0), ("D", 1)):
                active = {"seed": seed, "arm": label, "lambda_value": lam}
                attempts.append(dict(active))
                event("fit_started", **active)
                target = run / "cases" / f"s{seed}" / label.lower()
                target.mkdir(exist_ok=False)
                fit = fit_protected_candidate(prepared, lambda_value=lam)
                fit_returns.append(dict(active))
                write_json(target / "diagnostics.json", fit["diagnostics"])
                raw_candidate = fit["candidate"].tobytes(order="C")
                write_bytes(target / "candidate.f32", raw_candidate)
                write_json(target / "candidate.json", {
                    "shape": [3, 3], "dtype": str(fit["candidate"].dtype), "dtype_str": fit["candidate"].dtype.str,
                    "byteorder": sys.byteorder, "order": "C",
                    "values": fit["candidate"].astype(float).tolist(), "raw_file_sha256": sha(raw_candidate),
                    "scientific_candidate_sha256": fit["diagnostics"]["candidate_sha256"],
                    "prepared_id": fit["prepared_id"], "lambda_value": lam,
                })
                score = encode_and_score_stage0(prepared, fit, binding="wildcard")
                metrics = score["metrics"]
                write_json(target / "metrics.json", metrics)
                for key, filename, hashkey, sizekey in (
                    ("actor_bytes", "actor.bin", "actor_sha256", "actor_bytes"),
                    ("checkpoint_bytes", "checkpoint.pt", "checkpoint_sha256", "checkpoint_bytes"),
                ):
                    data = score["deployment"][key]
                    if len(data) != metrics["capacity"][sizekey] or sha(data) != metrics["artifacts"][hashkey]:
                        raise ValueError("deployment bytes differ from scientific measurement")
                    write_bytes(target / filename, data)
                evidence_ref = next(r for r in spec["input_files"] if r["seed"] == seed and r["file"] == "evidence.npz")
                evidence = score["deployment"]["evidence_bytes"]
                if (sha(evidence) != evidence_ref["sha256"] or len(evidence) != evidence_ref["bytes"]
                        or metrics["artifacts"]["evidence_sha256"] != evidence_ref["sha256"]):
                    raise ValueError("returned evidence differs from exact original byte identity")
                write_json(target / "persistent_references.json", {
                    "evidence": evidence_ref, "foundation": spec["foundation"],
                    "full_persistent_bytes": metrics["capacity"]["full_persistent_bytes"],
                    "deployment_checkpoint_saved_verbatim": True,
                })
                candidates[label] = {"metrics": metrics, "diagnostics": fit["diagnostics"]}
                completed.append({**active, "candidate_sha256": metrics["candidate_sha256"]})
                event("fit_and_score_completed", **active)
                del score, fit
            c, d = candidates["C"]["metrics"], candidates["D"]["metrics"]
            if c["K_new_W0"] != d["K_new_W0"]:
                raise ValueError("C/D W0 controls differ")
            for m in (c, d):
                if (m["K_old"] != m["encoded_fresh"]["old_holdout"]
                        or m["K_new"] != m["encoded_fresh"]["new_holdout"]
                        or m["K_new_W0"] != m["W0_fresh"]["new_holdout"]
                        or m["G_new"] != m["K_new_W0"] - m["K_new"]
                        or m["binding"] != "wildcard" or m["fresh_local_is_zero"] is not True):
                    raise ValueError("gate mapping does not match frozen encoded holdout outputs")
            rows.append({"seed": seed, "K_old_C": c["K_old"], "K_old_D": d["K_old"],
                         "K_new_W0": c["K_new_W0"], "K_new_C": c["K_new"], "K_new_D": d["K_new"],
                         "fallback_flags": {a: candidates[a]["diagnostics"]["fallback_applied"] for a in ("C", "D")}})
        stage, active = "report", None
        gate = evaluate_batch(rows)
        if len(completed) != 32:
            raise BatchStateError("INCOMPLETE", "fixed32fit batch not completed")
        if gate["scored_count"] != 16:
            raise BatchStateError(gate["status"] if gate["status"] in ("INVALID", "INCOMPLETE") else "INCOMPLETE",
                                  "gate batch is not complete and valid")
        gate["scientific_mapping"] = spec["gate_mapping"]
        gate["completed_fits"] = completed
        write_json(run / "gate_inputs.json", rows)
        write_json(run / "scientific_report.json", gate)
        lines = ["# E163 Stage 0", "", f"Status: **{gate['status']}**; passes: **{gate['pass_count']}/16**.",
                 "", "32 fits; zero environment interactions. Encoded fresh-local-zero holdout KL mechanism screen only.",
                 "No reward effect, function-class compatibility proof or follow-up authorization is inferred.", "",
                 "| seed | K_old C | K_old D | G_new C | G_new D | pass |", "|---|---:|---:|---:|---:|---|"]
        lines += [f"| {r['seed']} | {r['K_old_C']:.9g} | {r['K_old_D']:.9g} | {r['G_new_C']:.9g} | {r['G_new_D']:.9g} | {r['passed']} |" for r in gate["evaluations"]]
        lines += ["", "Full C0/fallback/termination, train/holdout and codec metrics are retained per fit.",
                  "Process-tree CPU, wall, peak memory and stopping evidence are in the supervisor records."]
        write_bytes(run / "scientific_report.md", ("\n".join(lines) + "\n").encode())
        final_status = "COMPLETE"
        event("scientific_report_completed", status=gate["status"], passes=gate["pass_count"])
        exit_code = 0
    except BaseException as exc:
        final_status = ("INCOMPLETE" if isinstance(exc, (CampaignStopped, KeyboardInterrupt))
                        else getattr(exc, "scientific_status", "INVALID"))
        failure = {"status": final_status, "stage": stage, "active_fit": active,
                   "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                   "fit_attempts": attempts, "fit_returns": fit_returns,
                   "completed_fits": completed, "completed_seed_gate_rows": rows}
        if hasattr(exc, "partial_diagnostics"):
            failure["partial_diagnostics"] = exc.partial_diagnostics
        write_json(run / "failure_worker.json", failure)
        exit_code = 2
    write_json(run / "state_worker.json", {"status": final_status, "fit_attempts": len(attempts),
               "fit_returns": len(fit_returns), "completed_fits": len(completed),
               "environment_interactions": 0, "worker_body_cpu_seconds": time.process_time()-started_cpu,
               "worker_body_wall_seconds": time.monotonic()-started_wall,
               "cost_scope": "worker-body subtotal only; complete lifecycle is supervisor wait4 accounting"})
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

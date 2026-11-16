"""Compose the accepted E164 APIs; never run scientific work on import.

Local static preparation only until the separately admitted frozen run spec
and single-attempt authorization pass every gate below.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback


SOURCE_BINDING_STATUS = "BOUND_FROZEN_SOURCE"
SOURCE_ID = "baae9a6a766518105a6d5b2e9bc64fef3488ff581870931cf83ff2d7b23a4c9a"
MANIFEST_SHA = "76073211ec4ff37af2677e0d3cae33c87e46a703a25c306ccb0257c03a2ef379"
PROJECT = Path("/data/robot/p001")
E_ROOT = PROJECT / "e164"
ORCHESTRATION_ROOT = E_ROOT / "o001"
SOURCE_ROOT = E_ROOT / "s001"
RUN_ROOT = E_ROOT / "run001"
SEEDS = tuple(range(26201, 26214)) + (26215, 26216, 26218)
RECOVERY_SEEDS = (26204, 26206, 26209, 26212, 26213)
EXPECTED_LIMITS = {
    "cpu_seconds": 60, "campaign_wall_seconds": 120,
    "new_artifact_cache_temp_bytes": 67108864,
    "workers": 1, "numeric_threads_per_worker": 1,
    "gpu_devices": 0, "environment_interactions": 0,
    "candidate_fit_calls": 5, "new_complete_scoring_packages": 15,
    "logical_paired_prefixes": 16, "aggregate_remote_project_bytes": 100000000000,
}


class PendingSourceFreeze(RuntimeError):
    pass


class CampaignStopped(RuntimeError):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _safe_path(path_value, root=PROJECT, *, exists=True):
    path = Path(path_value)
    if not path.is_absolute() or not path.is_relative_to(root) or ".." in path.parts:
        raise PermissionError("path outside its exact allocated root")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise PermissionError("symlink rejected")
    if path.resolve(strict=exists) != path:
        raise PermissionError("noncanonical path rejected")
    return path


def _safe_exact(path_value, expected):
    path = _safe_path(path_value)
    if path != expected:
        raise PermissionError("path differs from the proposed exact E164 placement")
    return path


def _write_bytes(path, data):
    path = _safe_path(path, RUN_ROOT, exists=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return {"path": path.relative_to(RUN_ROOT).as_posix(), "bytes": len(data), "sha256": _sha(data)}


def _write_json(path, value):
    return _write_bytes(path, _json_bytes(value))


def _read_identity(path, expected):
    path = _safe_path(path)
    digest = hashlib.sha256()
    size = 0
    # Only the explicitly mapped file is opened. Models are hashed opaquely.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
            size += len(chunk)
    if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"file identity mismatch: {path}")
    return path


def _reference_key(descriptor):
    return descriptor.get("key") or descriptor.get("remote_path") or descriptor.get("path")


def _binding_references(value, result=None):
    result = {} if result is None else result
    if isinstance(value, dict):
        if "sha256" in value and "bytes" in value and _reference_key(value):
            key = _reference_key(value)
            if key in result and any(result[key][k] != value[k] for k in ("bytes", "sha256")):
                raise ValueError("conflicting reference identities")
            result[key] = value
        for item in value.values():
            _binding_references(item, result)
    elif isinstance(value, list):
        for item in value:
            _binding_references(item, result)
    return result


def _validate_authorization(spec_path, authorization_path):
    # This remains a hard code-level stop even if a draft JSON is edited.
    # Removing it requires the accepted source binding and a new code identity.
    if SOURCE_BINDING_STATUS != "BOUND_FROZEN_SOURCE":
        raise PendingSourceFreeze("PENDING_FINAL_SOURCE_MANIFEST: final local binding review is incomplete")
    spec_path = _safe_exact(spec_path, ORCHESTRATION_ROOT / "run_spec.json")
    authorization_path = _safe_exact(authorization_path, ORCHESTRATION_ROOT / "authorization.json")
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if (spec.get("preparation_state") != "FROZEN"
            or spec.get("resource_authorized") is not True
            or authorization.get("resource_authorized") is not True
            or authorization.get("execution_authorized") is not True
            or authorization.get("experiment_id") != "E164"
            or authorization.get("run_spec_sha256") != hashlib.sha256(spec_bytes).hexdigest()
            or not authorization.get("coordinator_directive")):
        raise PermissionError("frozen single-attempt resource/execution admission is missing")
    if spec.get("limits") != EXPECTED_LIMITS or authorization.get("limits") != EXPECTED_LIMITS:
        raise PermissionError("limits differ from the proposed E164 limits")
    if (spec.get("experiment_id") != "E164"
            or spec.get("project_root") != str(PROJECT)
            or spec.get("e_root") != str(E_ROOT)
            or spec.get("orchestration_root") != str(ORCHESTRATION_ROOT)
            or spec.get("source_cwd") != str(SOURCE_ROOT)
            or spec.get("run_root") != str(RUN_ROOT)
            or spec.get("seeds") != list(SEEDS)
            or spec.get("recovery_seed_ids") != list(RECOVERY_SEEDS)):
        raise PermissionError("E164 placements or fixed phase differ")
    if spec.get("source_identity_sha256") != SOURCE_ID or spec.get("scientific_manifest_sha256") != MANIFEST_SHA:
        raise PermissionError("scientific snapshot identity differs from the accepted delivery")
    if os.environ.get("E164_SUPERVISED") != "1":
        raise PermissionError("worker requires the separately admitted supervisor")
    return spec, authorization


def _run_bound_campaign(spec, authorization):
    """Call the frozen APIs in the METHOD/HANDOFF order, with no selector here."""
    started_cpu, started_wall = time.process_time(), time.monotonic()
    state = {"format": "e164_worker_state_v1", "status": "INCOMPLETE", "fit_attempts": 0,
             "fit_returns": 0, "completed_fits": 0, "score_calls": 0, "new_scoring_packages": 0,
             "logical_prefixes": 0, "anchors_confirmed_count": 0, "cohort_lock_sha256": None,
             "environment_interactions": 0, "completed_scores": []}
    events = None
    stage, active = "output_setup", None

    def event(name, **fields):
        events.write(json.dumps({"event": name, "worker_body_wall_seconds": time.monotonic()-started_wall,
                                **fields}, sort_keys=True, allow_nan=False) + "\n")
        events.flush()

    def stop(signum, frame):
        raise CampaignStopped(f"owned worker received signal {signum}")

    try:
        RUN_ROOT.mkdir(exist_ok=True)
        events = _safe_path(RUN_ROOT / "events_worker.jsonl", RUN_ROOT, exists=False).open("x", encoding="utf-8", newline="\n")
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        stage = "identity"
        if sys.platform != "linux" or sys.byteorder != "little":
            raise RuntimeError("the admitted interpreter must be little-endian Linux")
        for key, value in spec["environment"].items():
            if os.environ.get(key) != value:
                raise RuntimeError(f"worker environment mismatch: {key}")
        if platform.python_version() != spec["expected_python_version"]:
            raise RuntimeError("compute interpreter version differs from the admitted runtime")
        # Verify every imported project file before placing the frozen src on sys.path.
        for member in spec["source_files"]:
            _read_identity(SOURCE_ROOT / member["path"], member)
        for member in spec["orchestration_files"]:
            _read_identity(ORCHESTRATION_ROOT / member["path"], member)
        manifest_bytes = _read_identity(spec["scientific_manifest"]["path"], spec["scientific_manifest"]).read_bytes()
        if _sha(manifest_bytes) != MANIFEST_SHA:
            raise ValueError("scientific manifest changed")
        installed = {name: importlib.metadata.version(name) for name in spec["expected_packages"]}
        if installed != spec["expected_packages"]:
            raise ValueError("locked dependency versions differ")
        bindings_ref = spec["bindings"]
        bindings_bytes = _read_identity(bindings_ref["path"], bindings_ref).read_bytes()
        bindings = json.loads(bindings_bytes)
        _write_bytes(RUN_ROOT / "bindings.json", bindings_bytes)
        descriptors = _binding_references(bindings)
        mapping = spec["reference_map"]
        if set(descriptors) != set(mapping):
            raise ValueError("explicit reference map does not equal the frozen binding descriptor set")
        verified, json_cache, receipt = {}, {}, {"format": "e164_reference_receipt_v1", "files": []}
        for key in sorted(descriptors):
            descriptor, location = descriptors[key], mapping[key]
            if any(location[k] != descriptor[k] for k in ("bytes", "sha256")):
                raise ValueError("reference map attempted to replace an expected identity")
            path = _read_identity(location["actual_path"], descriptor)
            verified[key] = path
            item = {"key": key, "actual_path": str(path), "bytes": descriptor["bytes"], "sha256": descriptor["sha256"]}
            if path.suffix == ".json":
                data = path.read_bytes()
                relative = f"reference_json/{descriptor['sha256']}.json"
                copy_path = RUN_ROOT / relative
                if not copy_path.exists():
                    _write_bytes(copy_path, data)
                elif copy_path.read_bytes() != data:
                    raise ValueError("conflicting JSON reference copy")
                item["local_json_copy"] = relative
                json_cache[key] = data
            receipt["files"].append(item)
        _write_json(RUN_ROOT / "reference_receipt.json", receipt)

        def old_json(descriptor):
            key = _reference_key(descriptor)
            if key not in json_cache:
                raise ValueError("required original JSON was not explicitly verified")
            return json.loads(json_cache[key])

        sys.path.insert(0, str(SOURCE_ROOT / "src"))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from online_rl import retention_acceptance_replay as replay
        from online_rl import retention_acceptance_diagnostic as diagnostic
        from online_rl.retention_protected_inputs import prepare_inputs
        from report import build_report
        imported = {}
        for name in ("online_rl.retention_acceptance_replay", "online_rl.retention_acceptance_diagnostic",
                     "online_rl.retention_protected_inputs", "online_rl.retention_protected_candidate"):
            path = Path(sys.modules[name].__file__).resolve(strict=True)
            if not path.is_relative_to(SOURCE_ROOT / "src"):
                raise ValueError("scientific import escaped the accepted frozen snapshot")
            imported[name] = str(path)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("numeric thread count differs")
        if torch.version.cuda is not None:
            raise RuntimeError("E164 requires the locked CPU-only torch build")
        _write_json(RUN_ROOT / "environment.json", {"source_identity_sha256": SOURCE_ID,
                    "scientific_manifest_sha256": MANIFEST_SHA, "python": sys.version,
                    "executable": sys.executable, "platform": platform.platform(), "dependencies": installed,
                    "scientific_modules": imported, "numeric_environment": spec["environment"],
                    "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                    "torch_build": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
                    "requested_device": "cpu", "gpu_devices_used": 0,
                    "environment_interactions": 0})

        original = bindings["records"]
        old_diagnostics = {seed: old_json(original[str(seed)]["arms"]["D"]["artifacts"]["diagnostics"]) for seed in SEEDS}
        stage = "training_replay"
        recovery = replay.build_recovery_manifest(old_diagnostics, config=SOURCE_ROOT / "configs/h036_acceptance_diagnostic_v1.json")
        if recovery["recovery_seeds"] != list(RECOVERY_SEEDS) or len(recovery["reuse_seeds"]) != 11:
            raise ValueError("frozen replay changed the fixed cohort partition")
        _write_json(RUN_ROOT / "recovery_manifest.json", recovery)
        prepared_by_seed, traces, locks = {}, {}, {}

        def save_array(path, array, dtype):
            if array.dtype.str != dtype or tuple(array.shape) != (3, 3):
                raise ValueError("returned candidate dtype/shape differs from the frozen method")
            data = array.tobytes(order="C")
            file_ref = _write_bytes(path, data)
            scientific = _sha(dtype.encode() + b"\0" + repr((3, 3)).encode() + b"\0" + data)
            return {**file_ref, "dtype": dtype, "shape": [3, 3], "order": "C", "scientific_sha256": scientific}

        stage = "recover_and_lock"
        for seed in RECOVERY_SEEDS:
            active = {"seed": seed}
            record = original[str(seed)]
            input_paths = {name: verified[_reference_key(ref)] for name, ref in record["inputs"].items()}
            parents = {path.parent for path in input_paths.values()}
            if len(parents) != 1 or set(input_paths) != {"actor.bin", "checkpoint.pt", "evidence.npz"}:
                raise ValueError("prefix components are not the exact three colocated files")
            prefix_dir = next(iter(parents))
            if any(path != prefix_dir / name for name, path in input_paths.items()):
                raise ValueError("verified prefix paths differ from the fixed filenames read by prepare_inputs")
            prepared = prepare_inputs(prefix_dir, expected_seed=seed,
                expected_identity={name: ref["sha256"] for name, ref in record["inputs"].items()},
                logical_case=record["logical_case"])
            if prepared["prepared_id"] != record["prepared_id"]:
                raise replay.ReplayMismatchError("prepared identity differs from the original prefix")
            case = RUN_ROOT / "cases" / f"s{seed}"
            _write_json(case / "preparation.json", {"seed": seed, "prepared_id": prepared["prepared_id"],
                        "input_identity": prepared["input_identity"], "memory_cpu_wall": prepared["memory_cpu_wall"]})
            view = diagnostic.make_training_view(prepared)
            if state["fit_attempts"] >= 5:
                raise RuntimeError("candidate fit-call cap reached")
            state["fit_attempts"] += 1
            event("fit_started", seed=seed, call=state["fit_attempts"])
            trace = diagnostic.fit_selection_trace(view)
            state["fit_returns"] += 1
            event("fit_returned", seed=seed, trace_sha256=trace["trace_sha256"])
            _write_json(case / "trace.json", {key: value for key, value in trace.items() if key != "candidates"})
            artifacts = {name: save_array(case / "arrays" / (name + ".f64"), array, "<f8")
                         for name, array in trace["candidates"].items()}
            _write_json(case / "trace_artifacts.json", artifacts)
            replay.assert_replay_matches(old_diagnostics[seed], trace)
            state["completed_fits"] += 1
            event("trace_matched", seed=seed, path_sha256=trace["path_sha256"])
            locked = diagnostic.lock_acceptance_probes(view, trace)
            lock_meta = {key: value for key, value in locked.items() if key != "probes"}
            lock_meta["probes"] = {}
            for label, probe in locked["probes"].items():
                lock_meta["probes"][label] = {key: value for key, value in probe.items()
                                               if key not in {"candidate", "raw_cast_candidate"}}
                save_array(case / "probes" / label / "candidate.f32", probe["candidate"], "<f4")
                save_array(case / "probes" / label / "raw_cast.f32", probe["raw_cast_candidate"], "<f4")
            _write_json(case / "lock.json", lock_meta)
            _write_json(case / "lock_identity.json", diagnostic.locked_probe_manifest(locked))
            prepared_by_seed[seed], traces[seed], locks[seed] = prepared, trace, locked

        cohort = replay.lock_cohort(recovery, {seed: diagnostic.locked_probe_manifest(lock) for seed, lock in locks.items()}, bindings)
        cohort_ref = _write_json(RUN_ROOT / "cohort.json", cohort)
        state["logical_prefixes"] = 16
        state["cohort_lock_sha256"] = cohort["cohort_lock_sha256"]
        event("cohort_locked", cohort_lock_sha256=cohort["cohort_lock_sha256"], file_sha256=cohort_ref["sha256"])
        anchor_permit = replay.make_anchor_permit(cohort)
        _write_json(RUN_ROOT / "anchor_permit.json", anchor_permit)
        cached, score_for_label, alias_rows = {}, {}, []

        def score_once(seed, label, permit):
            nonlocal active
            active = {"seed": seed, "label": label}
            if state["score_calls"] >= 15:
                raise RuntimeError("scientific scoring-call cap reached")
            state["score_calls"] += 1
            event("anchor_score_started" if label == "V" else "counterfactual_score_started",
                  seed=seed, label=label, permit_sha256=permit["permit_sha256"])
            score = diagnostic.encode_and_score_locked_probe(prepared_by_seed[seed], locks[seed], label, stage_permit=permit)
            directory = f"scores/s{seed}/{label}"
            folder = RUN_ROOT / directory
            metrics = score["metrics"]
            metrics_ref = _write_json(folder / "metrics.json", metrics)
            _write_json(folder / "selection_source.json", score["selection_source"])
            artifacts = {}
            for name, key, filename, metric_key in (
                ("actor", "actor_bytes", "actor.bin", "actor_sha256"),
                ("checkpoint", "checkpoint_bytes", "checkpoint.pt", "checkpoint_sha256"),
                ("evidence", "evidence_bytes", "evidence.npz", "evidence_sha256"),
            ):
                data = score["deployment"][key]
                if not isinstance(data, bytes) or _sha(data) != metrics["artifacts"][metric_key]:
                    raise ValueError("deployment bytes differ from the actual scientific score")
                artifacts[name] = _write_bytes(folder / filename, data)
            expected_evidence = original[str(seed)]["inputs"]["evidence.npz"]
            if any(artifacts["evidence"][key] != expected_evidence[key] for key in ("bytes", "sha256")):
                raise ValueError("returned evidence is not the original exact bytes")
            foundation = bindings["foundation"]
            artifacts["foundation"] = {"bytes": foundation["bytes"], "sha256": foundation["sha256"],
                "reference_key": _reference_key(foundation), "remote_path": verified[_reference_key(foundation)].as_posix(), "reference_only": True}
            full_bytes = sum(entry["bytes"] for entry in artifacts.values())
            capacity = metrics["capacity"]
            if (artifacts["actor"]["bytes"] != 89 or full_bytes > 262144
                    or capacity["actor_bytes"] != 89 or capacity["full_persistent_bytes"] != full_bytes
                    or capacity["checkpoint_bytes"] != artifacts["checkpoint"]["bytes"]
                    or capacity["evidence_npz_bytes"] != artifacts["evidence"]["bytes"]
                    or capacity["foundation_bytes"] != foundation["bytes"]):
                raise ValueError("actual complete deployment package violates its measured layout")
            probe = locks[seed]["probes"][label]
            candidate_raw = probe["candidate"].tobytes(order="C")
            if _sha(candidate_raw) != probe["raw_sha256"] or metrics["candidate_sha256"] != probe["candidate_sha256"]:
                raise ValueError("score candidate differs from the locked bytes")
            package = {"format": "e164_score_package_v1", "seed": seed, "label": label,
                "prepared_id": metrics["prepared_id"], "candidate_sha256": probe["candidate_sha256"],
                "candidate_raw_sha256": probe["raw_sha256"], "diagnostic_only": score["diagnostic_only"],
                "artifacts": artifacts, "full_persistent_bytes": full_bytes}
            package_ref = _write_json(folder / "package.json", package)
            reference = {"package_path": package_ref["path"], "metrics": metrics_ref, "canonical_label": label,
                         "candidate_sha256": probe["candidate_sha256"], "candidate_raw_sha256": probe["raw_sha256"]}
            completed = {"seed": seed, "label": label, "relative_dir": directory,
                         "candidate_raw_sha256": probe["raw_sha256"], "candidate_sha256": probe["candidate_sha256"]}
            state["completed_scores"].append(completed)
            state["new_scoring_packages"] += 1
            event("score_completed", **completed, metrics_sha256=metrics_ref["sha256"], package_sha256=package_ref["sha256"])
            cached[(seed, candidate_raw)] = (score, reference)
            score_for_label[(seed, label)] = (score, reference, "NEW_SCORE")
            return score

        stage = "anchors"
        anchors = {seed: score_once(seed, "V", anchor_permit) for seed in RECOVERY_SEEDS}
        old_d_metrics_bytes = {seed: json_cache[_reference_key(original[str(seed)]["arms"]["D"]["artifacts"]["metrics"])] for seed in RECOVERY_SEEDS}
        counter_permit = replay.confirm_anchors(cohort, anchors, old_d_metrics_bytes, bindings)
        _write_json(RUN_ROOT / "counterfactual_permit.json", counter_permit)
        state["anchors_confirmed_count"] = 5
        event("anchors_confirmed", count=5, permit_sha256=counter_permit["permit_sha256"],
              cohort_lock_sha256=cohort["cohort_lock_sha256"])
        stage = "counterfactual_scores"
        for seed in RECOVERY_SEEDS:
            for label in ("UG", "UR"):
                raw = locks[seed]["probes"][label]["candidate"].tobytes(order="C")
                if (seed, raw) in cached:
                    score, reference = cached[(seed, raw)]
                    score_for_label[(seed, label)] = (score, reference, "ALIAS")
                    event("score_alias", seed=seed, label=label, **reference)
                else:
                    score_once(seed, label, counter_permit)

        stage, active = "report", None
        prefixes = []
        for seed in SEEDS:
            record = original[str(seed)]
            c_ref = record["arms"]["C"]["artifacts"]["metrics"]
            logical = {"seed": seed, "C": old_json(c_ref), "C_reference": c_ref}
            for label in ("V", "UG", "UR"):
                if seed in RECOVERY_SEEDS:
                    score, reference, origin = score_for_label[(seed, label)]
                    metrics = score["metrics"]
                else:
                    d_arm = record["arms"]["D"]
                    metrics = old_json(d_arm["artifacts"]["metrics"])
                    reference = {"cohort_ref": {"path": "cohort.json", "cohort_lock_sha256": cohort["cohort_lock_sha256"]},
                                 "metrics_key": _reference_key(d_arm["artifacts"]["metrics"]),
                                 "artifacts": d_arm["artifacts"], "candidate_sha256": d_arm["helper_candidate_sha256"],
                                 "candidate_raw_sha256": d_arm["raw_candidate_sha256"]}
                    origin = "REUSED_E163"
                logical[label] = {"metrics": metrics, "origin": origin, "reference": reference}
                alias_rows.append({"seed": seed, "label": label, "origin": origin, "reference": reference})
            prefixes.append(logical)
        logical_scores = {"format": "e164_logical_scores_v1", "source_freeze_state": "FROZEN", "source_freeze_id": SOURCE_ID, "prefixes": prefixes}
        _write_json(RUN_ROOT / "aliases.json", {"format": "e164_score_aliases_v1", "records": alias_rows})
        _write_json(RUN_ROOT / "logical_scores.json", logical_scores)
        report = build_report(logical_scores)
        _write_json(RUN_ROOT / "scientific_report.json", report)
        _write_bytes(RUN_ROOT / "scientific_report.md", ("# E164 exploratory paired diagnosis\n\n"
                    "16 logical prefixes; UG-V and UR-UG. No advancement decision.\n"
                    "Exact per-prefix values, aliases and count summaries are in scientific_report.json.\n").encode())
        state["status"] = "COMPLETE"
        event("report_completed", logical_prefixes=16, new_scoring_packages=state["new_scoring_packages"])
        exit_code = 0
    except BaseException as exc:
        state["status"] = "INCOMPLETE" if isinstance(exc, (CampaignStopped, KeyboardInterrupt)) else (
            "REPLAY_MISMATCH" if type(exc).__name__ == "ReplayMismatchError" else "INVALID")
        failure = {"status": state["status"], "stage": stage,
                    "active": active, "exception_type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(), "counts": state}
        try:
            _write_json(RUN_ROOT / "failure_worker.json", failure)
        except (OSError, PermissionError, ValueError) as record_error:
            print(json.dumps({"worker_failure": failure, "failure_record_error": str(record_error)}), file=sys.stderr)
        exit_code = 2
    finally:
        state["worker_body_cpu_seconds"] = time.process_time() - started_cpu
        state["worker_body_wall_seconds"] = time.monotonic() - started_wall
        state["cost_scope"] = "worker body subtotal only; complete lifecycle is supervisor wait4"
        try:
            _write_json(RUN_ROOT / "state_worker.json", state)
        except (OSError, PermissionError, ValueError) as record_error:
            print(json.dumps({"worker_state": state, "state_record_error": str(record_error)}), file=sys.stderr)
        if events is not None:
            events.close()
    return exit_code


def main():
    parser = argparse.ArgumentParser(description="E164 frozen scientific API orchestration")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    try:
        spec, authorization = _validate_authorization(args.spec, args.authorization)
        return _run_bound_campaign(spec, authorization)
    except (PendingSourceFreeze, PermissionError) as exc:
        # No output directory or scientific imports are created on refusal.
        print(json.dumps({"experiment_id": "E164", "status": "PREPARATION_REFUSED",
                          "reason": str(exc)}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

"""One-shot allowlisted H036 source freeze; no model or environment execution."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
DELIVERY = ROOT / "research/appendix/H036/delivery_v1"
SNAPSHOT = DELIVERY / "source_snapshot"
PACKAGE = ROOT / "src/online_rl"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def reference(relative: str) -> dict:
    path = ROOT / relative
    data = path.read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": sha(data)}


def closure() -> list[str]:
    pending = ["retention_protected_candidate", "retention_protected_inputs", "__init__"]
    found: set[str] = set()
    while pending:
        module = pending.pop()
        if module in found:
            continue
        path = PACKAGE / (module + ".py")
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unsupported local import {module}")
        found.add(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1:
                    raise RuntimeError("unexpected relative import level")
                if node.module:
                    pending.append(node.module)
                else:
                    pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("online_rl."):
                pending.append(node.module.split(".", 1)[1])
    return sorted(f"src/online_rl/{module}.py" for module in found)


def main() -> None:
    if DELIVERY.exists():
        raise RuntimeError("delivery_v1 already exists; never overwrite a frozen delivery")
    source_paths = closure() + [
        "configs/h036_protected_v1.json", "requirements.txt", "requirements-lock.txt",
        "tests/test_retention_protected_candidate.py", "tests/test_retention_protected_inputs.py",
        "tests/test_retention_protected_score.py", "research/appendix/H036/METHOD_V1.md",
        "research/appendix/H036/HANDOFF_V1.md",
    ]
    source_paths = sorted(set(source_paths))
    if any(Path(name).name.lower() in {"agent.md", "agents.md"} for name in source_paths):
        raise RuntimeError("local instruction file cannot be exported")
    validation_manifest_path = "experiments/exp163/preparation/validation/input_manifest_planned.json"
    planned = json.loads((ROOT / validation_manifest_path).read_text(encoding="utf-8-sig"))
    config = json.loads((ROOT / "configs/h036_protected_v1.json").read_text(encoding="utf-8"))
    lookup = {(int(row["seed"]), row["file"]): row for row in planned["rows"]}
    names = ("checkpoint.pt", "actor.bin", "evidence.npz")
    expected_keys = {(seed, name) for seed in config["seeds"] for name in names}
    if len(lookup) != 48 or set(lookup) != expected_keys or len(planned["rows"]) != 48:
        raise RuntimeError("validation manifest is not the fixed 16 by 3 inputs")
    input_files = []
    for seed in config["seeds"]:
        logical_case = f"s{seed}_dynamics_late"
        for name in names:
            row = lookup[(seed, name)]
            relative = f"experiments/exp161/runs/run001/producer/cases/{logical_case}/prefix/{name}"
            path = ROOT / relative
            if path.resolve() != Path(row["local_path"]).resolve() or path.is_symlink():
                raise RuntimeError("input placement differs from the explicit allowlist")
            data = path.read_bytes()  # Opaque identity check only; never deserialize.
            if len(data) != row["bytes"] or sha(data) != row["sha256"]:
                raise RuntimeError(f"input hash differs: {logical_case}/{name}")
            input_files.append({"seed": seed, "logical_case": logical_case, "file": name,
                                "relative_path": relative, "local_path": str(path),
                                "bytes": len(data), "sha256": sha(data)})
    source_manifest_path = "experiments/exp161/preparation/source-manifest.json"
    historical = json.loads((ROOT / source_manifest_path).read_text(encoding="utf-8"))
    historical_lookup = {row["path"]: row for row in historical["files"]}
    fitter_path = "src/online_rl/retention_behavioral_candidate.py"
    fitter_matches = sha((ROOT / fitter_path).read_bytes()) == historical_lookup[fitter_path]["sha256"]
    if not fitter_matches or config["foundation"] != {
        key: historical_lookup["data/foundation.pt"][key] for key in ("bytes", "sha256")
    }:
        raise RuntimeError("old fitter or foundation metadata differs from E161")

    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    files = []
    for name in source_paths:
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source is not a regular file: {name}")
        data = path.read_bytes()
        target = SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append({"path": name, "bytes": len(data), "sha256": sha(data),
                      "working_source_sha256": sha(data)})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *source_paths], cwd=ROOT)
    (DELIVERY / "tracked_changes.patch").write_bytes(patch)
    identity_rows = [{key: row[key] for key in ("path", "bytes", "sha256")} for row in files]
    dependencies = {}
    for package in ("numpy", "torch", "psutil", "pytest", "stable-baselines3", "gymnasium"):
        dependencies[package] = importlib.metadata.version(package)
    contracts = [
        "research/appendix/H036/COORDINATOR_CONTRACT_20261022.md",
        "research/appendix/H036/DESIGN_AND_VALIDATION_PROPOSAL_20261022.md",
        "experiments/exp163/README.md", "experiments/exp163/budget.json",
        "experiments/exp163/preparation/registration.json", validation_manifest_path,
        source_manifest_path, "experiments/exp161/runs/run001/artifact-manifest.json",
    ]
    manifest = {
        "format": "h036_method_delivery_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "H036", "experiment_id": "E163",
        "phase": "implementation_only_synthetic_checks_no_real_fits",
        "actual_checkout": str(ROOT), "frozen_execution_root": str(SNAPSHOT),
        "git_commit": commit, "git_tree": "dirty; allowlisted full snapshots identify the code",
        "source_identity_sha256": sha(canonical(identity_rows)),
        "source_identity_algorithm": "SHA256 of UTF-8 compact sorted-key JSON [{path,bytes,sha256}] sorted by path",
        "snapshot_files": files,
        "tracked_diff": {"path": "tracked_changes.patch", "bytes": len(patch), "sha256": sha(patch),
                         "scope": "only allowlisted tracked changes; untracked source is preserved in full snapshot"},
        "packaging_script": reference("research/appendix/H036/development_checks/freeze_v1.py"),
        "input_files": input_files, "input_file_count": len(input_files),
        "input_total_bytes": sum(row["bytes"] for row in input_files),
        "input_usage_this_turn": "opaque SHA/byte checks only; no real prefix deserialization, feature extraction or fitting",
        "input_identity": "logical_case + basename + SHA256 + bytes; placement paths are provenance only",
        "foundation": {**config["foundation"], "source_manifest_path": source_manifest_path,
                       "entry": "data/foundation.pt", "content_read_this_turn": False},
        "h026_fitter_matches_e161_snapshot": fitter_matches,
        "reference_files": [reference(name) for name in contracts],
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "dependencies": dependencies,
                        "numeric_threads": 1, "remote_environment_verified": False},
        "engineering_checks": [reference(f"research/appendix/H036/development_checks/attempt{attempt}.{extension}")
                               for attempt in ("001", "002") for extension in ("txt", "json")],
        "engineering_read_scope_note": "During adapter development one overly broad historical schema search also returned existing outcome fields. It was stopped. No returned outcome was used as an algorithm input, selection rule or parameter-tuning criterion. Final scientific APIs read only the three prefix files.",
        "excluded": ["Agent.md", "AGENTS.md", "credentials", "Git history", "real input copies", "historical outcome copies"],
        "scientific_results": "NOT_RUN: synthetic engineering checks are not E163 evidence",
        "figure_paths": "NOT_APPLICABLE: no scientific results or figures in implementation delivery",
        "real_execution": "NOT_STARTED; validator owns readiness-gated SSH within robot project allocation; this manifest grants no extra scope",
    }
    (DELIVERY / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"snapshot_files": len(files), "source_identity_sha256": manifest["source_identity_sha256"],
                      "input_files": len(input_files), "input_bytes": manifest["input_total_bytes"],
                      "git_commit": commit}, indent=2))


if __name__ == "__main__":
    main()

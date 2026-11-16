"""One-shot explicit H036 acceptance source freeze; no model execution."""
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
DELIVERY = ROOT / "research/appendix/H036/delivery_acceptance_v1"
SNAPSHOT = DELIVERY / "source_snapshot"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference(relative: str) -> dict:
    data = (ROOT / relative).read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": sha(data)}


def local_closure() -> list[str]:
    pending = ["retention_acceptance_replay", "retention_acceptance_diagnostic", "__init__"]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = ROOT / "src/online_rl" / (name + ".py")
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unexpected local dependency {name}")
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1:
                    raise RuntimeError("unexpected relative import")
                pending.extend([node.module] if node.module else [alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("online_rl."):
                pending.append(node.module.split(".", 1)[1])
    return sorted(f"src/online_rl/{name}.py" for name in seen)


def main() -> None:
    if DELIVERY.exists():
        raise RuntimeError("never overwrite a frozen delivery")
    bindings_path = "research/appendix/H036/ACCEPTANCE_BINDINGS_V1.json"
    bindings = json.loads((ROOT / bindings_path).read_text(encoding="utf-8"))
    original = json.loads((ROOT / "research/appendix/H036/delivery_v1/manifest.json").read_text(encoding="utf-8"))
    if bindings["v1_source_identity_sha256"] != original["source_identity_sha256"]:
        raise RuntimeError("old source identity differs")
    for row in bindings["original_scientific_files"]:
        for path in (ROOT / row["path"], ROOT / "research/appendix/H036/delivery_v1/source_snapshot" / row["path"]):
            data = path.read_bytes()
            if len(data) != row["bytes"] or sha(data) != row["sha256"]:
                raise RuntimeError(f"old science file changed: {row['path']}")
    paths = sorted(set(local_closure() + [
        "configs/h036_protected_v1.json", "configs/h036_acceptance_diagnostic_v1.json",
        "requirements.txt", "requirements-lock.txt",
        "tests/test_retention_acceptance_diagnostic.py",
        "tests/test_retention_protected_inputs.py", "tests/test_retention_protected_score.py",
        "research/appendix/H036/METHOD_ACCEPTANCE_V1.md",
        "research/appendix/H036/HANDOFF_ACCEPTANCE_V1.md", bindings_path,
        "research/appendix/H036/NEXT_DESIGN_20261023.md",
        "research/appendix/H036/NEXT_SCALAR_REPLAY_20261023.json",
    ]))
    if any(Path(name).name.lower() in {"agent.md", "agents.md"} for name in paths):
        raise RuntimeError("local instructions cannot be exported")
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    files = []
    for name in paths:
        source = ROOT / name
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"not a regular source file: {name}")
        data = source.read_bytes()
        target = SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append({"path": name, "bytes": len(data), "sha256": sha(data), "working_source_sha256": sha(data)})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *paths], cwd=ROOT)
    (DELIVERY / "tracked_changes.patch").write_bytes(patch)
    rows = [{key: row[key] for key in ("path", "bytes", "sha256")} for row in files]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    refs = [
        "research/appendix/H036/COORDINATOR_ACCEPTANCE_IMPLEMENTATION_20261023.md",
        "research/appendix/H036/COORDINATOR_DESIGN_ACCEPTANCE_20261023.json",
        "research/appendix/H036/NEXT_DESIGN_REVIEW_20261023.md",
        "experiments/exp164/README.md", "experiments/exp164/budget.json",
        "research/appendix/H036/delivery_v1/manifest.json",
        "experiments/exp163/runs/run002/provenance.json",
        "experiments/exp163/runs/run002/mirror_manifest.json",
    ]
    packages = {name: importlib.metadata.version(name) for name in (
        "numpy", "torch", "psutil", "pytest", "gymnasium", "stable-baselines3",
    )}
    manifest = {
        "format": "h036_acceptance_method_delivery_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "H036", "experiment_id": "E164",
        "phase": "local_implementation_and_synthetic_checks_only",
        "actual_checkout": str(ROOT), "frozen_execution_root": str(SNAPSHOT),
        "git_commit": commit, "git_tree": "dirty; full allowlisted snapshots identify code",
        "source_identity_sha256": sha(canonical),
        "source_identity_algorithm": "SHA256 compact sorted-key UTF-8 JSON [{path,bytes,sha256}] sorted by path",
        "snapshot_files": files,
        "tracked_diff": {"path": "tracked_changes.patch", "bytes": len(patch), "sha256": sha(patch),
                         "scope": "allowlisted tracked diff only; untracked sources/configs preserved in full"},
        "old_scientific_source_identity_sha256": original["source_identity_sha256"],
        "old_scientific_dependency_file_count": len(bindings["original_scientific_files"]),
        "all_old_scientific_dependency_bytes_unchanged": True,
        "bindings": {**reference(bindings_path), "scope": "16 logical cases / 32 original arms / all candidate-input-deployment-score identities; no raw model copies"},
        "logical_prefixes": 16, "exact_reuse_prefixes": 11,
        "planned_recovery_seeds": [26204, 26206, 26209, 26212, 26213],
        "planned_maximum_real_fit_calls": 5, "planned_maximum_new_scoring_packages": 15,
        "input_usage_this_delivery": "existing JSON identity metadata and opaque SHA/size checks; no binary model decoding, feature extraction, new real fits or policy scores",
        "reference_files": [reference(name) for name in refs],
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "packages": packages, "numeric_thread_environment": 1, "remote_check": "NOT_PERFORMED"},
        "engineering_checks": [reference("research/appendix/H036/acceptance_development_checks/attempt001." + ext) for ext in ("txt", "json")],
        "packaging_scripts": [reference("research/appendix/H036/acceptance_development_checks/" + name) for name in ("build_bindings_v1.py", "freeze_acceptance_v1.py")],
        "review": "Read-only bounded review found source-index validation gap; corrected before final checks. Training loop/gradient helper/initialization/stopping/U noninterference/legacy adapter reviewed. This is engineering review, not E164 scientific validation.",
        "test_scope": "only new test_retention_acceptance_diagnostic.py; old fixture helpers imported, old23/1876 not rerun",
        "scientific_results": "NOT_RUN; synthetic engineering is not E164 mechanism evidence",
        "figure_paths": "NOT_APPLICABLE: no new scientific results or figures",
        "resource_state": "PLANNED_ONLY, no campaign/reservation authority conferred",
        "real_recovery_fits": 0, "new_real_policy_scores": 0, "environment_interactions": 0,
        "remote_operations": 0,
        "excluded": ["Agent.md", "AGENTS.md", "credentials", "Git history", "raw model/input/deployment copies", "experiment runner"],
    }
    (DELIVERY / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "source_identity_sha256": manifest["source_identity_sha256"],
                      "original_dependency_files_unchanged": len(bindings["original_scientific_files"]),
                      "snapshot_bytes": sum(row["bytes"] for row in files)}, indent=2))


if __name__ == "__main__":
    main()

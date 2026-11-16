"""Package only allowlisted source bytes; never import or run research code.

This is a local text/AST/manifest operation, not an E165 runtime check or audit.
It deliberately refuses to replace an existing frozen manifest.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
DELIVERY = Path(__file__).resolve().parent
LEGACY_ID = "baae9a6a766518105a6d5b2e9bc64fef3488ff581870931cf83ff2d7b23a4c9a"
NEW_PATHS = (
    "src/online_rl/retention_budgeted_search.py",
    "configs/h036_budgeted_search_v1.json",
    "tests/test_retention_budgeted_search.py",
    "research/appendix/H036/METHOD_BUDGETED_V1.md",
    "research/appendix/H036/HANDOFF_BUDGETED_V1.md",
    "research/appendix/H036/NEXT_DESIGN_20261024.md",
    "research/appendix/H036/COORDINATOR_E165_CONTRACT_20261024.md",
    "research/appendix/H036/delivery_budgeted_v1/anchor_manifest.json",
    "research/appendix/H036/delivery_budgeted_v1/CHECKS_SPEC.json",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def checked_path(relative):
    candidate = (ROOT / relative).resolve(strict=True)
    if not candidate.is_relative_to(ROOT.resolve()) or candidate.name.lower() in {"agent.md", "agents.md"}:
        raise ValueError("source path escapes root or is a local instruction file")
    if ".git" in candidate.relative_to(ROOT).parts:
        raise ValueError("Git history is not an exportable source artifact")
    return candidate


def main():
    target_manifest = DELIVERY / "manifest.json"
    if target_manifest.exists():
        raise FileExistsError("A frozen manifest already exists; preserve it and version any later repair")
    old_path = ROOT / "research/appendix/H036/delivery_acceptance_v1/manifest.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if old["source_identity_sha256"] != LEGACY_ID:
        raise ValueError("legacy source manifest identity changed")
    legacy_rows = old["snapshot_files"]
    for entry in legacy_rows:
        data = checked_path(entry["path"]).read_bytes()
        if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
            raise ValueError("legacy allowlisted dependency changed: " + entry["path"])
    anchors_data = (DELIVERY / "anchor_manifest.json").read_bytes()
    checks_data = (DELIVERY / "CHECKS_SPEC.json").read_bytes()
    config = json.loads(checked_path("configs/h036_budgeted_search_v1.json").read_text(encoding="utf-8"))
    check_spec = json.loads(checks_data.decode("utf-8"))
    test_path = "tests/test_retention_budgeted_search.py"
    test_tree = ast.parse(checked_path(test_path).read_text(encoding="utf-8-sig"), filename=test_path)
    test_names = [node.name for node in test_tree.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]
    expected_nodes = [test_path + "::" + name for name in test_names]
    if check_spec.get("expected_test_nodeids") != expected_nodes or check_spec.get("expected_count") != len(expected_nodes):
        raise ValueError("frozen CHECKS node IDs differ from static test function order")
    if config.get("anchor_manifest_sha256") != digest(anchors_data):
        raise ValueError("configuration does not pin the exact anchor manifest file bytes")
    if config.get("e164_source_identity_sha256") != LEGACY_ID:
        raise ValueError("configuration does not pin accepted E164 science source identity")
    core_path = "src/online_rl/retention_budgeted_search.py"
    core_tree = ast.parse(checked_path(core_path).read_text(encoding="utf-8-sig"), filename=core_path)
    literal_pins = {
        node.targets[0].id: node.value.value for node in core_tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)
    }
    if literal_pins.get("_ANCHOR_MANIFEST_SHA256") != config["anchor_manifest_sha256"]:
        raise ValueError("core anchor pin differs from configuration")
    if literal_pins.get("_E164_SOURCE_IDENTITY_SHA256") != LEGACY_ID:
        raise ValueError("core source pin differs from accepted E164 source")
    paths = sorted({entry["path"] for entry in legacy_rows} | set(NEW_PATHS))
    snapshot = DELIVERY / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    rows = []
    parsed = []
    for relative in paths:
        source = checked_path(relative)
        data = source.read_bytes()
        if source.suffix == ".py":
            tree = ast.parse(data.decode("utf-8-sig"), filename=relative)
            parsed.append({"path": relative, "ast_parse": "PASS_STATIC_ONLY",
                           "top_level_functions": [node.name for node in tree.body
                                                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]})
        if source.suffix == ".json":
            json.loads(data.decode("utf-8-sig"))
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        copied = destination.read_bytes()
        if copied != data:
            raise OSError("source copy differs: " + relative)
        rows.append({"path": relative, "bytes": len(data), "sha256": digest(data)})
    # These are exact byte references, not newly generated scientific evidence.
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked_paths = [path for path in paths if path.startswith("src/") or path.startswith("configs/")
                     or path.startswith("tests/") or path.startswith("requirements")]
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *tracked_paths], cwd=ROOT)
    (DELIVERY / "tracked_source.patch").write_bytes(patch)
    static_report = {
        "format": "h036_budgeted_static_packaging_v1",
        "scope": "Text/AST/JSON/allowlisted byte identity only; no imports, runtime checks or scientific audit",
        "research_runtime_executed_locally": False,
        "pytest_executed_locally": False,
        "remote_checks": "PENDING",
        "legacy_source_files_unchanged": len(legacy_rows),
        "ast_files": parsed,
        "source_snapshot_files": len(rows),
        "source_snapshot_bytes": sum(row["bytes"] for row in rows),
    }
    save_json(DELIVERY / "static_checks.json", static_report)
    manifest = {
        "format": "h036_budgeted_method_delivery_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "H036", "experiment_id": "E165",
        "state": "SOURCE_FROZEN_REMOTE_CHECKS_PENDING",
        "actual_checkout": str(ROOT), "git_commit": git_head,
        "git_tree": "dirty; complete allowlisted source snapshots and tracked patch identify execution code",
        "source_identity_algorithm": "SHA256 compact sorted-key UTF-8 JSON [{path,bytes,sha256}] sorted by path",
        "source_identity_sha256": digest(canonical(rows)),
        "snapshot_files": rows,
        "snapshot_bytes": sum(row["bytes"] for row in rows),
        "legacy_source_identity_sha256": LEGACY_ID,
        "anchor_manifest": {"path": "research/appendix/H036/delivery_budgeted_v1/anchor_manifest.json",
                            "bytes": len(anchors_data), "sha256": digest(anchors_data)},
        "checks_spec": {"path": "research/appendix/H036/delivery_budgeted_v1/CHECKS_SPEC.json",
                        "bytes": len(checks_data), "sha256": digest(checks_data)},
        "packaging": [{"path": "research/appendix/H036/delivery_budgeted_v1/" + name,
                       "bytes": (DELIVERY / name).stat().st_size,
                       "sha256": digest((DELIVERY / name).read_bytes())}
                      for name in ("freeze_static.py", "static_checks.json", "tracked_source.patch", "README.md")],
        "runtime_checks": "NOT_RUN_LOCAL; pending single admitted remote CHECKS receipt",
        "scientific_results": "NOT_RUN",
        "new_local_research_oracle_calls": 0, "new_local_model_score_calls": 0,
        "real_fits": 0, "remote_operations_by_algorithm_chat": 0,
        "resource_state": "No execution authorization conferred by this artifact",
        "excluded": ["Agent.md", "AGENTS.md", "credentials", "Git history",
                     "raw input/checkpoint/deployment copies", "experiment runner"],
    }
    save_json(target_manifest, manifest)
    print(json.dumps({"state": manifest["state"], "source_identity_sha256": manifest["source_identity_sha256"],
                      "snapshot_files": len(rows), "snapshot_bytes": manifest["snapshot_bytes"],
                      "remote_checks": "PENDING"}))


if __name__ == "__main__":
    main()

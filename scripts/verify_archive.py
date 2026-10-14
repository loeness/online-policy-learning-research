"""Check all archived artifact bytes against the committed SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path, manifest_path: Path) -> dict:
    root = root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest version")
    failures, seen = [], set()
    for row in manifest["files"]:
        relative = row["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or relative in seen:
            failures.append({"path": relative, "reason": "unsafe or duplicate path"})
            continue
        seen.add(relative)
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            failures.append({"path": relative, "reason": "byte count or SHA-256 mismatch"})
    if len(manifest["files"]) != manifest["file_count"]:
        failures.append({"path": "manifest", "reason": "file count mismatch"})
    if sum(row["bytes"] for row in manifest["files"]) != manifest["total_bytes"]:
        failures.append({"path": "manifest", "reason": "total bytes mismatch"})
    return {"valid": not failures, "checked_files": len(seen),
            "expected_total_bytes": manifest["total_bytes"], "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.manifest or args.root / "archive/manifest.json")
    print(json.dumps(result, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

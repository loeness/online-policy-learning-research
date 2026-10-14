#!/usr/bin/env python3
"""Build a deterministic SHA-256 inventory for the research archive."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "archive" / "manifest.json"
EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "tmp",
    ".archive-verification",
}
MANIFEST_PATH = "archive/manifest.json"
CREDENTIAL_FILENAME_PATTERNS = (".env", ".env.*", "*.pem", "*.key")

EXCLUSIONS = [
    {"pattern": ".git/**", "reason": "Git object database and repository metadata."},
    {"pattern": ".venv/**", "reason": "Locally installed Python environment."},
    {
        "pattern": ".archive-verification/**",
        "reason": "Temporary clone used to verify the private repository upload.",
    },
    {
        "pattern": "**/__pycache__/**, **/.pytest_cache/**, **/*.pyc",
        "reason": "Generated Python bytecode and test caches.",
    },
    {
        "pattern": "tmp/**",
        "reason": "Local-only confidential source-document extraction; not experiment data or reproducibility code.",
    },
    {
        "patterns": list(CREDENTIAL_FILENAME_PATTERNS),
        "reason": "Credential or private-key filename patterns are excluded by name only; contents are not read.",
    },
    {
        "path": MANIFEST_PATH,
        "reason": "The generated manifest omits itself to avoid recursive hashing.",
    },
]


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def iter_archive_files():
    for directory, subdirectories, filenames in os.walk(ROOT):
        relative_directory = Path(directory).relative_to(ROOT)
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        )
        for filename in sorted(filenames):
            path = Path(directory) / filename
            relative_path = path.relative_to(ROOT).as_posix()
            if (
                relative_path == MANIFEST_PATH
                or path.suffix.lower() == ".pyc"
                or any(
                    fnmatch.fnmatchcase(path.name, pattern)
                    for pattern in CREDENTIAL_FILENAME_PATTERNS
                )
            ):
                continue
            yield relative_path, path


def build_manifest() -> dict[str, object]:
    files = []
    for relative_path, path in iter_archive_files():
        digest, size = sha256_and_size(path)
        files.append({"path": relative_path, "bytes": size, "sha256": digest})

    files.sort(key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "generated_by": "scripts/build_archive_manifest.py",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "exclusions": EXCLUSIONS,
        "files": files,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

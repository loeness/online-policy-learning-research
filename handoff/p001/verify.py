"""Compare a returned o001 directory with its SHA-256 transfer manifest."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat

root = Path(__file__).resolve().parent
data = root / "o001"
manifest = root / "o001.sha256"
if not data.is_dir() or not manifest.is_file():
    raise SystemExit("Missing o001 directory or o001.sha256 manifest.")
for path in (data, manifest):
    st = path.lstat()
    if path.is_symlink() or getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise SystemExit("The returned directory or manifest is a link or reparse point; stop.")
expected = {}
for line in manifest.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
    if not match:
        raise SystemExit("Invalid or unsupported manifest line; preserve the files and stop.")
    rel = PurePosixPath(match.group(2))
    if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != match.group(2):
        raise SystemExit("Unsafe manifest path; preserve the files and stop.")
    if str(rel) in expected:
        raise SystemExit("Duplicate manifest path.")
    expected[str(rel)] = match.group(1)
observed = {}
for folder, dirs, files in os.walk(data, followlinks=False):
    for name in dirs + files:
        path = Path(folder) / name
        st = path.lstat()
        if path.is_symlink() or getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise SystemExit("A returned path is a link or reparse point; stop.")
    for name in files:
        path = Path(folder) / name
        rel = path.relative_to(data).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        observed[rel] = digest.hexdigest()
if not expected or observed != expected:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    changed = sorted(name for name in set(expected) & set(observed) if expected[name] != observed[name])
    raise SystemExit(f"FAIL: missing={missing}, extra={extra}, hash_mismatch={changed}")
print(f"PASS: {len(expected)} files match the remote SHA-256 manifest.")

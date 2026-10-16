"""Repair only the missing top-level metadata in the resumed unknown cohort.

The repair validates the frozen configuration hash, every run record/config,
and every frozen protocol snapshot before writing. The original cohort summary
is archived byte-for-byte; run summaries, ledgers, and snapshots are untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "outputs" / "write_stream_unknown"
FIELDS = ("base_sequence", "sequence", "protocol_sha256")
PROTOCOL_RELATIVE = Path("docs") / "WRITE_STREAM_PROTOCOL.md"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_source(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    frozen_path = output / "frozen_config.json"
    resume_note_path = output / "RESUME_NOTE.json"
    summary_path = output / "summary.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = _read_object(frozen_path)
    resume_note = _read_object(resume_note_path)
    summary = _read_object(summary_path)

    frozen_hash = _sha256(frozen_bytes)
    if resume_note.get("frozen_config_sha256") != frozen_hash:
        raise ValueError("RESUME_NOTE frozen_config_sha256 does not match frozen_config.json")
    if frozen.get("phase") != "unknown":
        raise ValueError("repair is restricted to an unknown-phase frozen config")
    if any(field not in frozen for field in FIELDS):
        raise ValueError("frozen_config.json is missing required cohort metadata")
    if not isinstance(frozen["base_sequence"], list) or not isinstance(frozen["sequence"], list):
        raise ValueError("frozen sequence metadata must be lists")
    if frozen["sequence"] != frozen["base_sequence"] + list(reversed(frozen["base_sequence"])):
        raise ValueError("frozen sequence is not base_sequence followed by its reverse")
    if not isinstance(frozen["protocol_sha256"], str) or not frozen["protocol_sha256"]:
        raise ValueError("frozen protocol SHA-256 is missing")

    expected_seeds = frozen.get("seeds")
    expected_conditions = frozen.get("conditions")
    if not isinstance(expected_seeds, list) or not isinstance(expected_conditions, list):
        raise ValueError("frozen seed and condition grids must be lists")
    expected_keys = {
        (int(seed), str(condition))
        for seed in expected_seeds
        for condition in expected_conditions
    }
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 12 or len(records) != len(expected_keys):
        raise ValueError("resumed summary must contain its complete 12-record frozen grid")
    actual_keys: set[tuple[int, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("cohort records must be JSON objects")
        key = (int(record.get("seed", -1)), str(record.get("condition", "")))
        if key in actual_keys:
            raise ValueError(f"duplicate cohort record: {key}")
        actual_keys.add(key)
        if record.get("sequence") != frozen["sequence"]:
            raise ValueError(f"record sequence does not match frozen config: {key}")

        run_dir = output / f"seed_{key[0]}" / key[1]
        run_config = _read_object(run_dir / "config.json")
        if run_config.get("sequence") != frozen["sequence"]:
            raise ValueError(f"run config sequence does not match frozen config: {key}")
        if (run_config.get("phase"), run_config.get("seed"), run_config.get("condition")) != (
            "unknown", key[0], key[1]
        ):
            raise ValueError(f"run config identity does not match frozen cohort: {key}")
        run_summary = _read_object(run_dir / "summary.json")
        if run_summary != record:
            raise ValueError(f"top-level record differs from its run summary: {key}")

        source_manifest = record.get("source_manifest")
        sources = source_manifest.get("sources") if isinstance(source_manifest, dict) else None
        protocol_metadata = sources.get(PROTOCOL_RELATIVE.as_posix()) if isinstance(sources, dict) else None
        if not isinstance(protocol_metadata, dict):
            raise ValueError(f"protocol snapshot metadata is missing: {key}")
        if protocol_metadata.get("sha256") != frozen["protocol_sha256"]:
            raise ValueError(f"protocol snapshot hash differs from frozen config: {key}")

        snapshot_dir = run_dir / "source_snapshot"
        protocol_snapshot = snapshot_dir / PROTOCOL_RELATIVE
        actual_protocol_bytes = protocol_snapshot.read_bytes()
        snapshot_hash = _sha256(actual_protocol_bytes)
        snapshot_manifest = _read_object(snapshot_dir / "manifest.json")
        if snapshot_manifest != source_manifest:
            raise ValueError(f"snapshot manifest differs from run record: {key}")
        if (snapshot_hash != frozen["protocol_sha256"]
                or protocol_metadata.get("bytes") != len(actual_protocol_bytes)):
            raise ValueError(f"protocol snapshot bytes/hash differ from frozen config: {key}")

    if actual_keys != expected_keys:
        raise ValueError("resumed summary records do not match the frozen seed-condition grid")

    for field in FIELDS:
        if field in summary and summary[field] != frozen[field]:
            raise ValueError(f"summary field {field} conflicts with frozen_config.json")
    return frozen, resume_note, summary, frozen_bytes


def _repair_summary(output: Path) -> dict[str, Any]:
    """Add the three stock cohort metadata fields after validating provenance."""

    output = output.resolve()
    frozen, resume_note, summary, frozen_bytes = _validate_source(output)
    summary_path = output / "summary.json"
    original_bytes = summary_path.read_bytes()
    before_hash = _sha256(original_bytes)
    after_summary = dict(summary)
    records_before = json.loads(json.dumps(summary["records"], ensure_ascii=False))
    for field in FIELDS:
        after_summary[field] = frozen[field]
    if after_summary["records"] != records_before:
        raise AssertionError("metadata repair must preserve the records list")
    repaired_bytes = json.dumps(after_summary, indent=2, ensure_ascii=False).encode("utf-8")
    after_hash = _sha256(repaired_bytes)

    history_dir = output / "metadata_history"
    history_path = history_dir / "summary_before_schema_repair.json"
    note_path = output / "METADATA_REPAIR_NOTE.json"
    summary_tmp = output / "summary.json.repair-tmp"
    note_tmp = output / "METADATA_REPAIR_NOTE.json.tmp"
    if history_path.exists() or note_path.exists() or summary_tmp.exists() or note_tmp.exists():
        raise FileExistsError("preserve existing metadata repair artifacts")

    note = {
        "schema": "write_resume_metadata_repair_v1",
        "cohort": "outputs/write_stream_unknown",
        "reason": "resume_write_stream.py omitted stock top-level cohort metadata fields",
        "copied_fields": {field: "frozen_config.json" for field in FIELDS},
        "frozen_config": {
            "path": "frozen_config.json",
            "sha256": _sha256(frozen_bytes),
            "resume_note_path": "RESUME_NOTE.json",
            "resume_note_sha256": _sha256((output / "RESUME_NOTE.json").read_bytes()),
            "resume_note_frozen_config_sha256": resume_note["frozen_config_sha256"],
        },
        "protocol_sha256": frozen["protocol_sha256"],
        "summary": {
            "path": "summary.json",
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "before_copy": "metadata_history/summary_before_schema_repair.json",
        },
        "records_list_unchanged": True,
        "only_top_level_fields_added": list(FIELDS),
        "raw_run_summaries_ledgers_and_source_snapshots_modified": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    history_dir.mkdir(exist_ok=True)
    with history_path.open("xb") as handle:
        handle.write(original_bytes)
    summary_tmp.write_bytes(repaired_bytes)
    note_tmp.write_text(json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(summary_tmp, summary_path)
    os.replace(note_tmp, note_path)
    return note


def repair_target() -> dict[str, Any]:
    """Run the repair against the one explicitly authorized cohort."""

    expected = (ROOT / "outputs" / "write_stream_unknown").resolve()
    target = TARGET.resolve()
    if target != expected:
        raise RuntimeError("repair target safety check failed")
    return _repair_summary(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    note = repair_target()
    print(json.dumps(note, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

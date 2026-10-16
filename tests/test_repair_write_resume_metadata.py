from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.repair_write_resume_metadata import FIELDS, _repair_summary


BASE_SEQUENCE = [f"T{index:02d}+" for index in range(12)]
SEQUENCE = BASE_SEQUENCE + list(reversed(BASE_SEQUENCE))
SEEDS = [1031, 1033, 1039]
CONDITIONS = ["independent7_lfu", "independent7_lru", "reward_shared", "trial_gate_relaxed"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_cohort(root: Path) -> Path:
    cohort = root / "outputs" / "write_stream_unknown"
    cohort.mkdir(parents=True)
    protocol = b"frozen protocol fixture\n"
    protocol_hash = hashlib.sha256(protocol).hexdigest()
    frozen = {
        "phase": "unknown",
        "base_sequence": BASE_SEQUENCE,
        "sequence": SEQUENCE,
        "protocol_sha256": protocol_hash,
        "seeds": SEEDS,
        "conditions": CONDITIONS,
    }
    frozen_bytes = json.dumps(frozen, indent=2, ensure_ascii=False).encode("utf-8")
    (cohort / "frozen_config.json").write_bytes(frozen_bytes)
    _write_json(cohort / "RESUME_NOTE.json", {
        "frozen_config_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
    })

    records = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            run_dir = cohort / f"seed_{seed}" / condition
            snapshot_dir = run_dir / "source_snapshot"
            snapshot_protocol = snapshot_dir / "docs" / "WRITE_STREAM_PROTOCOL.md"
            snapshot_protocol.parent.mkdir(parents=True)
            snapshot_protocol.write_bytes(protocol)
            manifest = {
                "sources": {
                    "docs/WRITE_STREAM_PROTOCOL.md": {
                        "sha256": protocol_hash,
                        "bytes": len(protocol),
                    },
                },
            }
            _write_json(snapshot_dir / "manifest.json", manifest)
            record = {
                "phase": "unknown",
                "seed": seed,
                "condition": condition,
                "sequence": SEQUENCE,
                "source_manifest": manifest,
            }
            _write_json(run_dir / "config.json", {
                "phase": "unknown",
                "seed": seed,
                "condition": condition,
                "sequence": SEQUENCE,
            })
            _write_json(run_dir / "summary.json", record)
            records.append(record)
    _write_json(cohort / "summary.json", {
        "phase": "unknown",
        "conditions": CONDITIONS,
        "seeds": SEEDS,
        "records": records,
        "cohort_cost": {"training_steps": 12},
    })
    return cohort


def test_repair_copies_frozen_fields_and_preserves_records_and_original_bytes(tmp_path: Path) -> None:
    cohort = _make_cohort(tmp_path)
    summary_path = cohort / "summary.json"
    before_bytes = summary_path.read_bytes()
    before_summary = json.loads(before_bytes)
    before_records = before_summary["records"]
    run_summary_bytes = {
        path: path.read_bytes()
        for path in cohort.glob("seed_*/*/summary.json")
    }

    note = _repair_summary(cohort)

    after_bytes = summary_path.read_bytes()
    after_summary = json.loads(after_bytes)
    assert {field: after_summary[field] for field in FIELDS} == {
        field: json.loads((cohort / "frozen_config.json").read_text(encoding="utf-8"))[field]
        for field in FIELDS
    }
    assert after_summary["records"] == before_records
    assert set(after_summary) - set(before_summary) == set(FIELDS)
    assert (cohort / "metadata_history" / "summary_before_schema_repair.json").read_bytes() == before_bytes
    assert note["summary"]["before_sha256"] == hashlib.sha256(before_bytes).hexdigest()
    assert note["summary"]["after_sha256"] == hashlib.sha256(after_bytes).hexdigest()
    assert note["records_list_unchanged"] is True
    assert {
        path: path.read_bytes()
        for path in cohort.glob("seed_*/*/summary.json")
    } == run_summary_bytes


@pytest.mark.parametrize("mismatch", ["resume_note_hash", "protocol_snapshot"])
def test_repair_refuses_mismatched_provenance_without_writing(mismatch: str, tmp_path: Path) -> None:
    cohort = _make_cohort(tmp_path)
    summary_path = cohort / "summary.json"
    before_summary = summary_path.read_bytes()
    if mismatch == "resume_note_hash":
        _write_json(cohort / "RESUME_NOTE.json", {"frozen_config_sha256": "0" * 64})
    else:
        snapshot = cohort / "seed_1031" / CONDITIONS[0] / "source_snapshot" / "docs" / "WRITE_STREAM_PROTOCOL.md"
        snapshot.write_bytes(b"different protocol\n")

    with pytest.raises(ValueError):
        _repair_summary(cohort)

    assert summary_path.read_bytes() == before_summary
    assert not (cohort / "metadata_history" / "summary_before_schema_repair.json").exists()
    assert not (cohort / "METADATA_REPAIR_NOTE.json").exists()

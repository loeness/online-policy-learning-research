"""Read-only coefficient-magnitude inventory for completed retention streams.

This diagnostic reads only the frozen calibration/comparison outputs and the
successful mechanism pilot v2 prefix. It does not load raw reward ledgers or
inspect the in-progress mechanism main run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
import re
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from online_rl.storage_decision import common_mean_effective, semantic_hash


CALIBRATION_ROOT = ROOT / "outputs" / "retention_value_calibration_main"
COMPARISON_ROOT = ROOT / "outputs" / "retention_value_comparison_main"
PILOT_V2_ROOT = ROOT / "outputs" / "retention_mechanism_pilot_v2"
OUTPUT_ROOT = ROOT / "outputs" / "retention_content_diagnostics_v2"
TARGET_METHODS = ("keep", "replace", "mean", "shared_ppo")
LOCAL_KEY = re.compile(r"^_nodes\.(\d+)\.A$")
SHARED_KEY = re.compile(r"^_shared_coefficients\.(\d+)\.C$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64).reshape(-1)))


def _core_actor_arrays(learner: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    core = learner["core_state"]
    locals_by_slot: dict[int, np.ndarray] = {}
    shared_by_group: dict[int, np.ndarray] = {}
    for name, value in core.items():
        local_match = LOCAL_KEY.match(str(name))
        shared_match = SHARED_KEY.match(str(name))
        if local_match:
            slot = int(local_match.group(1)) - 1
            locals_by_slot[slot] = np.asarray(value.detach().cpu(), dtype=np.float32).reshape(27)
        elif shared_match:
            group = int(shared_match.group(1))
            shared_by_group[group] = np.asarray(value.detach().cpu(), dtype=np.float32).reshape(9)
    if sorted(locals_by_slot) != [0, 1] or sorted(shared_by_group) != [0]:
        raise ValueError("checkpoint does not have the expected two-local, one-shared actor schema")
    local = np.stack([locals_by_slot[index] for index in range(2)]).astype(np.float32)
    shared = np.stack([shared_by_group[index] for index in range(1)]).astype(np.float32)
    physical_groups = np.asarray(core["slot_groups"].detach().cpu(), dtype=np.int8)
    if physical_groups.shape != (2,):
        raise ValueError("checkpoint physical group binding must have two slots")
    return local, shared, physical_groups


def _checkpoint_row(
    *, root: Path, stream: Mapping[str, Any], stream_summary: Mapping[str, Any],
    checkpoint: Path, experiment: str, bout_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    learner = payload.get("learner")
    if not isinstance(learner, Mapping):
        raise ValueError(f"checkpoint has no learner state: {checkpoint}")
    local, shared, physical = _core_actor_arrays(learner)
    tags = np.asarray(learner["slot_group_tags"], dtype=np.int8)
    cache = learner.get("cache", {})
    tasks = list(cache.get("slot_to_task", []))
    generations = np.asarray(learner.get("slot_generation", []), dtype=np.int64)
    fitted_steps = np.asarray(learner.get("fitted_steps", []), dtype=np.int64)
    if tags.shape != (2,) or len(tasks) != 2 or generations.shape != (2,) or fitted_steps.shape != (2,):
        raise ValueError(f"checkpoint resident metadata is malformed: {checkpoint}")

    provenance = payload.get("last_candidate_provenance") or {}
    prior_candidate_group = provenance.get("candidate_group")
    decision_match = re.search(r"decision_(\d+)$", checkpoint.parent.name)
    if decision_match is None:
        raise ValueError(f"unexpected checkpoint directory name: {checkpoint.parent.name}")
    decision_index = int(decision_match.group(1))

    # The last two outcome bouts are the source pair for the next decision.
    # Read only their task/generation metadata; returns are deliberately unused.
    source_bouts = [
        row for row in bout_metadata
        if row.get("decision_index") is not None
        and int(row["decision_index"]) == decision_index - 1
        and row.get("phase") == "outcome"
    ][-2:]
    candidate_group: int | None = None
    current_slots: list[int] = []
    candidate_source_records: list[dict[str, Any]] = []
    current_candidate_norm: float | None = None
    current_candidate_status = "unavailable_current_source_pair"
    if len(source_bouts) == 2:
        candidate_source_records = [
            {key: row.get(key) for key in (
                "task", "group", "variant", "slot", "generation", "fitted_steps_before",
                "fitted_steps_after", "steps_completed",
            )}
            for row in source_bouts
        ]
        source_groups = {int(row["group"]) for row in source_bouts}
        source_variants = {int(row["variant"]) for row in source_bouts}
        source_slots = [int(row["slot"]) for row in source_bouts]
        if len(source_groups) == 1 and len(source_variants) == 2 and len(set(source_slots)) == 2:
            candidate_group = next(iter(source_groups))
            current_slots = source_slots
            source_rows_match = all(
                slot in range(2)
                and tasks[slot] == str(row["task"])
                and int(generations[slot]) == int(row["generation"])
                and int(tags[slot]) == int(row["group"])
                and int(fitted_steps[slot]) >= 512
                for slot, row in zip(source_slots, source_bouts)
            )
            if source_rows_match:
                current_mean = common_mean_effective(local, shared, physical, current_slots)
                current_candidate_norm = _norm(current_mean)
                current_candidate_status = "reconstructed_current_next_decision_source_pair"
            else:
                current_candidate_status = "unavailable_source_pair_not_current_generation"
        else:
            current_candidate_status = "unavailable_source_pair_group_or_variant_contract"

    source_rows = provenance.get("sources", [])
    exact_source_slots: list[int] = []
    exact_sources_current = len(source_rows) == 2
    for source in source_rows:
        slot = int(source.get("slot", -1))
        if (
            slot not in range(2)
            or tasks[slot] != source.get("task")
            or int(generations[slot]) != int(source.get("generation", -1))
            or int(tags[slot]) != int(source.get("group", -2))
        ):
            exact_sources_current = False
            break
        exact_source_slots.append(slot)
    exact_last_candidate_norm: float | None = None
    if exact_sources_current and len(set(exact_source_slots)) == 2:
        exact_mean = common_mean_effective(local, shared, physical, exact_source_slots)
        exact_last_candidate_norm = _norm(exact_mean)

    parent_meta = payload.get("parent_metadata", {})
    parent_binding = parent_meta.get("binding")
    learner_parent = learner.get("parent_group")

    return {
        "experiment": experiment,
        "stage": str(stream.get("stage", stream_summary.get("stage", "unknown"))),
        "method": str(stream.get("method", stream_summary.get("method", "unknown"))),
        "stream_id": str(stream.get("stream_id", stream_summary.get("stream_id", "unknown"))),
        "pair_kind": str(stream.get("pair_kind", stream_summary.get("pair_kind", "unknown"))),
        "seed": int(stream.get("seed", stream_summary.get("seed", -1))),
        "status": str(stream.get("status", stream_summary.get("status", "unknown"))),
        "decision_index_next": decision_index,
        "checkpoint_path": _relative(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "parent_binding": parent_binding,
        "learner_parent_group": learner_parent,
        "parent_binding_consistent": bool(parent_binding == learner_parent),
        "parent_version": parent_meta.get("version"),
        "parent_written_decision": parent_meta.get("written_decision"),
        "parent_source_fit_steps": parent_meta.get("source_fit_steps"),
        "shared_parent_frobenius_norm_9": _norm(shared[0]),
        "candidate_group": candidate_group,
        "candidate_current_slots": json.dumps(current_slots),
        "candidate_current_tasks": json.dumps([tasks[index] for index in current_slots]),
        "candidate_current_source_records": json.dumps(candidate_source_records, sort_keys=True),
        "current_candidate_mean_effective_frobenius_norm_9": current_candidate_norm,
        "current_candidate_reconstruction_status": current_candidate_status,
        "last_candidate_matrix_persisted": bool(payload.get("candidate_coefficients_persisted", True)),
        "last_candidate_provenance_sources_current": bool(exact_sources_current and len(set(exact_source_slots)) == 2),
        "last_candidate_mean_effective_frobenius_norm_9": exact_last_candidate_norm,
        "logical_group_tags": json.dumps([int(value) for value in tags]),
        "physical_shared_group_tags": json.dumps([int(value) for value in physical]),
        "resident_tasks": json.dumps(tasks),
        "slot_generations": json.dumps([int(value) for value in generations]),
        "fitted_steps": json.dumps([int(value) for value in fitted_steps]),
    }


def _completed_streams(root: Path, expected_stage: str, expected_count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_summary_path = root / "summary.json"
    root_summary = _load_json(root_summary_path)
    if root_summary.get("stage") != expected_stage:
        raise ValueError(f"{root.name} is not a {expected_stage} run")
    if int(root_summary.get("completed_stream_count", -1)) != expected_count:
        raise ValueError(f"{root.name} is not the expected completed stream inventory")
    if int(root_summary.get("requested_stream_count", -1)) != expected_count or root_summary.get("failures"):
        raise ValueError(f"{root.name} has incomplete or failed streams")
    rows = root_summary.get("streams", [])
    if len(rows) != expected_count or any(row.get("status") != "completed" for row in rows):
        raise ValueError(f"{root.name} stream rows are not all completed")
    return root_summary, rows


def _stream_summary_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        candidate = root / path
        if candidate.is_file():
            return candidate
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_stream_checkpoints(
    root: Path, rows: list[dict[str, Any]], *, experiment: str,
    methods: set[str], inputs: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventories: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for stream in rows:
        if stream.get("method") not in methods:
            continue
        stream_summary_path = _stream_summary_path(root, str(stream["summary_path"]))
        stream_summary = _load_json(stream_summary_path)
        inputs[_relative(stream_summary_path)] = sha256(stream_summary_path)
        bouts_path = stream_summary_path.parent / "bouts.jsonl"
        bout_metadata: list[dict[str, Any]] = []
        safe_bout_keys = (
            "decision_index", "phase", "task", "group", "variant", "slot", "generation",
            "fitted_steps_before", "fitted_steps_after", "steps_completed",
        )
        with bouts_path.open("r", encoding="utf-8") as bout_stream:
            for line in bout_stream:
                if line.strip():
                    raw = json.loads(line)
                    bout_metadata.append({key: raw.get(key) for key in safe_bout_keys})
        inputs[_relative(bouts_path)] = sha256(bouts_path)
        checkpoint_root = stream_summary_path.parent / "checkpoints"
        checkpoints = sorted(checkpoint_root.glob("decision_*/checkpoint.pt"))
        decision_indices = [int(path.parent.name.split("_")[1]) for path in checkpoints]
        if not checkpoints:
            raise ValueError(f"no persisted decisions in completed stream {stream.get('stream_id')}")
        inventories.append({
            "experiment": experiment,
            "stream_id": stream["stream_id"],
            "method": stream["method"],
            "pair_kind": stream["pair_kind"],
            "seed": int(stream["seed"]),
            "completed_decisions": int(stream["completed_decisions"]),
            "available_checkpoint_decisions": decision_indices,
            "checkpoint_count": len(checkpoints),
        })
        for checkpoint in checkpoints:
            record = _checkpoint_row(
                root=root,
                stream=stream,
                stream_summary=stream_summary,
                checkpoint=checkpoint,
                experiment=experiment,
                bout_metadata=bout_metadata,
            )
            checkpoint_rows.append(record)
            inputs[record["checkpoint_path"]] = record["checkpoint_sha256"]
    return inventories, checkpoint_rows


def _pilot_prefix_reference(inputs: dict[str, str]) -> dict[str, Any]:
    summary_path = PILOT_V2_ROOT / "cases" / "s25089_dynamics_old_major" / "prefix" / "summary.json"
    observer_path = PILOT_V2_ROOT / "cases" / "s25089_dynamics_old_major" / "prefix" / "observer_prefix.json"
    summary = _load_json(summary_path)
    if int(summary.get("steps", -1)) != 7168 or not summary.get("source_fits", {}).get("verified"):
        raise ValueError("successful pilot v2 prefix verification did not match the expected contract")
    if summary.get("observer_artifact", {}).get("sha256") != sha256(observer_path):
        raise ValueError("pilot v2 observer prefix hash does not match its summary record")
    observer = _load_json(observer_path)
    arrays = observer["prefix_actor_arrays_before_action"]
    shared = np.asarray(arrays["shared"], dtype=np.float64).reshape(-1)
    candidate = np.asarray(observer["candidate_mean_effective"], dtype=np.float32)
    source_effective = np.asarray([
        row["effective_matrix"]
        for row in observer["candidate_current_effective_matrices_before_averaging"]
    ], dtype=np.float32)
    if source_effective.shape != (2, 3, 3, 3) or candidate.shape != (3, 3):
        raise ValueError("pilot candidate must be the producer's nine-coefficient mean from 2 slots by 3 bands")
    recomputed_candidate = source_effective.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    recomputed_hash = semantic_hash(recomputed_candidate)
    if not np.array_equal(recomputed_candidate, candidate):
        raise ValueError("pilot candidate does not match the 2-slot, 3-band common mean")
    if recomputed_hash != observer.get("candidate_mean_sha256"):
        raise ValueError("pilot common-mean candidate hash does not match observer evidence")
    candidate_sources = [
        {key: row[key] for key in ("task", "slot", "generation", "fitted_steps")}
        for row in observer["candidate_current_effective_matrices_before_averaging"]
    ]
    reference = {
        "experiment": "retention_mechanism_pilot_v2",
        "case_id": "s25089_dynamics_old_major",
        "prefix_steps": 7168,
        "parent_binding": int(arrays["parent_group"]),
        "shared_parent_frobenius_norm_9": _norm(shared),
        "candidate_group": 1,
        "candidate_current_source_records": candidate_sources,
        "current_candidate_mean_effective_frobenius_norm_9": _norm(candidate),
        "candidate_matrix_sha256": observer["candidate_mean_sha256"],
        "candidate_recomputed_sha256": recomputed_hash,
        "observer_prefix_path": _relative(observer_path),
        "observer_prefix_sha256": sha256(observer_path),
        "prefix_summary_sha256": sha256(summary_path),
    }
    inputs[reference["observer_prefix_path"]] = reference["observer_prefix_sha256"]
    inputs[_relative(summary_path)] = reference["prefix_summary_sha256"]
    return reference


def _summary_by_cohort(rows: list[dict[str, Any]], reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    cohorts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["stage"] != "pilot_prefix_reference":
            cohorts.setdefault((row["experiment"], row["method"]), []).append(row)
    result = []
    for (experiment, method), group in sorted(cohorts.items()):
        by_stream: dict[str, list[dict[str, Any]]] = {}
        for row in group:
            by_stream.setdefault(row["stream_id"], []).append(row)
        early = [min(items, key=lambda item: item["decision_index_next"]) for items in by_stream.values()]
        mature = [max(items, key=lambda item: item["decision_index_next"]) for items in by_stream.values()]

        def vals(items: list[dict[str, Any]], key: str) -> list[float]:
            return [float(item[key]) for item in items if item.get(key) is not None]

        for label, items in (("earliest_saved_checkpoint", early), ("latest_saved_checkpoint", mature)):
            parent_values = vals(items, "shared_parent_frobenius_norm_9")
            candidate_values = vals(items, "current_candidate_mean_effective_frobenius_norm_9")
            result.append({
                "experiment": experiment,
                "method": method,
                "snapshot_set": label,
                "stream_count": len(items),
                "checkpoint_decisions": sorted({int(item["decision_index_next"]) for item in items}),
                "parent_binding_counts": {
                    str(binding): sum(str(item.get("parent_binding")) == str(binding) for item in items)
                    for binding in sorted({item.get("parent_binding") for item in items}, key=lambda value: str(value))
                },
                "parent_norm_reconstructable_count": len(parent_values),
                "parent_norm_9_mean": statistics.fmean(parent_values) if parent_values else None,
                "parent_norm_9_median": statistics.median(parent_values) if parent_values else None,
                "candidate_norm_reconstructable_count": len(candidate_values),
                "current_candidate_norm_9_mean": statistics.fmean(candidate_values) if candidate_values else None,
                "current_candidate_norm_9_median": statistics.median(candidate_values) if candidate_values else None,
                "last_candidate_provenance_sources_current_count": sum(
                    bool(item.get("last_candidate_provenance_sources_current")) for item in items
                ),
                "pilot_v2_prefix_reference_parent_norm_9": reference["shared_parent_frobenius_norm_9"],
                "pilot_v2_prefix_reference_candidate_norm_9": reference["current_candidate_mean_effective_frobenius_norm_9"],
            })
    return result


def run() -> dict[str, Any]:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Preserve existing diagnostic output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_dir = OUTPUT_ROOT / "code_snapshot"
    snapshot_dir.mkdir()
    script_snapshot = snapshot_dir / Path(__file__).name
    script_snapshot.write_bytes(Path(__file__).read_bytes())
    script_snapshot_info = {
        "path": _relative(script_snapshot),
        "sha256": sha256(script_snapshot),
    }

    inputs: dict[str, str] = {}
    calibration_summary, calibration_streams = _completed_streams(
        CALIBRATION_ROOT, "calibration", 12,
    )
    comparison_summary, comparison_streams = _completed_streams(
        COMPARISON_ROOT, "comparison", 80,
    )
    inputs[_relative(CALIBRATION_ROOT / "summary.json")] = sha256(CALIBRATION_ROOT / "summary.json")
    inputs[_relative(COMPARISON_ROOT / "summary.json")] = sha256(COMPARISON_ROOT / "summary.json")

    inventory, rows = _read_stream_checkpoints(
        CALIBRATION_ROOT, calibration_streams, experiment="retention_value_calibration_main",
        methods={"random"}, inputs=inputs,
    )
    comparison_inventory, comparison_rows = _read_stream_checkpoints(
        COMPARISON_ROOT, comparison_streams, experiment="retention_value_comparison_main",
        methods=set(TARGET_METHODS), inputs=inputs,
    )
    inventory.extend(comparison_inventory)
    rows.extend(comparison_rows)

    reference = _pilot_prefix_reference(inputs)
    rows.append({
        "experiment": "retention_mechanism_pilot_v2",
        "stage": "pilot_prefix_reference",
        "method": "prefix_reference",
        "stream_id": "s25089_dynamics_old_major_prefix",
        "pair_kind": "dynamics",
        "seed": 25089,
        "status": "completed",
        "decision_index_next": None,
        "checkpoint_path": reference["observer_prefix_path"],
        "checkpoint_sha256": reference["observer_prefix_sha256"],
        "parent_binding": reference["parent_binding"],
        "learner_parent_group": reference["parent_binding"],
        "parent_binding_consistent": True,
        "parent_version": None,
        "parent_written_decision": None,
        "parent_source_fit_steps": None,
        "shared_parent_frobenius_norm_9": reference["shared_parent_frobenius_norm_9"],
        "candidate_group": 1,
        "candidate_current_slots": json.dumps([int(row["slot"]) for row in reference["candidate_current_source_records"]]),
        "candidate_current_tasks": json.dumps([str(row["task"]) for row in reference["candidate_current_source_records"]]),
        "candidate_current_source_records": json.dumps(reference["candidate_current_source_records"], sort_keys=True),
        "current_candidate_mean_effective_frobenius_norm_9": reference["current_candidate_mean_effective_frobenius_norm_9"],
        "current_candidate_reconstruction_status": "observer_prefix_exact_candidate_mean",
        "last_candidate_matrix_persisted": False,
        "last_candidate_provenance_sources_current": True,
        "last_candidate_mean_effective_frobenius_norm_9": reference["current_candidate_mean_effective_frobenius_norm_9"],
        "logical_group_tags": "[]",
        "physical_shared_group_tags": "[]",
        "resident_tasks": "[]",
        "slot_generations": "[]",
        "fitted_steps": "[]",
    })

    column_names = list(rows[0])
    csv_path = OUTPUT_ROOT / "checkpoint_magnitudes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "format": "retention_content_diagnostic_v2",
        "scope": "Read-only coefficient magnitude inventory. No rewards, raw step ledgers, or in-progress main-run outputs were used.",
        "supersedes": {
            "path": "outputs/retention_content_diagnostics/summary.json",
            "reason": "The v1 candidate calculation averaged only across two task matrices, yielding 27 coefficients, while the pilot candidate is the producer's 9-coefficient mean across two slots and three bands. V1 candidate comparisons across those dimensions are invalid and must not be used. V1 parent norms remain valid.",
        },
        "code_snapshot": script_snapshot_info,
        "definitions": {
            "parent_norm": "Frobenius norm of the stored nine-coefficient shared matrix, regardless of binding state.",
            "current_candidate_norm": "Frobenius norm of the next decision's nine-coefficient common_mean_effective candidate. Its source pair is the final two outcome bout metadata rows before the checkpoint; the pair is matched to current resident task generations and reconstructed by averaging across the two slots and three bands, then casting to float32 as the producer does. Bout reward fields are not read into the diagnostic record or used.",
            "last_candidate_matrix": "The most recent transient candidate matrix is not persisted in calibration/comparison checkpoints. Its provenance is tested against current resident generations; exact reconstruction is reported only if both sources still match.",
            "early_late": "Earliest and latest persisted decision checkpoints available per stream, rather than chosen decision thresholds.",
        },
        "inputs": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.items())],
        "input_run_inventory": {
            "calibration": {
                "requested_stream_count": calibration_summary["requested_stream_count"],
                "completed_stream_count": calibration_summary["completed_stream_count"],
                "selected_method": "random",
                "selected_stream_count": len(inventory) - len(comparison_inventory),
                "checkpoint_count": sum(item["checkpoint_count"] for item in inventory if item["experiment"] == "retention_value_calibration_main"),
            },
            "comparison": {
                "requested_stream_count": comparison_summary["requested_stream_count"],
                "completed_stream_count": comparison_summary["completed_stream_count"],
                "selected_methods": list(TARGET_METHODS),
                "selected_stream_count": len(comparison_inventory),
                "checkpoint_count": sum(item["checkpoint_count"] for item in comparison_inventory),
            },
            "pilot_v2_prefix_reference": reference,
        },
        "stream_checkpoint_inventory": inventory,
        "cohort_summaries": _summary_by_cohort(rows, reference),
        "checkpoint_rows": len(rows) - 1,
        "pilot_reference_rows": 1,
        "csv_path": _relative(csv_path),
        "csv_sha256": sha256(csv_path),
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": _relative(OUTPUT_ROOT),
        "calibration_streams": result["input_run_inventory"]["calibration"]["selected_stream_count"],
        "comparison_streams": result["input_run_inventory"]["comparison"]["selected_stream_count"],
        "checkpoint_rows": result["checkpoint_rows"],
        "pilot_v2_prefix_reference": result["input_run_inventory"]["pilot_v2_prefix_reference"],
    }, indent=2), flush=True)

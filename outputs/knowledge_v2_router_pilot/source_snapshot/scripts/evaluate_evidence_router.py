"""Replay saved unlabeled C1 streams through the evidence-gated router.

No environment interactions are generated here. The saved observations,
actions, rewards, and next observations are the only fields passed to the
router; context IDs and segment boundaries are used only by the shared offline
metrics function after each replay is complete.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from online_rl.evidence_router import EvidenceRouter  # noqa: E402
import run_task_free_identification as reference  # noqa: E402


INPUTS = {
    "pilot": ROOT / "outputs" / "knowledge_identification_pilot",
    "formal": ROOT / "outputs" / "knowledge_identification_formal",
}
OUTPUTS = {
    "pilot": ROOT / "outputs" / "knowledge_v2_router_pilot",
    "formal": ROOT / "outputs" / "knowledge_v2_router_formal",
}
WINDOW = 32
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snapshot_sources(output_dir: Path) -> dict[str, Any]:
    files = (
        ROOT / "src" / "online_rl" / "evidence_router.py",
        ROOT / "scripts" / "evaluate_evidence_router.py",
        ROOT / "scripts" / "run_task_free_identification.py",
    )
    snapshot_root = output_dir / "source_snapshot"
    hashes: dict[str, str] = {}
    for source in files:
        relative = source.relative_to(ROOT)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = _sha256_file(source)
    return {"source_snapshot_directory": str(snapshot_root.relative_to(ROOT)), "source_sha256": hashes}


def _movement(obs: np.ndarray, next_obs: np.ndarray) -> int | None:
    before = (int(round(float(obs[0]) * 8)), int(round(float(obs[1]) * 8)))
    after = (int(round(float(next_obs[0]) * 8)), int(round(float(next_obs[1]) * 8)))
    delta = (after[0] - before[0], after[1] - before[1])
    if delta == (0, 0):
        return None
    return DELTAS.index(delta)


def _evaluate_stream(
    stream: dict[str, np.ndarray],
    *,
    seed: int,
    output_dir: Path,
    schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    router = EvidenceRouter(max_contexts=6, window=WINDOW)
    observations = stream["observations"]
    actions = stream["actions"]
    rewards = stream["rewards"]
    next_observations = stream["next_observations"]
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    correct: list[bool] = []
    movement_predictions = 0
    censored_count = 0
    slot_generations = [-1] * router.max_contexts
    next_generation = 0
    start_time = time.perf_counter()
    trace_path = output_dir / f"steps_seed_{seed}_evidence_router_w{WINDOW}.csv.gz"

    with gzip.open(trace_path, "wt", encoding="utf-8", newline="") as zipped:
        writer = None
        for index in range(len(actions)):
            obs = observations[index]
            action = int(actions[index])
            route = router.route(obs)
            probabilities = router.predict_direction(obs, action)
            routed_slot = int(route["slot"])
            generation = slot_generations[routed_slot] if routed_slot >= 0 else -1
            confidence = float(route["confidence"])
            support = float(route["support"])
            reason = str(route["reason"])
            if routed_slot >= 0:
                x = int(round(float(obs[0]) * 8))
                untried = router._cell_total(routed_slot, x // 3, action) == 0
            else:
                untried = True

            nxt = next_observations[index]
            actual_direction = _movement(obs, nxt)
            is_censored = actual_direction is None
            if is_censored:
                censored_count += 1
                predicted_correct = None
                logloss = None
            else:
                movement_predictions += 1
                logloss = -math.log(max(float(probabilities[actual_direction]), 1e-12))
                predicted_correct = int(int(np.argmax(probabilities)) == actual_direction)
                losses.append(float(logloss))
                correct.append(bool(predicted_correct))

            observe_info = router.observe(obs, action, float(rewards[index]), nxt)
            allocated_slot = observe_info.get("allocated_slot")
            allocated_generation = None
            if allocated_slot is not None:
                slot_generations[int(allocated_slot)] = next_generation
                allocated_generation = next_generation
                next_generation += 1
            before_x = int(round(float(obs[0]) * 8))
            row = {
                "step": index,
                "action": action,
                "band": int(before_x // 3),
                "slot": routed_slot,
                "generation": int(generation),
                "allocated_generation": allocated_generation,
                "confidence": confidence,
                "support": support,
                "reason": reason,
                "untried_action": bool(untried),
                "actual_direction_observer_only": actual_direction,
                "predicted_direction": int(np.argmax(probabilities)),
                "transition_correct": predicted_correct,
                "transition_logloss": logloss,
                "wall_censored": is_censored,
                "allocated_slot": allocated_slot,
                "evicted_slot": observe_info.get("evicted_slot"),
                "switched": bool(observe_info.get("switched", False)),
            }
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(zipped, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)

    elapsed = time.perf_counter() - start_time
    memory = router.nbytes()
    memory["state_dict_json_bytes"] = len(
        json.dumps(router.state_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    metadata = {
        "schedule": schedule,
        "allocations": router.allocations,
        "evictions": router.evictions,
        "switches": router.switches,
        "censored": router._censored,
        "active_slots": list(router.active_slots),
    }
    metrics = reference._posthoc_metrics(
        stream,
        rows,
        losses,
        correct,
        movement_predictions,
        censored_count,
        memory,
        elapsed,
        metadata,
    )
    metrics.update({
        "algorithm": "evidence_router",
        "window": WINDOW,
        "seed": seed,
        "per_step_output": str(trace_path.relative_to(ROOT)),
    })
    return metrics


def run_phase(phase: str, output_dir: Path | None = None, input_dir: Path | None = None) -> dict[str, Any]:
    source_dir = INPUTS[phase] if input_dir is None else input_dir.resolve()
    destination = OUTPUTS[phase] if output_dir is None else output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"saved stream directory does not exist: {source_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {destination}")

    source_manifest = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_streams = {int(item["seed"]): item for item in source_manifest["streams"]}
    seeds = [331] if phase == "pilot" else [331, 347, 359]
    provenance = _snapshot_sources(destination)
    records: list[dict[str, Any]] = []
    stream_provenance: list[dict[str, Any]] = []
    for seed in seeds:
        details = manifest_streams[seed]
        stream_path = source_dir / f"stream_seed_{seed}.npz"
        expected_hash = details.get("stream_sha256")
        actual_hash = _sha256_file(stream_path)
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError(f"saved stream hash mismatch for seed {seed}")
        with np.load(stream_path, allow_pickle=False) as saved:
            stream = {key: saved[key] for key in saved.files}
        record = _evaluate_stream(
            stream,
            seed=seed,
            output_dir=destination,
            schedule=details["schedule"],
        )
        record["stream_file"] = str(stream_path.relative_to(ROOT)) if stream_path.is_relative_to(ROOT) else str(stream_path)
        records.append(record)
        stream_provenance.append({
            "seed": seed,
            "stream_file": record["stream_file"],
            "stream_sha256": actual_hash,
            "steps": int(len(stream["actions"])),
            "source_schedule": details["schedule"],
        })
        print(json.dumps({
            "phase": phase,
            "seed": seed,
            "route_coverage": record["route_coverage"],
            "ari": record["offline_ari_including_unresolved"],
            "allocations": record["allocations"],
            "evictions": record["evictions"],
            "reuse": record["known_context_revisit_generation_reuse"],
        }), flush=True)

    baseline_by_seed = {
        int(record["seed"]): record
        for record in source_manifest["records"]
        if record.get("algorithm") == "robust_bank" and int(record.get("window", -1)) == WINDOW
    }
    comparison_fields = (
        "route_coverage",
        "offline_ari_including_unresolved",
        "resolved_route_purity_by_slot_generation",
        "resolved_route_completeness_macro_by_true_context",
        "known_context_revisit_generation_reuse",
        "allocations",
        "evictions",
    )
    baseline_comparison = {}
    for record in records:
        baseline = baseline_by_seed.get(int(record["seed"]))
        if baseline is None:
            continue
        baseline_comparison[str(record["seed"])] = {
            field: {
                "evidence_router": record.get(field),
                "robust_bank_w32": baseline.get(field),
            }
            for field in comparison_fields
        }

    configuration = {
        "router": "EvidenceRouter",
        "max_contexts": 6,
        "window": WINDOW,
        "min_seed_events": 4,
        "min_seed_cells": 2,
        "min_cell_support": 2,
        "min_conflict_cells_per_profile_for_allocation": 2,
        "profile_counts": "uint16 with bounded per-cell support",
        "stationary_wall_transitions": "censored",
    }
    _write_json(destination / "router_configuration.json", configuration)
    _write_json(destination / "metrics.json", {"records": records})
    manifest = {
        "phase": phase,
        **provenance,
        "input_directory": str(source_dir),
        "input_manifest_sha256": _sha256_file(source_dir / "summary.json"),
        "streams": stream_provenance,
        "generated_environment_interactions": 0,
        "offline_algorithm_evaluations": len(records),
        "baseline": "robust_bank, window 32 from saved identification output",
        "baseline_comparison": baseline_comparison,
        "records": records,
    }
    _write_json(destination / "summary.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_phase(args.phase, output_dir=args.output, input_dir=args.input)


if __name__ == "__main__":
    main()

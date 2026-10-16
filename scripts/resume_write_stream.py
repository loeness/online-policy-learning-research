"""Resume an interrupted cohort by retaining complete cells and rerunning partial cells.

Partial environments cannot be restored exactly. Their raw records are moved
to a separate, counted interruption archive before a full cell is rerun.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_write_stream as stream


def resume(output: Path, interrupted: Path, selection_path: Path, workers: int = 2):
    output, interrupted = output.resolve(), interrupted.resolve()
    intended = (ROOT / "outputs").resolve()
    for target in (output, interrupted):
        if not target.is_relative_to(intended) or target == intended:
            raise ValueError(f"Target escapes the intended experiment outputs: {target}")
    if (output / "summary.json").exists():
        raise FileExistsError("Cohort already has a complete summary")
    if interrupted.exists() and any(interrupted.iterdir()):
        raise FileExistsError("Preserve previous interruption archive")
    config_path = output / "frozen_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = stream._load_selection(selection_path)
    records, tasks, partials = [], [], []
    for seed in config["seeds"]:
        for condition in config["conditions"]:
            folder = output / f"seed_{seed}" / condition
            if (folder / "summary.json").exists():
                record = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                if record["seed"] != seed or record["condition"] != condition or record["bout_steps"] != config["bout_steps"]:
                    raise ValueError("Completed-cell identity differs from frozen grid")
                records.append(record)
                continue
            if folder.exists() and any(folder.iterdir()):
                target = (interrupted / f"seed_{seed}" / condition).resolve()
                if not folder.resolve().is_relative_to(output) or not target.is_relative_to(interrupted):
                    raise ValueError("Unsafe partial-cell move")
                counts = collections.Counter()
                if (folder / "steps.csv").exists():
                    with (folder / "steps.csv").open(encoding="utf-8", newline="") as handle:
                        for row in csv.DictReader(handle):
                            if row.get("interaction_kind"):
                                counts[row["interaction_kind"]] += 1
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(target))
                partials.append({"path": str(target.relative_to(interrupted)), "raw_interactions": dict(counts)})
            cell = {**config, "condition": condition, "seed": seed}
            tasks.append((cell, selection, folder))
    if partials:
        totals = {key: sum(row["raw_interactions"].get(key, 0) for row in partials)
                  for key in ("training", "probe", "evaluation")}
        stream._write_json(interrupted / "FAILED_COHORT.json", {
            "status": "externally_interrupted_partial_cells",
            "reason": "Execution session was unavailable and no training processes remained when work resumed. Exact in-flight environment state was not saved. Complete cells remain in the main cohort; only these partial cells are rerun.",
            "raw_interactions": totals, "total_interactions_recorded": sum(totals.values()),
            "count_is_lower_bound": True, "recorded_count_exact": False, "partials": partials,
        })
    note = {"retained_complete_cells": len(records), "rerun_cells": len(tasks),
            "interrupted_archive": str(interrupted), "frozen_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "algorithm_and_hyperparameters_unchanged": True}
    stream._write_json(output / "RESUME_NOTE.json", note)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(stream._run_one, cell, selection, folder) for cell, selection, folder in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"completed {record['seed']} {record['condition']}", flush=True)
    records.sort(key=lambda row: (row["seed"], row["condition"]))
    costs = {key: sum(row["costs"][key] for row in records)
             for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")}
    result = {"phase": config["phase"], "base_sequence": config["base_sequence"],
              "sequence": config["sequence"], "protocol_sha256": config["protocol_sha256"],
              "conditions": config["conditions"], "seeds": config["seeds"],
              "records": records, "cohort_cost": costs, "resume_note": note}
    stream._write_json(output / "summary.json", result)
    print(json.dumps({"phase": result["phase"], "cohort_cost": costs}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interrupted", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    resume(args.output, args.interrupted, args.selection, args.workers)

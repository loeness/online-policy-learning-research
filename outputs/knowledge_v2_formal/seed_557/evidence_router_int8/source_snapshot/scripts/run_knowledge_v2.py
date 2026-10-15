"""New-seed controls for fixed allocation, evidence routing, and consolidation."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from scripts import run_knowledge_loop as loop
from scripts.knowledge_v2_components import PreservingTrialArm, nonblocking_consolidate
from online_rl.evidence_router import EvidenceRouter


CONDITIONS = {
    "single_head": ("single_head_online", False, False),
    "fixed_old_router_int8": ("int8", False, False),
    "evidence_router_int8": ("int8", True, False),
    "evidence_router_old_gate": ("evidence_gate", True, False),
    "evidence_router_nonblocking": ("evidence_gate", True, True),
}
PACKAGE_CAP = 256 * 1024


@contextlib.contextmanager
def variant(condition):
    arm, use_new_router, nonblocking = CONDITIONS[condition]
    old = loop.OnlineArm, loop.TaskFreeRouter, loop._consolidate, loop.save_checkpoint
    original_save = loop.save_checkpoint
    def guarded_save(agent, path):
        size = original_save(agent, path)
        package = size + Path(agent.base_config["foundation"]).stat().st_size
        if package > PACKAGE_CAP:
            raise RuntimeError(f"committed package exceeds predeclared256KiB: {package}")
        return size
    try:
        loop.OnlineArm = PreservingTrialArm
        if use_new_router:
            loop.TaskFreeRouter = EvidenceRouter
        if nonblocking:
            loop._consolidate = nonblocking_consolidate
        loop.save_checkpoint = guarded_save
        yield arm
    finally:
        loop.OnlineArm, loop.TaskFreeRouter, loop._consolidate, loop.save_checkpoint = old


def snapshot(folder, config, condition):
    loop._snapshot_sources(folder, config)
    files = ("scripts/run_knowledge_v2.py", "scripts/knowledge_v2_components.py",
             "src/online_rl/evidence_router.py", "docs/KNOWLEDGE_V2_VALIDATION.md")
    manifest = {}
    for relative in files:
        src, dst = ROOT / relative, folder / "source_snapshot" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        assert digest == hashlib.sha256(src.read_bytes()).hexdigest()
        manifest[relative] = digest
    loop._write_json(folder / "runtime_patches.json", {
        "condition": condition, "underlying_arm": CONDITIONS[condition][0],
        "preserve_unallocated_trial_rows": True, "new_router": CONDITIONS[condition][1],
        "nonblocking_consolidation": CONDITIONS[condition][2],
        "whole_committed_package_cap_bytes": PACKAGE_CAP,
        "package_definition": "W0 file plus saved actor/critic/optimizer/router/replay/RNG checkpoint; workspace separately measured",
        "extra_source_sha256": manifest,
    })


def trace_metrics(folder):
    n, resolved, allocation, evictions = 0, 0, 0, 0
    with (folder / "steps.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            n += 1
            resolved += int(int(row["route_slot"]) >= 0)
            allocation += int(row["allocation_slot"] != "")
            evictions += int(row["evicted_slot"] != "")
    return {"trace_steps": n, "route_coverage": resolved / n,
            "trace_allocations": allocation, "trace_evictions": evictions}


def run(phase):
    torch.set_num_threads(1)
    output = ROOT / f"outputs/knowledge_v2_{phase}"
    seeds, steps = ((787,), 8192) if phase == "pilot" else ((541, 557, 569), 32768)
    conditions = tuple(CONDITIONS) if phase == "formal" else (
        "evidence_router_int8", "evidence_router_old_gate", "evidence_router_nonblocking")
    records = []
    start, cpu = time.perf_counter(), time.process_time()
    config = loop.load_base_config()
    for seed in seeds:
        schedule = loop.make_hidden_schedule(steps, seed + 7000)
        for condition in conditions:
            folder = output / f"seed_{seed}" / condition
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"Preserve existing run: {folder}")
            folder.mkdir(parents=True)
            snapshot(folder, config, condition)
            loop._write_json(folder / "condition.json", {"seed": seed, "condition": condition,
                                                         "steps": steps, "schedule": schedule})
            with variant(condition) as arm:
                summary = loop.run_arm(arm, seed=seed, total_steps=steps, schedule=schedule,
                                       base_config=config, output=folder, phase="v2_" + phase, run_revisit=False)
            summary.update({"condition": condition, **trace_metrics(folder)})
            summary["whole_committed_package_bytes"] = summary["checkpoint_file_bytes"] + Path(config["foundation"]).stat().st_size
            assert summary["whole_committed_package_bytes"] <= PACKAGE_CAP
            loop._write_json(folder / "summary.json", summary)
            records.append(summary)
            print(json.dumps({"seed": seed, "condition": condition, "reward": summary["reward_per_step"],
                              "success": summary["success_rate"], "coverage": summary["route_coverage"],
                              "allocations": summary["trace_allocations"]}), flush=True)
    result = {"phase": phase, "records": records, "steps_per_run": steps,
              "environment_steps": sum(r["training_environment_steps"] for r in records),
              "run_revisit": False, "package_cap_bytes": PACKAGE_CAP,
              "wall_s": time.perf_counter() - start, "cpu_s": time.process_time() - cpu}
    loop._write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    run(parser.parse_args().phase)

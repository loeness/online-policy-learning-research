"""Slot-safe, episode-paired evaluation of the saved online-loop policies."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import psutil
import torch

from online_rl.c1_envs import C1NavigationEnv, c1_contexts
from online_rl.c1_experiment import augment
from online_rl.task_free_router import TaskFreeRouter
from run_knowledge_loop import (
    ALL_ARMS, _install_matrix, load_base_config, restore_agent_checkpoint,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_allocation(matrix, allocation, *, shared):
    """Forget an evicted slot's correction without gradient learning."""
    slot = allocation.get("allocated_slot")
    if shared or slot is None:
        return False
    matrix[int(slot)].fill(0)
    return True


def episode_plan(seed, context_index, episode_index):
    reset_seed = seed + 88000 + context_index * 100 + episode_index
    action_seed = seed + 99000 + context_index * 200 + episode_index
    return reset_seed, action_seed, np.random.default_rng(action_seed).random(64)


def evaluate(agent, output):
    start, cpu = time.perf_counter(), time.process_time()
    base_router_state = agent.router.state_dict()
    retained = agent.committed_matrix.copy()
    rows = []
    for context_index, (name, spec) in enumerate(c1_contexts().items()):
        for condition in ("retain_parameters", "zero_corrections"):
            router = TaskFreeRouter.from_state_dict(base_router_state)
            router.reset_history()
            matrix = retained.copy() if condition == "retain_parameters" else np.zeros_like(retained)
            _install_matrix(agent.core, matrix, shared=agent.shared_head)
            env = C1NavigationEnv(spec)
            for episode_index in range(8):
                reset_seed, action_seed, uniforms = episode_plan(agent.seed, context_index, episode_index)
                obs, _ = env.reset(seed=reset_seed)
                episode_return, clears = 0., 0
                old_allocations, old_evictions = router.allocations, router.evictions
                for step, uniform in enumerate(uniforms):
                    route = router.route(obs)
                    slot = int(route["slot"])
                    route_id = 1 if agent.shared_head else slot + 1 if slot >= 0 else 0
                    tensor = torch.as_tensor(augment(obs, route_id)[None], dtype=torch.float32)
                    with torch.no_grad():
                        p = agent.core.get_distribution(tensor).distribution.probs[0].numpy().astype(np.float64)
                    p /= p.sum()
                    action = min(int(np.searchsorted(np.cumsum(p), uniform, side="right")), 3)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    allocation = router.observe(obs, action, float(reward), next_obs)
                    if apply_allocation(matrix, allocation, shared=agent.shared_head):
                        clears += 1
                        _install_matrix(agent.core, matrix, shared=agent.shared_head)
                    episode_return += float(reward)
                    obs = next_obs
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("C1 did not terminate by step64")
                rows.append({
                    "seed": agent.seed, "arm": agent.arm, "context": name,
                    "condition": condition, "episode": episode_index + 1,
                    "reset_seed": reset_seed, "action_seed": action_seed,
                    "return": episode_return, "success": int(info["success"]), "length": step + 1,
                    "new_allocations": router.allocations - old_allocations,
                    "new_evictions": router.evictions - old_evictions,
                    "structural_correction_resets": clears,
                })
            env.close()
    _install_matrix(agent.core, retained, shared=agent.shared_head)
    # Paired rows must have identical episode-level random plans.
    for context in c1_contexts():
        keep = [r for r in rows if r["context"] == context and r["condition"] == "retain_parameters"]
        zero = [r for r in rows if r["context"] == context and r["condition"] == "zero_corrections"]
        assert [(r["reset_seed"], r["action_seed"]) for r in keep] == [(r["reset_seed"], r["action_seed"]) for r in zero]
    keep = [r for r in rows if r["condition"] == "retain_parameters"]
    zero = [r for r in rows if r["condition"] == "zero_corrections"]
    summary = {
        "seed": agent.seed, "arm": agent.arm, "episodes": len(rows),
        "retain_return": float(np.mean([r["return"] for r in keep])),
        "zero_return": float(np.mean([r["return"] for r in zero])),
        "retain_success": float(np.mean([r["success"] for r in keep])),
        "zero_success": float(np.mean([r["success"] for r in zero])),
        "mean_paired_return_gain": float(np.mean([a["return"] - b["return"] for a, b in zip(keep, zero)])),
        "structural_correction_resets": sum(r["structural_correction_resets"] for r in rows),
        "policy_gradient_updates": 0, "environment_steps": sum(r["length"] for r in rows),
        "wall_s": time.perf_counter() - start, "cpu_s": time.process_time() - cpu,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    torch.set_num_threads(1)
    out = ROOT / "outputs/knowledge_loop_revisit_corrected"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Corrected evaluation already exists; preserve it")
    source = out / "source_snapshot"
    source.mkdir(parents=True)
    paths = [Path(__file__), ROOT / "scripts/run_knowledge_loop.py", ROOT / "src/online_rl/task_free_router.py",
             ROOT / "src/online_rl/compact_knowledge.py", ROOT / "docs/KNOWLEDGE_REVISIT_CORRECTION.md"]
    for path in paths:
        shutil.copyfile(path, source / path.name)
    started, cpu = time.perf_counter(), time.process_time()
    records = []
    inputs = {}
    config = load_base_config()
    for seed in (401, 419, 433):
        for arm in ALL_ARMS:
            if arm == "adaptive_assignment":
                continue
            checkpoint = ROOT / f"outputs/knowledge_loop_formal/seed_{seed}/{arm}/checkpoint_latest.pt"
            inputs[str(checkpoint.relative_to(ROOT))] = sha(checkpoint)
            agent = restore_agent_checkpoint(checkpoint, config)
            records.append(evaluate(agent, out / f"seed_{seed}" / arm))
            print(json.dumps(records[-1]), flush=True)
    summary = {
        "status": "complete", "scope": "Corrected revisit only; no training or parameter optimization.",
        "legacy_evaluation": "Preserved in knowledge_loop_formal; excluded from persistence conclusions.",
        "records": records, "environment_steps": sum(r["environment_steps"] for r in records),
        "episodes": sum(r["episodes"] for r in records),
        "wall_s": time.perf_counter() - started, "cpu_s": time.process_time() - cpu,
        "rss_end_bytes": psutil.Process().memory_info().rss,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}, "input_sha256": inputs,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Paid, observer-only counterfactual audit of Step 1 shared-write decisions.

These simulator resets are a diagnostic cost. No resulting observation, return
or ranking is supplied to a deployment learner or used for rule selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from scripts import run_write_decision as wd
from online_rl.write_decision_envs import make_env, task_from_name


def chosen_decisions(records):
    """First and last proposals of each decision outcome, fixed before audit."""
    selected = {}
    for accepted in (False, True):
        candidates = [r for r in records if bool(r.get("accepted")) == accepted
                      and r.get("delta_proposed") is not None
                      and np.linalg.norm(r["delta_proposed"]) > 1e-8]
        if candidates:
            for r in (candidates[0], candidates[-1]):
                selected[int(r["step"])] = r
    return [selected[k] for k in sorted(selected)]


def paired_returns(agent, name, slot, local, shared, group, c_start, delta,
                   *, episodes, seed, cold, domain, rows_writer):
    returns = [[], []]
    steps = 0
    for candidate in (0, 1):
        a, c = local.copy(), shared.copy()
        c[group] = (c_start + candidate * delta).reshape(9)
        if cold:
            a[slot] = 0
        with wd._temporary_actor(agent.core, a, c):
            for episode in range(episodes):
                env = make_env(task_from_name(name), domain)
                rng = np.random.default_rng(seed + episode * 17)
                obs, _ = env.reset(seed=seed + episode * 31)
                total, length, done = 0.0, 0, False
                while not done:
                    augmented = wd.loop.augment_observation(obs, slot + 1)
                    probs = wd._policy_probs(agent.core, augmented[None])[0]
                    action = wd._action_from_uniform(probs, rng.random())
                    obs, reward, term, trunc, _ = env.step(action)
                    total += float(reward)
                    length += 1
                    done = bool(term or trunc)
                env.close()
                steps += length
                returns[candidate].append(total)
                rows_writer.writerow({
                    "taskobserver": name, "slot": slot, "cold": int(cold),
                    "candidate": candidate, "episode": episode,
                    "seed": seed, "return_value": total, "steps": length,
                })
    differences = np.asarray(returns[1]) - np.asarray(returns[0])
    return {
        "taskobserver": name, "slot": slot, "cold": cold,
        "mean_incumbent": float(np.mean(returns[0])),
        "mean_candidate": float(np.mean(returns[1])),
        "paired_delta": float(differences.mean()),
        "paired_deltas": differences.tolist(),
        "episodes_per_policy": episodes, "evaluation_steps": steps,
    }


def audit(cohort, output, *, episodes=8):
    torch.set_num_threads(1)
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing audit: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started, cpu_start = time.perf_counter(), time.process_time()
    import shutil
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__).resolve(), ROOT / "scripts/run_write_decision.py"):
        shutil.copy2(source, snapshot / source.name)
    wd._write_json(output / "audit_config.json", {
        "cohort": str(cohort), "cohort_summary_sha256": wd._sha(cohort / "summary.json"),
        "episodes_per_policy": episodes,
        "selection": "First and last nonzero accepted/rejected proposals per run, deduplicated.",
        "source_sha256": {p.name: wd._sha(p) for p in snapshot.iterdir()},
        "benefit_threshold": 0.05, "harm_threshold": -0.05,
        "no_learner_feedback": True,
    })
    summary = json.loads((cohort / "summary.json").read_text(encoding="utf-8"))
    results = []
    for record in summary["records"]:
        condition, seed = record["condition"], int(record["seed"])
        if condition not in wd.PROPOSED_CONDITIONS:
            continue
        folder = cohort / f"seed_{seed}" / condition
        if not folder.is_dir():
            # The runner may use a flat condition_seed layout; discover only
            # by its saved summary identity, never by guessed task labels.
            matches = [p.parent for p in cohort.rglob("summary.json")
                       if p.parent != cohort and
                       (lambda r: r.get("condition") == condition and
                        int(r.get("seed", -1)) == seed)(json.loads(p.read_text(encoding="utf-8")))]
            if len(matches) != 1:
                raise ValueError(f"Cannot identify run folder for {condition}/{seed}: {matches}")
            folder = matches[0]
        decision_path = folder / "decisions.jsonl"
        decisions = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line]
        for decision in chosen_decisions(decisions):
            step, group = int(decision["step"]), int(decision["group"])
            path = folder / "checkpoints" / f"checkpoint_step_{step:08d}.pt"
            checkpoint_hash = wd._sha(path)
            agent = wd.restore_agent_checkpoint(path)
            local, shared, groups = wd._actor_arrays(agent.core)
            delta = np.asarray(decision["delta_proposed"], dtype=np.float32).reshape(3, 3)
            c_start = np.asarray(decision["c_start"], dtype=np.float32).reshape(3, 3).copy()
            expected_post = c_start + (delta if decision["accepted"] else 0)
            if not np.allclose(shared[group].reshape(3, 3), expected_post, atol=1e-6):
                raise ValueError("Post-checkpoint does not match the recorded write decision")
            current_slot = int(decision["slot"])
            current_task = decision["taskobserver"]
            targets = [(current_task, current_slot, True)]
            for name, slot in sorted(agent.slot_map.items()):
                if slot != current_slot and agent.profiles["seen"][slot] and groups[slot] == group:
                    targets.append((name, int(slot), False))
            per_target = []
            audit_dir = output / f"{condition}_seed_{seed}_step_{step}"
            audit_dir.mkdir()
            with (audit_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["taskobserver", "slot", "cold", "candidate", "episode", "seed", "return_value", "steps"])
                writer.writeheader()
                for index, (name, slot, cold) in enumerate(targets):
                    per_target.append(paired_returns(
                        agent, name, slot, local, shared, group, c_start, delta,
                        episodes=episodes, seed=seed + 900000 + step * 13 + index * 1000,
                        cold=cold, domain=agent.run_config.get("domain", "navigation"),
                        rows_writer=writer,
                    ))
            cold_gain = per_target[0]["paired_delta"]
            old = [r["paired_delta"] for r in per_target[1:]]
            harmed = bool(old and min(old) < -0.05)
            beneficial = bool(cold_gain > 0.05 and old and min(old) >= -0.05)
            result = {
                "condition": condition, "seed": seed, "step": step,
                "accepted": bool(decision["accepted"]), "reason": decision["reason"],
                "checkpoint_sha256": checkpoint_hash, "targets": per_target,
                "current_cold_gain": cold_gain, "old_worst_delta": min(old) if old else None,
                "harmful_on_audited_old_tasks": harmed,
                "beneficial_by_diagnostic_threshold": beneficial,
                "accepted_harmful": bool(decision["accepted"] and harmed),
                "rejected_beneficial": bool(not decision["accepted"] and beneficial),
                "evaluation_steps": sum(r["evaluation_steps"] for r in per_target),
                "diagnostic_only": True,
            }
            if wd._sha(path) != checkpoint_hash:
                raise RuntimeError("Audit modified input checkpoint")
            wd._write_json(audit_dir / "summary.json", result)
            results.append(result)
    final = {
        "results": results, "evaluation_steps": sum(r["evaluation_steps"] for r in results),
        "training_steps": 0, "probe_steps": 0,
        "accepted_harmful": sum(r["accepted_harmful"] for r in results),
        "rejected_beneficial": sum(r["rejected_beneficial"] for r in results),
        "wall_s": time.perf_counter() - started, "cpu_s": time.process_time() - cpu_start,
        "selection_rule": "First and last nonzero accepted and rejected proposals per run; duplicates removed.",
        "thresholds": {"cold_benefit": 0.05, "old_harm": -0.05},
        "limitations": "Paid simulator diagnostic, not deployment evidence or selection data. Few paired episodes do not establish true policy-value labels or calibrated error rates. Old targets are resident routes sharing the affected parent.",
    }
    wd._write_json(output / "summary.json", final)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    result = audit(args.input, args.output, episodes=args.episodes)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, ensure_ascii=False))

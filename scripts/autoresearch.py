"""Bounded RL adaptation of the propose/run/measure/keep experimental loop.

This runner searches declared validation configurations. Code revisions remain
auditable in the coding-agent session; it is not Karpathy's unmodified LLM
trainer and does not claim to autonomously prove a research hypothesis.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.experiment import run_one, read_episodes, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/autoresearch_validation")
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--dataset", default="data/pretrain_source.npz")
    p.add_argument("--checkpoint", default="data/foundation.pt")
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    original = json.loads(Path(args.config).read_text(encoding="utf-8"))
    candidates = [(.001, .04), (.003, .04), (.01, .06)]
    best_score, selected = float("-inf"), None
    fieldnames = ["trial", "learning_rate", "ent_coef", "score", "mean_online_return", "final_success",
                  "online_steps", "evaluation_steps", "elapsed_s", "agent_bytes", "status", "config_hash"]
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trial, (lr, entropy) in enumerate(candidates):
            config = {**original, "learning_rate": lr, "ent_coef": entropy, "a_steps": 8192}
            directory = root / f"trial_{trial}"
            print(f"VALIDATION trial={trial} lr={lr} entropy={entropy}", flush=True)
            summary = run_one(directory, "A", "global_residual", 901,
                              args.dataset, args.checkpoint, config)
            rows = read_episodes(directory / "episodes.csv")
            mean_return = sum(float(r["return"]) for r in rows) / max(1, len(rows))
            success = summary["evaluation"]["shift_cw"]["success_rate"]
            # Fixed before seeing validation outcomes. The environment return is
            # bounded but episode lengths vary, so all raw interactions are logged.
            score = success + .1 * mean_return
            status = "discard"
            if score > best_score:
                best_score, selected = score, (lr, entropy)
                status = "keep"
            writer.writerow({"trial": trial, "learning_rate": lr, "ent_coef": entropy,
                             "score": score, "mean_online_return": mean_return, "final_success": success,
                             "online_steps": summary["online_steps"], "evaluation_steps": summary["evaluation_steps"],
                             "elapsed_s": summary["online_elapsed_s"], "agent_bytes": summary["peak_agent_state_bytes"],
                             "status": status, "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
            file.flush()
            print(f"RESULT score={score:.4f} success={success:.3f} mean_return={mean_return:.3f} {status}", flush=True)
            gc.collect()
    chosen = {**original, "learning_rate": selected[0], "ent_coef": selected[1]}
    write_json("configs/selected.json", chosen)
    write_json(root / "selection.json", {"selected": chosen, "validation_seed": 901,
                                         "heldout_main_seeds": [11, 23, 37], "best_score": best_score,
                                         "scope": "declared PPO configuration search; no test-seed selection"})
    print("Selected configuration saved to configs/selected.json", flush=True)


if __name__ == "__main__":
    main()

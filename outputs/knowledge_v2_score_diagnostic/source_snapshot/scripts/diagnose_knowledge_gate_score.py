"""Observe two surrogate scores while preserving every legacy gate decision."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from scripts import run_knowledge_loop as loop
from scripts.knowledge_v2_components import behavior_relative_surrogate
from scripts.run_knowledge_v2 import variant, snapshot


def main():
    torch.set_num_threads(1)
    output = ROOT / "outputs/knowledge_v2_score_diagnostic"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing diagnostic")
    output.mkdir(parents=True)
    config = loop.load_base_config()
    snapshot(output, config, "evidence_router_old_gate")
    source = Path(__file__).read_bytes()
    (output / "source_snapshot/scripts/diagnose_knowledge_gate_score.py").write_bytes(source)
    old = loop._local_surrogate
    scores = []
    def observe_score(agent, matrix, validation):
        legacy, count = old(agent, matrix, validation)
        # Match legacy advantage normalization; change only the importance
        # reference to the probabilities that actually generated these actions.
        selected = [r for r in validation if r.slot >= 0 and
                    r.generation == agent.slot_generation[r.slot]]
        advantages = np.asarray([r.advantage for r in selected], dtype=np.float64)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cloned = [SimpleNamespace(**{**vars(r), "advantage": float(a)}) for r, a in zip(selected, advantages)]
        corrected, _ = behavior_relative_surrogate(agent, matrix, cloned)
        scores.append({"probe": len(scores), "samples": count,
                       "legacy_score": legacy, "behavior_relative_score": corrected,
                       "sign_differs": bool(np.isfinite(legacy) and np.isfinite(corrected) and
                                            (legacy >= 0) != (corrected >= 0))})
        return legacy, count
    try:
        loop._local_surrogate = observe_score
        with variant("evidence_router_old_gate") as arm:
            summary = loop.run_arm(arm, seed=787, total_steps=8192,
                                   schedule=loop.make_hidden_schedule(8192, 7787),
                                   base_config=config, output=output, phase="score_diagnostic", run_revisit=False)
    finally:
        loop._local_surrogate = old
    original = ROOT / "outputs/knowledge_v2_pilot/seed_787/evidence_router_old_gate/steps.csv"
    a = list(csv.DictReader(original.open(encoding="utf-8")))
    b = list(csv.DictReader((output / "steps.csv").open(encoding="utf-8")))
    identical = all((r["action"], r["reward"]) == (s["action"], s["reward"]) for r, s in zip(a, b)) and len(a) == len(b) == 8192
    assert identical, "Diagnostic hook changed the action/reward trajectory"
    summary.update({"scores": scores, "sign_disagreements": sum(r["sign_differs"] for r in scores),
                    "original_pilot_trajectory_identical": identical,
                    "source_sha256": hashlib.sha256(source).hexdigest(),
                    "interpretation": "Auxiliary score diagnostic only; original score controls every gate decision. Neither score is a value guarantee."})
    loop._write_json(output / "summary.json", summary)
    print(json.dumps({"probes": len(scores), "sign_disagreements": summary["sign_disagreements"],
                      "trajectory_identical": identical, "interactions": 8192}))


if __name__ == "__main__":
    main()

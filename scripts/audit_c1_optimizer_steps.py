"""Recover exact LeafAdam update counts without modifying archived run results.

PyTorch can invoke a post-hook twice when LeafAdam.step calls a separately
wrapped Adam.step. LeafAdam's checkpoint counter increments exactly once.
This diagnostic leaves all result.json files and checkpoints unchanged.
"""
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/c1_supplement"

def audit(path):
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["status"] != "complete":
        return None
    cumulative = {}
    for phase in ("seen", "heldout", "revisit", "history"):
        payload = torch.load(path.parent/phase/"agent.pt", map_location="cpu", weights_only=False)
        optimizer = payload["optimizer"]
        if "step_count" not in optimizer:
            return {"result": path.relative_to(ROOT).as_posix(),
                    "counter_kind": "ordinary_Adam_post_hook",
                    "raw_optimizer_steps": row["actual_optimizer_steps"],
                    "corrected_optimizer_steps": row["actual_optimizer_steps"],
                    "correction_needed": False}
        cumulative[phase] = int(optimizer["step_count"])
    exact = cumulative["history"]
    result = {"result": path.relative_to(ROOT).as_posix(),
              "counter_kind": "LeafAdam_checkpoint_single_increment",
              "raw_optimizer_steps": row["actual_optimizer_steps"],
              "corrected_optimizer_steps": exact,
              "checkpoint_cumulative_steps": cumulative,
              "phase_optimizer_steps": {
                  "seen": cumulative["seen"],
                  "heldout": cumulative["heldout"]-cumulative["seen"],
                  "revisit": cumulative["revisit"]-cumulative["heldout"],
                  "history": cumulative["history"]-cumulative["revisit"]},
              "correction_needed": exact != row["actual_optimizer_steps"]}
    (path.parent/"optimizer_step_audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result

def main():
    rows = []
    for path in sorted(OUT.rglob("result.json")):
        if "formal" not in path.parts and "validation" not in path.parts:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if "actual_optimizer_steps" not in row:
            continue
        result = audit(path)
        if result is not None:
            rows.append(result)
    report = {"scope": "Counter audit only. Raw metrics/checkpoints unchanged; policy updates were not duplicated.",
              "corrected_runs": sum(r["correction_needed"] for r in rows), "runs": rows}
    (OUT/"optimizer_step_audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"audited_runs": len(rows), "corrected_runs": report["corrected_runs"]}))

if __name__ == "__main__":
    main()

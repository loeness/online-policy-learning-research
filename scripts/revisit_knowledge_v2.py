"""Factory-aware, slot-safe persistence test for the completed V2 endpoints."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from scripts import run_knowledge_loop as loop
from scripts import run_knowledge_revisit_corrected as evaluation
from scripts.restore_knowledge_v2 import restore_v2


def main():
    torch.set_num_threads(1)
    cohort = json.loads((ROOT / "outputs/knowledge_v2_formal/summary.json").read_text())
    output = ROOT / "outputs/knowledge_v2_revisit"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing V2 revisit")
    output.mkdir(parents=True)
    sources = {}
    for relative in ("scripts/revisit_knowledge_v2.py", "scripts/restore_knowledge_v2.py",
                     "scripts/run_knowledge_revisit_corrected.py", "scripts/run_knowledge_v2.py",
                     "scripts/knowledge_v2_components.py", "src/online_rl/evidence_router.py"):
        source = ROOT / relative
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    records = []
    inputs = {}
    for row in cohort["records"]:
        seed, condition = row["seed"], row["condition"]
        folder = ROOT / f"outputs/knowledge_v2_formal/seed_{seed}/{condition}"
        inputs[str((folder / "checkpoint_latest.pt").relative_to(ROOT))] = hashlib.sha256((folder / "checkpoint_latest.pt").read_bytes()).hexdigest()
        agent = restore_v2(folder)
        old_factory = evaluation.TaskFreeRouter
        try:
            evaluation.TaskFreeRouter = type(agent.router)
            result = evaluation.evaluate(agent, output / f"seed_{seed}" / condition)
        finally:
            evaluation.TaskFreeRouter = old_factory
        result["condition"] = condition
        records.append(result)
        print(json.dumps({"seed": seed, "condition": condition, "gain": result["mean_paired_return_gain"]}), flush=True)
    loop._write_json(output / "summary.json", {"records": records,
                    "environment_steps": sum(r["environment_steps"] for r in records),
                    "episodes": sum(r["episodes"] for r in records), "policy_updates": 0,
                    "source_sha256": sources, "input_sha256": inputs,
                    "scope": "Paired persistence after clearing short history; structural resets on slot reuse. No gradients or tuning."})


if __name__ == "__main__":
    main()

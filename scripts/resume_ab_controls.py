"""Resume the interrupted controls without overwriting completed trials."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from online_rl.controls import _run_analytic_variant
from online_rl.experiment import write_json

root = Path("outputs/ab_plus")
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
no_gradient = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "no_gradient").rglob("summary.json"))]
optimizer = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "optimizer_isolation").rglob("summary.json"))]
source_responses = np.load(config["dataset"])["source_responses"]
analytic = []
reused = new = 0
for stage in config["analytic_stages"]:
    for seed in config["source_seeds"]:
        for variant in ["analytic", "frozen_foundation"]:
            output = root / "analytic" / stage / f"s{seed}" / variant
            summary_path = output / "summary.json"
            if summary_path.exists():
                record = json.loads(summary_path.read_text(encoding="utf-8"))
                reused += 1
            else:
                record = _run_analytic_variant(output, config, stage, seed,
                                               source_responses, Path(config["foundation"]), variant)
                new += 1
                print(json.dumps({"stage": stage, "seed": seed, "variant": variant,
                                  "steps": record["steps"], "return": record["full_cumulative_reward"]}), flush=True)
            analytic.append(record)
expected = len(config["source_seeds"]) * len(config["deployment_contexts"]) * 3
if len(no_gradient) != expected or len(optimizer) != len(config["source_seeds"]) * 2:
    raise RuntimeError("Completed control records missing; do not label suite complete")
write_json(root / "summary.json", {
    "config": config, "no_gradient_controls": no_gradient, "optimizer_isolation": optimizer,
    "analytic_controls": analytic,
    "source_checkpoint_reconstruction": "policy, Adam and router restored; original historic RNG unavailable",
    "deployment_rng": "new declared common environment/action seeds",
    "budget_accounting": "serialized deployed core, optimizer where retained, active fixed training buffer where used; process RSS separately sampled",
    "execution_repair": {"error": "analytic B bout_steps initialized after default expression; reordered",
                         "completed_records_reused": len(no_gradient) + len(optimizer) + reused,
                         "remaining_analytic_records_run": new},
})
print("A/B control suite complete", flush=True)

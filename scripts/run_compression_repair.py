"""A fixed 50%-support repair diagnostic after the primary screen failed."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.compression_experiment import evaluate_deployment, run_continuation, select_candidate
from online_rl.experiment import write_json

base = Path("outputs/compression_attempt")
out = base / "repair"
out.mkdir(parents=True, exist_ok=True)
primary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
repair = json.loads(Path("configs/compression_repair.json").read_text(encoding="utf-8"))
config = {**primary["config"], **repair}
write_json(out / "config.json", config)
before, after, training = [], [], []
for seed in config["source_seeds"]:
    for method in [config["reference"], config["method"]]:
        path = base / "bundles" / f"{method}_s{seed}" / "agent.pt"
        for context in config["contexts"]:
            before.append(evaluate_deployment(path, out / "before" / f"{method}_s{seed}_{context}",
                                              config, seed, context, "test", method))
        continuation = out / "continuation" / f"{method}_s{seed}"
        training.append(run_continuation(path, continuation, config, seed, method))
        for context in config["contexts"]:
            after.append(evaluate_deployment(continuation / "agent.pt",
                                             out / "after" / f"{method}_s{seed}_{context}",
                                             config, seed, context, "test", method))
        print(json.dumps({"phase": "repair", "source_seed": seed, "method": method,
                          "training_return": training[-1]["online_cumulative_return"]}), flush=True)
screen = select_candidate(after, primary["representations"],
                          {**config, "retain_fractions": [repair["retain_fraction"]]})
write_json(out / "summary.json", {"scope": repair["scope"], "config": config,
                                  "before": before, "training": training, "after": after,
                                  "postrepair_empirical_screen": screen,
                                  "primary_selection_unchanged": primary["selection"],
                                  "total_interactions": sum(r["online_steps"] for r in before + after + training)})
print(json.dumps({"repair_interactions": sum(r["online_steps"] for r in before + after + training),
                  "postrepair_screen": screen["selected_method"]}), flush=True)

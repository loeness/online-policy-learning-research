"""Run the fixed C1 state-conditioned online-model controls."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_rl.c1_model_experiment import run_step1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "validation", "formal", "tree-diagnostic"])
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/c1_supplement.json").read_text())
    base = json.loads((ROOT / config["base_config"]).read_text())
    result = run_step1(base, config, ROOT / config["output"] / "step1_models", mode=args.mode)
    print(json.dumps({"status": result["status"], "formal_interactions": result["formal_interactions"]}), flush=True)

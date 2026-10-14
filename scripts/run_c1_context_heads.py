"""Run the validated C1 full shared actor with context-specific small heads."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.c1_context_head_experiment import run_context_head_alternatives


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/c1_supplement.json")
    args = parser.parse_args()
    run_context_head_alternatives(args.config)

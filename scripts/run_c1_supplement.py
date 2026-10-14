"""Run the ordered C1 neural supplements."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_rl.c1_supplement_experiment import run_stage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["step2", "step3"])
    args = parser.parse_args()
    run_stage(args.stage)

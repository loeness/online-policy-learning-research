"""Run the predeclared C1 tensor source-basis alternatives."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.c1_tensor_experiment import run_alternatives  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/c1_supplement.json")
    args = parser.parse_args()
    run_alternatives(args.config)

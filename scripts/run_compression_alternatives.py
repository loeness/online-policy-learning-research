"""Run the predeclared A/B-checkpoint compression comparison."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.alternative_experiment import run_suite

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/compression_alternatives.json")
parser.add_argument("--output", default="outputs/compression_alternatives")
arguments = parser.parse_args()
result = run_suite(json.loads(Path(arguments.config).read_text(encoding="utf-8")), arguments.output)
print(json.dumps({"complete": result["complete"], "total_interactions": result["total_interactions"],
                  "errors": result["errors"]}), flush=True)
if not result["complete"]:
    raise SystemExit(1)

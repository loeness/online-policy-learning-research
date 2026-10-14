from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.compression_experiment import run_suite

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/compression.json")
parser.add_argument("--output", default="outputs/compression_attempt")
args = parser.parse_args()
result = run_suite(json.loads(Path(args.config).read_text(encoding="utf-8")), args.output)
print(json.dumps({"selected": result["selection"]["selected_method"],
                  "total_interactions": result["total_interactions"]}), flush=True)

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.c1_experiment import run_precheck, run_formal

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["precheck", "formal"], required=True)
parser.add_argument("--config", default="configs/c1.json")
parser.add_argument("--output", default="outputs/c1")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text(encoding="utf-8"))
result = (run_precheck if args.mode == "precheck" else run_formal)(config, Path(args.output))
print(json.dumps(result, ensure_ascii=False), flush=True)

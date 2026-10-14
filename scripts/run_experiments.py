from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rl.experiment import generate_dataset, pretrain, run_suite, run_one, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "suite", "one", "report"])
    parser.add_argument("--output", default="outputs/pilot")
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--dataset", default="data/pretrain_source.npz")
    parser.add_argument("--checkpoint", default="data/foundation.pt")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--stage", default="A")
    parser.add_argument("--method", default="global_residual")
    parser.add_argument("--seed", type=int, default=901)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.command == "prepare":
        metadata = generate_dataset(args.dataset, config.get("pretrain_transitions", 20000))
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        metrics = pretrain(args.dataset, args.checkpoint, config)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    elif args.command == "suite":
        result = run_suite(args.output, args.dataset, args.checkpoint, config,
                           seeds=tuple(int(s) for s in args.seeds.split(",")))
        print(json.dumps(result["stage_A_gate"], ensure_ascii=False), flush=True)
        from online_rl.reporting import generate_report
        print(generate_report(args.output), flush=True)
    elif args.command == "one":
        result = run_one(args.output, args.stage, args.method, args.seed, args.dataset, args.checkpoint, config)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        from online_rl.reporting import generate_report
        print(generate_report(args.output), flush=True)


if __name__ == "__main__":
    main()

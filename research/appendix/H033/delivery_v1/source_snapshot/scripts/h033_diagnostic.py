"""Three explicit, offline stages for the frozen H033 diagnostic contract.

This entry point does not launch environments or select a research conclusion.
Its use on actual evidence requires a separately accepted experimental plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/h033_diagnostic_v1.json")
    commands = parser.add_subparsers(dest="stage", required=True)
    features = commands.add_parser("features", help="pre-action allowlisted prefix inputs only")
    features.add_argument("--producer", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    outcomes = commands.add_parser("outcomes", help="separate paired outcome-label packet")
    outcomes.add_argument("--analysis", type=Path, required=True)
    outcomes.add_argument("--output", type=Path, required=True)
    crossfit = commands.add_parser("crossfit", help="four-fold predictions and factual selected values")
    crossfit.add_argument("--features", type=Path, required=True)
    crossfit.add_argument("--outcomes", type=Path, required=True)
    crossfit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent must already exist: {output.parent}")
    # The pure core never imports policy constructors or an environment runner.
    from online_rl.h033_diagnostic import crossfit_diagnostic, validate_config

    config_path = args.config.resolve(strict=True)
    config = validate_config(_read_json(config_path))
    cpu, wall = time.process_time(), time.perf_counter()
    if args.stage == "features":
        import torch
        from online_rl.h033_inputs import build_feature_packet

        torch.set_num_threads(1)
        packet = build_feature_packet(args.producer.resolve(strict=True), config)
    elif args.stage == "outcomes":
        from online_rl.h033_inputs import load_outcome_packet

        packet = load_outcome_packet(args.analysis.resolve(strict=True), config)
    else:
        feature_path = args.features.resolve(strict=True)
        outcome_path = args.outcomes.resolve(strict=True)
        packet = crossfit_diagnostic(_read_json(feature_path), _read_json(outcome_path), config)
        packet["input_packets"] = {
            "features": {"path": str(feature_path), "sha256": _sha256(feature_path)},
            "outcomes": {"path": str(outcome_path), "sha256": _sha256(outcome_path)},
        }
    execution = {
        "stage": args.stage,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "cpu_s": time.process_time() - cpu,
        "wall_s": time.perf_counter() - wall,
        "environment_interactions": 0,
        "formal_statistics": "owned_by_experiment_chat",
    }
    if args.stage == "crossfit":
        packet["execution"] = execution
    else:
        packet["metadata"]["execution"] = execution
    # JSON encoding completes before creating the output, so invalid values do
    # not leave an apparently valid partial packet.
    encoded = json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    print(f"{args.stage}: wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

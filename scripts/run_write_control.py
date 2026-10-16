"""Post-selection causal control: retain all trial learning but forbid shared writes.

This arm was a validation diagnostic, was not selected as a proposal rule, and
is added to the formal seeds as an explicitly declared causal control.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_write_supported

wd = run_write_supported.runner
wd.SOURCE_FILES = (*wd.SOURCE_FILES, Path("scripts/run_write_control.py"))


def run(output, workers):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve control output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = wd._load_config()
    seeds = config["formal_seeds"]
    frozen = {**config, "phase": "post_selection_causal_control",
              "condition": "trial_rollback", "conditions": ["trial_rollback"],
              "seeds": seeds, "bout_steps": config["formal_bout_steps"],
              "output_root": str(output),
              "reason": "Separate shared-write benefit from jointly trained gradients retained locally; not used for method selection."}
    wd._write_json(output / "frozen_config.json", frozen)
    cells = [{**frozen, "seed": seed} for seed in seeds]
    if workers == 1:
        records = [wd.run_one(cell) for cell in cells]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(wd.run_one, cells))
    costs = {key: sum(row["costs"][key] for row in records)
             for key in ("training_steps", "probe_steps", "evaluation_steps", "total_interactions")}
    result = {"phase": frozen["phase"], "conditions": ["trial_rollback"],
              "seeds": seeds, "records": records, "cohort_cost": costs,
              "selection_eligible": False}
    wd._write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    result = run(args.output, args.workers)
    print(json.dumps({"phase": result["phase"], "cohort_cost": result["cohort_cost"]}))

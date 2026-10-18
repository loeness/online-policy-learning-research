"""Read-only diagnostics of the completed single-decision experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch


def diagnose(folder: Path) -> dict:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["errors"] or summary["case_count"] != 32 or summary["branch_count"] != 256:
        raise ValueError("diagnostics require the complete frozen main cohort")
    drift = defaultdict(list)
    choices = defaultdict(lambda: defaultdict(Counter))
    support = Counter()
    fits = []
    windows = defaultdict(lambda: defaultdict(list))
    for path in sorted((folder / "cases").glob("*/decision.json")):
        decision = json.loads(path.read_text(encoding="utf-8"))
        pair, history = decision["pair_kind"], decision["history_regime"]
        for method, action in decision["selectors"].items():
            choices[pair][method][action] += 1
        scores = decision["selector_diagnostics"]["surrogate"]
        for name in ("old_zero_vs_old_parent", "new_parent_vs_zero"):
            support[f"{name}:{scores[name]['status']}"] += 1
        checkpoint = torch.load(path.parent / "prefix/checkpoint.pt", map_location="cpu", weights_only=False)
        fits.append({
            "case_id": decision["case_id"],
            "old_source_steps_each": [entry["steps"] for entry in decision["bootstrap_schedule"]],
            "new_live_slot_fit_steps": checkpoint["fitted_steps"].tolist(),
        })
        # One future suffices for the pre-future intervention drift; the same
        # action is restored and installed identically for both futures.
        for action in ("keep", "replace", "mean", "none"):
            branch_path = path.parent / "branches" / f"{action}__old_major/summary.json"
            branch = json.loads(branch_path.read_text(encoding="utf-8"))
            drift[action].append({
                "case_id": decision["case_id"],
                "prequant_probability": branch["prequantized_behavior_drift"]["max_probability_absolute_drift"],
                "postquant_probability": branch["postquantized_behavior_drift"]["max_probability_absolute_drift"],
                "postquant_kl": branch["postquantized_behavior_drift"]["max_kl_reference_to_candidate"],
            })
        branches = {
            action: json.loads((path.parent / "branches" / f"{action}__{history}/summary.json").read_text(encoding="utf-8"))
            for action in ("keep", "replace", "mean", "none")
        }
        # Prespecified early windows are summed over the same twelve task
        # bouts, then divided by the corresponding number of real steps.
        for method in ("frequency", "surrogate_gain", "surrogate_lower"):
            chosen = branches[decision["selectors"][method]]
            baseline = branches["keep"]
            for width in (256, 512):
                key = f"return_first_{width}"
                difference = sum(b[key] for b in chosen["per_bout"]) - sum(b[key] for b in baseline["per_bout"])
                windows[pair][f"{method}_minus_keep_first_{width}"].append({
                    "seed": decision["seed"], "history": history,
                    "reward_per_step_difference": difference / (12 * width),
                })
    drift_summary = {
        action: {
            "unique_prefix_action_count": len(rows),
            "max_prequant_probability_drift": max(row["prequant_probability"] for row in rows),
            "max_postquant_probability_drift": max(row["postquant_probability"] for row in rows),
            "max_postquant_kl": max(row["postquant_kl"] for row in rows),
            "records": rows,
        }
        for action, rows in drift.items()
    }
    early = {}
    for pair, contrasts in windows.items():
        early[pair] = {}
        for contrast, rows in contrasts.items():
            by_seed = defaultdict(list)
            for row in rows:
                by_seed[row["seed"]].append(row["reward_per_step_difference"])
            values = {str(seed): float(np.mean(items)) for seed, items in sorted(by_seed.items())}
            early[pair][contrast] = {"per_seed": values, "mean": float(np.mean(list(values.values())))}
    return {
        "format": "storage_decision_readonly_diagnostics_v1",
        "environment_steps": 0,
        "intervention_drift": drift_summary,
        "selector_action_counts": {pair: {name: dict(counts) for name, counts in methods.items()} for pair, methods in choices.items()},
        "evidence_support": dict(support),
        "source_fit_budget": fits,
        "primary_early_window_contrasts": early,
        "limits": [
            "Probability drift is measured only at the actual fork observation under each resident route, not uniformly over all states.",
            "Action effects include codec changes and subsequent PPO; small observed drift does not prove a pure retention-only causal effect.",
            "Old and new module source fit budgets differ by design. This is a possible explanation of asymmetry, not an isolated causal finding.",
            "Early-window contrasts are descriptive secondary diagnostics; no new selector or threshold is selected from these results.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = diagnose(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "environment_steps": 0}))


if __name__ == "__main__":
    main()

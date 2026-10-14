"""Generate a compact ledger from completed C1 tensor experiment outputs."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "frozen_settings.json"


def _score_result(result: dict) -> tuple[float, float, float, float]:
    final = result["final_eval"]
    adapted = result["adapted_eval"]
    final_mean = statistics.mean(row["success_rate"] for row in final.values())
    old_mean = statistics.mean(adapted[name]["success_rate"] for name in ("A0", "A1", "B0", "B1"))
    heldout_mean = statistics.mean(adapted[name]["success_rate"] for name in ("A2", "B2"))
    score = 0.5 * final_mean + 0.25 * old_mean + 0.25 * heldout_mean
    return score, final_mean, old_mean, heldout_mean


def build_summary() -> dict:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    validation_ledger: dict[str, dict] = {}
    formal_ledger: dict[str, dict] = {}
    failures: list[dict] = []
    all_runs: list[dict] = []

    for method in ("tensor_learned", "output_mixer", "hierarchical_output_mixer"):
        selection = settings["selected"][method]
        candidates = []
        for candidate in selection["candidates"]:
            result_path = Path(candidate["result_path"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            candidates.append({
                "candidate": candidate["candidate"],
                "params": candidate["params"],
                "status": result["status"],
                "score": candidate["score"],
                "error": result.get("error"),
                "training_process_cpu_s": result.get("training_process_cpu_s"),
                "result_path": candidate["result_path"],
            })
            all_runs.append(result)
            if result["status"] != "complete":
                failures.append({
                    "suite": "validation", "method": method,
                    "candidate": candidate["candidate"], "seed": 919,
                    "status": result["status"], "error": result.get("error"),
                })
        validation_ledger[method] = {
            "seed": 919,
            "candidates": candidates,
            "selected_candidate": selection["selected_candidate"],
            "selected_params": selection["selected_params"],
            "selected_score": selection["selected_score"],
        }

    for method, results in settings["formal_results"].items():
        rows = []
        for result in results:
            score, final_mean, old_mean, heldout_mean = _score_result(result)
            rows.append({
                "seed": result["seed"],
                "status": result["status"],
                "score": score,
                "final_mean_success": final_mean,
                "adapted_old_mean_success": old_mean,
                "adapted_heldout_mean_success": heldout_mean,
                "history_mean_success": statistics.mean(
                    item["success_rate"] for item in result["history_eval"].values()
                ),
                "params": result["params"],
                "training_interactions": result["training_interactions"],
                "evaluation_interactions": result["evaluation_interactions"],
                "training_elapsed_s": result["training_elapsed_s"],
                "training_process_cpu_s": result["training_process_cpu_s"],
                "optimizer_steps": result["actual_optimizer_steps"],
                "adaptive_bytes_peak": result["representation"]["adaptive_parameter_and_optimizer_bytes"],
                "adaptive_cap_bytes": result["adaptive_parameter_optimizer_cap_bytes"],
                "complete_agent_bytes_peak": result["max_complete_agent_bytes"],
                "complete_agent_cap_bytes": result["complete_agent_cap_bytes"],
                "revisit_checkpoint_bytes": result["revisit_checkpoint_bytes"],
                "history_checkpoint_bytes": result["history_checkpoint_bytes"],
            })
            all_runs.append(result)
            if result["status"] != "complete":
                failures.append({
                    "suite": "formal", "method": method, "seed": result["seed"],
                    "status": result["status"], "error": result.get("error"),
                })
        formal_ledger[method] = {
            "runs": rows,
            "mean_score": statistics.mean(row["score"] for row in rows) if rows else None,
            "score_population_sd": statistics.pstdev(row["score"] for row in rows) if len(rows) > 1 else 0.0,
            "mean_final_success": statistics.mean(row["final_mean_success"] for row in rows) if rows else None,
            "mean_adapted_old_success": statistics.mean(row["adapted_old_mean_success"] for row in rows) if rows else None,
            "mean_adapted_heldout_success": statistics.mean(row["adapted_heldout_mean_success"] for row in rows) if rows else None,
            "mean_history_success": statistics.mean(row["history_mean_success"] for row in rows) if rows else None,
        }

    summary = {
        "stage": "representation_alternatives",
        "scope": settings["scope"],
        "validation_seeds": settings["validation_seeds"],
        "formal_seeds": settings["formal_seeds"],
        "validation_ledger": validation_ledger,
        "formal_ledger": formal_ledger,
        "run_counts": {
            "validation": sum(len(row["candidates"]) for row in validation_ledger.values()),
            "formal": sum(len(row["runs"]) for row in formal_ledger.values()),
            "complete": sum(result.get("status") == "complete" for result in all_runs),
            "failures": failures,
        },
        "total_training_interactions": sum(result.get("training_interactions", 0) for result in all_runs),
        "total_evaluation_interactions": sum(result.get("evaluation_interactions", 0) for result in all_runs),
        "summed_training_elapsed_s": sum(result.get("training_elapsed_s", 0.0) for result in all_runs),
        "summed_training_process_cpu_s": sum(result.get("training_process_cpu_s", 0.0) for result in all_runs),
        "all_formal_runs_within_caps": all(
            row["adaptive_bytes_peak"] <= row["adaptive_cap_bytes"]
            and row["complete_agent_bytes_peak"] <= row["complete_agent_cap_bytes"]
            for rows in formal_ledger.values() for row in rows["runs"]
        ),
        "frozen_settings_path": str(SETTINGS_PATH),
        "execution_log_path": str(ROOT / "execution.log"),
        "launch_source_sha256": settings["source_sha256"],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# C1 tensor alternatives: execution summary", "",
        "All runs used low capacity. Validation used seed 919; formal runs used seeds 181, 193, and 211. "
        "Each tensor method was selected from the three frozen candidates; the tree reference used the Step 2 selection.", "",
        "| Arm | Validation scores (candidates 0 / 1 / 2) | Selected | Formal score by seed | Mean formal score | Mean final success | Mean history success |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for method in ("tensor_learned", "output_mixer", "hierarchical_output_mixer"):
        validation = validation_ledger[method]
        formal = formal_ledger[method]
        values = ", ".join(f"{row['score']:.4f}" for row in validation["candidates"])
        seeds = ", ".join(f"{row['seed']}: {row['score']:.4f}" for row in formal["runs"])
        lines.append(
            f"| `{method}` | {values} | candidate {validation['selected_candidate']} `{validation['selected_params']}` | "
            f"{seeds} | {formal['mean_score']:.4f} | {formal['mean_final_success']:.4f} | {formal['mean_history_success']:.4f} |"
        )
    tree = formal_ledger["tree_reference"]
    tree_seeds = ", ".join(f"{row['seed']}: {row['score']:.4f}" for row in tree["runs"])
    lines.append(
        f"| `tree_reference` | Prior Step 2 selection | `{settings['tree_reference_params']}` | {tree_seeds} | "
        f"{tree['mean_score']:.4f} | {tree['mean_final_success']:.4f} | {tree['mean_history_success']:.4f} |"
    )
    lines.extend([
        "",
        f"Completed {summary['run_counts']['complete']} of {summary['run_counts']['validation'] + summary['run_counts']['formal']} jobs; "
        f"failures: {len(failures)}.",
        f"Training interactions: {summary['total_training_interactions']:,}; evaluation interactions: {summary['total_evaluation_interactions']:,}.",
        f"Summed training CPU time: {summary['summed_training_process_cpu_s'] / 3600:.2f} hours; "
        f"summed measured training elapsed time: {summary['summed_training_elapsed_s'] / 3600:.2f} hours.",
        f"All formal runs stayed within both configured caps: {summary['all_formal_runs_within_caps']}.",
        "",
        "The detailed seed-level ledger, candidate results, byte peaks, checkpoints, and source hashes remain in "
        "`frozen_settings.json` and the per-run result JSON files. `execution.log` contains the complete progress stream.",
        "",
    ])
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = build_summary()
    print(json.dumps({
        "run_counts": summary["run_counts"],
        "total_training_interactions": summary["total_training_interactions"],
        "total_evaluation_interactions": summary["total_evaluation_interactions"],
        "summed_training_elapsed_s": round(summary["summed_training_elapsed_s"], 2),
        "summed_training_process_cpu_s": round(summary["summed_training_process_cpu_s"], 2),
        "all_formal_runs_within_caps": summary["all_formal_runs_within_caps"],
        "summary_json": str(ROOT / "summary.json"),
        "summary_md": str(ROOT / "summary.md"),
    }, indent=2))
